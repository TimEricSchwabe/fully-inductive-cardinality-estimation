import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import GATv2Conv, GINEConv, GINConv
from torch_geometric.utils import degree

from easydict import EasyDict

from fice.gnce.models import QueryGNN, TripleModel, QueryCardinalityGNN, IndependentQueryGNN
from fice.ultra.models import RelNBFNet, EntityNBFNet

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA



### Helper Functions ###
def analyze_vector_similarity(embeddings, name="Embeddings", sample_size=2000):
    """
    Analyzes pairwise cosine similarity of a set of vectors.
    
    Args:
        embeddings: Tensor of shape [N, D]
        name: String identifier for plots/logs
        sample_size: Max number of vectors to use (to avoid OOM on large N*N matrices)
    """
    if embeddings.size(0) < 2:
        print(f"[{name}] Not enough vectors to compute similarity.")
        return

    # Detach and move to computation device (can stay on GPU for speed)
    vecs = embeddings.detach()
    
    # Optional: Downsample if too many vectors to avoid creating a massive NxN matrix
    if vecs.size(0) > sample_size:
        perm = torch.randperm(vecs.size(0))
        vecs = vecs[perm[:sample_size]]
    
    # 1. Normalize vectors to unit length (L2 norm)
    # cosine_similarity(a, b) = dot(a, b) / (|a| * |b|)
    # If we normalize first, it becomes just dot(a, b)
    vecs_norm = torch.nn.functional.normalize(vecs, p=2, dim=1)
    
    # 2. Compute Pairwise Similarity Matrix (N x N)
    similarity_matrix = torch.mm(vecs_norm, vecs_norm.t())
    
    # 3. Extract Upper Triangle (off-diagonal)
    # We only care about unique pairs (i, j) where i < j
    triu_indices = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), offset=1)
    similarities = similarity_matrix[triu_indices[0], triu_indices[1]]
    
    # --- Metrics ---
    avg_sim = similarities.mean().item()
    std_sim = similarities.std().item()
    min_sim = similarities.min().item()
    max_sim = similarities.max().item()
    
    print(f"--- {name} Analysis ---")
    print(f"Avg Cosine Similarity: {avg_sim:.4f} ± {std_sim:.4f}")
    print(f"Range: [{min_sim:.4f}, {max_sim:.4f}]")
    
    # --- Plotting ---
    # Move to CPU for plotting
    sim_values = similarities.cpu().numpy()
    
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(sim_values, bins=50, range=(-1, 1), alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(avg_sim, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_sim:.2f}')
        plt.title(f'Pairwise Cosine Similarity Distribution: {name}')
        plt.xlabel('Cosine Similarity (-1 to 1)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure instead of showing to avoid blocking execution
        plt.savefig(f'similarity_hist_{name}.png')
        plt.close()
        print(f"Saved histogram to similarity_hist_{name}.png")
    except Exception as e:
        print(f"Could not plot histogram: {e}")


    vecs_np = vecs.cpu().numpy()
    
    # Project to 2D
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs_np)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], alpha=0.6, s=10, color='teal')
    plt.title(f'PCA Projection (2D): {name}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'pca_projection_{name}.png')
    plt.close()
    print(f"Saved PCA plot to pca_projection_{name}.png")

class EntityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_relation=1, message_func='gine', conditioning='all', layer_norm=True, **kwargs):
        super(EntityGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.message_func = message_func
        self.conditioning = conditioning

        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            if message_func == 'gine':
                self.layers.append(
                    GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            else:
                self.layers.append(
                    GATv2Conv(dims[i], dims[i + 1], heads=1, concat=False, edge_dim=input_dim)
                )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if layer_norm else []
        )


    def forward(self, data, batch=None, relation_representations=None):
        if batch is None:
            batch_size = 1
            device = data.edge_index.device
        else:
            batch_size = batch.batch_size
            device = batch.edge_index.device

        # Preparing node features based on conditioning strategy
        # Conditioning choices: "false" (all 0.1), "seed only" (seeds=1, rest=0), "all" (seeds=1, rest=0.1)
        if self.conditioning in ["false", "all"]:
            x = torch.ones(batch_size, data.num_nodes, self.input_dim, device=device) * 0.1
        elif self.conditioning == "seed only":
            x = torch.zeros(batch_size, data.num_nodes, self.input_dim, device=device)
        else:
            raise ValueError(f"Invalid conditioning value: {self.conditioning}")

        # --- FIX 1: Symmetry Breaking with Degrees ---
        # Inject log(degree) into the first dimension of the features to distinguish nodes structurally
        if data.edge_index.numel() > 0:
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
            # Normalize/Log degree to keep it in a reasonable range
            deg_feature = torch.log(deg + 1).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1).to(device)
            # Add to the first dimension of x (broadcast over batch)
            x = x.clone()
            x[:, :, 0] += deg_feature.squeeze(-1)

        # Reshape once at the beginning: (batch_size, num_nodes, input_dim) -> (batch_size * num_nodes, input_dim)
        x = x.view(-1, x.shape[-1])  # (batch_size * num_nodes, input_dim)




        # Set input of nodes appearing in query based on conditioning (labelling trick)
        if self.conditioning != "false":
        # Indices of the nodes that appear in the query batch
            node_indices = batch.node_embedding_indices 
            valid = node_indices != -1
            graph_idx = batch.batch[valid] # gives the batch index for the bound nodes
            node_pos = node_indices[valid] # gives the global node ids of the bound nodes
            rows = graph_idx * data.num_nodes + node_pos      # the indices in x of the bound nodes
            x[rows] = 1.0   
        
        # Create batched edge_index once: replicate edges for each batch element
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        # Create edge_index for batched graph: replicate edges for each batch element
        batch_edge_list = []
        # also replicate the edge types for each batch element
        batched_edge_type_list = []
        for b in range(batch_size):
            offset = b * num_nodes
            batch_edges = edge_index + offset
            batch_edge_list.append(batch_edges)
            batched_edge_type_list.append(data.edge_type)
        batched_edge_index = torch.cat(batch_edge_list, dim=1)
        batched_edge_type = torch.cat(batched_edge_type_list, dim=0)

        
        # Shape (batch_size * num_edges) - maps each edge to its batch index
        all_edges = torch.repeat_interleave(torch.arange(batch_size, device=device), data.edge_index.size(1))

        # Since relation embeddings are different for each batch, we now use all_edges to pick the correct batch and batched_edge_type to get the global id of the relation
        edge_attr = relation_representations[all_edges, batched_edge_type] # Shape (batch_size * num_edges, embedding_dim)
        
        for i, layer in enumerate(self.layers):
            x_in = x # Save input for residual
            x = layer(x, batched_edge_index, edge_attr=edge_attr)
            if self.norms:
                x = self.norms[i](x)
            if self.message_func != 'gine':
                x = F.gelu(x)
            

            x = x + x_in
        
        # Reshape once at the end: (batch_size * num_nodes, output_dim) -> (batch_size, num_nodes, output_dim)
        x = x.view(batch_size, num_nodes, -1)
        
        return x

class RelationGNN(nn.Module):
    """
    Replacement for RelNBFNet operating on the **relation graph**.
    Exactly four directed edge types exist and we learn a fixed
    embedding vector for each of them.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple = (128, 128),
        num_relation: int = 4,
        heads: int = 4,
        conditioning: str = "all",      
        layer_norm: bool = False,
    ):
        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )


        super().__init__()
        self.num_relation = num_relation
        self.conditioning = conditioning     

        # 4 trainable relation embeddings
        self.rel_emb = nn.Embedding(num_relation, input_dim)

        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            self.layers.append(
                GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
            )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if layer_norm else []
        )
        self.act = nn.GELU()

    def forward(self, data, batch=None):
        """
        data  : PyG Data of the *relation* graph
        batch : batched query graph (only batch_size &  edge_embedding_indices  matter)
        """
        if batch is None:
            bs = 1
            device = data.edge_index.device
        else:
            bs = batch.batch_size
            device = batch.edge_index.device
        num_nodes = data.num_nodes


        if   self.conditioning == "false":
            h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1
        elif self.conditioning == "seed only":
            h = torch.zeros(bs, num_nodes, self.rel_emb.embedding_dim, device=device)
        else:  # "all"
            h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1

        # --- FIX: Symmetry Breaking with Degrees for Relation Graph ---
        # Inject log(degree) into the first dimension to distinguish relations structurally
        if data.edge_index.numel() > 0:
            deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
            deg_feature = torch.log(deg + 1).unsqueeze(0).expand(bs, -1).unsqueeze(-1).to(device)
            h = h.clone()
            h[:, :, 0] += deg_feature.squeeze(-1)

        # Identify which relation IDs appear in each query and mark them as 1 (if conditioning allows)
        if self.conditioning != "false":
            valid_mask   = batch.edge_embedding_indices != -1       # which edges are bound (global id of the edge) and wich arenot (-1)
            edge2graph   = batch.batch[batch.edge_index[0]]         # which batch element (query) does every edge belong to?
            graph_idx    = edge2graph[valid_mask]                   # batch index of the bound edges
            rel_idx      = batch.edge_embedding_indices[valid_mask] # global relation id of the bound edges
            h[graph_idx, rel_idx] = 1.0

        rel_edges  = data.edge_index        # (2, E)
        rel_types  = data.edge_type         # (E,)
        batch_edge_list  = []
        batch_type_list  = []
        # Since GAT etc. dont accept an explicit batch dimension, we need to reshape the batch_size different relation graphs into a large single graph. hence we need
        # to increase the edge index per batch, but leave the edge type the same.
        for b in range(bs):                 # identical to EntityGNN lines 72-84
            offset = b * num_nodes
            batch_edge_list.append(rel_edges + offset)
            batch_type_list.append(rel_types)
        edge_index = torch.cat(batch_edge_list, dim=1)   # (2, bs·E)
        edge_type  = torch.cat(batch_type_list, dim=0)   # (bs·E,)

        edge_attr  = self.rel_emb(edge_type)    


        x = h.view(bs * num_nodes, -1)

        # Message passing
        for i, layer in enumerate(self.layers):
            x_in = x # Save input for residual
            x   = layer(x, edge_index, edge_attr)
            if self.norms:
                 x = self.norms[i](x)
            x = self.act(x)
            
            # --- FIX 2: Residual Connection ---
            if x.shape == x_in.shape:
                x = x + x_in

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        out = x.view(bs, num_nodes, -1)
        return out



class FICE(nn.Module):

    def __init__(self, entity_message_func='rotate', relation_message_func='rotate', conditioning='all', layer_norm=True, query_gnn='TripleModel',
     vector_dim=128, n_embedding_layers=2):
        super(FICE, self).__init__()
        self.vector_dim = vector_dim
        # Use n_embedding_layers to create the list of hidden dimensions
        self.hidden_dims_relnet = self.hidden_dims_entitynet = [self.vector_dim] * n_embedding_layers
        
        # Store message functions and conditioning for later use
        self.entity_message_func = entity_message_func
        self.relation_message_func = relation_message_func
        self.conditioning = conditioning
        self.query_gnn_type = query_gnn


        if relation_message_func in ['rotate', 'distmult', 'transe']:
            self.relation_model = RelNBFNet(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_relnet,
                message_func=relation_message_func,
                aggregate_func='pna',
                short_cut=True,
                layer_norm=True,
                conditioning=conditioning
            )
        elif relation_message_func == 'GAT':
            self.relation_model = RelationGNN(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_relnet,
                num_relation=4,
                heads=4,
                layer_norm=layer_norm,
                conditioning=conditioning
            )
        else:
            raise ValueError(f"Invalid relation message function: {relation_message_func}")
 
        if entity_message_func in ['rotate', 'distmult', 'transe']:
            self.entity_model = EntityNBFNet(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_entitynet,
                message_func=entity_message_func,
                aggregate_func='sum',
                short_cut=True,
                layer_norm=True,
                conditioning=conditioning
            )
        elif entity_message_func in ['gine', 'GAT']:
            self.entity_model = EntityGNN(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_entitynet,
                message_func=entity_message_func,
                conditioning=conditioning,
                layer_norm=layer_norm
            )
        else:
            raise ValueError(f"Invalid entity message function: {entity_message_func}")

        # Dataset creates node/edge features of size (vector_dim + 3)
        if self.query_gnn_type == 'QueryCardinalityGNN':
            self.query_gnn = QueryCardinalityGNN(feat_dim=self.vector_dim + 3)
        elif self.query_gnn_type == 'TripleModel':
            self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3)
        else:
            raise ValueError(f"Invalid query GNN type: {self.query_gnn_type}")

    def forward(self, data, relation_graph, batch, graph_size, plot_embeddings=False, epoch=0):

        # All inputs should already be on the correct device
        device = batch.x.device
        if data is not None:
            assert data.edge_index.device == device

        assert relation_graph.edge_index.device == device
        assert batch.x.device == device
        assert graph_size.device == device

        # Recreate query_features with correct vector_dim
        # Keep the first 3 dimensions (counts) and create new zero placeholders for the embedding part
        query_counts = batch.x[:, :3]
        query_features = torch.zeros(batch.x.size(0), self.vector_dim + 3, device=device)
        query_features[:, :3] = query_counts
        
        # Same for edge_attr
        edge_counts = batch.edge_attr[:, :3]
        edge_attr = torch.zeros(batch.edge_attr.size(0), self.vector_dim + 3, device=device)
        edge_attr[:, :3] = edge_counts

        # IMPORTANT: Update the batch object immediately so subsequent code uses correct shapes
        batch.x = query_features
        batch.edge_attr = edge_attr

        embedding_indices = batch.node_embedding_indices
        edge_embedding_indices = batch.edge_embedding_indices

        # Provide shape [1, K] to seed K relations at once
        # No need to call .to(device) - already on correct device
        relation_representations = self.relation_model(batch=batch, data=relation_graph) # shape (batch_size, num_relations, embedding_dim)
        assert relation_representations.shape[0] == batch.batch_size, "Relation representations shape does not match batch size"

        if data is not None:
            node_representations = self.entity_model(batch=batch, data=data, relation_representations=relation_representations)
            assert node_representations.shape[0] == batch.batch_size, "Node representations shape does not match batch size"


            # If there are no node representations, we cannot map anything, so we keep the original features
            bound_nodes = embedding_indices >= 0
            graph_indices = batch.batch
            # HACK: not the most efficient
            node_embeddings = node_representations[graph_indices, embedding_indices]

            # Selectively replace last x dims for non-variable nodes and keeping  first 3 dims for all
            last = torch.where(bound_nodes.unsqueeze(-1),          
                                node_embeddings,
                                query_features[:, 3:])         

            query_features = torch.cat([query_features[:, :3], last], dim=1)
            batch.x = query_features


        # Handle edge embeddings similar to node embeddings
        # Get relation representations for edges
        bound_edges = (edge_embedding_indices >= 0)  # bools indicating which edges need relation embeddings
        edge2graph = batch.batch[batch.edge_index[0]] # which batch element (query) does every edge belong to
        edge_embeddings = relation_representations[edge2graph, edge_embedding_indices]

        if plot_embeddings:
            analyze_vector_similarity(node_embeddings, "Node_Embeddings Epoch " + str(epoch))
            analyze_vector_similarity(edge_embeddings, "Edge_Embeddings Epoch " + str(epoch))

        # Selectively replace last 64 dims for edges with actual relations and keeping first 3 dims for all
        edge_last = torch.where(bound_edges.unsqueeze(-1),
                                  edge_embeddings,
                                  edge_attr[:, 3:])

        edge_features = torch.cat([edge_attr[:, :3], edge_last], dim=1)
        batch.edge_attr = edge_features

        # TODO:
        
        if self.query_gnn_type == 'QueryCardinalityGNN':
            # update Counts for log:
            
            # Process node features (batch.x)
            unbound_nodes = batch.node_embedding_indices < 0
            bound_nodes = ~unbound_nodes
            
            # For unbound nodes: set first 3 dims to log(graph_size)
            log_graph_size = torch.log(graph_size.float())
            batch.x[unbound_nodes, :3] = log_graph_size
            
            # For bound nodes: replace 0s with 1s in first 3 dims, then apply log
            first_3_dims = batch.x[bound_nodes, :3]
            first_3_dims = torch.where(first_3_dims == 0, torch.ones_like(first_3_dims), first_3_dims)
            batch.x[bound_nodes, :3] = torch.log(first_3_dims)
            
            # Process edge features (batch.edge_attr)
            unbound_edges = batch.edge_embedding_indices < 0
            bound_edges = ~unbound_edges
            
            # For unbound edges: set first 3 dims to log(graph_size)
            batch.edge_attr[unbound_edges, :3] = log_graph_size
            
            # For bound edges: replace 0s with 1s in first 3 dims, then apply log
            edge_first_3_dims = batch.edge_attr[bound_edges, :3]
            edge_first_3_dims = torch.where(edge_first_3_dims == 0, torch.ones_like(edge_first_3_dims), edge_first_3_dims)
            batch.edge_attr[bound_edges, :3] = torch.log(edge_first_3_dims)
            
            log_cardinality_pred = self.query_gnn(batch=batch, N=graph_size)
        elif self.query_gnn_type == 'TripleModel':
            log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch)
        else:
            raise ValueError(f"Invalid query GNN type: {self.query_gnn_type}")

        return log_cardinality_pred




class FiceBinary(nn.Module):
    """Ablation model that encodes only the IDs actually present in the
    *query batch* instead of generating embeddings for the entire entity /
    relation universe.  ``data`` and ``relation_graph`` inputs are therefore
    optional (they can be ``None``).
    """

    def __init__(self, vector_dim: int = 128):
        super().__init__()
        self.vector_dim = vector_dim
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3)

    # --- binary encoding helper ------------------------------------------------
    def _binary_encode(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        ids_long = ids.to(torch.long)
        bits = (ids_long.unsqueeze(-1) >> torch.arange(self.vector_dim, device=device)) & 1
        return bits.float()

    # --- forward ----------------------------------------------------------------
    def forward(self, data, relation_graph, batch, graph_size):
        """Same signature as *FICE* for drop-in use, but `data` and
        `relation_graph` may be ``None``.  Only the indices stored in
        ``batch`` are required.
        """
        device = batch.x.device

        # ------------------------------------------------------------------
        # Entity embeddings – encode only *used* entity IDs
        # ------------------------------------------------------------------
        embedding_indices = batch.node_embedding_indices  # (N_q,)
        bound_nodes_mask = embedding_indices >= 0
        if bound_nodes_mask.any():
            used_node_ids = embedding_indices[bound_nodes_mask]
            unique_node_ids, inverse = torch.unique(used_node_ids, return_inverse=True)
            node_emb_unique = self._binary_encode(unique_node_ids)  # (U, D)
            node_emb_flat = node_emb_unique[inverse]  # (num_bound_nodes, D)
        else:
            node_emb_flat = None  # no bound nodes in this query batch

        # Start with original features
        query_features = batch.x
        if node_emb_flat is not None:
            last = query_features[:, 3:].clone()  # copy to preserve autograd graph
            last[bound_nodes_mask] = node_emb_flat
            query_features = torch.cat([query_features[:, :3], last], dim=1)
            batch.x = query_features

        # ------------------------------------------------------------------
        # Relation embeddings – encode only *used* relation IDs for edges
        # ------------------------------------------------------------------
        edge_embedding_indices = batch.edge_embedding_indices  # (E_q,)
        bound_edges_mask = edge_embedding_indices >= 0
        if bound_edges_mask.any():
            used_rel_ids = edge_embedding_indices[bound_edges_mask]
            unique_rel_ids, inverse_e = torch.unique(used_rel_ids, return_inverse=True)
            rel_emb_unique = self._binary_encode(unique_rel_ids)  # (R_u, D)
            rel_emb_flat = rel_emb_unique[inverse_e]
        else:
            rel_emb_flat = None

        edge_attr = batch.edge_attr
        if rel_emb_flat is not None:
            edge_last = edge_attr[:, 3:].clone()
            edge_last[bound_edges_mask] = rel_emb_flat
            edge_attr = torch.cat([edge_attr[:, :3], edge_last], dim=1)
            batch.edge_attr = edge_attr

        # ------------------------------------------------------------------
        # Cardinality prediction through query GNN
        # ------------------------------------------------------------------
        log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch)
        return log_cardinality_pred



class FiceIndependent(nn.Module):
    """Ablation model calculates cardinalities based on the independent assumption
    using only the counts.
    """

    def __init__(self, vector_dim: int = 128):
        super().__init__()
        self.vector_dim = vector_dim
        self.query_gnn = IndependentQueryGNN(feat_dim=self.vector_dim + 3)


    # --- forward ----------------------------------------------------------------
    def forward(self, data, relation_graph, batch, graph_size):


        log_cardinality_pred = self.query_gnn(batch=batch, N=graph_size)
        return log_cardinality_pred
