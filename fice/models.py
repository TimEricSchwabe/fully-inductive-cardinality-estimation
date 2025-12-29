from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.utils import degree
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from fice.gnce.models import TripleModel, QueryCardinalityGNN, IndependentQueryGNN, QueryTransformer, SimpleSumGNN


class EntityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, message_func='gine', conditioning='all', layer_norm=True, skip_connection=True, **kwargs):
        super(EntityGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.message_func = message_func
        self.conditioning = conditioning
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm

        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
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
            [nn.LayerNorm(d) for d in hidden_dims] if self.enable_layer_norm else []
        )

        self.laplacian_k = 0
        if self.laplacian_k > 0:
            self.le_pw_transform = AddLaplacianEigenvectorPE(k=self.laplacian_k)


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
            x = torch.randn(batch_size, data.num_nodes, self.input_dim, device=device) # todo !!
            #x = torch.ones(batch_size, data.num_nodes, self.input_dim, device=device) * 0.1
        elif self.conditioning == "seed only":
            x = torch.zeros(batch_size, data.num_nodes, self.input_dim, device=device)
        else:
            raise ValueError(f"Invalid conditioning value: {self.conditioning}")

        # --- FIX 1: Symmetry Breaking with Degrees ---
        # Inject log(degree) into the first dimension of the features to distinguish nodes structurally
        if data.edge_index.numel() > 0:
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
            # Normalize/Log degree to keep it in a reasonable range
            deg_feature = torch.log1p(deg + 1).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1).to(device)
            # Add to the first dimension of x (broadcast over batch)
            x = x.clone()
            x[:, :, 0] += deg_feature.squeeze(-1)

        if self.laplacian_k > 0:
            try:
                data = self.le_pw_transform(data)
                # Extract PE: shape (num_nodes, walk_length)
                rw_pe = data.laplacian_eigenvector_pe.to(device)
                
                # Expand to batch size: (batch_size, num_nodes, walk_length)
                rw_pe_expanded = rw_pe.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Add to x, skipping the 0-th dimension used by degree
                # Ensure we don't exceed embedding dimensions
                dim_limit = min(self.laplacian_k, x.size(-1) - 1)
                x[:, :, 1:1+dim_limit] = rw_pe_expanded[:, :, :dim_limit]
            except Exception:
                print(f"Laplacian Eigenvector PE computation failed")
                pass # Fallback to random features if Eigendecomposition fails

        x = x.view(-1, x.shape[-1])  # (batch_size * num_nodes, input_dim)

        # Set input of nodes appearing in query based on conditioning (labelling trick)
        if self.conditioning != "false" and batch is not None:
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

            if self.enable_layer_norm:
                 x = self.norms[i](x)

            if self.skip_connection:
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
        message_func: str = 'gine',
        conditioning: str = "all",      
        layer_norm: bool = False,
        skip_connection: bool = True,
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
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        self.message_func = message_func

        # 4 trainable relation embeddings
        self.rel_emb = nn.Embedding(num_relation, input_dim)

        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            if self.message_func == 'gine':
                self.layers.append(
                    GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            else:
                # GATv2Conv
                self.layers.append(
                    GATv2Conv(dims[i], dims[i + 1], heads=1, concat=False, edge_dim=input_dim)
                )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if self.enable_layer_norm else []
        )

        self.laplacian_k = 0
        if self.laplacian_k > 0:
            self.le_pw_transform = AddLaplacianEigenvectorPE(k=self.laplacian_k)

    def forward(self, data, batch=None):
        """
        data  : PyG Data of the *relation* graph
        batch : batched query graph (only batch_size &  edge_embedding_indices  matter)
        """
        assert len(data.edge_index) % 2 == 0, "Bipartit graph must have even number of edges (directed pairs)."

        if batch is None:
            bs = 1
            device = data.edge_index.device
        else:
            bs = batch.batch_size
            device = batch.edge_index.device
        num_nodes = data.num_nodes

        if self.conditioning == "false":
            # random features 
            h = torch.randn(bs, num_nodes, self.rel_emb.embedding_dim, device=device) # todo !!
            #h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1
        elif self.conditioning == "seed only":
            h = torch.zeros(bs, num_nodes, self.rel_emb.embedding_dim, device=device)
        else:  # "all"
            h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1

        # --- FIX: Symmetry Breaking with Degrees for Relation Graph ---
        # Inject log(degree) into the first dimension to distinguish relations structurally
        if data.edge_index.numel() > 0:
            deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
            deg_feature = torch.log1p(deg + 1).unsqueeze(0).expand(bs, -1).unsqueeze(-1).to(device)
            h = h.clone()
            h[:, :, 0] += deg_feature.squeeze(-1)

        if self.laplacian_k > 0:
            try:
                data = self.le_pw_transform(data)
                # Extract PE: shape (num_nodes, walk_length)
                rw_pe = data.laplacian_eigenvector_pe.to(device)
                
                # Expand to batch size: (batch_size, num_nodes, walk_length)
                rw_pe_expanded = rw_pe.unsqueeze(0).expand(bs, -1, -1)

                # Add to h, skipping the 0-th dimension used by degree
                # Ensure we don't exceed embedding dimensions
                dim_limit = min(self.laplacian_k, h.size(-1) - 1)
                h[:, :, 1:1+dim_limit] = rw_pe_expanded[:, :, :dim_limit]
            except Exception as e:
                pass # Fallback to random features if Eigendecomposition fails

        # Identify which relation IDs appear in each query and mark them as 1 (if conditioning allows)
        if self.conditioning != "false" and batch is not None:
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
            x   = layer(x, edge_index, edge_attr=edge_attr)  # edge_attr is not used in GINEConv if not provided

            if self.enable_layer_norm:
                 x = self.norms[i](x)
           
            # --- FIX 2: Residual Connection ---
            if self.skip_connection:
                x = x + x_in

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        out = x.view(bs, num_nodes, -1)
        return out


class BipartiteGraphGNN(nn.Module):
    """
        A unified embedding model that produces embeddings for both entities and relations
        on the bipartite (entity-relation) graph of the KG.
        (Renamed from `EmbeddingGNNv1`.)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple = (128, 128),
        conditioning: str = "false",      
        layer_norm: bool = False,
        skip_connection: bool = True,
    ):
        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(), 
                nn.Linear(out_dim, out_dim),
            )

        super().__init__()
        self.conditioning = conditioning     
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm

        # Add projection for edge attributes (Count, Direction) -> input_dim
        self.edge_encoder = nn.Linear(2, input_dim)


        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            self.layers.append(
                GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
            )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if self.enable_layer_norm else []
        )

        self.laplacian_k = 0
        if self.laplacian_k > 0:
            self.le_pw_transform = AddLaplacianEigenvectorPE(k=self.laplacian_k)

    def forward(self, data, batch=None):
        """
        data  : PyG Data (sample) of the bipartite graph of the KG
        batch : batched query graph (only batch_size &  edge_embedding_indices  matter)
        """
        assert len(data.edge_index) % 2 == 0, "Bipartit graph must have even number of edges (directed pairs)."

        if batch is None:
            bs = 1
            device = data.edge_index.device
        else:
            bs = batch.batch_size
            device = batch.edge_index.device
        num_nodes = data.num_nodes


        if   self.conditioning == "false":
            # random features, random gauss wegen randn
            # h = torch.zeros(bs, num_nodes, self.input_dim, device=device)
            #h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1

            # TODO: extra condition self.conditioning == "random_relations"
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            bound_nodes = data.mapped_entity_indices[data.mapped_entity_indices >= 0]
            mask[bound_nodes] = True

            h = torch.zeros((num_nodes, self.input_dim), device=device)
            h[mask] = torch.randn((bound_nodes.size(0), self.input_dim), device=device)
            h = h.unsqueeze(0)  # (1, num_nodes, input_dim)
            h = h.repeat(bs, 1, 1)


        elif self.conditioning == "seed only":
            h = torch.zeros(bs, num_nodes, self.input_dim, device=device)
        else:  # "all"
            h = torch.ones(bs, num_nodes, self.input_dim, device=device) * 0.1

        # --- FIX: Symmetry Breaking with Degrees for Relation Graph ---
        # Inject log(degree) into the first dimension to distinguish relations structurally
        if data.edge_index.numel() > 0:
            deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
            deg_feature = torch.log1p(deg + 1).unsqueeze(0).expand(bs, -1).unsqueeze(-1).to(device)
            h = h.clone()
            h[:, :, 0] += deg_feature.squeeze(-1)

        if self.laplacian_k > 0:
            try:
                data = self.le_pw_transform(data)
                # Extract PE: shape (num_nodes, walk_length)
                rw_pe = data.laplacian_eigenvector_pe.to(device)
                
                # Expand to batch size: (batch_size, num_nodes, walk_length)
                rw_pe_expanded = rw_pe.unsqueeze(0).expand(bs, -1, -1)

                # Add to h, skipping the 0-th dimension used by degree
                # Ensure we don't exceed embedding dimensions
                dim_limit = min(self.laplacian_k, h.size(-1) - 1)
                h[:, :, 1:1+dim_limit] = rw_pe_expanded[:, :, :dim_limit]
            except Exception as e:
                pass # Fallback to random features if Eigendecomposition fails

        # Identify which relation IDs appear in each query and mark them as 1 (if conditioning allows)
        if self.conditioning != "false" and batch is not None:
            valid_mask   = batch.edge_embedding_indices != -1       # which edges are bound (global id of the edge) and wich arenot (-1)
            edge2graph   = batch.batch[batch.edge_index[0]]         # which batch element (query) does every edge belong to?
            graph_idx    = edge2graph[valid_mask]                   # batch index of the bound edges
            rel_idx      = batch.edge_embedding_indices[valid_mask] # global relation id of the bound edges
            h[graph_idx, rel_idx] = 1.0

        # TODO: Write without for loops (vectorize it)
        rel_edges  = data.edge_index        # (2, E)
        batch_edge_list  = []
        # Since GAT etc. dont accept an explicit batch dimension, we need to reshape the batch_size different relation graphs into a large single graph. hence we need
        # to increase the edge index per batch, but leave the edge type the same.
        for b in range(bs):                 # identical to EntityGNN lines 72-84
            offset = b * num_nodes
            batch_edge_list.append(rel_edges + offset)
        edge_index = torch.cat(batch_edge_list, dim=1)   # (2, bs·E)

        # Encode edge attributes (Count, Direction)
        # 1. Replicate edge attributes for batch
        rel_edge_attr = data.edge_attr.to(torch.float32) # Shape [E, 2]
        batch_edge_attr_list = []
        for b in range(bs):
            batch_edge_attr_list.append(rel_edge_attr)
        edge_attr_stacked = torch.cat(batch_edge_attr_list, dim=0) # Shape [bs*E, 2]
        
        # 2. Project to embedding dimension
        edge_attr = self.edge_encoder(edge_attr_stacked) # Shape [bs*E, input_dim]

        x = h.view(bs * num_nodes, -1)

        # Message passing
        for i, layer in enumerate(self.layers):
            x_in = x # Save input for residual
            x   = layer(x, edge_index, edge_attr = edge_attr)  # edge_attr is not used in GINEConv if not provided

            if self.enable_layer_norm:
                 x = self.norms[i](x)
           
            # --- FIX 2: Residual Connection ---
            if x.shape == x_in.shape and self.skip_connection:
                x = x + x_in

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        # and numnodes is total number of nodes (entities + relations)
        out = x.view(bs, num_nodes, -1)
        return out


class FICE(nn.Module):

    def __init__(self, 
                 vector_dim=128, 
                 n_embedding_layers=2, 
                 conditioning='all',
                 entity_graph_mode: Literal['bipartite', 'KG'] = "KG", 
                 relation_graph_mode: Literal['bipartite', 'RelationGraph'] = "RelationGraph",
                 # Entity GNN params
                 entity_message_func='gine',
                 entity_layer_norm=False,
                 entity_skip_connection=True,
                 # Relation GNN params
                 relation_message_func='gine',
                 relation_layer_norm=False,
                 relation_skip_connection=True,
                 # Query GNN params
                 query_gnn_type='TripleModel',
                 query_layer_norm=True,
                 query_skip_connection=False,
                 query_n_layers=6,
                 query_jk_mode='cat'
                 ):
        super(FICE, self).__init__()
        self.vector_dim = vector_dim
        self.n_layers = query_n_layers
        self.skip_connection = entity_skip_connection # Default legacy behavior, though now we use specific ones
        self.use_unified_bipartite_graph = (entity_graph_mode == "bipartite" and relation_graph_mode == "bipartite")
        # Use n_embedding_layers to create the list of hidden dimensions
        self.hidden_dims_relnet = self.hidden_dims_entitynet = [self.vector_dim] * n_embedding_layers
        
        self.entity_message_func = entity_message_func
        self.relation_message_func = relation_message_func
        self.conditioning = conditioning
        self.query_gnn_type = query_gnn_type
        self.relation_graph_mode = relation_graph_mode
        self.entity_graph_mode = entity_graph_mode


        if self.use_unified_bipartite_graph:
            self.embedding_model = BipartiteGraphGNN(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_relnet,
                layer_norm=relation_layer_norm, # Use relation norm as default for unified
                skip_connection=relation_skip_connection,
                conditioning=conditioning,
            )

        else:
            
            if relation_message_func in ['GAT', 'gine'] and self.relation_graph_mode == "RelationGraph":
                self.relation_model = RelationGNN(
                    input_dim=self.vector_dim,
                    hidden_dims=self.hidden_dims_relnet,
                    num_relation=4,
                    message_func=relation_message_func,
                    layer_norm=relation_layer_norm,
                    skip_connection=relation_skip_connection,
                    conditioning=conditioning
                )
            elif self.relation_graph_mode == "bipartite":
                # If using bipartite graph for relations, we use the unified model
                self.relation_model = BipartiteGraphGNN(
                    input_dim=self.vector_dim,
                    hidden_dims=self.hidden_dims_relnet,
                    layer_norm=relation_layer_norm,
                    skip_connection=relation_skip_connection,
                    conditioning=conditioning,
                )
            else:
                raise ValueError(f"Invalid relation message function: {relation_message_func}")
    
            
            if entity_message_func in ['gine', 'GAT'] and self.entity_graph_mode == "KG":
                self.entity_model = EntityGNN(
                    input_dim=self.vector_dim,
                    hidden_dims=self.hidden_dims_entitynet,
                    message_func=entity_message_func,
                    conditioning=conditioning,
                    layer_norm=entity_layer_norm,
                    skip_connection=entity_skip_connection
                )
            elif self.entity_graph_mode == "bipartite" and self.relation_graph_mode == "RelationGraph":
                raise ValueError(f"Entity graph bipartite but relation graph standard is not supported.")
            else:
                raise ValueError(f"Invalid entity message function: {entity_message_func}")

        if self.query_gnn_type == 'QueryCardinalityGNN':
            self.query_gnn = QueryCardinalityGNN(feat_dim=self.vector_dim + 3, num_layers=3)
        elif self.query_gnn_type == 'TripleModel':
            self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3, layer_norm=query_layer_norm, n_layers=self.n_layers, skip_connection=query_skip_connection, jk_mode=query_jk_mode)
        elif self.query_gnn_type == 'QueryTransformer':
            self.query_gnn = QueryTransformer(feat_dim=self.vector_dim + 3)
        elif self.query_gnn_type == 'SimpleSumGNN':
            self.query_gnn = SimpleSumGNN(feat_dim=self.vector_dim + 3)
        else:
            raise ValueError(f"Invalid query GNN type: {self.query_gnn_type}")

    def forward(self, data, relation_graph, batch, graph_size, epoch=0):
        device = batch.x.device
        
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

        # -----------------------------------------------------------
        # LOGIC SPLIT: 
        # Case A: We use a single BIPARTITE graph for both entities and relations.
        # Case B: We use separate graphs (KG for entities, RelationGraph for relations).
        # -----------------------------------------------------------
        
        if self.use_unified_bipartite_graph:
            # Check integrity just in case: data should be the bipartite graph
            if data is None and relation_graph is not None:
                 # If train.py passes data=None but relation_graph is set, assume relation_graph is the one
                 data = relation_graph
            
            # --- SINGLE GNN PASS ---
            # Run relation_model on the bipartite graph.
            # It returns embeddings for ALL nodes (Entities + Relations).
            # shape: (batch_size, num_bipartite_nodes, embedding_dim)
            all_embeddings = self.embedding_model(batch=batch, data=data)

            # 1. Extract Entity Embeddings
            bound_nodes = embedding_indices >= 0
            graph_indices = batch.batch
            
            node_embeddings_gathered = torch.zeros(embedding_indices.size(0), self.vector_dim, device=device)
            if bound_nodes.any():
                valid_graph_idx = graph_indices[bound_nodes]
                valid_node_idx  = embedding_indices[bound_nodes]
                node_embeddings_gathered[bound_nodes] = all_embeddings[valid_graph_idx, valid_node_idx]

            # Merge with query features
            last = torch.where(bound_nodes.unsqueeze(-1),          
                                node_embeddings_gathered,
                                query_features[:, 3:])         
            query_features = torch.cat([query_features[:, :3], last], dim=1)
            batch.x = query_features

            # 2. Extract Relation Embeddings (for edges)
            bound_edges = (edge_embedding_indices >= 0)
            edge2graph = batch.batch[batch.edge_index[0]]
            
            edge_embeddings_gathered = torch.zeros(edge_embedding_indices.size(0), self.vector_dim, device=device)
            
            if bound_edges.any():
                valid_edge_graph_idx = edge2graph[bound_edges]
                valid_edge_node_idx  = edge_embedding_indices[bound_edges]
                edge_embeddings_gathered[bound_edges] = all_embeddings[valid_edge_graph_idx, valid_edge_node_idx]

            edge_embeddings = edge_embeddings_gathered # For plotting variable name consistency
            node_embeddings = node_embeddings_gathered # For plotting variable name consistency

            # Merge with edge attributes
            edge_last = torch.where(bound_edges.unsqueeze(-1),
                                    edge_embeddings_gathered,
                                    edge_attr[:, 3:])
            edge_features = torch.cat([edge_attr[:, :3], edge_last], dim=1)
            batch.edge_attr = edge_features

        else:
            # --- ORIGINAL TWO-PASS LOGIC ---
            # Provide shape [1, K] to seed K relations at once
            # No need to call .to(device) - already on correct device
            relation_representations = self.relation_model(batch=batch, data=relation_graph) # shape (batch_size, num_relations, embedding_dim)

            if self.relation_graph_mode == "bipartite" and self.entity_graph_mode == "KG":
                # TODO: the returned relation_represtaitons contain both entity and relation embeddings stacked
                # We need to split them here
                # TODO: extract the entities embedding on their own
                # batch.edge_embedding_indices contains the global relation ids
                relation_indices = batch.edge_embedding_indices
                edge2graph = batch.batch[batch.edge_index[0]] # which batch element (query) does every edge belong to
                # maske funktioniert nicht so wie ich es will
                extracted_relation_representation = relation_representations[edge2graph, relation_indices]

                num_relations = relation_graph.original_num_relations # TODO: keine ahung wie stabil das ist, Antwort überhaupt nicht stabil, besser lösung finden
            

                new_relation_representations = torch.zeros((relation_representations.shape[0], num_relations, relation_representations.shape[2]), device=device) 
                new_relation_representations[edge2graph, batch.original_edge_embedding_indices] = extracted_relation_representation
                relation_representations = new_relation_representations

            if data is not None:
                node_representations = self.entity_model(batch=batch, data=data, relation_representations=relation_representations)
                
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
            if self.relation_graph_mode == "bipartite" and self.entity_graph_mode == "KG":
                # edge_embedding_indices contains the local bipartite graph indices
                edge_embedding_indices = batch.original_edge_embedding_indices

            edge_embeddings = relation_representations[edge2graph, edge_embedding_indices]

            # Selectively replace last 64 (128) dims for edges with actual relations and keeping first 3 dims for all
            edge_last = torch.where(bound_edges.unsqueeze(-1),
                                      edge_embeddings,
                                      edge_attr[:, 3:])

            edge_features = torch.cat([edge_attr[:, :3], edge_last], dim=1)
            batch.edge_attr = edge_features
        
        if self.query_gnn_type == 'QueryCardinalityGNN':
            unbound_nodes = batch.node_embedding_indices < 0
            bound_nodes = ~unbound_nodes
            
            # For unbound nodes: set first 3 dims to log(graph_size)
            log_graph_size = torch.log1p(graph_size.float())
            batch.x[unbound_nodes, :3] = log_graph_size
            
            # For bound nodes: replace 0s with 1s in first 3 dims, then apply log
            first_3_dims = batch.x[bound_nodes, :3]
            first_3_dims = torch.where(first_3_dims == 0, torch.ones_like(first_3_dims), first_3_dims)
            batch.x[bound_nodes, :3] = torch.log1p(first_3_dims)
            
            # Process edge features (batch.edge_attr)
            unbound_edges = batch.edge_embedding_indices < 0
            bound_edges = ~unbound_edges
            
            # For unbound edges: set first 3 dims to log(graph_size)
            batch.edge_attr[unbound_edges, :3] = log_graph_size
            
            # For bound edges: replace 0s with 1s in first 3 dims, then apply log
            edge_first_3_dims = batch.edge_attr[bound_edges, :3]
            edge_first_3_dims = torch.where(edge_first_3_dims == 0, torch.ones_like(edge_first_3_dims), edge_first_3_dims)
            batch.edge_attr[bound_edges, :3] = torch.log1p(edge_first_3_dims)
            
            log_cardinality_pred = self.query_gnn(batch=batch, N=graph_size)
        elif self.query_gnn_type == 'TripleModel':
            log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch)
        elif self.query_gnn_type == 'QueryTransformer':
            log_cardinality_pred = self.query_gnn(batch=batch, N=graph_size)
        elif self.query_gnn_type == 'SimpleSumGNN':
            log_cardinality_pred = self.query_gnn(batch=batch, N=graph_size)
        else:
            raise ValueError(f"Invalid query GNN type: {self.query_gnn_type}")

        return log_cardinality_pred


class FiceBinary(nn.Module):
    """Ablation model that encodes only the IDs actually present in the
    *query batch* instead of generating embeddings for the entire entity /
    relation universe.  ``data`` and ``relation_graph`` inputs are therefore
    optional (they can be ``None``).
    """

    def __init__(self, vector_dim: int = 128, n_layers: int = 6, skip_connection: bool = True):
        super().__init__()
        self.vector_dim = vector_dim
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3, n_layers=n_layers, skip_connection=skip_connection)

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
