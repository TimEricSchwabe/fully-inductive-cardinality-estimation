import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import GATv2Conv, GINEConv

from easydict import EasyDict

from fice.gnce.models import QueryGNN, TripleModel
from fice.ultra.models import RelNBFNet, EntityNBFNet


class EntityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_relation=1, message_func='gine', conditioning='all', **kwargs):
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

        if message_func == 'gine':
            self.layers = nn.ModuleList([
                GINEConv(gine_mlp(input_dim, hidden_dims[0]), edge_dim=input_dim, train_eps=False),
                GINEConv(gine_mlp(hidden_dims[0], hidden_dims[1]), edge_dim=input_dim, train_eps=False),
            ])
        else:
            self.layers = nn.ModuleList([
                GATv2Conv(input_dim, hidden_dims[0], heads=1, concat=False, edge_dim=input_dim),
                GATv2Conv(hidden_dims[0], hidden_dims[1], heads=1, concat=False, edge_dim=input_dim),
            ])


    def forward(self, data, batch, relation_representations=None):
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

        # Reshape once at the beginning: (batch_size, num_nodes, input_dim) -> (batch_size * num_nodes, input_dim)
        x = x.view(-1, x.shape[-1])  # (batch_size * num_nodes, input_dim)


        # Indices of the nodes that appear in the query batch
        node_indices = batch.node_embedding_indices 
        valid = node_indices != -1
        graph_idx = batch.batch[valid] # gives the graph index for each node
        node_pos = node_indices[valid] # gives the position of each node in the graph

        # Set input of nodes appearing in query based on conditioning (labelling trick)
        if self.conditioning != "false":
            rows = graph_idx * data.num_nodes + node_pos      # linear node index in flattened tensor
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

        all_edges = torch.repeat_interleave(torch.arange(batch_size, device=device), data.edge_index.size(1))

        edge_attr = relation_representations[all_edges, batched_edge_type]
        
        for layer in self.layers:
            x = layer(x, batched_edge_index, edge_attr=edge_attr)
            if self.message_func != 'gine':
                x = F.gelu(x)
        
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
        super().__init__()
        self.num_relation = num_relation
        self.conditioning = conditioning     

        # 4 trainable relation embeddings
        self.rel_emb = nn.Embedding(num_relation, input_dim)

        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            self.layers.append(
                GATv2Conv(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    heads=heads,
                    concat=False,               
                    edge_dim=input_dim,         # relation embedding as edge_attr
                )
            )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if layer_norm else []
        )
        self.act = nn.GELU()

    def forward(self, data, batch):
        """
        data  : PyG Data of the *relation* graph
        batch : batched query graph (only batch_size &  edge_embedding_indices  matter)
        """
        bs        = batch.batch_size
        num_nodes = data.num_nodes
        device    = data.edge_index.device


        if   self.conditioning == "false":
            h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1
        elif self.conditioning == "seed only":
            h = torch.zeros(bs, num_nodes, self.rel_emb.embedding_dim, device=device)
        else:  # "all"
            h = torch.ones(bs, num_nodes, self.rel_emb.embedding_dim, device=device) * 0.1

        # Identify which relation IDs appear in each query and mark them as 1 (if conditioning allows)
        if self.conditioning != "false":
            valid_mask   = batch.edge_embedding_indices != -1
            edge2graph   = batch.batch[batch.edge_index[0]]         # which query does every edge belong to?
            graph_idx    = edge2graph[valid_mask]
            rel_idx      = batch.edge_embedding_indices[valid_mask] # local relation-ID inside relation graph
            h[graph_idx, rel_idx] = 1.0

        rel_edges  = data.edge_index        # (2, E)
        rel_types  = data.edge_type         # (E,)
        batch_edge_list  = []
        batch_type_list  = []
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
            x   = layer(x, edge_index, edge_attr)
            if self.norms: x = self.norms[i](x)
            x = self.act(x)

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        out = x.view(bs, num_nodes, -1)
        return out



class FICE(nn.Module):

    def __init__(self, entity_message_func='rotate', relation_message_func='rotate', conditioning='all'):
        super(FICE, self).__init__()
        self.vector_dim = 128
        self.hidden_dims_relnet = self.hidden_dims_entitynet = [self.vector_dim] * 2 # n layers with 64 hidden units each
        
        # Store message functions and conditioning for later use
        self.entity_message_func = entity_message_func
        self.relation_message_func = relation_message_func
        self.conditioning = conditioning


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
                layer_norm=False,
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
                hidden_dims=[128, 128],
                message_func=entity_message_func,
                conditioning=conditioning
            )
        else:
            raise ValueError(f"Invalid entity message function: {entity_message_func}")

        # Dataset creates node/edge features of size (vector_dim + 3)
        # self.query_gnn = QueryGNN(feat_dim=self.vector_dim + 3)
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3)

    def forward(self, data, relation_graph, batch, graph_size):

        # All inputs should already be on the correct device
        device = batch.x.device
        if data is not None:
            assert data.edge_index.device == device

        assert relation_graph.edge_index.device == device
        assert batch.x.device == device
        assert graph_size.device == device

        query_features = batch.x
        edge_attr = batch.edge_attr
        embedding_indices = batch.node_embedding_indices
        edge_embedding_indices = batch.edge_embedding_indices

        # Provide shape [1, K] to seed K relations at once
        # No need to call .to(device) - already on correct device
        relation_representations = self.relation_model(batch=batch, data=relation_graph)
        assert relation_representations.shape[0] == batch.batch_size, "Relation representations shape does not match batch size"

        # TODO: Always passes the large relation graph with it (in data-obejct) even though not needed
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
        edge2graph = batch.batch[batch.edge_index[0]]
        edge_embeddings = relation_representations[edge2graph, edge_embedding_indices]

        # Selectively replace last 64 dims for edges with actual relations and keeping first 3 dims for all
        edge_last = torch.where(bound_edges.unsqueeze(-1),
                                  edge_embeddings,
                                  edge_attr[:, 3:])

        edge_features = torch.cat([edge_attr[:, :3], edge_last], dim=1)
        batch.edge_attr = edge_features

        # TODO:

        log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch)

        return log_cardinality_pred




class FiceBinary(nn.Module):
    """Ablation model that replaces learned entity / relation encoders with
    simple *binary* encodings of the corresponding ID.

    The idea is to convert an integer id (e.g. 12 -> 1100) into a fixed-length
    binary vector (least significant bit first) and pad the remaining
    dimensions with zeros until the desired embedding size (``vector_dim``).
    """

    def __init__(self, vector_dim: int = 128):
        super().__init__()
        self.vector_dim = vector_dim
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3)


    def _binary_encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of integer IDs into *binary* vectors.

        Parameters
        ----------
        ids: torch.Tensor (any shape) of non-negative ints

        Returns
        -------
        torch.FloatTensor with shape ``ids.shape + (vector_dim,)`` containing
        0/1 entries (least significant bit first).
        """
        device = ids.device
        # Ensure long dtype for bit-wise operations
        ids_long = ids.to(torch.long)
        # Create a column vector of bit positions: 0 .. vector_dim-1
        bits = (ids_long.unsqueeze(-1) >> torch.arange(self.vector_dim, device=device)) & 1
        return bits.float()  # convert {0,1} -> {0.,1.}


    def forward(self, data, relation_graph, batch, graph_size):

        device = batch.x.device
        bs = batch.batch_size

        # Relation *node* encodings – shape: (bs, |R|, D)
        rel_ids = torch.arange(relation_graph.num_nodes, device=device)
        rel_emb_single = self._binary_encode(rel_ids)                    # (|R|, D)
        relation_representations = rel_emb_single.unsqueeze(0).repeat(bs, 1, 1)

        # Entity *node* encodings – shape: (bs, |V|, D)
        if data is not None:
            node_ids = torch.arange(data.num_nodes, device=device)
            node_emb_single = self._binary_encode(node_ids)              # (|V|, D)
            node_representations = node_emb_single.unsqueeze(0).repeat(bs, 1, 1)

            # Inject entity embeddings into the query graph
            query_features = batch.x                                    # (N_q, 3 + D)
            embedding_indices = batch.node_embedding_indices            # (N_q,)
            bound_nodes = embedding_indices >= 0

            # Replace invalid indices (-1) temporarily with 0 to allow gather
            safe_indices = embedding_indices.clone()
            safe_indices[~bound_nodes] = 0

            graph_indices = batch.batch                                 # (N_q,)
            node_embeddings = node_representations[graph_indices, safe_indices]  # (N_q, D)

            # Keep original features for variables, replace for bound nodes
            last = torch.where(bound_nodes.unsqueeze(-1),
                               node_embeddings,
                               query_features[:, 3:])
            query_features = torch.cat([query_features[:, :3], last], dim=1)
            batch.x = query_features

        # Inject relation embeddings for *edges*
        edge_attr = batch.edge_attr                                      # (E_q, 3 + D)
        edge_embedding_indices = batch.edge_embedding_indices            # (E_q,)
        bound_edges = edge_embedding_indices >= 0
        edge2graph = batch.batch[batch.edge_index[0]]                    # (E_q,)

        # As above: avoid negative indices when gathering
        safe_edge_idx = edge_embedding_indices.clone()
        safe_edge_idx[~bound_edges] = 0

        edge_embeddings = relation_representations[edge2graph, safe_edge_idx]    # (E_q, D)

        edge_last = torch.where(bound_edges.unsqueeze(-1),
                                edge_embeddings,
                                edge_attr[:, 3:])
        edge_features = torch.cat([edge_attr[:, :3], edge_last], dim=1)
        batch.edge_attr = edge_features

        # Pass through the query GNN for cardinality prediction
        log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch)
        return log_cardinality_pred



