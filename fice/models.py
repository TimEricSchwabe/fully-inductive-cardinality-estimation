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



class FICE(nn.Module):

    def __init__(self, entity_message_func='rotate', relation_message_func='rotate', conditioning='all'):
        super(FICE, self).__init__()
        self.vector_dim = 128
        self.hidden_dims_relnet = self.hidden_dims_entitynet = [self.vector_dim] * 2 # n layers with 64 hidden units each
        
        # Store message functions and conditioning for later use
        self.entity_message_func = entity_message_func
        self.relation_message_func = relation_message_func
        self.conditioning = conditioning

        # Initialize models directly without config dictionaries
        self.relation_model = RelNBFNet(
            input_dim=self.vector_dim,
            hidden_dims=self.hidden_dims_relnet,
            message_func=relation_message_func,
            aggregate_func='pna',
            short_cut=True,
            layer_norm=True,
            conditioning=conditioning
        )
 
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
        else:
            self.entity_model = EntityGNN(
                input_dim=self.vector_dim,
                hidden_dims=[128, 128],
                message_func=entity_message_func,
                conditioning=conditioning
            )

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



