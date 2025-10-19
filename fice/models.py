import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import NeighborLoader

from easydict import EasyDict

from fice.gnce.models import QueryGNN, TripleModel
from fice.ultra.models import RelNBFNet, EntityNBFNet, map_global_to_neighborhood


class FICE(nn.Module):

    def __init__(self):
        super(FICE, self).__init__()
        self.vector_dim = 128
        self.hidden_dims_relnet = self.hidden_dims_entitynet = [self.vector_dim] * 6  # 6 layers with 64 hidden units each


        # Initialize models directly without config dictionaries
        self.relation_model = RelNBFNet(
            input_dim=self.vector_dim,
            hidden_dims=self.hidden_dims_relnet,
            message_func='distmult',
            aggregate_func='sum',
            short_cut=True,
            layer_norm=True
        )
        
        self.entity_model = EntityNBFNet(
            input_dim=self.vector_dim,
            hidden_dims=self.hidden_dims_entitynet,
            message_func='distmult',
            aggregate_func='sum',
            short_cut=True,
            layer_norm=True
        )
        # Dataset creates node/edge features of size (vector_dim + 3)
        # self.query_gnn = QueryGNN(feat_dim=self.vector_dim + 3)
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3)

    def forward(self, data, batch):
        # Get model device from model parameters
        device = next(self.parameters()).device

        entity_graph = data
        
        # Move only the batch to GPU (data/graph is already on GPU)
        batch = batch.to(device)
        query_features = batch.x.to(device)
        edge_attr = batch.edge_attr.to(device)
        embedding_indices = batch.node_embedding_indices.to(device)
        edge_embedding_indices = batch.edge_embedding_indices.to(device)

        # Provide shape [1, K] to seed K relations at once
        relation_representations = self.relation_model(batch=batch, data=data.relation_graph)
        assert relation_representations.shape[0] == batch.batch_size, "Relation representations shape does not match batch size"

        input_nodes = batch.node_embedding_indices[batch.node_embedding_indices >= 0].unique()
        neighbor_loader = NeighborLoader(
            data=data,
            num_neighbors=[-1] * 2,
            batch_size=input_nodes.size(0),
            shuffle=True,
            disjoint=False,
            input_nodes=input_nodes
        )
        entity_batch = next(iter(neighbor_loader)).to(device) 

        # TODO: Always passes the large relation graph with it even though not needed
        node_representations = self.entity_model(batch=batch, data=entity_batch, relation_representations=relation_representations)
        assert node_representations.shape[0] == batch.batch_size, "Node representations shape does not match batch size"


        # Merging ULTRA embeddings with precomputed query features
        bound_nodes = embedding_indices >= 0
        graph_indices = batch.batch
        # HACK: not the most efficient
        neighborhood_graph_mapped_indices = map_global_to_neighborhood(entity_batch.n_id, embedding_indices)
        node_embeddings = node_representations[graph_indices, neighborhood_graph_mapped_indices]

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

        graph_size = torch.tensor([data.num_edges], device=data.edge_index.device)
        log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch) 

        return log_cardinality_pred



