from torch import nn

from easydict import EasyDict
from fice.ultra.models import RelNBFNet, EntityNBFNet
from fice.gnce.models import QueryGNN
import torch.nn.functional as F
import torch
import torch.nn as nn


class FICE(nn.Module):

    def __init__(self, data):
        super(FICE, self).__init__()
        self.vector_dim = 64
        self.data = data
        self.device = data.edge_index.device


        # TODO: Temp, diese statischen Configs sollen irgendwann durch dynamische ersetzt werden
        rel_model_cfg = EasyDict({
            'class': 'RelNBFNet',
            'input_dim': self.vector_dim,
            'hidden_dims': [self.vector_dim] * 6,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'short_cut': True,
            'layer_norm': True,
            'data': self.data.relation_graph
        })

        entity_model_cfg = EasyDict({
            'class': 'EntityNBFNet',
            'input_dim': self.vector_dim,
            'hidden_dims': [self.vector_dim] * 6,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'short_cut': True,
            'layer_norm': True,
            'data': self.data
        })


        # TODO: total unleserlich, schöner machen. 
        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)
        self.query_gnn = QueryGNN(hidden_dim=self.vector_dim+3)

        
    def forward(self, batch):
        batch = batch.to(self.device)
        query_features = batch.x.to(self.device)
        edge_attr = batch.edge_attr.to(self.device)
        embedding_indices = batch.node_embedding_indices.to(self.device)
        edge_embedding_indices = batch.edge_embedding_indices.to(self.device)

        # Provide shape [1, K] to seed K relations at once
        relation_representations = self.relation_model(batch)
        assert relation_representations.shape[0] == batch.batch_size, "Relation representations shape does not match batch size"

        node_representations = self.entity_model(batch, relation_representations)
        assert node_representations.shape[0] == batch.batch_size, "Node representations shape does not match batch size"


        # Merging ULTRA embeddings with precomputed query features
        bound_nodes = embedding_indices >= 0
        graph_indices = batch.batch
        embedding_indices = embedding_indices
        # HACK: not the most efficient
        node_embeddings = node_representations[graph_indices, embedding_indices]

        # Selectively replace last 64 dims for non-variable nodes and keeping  first 3 dims for all
        last64 = torch.where(bound_nodes.unsqueeze(-1),          
                             node_embeddings,
                             query_features[:, 3:])         

        query_features = torch.cat([query_features[:, :3], last64], dim=1)
        batch.x = query_features


        # Handle edge embeddings similar to node embeddings
        # Get relation representations for edges
        bound_edges = (edge_embedding_indices >= 0)  # bools indicating which edges need relation embeddings
        edge2graph = batch.batch[batch.edge_index[0]]
        edge_embeddings = relation_representations[edge2graph, edge_embedding_indices]

        # Selectively replace last 64 dims for edges with actual relations and keeping first 3 dims for all
        edge_last64 = torch.where(bound_edges.unsqueeze(-1),
                                  edge_embeddings,
                                  edge_attr[:, 3:])

        edge_features = torch.cat([edge_attr[:, :3], edge_last64], dim=1)
        batch.edge_attr = edge_features

        graph_size = torch.tensor([self.data.num_edges], device=self.data.edge_index.device)
        log_cardinality_pred = self.query_gnn(graph_size=graph_size, batch=batch) 
        # log_cardinality_pred = self.query_gnn(query_features, query_edge_index, edge_attr=edge_features, graph_size=graph_size) 



        # return score
        return log_cardinality_pred



