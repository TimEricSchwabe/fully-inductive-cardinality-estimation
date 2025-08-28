from torch import nn

from easydict import EasyDict
from fice.ultra.models import RelNBFNet, EntityNBFNet
from fice.gnce.models import QueryGNN
import torch.nn.functional as F
import torch
import torch.nn as nn
# from fice.gnce.cardinality_estimation import GNCE


class FICE(nn.Module):

    def __init__(self, graph):
        super(FICE, self).__init__()
        # TODO: calculate occurences for each atom
        vector_dim = 64

        rel_model_cfg = EasyDict({
            'class': 'RelNBFNet',
            'input_dim': vector_dim,
            'hidden_dims': [vector_dim] * 6,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'short_cut': True,
            'layer_norm': True,
        })

        entity_model_cfg = EasyDict({
            'class': 'EntityNBFNet',
            'input_dim': vector_dim,
            'hidden_dims': [vector_dim] * 6,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'short_cut': True,
            'layer_norm': True,
        })



        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)
        # self.gnce_model = GNCE()
        self.query_gnn = QueryGNN(hidden_dim=vector_dim+3)

        
    # def forward(self, query):
    def forward(self, data, batch, query_features, query_edge_index, embedding_indices, edge_attr, edge_embedding_indices, graph_size):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        node_representations = self.entity_model(data, relation_representations, batch)

        if node_representations.dim() == 3 and node_representations.size(0) == 1:           
            node_representations = node_representations.squeeze(0)

        if relation_representations.dim() == 3 and relation_representations.size(0) == 1:           
            relation_representations = relation_representations.squeeze(0)

        # Merging ULTRA embeddings with precomputed query features
        bound_nodes = (embedding_indices >= 0) # bools indicating which nodes need ULTRA embeddings
        idx = embedding_indices.clamp(min=0) # clamping is necessary for F.embedding to work
        node_embeddings = F.embedding(idx, node_representations)

        # Selectively replace last 64 dims for non-variable nodes and keeping  first 3 dims for all
        last64 = torch.where(bound_nodes.unsqueeze(-1),          
                             node_embeddings,
                             query_features[:, 3:])         

        query_features = torch.cat([query_features[:, :3], last64], dim=1)

        # Handle edge embeddings similar to node embeddings
        # Get relation representations for edges
        bound_edges = (edge_embedding_indices >= 0)  # bools indicating which edges need relation embeddings
        edge_idx = edge_embedding_indices.clamp(min=0)  # clamping is necessary for F.embedding to work
        edge_embeddings = F.embedding(edge_idx, relation_representations)

        # Selectively replace last 64 dims for edges with actual relations and keeping first 3 dims for all
        edge_last64 = torch.where(bound_edges.unsqueeze(-1),
                                  edge_embeddings,
                                  edge_attr[:, 3:])

        edge_features = torch.cat([edge_attr[:, :3], edge_last64], dim=1)

        log_cardinality_pred = self.query_gnn(query_features, query_edge_index, edge_attr=edge_features, graph_size=graph_size) 


        # return score
        return log_cardinality_pred



