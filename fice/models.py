from torch import nn

from easydict import EasyDict
from fice.ultra.models import RelNBFNet, EntityNBFNet
from fice.gnce.models import QueryGNN
import torch.nn.functional as F
import torch
import torch.nn as nn


class FICE(nn.Module):

    # TODO: __init__(self, graph)       Graph wird nur einmal am Anfang übergeben und danach in forward nur die queries
    def __init__(self):
        super(FICE, self).__init__()
        vector_dim = 64


        # TODO: Temp, diese statischen Configs sollen irgendwann durch dynamische ersetzt werden
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
        self.query_gnn = QueryGNN(hidden_dim=vector_dim+3)

        
    def forward(self, data, batch, graph_size ):
        # TODO: Add None Checks 

        query_features = batch.x
        edge_attr = batch.edge_attr
        query_edge_index = batch.edge_index
        embedding_indices = batch.node_embedding_indices
        edge_embedding_indices = batch.edge_embedding_indices

        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch.edge_type

        #TODO: Testing conditioning on multiple relations within a single query (batch size = 1)
        # Provide shape [1, K] to seed K relations at once
        # query_rels = torch.tensor([[83, 44]], device=data.relation_graph.edge_index.device) # TODO: Statisch gesetyte 83 und 44
        relation_representations = self.relation_model(data.relation_graph, batch)
        assert relation_representations.shape[0] == batch.batch_size, "Relation representations shape does not match batch size"

        node_representations = self.entity_model(data, batch, relation_representations)

        # if node_representations.dim() == 3 and node_representations.size(0) == 1:           
        #     node_representations = node_representations.squeeze(0)

        # if relation_representations.dim() == 3 and relation_representations.size(0) == 1:           
        #     relation_representations = relation_representations.squeeze(0)

        # Merging ULTRA embeddings with precomputed query features
        bound_nodes = embedding_indices >= 0
        graph_indices = batch.batch
        embedding_indices = embedding_indices
        # HACK: not the most efficient
        node_embeddings = node_representations[graph_indices, embedding_indices]

        # bound_nodes = (embedding_indices >= 0) # bools indicating which nodes need ULTRA embeddings
        # idx = embedding_indices.clamp(min=0) # clamping is necessary for F.embedding to work

        # Selectively replace last 64 dims for non-variable nodes and keeping  first 3 dims for all
        last64 = torch.where(bound_nodes.unsqueeze(-1),          
                             node_embeddings,
                             query_features[:, 3:])         

        query_features = torch.cat([query_features[:, :3], last64], dim=1)


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

        log_cardinality_pred = self.query_gnn(query_features, query_edge_index, edge_attr=edge_features, graph_size=graph_size) 


        # return score
        return log_cardinality_pred



