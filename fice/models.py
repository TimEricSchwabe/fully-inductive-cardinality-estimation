from torch import nn

from easydict import EasyDict
from fice.ultra.models import RelNBFNet, EntityNBFNet
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
    

        
    # def forward(self, query):
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        node_representations = self.entity_model(data, relation_representations, batch)

        # score = self.gnce_model(node_representations, relation_representations)

        # return score
        return relation_representations, node_representations



