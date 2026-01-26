from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GINConv
from torch_geometric.utils import degree




class AblationEmbeddingGNN(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(AblationEmbeddingGNN, self).__init__()

        self.input_dim = input_dim

    def forward(self, data, batch=None, relation_representations=None):
        if batch is None:
            batch_size = 1
            device = data.edge_index.device
        else:
            batch_size = batch.batch_size
            device = batch.edge_index.device

        num_nodes = data.num_nodes
      
        x = torch.ones(batch_size, num_nodes, self.input_dim, device=device)
        
        # Clean up large batched tensors
        del batched_edge_index, edge_attr
        
        return x