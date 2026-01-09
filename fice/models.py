from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.utils import degree

from fice.gnce.models import TripleModel, QueryCardinalityGNN, IndependentQueryGNN, QueryTransformer, SimpleSumGNN


class EntityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, message_func='gine', conditioning='all', layer_norm=True, skip_connection=True, laplacian_k=0, dropout=0.0, **kwargs):
        super(EntityGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.message_func = message_func
        self.conditioning = conditioning
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        # Feature encoder to project structural features to hidden dimension
        self.dropout = dropout
        self.laplacian_k = laplacian_k
        
        # Base Feature Dimension
        if conditioning == 'random':
             self.base_dim = input_dim
        else:
             self.base_dim = 1 # Constant 0.1
             
        # Input features: [Base + LogDegree(1) + LaplacianPE(k) + SeedFlag(1)]
        self.raw_in_dim = self.base_dim + 1 + self.laplacian_k + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(p=self.dropout),
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


    def forward(self, data, batch=None, relation_representations=None):
        if batch is None:
            batch_size = 1
            device = data.edge_index.device
        else:
            batch_size = batch.batch_size
            device = batch.edge_index.device

        num_nodes = data.num_nodes

        # --- Structural Feature Construction ---
        # 1. Log Degree
        deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
        feat_deg = torch.log1p(deg).unsqueeze(-1).to(device)  # (num_nodes, 1)

        # 2. Laplacian PE (assumed pre-computed)
        if self.laplacian_k > 0 and hasattr(data, 'laplacian_eigenvector_pe'):
            pe = data.laplacian_eigenvector_pe.to(device)
            if pe.shape[-1] > self.laplacian_k:
                pe = pe[:, :self.laplacian_k]
            elif pe.shape[-1] < self.laplacian_k:
                padding = torch.zeros(pe.shape[0], self.laplacian_k - pe.shape[-1], device=device)
                pe = torch.cat([pe, padding], dim=-1)
            feat_pe = pe
        else:
            feat_pe = torch.zeros(num_nodes, self.laplacian_k, device=device)
        
        # 3. Seed Flag (varies per batch element)
        feat_seed = torch.zeros(batch_size, num_nodes, 1, device=device)
        if self.conditioning != "false" and batch is not None:
            node_indices = batch.node_embedding_indices 
            valid = node_indices != -1
            graph_idx = batch.batch[valid]
            node_pos = node_indices[valid]
            feat_seed[graph_idx, node_pos] = 1.0

        # 4. Base Features
        if self.conditioning == 'random':
            feat_base = torch.randn(batch_size, num_nodes, self.input_dim, device=device)
        else:
            feat_base = torch.ones(batch_size, num_nodes, 1, device=device) * 0.1

        # Expand shared features to batch
        feat_deg = feat_deg.unsqueeze(0).expand(batch_size, -1, -1)
        feat_pe = feat_pe.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine: [Base | Degree | PE | Seed]
        raw_x = torch.cat([feat_base, feat_deg, feat_pe, feat_seed], dim=-1)
        
        # Project to hidden dimension
        x = self.feature_encoder(raw_x)
        x = self.input_norm(x)
        x = F.silu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(-1, x.shape[-1])  # (batch_size * num_nodes, input_dim)

        # Create batched edge_index once: replicate edges for each batch element
        edge_index = data.edge_index
        
        # New vectorized implementation
        offset = (torch.arange(batch_size, device=device) * num_nodes).view(-1, 1, 1)
        edge_index_batched = edge_index.unsqueeze(0) + offset
        batched_edge_index = edge_index_batched.permute(1, 0, 2).reshape(2, -1)

        batched_edge_type = data.edge_type.repeat(batch_size)

        # Old implementation
        # # Create edge_index for batched graph: replicate edges for each batch element
        # batch_edge_list = []
        # # also replicate the edge types for each batch element
        # batched_edge_type_list = []
        # for b in range(batch_size):
        #     offset = b * num_nodes
        #     batch_edges = edge_index + offset
        #     batch_edge_list.append(batch_edges)
        #     batched_edge_type_list.append(data.edge_type)
        # batched_edge_index = torch.cat(batch_edge_list, dim=1)
        # batched_edge_type = torch.cat(batched_edge_type_list, dim=0)

        # del batch_edge_list
        # del batched_edge_type_list

        # Shape (batch_size * num_edges) - maps each edge to its batch index
        all_edges = torch.repeat_interleave(torch.arange(batch_size, device=device), data.edge_index.size(1))

        # Since relation embeddings are different for each batch, we now use all_edges to pick the correct batch and batched_edge_type to get the global id of the relation
        edge_attr = relation_representations[all_edges, batched_edge_type] # Shape (batch_size * num_edges, embedding_dim)
        
        del all_edges
        del batched_edge_type

        for i, layer in enumerate(self.layers):
            x_in = x # Save input for residual
            x = layer(x, batched_edge_index, edge_attr=edge_attr)

            if self.enable_layer_norm:
                 x = self.norms[i](x)

            if self.skip_connection:
                x = x + x_in
            
            x = F.dropout(x, p=self.dropout, training=self.training)
       
        # Reshape once at the end: (batch_size * num_nodes, output_dim) -> (batch_size, num_nodes, output_dim)
        x = x.view(batch_size, num_nodes, -1)
        
        # Clean up large batched tensors
        del batched_edge_index, edge_attr
        
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
        laplacian_k: int = 0,
        dropout: float = 0.0,
    ):
        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(p=self.dropout),
            )

        super().__init__()
        self.num_relation = num_relation
        self.conditioning = conditioning     
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        self.message_func = message_func
        self.dropout = dropout

        # 4 trainable relation embeddings
        self.rel_emb = nn.Embedding(num_relation, input_dim)

        # Feature encoder to project structural features + base embeddings
        # Input features: [BaseRelEmb(input_dim) + LogDegree(1) + LaplacianPE(k) + SeedFlag(1)]
        self.laplacian_k = laplacian_k
        self.raw_in_dim = input_dim + 1 + self.laplacian_k + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

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

        # --- Structural Feature Construction ---
        # 0. Base Relation Embeddings (fixed for all batch elements)
        # Assuming nodes in RelationGraph correspond to relation types [0..num_nodes-1]
        node_ids = torch.arange(num_nodes, device=device)
        feat_base = self.rel_emb(node_ids) # (num_nodes, input_dim)

        # 1. Log Degree
        deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
        feat_deg = torch.log1p(deg).unsqueeze(-1) # (num_nodes, 1)

        # 2. Laplacian PE
        if self.laplacian_k > 0 and hasattr(data, 'laplacian_eigenvector_pe'):
            pe = data.laplacian_eigenvector_pe.to(device)
            if pe.shape[-1] > self.laplacian_k:
                pe = pe[:, :self.laplacian_k]
            elif pe.shape[-1] < self.laplacian_k:
                padding = torch.zeros(pe.shape[0], self.laplacian_k - pe.shape[-1], device=device)
                pe = torch.cat([pe, padding], dim=-1)
            feat_pe = pe
        else:
            feat_pe = torch.zeros(num_nodes, self.laplacian_k, device=device)

        # 3. Seed Flag (varies per batch element)
        feat_seed = torch.zeros(bs, num_nodes, 1, device=device)
        if self.conditioning != "false" and batch is not None:
            valid_mask   = batch.edge_embedding_indices != -1
            edge2graph   = batch.batch[batch.edge_index[0]]
            graph_idx    = edge2graph[valid_mask]
            rel_idx      = batch.edge_embedding_indices[valid_mask]
            feat_seed[graph_idx, rel_idx] = 1.0

        # Expand shared features to batch
        feat_base = feat_base.unsqueeze(0).expand(bs, -1, -1)
        feat_deg = feat_deg.unsqueeze(0).expand(bs, -1, -1)
        feat_pe = feat_pe.unsqueeze(0).expand(bs, -1, -1)

         # Combine: [Base | Degree | PE | Seed]
        raw_h = torch.cat([feat_base, feat_deg, feat_pe, feat_seed], dim=-1)

        # Project to hidden dimension
        h = self.feature_encoder(raw_h)
        h = self.input_norm(h)
        h = F.silu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        rel_edges  = data.edge_index        # (2, E)
        rel_types  = data.edge_type         # (E,)
        
        # New vectorized implementation
        offset = (torch.arange(bs, device=device) * num_nodes).view(-1, 1, 1)
        edge_index_batched = rel_edges.unsqueeze(0) + offset
        edge_index = edge_index_batched.permute(1, 0, 2).reshape(2, -1)
        
        edge_type = rel_types.repeat(bs)

        # Old implementation
        # batch_edge_list  = []
        # batch_type_list  = []
        # # Since GAT etc. dont accept an explicit batch dimension, we need to reshape the batch_size different relation graphs into a large single graph. hence we need
        # # to increase the edge index per batch, but leave the edge type the same.
        # for b in range(bs):                 # identical to EntityGNN lines 72-84
        #     offset = b * num_nodes
        #     batch_edge_list.append(rel_edges + offset)
        #     batch_type_list.append(rel_types)
        # edge_index = torch.cat(batch_edge_list, dim=1)   # (2, bs·E)
        # edge_type  = torch.cat(batch_type_list, dim=0)   # (bs·E,)

        # del batch_edge_list
        # del batch_type_list

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
            
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        out = x.view(bs, num_nodes, -1)
        
        # Clean up large batched tensors
        del edge_index, edge_type, edge_attr
        
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
        message_func: str = 'gine',
        conditioning: str = "false",      
        layer_norm: bool = False,
        skip_connection: bool = True,
        laplacian_k: int = 0,
        dropout: float = 0.0,
    ):
        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(), 
                nn.Dropout(p=self.dropout),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(p=self.dropout),
            )

        super().__init__()
        self.conditioning = conditioning     
        self.input_dim = input_dim
        self.message_func = message_func
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        self.dropout = dropout

        # Add projection for edge attributes (Count, Direction) -> input_dim
        self.edge_encoder = nn.Linear(2, input_dim)

        # Feature encoder to project structural features to hidden dimension
        # Input features: [Base + LogDegree(1) + LaplacianPE(k) + SeedFlag(1)]
        self.laplacian_k = laplacian_k
        if conditioning == "random":
            self.base_dim = input_dim
        else:
            self.base_dim = 1
        self.raw_in_dim = self.base_dim + 1 + self.laplacian_k + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            if self.message_func == 'gine':
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

        # --- Structural Feature Construction ---
        # 1. Log Degree
        deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
        feat_deg = torch.log1p(deg).unsqueeze(-1).to(device) # (num_nodes, 1)

        # 2. Laplacian PE
        if self.laplacian_k > 0 and hasattr(data, 'laplacian_eigenvector_pe'):
            pe = data.laplacian_eigenvector_pe.to(device)
            if pe.shape[-1] > self.laplacian_k:
                pe = pe[:, :self.laplacian_k]
            elif pe.shape[-1] < self.laplacian_k:
                padding = torch.zeros(pe.shape[0], self.laplacian_k - pe.shape[-1], device=device)
                pe = torch.cat([pe, padding], dim=-1)
            feat_pe = pe
        else:
            feat_pe = torch.zeros(num_nodes, self.laplacian_k, device=device)

        # 3. Seed Flag (varies per batch element)
        feat_seed = torch.zeros(bs, num_nodes, 1, device=device)
        if self.conditioning != "false" and batch is not None:
            # entities
            valid_mask_n = batch.node_embedding_indices != -1
            graph_idx_n  = batch.batch[valid_mask_n]
            node_idx_n   = batch.node_embedding_indices[valid_mask_n]
            feat_seed[graph_idx_n, node_idx_n] = 1.0
            
            # relations
            valid_mask_e = batch.edge_embedding_indices != -1
            edge2graph_e = batch.batch[batch.edge_index[0]]
            graph_idx_e  = edge2graph_e[valid_mask_e]
            rel_idx_e    = batch.edge_embedding_indices[valid_mask_e]
            feat_seed[graph_idx_e, rel_idx_e] = 1.0

        # 4. Base Features
        if self.conditioning == 'random':
            feat_base = torch.randn(bs, num_nodes, self.input_dim, device=device)
        else:
            feat_base = torch.ones(bs, num_nodes, 1, device=device) * 0.1

        # Expand shared features to batch
        feat_deg = feat_deg.unsqueeze(0).expand(bs, -1, -1)
        feat_pe = feat_pe.unsqueeze(0).expand(bs, -1, -1)

        # Combine: [Base | Degree | PE | Seed]
        raw_h = torch.cat([feat_base, feat_deg, feat_pe, feat_seed], dim=-1)
        
        # Project to hidden dimension
        h = self.feature_encoder(raw_h)
        h = self.input_norm(h)
        h = F.silu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        rel_edges  = data.edge_index        # (2, E)
        batch_edge_list  = []
        # Since GAT etc. dont accept an explicit batch dimension, we need to reshape the batch_size different relation graphs into a large single graph. hence we need
        # to increase the edge index per batch, but leave the edge type the same.

        # new
        offset = (torch.arange(bs, device=device) * num_nodes).view(-1, 1, 1) 
        edge_index_batched = rel_edges.unsqueeze(0) + offset  # (bs, 2, E)
        edge_index = edge_index_batched.permute(1, 0, 2).reshape(2, -1)  # (2, bs·E)

        # old
        # for b in range(bs):                 # identical to EntityGNN lines 72-84
        #     offset = b * num_nodes
        #     batch_edge_list.append(rel_edges + offset)
        # edge_index = torch.cat(batch_edge_list, dim=1)   # (2, bs·E)
        del batch_edge_list

        # Encode edge attributes (Count, Direction)
        # 1. Replicate edge attributes for batch
        # new
        edge_attr_stacked = data.edge_attr.to(torch.float32).repeat(bs, 1) # Shape [bs*E, 2]

        # old
        # rel_edge_attr = data.edge_attr.to(torch.float32) # Shape [E, 2]
        # batch_edge_attr_list = []
        # for b in range(bs):
        #     batch_edge_attr_list.append(rel_edge_attr)
        # edge_attr_stacked = torch.cat(batch_edge_attr_list, dim=0) # Shape [bs*E, 2]

    # del batch_edge_attr_list
        
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
            
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        # and numnodes is total number of nodes (entities + relations)
        out = x.view(bs, num_nodes, -1)
        
        # Clean up large batched tensors
        del edge_index, edge_attr, edge_attr_stacked
        
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
                 query_gnn_dropout=0.0,
                 # Global params
                 laplacian_k=0,
                 entity_dropout=0.0,
                 relation_dropout=0.0
                 ):
        super(FICE, self).__init__()
        self.vector_dim = vector_dim
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
                message_func=relation_message_func,
                layer_norm=relation_layer_norm, # Use relation norm as default for unified
                skip_connection=relation_skip_connection,
                conditioning=conditioning,
                laplacian_k=laplacian_k,
                dropout=relation_dropout # Using relation_dropout as default for unified
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
                    conditioning=conditioning,
                    laplacian_k=laplacian_k,
                    dropout=relation_dropout
                )
            elif self.relation_graph_mode == "bipartite":
                # If using bipartite graph for relations, we use the unified model
                self.relation_model = BipartiteGraphGNN(
                    input_dim=self.vector_dim,
                    hidden_dims=self.hidden_dims_relnet,
                    message_func=relation_message_func,
                    layer_norm=relation_layer_norm,
                    skip_connection=relation_skip_connection,
                    conditioning=conditioning,
                    laplacian_k=laplacian_k,
                    dropout=relation_dropout
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
                    skip_connection=entity_skip_connection,
                    laplacian_k=laplacian_k,
                    dropout=entity_dropout
                )
            elif self.entity_graph_mode == "bipartite" and self.relation_graph_mode == "RelationGraph":
                raise ValueError("Entity graph bipartite but relation graph standard is not supported.")
            else:
                raise ValueError(f"Invalid entity message function: {entity_message_func}")

        if self.query_gnn_type == 'QueryCardinalityGNN':
            self.query_gnn = QueryCardinalityGNN(feat_dim=self.vector_dim + 3, num_layers=3)
        elif self.query_gnn_type == 'TripleModel':
            self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3, layer_norm=query_layer_norm, skip_connection=query_skip_connection, dropout=query_gnn_dropout)
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

            del all_embeddings

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
                
                del node_representations

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

            del relation_representations

            # Selectively replace last 64 (128) dims for edges with actual relations and keeping first 3 dims for all
            edge_last = torch.where(bound_edges.unsqueeze(-1),
                                      edge_embeddings,
                                      edge_attr[:, 3:])

            edge_features = torch.cat([edge_attr[:, :3], edge_last], dim=1)
            batch.edge_attr = edge_features
        
        # Universal log-transform for counts
        x_counts = torch.log1p(batch.x[:, :3])
        e_counts = torch.log1p(batch.edge_attr[:, :3])

        if self.query_gnn_type == 'QueryCardinalityGNN':
            log_graph_size = torch.log1p(graph_size.float())
            
            unbound_nodes = batch.node_embedding_indices < 0
            x_counts = torch.where(unbound_nodes.unsqueeze(-1), log_graph_size, x_counts)
            
            unbound_edges = batch.edge_embedding_indices < 0
            e_counts = torch.where(unbound_edges.unsqueeze(-1), log_graph_size, e_counts)

        batch.x = torch.cat([x_counts, batch.x[:, 3:]], dim=1)
        batch.edge_attr = torch.cat([e_counts, batch.edge_attr[:, 3:]], dim=1)
        
        if self.query_gnn_type == 'QueryCardinalityGNN':
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

    def __init__(self, vector_dim: int = 128, skip_connection: bool = True):
        super().__init__()
        self.vector_dim = vector_dim
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3, skip_connection=skip_connection)

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

        x_counts = torch.log1p(batch.x[:, :3])
        batch.x = torch.cat([x_counts, batch.x[:, 3:]], dim=1)

        e_counts = torch.log1p(batch.edge_attr[:, :3])
        batch.edge_attr = torch.cat([e_counts, batch.edge_attr[:, 3:]], dim=1)

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
        
        x_counts = torch.log1p(batch.x[:, :3])
        batch.x = torch.cat([x_counts, batch.x[:, 3:]], dim=1)

        e_counts = torch.log1p(batch.edge_attr[:, :3])
        batch.edge_attr = torch.cat([e_counts, batch.edge_attr[:, 3:]], dim=1)

        log_cardinality_pred = self.query_gnn(batch=batch, N=graph_size)
        return log_cardinality_pred
