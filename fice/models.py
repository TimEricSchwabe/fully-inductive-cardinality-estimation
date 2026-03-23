from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GINConv, GPSConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
import time 
from fice.gnce.models import TripleModel, QueryCardinalityGNN, IndependentQueryGNN, QueryTransformer, SimpleSumGNN
from fice.embedding_spaces import EmbeddingProjection


class DeepSetConv(MessagePassing):
    def __init__(self, nn_module, edge_dim, eps=0.0, train_eps=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn_module
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.phi_with_edge = nn.Sequential(
            nn.Linear(2 * edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.phi_no_edge = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        for module in self.phi_with_edge:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.phi_no_edge:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j, edge_attr=None):
        if edge_attr is None:
            return self.phi_no_edge(x_j)
        return self.phi_with_edge(torch.cat([x_j, edge_attr], dim=-1))


class RelationalGINELayer(nn.Module):
    """Per-edge-type GINE: separate message MLPs for subject, predicate, object roles."""

    def __init__(self, in_dim, out_dim, edge_dim, num_roles=3, dropout=0.0):
        super().__init__()
        self.num_roles = num_roles

        # Per-role message transforms: (source_feat || edge_feat) → in_dim
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + edge_dim, in_dim),
                nn.SiLU(),
            )
            for _ in range(num_roles)
        ])

        # Shared update MLP (GINE-style)
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(p=dropout),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, edge_roles):
        """
        x: (N, in_dim), edge_index: (2, E), edge_attr: (E, edge_dim),
        edge_roles: (E,) absolute role values (1, 2, 3).
        """
        num_nodes = x.size(0)
        role_aggs = []

        for role_idx in range(self.num_roles):
            role = role_idx + 1
            mask = (edge_roles == role)
            if not mask.any():
                continue
            src_idx = edge_index[0, mask]
            dst_idx = edge_index[1, mask]
            src_feats = x[src_idx]
            e_feats = edge_attr[mask]

            msg = self.msg_mlps[role_idx](torch.cat([src_feats, e_feats], dim=-1))
            role_agg = torch.zeros(num_nodes, x.size(1), device=x.device)
            role_agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)
            role_aggs.append(role_agg)

        agg = sum(role_aggs) if role_aggs else torch.zeros(num_nodes, x.size(1), device=x.device)
        out = self.update_mlp((1 + self.eps) * x + agg)
        return out


class EntityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, message_func='gine', conditioning='all', layer_norm=True, skip_connection=True, pe_dim=0, dropout=0.0, embedding_space='euclidean', **kwargs):
        super(EntityGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.message_func = message_func
        self.conditioning = conditioning
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        # Feature encoder to project structural features to hidden dimension
        self.dropout = dropout
        self.pe_dim = pe_dim

        # Base Feature Dimension
        if conditioning == 'random':
             self.base_dim = 32
        else:
             self.base_dim = 1 # Constant 0.1

        # Input features: [Base + LogDegree(1) + PE(k) + SeedFlag(1)]
        self.raw_in_dim = self.base_dim + 1 + self.pe_dim + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.LayerNorm(out_dim),
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
            elif message_func == 'deepset':
                self.layers.append(
                    DeepSetConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            else:
                self.layers.append(
                    GATv2Conv(dims[i], dims[i + 1], heads=1, concat=False, edge_dim=input_dim)
                )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if self.enable_layer_norm else []
        )

        # Geometric embedding space projection at output
        self.embed_proj = EmbeddingProjection(hidden_dims[-1], space=embedding_space)

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

        # 2. Positional Encoding (assumed pre-computed)
        if self.pe_dim > 0 and hasattr(data, 'random_walk_pe'):
            pe = data.random_walk_pe.to(device)
            if pe.shape[-1] > self.pe_dim:
                pe = pe[:, :self.pe_dim]
            elif pe.shape[-1] < self.pe_dim:
                padding = torch.zeros(pe.shape[0], self.pe_dim - pe.shape[-1], device=device)
                pe = torch.cat([pe, padding], dim=-1)
            feat_pe = pe
        else:
            feat_pe = torch.zeros(num_nodes, self.pe_dim, device=device)

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
            feat_base = torch.randn(batch_size, num_nodes, 32, device=device)
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
        # TODO: Was wenn es hier einen boundary case gibt, dass eine relation in einem batch element nicht existiert?

        # relation_representation stores the embeddings on a global scale.
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

        return self.embed_proj(x)


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
        pe_dim: int = 0,
        dropout: float = 0.0,
        embedding_space: str = 'euclidean',
    ):
        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.LayerNorm(out_dim),
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
        # Input features: [BaseRelEmb(input_dim) + LogDegree(1) + PE(k) + SeedFlag(1)]
        self.pe_dim = pe_dim
        self.raw_in_dim = input_dim + 1 + self.pe_dim + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        self.layers = nn.ModuleList()
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            if self.message_func == 'gine':
                self.layers.append(
                    GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            elif self.message_func == 'deepset':
                self.layers.append(
                    DeepSetConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            else:
                # GATv2Conv
                self.layers.append(
                    GATv2Conv(dims[i], dims[i + 1], heads=1, concat=False, edge_dim=input_dim)
                )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if self.enable_layer_norm else []
        )

        # Geometric embedding space projection at output
        self.embed_proj = EmbeddingProjection(hidden_dims[-1], space=embedding_space)

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

        # 2. Positional Encoding
        if self.pe_dim > 0 and hasattr(data, 'random_walk_pe'):
            pe = data.random_walk_pe.to(device)
            if pe.shape[-1] > self.pe_dim:
                pe = pe[:, :self.pe_dim]
            elif pe.shape[-1] < self.pe_dim:
                padding = torch.zeros(pe.shape[0], self.pe_dim - pe.shape[-1], device=device)
                pe = torch.cat([pe, padding], dim=-1)
            feat_pe = pe
        else:
            feat_pe = torch.zeros(num_nodes, self.pe_dim, device=device)

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

        return self.embed_proj(out)


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
        pe_dim: int = 0,
        dropout: float = 0.0,
        relation_attention: bool = False,
        relational: bool = False,
        role_aware_degrees: bool = True,
        embedding_space: str = 'euclidean',
    ):
        def gine_mlp(in_dim, out_dim):
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.LayerNorm(out_dim),
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
        self.relational = relational
        self.role_aware_degrees = role_aware_degrees

        # Add projection for edge attributes (Count, Direction) -> input_dim
        self.edge_encoder = nn.Linear(2, input_dim)

        # Node type embeddings for factor graph (entity=0, relation=1, triple=2)
        if relational:
            self.node_type_emb = nn.Embedding(3, input_dim)

        # Feature encoder to project structural features to hidden dimension
        self.pe_dim = pe_dim
        if conditioning == "random":
            self.base_dim = input_dim
        else:
            self.base_dim = 1
        self.degree_dim = 6 if role_aware_degrees else 1
        self.raw_in_dim = self.base_dim + self.degree_dim + self.pe_dim + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        self.layers = nn.ModuleList()
        self.use_gps = (message_func == 'gps')
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            if relational:
                self.layers.append(
                    RelationalGINELayer(dims[i], dims[i + 1], edge_dim=input_dim, dropout=dropout)
                )
            elif self.message_func == 'gps':
                # GPS: local GINEConv + global self-attention in each layer
                local_conv = GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                self.layers.append(
                    GPSConv(dims[i + 1], local_conv, heads=4, dropout=0.0)
                )
            elif self.message_func == 'gine':
                self.layers.append(
                    GINEConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            elif self.message_func == 'deepset':
                self.layers.append(
                    DeepSetConv(gine_mlp(dims[i], dims[i + 1]), edge_dim=input_dim, train_eps=False)
                )
            else:
                self.layers.append(
                    GATv2Conv(dims[i], dims[i + 1], heads=1, concat=False, edge_dim=input_dim)
                )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in hidden_dims] if self.enable_layer_norm else []
        )

        # Post-GNN relation self-attention
        # Geometric embedding space projection at output
        self.embed_proj = EmbeddingProjection(hidden_dims[-1], space=embedding_space)

        self.relation_attention = relation_attention
        if relation_attention:
            out_dim = hidden_dims[-1]
            self.rel_attn_norm = nn.LayerNorm(out_dim)
            self.rel_attn = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
            self.rel_ffn_norm = nn.LayerNorm(out_dim)
            self.rel_ffn = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.SiLU(),
                nn.Linear(out_dim * 2, out_dim),
            )

    def forward(self, data, batch=None):
        """
        data  : PyG Data (sample) of the bipartite/factor graph of the KG
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
        if self.role_aware_degrees and hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[1] >= 2:
            # Role-aware Degree (6 dims: subject_in, predicate_in, object_in, subject_out, predicate_out, object_out)
            edge_roles = data.edge_attr[:, 1].to(device)
            dst = data.edge_index[1].to(device)
            feat_deg = torch.zeros(num_nodes, 6, device=device)
            for i, role_val in enumerate([1, 2, 3, -1, -2, -3]):
                mask = (edge_roles == role_val)
                if mask.any():
                    feat_deg[:, i].scatter_add_(0, dst[mask], torch.ones(mask.sum(), device=device))
            feat_deg = torch.log1p(feat_deg)
        else:
            # Simple 1-dim log-degree
            deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
            feat_deg = torch.log1p(deg).unsqueeze(-1).to(device)

        # 2. Positional Encoding
        if self.pe_dim > 0 and hasattr(data, 'random_walk_pe'):
            pe = data.random_walk_pe.to(device)
            if pe.shape[-1] > self.pe_dim:
                pe = pe[:, :self.pe_dim]
            elif pe.shape[-1] < self.pe_dim:
                padding = torch.zeros(pe.shape[0], self.pe_dim - pe.shape[-1], device=device)
                pe = torch.cat([pe, padding], dim=-1)
            feat_pe = pe
        else:
            feat_pe = torch.zeros(num_nodes, self.pe_dim, device=device)

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

        # Add node type embeddings for relational mode (entity=0, relation=1, triple=2)
        if self.relational and hasattr(data, 'original_num_nodes') and hasattr(data, 'original_num_relations'):
            node_types = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_ent = data.original_num_nodes
            n_rel = data.original_num_relations
            node_types[n_ent:n_ent + n_rel] = 1
            node_types[n_ent + n_rel:] = 2
            h = h + self.node_type_emb(node_types).unsqueeze(0)

        rel_edges  = data.edge_index        # (2, E)

        # Batch edge indices
        offset = (torch.arange(bs, device=device) * num_nodes).view(-1, 1, 1)
        edge_index_batched = rel_edges.unsqueeze(0) + offset  # (bs, 2, E)
        edge_index = edge_index_batched.permute(1, 0, 2).reshape(2, -1)  # (2, bs·E)

        # Encode edge attributes (Count, Direction/Role) -> input_dim
        edge_attr_stacked = data.edge_attr.to(torch.float32).repeat(bs, 1) # Shape [bs*E, 2]
        edge_attr = self.edge_encoder(edge_attr_stacked) # Shape [bs*E, input_dim]

        # Compute absolute edge roles for relational layers (1=subject, 2=predicate, 3=object)
        if self.relational:
            edge_roles = data.edge_attr[:, 1].abs().long().repeat(bs)  # (bs*E,)

        x = h.view(bs * num_nodes, -1)

        # Batch vector for GPS (maps each node to its graph in the batch)
        if self.use_gps:
            batch_vec = torch.arange(bs, device=device).repeat_interleave(num_nodes)

        # Message passing
        for i, layer in enumerate(self.layers):
            x_in = x # Save input for residual
            if self.relational:
                x = layer(x, edge_index, edge_attr=edge_attr, edge_roles=edge_roles)
            elif self.use_gps:
                x = layer(x, edge_index, batch_vec, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index, edge_attr=edge_attr)

            if self.enable_layer_norm and not self.use_gps:  # GPS has its own internal norms
                 x = self.norms[i](x)

            if x.shape == x_in.shape and self.skip_connection and not self.use_gps:  # GPS has its own residual
                x = x + x_in

            x = F.dropout(x, p=self.dropout, training=self.training)

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        # and numnodes is total number of nodes (entities + relations)
        out = x.view(bs, num_nodes, -1)

        # Post-GNN relation self-attention
        if self.relation_attention and hasattr(data, 'original_num_nodes') and hasattr(data, 'original_num_relations'):
            rel_start = data.original_num_nodes
            rel_end = rel_start + data.original_num_relations
            if rel_end > rel_start:
                rel_emb = out[:, rel_start:rel_end, :]  # (bs, num_rels, dim)

                # Pre-norm self-attention + residual
                rel_normed = self.rel_attn_norm(rel_emb)
                rel_attn_out, _ = self.rel_attn(rel_normed, rel_normed, rel_normed)
                rel_emb = rel_emb + rel_attn_out

                # Pre-norm FFN + residual
                rel_normed2 = self.rel_ffn_norm(rel_emb)
                rel_emb = rel_emb + self.rel_ffn(rel_normed2)

                # Avoid in-place modification for autograd
                out = torch.cat([out[:, :rel_start, :], rel_emb, out[:, rel_end:, :]], dim=1)

        # Clean up large batched tensors
        del edge_index, edge_attr, edge_attr_stacked

        return self.embed_proj(out)


class FICE(nn.Module):

    def __init__(self,
                 vector_dim=128,
                 n_embedding_layers=2,
                 conditioning='all',
                 entity_graph_mode: Literal['bipartite', 'KG'] = "KG",
                 relation_graph_mode: Literal['bipartite'] = "bipartite",
                 # Entity GNN params
                 entity_message_func='gine',
                 entity_layer_norm=False,
                 entity_skip_connection=True,
                 # Relation GNN params
                 relation_message_func='gine',
                 relation_layer_norm=False,
                 relation_skip_connection=True,
                 relation_attention=False,
                 relational=False,
                 role_aware_degrees=True,
                 # Query GNN params
                 query_gnn_type='TripleModel',
                 query_message_func='gine',
                 query_layer_norm=True,
                 query_skip_connection=False,
                 query_gnn_dropout=0.0,
                 query_gnn_num_layers=2,
                 query_gnn_readout='sum',
                 query_gnn_message_activation='relu',
                 # Global params
                 pe_dim=0,
                 entity_dropout=0.0,
                 relation_dropout=0.0,
                 prediction_mode='log-cardinality',
                 embedding_space='euclidean',
                 ):
        super(FICE, self).__init__()
        self.vector_dim = vector_dim
        self.prediction_mode = prediction_mode
        self.skip_connection = entity_skip_connection # Default legacy behavior, though now we use specific ones
        self.use_unified_bipartite_graph = (entity_graph_mode == "bipartite")
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
                pe_dim=pe_dim,
                dropout=relation_dropout, # Using relation_dropout as default for unified
                relation_attention=relation_attention,
                relational=relational,
                role_aware_degrees=role_aware_degrees,
                embedding_space=embedding_space,
            )

        else:
            self.relation_model = BipartiteGraphGNN(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_relnet,
                message_func=relation_message_func,
                layer_norm=relation_layer_norm,
                skip_connection=relation_skip_connection,
                conditioning=conditioning,
                pe_dim=pe_dim,
                dropout=relation_dropout,
                relation_attention=relation_attention,
                relational=relational,
                role_aware_degrees=role_aware_degrees,
                embedding_space=embedding_space,
            )

            self.entity_model = EntityGNN(
                input_dim=self.vector_dim,
                hidden_dims=self.hidden_dims_entitynet,
                message_func=entity_message_func,
                conditioning=conditioning,
                layer_norm=entity_layer_norm,
                skip_connection=entity_skip_connection,
                pe_dim=pe_dim,
                dropout=entity_dropout,
                embedding_space=embedding_space,
            )

        if self.query_gnn_type == 'QueryCardinalityGNN':
            self.query_gnn = QueryCardinalityGNN(feat_dim=self.vector_dim + 3, num_layers=3, embedding_space=embedding_space)
        elif self.query_gnn_type == 'TripleModel':
            use_gs = (prediction_mode != 'log-selectivity')
            use_sp = (prediction_mode == 'log-selectivity')
            self.query_gnn = TripleModel(
                feat_dim=self.vector_dim + 3,
                layer_norm=query_layer_norm,
                skip_connection=query_skip_connection,
                dropout=query_gnn_dropout,
                message_func=query_message_func,
                use_graph_size=use_gs,
                use_softplus=use_sp,
                num_layers=query_gnn_num_layers,
                readout=query_gnn_readout,
                message_activation=query_gnn_message_activation,
            )
        elif self.query_gnn_type == 'QueryTransformer':
            self.query_gnn = QueryTransformer(feat_dim=self.vector_dim + 3)
        elif self.query_gnn_type == 'SimpleSumGNN':
            self.query_gnn = SimpleSumGNN(feat_dim=self.vector_dim + 3)
        else:
            raise ValueError(f"Invalid query GNN type: {self.query_gnn_type}")
        

    def forward(self, data, relation_graph, batch, graph_size, epoch=0, repeat: int = 1):
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
                extracted_relation_representation = torch.where(relation_indices.unsqueeze(-1) >=0,
                                                                extracted_relation_representation,
                                                                torch.zeros_like(extracted_relation_representation))

                # entity_indices = batch.node_embedding_indices 
                # graph_indices = batch.batch
                # extracted_entity_representation = relation_representations[graph_indices, entity_indices]
                # extracted_entity_representation =torch.where(entity_indices.unsqueeze(-1) >=0,
                #                         extracted_entity_representation,
                #                         torch.zeros_like(extracted_entity_representation))
                
                # new_entity_representations = torch.zeros((relation_representations.shape[0], relation_graph.original_num_nodes, relation_representations.shape[2]), device=device)
                # new_entity_representations[graph_indices, batch.original_node_embedding_indices] = extracted_entity_representation
                # entity_representations = new_entity_representations


                num_relations = relation_graph.original_num_relations # TODO: keine ahung wie stabil das ist, Antwort überhaupt nicht stabil, besser lösung finden
            

                new_relation_representations = torch.zeros((relation_representations.shape[0], num_relations, relation_representations.shape[2]), device=device) 
                new_relation_representations[edge2graph, batch.original_edge_embedding_indices] = extracted_relation_representation
                relation_representations = new_relation_representations

            if data is not None:
                node_representations = self.entity_model(batch=batch, data=data, relation_representations=relation_representations)
                
                bound_nodes = embedding_indices >= 0
                graph_indices = batch.batch

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

        if self.prediction_mode == 'log-selectivity':
            src = batch.edge_index[0]  # subject nodes
            tgt = batch.edge_index[1]  # object nodes

            log_N = torch.log(graph_size.float().clamp(min=1))

            # Per-triple counts: pick the role-appropriate dimension
            count_s = batch.x[src, 0]       # head count of subject
            count_p = batch.edge_attr[:, 1]  # relation count of predicate
            count_t = batch.x[tgt, 2]       # tail count of object

            # Unbound (variable) nodes/edges have count=0 → use N
            count_s = torch.where(count_s > 0, count_s, graph_size.float())
            count_p = torch.where(count_p > 0, count_p, graph_size.float())
            count_t = torch.where(count_t > 0, count_t, graph_size.float())

            # Independence cardinality per triple pattern (in log space)
            log_card_per_triple = torch.log(count_s) + torch.log(count_p) + torch.log(count_t) - 2 * log_N

            # Sum across triples per query
            edge2graph = batch.batch[src]
            count_upper_bound = torch.zeros(batch.num_graphs, device=device)
            count_upper_bound.scatter_add_(0, edge2graph, log_card_per_triple)

            x_counts = torch.zeros_like(x_counts)
            e_counts = torch.zeros_like(e_counts)

        if self.query_gnn_type == 'QueryCardinalityGNN':
            log_graph_size = torch.log1p(graph_size.float())

            unbound_nodes = batch.node_embedding_indices < 0
            x_counts = torch.where(unbound_nodes.unsqueeze(-1), log_graph_size, x_counts)

            unbound_edges = batch.edge_embedding_indices < 0
            e_counts = torch.where(unbound_edges.unsqueeze(-1), log_graph_size, e_counts)

        batch.x = torch.cat([x_counts, batch.x[:, 3:]], dim=1)
        batch.edge_attr = torch.cat([e_counts, batch.edge_attr[:, 3:]], dim=1)

        start = time.perf_counter()
        for _ in range(repeat):
            if self.query_gnn_type == 'QueryCardinalityGNN':
                querygnn_output = self.query_gnn(batch=batch, N=graph_size)
            elif self.query_gnn_type == 'TripleModel':
                querygnn_output = self.query_gnn(graph_size=graph_size, batch=batch)
            elif self.query_gnn_type == 'QueryTransformer':
                querygnn_output = self.query_gnn(batch=batch, N=graph_size)
            elif self.query_gnn_type == 'SimpleSumGNN':
                querygnn_output = self.query_gnn(batch=batch, N=graph_size)
            else:
                raise ValueError(f"Invalid query GNN type: {self.query_gnn_type}")
        end = time.perf_counter()
        elapsed = end - start
        avg_time = elapsed / repeat

        if self.prediction_mode == 'log-selectivity':
            log_cardinality_pred = count_upper_bound + F.logsigmoid(querygnn_output)
        else:
            log_cardinality_pred = querygnn_output

        if repeat > 1:
            return log_cardinality_pred, avg_time
        return log_cardinality_pred


class FiceBinary(nn.Module):
    """Ablation model that encodes only the IDs actually present in the
    *query batch* instead of generating embeddings for the entire entity /
    relation universe.  ``data`` and ``relation_graph`` inputs are therefore
    optional (they can be ``None``).
    """

    def __init__(self, vector_dim: int = 128, skip_connection: bool = True, query_layer_norm: bool = True, query_skip_connection: bool = False, query_gnn_dropout: float = 0.0):
        super().__init__()
        self.vector_dim = vector_dim
        self.query_gnn = TripleModel(feat_dim=self.vector_dim + 3, layer_norm=query_layer_norm, skip_connection=query_skip_connection, dropout=query_gnn_dropout)

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
        # x_counts = torch.zeros_like(x_counts)
        batch.x = torch.cat([x_counts, batch.x[:, 3:]], dim=1)

        e_counts = torch.log1p(batch.edge_attr[:, :3])
        # x_counts = torch.zeros_like(e_counts)
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
