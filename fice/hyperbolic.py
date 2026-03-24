"""Hyperbolic BipartiteGraphGNN using HGCN-style message passing on the Poincaré ball.

Each GNN layer operates as: logmap0 → Euclidean GNN op → expmap0.
The curvature-dependent exp/log maps act as non-linear warping between layers,
helping model hierarchical structure in knowledge graphs.

Output is mapped back to tangent space (Euclidean) so the existing TripleModel
query GNN works unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATv2Conv, GPSConv
from torch_geometric.utils import degree

from fice.models import RelationalGINELayer, DeepSetConv


def _poincare_expmap0(v, c):
    """Exponential map at origin of the Poincaré ball with curvature c.

    Maps tangent vector v ∈ R^d to point on the ball.
    Formula: tanh(√c · ||v|| / 2) · v / (√c · ||v||)
    """
    sqrt_c = c.sqrt()
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    return torch.tanh(sqrt_c * norm / 2) * v / (sqrt_c * norm)


def _poincare_logmap0(y, c):
    """Logarithmic map at origin of the Poincaré ball with curvature c.

    Maps point y on the ball to tangent vector ∈ R^d.
    Formula: 2 / √c · arctanh(√c · ||y||) · y / ||y||
    """
    sqrt_c = c.sqrt()
    norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    # Clamp argument to arctanh to (-1, 1) for stability
    return 2.0 / sqrt_c * torch.atanh((sqrt_c * norm).clamp(max=1 - 1e-5)) * y / norm


class HyperbolicBipartiteGraphGNN(nn.Module):
    """BipartiteGraphGNN with HGCN-style hyperbolic message passing.

    Identical feature encoding to BipartiteGraphGNN. The difference is in
    the message passing loop: features are mapped to the Poincaré ball,
    and each layer does logmap0 → Euclidean conv → expmap0.
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
        embedding_space: str = 'hyperbolic',  # ignored — always hyperbolic internally
    ):
        super().__init__()

        def gine_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(p=self.dropout),
            )

        self.conditioning = conditioning
        self.input_dim = input_dim
        self.message_func = message_func
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        self.dropout = dropout
        self.relational = relational
        self.role_aware_degrees = role_aware_degrees

        # Learnable curvature (softplus ensures c > 0)
        self.c_param = nn.Parameter(torch.tensor(1.0))

        # Edge encoder
        self.edge_encoder = nn.Linear(2, input_dim)

        # Node type embeddings for factor graph
        if relational:
            self.node_type_emb = nn.Embedding(3, input_dim)

        # Feature encoder (identical to BipartiteGraphGNN)
        self.pe_dim = pe_dim
        if conditioning == "random":
            self.base_dim = input_dim
        else:
            self.base_dim = 1
        self.degree_dim = 6 if role_aware_degrees else 1
        self.raw_in_dim = self.base_dim + self.degree_dim + self.pe_dim + 1
        self.feature_encoder = nn.Linear(self.raw_in_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        # GNN layers (operate in tangent space = Euclidean)
        self.layers = nn.ModuleList()
        self.use_gps = (message_func == 'gps')
        dims = (input_dim, *hidden_dims)
        for i in range(len(dims) - 1):
            if relational:
                self.layers.append(
                    RelationalGINELayer(dims[i], dims[i + 1], edge_dim=input_dim, dropout=dropout)
                )
            elif self.use_gps:
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

        # Post-GNN relation self-attention (same as BipartiteGraphGNN)
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

    @property
    def curvature(self):
        """Positive curvature via softplus."""
        return F.softplus(self.c_param)

    def _expmap0(self, v):
        """Map tangent vector at origin → Poincaré ball. Gradients flow through curvature."""
        return _poincare_expmap0(v, self.curvature)

    def _logmap0(self, y):
        """Map Poincaré ball point → tangent vector at origin. Gradients flow through curvature."""
        # Clamp norm to stay strictly inside the ball: ||y|| < 1/√c
        c = self.curvature
        max_norm = (1.0 / c.sqrt()) * 0.95
        norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        y = torch.where(norm > max_norm, y * max_norm / norm, y)
        return _poincare_logmap0(y, c)

    def forward(self, data, batch=None):
        """
        data  : PyG Data (sample) of the bipartite/factor graph of the KG
        batch : batched query graph (only batch_size & edge_embedding_indices matter)
        """
        assert len(data.edge_index) % 2 == 0, "Bipartite graph must have even number of edges."

        if batch is None:
            bs = 1
            device = data.edge_index.device
        else:
            bs = batch.batch_size
            device = batch.edge_index.device
        num_nodes = data.num_nodes

        # --- Structural Feature Construction (identical to BipartiteGraphGNN) ---
        if self.role_aware_degrees and hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[1] >= 2:
            edge_roles = data.edge_attr[:, 1].to(device)
            dst = data.edge_index[1].to(device)
            feat_deg = torch.zeros(num_nodes, 6, device=device)
            for i, role_val in enumerate([1, 2, 3, -1, -2, -3]):
                mask = (edge_roles == role_val)
                if mask.any():
                    feat_deg[:, i].scatter_add_(0, dst[mask], torch.ones(mask.sum(), device=device))
            feat_deg = torch.log1p(feat_deg)
        else:
            deg = degree(data.edge_index[0], num_nodes, dtype=torch.float)
            feat_deg = torch.log1p(deg).unsqueeze(-1).to(device)

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

        feat_seed = torch.zeros(bs, num_nodes, 1, device=device)
        if self.conditioning != "false" and batch is not None:
            valid_mask_n = batch.node_embedding_indices != -1
            graph_idx_n = batch.batch[valid_mask_n]
            node_idx_n = batch.node_embedding_indices[valid_mask_n]
            feat_seed[graph_idx_n, node_idx_n] = 1.0

            valid_mask_e = batch.edge_embedding_indices != -1
            edge2graph_e = batch.batch[batch.edge_index[0]]
            graph_idx_e = edge2graph_e[valid_mask_e]
            rel_idx_e = batch.edge_embedding_indices[valid_mask_e]
            feat_seed[graph_idx_e, rel_idx_e] = 1.0

        if self.conditioning == 'random':
            feat_base = torch.randn(bs, num_nodes, self.input_dim, device=device)
        else:
            feat_base = torch.ones(bs, num_nodes, 1, device=device) * 0.1

        feat_deg = feat_deg.unsqueeze(0).expand(bs, -1, -1)
        feat_pe = feat_pe.unsqueeze(0).expand(bs, -1, -1)

        raw_h = torch.cat([feat_base, feat_deg, feat_pe, feat_seed], dim=-1)

        h = self.feature_encoder(raw_h)
        h = self.input_norm(h)
        h = F.silu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.relational and hasattr(data, 'original_num_nodes') and hasattr(data, 'original_num_relations'):
            node_types = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_ent = data.original_num_nodes
            n_rel = data.original_num_relations
            node_types[n_ent:n_ent + n_rel] = 1
            node_types[n_ent + n_rel:] = 2
            h = h + self.node_type_emb(node_types).unsqueeze(0)

        rel_edges = data.edge_index

        # Batch edge indices
        offset = (torch.arange(bs, device=device) * num_nodes).view(-1, 1, 1)
        edge_index_batched = rel_edges.unsqueeze(0) + offset
        edge_index = edge_index_batched.permute(1, 0, 2).reshape(2, -1)

        edge_attr_stacked = data.edge_attr.to(torch.float32).repeat(bs, 1)
        edge_attr = self.edge_encoder(edge_attr_stacked)

        if self.relational:
            edge_roles = data.edge_attr[:, 1].abs().long().repeat(bs)

        x = h.view(bs * num_nodes, -1)

        if self.use_gps:
            batch_vec = torch.arange(bs, device=device).repeat_interleave(num_nodes)

        # --- HYPERBOLIC MESSAGE PASSING ---
        # Map encoded features to Poincaré ball
        x = self._expmap0(x)

        for i, layer in enumerate(self.layers):
            # Project to tangent space at origin (Euclidean)
            x_tangent = self._logmap0(x)
            x_tangent_in = x_tangent  # for skip connection

            # Standard Euclidean GNN layer
            if self.relational:
                x_tangent = layer(x_tangent, edge_index, edge_attr=edge_attr, edge_roles=edge_roles)
            elif self.use_gps:
                x_tangent = layer(x_tangent, edge_index, batch_vec, edge_attr=edge_attr)
            else:
                x_tangent = layer(x_tangent, edge_index, edge_attr=edge_attr)

            if self.enable_layer_norm and not self.use_gps:
                x_tangent = self.norms[i](x_tangent)

            if x_tangent.shape == x_tangent_in.shape and self.skip_connection and not self.use_gps:
                x_tangent = x_tangent + x_tangent_in

            x_tangent = F.dropout(x_tangent, p=self.dropout, training=self.training)

            # Project back to Poincaré ball
            x = self._expmap0(x_tangent)

        # Final output: project to tangent space (Euclidean vectors)
        out = self._logmap0(x)

        # Reshape back to (batch_size, num_nodes, hidden_dim)
        out = out.view(bs, num_nodes, -1)

        # Post-GNN relation self-attention
        if self.relation_attention and hasattr(data, 'original_num_nodes') and hasattr(data, 'original_num_relations'):
            rel_start = data.original_num_nodes
            rel_end = rel_start + data.original_num_relations
            if rel_end > rel_start:
                rel_emb = out[:, rel_start:rel_end, :]
                rel_normed = self.rel_attn_norm(rel_emb)
                rel_attn_out, _ = self.rel_attn(rel_normed, rel_normed, rel_normed)
                rel_emb = rel_emb + rel_attn_out
                rel_normed2 = self.rel_ffn_norm(rel_emb)
                rel_emb = rel_emb + self.rel_ffn(rel_normed2)
                out = torch.cat([out[:, :rel_start, :], rel_emb, out[:, rel_end:, :]], dim=1)

        del edge_index, edge_attr, edge_attr_stacked

        # No EmbeddingProjection — output is already Euclidean tangent vectors
        return out
