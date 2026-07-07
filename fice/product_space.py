"""Product Space (Euclidean × Torus) models for FICE.

Contains both the relation encoder (ProductSpaceBipartiteGraphGNN) and the
query decoder (ProductTripleModel). Both split embedding dimensions into a
Euclidean branch and a torus branch with rotation-based message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from fice.models import RelationalGINELayer


class TorusRelationalLayer(nn.Module):
    """Rotation-based message passing on the torus T^d.

    Per role: messages are source embeddings rotated by a learned phase vector.
    Forward edges use the rotation, reverse edges use the conjugate (inverse).
    Aggregation: scatter_sum → project to unit circle.
    Update: α-blend self with aggregated → project to unit circle.
    """

    def __init__(self, complex_dim, num_roles=3):
        super().__init__()
        self.complex_dim = complex_dim
        self.num_roles = num_roles

        # Per-role learned phase vectors (rotation angles)
        self.role_phases = nn.ParameterList([
            nn.Parameter(torch.randn(complex_dim) * 0.1)
            for _ in range(num_roles)
        ])
        # Per-role amplitude scaling (initialized at 1.0 = pure rotation)
        self.role_log_amplitudes = nn.ParameterList([
            nn.Parameter(torch.zeros(complex_dim))
            for _ in range(num_roles)
        ])
        # Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_complex, edge_index, signed_roles):
        """
        x_complex: (N, complex_dim) complex64 — on the torus
        edge_index: (2, E)
        signed_roles: (E,) values in {-3, -2, -1, 1, 2, 3}
        Returns: (N, complex_dim) complex64 — on the torus
        """
        N = x_complex.size(0)
        agg = torch.zeros(N, self.complex_dim, dtype=torch.cfloat, device=x_complex.device)
        src, dst = edge_index

        for role_idx in range(self.num_roles):
            phase = self.role_phases[role_idx]
            amp = self.role_log_amplitudes[role_idx].exp().clamp(max=10.0)

            # Forward edges: rotation
            fwd_mask = (signed_roles == (role_idx + 1))
            if fwd_mask.any():
                rotation = amp * torch.exp(1j * phase)
                msg = x_complex[src[fwd_mask]] * rotation
                agg.scatter_add_(0, dst[fwd_mask].unsqueeze(-1).expand_as(msg), msg)

            # Reverse edges: conjugate rotation (inverse)
            rev_mask = (signed_roles == -(role_idx + 1))
            if rev_mask.any():
                rotation = amp * torch.exp(-1j * phase)
                msg = x_complex[src[rev_mask]] * rotation
                agg.scatter_add_(0, dst[rev_mask].unsqueeze(-1).expand_as(msg), msg)

        # Project aggregated messages to torus
        agg = agg / (agg.abs() + 1e-8)

        # α-blend self with aggregated, then project
        alpha = torch.sigmoid(self.alpha)
        out = alpha * x_complex + (1 - alpha) * agg
        return out / (out.abs() + 1e-8)


class ProductSpaceBipartiteGraphGNN(nn.Module):
    """Hybrid Euclidean + Torus BipartiteGraphGNN.

    Euclidean branch: RelationalGINELayer (proven GINE with per-role MLPs).
    Torus branch: TorusRelationalLayer (RotatE-style rotations).
    Output: cat[h_eucl, h_complex.real, h_complex.imag] of dim hidden_dims[-1].
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple = (128, 128),
        message_func: str = 'gine',  # ignored — eucl branch always uses RelationalGINELayer
        conditioning: str = "false",
        layer_norm: bool = False,
        skip_connection: bool = True,
        pe_dim: int = 0,
        dropout: float = 0.0,
        relation_attention: bool = False,
        relational: bool = False,  # ignored — always relational
        role_aware_degrees: bool = True,
        embedding_space: str = 'product',  # ignored
    ):
        super().__init__()
        self.conditioning = conditioning
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.enable_layer_norm = layer_norm
        self.dropout = dropout
        self.role_aware_degrees = role_aware_degrees

        # Dimension split
        out_dim = hidden_dims[-1]
        self.complex_dim = out_dim // 8
        self.eucl_dim = out_dim - 2 * self.complex_dim

        # Edge encoder (shared, projects [count, role] → input_dim)
        self.edge_encoder = nn.Linear(2, input_dim)

        # Node type embeddings for factor graph (entity=0, relation=1, triple=2)
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

        # Split projections: shared features → branch-specific dims
        self.proj_eucl = nn.Linear(input_dim, self.eucl_dim)
        self.proj_phase = nn.Linear(input_dim, self.complex_dim)

        # Euclidean branch: RelationalGINELayer per layer
        num_layers = len(hidden_dims)
        self.eucl_layers = nn.ModuleList([
            RelationalGINELayer(self.eucl_dim, self.eucl_dim, edge_dim=input_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Torus branch: TorusRelationalLayer per layer
        self.torus_layers = nn.ModuleList([
            TorusRelationalLayer(self.complex_dim)
            for _ in range(num_layers)
        ])

        # LayerNorms for Euclidean branch only
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.eucl_dim) for _ in hidden_dims] if self.enable_layer_norm else []
        )

        # Post-GNN relation self-attention
        self.relation_attention = relation_attention
        if relation_attention:
            self.rel_attn_norm = nn.LayerNorm(out_dim)
            self.rel_attn = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
            self.rel_ffn_norm = nn.LayerNorm(out_dim)
            self.rel_ffn = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.SiLU(),
                nn.Linear(out_dim * 2, out_dim),
            )

    @staticmethod
    def _project_to_torus(z, eps=1e-8):
        return z / (z.abs() + eps)

    def forward(self, data, batch=None):
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
            edge_roles_raw = data.edge_attr[:, 1].to(device)
            dst = data.edge_index[1].to(device)
            feat_deg = torch.zeros(num_nodes, 6, device=device)
            for i, role_val in enumerate([1, 2, 3, -1, -2, -3]):
                mask = (edge_roles_raw == role_val)
                if mask.any():
                    feat_deg[:, i].scatter_add_(0, dst[mask], torch.ones(mask.sum(), device=device))
            feat_deg = torch.log1p(feat_deg)
        else:
            feat_deg = torch.log1p(degree(data.edge_index[0], num_nodes, dtype=torch.float)).unsqueeze(-1).to(device)

        if self.pe_dim > 0 and hasattr(data, 'random_walk_pe'):
            pe = data.random_walk_pe.to(device)
            if pe.shape[-1] > self.pe_dim:
                pe = pe[:, :self.pe_dim]
            elif pe.shape[-1] < self.pe_dim:
                pe = torch.cat([pe, torch.zeros(pe.shape[0], self.pe_dim - pe.shape[-1], device=device)], dim=-1)
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

        # Node type embeddings
        if hasattr(data, 'original_num_nodes') and hasattr(data, 'original_num_relations'):
            node_types = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_ent = data.original_num_nodes
            n_rel = data.original_num_relations
            node_types[n_ent:n_ent + n_rel] = 1
            node_types[n_ent + n_rel:] = 2
            h = h + self.node_type_emb(node_types).unsqueeze(0)

        # Batch edge indices
        rel_edges = data.edge_index
        offset = (torch.arange(bs, device=device) * num_nodes).view(-1, 1, 1)
        edge_index = (rel_edges.unsqueeze(0) + offset).permute(1, 0, 2).reshape(2, -1)

        # Encode edge attributes for Euclidean branch
        edge_attr = self.edge_encoder(data.edge_attr.to(torch.float32).repeat(bs, 1))
        # Absolute roles for Euclidean branch, signed roles for Torus branch
        abs_roles = data.edge_attr[:, 1].abs().long().repeat(bs)
        signed_roles = data.edge_attr[:, 1].long().repeat(bs)

        x = h.view(bs * num_nodes, -1)

        # --- Split into branches ---
        x_eucl = self.proj_eucl(x)
        x_phase = self.proj_phase(x)
        x_complex = self._project_to_torus(torch.exp(1j * x_phase))

        # --- Message passing ---
        for i in range(len(self.eucl_layers)):
            x_eucl_prev = x_eucl
            x_complex_prev = x_complex

            # Euclidean branch
            x_eucl = self.eucl_layers[i](x_eucl, edge_index, edge_attr=edge_attr, edge_roles=abs_roles)

            # Torus branch
            x_complex = self.torus_layers[i](x_complex, edge_index, signed_roles)

            # LayerNorm (Euclidean only)
            if self.enable_layer_norm and i < len(self.norms):
                x_eucl = self.norms[i](x_eucl)

            # Skip connections
            if self.skip_connection:
                x_eucl = x_eucl + x_eucl_prev
                x_complex = self._project_to_torus(x_complex * x_complex_prev)

            x_eucl = F.dropout(x_eucl, p=self.dropout, training=self.training)

        # --- Flatten to real output ---
        out = torch.cat([x_eucl, x_complex.real, x_complex.imag], dim=-1)
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

        return out


# =============================================================================
# Product Space Query GNN
# =============================================================================

class TorusQueryLayer(nn.Module):
    """Rotation-based message passing on the torus for query graphs.

    Unlike TorusRelationalLayer (which has per-role learned rotations),
    this layer uses the edge embedding's torus part AS the rotation.
    This is the RotatE principle: the relation embedding defines how
    a head entity is rotated to predict the tail entity.
    """

    def __init__(self, complex_dim):
        super().__init__()
        self.complex_dim = complex_dim
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_complex, edge_index, edge_complex):
        """
        x_complex: (N, complex_dim) complex64 — node embeddings on torus
        edge_index: (2, E)
        edge_complex: (E, complex_dim) complex64 — edge rotations
        Returns: (N, complex_dim) complex64 — on torus
        """
        N = x_complex.size(0)
        agg = torch.zeros(N, self.complex_dim, dtype=torch.cfloat, device=x_complex.device)
        src, dst = edge_index

        # Message: rotate source by edge rotation
        msg = x_complex[src] * edge_complex
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        # Project to torus
        agg = agg / (agg.abs() + 1e-8)

        # Alpha-blend self with aggregated
        alpha = torch.sigmoid(self.alpha)
        out = alpha * x_complex + (1 - alpha) * agg
        return out / (out.abs() + 1e-8)


class ProductTripleModel(nn.Module):
    """Product space (Euclidean × Torus) query GNN.

    Splits query features into Euclidean and torus parts. The Euclidean branch
    uses GINEConv (same as TripleModel). The torus branch uses edge embeddings
    as RotatE-style rotations. Both branches run in parallel, then flatten
    and pool for the final cardinality prediction.

    Drop-in replacement for TripleModel when using product space embeddings.
    """

    def __init__(
        self,
        feat_dim: int = 131,
        layer_norm=True,
        skip_connection=False,
        dropout=0.0,
        message_func='gine',
        use_graph_size: bool = True,
        use_softplus: bool = False,
        num_layers: int = 2,
        readout: str = 'sum',
        message_activation: str = 'relu',
        embedding_space: str = 'euclidean',  # unused
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.layer_norm = layer_norm
        self.skip_connection = skip_connection
        self.use_graph_size = use_graph_size
        self.use_softplus = use_softplus

        # Dimension split (must match ProductSpaceBipartiteGraphGNN)
        embedding_dim = feat_dim - 3  # remove 3 count dims
        self.complex_dim = embedding_dim // 8
        self.eucl_dim = embedding_dim - 2 * self.complex_dim
        self.eucl_feat_dim = 3 + self.eucl_dim  # counts go with Euclidean branch

        readout_dim = 200

        # --- Euclidean branch: GINEConv layers ---
        def make_gine(in_dim, out_dim, edge_dim):
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(p=dropout),
            )
            return GINEConv(nn=mlp, edge_dim=edge_dim)

        self.eucl_convs = nn.ModuleList()
        self.eucl_norms = nn.ModuleList()
        for i in range(num_layers):
            self.eucl_convs.append(make_gine(self.eucl_feat_dim, self.eucl_feat_dim, self.eucl_feat_dim))
            self.eucl_norms.append(nn.LayerNorm(self.eucl_feat_dim))

        # --- Torus branch: TorusQueryLayer per layer ---
        self.torus_layers = nn.ModuleList([
            TorusQueryLayer(self.complex_dim) for _ in range(num_layers)
        ])

        # --- Readout ---
        # Project flattened [eucl_feat | torus_real | torus_imag] → readout_dim
        combined_dim = self.eucl_feat_dim + 2 * self.complex_dim  # = feat_dim
        self.readout_proj = nn.Linear(combined_dim, readout_dim)

        self.readout_type = readout
        if readout == 'attention':
            gate_nn = nn.Sequential(
                nn.Linear(readout_dim, readout_dim // 2),
                nn.SiLU(),
                nn.Linear(readout_dim // 2, 1),
            )
            self.pool = AttentionalAggregation(gate_nn)

        lin_input_dim = readout_dim + 1 if use_graph_size else readout_dim
        self.lin = nn.Linear(lin_input_dim, 50)
        self.lin2 = nn.Linear(50, 1)
        self.dropout_layer = nn.Dropout(p=dropout)

    def _split_features(self, features):
        """Split [counts(3) | eucl(d_E) | torus_real(d_T) | torus_imag(d_T)] into branches."""
        eucl = features[:, :3 + self.eucl_dim]  # counts + euclidean part
        torus_real = features[:, 3 + self.eucl_dim:3 + self.eucl_dim + self.complex_dim]
        torus_imag = features[:, 3 + self.eucl_dim + self.complex_dim:]
        torus_complex = torch.complex(torus_real, torus_imag)
        # Project to torus; replace zero-norm (unbound nodes/edges) with identity
        norm = torus_complex.abs()
        torus_complex = torch.where(
            norm > 1e-8,
            torus_complex / (norm + 1e-8),
            torch.ones_like(torus_complex),
        )
        return eucl, torus_complex

    def forward(self, batch, graph_size):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # Split node and edge features
        x_eucl, x_torus = self._split_features(x)
        e_eucl, e_torus = self._split_features(edge_attr)

        # Message passing
        for i in range(len(self.eucl_convs)):
            x_eucl_prev = x_eucl
            x_torus_prev = x_torus

            # Euclidean branch
            x_eucl = F.silu(self.eucl_convs[i](x_eucl, edge_index, edge_attr=e_eucl))
            if self.layer_norm:
                x_eucl = self.eucl_norms[i](x_eucl)

            # Torus branch
            x_torus = self.torus_layers[i](x_torus, edge_index, e_torus)

            # Skip connections
            if self.skip_connection:
                x_eucl = x_eucl + x_eucl_prev
                x_torus = x_torus * x_torus_prev
                x_torus = x_torus / (x_torus.abs() + 1e-8)

        # Flatten: [eucl | torus_real | torus_imag]
        x = torch.cat([x_eucl, x_torus.real, x_torus.imag], dim=-1)
        x = F.silu(self.readout_proj(x))

        # Pool to graph level
        if self.readout_type == 'attention':
            x = self.pool(x, batch.batch)
        else:
            x = global_add_pool(x, batch.batch)

        if self.use_graph_size:
            log_gs = torch.log1p(graph_size.float()).unsqueeze(-1).expand(x.size(0), -1)
            x = torch.cat([x, log_gs], dim=-1)

        x = self.dropout_layer(x)
        x = self.lin(x)
        x = F.silu(x)
        x = self.lin2(x)

        if self.use_softplus:
            return x.squeeze(-1)
        return torch.abs(x.squeeze(-1))
