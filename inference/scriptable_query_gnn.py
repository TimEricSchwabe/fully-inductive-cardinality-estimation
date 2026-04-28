"""TorchScript-compatible reimplementation of TripleModel (query GNN).

Reimplements GINEConv + global pooling + MLP head using only pure PyTorch ops
(scatter_add_, Linear, LayerNorm, etc.) so the module can be exported via
torch.jit.script or torch.jit.trace for use with libtorch in C++.

Use transfer_weights() to copy weights from a trained PyG TripleModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScriptableGINEConv(nn.Module):
    """Pure-PyTorch reimplementation of torch_geometric.nn.GINEConv.

    GINEConv:
      message_j = relu(x[src] + lin(edge_attr))
      agg       = scatter_add(message_j, dst)
      out       = nn(agg + (1 + eps) * x)
    """

    def __init__(self, nn_module: nn.Sequential, edge_dim: int, in_channels: int):
        super().__init__()
        self.nn = nn_module
        self.lin = nn.Linear(edge_dim, in_channels)
        self.eps = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        # Message: relu(x[source] + lin(edge_attr))
        edge_feat = self.lin(edge_attr)
        messages = F.relu(x[src] + edge_feat)

        # Aggregate at destination via scatter_add
        agg = torch.zeros(x.size(0), messages.size(1),
                          device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # Self-loop + MLP
        out = self.nn(agg + (1.0 + self.eps) * x)
        return out


class ScriptableAttentionPool(nn.Module):
    """Pure-PyTorch reimplementation of AttentionalAggregation.

    gate = softmax(gate_nn(x), per graph)
    output = scatter_add(gate * x, batch)
    """

    def __init__(self, gate_nn: nn.Sequential):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                num_graphs: int) -> torch.Tensor:
        gate = self.gate_nn(x)  # (N, 1)

        # Per-graph softmax via log-sum-exp trick
        # Compute max per graph for numerical stability
        gate_max = torch.full((num_graphs, 1), float('-inf'),
                              device=x.device, dtype=x.dtype)
        gate_max.scatter_reduce_(0, batch.unsqueeze(1), gate, reduce='amax',
                                 include_self=False)
        gate_shifted = gate - gate_max[batch]
        gate_exp = gate_shifted.exp()

        # Sum of exp per graph
        gate_sum = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        gate_sum.scatter_add_(0, batch.unsqueeze(1), gate_exp)

        # Normalized attention weights
        gate_softmax = gate_exp / (gate_sum[batch] + 1e-16)

        # Weighted sum
        weighted = gate_softmax * x
        out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted), weighted)
        return out


class _ConvBlock(nn.Module):
    """Pairs a GINEConv with its LayerNorm for TorchScript-friendly iteration."""

    def __init__(self, conv: ScriptableGINEConv, norm: nn.LayerNorm):
        super().__init__()
        self.conv = conv
        self.norm = norm


class ScriptableTripleModel(nn.Module):
    """TorchScript-compatible reimplementation of TripleModel.

    Forward signature takes raw tensors (not PyG Batch objects):
        x:          (num_query_nodes, feat_dim)
        edge_index: (2, num_query_edges)
        edge_attr:  (num_query_edges, feat_dim)
        batch:      (num_query_nodes,) int64 — graph assignment per node
        graph_size: (1,) or scalar — total edges in the KG

    Weight transfer from PyG TripleModel uses a key remapping
    (convs.N.* → blocks.N.conv.*, norms.N.* → blocks.N.norm.*).
    """

    def __init__(self, feat_dim: int, num_layers: int = 2,
                 layer_norm: bool = True, skip_connection: bool = False,
                 dropout: float = 0.0, use_graph_size: bool = True,
                 readout: str = 'sum'):
        super().__init__()
        self.feat_dim = feat_dim
        self.layer_norm = layer_norm
        self.skip_connection = skip_connection
        self.use_graph_size = use_graph_size
        self.readout_type = readout
        self.dropout_p = dropout
        readout_dim = 200

        blocks: list[_ConvBlock] = []
        for i in range(num_layers):
            out_dim = readout_dim if i == num_layers - 1 else feat_dim
            mlp = nn.Sequential(
                nn.Linear(feat_dim, out_dim),
                nn.Dropout(p=dropout),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(p=dropout),
            )
            conv = ScriptableGINEConv(mlp, edge_dim=feat_dim,
                                      in_channels=feat_dim)
            norm = nn.LayerNorm(out_dim)
            blocks.append(_ConvBlock(conv, norm))
        self.blocks = nn.ModuleList(blocks)

        self.skip_proj = nn.Linear(feat_dim, readout_dim)

        # Readout — always create pool so TorchScript sees the attribute
        self.use_attention = (readout == 'attention')
        gate_nn = nn.Sequential(
            nn.Linear(readout_dim, readout_dim // 2),
            nn.SiLU(),
            nn.Linear(readout_dim // 2, 1),
        )
        self.pool = ScriptableAttentionPool(gate_nn)

        lin_input_dim = readout_dim + 1 if use_graph_size else readout_dim
        self.lin = nn.Linear(lin_input_dim, 50)
        self.lin2 = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                graph_size: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1

        for block in self.blocks:
            x_input = x
            x = F.silu(block.conv(x, edge_index, edge_attr))
            if self.layer_norm:
                x = block.norm(x)
            if self.skip_connection:
                if x.size(-1) != x_input.size(-1):
                    x = x + self.skip_proj(x_input)
                else:
                    x = x + x_input

        # Global pooling
        if self.use_attention:
            x = self.pool(x, batch, num_graphs)
        else:
            pooled = torch.zeros(num_graphs, x.size(1),
                                 device=x.device, dtype=x.dtype)
            pooled.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            x = pooled

        if self.use_graph_size:
            log_gs = torch.log1p(graph_size.float()).unsqueeze(-1).expand(
                x.size(0), -1)
            x = torch.cat([x, log_gs], dim=-1)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin(x)
        x = F.silu(x)
        x = self.lin2(x)
        return torch.abs(x.squeeze(-1))


def transfer_weights(original_model: nn.Module,
                     scriptable_model: ScriptableTripleModel) -> None:
    """Copy weights from a PyG TripleModel into a ScriptableTripleModel.

    Handles the key remapping:
      convs.N.*  → blocks.N.conv.*
      norms.N.*  → blocks.N.norm.*
    Other keys (skip_proj, lin, lin2, pool) are identical.
    """
    src_sd = original_model.state_dict()
    tgt_sd = scriptable_model.state_dict()

    def _remap_key(orig_key: str) -> str:
        """Map original TripleModel key to ScriptableTripleModel key."""
        # convs.0.nn.0.weight → blocks.0.conv.nn.0.weight
        if orig_key.startswith("convs."):
            parts = orig_key.split(".", 2)  # ['convs', '0', 'nn.0.weight']
            return f"blocks.{parts[1]}.conv.{parts[2]}"
        # norms.0.weight → blocks.0.norm.weight
        if orig_key.startswith("norms."):
            parts = orig_key.split(".", 2)  # ['norms', '0', 'weight']
            return f"blocks.{parts[1]}.norm.{parts[2]}"
        # pool.gate_nn.* → pool.gate_nn.* (identical)
        return orig_key

    new_sd = {}
    for orig_key, value in src_sd.items():
        new_key = _remap_key(orig_key)
        if new_key in tgt_sd:
            new_sd[new_key] = value
        else:
            raise RuntimeError(
                f"Remapped key '{new_key}' (from '{orig_key}') "
                f"not found in ScriptableTripleModel"
            )

    # Fill any remaining target keys with their defaults
    # (e.g. pool.* when readout='sum' — pool exists but isn't in source)
    for k in tgt_sd:
        if k not in new_sd:
            new_sd[k] = tgt_sd[k]

    scriptable_model.load_state_dict(new_sd)


def build_scriptable_from_config(feat_dim: int, num_layers: int = 2,
                                 layer_norm: bool = True,
                                 skip_connection: bool = False,
                                 dropout: float = 0.0,
                                 use_graph_size: bool = True,
                                 readout: str = 'sum') -> ScriptableTripleModel:
    """Create a ScriptableTripleModel with the given config."""
    return ScriptableTripleModel(
        feat_dim=feat_dim,
        num_layers=num_layers,
        layer_norm=layer_norm,
        skip_connection=skip_connection,
        dropout=dropout,
        use_graph_size=use_graph_size,
        readout=readout,
    )
