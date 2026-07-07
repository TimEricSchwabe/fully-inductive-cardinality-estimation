"""Geometric embedding space projections for cardinality estimation.

Provides projection layers that map raw GNN outputs into structured geometric
spaces (box, Gaussian) where volume and intersection operations have natural
interpretations for cardinality estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbeddingProjection(nn.Module):
    """Projects raw GNN output into a geometric embedding space.

    The GNN operates in Euclidean space internally. This thin layer at the
    output imposes geometric structure on the embedding vector:
      - 'euclidean': identity (no-op)
      - 'box': first D/2 = min, last D/2 = max  (hyperrectangle)
      - 'gaussian': first D/2 = μ, last D/2 = log(σ)
    """

    def __init__(self, embed_dim: int, space: str = 'euclidean'):
        super().__init__()
        self.space = space
        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 2

        if space not in ('euclidean', 'box', 'gaussian'):
            raise ValueError(f"Unknown embedding space: {space}")
        if space != 'euclidean' and embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for {space} space, got {embed_dim}")

    def forward(self, h: Tensor) -> Tensor:
        if self.space == 'euclidean':
            return h

        center = h[..., :self.half_dim]
        raw_offset = h[..., self.half_dim:]

        if self.space == 'box':
            # offset > 0 via softplus; box = (center - offset, center + offset)
            offset = F.softplus(raw_offset)
            return torch.cat([center - offset, center + offset], dim=-1)

        if self.space == 'gaussian':
            # mu = center, log_sigma = raw (unrestricted)
            return torch.cat([center, raw_offset], dim=-1)


# --------------- Geometry utilities ---------------

def box_log_volume(emb: Tensor, half_dim: int) -> Tensor:
    """Log-volume of a box embedding. Shape: (...,)"""
    mins, maxs = emb[..., :half_dim], emb[..., half_dim:]
    widths = F.softplus(maxs - mins)  # positive even after perturbation
    return widths.log().sum(dim=-1)


def box_intersection_log_volume(emb_a: Tensor, emb_b: Tensor, half_dim: int) -> Tensor:
    """Log-volume of the intersection of two box embeddings. Shape: (...,)"""
    min_a, max_a = emb_a[..., :half_dim], emb_a[..., half_dim:]
    min_b, max_b = emb_b[..., :half_dim], emb_b[..., half_dim:]
    inter_min = torch.max(min_a, min_b)
    inter_max = torch.min(max_a, max_b)
    widths = F.softplus(inter_max - inter_min)
    return widths.log().sum(dim=-1)


def box_center(emb: Tensor, half_dim: int) -> Tensor:
    """Center of a box embedding. Shape: (..., half_dim)"""
    return (emb[..., :half_dim] + emb[..., half_dim:]) / 2


def gaussian_product(mu_a: Tensor, log_sigma_a: Tensor,
                     mu_b: Tensor, log_sigma_b: Tensor):
    """Product of two diagonal Gaussians (precision-weighted combination).

    Returns (mu_prod, log_sigma_prod).
    """
    prec_a = (-2 * log_sigma_a).exp()
    prec_b = (-2 * log_sigma_b).exp()
    prec_sum = prec_a + prec_b
    mu_prod = (prec_a * mu_a + prec_b * mu_b) / prec_sum
    log_sigma_prod = -0.5 * prec_sum.log()
    return mu_prod, log_sigma_prod
