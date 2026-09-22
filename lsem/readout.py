"""LSEM readout — structure-regularized grid readout over the intact latent spatial grid.

Given the latent spatial evidence grid produced by :class:`lsem.extract.LSEMExtractor`
(shape ``(M, R, C, D)`` = modalities x rows x cols x hidden), LSEM reads each clinical
concept with a *per-concept weight field* over the grid::

    s_c = sum_i <e_i, w_{c,i}>            # e_i = P h_i, one weight vector per grid cell i

The grid is never pooled, routed, or rewritten (all spatial evidence is preserved).
Structure is injected *only* by regularizing the weight FIELD:

    Omega(W_c) = lambda_group * sum_i ||w_{c,i}||                              (which cells matter)
               + lambda_fused * sum_{(i,j) in grid edges} ||w_{c,i} - w_{c,j}||  (contiguous regions)
               + lambda_bilateral * sum_{(i,i') in mirror pairs}
                     ( ||w_{c,i}+w_{c,i'}|| + ||w_{c,i}-w_{c,i'}|| )            (symmetric / antisymmetric)

Setting all lambdas to 0 recovers the plain full-grid linear readout, so the regularized
readout can never underperform it by construction. The learned weight field is interpretable:
``region_map`` returns a per-cell importance heatmap (regions emerge automatically, no ROIs /
no routing) and ``bilateral_energy`` reports how much a concept relies on left-right asymmetry.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _grid_edges(M: int, R: int, C: int):
    """4-neighbour edges within each modality and left-right mirror pairs, as flat cell indices."""
    cid = lambda m, r, c: m * R * C + r * C + c
    edges = [(cid(m, r, c), cid(m, r2, c2))
             for m in range(M) for r in range(R) for c in range(C)
             for (r2, c2) in ((r + 1, c), (r, c + 1)) if r2 < R and c2 < C]
    mirror = [(cid(m, r, c), cid(m, r, C - 1 - c))
              for m in range(M) for r in range(R) for c in range(C // 2)]
    return edges, mirror


class LSEMReadout(nn.Module):
    """Structure-regularized grid readout.

    Parameters
    ----------
    in_dim : hidden size D of the latent grid cells.
    num_classes : dict {concept_name: n_classes}.
    grid_shape : (M, R, C) of the latent spatial grid (default (2, 6, 4)).
    proj_dim : cell projection width d (default 96).
    lambda_group / lambda_fused / lambda_bilateral : structure-regularization strengths.
        Defaults (group+fused on, bilateral off) = the ``grid_fused`` configuration used in the paper.
    """

    def __init__(self, in_dim: int, num_classes: dict, grid_shape=(2, 6, 4), proj_dim: int = 96,
                 lambda_group: float = 0.01, lambda_fused: float = 0.01, lambda_bilateral: float = 0.0):
        super().__init__()
        self.M, self.R, self.C = grid_shape
        self.ncell = self.M * self.R * self.C
        self.d = proj_dim
        self.lg, self.lf, self.lb = lambda_group, lambda_fused, lambda_bilateral
        self.proj = nn.Linear(in_dim, proj_dim)
        self.W = nn.ParameterDict({c: nn.Parameter(torch.randn(k, self.ncell, proj_dim) * 0.01)
                                   for c, k in num_classes.items()})
        self.b = nn.ParameterDict({c: nn.Parameter(torch.zeros(k)) for c, k in num_classes.items()})
        edges, mirror = _grid_edges(self.M, self.R, self.C)
        self.register_buffer("ei", torch.tensor([a for a, _ in edges]))
        self.register_buffer("ej", torch.tensor([b for _, b in edges]))
        self.register_buffer("mi", torch.tensor([a for a, _ in mirror]))
        self.register_buffer("mj", torch.tensor([b for _, b in mirror]))

    def _cells(self, grid: torch.Tensor) -> torch.Tensor:
        """(N, M, R, C, in_dim) or (N, ncell, in_dim) -> projected cells (N, ncell, d)."""
        x = grid.reshape(grid.shape[0], self.ncell, grid.shape[-1])
        return self.proj(x)

    def forward(self, grid: torch.Tensor, concept: str) -> torch.Tensor:
        """Return class logits (N, n_classes) for one concept. The grid is used intact."""
        e = self._cells(grid)
        return torch.einsum("ncd,kcd->nk", e, self.W[concept]) + self.b[concept]

    def structure_penalty(self, concept: str) -> torch.Tensor:
        """Group + fused + bilateral penalty on the concept's weight field (add to the loss)."""
        W = self.W[concept]
        pen = W.new_zeros(())
        if self.lg > 0:
            pen = pen + self.lg * W.norm(dim=(0, 2)).sum()
        if self.lf > 0:
            pen = pen + self.lf * (W[:, self.ei] - W[:, self.ej]).norm(dim=(0, 2)).sum()
        if self.lb > 0:
            sym = W[:, self.mi] + W[:, self.mj]
            anti = W[:, self.mi] - W[:, self.mj]
            pen = pen + self.lb * (sym.norm(dim=(0, 2)).sum() + anti.norm(dim=(0, 2)).sum())
        return pen

    @torch.no_grad()
    def region_map(self, concept: str) -> torch.Tensor:
        """Per-cell readout importance ||w_i|| reshaped to (M, R, C) — an interpretable region heatmap."""
        return self.W[concept].norm(dim=(0, 2)).reshape(self.M, self.R, self.C)

    @torch.no_grad()
    def bilateral_energy(self, concept: str):
        """(symmetric_energy, antisymmetric_energy) of the weight field over mirror pairs."""
        W = self.W[concept]
        sym = (W[:, self.mi] + W[:, self.mj]).norm().item()
        anti = (W[:, self.mi] - W[:, self.mj]).norm().item()
        return sym, anti
