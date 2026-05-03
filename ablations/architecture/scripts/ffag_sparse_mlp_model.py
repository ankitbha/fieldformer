#!/usr/bin/env python3
"""Sparse FieldFormer architecture ablation with an MLP token mixer."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn


class _TokenMLP(nn.Module):
    def __init__(self, d_model: int, layers: int, d_ff: int, out_dim: int):
        super().__init__()
        depth = max(1, int(layers))
        token_layers: list[nn.Module] = [nn.Linear(4, d_model), nn.GELU()]
        for _ in range(depth - 1):
            token_layers.extend([nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.GELU()])
        self.token_mlp = nn.Sequential(*token_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.token_mlp(tokens)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        return self.head(pooled)


class FieldFormerMLPSparse(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, d_ff: int):
        super().__init__()
        del nhead
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.mlp = _TokenMLP(d_model=d_model, layers=layers, d_ff=d_ff, out_dim=1)

    def _tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        dx = (nb_xyt[..., 0] - xyt_q[:, None, 0] + 0.5 * Lx) % Lx - 0.5 * Lx
        dy = (nb_xyt[..., 1] - xyt_q[:, None, 1] + 0.5 * Ly) % Ly - 0.5 * Ly
        dt = nb_xyt[..., 2] - xyt_q[:, None, 2]
        rel = torch.stack([dx, dy, dt], dim=-1) * torch.exp(self.log_gammas)[None, None, :]
        return torch.cat([rel, nb_vals[..., None]], dim=-1)

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        return self.mlp(self._tokens(xyt_q, nb_xyt, nb_vals, Lx, Ly)).squeeze(-1)

    def forward_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        return self._forward_tokens(obs_coords[q_lin], obs_coords[nb_idx], obs_vals[nb_idx], Lx=Lx, Ly=Ly)

    def forward_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        return self._forward_tokens(xyt_q, obs_coords[nb_idx], obs_vals[nb_idx], Lx=Lx, Ly=Ly)


class FieldFormerMLPSparseSWE(FieldFormerMLPSparse):
    def __init__(self, d_model: int, nhead: int, layers: int, d_ff: int):
        nn.Module.__init__(self)
        del nhead
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.mlp = _TokenMLP(d_model=d_model, layers=layers, d_ff=d_ff, out_dim=3)

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        return self.mlp(self._tokens(xyt_q, nb_xyt, nb_vals, Lx, Ly))


class FieldFormerMLPSparsePollution(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, d_ff: int):
        super().__init__()
        del nhead
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.mlp = _TokenMLP(d_model=d_model, layers=layers, d_ff=d_ff, out_dim=1)

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor) -> torch.Tensor:
        rel = (nb_xyt - xyt_q[:, None, :]) * torch.exp(self.log_gammas)[None, None, :]
        mu = nb_vals.mean(dim=1, keepdim=True)
        sigma = nb_vals.std(dim=1, keepdim=True).clamp_min(1e-3)
        nb_vals_norm = ((nb_vals - mu) / sigma)[..., None]
        tokens = torch.cat([rel, nb_vals_norm], dim=-1)
        amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
        with amp_ctx:
            u_std_res = self.mlp(tokens).squeeze(-1)
        return u_std_res * sigma.squeeze(1) + mu.squeeze(1)

    def forward_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
        return self._forward_tokens(obs_coords[q_lin], obs_coords[nb_idx], obs_vals[nb_idx])

    def forward_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
        return self._forward_tokens(xyt_q, obs_coords[nb_idx], obs_vals[nb_idx])


def class_for_dataset(dataset_key: str) -> type[nn.Module]:
    return {
        "heat": FieldFormerMLPSparse,
        "swe": FieldFormerMLPSparseSWE,
        "pol": FieldFormerMLPSparsePollution,
    }[dataset_key]
