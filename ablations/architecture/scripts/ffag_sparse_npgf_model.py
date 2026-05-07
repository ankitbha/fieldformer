#!/usr/bin/env python3
"""Sparse FieldFormer ablation with query-dependent gamma fields."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.backends.cuda import sdp_kernel


POLLUTION_KEYS = {"pol", "govpol", "atm", "govpolsplit", "atmsplit"}


class GammaFieldMLP(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        layers: int = 2,
        base_gammas: tuple[float, float, float] = (1.0, 1.0, 1.0),
        max_delta: float = 2.0,
    ):
        super().__init__()
        depth = max(1, int(layers))
        width = max(1, int(hidden))
        modules: list[nn.Module] = [nn.Linear(3, width), nn.GELU()]
        for _ in range(depth - 1):
            modules.extend([nn.Linear(width, width), nn.GELU()])
        final = nn.Linear(width, 3)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        modules.append(final)
        self.net = nn.Sequential(*modules)
        self.max_delta = float(max_delta)
        base = torch.tensor(base_gammas, dtype=torch.float32).clamp_min(1e-6).log()
        self.register_buffer("base_log_gammas", base)

    def forward(self, xyt01: torch.Tensor) -> torch.Tensor:
        log_gamma = self.base_log_gammas + self.max_delta * torch.tanh(self.net(xyt01))
        return torch.exp(log_gamma)


class GammaFieldMixin:
    def _init_gamma_field(
        self,
        *,
        gamma_hidden: int,
        gamma_layers: int,
        gamma_max_delta: float,
        base_gammas: tuple[float, float, float],
    ) -> None:
        self.gamma_field = GammaFieldMLP(
            hidden=gamma_hidden,
            layers=gamma_layers,
            base_gammas=base_gammas,
            max_delta=gamma_max_delta,
        )
        self.register_buffer("coord_min", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("coord_span", torch.ones(3, dtype=torch.float32))

    def set_coord_ranges(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        t_min: float,
        t_max: float,
    ) -> None:
        mins = torch.tensor([x_min, y_min, t_min], dtype=self.coord_min.dtype, device=self.coord_min.device)
        spans = torch.tensor(
            [max(float(x_max) - float(x_min), 1e-6), max(float(y_max) - float(y_min), 1e-6), max(float(t_max) - float(t_min), 1e-6)],
            dtype=self.coord_span.dtype,
            device=self.coord_span.device,
        )
        self.coord_min.copy_(mins)
        self.coord_span.copy_(spans)

    def query_gammas(self, xyt_q: torch.Tensor) -> torch.Tensor:
        q01 = ((xyt_q - self.coord_min.to(xyt_q.device)) / self.coord_span.to(xyt_q.device)).clamp(0.0, 1.0)
        return self.gamma_field(q01)

    def gamma_metadata(self) -> dict[str, object]:
        return {
            "base_gammas": torch.exp(self.gamma_field.base_log_gammas.detach()).cpu().tolist(),
            "coord_min": self.coord_min.detach().cpu().tolist(),
            "coord_span": self.coord_span.detach().cpu().tolist(),
            "max_delta": float(self.gamma_field.max_delta),
        }


class FieldFormerNPGFSparse(GammaFieldMixin, nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        layers: int,
        d_ff: int,
        *,
        gamma_hidden: int = 32,
        gamma_layers: int = 2,
        gamma_max_delta: float = 2.0,
        base_gammas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self._init_gamma_field(
            gamma_hidden=gamma_hidden,
            gamma_layers=gamma_layers,
            gamma_max_delta=gamma_max_delta,
            base_gammas=base_gammas,
        )
        self.input_proj = nn.Linear(4, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        dx = (nb_xyt[..., 0] - xyt_q[:, None, 0] + 0.5 * Lx) % Lx - 0.5 * Lx
        dy = (nb_xyt[..., 1] - xyt_q[:, None, 1] + 0.5 * Ly) % Ly - 0.5 * Ly
        dt = nb_xyt[..., 2] - xyt_q[:, None, 2]
        rel = torch.stack([dx, dy, dt], dim=-1) * self.query_gammas(xyt_q)[:, None, :]
        tokens = torch.cat([rel, nb_vals[..., None]], dim=-1)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                h = self.input_proj(tokens)
                h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)

    def forward_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        return self._forward_tokens(obs_coords[q_lin], obs_coords[nb_idx], obs_vals[nb_idx], Lx=Lx, Ly=Ly)

    def forward_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        return self._forward_tokens(xyt_q, obs_coords[nb_idx], obs_vals[nb_idx], Lx=Lx, Ly=Ly)


class FieldFormerNPGFSparseSWE(FieldFormerNPGFSparse):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        layers: int,
        d_ff: int,
        *,
        gamma_hidden: int = 32,
        gamma_layers: int = 2,
        gamma_max_delta: float = 2.0,
        base_gammas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        nn.Module.__init__(self)
        self._init_gamma_field(
            gamma_hidden=gamma_hidden,
            gamma_layers=gamma_layers,
            gamma_max_delta=gamma_max_delta,
            base_gammas=base_gammas,
        )
        self.input_proj = nn.Linear(4, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_eta: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
        dx = (nb_xyt[..., 0] - xyt_q[:, None, 0] + 0.5 * Lx) % Lx - 0.5 * Lx
        dy = (nb_xyt[..., 1] - xyt_q[:, None, 1] + 0.5 * Ly) % Ly - 0.5 * Ly
        dt = nb_xyt[..., 2] - xyt_q[:, None, 2]
        rel = torch.stack([dx, dy, dt], dim=-1) * self.query_gammas(xyt_q)[:, None, :]
        tokens = torch.cat([rel, nb_eta[..., None]], dim=-1)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                h = self.input_proj(tokens)
                h = self.encoder(h)
        return self.head(h.mean(dim=1))


class FieldFormerNPGFSparsePollution(GammaFieldMixin, nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        layers: int,
        d_ff: int,
        out_dim: int = 1,
        *,
        gamma_hidden: int = 32,
        gamma_layers: int = 2,
        gamma_max_delta: float = 2.0,
        base_gammas: tuple[float, float, float] = (1.0, 1.0, 0.5),
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self._init_gamma_field(
            gamma_hidden=gamma_hidden,
            gamma_layers=gamma_layers,
            gamma_max_delta=gamma_max_delta,
            base_gammas=base_gammas,
        )
        self.input_proj = nn.Linear(3 + self.out_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.out_dim),
        )

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor) -> torch.Tensor:
        rel = (nb_xyt - xyt_q[:, None, :]) * self.query_gammas(xyt_q)[:, None, :]
        if nb_vals.ndim == 2:
            nb_vals = nb_vals[..., None]
        mu = nb_vals.mean(dim=1, keepdim=True)
        sigma = nb_vals.std(dim=1, keepdim=True).clamp_min(1e-3)
        nb_vals_norm = (nb_vals - mu) / sigma
        tokens = torch.cat([rel, nb_vals_norm], dim=-1)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                h = self.input_proj(tokens)
                h = self.encoder(h)
        out = self.head(h.mean(dim=1)) * sigma.squeeze(1) + mu.squeeze(1)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def forward_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
        return self._forward_tokens(obs_coords[q_lin], obs_coords[nb_idx], obs_vals[nb_idx])

    def forward_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
        return self._forward_tokens(xyt_q, obs_coords[nb_idx], obs_vals[nb_idx])


def class_for_dataset(dataset_key: str) -> type[nn.Module]:
    if dataset_key == "heat":
        return FieldFormerNPGFSparse
    if dataset_key == "swe":
        return FieldFormerNPGFSparseSWE
    if dataset_key in POLLUTION_KEYS:
        return FieldFormerNPGFSparsePollution
    raise KeyError(dataset_key)
