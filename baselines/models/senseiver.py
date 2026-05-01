from __future__ import annotations

import torch
import torch.nn as nn
from einops import repeat


class Sequential(nn.Sequential):
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for module in self:
            x = module(*x) if isinstance(x, tuple) else module(x)
        return x


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.module(*args, **kwargs)) + args[0]


def mlp(channels: int) -> Sequential:
    return Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))


class MultiHeadAttention(nn.Module):
    def __init__(self, q_channels: int, kv_channels: int, heads: int, dropout: float):
        super().__init__()
        if q_channels % heads != 0:
            raise ValueError(f"q_channels={q_channels} must be divisible by heads={heads}")
        self.attn = nn.MultiheadAttention(q_channels, heads, kdim=kv_channels, vdim=kv_channels, dropout=dropout, batch_first=True)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        return self.attn(x_q, x_kv, x_kv)[0]


class CrossAttention(nn.Module):
    def __init__(self, q_channels: int, kv_channels: int, heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_channels)
        self.kv_norm = nn.LayerNorm(kv_channels)
        self.attn = MultiHeadAttention(q_channels, kv_channels, heads, dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        return self.attn(self.q_norm(x_q), self.kv_norm(x_kv))


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = MultiHeadAttention(channels, channels, heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return self.attn(z, z)


def cross_layer(q_channels: int, kv_channels: int, heads: int, dropout: float) -> Sequential:
    return Sequential(Residual(CrossAttention(q_channels, kv_channels, heads, dropout), dropout), Residual(mlp(q_channels), dropout))


def self_block(layers: int, channels: int, heads: int, dropout: float) -> Sequential:
    return Sequential(*[Sequential(Residual(SelfAttention(channels, heads, dropout), dropout), Residual(mlp(channels), dropout)) for _ in range(layers)])


class Senseiver(nn.Module):
    """Sensor-set encoder/query-decoder baseline without per-query local neighborhoods."""

    def __init__(
        self,
        sensor_feature_dim: int,
        query_feature_dim: int,
        num_latents: int = 16,
        latent_channels: int = 64,
        num_layers: int = 3,
        cross_heads: int = 4,
        decoder_heads: int = 1,
        self_heads: int = 4,
        self_layers: int = 2,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(num_latents, latent_channels))
        nn.init.normal_(self.latent, 0.0, 0.02)
        self.encoder_1 = Sequential(cross_layer(latent_channels, sensor_feature_dim, cross_heads, dropout), self_block(self_layers, latent_channels, self_heads, dropout))
        self.encoder_n = Sequential(cross_layer(latent_channels, sensor_feature_dim, cross_heads, dropout), self_block(self_layers, latent_channels, self_heads, dropout))
        self.num_layers = int(num_layers)
        self.query_seed = nn.Parameter(torch.empty(1, latent_channels))
        nn.init.normal_(self.query_seed, 0.0, 0.02)
        q_channels = query_feature_dim + latent_channels
        self.decoder = cross_layer(q_channels, latent_channels, decoder_heads, dropout)
        self.readout = nn.Linear(q_channels, out_dim)

    def forward(self, sensor_tokens: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        bsz, n_query, _ = query_tokens.shape
        latents = repeat(self.latent, "n c -> b n c", b=bsz)
        latents = self.encoder_1(latents, sensor_tokens)
        for _ in range(self.num_layers - 1):
            latents = self.encoder_n(latents, sensor_tokens)
        seed = self.query_seed.expand(bsz, n_query, -1)
        q = torch.cat([query_tokens, seed], dim=-1)
        out = self.decoder(q, latents)
        out = self.readout(out)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def old_checkpoint_message() -> str:
    return "Senseiver was refactored to sensor-set conditioning; old local-neighborhood checkpoints must be retrained."
