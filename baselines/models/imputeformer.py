from __future__ import annotations

import torch
import torch.nn as nn
from einops import repeat


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool | None = False):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.mask = mask
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = torch.cat(torch.split(self.FC_Q(query), self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(self.FC_K(key), self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(self.FC_V(value), self.head_dim, dim=-1), dim=0)
        score = (query @ key.transpose(-1, -2)) / self.head_dim**0.5
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            score.masked_fill_(~mask, -torch.inf)
        out = torch.softmax(score, dim=-1) @ value
        out = torch.cat(torch.split(out, batch, dim=0), dim=-1)
        return self.out_proj(out)


class ProjectedAttentionLayer(nn.Module):
    def __init__(self, seq_len: int, dim_proj: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))
        self.out_attn = AttentionLayer(d_model, n_heads)
        self.in_attn = AttentionLayer(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        projector = repeat(self.projector, "p d -> b s p d", b=batch, s=self.seq_len)
        msg_out = self.out_attn(projector, x, x)
        msg_in = self.in_attn(x, projector, msg_out)
        x = self.norm1(x + self.dropout(msg_in))
        return self.norm2(x + self.dropout(self.mlp(x)))


class FixedNodeImputeFormer(nn.Module):
    """Fixed sensor/time masked-imputation baseline without query-specific neighborhoods."""

    def __init__(
        self,
        num_nodes: int,
        windows: int,
        input_dim: int = 2,
        output_dim: int = 1,
        input_embedding_dim: int = 32,
        learnable_embedding_dim: int = 96,
        num_layers: int = 3,
        num_temporal_heads: int = 4,
        dim_proj: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.windows = int(windows)
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(windows, num_nodes, learnable_embedding_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        model_dim = input_embedding_dim + learnable_embedding_dim
        self.layers = nn.ModuleList(
            [ProjectedAttentionLayer(windows, dim_proj, model_dim, num_temporal_heads, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(inplace=True), nn.Linear(model_dim, output_dim))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([values * mask, mask], dim=-1)
        x = self.input_proj(x)
        emb = self.node_embedding.permute(1, 0, 2).unsqueeze(0).expand(values.shape[0], self.num_nodes, self.windows, -1)
        x = torch.cat([x, emb], dim=-1).permute(0, 2, 1, 3)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1, 3)
        out = self.readout(x)
        return out


def old_checkpoint_message() -> str:
    return "ImputeFormer was refactored to fixed-node masked imputation; old local-neighborhood checkpoints must be retrained."
