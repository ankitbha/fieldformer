from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False):
        super().__init__(in_features, out_features)
        self.w0 = float(w0)
        with torch.no_grad():
            bound = 1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / self.w0
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * F.linear(x, self.weight, self.bias))


class SIREN(nn.Module):
    """Coordinate-only SIREN baseline."""

    def __init__(
        self,
        in_dim: int = 3,
        width: int = 256,
        depth: int = 6,
        out_dim: int = 1,
        w0: float = 30.0,
        w0_hidden: float = 1.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        layers = [SineLayer(in_dim, width, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(width, width, w0=w0_hidden, is_first=False))
        self.hidden = nn.ModuleList(layers)
        self.final = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / width) / w0_hidden
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        h = xyt
        for layer in self.hidden:
            h = layer(h)
        out = self.final(h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class SIRENSWE(SIREN):
    def __init__(self, width: int = 256, depth: int = 6, w0: float = 30.0, w0_hidden: float = 1.0):
        super().__init__(in_dim=3, width=width, depth=depth, out_dim=3, w0=w0, w0_hidden=w0_hidden)


class SIRENPollution(SIREN):
    def __init__(
        self,
        in_dim: int = 3,
        width: int = 256,
        depth: int = 6,
        out_dim: int = 1,
        w0: float = 30.0,
        w0_hidden: float = 1.0,
    ):
        super().__init__(in_dim=in_dim, width=width, depth=depth, out_dim=out_dim, w0=w0, w0_hidden=w0_hidden)
