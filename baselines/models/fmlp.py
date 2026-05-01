from __future__ import annotations

import math

import torch
import torch.nn as nn


def _harmonics(k_max: int, device: torch.device) -> torch.Tensor:
    return torch.arange(1, k_max + 1, dtype=torch.float32, device=device)


class FourierMLP(nn.Module):
    """Coordinate-only Fourier-feature MLP baseline."""

    def __init__(
        self,
        width: int = 256,
        depth: int = 6,
        kx: int = 16,
        ky: int = 16,
        kt: int = 8,
        out_dim: int = 1,
    ):
        super().__init__()
        self.kx = int(kx)
        self.ky = int(ky)
        self.kt = int(kt)
        self.out_dim = int(out_dim)
        in_dim = 2 * (self.kx + self.ky + self.kt)
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, self.out_dim))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _encode_1d(x: torch.Tensor, ks: torch.Tensor, length: float) -> torch.Tensor:
        z = (2.0 * math.pi) * (x[..., None] / max(float(length), 1e-6)) * ks[None, :]
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    def forward(self, xyt: torch.Tensor, *, Lx: float = 1.0, Ly: float = 1.0, Tt: float = 1.0) -> torch.Tensor:
        dev = xyt.device
        feat = torch.cat(
            [
                self._encode_1d(xyt[:, 0], _harmonics(self.kx, dev), Lx),
                self._encode_1d(xyt[:, 1], _harmonics(self.ky, dev), Ly),
                self._encode_1d(xyt[:, 2], _harmonics(self.kt, dev), Tt),
            ],
            dim=-1,
        )
        out = self.net(feat)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class FourierMLPSWE(FourierMLP):
    def __init__(self, width: int = 256, depth: int = 6, kx: int = 16, ky: int = 16, kt: int = 8):
        super().__init__(width=width, depth=depth, kx=kx, ky=ky, kt=kt, out_dim=3)


class FourierMLPPollution(FourierMLP):
    def __init__(
        self,
        width: int = 256,
        depth: int = 6,
        kx: int = 16,
        ky: int = 16,
        kt: int = 8,
        out_dim: int = 1,
    ):
        super().__init__(width=width, depth=depth, kx=kx, ky=ky, kt=kt, out_dim=out_dim)
