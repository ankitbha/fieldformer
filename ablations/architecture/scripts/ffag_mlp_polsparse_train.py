#!/usr/bin/env python3
"""FieldFormer sparse pollution architecture ablation: MLP token mixer."""

from __future__ import annotations

from dataclasses import dataclass

from ffag_sparse_mlp_common import train_pollution_mlp


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
    gamma_lr: float = 1e-3
    weight_decay: float = 1e-4
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    k_neighbors: int = 128
    time_radius: int = 3
    d_model: int = 128
    nhead: int = 4
    layers: int = 3
    d_ff: int = 256
    lambda_sponge: float = 0.03
    lambda_rad: float = 0.01
    sponge_samples: int = 512
    rad_samples: int = 512
    sponge_border_frac: float = 0.05
    rad_warmup: int = 5
    rad_ramp: int = 20
    sponge_warmup: int = 0
    sponge_ramp: int = 10
    c_cap: float = 2.0
    huber_delta: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 0.5
    patience: int = 10
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_mlp_polsparse_best.pt"


CFG = Config()


if __name__ == "__main__":
    train_pollution_mlp(CFG)
