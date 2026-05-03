#!/usr/bin/env python3
"""FieldFormer-Autograd sparse pollution ablation: observation-only training."""

from __future__ import annotations

from dataclasses import dataclass

from ffag_sparse_nophys_common import train_sparse_nophys


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
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
    grad_clip: float = 0.5
    patience: int = 10
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_polsparse_nophys_best.pt"


CFG = Config()


if __name__ == "__main__":
    train_sparse_nophys("pol", CFG)
