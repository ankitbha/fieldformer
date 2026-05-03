#!/usr/bin/env python3
"""FieldFormer sparse heat architecture ablation: MLP token mixer."""

from __future__ import annotations

from dataclasses import dataclass

from ffag_sparse_mlp_common import train_periodic_mlp


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
    obs_key: str = "sensor_noisy"
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 80
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
    lambda_phys: float = 0.2
    lambda_bc: float = 0.2
    phys_samples: int = 1024
    bc_samples: int = 512
    match_grad_bc: bool = False
    grad_clip: float = 1.0
    patience: int = 12
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_mlp_heatsparse_best.pt"


CFG = Config()


if __name__ == "__main__":
    train_periodic_mlp("heat", CFG)
