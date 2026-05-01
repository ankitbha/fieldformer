#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.models.imputeformer import FixedNodeImputeFormer
from baselines.scripts.fair_sparse_train import train_imputeformer


@dataclass
class Config:
    dataset: str = "pol"
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"
    save: str = ""
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 1e-4
    windows: int = 128
    window_stride: int = 64
    mask_rate: float = 0.25
    input_embedding_dim: int = 32
    learnable_embedding_dim: int = 96
    num_layers: int = 3
    num_temporal_heads: int = 4
    dim_proj: int = 8
    dropout: float = 0.1
    grad_clip: float = 1.0
    patience: int = 12


CFG = Config()
MODEL_CLASS = FixedNodeImputeFormer


def main(cfg: Config = CFG) -> None:
    train_imputeformer(cfg)


if __name__ == "__main__":
    main(CFG)
