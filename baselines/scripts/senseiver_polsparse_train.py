#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.models.senseiver import Senseiver
from baselines.scripts.fair_sparse_train import train_senseiver


@dataclass
class Config:
    dataset: str = "pol"
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"
    save: str = ""
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    batch_size: int = 256
    val_batch_size: int = 512
    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_latents: int = 16
    latent_channels: int = 64
    num_layers: int = 3
    cross_heads: int = 4
    decoder_heads: int = 1
    self_heads: int = 4
    self_layers: int = 2
    dropout: float = 0.0
    grad_clip: float = 1.0
    patience: int = 12


CFG = Config()
MODEL_CLASS = Senseiver


def main(cfg: Config = CFG) -> None:
    train_senseiver(cfg)


if __name__ == "__main__":
    main(CFG)
