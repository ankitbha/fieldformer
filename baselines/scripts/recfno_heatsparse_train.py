#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.models.recfno import VoronoiFNO2d
from baselines.scripts.fair_sparse_train import train_recfno


@dataclass
class Config:
    dataset: str = "heat"
    data: str = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
    obs_key: str = "sensor_noisy"
    save: str = ""
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    batch_size: int = 64
    val_batch_size: int = 128
    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 1e-4
    modes1: int = 12
    modes2: int = 12
    width: int = 32
    grad_clip: float = 1.0
    patience: int = 12


CFG = Config()
MODEL_CLASS = VoronoiFNO2d


def main(cfg: Config = CFG) -> None:
    train_recfno(cfg)


if __name__ == "__main__":
    main(CFG)
