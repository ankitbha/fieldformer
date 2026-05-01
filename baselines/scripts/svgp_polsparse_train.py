#!/usr/bin/env python3
"""SVGP sparse pollution training without FieldFormer local context."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.models.svgp import PollutionSVGP
from baselines.scripts.svgp_sparse_train import train_svgp_sparse


@dataclass
class Config:
    dataset: str = "pol"
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"
    save: str = ""
    pinn: bool = False
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    batch_size: int = 2048
    val_batch_size: int = 4096
    epochs: int = 300
    lr: float = 3e-4
    lr_noise: float = 5e-3
    weight_decay: float = 1e-4
    inducing_points: int = 512
    noise: float = 1e-3
    lambda_phys: float = 0.0
    lambda_bc: float = 0.0
    phys_samples: int = 1024
    bc_samples: int = 512
    phys_warmup: int = 0
    phys_ramp: int = 1
    bc_warmup: int = 0
    bc_ramp: int = 1
    match_grad_bc: bool = False
    lambda_sponge: float = 0.0
    lambda_rad: float = 0.0
    ard_lengthscale_init: tuple[float, float, float] = (0.2, 0.2, 0.1)
    outputscale_init: float = 1.0
    grad_clip: float = 1.0
    patience: int = 10


CFG = Config()
MODEL_CLASS = PollutionSVGP


def main(cfg: Config = CFG) -> None:
    train_svgp_sparse(cfg)


if __name__ == "__main__":
    main(CFG)
