#!/usr/bin/env python3
"""SIREN sparse pollution training without FieldFormer local context."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.models.siren import SIRENPollution
from baselines.scripts.coordinate_sparse_train import train_coordinate_sparse


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
    weight_decay: float = 1e-4
    width: int = 256
    depth: int = 6
    w0: float = 30.0
    w0_hidden: float = 1.0
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
    sponge_samples: int = 512
    rad_samples: int = 512
    sponge_border_frac: float = 0.05
    sponge_warmup: int = 0
    sponge_ramp: int = 10
    rad_warmup: int = 5
    rad_ramp: int = 20
    c_cap: float = 2.0
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    patience: int = 10


CFG = Config()
MODEL_CLASS = SIRENPollution


def main(cfg: Config = CFG) -> None:
    train_coordinate_sparse(
        cfg,
        "siren",
        lambda c: SIRENPollution(in_dim=3, width=c.width, depth=c.depth, out_dim=1, w0=c.w0, w0_hidden=c.w0_hidden),
    )


if __name__ == "__main__":
    main(CFG)
