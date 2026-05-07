#!/usr/bin/env python3
"""FieldFormer sparse real pollution gamma-field ablation: observation-only training."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ffag_sparse_npgf_common import train_sparse_npgf

_DEBUG_START = time.monotonic()


def _debug(msg: str) -> None:
    elapsed = time.monotonic() - _DEBUG_START
    print(f"[debug:govpol-npgf-entry +{elapsed:7.2f}s] {msg}", flush=True)


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/gov_sensor_dataset.npz"
    obs_key: str = "U_sensor"
    mask_key: str = "U_sensor_mask"
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
    gamma_field_lr: float = 3e-4
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
    gamma_hidden: int = 32
    gamma_layers: int = 2
    gamma_max_delta: float = 2.0
    grad_clip: float = 0.5
    patience: int = 10
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_npgf_govpolsparse_best.pt"


CFG = Config()


if __name__ == "__main__":
    _debug(f"starting train_sparse_npgf with data={CFG.data}")
    train_sparse_npgf("govpol", CFG)
