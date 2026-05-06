#!/usr/bin/env python3
"""FieldFormer sparse atmospheric training: data loss only."""

from __future__ import annotations

import time
from dataclasses import dataclass

_DEBUG_START = time.monotonic()


def _debug(msg: str) -> None:
    elapsed = time.monotonic() - _DEBUG_START
    print(f"[debug:atm-entry +{elapsed:7.2f}s] {msg}", flush=True)


_debug("entrypoint module import started")
_debug("importing shared no-physics trainer")
from ffag_sparse_nophys_common import train_sparse_nophys
_debug("imported shared no-physics trainer")


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/gov_atm_dataset.npz"
    obs_key: str = "U_sensor"
    mask_key: str = "U_sensor_mask"
    normalize_values: bool = True
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
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_atmsparse_nophys_best.pt"


CFG = Config()


if __name__ == "__main__":
    _debug(f"starting train_sparse_nophys with data={CFG.data}")
    train_sparse_nophys("atm", CFG)
