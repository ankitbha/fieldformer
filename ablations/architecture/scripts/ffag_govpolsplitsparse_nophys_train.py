#!/usr/bin/env python3
"""FieldFormer sparse govpol held-out-sensor training: data loss only."""

from __future__ import annotations

from dataclasses import dataclass

from ffag_govpolsparse_nophys_train import Config as BaseConfig
from ffag_sparse_nophys_common import train_sparse_nophys


@dataclass
class Config(BaseConfig):
    data: str = "/scratch/ab9738/fieldformer/data/gov_sensor_dataset.npz"
    normalize_values: bool = True
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_govpolsplitsparse_nophys_best.pt"


CFG = Config()


if __name__ == "__main__":
    train_sparse_nophys("govpolsplit", CFG)
