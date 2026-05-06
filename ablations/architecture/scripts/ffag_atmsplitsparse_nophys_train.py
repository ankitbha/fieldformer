#!/usr/bin/env python3
"""FieldFormer sparse atmospheric held-out-sensor training: data loss only."""

from __future__ import annotations

from dataclasses import dataclass

from ffag_atmsparse_nophys_train import Config as BaseConfig
from ffag_sparse_nophys_common import train_sparse_nophys


@dataclass
class Config(BaseConfig):
    data: str = "/scratch/ab9738/fieldformer/data/gov_atm_dataset.npz"
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_atmsplitsparse_nophys_best.pt"


CFG = Config()


if __name__ == "__main__":
    train_sparse_nophys("atmsplit", CFG)
