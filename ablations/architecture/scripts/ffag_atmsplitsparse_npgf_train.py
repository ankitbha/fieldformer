#!/usr/bin/env python3
"""FieldFormer sparse atmospheric held-out-sensor gamma-field ablation."""

from __future__ import annotations

from dataclasses import dataclass

from ffag_atmsparse_npgf_train import Config as BaseConfig
from ffag_sparse_npgf_common import train_sparse_npgf


@dataclass
class Config(BaseConfig):
    save: str = "/scratch/ab9738/fieldformer/ablations/architecture/checkpoints/ffag_npgf_atmsplitsparse_best.pt"


CFG = Config()


if __name__ == "__main__":
    train_sparse_npgf("atmsplit", CFG)
