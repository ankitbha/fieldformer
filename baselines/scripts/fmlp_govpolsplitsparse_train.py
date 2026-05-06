#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.scripts.fmlp_govpolsparse_train import Config as BaseConfig, main


@dataclass
class Config(BaseConfig):
    dataset: str = "govpolsplit"
    data: str = "/scratch/ab9738/fieldformer/data/gov_sensor_dataset.npz"
    normalize_values: bool = True


CFG = Config()


if __name__ == "__main__":
    main(CFG)
