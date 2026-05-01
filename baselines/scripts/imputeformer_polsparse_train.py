#!/usr/bin/env python3
"""
ImputeFormer (sparse pollution): train only from sensor observations.

This is a standalone sparse adaptation of the ImputeFormer attention blocks for
the pollution sparse datasets used in this repository. It follows the same train
script structure as the Senseiver sparse baselines and does not add physics loss.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from imputeformer_sparse_common import add_config_args, run_sparse_experiment


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"  # or "U_sensor_clean"
    batch_size: int = 128
    val_batch_size: int = 256
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42
    k_neighbors: int = 128
    time_radius: int = 3
    input_dim: int = 5
    output_dim: int = 1
    input_embedding_dim: int = 32
    learnable_embedding_dim: int = 96
    feed_forward_dim: int = 256
    num_temporal_heads: int = 4
    num_layers: int = 3
    dim_proj: int = 8
    dropout: float = 0.1
    f1_loss_weight: float = 0.01
    grad_clip: float = 5.0
    patience: int = 20
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/imputeformer_polsparse_best.pt"


CFG = Config()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train ImputeFormer on sparse pollution sensor observations.")
    add_config_args(parser, CFG)
    return Config(**vars(parser.parse_args()))


def main(cfg: Config) -> None:
    run_sparse_experiment(
        cfg,
        variant="imputeformer_pollution_sparse",
        description="pollution",
        fallback_keys=("U_sensor_noisy", "U_sensor_clean"),
        error_name="pollution",
    )


if __name__ == "__main__":
    main(parse_args())
