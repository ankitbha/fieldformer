from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


HEAT_DATA = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
SWE_DATA = "/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz"
POLLUTION_DATA = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"


class ObservedIndexDataset(Dataset):
    def __init__(self, n_obs: int, train_frac: float, val_frac: float, seed: int):
        rng = np.random.default_rng(seed)
        all_idx = np.arange(n_obs)
        rng.shuffle(all_idx)
        n_train = int(train_frac * n_obs)
        n_val = int(val_frac * n_obs)
        self.train_idx = torch.from_numpy(all_idx[:n_train]).long()
        self.val_idx = torch.from_numpy(all_idx[n_train:n_train + n_val]).long()
        self.test_idx = torch.from_numpy(all_idx[n_train + n_val:]).long()
        self.split = "train"

    def set_split(self, split: str) -> None:
        assert split in {"train", "val", "test"}
        self.split = split

    def __len__(self) -> int:
        return len(getattr(self, f"{self.split}_idx"))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return getattr(self, f"{self.split}_idx")[idx]


def build_observed_tuples(sensors_xy: np.ndarray, t_grid: np.ndarray, sensor_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_count, nt_count = sensor_values.shape
    coords = np.stack(
        [
            np.repeat(sensors_xy[:, 0], nt_count),
            np.repeat(sensors_xy[:, 1], nt_count),
            np.tile(t_grid, s_count),
        ],
        axis=1,
    ).astype(np.float32)
    vals = sensor_values.reshape(-1).astype(np.float32)
    return coords, vals


def sensor_key(pack: np.lib.npyio.NpzFile, dataset: str, override: str = "") -> str:
    if override:
        return override
    candidates = {
        "heat": ("sensor_noisy", "sensor_clean"),
        "swe": ("eta_sensor_noisy", "eta_sensor_clean"),
        "pol": ("U_sensor_noisy", "U_sensor_clean", "sensor_noisy", "sensor_clean"),
    }[dataset]
    for key in candidates:
        if key in pack:
            return key
    raise KeyError(f"No sensor key found for {dataset}; tried {candidates}")

