from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


HEAT_DATA = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
SWE_DATA = "/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz"
POLLUTION_DATA = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"


class ObservedIndexDataset(Dataset):
    def __init__(
        self,
        n_obs: int,
        train_frac: float,
        val_frac: float,
        seed: int,
        valid_idx: np.ndarray | torch.Tensor | None = None,
    ):
        rng = np.random.default_rng(seed)
        all_idx = np.asarray(valid_idx if valid_idx is not None else np.arange(n_obs), dtype=np.int64)
        rng.shuffle(all_idx)
        n_split = int(all_idx.shape[0])
        n_train = int(train_frac * n_split)
        n_val = int(val_frac * n_split)
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


def build_observed_tuples(
    sensors_xy: np.ndarray,
    t_grid: np.ndarray,
    sensor_values: np.ndarray,
    sensor_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sensor_values.ndim == 2:
        s_count, nt_count = sensor_values.shape
    elif sensor_values.ndim == 3:
        s_count, nt_count, _ = sensor_values.shape
    else:
        raise ValueError("sensor_values must have shape (S,T) or (S,T,C)")
    coords = np.stack(
        [
            np.repeat(sensors_xy[:, 0], nt_count),
            np.repeat(sensors_xy[:, 1], nt_count),
            np.tile(t_grid, s_count),
        ],
        axis=1,
    ).astype(np.float32)
    vals = sensor_values.reshape(s_count * nt_count, *sensor_values.shape[2:]).astype(np.float32)
    if sensor_mask is None:
        return coords, vals
    if sensor_mask.shape != sensor_values.shape:
        raise ValueError(f"sensor_mask shape {sensor_mask.shape} does not match values {sensor_values.shape}")
    mask = sensor_mask.reshape(s_count * nt_count, *sensor_mask.shape[2:]).astype(np.float32)
    return coords, vals, mask


def sensor_key(pack: np.lib.npyio.NpzFile, dataset: str, override: str = "") -> str:
    if override:
        return override
    candidates = {
        "heat": ("sensor_noisy", "sensor_clean"),
        "swe": ("eta_sensor_noisy", "eta_sensor_clean"),
        "pol": ("U_sensor_noisy", "U_sensor_clean", "sensor_noisy", "sensor_clean"),
        "govpol": ("U_sensor", "U_sensor_noisy", "U_sensor_clean"),
    }[dataset]
    for key in candidates:
        if key in pack:
            return key
    raise KeyError(f"No sensor key found for {dataset}; tried {candidates}")


def mask_key(pack: np.lib.npyio.NpzFile, override: str = "") -> str:
    if override:
        return override
    for key in ("U_sensor_mask", "sensor_mask", "obs_mask"):
        if key in pack:
            return key
    return ""
