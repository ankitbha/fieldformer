from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


HEAT_DATA = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
SWE_DATA = "/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz"
POLLUTION_DATA = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
ATM_DATA = "/scratch/ab9738/fieldformer/data/gov_atm_dataset.npz"
SENSOR_SPLIT_DATASETS = {"govpolsplit", "atmsplit"}
SPLIT_BASE_DATASET = {"govpolsplit": "govpol", "atmsplit": "atm"}


def base_dataset_key(dataset: str) -> str:
    return SPLIT_BASE_DATASET.get(str(dataset), str(dataset))


def is_sensor_split_dataset(dataset: str) -> bool:
    return str(dataset) in SENSOR_SPLIT_DATASETS


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


class PrecomputedObservedIndexDataset(Dataset):
    def __init__(self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, meta: dict | None = None):
        self.train_idx = torch.from_numpy(np.asarray(train_idx, dtype=np.int64)).long()
        self.val_idx = torch.from_numpy(np.asarray(val_idx, dtype=np.int64)).long()
        self.test_idx = torch.from_numpy(np.asarray(test_idx, dtype=np.int64)).long()
        self.meta = meta or {}
        self.split = "train"

    def set_split(self, split: str) -> None:
        assert split in {"train", "val", "test"}
        self.split = split

    def __len__(self) -> int:
        return len(getattr(self, f"{self.split}_idx"))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return getattr(self, f"{self.split}_idx")[idx]


def _sensor_ids(pack: np.lib.npyio.NpzFile, n_sensors: int) -> list[str]:
    if "monitor_ids" in pack:
        return [str(x) for x in pack["monitor_ids"].tolist()]
    return [f"sensor_{i}" for i in range(n_sensors)]


def build_sensor_holdout_split(
    *,
    pack: np.lib.npyio.NpzFile,
    sensor_mask: np.ndarray,
    seed: int,
    val_sensors: int = 3,
    test_sensors: int = 3,
    min_valid_frac: float = 0.10,
    require_all_channels: bool = False,
) -> tuple[PrecomputedObservedIndexDataset, dict]:
    mask = sensor_mask.astype(bool)
    if mask.ndim == 2:
        mask = mask[..., None]
    n_sensors, n_times, _ = mask.shape
    threshold = max(1, int(np.ceil(float(min_valid_frac) * n_times)))
    per_sensor_channel_counts = mask.sum(axis=1)
    eligible = np.flatnonzero((per_sensor_channel_counts >= threshold).all(axis=1))
    need = int(val_sensors) + int(test_sensors)
    if eligible.shape[0] < need:
        raise ValueError(f"Need {need} eligible sensors, found {eligible.shape[0]} with min_valid_frac={min_valid_frac}")

    rng = np.random.default_rng(seed)
    shuffled = eligible.copy()
    rng.shuffle(shuffled)
    val_sensor_idx = np.sort(shuffled[: int(val_sensors)])
    test_sensor_idx = np.sort(shuffled[int(val_sensors):need])
    held = set(val_sensor_idx.tolist()) | set(test_sensor_idx.tolist())
    train_sensor_idx = np.asarray([i for i in range(n_sensors) if i not in held], dtype=np.int64)

    flat_valid = mask.reshape(n_sensors, n_times, -1)
    point_valid = flat_valid.all(axis=2) if require_all_channels else flat_valid.any(axis=2)

    def flat_indices(sensor_idx: np.ndarray) -> np.ndarray:
        if sensor_idx.size == 0:
            return np.asarray([], dtype=np.int64)
        rows = point_valid[sensor_idx]
        ss, tt = np.nonzero(rows)
        return (sensor_idx[ss] * n_times + tt).astype(np.int64)

    train_idx = flat_indices(train_sensor_idx)
    val_idx = flat_indices(val_sensor_idx)
    test_idx = flat_indices(test_sensor_idx)
    sensor_ids = _sensor_ids(pack, n_sensors)
    meta = {
        "split_type": "sensor_holdout",
        "sensor_split_seed": int(seed),
        "val_sensors": int(val_sensors),
        "test_sensors": int(test_sensors),
        "min_valid_frac": float(min_valid_frac),
        "require_all_channels": bool(require_all_channels),
        "min_valid_per_channel": int(threshold),
        "eligible_sensor_indices": eligible.astype(int).tolist(),
        "eligible_sensor_ids": [sensor_ids[i] for i in eligible],
        "train_sensor_indices": train_sensor_idx.astype(int).tolist(),
        "val_sensor_indices": val_sensor_idx.astype(int).tolist(),
        "test_sensor_indices": test_sensor_idx.astype(int).tolist(),
        "train_sensor_ids": [sensor_ids[i] for i in train_sensor_idx],
        "val_sensor_ids": [sensor_ids[i] for i in val_sensor_idx],
        "test_sensor_ids": [sensor_ids[i] for i in test_sensor_idx],
    }
    return PrecomputedObservedIndexDataset(train_idx, val_idx, test_idx, meta), meta


def build_observed_index_dataset(
    *,
    dataset_key: str,
    pack: np.lib.npyio.NpzFile,
    n_obs: int,
    train_frac: float,
    val_frac: float,
    seed: int,
    valid_idx: np.ndarray | torch.Tensor | None = None,
    sensor_mask: np.ndarray | None = None,
    sensor_split_seed: int | None = None,
    val_sensors: int = 3,
    test_sensors: int = 3,
    min_valid_frac: float = 0.10,
    require_all_channels: bool = False,
) -> PrecomputedObservedIndexDataset | ObservedIndexDataset:
    if is_sensor_split_dataset(dataset_key):
        if sensor_mask is None:
            raise ValueError(f"{dataset_key} requires sensor_mask for sensor holdout split")
        split_seed = int(seed if sensor_split_seed is None else sensor_split_seed)
        split, _ = build_sensor_holdout_split(
            pack=pack,
            sensor_mask=sensor_mask,
            seed=split_seed,
            val_sensors=val_sensors,
            test_sensors=test_sensors,
            min_valid_frac=min_valid_frac,
            require_all_channels=require_all_channels,
        )
        return split
    return ObservedIndexDataset(n_obs, train_frac, val_frac, seed, valid_idx=valid_idx)


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
        "atm": ("U_sensor", "U_sensor_noisy", "U_sensor_clean"),
    }[base_dataset_key(dataset)]
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
