#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **_: x

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from sparse_models import build_sparse_model


@dataclass
class Config:
    dataset: str = "heat"
    model: str = "siren"
    batch_size: int = 1024
    output_path: str = ""
    device: str = "cuda"
    obs_key: str = ""


DATASETS = {
    "heat": ROOT / "data" / "heat_periodic_dataset_sharp.npz",
    "swe": ROOT / "data" / "swe_periodic_dataset.npz",
    "pol": ROOT / "data" / "pollution_dataset.npz",
}
PINN_ALIASES = {"fmlp-pinn": "fmlp", "siren-pinn": "siren", "svgp-pinn": "svgp"}
MODELS = {
    "ffag",
    "siren",
    "siren-pinn",
    "fmlp",
    "fmlp-pinn",
    "svgp",
    "svgp-pinn",
    "recfno",
    "senseiver",
    "imputeformer",
}


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

    def __len__(self) -> int:
        return int(self.test_idx.numel())

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.test_idx[idx]


class FieldFormerNeighborIndexer:
    def __init__(self, sensors_xy: torch.Tensor, t_grid: torch.Tensor, time_radius: int, k_neighbors: int):
        self.sensors_xy = sensors_xy
        self.t_grid = t_grid
        self.S = sensors_xy.shape[0]
        self.Nt = t_grid.shape[0]
        self.time_radius = int(time_radius)
        self.k_neighbors = int(k_neighbors)

        sensor_ids = torch.arange(self.S, dtype=torch.long)
        offsets = torch.arange(-self.time_radius, self.time_radius + 1, dtype=torch.long)
        s_mesh, dt_mesh = torch.meshgrid(sensor_ids, offsets, indexing="ij")
        self.base_sensor = s_mesh.reshape(-1)
        self.base_dt = dt_mesh.reshape(-1)

    def lin_to_sk(self, lin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return lin // self.Nt, lin % self.Nt

    def sk_to_lin(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return s * self.Nt + k

    def gather_observed_neighbors(self, lin_q: torch.Tensor, exclude_self: bool = True) -> torch.Tensor:
        _, k_q = self.lin_to_sk(lin_q)
        bsz = lin_q.shape[0]
        s_nb = self.base_sensor.to(lin_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_q[:, None] + self.base_dt.to(lin_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)
        if exclude_self:
            lin_nb = lin_nb.masked_fill(lin_nb == lin_q[:, None], -1)
        if lin_nb.shape[1] > self.k_neighbors:
            lin_nb = lin_nb[:, : self.k_neighbors]
        if lin_nb.shape[1] < self.k_neighbors:
            pad = lin_nb[:, -1:].expand(-1, self.k_neighbors - lin_nb.shape[1])
            lin_nb = torch.cat([lin_nb, pad], dim=1)
        if (lin_nb < 0).any():
            lin_nb = torch.where(lin_nb < 0, lin_q[:, None].expand_as(lin_nb), lin_nb)
        return lin_nb

    def gather_continuous_neighbors(self, xyt_q: torch.Tensor) -> torch.Tensor:
        t_q = xyt_q[:, 2]
        dist = torch.abs(t_q[:, None] - self.t_grid[None, :].to(xyt_q.device))
        k_hat = torch.argmin(dist, dim=1)
        bsz = xyt_q.shape[0]
        s_nb = self.base_sensor.to(xyt_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_hat[:, None] + self.base_dt.to(xyt_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)
        if lin_nb.shape[1] > self.k_neighbors:
            lin_nb = lin_nb[:, : self.k_neighbors]
        if lin_nb.shape[1] < self.k_neighbors:
            pad = lin_nb[:, -1:].expand(-1, self.k_neighbors - lin_nb.shape[1])
            lin_nb = torch.cat([lin_nb, pad], dim=1)
        return lin_nb


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Evaluate sparse-trained models on sparse test and full-field metrics.")
    for field in fields(Config):
        value = getattr(cfg, field.name)
        parser.add_argument(f"--{field.name}", type=type(value), default=value)
    args = parser.parse_args()
    return Config(**vars(args))


def ckpt_path(model_key: str, dataset_key: str) -> Path:
    dataset_slug = {"pol": "pol"}.get(dataset_key, dataset_key)
    if model_key == "ffag":
        return ROOT / "fieldformer_core" / "checkpoints" / f"ffag_{dataset_slug}sparse_best.pt"
    return ROOT / "baselines" / "checkpoints" / f"{model_key}_{dataset_slug}sparse_best.pt"


def implementation_key(model_key: str) -> str:
    return PINN_ALIASES.get(model_key, model_key)


def available_checkpoints() -> str:
    roots = [ROOT / "fieldformer_core" / "checkpoints", ROOT / "baselines" / "checkpoints"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(str(p.relative_to(ROOT)) for p in sorted(root.glob("*sparse*_best.pt")))
    return "\n".join(files)


def choose_obs_key(pack: Any, dataset_key: str, override: str = "") -> str:
    if override:
        if override not in pack:
            raise KeyError(f"--obs_key {override!r} not found in dataset")
        return override
    candidates = {
        "swe": ("eta_sensor_noisy", "eta_sensor_clean"),
        "pol": ("U_sensor_noisy", "U_sensor_clean", "sensor_noisy", "sensor_clean"),
        "heat": ("sensor_noisy", "sensor_clean"),
    }[dataset_key]
    for key in candidates:
        if key in pack:
            return key
    raise KeyError(f"Could not find a sensor observation key. Tried: {candidates}")


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


def sparse_observations(pack: Any, dataset_key: str, obs_key: str) -> tuple[np.ndarray, np.ndarray]:
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    sensor_values = pack[obs_key].astype(np.float32)
    coords, vals = build_observed_tuples(sensors_xy, t_np, sensor_values)
    return coords, vals


def full_field(pack: Any, dataset_key: str) -> tuple[np.ndarray, np.ndarray]:
    x_np = pack["x"].astype(np.float32)
    y_np = pack["y"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    xx, yy, tt = np.meshgrid(x_np, y_np, t_np, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1).astype(np.float32)
    if dataset_key == "swe":
        vals = np.stack([pack["eta"], pack["u"], pack["v"]], axis=-1).reshape(-1, 3).astype(np.float32)
    elif dataset_key == "pol":
        vals = pack["U"].reshape(-1).astype(np.float32)
    else:
        vals = pack["u"].reshape(-1).astype(np.float32)
    return coords, vals


def align_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 1 and target.ndim == 2:
        return pred[:, None]
    if pred.ndim == 2 and target.ndim == 1:
        return pred[:, 0]
    if pred.ndim == 2 and target.ndim == 2 and pred.shape[1] != target.shape[1]:
        n = min(pred.shape[1], target.shape[1])
        return pred[:, :n]
    return pred


def metric_sums(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int]:
    pred = align_prediction(pred, target)
    target = align_prediction(target, pred)
    diff = pred - target
    return float((diff * diff).sum().item()), float(diff.abs().sum().item()), int(diff.numel())


def finish_metrics(se_sum: float, ae_sum: float, n_sum: int) -> dict[str, float]:
    n = max(1, n_sum)
    return {"rmse": math.sqrt(se_sum / n), "mae": ae_sum / n}


@torch.no_grad()
def eval_sparse_test(
    adapter: Any,
    indexer: FieldFormerNeighborIndexer | None,
    obs_coords: torch.Tensor,
    obs_vals: torch.Tensor,
    test_idx: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    se_sum, ae_sum, n_sum = 0.0, 0.0, 0
    starts = range(0, test_idx.numel(), batch_size)
    for start in tqdm(starts, desc="sparse-test", leave=False):
        q_lin = test_idx[start:start + batch_size].to(device)
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True) if adapter.needs_sensor_context else None
        pred = adapter.predict_observed(q_lin, obs_coords, obs_vals, nb_idx)
        tgt = obs_vals[q_lin]
        se, ae, n = metric_sums(pred, tgt)
        se_sum += se
        ae_sum += ae
        n_sum += n
    return finish_metrics(se_sum, ae_sum, n_sum)


@torch.no_grad()
def eval_full_field(
    adapter: Any,
    indexer: FieldFormerNeighborIndexer | None,
    obs_coords: torch.Tensor,
    obs_vals: torch.Tensor,
    full_coords: torch.Tensor,
    full_vals: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    se_sum, ae_sum, n_sum = 0.0, 0.0, 0
    starts = range(0, full_coords.shape[0], batch_size)
    for start in tqdm(starts, desc="full-field", leave=False):
        xyt = full_coords[start:start + batch_size].to(device)
        tgt = full_vals[start:start + batch_size].to(device)
        nb_idx = indexer.gather_continuous_neighbors(xyt) if adapter.needs_sensor_context else None
        pred = adapter.predict_continuous(xyt, obs_coords, obs_vals, nb_idx)
        se, ae, n = metric_sums(pred, tgt)
        se_sum += se
        ae_sum += ae
        n_sum += n
    return finish_metrics(se_sum, ae_sum, n_sum)


def write_output(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        flat = {
            "dataset": result["dataset"],
            "model": result["model"],
            "checkpoint": result["checkpoint"],
            "sparse_test_rmse": result["sparse_test"]["rmse"],
            "sparse_test_mae": result["sparse_test"]["mae"],
            "full_field_rmse": result["full_field"]["rmse"],
            "full_field_mae": result["full_field"]["mae"],
        }
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        return
    path.write_text(json.dumps(result, indent=2) + "\n")


def main(cfg: Config) -> None:
    dataset_key = cfg.dataset.lower()
    model_key = cfg.model.lower()
    if dataset_key not in DATASETS:
        raise SystemExit(f"Unknown dataset {cfg.dataset!r}. Expected one of: {sorted(DATASETS)}")
    if model_key not in MODELS:
        raise SystemExit(f"Unknown model {cfg.model!r}. Expected one of: {sorted(MODELS)}")

    path = ckpt_path(model_key, dataset_key)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}\n\nAvailable sparse checkpoints:\n{available_checkpoints()}")

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    pack = np.load(DATASETS[dataset_key])
    ckpt = torch.load(path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    obs_key = choose_obs_key(pack, dataset_key, cfg.obs_key)
    impl_model_key = implementation_key(model_key)
    obs_coords_np, obs_vals_np = sparse_observations(pack, dataset_key, obs_key)

    train_frac = float(ckpt_cfg.get("train_frac", 0.8))
    val_frac = float(ckpt_cfg.get("val_frac", 0.1))
    seed = int(ckpt_cfg.get("seed", 123))
    split = ObservedIndexDataset(obs_coords_np.shape[0], train_frac, val_frac, seed)
    train_vals = obs_vals_np[split.train_idx.numpy()]
    obs_mean = train_vals.mean(axis=0) if train_vals.ndim == 2 else float(train_vals.mean())
    obs_std = train_vals.std(axis=0) + 1e-6 if train_vals.ndim == 2 else float(train_vals.std() + 1e-6)
    if impl_model_key == "svgp" and dataset_key in {"heat", "swe"}:
        obs_mean = obs_vals_np.mean(axis=0) if obs_vals_np.ndim == 2 else float(obs_vals_np.mean())
        obs_std = obs_vals_np.std(axis=0) + 1e-8 if obs_vals_np.ndim == 2 else float(obs_vals_np.std() + 1e-8)
    meta = ckpt.get("meta", {})
    if impl_model_key == "svgp" and "obs_mean" in meta and "obs_std" in meta:
        obs_mean = meta["obs_mean"]
        obs_std = meta["obs_std"]

    x_np = pack["x"].astype(np.float32)
    y_np = pack["y"].astype(np.float32)
    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())
    Lx, Ly, Tt = max(1e-6, x_max - x_min), max(1e-6, y_max - y_min), max(1e-6, t_max - t_min)

    obs_coords = torch.from_numpy(obs_coords_np).float().to(device)
    obs_vals = torch.from_numpy(obs_vals_np).float().to(device)
    full_coords_np, full_vals_np = full_field(pack, dataset_key)
    full_coords = torch.from_numpy(full_coords_np).float()
    full_vals = torch.from_numpy(full_vals_np).float()

    indexer = None
    if impl_model_key == "ffag":
        indexer = FieldFormerNeighborIndexer(
            torch.from_numpy(sensors_xy).float().to(device),
            torch.from_numpy(t_np).float().to(device),
            int(ckpt_cfg.get("time_radius", 3)),
            int(ckpt_cfg.get("k_neighbors", 128)),
        )
    adapter = build_sparse_model(
        model_key=impl_model_key,
        dataset_key=dataset_key,
        ckpt=ckpt,
        data=pack,
        device=device,
        obs_mean=obs_mean,
        obs_std=obs_std,
        x_min=x_min,
        y_min=y_min,
        t_min=t_min,
        Lx=Lx,
        Ly=Ly,
        Tt=Tt,
        nt_count=t_np.shape[0],
    )

    sparse_metrics = eval_sparse_test(adapter, indexer, obs_coords, obs_vals, split.test_idx, cfg.batch_size, device)
    full_metrics = eval_full_field(adapter, indexer, obs_coords, obs_vals, full_coords, full_vals, cfg.batch_size, device)
    result = {
        "dataset": dataset_key,
        "model": model_key,
        "checkpoint": str(path),
        "obs_key": obs_key,
        "num_sparse_test": int(split.test_idx.numel()),
        "num_full_field": int(full_coords.shape[0]),
        "sparse_test": sparse_metrics,
        "full_field": full_metrics,
    }

    print(f"[eval] dataset={dataset_key} model={model_key} checkpoint={path.relative_to(ROOT)}")
    print("")
    print(f"{'metric':<18} {'rmse':>14} {'mae':>14}")
    print(f"{'-' * 48}")
    print(f"{'sparse_test':<18} {sparse_metrics['rmse']:>14.6g} {sparse_metrics['mae']:>14.6g}")
    print(f"{'full_field':<18} {full_metrics['rmse']:>14.6g} {full_metrics['mae']:>14.6g}")

    if cfg.output_path:
        out = Path(cfg.output_path)
        write_output(out, result)
        print(f"\n[write] {out}")


if __name__ == "__main__":
    main(parse_args())
