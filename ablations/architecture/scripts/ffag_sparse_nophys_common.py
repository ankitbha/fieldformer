#!/usr/bin/env python3
"""Shared observation-only trainer for sparse FieldFormer architecture ablations."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fieldformer_core.models.ffag import class_for_dataset, module_for_dataset


def _core_symbols(dataset_key: str) -> dict[str, Any]:
    mod = module_for_dataset(dataset_key)
    return {
        "set_seed": mod.set_seed,
        "ObservedIndexDataset": mod.ObservedIndexDataset,
        "SparseNeighborIndexer": mod.SparseNeighborIndexer,
        "EarlyStopping": mod.EarlyStopping,
        "build_observed_tuples": mod.build_observed_tuples,
    }


def _load_observations(pack: np.lib.npyio.NpzFile, dataset_key: str, obs_key: str) -> np.ndarray:
    fallbacks = {
        "heat": ("sensor_noisy", "sensor_clean"),
        "swe": ("eta_sensor_noisy", "eta_sensor_clean"),
        "pol": ("U_sensor_noisy", "U_sensor_clean", "sensor_noisy", "sensor_clean"),
    }[dataset_key]
    for key in (obs_key, *fallbacks):
        if key in pack:
            return pack[key].astype(np.float32)
    raise KeyError(f"Could not find observation key. Tried: {(obs_key, *fallbacks)}")


def _domain_extents(pack: np.lib.npyio.NpzFile, sensors_xy: np.ndarray, t_np: np.ndarray) -> dict[str, float]:
    if "x" in pack and "y" in pack:
        x_np = pack["x"].astype(np.float32)
        y_np = pack["y"].astype(np.float32)
        x_min, x_max = float(x_np.min()), float(x_np.max())
        y_min, y_max = float(y_np.min()), float(y_np.max())
        dx = float(x_np[1] - x_np[0]) if x_np.size > 1 else 1.0
        dy = float(y_np[1] - y_np[0]) if y_np.size > 1 else 1.0
    else:
        x_min, x_max = float(sensors_xy[:, 0].min()), float(sensors_xy[:, 0].max())
        y_min, y_max = float(sensors_xy[:, 1].min()), float(sensors_xy[:, 1].max())
        dx = dy = 1.0

    t_min, t_max = float(t_np.min()), float(t_np.max())
    dt = float(t_np[1] - t_np[0]) if t_np.size > 1 else 1.0
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "t_min": t_min,
        "t_max": t_max,
        "Lx": max(1e-6, x_max - x_min),
        "Ly": max(1e-6, y_max - y_min),
        "dx": dx,
        "dy": dy,
        "dt": dt,
    }


def _sensors_are_aligned(pack: np.lib.npyio.NpzFile, sensors_xy: np.ndarray, dataset_key: str) -> None:
    if not all(k in pack for k in ["sensors_idx", "x", "y"]):
        return
    try:
        sidx = pack["sensors_idx"]
        x_np, y_np = pack["x"], pack["y"]
        if dataset_key == "pol":
            okx = np.allclose(sensors_xy[:, 0], x_np[sidx[:, 1]], atol=1e-6)
            oky = np.allclose(sensors_xy[:, 1], y_np[sidx[:, 0]], atol=1e-6)
        else:
            okx = np.allclose(sensors_xy[:, 0], x_np[sidx[:, 0]], atol=1e-6)
            oky = np.allclose(sensors_xy[:, 1], y_np[sidx[:, 1]], atol=1e-6)
        if not (okx and oky):
            print("[warn] sensors_xy and sensors_idx/x/y are not exactly aligned.")
    except Exception:
        print("[warn] skipped sensors_idx consistency check due to shape/index mismatch.")


def train_sparse_nophys(dataset_key: str, cfg: Any) -> None:
    core = _core_symbols(dataset_key)
    core["set_seed"](cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    sensor_values = _load_observations(pack, dataset_key, cfg.obs_key)

    assert sensors_xy.ndim == 2 and sensors_xy.shape[1] == 2, "sensors_xy must be (S,2)"
    assert sensor_values.ndim == 2, "sensor values must be (S,Nt)"
    n_sensors, n_times = sensor_values.shape
    assert t_np.shape[0] == n_times, "time grid length must match sensor series"
    _sensors_are_aligned(pack, sensors_xy, dataset_key)

    coords_np, vals_np = core["build_observed_tuples"](sensors_xy, t_np, sensor_values)
    n_obs = coords_np.shape[0]
    domain = _domain_extents(pack, sensors_xy, t_np)

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    sensors_xy_t = torch.from_numpy(sensors_xy).float().to(device)
    t_grid_t = torch.from_numpy(t_np).float().to(device)

    dataset_cls = core["ObservedIndexDataset"]
    ds = dataset_cls(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    ds_val = dataset_cls(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    indexer = core["SparseNeighborIndexer"](sensors_xy_t, t_grid_t, cfg.time_radius, cfg.k_neighbors)
    model_cls = class_for_dataset(dataset_key)
    model = model_cls(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff).to(device)
    if dataset_key == "pol":
        with torch.no_grad():
            model.log_gammas[:] = torch.log(torch.tensor([1.0, 1.0, 0.5], device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    stopper = core["EarlyStopping"](patience=cfg.patience)

    def predict_observed(q_lin: torch.Tensor) -> torch.Tensor:
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        if dataset_key == "pol":
            return model.forward_observed(q_lin, obs_coords, obs_vals, nb_idx)
        return model.forward_observed(q_lin, obs_coords, obs_vals, nb_idx, Lx=domain["Lx"], Ly=domain["Ly"])

    def observed_channel(pred: torch.Tensor) -> torch.Tensor:
        return pred[:, 0] if dataset_key == "swe" else pred

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred = observed_channel(predict_observed(q_lin))
            tgt = obs_vals[q_lin]
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_data, n_batches = 0.0, 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred = observed_channel(predict_observed(q_lin))
            loss = F.mse_loss(pred, obs_vals[q_lin])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_data += loss.item()
            n_batches += 1
            pbar.set_postfix({"data": f"{running_data / max(1, n_batches):.4e}"})

        scheduler.step()
        rmse = val_rmse()
        print(
            f"[epoch {epoch:03d}] train_data={running_data/max(1,n_batches):.4e} "
            f"val_rmse={rmse:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_rmse": best_rmse,
                    "config": asdict(cfg),
                    "meta": {
                        "variant": f"fieldformer_autograd_{dataset_key}_sparse_no_physics",
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(n_sensors),
                        "num_times": int(n_times),
                        "num_observations": int(n_obs),
                        "x_range": [domain["x_min"], domain["x_max"]],
                        "y_range": [domain["y_min"], domain["y_max"]],
                        "t_range": [domain["t_min"], domain["t_max"]],
                        "dx": domain["dx"],
                        "dy": domain["dy"],
                        "dt": domain["dt"],
                        "physics_loss": False,
                    },
                },
                best_path,
            )
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_rmse:.6f})")

        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break

    print(f"Done. Best val RMSE: {best_rmse:.6f}")
