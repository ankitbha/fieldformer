#!/usr/bin/env python3
"""Shared observation-only trainer for sparse FieldFormer gamma-field ablations."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_DEBUG_START = time.monotonic()


def _debug(msg: str) -> None:
    elapsed = time.monotonic() - _DEBUG_START
    print(f"[debug:npgf-common +{elapsed:7.2f}s] {msg}", flush=True)


_debug("module import started")
_debug("importing numpy")
import numpy as np
_debug("imported numpy")
_debug("importing torch")
import torch
_debug("imported torch")
_debug("importing torch.nn.functional")
import torch.nn.functional as F
_debug("imported torch.nn.functional")
_debug("importing DataLoader")
from torch.utils.data import DataLoader
_debug("imported DataLoader")
_debug("importing tqdm")
from tqdm.auto import tqdm
_debug("imported tqdm")


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_debug("importing FieldFormer dataset dispatch")
from fieldformer_core.models.ffag import module_for_dataset
_debug("imported FieldFormer dataset dispatch")
from ffag_sparse_npgf_model import class_for_dataset
_debug("importing baseline mask_key helper")
from baselines.models.data import build_observed_index_dataset, mask_key
_debug("imported baseline mask_key helper")
from baselines.scripts.training_cli import apply_cli_overrides, maybe_load_checkpoint


def _core_symbols(dataset_key: str) -> dict[str, Any]:
    _debug(f"loading core symbols for dataset={dataset_key}")
    mod = module_for_dataset(dataset_key)
    _debug(f"loaded core module {mod.__name__}")
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
        "govpol": ("U_sensor", "U_sensor_noisy", "U_sensor_clean"),
        "atm": ("U_sensor", "U_sensor_noisy", "U_sensor_clean"),
        "govpolsplit": ("U_sensor", "U_sensor_noisy", "U_sensor_clean"),
        "atmsplit": ("U_sensor", "U_sensor_noisy", "U_sensor_clean"),
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


def train_sparse_npgf(dataset_key: str, cfg: Any) -> None:
    cfg = apply_cli_overrides(cfg)
    _debug(f"train_sparse_npgf start dataset={dataset_key}")
    core = _core_symbols(dataset_key)
    _debug(f"setting seed={cfg.seed}")
    core["set_seed"](cfg.seed)
    _debug("setting matmul precision")
    torch.set_float32_matmul_precision("high")
    _debug("checking CUDA availability")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _debug(f"selected device={device}")

    _debug(f"loading data npz {cfg.data}")
    pack = np.load(cfg.data)
    _debug(f"loaded npz keys={list(pack.files)}")
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    _debug(f"loaded sensors_xy shape={sensors_xy.shape}, t shape={t_np.shape}")
    sensor_values = _load_observations(pack, dataset_key, cfg.obs_key)
    _debug(f"loaded observations key={cfg.obs_key}, shape={sensor_values.shape}")
    obs_mask_key = mask_key(pack, getattr(cfg, "mask_key", ""))
    _debug(f"resolved mask key={obs_mask_key or '<none>'}")
    sensor_mask = pack[obs_mask_key].astype(np.float32) if obs_mask_key else None
    if sensor_mask is not None:
        _debug(f"loaded sensor mask shape={sensor_mask.shape}")

    assert sensors_xy.ndim == 2 and sensors_xy.shape[1] == 2, "sensors_xy must be (S,2)"
    assert sensor_values.ndim in {2, 3}, "sensor values must be (S,Nt) or (S,Nt,C)"
    n_sensors, n_times = sensor_values.shape[:2]
    assert t_np.shape[0] == n_times, "time grid length must match sensor series"
    _debug(f"validated shapes: sensors={n_sensors}, times={n_times}")
    _sensors_are_aligned(pack, sensors_xy, dataset_key)

    if sensor_mask is not None:
        _debug("building observed tuples with mask")
        coords_np, vals_np, mask_np = core["build_observed_tuples"](sensors_xy, t_np, sensor_values, sensor_mask)
        valid_idx = np.flatnonzero(mask_np.reshape(mask_np.shape[0], -1).any(axis=1))
        _debug(f"built observed tuples coords={coords_np.shape}, vals={vals_np.shape}, valid={valid_idx.shape[0]}")
    else:
        _debug("building observed tuples without mask")
        coords_np, vals_np = core["build_observed_tuples"](sensors_xy, t_np, sensor_values)
        mask_np = np.ones_like(vals_np, dtype=np.float32)
        valid_idx = None
        _debug(f"built observed tuples coords={coords_np.shape}, vals={vals_np.shape}")
    n_obs = coords_np.shape[0]
    domain = _domain_extents(pack, sensors_xy, t_np)
    _debug(f"domain extents={domain}")

    _debug("moving observed tensors to device")
    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_mask = torch.from_numpy(mask_np).float().to(device)
    sensors_xy_t = torch.from_numpy(sensors_xy).float().to(device)
    t_grid_t = torch.from_numpy(t_np).float().to(device)
    _debug("moved observed tensors to device")

    _debug("constructing train split and DataLoader")
    ds = build_observed_index_dataset(
        dataset_key=dataset_key,
        pack=pack,
        n_obs=n_obs,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        valid_idx=valid_idx,
        sensor_mask=sensor_mask,
        sensor_split_seed=getattr(cfg, "sensor_split_seed", None),
        val_sensors=int(getattr(cfg, "val_sensors", 3)),
        test_sensors=int(getattr(cfg, "test_sensors", 3)),
        min_valid_frac=float(getattr(cfg, "sensor_min_valid_frac", 0.10)),
    )
    ds.set_split("train")
    split_meta = getattr(ds, "meta", {})
    out_dim = int(sensor_values.shape[2]) if sensor_values.ndim == 3 else 1
    normalize_values = bool(getattr(cfg, "normalize_values", False))
    vals_mean = np.zeros(out_dim, dtype=np.float32)
    vals_std = np.ones(out_dim, dtype=np.float32)
    vals_raw = vals_np.reshape(n_obs, out_dim)
    mask_raw = mask_np.reshape(n_obs, out_dim)
    train_idx_np = ds.train_idx.detach().cpu().numpy()
    if normalize_values:
        for c in range(out_dim):
            valid = mask_raw[train_idx_np, c].astype(bool)
            if valid.any():
                vals_mean[c] = float(vals_raw[train_idx_np, c][valid].mean())
                vals_std[c] = float(vals_raw[train_idx_np, c][valid].std() + 1e-6)
        vals_np = ((vals_raw - vals_mean) / vals_std).reshape(vals_np.shape).astype(np.float32)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    vals_mean_t = torch.from_numpy(vals_mean).float().to(device)
    vals_std_t = torch.from_numpy(vals_std).float().to(device)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    _debug(f"train split size={len(ds)}, train batches={len(dl)}")

    _debug("constructing val split and DataLoader")
    ds_val = build_observed_index_dataset(
        dataset_key=dataset_key,
        pack=pack,
        n_obs=n_obs,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        valid_idx=valid_idx,
        sensor_mask=sensor_mask,
        sensor_split_seed=getattr(cfg, "sensor_split_seed", None),
        val_sensors=int(getattr(cfg, "val_sensors", 3)),
        test_sensors=int(getattr(cfg, "test_sensors", 3)),
        min_valid_frac=float(getattr(cfg, "sensor_min_valid_frac", 0.10)),
    )
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)
    _debug(f"val split size={len(ds_val)}, val batches={len(dl_val)}")

    _debug("constructing sparse neighbor indexer")
    indexer = core["SparseNeighborIndexer"](sensors_xy_t, t_grid_t, cfg.time_radius, cfg.k_neighbors, allowed_indices=ds.train_idx.to(device))
    _debug("constructed sparse neighbor indexer")
    _debug("loading model class")
    model_cls = class_for_dataset(dataset_key)
    _debug(f"constructing model class={model_cls.__name__}, out_dim={out_dim}")
    gamma_kwargs = {
        "gamma_hidden": int(getattr(cfg, "gamma_hidden", 32)),
        "gamma_layers": int(getattr(cfg, "gamma_layers", 2)),
        "gamma_max_delta": float(getattr(cfg, "gamma_max_delta", 2.0)),
    }
    if dataset_key in {"pol", "govpol", "atm", "govpolsplit", "atmsplit"}:
        gamma_kwargs["base_gammas"] = tuple(float(x) for x in getattr(cfg, "base_gammas", (1.0, 1.0, 0.5)))
    else:
        gamma_kwargs["base_gammas"] = tuple(float(x) for x in getattr(cfg, "base_gammas", (1.0, 1.0, 1.0)))
    try:
        model = model_cls(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff, out_dim=out_dim, **gamma_kwargs).to(device)
    except TypeError:
        model = model_cls(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff, **gamma_kwargs).to(device)
    if hasattr(model, "set_coord_ranges"):
        model.set_coord_ranges(
            x_min=domain["x_min"],
            x_max=domain["x_max"],
            y_min=domain["y_min"],
            y_max=domain["y_max"],
            t_min=domain["t_min"],
            t_max=domain["t_max"],
        )
    _debug("constructed model on device")
    _debug(f"initialized gamma field kwargs={gamma_kwargs}")

    _debug("constructing optimizer/scheduler/early stopper")
    gamma_param_ids = {id(param) for param in model.gamma_field.parameters()} if hasattr(model, "gamma_field") else set()
    main_params = [param for param in model.parameters() if id(param) not in gamma_param_ids]
    gamma_params = [param for param in model.parameters() if id(param) in gamma_param_ids]
    param_groups: list[dict[str, Any]] = [{"params": main_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    if gamma_params:
        param_groups.append(
            {
                "params": gamma_params,
                "lr": float(getattr(cfg, "gamma_field_lr", cfg.lr)),
                "weight_decay": cfg.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    stopper = core["EarlyStopping"](patience=cfg.patience)
    _debug("constructed optimizer/scheduler/early stopper")

    def predict_observed(q_lin: torch.Tensor) -> torch.Tensor:
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        if dataset_key in {"pol", "govpol", "atm", "govpolsplit", "atmsplit"}:
            return model.forward_observed(q_lin, obs_coords, obs_vals, nb_idx)
        return model.forward_observed(q_lin, obs_coords, obs_vals, nb_idx, Lx=domain["Lx"], Ly=domain["Ly"])

    def observed_channel(pred: torch.Tensor) -> torch.Tensor:
        return pred[:, 0] if dataset_key == "swe" else pred

    def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = observed_channel(pred)
        pred = pred.unsqueeze(-1) if pred.ndim == 1 and tgt.ndim == 2 else pred
        tgt = tgt.unsqueeze(-1) if tgt.ndim == 1 and pred.ndim == 2 else tgt
        mask = mask.unsqueeze(-1) if mask.ndim == 1 and pred.ndim == 2 else mask
        return (((pred - tgt) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred = predict_observed(q_lin)
            tgt = torch.from_numpy(vals_raw[q_lin.detach().cpu().numpy()]).float().to(device)
            if out_dim == 1:
                tgt = tgt[:, 0]
            mask = obs_mask[q_lin]
            pred = observed_channel(pred)
            if normalize_values:
                pred_m = pred.unsqueeze(-1) if pred.ndim == 1 else pred
                pred = pred_m * vals_std_t + vals_mean_t
                if out_dim == 1:
                    pred = pred[:, 0]
            pred = pred.unsqueeze(-1) if pred.ndim == 1 and tgt.ndim == 2 else pred
            tgt = tgt.unsqueeze(-1) if tgt.ndim == 1 and pred.ndim == 2 else tgt
            mask = mask.unsqueeze(-1) if mask.ndim == 1 and pred.ndim == 2 else mask
            se_sum += ((((pred - tgt) ** 2) * mask).sum()).item()
            n_sum += int(mask.sum().item())
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best_rmse = maybe_load_checkpoint(
        cfg,
        best_path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        strict=True,
    )
    stopper.best = best_rmse
    _debug(f"checkpoint path ready: {best_path}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        _debug(f"entering epoch {epoch}/{cfg.epochs}")
        model.train()
        running_data, n_batches = 0.0, 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            loss = masked_mse(predict_observed(q_lin), obs_vals[q_lin], obs_mask[q_lin])

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
                        "variant": f"fieldformer_autograd_{dataset_key}_sparse_no_physics_gamma_field",
                        "architecture": "ffag_npgf",
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(n_sensors),
                        "num_times": int(n_times),
                        "output_dim": int(out_dim),
                        "val_mean": vals_mean.tolist() if normalize_values else None,
                        "val_std": vals_std.tolist() if normalize_values else None,
                        "normalizes_values": normalize_values,
                        "split": split_meta or None,
                        "num_observations": int(n_obs),
                        "num_valid_observations": int(valid_idx.shape[0]) if valid_idx is not None else int(n_obs),
                        "x_range": [domain["x_min"], domain["x_max"]],
                        "y_range": [domain["y_min"], domain["y_max"]],
                        "t_range": [domain["t_min"], domain["t_max"]],
                        "dx": domain["dx"],
                        "dy": domain["dy"],
                        "dt": domain["dt"],
                        "physics_loss": False,
                        "gamma_field": {
                            **(model.gamma_metadata() if hasattr(model, "gamma_metadata") else {}),
                            "hidden": int(getattr(cfg, "gamma_hidden", 32)),
                            "layers": int(getattr(cfg, "gamma_layers", 2)),
                            "gamma_field_lr": float(getattr(cfg, "gamma_field_lr", cfg.lr)),
                        },
                        "mask_key": obs_mask_key,
                        "channel_names": pack["pollutant_names"].tolist() if "pollutant_names" in pack else None,
                        "merged_sensor_names": pack["merged_sensor_names"].tolist() if "merged_sensor_names" in pack else None,
                        "merged_sensor_sources": pack["merged_sensor_sources"].tolist() if "merged_sensor_sources" in pack else None,
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
