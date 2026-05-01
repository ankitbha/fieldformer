from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from baselines.models.data import ObservedIndexDataset, build_observed_tuples, sensor_key
from baselines.models.svgp import MultitaskPeriodicSVGP, PeriodicSVGP, PollutionSVGP, gpytorch, make_likelihood


def _dataset_key(cfg: Any) -> str:
    if hasattr(cfg, "dataset"):
        return str(cfg.dataset)
    data = str(cfg.data).lower()
    if "pollution" in data:
        return "pol"
    if "swe" in data:
        return "swe"
    return "heat"


def _save_path(cfg: Any, dataset_key: str) -> str:
    if getattr(cfg, "save", ""):
        return str(cfg.save)
    suffix = "-pinn" if getattr(cfg, "pinn", False) else ""
    return f"/scratch/ab9738/fieldformer/baselines/checkpoints/svgp{suffix}_{dataset_key}sparse_best.pt"


def train_svgp_sparse(cfg: Any) -> None:
    if gpytorch is None:
        raise ImportError("gpytorch is required to train SVGP baselines")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_key = _dataset_key(cfg)

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    obs_key = sensor_key(pack, dataset_key, getattr(cfg, "obs_key", ""))
    sensor_values = pack[obs_key].astype(np.float32)
    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)

    x_min, y_min, t_min = coords_np.min(axis=0)
    x_max, y_max, t_max = coords_np.max(axis=0)
    span = np.maximum(np.array([x_max - x_min, y_max - y_min, t_max - t_min], dtype=np.float32), 1e-6)
    coords_np = (coords_np - np.array([x_min, y_min, t_min], dtype=np.float32)) / span
    obs_mean = vals_np.mean(axis=0).astype(np.float32) if vals_np.ndim == 2 else np.float32(vals_np.mean())
    obs_std = (vals_np.std(axis=0) + 1e-6).astype(np.float32) if vals_np.ndim == 2 else np.float32(vals_np.std() + 1e-6)
    vals_np = (vals_np - obs_mean) / obs_std

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    n_obs = int(obs_coords.shape[0])

    ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    val_ds.set_split("val")
    val_dl = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    m = min(int(cfg.inducing_points), n_obs)
    inducing_idx = torch.randperm(n_obs, device=device)[:m]
    inducing = obs_coords[inducing_idx].detach().clone()
    if dataset_key == "pol":
        model = PollutionSVGP(inducing, tuple(cfg.ard_lengthscale_init), cfg.outputscale_init).to(device)
    elif dataset_key == "swe":
        model = MultitaskPeriodicSVGP(inducing, num_tasks=3).to(device)
    else:
        model = PeriodicSVGP(inducing).to(device)
    likelihood_key = "heat" if dataset_key == "swe" else dataset_key
    likelihood = make_likelihood(likelihood_key).to(device)
    if hasattr(likelihood, "noise"):
        likelihood.noise = torch.tensor(float(cfg.noise), device=device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}, {"params": likelihood.parameters(), "lr": cfg.lr_noise}],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(ds))
    save_path = Path(_save_path(cfg, dataset_key))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")
    bad_epochs = 0

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        likelihood.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in val_dl:
            q_lin = q_lin.to(device)
            output = model(obs_coords[q_lin])
            if dataset_key == "swe":
                output = output[..., 0]
            pred = likelihood(output).mean
            tgt = obs_vals[q_lin]
            se_sum += torch.sum((pred - tgt) ** 2).item()
            n_sum += int(q_lin.numel())
        model.train()
        likelihood.train()
        scale = float(np.sqrt(np.mean(np.square(obs_std)))) if np.ndim(obs_std) else float(obs_std)
        return math.sqrt(se_sum / max(1, n_sum)) * scale

    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        n_batches = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(obs_coords[q_lin])
            if dataset_key == "swe":
                output = output[..., 0]
            loss = -mll(output, obs_vals[q_lin])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1
            pbar.set_postfix({"nll": f"{running / max(1, n_batches):.4e}"})

        rmse = val_rmse()
        print(f"[epoch {epoch:03d}] nll={running/max(1,n_batches):.4e} val_rmse={rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "likelihood_state_dict": likelihood.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse": best_rmse,
                    "config": asdict(cfg),
                    "meta": {
                        "variant": f"svgp_{dataset_key}_sparse",
                        "obs_key": obs_key,
                        "obs_mean": [float(obs_mean), 0.0, 0.0] if dataset_key == "swe" else np.asarray(obs_mean).tolist(),
                        "obs_std": [float(obs_std), 1.0, 1.0] if dataset_key == "swe" else np.asarray(obs_std).tolist(),
                        "output_dim": 3 if dataset_key == "swe" else (int(vals_np.shape[-1]) if vals_np.ndim == 2 else 1),
                        "supervised_channels": ["eta"] if dataset_key == "swe" else None,
                    },
                },
                save_path,
            )
            print(f"[save] best checkpoint -> {save_path} (val_rmse={best_rmse:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[early-stop] patience={cfg.patience} reached.")
                break

    print(f"Done. Best val RMSE: {best_rmse:.6f}")
