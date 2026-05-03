#!/usr/bin/env python3
"""Shared trainer for sparse FieldFormer-MLP architecture ablations."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ffag_sparse_mlp_model import class_for_dataset
from ffag_sparse_nophys_common import _core_symbols, _domain_extents, _load_observations, _sensors_are_aligned


@torch.no_grad()
def _ema_update(ema: nn.Module, online: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema.parameters(), online.parameters()):
        ema_param.copy_(decay * ema_param + (1.0 - decay) * param)
    for ema_buf, buf in zip(ema.buffers(), online.buffers()):
        ema_buf.copy_(buf)


def _huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch.where(x.abs() <= delta, 0.5 * x * x, delta * (x.abs() - 0.5 * delta))


def _cosine_ramp(epoch: int, warmup: int, ramp_epochs: int, max_value: float) -> float:
    if epoch <= warmup:
        return 0.0
    z = min(1.0, (epoch - warmup) / max(1, ramp_epochs))
    return float(max_value) * 0.5 * (1.0 - math.cos(math.pi * z))


def _params_by_name(pack: Any) -> dict[str, float]:
    if "params" not in pack or "param_names" not in pack:
        return {}
    params = pack["params"]
    names = list(pack["param_names"])
    return {str(name): float(params[i]) for i, name in enumerate(names)}


def _load_sparse_context(dataset_key: str, cfg: Any) -> dict[str, Any]:
    import numpy as np

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
    domain = _domain_extents(pack, sensors_xy, t_np)
    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)

    ds_cls = core["ObservedIndexDataset"]
    ds = ds_cls(n_obs=coords_np.shape[0], train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    ds_val = ds_cls(n_obs=coords_np.shape[0], train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    indexer = core["SparseNeighborIndexer"](
        torch.from_numpy(sensors_xy).float().to(device),
        torch.from_numpy(t_np).float().to(device),
        cfg.time_radius,
        cfg.k_neighbors,
        allowed_indices=ds.train_idx.to(device),
    )

    return {
        "core": core,
        "pack": pack,
        "device": device,
        "domain": domain,
        "obs_coords": obs_coords,
        "obs_vals": obs_vals,
        "dl": dl,
        "dl_val": dl_val,
        "indexer": indexer,
        "n_sensors": n_sensors,
        "n_times": n_times,
        "n_obs": coords_np.shape[0],
        "params": _params_by_name(pack),
    }


def _new_model(dataset_key: str, cfg: Any, device: torch.device) -> nn.Module:
    model = class_for_dataset(dataset_key)(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff).to(device)
    if dataset_key == "pol":
        with torch.no_grad():
            model.log_gammas[:] = torch.log(torch.tensor([1.0, 1.0, 0.5], device=device))
    return model


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_periodic_mlp(dataset_key: str, cfg: Any) -> None:
    assert dataset_key in {"heat", "swe"}
    ctx = _load_sparse_context(dataset_key, cfg)
    device, domain = ctx["device"], ctx["domain"]
    obs_coords, obs_vals, indexer = ctx["obs_coords"], ctx["obs_vals"], ctx["indexer"]
    model = _new_model(dataset_key, cfg, device)

    params = ctx["params"]
    alpha_x = params.get("alpha_x", 0.01)
    alpha_y = params.get("alpha_y", 0.001)
    g = params.get("g", 9.81)
    h_depth = params.get("H", 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    stopper = ctx["core"]["EarlyStopping"](patience=cfg.patience)

    def forward_observed(q_lin: torch.Tensor) -> torch.Tensor:
        nb = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return model.forward_observed(q_lin, obs_coords, obs_vals, nb, Lx=domain["Lx"], Ly=domain["Ly"])

    def forward_continuous(xyt: torch.Tensor) -> torch.Tensor:
        nb = indexer.gather_continuous_neighbors(xyt)
        return model.forward_continuous(xyt, obs_coords, obs_vals, nb, Lx=domain["Lx"], Ly=domain["Ly"])

    def pde_residual(xyt: torch.Tensor) -> torch.Tensor:
        xyt = xyt.requires_grad_(True)
        pred = forward_continuous(xyt)
        if dataset_key == "heat":
            u = pred
            grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
            ux, uy, ut = grads[:, 0], grads[:, 1], grads[:, 2]
            uxx = torch.autograd.grad(ux, xyt, torch.ones_like(ux), create_graph=True)[0][:, 0]
            uyy = torch.autograd.grad(uy, xyt, torch.ones_like(uy), create_graph=True)[0][:, 1]
            forcing = 5.0 * torch.cos(torch.pi * xyt[:, 0]) * torch.cos(torch.pi * xyt[:, 1]) * torch.sin(4 * torch.pi * xyt[:, 2] / 20.0)
            return ut - (alpha_x * uxx + alpha_y * uyy) - forcing

        eta_hat, u_hat, v_hat = pred[:, 0], pred[:, 1], pred[:, 2]
        grads_eta = torch.autograd.grad(eta_hat, xyt, torch.ones_like(eta_hat), create_graph=True)[0]
        grads_u = torch.autograd.grad(u_hat, xyt, torch.ones_like(u_hat), create_graph=True)[0]
        grads_v = torch.autograd.grad(v_hat, xyt, torch.ones_like(v_hat), create_graph=True)[0]
        r_u = grads_u[:, 2] + g * grads_eta[:, 0]
        r_v = grads_v[:, 2] + g * grads_eta[:, 1]
        r_eta = grads_eta[:, 2] + h_depth * (grads_u[:, 0] + grads_v[:, 1])
        return torch.stack([r_u, r_v, r_eta], dim=-1)

    def periodic_bc_loss(n_bc: int, match_grad: bool) -> torch.Tensor:
        yb = torch.empty(n_bc, device=device).uniform_(domain["y_min"], domain["y_max"])
        tb = torch.empty(n_bc, device=device).uniform_(domain["t_min"], domain["t_max"])
        x0 = torch.full_like(yb, domain["x_min"])
        x_l = torch.full_like(yb, domain["x_max"])
        a0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
        a_l = torch.stack([x_l, yb, tb], dim=-1).requires_grad_(match_grad)
        u0, u_l = forward_continuous(a0), forward_continuous(a_l)
        loss_x = F.mse_loss(u0, u_l)

        xb = torch.empty(n_bc, device=device).uniform_(domain["x_min"], domain["x_max"])
        y0 = torch.full_like(xb, domain["y_min"])
        y_l = torch.full_like(xb, domain["y_max"])
        b0 = torch.stack([xb, y0, tb], dim=-1).requires_grad_(match_grad)
        b_l = torch.stack([xb, y_l, tb], dim=-1).requires_grad_(match_grad)
        v0, v_l = forward_continuous(b0), forward_continuous(b_l)
        loss = 0.5 * (loss_x + F.mse_loss(v0, v_l))
        if match_grad:
            c0 = u0[:, 0] if dataset_key == "swe" else u0
            c_l = u_l[:, 0] if dataset_key == "swe" else u_l
            d0 = v0[:, 0] if dataset_key == "swe" else v0
            d_l = v_l[:, 0] if dataset_key == "swe" else v_l
            gx0 = torch.autograd.grad(c0, a0, torch.ones_like(c0), create_graph=True)[0][:, 0]
            gx_l = torch.autograd.grad(c_l, a_l, torch.ones_like(c_l), create_graph=True)[0][:, 0]
            gy0 = torch.autograd.grad(d0, b0, torch.ones_like(d0), create_graph=True)[0][:, 1]
            gy_l = torch.autograd.grad(d_l, b_l, torch.ones_like(d_l), create_graph=True)[0][:, 1]
            loss = loss + 0.5 * (F.mse_loss(gx0, gx_l) + F.mse_loss(gy0, gy_l))
        return loss

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in ctx["dl_val"]:
            q_lin = q_lin.to(device)
            pred = forward_observed(q_lin)
            pred_obs = pred[:, 0] if dataset_key == "swe" else pred
            se_sum += F.mse_loss(pred_obs, obs_vals[q_lin], reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_rmse = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"data": 0.0, "phys": 0.0, "bc": 0.0, "total": 0.0}
        n_batches = 0
        pbar = tqdm(ctx["dl"], desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred = forward_observed(q_lin)
            pred_obs = pred[:, 0] if dataset_key == "swe" else pred
            data_loss = F.mse_loss(pred_obs, obs_vals[q_lin])
            xyt_phys = torch.stack(
                [
                    torch.empty(cfg.phys_samples, device=device).uniform_(domain["x_min"], domain["x_max"]),
                    torch.empty(cfg.phys_samples, device=device).uniform_(domain["y_min"], domain["y_max"]),
                    torch.empty(cfg.phys_samples, device=device).uniform_(domain["t_min"], domain["t_max"]),
                ],
                dim=-1,
            )
            phys_loss = pde_residual(xyt_phys).pow(2).mean()
            bc_loss = periodic_bc_loss(cfg.bc_samples, cfg.match_grad_bc)
            loss = data_loss + cfg.lambda_phys * phys_loss + cfg.lambda_bc * bc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            for key, value in {"data": data_loss, "phys": phys_loss, "bc": bc_loss, "total": loss}.items():
                running[key] += value.item()
            n_batches += 1
            pbar.set_postfix({k: f"{v/max(1,n_batches):.4e}" for k, v in running.items()})

        scheduler.step()
        rmse = val_rmse()
        print(f"[epoch {epoch:03d}] train_total={running['total']/max(1,n_batches):.4e} val_rmse={rmse:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
        if rmse < best_rmse:
            best_rmse = rmse
            _save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_rmse": best_rmse,
                    "config": asdict(cfg),
                    "meta": {
                        "variant": f"fieldformer_mlp_{dataset_key}_sparse",
                        "architecture": "mlp_token_mixer",
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(ctx["n_sensors"]),
                        "num_times": int(ctx["n_times"]),
                        "num_observations": int(ctx["n_obs"]),
                        "x_range": [domain["x_min"], domain["x_max"]],
                        "y_range": [domain["y_min"], domain["y_max"]],
                        "t_range": [domain["t_min"], domain["t_max"]],
                        "physics_loss": True,
                    },
                },
            )
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_rmse:.6f})")
        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break
    print(f"Done. Best val RMSE: {best_rmse:.6f}")


def train_pollution_mlp(cfg: Any) -> None:
    ctx = _load_sparse_context("pol", cfg)
    device, domain = ctx["device"], ctx["domain"]
    obs_coords, obs_vals, indexer = ctx["obs_coords"], ctx["obs_vals"], ctx["indexer"]
    model = _new_model("pol", cfg, device)
    ema_model = _new_model("pol", cfg, device)
    ema_model.load_state_dict(model.state_dict())

    base_params = [p for name, p in model.named_parameters() if name != "log_gammas"]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": [model.log_gammas], "lr": cfg.gamma_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    stopper = ctx["core"]["EarlyStopping"](patience=cfg.patience)

    def predict_observed(q_lin: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        net = ema_model if use_ema else model
        nb = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return net.forward_observed(q_lin, obs_coords, obs_vals, nb)

    def sample_interior(n_samples: int) -> torch.Tensor:
        return torch.stack(
            [
                torch.empty(n_samples, device=device).uniform_(domain["x_min"], domain["x_max"]),
                torch.empty(n_samples, device=device).uniform_(domain["y_min"], domain["y_max"]),
                torch.empty(n_samples, device=device).uniform_(domain["t_min"], domain["t_max"]),
            ],
            dim=-1,
        )

    def sponge_loss(n_samples: int) -> torch.Tensor:
        xyt = sample_interior(n_samples)
        pred = model.forward_continuous(xyt, obs_coords, obs_vals, indexer.gather_continuous_neighbors(xyt))
        x01 = (xyt[:, 0] - domain["x_min"]) / domain["Lx"]
        y01 = (xyt[:, 1] - domain["y_min"]) / domain["Ly"]
        d_edge = torch.minimum(torch.minimum(x01, 1.0 - x01), torch.minimum(y01, 1.0 - y01))
        ramp = ((cfg.sponge_border_frac - d_edge).clamp(min=0.0) / cfg.sponge_border_frac).pow(2)
        return (ramp * pred.pow(2)).mean()

    def radiation_bc_loss(n_samples: int) -> torch.Tensor:
        n_side = max(1, n_samples // 4)
        tb = torch.empty(n_side, device=device).uniform_(domain["t_min"], domain["t_max"])
        yb = torch.empty(n_side, device=device).uniform_(domain["y_min"], domain["y_max"])
        xb = torch.empty(n_side, device=device).uniform_(domain["x_min"], domain["x_max"])
        pts = [
            torch.stack([torch.full_like(yb, domain["x_min"]), yb, tb], dim=-1),
            torch.stack([torch.full_like(yb, domain["x_max"]), yb, tb], dim=-1),
            torch.stack([xb, torch.full_like(xb, domain["y_min"]), tb], dim=-1),
            torch.stack([xb, torch.full_like(xb, domain["y_max"]), tb], dim=-1),
        ]
        xyt = torch.cat(pts, dim=0).detach().requires_grad_(True)
        u = model.forward_continuous(xyt, obs_coords, obs_vals, indexer.gather_continuous_neighbors(xyt))
        grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
        ux, uy, ut = grads[:, 0], grads[:, 1], grads[:, 2]
        eps = 1e-6
        left = (xyt[:, 0] <= domain["x_min"] + eps).float()
        right = (xyt[:, 0] >= domain["x_max"] - eps).float()
        bottom = (xyt[:, 1] <= domain["y_min"] + eps).float()
        top = (xyt[:, 1] >= domain["y_max"] - eps).float()
        un = left * (-ux) + right * ux + bottom * (-uy) + top * uy
        c_eff = (-ut / un.abs().clamp(min=1e-6)).clamp(0.0, cfg.c_cap).detach()
        rad_res = ut + c_eff * un
        scale = (torch.sqrt(ut.pow(2) + un.pow(2)) + 1e-3).detach()
        return _huber(rad_res / scale, delta=cfg.huber_delta).mean()

    @torch.no_grad()
    def val_rmse() -> float:
        ema_model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in ctx["dl_val"]:
            q_lin = q_lin.to(device)
            se_sum += F.mse_loss(predict_observed(q_lin, use_ema=True), obs_vals[q_lin], reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_rmse = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        lam_sp = _cosine_ramp(epoch, cfg.sponge_warmup, cfg.sponge_ramp, cfg.lambda_sponge)
        lam_rad = _cosine_ramp(epoch, cfg.rad_warmup, cfg.rad_ramp, cfg.lambda_rad)
        model.log_gammas.requires_grad_(epoch > 6)
        running = {"data": 0.0, "sponge": 0.0, "rad": 0.0, "total": 0.0}
        n_batches = 0
        pbar = tqdm(ctx["dl"], desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            amp_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available()) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                data_loss = F.mse_loss(predict_observed(q_lin), obs_vals[q_lin])
                sp_loss = sponge_loss(cfg.sponge_samples)
                rad_loss = radiation_bc_loss(cfg.rad_samples) if lam_rad > 0.0 else torch.tensor(0.0, device=device)
                loss = data_loss + lam_sp * sp_loss + lam_rad * rad_loss
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.log_gammas.clamp_(-2.0, 2.0)
            _ema_update(ema_model, model, cfg.ema_decay)
            for key, value in {"data": data_loss, "sponge": sp_loss, "rad": rad_loss, "total": loss}.items():
                running[key] += value.item()
            n_batches += 1
            pbar.set_postfix({k: f"{v/max(1,n_batches):.4e}" for k, v in running.items()})

        rmse = val_rmse()
        scheduler.step(rmse)
        print(f"[epoch {epoch:03d}] train_total={running['total']/max(1,n_batches):.4e} val_rmse={rmse:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if rmse < best_rmse:
            best_rmse = rmse
            _save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_rmse": best_rmse,
                    "gammas": model.log_gammas.detach().exp().cpu().numpy(),
                    "config": asdict(cfg),
                    "meta": {
                        "variant": "fieldformer_mlp_pol_sparse",
                        "architecture": "mlp_token_mixer",
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(ctx["n_sensors"]),
                        "num_times": int(ctx["n_times"]),
                        "num_observations": int(ctx["n_obs"]),
                        "x_range": [domain["x_min"], domain["x_max"]],
                        "y_range": [domain["y_min"], domain["y_max"]],
                        "t_range": [domain["t_min"], domain["t_max"]],
                        "open_boundary": True,
                    },
                },
            )
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_rmse:.6f})")
        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break
    print(f"Done. Best val RMSE: {best_rmse:.6f}")
