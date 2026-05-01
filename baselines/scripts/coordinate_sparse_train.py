from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from baselines.models.data import ObservedIndexDataset, build_observed_tuples, sensor_key


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_ramp(epoch: int, warmup: int, ramp_epochs: int, max_value: float) -> float:
    if epoch <= warmup:
        return 0.0
    z = min(1.0, (epoch - warmup) / max(1, ramp_epochs))
    return float(max_value) * 0.5 * (1.0 - math.cos(math.pi * z))


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(x.abs() <= delta, 0.5 * x * x, delta * (x.abs() - 0.5 * delta))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best = float("inf")
        self.bad_epochs = 0
        self.stopped = False

    def step(self, metric: float) -> None:
        if metric < self.best - 1e-8:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience


def _dataset_key(cfg: Any) -> str:
    if hasattr(cfg, "dataset"):
        return str(cfg.dataset)
    data = str(cfg.data).lower()
    if "pollution" in data:
        return "pol"
    if "swe" in data:
        return "swe"
    return "heat"


def _make_save_path(cfg: Any, dataset_key: str, model_key: str) -> str:
    if getattr(cfg, "save", ""):
        return str(cfg.save)
    suffix = "-pinn" if getattr(cfg, "pinn", False) else ""
    return f"/scratch/ab9738/fieldformer/baselines/checkpoints/{model_key}{suffix}_{dataset_key}sparse_best.pt"


def _ranges(pack: Any, sensors_xy: np.ndarray, t_np: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float]:
    x_np = pack["x"].astype(np.float32) if "x" in pack else sensors_xy[:, 0]
    y_np = pack["y"].astype(np.float32) if "y" in pack else sensors_xy[:, 1]
    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())
    return x_min, x_max, y_min, y_max, t_min, t_max, max(x_max - x_min, 1e-6), max(y_max - y_min, 1e-6), max(t_max - t_min, 1e-6)


def _heat_params(pack: Any) -> tuple[float, float]:
    alpha_x, alpha_y = 0.01, 0.001
    if "params" in pack and "param_names" in pack:
        params = pack["params"]
        names = [str(x) for x in pack["param_names"]]
        if "alpha_x" in names:
            alpha_x = float(params[names.index("alpha_x")])
        if "alpha_y" in names:
            alpha_y = float(params[names.index("alpha_y")])
    return alpha_x, alpha_y


def _swe_params(pack: Any) -> tuple[float, float]:
    g, H = 9.81, 1.0
    if "params" in pack and "param_names" in pack:
        params = pack["params"]
        names = [str(x) for x in pack["param_names"]]
        if "g" in names:
            g = float(params[names.index("g")])
        if "H" in names:
            H = float(params[names.index("H")])
    return g, H


def train_coordinate_sparse(cfg: Any, model_key: str, model_factory: Callable[[Any], torch.nn.Module]) -> None:
    set_seed(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_key = _dataset_key(cfg)
    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    obs_key = sensor_key(pack, dataset_key, getattr(cfg, "obs_key", ""))
    sensor_values = pack[obs_key].astype(np.float32)
    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)
    x_min, x_max, y_min, y_max, t_min, t_max, Lx, Ly, Tt = _ranges(pack, sensors_xy, t_np)

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    n_obs = int(obs_coords.shape[0])

    train_ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    train_ds.set_split("train")
    val_ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    val_ds.set_split("val")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    model = model_factory(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    stopper = EarlyStopping(cfg.patience)

    def predict_all(xyt: torch.Tensor) -> torch.Tensor:
        if model_key == "fmlp":
            pred = model(xyt, Lx=Lx, Ly=Ly, Tt=Tt)
        else:
            pred = model(xyt)
        return pred.unsqueeze(-1) if pred.ndim == 1 else pred

    def predict(xyt: torch.Tensor) -> torch.Tensor:
        pred = predict_all(xyt)
        if pred.ndim == 2:
            pred = pred[:, 0]
        return pred

    def sample_interior(n: int) -> torch.Tensor:
        return torch.stack(
            [
                torch.empty(n, device=device).uniform_(x_min, x_max),
                torch.empty(n, device=device).uniform_(y_min, y_max),
                torch.empty(n, device=device).uniform_(t_min, t_max),
            ],
            dim=-1,
        )

    def heat_pde_loss(n_samples: int) -> torch.Tensor:
        alpha_x, alpha_y = _heat_params(pack)
        xyt = sample_interior(n_samples).requires_grad_(True)
        u = predict(xyt)
        grads = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux, uy, ut = grads[:, 0], grads[:, 1], grads[:, 2]
        uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:, 1]
        forcing = 5.0 * torch.cos(torch.pi * xyt[:, 0]) * torch.cos(torch.pi * xyt[:, 1]) * torch.sin(4.0 * torch.pi * xyt[:, 2] / 20.0)
        return (ut - (alpha_x * uxx + alpha_y * uyy) - forcing).pow(2).mean()

    def swe_pde_loss(n_samples: int) -> torch.Tensor:
        g, H = _swe_params(pack)
        xyt = sample_interior(n_samples).requires_grad_(True)
        pred = predict_all(xyt)
        if pred.shape[-1] < 3:
            raise ValueError("SWE PINN loss requires model output [eta, u, v].")
        eta_hat, u_hat, v_hat = pred[:, 0], pred[:, 1], pred[:, 2]
        grads_eta = torch.autograd.grad(eta_hat, xyt, grad_outputs=torch.ones_like(eta_hat), create_graph=True)[0]
        grads_u = torch.autograd.grad(u_hat, xyt, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0]
        grads_v = torch.autograd.grad(v_hat, xyt, grad_outputs=torch.ones_like(v_hat), create_graph=True)[0]
        eta_x, eta_y, eta_t = grads_eta[:, 0], grads_eta[:, 1], grads_eta[:, 2]
        u_x, u_t = grads_u[:, 0], grads_u[:, 2]
        v_y, v_t = grads_v[:, 1], grads_v[:, 2]
        r_u = u_t + g * eta_x
        r_v = v_t + g * eta_y
        r_eta = eta_t + H * (u_x + v_y)
        return torch.stack([r_u, r_v, r_eta], dim=-1).pow(2).mean()

    def periodic_bc_loss(n_bc: int, match_grad: bool) -> torch.Tensor:
        yb = torch.empty(n_bc, device=device).uniform_(y_min, y_max)
        tb = torch.empty(n_bc, device=device).uniform_(t_min, t_max)
        x0 = torch.stack([torch.full_like(yb, x_min), yb, tb], dim=-1).requires_grad_(match_grad)
        xL = torch.stack([torch.full_like(yb, x_max), yb, tb], dim=-1).requires_grad_(match_grad)
        xb = torch.empty(n_bc, device=device).uniform_(x_min, x_max)
        tb2 = torch.empty(n_bc, device=device).uniform_(t_min, t_max)
        y0 = torch.stack([xb, torch.full_like(xb, y_min), tb2], dim=-1).requires_grad_(match_grad)
        yL = torch.stack([xb, torch.full_like(xb, y_max), tb2], dim=-1).requires_grad_(match_grad)
        u0, uL, v0, vL = predict_all(x0), predict_all(xL), predict_all(y0), predict_all(yL)
        loss = 0.5 * (F.mse_loss(u0, uL) + F.mse_loss(v0, vL))
        if match_grad:
            gx0 = torch.autograd.grad(u0[:, 0], x0, torch.ones_like(u0[:, 0]), create_graph=True)[0][:, 0]
            gxL = torch.autograd.grad(uL[:, 0], xL, torch.ones_like(uL[:, 0]), create_graph=True)[0][:, 0]
            gy0 = torch.autograd.grad(v0[:, 0], y0, torch.ones_like(v0[:, 0]), create_graph=True)[0][:, 1]
            gyL = torch.autograd.grad(vL[:, 0], yL, torch.ones_like(vL[:, 0]), create_graph=True)[0][:, 1]
            loss = loss + 0.5 * (F.mse_loss(gx0, gxL) + F.mse_loss(gy0, gyL))
        return loss

    def sponge_loss(n_samples: int) -> torch.Tensor:
        xyt = sample_interior(n_samples)
        pred = predict(xyt)
        x01 = (xyt[:, 0] - x_min) / Lx
        y01 = (xyt[:, 1] - y_min) / Ly
        d_edge = torch.minimum(torch.minimum(x01, 1.0 - x01), torch.minimum(y01, 1.0 - y01))
        ramp = ((cfg.sponge_border_frac - d_edge).clamp(min=0.0) / cfg.sponge_border_frac) ** 2
        return (ramp * pred.pow(2)).mean()

    def radiation_bc_loss(n_samples: int) -> torch.Tensor:
        n_side = max(1, n_samples // 4)
        tb = torch.empty(n_side, device=device).uniform_(t_min, t_max)
        yb = torch.empty(n_side, device=device).uniform_(y_min, y_max)
        xb = torch.empty(n_side, device=device).uniform_(x_min, x_max)
        xyt = torch.cat(
            [
                torch.stack([torch.full_like(yb, x_min), yb, tb], dim=-1),
                torch.stack([torch.full_like(yb, x_max), yb, tb], dim=-1),
                torch.stack([xb, torch.full_like(xb, y_min), tb], dim=-1),
                torch.stack([xb, torch.full_like(xb, y_max), tb], dim=-1),
            ],
            dim=0,
        ).detach().requires_grad_(True)
        u = predict(xyt)
        grads = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux, uy, ut = grads[:, 0], grads[:, 1], grads[:, 2]
        left = (xyt[:, 0] <= x_min + 1e-6).float()
        right = (xyt[:, 0] >= x_max - 1e-6).float()
        bottom = (xyt[:, 1] <= y_min + 1e-6).float()
        top = (xyt[:, 1] >= y_max - 1e-6).float()
        un = left * (-ux) + right * ux + bottom * (-uy) + top * uy
        c_eff = (-ut / un.abs().clamp(min=1e-6)).clamp(0.0, cfg.c_cap).detach()
        rad_res = ut + c_eff * un
        scale = (torch.sqrt(ut.pow(2) + un.pow(2)) + 1e-3).detach()
        return huber(rad_res / scale, delta=cfg.huber_delta).mean()

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in val_dl:
            q_lin = q_lin.to(device)
            pred = predict(obs_coords[q_lin])
            tgt = obs_vals[q_lin]
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum += int(q_lin.numel())
        return math.sqrt(se_sum / max(1, n_sum))

    save_path = Path(_make_save_path(cfg, dataset_key, model_key))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"data": 0.0, "phys": 0.0, "bc": 0.0, "total": 0.0}
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        lam_phys = cosine_ramp(epoch, cfg.phys_warmup, cfg.phys_ramp, cfg.lambda_phys)
        lam_bc = cosine_ramp(epoch, cfg.bc_warmup, cfg.bc_ramp, cfg.lambda_bc)
        lam_sp = cosine_ramp(epoch, cfg.sponge_warmup, cfg.sponge_ramp, cfg.lambda_sponge)
        lam_rad = cosine_ramp(epoch, cfg.rad_warmup, cfg.rad_ramp, cfg.lambda_rad)

        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred = predict(obs_coords[q_lin])
            data_loss = F.mse_loss(pred, obs_vals[q_lin])
            phys_term = torch.tensor(0.0, device=device)
            bc_term = torch.tensor(0.0, device=device)
            sponge_term = torch.tensor(0.0, device=device)
            rad_term = torch.tensor(0.0, device=device)
            if dataset_key == "heat" and lam_phys > 0.0:
                phys_term = heat_pde_loss(cfg.phys_samples)
            if dataset_key == "heat" and lam_bc > 0.0:
                bc_term = periodic_bc_loss(cfg.bc_samples, cfg.match_grad_bc)
            if dataset_key == "swe" and lam_phys > 0.0:
                phys_term = swe_pde_loss(cfg.phys_samples)
            if dataset_key == "swe" and lam_bc > 0.0:
                bc_term = periodic_bc_loss(cfg.bc_samples, cfg.match_grad_bc)
            if dataset_key == "pol" and lam_sp > 0.0:
                sponge_term = sponge_loss(cfg.sponge_samples)
            if dataset_key == "pol" and lam_rad > 0.0:
                rad_term = radiation_bc_loss(cfg.rad_samples)
            loss = data_loss + lam_phys * phys_term + lam_bc * bc_term + lam_sp * sponge_term + lam_rad * rad_term

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running["data"] += data_loss.item()
            running["phys"] += (phys_term + rad_term).item()
            running["bc"] += (bc_term + sponge_term).item()
            running["total"] += loss.item()
            n_batches += 1
            pbar.set_postfix({k: f"{v / max(1, n_batches):.4e}" for k, v in running.items()})

        rmse = val_rmse()
        scheduler.step(rmse)
        print(
            f"[epoch {epoch:03d}] total={running['total']/max(1,n_batches):.4e} "
            f"data={running['data']/max(1,n_batches):.4e} phys={running['phys']/max(1,n_batches):.4e} "
            f"bc={running['bc']/max(1,n_batches):.4e} val_rmse={rmse:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
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
                        "variant": f"{model_key}_{dataset_key}_sparse",
                        "obs_key": obs_key,
                        "num_sensors": int(sensor_values.shape[0]),
                        "num_times": int(sensor_values.shape[1]),
                        "num_observations": n_obs,
                        "x_range": [x_min, x_max],
                        "y_range": [y_min, y_max],
                        "t_range": [t_min, t_max],
                        "swe_params": list(_swe_params(pack)) if dataset_key == "swe" else None,
                    },
                },
                save_path,
            )
            print(f"[save] best checkpoint -> {save_path} (val_rmse={best_rmse:.6f})")

        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break

    print(f"Done. Best val RMSE: {best_rmse:.6f}")
