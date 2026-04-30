#!/usr/bin/env python3
"""
Fourier-MLP (sparse SWE): train from sensor observations.

Supervision uses observed tuples (sensor_x, sensor_y, t) -> eta_sensor_value.
Predictions are 3-channel [eta, u, v], with u/v learned through physics + BC losses.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz"
    obs_key: str = "eta_sensor_noisy"  # or eta_sensor_clean
    batch_size: int = 2048
    val_batch_size: int = 4096
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    k_neighbors: int = 128
    time_radius: int = 3
    d_model: int = 128
    nhead: int = 4
    layers: int = 3
    d_ff: int = 256
    lambda_phys: float = 0.0
    lambda_bc: float = 0.0
    phys_samples: int = 1024
    bc_samples: int = 512
    match_grad_bc: bool = False
    grad_clip: float = 1.0
    patience: int = 12
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/fmlp_swesparse_nophysics_best.pt"


CFG = Config()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        if self.split == "train":
            return len(self.train_idx)
        if self.split == "val":
            return len(self.val_idx)
        return len(self.test_idx)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.split == "train":
            return self.train_idx[idx]
        if self.split == "val":
            return self.val_idx[idx]
        return self.test_idx[idx]


class SparseNeighborIndexer:
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
        s = lin // self.Nt
        k = lin % self.Nt
        return s, k

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
        return lin_nb


class FourierMLPSparseSWE(nn.Module):
    def __init__(self, width: int = 256, depth: int = 6, kx: int = 16, ky: int = 16, kt: int = 8):
        super().__init__()
        in_dim = 2*(kx+ky+kt)
        self.kx,self.ky,self.kt=kx,ky,kt
        layers=[nn.Linear(in_dim,width), nn.GELU()]
        for _ in range(depth-2): layers += [nn.Linear(width,width), nn.GELU()]
        layers += [nn.Linear(width,3)]
        self.net=nn.Sequential(*layers)

    def _enc1(self,x,ks,L):
        z=2*math.pi*(x[...,None]/(L if L>0 else 1.0))*ks[None,:]
        return torch.cat([torch.sin(z),torch.cos(z)],dim=-1)

    def _forward_xyt(self,xyt,Lx,Ly,Tt):
        dev=xyt.device
        kx=torch.arange(1,self.kx+1,device=dev,dtype=torch.float32)
        ky=torch.arange(1,self.ky+1,device=dev,dtype=torch.float32)
        kt=torch.arange(1,self.kt+1,device=dev,dtype=torch.float32)
        feat=torch.cat([self._enc1(xyt[:,0],kx,Lx),self._enc1(xyt[:,1],ky,Ly),self._enc1(xyt[:,2],kt,Tt)],dim=-1)
        return self.net(feat)

    def forward_observed(self, q_lin, obs_coords, obs_eta, nb_idx, Lx, Ly):
        del obs_eta, nb_idx
        Tt=max(1e-6,float(obs_coords[:,2].max()-obs_coords[:,2].min()))
        return self._forward_xyt(obs_coords[q_lin],Lx,Ly,Tt)

    def forward_continuous(self, xyt_q, obs_coords, obs_eta, nb_idx, Lx, Ly):
        del obs_coords, obs_eta, nb_idx
        Tt=max(1e-6,float(xyt_q[:,2].max()-xyt_q[:,2].min()) if xyt_q.shape[0]>1 else 1.0)
        return self._forward_xyt(xyt_q,Lx,Ly,Tt)


@dataclass
class EarlyStopping:
    patience: int = 12
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, metric: float) -> None:
        if metric < self.best - 1e-8:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True


def build_observed_tuples(sensors_xy: np.ndarray, t_grid: np.ndarray, eta_sensor_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_count, nt_count = eta_sensor_values.shape
    coords = np.stack([
        np.repeat(sensors_xy[:, 0], nt_count),
        np.repeat(sensors_xy[:, 1], nt_count),
        np.tile(t_grid, s_count),
    ], axis=1).astype(np.float32)
    vals = eta_sensor_values.reshape(-1).astype(np.float32)
    return coords, vals


def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    if cfg.obs_key in pack:
        eta_sensor = pack[cfg.obs_key].astype(np.float32)
    elif "eta_sensor_noisy" in pack:
        eta_sensor = pack["eta_sensor_noisy"].astype(np.float32)
    else:
        eta_sensor = pack["eta_sensor_clean"].astype(np.float32)

    s_count, nt_count = eta_sensor.shape
    assert t_np.shape[0] == nt_count

    coords_np, eta_obs_np = build_observed_tuples(sensors_xy, t_np, eta_sensor)
    n_obs = coords_np.shape[0]

    x_np, y_np = pack["x"], pack["y"]
    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())
    Lx = max(1e-6, x_max - x_min)
    Ly = max(1e-6, y_max - y_min)

    g = 9.81
    H = 1.0
    if "params" in pack and "param_names" in pack:
        p = pack["params"]
        names = list(pack["param_names"])
        if "g" in names:
            g = float(p[names.index("g")])
        if "H" in names:
            H = float(p[names.index("H")])

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_eta = torch.from_numpy(eta_obs_np).float().to(device)
    indexer = SparseNeighborIndexer(torch.from_numpy(sensors_xy).float().to(device), torch.from_numpy(t_np).float().to(device), cfg.time_radius, cfg.k_neighbors)

    ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    ds_val = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False)

    model = FourierMLPSparseSWE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    stopper = EarlyStopping(cfg.patience)

    def predict_observed(q_lin: torch.Tensor) -> torch.Tensor:
        nb = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return model.forward_observed(q_lin, obs_coords, obs_eta, nb, Lx=Lx, Ly=Ly)

    def pde_residual_autograd(xyt: torch.Tensor) -> torch.Tensor:
        xyt = xyt.requires_grad_(True)
        nb = indexer.gather_continuous_neighbors(xyt)
        pred = model.forward_continuous(xyt, obs_coords, obs_eta, nb, Lx=Lx, Ly=Ly)
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
        return torch.stack([r_u, r_v, r_eta], dim=-1)

    def periodic_bc_loss(n_bc: int, match_grad: bool) -> torch.Tensor:
        yb = torch.empty(n_bc, device=device).uniform_(y_min, y_max)
        tb = torch.empty(n_bc, device=device).uniform_(t_min, t_max)
        x0 = torch.full_like(yb, x_min)
        xL = torch.full_like(yb, x_max)
        a0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
        aL = torch.stack([xL, yb, tb], dim=-1).requires_grad_(match_grad)
        u0 = model.forward_continuous(a0, obs_coords, obs_eta, indexer.gather_continuous_neighbors(a0), Lx=Lx, Ly=Ly)
        uL = model.forward_continuous(aL, obs_coords, obs_eta, indexer.gather_continuous_neighbors(aL), Lx=Lx, Ly=Ly)
        loss_x = F.mse_loss(u0, uL)

        xb = torch.empty(n_bc, device=device).uniform_(x_min, x_max)
        y0 = torch.full_like(xb, y_min)
        yL = torch.full_like(xb, y_max)
        b0 = torch.stack([xb, y0, tb], dim=-1).requires_grad_(match_grad)
        bL = torch.stack([xb, yL, tb], dim=-1).requires_grad_(match_grad)
        v0 = model.forward_continuous(b0, obs_coords, obs_eta, indexer.gather_continuous_neighbors(b0), Lx=Lx, Ly=Ly)
        vL = model.forward_continuous(bL, obs_coords, obs_eta, indexer.gather_continuous_neighbors(bL), Lx=Lx, Ly=Ly)
        loss_y = F.mse_loss(v0, vL)

        loss = 0.5 * (loss_x + loss_y)
        if match_grad:
            gx0 = torch.autograd.grad(u0[:, 0], a0, torch.ones_like(u0[:, 0]), create_graph=True)[0][:, 0]
            gxL = torch.autograd.grad(uL[:, 0], aL, torch.ones_like(uL[:, 0]), create_graph=True)[0][:, 0]
            gy0 = torch.autograd.grad(v0[:, 0], b0, torch.ones_like(v0[:, 0]), create_graph=True)[0][:, 1]
            gyL = torch.autograd.grad(vL[:, 0], bL, torch.ones_like(vL[:, 0]), create_graph=True)[0][:, 1]
            loss = loss + 0.5 * (F.mse_loss(gx0, gxL) + F.mse_loss(gy0, gyL))
        return loss

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred_eta = predict_observed(q_lin)[:, 0]
            tgt_eta = obs_eta[q_lin]
            se_sum += F.mse_loss(pred_eta, tgt_eta, reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n_batches = {"data": 0.0, "phys": 0.0, "bc": 0.0, "total": 0.0}, 0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred = predict_observed(q_lin)
            data_loss = F.mse_loss(pred[:, 0], obs_eta[q_lin])

            xyt_phys = torch.stack([
                torch.empty(cfg.phys_samples, device=device).uniform_(x_min, x_max),
                torch.empty(cfg.phys_samples, device=device).uniform_(y_min, y_max),
                torch.empty(cfg.phys_samples, device=device).uniform_(t_min, t_max),
            ], dim=-1)
            phys_loss = (pde_residual_autograd(xyt_phys) ** 2).mean()
            bc_loss = periodic_bc_loss(cfg.bc_samples, cfg.match_grad_bc)
            loss = data_loss + cfg.lambda_phys * phys_loss + cfg.lambda_bc * bc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running["data"] += data_loss.item(); running["phys"] += phys_loss.item()
            running["bc"] += bc_loss.item(); running["total"] += loss.item(); n_batches += 1

            pbar.set_postfix({k: f"{v/max(1,n_batches):.4e}" for k, v in running.items()})

        scheduler.step()
        rmse = val_rmse()
        print(f"[epoch {epoch:03d}] train_total={running['total']/max(1,n_batches):.4e} val_rmse={rmse:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_rmse": best_rmse,
                "config": asdict(cfg),
                "meta": {
                    "obs_key": cfg.obs_key,
                    "num_sensors": int(s_count),
                    "num_times": int(nt_count),
                    "num_observations": int(n_obs),
                    "x_range": [x_min, x_max],
                    "y_range": [y_min, y_max],
                    "t_range": [t_min, t_max],
                    "g": g,
                    "H": H,
                },
            }, best_path)
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_rmse:.6f})")

        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break

    print(f"Done. Best val RMSE: {best_rmse:.6f}")


if __name__ == "__main__":
    main(CFG)
