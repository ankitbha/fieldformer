#!/usr/bin/env python3
"""
SVGP (sparse heat): train only from sensor observations.

Supervision uses observed tuples (sensor_x, sensor_y, t) -> sensor_value.
No full-field targets are used anywhere in training/eval.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import gpytorch
from gpytorch.kernels import PeriodicKernel, ProductKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset.npz"
    obs_key: str = "sensor_noisy"  # or "sensor_clean"
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
    inducing_points: int = 512
    lr_noise: float = 5e-3
    noise: float = 1e-3
    min_noise: float = 1e-6
    max_noise: float = 1e-1
    lambda_phys: float = 0.2
    lambda_bc: float = 0.2
    phys_samples: int = 1024
    bc_samples: int = 512
    match_grad_bc: bool = False
    grad_clip: float = 1.0
    patience: int = 12
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/svgp_heatsparse_best.pt"


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
        s_q, k_q = self.lin_to_sk(lin_q)
        bsz = lin_q.shape[0]

        s_nb = self.base_sensor.to(lin_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_q[:, None] + self.base_dt.to(lin_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)

        if exclude_self:
            mask_self = lin_nb == lin_q[:, None]
            lin_nb = lin_nb.masked_fill(mask_self, -1)

        if lin_nb.shape[1] > self.k_neighbors:
            lin_nb = lin_nb[:, : self.k_neighbors]

        if (lin_nb < 0).any():
            replacement = lin_q[:, None].expand_as(lin_nb)
            lin_nb = torch.where(lin_nb < 0, replacement, lin_nb)

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


class SVGPSparse(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        m = inducing_points.size(0)
        q = CholeskyVariationalDistribution(m)
        vs = VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ConstantMean()
        kx, ky, kt = PeriodicKernel(), PeriodicKernel(), PeriodicKernel()
        kx.initialize(period_length=1.0)
        ky.initialize(period_length=1.0)
        kt.initialize(period_length=1.0)
        self.covar_module = ScaleKernel(ProductKernel(kx, ky, kt))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGPWrapper:
    def __init__(self, model: SVGPSparse, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        self.model = model
        self.likelihood = likelihood

    def forward_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor, Lx: float, Ly: float, Tt: float) -> torch.Tensor:
        del obs_vals, nb_idx, Lx, Ly, Tt
        return self.likelihood(self.model(obs_coords[q_lin])).mean

    def forward_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor, Lx: float, Ly: float, Tt: float) -> torch.Tensor:
        del obs_coords, obs_vals, nb_idx, Lx, Ly, Tt
        return self.likelihood(self.model(xyt_q)).mean


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


def build_observed_tuples(sensors_xy: np.ndarray, t_grid: np.ndarray, sensor_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_count, nt_count = sensor_values.shape
    x = np.repeat(sensors_xy[:, 0], nt_count)
    y = np.repeat(sensors_xy[:, 1], nt_count)
    t = np.tile(t_grid, s_count)
    coords = np.stack([x, y, t], axis=1).astype(np.float32)
    vals = sensor_values.reshape(-1).astype(np.float32)

    for s in [0, min(s_count - 1, 1), s_count - 1]:
        for k in [0, min(nt_count - 1, 1), nt_count - 1]:
            lin = s * nt_count + k
            assert np.allclose(coords[lin, :2], sensors_xy[s], atol=1e-7)
            assert np.allclose(coords[lin, 2], t_grid[k], atol=1e-7)
            assert np.allclose(vals[lin], sensor_values[s, k], atol=1e-7)

    return coords, vals


def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)

    if cfg.obs_key in pack:
        sensor_values = pack[cfg.obs_key].astype(np.float32)
    elif "sensor_noisy" in pack:
        sensor_values = pack["sensor_noisy"].astype(np.float32)
    else:
        sensor_values = pack["sensor_clean"].astype(np.float32)

    assert sensors_xy.ndim == 2 and sensors_xy.shape[1] == 2, "sensors_xy must be (S,2)"
    assert sensor_values.ndim == 2, "sensor values must be (S,Nt)"
    s_count, nt_count = sensor_values.shape
    assert t_np.shape[0] == nt_count, "time grid length must match sensor series"

    if all(k in pack for k in ["sensors_idx", "x", "y"]):
        sidx, x_np, y_np = pack["sensors_idx"], pack["x"], pack["y"]
        try:
            okx = np.allclose(sensors_xy[:, 0], x_np[sidx[:, 0]], atol=1e-6)
            oky = np.allclose(sensors_xy[:, 1], y_np[sidx[:, 1]], atol=1e-6)
            if not (okx and oky):
                print("[warn] sensors_xy and sensors_idx/x/y are not exactly aligned.")
        except Exception:
            print("[warn] skipped sensors_idx consistency check due to shape/index mismatch.")

    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)
    n_obs = coords_np.shape[0]

    if "x" in pack and "y" in pack:
        x_np, y_np = pack["x"], pack["y"]
        x_min, x_max = float(x_np.min()), float(x_np.max())
        y_min, y_max = float(y_np.min()), float(y_np.max())
    else:
        x_min, x_max = float(sensors_xy[:, 0].min()), float(sensors_xy[:, 0].max())
        y_min, y_max = float(sensors_xy[:, 1].min()), float(sensors_xy[:, 1].max())

    t_min, t_max = float(t_np.min()), float(t_np.max())
    Lx = max(1e-6, x_max - x_min)
    Ly = max(1e-6, y_max - y_min)
    Tt = max(1e-6, t_max - t_min)

    alpha_x = 0.01
    alpha_y = 0.001
    if "params" in pack and "param_names" in pack:
        p = pack["params"]
        names = list(pack["param_names"])
        if "alpha_x" in names:
            alpha_x = float(p[names.index("alpha_x")])
        if "alpha_y" in names:
            alpha_y = float(p[names.index("alpha_y")])

    obs_coords = torch.from_numpy(coords_np).float()
    y_mean = float(vals_np.mean())
    y_std = float(vals_np.std() + 1e-8)
    vals_z = ((vals_np - y_mean) / y_std).astype(np.float32)
    obs_vals = torch.from_numpy(vals_z).float()
    coords01 = obs_coords.clone()
    coords01[:, 0] = (coords01[:, 0] - x_min) / Lx
    coords01[:, 1] = (coords01[:, 1] - y_min) / Ly
    coords01[:, 2] = (coords01[:, 2] - t_min) / Tt

    obs_coords = obs_coords.to(device)
    obs_coords01 = coords01.to(device)
    obs_vals = obs_vals.to(device)
    sensors_xy_t = torch.from_numpy(sensors_xy).float().to(device)
    t_grid_t = torch.from_numpy(t_np).float().to(device)

    ds = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    ds_val = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    indexer = SparseNeighborIndexer(sensors_xy=sensors_xy_t, t_grid=t_grid_t, time_radius=cfg.time_radius, k_neighbors=cfg.k_neighbors)
    m = min(cfg.inducing_points, ds.train_idx.numel())
    with torch.no_grad():
        perm = torch.randperm(ds.train_idx.numel())
        z_init = obs_coords01[ds.train_idx[perm[:m]]].clone()

    gp_model = SVGPSparse(z_init).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    with torch.no_grad():
        likelihood.noise = torch.as_tensor(cfg.noise, device=device).clamp(cfg.min_noise, cfg.max_noise)

    model = SVGPWrapper(gp_model, likelihood)
    optimizer = torch.optim.Adam([
        {"params": gp_model.parameters(), "lr": cfg.lr},
        {"params": likelihood.parameters(), "lr": cfg.lr_noise},
    ], weight_decay=cfg.weight_decay)
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=ds.train_idx.numel())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-4)
    stopper = EarlyStopping(patience=cfg.patience)

    def predict_observed(q_lin: torch.Tensor) -> torch.Tensor:
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return model.forward_observed(q_lin, obs_coords01, obs_vals, nb_idx, Lx=Lx, Ly=Ly, Tt=Tt)

    def pde_residual_autograd(xyt: torch.Tensor) -> torch.Tensor:
        xyt = xyt.requires_grad_(True)
        nb_idx = indexer.gather_continuous_neighbors(xyt)
        xyt01 = xyt.clone()
        xyt01[:, 0] = (xyt01[:, 0] - x_min) / Lx
        xyt01[:, 1] = (xyt01[:, 1] - y_min) / Ly
        xyt01[:, 2] = (xyt01[:, 2] - t_min) / Tt
        u = model.forward_continuous(xyt01, obs_coords01, obs_vals, nb_idx, Lx=Lx, Ly=Ly, Tt=Tt)

        ones = torch.ones_like(u)
        grads = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
        ux, uy, ut = grads[:, 0], grads[:, 1], grads[:, 2]
        uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:, 1]

        amp, period = 5.0, 20.0
        forcing = amp * torch.cos(torch.pi * xyt[:, 0]) * torch.cos(torch.pi * xyt[:, 1]) * torch.sin(4 * torch.pi * xyt[:, 2] / period)
        return ut - (alpha_x * uxx + alpha_y * uyy) - forcing

    def periodic_bc_loss(n_bc: int, match_grad: bool) -> torch.Tensor:
        yb = torch.empty(n_bc, device=device).uniform_(y_min, y_max)
        tb = torch.empty(n_bc, device=device).uniform_(t_min, t_max)
        x0 = torch.full_like(yb, x_min)
        xL = torch.full_like(yb, x_max)

        xyt0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
        xytL = torch.stack([xL, yb, tb], dim=-1).requires_grad_(match_grad)

        nb0 = indexer.gather_continuous_neighbors(xyt0)
        nbL = indexer.gather_continuous_neighbors(xytL)
        xyt0n = xyt0.clone(); xyt0n[:,0]=(xyt0n[:,0]-x_min)/Lx; xyt0n[:,1]=(xyt0n[:,1]-y_min)/Ly; xyt0n[:,2]=(xyt0n[:,2]-t_min)/Tt
        xytLn = xytL.clone(); xytLn[:,0]=(xytLn[:,0]-x_min)/Lx; xytLn[:,1]=(xytLn[:,1]-y_min)/Ly; xytLn[:,2]=(xytLn[:,2]-t_min)/Tt
        u0 = model.forward_continuous(xyt0n, obs_coords01, obs_vals, nb0, Lx=Lx, Ly=Ly, Tt=Tt)
        uL = model.forward_continuous(xytLn, obs_coords01, obs_vals, nbL, Lx=Lx, Ly=Ly, Tt=Tt)
        loss_x = F.mse_loss(u0, uL)

        xb = torch.empty(n_bc, device=device).uniform_(x_min, x_max)
        tb = torch.empty(n_bc, device=device).uniform_(t_min, t_max)
        y0 = torch.full_like(xb, y_min)
        yL = torch.full_like(xb, y_max)

        xyt0y = torch.stack([xb, y0, tb], dim=-1).requires_grad_(match_grad)
        xytLy = torch.stack([xb, yL, tb], dim=-1).requires_grad_(match_grad)

        nb0y = indexer.gather_continuous_neighbors(xyt0y)
        nbLy = indexer.gather_continuous_neighbors(xytLy)
        xyt0yn = xyt0y.clone(); xyt0yn[:,0]=(xyt0yn[:,0]-x_min)/Lx; xyt0yn[:,1]=(xyt0yn[:,1]-y_min)/Ly; xyt0yn[:,2]=(xyt0yn[:,2]-t_min)/Tt
        xytLyn = xytLy.clone(); xytLyn[:,0]=(xytLyn[:,0]-x_min)/Lx; xytLyn[:,1]=(xytLyn[:,1]-y_min)/Ly; xytLyn[:,2]=(xytLyn[:,2]-t_min)/Tt
        uy0 = model.forward_continuous(xyt0yn, obs_coords01, obs_vals, nb0y, Lx=Lx, Ly=Ly, Tt=Tt)
        uyL = model.forward_continuous(xytLyn, obs_coords01, obs_vals, nbLy, Lx=Lx, Ly=Ly, Tt=Tt)
        loss_y = F.mse_loss(uy0, uyL)

        loss = 0.5 * (loss_x + loss_y)
        if match_grad:
            gx0 = torch.autograd.grad(u0, xyt0, torch.ones_like(u0), create_graph=True)[0][:, 0]
            gxL = torch.autograd.grad(uL, xytL, torch.ones_like(uL), create_graph=True)[0][:, 0]
            gy0 = torch.autograd.grad(uy0, xyt0y, torch.ones_like(uy0), create_graph=True)[0][:, 1]
            gyL = torch.autograd.grad(uyL, xytLy, torch.ones_like(uyL), create_graph=True)[0][:, 1]
            loss = loss + 0.5 * (F.mse_loss(gx0, gxL) + F.mse_loss(gy0, gyL))

        return loss

    @torch.no_grad()
    def val_rmse() -> float:
        gp_model.eval(); likelihood.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred = predict_observed(q_lin)
            tgt = obs_vals[q_lin] * y_std + y_mean
            pred = pred * y_std + y_mean
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        gp_model.train(); likelihood.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        running = {"data": 0.0, "phys": 0.0, "bc": 0.0, "total": 0.0}
        n_batches = 0

        for q_lin in pbar:
            q_lin = q_lin.to(device)

            nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
            xb = obs_coords01[q_lin]
            yb = obs_vals[q_lin]
            data_loss = -mll(gp_model(xb), yb)

            xyt_phys = torch.stack([
                torch.empty(cfg.phys_samples, device=device).uniform_(x_min, x_max),
                torch.empty(cfg.phys_samples, device=device).uniform_(y_min, y_max),
                torch.empty(cfg.phys_samples, device=device).uniform_(t_min, t_max),
            ], dim=-1)
            phys_loss = (pde_residual_autograd(xyt_phys) ** 2).mean()

            bc_loss = periodic_bc_loss(cfg.bc_samples, match_grad=cfg.match_grad_bc)
            loss = data_loss + cfg.lambda_phys * phys_loss + cfg.lambda_bc * bc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gp_model.parameters(), cfg.grad_clip)
            optimizer.step()

            running["data"] += (-data_loss).item()
            running["phys"] += phys_loss.item()
            running["bc"] += bc_loss.item()
            running["total"] += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "data": f"{running['data']/n_batches:.4e}",
                "phys": f"{running['phys']/n_batches:.4e}",
                "bc": f"{running['bc']/n_batches:.4e}",
                "tot": f"{running['total']/n_batches:.4e}",
            })

        rmse = val_rmse()
        scheduler.step(rmse)
        print(f"[epoch {epoch:03d}] train_total={running['total']/max(1,n_batches):.4e} val_rmse={rmse:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

        if rmse < best_rmse:
            best_rmse = rmse
            ckpt = {
                "epoch": epoch,
                "model_state_dict": gp_model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
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
                    "alpha_x": alpha_x,
                    "alpha_y": alpha_y,
                },
            }
            torch.save(ckpt, best_path)
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_rmse:.6f})")

        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break

    print(f"Done. Best val RMSE: {best_rmse:.6f}")


if __name__ == "__main__":
    main(CFG)
