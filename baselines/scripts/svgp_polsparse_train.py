#!/usr/bin/env python3
"""
SVGP (sparse pollution): train from sensor observations.

Supervision uses observed tuples (sensor_x, sensor_y, t) -> U_sensor_value.
This mirrors the sparse trainers' observed-tuple data path and the pollution
SVGP baseline's data-only objective.
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
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"  # or U_sensor_clean
    batch_size: int = 65536
    val_batch_size: int = 65536
    epochs: int = 300
    lr: float = 3e-3
    lr_noise: float = 1e-3
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 1337
    inducing_points: int = 1024
    noise: float = 1e-2
    min_noise: float = 1e-6
    ard_lengthscale_init: tuple[float, float, float] = (0.2, 0.2, 0.1)
    outputscale_init: float = 1.0
    grad_clip: float = 2.0
    jitter: float = 1e-4
    patience: int = 10
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/svgp_polsparse_best.pt"


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


class SVGPSparsePollution(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, ard_lengthscale_init: tuple[float, float, float], outputscale_init: float):
        m, d = inducing_points.shape
        q = CholeskyVariationalDistribution(m)
        vs = VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=d))
        with torch.no_grad():
            ls = torch.tensor(ard_lengthscale_init, dtype=torch.float32, device=inducing_points.device)
            self.covar_module.base_kernel.lengthscale = ls
            self.covar_module.outputscale = torch.tensor(float(outputscale_init), dtype=torch.float32, device=inducing_points.device)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@dataclass
class EarlyStopping:
    patience: int = 10
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


def normalize_coords(coords: torch.Tensor, x_min: float, y_min: float, t_min: float, Lx: float, Ly: float, Tt: float) -> torch.Tensor:
    out = coords.clone()
    out[:, 0] = (out[:, 0] - x_min) / Lx
    out[:, 1] = (out[:, 1] - y_min) / Ly
    out[:, 2] = (out[:, 2] - t_min) / Tt
    return out


def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    if cfg.obs_key in pack:
        sensor_values = pack[cfg.obs_key].astype(np.float32)
    elif "U_sensor_noisy" in pack:
        sensor_values = pack["U_sensor_noisy"].astype(np.float32)
    elif "U_sensor_clean" in pack:
        sensor_values = pack["U_sensor_clean"].astype(np.float32)
    else:
        raise KeyError("pollution sparse dataset must contain U_sensor_noisy or U_sensor_clean")

    s_count, nt_count = sensor_values.shape
    assert t_np.shape[0] == nt_count

    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)
    n_obs = coords_np.shape[0]

    x_np = pack["x"].astype(np.float32) if "x" in pack else sensors_xy[:, 0]
    y_np = pack["y"].astype(np.float32) if "y" in pack else sensors_xy[:, 1]
    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())
    Lx = max(1e-6, x_max - x_min)
    Ly = max(1e-6, y_max - y_min)
    Tt = max(1e-6, t_max - t_min)

    obs_coords = torch.from_numpy(coords_np).float()
    obs_coords01 = normalize_coords(obs_coords, x_min, y_min, t_min, Lx, Ly, Tt).to(device)
    vals = torch.from_numpy(vals_np).float().to(device)

    ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    ds_val = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    train_idx = ds.train_idx.to(device)
    y_train = vals[train_idx]
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-6)
    obs_vals = (vals - y_mean) / y_std

    m = min(cfg.inducing_points, ds.train_idx.numel())
    with torch.no_grad():
        perm = torch.randperm(ds.train_idx.numel(), device=device)
        z_init = obs_coords01[train_idx[perm[:m]]].clone().contiguous()

    gp_model = SVGPSparsePollution(z_init, cfg.ard_lengthscale_init, cfg.outputscale_init).to(device)
    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(cfg.min_noise)).to(device)
    with torch.no_grad():
        likelihood.noise = torch.tensor(cfg.noise, device=device)

    optimizer = torch.optim.Adam(
        [
            {"params": gp_model.parameters(), "lr": cfg.lr},
            {"params": likelihood.parameters(), "lr": cfg.lr_noise},
        ]
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=ds.train_idx.numel())
    stopper = EarlyStopping(cfg.patience)

    @torch.no_grad()
    def val_rmse() -> float:
        gp_model.eval()
        likelihood.eval()
        se_sum, n_sum = 0.0, 0
        with gpytorch.settings.fast_pred_var(True), gpytorch.settings.cholesky_jitter(cfg.jitter):
            for q_lin in dl_val:
                q_lin = q_lin.to(device)
                pred_std = likelihood(gp_model(obs_coords01[q_lin])).mean
                pred = pred_std * y_std + y_mean
                tgt = obs_vals[q_lin] * y_std + y_mean
                se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
                n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        gp_model.train()
        likelihood.train()
        running_elbo = 0.0
        n_batches = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            xb = obs_coords01[q_lin]
            yb = obs_vals[q_lin]

            optimizer.zero_grad(set_to_none=True)
            with gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(cfg.jitter):
                loss = -mll(gp_model(xb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gp_model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_elbo += -loss.item()
            n_batches += 1
            pbar.set_postfix({"elbo": f"{running_elbo / max(1, n_batches):.4e}"})

        rmse = val_rmse()
        print(f"[epoch {epoch:03d}] train_elbo={running_elbo/max(1,n_batches):.4e} val_rmse={rmse:.6f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": gp_model.state_dict(),
                    "likelihood_state_dict": likelihood.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse": best_rmse,
                    "inducing_points": z_init.detach().cpu(),
                    "config": asdict(cfg),
                    "meta": {
                        "variant": "svgp_pollution_sparse",
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(s_count),
                        "num_times": int(nt_count),
                        "num_observations": int(n_obs),
                        "x_range": [x_min, x_max],
                        "y_range": [y_min, y_max],
                        "t_range": [t_min, t_max],
                        "y_mean": float(y_mean.detach().cpu()),
                        "y_std": float(y_std.detach().cpu()),
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


if __name__ == "__main__":
    main(CFG)
