#!/usr/bin/env python3
"""
SIREN (sparse pollution): train from sensor observations.

Supervision uses observed tuples (sensor_x, sensor_y, t) -> U_sensor_value.
No full-field targets are used for training or validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"  # or U_sensor_clean
    batch_size: int = 1024
    val_batch_size: int = 1024
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    width: int = 256
    depth: int = 6
    w0: float = 30.0
    w0_hidden: float = 1.0
    lambda_sponge: float = 0.0
    lambda_rad: float = 0.0
    sponge_samples: int = 512
    rad_samples: int = 512
    sponge_border_frac: float = 0.05
    rad_warmup: int = 5
    rad_ramp: int = 20
    sponge_warmup: int = 0
    sponge_ramp: int = 10
    c_cap: float = 2.0
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    patience: int = 10
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/siren_polsparse_nophysics_best.pt"


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


class SineLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False):
        super().__init__(in_features, out_features)
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.w0
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * F.linear(x, self.weight, self.bias))


class SIRENSparsePollution(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int, w0: float, w0_hidden: float):
        super().__init__()
        assert depth >= 2
        layers = [SineLayer(in_dim, width, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(width, width, w0=w0_hidden, is_first=False))
        self.hidden = nn.ModuleList(layers)
        self.final = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / width) / w0_hidden
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        h = xyt
        for layer in self.hidden:
            h = layer(h)
        return self.final(h).squeeze(-1)


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


def cosine_ramp(epoch: int, warmup: int, ramp_epochs: int, max_value: float) -> float:
    if epoch <= warmup:
        return 0.0
    z = min(1.0, (epoch - warmup) / max(1, ramp_epochs))
    return float(max_value) * 0.5 * (1.0 - math.cos(math.pi * z))


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(x.abs() <= delta, 0.5 * x * x, delta * (x.abs() - 0.5 * delta))


def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
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
    dx = float(x_np[1] - x_np[0]) if x_np.size > 1 else 1.0
    dy = float(y_np[1] - y_np[0]) if y_np.size > 1 else 1.0
    dt = float(t_np[1] - t_np[0]) if t_np.size > 1 else 1.0

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)

    ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    ds_val = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    model = SIRENSparsePollution(3, cfg.width, cfg.depth, 1, cfg.w0, cfg.w0_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    stopper = EarlyStopping(cfg.patience)

    def sample_interior(n: int) -> torch.Tensor:
        return torch.stack(
            [
                torch.empty(n, device=device).uniform_(x_min, x_max),
                torch.empty(n, device=device).uniform_(y_min, y_max),
                torch.empty(n, device=device).uniform_(t_min, t_max),
            ],
            dim=-1,
        )

    def sponge_loss(n_samples: int) -> torch.Tensor:
        xyt = sample_interior(n_samples)
        pred = model(xyt)
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
        u = model(xyt)
        grads = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux, uy, ut = grads[:, 0], grads[:, 1], grads[:, 2]
        eps = 1e-6
        left = (xyt[:, 0] <= x_min + eps).float()
        right = (xyt[:, 0] >= x_max - eps).float()
        bottom = (xyt[:, 1] <= y_min + eps).float()
        top = (xyt[:, 1] >= y_max - eps).float()
        un = left * (-ux) + right * ux + bottom * (-uy) + top * uy
        c_eff = (-ut / un.abs().clamp(min=1e-6)).clamp(0.0, cfg.c_cap).detach()
        rad_res = ut + c_eff * un
        scale = (torch.sqrt(ut.pow(2) + un.pow(2)) + 1e-3).detach()
        return huber(rad_res / scale, delta=cfg.huber_delta).mean()

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred = model(obs_coords[q_lin])
            tgt = obs_vals[q_lin]
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        lam_sp = cosine_ramp(epoch, cfg.sponge_warmup, cfg.sponge_ramp, cfg.lambda_sponge)
        lam_rad = cosine_ramp(epoch, cfg.rad_warmup, cfg.rad_ramp, cfg.lambda_rad)
        running = {"data": 0.0, "sponge": 0.0, "rad": 0.0, "total": 0.0}
        n_batches = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred = model(obs_coords[q_lin])
            data_loss = F.mse_loss(pred, obs_vals[q_lin])
            sp_loss = sponge_loss(cfg.sponge_samples)
            rad_loss = radiation_bc_loss(cfg.rad_samples) if lam_rad > 0.0 else torch.tensor(0.0, device=device)
            loss = data_loss + lam_sp * sp_loss + lam_rad * rad_loss

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running["data"] += data_loss.item()
            running["sponge"] += sp_loss.item()
            running["rad"] += rad_loss.item()
            running["total"] += loss.item()
            n_batches += 1
            pbar.set_postfix({k: f"{v / max(1, n_batches):.4e}" for k, v in running.items()})

        rmse = val_rmse()
        scheduler.step(rmse)
        print(
            f"[epoch {epoch:03d}] train_total={running['total']/max(1,n_batches):.4e} "
            f"data={running['data']/max(1,n_batches):.4e} "
            f"sponge={running['sponge']/max(1,n_batches):.4e} "
            f"rad={running['rad']/max(1,n_batches):.4e} "
            f"val_rmse={rmse:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
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
                        "variant": "siren_pollution_sparse",
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(s_count),
                        "num_times": int(nt_count),
                        "num_observations": int(n_obs),
                        "x_range": [x_min, x_max],
                        "y_range": [y_min, y_max],
                        "t_range": [t_min, t_max],
                        "dx": dx,
                        "dy": dy,
                        "dt": dt,
                        "open_boundary": True,
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
