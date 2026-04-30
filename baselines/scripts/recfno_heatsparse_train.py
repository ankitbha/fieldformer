#!/usr/bin/env python3
"""
RecFNO (sparse heat): train only from sensor observations.

Supervision uses observed tuples (sensor_x, sensor_y, t) -> sensor_value.
No full-field targets are used anywhere in training/eval.
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
    data: str = "/scratch/ab9738/fieldformer/data/heat_sharp_dataset.npz"
    obs_key: str = "sensor_noisy"  # or "sensor_clean"
    batch_size: int = 16
    val_batch_size: int = 32
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-6
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    k_neighbors: int = 128
    time_radius: int = 3
    modes1: int = 24
    modes2: int = 24
    width: int = 32
    grad_clip: float = 1.0
    patience: int = 12
    temporal_decay: float = 4.0
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/recfno_heatsparse_best.pt"


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


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class VoronoiFNO2dCore(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)


class VoronoiFNO2dSparse(nn.Module):
    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        grid_shape: tuple[int, int],
        sensors_idx: torch.Tensor,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
        nt_count: int,
        temporal_decay: float,
    ):
        super().__init__()
        self.core = VoronoiFNO2dCore(modes1=modes1, modes2=modes2, width=width, in_channels=3, out_channels=1)
        self.nx, self.ny = grid_shape
        self.nt_count = int(nt_count)
        self.temporal_decay = float(temporal_decay)
        self.register_buffer("sensor_iy", sensors_idx[:, 0].long())
        self.register_buffer("sensor_ix", sensors_idx[:, 1].long())
        self.register_buffer("x_grid", x_grid.float())
        self.register_buffer("y_grid", y_grid.float())

    def _scatter_sparse_grid(self, xyt_q: torch.Tensor, nb_idx: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor) -> torch.Tensor:
        bsz, k_count = nb_idx.shape
        sparse = torch.zeros(bsz, self.nx, self.ny, device=xyt_q.device, dtype=obs_vals.dtype)
        counts = torch.zeros_like(sparse)

        sensor_ids = torch.div(nb_idx, self.nt_count, rounding_mode="floor")
        t_nb = obs_coords[nb_idx, 2]
        t_q = xyt_q[:, None, 2]
        weights = torch.exp(-self.temporal_decay * torch.abs(t_nb - t_q))
        vals = obs_vals[nb_idx] * weights

        iy = self.sensor_iy[sensor_ids]
        ix = self.sensor_ix[sensor_ids]
        linear = iy * self.ny + ix
        sparse_flat = sparse.view(bsz, -1)
        counts_flat = counts.view(bsz, -1)
        sparse_flat.scatter_add_(1, linear, vals)
        counts_flat.scatter_add_(1, linear, weights)
        sparse = sparse_flat.view(bsz, self.nx, self.ny)
        counts = counts_flat.view(bsz, self.nx, self.ny)
        sparse = sparse / counts.clamp_min(1e-6)
        sparse = torch.where(counts > 0, sparse, torch.zeros_like(sparse))
        return sparse

    def _run_core(self, sparse_grid: torch.Tensor) -> torch.Tensor:
        bsz = sparse_grid.shape[0]
        x_grid = self.x_grid.unsqueeze(0).expand(bsz, -1, -1)
        y_grid = self.y_grid.unsqueeze(0).expand(bsz, -1, -1)
        inp = torch.stack([sparse_grid, x_grid, y_grid], dim=1)
        return self.core(inp)

    def forward_observed(
        self,
        q_lin: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals: torch.Tensor,
        nb_idx: torch.Tensor,
        Lx: float,
        Ly: float,
    ) -> torch.Tensor:
        del Lx, Ly
        xyt_q = obs_coords[q_lin]
        sparse_grid = self._scatter_sparse_grid(xyt_q, nb_idx, obs_coords, obs_vals)
        field = self._run_core(sparse_grid).squeeze(1)
        sensor_ids = torch.div(q_lin, self.nt_count, rounding_mode="floor")
        iy = self.sensor_iy[sensor_ids]
        ix = self.sensor_ix[sensor_ids]
        return field[torch.arange(q_lin.shape[0], device=q_lin.device), iy, ix]

    def forward_continuous(
        self,
        xyt_q: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals: torch.Tensor,
        nb_idx: torch.Tensor,
        Lx: float,
        Ly: float,
    ) -> torch.Tensor:
        sparse_grid = self._scatter_sparse_grid(xyt_q, nb_idx, obs_coords, obs_vals)
        field = self._run_core(sparse_grid)
        x01 = (xyt_q[:, 0] - obs_coords[:, 0].min()) / max(Lx, 1e-6)
        y01 = (xyt_q[:, 1] - obs_coords[:, 1].min()) / max(Ly, 1e-6)
        sample_grid = torch.stack([2.0 * x01 - 1.0, 2.0 * y01 - 1.0], dim=-1).view(-1, 1, 1, 2)
        sampled = F.grid_sample(field, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled[:, 0, 0, 0]


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
    torch.set_float32_matmul_precision("high")
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

    x_np = pack["x"].astype(np.float32)
    y_np = pack["y"].astype(np.float32)
    nx = x_np.shape[0]
    ny = y_np.shape[0]
    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())
    Lx = max(1e-6, x_max - x_min)
    Ly = max(1e-6, y_max - y_min)

    if "sensors_idx" in pack:
        sensors_idx = pack["sensors_idx"].astype(np.int64)
    else:
        ix = np.abs(x_np[None, :] - sensors_xy[:, 0:1]).argmin(axis=1)
        iy = np.abs(y_np[None, :] - sensors_xy[:, 1:2]).argmin(axis=1)
        sensors_idx = np.stack([iy, ix], axis=1).astype(np.int64)

    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)
    n_obs = coords_np.shape[0]

    train_obs = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    train_vals = vals_np[train_obs.train_idx.numpy()]
    mean = float(train_vals.mean())
    std = float(train_vals.std() + 1e-8)
    vals_np = ((vals_np - mean) / std).astype(np.float32)

    xx, yy = np.meshgrid(x_np, y_np, indexing="ij")
    x_grid = ((xx - x_min) / Lx).astype(np.float32)
    y_grid = ((yy - y_min) / Ly).astype(np.float32)

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    sensors_xy_t = torch.from_numpy(sensors_xy).float().to(device)
    t_grid_t = torch.from_numpy(t_np).float().to(device)
    sensors_idx_t = torch.from_numpy(sensors_idx).long().to(device)
    x_grid_t = torch.from_numpy(x_grid).float().to(device)
    y_grid_t = torch.from_numpy(y_grid).float().to(device)

    ds = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    ds_val = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)
    ds_test = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds_test.set_split("test")
    dl_test = DataLoader(ds_test, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    indexer = SparseNeighborIndexer(sensors_xy=sensors_xy_t, t_grid=t_grid_t, time_radius=cfg.time_radius, k_neighbors=cfg.k_neighbors)
    model = VoronoiFNO2dSparse(
        modes1=cfg.modes1,
        modes2=cfg.modes2,
        width=cfg.width,
        grid_shape=(nx, ny),
        sensors_idx=sensors_idx_t,
        x_grid=x_grid_t,
        y_grid=y_grid_t,
        nt_count=nt_count,
        temporal_decay=cfg.temporal_decay,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    stopper = EarlyStopping(patience=cfg.patience)

    def predict_observed(q_lin: torch.Tensor) -> torch.Tensor:
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return model.forward_observed(q_lin, obs_coords, obs_vals, nb_idx, Lx=Lx, Ly=Ly)

    @torch.no_grad()
    def eval_split(loader: DataLoader) -> tuple[float, float]:
        model.eval()
        se_sum = 0.0
        l1_sum = 0.0
        n_sum = 0
        for q_lin in loader:
            q_lin = q_lin.to(device)
            pred = predict_observed(q_lin)
            tgt = obs_vals[q_lin]
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            l1_sum += F.l1_loss(pred, tgt, reduction="sum").item()
            n_sum += q_lin.numel()
        rmse = math.sqrt(se_sum / max(1, n_sum)) * std
        l1 = (l1_sum / max(1, n_sum)) * std
        return l1, rmse

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_rmse = float("inf")

    print(
        f"[info] sparse heat tuples={n_obs}, sensors={s_count}, Nt={nt_count}, "
        f"split(train/val/test)=({len(ds.train_idx)}/{len(ds_val.val_idx)}/{len(ds_test.test_idx)})"
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_l1_sum = 0.0
        train_rmse_sum = 0.0
        n_batches = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred = predict_observed(q_lin)
            tgt = obs_vals[q_lin]
            loss = F.l1_loss(pred, tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_l1_sum += loss.item() * std
            train_rmse_sum += math.sqrt(F.mse_loss(pred.detach(), tgt).item()) * std
            n_batches += 1
            pbar.set_postfix({"l1": f"{train_l1_sum/max(1,n_batches):.4e}", "rmse": f"{train_rmse_sum/max(1,n_batches):.4e}"})

        train_l1 = train_l1_sum / max(1, n_batches)
        train_rmse = train_rmse_sum / max(1, n_batches)
        val_l1, val_rmse = eval_split(dl_val)

        print(
            f"epoch {epoch:03d} | "
            f"train_l1={train_l1:.6f} train_rmse={train_rmse:.6f} | "
            f"val_l1={val_l1:.6f} val_rmse={val_rmse:.6f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_rmse": best_val_rmse,
                    "config": asdict(cfg),
                    "meta": {
                        "obs_key": cfg.obs_key,
                        "num_sensors": int(s_count),
                        "num_times": int(nt_count),
                        "num_observations": int(n_obs),
                        "x_range": [x_min, x_max],
                        "y_range": [y_min, y_max],
                        "t_range": [t_min, t_max],
                        "mean": mean,
                        "std": std,
                    },
                },
                best_path,
            )
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_val_rmse:.6f})")

        scheduler.step()
        stopper.step(val_rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[info] loaded best checkpoint from epoch {ckpt['epoch']}")

    test_l1, test_rmse = eval_split(dl_test)
    print(f"[test] l1={test_l1:.6f} rmse={test_rmse:.6f}")


if __name__ == "__main__":
    main(CFG)
