#!/usr/bin/env python3
"""
FieldFormer-Autograd (sparse pollution): train from sensor observations.

Supervision uses observed tuples (sensor_x, sensor_y, t) -> U_sensor_value.
No full-field targets are used for training or validation.

Dataset contract (npz produced by data/pollution.py):
  - sensors_xy: (S, 2)
  - U_sensor_noisy or U_sensor_clean: (S, Nt)
  - t: (Nt,)
  - x, y: optional domain grids, expected to span [0, 1]
  - sensors_idx: optional (S, 2) as (iy, ix), used only for a consistency check
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from sparse_neighbor_indexer import SplitAwareSparseNeighborIndexer


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    obs_key: str = "U_sensor_noisy"  # or U_sensor_clean
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
    gamma_lr: float = 1e-3
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
    lambda_sponge: float = 0.03
    lambda_rad: float = 0.01
    sponge_samples: int = 512
    rad_samples: int = 512
    sponge_border_frac: float = 0.05
    rad_warmup: int = 5
    rad_ramp: int = 20
    sponge_warmup: int = 0
    sponge_ramp: int = 10
    c_cap: float = 2.0
    huber_delta: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 0.5
    patience: int = 10
    save: str = "/scratch/ab9738/fieldformer/model/ffag_polsparse_best.pt"


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
    """
    Neighborhood over observed sensor-time tuples only.

    Observation linear index maps as: lin = s * Nt + k.
    Neighbors are generated from all sensors and nearby time offsets.
    """

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


SparseNeighborIndexer = SplitAwareSparseNeighborIndexer


class FieldFormerSparsePollution(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, d_ff: int):
        super().__init__()
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.input_proj = nn.Linear(4, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor) -> torch.Tensor:
        rel = nb_xyt - xyt_q[:, None, :]
        rel = rel * torch.exp(self.log_gammas)[None, None, :]

        mu = nb_vals.mean(dim=1, keepdim=True)
        sigma = nb_vals.std(dim=1, keepdim=True).clamp_min(1e-3)
        nb_vals_norm = ((nb_vals - mu) / sigma)[..., None]
        tokens = torch.cat([rel, nb_vals_norm], dim=-1)

        kernel_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
        with kernel_ctx, amp_ctx:
            h = self.input_proj(tokens)
            h = self.encoder(h)
            u_std_res = self.head(h.mean(dim=1)).squeeze(-1)
        return u_std_res * sigma.squeeze(1) + mu.squeeze(1)

    def forward_observed(
        self,
        q_lin: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals: torch.Tensor,
        nb_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_tokens(obs_coords[q_lin], obs_coords[nb_idx], obs_vals[nb_idx])

    def forward_continuous(
        self,
        xyt_q: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals: torch.Tensor,
        nb_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_tokens(xyt_q, obs_coords[nb_idx], obs_vals[nb_idx])


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


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch.where(x.abs() <= delta, 0.5 * x * x, delta * (x.abs() - 0.5 * delta))


@torch.no_grad()
def ema_update(ema: nn.Module, online: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema.parameters(), online.parameters()):
        ema_param.copy_(decay * ema_param + (1.0 - decay) * param)
    for ema_buf, buf in zip(ema.buffers(), online.buffers()):
        ema_buf.copy_(buf)


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
    elif "sensor_noisy" in pack:
        sensor_values = pack["sensor_noisy"].astype(np.float32)
    else:
        sensor_values = pack["sensor_clean"].astype(np.float32)

    assert sensors_xy.ndim == 2 and sensors_xy.shape[1] == 2, "sensors_xy must be (S,2)"
    assert sensor_values.ndim == 2, "sensor values must be (S,Nt)"
    s_count, nt_count = sensor_values.shape
    assert t_np.shape[0] == nt_count, "time grid length must match sensor series"

    if all(k in pack for k in ["sensors_idx", "x", "y"]):
        sidx = pack["sensors_idx"]
        x_np_check, y_np_check = pack["x"], pack["y"]
        try:
            okx = np.allclose(sensors_xy[:, 0], x_np_check[sidx[:, 1]], atol=1e-6)
            oky = np.allclose(sensors_xy[:, 1], y_np_check[sidx[:, 0]], atol=1e-6)
            if not (okx and oky):
                print("[warn] sensors_xy and sensors_idx/x/y are not exactly aligned.")
        except Exception:
            print("[warn] skipped sensors_idx consistency check due to shape/index mismatch.")

    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)
    n_obs = coords_np.shape[0]

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
    Lx = max(1e-6, x_max - x_min)
    Ly = max(1e-6, y_max - y_min)

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    sensors_xy_t = torch.from_numpy(sensors_xy).float().to(device)
    t_grid_t = torch.from_numpy(t_np).float().to(device)

    ds = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    ds_val = ObservedIndexDataset(n_obs=n_obs, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    indexer = SparseNeighborIndexer(sensors_xy_t, t_grid_t, cfg.time_radius, cfg.k_neighbors, allowed_indices=ds.train_idx.to(device))

    model = FieldFormerSparsePollution(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff).to(device)
    with torch.no_grad():
        model.log_gammas[:] = torch.log(torch.tensor([1.0, 1.0, 0.5], device=device))

    ema_model = FieldFormerSparsePollution(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff).to(device)
    ema_model.load_state_dict(model.state_dict())

    base_params = [p for n, p in model.named_parameters() if n != "log_gammas"]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": [model.log_gammas], "lr": cfg.gamma_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    stopper = EarlyStopping(patience=cfg.patience)

    def predict_observed(q_lin: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        net = ema_model if use_ema else model
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return net.forward_observed(q_lin, obs_coords, obs_vals, nb_idx)

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
        nb_idx = indexer.gather_continuous_neighbors(xyt)
        pred = model.forward_continuous(xyt, obs_coords, obs_vals, nb_idx)

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

        pts = [
            torch.stack([torch.full_like(yb, x_min), yb, tb], dim=-1),
            torch.stack([torch.full_like(yb, x_max), yb, tb], dim=-1),
            torch.stack([xb, torch.full_like(xb, y_min), tb], dim=-1),
            torch.stack([xb, torch.full_like(xb, y_max), tb], dim=-1),
        ]
        xyt = torch.cat(pts, dim=0).detach().requires_grad_(True)
        nb_idx = indexer.gather_continuous_neighbors(xyt)
        u = model.forward_continuous(xyt, obs_coords, obs_vals, nb_idx)
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
        ema_model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred = predict_observed(q_lin, use_ema=True)
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

        if epoch <= 6:
            model.log_gammas.requires_grad_(False)
        else:
            model.log_gammas.requires_grad_(True)

        running = {"data": 0.0, "sponge": 0.0, "rad": 0.0, "total": 0.0}
        n_batches = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for q_lin in pbar:
            q_lin = q_lin.to(device)
            amp_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available()) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                pred = predict_observed(q_lin)
                data_loss = F.mse_loss(pred, obs_vals[q_lin])
                sp_loss = sponge_loss(cfg.sponge_samples)
                rad_loss = radiation_bc_loss(cfg.rad_samples) if lam_rad > 0.0 else torch.tensor(0.0, device=device)
                loss = data_loss + lam_sp * sp_loss + lam_rad * rad_loss

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.log_gammas.clamp_(-2.0, 2.0)
            ema_update(ema_model, model, cfg.ema_decay)

            running["data"] += data_loss.item()
            running["sponge"] += sp_loss.item()
            running["rad"] += rad_loss.item()
            running["total"] += loss.item()
            n_batches += 1
            pbar.set_postfix({k: f"{v / max(1, n_batches):.4e}" for k, v in running.items()})

        rmse = val_rmse()
        scheduler.step(rmse)
        lr0 = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}] train_total={running['total']/max(1,n_batches):.4e} "
            f"data={running['data']/max(1,n_batches):.4e} "
            f"sponge={running['sponge']/max(1,n_batches):.4e} "
            f"rad={running['rad']/max(1,n_batches):.4e} "
            f"val_rmse={rmse:.6f} lr={lr0:.2e}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(
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
                        "variant": "fieldformer_autograd_pollution_sparse",
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
