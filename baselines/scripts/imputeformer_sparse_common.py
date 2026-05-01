from __future__ import annotations

import math
import argparse
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from Attention_layers import AttentionLayer, EmbeddedAttention

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - convenience fallback for lean envs
    tqdm = lambda x, **_: x


def add_config_args(parser: Any, cfg: Any) -> Any:
    for field in fields(type(cfg)):
        value = getattr(cfg, field.name)
        arg = f"--{field.name}"
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    return parser


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
        return lin // self.Nt, lin % self.Nt

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
        if lin_nb.shape[1] < self.k_neighbors:
            pad = lin_nb[:, -1:].expand(-1, self.k_neighbors - lin_nb.shape[1])
            lin_nb = torch.cat([lin_nb, pad], dim=1)
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
        if lin_nb.shape[1] < self.k_neighbors:
            pad = lin_nb[:, -1:].expand(-1, self.k_neighbors - lin_nb.shape[1])
            lin_nb = torch.cat([lin_nb, pad], dim=1)
        return lin_nb


class EmbeddedAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, adaptive_embedding_dim: int, feed_forward_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.attn = EmbeddedAttention(model_dim, adaptive_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        out = self.dropout1(self.attn(x, emb))
        out = self.ln1(residual + out)
        residual = out
        out = self.dropout2(self.feed_forward(out))
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class ProjectedAttentionLayer(nn.Module):
    def __init__(self, seq_len: int, dim_proj: int, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        projector = repeat(self.projector, "dim_proj d_model -> repeat seq_len dim_proj d_model",
                           repeat=batch, seq_len=self.seq_len)
        message_out = self.out_attn(projector, x, x)
        message_in = self.in_attn(x, projector, message_out)
        message = self.norm1(x + self.dropout(message_in))
        message = self.norm2(message + self.dropout(self.MLP(message)))
        return message


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImputeFormerSparse(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.context_nodes = int(cfg.k_neighbors) + 1
        self.input_embedding_dim = cfg.input_embedding_dim
        self.learnable_embedding_dim = cfg.learnable_embedding_dim
        self.model_dim = cfg.input_embedding_dim + cfg.learnable_embedding_dim

        self.input_proj = nn.Linear(cfg.input_dim, cfg.input_embedding_dim)
        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(1, self.context_nodes, cfg.learnable_embedding_dim))
        )
        self.readout = MLP(self.model_dim, self.model_dim, cfg.output_dim)
        self.attn_layers_t = nn.ModuleList(
            [
                ProjectedAttentionLayer(
                    self.context_nodes,
                    cfg.dim_proj,
                    self.model_dim,
                    cfg.num_temporal_heads,
                    self.model_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                EmbeddedAttentionLayer(self.model_dim, cfg.learnable_embedding_dim, cfg.feed_forward_dim, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )

    def forward(self, query_coords: torch.Tensor, sensor_coords: torch.Tensor, sensor_values: torch.Tensor) -> torch.Tensor:
        rel_coords = sensor_coords - query_coords[:, None, :]
        rel_scale = rel_coords.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        rel_coords = rel_coords / rel_scale

        query_token = torch.zeros(query_coords.shape[0], 1, self.input_embedding_dim, device=query_coords.device)
        sensor_mask = torch.ones(sensor_values.shape[0], sensor_values.shape[1], 1, device=sensor_values.device)
        sensor_features = torch.cat([sensor_values.unsqueeze(-1), sensor_mask, rel_coords], dim=-1)
        x = self.input_proj(sensor_features)
        x = torch.cat([query_token, x], dim=1).unsqueeze(1)

        node_emb = self.learnable_embedding.expand(x.shape[0], *self.learnable_embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)
        x = x.permute(0, 2, 1, 3)
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)
        x = x.permute(0, 2, 1, 3)
        return self.readout(x)[:, 0, 0, 0]

    def forward_observed(
        self,
        q_lin: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals_norm: torch.Tensor,
        nb_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self(obs_coords[q_lin], obs_coords[nb_idx], obs_vals_norm[nb_idx])

    def forward_continuous(
        self,
        xyt_q: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals_norm: torch.Tensor,
        nb_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self(xyt_q, obs_coords[nb_idx], obs_vals_norm[nb_idx])


class EarlyStopping:
    def __init__(self, patience: int = 12):
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0
        self.stopped = False

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
    return coords, vals


def spectral_freg(values: torch.Tensor) -> torch.Tensor:
    coeffs = torch.fft.fft(values)
    return torch.mean(torch.abs(coeffs))


def run_sparse_experiment(cfg: Any, *, variant: str, description: str, fallback_keys: tuple[str, str], error_name: str) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)

    if cfg.obs_key in pack:
        sensor_values = pack[cfg.obs_key].astype(np.float32)
    elif fallback_keys[0] in pack:
        sensor_values = pack[fallback_keys[0]].astype(np.float32)
    elif fallback_keys[1] in pack:
        sensor_values = pack[fallback_keys[1]].astype(np.float32)
    else:
        raise KeyError(f"{error_name} sparse dataset must contain {fallback_keys[0]} or {fallback_keys[1]}")

    assert sensors_xy.ndim == 2 and sensors_xy.shape[1] == 2, "sensors_xy must be (S,2)"
    assert sensor_values.ndim == 2, "sensor values must be (S,Nt)"
    s_count, nt_count = sensor_values.shape
    assert t_np.shape[0] == nt_count, "time grid length must match sensor series"

    coords_np, vals_np = build_observed_tuples(sensors_xy, t_np, sensor_values)
    n_obs = coords_np.shape[0]

    ds = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds.set_split("train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    ds_val = ObservedIndexDataset(n_obs, cfg.train_frac, cfg.val_frac, cfg.seed)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False)

    train_vals = vals_np[ds.train_idx.numpy()]
    value_mean = float(train_vals.mean())
    value_std = float(train_vals.std() + 1e-6)
    vals_norm_np = (vals_np - value_mean) / value_std

    obs_coords = torch.from_numpy(coords_np).float().to(device)
    obs_vals = torch.from_numpy(vals_np).float().to(device)
    obs_vals_norm = torch.from_numpy(vals_norm_np).float().to(device)
    sensors_xy_t = torch.from_numpy(sensors_xy).float().to(device)
    t_grid_t = torch.from_numpy(t_np).float().to(device)

    indexer = SparseNeighborIndexer(sensors_xy_t, t_grid_t, cfg.time_radius, cfg.k_neighbors)
    model = ImputeFormerSparse(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    stopper = EarlyStopping(patience=cfg.patience)

    def predict_observed_norm(q_lin: torch.Tensor) -> torch.Tensor:
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return model.forward_observed(q_lin, obs_coords, obs_vals_norm, nb_idx)

    @torch.no_grad()
    def val_rmse() -> float:
        model.eval()
        se_sum, n_sum = 0.0, 0
        for q_lin in dl_val:
            q_lin = q_lin.to(device)
            pred = predict_observed_norm(q_lin) * value_std + value_mean
            tgt = obs_vals[q_lin]
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum += q_lin.numel()
        return math.sqrt(se_sum / max(1, n_sum))

    best_path = Path(cfg.save)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[data] {description} obs={n_obs} sensors={s_count} Nt={nt_count} value_mean={value_mean:.4e} value_std={value_std:.4e}")
    print(f"[model] ImputeFormerSparse parameters={n_params}")

    best_rmse = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        running = {"data": 0.0, "freg": 0.0}
        n_batches = 0

        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred_norm = predict_observed_norm(q_lin)
            tgt_norm = obs_vals_norm[q_lin]
            data_loss = F.mse_loss(pred_norm, tgt_norm)
            freg_loss = spectral_freg(pred_norm) if cfg.f1_loss_weight > 0 else pred_norm.new_tensor(0.0)
            loss = data_loss + cfg.f1_loss_weight * freg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running["data"] += data_loss.item()
            running["freg"] += freg_loss.item()
            n_batches += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "data": f"{running['data']/n_batches:.4e}",
                    "freg": f"{running['freg']/n_batches:.4e}",
                })

        scheduler.step()
        rmse = val_rmse()
        print(
            f"[epoch {epoch:03d}] train_data={running['data']/max(1,n_batches):.4e} "
            f"train_freg={running['freg']/max(1,n_batches):.4e} "
            f"val_rmse={rmse:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_rmse": best_rmse,
                "config": asdict(cfg),
                "normalization": {"value_mean": value_mean, "value_std": value_std},
                "meta": {
                    "variant": variant,
                    "obs_key": cfg.obs_key,
                    "num_sensors": int(s_count),
                    "num_times": int(nt_count),
                    "num_observations": int(n_obs),
                },
            }
            torch.save(ckpt, best_path)
            print(f"[save] best checkpoint -> {best_path} (val_rmse={best_rmse:.6f})")

        stopper.step(rmse)
        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached.")
            break

    print(f"Done. Best val RMSE: {best_rmse:.6f}")
