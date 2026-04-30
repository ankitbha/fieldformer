#!/usr/bin/env python3
"""
Senseiver (sparse heat): train only from sensor observations.

This is a standalone adaptation of the core Senseiver encoder/decoder idea for
the heat sparse datasets used in this repository. It does not use the original
Senseiver dataset loader, plotting code, argument parser, or PyTorch Lightning.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - convenience fallback for lean envs
    tqdm = lambda x, **_: x


@dataclass
class Config:
    data: str = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset.npz"
    obs_key: str = "sensor_noisy"  # or "sensor_clean"
    batch_size: int = 128
    val_batch_size: int = 256
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 123
    k_neighbors: int = 128
    time_radius: int = 3
    space_bands: int = 32
    max_frequency: float = 32.0
    enc_preproc_ch: int = 128
    num_latents: int = 16
    latent_channels: int = 64
    num_layers: int = 3
    num_cross_attention_heads: int = 4
    num_self_attention_heads: int = 4
    num_self_attention_layers_per_block: int = 2
    dec_preproc_ch: int = 128
    dec_num_cross_attention_heads: int = 4
    dropout: float = 0.0
    grad_clip: float = 1.0
    patience: int = 12
    save: str = "/scratch/ab9738/fieldformer/baselines/checkpoints/senseiver_heatsparse_best.pt"


CFG = Config()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Senseiver on sparse heat sensor observations.")
    for field in fields(Config):
        value = getattr(CFG, field.name)
        arg = f"--{field.name}"
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    return Config(**vars(parser.parse_args()))


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


class Sequential(nn.Sequential):
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x: Any = inputs
        for module in self:
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x


def mlp(num_channels: int) -> Sequential:
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.module(*args, **kwargs)) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(num_q_channels, num_kv_channels, num_heads, dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.attention(self.q_norm(x_q), self.kv_norm(x_kv), pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(num_channels, num_channels, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.attention(self.norm(x), self.norm(x), pad_mask=pad_mask, attn_mask=attn_mask)


def cross_attention_layer(num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float) -> Sequential:
    return Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )


def self_attention_layer(num_channels: int, num_heads: int, dropout: float) -> Sequential:
    return Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout),
    )


def self_attention_block(num_layers: int, num_channels: int, num_heads: int, dropout: float) -> Sequential:
    return Sequential(*[self_attention_layer(num_channels, num_heads, dropout) for _ in range(num_layers)])


class Encoder(nn.Module):
    def __init__(
        self,
        input_ch: int,
        preproc_ch: int | None,
        num_latents: int,
        num_latent_channels: int,
        num_layers: int,
        num_cross_attention_heads: int,
        num_self_attention_heads: int,
        num_self_attention_layers_per_block: int,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        if preproc_ch:
            self.preproc = nn.Linear(input_ch, preproc_ch)
        else:
            self.preproc = None
            preproc_ch = input_ch

        def create_layer() -> Sequential:
            return Sequential(
                cross_attention_layer(num_latent_channels, preproc_ch, num_cross_attention_heads, dropout),
                self_attention_block(
                    num_self_attention_layers_per_block,
                    num_latent_channels,
                    num_self_attention_heads,
                    dropout,
                ),
            )

        self.layer_1 = create_layer()
        if num_layers > 1:
            self.layer_n = create_layer()
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz = x.shape[0]
        if self.preproc:
            x = self.preproc(x)
        x_latent = self.latent.unsqueeze(0).expand(bsz, -1, -1)
        x_latent = self.layer_1(x_latent, x, pad_mask)
        for _ in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)
        return x_latent


class Decoder(nn.Module):
    def __init__(
        self,
        coord_channels: int,
        preproc_ch: int | None,
        num_latent_channels: int,
        num_output_channels: int,
        num_cross_attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        q_channels = coord_channels + num_latent_channels
        if preproc_ch:
            self.preproc = nn.Linear(q_channels, preproc_ch)
            q_in = preproc_ch
        else:
            self.preproc = None
            q_in = q_channels

        self.cross_attention = cross_attention_layer(q_in, num_latent_channels, num_cross_attention_heads, dropout)
        self.postproc = nn.Linear(q_in, num_output_channels)
        self.output = nn.Parameter(torch.empty(1, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, latents: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz, n_query, _ = coords.shape
        output = self.output.unsqueeze(0).expand(bsz, n_query, -1)
        output = torch.cat([coords, output], dim=-1)
        if self.preproc:
            output = self.preproc(output)
        output = self.cross_attention(output, latents)
        return self.postproc(output)


class FourierPositionEncoder(nn.Module):
    def __init__(
        self,
        coord_min: torch.Tensor,
        coord_max: torch.Tensor,
        num_bands: int,
        max_frequency: float,
    ):
        super().__init__()
        self.register_buffer("coord_min", coord_min.float())
        self.register_buffer("coord_range", (coord_max - coord_min).float().clamp_min(1e-6))
        self.register_buffer("freqs", torch.linspace(1.0, max_frequency / 2.0, num_bands).float())
        self.out_channels = int(coord_min.numel() * num_bands * 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (coords - self.coord_min) / self.coord_range - 1.0
        angles = math.pi * z.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return enc.flatten(start_dim=-2)


class SenseiverSparse(nn.Module):
    def __init__(
        self,
        coord_min: torch.Tensor,
        coord_max: torch.Tensor,
        cfg: Config,
    ):
        super().__init__()
        self.pos_encoder = FourierPositionEncoder(coord_min, coord_max, cfg.space_bands, cfg.max_frequency)
        pos_ch = self.pos_encoder.out_channels

        self.encoder = Encoder(
            input_ch=1 + pos_ch,
            preproc_ch=cfg.enc_preproc_ch,
            num_latents=cfg.num_latents,
            num_latent_channels=cfg.latent_channels,
            num_layers=cfg.num_layers,
            num_cross_attention_heads=cfg.num_cross_attention_heads,
            num_self_attention_heads=cfg.num_self_attention_heads,
            num_self_attention_layers_per_block=cfg.num_self_attention_layers_per_block,
            dropout=cfg.dropout,
        )
        self.decoder = Decoder(
            coord_channels=pos_ch,
            preproc_ch=cfg.dec_preproc_ch,
            num_latent_channels=cfg.latent_channels,
            num_output_channels=1,
            num_cross_attention_heads=cfg.dec_num_cross_attention_heads,
            dropout=cfg.dropout,
        )

    def forward(self, query_coords: torch.Tensor, sensor_coords: torch.Tensor, sensor_values: torch.Tensor) -> torch.Tensor:
        sensor_pos = self.pos_encoder(sensor_coords)
        sensor_tokens = torch.cat([sensor_values.unsqueeze(-1), sensor_pos], dim=-1)
        latents = self.encoder(sensor_tokens)
        query_pos = self.pos_encoder(query_coords)
        return self.decoder(latents, query_pos).squeeze(-1)

    def forward_observed(
        self,
        q_lin: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals_norm: torch.Tensor,
        nb_idx: torch.Tensor,
    ) -> torch.Tensor:
        query_coords = obs_coords[q_lin].unsqueeze(1)
        sensor_coords = obs_coords[nb_idx]
        sensor_values = obs_vals_norm[nb_idx]
        return self(query_coords, sensor_coords, sensor_values).squeeze(1)

    def forward_continuous(
        self,
        xyt_q: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals_norm: torch.Tensor,
        nb_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self(xyt_q.unsqueeze(1), obs_coords[nb_idx], obs_vals_norm[nb_idx]).squeeze(1)


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


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)

    if cfg.obs_key in pack:
        sensor_values = pack[cfg.obs_key].astype(np.float32)
    elif "sensor_noisy" in pack:
        sensor_values = pack["sensor_noisy"].astype(np.float32)
    elif "sensor_clean" in pack:
        sensor_values = pack["sensor_clean"].astype(np.float32)
    else:
        raise KeyError("heat sparse dataset must contain sensor_noisy or sensor_clean")

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
    coord_min = torch.tensor([x_min, y_min, t_min], dtype=torch.float32, device=device)
    coord_max = torch.tensor([x_max, y_max, t_max], dtype=torch.float32, device=device)

    indexer = SparseNeighborIndexer(sensors_xy_t, t_grid_t, cfg.time_radius, cfg.k_neighbors)
    model = SenseiverSparse(coord_min, coord_max, cfg).to(device)

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
    print(f"[data] obs={n_obs} sensors={s_count} Nt={nt_count} value_mean={value_mean:.4e} value_std={value_std:.4e}")
    print(f"[model] SenseiverSparse parameters={n_params}")

    best_rmse = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        running = {"data": 0.0}
        n_batches = 0

        for q_lin in pbar:
            q_lin = q_lin.to(device)
            pred_norm = predict_observed_norm(q_lin)
            tgt_norm = obs_vals_norm[q_lin]
            loss = F.mse_loss(pred_norm, tgt_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running["data"] += loss.item()
            n_batches += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"data": f"{running['data']/n_batches:.4e}"})

        scheduler.step()
        rmse = val_rmse()
        print(
            f"[epoch {epoch:03d}] train_data={running['data']/max(1,n_batches):.4e} "
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
                    "variant": "senseiver_heat_sparse",
                    "obs_key": cfg.obs_key,
                    "num_sensors": int(s_count),
                    "num_times": int(nt_count),
                    "num_observations": int(n_obs),
                    "x_range": [x_min, x_max],
                    "y_range": [y_min, y_max],
                    "t_range": [t_min, t_max],
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
    main(parse_args())
