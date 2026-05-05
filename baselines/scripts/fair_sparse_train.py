from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from baselines.models.data import ObservedIndexDataset, build_observed_tuples, mask_key, sensor_key
from baselines.models.imputeformer import FixedNodeImputeFormer
from baselines.models.recfno import VoronoiFNO2d
from baselines.models.senseiver import Senseiver


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_key(cfg: Any) -> str:
    if hasattr(cfg, "dataset"):
        return str(cfg.dataset)
    data = str(cfg.data).lower()
    if "gov_sensor" in data or "govdata" in data:
        return "govpol"
    if "pollution" in data:
        return "pol"
    if "swe" in data:
        return "swe"
    return "heat"


def save_path(cfg: Any, model_key: str, key: str) -> Path:
    if getattr(cfg, "save", ""):
        return Path(cfg.save)
    return Path(f"/scratch/ab9738/fieldformer/baselines/checkpoints/{model_key}_{key}sparse_best.pt")


def load_sparse_arrays(cfg: Any) -> dict[str, Any]:
    key = dataset_key(cfg)
    pack = np.load(cfg.data)
    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    obs_key = sensor_key(pack, key, getattr(cfg, "obs_key", ""))
    obs_mask_key = mask_key(pack, getattr(cfg, "mask_key", ""))
    values = pack[obs_key].astype(np.float32)
    sensor_mask = pack[obs_mask_key].astype(np.float32) if obs_mask_key else None
    if sensor_mask is not None:
        coords, vals, masks = build_observed_tuples(sensors_xy, t_np, values, sensor_mask)
        valid_idx = np.flatnonzero(masks.reshape(masks.shape[0], -1).any(axis=1))
    else:
        coords, vals = build_observed_tuples(sensors_xy, t_np, values)
        masks = np.ones_like(vals, dtype=np.float32)
        valid_idx = None
    split = ObservedIndexDataset(coords.shape[0], cfg.train_frac, cfg.val_frac, cfg.seed, valid_idx=valid_idx)
    return {
        "key": key,
        "pack": pack,
        "obs_key": obs_key,
        "mask_key": obs_mask_key,
        "sensors_xy": sensors_xy,
        "t": t_np,
        "values": values,
        "masks": sensor_mask if sensor_mask is not None else np.ones_like(values, dtype=np.float32),
        "coords": coords,
        "vals": vals,
        "flat_masks": masks,
        "split": split,
    }


def sensor_grid_indices(pack: Any, sensors_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_np = pack["x"].astype(np.float32)
    y_np = pack["y"].astype(np.float32)
    ix = np.abs(x_np[None, :] - sensors_xy[:, 0:1]).argmin(axis=1).astype(np.int64)
    iy = np.abs(y_np[None, :] - sensors_xy[:, 1:2]).argmin(axis=1).astype(np.int64)
    return ix, iy


def split_masks(split: ObservedIndexDataset, n_sensors: int, n_times: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = []
    for idx in (split.train_idx.numpy(), split.val_idx.numpy(), split.test_idx.numpy()):
        mask = np.zeros((n_sensors, n_times), dtype=bool)
        mask[idx // n_times, idx % n_times] = True
        masks.append(mask)
    return masks[0], masks[1], masks[2]


class IndexDataset(Dataset):
    def __init__(self, idx: torch.Tensor):
        self.idx = idx.long()

    def __len__(self) -> int:
        return int(self.idx.numel())

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.idx[item]


def train_recfno(cfg: Any) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sparse_arrays(cfg)
    key, pack = data["key"], data["pack"]
    values = data["values"]
    masks = data["masks"].astype(np.float32)
    if values.ndim == 2:
        values = values[..., None]
        masks = masks[..., None]
    out_dim = int(values.shape[-1])
    sensors_xy, t_np = data["sensors_xy"], data["t"]
    n_sensors, n_times = values.shape[:2]
    train_mask, val_mask, _ = split_masks(data["split"], n_sensors, n_times)
    ix, iy = sensor_grid_indices(pack, sensors_xy)
    x_np, y_np = pack["x"].astype(np.float32), pack["y"].astype(np.float32)
    nx, ny = len(x_np), len(y_np)
    xx, yy = np.meshgrid((x_np - x_np.min()) / max(np.ptp(x_np), 1e-6), (y_np - y_np.min()) / max(np.ptp(y_np), 1e-6), indexing="ij")
    coord_grid = torch.from_numpy(np.stack([xx, yy], axis=0).astype(np.float32)).to(device)

    model = VoronoiFNO2d(cfg.modes1, cfg.modes2, cfg.width, in_channels=2 * out_dim + 2, out_channels=3 if key == "swe" else out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    bad = 0
    path = save_path(cfg, "recfno", key)
    path.parent.mkdir(parents=True, exist_ok=True)

    def make_input(times: torch.Tensor, context_mask_np: np.ndarray, drop_lin: torch.Tensor | None = None) -> torch.Tensor:
        bsz = int(times.numel())
        val_grid = torch.zeros(bsz, out_dim, nx, ny, device=device)
        mask_grid = torch.zeros(bsz, out_dim, nx, ny, device=device)
        for b, k_t in enumerate(times.detach().cpu().numpy()):
            m = context_mask_np[:, k_t].copy()
            if drop_lin is not None:
                q = drop_lin[(drop_lin % n_times) == int(k_t)].detach().cpu().numpy()
                m[q // n_times] = False
            channel_mask = masks[m, k_t].astype(np.float32)
            val_grid[b, :, ix[m], iy[m]] = torch.from_numpy(values[m, k_t].T).float().to(device)
            mask_grid[b, :, ix[m], iy[m]] = torch.from_numpy(channel_mask.T).float().to(device)
        coords = coord_grid.unsqueeze(0).expand(bsz, -1, -1, -1)
        return torch.cat([val_grid, mask_grid, coords], dim=1)

    train_dl = DataLoader(IndexDataset(data["split"].train_idx), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(IndexDataset(data["split"].val_idx), batch_size=cfg.val_batch_size, shuffle=False)

    def eval_rmse() -> float:
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for q_lin in val_dl:
                q_lin = q_lin.to(device)
                times = torch.unique(q_lin % n_times)
                pred_grid = model(make_input(times, train_mask))
                time_to_b = {int(k.item()): b for b, k in enumerate(times)}
                pred = torch.stack([pred_grid[time_to_b[int((q % n_times).item())], :, ix[int((q // n_times).item())], iy[int((q // n_times).item())]] for q in q_lin])
                tgt = torch.from_numpy(values[(q_lin // n_times).cpu().numpy(), (q_lin % n_times).cpu().numpy()]).float().to(device)
                m = torch.from_numpy(masks[(q_lin // n_times).cpu().numpy(), (q_lin % n_times).cpu().numpy()]).float().to(device)
                se += ((((pred - tgt) ** 2) * m).sum()).item()
                n += int(m.sum().item())
        model.train()
        return math.sqrt(se / max(1, n))

    for epoch in range(1, cfg.epochs + 1):
        total, batches = 0.0, 0
        for q_lin in tqdm(train_dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False):
            q_lin = q_lin.to(device)
            times = torch.unique(q_lin % n_times)
            pred_grid = model(make_input(times, train_mask, drop_lin=q_lin))
            time_to_b = {int(k.item()): b for b, k in enumerate(times)}
            pred = torch.stack([pred_grid[time_to_b[int((q % n_times).item())], :, ix[int((q // n_times).item())], iy[int((q // n_times).item())]] for q in q_lin])
            tgt = torch.from_numpy(values[(q_lin // n_times).cpu().numpy(), (q_lin % n_times).cpu().numpy()]).float().to(device)
            m = torch.from_numpy(masks[(q_lin // n_times).cpu().numpy(), (q_lin % n_times).cpu().numpy()]).float().to(device)
            loss = (((pred - tgt) ** 2) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item()
            batches += 1
        rmse = eval_rmse()
        print(f"[epoch {epoch:03d}] train_mse={total/max(1,batches):.4e} val_rmse={rmse:.6f}")
        if rmse < best:
            best, bad = rmse, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_rmse": best, "config": asdict(cfg), "meta": {"variant": f"recfno_{key}_sparse", "obs_key": data["obs_key"], "mask_key": data["mask_key"], "out_dim": 3 if key == "swe" else out_dim, "output_dim": 3 if key == "swe" else out_dim}}, path)
            print(f"[save] best checkpoint -> {path}")
        else:
            bad += 1
            if bad >= cfg.patience:
                break


def train_senseiver(cfg: Any) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sparse_arrays(cfg)
    key = data["key"]
    values = data["values"]
    masks = data["masks"].astype(np.float32)
    if values.ndim == 2:
        values = values[..., None]
        masks = masks[..., None]
    out_dim = int(values.shape[-1])
    sensors_xy, t_np = data["sensors_xy"], data["t"]
    n_sensors, n_times = values.shape[:2]
    train_mask, _, _ = split_masks(data["split"], n_sensors, n_times)
    x_min, y_min, t_min = data["coords"].min(axis=0)
    span = np.maximum(data["coords"].max(axis=0) - data["coords"].min(axis=0), 1e-6)
    vals_mean = values.reshape(-1, out_dim)[masks.reshape(-1, out_dim).astype(bool)].mean() if out_dim == 1 else np.nan_to_num(
        np.divide((values * masks).sum(axis=(0, 1)), masks.sum(axis=(0, 1)), out=np.zeros(out_dim, dtype=np.float32), where=masks.sum(axis=(0, 1)) > 0)
    ).astype(np.float32)
    vals_std = np.ones(out_dim, dtype=np.float32)
    for c in range(out_dim):
        valid = masks[..., c].astype(bool)
        vals_std[c] = float(values[..., c][valid].std() + 1e-6) if valid.any() else 1.0
    vals_mean = np.asarray(vals_mean, dtype=np.float32).reshape(out_dim)
    coords_t = torch.from_numpy(((data["coords"] - np.array([x_min, y_min, t_min], dtype=np.float32)) / span).astype(np.float32)).to(device)
    vals_t = torch.from_numpy(((data["vals"] - vals_mean) / vals_std).astype(np.float32)).to(device)
    flat_masks_t = torch.from_numpy(data["flat_masks"].astype(np.float32)).to(device)
    model = Senseiver(
        sensor_feature_dim=3 + 2 * out_dim,
        query_feature_dim=3,
        num_latents=cfg.num_latents,
        latent_channels=cfg.latent_channels,
        num_layers=cfg.num_layers,
        cross_heads=cfg.cross_heads,
        decoder_heads=getattr(cfg, "decoder_heads", 1),
        self_heads=cfg.self_heads,
        self_layers=cfg.self_layers,
        out_dim=3 if key == "swe" else out_dim,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_dl = DataLoader(IndexDataset(data["split"].train_idx), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(IndexDataset(data["split"].val_idx), batch_size=cfg.val_batch_size, shuffle=False)
    path = save_path(cfg, "senseiver", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    best, bad = float("inf"), 0

    def sensor_tokens(times: torch.Tensor, drop_lin: torch.Tensor | None = None) -> torch.Tensor:
        toks = []
        sxy = torch.from_numpy(((sensors_xy - np.array([x_min, y_min], dtype=np.float32)) / span[:2]).astype(np.float32)).to(device)
        for k in times.detach().cpu().numpy():
            m = train_mask[:, k].copy()
            if drop_lin is not None:
                q = drop_lin[(drop_lin % n_times) == int(k)].detach().cpu().numpy()
                m[q // n_times] = False
            base_mask = torch.from_numpy(m.astype(np.float32))[:, None].to(device)
            channel_mask = torch.from_numpy(masks[:, k].astype(np.float32)).to(device) * base_mask
            val = torch.from_numpy(((values[:, k] - vals_mean) / vals_std).astype(np.float32)).to(device) * channel_mask
            tt = torch.full((n_sensors, 1), (t_np[k] - t_min) / span[2], device=device)
            toks.append(torch.cat([sxy, tt, val, channel_mask], dim=-1))
        return torch.stack(toks, dim=0)

    def first_query(out: torch.Tensor) -> torch.Tensor:
        if out.ndim == 2:
            return out
        return out[:, 0, :]

    def run_eval() -> float:
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for q_lin in val_dl:
                q_lin = q_lin.to(device)
                times = torch.unique(q_lin % n_times)
                time_to_b = {int(k.item()): b for b, k in enumerate(times)}
                queries = torch.stack([coords_t[q] for q in q_lin]).unsqueeze(1)
                preds = []
                toks = sensor_tokens(times)
                for i, q in enumerate(q_lin):
                    b = time_to_b[int((q % n_times).item())]
                    preds.append(first_query(model(toks[b:b + 1], queries[i:i + 1])))
                pred = torch.cat(preds) * torch.from_numpy(vals_std).to(device) + torch.from_numpy(vals_mean).to(device)
                tgt = torch.from_numpy(values[(q_lin // n_times).cpu().numpy(), (q_lin % n_times).cpu().numpy()]).float().to(device)
                m = torch.from_numpy(masks[(q_lin // n_times).cpu().numpy(), (q_lin % n_times).cpu().numpy()]).float().to(device)
                se += ((((pred - tgt) ** 2) * m).sum()).item()
                n += int(m.sum().item())
        model.train()
        return math.sqrt(se / max(1, n))

    for epoch in range(1, cfg.epochs + 1):
        total, batches = 0.0, 0
        for q_lin in tqdm(train_dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False):
            q_lin = q_lin.to(device)
            times = torch.unique(q_lin % n_times)
            time_to_b = {int(k.item()): b for b, k in enumerate(times)}
            toks = sensor_tokens(times, drop_lin=q_lin)
            queries = torch.stack([coords_t[q] for q in q_lin]).unsqueeze(1)
            preds = []
            for i, q in enumerate(q_lin):
                b = time_to_b[int((q % n_times).item())]
                preds.append(first_query(model(toks[b:b + 1], queries[i:i + 1])))
            pred = torch.cat(preds)
            loss = (((pred - vals_t[q_lin]) ** 2) * flat_masks_t[q_lin]).sum() / flat_masks_t[q_lin].sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item()
            batches += 1
        rmse = run_eval()
        print(f"[epoch {epoch:03d}] train_mse={total/max(1,batches):.4e} val_rmse={rmse:.6f}")
        if rmse < best:
            best, bad = rmse, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_rmse": best, "config": asdict(cfg), "meta": {"variant": f"senseiver_{key}_sparse", "obs_key": data["obs_key"], "mask_key": data["mask_key"], "val_mean": vals_mean.tolist(), "val_std": vals_std.tolist(), "out_dim": 3 if key == "swe" else out_dim, "output_dim": 3 if key == "swe" else out_dim}}, path)
            print(f"[save] best checkpoint -> {path}")
        else:
            bad += 1
            if bad >= cfg.patience:
                break


def train_imputeformer(cfg: Any) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sparse_arrays(cfg)
    key = data["key"]
    values = data["values"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    if values.ndim == 2:
        values = values[..., None]
        masks = masks[..., None]
    out_dim = int(values.shape[-1])
    n_sensors, n_times = values.shape[:2]
    train_mask, val_mask, _ = split_masks(data["split"], n_sensors, n_times)
    train_channel_mask = train_mask[..., None] * masks
    mean = np.divide(
        (values * train_channel_mask).sum(axis=(0, 1)),
        train_channel_mask.sum(axis=(0, 1)),
        out=np.zeros(out_dim, dtype=np.float32),
        where=train_channel_mask.sum(axis=(0, 1)) > 0,
    ).astype(np.float32)
    std = np.ones(out_dim, dtype=np.float32)
    for c in range(out_dim):
        valid = train_channel_mask[..., c].astype(bool)
        std[c] = float(values[..., c][valid].std() + 1e-6) if valid.any() else 1.0
    norm_values = ((values - mean) / std).astype(np.float32)
    windows = min(cfg.windows, n_times)
    model = FixedNodeImputeFormer(n_sensors, windows, input_dim=2 * out_dim, output_dim=3 if key == "swe" else out_dim, input_embedding_dim=cfg.input_embedding_dim, learnable_embedding_dim=cfg.learnable_embedding_dim, num_layers=cfg.num_layers, num_temporal_heads=cfg.num_temporal_heads, dim_proj=cfg.dim_proj, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    starts = np.arange(0, max(1, n_times - windows + 1), max(1, cfg.window_stride), dtype=np.int64)
    path = save_path(cfg, "imputeformer", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    best, bad = float("inf"), 0

    def batch_window(starts_np: np.ndarray, target_mask_np: np.ndarray, context_mask_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vals, ctx, tgt = [], [], []
        for st in starts_np:
            sl = slice(int(st), int(st) + windows)
            vals.append(norm_values[:, sl])
            ctx.append((context_mask_np[:, sl, None] * masks[:, sl]).astype(np.float32))
            tgt.append((target_mask_np[:, sl, None] * masks[:, sl]).astype(np.float32))
        return (
            torch.from_numpy(np.stack(vals)).float().to(device),
            torch.from_numpy(np.stack(ctx)).float().to(device),
            torch.from_numpy(np.stack(tgt)).float().to(device),
        )

    for epoch in range(1, cfg.epochs + 1):
        rng = np.random.default_rng(cfg.seed + epoch)
        rng.shuffle(starts)
        total, batches = 0.0, 0
        for i in tqdm(range(0, len(starts), cfg.batch_size), desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False):
            bstarts = starts[i:i + cfg.batch_size]
            target_mask = train_mask.copy()
            drop = rng.random(target_mask.shape) < cfg.mask_rate
            target_mask &= drop
            context_mask = train_mask & ~target_mask
            vals, ctx, tgt = batch_window(bstarts, target_mask, context_mask)
            pred = model(vals, ctx)
            if tgt.sum() <= 0:
                continue
            loss = (((pred - vals) ** 2) * tgt).sum() / tgt.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item()
            batches += 1
        with torch.no_grad():
            se, n = 0.0, 0
            for i in range(0, len(starts), cfg.val_batch_size):
                vals, ctx, tgt = batch_window(starts[i:i + cfg.val_batch_size], val_mask, train_mask)
                pred = model(vals, ctx)
                std_t = torch.from_numpy(std).float().to(device)
                se += ((((pred - vals) * std_t) ** 2) * tgt).sum().item()
                n += int(tgt.sum().item())
            rmse = math.sqrt(se / max(1, n))
        print(f"[epoch {epoch:03d}] train_mse={total/max(1,batches):.4e} val_rmse={rmse:.6f}")
        if rmse < best:
            best, bad = rmse, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_rmse": best, "config": asdict(cfg), "meta": {"variant": f"imputeformer_{key}_sparse", "obs_key": data["obs_key"], "mask_key": data["mask_key"], "val_mean": mean.tolist(), "val_std": std.tolist(), "out_dim": 3 if key == "swe" else out_dim, "output_dim": 3 if key == "swe" else out_dim}}, path)
            print(f"[save] best checkpoint -> {path}")
        else:
            bad += 1
            if bad >= cfg.patience:
                break
