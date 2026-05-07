#!/usr/bin/env python3
"""Model adapters for sparse architecture-ablation eval."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
MAIN_EVAL_DIR = ROOT / "eval" / "main"
ABLATION_SCRIPT_DIR = ROOT / "ablations" / "architecture" / "scripts"
for path in (ROOT, MAIN_EVAL_DIR, ABLATION_SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval.main.sparse_models import EvalAdapter, build_sparse_model as build_main_sparse_model, checkpoint_meta, cfg_obj, output_dim_for, _get


ABLATION_MODELS = {
    "ffag_nophys",
    "ffag_mlp",
    "ffag_npgf",
}
ALIASES = {
    "ffag-no-physics": "ffag_nophys",
    "ffag_nophysics": "ffag_nophys",
    "ffag-nophys": "ffag_nophys",
    "ffag-mlp": "ffag_mlp",
    "npgf": "ffag_npgf",
    "ffag-npgf": "ffag_npgf",
    "ffag_gamma_field": "ffag_npgf",
    "ffag-gamma-field": "ffag_npgf",
}


def canonical_model_key(model_key: str) -> str:
    return ALIASES.get(model_key.lower(), model_key.lower())


def build_ablation_sparse_model(
    *,
    model_key: str,
    dataset_key: str,
    ckpt: dict[str, Any],
    data: Any,
    device: torch.device,
    obs_mean: float | np.ndarray,
    obs_std: float | np.ndarray,
    x_min: float,
    y_min: float,
    t_min: float,
    Lx: float,
    Ly: float,
    Tt: float,
    nt_count: int,
    sensors_xy: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
    obs_coords_np: np.ndarray | None = None,
    obs_vals_np: np.ndarray | None = None,
    obs_mask_np: np.ndarray | None = None,
) -> EvalAdapter:
    model_key = canonical_model_key(model_key)
    meta = checkpoint_meta(ckpt)
    architecture = str(meta.get("architecture", "")).lower()
    if model_key == "ffag" and architecture == "mlp_token_mixer":
        model_key = "ffag_mlp"
    if model_key == "ffag" and architecture == "ffag_npgf":
        model_key = "ffag_npgf"
    if model_key in {"ffag_nophys", "ffag"}:
        return build_main_sparse_model(
            model_key="ffag",
            dataset_key=dataset_key,
            ckpt=ckpt,
            data=data,
            device=device,
            obs_mean=obs_mean,
            obs_std=obs_std,
            x_min=x_min,
            y_min=y_min,
            t_min=t_min,
            Lx=Lx,
            Ly=Ly,
            Tt=Tt,
            nt_count=nt_count,
            sensors_xy=sensors_xy,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
            train_idx=train_idx,
            obs_coords_np=obs_coords_np,
            obs_vals_np=obs_vals_np,
            obs_mask_np=obs_mask_np,
        )
    if model_key == "ffag_npgf":
        from ffag_sparse_npgf_model import class_for_dataset

        cfg = cfg_obj(ckpt.get("config"))
        state = ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict", ckpt)
        meta_gamma = meta.get("gamma_field", {})
        if not isinstance(meta_gamma, dict):
            meta_gamma = {}
        base_gammas = meta_gamma.get("base_gammas", _get(cfg, "base_gammas", (1.0, 1.0, 0.5 if dataset_key in {"pol", "govpol", "atm", "govpolsplit", "atmsplit"} else 1.0)))
        model_cls = class_for_dataset(dataset_key)
        kwargs = {
            "gamma_hidden": int(_get(cfg, "gamma_hidden", meta_gamma.get("hidden", 32))),
            "gamma_layers": int(_get(cfg, "gamma_layers", meta_gamma.get("layers", 2))),
            "gamma_max_delta": float(_get(cfg, "gamma_max_delta", meta_gamma.get("max_delta", 2.0))),
            "base_gammas": tuple(float(x) for x in base_gammas),
        }
        try:
            model = model_cls(
                _get(cfg, "d_model", 128),
                _get(cfg, "nhead", 4),
                _get(cfg, "layers", 3),
                _get(cfg, "d_ff", 256),
                out_dim=output_dim_for(dataset_key, ckpt, cfg),
                **kwargs,
            ).to(device)
        except TypeError:
            model = model_cls(
                _get(cfg, "d_model", 128),
                _get(cfg, "nhead", 4),
                _get(cfg, "layers", 3),
                _get(cfg, "d_ff", 256),
                **kwargs,
            ).to(device)
        if hasattr(model, "set_coord_ranges"):
            model.set_coord_ranges(
                x_min=x_min,
                x_max=x_min + Lx,
                y_min=y_min,
                y_max=y_min + Ly,
                t_min=t_min,
                t_max=t_min + Tt,
            )
        model.load_state_dict(state)
        return EvalAdapter(
            model,
            model_key="ffag",
            dataset_key=dataset_key,
            obs_mean=obs_mean,
            obs_std=obs_std,
            x_min=x_min,
            y_min=y_min,
            t_min=t_min,
            Lx=Lx,
            Ly=Ly,
            Tt=Tt,
        ).to(device)

    if model_key != "ffag_mlp":
        raise KeyError(model_key)

    from ffag_sparse_mlp_model import class_for_dataset

    cfg = cfg_obj(ckpt.get("config"))
    state = ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict", ckpt)
    model = class_for_dataset(dataset_key)(
        _get(cfg, "d_model", 128),
        _get(cfg, "nhead", 4),
        _get(cfg, "layers", 3),
        _get(cfg, "d_ff", 256),
    ).to(device)
    model.load_state_dict(state)
    return EvalAdapter(
        model,
        model_key="ffag",
        dataset_key=dataset_key,
        obs_mean=obs_mean,
        obs_std=obs_std,
        x_min=x_min,
        y_min=y_min,
        t_min=t_min,
        Lx=Lx,
        Ly=Ly,
        Tt=Tt,
    ).to(device)


build_sparse_model = build_ablation_sparse_model
