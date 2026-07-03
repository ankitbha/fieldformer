#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
CORE_SCRIPT_DIR = ROOT / "fieldformer_core" / "scripts"
for path in (ROOT, THIS_DIR, CORE_SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from baselines.models.data import build_observed_index_dataset, is_sensor_split_dataset
from sparse_eval import (
    DATASETS,
    MODELS,
    available_checkpoints,
    checkpoint_meta,
    choose_mask_key,
    choose_obs_key,
    ckpt_path,
    ensemble_ckpt_paths,
    implementation_key,
    load_checkpoint,
    parse_ensemble_seeds,
    sparse_observations,
)
from sparse_models import build_fmlp_ensemble_adapter, build_sparse_model
from sparse_neighbor_indexer import SplitAwareSparseNeighborIndexer


@dataclass
class Config:
    dataset: str = "heat"
    model: str = "ffag"
    batch_size: int = 4096
    max_queries: int = 50000
    warmup_batches: int = 5
    timed_repeats: int = 3
    output_path: str = ""
    output_dir: str = str(ROOT / "eval" / "main" / "timing_outputs")
    device: str = "cuda"
    obs_key: str = ""
    mask_key: str = ""
    ensemble_seeds: str = "101,102,103,104,105"
    ensemble_dir: str = ""
    slurm_array: bool = False
    tasks: str = ""


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = raw.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Benchmark sparse-test inference time for sparse-trained baselines.")
    for field in fields(Config):
        value = getattr(cfg, field.name)
        arg_type = parse_bool if isinstance(value, bool) else type(value)
        if isinstance(value, bool):
            parser.add_argument(f"--{field.name}", type=arg_type, nargs="?", const=True, default=value)
        else:
            parser.add_argument(f"--{field.name}", type=arg_type, default=value)
    args = parser.parse_args()
    return Config(**vars(args))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_slurm_array_selection(cfg: Config) -> Config:
    if not cfg.slurm_array:
        return cfg
    tasks = [part.strip() for part in cfg.tasks.split(",") if part.strip()]
    if not tasks:
        raise SystemExit("--slurm_array requires --tasks model:dataset[,model:dataset...]")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"SLURM_ARRAY_TASK_ID={task_id} outside task range 0..{len(tasks) - 1}")
    try:
        model, dataset = tasks[task_id].split(":", 1)
    except ValueError as exc:
        raise SystemExit(f"Malformed task {tasks[task_id]!r}; expected model:dataset") from exc
    cfg.model = model
    cfg.dataset = dataset
    return cfg


def output_path_for(cfg: Config, model_key: str, dataset_key: str) -> Path:
    if cfg.output_path:
        return Path(cfg.output_path)
    return Path(cfg.output_dir) / f"{model_key}-{dataset_key}.json"


def tensor_output_dim(pred: torch.Tensor) -> int:
    if pred.ndim <= 1:
        return 1
    return int(pred.shape[-1])


def prepare_runtime(cfg: Config) -> dict[str, Any]:
    dataset_key = cfg.dataset.lower()
    model_key = cfg.model.lower()
    if dataset_key not in DATASETS:
        raise SystemExit(f"Unknown dataset {cfg.dataset!r}. Expected one of: {sorted(DATASETS)}")
    if model_key not in MODELS:
        raise SystemExit(f"Unknown model {cfg.model!r}. Expected one of: {sorted(MODELS)}")

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    load_start = time.perf_counter()
    pack = np.load(DATASETS[dataset_key])
    ensemble_paths: list[Path] = []
    ensemble_ckpts: list[dict[str, Any]] = []
    if model_key == "fmlp_ensemble":
        ensemble_paths = ensemble_ckpt_paths(dataset_key, cfg.ensemble_seeds, cfg.ensemble_dir)
        missing = [str(path) for path in ensemble_paths if not path.exists()]
        if missing:
            raise SystemExit(
                "Ensemble checkpoint(s) not found:\n"
                + "\n".join(missing)
                + f"\n\nAvailable sparse checkpoints:\n{available_checkpoints()}"
            )
        ensemble_ckpts = [load_checkpoint(path, device) for path in ensemble_paths]
        ckpt = ensemble_ckpts[0]
        path = ensemble_paths[0]
    else:
        path = ckpt_path(model_key, dataset_key)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}\n\nAvailable sparse checkpoints:\n{available_checkpoints()}")
        ckpt = load_checkpoint(path, device)
    load_seconds = time.perf_counter() - load_start

    ckpt_cfg = ckpt.get("config", {})
    ckpt_data_path = Path(str(ckpt_cfg.get("data", "")))
    if model_key in {"recfno", "imputeformer", "senseiver"} and ckpt_data_path.exists():
        pack = np.load(ckpt_data_path)
    meta = checkpoint_meta(ckpt)
    split_ckpt_meta = meta.get("split", {})
    if not isinstance(split_ckpt_meta, dict):
        split_ckpt_meta = {}

    sensors_xy = pack["sensors_xy"].astype(np.float32)
    t_np = pack["t"].astype(np.float32)
    obs_key = choose_obs_key(pack, dataset_key, cfg.obs_key)
    obs_mask_key = choose_mask_key(pack, cfg.mask_key)
    impl_model_key = implementation_key(model_key)
    obs_coords_np, obs_vals_np, obs_mask_np, valid_idx = sparse_observations(pack, dataset_key, obs_key, obs_mask_key)

    train_frac = float(ckpt_cfg.get("train_frac", 0.8))
    val_frac = float(ckpt_cfg.get("val_frac", 0.1))
    seed = int(ckpt_cfg.get("split_seed", ckpt_cfg.get("seed", 123)))
    sensor_mask_np = pack[obs_mask_key].astype(np.float32) if obs_mask_key else None
    split = build_observed_index_dataset(
        dataset_key=dataset_key,
        pack=pack,
        n_obs=obs_coords_np.shape[0],
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        valid_idx=valid_idx,
        sensor_mask=sensor_mask_np,
        sensor_split_seed=ckpt_cfg.get("sensor_split_seed", split_ckpt_meta.get("sensor_split_seed")),
        val_sensors=int(ckpt_cfg.get("val_sensors", split_ckpt_meta.get("val_sensors", 3))),
        test_sensors=int(ckpt_cfg.get("test_sensors", split_ckpt_meta.get("test_sensors", 3))),
        min_valid_frac=float(ckpt_cfg.get("sensor_min_valid_frac", split_ckpt_meta.get("min_valid_frac", 0.10))),
    )
    split_meta = getattr(split, "meta", {})
    train_vals = obs_vals_np[split.train_idx.numpy()]
    obs_mean = train_vals.mean(axis=0) if train_vals.ndim == 2 else float(train_vals.mean())
    obs_std = train_vals.std(axis=0) + 1e-6 if train_vals.ndim == 2 else float(train_vals.std() + 1e-6)
    if impl_model_key == "svgp" and dataset_key in {"heat", "swe"}:
        obs_mean = obs_vals_np.mean(axis=0) if obs_vals_np.ndim == 2 else float(obs_vals_np.mean())
        obs_std = obs_vals_np.std(axis=0) + 1e-8 if obs_vals_np.ndim == 2 else float(obs_vals_np.std() + 1e-8)
    if impl_model_key == "svgp" and "obs_mean" in meta and "obs_std" in meta:
        obs_mean = meta["obs_mean"]
        obs_std = meta["obs_std"]
    if "val_mean" in meta and "val_std" in meta and meta["val_mean"] is not None and meta["val_std"] is not None:
        obs_mean = meta["val_mean"]
        obs_std = meta["val_std"]

    x_np = pack["x"].astype(np.float32)
    y_np = pack["y"].astype(np.float32)
    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())
    Lx, Ly, Tt = max(1e-6, x_max - x_min), max(1e-6, y_max - y_min), max(1e-6, t_max - t_min)

    obs_coords = torch.from_numpy(obs_coords_np).float().to(device)
    obs_vals = torch.from_numpy(obs_vals_np).float().to(device)
    sparse_test_idx = split.test_idx
    if cfg.max_queries > 0:
        sparse_test_idx = sparse_test_idx[: cfg.max_queries]
    if sparse_test_idx.numel() == 0:
        raise SystemExit(f"No sparse test queries available for dataset={dataset_key} model={model_key}")

    context_idx = split.train_idx
    if is_sensor_split_dataset(dataset_key) or impl_model_key == "ffag":
        context_idx = torch.cat([split.train_idx, split.val_idx])

    sync(device)
    setup_start = time.perf_counter()
    indexer = None
    if impl_model_key == "ffag":
        indexer = SplitAwareSparseNeighborIndexer(
            torch.from_numpy(sensors_xy).float().to(device),
            torch.from_numpy(t_np).float().to(device),
            int(ckpt_cfg.get("time_radius", 3)),
            int(ckpt_cfg.get("k_neighbors", 128)),
            allowed_indices=context_idx.to(device),
        )
    if model_key == "fmlp_ensemble":
        adapter = build_fmlp_ensemble_adapter(
            dataset_key=dataset_key,
            ckpts=ensemble_ckpts,
            device=device,
            obs_mean=obs_mean,
            obs_std=obs_std,
            x_min=x_min,
            y_min=y_min,
            t_min=t_min,
            Lx=Lx,
            Ly=Ly,
            Tt=Tt,
        )
    else:
        adapter = build_sparse_model(
            model_key=impl_model_key,
            dataset_key=dataset_key,
            ckpt=ckpt,
            data=pack,
            device=device,
            obs_mean=obs_mean,
            obs_std=obs_std,
            x_min=x_min,
            y_min=y_min,
            t_min=t_min,
            Lx=Lx,
            Ly=Ly,
            Tt=Tt,
            nt_count=t_np.shape[0],
            sensors_xy=sensors_xy,
            x_grid=x_np,
            y_grid=y_np,
            t_grid=t_np,
            train_idx=context_idx.numpy(),
            obs_coords_np=obs_coords_np,
            obs_vals_np=obs_vals_np,
            obs_mask_np=obs_mask_np,
        )
    adapter.eval()
    sync(device)
    setup_seconds = time.perf_counter() - setup_start

    return {
        "dataset_key": dataset_key,
        "model_key": model_key,
        "impl_model_key": impl_model_key,
        "device": device,
        "path": path,
        "ensemble_paths": ensemble_paths,
        "obs_key": obs_key,
        "mask_key": obs_mask_key,
        "split_meta": split_meta,
        "context_idx": context_idx,
        "obs_coords": obs_coords,
        "obs_vals": obs_vals,
        "test_idx": sparse_test_idx,
        "adapter": adapter,
        "indexer": indexer,
        "load_seconds": load_seconds,
        "setup_seconds": setup_seconds,
    }


def run_prediction_pass(
    *,
    adapter: Any,
    indexer: Any,
    obs_coords: torch.Tensor,
    obs_vals: torch.Tensor,
    test_idx: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[int, int]:
    seen = 0
    output_dim = 0
    with torch.inference_mode():
        for batch_no, start in enumerate(range(0, int(test_idx.numel()), batch_size)):
            if max_batches > 0 and batch_no >= max_batches:
                break
            q_lin = test_idx[start:start + batch_size].to(device)
            nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True) if adapter.needs_sensor_context else None
            pred = adapter.predict_observed(q_lin, obs_coords, obs_vals, nb_idx)
            seen += int(q_lin.numel())
            if output_dim <= 0:
                output_dim = tensor_output_dim(pred)
    return seen, output_dim


def benchmark(cfg: Config) -> dict[str, Any]:
    cfg = apply_slurm_array_selection(cfg)
    runtime = prepare_runtime(cfg)
    device: torch.device = runtime["device"]
    adapter = runtime["adapter"]
    indexer = runtime["indexer"]
    obs_coords = runtime["obs_coords"]
    obs_vals = runtime["obs_vals"]
    test_idx = runtime["test_idx"]

    warmup_batches = max(0, int(cfg.warmup_batches))
    timed_repeats = max(1, int(cfg.timed_repeats))
    batch_size = max(1, int(cfg.batch_size))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup_seen = 0
    warmup_output_dim = 0
    if warmup_batches > 0:
        warmup_seen, warmup_output_dim = run_prediction_pass(
            adapter=adapter,
            indexer=indexer,
            obs_coords=obs_coords,
            obs_vals=obs_vals,
            test_idx=test_idx,
            batch_size=batch_size,
            device=device,
            max_batches=warmup_batches,
        )
        sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    repeat_seconds: list[float] = []
    repeat_queries: list[int] = []
    output_dim = warmup_output_dim
    for _ in range(timed_repeats):
        sync(device)
        start = time.perf_counter()
        seen, dim = run_prediction_pass(
            adapter=adapter,
            indexer=indexer,
            obs_coords=obs_coords,
            obs_vals=obs_vals,
            test_idx=test_idx,
            batch_size=batch_size,
            device=device,
        )
        sync(device)
        repeat_seconds.append(time.perf_counter() - start)
        repeat_queries.append(seen)
        output_dim = dim or output_dim

    seconds_total = float(sum(repeat_seconds))
    num_timed_queries = int(sum(repeat_queries))
    ms_per_query = 1000.0 * seconds_total / max(1, num_timed_queries)
    queries_per_second = num_timed_queries / max(seconds_total, 1e-12)
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    dataset_key = runtime["dataset_key"]
    model_key = runtime["model_key"]
    result: dict[str, Any] = {
        "dataset": dataset_key,
        "model": model_key,
        "implementation_model": runtime["impl_model_key"],
        "checkpoint": [str(path) for path in runtime["ensemble_paths"]] if model_key == "fmlp_ensemble" else str(runtime["path"]),
        "obs_key": runtime["obs_key"],
        "mask_key": runtime["mask_key"],
        "device": str(device),
        "batch_size": batch_size,
        "max_queries": int(cfg.max_queries),
        "num_queries_per_repeat": int(test_idx.numel()),
        "num_timed_queries": num_timed_queries,
        "warmup_batches": warmup_batches,
        "warmup_queries": int(warmup_seen),
        "timed_repeats": timed_repeats,
        "repeat_seconds": repeat_seconds,
        "repeat_queries": repeat_queries,
        "seconds_total": seconds_total,
        "ms_per_query": ms_per_query,
        "queries_per_second": queries_per_second,
        "load_seconds": float(runtime["load_seconds"]),
        "setup_seconds": float(runtime["setup_seconds"]),
        "needs_sensor_context": bool(adapter.needs_sensor_context),
        "output_dim": int(output_dim),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_mb": peak_gpu_memory_bytes / (1024.0 * 1024.0),
    }
    if model_key == "fmlp_ensemble":
        result["ensemble_seeds"] = parse_ensemble_seeds(cfg.ensemble_seeds)
        result["ensemble_aggregation"] = "mean"
    split_meta = runtime["split_meta"]
    if split_meta:
        result["split"] = split_meta
        result["context_sensor_ids"] = split_meta.get("train_sensor_ids", []) + split_meta.get("val_sensor_ids", [])
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main(cfg: Config) -> None:
    cfg = apply_slurm_array_selection(cfg)
    result = benchmark(cfg)
    out = output_path_for(cfg, result["model"], result["dataset"])
    write_result(out, result)
    print(
        f"[timing] dataset={result['dataset']} model={result['model']} "
        f"ms/query={result['ms_per_query']:.6g} qps={result['queries_per_second']:.6g} "
        f"queries/repeat={result['num_queries_per_repeat']} repeats={result['timed_repeats']}"
    )
    print(f"[write] {out}")


if __name__ == "__main__":
    main(parse_args())
