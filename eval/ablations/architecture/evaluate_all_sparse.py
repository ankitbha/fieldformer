#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
MAIN_EVAL_DIR = ROOT / "eval" / "main"
for path in (ROOT, THIS_DIR, MAIN_EVAL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval.main import sparse_eval

_SPEC = importlib.util.spec_from_file_location("architecture_ablation_sparse_models", THIS_DIR / "sparse_models.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load ablation sparse models from {THIS_DIR / 'sparse_models.py'}")
_ABLATION_SPARSE_MODELS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ABLATION_SPARSE_MODELS)
ABLATION_MODELS = _ABLATION_SPARSE_MODELS.ABLATION_MODELS
canonical_model_key = _ABLATION_SPARSE_MODELS.canonical_model_key
build_ablation_sparse_model = _ABLATION_SPARSE_MODELS.build_ablation_sparse_model


DEFAULT_DATASETS = ("heat", "pol", "swe")
DEFAULT_MODELS = ("ffag_nophys", "ffag_mlp")
CHECKPOINT_DIR = ROOT / "ablations" / "architecture" / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse eval for architecture ablation models/datasets.")
    parser.add_argument("--output_dir", default=str(ROOT / "eval" / "ablations" / "architecture" / "outputs"), help="Directory for per-run JSON files.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max_sparse_test", type=int, default=0, help="Optional cap on sparse test points per run; 0 evaluates all.")
    parser.add_argument("--max_full_field", type=int, default=0, help="Optional cap on full-field points per run; 0 evaluates all.")
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Bootstrap resamples for metric standard deviations; 0 disables.")
    parser.add_argument("--bootstrap_seed", type=int, default=123, help="Seed for bootstrap resampling.")
    parser.add_argument("--slurm_array", action="store_true", help="Run one model/dataset pair selected by SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def ckpt_path(model_key: str, dataset_key: str) -> Path:
    model_key = canonical_model_key(model_key)
    if model_key == "ffag_nophys":
        return CHECKPOINT_DIR / f"ffag_{dataset_key}sparse_nophys_best.pt"
    if model_key == "ffag_mlp":
        return CHECKPOINT_DIR / f"ffag_mlp_{dataset_key}sparse_best.pt"
    raise KeyError(model_key)


def available_checkpoints() -> str:
    if not CHECKPOINT_DIR.exists():
        return ""
    return "\n".join(str(p.relative_to(ROOT)) for p in sorted(CHECKPOINT_DIR.glob("*sparse*_best.pt")))


def implementation_key(model_key: str) -> str:
    model_key = canonical_model_key(model_key)
    if model_key in ABLATION_MODELS:
        return "ffag"
    return model_key


def apply_slurm_array_selection(args: argparse.Namespace) -> None:
    if not args.slurm_array:
        return

    task_id_raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_raw is None:
        raise SystemExit("--slurm_array requires SLURM_ARRAY_TASK_ID to be set")
    task_id = int(task_id_raw)
    combos = [(dataset, model) for dataset in args.datasets for model in args.models]
    if task_id < 0 or task_id >= len(combos):
        raise SystemExit(f"SLURM_ARRAY_TASK_ID={task_id} is out of range for {len(combos)} tasks")

    dataset, model = combos[task_id]
    args.datasets = [dataset]
    args.models = [model]
    print(f"[array] task_id={task_id} dataset={dataset} model={model}")


def install_ablation_hooks() -> None:
    sparse_eval.MODELS = set(ABLATION_MODELS)
    sparse_eval.ckpt_path = ckpt_path
    sparse_eval.available_checkpoints = available_checkpoints
    sparse_eval.implementation_key = implementation_key
    sparse_eval.build_sparse_model = build_ablation_sparse_model


def main() -> None:
    args = parse_args()
    args.models = [canonical_model_key(model) for model in args.models]
    install_ablation_hooks()
    apply_slurm_array_selection(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(args.models)
    done = 0
    ok = 0
    for dataset in args.datasets:
        for model in args.models:
            done += 1
            exp = f"{model}-{dataset}"
            result_path = out_dir / f"{exp}.json"
            print(f"\n[{done}/{total}] evaluating {exp}")
            cfg = sparse_eval.Config(
                dataset=dataset,
                model=model,
                batch_size=args.batch_size,
                output_path=str(result_path),
                device=args.device,
                obs_key="",
                max_sparse_test=args.max_sparse_test,
                max_full_field=args.max_full_field,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            try:
                sparse_eval.main(cfg)
                ok += 1
            except BaseException as exc:
                print(f"[error] {exp}: {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc()
                if args.stop_on_error:
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\n[done] {ok}/{total} evaluations succeeded.")


if __name__ == "__main__":
    main()
