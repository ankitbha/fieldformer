#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for path in (ROOT, THIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sparse_eval import Config, main as run_sparse_eval


DEFAULT_DATASETS = ("heat", "pol", "swe")
DEFAULT_MODELS = ("ffag", "fmlp", "fmlp_pinn", "siren", "siren_pinn", "svgp", "recfno", "imputeformer", "senseiver")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse eval for all selected models/datasets.")
    parser.add_argument("--output_dir", default=str(ROOT / "eval" / "main" / "outputs"), help="Directory for per-run JSON files.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max_sparse_test", type=int, default=0, help="Optional cap on sparse test points per run; 0 evaluates all.")
    parser.add_argument("--max_full_field", type=int, default=0, help="Optional cap on full-field points per run; 0 evaluates all.")
    parser.add_argument("--senseiver_full_field_fraction", type=float, default=0.10, help="Fraction of the full field to sample for Senseiver full-field eval.")
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Bootstrap resamples for metric standard deviations; 0 disables.")
    parser.add_argument("--bootstrap_seed", type=int, default=123, help="Seed for bootstrap resampling.")
    parser.add_argument("--slurm_array", action="store_true", help="Run one model/dataset pair selected by SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
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
            cfg = Config(
                dataset=dataset,
                model=model,
                batch_size=args.batch_size,
                output_path=str(result_path),
                device=args.device,
                obs_key="",
                max_sparse_test=args.max_sparse_test,
                max_full_field=args.max_full_field,
                senseiver_full_field_fraction=args.senseiver_full_field_fraction,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            try:
                run_sparse_eval(cfg)
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
