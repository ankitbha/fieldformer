#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from sparse_eval import Config, main as run_sparse_eval


DEFAULT_DATASETS = ("heat", "pol", "swe")
DEFAULT_MODELS = ("ffag", "fmlp", "fmlp_pinn", "siren", "siren_pinn", "svgp", "recfno", "imputeformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse eval for all selected models/datasets.")
    parser.add_argument("--output_dir", default=str(ROOT / "eval" / "main" / "outputs"), help="Directory for per-run JSON and summary files.")
    parser.add_argument("--summary_name", default="sparse_eval_all", help="Basename for summary CSV/JSONL.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def flatten_result(result: dict) -> dict:
    return {
        "dataset": result["dataset"],
        "model": result["model"],
        "status": "ok",
        "checkpoint": result["checkpoint"],
        "obs_key": result.get("obs_key", ""),
        "num_sparse_test": result.get("num_sparse_test", ""),
        "num_full_field": result.get("num_full_field", ""),
        "sparse_test_rmse": result["sparse_test"]["rmse"],
        "sparse_test_mae": result["sparse_test"]["mae"],
        "full_field_rmse": result["full_field"]["rmse"],
        "full_field_mae": result["full_field"]["mae"],
        "error": "",
    }


def error_row(dataset: str, model: str, exc: BaseException) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "status": "error",
        "checkpoint": "",
        "obs_key": "",
        "num_sparse_test": "",
        "num_full_field": "",
        "sparse_test_rmse": "",
        "sparse_test_mae": "",
        "full_field_rmse": "",
        "full_field_mae": "",
        "error": f"{type(exc).__name__}: {exc}",
    }


def write_summary(rows: list[dict], out_dir: Path, summary_name: str) -> None:
    fields = [
        "dataset",
        "model",
        "status",
        "checkpoint",
        "obs_key",
        "num_sparse_test",
        "num_full_field",
        "sparse_test_rmse",
        "sparse_test_mae",
        "full_field_rmse",
        "full_field_mae",
        "error",
    ]
    csv_path = out_dir / f"{summary_name}.csv"
    jsonl_path = out_dir / f"{summary_name}.jsonl"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[summary] {csv_path}")
    print(f"[summary] {jsonl_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    total = len(args.datasets) * len(args.models)
    done = 0
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
            )
            try:
                run_sparse_eval(cfg)
                result = json.loads(result_path.read_text())
                rows.append(flatten_result(result))
            except BaseException as exc:
                print(f"[error] {exp}: {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc()
                rows.append(error_row(dataset, model, exc))
                if args.stop_on_error:
                    write_summary(rows, out_dir, args.summary_name)
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write_summary(rows, out_dir, args.summary_name)

    ok = sum(row["status"] == "ok" for row in rows)
    print(f"\n[done] {ok}/{len(rows)} evaluations succeeded.")


if __name__ == "__main__":
    main()
