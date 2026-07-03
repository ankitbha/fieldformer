#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DATASET_ORDER = ["heat", "pol", "swe", "atm", "govpol", "atmsplit", "govpolsplit"]
MODEL_ORDER = [
    "ffag",
    "fmlp",
    "fmlp_ensemble",
    "fmlp_pinn",
    "siren",
    "siren_pinn",
    "svgp",
    "recfno",
    "imputeformer",
    "senseiver",
]


def order_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    dataset = str(row.get("dataset", ""))
    model = str(row.get("model", ""))
    dataset_rank = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
    model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    return dataset_rank, model_rank, dataset, model


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text())
        rows.append({
            "dataset": data.get("dataset", ""),
            "model": data.get("model", ""),
            "ms_per_query": data.get("ms_per_query", ""),
            "queries_per_second": data.get("queries_per_second", ""),
            "seconds_total": data.get("seconds_total", ""),
            "num_queries_per_repeat": data.get("num_queries_per_repeat", ""),
            "timed_repeats": data.get("timed_repeats", ""),
            "batch_size": data.get("batch_size", ""),
            "setup_seconds": data.get("setup_seconds", ""),
            "load_seconds": data.get("load_seconds", ""),
            "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb", ""),
            "needs_sensor_context": data.get("needs_sensor_context", ""),
            "output_dim": data.get("output_dim", ""),
            "source": str(path),
        })
    rows.sort(key=order_key)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path | None) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0])
    out_lines = []
    writer_target = out_lines

    class ListWriter:
        def write(self, text: str) -> None:
            writer_target.append(text)

    writer = csv.DictWriter(ListWriter(), fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_text = "".join(out_lines)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv_text)
    return csv_text


def format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path | None) -> str:
    headers = ["Dataset", "Model", "ms/query", "queries/s", "Peak GPU MB", "Setup s", "N/query"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["model"]),
                    format_float(row["ms_per_query"]),
                    format_float(row["queries_per_second"]),
                    format_float(row["peak_gpu_memory_mb"]),
                    format_float(row["setup_seconds"]),
                    str(row["num_queries_per_repeat"]),
                ]
            )
            + " |"
        )
    text = "\n".join(lines) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sparse inference timing JSON outputs.")
    parser.add_argument("--input_dir", type=Path, default=Path("eval/main/timing_outputs"))
    parser.add_argument("--csv", type=Path, default=Path("eval/main/timing_outputs/timing_summary.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("eval/main/timing_outputs/timing_summary.md"))
    parser.add_argument("--stdout", choices=("csv", "markdown", "none"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_dir)
    if not rows:
        raise SystemExit(f"No timing JSON files found in {args.input_dir}")
    csv_text = write_csv(rows, args.csv)
    markdown_text = write_markdown(rows, args.markdown)
    if args.stdout == "csv":
        sys.stdout.write(csv_text)
    elif args.stdout == "markdown":
        sys.stdout.write(markdown_text)
    print(f"[summary] rows={len(rows)} csv={args.csv} markdown={args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
