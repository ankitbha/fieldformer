#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRS = (
    ROOT / "eval" / "main" / "outputs",
    ROOT / "eval" / "ablations" / "architecture" / "outputs",
)
DEFAULT_OUTPUT = ROOT / "eval" / "sparse_summary_tables.tex"

BULKY_KEYS = {"checkpoint", "context_sensor_ids"}
BULKY_SUFFIXES = (
    "sensor_ids",
    "sensor_indices",
    "eligible_sensor_ids",
    "eligible_sensor_indices",
)
CAPTION_KEYS = (
    "dataset",
    "obs_key",
    "mask_key",
    "num_sparse_test",
    "num_full_field",
    "num_full_field_total",
    "full_field_sample_fraction",
    "bootstrap_samples",
    "bootstrap_seed",
    "split.split_type",
    "split.sensor_split_seed",
    "split.val_sensors",
    "split.test_sensors",
)
METRIC_ORDER = {
    "rmse": 0,
    "mae": 1,
    "count": 2,
    "rmse_bootstrap_std": 3,
    "mae_bootstrap_std": 4,
}
MODEL_ORDER = {
    "ffag": 0,
    "ffag_nophys": 1,
    "ffag_npgf": 2,
    "ffag_mlp": 3,
    "fmlp": 10,
    "fmlp_pinn": 11,
    "siren": 20,
    "siren_pinn": 21,
    "svgp": 30,
    "recfno": 40,
    "imputeformer": 50,
    "senseiver": 60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LaTeX summary tables from sparse eval JSON outputs.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUT_DIRS],
        help="Directories containing per-run JSON outputs.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Combined LaTeX table output path.")
    parser.add_argument("--precision", type=int, default=4, help="Significant digits for numeric values.")
    return parser.parse_args()


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def is_bulky_key(path: str) -> bool:
    tail = path.split(".")[-1]
    return path in BULKY_KEYS or any(tail.endswith(suffix) for suffix in BULKY_SUFFIXES)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, child))
        return out
    return {prefix: value}


def table_values(result: dict[str, Any]) -> dict[str, Any]:
    flat = flatten(result)
    return {key: value for key, value in flat.items() if is_scalar(value) and not is_bulky_key(key)}


def common_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = set(rows[0])
    for row in rows[1:]:
        keys &= set(row)
    return {key: rows[0][key] for key in keys if all(row[key] == rows[0][key] for row in rows)}


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_value(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{precision}g}"
    return latex_escape(value)


def metric_suffix(key: str) -> str:
    return key.split(".")[-1]


def column_sort_key(key: str) -> tuple[int, str, int, str]:
    if key == "model":
        return (-1, "", -1, "")
    if key.startswith("sparse_test.channels."):
        parts = key.split(".")
        channel = parts[2] if len(parts) > 2 else ""
        metric = parts[-1]
        return (2, channel, METRIC_ORDER.get(metric, 99), metric)
    if key.startswith("sparse_test."):
        metric = metric_suffix(key)
        return (0, "", METRIC_ORDER.get(metric, 99), metric)
    if key.startswith("full_field."):
        metric = metric_suffix(key)
        return (1, "", METRIC_ORDER.get(metric, 99), metric)
    return (3, key, 99, key)


def human_metric(metric: str) -> str:
    labels = {
        "rmse": "RMSE",
        "mae": "MAE",
        "count": "Count",
        "rmse_bootstrap_std": "RMSE Std.",
        "mae_bootstrap_std": "MAE Std.",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def human_header(key: str) -> str:
    if key == "model":
        return "Model"
    if key.startswith("sparse_test.channels."):
        parts = key.split(".")
        channel = parts[2] if len(parts) > 2 else ""
        return f"{channel} {human_metric(parts[-1])}".strip()
    if key.startswith("sparse_test."):
        return f"Sparse {human_metric(metric_suffix(key))}"
    if key.startswith("full_field."):
        return f"Full {human_metric(metric_suffix(key))}"
    return key.replace(".", " ").replace("_", " ").title()


def caption_value(key: str, value: Any, precision: int) -> str:
    labels = {
        "dataset": "dataset",
        "obs_key": "obs",
        "mask_key": "mask",
        "num_sparse_test": "sparse n",
        "num_full_field": "full n",
        "num_full_field_total": "full total",
        "full_field_sample_fraction": "full fraction",
        "bootstrap_samples": "bootstrap samples",
        "bootstrap_seed": "bootstrap seed",
        "split.split_type": "split",
        "split.sensor_split_seed": "split seed",
        "split.val_sensors": "val sensors",
        "split.test_sensors": "test sensors",
    }
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    elif isinstance(value, float):
        rendered = f"{value:.{precision}g}" if math.isfinite(value) else str(value)
    else:
        rendered = "" if value is None else str(value)
    return f"{labels.get(key, key)}={rendered}"


def build_caption(dataset: str, common: dict[str, Any], precision: int) -> str:
    parts = [f"Sparse evaluation results for {dataset}."]
    metadata = [caption_value(key, common[key], precision) for key in CAPTION_KEYS if key in common]
    if metadata:
        parts.append("Common settings: " + "; ".join(metadata) + ".")
    return latex_escape(" ".join(parts))


def model_sort_key(result: dict[str, Any]) -> tuple[int, str]:
    model = str(result.get("model", ""))
    return (MODEL_ORDER.get(model, 1000), model)


def make_table(dataset: str, results: list[dict[str, Any]], precision: int) -> str:
    rows = [table_values(result) for result in sorted(results, key=model_sort_key)]
    common = common_values(rows)
    columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key == "model" or (key not in common and not is_bulky_key(key))
        },
        key=column_sort_key,
    )
    if "model" not in columns:
        columns.insert(0, "model")

    header = " & ".join(latex_escape(human_header(column)) for column in columns) + r" \\"
    body_lines = []
    for row in rows:
        cells = [format_value(row.get(column, ""), precision) for column in columns]
        body_lines.append(" & ".join(cells) + r" \\")

    alignment = "l" + "r" * (len(columns) - 1)
    label_slug = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{build_caption(dataset, common, precision)}}}",
        rf"\label{{tab:sparse-{label_slug}}}",
        rf"\resizebox{{\textwidth}}{{!}}{{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        header,
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def load_results(input_dirs: list[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for input_dir in input_dirs:
        for path in sorted(Path(input_dir).glob("*.json")):
            with path.open() as handle:
                result = json.load(handle)
            dataset = result.get("dataset")
            if not dataset:
                raise ValueError(f"{path} is missing a dataset field")
            grouped[str(dataset)].append(result)
    return grouped


def main() -> None:
    args = parse_args()
    grouped = load_results(args.input_dirs)
    if not grouped:
        raise SystemExit("No JSON outputs found.")

    chunks = [
        "% Generated by eval/make_sparse_summary_tables.py",
        "% Requires \\usepackage{graphicx} for \\resizebox.",
        "",
    ]
    for dataset in sorted(grouped):
        chunks.append(make_table(dataset, grouped[dataset], args.precision))
        chunks.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(chunks))
    print(f"Wrote {output} with {len(grouped)} dataset tables.")
    for dataset in sorted(grouped):
        print(f"  {dataset}: {len(grouped[dataset])} models")


if __name__ == "__main__":
    main()
