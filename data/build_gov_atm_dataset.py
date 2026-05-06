#!/usr/bin/env python3
"""Build a sparse real atmospheric sensor dataset from Delhi government CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ATM_VARS = ("AT", "RH", "U_x", "V_y")
SOURCE_VARS = ("AT", "RH", "WD", "WS")
PUSA_SOURCES = ("Pusa_IMD", "Pusa_DPCC")
PUSA_OUTPUT = "Pusa_averaged"


def _match_monitor_ids(ids: pd.Index, wanted: tuple[str, ...]) -> list[str]:
    by_lower = {str(mid).lower(): str(mid) for mid in ids}
    return [by_lower[name.lower()] for name in wanted if name.lower() in by_lower]


def _prepare_atm_readings(readings: pd.DataFrame) -> pd.DataFrame:
    out = readings.copy()
    for col in SOURCE_VARS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    wd_rad = np.deg2rad(out["WD"].to_numpy(dtype=np.float64))
    ws = out["WS"].to_numpy(dtype=np.float64)
    out["U_x"] = (-ws * np.sin(wd_rad)).astype(np.float32)
    out["V_y"] = (-ws * np.cos(wd_rad)).astype(np.float32)
    return out


def _series_for_monitor(readings: pd.DataFrame, monitor_ids: list[str], timestamps: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    subset = readings.loc[readings["monitor_id"].isin(monitor_ids), ["timestamp_round", *ATM_VARS]]
    by_time = subset.groupby("timestamp_round", sort=True)[list(ATM_VARS)].mean()
    values = by_time.reindex(timestamps).to_numpy(dtype=np.float32)
    mask = np.isfinite(values)
    return np.nan_to_num(values, nan=0.0).astype(np.float32), mask


def build_dataset(data_csv: Path, locations_csv: Path, output_npz: Path) -> None:
    locs = pd.read_csv(locations_csv)
    readings = pd.read_csv(data_csv, parse_dates=["timestamp_round"])

    required_locs = {"Monitor ID", "Latitude", "Longitude", "Location"}
    missing_locs = required_locs.difference(locs.columns)
    if missing_locs:
        raise ValueError(f"{locations_csv} is missing columns: {sorted(missing_locs)}")
    required_readings = {"monitor_id", "timestamp_round", *SOURCE_VARS}
    missing_readings = required_readings.difference(readings.columns)
    if missing_readings:
        raise ValueError(f"{data_csv} is missing columns: {sorted(missing_readings)}")

    locs = locs.drop_duplicates("Monitor ID").set_index("Monitor ID", drop=False)
    readings = readings[readings["monitor_id"].isin(locs.index)].copy()
    readings = _prepare_atm_readings(readings)

    timestamps = pd.DatetimeIndex(sorted(readings["timestamp_round"].dropna().unique()))
    if timestamps.empty:
        raise ValueError("No timestamps found in readings CSV")
    t0 = timestamps[0]
    t_hours = ((timestamps - t0) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32)

    pusa_ids = _match_monitor_ids(locs.index, PUSA_SOURCES)
    output_ids = [str(mid) for mid in locs.index if str(mid) not in set(pusa_ids)]
    if pusa_ids:
        output_ids.append(PUSA_OUTPUT)

    sensors_xy = np.empty((len(output_ids), 2), dtype=np.float32)
    values = np.empty((len(output_ids), len(timestamps), len(ATM_VARS)), dtype=np.float32)
    mask = np.empty_like(values, dtype=bool)
    location_names: list[str] = []
    merged_sensors: dict[str, list[str]] = {}

    for i, monitor_id in enumerate(output_ids):
        if monitor_id == PUSA_OUTPUT:
            src_locs = locs.loc[pusa_ids]
            sensors_xy[i, 0] = float(src_locs["Longitude"].mean())
            sensors_xy[i, 1] = float(src_locs["Latitude"].mean())
            location_names.append("Averaged Pusa sensor from IMD and DPCC monitors")
            vals_i, mask_i = _series_for_monitor(readings, pusa_ids, timestamps)
            merged_sensors[PUSA_OUTPUT] = pusa_ids
        else:
            row = locs.loc[monitor_id]
            sensors_xy[i, 0] = float(row["Longitude"])
            sensors_xy[i, 1] = float(row["Latitude"])
            location_names.append(str(row["Location"]))
            vals_i, mask_i = _series_for_monitor(readings, [monitor_id], timestamps)
        values[i] = vals_i
        mask[i] = mask_i

    valid_counts = mask.sum(axis=(0, 1)).astype(np.int64)
    per_sensor_valid_counts = mask.sum(axis=1).astype(np.int64)
    x_grid = np.unique(sensors_xy[:, 0]).astype(np.float32)
    y_grid = np.unique(sensors_xy[:, 1]).astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        sensors_xy=sensors_xy,
        x=x_grid,
        y=y_grid,
        t=t_hours,
        U_sensor=values,
        U_sensor_mask=mask,
        pollutant_names=np.asarray(ATM_VARS, dtype="<U16"),
        monitor_ids=np.asarray(output_ids, dtype="<U64"),
        location_names=np.asarray(location_names, dtype="<U256"),
        timestamps=np.asarray([ts.isoformat() for ts in timestamps], dtype="<U32"),
        coordinate_units=np.asarray(["longitude", "latitude"]),
        time_units=np.asarray(["elapsed_hours"]),
        source_data_csv=np.asarray([str(data_csv)]),
        source_locations_csv=np.asarray([str(locations_csv)]),
        wind_direction_convention=np.asarray(["meteorological_from_degrees"]),
        wind_vector_formula=np.asarray(["U_x=-WS*sin(WD*pi/180); V_y=-WS*cos(WD*pi/180)"]),
        merged_sensor_names=np.asarray(list(merged_sensors), dtype="<U64"),
        merged_sensor_sources=np.asarray([json.dumps(v) for v in merged_sensors.values()], dtype="<U256"),
        valid_counts=valid_counts,
        per_sensor_valid_counts=per_sensor_valid_counts,
    )
    print(f"[save] {output_npz}")
    print(f"       U_sensor shape: {values.shape}")
    print(f"       valid counts: {dict(zip(ATM_VARS, valid_counts.tolist()))}")
    if merged_sensors:
        print(f"       merged sensors: {merged_sensors}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", type=Path, default=root / "govdata_1H_current.csv")
    parser.add_argument("--locations_csv", type=Path, default=root / "govdata_locations.csv")
    parser.add_argument("--output", type=Path, default=root / "gov_atm_dataset.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.data_csv, args.locations_csv, args.output)


if __name__ == "__main__":
    main()
