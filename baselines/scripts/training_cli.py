from __future__ import annotations

import argparse
import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _str_to_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def apply_cli_overrides(cfg: Any) -> Any:
    """Apply simple --field value overrides for dataclass training configs."""

    parser = argparse.ArgumentParser()
    known_names = set()
    if is_dataclass(cfg):
        for field in fields(cfg):
            name = field.name
            known_names.add(name)
            default = getattr(cfg, name)
            arg = f"--{name.replace('_', '-')}"
            if isinstance(default, bool):
                parser.add_argument(arg, dest=name, nargs="?", const=True, type=_str_to_bool)
                parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
            elif isinstance(default, tuple):
                parser.add_argument(arg, dest=name, type=lambda value: tuple(ast.literal_eval(value)))
            else:
                parser.add_argument(arg, dest=name, type=type(default))

    if "load_checkpoint" not in known_names:
        parser.add_argument("--load-checkpoint", dest="load_checkpoint", action="store_true")
    if "checkpoint" not in known_names:
        parser.add_argument("--checkpoint", dest="checkpoint", default="")

    args, unknown = parser.parse_known_args()
    for name, value in vars(args).items():
        if value is not None:
            setattr(cfg, name, value)
    if unknown:
        print(f"[warn] ignoring unknown args: {' '.join(unknown)}")
    return cfg


def _load_torch_checkpoint(path: Path, device: torch.device | str | None) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def maybe_load_checkpoint(
    cfg: Any,
    default_path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    likelihood: torch.nn.Module | None = None,
    ema_model: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
    best_value: float = float("inf"),
) -> tuple[int, float]:
    """Load a training checkpoint when cfg.load_checkpoint is set and the file exists."""

    if not bool(getattr(cfg, "load_checkpoint", False)):
        return 1, best_value

    path = Path(str(getattr(cfg, "checkpoint", "") or default_path))
    if not path.exists():
        print(f"[checkpoint] --load-checkpoint set but checkpoint not found: {path}")
        return 1, best_value

    ckpt = _load_torch_checkpoint(path, device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    msg = model.load_state_dict(state, strict=strict)
    missing = getattr(msg, "missing_keys", [])
    unexpected = getattr(msg, "unexpected_keys", [])
    if missing or unexpected:
        print(f"[checkpoint] model load: missing={missing} unexpected={unexpected}")

    if ema_model is not None:
        ema_state = ckpt.get("ema_model_state_dict") if isinstance(ckpt, dict) else None
        ema_model.load_state_dict(ema_state or model.state_dict(), strict=strict)
    if likelihood is not None and isinstance(ckpt, dict) and "likelihood_state_dict" in ckpt:
        likelihood.load_state_dict(ckpt["likelihood_state_dict"], strict=strict)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if device is not None:
            _move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and isinstance(ckpt, dict) and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else 1
    best = float(ckpt.get("best_val_rmse", best_value)) if isinstance(ckpt, dict) else best_value
    print(f"[checkpoint] loaded {path} (start_epoch={start_epoch}, best_val_rmse={best:.6f})")
    return start_epoch, best
