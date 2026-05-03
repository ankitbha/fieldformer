from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "fieldformer_core" / "scripts"


def _load_legacy(stem: str) -> Any:
    path = SCRIPT_DIR / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing FieldFormer source: {path}")
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    name = f"_fieldformer_core_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def module_for_dataset(dataset_key: str) -> Any:
    stem = {"heat": "ffag_heatsparse_train", "swe": "ffag_swesparse_train", "pol": "ffag_polsparse_train"}[dataset_key]
    return _load_legacy(stem)


def class_for_dataset(dataset_key: str) -> Any:
    mod = module_for_dataset(dataset_key)
    name = {"heat": "FieldFormerSparse", "swe": "FieldFormerSparseSWE", "pol": "FieldFormerSparsePollution"}[dataset_key]
    return getattr(mod, name)
