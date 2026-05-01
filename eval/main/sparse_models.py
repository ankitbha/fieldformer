from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]


def cfg_obj(config: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(**(config or {}))


def _get(cfg: SimpleNamespace, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def infer_inducing_points(state: dict[str, torch.Tensor]) -> torch.Tensor:
    for key, value in state.items():
        if key.endswith("inducing_points"):
            z = value.detach().clone()
            return z[0] if z.ndim == 3 else z
    raise KeyError("checkpoint does not contain variational inducing_points")


def is_multitask_svgp_state(state: dict[str, torch.Tensor], ckpt: dict[str, Any]) -> bool:
    meta = ckpt.get("meta", {})
    if int(meta.get("output_dim", 1)) > 1:
        return True
    for key, value in state.items():
        if "base_variational_strategy" in key:
            return True
        if "variational" in key and value.ndim >= 2 and value.shape[0] == 3:
            return True
    return False


class EvalAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        model_key: str,
        dataset_key: str,
        likelihood: Any | None = None,
        normalizes_values: bool = False,
        normalizes_coords: bool = False,
        obs_mean: Any = 0.0,
        obs_std: Any = 1.0,
        x_min: float = 0.0,
        y_min: float = 0.0,
        t_min: float = 0.0,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Tt: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.model_key = model_key
        self.dataset_key = dataset_key
        self.normalizes_values = bool(normalizes_values)
        self.normalizes_coords = bool(normalizes_coords)
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.t_min = float(t_min)
        self.Lx = float(max(Lx, 1e-6))
        self.Ly = float(max(Ly, 1e-6))
        self.Tt = float(max(Tt, 1e-6))

    @property
    def needs_sensor_context(self) -> bool:
        return self.model_key == "ffag"

    def _coords(self, xyt: torch.Tensor) -> torch.Tensor:
        if not self.normalizes_coords:
            return xyt
        out = xyt.clone()
        out[:, 0] = (out[:, 0] - self.x_min) / self.Lx
        out[:, 1] = (out[:, 1] - self.y_min) / self.Ly
        out[:, 2] = (out[:, 2] - self.t_min) / self.Tt
        return out

    def _obs_values(self, obs_vals: torch.Tensor) -> torch.Tensor:
        if not self.normalizes_values:
            return obs_vals
        mean = torch.as_tensor(self.obs_mean, dtype=obs_vals.dtype, device=obs_vals.device)
        std = torch.as_tensor(self.obs_std, dtype=obs_vals.dtype, device=obs_vals.device)
        return (obs_vals - mean) / std

    def _denorm(self, pred: torch.Tensor) -> torch.Tensor:
        if not self.normalizes_values:
            return pred
        mean = torch.as_tensor(self.obs_mean, dtype=pred.dtype, device=pred.device)
        std = torch.as_tensor(self.obs_std, dtype=pred.dtype, device=pred.device)
        return pred * std + mean

    def _predict_points(self, xyt: torch.Tensor) -> torch.Tensor:
        xyt = self._coords(xyt)
        if self.likelihood is not None:
            return self._denorm(self.likelihood(self.model(xyt)).mean)
        if self.model_key == "fmlp":
            return self.model(xyt, Lx=self.Lx, Ly=self.Ly, Tt=self.Tt)
        return self.model(xyt)

    def predict_observed(
        self,
        q_lin: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals: torch.Tensor,
        nb_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.needs_sensor_context:
            return self._predict_points(obs_coords[q_lin])
        assert nb_idx is not None
        obs_vals_m = self._obs_values(obs_vals)
        return self.model.forward_observed(q_lin, obs_coords, obs_vals_m, nb_idx, Lx=self.Lx, Ly=self.Ly)

    def predict_continuous(
        self,
        xyt_q: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_vals: torch.Tensor,
        nb_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.needs_sensor_context:
            return self._predict_points(xyt_q)
        assert nb_idx is not None
        obs_vals_m = self._obs_values(obs_vals)
        return self.model.forward_continuous(xyt_q, obs_coords, obs_vals_m, nb_idx, Lx=self.Lx, Ly=self.Ly)

    def eval(self) -> "EvalAdapter":
        super().eval()
        self.model.eval()
        if self.likelihood is not None:
            self.likelihood.eval()
        return self


def build_sparse_model(
    *,
    model_key: str,
    dataset_key: str,
    ckpt: dict[str, Any],
    data: Any,
    device: torch.device,
    obs_mean: float,
    obs_std: float,
    x_min: float,
    y_min: float,
    t_min: float,
    Lx: float,
    Ly: float,
    Tt: float,
    nt_count: int,
) -> EvalAdapter:
    del data, nt_count
    cfg = cfg_obj(ckpt.get("config"))
    state = ckpt.get("model_state_dict", ckpt)
    likelihood = None
    normalizes_values = False
    normalizes_coords = False

    if model_key == "siren":
        from baselines.models.siren import SIREN, SIRENSWE

        if dataset_key == "swe":
            model = SIRENSWE(_get(cfg, "width", 256), _get(cfg, "depth", 6), _get(cfg, "w0", 30.0), _get(cfg, "w0_hidden", 1.0))
        else:
            model = SIREN(3, _get(cfg, "width", 256), _get(cfg, "depth", 6), 1, _get(cfg, "w0", 30.0), _get(cfg, "w0_hidden", 1.0))
    elif model_key == "fmlp":
        from baselines.models.fmlp import FourierMLP, FourierMLPSWE

        if dataset_key == "swe":
            model = FourierMLPSWE(_get(cfg, "width", 256), _get(cfg, "depth", 6), _get(cfg, "kx", 16), _get(cfg, "ky", 16), _get(cfg, "kt", 8))
        else:
            model = FourierMLP(_get(cfg, "width", 256), _get(cfg, "depth", 6), _get(cfg, "kx", 16), _get(cfg, "ky", 16), _get(cfg, "kt", 8), 1)
    elif model_key == "svgp":
        from baselines.models.svgp import MultitaskPeriodicSVGP, PeriodicSVGP, PollutionSVGP, make_likelihood

        z = infer_inducing_points(state).to(device)
        if dataset_key == "pol":
            model = PollutionSVGP(z, tuple(_get(cfg, "ard_lengthscale_init", (0.2, 0.2, 0.1))), _get(cfg, "outputscale_init", 1.0))
        elif dataset_key == "swe" and is_multitask_svgp_state(state, ckpt):
            model = MultitaskPeriodicSVGP(z, num_tasks=3)
        else:
            model = PeriodicSVGP(z)
        if isinstance(model, MultitaskPeriodicSVGP):
            likelihood = None
        else:
            likelihood_key = "heat" if dataset_key == "swe" else dataset_key
            likelihood = make_likelihood(likelihood_key).to(device)
        if likelihood is not None and "likelihood_state_dict" in ckpt:
            likelihood.load_state_dict(ckpt["likelihood_state_dict"])
        normalizes_values = True
        normalizes_coords = True
    elif model_key == "ffag":
        from fieldformer_core.models.ffag import class_for_dataset

        cls = class_for_dataset(dataset_key)
        model = cls(_get(cfg, "d_model", 128), _get(cfg, "nhead", 4), _get(cfg, "layers", 3), _get(cfg, "d_ff", 256))
    elif model_key in {"recfno", "senseiver", "imputeformer"}:
        raise RuntimeError(
            f"{model_key} was refactored to remove FieldFormer local-neighbor conditioning. "
            "Retrain this baseline with the new fair model before evaluating it."
        )
    else:
        raise KeyError(model_key)

    model.to(device)
    model.load_state_dict(state)
    return EvalAdapter(
        model,
        model_key=model_key,
        dataset_key=dataset_key,
        likelihood=likelihood,
        normalizes_values=normalizes_values,
        normalizes_coords=normalizes_coords,
        obs_mean=obs_mean,
        obs_std=obs_std,
        x_min=x_min,
        y_min=y_min,
        t_min=t_min,
        Lx=Lx,
        Ly=Ly,
        Tt=Tt,
    ).to(device)
