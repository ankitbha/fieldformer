from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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
        if self.model_key == "svgp":
            return self._denorm(self.model(xyt).mean)
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
        if self.dataset_key == "pol":
            return self.model.forward_observed(q_lin, obs_coords, obs_vals_m, nb_idx)
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
        if self.dataset_key == "pol":
            return self.model.forward_continuous(xyt_q, obs_coords, obs_vals_m, nb_idx)
        return self.model.forward_continuous(xyt_q, obs_coords, obs_vals_m, nb_idx, Lx=self.Lx, Ly=self.Ly)

    def eval(self) -> "EvalAdapter":
        super().eval()
        self.model.eval()
        if self.likelihood is not None:
            self.likelihood.eval()
        return self


def split_train_mask(train_idx: np.ndarray, n_sensors: int, n_times: int) -> np.ndarray:
    mask = np.zeros((n_sensors, n_times), dtype=bool)
    mask[train_idx // n_times, train_idx % n_times] = True
    return mask


def sensor_grid_indices(x_grid: np.ndarray, y_grid: np.ndarray, sensors_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.abs(x_grid[None, :] - sensors_xy[:, 0:1]).argmin(axis=1).astype(np.int64)
    iy = np.abs(y_grid[None, :] - sensors_xy[:, 1:2]).argmin(axis=1).astype(np.int64)
    return ix, iy


class RecFNOEvalAdapter(EvalAdapter):
    def __init__(
        self,
        model: nn.Module,
        *,
        dataset_key: str,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        t_grid: np.ndarray,
        sensors_xy: np.ndarray,
        obs_vals_np: np.ndarray,
        train_idx: np.ndarray,
        device: torch.device,
    ):
        super().__init__(model, model_key="recfno", dataset_key=dataset_key)
        self.device_ref = device
        self.n_sensors, self.n_times = int(sensors_xy.shape[0]), int(t_grid.shape[0])
        self.values = obs_vals_np.reshape(self.n_sensors, self.n_times).astype(np.float32)
        self.train_mask = split_train_mask(train_idx, self.n_sensors, self.n_times)
        self.ix_np, self.iy_np = sensor_grid_indices(x_grid, y_grid, sensors_xy)
        self.ix = torch.from_numpy(self.ix_np).long().to(device)
        self.iy = torch.from_numpy(self.iy_np).long().to(device)
        self.x_grid = torch.from_numpy(x_grid.astype(np.float32)).to(device)
        self.y_grid = torch.from_numpy(y_grid.astype(np.float32)).to(device)
        self.t_grid = torch.from_numpy(t_grid.astype(np.float32)).to(device)
        xx, yy = np.meshgrid(
            (x_grid - x_grid.min()) / max(np.ptp(x_grid), 1e-6),
            (y_grid - y_grid.min()) / max(np.ptp(y_grid), 1e-6),
            indexing="ij",
        )
        self.coord_grid = torch.from_numpy(np.stack([xx, yy], axis=0).astype(np.float32)).to(device)

    def _make_input(self, times: torch.Tensor) -> torch.Tensor:
        bsz = int(times.numel())
        nx, ny = int(self.x_grid.numel()), int(self.y_grid.numel())
        val_grid = torch.zeros(bsz, nx, ny, device=self.device_ref)
        mask_grid = torch.zeros(bsz, nx, ny, device=self.device_ref)
        for b, k_t in enumerate(times.detach().cpu().numpy()):
            m = self.train_mask[:, int(k_t)]
            ix = torch.from_numpy(self.ix_np[m]).long().to(self.device_ref)
            iy = torch.from_numpy(self.iy_np[m]).long().to(self.device_ref)
            val_grid[b, ix, iy] = torch.from_numpy(self.values[m, int(k_t)]).float().to(self.device_ref)
            mask_grid[b, ix, iy] = 1.0
        coords = self.coord_grid.unsqueeze(0).expand(bsz, -1, -1, -1)
        return torch.cat([val_grid[:, None], mask_grid[:, None], coords], dim=1)

    def _grid_for_times(self, times: torch.Tensor) -> tuple[torch.Tensor, dict[int, int]]:
        times = torch.unique(times.long())
        pred_grid = self.model(self._make_input(times))
        return pred_grid, {int(k.item()): b for b, k in enumerate(times)}

    def predict_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor | None = None) -> torch.Tensor:
        del obs_coords, obs_vals, nb_idx
        s_q = q_lin // self.n_times
        k_q = q_lin % self.n_times
        pred_grid, time_to_b = self._grid_for_times(k_q)
        return torch.stack([pred_grid[time_to_b[int(k.item())], :, self.ix[int(s.item())], self.iy[int(s.item())]] for s, k in zip(s_q, k_q)])

    def predict_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor | None = None) -> torch.Tensor:
        del obs_coords, obs_vals, nb_idx
        ix = torch.argmin(torch.abs(xyt_q[:, 0:1] - self.x_grid[None, :]), dim=1)
        iy = torch.argmin(torch.abs(xyt_q[:, 1:2] - self.y_grid[None, :]), dim=1)
        kt = torch.argmin(torch.abs(xyt_q[:, 2:3] - self.t_grid[None, :]), dim=1)
        pred_grid, time_to_b = self._grid_for_times(kt)
        return torch.stack([pred_grid[time_to_b[int(k.item())], :, i, j] for i, j, k in zip(ix, iy, kt)])


class SenseiverEvalAdapter(EvalAdapter):
    def __init__(
        self,
        model: nn.Module,
        *,
        dataset_key: str,
        sensors_xy: np.ndarray,
        t_grid: np.ndarray,
        obs_coords_np: np.ndarray,
        obs_vals_np: np.ndarray,
        train_idx: np.ndarray,
        device: torch.device,
    ):
        vals_mean = float(obs_vals_np.mean())
        vals_std = float(obs_vals_np.std() + 1e-6)
        super().__init__(model, model_key="senseiver", dataset_key=dataset_key, normalizes_values=True, obs_mean=vals_mean, obs_std=vals_std)
        self.device_ref = device
        self.n_sensors, self.n_times = int(sensors_xy.shape[0]), int(t_grid.shape[0])
        self.values = obs_vals_np.reshape(self.n_sensors, self.n_times).astype(np.float32)
        self.train_mask = split_train_mask(train_idx, self.n_sensors, self.n_times)
        mins = obs_coords_np.min(axis=0).astype(np.float32)
        spans = np.maximum(obs_coords_np.max(axis=0).astype(np.float32) - mins, 1e-6)
        self.coord_min = torch.from_numpy(mins).to(device)
        self.coord_span = torch.from_numpy(spans).to(device)
        self.t_grid = torch.from_numpy(t_grid.astype(np.float32)).to(device)
        sxy_norm = ((sensors_xy.astype(np.float32) - mins[:2]) / spans[:2]).astype(np.float32)
        self.sxy = torch.from_numpy(sxy_norm).to(device)
        self.t_norm = torch.from_numpy(((t_grid.astype(np.float32) - mins[2]) / spans[2]).astype(np.float32)).to(device)
        self.vals_norm = torch.from_numpy(((self.values - vals_mean) / vals_std).astype(np.float32)).to(device)

    def _sensor_tokens(self, times: torch.Tensor) -> torch.Tensor:
        toks = []
        for k in times.detach().cpu().numpy():
            k = int(k)
            mask = torch.from_numpy(self.train_mask[:, k].astype(np.float32))[:, None].to(self.device_ref)
            val = self.vals_norm[:, k:k + 1] * mask
            tt = torch.full((self.n_sensors, 1), float(self.t_norm[k].item()), device=self.device_ref)
            toks.append(torch.cat([self.sxy, tt, val, mask], dim=-1))
        return torch.stack(toks, dim=0)

    def _query_norm(self, xyt_q: torch.Tensor) -> torch.Tensor:
        return (xyt_q - self.coord_min) / self.coord_span

    def _predict_queries(self, xyt_q: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
        out_shape = 3 if self.dataset_key == "swe" else 1
        pred = torch.empty(xyt_q.shape[0], out_shape, device=self.device_ref)
        for k in torch.unique(kt):
            mask = kt == k
            q = self._query_norm(xyt_q[mask]).unsqueeze(0)
            toks = self._sensor_tokens(k.reshape(1))
            out = self.model(toks, q)
            if out.ndim == 2:
                out = out.unsqueeze(-1)
            out = out[0] * float(self.obs_std) + float(self.obs_mean)
            pred[mask] = out if out_shape > 1 else out[:, :1]
        return pred[:, 0] if out_shape == 1 else pred

    def predict_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor | None = None) -> torch.Tensor:
        del obs_vals, nb_idx
        kt = q_lin % self.n_times
        return self._predict_queries(obs_coords[q_lin], kt)

    def predict_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor | None = None) -> torch.Tensor:
        del obs_coords, obs_vals, nb_idx
        kt = torch.argmin(torch.abs(xyt_q[:, 2:3] - self.t_grid[None, :]), dim=1)
        return self._predict_queries(xyt_q, kt)


class ImputeFormerEvalAdapter(EvalAdapter):
    def __init__(
        self,
        model: nn.Module,
        *,
        dataset_key: str,
        t_grid: np.ndarray,
        sensors_xy: np.ndarray,
        obs_vals_np: np.ndarray,
        train_idx: np.ndarray,
        cfg: SimpleNamespace,
        device: torch.device,
    ):
        mean = float(obs_vals_np.reshape(sensors_xy.shape[0], t_grid.shape[0])[split_train_mask(train_idx, sensors_xy.shape[0], t_grid.shape[0])].mean())
        std = float(obs_vals_np.reshape(sensors_xy.shape[0], t_grid.shape[0])[split_train_mask(train_idx, sensors_xy.shape[0], t_grid.shape[0])].std() + 1e-6)
        super().__init__(model, model_key="imputeformer", dataset_key=dataset_key, normalizes_values=True, obs_mean=mean, obs_std=std)
        self.device_ref = device
        self.n_sensors, self.n_times = int(sensors_xy.shape[0]), int(t_grid.shape[0])
        self.sensors_xy = torch.from_numpy(sensors_xy.astype(np.float32)).to(device)
        self.t_grid = torch.from_numpy(t_grid.astype(np.float32)).to(device)
        values = obs_vals_np.reshape(self.n_sensors, self.n_times).astype(np.float32)
        train_mask = split_train_mask(train_idx, self.n_sensors, self.n_times)
        self.pred_series = self._predict_series(values, train_mask, int(_get(cfg, "windows", min(128, self.n_times))), int(_get(cfg, "window_stride", 64)), mean, std)

    def _predict_series(self, values: np.ndarray, train_mask: np.ndarray, windows: int, stride: int, mean: float, std: float) -> torch.Tensor:
        windows = min(windows, self.n_times)
        starts = np.arange(0, max(1, self.n_times - windows + 1), max(1, stride), dtype=np.int64)
        if starts[-1] != self.n_times - windows:
            starts = np.append(starts, self.n_times - windows)
        out_dim = 3 if self.dataset_key == "swe" else 1
        acc = torch.zeros(self.n_sensors, self.n_times, out_dim, device=self.device_ref)
        cnt = torch.zeros(self.n_sensors, self.n_times, 1, device=self.device_ref)
        norm_values = ((values - mean) / std).astype(np.float32)
        with torch.no_grad():
            for st in starts:
                sl = slice(int(st), int(st) + windows)
                vals = torch.from_numpy(norm_values[:, sl][None, ..., None]).float().to(self.device_ref)
                ctx = torch.from_numpy(train_mask[:, sl][None, ..., None].astype(np.float32)).float().to(self.device_ref)
                pred = self.model(vals, ctx)[0] * std + mean
                acc[:, sl, :] += pred
                cnt[:, sl, :] += 1.0
        return acc / cnt.clamp_min(1.0)

    def _nearest_sensor_time(self, xyt_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d2 = ((xyt_q[:, None, :2] - self.sensors_xy[None, :, :]) ** 2).sum(dim=-1)
        s = torch.argmin(d2, dim=1)
        k = torch.argmin(torch.abs(xyt_q[:, 2:3] - self.t_grid[None, :]), dim=1)
        return s, k

    def predict_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor | None = None) -> torch.Tensor:
        del obs_coords, obs_vals, nb_idx
        s = q_lin // self.n_times
        k = q_lin % self.n_times
        pred = self.pred_series[s, k]
        return pred[:, 0] if pred.shape[-1] == 1 else pred

    def predict_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor | None = None) -> torch.Tensor:
        del obs_coords, obs_vals, nb_idx
        s, k = self._nearest_sensor_time(xyt_q)
        pred = self.pred_series[s, k]
        return pred[:, 0] if pred.shape[-1] == 1 else pred


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
    sensors_xy: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
    obs_coords_np: np.ndarray | None = None,
    obs_vals_np: np.ndarray | None = None,
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
    elif model_key == "recfno":
        from baselines.models.recfno import VoronoiFNO2d

        model = VoronoiFNO2d(
            _get(cfg, "modes1", 12),
            _get(cfg, "modes2", 12),
            _get(cfg, "width", 32),
            in_channels=4,
            out_channels=3 if dataset_key == "swe" else 1,
        )
    elif model_key == "senseiver":
        from baselines.models.senseiver import Senseiver

        model = Senseiver(
            sensor_feature_dim=5,
            query_feature_dim=3,
            num_latents=_get(cfg, "num_latents", 16),
            latent_channels=_get(cfg, "latent_channels", 64),
            num_layers=_get(cfg, "num_layers", 3),
            cross_heads=_get(cfg, "cross_heads", 4),
            decoder_heads=_get(cfg, "decoder_heads", 1),
            self_heads=_get(cfg, "self_heads", 4),
            self_layers=_get(cfg, "self_layers", 2),
            out_dim=3 if dataset_key == "swe" else 1,
            dropout=_get(cfg, "dropout", 0.0),
        )
    elif model_key == "imputeformer":
        from baselines.models.imputeformer import FixedNodeImputeFormer

        if sensors_xy is None:
            raise ValueError("ImputeFormer eval requires sensors_xy")
        model = FixedNodeImputeFormer(
            int(sensors_xy.shape[0]),
            min(_get(cfg, "windows", 128), int(t_grid.shape[0]) if t_grid is not None else 128),
            output_dim=3 if dataset_key == "swe" else 1,
            input_embedding_dim=_get(cfg, "input_embedding_dim", 32),
            learnable_embedding_dim=_get(cfg, "learnable_embedding_dim", 96),
            num_layers=_get(cfg, "num_layers", 3),
            num_temporal_heads=_get(cfg, "num_temporal_heads", 4),
            dim_proj=_get(cfg, "dim_proj", 8),
            dropout=_get(cfg, "dropout", 0.1),
        )
    else:
        raise KeyError(model_key)

    model.to(device)
    model.load_state_dict(state)
    if model_key == "recfno":
        if any(v is None for v in (sensors_xy, x_grid, y_grid, t_grid, train_idx, obs_vals_np)):
            raise ValueError("RecFNO eval requires sensor/grid/train context")
        return RecFNOEvalAdapter(
            model,
            dataset_key=dataset_key,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
            sensors_xy=sensors_xy,
            obs_vals_np=obs_vals_np,
            train_idx=train_idx,
            device=device,
        ).to(device)
    if model_key == "senseiver":
        if any(v is None for v in (sensors_xy, t_grid, train_idx, obs_coords_np, obs_vals_np)):
            raise ValueError("Senseiver eval requires sensor/train context")
        return SenseiverEvalAdapter(
            model,
            dataset_key=dataset_key,
            sensors_xy=sensors_xy,
            t_grid=t_grid,
            obs_coords_np=obs_coords_np,
            obs_vals_np=obs_vals_np,
            train_idx=train_idx,
            device=device,
        ).to(device)
    if model_key == "imputeformer":
        if any(v is None for v in (sensors_xy, t_grid, train_idx, obs_vals_np)):
            raise ValueError("ImputeFormer eval requires sensor/train context")
        return ImputeFormerEvalAdapter(
            model,
            dataset_key=dataset_key,
            t_grid=t_grid,
            sensors_xy=sensors_xy,
            obs_vals_np=obs_vals_np,
            train_idx=train_idx,
            cfg=cfg,
            device=device,
        ).to(device)
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
