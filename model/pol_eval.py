#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# pol_eval.py
"""
Evaluation for the pollution (advection–diffusion, open BC) dataset.
Models: FieldFormer-Autograd (FFAG), SIREN, Fourier-MLP, SVGP (single-output).
- Loads pollution_dataset.npz produced by pollution.py
- Computes RMSE/MAE on:
   (A) sensor coordinates (held-out test split indices)
   (B) entire field (all grid points)
- Bootstrap (N=5 by default) on the test split: mean ± std
- Physics residuals intentionally skipped.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from torch.backends.cuda import sdp_kernel

import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ProductKernel  # generic; state_dict will refine
# If your SVGP used a different kernel topology, loading will still override hyperparams via state_dict.


# In[2]:


# ----------------------
# Config
# ----------------------
class Config:
    # Data
    data_path = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
    seed = 123
    train_frac = 0.80
    val_frac = 0.10

    # Checkpoints
    ckpt_fieldformer = "/scratch/ab9738/fieldformer/model/ffag_pol_bestv2.pt"
    ckpt_siren       = "/scratch/ab9738/fieldformer/model/siren_pol_best.pt"
    ckpt_fmlp        = "/scratch/ab9738/fieldformer/model/fmlp_pol_best.pt"
    ckpt_svgp        = "/scratch/ab9738/fieldformer/model/svgp_pol_best.pt"

    # Eval batch sizes
    batch_eval_idx  = 2048    # FieldFormer index-batch
    batch_eval_nf   = 16384   # SIREN/FMLP direct eval
    batch_eval_svgp = 4096    # SVGP chunks

    # SVGP numerical safety
    svgp_autocast    = False
    svgp_max_root    = 16
    svgp_max_precond = 8
    svgp_jitter      = 1e-4
    svgp_device      = "cuda" if torch.cuda.is_available() else "cpu"

    # Bootstrap
    bootstrap_n = 5
    bootstrap_frac = 0.8         # use 1.0 for textbook bootstrap
    bootstrap_with_replacement = True
    bootstrap_seed = 123

    device = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


# ----------------------
# Data loading (pollution)
# ----------------------
def load_pollution_data(cfg: Config):
    pack = np.load(cfg.data_path)

    # Stored as U (Nx,Ny,Nt) + x,y,t; mirrors SWE/heat style
    U_np = pack["U"]
    x_np, y_np, t_np = pack["x"], pack["y"], pack["t"]

    # Optional params (k,T,Nx,Ny,Nt,Lx,Ly,dx,dy,dt)
    params = pack.get("params", None)
    names  = list(pack.get("param_names", [])) if pack.get("param_names", None) is not None else []
    def _get(name, default=None):
        if params is None or name not in names: return default
        return float(params[names.index(name)])

    dx = _get("dx", float(x_np[1]-x_np[0]) if len(x_np)>1 else 1.0)
    dy = _get("dy", float(y_np[1]-y_np[0]) if len(y_np)>1 else 1.0)
    dt = _get("dt", float(t_np[1]-t_np[0]) if len(t_np)>1 else 1.0)

    Nx, Ny, Nt = U_np.shape
    XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
    coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1).astype(np.float32)  # (N,3)
    vals_np    = U_np.reshape(-1).astype(np.float32)                                        # (N,)

    device = torch.device(cfg.device)
    coords = torch.from_numpy(coords_np).to(device)
    vals   = torch.from_numpy(vals_np).to(device)
    x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
    t_t = torch.from_numpy(t_np.astype(np.float32)).to(device)

    extents = {
        "Lx": float(x_np.max() - x_np.min()) if len(x_np)>1 else 1.0,
        "Ly": float(y_np.max() - y_np.min()) if len(y_np)>1 else 1.0,
        "Tt": float(t_np.max() - t_np.min()) if len(t_np)>1 else 1.0,
    }
    steps = {"dx": dx, "dy": dy, "dt": dt}

    return (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps


class OpenDataset(Dataset):
    """Matches your SplitDataset logic: random split over linear indices."""
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123):
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        rng = np.random.default_rng(seed)
        all_lin = np.arange(Nx * Ny * Nt)
        rng.shuffle(all_lin)
        n_total = len(all_lin)
        n_train = int(train_frac * n_total)
        n_val   = int(val_frac * n_total)
        self.train_idx = torch.from_numpy(all_lin[:n_train]).long()
        self.val_idx   = torch.from_numpy(all_lin[n_train:n_train + n_val]).long()
        self.test_idx  = torch.from_numpy(all_lin[n_train + n_val:]).long()
        self.split = "train"
    def set_split(self, s):
        assert s in ["train","val","test"]; self.split = s
    def __len__(self):
        return len(getattr(self, f"{self.split}_idx"))
    def __getitem__(self, i):
        return getattr(self, f"{self.split}_idx")[i]


# In[4]:


# ----------------------
# Metrics & Bootstrap
# ----------------------
def rmse_mae(pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[float,float]:
    se = F.mse_loss(pred, tgt, reduction="mean").item()
    ae = F.l1_loss(pred, tgt, reduction="mean").item()
    return math.sqrt(se), ae

def _make_bootstrap_indices(base_idx: torch.Tensor, n_boot: int, frac: float,
                            with_replacement: bool = True, seed: int = 123):
    N = base_idx.numel()
    m = max(1, int(round(frac * N)))
    g = torch.Generator(device=base_idx.device); g.manual_seed(seed)
    boots = []
    for _ in range(n_boot):
        if with_replacement:
            sel = torch.randint(low=0, high=N, size=(m,), generator=g, device=base_idx.device)
        else:
            sel = torch.randperm(N, generator=g, device=base_idx.device)[:m]
        boots.append(sel)
    return boots

def _agg_mean_std(values: torch.Tensor):
    if values.numel() == 1: return float(values.item()), 0.0
    return float(values.mean().item()), float(values.std(unbiased=True).item())

def _bootstrap_metrics(pred_all: torch.Tensor, tgt_all: torch.Tensor, boots: list):
    rmses, maes = [], []
    for sel in boots:
        p = pred_all.index_select(0, sel)
        t = tgt_all.index_select(0, sel)
        se = torch.mean((p - t) ** 2)
        ae = torch.mean((p - t).abs())
        rmses.append(torch.sqrt(se)); maes.append(ae)
    rmse_mean, rmse_std = _agg_mean_std(torch.stack(rmses))
    mae_mean,  mae_std  = _agg_mean_std(torch.stack(maes))
    return {"rmse_mean": rmse_mean, "rmse_std": rmse_std,
            "mae_mean": mae_mean,   "mae_std":  mae_std}


# In[5]:


# ----------------------
# Models
# ----------------------
class FFAG_Pollution(nn.Module):
    """
    Matches ffag_pol_trainv2.py (open BCs; neighbor value standardization).
    d_in=4 tokens = [dx,dy,dt,u_nb_norm]; wrap_x/y/t = False in pollution.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128,
                 Nx=None, Ny=None, Nt=None, Lx=None, Ly=None, Tt=None,
                 dx=None, dy=None, dt=None,
                 coords=None, vals=None):
        super().__init__()
        self.k = int(k_neighbors)
        self.wrap_x = False; self.wrap_y = False; self.wrap_t = False

        self.Nx, self.Ny, self.Nt = int(Nx), int(Ny), int(Nt)
        self.register_buffer("coords", coords)   # (N,3)
        self.register_buffer("vals",   vals)     # (N,)
        self.register_buffer("Lx_buf", torch.tensor(float(Lx)))
        self.register_buffer("Ly_buf", torch.tensor(float(Ly)))
        self.register_buffer("Tt_buf", torch.tensor(float(Tt)))
        self.register_buffer("dx_buf", torch.tensor(float(dx)))
        self.register_buffer("dy_buf", torch.tensor(float(dy)))
        self.register_buffer("dt_buf", torch.tensor(float(dt)))

        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.input_proj = nn.Linear(4, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    @property
    def Lx(self): return float(self.Lx_buf.item())
    @property
    def Ly(self): return float(self.Ly_buf.item())
    @property
    def dt(self): return float(self.dt_buf.item())

    def lin_to_ijk(self, lin):
        i = lin // (self.Ny * self.Nt)
        r = lin %  (self.Ny * self.Nt)
        j = r // self.Nt
        k = r %  self.Nt
        return i, j, k

    @staticmethod
    def build_offset_table(k, gammas, dx, dy, dt, frac_time=0.5, max_dt_radius=None):
        gx, gy, gt = gammas
        if max_dt_radius is None:
            max_dt_radius = int(3 * (1.0 / max(1e-6, gt)))
        rad = int(max(2, math.ceil((k**(1/3)) * 4)))

        cand = []
        for di in range(-rad, rad+1):
            for dj in range(-rad, rad+1):
                for dk in range(-max_dt_radius, max_dt_radius+1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    if dk > 0:                         # <<< enforce no future neighbors (match training)
                        continue
                    dxp = di * dx
                    dyp = dj * dy
                    dtp = dk * dt
                    dtp_eff = dtp / (1.0 + abs(dk) / 2.0)
                    d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp_eff)**2
                    cand.append((d2, di, dj, dk))

        cand.sort(key=lambda t: t[0])
        n_time = max(1, int(k * frac_time))

        # First fill with smallest |dk| (time-prioritized), then by distance
        sel, seen = [], set()
        for _, di, dj, dk in sorted(cand, key=lambda t: abs(t[3])):  # prioritize small |dk|
            key = (di, dj, dk)
            if key not in seen:
                sel.append(key); seen.add(key)
            if len(sel) >= n_time:
                break
        for _, di, dj, dk in cand:                                   # fill remainder by distance
            key = (di, dj, dk)
            if key not in seen:
                sel.append(key); seen.add(key)
            if len(sel) >= k:
                break
        return sel

    def _gather_neighbors(self, q_lin_idx, offsets_ijk):
        i, j, k0 = self.lin_to_ijk(q_lin_idx)
        sel = offsets_ijk[:self.k]
        dev, dtype = q_lin_idx.device, i.dtype
        di = torch.tensor([o[0] for o in sel], device=dev, dtype=dtype)
        dj = torch.tensor([o[1] for o in sel], device=dev, dtype=dtype)
        dk = torch.tensor([o[2] for o in sel], device=dev, dtype=dtype)
        I = i[:,None] + di[None,:]
        J = j[:,None] + dj[None,:]
        K = k0[:,None] + dk[None,:]
        I = I.clamp_(0, self.Nx-1)  # open boundaries: clamp
        J = J.clamp_(0, self.Ny-1)
        K = K.clamp_(0, self.Nt-1)
        return I*(self.Ny*self.Nt) + J*self.Nt + K

    def forward(self, q_lin_idx, offsets_ijk):
        nb_idx = self._gather_neighbors(q_lin_idx, offsets_ijk)
        q_xyz  = self.coords[q_lin_idx]
        nb_xyz = self.coords[nb_idx]

        dxv = nb_xyz[...,0] - q_xyz[:,None,0]
        dyv = nb_xyz[...,1] - q_xyz[:,None,1]
        dtv = nb_xyz[...,2] - q_xyz[:,None,2]
        rel = torch.stack([dxv, dyv, dtv], dim=-1) * torch.exp(self.log_gammas)[None,None,:]

        nb_vals = self.vals[nb_idx].to(torch.float32)
        mu = nb_vals.mean(dim=1, keepdim=True)
        sigma = nb_vals.std(dim=1, keepdim=True).clamp_min(1e-3)
        nb_vals_norm = ((nb_vals - mu) / sigma)[..., None]

        tokens = torch.cat([rel, nb_vals_norm], dim=-1)  # (B,k,4)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            h = self.input_proj(tokens)
            h = self.encoder(h)
            u_std_res = self.head(h.mean(dim=1)).squeeze(-1)  # standardized residual
        u_pred = u_std_res * sigma.squeeze(1) + mu.squeeze(1)
        return u_pred


# ----- SIREN (scalar) -----
class SineLayer(nn.Linear):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__(in_features, out_features)
        self.w0 = float(w0); self.is_first = bool(is_first)
        self._init()
    def _init(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
                self.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6 / self.in_features) / max(self.w0, 1e-6)
                self.weight.uniform_(-bound, bound)
            if self.bias is not None: self.bias.fill_(0.0)
    def forward(self, x):
        return torch.sin(self.w0 * F.linear(x, self.weight, self.bias))

class SIREN1(nn.Module):
    def __init__(self, in_dim=3, width=256, depth=6, out_dim=1, w0=30.0, w0_hidden=1.0):
        super().__init__()
        assert depth >= 2
        layers = [SineLayer(in_dim, width, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(width, width, w0=w0_hidden, is_first=False))
        self.hidden = nn.ModuleList(layers)
        self.final = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / max(w0_hidden, 1e-6)
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None: self.final.bias.fill_(0.0)
    def forward(self, x):
        y = x
        for layer in self.hidden: y = layer(y)
        return self.final(y).squeeze(-1)


# ----- Fourier-MLP (scalar) -----
def build_harmonics(K):
    if isinstance(K, int): return torch.arange(1, K+1, dtype=torch.float32)
    return torch.tensor(K, dtype=torch.float32)

def fourier_encode_1d(x, Ks, L):
    z = (2 * math.pi) * (x[..., None] / (L if L > 0 else 1.0)) * Ks[None, :].to(x)
    return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

def fourier_encode_3d(xyt, Kx, Ky, Kt, Lx, Ly, Tt):
    x, y, t = xyt[:,0], xyt[:,1], xyt[:,2]
    fx = fourier_encode_1d(x, Kx, Lx)
    fy = fourier_encode_1d(y, Ky, Ly)
    ft = fourier_encode_1d(t, Kt, Tt)
    return torch.cat([fx, fy, ft], dim=-1)

class FourierMLP1(nn.Module):
    def __init__(self, width=256, depth=6, in_dim=None, out_dim=1):
        super().__init__()
        assert in_dim is not None
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth-2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, feat): return self.net(feat).squeeze(-1)


# ----- SVGP (scalar) -----
class SVGPModel1(ApproximateGP):
    def __init__(self, Z):  # Z in [0,1]^3
        M = Z.size(0)
        q = CholeskyVariationalDistribution(M)
        vs = VariationalStrategy(self, Z, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ConstantMean()
        # Use generic RBF over (x,y,t); state_dict will set exact params
        self.covar_module = ScaleKernel(ProductKernel(RBFKernel(), RBFKernel(), RBFKernel()))
    def forward(self, X):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(X), self.covar_module(X))


# In[6]:


# ----------------------
# Evaluators
# ----------------------
def eval_fieldformer(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps = data
    device = torch.device(cfg.device)

    ff = FFAG_Pollution(
        d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128,
        Nx=Nx, Ny=Ny, Nt=Nt,
        Lx=extents['Lx'], Ly=extents['Ly'], Tt=extents['Tt'],
        dx=steps['dx'], dy=steps['dy'], dt=steps['dt'],
        coords=coords.to(device), vals=vals.to(device)
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # prefer EMA if present
    sd = state.get("ema_model_state_dict", state.get("model_state_dict", state))
    ff.load_state_dict(sd, strict=False)
    ff.eval()

    with torch.no_grad():
        gam = torch.exp(ff.log_gammas).detach().cpu().numpy()
        offsets_ijk = FFAG_Pollution.build_offset_table(k=ff.k, gammas=gam, dx=steps['dx'], dy=steps['dy'], dt=steps['dt'])

    def predict_over_indices(lin_idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        use_amp = (cfg.device == "cuda")
        for i in range(0, lin_idx.numel(), bs):
            q = lin_idx[i:i+bs]
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
                out = ff(q, offsets_ijk)    # (B,)
            outs.append(out.float())
        return torch.cat(outs, dim=0)

    ds = OpenDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(device)
    all_idx  = torch.arange(Nx*Ny*Nt, device=device, dtype=test_idx.dtype)

    pred_test = predict_over_indices(test_idx, cfg.batch_eval_idx)
    tgt_test  = vals[test_idx]
    rmse_s, mae_s = rmse_mae(pred_test, tgt_test)

    pred_all = predict_over_indices(all_idx, cfg.batch_eval_idx)
    tgt_all  = vals[all_idx]
    rmse_f, mae_f = rmse_mae(pred_all, tgt_all)

    boots = _make_bootstrap_indices(torch.arange(test_idx.numel(), device=device, dtype=test_idx.dtype),
                                    cfg.bootstrap_n, cfg.bootstrap_frac,
                                    cfg.bootstrap_with_replacement, cfg.bootstrap_seed)
    boot_stats = _bootstrap_metrics(pred_test, tgt_test, boots)

    return {
        "rmse_test": rmse_s, "mae_test": mae_s,
        "rmse_full": rmse_f, "mae_full": mae_f,
        "bootstrap_test": {"n": int(cfg.bootstrap_n), "frac": float(cfg.bootstrap_frac), **boot_stats},
    }


def eval_siren(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, *_ = data
    device = torch.device(cfg.device)
    state = torch.load(ckpt_path, map_location=device)
    conf = state.get("config", {})
    model = SIREN1(
        in_dim=3,
        width=int(conf.get("width", 256)),
        depth=int(conf.get("depth", 6)),
        out_dim=1,
        w0=float(conf.get("w0", 30.0)) if "w0" in conf else 30.0,
        w0_hidden=float(conf.get("w0_hidden", 1.0)) if "w0_hidden" in conf else 1.0,
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    def predict_over_indices(idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, idx.numel(), bs):
                ii = idx[i:i+bs]
                outs.append(model(coords[ii]))
        return torch.cat(outs, dim=0)

    ds = OpenDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(device)
    all_idx  = torch.arange(Nx*Ny*Nt, device=device, dtype=test_idx.dtype)

    pred_test = predict_over_indices(test_idx, cfg.batch_eval_nf)
    rmse_s, mae_s = rmse_mae(pred_test, vals[test_idx])
    pred_all = predict_over_indices(all_idx, cfg.batch_eval_nf)
    rmse_f, mae_f = rmse_mae(pred_all, vals[all_idx])

    boots = _make_bootstrap_indices(torch.arange(test_idx.numel(), device=device, dtype=test_idx.dtype),
                                    cfg.bootstrap_n, cfg.bootstrap_frac,
                                    cfg.bootstrap_with_replacement, cfg.bootstrap_seed)
    boot_stats = _bootstrap_metrics(pred_test, vals[test_idx], boots)

    return {"rmse_test": rmse_s, "mae_test": mae_s, "rmse_full": rmse_f, "mae_full": mae_f,
            "bootstrap_test": {"n": int(cfg.bootstrap_n), "frac": float(cfg.bootstrap_frac), **boot_stats}}


def eval_fmlp(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps = data
    device = torch.device(cfg.device)
    state = torch.load(ckpt_path, map_location=device)
    conf = state.get("config", {})

    Kx = int(conf.get("Kx", 16)); Ky = int(conf.get("Ky", 16)); Kt = int(conf.get("Kt", 8))
    Kx_list = build_harmonics(Kx).to(device)
    Ky_list = build_harmonics(Ky).to(device)
    Kt_list = build_harmonics(Kt).to(device)
    in_dim = 2*(len(Kx_list)+len(Ky_list)+len(Kt_list))
    model = FourierMLP1(width=int(conf.get("width", 256)), depth=int(conf.get("depth", 6)), in_dim=in_dim, out_dim=1).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    Lx, Ly, Tt = extents['Lx'], extents['Ly'], extents['Tt']
    def predict_over_indices(idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, idx.numel(), bs):
                ii = idx[i:i+bs]
                feat = fourier_encode_3d(coords[ii], Kx_list, Ky_list, Kt_list, Lx, Ly, Tt)
                outs.append(model(feat))
        return torch.cat(outs, dim=0)

    ds = OpenDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(device)
    all_idx  = torch.arange(Nx*Ny*Nt, device=device, dtype=test_idx.dtype)

    pred_test = predict_over_indices(test_idx, cfg.batch_eval_nf)
    rmse_s, mae_s = rmse_mae(pred_test, vals[test_idx])
    pred_all = predict_over_indices(all_idx, cfg.batch_eval_nf)
    rmse_f, mae_f = rmse_mae(pred_all, vals[all_idx])

    boots = _make_bootstrap_indices(torch.arange(test_idx.numel(), device=device, dtype=test_idx.dtype),
                                    cfg.bootstrap_n, cfg.bootstrap_frac,
                                    cfg.bootstrap_with_replacement, cfg.bootstrap_seed)
    boot_stats = _bootstrap_metrics(pred_test, vals[test_idx], boots)

    return {"rmse_test": rmse_s, "mae_test": mae_s, "rmse_full": rmse_f, "mae_full": mae_f,
            "bootstrap_test": {"n": int(cfg.bootstrap_n), "frac": float(cfg.bootstrap_frac), **boot_stats}}


def load_checkpoint(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        try:
            from torch.serialization import add_safe_globals
            import numpy.core.multiarray as _np_ma
            add_safe_globals([_np_ma._reconstruct])
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            return torch.load(path, map_location=device, weights_only=False)


def eval_svgp(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps = data
    dev = torch.device(getattr(cfg, "svgp_device", cfg.device))

    # Build normalized inputs [0,1]^3
    x_np = x_t.detach().cpu().numpy(); y_np = y_t.detach().cpu().numpy(); t_np = t_t.detach().cpu().numpy()
    XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
    coords_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)
    x_min, x_max = x_np.min(), x_np.max()
    y_min, y_max = y_np.min(), y_np.max()
    t_min, t_max = (t_np.min(), t_np.max()) if len(t_np)>1 else (0.0, 1.0)
    coords01 = np.empty_like(coords_np, dtype=np.float32)
    coords01[:,0] = (coords_np[:,0]-x_min)/max(1e-12, x_max-x_min)
    coords01[:,1] = (coords_np[:,1]-y_min)/max(1e-12, y_max-y_min)
    coords01[:,2] = (coords_np[:,2]-t_min)/max(1e-12, t_max-t_min)
    X01 = torch.from_numpy(coords01).float().to(dev)

    state = load_checkpoint(ckpt_path, dev)
    Z = state["inducing_points"].float().to(dev)
    conf = state.get("config", {})
    y_mean = float(conf.get("y_mean", 0.0))
    y_std  = float(conf.get("y_std", 1.0))

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model = SVGPModel1(Z).to(dev)
    model.load_state_dict(state["model_state_dict"], strict=False)
    likelihood.load_state_dict(state["likelihood_state_dict"], strict=False)
    model.eval(); likelihood.eval()

    ds = OpenDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(dev)
    all_idx  = torch.arange(Nx*Ny*Nt, device=dev, dtype=test_idx.dtype)

    def predict_over_indices(idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        use_amp = (dev.type == "cuda" and getattr(cfg, "svgp_autocast", False))
        max_root = int(getattr(cfg, "svgp_max_root", 16))
        max_prec = int(getattr(cfg, "svgp_max_precond", 8))
        jitter   = float(getattr(cfg, "svgp_jitter", 1e-4))

        def _predict_chunk(ii, amp_enabled, extra_jitter, device_override=None):
            Xii = X01[ii] if device_override is None else X01[ii].to(device_override)
            mdl = model if device_override is None else model.to(device_override)
            lik = likelihood if device_override is None else likelihood.to(device_override)
            ac = torch.amp.autocast("cuda", enabled=(amp_enabled and (device_override is None or device_override.type=="cuda")))
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.max_root_decomposition_size(max_root), \
                 gpytorch.settings.max_preconditioner_size(max_prec), \
                 gpytorch.settings.cholesky_jitter(extra_jitter), ac:
                mu = lik(mdl(Xii)).mean  # standardized
            return mu

        for i in range(0, idx.numel(), bs):
            ii = idx[i:i+bs]
            try:
                mu = _predict_chunk(ii, amp_enabled=use_amp, extra_jitter=jitter)
                if torch.isnan(mu).any() or torch.isinf(mu).any():
                    raise FloatingPointError("NaN/Inf in AMP chunk")
            except Exception:
                try:
                    mu = _predict_chunk(ii, amp_enabled=False, extra_jitter=jitter*10)
                    if torch.isnan(mu).any() or torch.isinf(mu).any():
                        raise FloatingPointError("NaN/Inf in fp32 chunk")
                except Exception:
                    mu = _predict_chunk(ii, amp_enabled=False, extra_jitter=jitter*10, device_override=torch.device("cpu"))
                    mu = mu.to(dev)
            outs.append(mu.float() * y_std + y_mean)
            if dev.type == "cuda":
                del mu; torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)

    pred_test = predict_over_indices(test_idx, cfg.batch_eval_svgp)
    tgt_test  = vals[test_idx].to(dev)
    rmse_s, mae_s = rmse_mae(pred_test, tgt_test)

    pred_all = predict_over_indices(all_idx, cfg.batch_eval_svgp)
    tgt_all  = vals[all_idx].to(dev)
    rmse_f, mae_f = rmse_mae(pred_all, tgt_all)

    boots = _make_bootstrap_indices(torch.arange(test_idx.numel(), device=dev, dtype=test_idx.dtype),
                                    cfg.bootstrap_n, cfg.bootstrap_frac,
                                    cfg.bootstrap_with_replacement, cfg.bootstrap_seed)
    boot_stats = _bootstrap_metrics(pred_test, tgt_test, boots)

    return {"rmse_test": rmse_s, "mae_test": mae_s, "rmse_full": rmse_f, "mae_full": mae_f,
            "bootstrap_test": {"n": int(cfg.bootstrap_n), "frac": float(cfg.bootstrap_frac), **boot_stats}}


# In[7]:


# ----------------------
# Main
# ----------------------
def main():
    cfg = Config()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    data = load_pollution_data(cfg)

    print("== Evaluating FieldFormer-Autograd (Pollution) ==")
    try:
        ff_metrics = eval_fieldformer(cfg, data, cfg.ckpt_fieldformer)
        print(ff_metrics)
    except FileNotFoundError:
        print("(skip) FieldFormer-Autograd checkpoint not found")
    except Exception as e:
        print(f"(error) FieldFormer-Autograd eval failed: {e}")

    print("== Evaluating SIREN (Pollution) ==")
    try:
        siren_metrics = eval_siren(cfg, data, cfg.ckpt_siren)
        print(siren_metrics)
    except FileNotFoundError:
        print("(skip) SIREN checkpoint not found")
    except Exception as e:
        print(f"(error) SIREN eval failed: {e}")

    print("== Evaluating Fourier-MLP (Pollution) ==")
    try:
        fmlp_metrics = eval_fmlp(cfg, data, cfg.ckpt_fmlp)
        print(fmlp_metrics)
    except FileNotFoundError:
        print("(skip) Fourier-MLP checkpoint not found")
    except Exception as e:
        print(f"(error) Fourier-MLP eval failed: {e}")

    print("== Evaluating SVGP (Pollution) ==")
    try:
        svgp_metrics = eval_svgp(cfg, data, cfg.ckpt_svgp)
        print(svgp_metrics)
    except FileNotFoundError:
        print("(skip) SVGP checkpoint not found")
    except Exception as e:
        print(f"(error) SVGP eval failed: {e}")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




