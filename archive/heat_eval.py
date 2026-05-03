#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
Evaluation script for FieldFormer-Autograd and SVGP on the periodic heat dataset.
- Loads data exactly like ff(ag)_heat_train.py
- Loads trained checkpoints (paths in Config)
- Computes RMSE and MAE on:
   (A) sensor coordinates (the held-out test split indices)
   (B) entire field (all grid points)
- Optional: physics residual metrics (R = u_t - (alpha_x u_xx + alpha_y u_yy) - f) via autograd

No CLI args: tweak the Config class below.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# For stable higher-order grads with Transformer attention
from torch.backends.cuda import sdp_kernel

import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, ProductKernel, PeriodicKernel


# In[2]:


# ----------------------
# Config (edit here)
# ----------------------
class Config:
    # Data
    data_path = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
    seed = 123
    train_frac = 0.80
    val_frac = 0.10

    # Checkpoints
    ckpt_fieldformer = "/scratch/ab9738/fieldformer/model/ffag_heat_best.pt"      # path to FieldFormer-Autograd .pt
    ckpt_svgp        = "/scratch/ab9738/fieldformer/model/svgp_heat_best.pt"         # path to SVGP .pt
    ckpt_siren       = "/scratch/ab9738/fieldformer/model/siren_heat_best.pt"
    ckpt_fmlp        = "/scratch/ab9738/fieldformer/model/fmlp_heat_best.pt"

    # Eval
    batch_eval_idx = 1024        # batch size for index-based eval (FieldFormer)
    batch_eval_nf = 16384
    batch_eval_svgp = 2048      # was 16384; drop further (1024/512) if needed
    svgp_autocast   = False     # start in fp32; turn True only if stable
    svgp_max_root   = 16        # smaller root decomposition (uses less VRAM, more stable)
    svgp_max_precond= 8         # smaller preconditioner
    svgp_jitter     = 1e-4      # extra numerical jitter during eval
    svgp_device     = "cuda"    # set "cpu" if you still OOM/NaN


    # Physics residual options
    compute_physics = True
    residual_points_cap = 10000  # when evaluating full-field residuals, cap #points for speed
    residual_chunk = 1024        # chunk residual eval to bound graph memory

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bootstrap settings
    bootstrap_n = 5
    bootstrap_frac = 0.8      # use 1.0 for textbook bootstrap
    bootstrap_with_replacement = True
    bootstrap_seed = 123


# In[3]:


# ----------------------
# Data loading (exactly as training)
# ----------------------

def load_heat_data(cfg: Config):
    pack = np.load(cfg.data_path)
    u_np = pack["u"]  # (Nx, Ny, Nt)
    x_np = pack["x"]; y_np = pack["y"]; t_np = pack["t"]

    # Optional params if present
    params = pack.get("params", None)
    names  = pack.get("param_names", None)
    if params is not None and names is not None:
        names = list(names)
        alpha_x = float(params[names.index("alpha_x")]) if "alpha_x" in names else 1.0
        alpha_y = float(params[names.index("alpha_y")]) if "alpha_y" in names else 1.0
        dx = float(params[names.index("dx")]) if "dx" in names else float(x_np[1]-x_np[0])
        dy = float(params[names.index("dy")]) if "dy" in names else float(y_np[1]-y_np[0])
        dt = float(params[names.index("dt")]) if "dt" in names else float(t_np[1]-t_np[0]) if len(t_np)>1 else 1.0
    else:
        # Fallbacks from arrays
        alpha_x = alpha_y = 1.0
        dx = float(x_np[1]-x_np[0])
        dy = float(y_np[1]-y_np[0])
        dt = float(t_np[1]-t_np[0]) if len(t_np)>1 else 1.0

    Nx, Ny, Nt = u_np.shape

    XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
    coords_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1).astype(np.float32)  # (N,3)
    vals_np = u_np.reshape(-1).astype(np.float32)                                          # (N,)

    device = torch.device(cfg.device)
    coords = torch.from_numpy(coords_np).to(device)
    vals   = torch.from_numpy(vals_np).to(device)
    x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
    t_t = torch.from_numpy(t_np.astype(np.float32)).to(device)

    extents = {
        "Lx": float(x_np.max() - x_np.min()),
        "Ly": float(y_np.max() - y_np.min()),
        "Tt": float(t_np.max() - t_np.min()) if Nt > 1 else 1.0,
    }
    steps = {"dx": dx, "dy": dy, "dt": dt}
    alphas = {"alpha_x": alpha_x, "alpha_y": alpha_y}

    return (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps, alphas

# Splitter matching training
class HeatPeriodicDataset(Dataset):
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
# FieldFormer-Autograd (subset: inference utilities)
# ----------------------
class FieldFormerAutograd(nn.Module):
    def __init__(self, d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128,
                 wrap=True,
                 # bind geometry & data so we don't rely on globals
                 Nx=None, Ny=None, Nt=None, Lx=None, Ly=None, Tt=None,
                 dx=None, dy=None, dt=None,
                 coords=None, vals=None):
        super().__init__()
        # arch
        self.k = k_neighbors
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        # wrap flags (no temporal wrap)
        self.wrap_x = bool(wrap)
        self.wrap_y = bool(wrap)
        self.wrap_t = False

        # bind problem geometry & tensors (buffers so .to(device)/ckpt-safe)
        self.Nx, self.Ny, self.Nt = int(Nx), int(Ny), int(Nt)
        self.register_buffer("coords", coords)   # (N,3)
        self.register_buffer("vals",   vals)     # (N,)
        self.register_buffer("Lx_buf", torch.tensor(float(Lx)))
        self.register_buffer("Ly_buf", torch.tensor(float(Ly)))
        self.register_buffer("Tt_buf", torch.tensor(float(Tt)))
        self.register_buffer("dx_buf", torch.tensor(float(dx)))
        self.register_buffer("dy_buf", torch.tensor(float(dy)))
        self.register_buffer("dt_buf", torch.tensor(float(dt)))

        # training uses this buffer; harmless in eval
        self.register_buffer("_huber_delta_ema", torch.tensor(1.0, dtype=torch.float32))

    # handy accessors
    @property
    def Lx(self): return float(self.Lx_buf.item())
    @property
    def Ly(self): return float(self.Ly_buf.item())
    @property
    def Tt(self): return float(self.Tt_buf.item())
    @property
    def dx(self): return float(self.dx_buf.item())
    @property
    def dy(self): return float(self.dy_buf.item())
    @property
    def dt(self): return float(self.dt_buf.item())

    # index helpers use bound dims
    def lin_to_ijk(self, lin):
        i = lin // (self.Ny * self.Nt)
        r = lin %  (self.Ny * self.Nt)
        j = r // self.Nt
        k = r %  self.Nt
        return i, j, k

    def ijk_to_lin(self, i, j, k):
        return (i % self.Nx) * (self.Ny * self.Nt) + (j % self.Ny) * self.Nt + (k % self.Nt)

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
                    dxp = di * dx; dyp = dj * dy; dtp = dk * dt
                    dtp_eff = dtp / (1.0 + abs(dk) / 2.0)   # softens far time neighbors
                    d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp_eff)**2
                    cand.append((d2, di, dj, dk))
        cand.sort(key=lambda t: t[0])
        n_time = max(1, int(k * frac_time))
        n_space = k - n_time
        cand_time  = sorted(cand, key=lambda t: abs(t[3]))
        cand_space = sorted(cand, key=lambda t: (t[0], abs(t[3])))
        sel, seen = [], set()
        for _, di, dj, dk in cand_time:
            key = (di, dj, dk)
            if key not in seen:
                sel.append(key); seen.add(key)
            if len(sel) >= n_time: break
        for _, di, dj, dk in cand_space:
            key = (di, dj, dk)
            if key not in seen:
                sel.append(key); seen.add(key)
            if len(sel) >= k: break
        return sel

    # axis-aware gather (no temporal wrap)
    def _gather_neighbors(self, q_lin_idx, offsets_ijk):
        i, j, k0 = self.lin_to_ijk(q_lin_idx)   # (B,)
        sel = offsets_ijk[:self.k]
        dev = q_lin_idx.device; dtype = i.dtype
        di = torch.tensor([o[0] for o in sel], device=dev, dtype=dtype)
        dj = torch.tensor([o[1] for o in sel], device=dev, dtype=dtype)
        dk = torch.tensor([o[2] for o in sel], device=dev, dtype=dtype)
        I = i[:, None] + di[None, :]
        J = j[:, None] + dj[None, :]
        K = k0[:, None] + dk[None, :]
        I = I % self.Nx if self.wrap_x else I.clamp_(0, self.Nx-1)
        J = J % self.Ny if self.wrap_y else J.clamp_(0, self.Ny-1)
        K = K % self.Nt if self.wrap_t else K.clamp_(0, self.Nt-1)
        return I * (self.Ny * self.Nt) + J * self.Nt + K

    # rel deltas (γ-scaled) with space wrap only
    @staticmethod
    def _delta_periodic(a, b, L):
        return (a - b + 0.5*L) % L - 0.5*L

    def _relative_deltas(self, q_xyz, nb_xyz):
        if self.wrap_x:
            dx_ = self._delta_periodic(nb_xyz[...,0], q_xyz[:,None,0], self.Lx)
        else:
            dx_ = nb_xyz[...,0] - q_xyz[:,None,0]
        if self.wrap_y:
            dy_ = self._delta_periodic(nb_xyz[...,1], q_xyz[:,None,1], self.Ly)
        else:
            dy_ = nb_xyz[...,1] - q_xyz[:,None,1]
        dt_ = nb_xyz[...,2] - q_xyz[:,None,2]  # no wrap in time
        rel = torch.stack([dx_, dy_, dt_], dim=-1)
        scale = torch.exp(self.log_gammas)[None,None,:]
        return rel * scale

    def forward(self, q_lin_idx, offsets_ijk):
        nb_idx = self._gather_neighbors(q_lin_idx, offsets_ijk)     # (B,k)
        q_xyz  = self.coords[q_lin_idx]                              # (B,3)
        nb_xyz = self.coords[nb_idx]                                 # (B,k,3)
        rel    = self._relative_deltas(q_xyz, nb_xyz)                # (B,k,3)
        nb_vals = self.vals[nb_idx][..., None].to(torch.float32)     # (B,k,1)
        tokens  = torch.cat([rel, nb_vals], dim=-1)                  # (B,k,4)
        h = self.input_proj(tokens); h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)

# Forcing (same as training)
pi = float(torch.acos(torch.zeros(1)).item()*2)

def forcing_torch(xx, yy, tt):
    return 5.0 * torch.cos(pi * xx) * torch.cos(pi * yy) * torch.sin(4 * pi * tt / 5.0)


# In[5]:


# ----------------------
# SVGP model def for loading
# ----------------------
class SVGPModel(ApproximateGP):
    def __init__(self, Z):  # Z in [0,1]^3
        M = Z.size(0)
        q = CholeskyVariationalDistribution(M)
        vs = VariationalStrategy(self, Z, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ConstantMean()
        kx, ky, kt = PeriodicKernel(), PeriodicKernel(), PeriodicKernel()
        kx.initialize(period_length=1.0); ky.initialize(period_length=1.0); kt.initialize(period_length=1.0)
        self.covar_module = ScaleKernel(ProductKernel(kx, ky, kt))
    def forward(self, X):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(X), self.covar_module(X))



# In[6]:


# ----------------------
# Extra: SIREN + FourierMLP evaluators (drop-in for trained checkpoints)
# ----------------------

# ===== SIREN defs =====
class SineLayer(nn.Linear):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__(in_features, out_features)
        self.w0 = float(w0); self.is_first = bool(is_first)
        self._init_weights()
    def _init_weights(self):
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

class SIREN(nn.Module):
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
    def forward(self, xyt):
        y = xyt
        for layer in self.hidden: y = layer(y)
        return self.final(y).squeeze(-1)

# ===== Fourier-MLP defs =====

def build_harmonics(K):
    if isinstance(K, int):
        return torch.arange(1, K+1, dtype=torch.float32)
    return torch.tensor(K, dtype=torch.float32)

def fourier_encode_1d(x, Ks, L):
    z = (2 * math.pi) * (x[..., None] / (L if L > 0 else 1.0)) * Ks[None, :].to(x)
    return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

def fourier_encode_3d(xyt, Kx_list, Ky_list, Kt_list, Lx, Ly, Tt):
    x, y, t = xyt[:,0], xyt[:,1], xyt[:,2]
    fx = fourier_encode_1d(x, Kx_list, Lx)
    fy = fourier_encode_1d(y, Ky_list, Ly)
    ft = fourier_encode_1d(t, Kt_list, Tt)
    return torch.cat([fx, fy, ft], dim=-1)

class FourierMLP(nn.Module):
    def __init__(self, width=256, depth=6, in_dim=None):
        super().__init__()
        assert in_dim is not None
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth-2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, feat):
        return self.net(feat).squeeze(-1)


# In[7]:


# ----------------------
# Metrics
# ----------------------

def rmse_mae(pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[float,float]:
    se = F.mse_loss(pred, tgt, reduction="mean").item()
    ae = F.l1_loss(pred, tgt, reduction="mean").item()
    return math.sqrt(se), ae


# In[8]:


# ----------------------
# Bootstrap Utilities
# ----------------------

def _make_bootstrap_indices(base_idx: torch.Tensor, n_boot: int, frac: float,
                            with_replacement: bool = True, seed: int = 123):
    """Return a list of index tensors into base_idx for bootstrap subsets."""
    N = base_idx.numel()
    m = max(1, int(round(frac * N)))
    g = torch.Generator(device=base_idx.device)
    g.manual_seed(seed)
    boots = []
    for b in range(n_boot):
        if with_replacement:
            sel = torch.randint(low=0, high=N, size=(m,), generator=g, device=base_idx.device)
        else:
            # sample without replacement if requested (not classic bootstrap)
            sel = torch.randperm(N, generator=g, device=base_idx.device)[:m]
        boots.append(sel)
    return boots

def _agg_mean_std(values: torch.Tensor):
    """Return (mean, std) as Python floats for a 1D tensor."""
    if values.numel() == 1:
        return float(values.item()), 0.0
    return float(values.mean().item()), float(values.std(unbiased=True).item())

def _bootstrap_metrics(pred_all: torch.Tensor, tgt_all: torch.Tensor, boots: list):
    """Compute RMSE/MAE for each bootstrap subset; return mean/std for each metric."""
    rmses, maes = [], []
    for sel in boots:
        p = pred_all.index_select(0, sel)
        t = tgt_all.index_select(0, sel)
        se = torch.mean((p - t) ** 2)
        ae = torch.mean((p - t).abs())
        rmses.append(torch.sqrt(se))
        maes.append(ae)
    rmse_mean, rmse_std = _agg_mean_std(torch.stack(rmses))
    mae_mean,  mae_std  = _agg_mean_std(torch.stack(maes))
    return {"rmse_mean": rmse_mean, "rmse_std": rmse_std,
            "mae_mean": mae_mean,   "mae_std": mae_std}


# In[12]:


# ----------------------
# Evaluation routines
# ----------------------

def eval_fieldformer(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps, alphas = data
    device = torch.device(cfg.device)

    # Build model and load weights
    ff = FieldFormerAutograd(
        d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128, wrap=True,
        Nx=Nx, Ny=Ny, Nt=Nt,
        Lx=extents['Lx'], Ly=extents['Ly'], Tt=extents['Tt'],
        dx=steps['dx'], dy=steps['dy'], dt=steps['dt'],
        coords=coords.to(device), vals=vals.to(device)
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ff.load_state_dict(state["model_state_dict"], strict=False)
    ff.eval()

    with torch.no_grad():
        gam = torch.exp(ff.log_gammas).detach().cpu().numpy()
        offsets_ijk = ff.build_offset_table(k=ff.k, gammas=gam, dx=ff.dx, dy=ff.dy, dt=ff.dt)

    def predict_over_indices(lin_idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        use_amp = (cfg.device == "cuda")
        for i in range(0, lin_idx.numel(), bs):
            q = lin_idx[i:i+bs]
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
                out = ff(q, offsets_ijk)          # <-- only (q, offsets_ijk)
            outs.append(out.float())
        return torch.cat(outs, dim=0)

    # Dataset split
    ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(device)

    # (A) Sensors: test split
    pred_test = predict_over_indices(test_idx, cfg.batch_eval_idx)
    tgt_test  = vals[test_idx]
    rmse_s, mae_s = rmse_mae(pred_test, tgt_test)

            # (A-b) Bootstrap over test split
    boots = _make_bootstrap_indices(
        base_idx=torch.arange(test_idx.numel(), device=test_idx.device, dtype=test_idx.dtype),
        n_boot=getattr(cfg, "bootstrap_n", 5),
        frac=getattr(cfg, "bootstrap_frac", 0.8),
        with_replacement=getattr(cfg, "bootstrap_with_replacement", True),
        seed=getattr(cfg, "bootstrap_seed", 123),
    )
    # We already have predictions for the whole test set (pred_test); index into that:
    boot_stats = _bootstrap_metrics(pred_test, tgt_test, boots)

    # (B) Entire field
    all_idx = torch.arange(Nx*Ny*Nt, device=device, dtype=test_idx.dtype)
    pred_all = predict_over_indices(all_idx, cfg.batch_eval_idx)
    tgt_all  = vals[all_idx]
    rmse_f, mae_f = rmse_mae(pred_all, tgt_all)

    # Physics residuals (optional)
    phys = {}
    if cfg.compute_physics:
        alpha_x, alpha_y = alphas['alpha_x'], alphas['alpha_y']

        def compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y, eps=1e-8):
            """
            Compute relative residual for the heat equation:
                ut - (alpha_x u_xx + alpha_y u_yy) - f
            normalized by magnitudes of governing terms.

            Returns: tensor of relative residuals, same shape as R.
            """
            denom = (ut.abs() +
                     (alpha_x * uxx).abs() +
                     (alpha_y * uyy).abs() +
                     f.abs() + eps)
            return R.abs() / denom

        def pde_residual_autograd_ff(q_lin_idx: torch.Tensor):
            # query coords as variables
            xyt = ff.coords[q_lin_idx].clone().detach().requires_grad_(True)  # (B,3)

            # neighbors are fixed, gathered with same axis-wrap as training
            with torch.no_grad():
                gam = torch.exp(ff.log_gammas).detach().cpu().numpy()
                offsets_ijk = ff.build_offset_table(k=ff.k, gammas=gam, dx=ff.dx, dy=ff.dy, dt=ff.dt)
            nb_idx = ff._gather_neighbors(q_lin_idx, offsets_ijk)
            nb_xyz = ff.coords[nb_idx]

            # relative deltas (space wrap only), γ-scaled
            if ff.wrap_x:
                dxv = FieldFormerAutograd._delta_periodic(nb_xyz[...,0], xyt[:,None,0], ff.Lx)
            else:
                dxv = nb_xyz[...,0] - xyt[:,None,0]
            if ff.wrap_y:
                dyv = FieldFormerAutograd._delta_periodic(nb_xyz[...,1], xyt[:,None,1], ff.Ly)
            else:
                dyv = nb_xyz[...,1] - xyt[:,None,1]
            dtv = nb_xyz[...,2] - xyt[:,None,2]  # no wrap in time
            rel = torch.stack([dxv, dyv, dtv], dim=-1) * torch.exp(ff.log_gammas)[None,None,:]

            nb_vals_tok = ff.vals[nb_idx][..., None].to(torch.float32)
            tokens = torch.cat([rel, nb_vals_tok], dim=-1)

            # fp32 + math SDPA for stable higher-order autograd
            with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), \
                 torch.cuda.amp.autocast(enabled=False):
                h = ff.input_proj(tokens)
                h = ff.encoder(h)
                u = ff.head(h.mean(dim=1)).squeeze(-1)

            ones  = torch.ones_like(u)
            grads = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
            ux, uy, ut = grads[:,0], grads[:,1], grads[:,2]
            uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:,0]
            uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:,1]

            f = forcing_torch(xyt[:,0], xyt[:,1], xyt[:,2])
            R = ut - (alphas['alpha_x'] * uxx + alphas['alpha_y'] * uyy) - f
            RelR = compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y)
            return R, RelR


        def residual_stats_chunked(index_tensor: torch.Tensor, chunk: int):
            ssq = 0.0; sa = 0.0; n = 0
            ssq_rel = 0.0; sa_rel = 0.0
            for i in range(0, index_tensor.numel(), chunk):
                ii = index_tensor[i:i+chunk]
                R, RelR = pde_residual_autograd_ff(ii)

                # raw residual accumulators
                ssq += float((R**2).sum().item())
                sa  += float(R.abs().sum().item())
                n   += R.numel()

                # relative residual accumulators
                ssq_rel += float((RelR**2).sum().item())
                sa_rel  += float(RelR.abs().sum().item())

                del R, RelR
                if cfg.device == "cuda":
                    torch.cuda.empty_cache()

            rmse = math.sqrt(ssq / max(1, n))
            mae  = sa / max(1, n)
            rel_rmse = math.sqrt(ssq_rel / max(1, n))
            rel_mae  = sa_rel / max(1, n)

            return rmse, mae, rel_rmse, rel_mae



        # Residual on sensors (test subset)
        ss = test_idx[::max(1, test_idx.numel() // min(test_idx.numel(), cfg.residual_points_cap))]
        phys['test_residual_rmse'], phys['test_residual_mae'], phys['test_relres_rmse'], phys['test_relres_mae'] = residual_stats_chunked(ss, cfg.residual_chunk)

        # Residual on full field (subsampled if huge)
        aa = all_idx[::max(1, all_idx.numel() // cfg.residual_points_cap)]
        phys['full_residual_rmse'], phys['full_residual_mae'], phys['full_relres_rmse'], phys['full_relres_mae'] = residual_stats_chunked(aa, cfg.residual_chunk)

    return {
        'rmse_test': rmse_s, 'mae_test': mae_s,
        'rmse_full': rmse_f, 'mae_full': mae_f,
        'bootstrap': boot_stats, 'physics': phys,
    }

from typing import Any
def load_checkpoint(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
    # Older torch that doesn't support weights_only
        return torch.load(path, map_location=device)
    except Exception:
        try:
            from torch.serialization import add_safe_globals
            import numpy.core.multiarray as _np_ma
            add_safe_globals([_np_ma._reconstruct])
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            # Last resort: re-raise the original error context for visibility
            return torch.load(path, map_location=device, weights_only=False)

def eval_svgp(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps, alphas = data
    dev = torch.device(getattr(cfg, "svgp_device", cfg.device))

    # Build normalized inputs [0,1]^3 on the chosen device
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

    # Load checkpoint
    state = load_checkpoint(ckpt_path, dev)
    Z = state["Z"].float().to(dev)
    y_mean = float(state.get("config", {}).get("y_mean", 0.0))
    y_std  = float(state.get("config", {}).get("y_std", 1.0))

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model = SVGPModel(Z).to(dev)
    model.load_state_dict(state["model_state_dict"], strict=False)
    likelihood.load_state_dict(state["likelihood_state_dict"], strict=False)
    model.eval(); likelihood.eval()

    # Split
    ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(dev)
    all_idx  = torch.arange(Nx*Ny*Nt, device=dev, dtype=test_idx.dtype)

    def predict_over_indices(idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        use_amp = (dev.type == "cuda" and getattr(cfg, "svgp_autocast", False))
        max_root = int(getattr(cfg, "svgp_max_root", 16))
        max_prec = int(getattr(cfg, "svgp_max_precond", 8))
        jitter   = float(getattr(cfg, "svgp_jitter", 1e-4))

        def _predict_chunk(ii, amp_enabled, extra_jitter, device_override=None):
            ctxs = [
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.max_root_decomposition_size(max_root),
                gpytorch.settings.max_preconditioner_size(max_prec),
                gpytorch.settings.cholesky_jitter(extra_jitter),
            ]
            # choose device if we fallback to CPU
            Xii = X01[ii] if device_override is None else X01[ii].to(device_override)
            mdl = model if device_override is None else model.to(device_override)
            lik = likelihood if device_override is None else likelihood.to(device_override)
            # AMP context only if we're on CUDA
            ac = torch.amp.autocast("cuda", enabled=(amp_enabled and (device_override is None or device_override.type=="cuda")))
            with torch.no_grad(), ctxs[0], ctxs[1], ctxs[2], ctxs[3], ac:
                mu = lik(mdl(Xii)).mean  # standardized
            return mu

        for i in range(0, idx.numel(), bs):
            ii = idx[i:i+bs]

            # 1) try fast path (maybe AMP)
            try:
                mu = _predict_chunk(ii, amp_enabled=use_amp, extra_jitter=jitter)
                if torch.isnan(mu).any() or torch.isinf(mu).any():
                    raise FloatingPointError("NaN/Inf in AMP chunk")
            except Exception:
                # 2) retry in full fp32 (no AMP), higher jitter
                try:
                    mu = _predict_chunk(ii, amp_enabled=False, extra_jitter=jitter*10)
                    if torch.isnan(mu).any() or torch.isinf(mu).any():
                        raise FloatingPointError("NaN/Inf in fp32 chunk")
                except Exception:
                    # 3) last resort: run this chunk on CPU
                    mu = _predict_chunk(ii, amp_enabled=False, extra_jitter=jitter*10, device_override=torch.device("cpu"))
                    mu = mu.to(dev)

            outs.append(mu.float())
            if dev.type == "cuda":
                del mu
                torch.cuda.empty_cache()

        return torch.cat(outs, dim=0) * y_std + y_mean  # back to original scale


    # (A) sensors (test split)
    pred_test = predict_over_indices(test_idx, cfg.batch_eval_svgp)
    tgt_test  = vals[test_idx].to(dev)
    rmse_s, mae_s = rmse_mae(pred_test, tgt_test)

    boots = _make_bootstrap_indices(
        base_idx=torch.arange(test_idx.numel(), device=test_idx.device, dtype=test_idx.dtype),
        n_boot=getattr(cfg, "bootstrap_n", 5),
        frac=getattr(cfg, "bootstrap_frac", 0.8),
        with_replacement=getattr(cfg, "bootstrap_with_replacement", True),
        seed=getattr(cfg, "bootstrap_seed", 123),
    )
    # We already have predictions for the whole test set (pred_test); index into that:
    boot_stats = _bootstrap_metrics(pred_test, tgt_test, boots)

    # (B) full field
    pred_all = predict_over_indices(all_idx, cfg.batch_eval_svgp)
    tgt_all  = vals[all_idx].to(dev)
    rmse_f, mae_f = rmse_mae(pred_all, tgt_all)

    # Physics residuals are extremely memory heavy; keep disabled unless you truly need them.
    # Physics residuals via autograd on predictive mean (CHUNKED + memory-safe)
    phys = {}
    if cfg.compute_physics:
        alpha_x, alpha_y = alphas['alpha_x'], alphas['alpha_y']
        dev = X01.device

        # scales for chain rule (normalized [0,1] -> physical units)
        sx = torch.as_tensor(x_max - x_min, device=dev, dtype=torch.float32)
        sy = torch.as_tensor(y_max - y_min, device=dev, dtype=torch.float32)
        st = torch.as_tensor(max(1e-12, t_max - t_min), device=dev, dtype=torch.float32)

        def compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y, eps=1e-8):
            """
            Compute relative residual for the heat equation:
                ut - (alpha_x u_xx + alpha_y u_yy) - f
            normalized by magnitudes of governing terms.

            Returns: tensor of relative residuals, same shape as R.
            """
            denom = (ut.abs() +
                     (alpha_x * uxx).abs() +
                     (alpha_y * uyy).abs() +
                     f.abs() + eps)
            return R.abs() / denom

        def residual_stats_chunked(index_tensor: torch.Tensor, chunk: int):
            """Compute RMSE/MAE of PDE residual in chunks to cap memory."""
            ssq = 0.0   # sum of squared residuals
            sa  = 0.0   # sum of absolute residuals
            n   = 0     # total count
            ssq_rel = 0.0
            sa_rel = 0.0

            for i in range(0, index_tensor.numel(), chunk):
                ii = index_tensor[i:i+chunk]

                # Inputs require grad (second derivatives needed)
                x = X01[ii].clone().detach().requires_grad_(True)

                # Predictive mean (standardized units). No AMP here—keep fp32 for stability.
                with gpytorch.settings.fast_pred_var():
                    m_std = likelihood(model(x)).mean

                # Back to original scale
                u = m_std * y_std + y_mean

                # First derivatives wrt normalized inputs
                ones  = torch.ones_like(u)
                grads = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
                ux = grads[:, 0] / sx
                uy = grads[:, 1] / sy
                ut = grads[:, 2] / st

                # Second derivatives (chain rule for normalized inputs)
                uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:, 0] / sx
                uyy = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:, 1] / sy

                # Forcing at physical coords
                x_phys = x[:, 0] * sx + x_min
                y_phys = x[:, 1] * sy + y_min
                t_phys = x[:, 2] * st + t_min
                f = forcing_torch(x_phys, y_phys, t_phys)

                # Residual
                R = ut - (alpha_x * uxx + alpha_y * uyy) - f
                RelR = compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y)

                # Accumulate stats (avoid storing all R)
                ssq += float((R ** 2).sum().item())
                sa  += float(R.abs().sum().item())
                n   += R.numel()

                ssq_rel += float((RelR**2).sum().item())
                sa_rel  += float(RelR.abs().sum().item())

                # Cleanup per-chunk to keep memory flat
                del x, m_std, u, ones, grads, ux, uy, ut, uxx, uyy, x_phys, y_phys, t_phys, f, R, RelR
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            rmse = math.sqrt(ssq / max(1, n))
            mae  = sa / max(1, n)
            rel_rmse = math.sqrt(ssq_rel / max(1, n))
            rel_mae  = sa_rel / max(1, n)
            return rmse, mae, rel_rmse, rel_mae

        # Sensors (subsampled)
        ss = test_idx[::max(1, test_idx.numel() // min(test_idx.numel(), cfg.residual_points_cap))]
        phys['test_residual_rmse'], phys['test_residual_mae'], phys['test_relres_rmse'], phys['test_relres_mae'] = residual_stats_chunked(
            ss, getattr(cfg, 'residual_chunk', 1024)
        )

        # Full field (subsampled)
        aa = all_idx[::max(1, all_idx.numel() // cfg.residual_points_cap)]
        phys['full_residual_rmse'], phys['full_residual_mae'], phys['full_relres_rmse'], phys['full_relres_mae'] = residual_stats_chunked(
            aa, getattr(cfg, 'residual_chunk', 1024)
        )


    return {
        'rmse_test': rmse_s, 'mae_test': mae_s,
        'rmse_full': rmse_f, 'mae_full': mae_f,
        'bootstrap': boot_stats, 'physics': phys,
    }


def eval_fmlp(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps, alphas = data
    device = torch.device(cfg.device)
    state = torch.load(ckpt_path, map_location=device)
    conf = state.get("config", {})

    Kx = int(conf.get("Kx", 16)); Ky = int(conf.get("Ky", 16)); Kt = int(conf.get("Kt", 8))
    Kx_list = build_harmonics(Kx).to(device)
    Ky_list = build_harmonics(Ky).to(device)
    Kt_list = build_harmonics(Kt).to(device)
    in_dim = 2*(len(Kx_list)+len(Ky_list)+len(Kt_list))

    model = FourierMLP(width=int(conf.get("width", 256)), depth=int(conf.get("depth", 6)), in_dim=in_dim).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    Lx, Ly, Tt = extents['Lx'], extents['Ly'], extents['Tt']

    def predict_over_indices(idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, idx.numel(), bs):
                ii = idx[i:i+bs]
                xyt = coords[ii]
                feat = fourier_encode_3d(xyt, Kx_list, Ky_list, Kt_list, Lx, Ly, Tt)
                outs.append(model(feat))
        return torch.cat(outs, dim=0)

    ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(device)
    all_idx  = torch.arange(Nx*Ny*Nt, device=device, dtype=test_idx.dtype)

    pred_test = predict_over_indices(test_idx, getattr(cfg, 'batch_eval_nf', 8192))
    tgt_test  = vals[test_idx]
    rmse_s, mae_s = rmse_mae(pred_test, tgt_test)

    boots = _make_bootstrap_indices(
        base_idx=torch.arange(test_idx.numel(), device=test_idx.device, dtype=test_idx.dtype),
        n_boot=getattr(cfg, "bootstrap_n", 5),
        frac=getattr(cfg, "bootstrap_frac", 0.8),
        with_replacement=getattr(cfg, "bootstrap_with_replacement", True),
        seed=getattr(cfg, "bootstrap_seed", 123),
    )
    # We already have predictions for the whole test set (pred_test); index into that:
    boot_stats = _bootstrap_metrics(pred_test, tgt_test, boots)

    pred_all = predict_over_indices(all_idx, getattr(cfg, 'batch_eval_nf', 8192))
    tgt_all  = vals[all_idx]
    rmse_f, mae_f = rmse_mae(pred_all, tgt_all)

    phys = {}
    if getattr(cfg, 'compute_physics', True):
        alpha_x, alpha_y = alphas['alpha_x'], alphas['alpha_y']
        def compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y, eps=1e-8):
            """
            Compute relative residual for the heat equation:
                ut - (alpha_x u_xx + alpha_y u_yy) - f
            normalized by magnitudes of governing terms.

            Returns: tensor of relative residuals, same shape as R.
            """
            denom = (ut.abs() +
                     (alpha_x * uxx).abs() +
                     (alpha_y * uyy).abs() +
                     f.abs() + eps)
            return R.abs() / denom

        def pde_residual(ii: torch.Tensor):
            xyt = coords[ii].clone().detach().requires_grad_(True)
            feat = fourier_encode_3d(xyt, Kx_list, Ky_list, Kt_list, Lx, Ly, Tt)
            u = model(feat)
            ones = torch.ones_like(u)
            g = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
            ux, uy, ut = g[:,0], g[:,1], g[:,2]
            uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:,0]
            uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:,1]
            f = forcing_torch(xyt[:,0], xyt[:,1], xyt[:,2])
            R = ut - (alpha_x * uxx + alpha_y * uyy) - f
            RelR = compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y)
            return R, RelR
        ss = test_idx[::max(1, test_idx.numel() // min(test_idx.numel(), getattr(cfg,'residual_points_cap',50000)))]
        aa = all_idx[::max(1, all_idx.numel() // getattr(cfg,'residual_points_cap',50000))]
        R_s, RelR_s = pde_residual(ss); R_f, RelR_f = pde_residual(aa)
        phys['test_residual_rmse'] = float(torch.sqrt(torch.mean(R_s**2)).item())
        phys['test_residual_mae']  = float(torch.mean(torch.abs(R_s)).item())
        phys['full_residual_rmse'] = float(torch.sqrt(torch.mean(R_f**2)).item())
        phys['full_residual_mae']  = float(torch.mean(torch.abs(R_f)).item())
        phys['test_relres_rmse'] = float(torch.sqrt(torch.mean(RelR_s**2)).item())
        phys['test_relres_mae']  = float(torch.mean(torch.abs(RelR_s)).item())
        phys['full_relres_rmse'] = float(torch.sqrt(torch.mean(RelR_f**2)).item())
        phys['full_relres_mae']  = float(torch.mean(torch.abs(RelR_f)).item())

    return {
        'rmse_test': rmse_s, 'mae_test': mae_s,
        'rmse_full': rmse_f, 'mae_full': mae_f,
        'bootstrap': boot_stats, 'physics': phys,
    }


def eval_siren(cfg: Config, data, ckpt_path: str):
    (Nx, Ny, Nt), coords, vals, x_t, y_t, t_t, extents, steps, alphas = data
    device = torch.device(cfg.device)
    state = torch.load(ckpt_path, map_location=device)
    conf = state.get("config", {})
    model = SIREN(
        in_dim=3,
        width=int(conf.get("width", 256)),
        depth=int(conf.get("depth", 6)),
        out_dim=1,
        w0=float(conf.get("w0", 30.0)),
        w0_hidden=float(conf.get("w0_hidden", 1.0)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    # batching helpers
    def predict_over_indices(idx: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, idx.numel(), bs):
                ii = idx[i:i+bs]
                xyt = coords[ii]
                outs.append(model(xyt))
        return torch.cat(outs, dim=0)

    ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    test_idx = ds.test_idx.to(device)
    all_idx  = torch.arange(Nx*Ny*Nt, device=device, dtype=test_idx.dtype)

    # (A) sensors
    pred_test = predict_over_indices(test_idx, getattr(cfg, 'batch_eval_nf', 8192))
    tgt_test  = vals[test_idx]
    rmse_s, mae_s = rmse_mae(pred_test, tgt_test)

    boots = _make_bootstrap_indices(
        base_idx=torch.arange(test_idx.numel(), device=test_idx.device, dtype=test_idx.dtype),
        n_boot=getattr(cfg, "bootstrap_n", 5),
        frac=getattr(cfg, "bootstrap_frac", 0.8),
        with_replacement=getattr(cfg, "bootstrap_with_replacement", True),
        seed=getattr(cfg, "bootstrap_seed", 123),
    )
    # We already have predictions for the whole test set (pred_test); index into that:
    boot_stats = _bootstrap_metrics(pred_test, tgt_test, boots)

    # (B) full field
    pred_all = predict_over_indices(all_idx, getattr(cfg, 'batch_eval_nf', 8192))
    tgt_all  = vals[all_idx]
    rmse_f, mae_f = rmse_mae(pred_all, tgt_all)

    phys = {}
    if getattr(cfg, 'compute_physics', True):
        alpha_x, alpha_y = alphas['alpha_x'], alphas['alpha_y']
        def compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y, eps=1e-8):
            """
            Compute relative residual for the heat equation:
                ut - (alpha_x u_xx + alpha_y u_yy) - f
            normalized by magnitudes of governing terms.

            Returns: tensor of relative residuals, same shape as R.
            """
            denom = (ut.abs() +
                     (alpha_x * uxx).abs() +
                     (alpha_y * uyy).abs() +
                     f.abs() + eps)
            return R.abs() / denom

        def pde_residual(ii: torch.Tensor):
            xyt = coords[ii].clone().detach().requires_grad_(True)
            u = model(xyt)
            ones = torch.ones_like(u)
            g = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
            ux, uy, ut = g[:,0], g[:,1], g[:,2]
            uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:,0]
            uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:,1]
            f = forcing_torch(xyt[:,0], xyt[:,1], xyt[:,2])
            R = ut - (alpha_x * uxx + alpha_y * uyy) - f
            RelR = compute_relative_residual(R, ut, uxx, uyy, f, alpha_x, alpha_y)
            return R, RelR
        ss = test_idx[::max(1, test_idx.numel() // min(test_idx.numel(), getattr(cfg,'residual_points_cap',50000)))]
        aa = all_idx[::max(1, all_idx.numel() // getattr(cfg,'residual_points_cap',50000))]
        R_s, RelR_s = pde_residual(ss); R_f, RelR_f = pde_residual(aa)
        phys['test_residual_rmse'] = float(torch.sqrt(torch.mean(R_s**2)).item())
        phys['test_residual_mae']  = float(torch.mean(torch.abs(R_s)).item())
        phys['full_residual_rmse'] = float(torch.sqrt(torch.mean(R_f**2)).item())
        phys['full_residual_mae']  = float(torch.mean(torch.abs(R_f)).item())
        phys['test_relres_rmse'] = float(torch.sqrt(torch.mean(RelR_s**2)).item())
        phys['test_relres_mae']  = float(torch.mean(torch.abs(RelR_s)).item())
        phys['full_relres_rmse'] = float(torch.sqrt(torch.mean(RelR_f**2)).item())
        phys['full_relres_mae']  = float(torch.mean(torch.abs(RelR_f)).item())

    return {
        'rmse_test': rmse_s, 'mae_test': mae_s,
        'rmse_full': rmse_f, 'mae_full': mae_f,
        'bootstrap': boot_stats, 'physics': phys,
    }


# In[13]:


# ----------------------
# Main
# ----------------------

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    data = load_heat_data(cfg)

    print("== Evaluating FieldFormer-Autograd ==")
    try:
        ff_metrics = eval_fieldformer(cfg, data, cfg.ckpt_fieldformer)
        print(ff_metrics)
    except FileNotFoundError:
        print("(skip) FieldFormer-Autograd checkpoint not found")
    except Exception as e:
        print(f"(error) FieldFormer-Autograd eval failed: {e}")

    print("== Evaluating SIREN ==")
    try:
        siren_metrics = eval_siren(cfg, data, getattr(cfg, 'ckpt_siren', 'siren_heat_best.pt'))
        print(siren_metrics)
    except FileNotFoundError:
        print("(skip) SIREN checkpoint not found")
    except Exception as e:
        print(f"(error) SIREN eval failed: {e}")

    print("== Evaluating Fourier-MLP ==")
    try:
        fmlp_metrics = eval_fmlp(cfg, data, getattr(cfg, 'ckpt_fmlp', 'fmlp_heat_best.pt'))
        print(fmlp_metrics)
    except FileNotFoundError:
        print("(skip) Fourier-MLP checkpoint not found")
    except Exception as e:
        print(f"(error) Fourier-MLP eval failed: {e}")

    print("== Evaluating SVGP ==")
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




