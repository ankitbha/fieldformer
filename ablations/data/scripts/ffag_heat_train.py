#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# ===== FieldFormer_Autograd: local transformer with PINN-style physics loss =====
# Minimal diffs from ff_heat_train.py: (1) add continuous dependence on (x,y,t)
# via relative deltas to query coords; (2) compute PDE residual with autograd.
# Neighbor gathering, loaders, optimizer, scheduler, early stop remain aligned.
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import math
from torch.backends.cuda import sdp_kernel

torch.pi = torch.acos(torch.zeros(1)).item() * 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# ----------------------
# Data: load the periodic heat dataset (same path as original)
# ----------------------
pack = np.load("/scratch/ab9738/fieldformer/data/heat_sharp_dataset.npz")
u_np   = pack["u"]           # (Nx, Ny, Nt)
x_np   = pack["x"]           # (Nx,)
y_np   = pack["y"]           # (Ny,)
t_np   = pack["t"]           # (Nt,)
X_np   = pack["X"]           # (Nx, Ny)
Y_np   = pack["Y"]           # (Nx, Ny)
params = pack["params"]
names  = pack["param_names"]

alpha_x = float(params[list(names).index("alpha_x")])
alpha_y = float(params[list(names).index("alpha_y")])
dx = float(params[list(names).index("dx")])
dy = float(params[list(names).index("dy")])
dt = float(params[list(names).index("dt")])

Nx, Ny, Nt = u_np.shape

# Flatten coordinates & values (mesh-free access)
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)   # (N,3)
vals_np    = u_np.reshape(-1)                                        # (N,)

coords = torch.from_numpy(coords_np).float().to(device)  # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)    # (N,)

x_t = torch.from_numpy(x_np).float().to(device)  # (Nx,)
y_t = torch.from_numpy(y_np).float().to(device)  # (Ny,)
t_t = torch.from_numpy(t_np).float().to(device)  # (Nt,)

# Domain extents (used for optional wrap deltas / BC loss)
Lx = float(x_np.max() - x_np.min())
Ly = float(y_np.max() - y_np.min())
Tt = float(t_np.max() - t_np.min()) if Nt > 1 else 1.0


# In[ ]:


# ----------------------
# Dataset that returns query indices (unchanged)
# ----------------------
class HeatPeriodicDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123):
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        rng = np.random.default_rng(seed)
        all_lin = np.arange(Nx*Ny*Nt)
        rng.shuffle(all_lin)
        n_total = len(all_lin)
        n_train = int(train_frac * n_total)
        n_val   = int(val_frac * n_total)
        self.train_idx = torch.from_numpy(all_lin[:n_train]).long()
        self.val_idx   = torch.from_numpy(all_lin[n_train:n_train+n_val]).long()
        self.test_idx  = torch.from_numpy(all_lin[n_train+n_val:]).long()
        self.split = "train"

    def set_split(self, split):
        assert split in ["train", "val", "test"]
        self.split = split

    def __len__(self):
        if self.split == "train": return len(self.train_idx)
        if self.split == "val":   return len(self.val_idx)
        if self.split == "test":  return len(self.test_idx)

    def __getitem__(self, idx):
        if self.split == "train": return self.train_idx[idx]
        if self.split == "val":   return self.val_idx[idx]
        return self.test_idx[idx]


# In[ ]:


# ----------------------
# Loaders (unchanged)
# ----------------------
ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds.set_split("train")
dl = DataLoader(ds, batch_size=8192, shuffle=True, drop_last=True)

ds_val = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=8192, shuffle=False)


# In[ ]:


# ----------------------
# Index <-> ijk helpers and periodic neighbor gather (unchanged)
# ----------------------

def lin_to_ijk(lin):
    i = lin // (Ny*Nt)
    r = lin %  (Ny*Nt)
    j = r // Nt
    k = r %  Nt
    return i, j, k

def ijk_to_lin(i, j, k):
    return (i % Nx) * (Ny*Nt) + (j % Ny) * Nt + (k % Nt)

def build_offset_table(k, gammas, dx, dy, dt, frac_time=0.5, max_dt_radius=None):
    """Return list of (di,dj,dk) with a quota on temporal neighbors.
       frac_time in [0,1]: fraction of k reserved for smallest |dk|."""
    gx, gy, gt = gammas
    # radius caps (optional)
    if max_dt_radius is None:
        max_dt_radius = int(3 * (1.0 / max(1e-6, gt)))  # heuristic

    # candidate pool
    rad = int(max(2, math.ceil((k**(1/3)) * 4)))
    cand = []
    for di in range(-rad, rad+1):
        for dj in range(-rad, rad+1):
            for dk in range(-max_dt_radius, max_dt_radius+1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                dxp = di * dx; dyp = dj * dy; dtp = dk * dt
                # soften time distance so edges don’t get starved
                dtp_eff = dtp / (1.0 + abs(dk) / 2.0)
                d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp_eff)**2
                cand.append((d2, di, dj, dk))
    cand.sort(key=lambda t: t[0])

    # split by |dk|
    n_time = max(1, int(k * frac_time))
    n_space = k - n_time
    cand_time  = sorted(cand, key=lambda t: abs(t[3]))  # prioritize small |dk|
    cand_space = sorted(cand, key=lambda t: (t[0], abs(t[3])), reverse=False)

    sel = []
    seen = set()
    # take time-quota
    for _, di, dj, dk in cand_time:
        key = (di, dj, dk)
        if key not in seen:
            sel.append(key); seen.add(key)
        if len(sel) >= n_time:
            break
    # fill with best remaining (mostly spatial)
    for _, di, dj, dk in cand_space:
        key = (di, dj, dk)
        if key not in seen:
            sel.append(key); seen.add(key)
        if len(sel) >= k:
            break
    return sel


# --- DROP-IN: axis-aware neighbor gather (no wrap in time) ---
def gather_neighbors_periodic(q_lin_idx, k, offsets_ijk, wrap_x=True, wrap_y=True, wrap_t=False):
    """Gather k neighbor indices using per-axis wrapping (x,y optional; t default False).
    Avoids the (i%Nx, j%Ny, k%Nt) periodicity for time by clamping instead."""
    i, j, k0 = lin_to_ijk(q_lin_idx)  # each is (B,)
    sel = offsets_ijk[:k]

    dev = q_lin_idx.device
    dtype = i.dtype
    di = torch.tensor([o[0] for o in sel], device=dev, dtype=dtype)
    dj = torch.tensor([o[1] for o in sel], device=dev, dtype=dtype)
    dk = torch.tensor([o[2] for o in sel], device=dev, dtype=dtype)

    I = i[:, None] + di[None, :]
    J = j[:, None] + dj[None, :]
    K = k0[:, None] + dk[None, :]

    if wrap_x:
        I = I % Nx
    else:
        I = I.clamp_(0, Nx - 1)

    if wrap_y:
        J = J % Ny
    else:
        J = J.clamp_(0, Ny - 1)

    if wrap_t:
        K = K % Nt
    else:
        K = K.clamp_(0, Nt - 1)

    # Inline ijk_to_lin to honor axis-specific wrapping/clamping we just applied
    nb_lin = I * (Ny * Nt) + J * Nt + K
    return nb_lin


# In[ ]:


# ----------------------
# FieldFormer (modified): include continuous relative deltas to query coords
# ----------------------
class FieldFormerAutograd(nn.Module):
    def __init__(self, d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128, wrap=True):
        super().__init__()
        self.k = k_neighbors
        self.wrap = wrap
        # in __init__ (or set once on the model)
        if wrap:
            self.wrap_x = True
            self.wrap_y = True
        else:
            self.wrap_x = False
            self.wrap_y = False
        self.wrap_t = False        # <-- time is NOT periodic

        self.register_buffer("_huber_delta_ema", torch.tensor(1.0, dtype=torch.float32))

        self.log_gammas = nn.Parameter(torch.zeros(3))  # learnable space-time scales
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    @staticmethod
    def _delta_periodic(a, b, L):
        # Smallest signed difference on a ring of length L (used only for wrapped axes)
        return (a - b + 0.5 * L) % L - 0.5 * L

    def _relative_deltas(self, q_xyz, nb_xyz):
        # q_xyz: (B,3), nb_xyz: (B,k,3)
        # x
        if self.wrap_x:
            dx_ = self._delta_periodic(nb_xyz[..., 0], q_xyz[:, None, 0], Lx)
        else:
            dx_ = nb_xyz[..., 0] - q_xyz[:, None, 0]
        # y
        if self.wrap_y:
            dy_ = self._delta_periodic(nb_xyz[..., 1], q_xyz[:, None, 1], Ly)
        else:
            dy_ = nb_xyz[..., 1] - q_xyz[:, None, 1]
        # t  (NO wrap)
        if self.wrap_t:
            dt_ = self._delta_periodic(nb_xyz[..., 2], q_xyz[:, None, 2], Tt)
        else:
            dt_ = nb_xyz[..., 2] - q_xyz[:, None, 2]

        rel = torch.stack([dx_, dy_, dt_], dim=-1)  # (B,k,3)
        scale = torch.exp(self.log_gammas)[None, None, :]  # (1,1,3)
        return rel * scale


    def forward(self, q_lin_idx, offsets_ijk):
        """Predict u at queries indexed by q_lin_idx (B,), using local neighbors.
        IMPORTANT CHANGE: tokens include continuous deltas to query coords, making
        output differentiable wrt (x,y,t) when we treat q coords as variables.
        """
        B = q_lin_idx.shape[0]
        # Neighbor indices (B,k)
        nb_idx = gather_neighbors_periodic(q_lin_idx, self.k, offsets_ijk)
        # Query & neighbor coordinates
        q_xyz  = coords[q_lin_idx]               # (B,3)
        nb_xyz = coords[nb_idx]                  # (B,k,3)
        # Relative deltas (wrap-aware) scaled by gammas
        rel = self._relative_deltas(q_xyz, nb_xyz)  # (B,k,3)
        # Neighbor values
        nb_vals = vals[nb_idx][..., None].to(torch.float32)  # (B,k,1)
        # Tokens: [rel_xyz, u_nb]
        tokens = torch.cat([rel, nb_vals], dim=-1)           # (B,k,4)
        h = self.input_proj(tokens)
        h = self.encoder(h)
        h_mean = h.mean(dim=1)
        out = self.head(h_mean).squeeze(-1)                  # (B,)
        return out


# In[ ]:


# ----------------------
# Forcing and utilities
# ----------------------

def forcing_torch(xx, yy, tt):
    return 5.0 * torch.cos(12*torch.pi * xx) * torch.cos(12*torch.pi * yy) * torch.sin(4 * torch.pi * tt / 5.0)

@torch.no_grad()
def batch_targets(q_lin_idx):
    return vals[q_lin_idx].to(device)

# Autograd PINN-style physics residual evaluated at grid queries
# (uses differentiable dependence on query coords via tokens

def huber(x, delta=1.0):
    return torch.where(x.abs() <= delta, 0.5*x*x, delta*(x.abs() - 0.5*delta))

def pde_residual_autograd(model, q_lin_idx):
    # Build a coordinate tensor tied to q_lin idx and enable grad on it
    xyt = coords[q_lin_idx].clone().detach().requires_grad_(True)  # (B,3)
    # We need predictions that depend on xyt. We temporarily override coords at q indices
    # by injecting xyt for the query part while neighbors remain fixed.
    # To achieve this without changing global coords, we compute tokens using xyt directly.

    # Gather neighbors
    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)
    nb_idx = gather_neighbors_periodic(q_lin_idx, model.k, offsets_ijk)
    nb_xyz = coords[nb_idx]  # (B,k,3) constants

    # Relative deltas built from the *variable* xyt
    wrap_space = model.wrap
    wrap_time = False  # <-- key change unless you truly have temporal periodicity

    if wrap_space:
        dxv = (nb_xyz[...,0] - xyt[:,None,0] + 0.5*Lx) % Lx - 0.5*Lx
        dyv = (nb_xyz[...,1] - xyt[:,None,1] + 0.5*Ly) % Ly - 0.5*Ly
    else:
        dxv = nb_xyz[...,0] - xyt[:,None,0]
        dyv = nb_xyz[...,1] - xyt[:,None,1]

    dtv = nb_xyz[...,2] - xyt[:,None,2]  # no wrap in time

    rel = torch.stack([dxv, dyv, dtv], dim=-1)
    scale = torch.exp(model.log_gammas)[None, None, :]
    rel = rel * scale  # (B,k,3)

    nb_vals_tok = vals[nb_idx][..., None].to(torch.float32)
    tokens = torch.cat([rel, nb_vals_tok], dim=-1)  # (B,k,4)

    # >>> Force math SDPA + disable AMP only for PDE path <<<
    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), \
         torch.cuda.amp.autocast(enabled=False):
        h = model.input_proj(tokens)
        h = model.encoder(h)
        h_mean = h.mean(dim=1)
        u = model.head(h_mean).squeeze(-1)

    ones = torch.ones_like(u)
    grads = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
    ux, uy, ut = grads[:,0], grads[:,1], grads[:,2]
    uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:,0]
    uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:,1]

    f = forcing_torch(xyt[:,0], xyt[:,1], xyt[:,2])
    R = ut - (alpha_x * uxx + alpha_y * uyy) - f
    return R


# In[ ]:


# ----------------------
# Early stopping (unchanged)
# ----------------------
@dataclass
class EarlyStopping:
    patience: int = 10
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False
    def step(self, metric: float):
        if metric < self.best - 1e-8:
            self.best = metric; self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True


# In[ ]:


# ----------------------
# Train setup (aligned with original)
# ----------------------

torch.set_float32_matmul_precision("high")
model = FieldFormerAutograd(d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128, wrap=True).to(device)
base_params = [p for n, p in model.named_parameters() if n != "log_gammas"]
optimizer = torch.optim.AdamW([
    {"params": base_params,        "lr": 3e-4, "weight_decay": 1e-4},
    {"params": [model.log_gammas], "lr": 3e-3, "weight_decay": 0.0},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
grad_clip = 1.0
max_epochs = 100
early = EarlyStopping(patience=10)
mse = nn.MSELoss()

USE_GRAD_NORM_BALANCE = True
BAL_CLAMP = (0.5, 2.0)  # narrower than before


# ---- Schedules ----
LAMBDA_PDE_MAX = 1.0          # tune (0.2–2.0 usually fine)
LAMBDA_BC_MAX  = 0.05         # keep BC small vs PDE
PDE_WARMUP_EPOCHS = 5
PDE_RAMP_EPOCHS   = 20        # length of cosine ramp after warmup

def cosine_ramp(t: float) -> float:
    # t in [0,1]
    return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, t))))

def lambdas_for_epoch(epoch: int):
    if epoch <= PDE_WARMUP_EPOCHS:
        lam_pde = 0.0
    else:
        prog = (epoch - PDE_WARMUP_EPOCHS) / max(1, PDE_RAMP_EPOCHS)
        lam_pde = LAMBDA_PDE_MAX * cosine_ramp(prog)
    lam_bc = min(LAMBDA_BC_MAX, 0.05 * max(lam_pde, 1e-6))
    return lam_pde, lam_bc

# lambda_pde = 0.1   # PINN residual weight
# lambda_bc  = 0.01  # optional periodic BC soft loss on function values
use_pde    = True
use_bc     = True
match_grad_bc = False

best_rmse = float("inf")
best_path = "ffag_heatsharp_best.pt"

@torch.no_grad()
def eval_val_rmse(offsets_ijk):
    se_sum = 0.0; n_sum = 0
    for q_lin in dl_val:
        q_lin = q_lin.to(device)
        pred = model(q_lin, offsets_ijk)
        tgt  = vals[q_lin]
        se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
        n_sum  += q_lin.numel()
    return math.sqrt(se_sum / max(1, n_sum))

# Periodic BC soft constraint (function equality on x and y boundaries)
def periodic_bc_loss(n_bc=1024, match_grad=False):
    j = torch.randint(0, Ny, (n_bc,), device=device)
    k = torch.randint(0, Nt, (n_bc,), device=device)
    yb, tb = y_t[j], t_t[k]
    x0, xL = x_t[0].expand_as(yb), x_t[-1].expand_as(yb)

    xyt0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
    xytL = torch.stack([xL, yb, tb], dim=-1).requires_grad_(match_grad)

    # Use model via nearest-grid queries for BC points
    # Map coords to nearest linear indices for neighbor gather
    def nearest_lin(xyt):
        # compute nearest grid indices
        ix = torch.clamp(((xyt[:,0] - x_t[0]) / dx).round().long(), 0, Nx-1)
        iy = torch.clamp(((xyt[:,1] - y_t[0]) / dy).round().long(), 0, Ny-1)
        it = torch.clamp(((xyt[:,2] - t_t[0]) / dt).round().long(), 0, Nt-1)
        return ijk_to_lin(ix, iy, it)

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    lin0 = nearest_lin(xyt0); linL = nearest_lin(xytL)

    # Build tokens using autograd xyt0/xytL as query coords (like in pde_residual_autograd)
    def predict_at_xyt(xyt, lin_idx):
        nb_idx = gather_neighbors_periodic(lin_idx, model.k, offsets_ijk)
        nb_xyz = coords[nb_idx]
        wrap_space = model.wrap
        wrap_time = False  # <-- key change unless you truly have temporal periodicity

        if wrap_space:
            dxv = (nb_xyz[...,0] - xyt[:,None,0] + 0.5*Lx) % Lx - 0.5*Lx
            dyv = (nb_xyz[...,1] - xyt[:,None,1] + 0.5*Ly) % Ly - 0.5*Ly
        else:
            dxv = nb_xyz[...,0] - xyt[:,None,0]
            dyv = nb_xyz[...,1] - xyt[:,None,1]

        dtv = nb_xyz[...,2] - xyt[:,None,2]  # no wrap in time

        rel = torch.stack([dxv, dyv, dtv], dim=-1) * torch.exp(model.log_gammas)[None,None,:]
        nb_vals_tok = vals[nb_idx][..., None].to(torch.float32)
        tokens = torch.cat([rel, nb_vals_tok], dim=-1)
        h = model.input_proj(tokens); h = model.encoder(h)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), \
         torch.cuda.amp.autocast(enabled=False):
            h = model.input_proj(tokens)
            h = model.encoder(h)
        return model.head(h.mean(dim=1)).squeeze(-1)

    u0 = predict_at_xyt(xyt0, lin0)
    uL = predict_at_xyt(xytL, linL)

    loss = F.mse_loss(u0, uL)
    if match_grad:
        gx0 = torch.autograd.grad(u0, xyt0, torch.ones_like(u0), create_graph=True)[0][:,0]
        gxL = torch.autograd.grad(uL, xytL, torch.ones_like(uL), create_graph=True)[0][:,0]
        loss = loss + F.mse_loss(gx0, gxL)

    # (Optionally mirror for y=0/Ly)
    return loss


# In[ ]:


# --------------------------
# Loading from a checkpoint
# --------------------------

def _move_optimizer_state_to_device(optimizer, device):
    """
    Ensure all tensors inside optimizer.state are moved to `device`.
    Call this right after `optimizer.load_state_dict(...)`.

    Args:
        optimizer (torch.optim.Optimizer): your optimizer
        device (torch.device or str): target device (e.g., torch.device("cuda"))
    """
    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        elif isinstance(x, list):
            return [ _to_device(y) for y in x ]
        elif isinstance(x, tuple):
            return tuple( _to_device(y) for y in x )
        elif isinstance(x, dict):
            return { k: _to_device(v) for k, v in x.items() }
        else:
            return x

    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in state.items():
                state[k] = _to_device(v)



def load_for_resume(path, model, optimizer=None, scaler=None, scheduler=None, device="cpu", strict=False):
    # PyTorch 2.6 changed default to weights_only=True; we want the old behavior here.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Accept both formats: {"model_state_dict": ...} or a bare state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # assume it's a raw state_dict

    # 1) Model
    msg = model.load_state_dict(state_dict, strict=strict)
    try:
        missing = getattr(msg, "missing_keys", [])
        unexpected = getattr(msg, "unexpected_keys", [])
        if missing or unexpected:
            print(f"[resume] model.load_state_dict: missing={missing}, unexpected={unexpected}")
    except Exception:
        pass

    # 2) Optimizer / Scaler / Scheduler (best ckpt likely doesn't have these; skip if absent)
    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, torch.device(device))
    if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            print(f"[resume] scaler load failed ({e}), reinitializing GradScaler.")
    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[resume] scheduler load failed ({e}), continuing without resume.")

    # 3) Book-keeping (only available if full ckpt dict was saved)
    start_epoch = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 1
    best_val_rmse = float(ckpt.get("best_val_rmse", float("inf"))) if isinstance(ckpt, dict) else float("inf")

    return start_epoch, best_val_rmse



LOAD_BEFORE_TRAIN = False
RESUME_PATH = '/scratch/ab9738/fieldformer/model/ffag_heatsharp_best.pt'
if LOAD_BEFORE_TRAIN:
    print(f"[resume] loading from {RESUME_PATH}")
    start_epoch, best_rmse = load_for_resume(
        RESUME_PATH, model, optimizer, scaler, scheduler, device=device, strict=False
    )


# In[ ]:


# ----------------------
# Training loop
# ----------------------

for epoch in range(1, max_epochs+1):
    model.train()
    total_loss = total_data = total_pde = total_bc = 0.0
    lam_pde, lam_bc = lambdas_for_epoch(epoch)

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
    offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    GAM_FZ_EPOCHS = 1  # try 5–10
    if epoch == 1:
        model.log_gammas.requires_grad_(False)
    elif epoch == GAM_FZ_EPOCHS + 1:
        model.log_gammas.requires_grad_(True)

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Data loss at grid queries
            pred = model(q_lin, offsets_ijk)
            tgt  = vals[q_lin]
            data_loss = mse(pred, tgt)

            loss = data_loss
            pde_loss = torch.tensor(0.0, device=device)
            bc_loss  = torch.tensor(0.0, device=device)

            if use_pde:
                subsample = q_lin[::8]
                R = pde_residual_autograd(model, subsample)
                # pde_loss = (R**2).mean()
                pde_loss = huber(R).mean()
                if not hasattr(model, "_huber_delta_ema"):
                    model._huber_delta_ema = torch.as_tensor(1.0, device=device)
                with torch.no_grad():
                    batch_med = R.detach().abs().median()
                    model._huber_delta_ema.mul_(0.98).add_(0.02 * batch_med)
                delta = model._huber_delta_ema.clamp(1e-3, 10.0)

                # loss = loss + lam_pde * pde_loss

            if use_bc and model.wrap:
                bc_loss = periodic_bc_loss(n_bc=512, match_grad=match_grad_bc)
                # loss = loss + lam_bc * bc_loss

            if USE_GRAD_NORM_BALANCE and use_pde and pde_loss.requires_grad:
                # compute grad norms w.r.t. base (non-geometry) params
                g_data = torch.autograd.grad(data_loss, base_params, retain_graph=True, allow_unused=True)
                gn_data = torch.sqrt(sum([(g.detach()**2).sum() for g in g_data if g is not None]) + 1e-12)

                g_phys = torch.autograd.grad(pde_loss,  base_params, retain_graph=True, allow_unused=True)
                gn_phys = torch.sqrt(sum([(g.detach()**2).sum() for g in g_phys if g is not None]) + 1e-12)

                bal = (gn_data / (gn_phys + 1e-12)).clamp(*BAL_CLAMP)
                lam_pde_eff = lam_pde * bal
            else:
                lam_pde_eff = lam_pde

            loss = data_loss + lam_pde_eff * pde_loss + lam_bc * bc_loss

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optional extra guard: ensure all grads are finite before stepping
        grads_finite = True
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_finite = False
                break

        if not grads_finite:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # back off scale even though we skipped the step
            continue

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_data += data_loss.item()
        total_pde  += pde_loss.item()
        total_bc   += bc_loss.item()

    # Validation
    model.eval()
    rmse = eval_val_rmse(offsets_ijk)
    scheduler.step(rmse)

    print(f"Epoch {epoch:03d} | train {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, pde {total_pde/len(dl):.4f}, bc {total_bc/len(dl):.4f}) | "
          f"val RMSE {rmse:.5f}")

    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "gammas": model.log_gammas.detach().exp().cpu().numpy(),
            "config": {
                "variant": "fieldformer_autograd",
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "k_neighbors": model.k,
                "d_model": 64, "nhead": 4, "num_layers": 2,
                "lambda_pde": lam_pde, "lambda_bc": lam_bc,
                "dx": dx, "dy": dy, "dt": dt, "alpha_x": alpha_x, "alpha_y": alpha_y,
                "wrap": model.wrap,
            }
        }, best_path)
        print(f"✓ Saved new best to {best_path} (val RMSE {best_rmse:.6f})")

    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
        break

print("Done.")


# In[ ]:




