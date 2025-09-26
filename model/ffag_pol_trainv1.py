#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8
# ===== FieldFormer_Autograd for Pollution (Advection–Diffusion, Open BCs) =====
# Minimal diffs from ffag_heat_train.py:
#   • wrap_x = wrap_y = False (open domain), wrap_t = False
#   • Boundary losses: sponge (absorbing rim) + radiation (Orlanski-style, STABLE)
#   • Loads vx, vy, S fields from dataset if present; else treats as zeros
#   • This version splices in a scale-stable radiation BC loss with γ-detach & Huber
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import math
from torch.backends.cuda import sdp_kernel
from contextlib import nullcontext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# Data: pollution dataset
# Expected keys (at minimum): u (Nx,Ny,Nt), x (Nx,), y (Ny,), t (Nt,)
# Optional: vx, vy, S as either (Nx,Ny,Nt) or (Nt,Nx,Ny)
# ----------------------
DATA_PATH = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"  # <-- update if needed
pack = np.load(DATA_PATH)

u_np   = pack["U"]              # (Nx, Ny, Nt)
x_np   = pack["x"]              # (Nx,)
y_np   = pack["y"]              # (Ny,)
t_np   = pack["t"]              # (Nt,)
Nx, Ny, Nt = u_np.shape

# Flatten for mesh-free access (match heat trainer’s style)
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)  # (N,3)
vals_np    = u_np.reshape(-1)                                       # (N,)

coords = torch.from_numpy(coords_np).float().to(device)             # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)               # (N,)
x_t    = torch.from_numpy(x_np).float().to(device)
y_t    = torch.from_numpy(y_np).float().to(device)
t_t    = torch.from_numpy(t_np).float().to(device)

# Spans
dx = float(x_np[1] - x_np[0]) if Nx > 1 else 1.0
dy = float(y_np[1] - y_np[0]) if Ny > 1 else 1.0
dt = float(t_np[1] - t_np[0]) if Nt > 1 else 1.0
Lx = float(x_np.max() - x_np.min()) if Nx > 1 else 1.0
Ly = float(y_np.max() - y_np.min()) if Ny > 1 else 1.0
Tt = float(t_np.max() - t_np.min()) if Nt > 1 else 1.0


# ----------------------
# Dataset / Loaders
# ----------------------
class PollutionDataset(Dataset):
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
        assert split in ["train","val","test"]; self.split = split
    def __len__(self):
        return len(getattr(self, f"{self.split}_idx"))
    def __getitem__(self, idx):
        return getattr(self, f"{self.split}_idx")[idx]

ds     = PollutionDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val = PollutionDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
dl     = DataLoader(ds,     batch_size=8192, shuffle=True,  drop_last=True)
ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=8192, shuffle=False)


# ----------------------
# Index helpers & neighbor gather with per-axis wrapping
# (Open in x,y => clamp; time never wraps)
# ----------------------
def lin_to_ijk(lin):
    i = lin // (Ny*Nt)
    r = lin %  (Ny*Nt)
    j = r // Nt
    k = r %  Nt
    return i, j, k

def ijk_to_lin(i, j, k):
    return (i.clamp(0, Nx-1)) * (Ny*Nt) + (j.clamp(0, Ny-1)) * Nt + (k.clamp(0, Nt-1))

def build_offset_table(k, gammas, dx, dy, dt, frac_time=0.5, max_dt_radius=None):
    gx, gy, gt = gammas
    if max_dt_radius is None:
        max_dt_radius = int(3 * (1.0 / max(1e-6, gt)))
    rad = int(max(2, math.ceil((k**(1/3)) * 4)))
    cand = []
    for di in range(-rad, rad+1):
        for dj in range(-rad, rad+1):
            for dk in range(-max_dt_radius, max_dt_radius+1):
                if di == 0 and dj == 0 and dk == 0: continue
                dxp = di*dx; dyp = dj*dy; dtp = dk*dt
                dtp_eff = dtp / (1.0 + abs(dk)/2.0)
                d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp_eff)**2
                cand.append((d2, di, dj, dk))
    cand.sort(key=lambda t: t[0])
    n_time = max(1, int(k * frac_time))
    sel, seen = [], set()
    # prioritize small |dk|
    for _, di, dj, dk in sorted(cand, key=lambda t: abs(t[3])):
        key = (di,dj,dk)
        if key not in seen:
            sel.append(key); seen.add(key)
        if len(sel) >= n_time: break
    for _, di, dj, dk in cand:
        key = (di,dj,dk)
        if key not in seen:
            sel.append(key); seen.add(key)
        if len(sel) >= k: break
    return sel

def gather_neighbors(q_lin_idx, k, offsets_ijk, wrap_x=False, wrap_y=False, wrap_t=False):
    i, j, k0 = lin_to_ijk(q_lin_idx)
    sel = offsets_ijk[:k]
    dev, dtype = q_lin_idx.device, i.dtype
    di = torch.tensor([o[0] for o in sel], device=dev, dtype=dtype)
    dj = torch.tensor([o[1] for o in sel], device=dev, dtype=dtype)
    dk = torch.tensor([o[2] for o in sel], device=dev, dtype=dtype)
    I = i[:,None] + di[None,:]
    J = j[:,None] + dj[None,:]
    K = k0[:,None] + dk[None,:]
    I = I % Nx if wrap_x else I.clamp_(0, Nx-1)
    J = J % Ny if wrap_y else J.clamp_(0, Ny-1)
    K = K % Nt if wrap_t else K.clamp_(0, Nt-1)
    return I*(Ny*Nt) + J*Nt + K


# ----------------------
# Model: FieldFormerAutograd (same as heat, but wrap_x/y=False by default)
# Tokens: [dx,dy,dt, u_nb]; dx,dy computed with or without wrap
# ----------------------
class FieldFormerAutograd(nn.Module):
    def __init__(self, d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128,
                 wrap_x=False, wrap_y=False, wrap_t=False):
        super().__init__()
        self.k = k_neighbors
        self.wrap_x, self.wrap_y, self.wrap_t = wrap_x, wrap_y, wrap_t
        self.log_gammas = nn.Parameter(torch.zeros(3))  # space-time scales
        self.input_proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model,1))

    @staticmethod
    def _delta_periodic(a, b, L):
        return (a - b + 0.5*L) % L - 0.5*L

    def _relative_deltas(self, q_xyz, nb_xyz):
        if self.wrap_x: dxv = self._delta_periodic(nb_xyz[...,0], q_xyz[:,None,0], Lx)
        else:           dxv = nb_xyz[...,0] - q_xyz[:,None,0]
        if self.wrap_y: dyv = self._delta_periodic(nb_xyz[...,1], q_xyz[:,None,1], Ly)
        else:           dyv = nb_xyz[...,1] - q_xyz[:,None,1]
        if self.wrap_t: dtv = self._delta_periodic(nb_xyz[...,2], q_xyz[:,None,2], Tt)
        else:           dtv = nb_xyz[...,2] - q_xyz[:,None,2]
        rel = torch.stack([dxv, dyv, dtv], dim=-1)
        return rel * torch.exp(self.log_gammas)[None,None,:]

    def forward(self, q_lin_idx, offsets_ijk):
        nb_idx = gather_neighbors(q_lin_idx, self.k, offsets_ijk, self.wrap_x, self.wrap_y, self.wrap_t)
        q_xyz  = coords[q_lin_idx]
        nb_xyz = coords[nb_idx]
        rel    = self._relative_deltas(q_xyz, nb_xyz)
        nb_vals = vals[nb_idx][...,None].to(torch.float32)
        tokens = torch.cat([rel, nb_vals], dim=-1)  # (B,k,4)
        h = self.input_proj(tokens)
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


# ----------------------
# Helpers
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

@torch.no_grad()
def eval_val_rmse(model, offsets_ijk):
    se_sum = 0.0; n_sum = 0
    for q_lin in dl_val:
        q_lin = q_lin.to(device)
        pred = model(q_lin, offsets_ijk)
        tgt  = vals[q_lin]
        se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
        n_sum  += q_lin.numel()
    return math.sqrt(se_sum / max(1, n_sum))

def huber(x, delta=1.0):
    return torch.where(x.abs() <= delta, 0.5*x*x, delta*(x.abs() - 0.5*delta))


# ----------------------
# Open-BC losses
# (1) Sponge rim: penalize deviation from background near edges
# (2) Radiation (Orlanski-style): ut + c_eff * ∂u/∂n ~ 0 on outflow edges (STABLE)
# ----------------------

def sponge_loss(u_pred, q_lin, border_frac=0.05, u_bg=0.0, power=2):
    xyt = coords[q_lin]
    x01 = (xyt[:,0] - x_np.min()) / max(1e-12, Lx)
    y01 = (xyt[:,1] - y_np.min()) / max(1e-12, Ly)
    d_edge = torch.minimum(torch.minimum(x01, 1-x01), torch.minimum(y01, 1-y01))
    mask = (d_edge < border_frac).float()
    ramp = ((border_frac - d_edge).clamp(min=0) / border_frac)**power
    w = mask * ramp
    return (w * (u_pred - u_bg)**2).mean()


def radiation_bc_loss_no_v_on01(
    model,
    coords01,          # (N,3) with x,y,t already in [0,1] (pollution data uses this)
    vals,              # (N,)
    q_lin_edge,        # (B,) linear indices for edge samples
    dx, dy, dt,        # spacings used by neighbor table
    device,
    build_offset_table,
    gather_neighbors,
    sdp_kernel=nullcontext,
    c_cap=2.0,         # cap on c_eff in normalized units (domain-lengths / time)
    huber_delta=1.0,
    eps=1e-6,
):
    """Stable radiation BC (Orlanski) on unit box.
    Key stability features:
      • grads wrt normalized coords; 
      • detach γ in neighbor scaling; 
      • safe ratio for c_eff with clamp+detach; 
      • Huber on normalized residual.
    Model is trained directly on physical-scale targets, so we don't (de)standardize here.
    """
    if q_lin_edge is None or q_lin_edge.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    xyz01 = coords01.to(device)
    v = vals.to(device).to(torch.float32)

    # Boundary query points with grad
    xyt01 = xyz01[q_lin_edge].clone().detach().requires_grad_(True)  # (B,3)

    # Neighbors using DETACHED gammas for stable geometry scaling
    with torch.no_grad():
        gam_det = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk = build_offset_table(k=model.k, gammas=gam_det, dx=dx, dy=dy, dt=dt)
        nb_idx = gather_neighbors(q_lin_edge, model.k, offsets_ijk, model.wrap_x, model.wrap_y, model.wrap_t)
        nb_xyz01 = xyz01[nb_idx]                                        # (B,k,3)

    # Relative positions in [0,1], then scale by DETACHED gammas
    drel = nb_xyz01 - xyt01[:, None, :]                                 # (B,k,3)
    rel  = drel * torch.exp(model.log_gammas).detach()[None, None, :]   # (B,k,3)
    nb_vals_tok = v[nb_idx][..., None]                                  # (B,k,1)

    # Forward (disable AMP here for stable autograd of grads)
    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), \
         torch.cuda.amp.autocast(enabled=False):
        h = model.input_proj(torch.cat([rel, nb_vals_tok], dim=-1))
        h = model.encoder(h)
        u = model.head(h.mean(dim=1)).squeeze(-1)        # (B,) in physical scale (matches training)

    # Grads wrt normalized coords (domain is unit-length in each dim)
    grads01 = torch.autograd.grad(u, xyt01, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ux, uy, ut = grads01[:, 0], grads01[:, 1], grads01[:, 2]

    # Outward normals on the unit box (x,y only)
    left   = (xyt01[:, 0] <= eps).float()
    right  = (xyt01[:, 0] >= 1.0 - eps).float()
    bottom = (xyt01[:, 1] <= eps).float()
    top    = (xyt01[:, 1] >= 1.0 - eps).float()
    un = left*(-ux) + right*(ux) + bottom*(-uy) + top*(uy)    # ∂u/∂n

    # c_eff = -ut / un  (safe, bounded, and DETACHED)
    denom = un.abs().clamp(min=1e-6)
    c_eff = (-ut / denom).clamp(0.0, c_cap).detach()

    # Radiation residual and normalized robust penalty
    rad_res = ut + c_eff * un
    scale = (torch.sqrt(ut.pow(2) + un.pow(2)) + 1e-3).detach()
    r = rad_res / scale
    return huber(r, delta=huber_delta).mean()


# ----------------------
# Train setup
# ----------------------
torch.set_float32_matmul_precision("high")
model = FieldFormerAutograd(
    d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128,
    wrap_x=False, wrap_y=False, wrap_t=False   # OPEN boundaries
).to(device)

base_params = [p for n, p in model.named_parameters() if n != "log_gammas"]
optimizer = torch.optim.AdamW(
    [{"params": base_params, "lr": 3e-4, "weight_decay": 1e-4},
     {"params": [model.log_gammas], "lr": 1e-3, "weight_decay": 0.0}]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                       factor=0.5, patience=3, min_lr=1e-6)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
mse = nn.MSELoss()
grad_clip = 0.5
max_epochs = 100
early = EarlyStopping(patience=10)

# Loss weights (ramp PDE; keep BC small)
# before training loop
RAD_WARMUP=5; RAD_RAMP=20
SP_WARMUP =0; SP_RAMP =10
RAD_MAX=0.01   # lower than before
SP_MAX =0.03

def bc_lambdas(epoch):
    def ramp(ep, warm, span, maxv):
        if ep <= warm: return 0.0
        z = min(1.0, (ep-warm)/max(1,span))
        return maxv * 0.5*(1 - math.cos(math.pi*z))
    return ramp(epoch, SP_WARMUP, SP_RAMP, SP_MAX), ramp(epoch, RAD_WARMUP, RAD_RAMP, RAD_MAX)

# Optional: resume

def _move_optimizer_state_to_device(optimizer, device):
    def _to_device(x):
        if isinstance(x, torch.Tensor): return x.to(device, non_blocking=True)
        if isinstance(x, list): return [ _to_device(y) for y in x ]
        if isinstance(x, tuple): return tuple(_to_device(y) for y in x)
        if isinstance(x, dict): return {k:_to_device(v) for k,v in x.items()}
        return x
    for st in optimizer.state.values():
        if isinstance(st, dict):
            for k,v in st.items(): st[k] = _to_device(v)

def load_for_resume(path, model, optimizer=None, scaler=None, scheduler=None, device="cpu", strict=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    msg = model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]); _move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try: scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception: pass
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception: pass
    start_epoch = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 1
    best_val_rmse = float(ckpt.get("best_val_rmse", float("inf"))) if isinstance(ckpt, dict) else float("inf")
    return start_epoch, best_val_rmse

LOAD_BEFORE_TRAIN = False
RESUME_PATH = "/scratch/ab9738/fieldformer/model/ffag_pol_bestv1.pt"
best_rmse = float("inf")
if LOAD_BEFORE_TRAIN:
    print(f"[resume] loading from {RESUME_PATH}")
    start_epoch, best_rmse = load_for_resume(RESUME_PATH, model, optimizer, scaler, scheduler, device=device, strict=False)


# ----------------------
# Training loop
# ----------------------
for epoch in range(1, max_epochs+1):
    model.train()
    lam_sp, lam_rad = bc_lambdas(epoch)
    total_loss = total_data = total_sp = total_rad = 0.0

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
    offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    # optional: freeze gammas in first few epochs
    if epoch <= 6:
        model.log_gammas.requires_grad_(False)
    else:
        model.log_gammas.requires_grad_(True)

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(q_lin, offsets_ijk)
            tgt  = vals[q_lin]
            data_loss = F.mse_loss(pred, tgt)

            # Edge sampling for BCs: exact-boundary points
            i, j, k0 = lin_to_ijk(q_lin)
            edge_mask = (i == 0) | (i == Nx-1) | (j == 0) | (j == Ny-1)
            q_edge = q_lin[edge_mask]

            sp_loss  = sponge_loss(pred, q_lin, border_frac=0.05, u_bg=0.0, power=2)
            rad_loss = radiation_bc_loss_no_v_on01(
                model=model,
                coords01=coords,
                vals=vals,
                q_lin_edge=q_edge,
                dx=dx, dy=dy, dt=dt,
                device=device,
                build_offset_table=build_offset_table,
                gather_neighbors=gather_neighbors,
                sdp_kernel=sdp_kernel,
                c_cap=2.0,
                huber_delta=1.0,
            ) if q_edge.numel() else torch.tensor(0.0, device=device)

            loss = data_loss + lam_sp*sp_loss + lam_rad*rad_loss

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()

        total_loss += loss.item()
        total_data += data_loss.item()
        total_sp   += float(sp_loss)
        total_rad  += float(rad_loss)

    # Validation
    model.eval()
    rmse = eval_val_rmse(model, offsets_ijk)
    scheduler.step(rmse)

    print(f"Epoch {epoch:03d} | train {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, "
          f"sponge {total_sp/len(dl):.4f}, rad {total_rad/len(dl):.4f}) | "
          f"val RMSE {rmse:.5f}")

    if rmse < best_rmse:
        best_rmse = rmse
        out_path = "/scratch/ab9738/fieldformer/model/ffag_pol_bestv1.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "gammas": model.log_gammas.detach().exp().cpu().numpy(),
            "config": {
                "variant": "fieldformer_autograd_pollution",
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "k_neighbors": model.k,
                "d_model": 64, "nhead": 4, "num_layers": 2,
                "dx": dx, "dy": dy, "dt": dt,
                # NOTE: removed undefined 'kappa' key which caused NameError on save
                "wrap_x": model.wrap_x, "wrap_y": model.wrap_y, "wrap_t": model.wrap_t,
                "lambda_sp": float(lam_sp), "lambda_rad": float(lam_rad)
            }
        }, out_path)
        print(f"✓ Saved new best to {out_path} (val RMSE {best_rmse:.6f})")

    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
        break

print("Done.")
