#!/usr/bin/env python
# coding: utf-8

# In[12]:


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


# In[2]:


# ----------------------
# Data: load the periodic heat dataset (same path as original)
# ----------------------
pack = np.load("/scratch/ab9738/fieldformer/data/heat_periodic_dataset.npz")
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


# In[3]:


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


# In[4]:


# ----------------------
# Loaders (unchanged)
# ----------------------
ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds.set_split("train")
dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)

ds_val = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)


# In[5]:


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

# Pre-sorted offsets table (same as original)
def build_offset_table(k, gammas, dx, dy, dt, max_rad=None):
    gx, gy, gt = [float(x) for x in gammas]
    c = 4.0
    base = (c * k) ** (1/3)
    Rx = max(1, int(base * (1.0/max(gx,1e-8))))
    Ry = max(1, int(base * (1.0/max(gy,1e-8))))
    Rt = max(1, int(base * (1.0/max(gt,1e-8))))
    if max_rad is not None:
        Rx = min(Rx, max_rad); Ry = min(Ry, max_rad); Rt = min(Rt, max_rad)
    offs = []
    for di in range(-Rx, Rx+1):
        for dj in range(-Ry, Ry+1):
            for dk in range(-Rt, Rt+1):
                dxp = di * dx; dyp = dj * dy; dtp = dk * dt
                d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp)**2
                offs.append((d2, di, dj, dk))
    offs.sort(key=lambda z: z[0])
    offs = [(di,dj,dk) for (d2,di,dj,dk) in offs if not (di==0 and dj==0 and dk==0)]
    return offs

def gather_neighbors_periodic(q_lin_idx, k, offsets_ijk):
    i, j, k0 = lin_to_ijk(q_lin_idx)
    sel = offsets_ijk[:k]
    di = torch.tensor([o[0] for o in sel], device=q_lin_idx.device, dtype=i.dtype)
    dj = torch.tensor([o[1] for o in sel], device=q_lin_idx.device, dtype=i.dtype)
    dk = torch.tensor([o[2] for o in sel], device=q_lin_idx.device, dtype=i.dtype)
    I = i[:, None] + di[None, :]
    J = j[:, None] + dj[None, :]
    K = k0[:, None] + dk[None, :]
    nb_lin = ijk_to_lin(I, J, K)
    return nb_lin


# In[6]:


# ----------------------
# FieldFormer (modified): include continuous relative deltas to query coords
# ----------------------
class FieldFormerAutograd(nn.Module):
    def __init__(self, d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128, wrap=True):
        super().__init__()
        self.k = k_neighbors
        self.wrap = wrap
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
        # smallest signed difference on a ring of length L
        return (a - b + 0.5*L) % L - 0.5*L

    def _relative_deltas(self, q_xyz, nb_xyz):
        # q_xyz: (B,3), nb_xyz: (B,k,3)
        if self.wrap:
            dx_ = self._delta_periodic(nb_xyz[...,0], q_xyz[:,None,0], Lx)
            dy_ = self._delta_periodic(nb_xyz[...,1], q_xyz[:,None,1], Ly)
            dt_ = self._delta_periodic(nb_xyz[...,2], q_xyz[:,None,2], Tt)
        else:
            dx_ = nb_xyz[...,0] - q_xyz[:,None,0]
            dy_ = nb_xyz[...,1] - q_xyz[:,None,1]
            dt_ = nb_xyz[...,2] - q_xyz[:,None,2]
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


# In[13]:


# ----------------------
# Forcing and utilities
# ----------------------

def forcing_torch(xx, yy, tt):
    return 5.0 * torch.cos(torch.pi * xx) * torch.cos(torch.pi * yy) * torch.sin(4 * torch.pi * tt / 5.0)

@torch.no_grad()
def batch_targets(q_lin_idx):
    return vals[q_lin_idx].to(device)

# Autograd PINN-style physics residual evaluated at grid queries
# (uses differentiable dependence on query coords via tokens)

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
    if model.wrap:
        dxv = (nb_xyz[...,0] - xyt[:,None,0] + 0.5*Lx) % Lx - 0.5*Lx
        dyv = (nb_xyz[...,1] - xyt[:,None,1] + 0.5*Ly) % Ly - 0.5*Ly
        dtv = (nb_xyz[...,2] - xyt[:,None,2] + 0.5*Tt) % Tt - 0.5*Tt
    else:
        dxv = nb_xyz[...,0] - xyt[:,None,0]
        dyv = nb_xyz[...,1] - xyt[:,None,1]
        dtv = nb_xyz[...,2] - xyt[:,None,2]
    rel = torch.stack([dxv, dyv, dtv], dim=-1)
    scale = torch.exp(model.log_gammas)[None, None, :]
    rel = rel * scale  # (B,k,3)

    nb_vals_tok = vals[nb_idx][..., None].to(torch.float32)
    tokens = torch.cat([rel, nb_vals_tok], dim=-1)  # (B,k,4)

    h = model.input_proj(tokens)
    h = model.encoder(h)
    h_mean = h.mean(dim=1)
    u = model.head(h_mean).squeeze(-1)             # (B,)

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


# In[8]:


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


# In[14]:


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

lambda_pde = 0.1   # PINN residual weight
lambda_bc  = 0.01  # optional periodic BC soft loss on function values
use_pde    = True
use_bc     = True
match_grad_bc = False

best_rmse = float("inf")
best_path = "ff_ag_heat_best.pt"

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
        if model.wrap:
            dxv = (nb_xyz[...,0] - xyt[:,None,0] + 0.5*Lx) % Lx - 0.5*Lx
            dyv = (nb_xyz[...,1] - xyt[:,None,1] + 0.5*Ly) % Ly - 0.5*Ly
            dtv = (nb_xyz[...,2] - xyt[:,None,2] + 0.5*Tt) % Tt - 0.5*Tt
        else:
            dxv = nb_xyz[...,0] - xyt[:,None,0]
            dyv = nb_xyz[...,1] - xyt[:,None,1]
            dtv = nb_xyz[...,2] - xyt[:,None,2]
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


# In[15]:


# ----------------------
# Training loop
# ----------------------

for epoch in range(1, max_epochs+1):
    model.train()
    total_loss = total_data = total_pde = total_bc = 0.0

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
    offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    # ramp physics/bc weights for stability
    ramp = min(1.0, epoch / 20.0)
    lam_pde = lambda_pde * ramp
    lam_bc  = lambda_bc  * ramp

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
                pde_loss = (R**2).mean()
                loss = loss + lam_pde * pde_loss

            if use_bc and model.wrap:
                bc_loss = periodic_bc_loss(n_bc=512, match_grad=match_grad_bc)
                loss = loss + lam_bc * bc_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                "lambda_pde": lambda_pde, "lambda_bc": lambda_bc,
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




