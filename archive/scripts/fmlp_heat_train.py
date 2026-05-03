#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

# ===== Fourier-MLP neural field with physics loss (periodic BC) =====
# Reuses the same dataset (.npz), tensors, and loaders pattern as ff_heat_train.py
# ====================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2
from dataclasses import dataclass


# In[2]:


# ----------------------
# Data: load the periodic heat dataset (same path as your FieldFormer script)
# ----------------------
pack = np.load("/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz")
u_np   = pack["u"]           # (Nx, Ny, Nt)
x_np   = pack["x"]           # (Nx,)
y_np   = pack["y"]           # (Ny,)
t_np   = pack["t"]           # (Nt,)
X_np   = pack["X"]           # (Nx, Ny)
Y_np   = pack["Y"]           # (Nx, Ny)
params = pack["params"]
names  = pack["param_names"]

# PDE / grid scalars
alpha_x = float(params[list(names).index("alpha_x")])
alpha_y = float(params[list(names).index("alpha_y")])
dx = float(params[list(names).index("dx")])
dy = float(params[list(names).index("dy")])
dt = float(params[list(names).index("dt")])

Nx, Ny, Nt = u_np.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flatten coordinates & values (identical pattern to ff_heat_train)
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)   # (N,3) in [x,y,t]
vals_np    = u_np.reshape(-1)                                         # (N,)

coords = torch.from_numpy(coords_np).float().to(device)  # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)    # (N,)

x_t = torch.from_numpy(x_np).float().to(device)  # (Nx,)
y_t = torch.from_numpy(y_np).float().to(device)  # (Ny,)
t_t = torch.from_numpy(t_np).float().to(device)  # (Nt,)

# Domain extents for periodicity (used in encodings & BC sampling)
Lx = float(x_np.max() - x_np.min())
Ly = float(y_np.max() - y_np.min())
Tt = float(t_np.max() - t_np.min()) if Nt > 1 else 1.0  # robust if Nt==1


# In[3]:


# ----------------------
# Dataset that returns query indices (same as ff_heat_train)
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
# Instantiate loaders (mirrors your script)
# ----------------------
ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds.set_split("train")
dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)

ds_val = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)


# In[5]:


# ----------------------
# Fourier features (integer harmonics; periodic-friendly but not hard constraints)
# ----------------------
def build_harmonics(K):
    # K can be an int (1..K) or a list of ints
    if isinstance(K, int):
        return torch.arange(1, K+1, dtype=torch.float32, device=device)
    return torch.tensor(K, dtype=torch.float32, device=device)

Kx_list = build_harmonics(16)   # tune
Ky_list = build_harmonics(16)   # tune
Kt_list = build_harmonics(8)    # tune

def fourier_encode_1d(x, Ks, L):
    """
    x: (B,) in physical units
    Returns: (B, 2*len(Ks)) [sin(2pi k x/L), cos(...)]
    """
    # Normalize to [0, L], then map to angles
    z = (2 * math.pi) * (x[..., None] / (L if L > 0 else 1.0)) * Ks[None, :]
    return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

def fourier_encode_3d(xyt):
    """
    xyt: (B,3) in physical units [x,y,t]
    returns: (B, 2*(Kx+Ky+Kt))
    """
    x, y, t = xyt[:,0], xyt[:,1], xyt[:,2]
    fx = fourier_encode_1d(x, Kx_list, Lx)
    fy = fourier_encode_1d(y, Ky_list, Ly)
    ft = fourier_encode_1d(t, Kt_list, Tt)
    return torch.cat([fx, fy, ft], dim=-1)


# In[6]:


# ----------------------
# Fourier-MLP model
# ----------------------
class FourierMLP(nn.Module):
    def __init__(self, width=256, depth=6, out_dim=1):
        super().__init__()
        in_dim = 2*(len(Kx_list)+len(Ky_list)+len(Kt_list))
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth-2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # Init: fan-in for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, xyt):
        # xyt: (B,3) (float, physical units)
        feat = fourier_encode_3d(xyt)
        return self.net(feat).squeeze(-1)  # (B,)

model = FourierMLP(width=256, depth=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
mse = nn.MSELoss()


# In[7]:


# ----------------------
# Forcing function (same as your FieldFormer script)
# ----------------------
def forcing_torch(xx, yy, tt):
    return 5.0 * torch.cos(12*torch.pi * xx) * torch.cos(12*torch.pi * yy) * torch.sin(4 * torch.pi * tt / 5.0)


# In[8]:


# ----------------------
# Utilities
# ----------------------
def lin_to_coords(q_lin):
    # q_lin: (B,) long -> (B,3) coords in physical units
    return coords[q_lin]  # already on device

def batch_targets(q_lin):
    return vals[q_lin].to(device)

# Physics residual: autograd PINN-style (no finite-diff needed)
def pde_residual_autograd(xyt):
    """
    Heat PDE residual: R = u_t - alpha_x u_xx - alpha_y u_yy - f(x,y,t)
    xyt: (B,3) requires_grad True
    """
    xyt = xyt.requires_grad_(True)
    u = model(xyt)  # (B,)
    ones = torch.ones_like(u)

    # First derivatives
    grads = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]  # (B,3)
    ux, uy, ut = grads[:,0], grads[:,1], grads[:,2]

    # Second derivatives
    uxx = torch.autograd.grad(ux, xyt, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:,0]
    uyy = torch.autograd.grad(uy, xyt, grad_outputs=torch.ones_like(uy), create_graph=True)[0][:,1]

    f = forcing_torch(xyt[:,0], xyt[:,1], xyt[:,2])
    R = ut - (alpha_x * uxx + alpha_y * uyy) - f
    return R

# Periodic BC loss on x=0/Lx and y=0/Ly (function equality; optional gradient equality)
def periodic_bc_loss(n_bc=1024, match_grad=False):
    """
    Sample random (y,t) and (x,t) pairs on boundaries and penalize mismatch.
    """
    # --- x-boundaries ---
    j = torch.randint(0, Ny, (n_bc,), device=device)
    k = torch.randint(0, Nt, (n_bc,), device=device)
    yb = y_t[j]
    tb = t_t[k]
    x0 = x_t[0].expand_as(yb)
    xL = x_t[-1].expand_as(yb)

    xyt0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
    xytL = torch.stack([xL, yb, tb], dim=-1).requires_grad_(match_grad)
    u0 = model(xyt0); uL = model(xytL)
    loss_x = F.mse_loss(u0, uL)

    if match_grad:
        ones0 = torch.ones_like(u0)
        onesL = torch.ones_like(uL)
        gx0 = torch.autograd.grad(u0, xyt0, grad_outputs=ones0, create_graph=True)[0][:,0]
        gxL = torch.autograd.grad(uL, xytL, grad_outputs=onesL, create_graph=True)[0][:,0]
        loss_x = loss_x + F.mse_loss(gx0, gxL)

    # --- y-boundaries ---
    i = torch.randint(0, Nx, (n_bc,), device=device)
    k = torch.randint(0, Nt, (n_bc,), device=device)
    xb = x_t[i]
    tb = t_t[k]
    y0 = y_t[0].expand_as(xb)
    yL = y_t[-1].expand_as(xb)

    xyt0 = torch.stack([xb, y0, tb], dim=-1).requires_grad_(match_grad)
    xytL = torch.stack([xb, yL, tb], dim=-1).requires_grad_(match_grad)
    u0 = model(xyt0); uL = model(xytL)
    loss_y = F.mse_loss(u0, uL)

    if match_grad:
        ones0 = torch.ones_like(u0)
        onesL = torch.ones_like(uL)
        gy0 = torch.autograd.grad(u0, xyt0, grad_outputs=ones0, create_graph=True)[0][:,1]
        gyL = torch.autograd.grad(uL, xytL, grad_outputs=onesL, create_graph=True)[0][:,1]
        loss_y = loss_y + F.mse_loss(gy0, gyL)

    return (loss_x + loss_y) * 0.5


# In[10]:


# ----------------------
# Early stopping
# ----------------------
@dataclass
class EarlyStopping:
    patience: int = 10
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, metric: float):
        if metric < self.best - 1e-8:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True


# In[11]:


# ----------------------
# Train setup
# ----------------------
torch.set_float32_matmul_precision("high")

lambda_pde = 0.1      # weight for PDE residual
lambda_bc  = 0.01     # weight for periodic boundary soft constraint
use_pde    = True
use_bc     = True
match_grad_bc = False

epochs = 100
best_rmse = float("inf")
best_path = "fmlp_heatsharp_best.pt"

# optimizer already defined above; add scheduler like FieldFormer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)
early = EarlyStopping(patience=10)
grad_clip = 1.0  # same as FieldFormer

# Helper: evaluate RMSE on the validation split (unchanged)
@torch.no_grad()
def eval_val_rmse():
    se_sum = 0.0
    n_sum = 0
    for q_lin in dl_val:
        q_lin = q_lin.to(device)
        xyt  = lin_to_coords(q_lin)
        pred = model(xyt)
        tgt  = batch_targets(q_lin)
        se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
        n_sum  += q_lin.numel()
    return math.sqrt(se_sum / max(n_sum, 1))


# In[12]:


# ----------------------
# Training loop
# ----------------------
for epoch in range(1, epochs+1):
    model.train()
    total_loss = total_data = total_phys = total_bc = 0.0

    # Optional: ramp physics/bc weights early
    ramp = min(1.0, epoch / 20.0)
    lam_pde = lambda_pde * ramp
    lam_bc  = lambda_bc  * ramp

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)

        # Data loss on observed points
        xyt = lin_to_coords(q_lin)           # (B,3)
        pred = model(xyt)                    # (B,)
        tgt  = batch_targets(q_lin)          # (B,)
        data_loss = F.mse_loss(pred, tgt)

        loss = data_loss
        phys_loss = torch.tensor(0.0, device=device)
        bc_loss   = torch.tensor(0.0, device=device)

        # PDE residual on a subsample of the batch (for speed)
        if use_pde:
            subsample = q_lin[::8]
            xyt_phys = lin_to_coords(subsample)
            phys_loss = (pde_residual_autograd(xyt_phys)**2).mean()
            loss = loss + lam_pde * phys_loss

        # Periodic BC penalty on boundary stripes
        if use_bc:
            bc_loss = periodic_bc_loss(n_bc=1024, match_grad=match_grad_bc)
            loss = loss + lam_bc * bc_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_data += data_loss.item()
        total_phys += phys_loss.item()
        total_bc   += bc_loss.item()

    # Validation
    model.eval()
    rmse = eval_val_rmse()
    # step LR scheduler on validation metric (same as FieldFormer)
    scheduler.step(rmse)

    print(f"Epoch {epoch:03d} | train {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, pde {total_phys/len(dl):.4f}, bc {total_bc/len(dl):.4f}) | "
          f"val RMSE {rmse:.5f}")

    # Save best
    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "config": {
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "width": 256, "depth": 6,
                "Kx": len(Kx_list), "Ky": len(Ky_list), "Kt": len(Kt_list),
                "lambda_pde": lambda_pde, "lambda_bc": lambda_bc,
                "dx": dx, "dy": dy, "dt": dt, "alpha_x": alpha_x, "alpha_y": alpha_y
            }
        }, best_path)
        print(f"✓ Saved new best to {best_path} (val RMSE {best_rmse:.6f})")

    # Early stopping (same logic as FieldFormer)
    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
        break

print("Done.")


# In[ ]:




