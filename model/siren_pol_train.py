#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# ===== SIREN neural field for Pollution (Open BCs; Data + BC losses) =====
# Mirrors `siren_heat_train.py` structure; dataset & BCs align with ffag_pol_train.py
# -------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.pi = torch.acos(torch.zeros(1)).item() * 2


# In[2]:


# ----------------------
# Data: pollution dataset
# Expected keys: U (Nx,Ny,Nt) or u (Nx,Ny,Nt); x (Nx,), y (Ny,), t (Nt,)
# ----------------------
DATA_PATH = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"  # <-- adjust if needed
pack = np.load(DATA_PATH)

u_key = "U" if "U" in pack.files else "u"
u_np = pack[u_key]                  # (Nx, Ny, Nt)
x_np = pack["x"]                    # (Nx,)
y_np = pack["y"]                    # (Ny,)
t_np = pack["t"]                    # (Nt,)

Nx, Ny, Nt = u_np.shape

# Flatten coordinates & values (x,y,t in physical units)
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)  # (N,3)
vals_np    = u_np.reshape(-1)                                       # (N,)

coords = torch.from_numpy(coords_np).float().to(device)  # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)    # (N,)

x_t = torch.from_numpy(x_np).float().to(device)  # (Nx,)
y_t = torch.from_numpy(y_np).float().to(device)  # (Ny,)
t_t = torch.from_numpy(t_np).float().to(device)  # (Nt,)

# Extents & steps
dx = float(x_np[1] - x_np[0]) if Nx > 1 else 1.0
dy = float(y_np[1] - y_np[0]) if Ny > 1 else 1.0
dt = float(t_np[1] - t_np[0]) if Nt > 1 else 1.0
x_min, x_max = float(x_np.min()), float(x_np.max())
y_min, y_max = float(y_np.min()), float(y_np.max())
t_min, t_max = float(t_np.min()), float(t_np.max())
Lx = max(1e-12, x_max - x_min)
Ly = max(1e-12, y_max - y_min)
Tt = max(1e-12, t_max - t_min)


# In[3]:


# ----------------------
# Dataset: returns linear indices (like your other trainers)
# ----------------------
class PollutionDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123):
        N = Nx * Ny * Nt
        rng = np.random.default_rng(seed)
        all_lin = np.arange(N)
        rng.shuffle(all_lin)
        n_train = int(train_frac * N)
        n_val   = int(val_frac * N)
        self.train_idx = torch.from_numpy(all_lin[:n_train]).long()
        self.val_idx   = torch.from_numpy(all_lin[n_train:n_train+n_val]).long()
        self.test_idx  = torch.from_numpy(all_lin[n_train+n_val:]).long()
        self.split = "train"

    def set_split(self, split):
        assert split in ["train", "val", "test"]
        self.split = split

    def __len__(self):
        return len(getattr(self, f"{self.split}_idx"))

    def __getitem__(self, i):
        return getattr(self, f"{self.split}_idx")[i]

ds = PollutionDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val = PollutionDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)
ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)


# In[4]:


# ----------------------
# Utilities
# ----------------------
def lin_to_coords(q_lin: torch.Tensor) -> torch.Tensor:
    return coords[q_lin]  # (B,3)

def batch_targets(q_lin: torch.Tensor) -> torch.Tensor:
    return vals[q_lin].to(device)

def lin_to_ijk(lin: torch.Tensor):
    i = lin // (Ny*Nt)
    r = lin %  (Ny*Nt)
    j = r // Nt
    k = r %  Nt
    return i, j, k


# In[5]:


# ----------------------
# SIREN model
# ----------------------
class SineLayer(nn.Linear):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__(in_features, out_features)
        self.w0 = float(w0); self.is_first = bool(is_first)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
                self.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.weight.uniform_(-bound, bound)
            if self.bias is not None: self.bias.zero_()

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
        self.final  = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / w0_hidden
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None: self.final.bias.zero_()

    def forward(self, xyt):
        y = xyt
        for layer in self.hidden:
            y = layer(y)
        return self.final(y).squeeze(-1)

model = SIREN(in_dim=3, width=256, depth=6, out_dim=1, w0=30.0, w0_hidden=1.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
mse = nn.MSELoss()


# In[6]:


# ----------------------
# Open-BC losses (no winds)
# ----------------------
def sponge_loss(u_pred: torch.Tensor, q_lin: torch.Tensor, border_frac=0.05, u_bg=0.0, power=2):
    """Absorbing rim: penalize deviation from background near edges."""
    xyt = coords[q_lin]
    x01 = (xyt[:,0] - x_min) / Lx
    y01 = (xyt[:,1] - y_min) / Ly
    d_edge = torch.minimum(torch.minimum(x01, 1-x01), torch.minimum(y01, 1-y01))
    mask = (d_edge < border_frac).float()
    ramp = ((border_frac - d_edge).clamp(min=0) / border_frac)**power
    w = mask * ramp
    return (w * (u_pred - u_bg)**2).mean()

def radiation_bc_loss_no_v(model: nn.Module, q_lin_edge: torch.Tensor):
    """Velocity-free Orlanski radiation: enforce ∂t u + c_eff ∂n u ≈ 0 on outflow (c_eff>0)."""
    if q_lin_edge.numel() == 0:
        return torch.tensor(0.0, device=device)
    xyt = coords[q_lin_edge].clone().detach().requires_grad_(True)
    u   = model(xyt)                           # (B,)
    ones = torch.ones_like(u)
    grads = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
    ux, uy, ut = grads[:,0], grads[:,1], grads[:,2]

    eps = 1e-8
    left   = (xyt[:,0] <= x_min + eps).float()
    right  = (xyt[:,0] >= x_max - eps).float()
    bottom = (xyt[:,1] <= y_min + eps).float()
    top    = (xyt[:,1] >= y_max - eps).float()

    un = left*(-ux) + right*(ux) + bottom*(-uy) + top*(uy)  # outward normal derivative
    c_eff = (-ut / (un.clamp(min=1e-6))).clamp(min=0)       # outflow-only speed
    rad_res = ut + c_eff * un
    return (rad_res**2).mean()


# In[7]:


# ----------------------
# Eval
# ----------------------
@torch.no_grad()
def eval_val_rmse():
    se_sum, n_sum = 0.0, 0
    for q_lin in dl_val:
        q_lin = q_lin.to(device)
        xyt = lin_to_coords(q_lin)
        pred = model(xyt)
        tgt  = batch_targets(q_lin)
        se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
        n_sum  += q_lin.numel()
    return math.sqrt(se_sum / max(1, n_sum))


# In[8]:


# ----------------------
# Train
# ----------------------
torch.set_float32_matmul_precision("high")

lam_sp  = 0.05   # sponge weight (0.02–0.10 typical)
lam_rad = 0.05   # radiation weight (0.01–0.05 typical)
epochs = 100
grad_clip = 1.0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)

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
            if self.bad_epochs >= self.patience: self.stopped = True

early = EarlyStopping(patience=10)
best_rmse = float("inf")
best_path = "/scratch/ab9738/fieldformer/model/siren_pol_best.pt"

for epoch in range(1, epochs+1):
    model.train()
    total_loss = total_data = total_sp = total_rad = 0.0

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)
        xyt  = lin_to_coords(q_lin)
        pred = model(xyt)
        tgt  = batch_targets(q_lin)

        data_loss = mse(pred, tgt)

        # Edge indices for BCs: exact boundaries
        i, j, k0 = lin_to_ijk(q_lin)
        edge_mask = (i == 0) | (i == Nx-1) | (j == 0) | (j == Ny-1)
        q_edge = q_lin[edge_mask]

        sp_loss  = sponge_loss(pred, q_lin, border_frac=0.05, u_bg=0.0, power=2)
        rad_loss = radiation_bc_loss_no_v(model, q_edge) if q_edge.numel() else torch.tensor(0.0, device=device)

        loss = data_loss + lam_sp*sp_loss + lam_rad*rad_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss)
        total_data += float(data_loss)
        total_sp   += float(sp_loss)
        total_rad  += float(rad_loss)

    # Validation
    model.eval()
    rmse = eval_val_rmse()
    scheduler.step(rmse)

    print(f"Epoch {epoch:03d} | train {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, sponge {total_sp/len(dl):.4f}, rad {total_rad/len(dl):.4f}) | "
          f"val RMSE {rmse:.5f}")

    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "config": {
                "model": "SIREN_pollution_openBC",
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "width": 256, "depth": 6,
                "w0": 30.0, "w0_hidden": 1.0,
                "lambda_sponge": lam_sp, "lambda_radiation": lam_rad,
                "dx": dx, "dy": dy, "dt": dt,
                "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max
            }
        }, best_path)
        print(f"✓ Saved new best to {best_path} (val RMSE {best_rmse:.6f})")

    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
        break

print("Done.")


# In[ ]:




