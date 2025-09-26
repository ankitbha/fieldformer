#!/usr/bin/env python
# coding: utf-8
# ===== Fourier-MLP neural field — Pollution (Open BCs; Data + BC losses) =====
# Mirrors fmlp_heat_train.py (arch/loader style) but:
#   • loads pollution dataset (open domain)
#   • no interior physics residual
#   • adds open-BC soft constraints: sponge + velocity-free radiation (Orlanski)
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from dataclasses import dataclass

torch.pi = torch.acos(torch.zeros(1)).item() * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Data: pollution dataset
# Expected keys: u (Nx,Ny,Nt) or U; x (Nx,), y (Ny,), t (Nt,)
# Optional keys (ignored here): vx, vy, S
# ----------------------
DATA_PATH = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"  # <-- adjust if needed
pack = np.load(DATA_PATH)

u_key = "U" if "U" in pack.files else "u"
u_np   = pack[u_key]            # (Nx, Ny, Nt)
x_np   = pack["x"]              # (Nx,)
y_np   = pack["y"]              # (Ny,)
t_np   = pack["t"]              # (Nt,)

Nx, Ny, Nt = u_np.shape

# Flatten coords & values (physical units)
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)   # (N,3)
vals_np    = u_np.reshape(-1)                                        # (N,)

coords = torch.from_numpy(coords_np).float().to(device)  # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)    # (N,)

x_t = torch.from_numpy(x_np).float().to(device)  # (Nx,)
y_t = torch.from_numpy(y_np).float().to(device)  # (Ny,)
t_t = torch.from_numpy(t_np).float().to(device)  # (Nt,)

# Extents & steps (open domain)
dx = float(x_np[1] - x_np[0]) if Nx > 1 else 1.0
dy = float(y_np[1] - y_np[0]) if Ny > 1 else 1.0
dt = float(t_np[1] - t_np[0]) if Nt > 1 else 1.0

x_min, x_max = float(x_np.min()), float(x_np.max())
y_min, y_max = float(y_np.min()), float(y_np.max())
t_min, t_max = float(t_np.min()), float(t_np.max())

Lx = max(1e-12, x_max - x_min)
Ly = max(1e-12, y_max - y_min)
Tt = max(1e-12, t_max - t_min)

# ----------------------
# Dataset returning linear indices (same pattern as fmlp_heat_train)
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
        assert split in ["train","val","test"]
        self.split = split

    def __len__(self):
        return len(getattr(self, f"{self.split}_idx"))

    def __getitem__(self, idx):
        return getattr(self, f"{self.split}_idx")[idx]

ds = PollutionDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val = PollutionDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)

ds.set_split("train")
dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)

ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)

# ----------------------
# Utilities
# ----------------------
def lin_to_coords(q_lin: torch.Tensor) -> torch.Tensor:
    return coords[q_lin]  # (B,3) physical units

def batch_targets(q_lin: torch.Tensor) -> torch.Tensor:
    return vals[q_lin].to(device)

def lin_to_ijk(lin: torch.Tensor):
    i = lin // (Ny*Nt)
    r = lin %  (Ny*Nt)
    j = r // Nt
    k = r %  Nt
    return i, j, k

# ----------------------
# Fourier features (periodic in each axis but used as features; domain is open)
# ----------------------
def build_harmonics(K):
    if isinstance(K, int): return torch.arange(1, K+1, dtype=torch.float32, device=device)
    return torch.tensor(K, dtype=torch.float32, device=device)

Kx_list = build_harmonics(16)   # tune
Ky_list = build_harmonics(16)   # tune
Kt_list = build_harmonics(8)    # tune

def fourier_encode_1d(x, Ks, L):
    # x in physical units; map to angles over [0, L]
    z = (2 * math.pi) * ( (x - x.min()) / (L if L > 0 else 1.0) ).unsqueeze(-1) * Ks.unsqueeze(0)
    return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

def fourier_encode_3d(xyt):
    x, y, t = xyt[:,0], xyt[:,1], xyt[:,2]
    # center at domain minima for stability
    x = (x - x_min); y = (y - y_min); t = (t - t_min)
    fx = fourier_encode_1d(x, Kx_list, Lx)
    fy = fourier_encode_1d(y, Ky_list, Ly)
    ft = fourier_encode_1d(t, Kt_list, Tt)
    return torch.cat([fx, fy, ft], dim=-1)

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
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, xyt):
        feat = fourier_encode_3d(xyt)     # (B, D)
        return self.net(feat).squeeze(-1) # (B,)

model = FourierMLP(width=256, depth=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
mse = nn.MSELoss()

# ----------------------
# Open-boundary losses (no winds)
# ----------------------
def sponge_loss(u_pred: torch.Tensor, q_lin: torch.Tensor, border_frac=0.05, u_bg=0.0, power=2):
    """
    Absorbing rim near edges; encourages values to relax toward u_bg.
    """
    xyt = coords[q_lin]
    x01 = (xyt[:,0] - x_min) / Lx
    y01 = (xyt[:,1] - y_min) / Ly
    d_edge = torch.minimum(torch.minimum(x01, 1-x01), torch.minimum(y01, 1-y01))
    mask = (d_edge < border_frac).float()
    ramp = ((border_frac - d_edge).clamp(min=0) / border_frac)**power
    w = mask * ramp
    return (w * (u_pred - u_bg)**2).mean()

def radiation_bc_loss_no_v(model: nn.Module, q_lin_edge: torch.Tensor):
    """
    Velocity-free Orlanski radiation:
        enforce ∂t u + c_eff ∂n u ≈ 0 for outflow where c_eff = max(0, -ut/un).
    """
    if q_lin_edge.numel() == 0:
        return torch.tensor(0.0, device=device)

    xyt = coords[q_lin_edge].clone().detach().requires_grad_(True)
    u   = model(xyt)
    ones = torch.ones_like(u)
    grads = torch.autograd.grad(u, xyt, grad_outputs=ones, create_graph=True)[0]
    ux, uy, ut = grads[:,0], grads[:,1], grads[:,2]

    eps = 1e-8
    left   = (xyt[:,0] <= x_min + eps).float()
    right  = (xyt[:,0] >= x_max - eps).float()
    bottom = (xyt[:,1] <= y_min + eps).float()
    top    = (xyt[:,1] >= y_max - eps).float()

    un = left*(-ux) + right*(ux) + bottom*(-uy) + top*(uy)  # outward-normal derivative
    c_eff = (-ut / (un.clamp(min=1e-6))).clamp(min=0)       # outflow-only speed
    rad_res = ut + c_eff * un
    return (rad_res**2).mean()

# ----------------------
# Validation RMSE
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

# ----------------------
# Training config
# ----------------------
torch.set_float32_matmul_precision("high")

lam_sp  = 0.05   # sponge weight (0.02–0.10 typical)
lam_rad = 0.03   # radiation weight (0.01–0.05 typical)
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
best_path = "/scratch/ab9738/fieldformer/model/fmlp_pol_best.pt"

# ----------------------
# Training loop
# ----------------------
for epoch in range(1, epochs+1):
    model.train()
    total_loss = total_data = total_sp = total_rad = 0.0

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)
        xyt  = lin_to_coords(q_lin)
        pred = model(xyt)
        tgt  = batch_targets(q_lin)

        data_loss = mse(pred, tgt)

        # Edge indices (exact boundary cells)
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

    # Save best
    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "config": {
                "variant": "fmlp_pollution_openBC",
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "width": 256, "depth": 6,
                "Kx": len(Kx_list), "Ky": len(Ky_list), "Kt": len(Kt_list),
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
