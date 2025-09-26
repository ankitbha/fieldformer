#!/usr/bin/env python
# coding: utf-8

# ===== SIREN neural field (periodic) for 2D linear SWE: joint 3-output [η, u, v] =====
# Minimal diffs from your heat SIREN:
# - Load swe_periodic_dataset.npz (eta,u,v,x,y,t,params...)
# - SIREN(out_dim=3) → predictions shape (B,3)
# - Targets shape (B,3)
# - PINN residuals: Ru = u_t + g*eta_x ; Rv = v_t + g*eta_y ; Rη = η_t + H*(u_x + v_y)
# - Periodic BC loss on function equality for all 3 channels (x=0/Lx and y=0/Ly)

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
# Data: SWE dataset (periodic)
# ----------------------
pack = np.load("/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz")
eta_np = pack["eta"]     # (Nx, Ny, Nt)
u_np   = pack["u"]       # (Nx, Ny, Nt)
v_np   = pack["v"]       # (Nx, Ny, Nt)
x_np   = pack["x"]
y_np   = pack["y"]
t_np   = pack["t"]
X_np   = pack["X"]
Y_np   = pack["Y"]
params = pack["params"]
names  = list(pack["param_names"])

g  = float(params[names.index("g")])
H  = float(params[names.index("H")])
dx = float(params[names.index("dx")])
dy = float(params[names.index("dy")])
dt = float(params[names.index("dt")])

Nx, Ny, Nt = eta_np.shape

# Flatten coordinates & values
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)       # (N,3)
vals_np    = np.stack([eta_np, u_np, v_np], axis=-1).reshape(-1, 3)       # (N,3)

coords = torch.from_numpy(coords_np).float().to(device)   # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)     # (N,3)

x_t = torch.from_numpy(x_np).float().to(device)
y_t = torch.from_numpy(y_np).float().to(device)
t_t = torch.from_numpy(t_np).float().to(device)

Lx = float(x_np.max() - x_np.min())
Ly = float(y_np.max() - y_np.min())
Tt = float(t_np.max() - t_np.min()) if Nt > 1 else 1.0

# ----------------------
# Dataset / Loaders (same pattern)
# ----------------------
class PeriodicDataset(Dataset):
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
        return len(self.train_idx if self.split=="train" else self.val_idx if self.split=="val" else self.test_idx)
    def __getitem__(self, idx):
        return (self.train_idx if self.split=="train" else self.val_idx if self.split=="val" else self.test_idx)[idx]

ds = PeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds.set_split("train"); dl = DataLoader(ds, batch_size=8192, shuffle=True, drop_last=True)
ds_val = PeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val.set_split("val"); dl_val = DataLoader(ds_val, batch_size=8192, shuffle=False)

# ----------------------
# SIREN model (out_dim = 3)
# ----------------------
class SineLayer(nn.Linear):
    """Linear layer + sine with SIREN init."""
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
    def __init__(self, in_dim=3, width=256, depth=6, out_dim=3, w0=30.0, w0_hidden=1.0):
        super().__init__()
        assert depth >= 2
        layers = [SineLayer(in_dim, width, w0=w0, is_first=True)]
        for _ in range(depth-2):
            layers.append(SineLayer(width, width, w0=w0_hidden, is_first=False))
        self.hidden = nn.ModuleList(layers)
        self.final = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / w0_hidden
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None: self.final.bias.zero_()
    def forward(self, xyt):
        y = xyt
        for layer in self.hidden:
            y = layer(y)
        return self.final(y)  # (B,3) -> [η̂, û, v̂]

model = SIREN(in_dim=3, width=256, depth=6, out_dim=3, w0=30.0, w0_hidden=1.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
mse = nn.MSELoss()

# ----------------------
# Utilities
# ----------------------
def lin_to_coords(q_lin):  # (B,) -> (B,3)
    return coords[q_lin]
def batch_targets(q_lin):  # (B,) -> (B,3)
    return vals[q_lin].to(device)

# ----------------------
# PINN residuals for linear SWE via autograd
#   Ru  = u_t  + g * eta_x
#   Rv  = v_t  + g * eta_y
#   Rη  = η_t  + H * (u_x + v_y)
# ----------------------
def pde_residual_autograd(xyt):
    xyt = xyt.requires_grad_(True)          # (B,3)
    pred = model(xyt)                       # (B,3)
    eta_hat, u_hat, v_hat = pred[:,0], pred[:,1], pred[:,2]

    ones_e = torch.ones_like(eta_hat)
    ones_u = torch.ones_like(u_hat)
    ones_v = torch.ones_like(v_hat)

    # First derivatives
    grads_e = torch.autograd.grad(eta_hat, xyt, grad_outputs=ones_e, create_graph=True)[0]
    grads_u = torch.autograd.grad(u_hat,   xyt, grad_outputs=ones_u, create_graph=True)[0]
    grads_v = torch.autograd.grad(v_hat,   xyt, grad_outputs=ones_v, create_graph=True)[0]

    eta_x, eta_y, eta_t = grads_e[:,0], grads_e[:,1], grads_e[:,2]
    u_x,   u_y,   u_t   = grads_u[:,0], grads_u[:,1], grads_u[:,2]
    v_x,   v_y,   v_t   = grads_v[:,0], grads_v[:,1], grads_v[:,2]

    R_u  = u_t   + g * eta_x
    R_v  = v_t   + g * eta_y
    R_et = eta_t + H * (u_x + v_y)

    return torch.stack([R_u, R_v, R_et], dim=-1)   # (B,3)

# ----------------------
# Periodic BC loss on x and y for all 3 channels
# ----------------------
def periodic_bc_loss(n_bc=1024, match_grad=False):
    # x-boundaries
    j = torch.randint(0, Ny, (n_bc,), device=device)
    k = torch.randint(0, Nt, (n_bc,), device=device)
    yb, tb = y_t[j], t_t[k]
    x0, xL = x_t[0].expand_as(yb), x_t[-1].expand_as(yb)
    xyt0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
    xytL = torch.stack([xL, yb, tb], dim=-1).requires_grad_(match_grad)
    u0 = model(xyt0); uL = model(xytL)         # (n_bc,3)
    loss_x = F.mse_loss(u0, uL)
    if match_grad:
        ones0 = torch.ones_like(u0.sum(dim=1))
        onesL = torch.ones_like(uL.sum(dim=1))
        gx0 = torch.autograd.grad(u0.sum(dim=1), xyt0, grad_outputs=ones0, create_graph=True)[0][:,0]
        gxL = torch.autograd.grad(uL.sum(dim=1), xytL, grad_outputs=onesL, create_graph=True)[0][:,0]
        loss_x = loss_x + F.mse_loss(gx0, gxL)

    # y-boundaries
    i = torch.randint(0, Nx, (n_bc,), device=device)
    k = torch.randint(0, Nt, (n_bc,), device=device)
    xb, tb = x_t[i], t_t[k]
    y0, yL = y_t[0].expand_as(xb), y_t[-1].expand_as(xb)
    xyt0 = torch.stack([xb, y0, tb], dim=-1).requires_grad_(match_grad)
    xytL = torch.stack([xb, yL, tb], dim=-1).requires_grad_(match_grad)
    u0 = model(xyt0); uL = model(xytL)
    loss_y = F.mse_loss(u0, uL)
    if match_grad:
        ones0 = torch.ones_like(u0.sum(dim=1))
        onesL = torch.ones_like(uL.sum(dim=1))
        gy0 = torch.autograd.grad(u0.sum(dim=1), xyt0, grad_outputs=ones0, create_graph=True)[0][:,1]
        gyL = torch.autograd.grad(uL.sum(dim=1), xytL, grad_outputs=onesL, create_graph=True)[0][:,1]
        loss_y = loss_y + F.mse_loss(gy0, gyL)

    return 0.5*(loss_x + loss_y)

# ----------------------
# Early stopping & scheduler
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
            if self.bad_epochs >= self.patience: self.stopped = True

torch.set_float32_matmul_precision("high")
epochs = 100
best_rmse = float("inf")
best_path = "siren_swe_best.pt"
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)
early = EarlyStopping(patience=10)
grad_clip = 1.0

# Physics/BC weights (simple ramp)
lambda_pde = 0.1
lambda_bc  = 0.01
use_pde = True
use_bc  = True
match_grad_bc = False

@torch.no_grad()
def eval_val_rmse():
    se_sum = 0.0; n_sum = 0
    for q_lin in dl_val:
        q_lin = q_lin.to(device)
        xyt  = lin_to_coords(q_lin)          # (B,3)
        pred = model(xyt)                    # (B,3)
        tgt  = batch_targets(q_lin)          # (B,3)
        se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
        n_sum  += tgt.numel()                # counts all 3 channels
    return math.sqrt(se_sum / max(n_sum, 1))

# ----------------------
# Training loop
# ----------------------
for epoch in range(1, epochs+1):
    model.train()
    total_loss = total_data = total_phys = total_bc = 0.0

    ramp = min(1.0, epoch / 20.0)
    lam_pde = lambda_pde * ramp
    lam_bc  = lambda_bc  * ramp

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)

        xyt  = lin_to_coords(q_lin)          # (B,3)
        pred = model(xyt)                    # (B,3)
        tgt  = batch_targets(q_lin)          # (B,3)
        data_loss = F.mse_loss(pred, tgt)

        loss = data_loss
        phys_loss = torch.tensor(0.0, device=device)
        bc_loss   = torch.tensor(0.0, device=device)

        if use_pde:
            subsample = q_lin[::8]
            xyt_phys = lin_to_coords(subsample)
            R = pde_residual_autograd(xyt_phys)   # (b,3)
            phys_loss = (R**2).mean()
            loss = loss + lam_pde * phys_loss

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
    scheduler.step(rmse)
    print(f"Epoch {epoch:03d} | train {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, pde {total_phys/len(dl):.4f}, bc {total_bc/len(dl):.4f}) | "
          f"val RMSE {rmse:.6f}")

    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "config": {
                "model": "SIREN",
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "width": 256, "depth": 6,
                "lambda_pde": lambda_pde, "lambda_bc": lambda_bc,
                "dx": dx, "dy": dy, "dt": dt, "g": g, "H": H
            }
        }, best_path)
        print(f"✓ Saved new best to {best_path} (val RMSE {best_rmse:.6f})")

    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
        break

print("Done.")
