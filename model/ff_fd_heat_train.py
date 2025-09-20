#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ========= FieldFormer (starter) for periodic heat dataset =========
# - Loads your .npz dataset
# - Builds a mesh-free (but grid-aware) query set
# - k-NN neighborhoods with learnable velocity-like scaling gammas
# - Transformer encoder over local tokens, mean pool, MLP head
# - Optional physics loss using finite-difference stencil (periodic)
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataclasses import dataclass
torch.pi = torch.acos(torch.zeros(1)).item() * 2


# In[ ]:


# ----------------------
# Data: load the periodic heat dataset you created
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


# In[ ]:


# pull a few scalars for physics loss & periodic wrapping
alpha_x = float(params[list(names).index("alpha_x")])
alpha_y = float(params[list(names).index("alpha_y")])
dx = float(params[list(names).index("dx")])
dy = float(params[list(names).index("dy")])
dt = float(params[list(names).index("dt")])


# In[ ]:


Nx, Ny, Nt = u_np.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Flatten full coordinate list and values for mesh-free access
# coords: (N, 3) with [x, y, t], values: (N,)
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)   # (N,3)
vals_np    = u_np.reshape(-1)                                        # (N,)

coords = torch.from_numpy(coords_np).float().to(device)
vals   = torch.from_numpy(vals_np).float().to(device)


# In[ ]:


# ----------------------
# Dataset that returns query indices (not raw coords)
# ----------------------
class HeatPeriodicDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=0):
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
        if self.split == "train":
            return self.train_idx[idx]
        elif self.split == "val":
            return self.val_idx[idx]
        else:
            return self.test_idx[idx]


# In[ ]:


# ----------------------
# Instantiate datasets + loaders
# ----------------------
ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)

# training loader
ds.set_split("train")
dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)

# validation loader
ds_val = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val.set_split("val")
dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)


# In[ ]:


# ---------------------- 
# Neighbor finder (k-NN in scaled space-time) 
# ----------------------

def lin_to_ijk(lin):
    i = lin // (Ny*Nt)
    r = lin %  (Ny*Nt)
    j = r // Nt
    k = r %  Nt
    return i, j, k

def ijk_to_lin(i, j, k):
    return (i % Nx) * (Ny*Nt) + (j % Ny) * Nt + (k % Nt)

# --- Build a global, metric-sorted offset table once per (approx) gamma ---
def build_offset_table(k, gammas, dx, dy, dt, max_rad=None):
    """
    Returns integer offsets [(di,dj,dk)] of length >= k, sorted by scaled distance.
    We then take the first k in gather_neighbors_periodic.
    """
    gx, gy, gt = [float(x) for x in gammas]            # gammas (positive)
    # Heuristic radius so we have more than k candidates
    # Volume ~ (2Rx+1)(2Ry+1)(2Rt+1) ≳ c*k. Use anisotropic radii by gamma.
    c = 4.0
    # choose radii proportional to 1/gamma so tighter gamma => smaller radius
    # also scale by grid spacing
    base = (c * k) ** (1/3)
    Rx = max(1, int(base * (1.0/max(gx,1e-8))))
    Ry = max(1, int(base * (1.0/max(gy,1e-8))))
    Rt = max(1, int(base * (1.0/max(gt,1e-8))))
    if max_rad is not None:
        Rx = min(Rx, max_rad); Ry = min(Ry, max_rad); Rt = min(Rt, max_rad)

    # Enumerate offsets and distances (include (0,0,0); we’ll skip center later)
    offs = []
    for di in range(-Rx, Rx+1):
        for dj in range(-Ry, Ry+1):
            for dk in range(-Rt, Rt+1):
                # physical deltas (use grid spacing)
                dxp = di * dx; dyp = dj * dy; dtp = dk * dt
                d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp)**2
                offs.append((d2, di, dj, dk))
    offs.sort(key=lambda z: z[0])  # increasing distance
    # drop the center if present
    offs = [(di,dj,dk) for (d2,di,dj,dk) in offs if not (di==0 and dj==0 and dk==0)]
    return offs  # long list; take first k where needed

# --- Fast periodic neighbor gather using pre-sorted offsets ---
def gather_neighbors_periodic(q_lin_idx, k, offsets_ijk):
    """
    q_lin_idx: (B,) long
    offsets_ijk: list[(di,dj,dk)] pre-sorted by the scaled metric; we take first k
    Returns: (B,k) neighbor linear indices with periodic wrap.
    Uses global Nx, Ny, Nt and ijk_to_lin / lin_to_ijk.
    """
    # unpack to i, j, k0
    i, j, k0 = lin_to_ijk(q_lin_idx)  # each (B,)

    sel = offsets_ijk[:k]
    di = torch.tensor([o[0] for o in sel], device=q_lin_idx.device, dtype=i.dtype)  # (k,)
    dj = torch.tensor([o[1] for o in sel], device=q_lin_idx.device, dtype=i.dtype)
    dk = torch.tensor([o[2] for o in sel], device=q_lin_idx.device, dtype=i.dtype)

    I = i[:, None] + di[None, :]   # (B,k)
    J = j[:, None] + dj[None, :]
    K = k0[:, None] + dk[None, :]

    nb_lin = ijk_to_lin(I, J, K)   # (B,k) with periodic wrap inside ijk_to_lin
    return nb_lin, di, dj, dk


# In[ ]:


class FieldFormer(nn.Module):
    def __init__(self, d_in, d_model=64, nhead=4, num_layers=2, k_neighbors=32, d_ff=128):
        super().__init__()
        self.k = k_neighbors
        # learnable log-gammas -> positive gammas for x,y,t (used when building offsets_ijk outside)
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, q_lin_idx, offsets_ijk):
        """
        q_lin_idx: (B,)
        offsets_ijk: list[(di,dj,dk)] sorted for current gammas; will take first self.k
        Returns:
          pred: (B,)
          extra: dict
        """
        # neighbors via periodic offsets (no BxN distances)
        nb_idx, di, dj, dk = gather_neighbors_periodic(q_lin_idx, self.k, offsets_ijk)  # (B,k)
        device_ = q_lin_idx.device
        rel_ijk = torch.stack([di, dj, dk], dim=1).to(torch.float32)       # (k,3)
        rel = rel_ijk[None, :, :].expand(q_lin_idx.shape[0], -1, -1)       # (B,k,3)
        # scale to physical deltas
        rel = rel * torch.tensor([dx, dy, dt], device=device_, dtype=torch.float32)  # (B,k,3)
        scale = torch.exp(self.log_gammas)[None, None, :]                  # (1,1,3)
        rel = rel * scale
        # neighbor values
        nb_vals = vals[nb_idx].to(device_)[..., None].to(torch.float32)    # (B,k,1)
        tokens = torch.cat([rel, nb_vals], dim=-1)                         # (B,k,4)
        h = self.input_proj(tokens)
        h = self.encoder(h)
        h_mean = h.mean(dim=1)
        out = self.head(h_mean).squeeze(-1)
        return out, {"gammas": torch.exp(self.log_gammas).detach()}


# In[ ]:


# ----------------------
# Physics loss (optional): periodic finite differences around query
# Stencil at the query's (i,j,k): use ±1 in x or y, ±1 in t with periodic wrap.
# ------------------------------------------------

@torch.no_grad()
def stencil_indices(q_lin_idx):
    # returns dict of linear indices for center and ±1 offsets
    i, j, k = lin_to_ijk(q_lin_idx)
    i_p = (i+1) % Nx; i_m = (i-1) % Nx
    j_p = (j+1) % Ny; j_m = (j-1) % Ny
    k_p = (k+1) % Nt; k_m = (k-1) % Nt
    return {
        "c": q_lin_idx,
        "ip": ijk_to_lin(i_p, j,   k),
        "im": ijk_to_lin(i_m, j,   k),
        "jp": ijk_to_lin(i,   j_p, k),
        "jm": ijk_to_lin(i,   j_m, k),
        "kp": ijk_to_lin(i,   j,   k_p),
        "km": ijk_to_lin(i,   j,   k_m),
    }

def forcing_torch(xx, yy, tt):
    return 5.0 * torch.cos(torch.pi * xx) * torch.cos(torch.pi * yy) * torch.sin(4 * torch.pi * tt / 5.0)


def physics_residual(model, q_lin_idx, k_neighbors, offsets_ijk):
    # Predict u at stencil points via the same model (using neighbors each time).
    # NOTE: This is more expensive but stays mesh-free in spirit.
    idx = stencil_indices(q_lin_idx)
    preds = {}
    for key, lin in idx.items():
        p, _ = model(lin, offsets_ijk)
        preds[key] = p

    # finite differences
    dudt = (preds["kp"] - preds["km"]) / (2.0 * dt)
    d2udx2 = (preds["ip"] - 2.0*preds["c"] + preds["im"]) / (dx*dx)
    d2udy2 = (preds["jp"] - 2.0*preds["c"] + preds["jm"]) / (dy*dy)

    # Residual of anisotropic heat equation without explicit forcing in the loss.
    # If you want to include known forcing f(x,y,t), add it here.
    # R = dudt - (alpha_x * d2udx2 + alpha_y * d2udy2)
    R = dudt - (alpha_x * d2udx2 + alpha_y * d2udy2) - forcing_torch(coords[q_lin_idx,0].to(preds["c"].device),
                                                                 coords[q_lin_idx,1].to(preds["c"].device),
                                                                 coords[q_lin_idx,2].to(preds["c"].device))
    return R


# In[ ]:


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


# In[ ]:


# ----------------------
# Train setup
# ----------------------
torch.set_float32_matmul_precision("high")
model = FieldFormer(d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128).to(device)
base_params = [p for n, p in model.named_parameters() if n != "log_gammas"]
optimizer = torch.optim.AdamW(
    [
        {"params": base_params,                 "lr": 3e-4, "weight_decay": 1e-4},
        {"params": [model.log_gammas],          "lr": 3e-3, "weight_decay": 0.0},
    ]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
grad_clip = 1.0
max_epochs = 100
early = EarlyStopping(patience=10)
mse = nn.MSELoss()

lambda_phys = 0.1   # weight for physics loss (tune)
use_physics = True  # toggle on/off

def batch_targets(q_lin_idx):
    return vals[q_lin_idx].to(device)


# In[ ]:


# ----------------------
# Training loop (few epochs as a smoke test)
# ----------------------

best_rmse = float("inf")
best_path = "ff_fd_heat_best.pt"

for epoch in range(max_epochs):
    model.train()
    total_loss = total_data = total_phys = 0.0

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
    offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    for q_lin in tqdm(dl, desc=f"Epoch {epoch+1} [train]", leave=False):
        q_lin = q_lin.to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred, _ = model(q_lin, offsets_ijk)
            tgt = batch_targets(q_lin)
            data_loss = mse(pred, tgt)
            loss = data_loss

            if use_physics:
                subsample = q_lin[::8]
                R = physics_residual(model, subsample, model.k, offsets_ijk)
                phys_loss = (R**2).mean()
                loss = loss + lambda_phys * phys_loss
            else:
                phys_loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_data += data_loss.item()
        total_phys += phys_loss.item()

        # rebuild offsets AFTER the step (gammas updated)
        with torch.no_grad():
            gam = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk_val = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

        se_sum = 0.0
        n_sum = 0
        for q_lin in tqdm(dl_val, desc=f"Epoch {epoch+1} [val]", leave=False):
            q_lin = q_lin.to(device)
            pred, _ = model(q_lin, offsets_ijk_val)
            tgt = batch_targets(q_lin)
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum  += q_lin.numel()
        rmse = np.sqrt(se_sum / n_sum)

    # step LR scheduler on validation metric
    scheduler.step(rmse)

    print(f"Epoch {epoch+1:02d} | train loss {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, phys {total_phys/len(dl):.4f}) | val RMSE {rmse:.4f}")

    # save best (unchanged)
    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "gammas": model.log_gammas.detach().exp().cpu().numpy(),
            "config": {
                "Nx": Nx, "Ny": Ny, "Nt": Nt,
                "k_neighbors": model.k,
                "d_model": 64, "nhead": 4, "num_layers": 2,
                "lambda_phys": lambda_phys, "use_physics": use_physics,
                "dx": dx, "dy": dy, "dt": dt, "alpha_x": alpha_x, "alpha_y": alpha_y
            }
        }, best_path)
        print(f"✓ Saved new best to {best_path} (val RMSE {best_rmse:.6f})")

    # early stopping
    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch+1} (best RMSE {early.best:.6f})")
        break

print("Gammas (learned space-time scales):", model.log_gammas.detach().exp().cpu().numpy())


# In[ ]:




