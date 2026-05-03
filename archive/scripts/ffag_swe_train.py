#!/usr/bin/env python
# coding: utf-8
# ===== FieldFormer_Autograd for SWE (η,u,v), joint 3-output =====
# Minimal diffs from ffag_heat_train.py: switch to SWE data, d_in=6 tokens,
# 3-output head, SWE autograd residuals, BC over 3 channels.

import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from torch.backends.cuda import sdp_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# ----------------------
# Data: load periodic SWE dataset
# ----------------------
pack = np.load("/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz")
eta_np = pack["eta"]   # (Nx, Ny, Nt)
u_np   = pack["u"]     # (Nx, Ny, Nt)
v_np   = pack["v"]     # (Nx, Ny, Nt)
x_np   = pack["x"]; y_np = pack["y"]; t_np = pack["t"]
X_np   = pack["X"]; Y_np = pack["Y"]
params = pack["params"]; names = list(pack["param_names"])

g  = float(params[names.index("g")])
H  = float(params[names.index("H")])
dx = float(params[names.index("dx")])
dy = float(params[names.index("dy")])
dt = float(params[names.index("dt")])

Nx, Ny, Nt = eta_np.shape

# Flatten coords & values
XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
coords_np  = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)       # (N,3)
vals_np    = np.stack([eta_np, u_np, v_np], axis=-1).reshape(-1, 3)       # (N,3)

coords = torch.from_numpy(coords_np).float().to(device)   # (N,3)
vals   = torch.from_numpy(vals_np).float().to(device)     # (N,3)
x_t    = torch.from_numpy(x_np).float().to(device)
y_t    = torch.from_numpy(y_np).float().to(device)
t_t    = torch.from_numpy(t_np).float().to(device)

Lx = float(x_np.max() - x_np.min())
Ly = float(y_np.max() - y_np.min())
Tt = float(t_np.max() - t_np.min()) if Nt > 1 else 1.0

# ----------------------
# Dataset / Loaders (unchanged split logic)
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
    def set_split(self, split): self.split = split
    def __len__(self):
        return len(self.train_idx if self.split=="train" else self.val_idx if self.split=="val" else self.test_idx)
    def __getitem__(self, idx):
        return (self.train_idx if self.split=="train" else self.val_idx if self.split=="val" else self.test_idx)[idx]

ds = PeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds.set_split("train"); dl = DataLoader(ds, batch_size=8192, shuffle=True, drop_last=True)
ds_val = PeriodicDataset(Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123)
ds_val.set_split("val"); dl_val = DataLoader(ds_val, batch_size=8192, shuffle=False)

# ----------------------
# Index helpers and neighbor gather (same logic)
# ----------------------
def lin_to_ijk(lin):
    i = lin // (Ny*Nt); r = lin % (Ny*Nt); j = r // Nt; k = r % Nt; return i, j, k
def ijk_to_lin(i,j,k): return (i % Nx) * (Ny*Nt) + (j % Ny) * Nt + (k % Nt)

def build_offset_table(k, gammas, dx, dy, dt, frac_time=0.5, max_dt_radius=None):
    gx, gy, gt = gammas
    if max_dt_radius is None:
        max_dt_radius = int(3 * (1.0 / max(1e-6, gt)))
    rad = int(max(2, math.ceil((k**(1/3)) * 4)))
    cand = []
    for di in range(-rad, rad+1):
        for dj in range(-rad, rad+1):
            for dk in range(-max_dt_radius, max_dt_radius+1):
                if di==0 and dj==0 and dk==0: continue
                dxp = di*dx; dyp = dj*dy; dtp = dk*dt
                dtp_eff = dtp / (1.0 + abs(dk)/2.0)
                d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp_eff)**2
                cand.append((d2, di, dj, dk))
    cand.sort(key=lambda t: t[0])
    n_time = max(1, int(k*frac_time)); n_space = k - n_time
    cand_time  = sorted(cand, key=lambda t: abs(t[3]))
    cand_space = sorted(cand, key=lambda t: (t[0], abs(t[3])))
    sel, seen = [], set()
    for _,di,dj,dk in cand_time:
        key=(di,dj,dk)
        if key not in seen: sel.append(key); seen.add(key)
        if len(sel)>=n_time: break
    for _,di,dj,dk in cand_space:
        key=(di,dj,dk)
        if key not in seen: sel.append(key); seen.add(key)
        if len(sel)>=k: break
    return sel

def gather_neighbors_periodic(q_lin_idx, k, offsets_ijk, wrap_x=True, wrap_y=True, wrap_t=False):
    i, j, k0 = lin_to_ijk(q_lin_idx)
    sel = offsets_ijk[:k]
    dev, dtype = q_lin_idx.device, i.dtype
    di = torch.tensor([o[0] for o in sel], device=dev, dtype=dtype)
    dj = torch.tensor([o[1] for o in sel], device=dev, dtype=dtype)
    dk = torch.tensor([o[2] for o in sel], device=dev, dtype=dtype)
    I = i[:,None] + di[None,:]; J = j[:,None] + dj[None,:]; K = k0[:,None] + dk[None,:]
    I = I % Nx if wrap_x else I.clamp_(0, Nx-1)
    J = J % Ny if wrap_y else J.clamp_(0, Ny-1)
    K = K % Nt if wrap_t else K.clamp_(0, Nt-1)
    return I * (Ny*Nt) + J * Nt + K

# ----------------------
# FieldFormer (d_in=6, head->3)
# ----------------------
class FieldFormerAutograd(nn.Module):
    def __init__(self, d_in=6, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128, wrap=True):
        super().__init__()
        self.k = k_neighbors
        self.wrap = wrap
        self.wrap_x = bool(wrap); self.wrap_y = bool(wrap); self.wrap_t = False
        self.register_buffer("_huber_delta_ema", torch.tensor(1.0, dtype=torch.float32))
        self.log_gammas = nn.Parameter(torch.zeros(3))  # x, y, t
        self.input_proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, 3))   # -> [eta,u,v]

    @staticmethod
    def _delta_periodic(a, b, L): return (a - b + 0.5*L) % L - 0.5*L

    def _relative_deltas(self, q_xyz, nb_xyz):
        if self.wrap_x: dx_ = self._delta_periodic(nb_xyz[...,0], q_xyz[:,None,0], Lx)
        else:           dx_ = nb_xyz[...,0] - q_xyz[:,None,0]
        if self.wrap_y: dy_ = self._delta_periodic(nb_xyz[...,1], q_xyz[:,None,1], Ly)
        else:           dy_ = nb_xyz[...,1] - q_xyz[:,None,1]
        dt_ = nb_xyz[...,2] - q_xyz[:,None,2]  # no wrap in time
        rel = torch.stack([dx_, dy_, dt_], dim=-1) * torch.exp(self.log_gammas)[None,None,:]
        return rel

    def forward(self, q_lin_idx, offsets_ijk):
        B = q_lin_idx.shape[0]
        nb_idx = gather_neighbors_periodic(q_lin_idx, self.k, offsets_ijk)
        q_xyz  = coords[q_lin_idx]     # (B,3)
        nb_xyz = coords[nb_idx]        # (B,k,3)
        rel    = self._relative_deltas(q_xyz, nb_xyz)          # (B,k,3)
        nb_vals = vals[nb_idx]                                  # (B,k,3)  <- η,u,v neighbors
        tokens = torch.cat([rel, nb_vals], dim=-1)              # (B,k,6)
        h = self.input_proj(tokens); h = self.encoder(h); h = h.mean(dim=1)
        out = self.head(h)                                      # (B,3)
        return out

# ----------------------
# Targets
# ----------------------
@torch.no_grad()
def batch_targets(q_lin_idx): return vals[q_lin_idx].to(device)  # (B,3)

def huber(x, delta=1.0): return torch.where(x.abs() <= delta, 0.5*x*x, delta*(x.abs()-0.5*delta))

# ----------------------
# SWE PINN residuals via autograd
# ----------------------
def pde_residual_autograd(model, q_lin_idx):
    xyt = coords[q_lin_idx].clone().detach().requires_grad_(True)  # (B,3)
    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)
    nb_idx = gather_neighbors_periodic(q_lin_idx, model.k, offsets_ijk)
    nb_xyz = coords[nb_idx]  # constants

    # tokens using variable xyt for query side
    if model.wrap:
        dxv = (nb_xyz[...,0] - xyt[:,None,0] + 0.5*Lx) % Lx - 0.5*Lx
        dyv = (nb_xyz[...,1] - xyt[:,None,1] + 0.5*Ly) % Ly - 0.5*Ly
    else:
        dxv = nb_xyz[...,0] - xyt[:,None,0]
        dyv = nb_xyz[...,1] - xyt[:,None,1]
    dtv = nb_xyz[...,2] - xyt[:,None,2]

    rel = torch.stack([dxv, dyv, dtv], dim=-1) * torch.exp(model.log_gammas)[None,None,:]
    nb_vals_tok = vals[nb_idx]  # (B,k,3)
    tokens = torch.cat([rel, nb_vals_tok], dim=-1)  # (B,k,6)

    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), \
         torch.cuda.amp.autocast(enabled=False):
        h = model.input_proj(tokens); h = model.encoder(h); h = h.mean(dim=1)
        pred = model.head(h)  # (B,3) -> [eta,u,v]

    eta_hat = pred[:,0]; u_hat = pred[:,1]; v_hat = pred[:,2]

    ones_eta = torch.ones_like(eta_hat)
    ones_u   = torch.ones_like(u_hat)
    ones_v   = torch.ones_like(v_hat)

    grads_eta = torch.autograd.grad(eta_hat, xyt, grad_outputs=ones_eta, create_graph=True)[0]
    grads_u   = torch.autograd.grad(u_hat,   xyt, grad_outputs=ones_u,   create_graph=True)[0]
    grads_v   = torch.autograd.grad(v_hat,   xyt, grad_outputs=ones_v,   create_graph=True)[0]

    eta_x, eta_y, eta_t = grads_eta[:,0], grads_eta[:,1], grads_eta[:,2]
    u_x,   u_y,   u_t   = grads_u[:,0],   grads_u[:,1],   grads_u[:,2]
    v_x,   v_y,   v_t   = grads_v[:,0],   grads_v[:,1],   grads_v[:,2]

    R_u  = u_t  + g * eta_x
    R_v  = v_t  + g * eta_y
    R_et = eta_t + H * (u_x + v_y)

    # stack residuals and return as a single tensor
    R = torch.stack([R_u, R_v, R_et], dim=-1)  # (B,3)
    return R

# ----------------------
# Early stopping (same)
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

# ----------------------
# Train setup (same schedules/optimizer)
# ----------------------
torch.set_float32_matmul_precision("high")
model = FieldFormerAutograd(d_in=6, d_model=64, nhead=4, num_layers=2, k_neighbors=128, d_ff=128, wrap=True).to(device)  # d_in=6
base_params = [p for n,p in model.named_parameters() if n != "log_gammas"]
optimizer = torch.optim.AdamW([
    {"params": base_params,        "lr": 3e-4, "weight_decay": 1e-4},
    {"params": [model.log_gammas], "lr": 3e-3, "weight_decay": 0.0},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
grad_clip = 1.0; max_epochs = 100; early = EarlyStopping(patience=10); mse = nn.MSELoss()

USE_GRAD_NORM_BALANCE = True
BAL_CLAMP = (0.5, 2.0)

LAMBDA_PDE_MAX = 1.0
LAMBDA_BC_MAX  = 0.05
PDE_WARMUP_EPOCHS = 5
PDE_RAMP_EPOCHS   = 20
def cosine_ramp(t: float): return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, t))))
def lambdas_for_epoch(epoch: int):
    if epoch <= PDE_WARMUP_EPOCHS: lam_pde = 0.0
    else: lam_pde = LAMBDA_PDE_MAX * cosine_ramp((epoch - PDE_WARMUP_EPOCHS) / max(1, PDE_RAMP_EPOCHS))
    lam_bc = min(LAMBDA_BC_MAX, 0.05 * max(lam_pde, 1e-6))
    return lam_pde, lam_bc
use_pde, use_bc, match_grad_bc = True, True, False

best_rmse = float("inf")
best_path = "/scratch/ab9738/fieldformer/model/ffag_swe_best.pt"

@torch.no_grad()
def eval_val_rmse(offsets_ijk):
    se_sum = 0.0; n_sum = 0
    for q_lin in dl_val:
        q_lin = q_lin.to(device)
        pred = model(q_lin, offsets_ijk)       # (B,3)
        tgt  = vals[q_lin]                    # (B,3)
        se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
        n_sum  += tgt.numel()
    return math.sqrt(se_sum / max(1, n_sum))

# Periodic BC on x for all 3 channels (mirror to y if desired)
def periodic_bc_loss(n_bc=1024, match_grad=False):
    j = torch.randint(0, Ny, (n_bc,), device=device)
    k = torch.randint(0, Nt, (n_bc,), device=device)
    yb, tb = y_t[j], t_t[k]
    x0, xL = x_t[0].expand_as(yb), x_t[-1].expand_as(yb)

    xyt0 = torch.stack([x0, yb, tb], dim=-1).requires_grad_(match_grad)
    xytL = torch.stack([xL, yb, tb], dim=-1).requires_grad_(match_grad)

    def nearest_lin(xyt):
        ix = torch.clamp(((xyt[:,0] - x_t[0]) / dx).round().long(), 0, Nx-1)
        iy = torch.clamp(((xyt[:,1] - y_t[0]) / dy).round().long(), 0, Ny-1)
        it = torch.clamp(((xyt[:,2] - t_t[0]) / dt).round().long(), 0, Nt-1)
        return ijk_to_lin(ix, iy, it)

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
        offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    def predict_at_xyt(xyt, lin_idx):
        nb_idx = gather_neighbors_periodic(lin_idx, model.k, offsets_ijk)
        nb_xyz = coords[nb_idx]
        if model.wrap:
            dxv = (nb_xyz[...,0] - xyt[:,None,0] + 0.5*Lx) % Lx - 0.5*Lx
            dyv = (nb_xyz[...,1] - xyt[:,None,1] + 0.5*Ly) % Ly - 0.5*Ly
        else:
            dxv = nb_xyz[...,0] - xyt[:,None,0]
            dyv = nb_xyz[...,1] - xyt[:,None,1]
        dtv = nb_xyz[...,2] - xyt[:,None,2]
        rel = torch.stack([dxv, dyv, dtv], dim=-1) * torch.exp(model.log_gammas)[None,None,:]
        nb_vals_tok = vals[nb_idx]  # (B,k,3)
        tokens = torch.cat([rel, nb_vals_tok], dim=-1)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), \
             torch.cuda.amp.autocast(enabled=False):
            h = model.input_proj(tokens); h = model.encoder(h); h = h.mean(dim=1)
            return model.head(h)  # (B,3)

    lin0, linL = nearest_lin(xyt0), nearest_lin(xytL)
    u0 = predict_at_xyt(xyt0, lin0)
    uL = predict_at_xyt(xytL, linL)

    loss = F.mse_loss(u0, uL)  # compares all 3 channels
    if match_grad:
        gx0 = torch.autograd.grad(u0.sum(dim=1), xyt0, torch.ones_like(u0[:,0]), create_graph=True)[0][:,0]
        gxL = torch.autograd.grad(uL.sum(dim=1), xytL, torch.ones_like(uL[:,0]), create_graph=True)[0][:,0]
        loss = loss + F.mse_loss(gx0, gxL)
    return loss

# ----------------------
# Training loop (unchanged flow)
# ----------------------
for epoch in range(1, max_epochs+1):
    model.train()
    total_loss = total_data = total_pde = total_bc = 0.0
    lam_pde, lam_bc = lambdas_for_epoch(epoch)

    with torch.no_grad():
        gam = torch.exp(model.log_gammas).detach().cpu().numpy()
    offsets_ijk = build_offset_table(k=model.k, gammas=gam, dx=dx, dy=dy, dt=dt)

    # optionally freeze gammas briefly
    GAM_FZ_EPOCHS = 3
    if epoch == 1: model.log_gammas.requires_grad_(False)
    elif epoch == GAM_FZ_EPOCHS + 1: model.log_gammas.requires_grad_(True)

    for q_lin in tqdm(dl, desc=f"Epoch {epoch:03d} [train]", leave=False):
        q_lin = q_lin.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(q_lin, offsets_ijk)   # (B,3)
            tgt  = vals[q_lin]                # (B,3)
            data_loss = F.mse_loss(pred, tgt)

            pde_loss = torch.tensor(0.0, device=device)
            bc_loss  = torch.tensor(0.0, device=device)
            if use_pde:
                subsample = q_lin[::8]
                R = pde_residual_autograd(model, subsample)   # (b,3)
                pde_loss = huber(R).mean()

            if use_bc and model.wrap:
                bc_loss = periodic_bc_loss(n_bc=512, match_grad=match_grad_bc)

            if USE_GRAD_NORM_BALANCE and use_pde and pde_loss.requires_grad:
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
        if not torch.isfinite(loss): continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()

        total_loss += float(loss)
        total_data += float(data_loss)
        total_pde  += float(pde_loss)
        total_bc   += float(bc_loss)

    model.eval()
    rmse = eval_val_rmse(offsets_ijk)
    scheduler.step(rmse)
    print(f"Epoch {epoch:03d} | train {total_loss/len(dl):.4f} "
          f"(data {total_data/len(dl):.4f}, pde {total_pde/len(dl):.4f}, bc {total_bc/len(dl):.4f}) | "
          f"val RMSE {rmse:.6f}")

    if rmse < best_rmse:
        best_rmse = rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_rmse,
            "gammas": model.log_gammas.detach().exp().cpu().numpy(),
            "config": {
                "variant": "fieldformer_autograd_swe",
                "Nx": Nx, "Ny": Ny, "Nt": Nt, "k_neighbors": model.k,
                "d_model": 64, "nhead": 4, "num_layers": 2,
                "dx": dx, "dy": dy, "dt": dt, "g": g, "H": H, "wrap": model.wrap,
            }
        }, best_path)
        print(f"✓ Saved new best to {best_path} (val RMSE {best_rmse:.6f})")

    early.step(rmse)
    if early.stopped:
        print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
        break

print("Done.")
