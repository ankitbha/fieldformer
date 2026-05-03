#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
SVGP baseline training for the periodic heat dataset (x, y, t -> u).
- Scales to large N via stochastic variational GP (mini-batch ELBO).
- Separable periodic kernel per dimension to honor wrap; inputs normalized to [0,1],
  so all kernel periods are fixed to 1.0 for stability.
- Matches FieldFormer dataset split (80/10/10) and path conventions.
- Saves best checkpoint by val RMSE.

Note: Requires PyTorch + GPyTorch. Train on GPU if available.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, ProductKernel, PeriodicKernel
# at top
from tqdm.auto import tqdm  # notebook-friendly


# In[ ]:


# ----------------------
# Config
# ----------------------
class Config:
    data = "/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"
    seed = 123
    train_frac = 0.80
    val_frac = 0.10
    batch_size = 32768
    epochs = 100
    M = 4000  # inducing points
    lr = 1e-2
    lr_noise = 5e-3
    noise = 1e-3
    min_noise = 1e-6
    max_noise = 1e-1
    patience = 10
    save = "svgp_heatsharp_best.pt"
    num_workers = 4


# In[ ]:


# ----------------------
# Dataset splitter
# ----------------------
class HeatPeriodicDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123):
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        rng = np.random.default_rng(seed)
        all_lin = np.arange(Nx * Ny * Nt)
        rng.shuffle(all_lin)
        n_total = len(all_lin)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        self.train_idx = torch.from_numpy(all_lin[:n_train]).long()
        self.val_idx = torch.from_numpy(all_lin[n_train:n_train + n_val]).long()
        self.test_idx = torch.from_numpy(all_lin[n_train + n_val:]).long()
        self.split = "train"

    def set_split(self, split):
        assert split in ["train", "val", "test"]
        self.split = split

    def __len__(self):
        if self.split == "train": return len(self.train_idx)
        if self.split == "val": return len(self.val_idx)
        return len(self.test_idx)

    def __getitem__(self, idx):
        if self.split == "train": return self.train_idx[idx]
        if self.split == "val": return self.val_idx[idx]
        return self.test_idx[idx]


# In[ ]:


# ----------------------
# Utilities
# ----------------------
@dataclass
class EarlyStopping:
    patience: int = 10
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False
    def step(self, metric: float):
        if metric < self.best - 1e-9:
            self.best = metric; self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True


# In[ ]:


# ----------------------
# SVGP Model
# ----------------------
class SVGPModel(ApproximateGP):
    def __init__(self, Z):
        M = Z.size(0)
        q = CholeskyVariationalDistribution(M)
        vs = VariationalStrategy(self, Z, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ConstantMean()
        kx, ky, kt = PeriodicKernel(), PeriodicKernel(), PeriodicKernel()
        # Initialize periods to 1.0 (inputs are normalized to [0,1])
        kx.initialize(period_length=1.0)
        ky.initialize(period_length=1.0)
        kt.initialize(period_length=1.0)
        self.covar_module = ScaleKernel(ProductKernel(kx, ky, kt))
    def forward(self, X):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(X), self.covar_module(X))


# In[ ]:


# ----------------------
# Main
# ----------------------
def main():
    args = Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(args.data)
    u_np = pack["u"]
    x_np, y_np, t_np = pack["x"], pack["y"], pack["t"]
    Nx, Ny, Nt = u_np.shape

    XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
    coords_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)
    vals_np = u_np.reshape(-1)

    x_min, x_max = x_np.min(), x_np.max()
    y_min, y_max = y_np.min(), y_np.max()
    t_min, t_max = (t_np.min(), t_np.max()) if Nt > 1 else (0.0, 1.0)
    coords01 = np.empty_like(coords_np)
    coords01[:, 0] = (coords_np[:, 0] - x_min) / max(1e-12, (x_max - x_min))
    coords01[:, 1] = (coords_np[:, 1] - y_min) / max(1e-12, (y_max - y_min))
    coords01[:, 2] = (coords_np[:, 2] - t_min) / max(1e-12, (t_max - t_min))

    y_mean, y_std = float(vals_np.mean()), float(vals_np.std() + 1e-8)
    vals_z = (vals_np - y_mean) / y_std

    X = torch.from_numpy(coords01).float()
    y = torch.from_numpy(vals_z).float()

    ds = HeatPeriodicDataset(Nx, Ny, Nt, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    train_idx, val_idx = ds.train_idx, ds.val_idx

    def make_loader(idx_tensor, bs, shuffle):
        return DataLoader(idx_tensor, batch_size=bs, shuffle=shuffle, drop_last=False,
                          num_workers=args.num_workers, pin_memory=True)

    dl_train = make_loader(train_idx, args.batch_size, True)
    dl_val = make_loader(val_idx, max(4096, args.batch_size), False)

    # ---- DROP-IN: resume from checkpoint (model-only) ----
    RESUME_FROM = "/scratch/ab9738/fieldformer/model/svgp_heat_best.pt"  # or None
    start_epoch = 1
    best_rmse = float("inf")
    Z_init = None

    if RESUME_FROM:
        print(f"[resume-svgp] Loading from {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location="cpu", weights_only=False)

        # 1) Make sure we use the SAME normalization as the checkpoint
        if isinstance(ckpt, dict) and "config" in ckpt:
            y_mean = float(ckpt["config"].get("y_mean", y_mean))
            y_std  = float(ckpt["config"].get("y_std",  y_std))
            # Recompute normalized targets with the checkpoint stats
            vals_z = (vals_np - y_mean) / y_std
            y = torch.from_numpy(vals_z).float()

        # 2) Inducing points: build model with the SAME M and Z as the checkpoint
        if "Z" in ckpt and ckpt["Z"] is not None:
            Z_init = ckpt["Z"].to(torch.float32)  # (M, 3) in [0,1]
        else:
            # Fallback: infer M from state_dict (variational_strategy.inducing_points)
            state_dict = ckpt.get("model_state_dict", ckpt)
            for k, v in state_dict.items():
                if k.endswith("variational_strategy.inducing_points"):
                    Z_init = v.detach().cpu().to(torch.float32)
                    break
        assert Z_init is not None, "Checkpoint missing inducing points; cannot resume SVGP safely."
        M = Z_init.size(0)
    else:
        M = min(Config.M, train_idx.numel())


    # Inducing points (either from ckpt or random subset)
    with torch.no_grad():
        if Z_init is None:
            perm = torch.randperm(train_idx.numel())
            Z_init = X[train_idx[perm[:M]]].clone()
        else:
            # Ensure Z_init is on the right device later
            pass

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SVGPModel(Z_init)  # SVGPModel takes Z at construction

    # Move to device
    model, likelihood = model.to(device), likelihood.to(device)

    # If resuming, load model/likelihood weights now
    if RESUME_FROM:
        state = ckpt.get("model_state_dict", ckpt)
        msg_m = model.load_state_dict(state, strict=True)
        try:
            if getattr(msg_m, "missing_keys", []) or getattr(msg_m, "unexpected_keys", []):
                print(f"[resume-svgp] model.load_state_dict: missing={msg_m.missing_keys}, unexpected={msg_m.unexpected_keys}")
        except Exception:
            pass

        if "likelihood_state_dict" in ckpt and ckpt["likelihood_state_dict"] is not None:
            try:
                likelihood.load_state_dict(ckpt["likelihood_state_dict"], strict=False)
            except Exception as e:
                print(f"[resume-svgp] likelihood load failed ({e}); continuing with freshly init likelihood.")

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_rmse   = float(ckpt.get("best_val_rmse", float("inf")))
        print(f"[resume-svgp] start_epoch={start_epoch}, best_val_rmse={best_rmse:.6f}")
    else:
        # Fresh run: set initial noise as before
        with torch.no_grad():
            likelihood.noise = torch.as_tensor(Config.noise).clamp(Config.min_noise, Config.max_noise).to(device)


    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": likelihood.parameters(), "lr": args.lr_noise},
    ])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_idx.numel())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-4)
    early = EarlyStopping(patience=args.patience)
    best_rmse = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); likelihood.train()
        total_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch:03d} [train]", total=len(dl_train), leave=False)
        for batch_lin in pbar:
            xb, yb = X[batch_lin].to(device), y[batch_lin].to(device)
            opt.zero_grad(set_to_none=True)
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            pbar.set_postfix(elbo=f"{-loss.item():.3f}")  # shows live per-batch metric

        model.eval(); likelihood.eval()
        se_sum, n_sum = 0.0, 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for batch_lin in dl_val:
                xb, y_true = X[batch_lin].to(device), y[batch_lin].to(device)
                pred = likelihood(model(xb)).mean
                pred_orig, true_orig = pred * y_std + y_mean, y_true * y_std + y_mean
                se_sum += F.mse_loss(pred_orig, true_orig, reduction="sum").item()
                n_sum += batch_lin.numel()
        rmse = math.sqrt(se_sum / max(1, n_sum))
        scheduler.step(rmse)
        print(f"Epoch {epoch:03d} | train ELBO {total_loss/len(dl_train):.4f} | val RMSE {rmse:.6f}")

        if rmse < best_rmse - 1e-9:
            best_rmse = rmse
            torch.save({
                "epoch": epoch,
                "best_val_rmse": best_rmse,
                "model_state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "Z": model.variational_strategy.inducing_points.detach().cpu(),  # (M, 3)
                "config": {
                    "Nx": Nx, "Ny": Ny, "Nt": Nt,
                    "M": int(model.variational_strategy.inducing_points.shape[0]),  # <-- fix
                    "batch_size": args.batch_size,
                    "lr": args.lr, "lr_noise": args.lr_noise,
                    "y_mean": y_mean, "y_std": y_std,
                    "train_frac": args.train_frac, "val_frac": args.val_frac,
                    "data": args.data,
                }
            }, args.save)
            print(f"✓ Saved new best to {args.save} (val RMSE {best_rmse:.6f})")

        early.step(rmse)
        if early.stopped:
            print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
            break

    print("Done.")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




