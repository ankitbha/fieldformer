#!/usr/bin/env python3
# coding: utf-8
"""
SVGP baseline training for the periodic SWE dataset (x, y, t -> [eta, u, v]).
- Inputs normalized to [0,1] so all periodic kernel periods are 1.0
- Independent multitask SVGP (3 tasks) sharing one input kernel
- MultitaskGaussianLikelihood with per-task noises (has_task_noise=True)
- Z-score per output channel, denormalize for val RMSE
"""

import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gpytorch
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
)
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, ProductKernel, PeriodicKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from tqdm.auto import tqdm


# ----------------------
# Config
# ----------------------
class Config:
    data = "/scratch/ab9738/fieldformer/data/swe_periodic_dataset.npz"
    seed = 123
    train_frac = 0.80
    val_frac   = 0.10
    batch_size = 32768
    epochs = 100
    M = 4000
    lr = 1e-2
    lr_noise = 5e-3
    noise = 1e-3
    min_noise = 1e-6
    max_noise = 1e-1
    patience = 10
    save = "/scratch/ab9738/fieldformer/model/svgp_swe_best.pt"
    num_workers = 4
    num_tasks = 3  # eta,u,v


# ----------------------
# Dataset splitter
# ----------------------
class PeriodicDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123):
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        rng = np.random.default_rng(seed)
        all_lin = np.arange(Nx * Ny * Nt)
        rng.shuffle(all_lin)
        n_total = len(all_lin)
        n_train = int(train_frac * n_total)
        n_val   = int(val_frac * n_total)
        self.train_idx = torch.from_numpy(all_lin[:n_train]).long()
        self.val_idx   = torch.from_numpy(all_lin[n_train:n_train + n_val]).long()
        self.test_idx  = torch.from_numpy(all_lin[n_train + n_val:]).long()
        self.split = "train"
    def set_split(self, split):
        assert split in ["train", "val", "test"]
        self.split = split
    def __len__(self):
        return len(getattr(self, f"{self.split}_idx"))
    def __getitem__(self, idx):
        return getattr(self, f"{self.split}_idx")[idx]


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


# ----------------------
# SVGP Model (Independent multitask: eta, u, v)
# ----------------------
class SVGP_Multitask(ApproximateGP):
    def __init__(self, Z, num_tasks=3):
        M = Z.size(0)
        base_q  = CholeskyVariationalDistribution(M)
        base_vs = VariationalStrategy(self, Z, base_q, learn_inducing_locations=True)
        vs = IndependentMultitaskVariationalStrategy(base_vs, num_tasks=num_tasks)
        super().__init__(vs)

        self.mean_module = ConstantMean()
        kx, ky, kt = PeriodicKernel(), PeriodicKernel(), PeriodicKernel()
        for k in (kx, ky, kt):
            k.initialize(period_length=1.0)  # inputs in [0,1]
        self.covar_module = ScaleKernel(ProductKernel(kx, ky, kt))

    def forward(self, X01):
        mean_x  = self.mean_module(X01)
        covar_x = self.covar_module(X01)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ----------------------
# Main
# ----------------------
def main():
    args = Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load SWE data ---
    pack = np.load(args.data)
    eta_np = pack["eta"]; u_np = pack["u"]; v_np = pack["v"]
    x_np, y_np, t_np = pack["x"], pack["y"], pack["t"]
    Nx, Ny, Nt = eta_np.shape

    # Flatten + normalize coords to [0,1]
    XX, YY, TT = np.meshgrid(x_np, y_np, t_np, indexing="ij")
    coords_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1).astype(np.float32)
    vals_np   = np.stack([eta_np, u_np, v_np], axis=-1).reshape(-1, 3).astype(np.float32)

    x_min, x_max = x_np.min(), x_np.max()
    y_min, y_max = y_np.min(), y_np.max()
    t_min, t_max = (t_np.min(), t_np.max()) if Nt > 1 else (0.0, 1.0)
    coords01 = np.empty_like(coords_np, dtype=np.float32)
    coords01[:, 0] = (coords_np[:, 0] - x_min) / max(1e-12, (x_max - x_min))
    coords01[:, 1] = (coords_np[:, 1] - y_min) / max(1e-12, (y_max - y_min))
    coords01[:, 2] = (coords_np[:, 2] - t_min) / max(1e-12, (t_max - t_min))

    # Per-channel standardization
    y_mean = vals_np.mean(axis=0).astype(np.float32)         # (3,)
    y_std  = (vals_np.std(axis=0) + 1e-8).astype(np.float32) # (3,)
    vals_z = (vals_np - y_mean) / y_std

    X = torch.from_numpy(coords01).float()
    Y = torch.from_numpy(vals_z).float()                     # (N,3)

    # Split/loaders
    ds = PeriodicDataset(Nx, Ny, Nt, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    train_idx, val_idx = ds.train_idx, ds.val_idx

    def make_loader(idx, bs, shuffle):
        return DataLoader(idx, batch_size=bs, shuffle=shuffle, drop_last=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_train = make_loader(train_idx, args.batch_size, True)
    dl_val   = make_loader(val_idx,   max(4096, args.batch_size), False)

    # Inducing points
    M = min(args.M, train_idx.numel())
    with torch.no_grad():
        perm = torch.randperm(train_idx.numel())
        Z_init = X[train_idx[perm[:M]]].clone()

    # Likelihood (per-task noise) + model
    likelihood = MultitaskGaussianLikelihood(num_tasks=args.num_tasks, has_task_noise=True)
    model = SVGP_Multitask(Z_init, num_tasks=args.num_tasks)
    model, likelihood = model.to(device), likelihood.to(device)

    # Correct noise initialization:
    # - keep global .noise as a scalar
    # - set the 3-vector on .task_noise (NOT .noise)
    with torch.no_grad():
        base = float(np.clip(args.noise, args.min_noise, args.max_noise))
        likelihood.noise = base  # global scalar
        try:
            likelihood.task_noise = torch.full((args.num_tasks,), base, device=device)
        except AttributeError:
            # older/newer API fallback
            likelihood.initialize(task_noise=torch.full((args.num_tasks,), base, device=device))

    opt = torch.optim.Adam([
        {"params": model.parameters(),      "lr": args.lr},
        {"params": likelihood.parameters(), "lr": args.lr_noise},
    ])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_idx.numel())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-4)
    early = EarlyStopping(patience=args.patience)
    best_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train(); likelihood.train()
        total_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch:03d} [train]", total=len(dl_train), leave=False)
        for batch_lin in pbar:
            xb = X[batch_lin].to(device)  # (B,3)
            yb = Y[batch_lin].to(device)  # (B,3)

            opt.zero_grad(set_to_none=True)
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            pbar.set_postfix(elbo=f"{-loss.item():.3f}")

        # Validation (denormalized RMSE across η,u,v)
        model.eval(); likelihood.eval()
        se_sum, n_sum = 0.0, 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for batch_lin in dl_val:
                xb = X[batch_lin].to(device)
                y_true = Y[batch_lin].to(device)
                pred_norm = likelihood(model(xb)).mean  # (B,3) normalized

                y_std_t, y_mean_t = torch.from_numpy(y_std).to(device), torch.from_numpy(y_mean).to(device)
                pred = pred_norm * y_std_t + y_mean_t
                true = y_true    * y_std_t + y_mean_t

                se_sum += F.mse_loss(pred, true, reduction="sum").item()
                n_sum  += true.numel()

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
                "Z": Z_init.cpu(),  # inducing inputs in [0,1]
                "config": {
                    "Nx": Nx, "Ny": Ny, "Nt": Nt,
                    "train_frac": args.train_frac, "val_frac": args.val_frac,
                    "M": int(M), "y_mean": y_mean, "y_std": y_std,
                    "period_length": [1.0, 1.0, 1.0], "num_tasks": args.num_tasks,
                    "x_min": float(x_min), "x_max": float(x_max),
                    "y_min": float(y_min), "y_max": float(y_max),
                    "t_min": float(t_min), "t_max": float(t_max),
                }
            }, args.save)
            print(f"✓ Saved new best to {args.save} (val RMSE {best_rmse:.6f})")

        early.step(rmse)
        if early.stopped:
            print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
            break

    print("Done.")


if __name__ == "__main__":
    main()
