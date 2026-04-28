#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVGP baseline for pollution (x, y, t -> u) with **no physics losses**.
- Uses your SplitDataset (returns linear indices) UNCHANGED.
- All settings are global variables (no CLI args).
- Targets are standardized on the train split; RMSE reported in physical units.

Run:
  python svgp_pol_train_nophys_globals_split.py
"""
import os, math, random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# ----------------------
# Global config (edit here)
# ----------------------
DATA_PATH = "/scratch/ab9738/fieldformer/data/pollution_dataset.npz"
OUT_PATH  = "/scratch/ab9738/fieldformer/model/svgp_pol_best.pt"

SEED   = 1337
DEVICE = "auto"           # 'cuda', 'cpu', or 'auto'

# SVGP hyperparams
M = 1024                   # number of inducing points
ARD_LENGTHSCALE_INIT = (0.2, 0.2, 0.1)  # for [x,y,t] in [0,1]
OUTPUTSCALE_INIT   = 1.0
NOISE_INIT         = 1e-2

# Training
EPOCHS      = 100
BATCH       = 65536
PATIENCE    = 10
LR          = 3e-3
LR_NOISE    = 1e-3
NUM_WORKERS = 2
PIN_MEMORY  = True
CLIP_NORM   = 2.0
JITTER      = 1e-4

# Splits (fractions)
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
# remaining is test (unused here)

# ----------------------
# Dataset splitter (linear indices) — EXACTLY as provided
# ----------------------
class SplitDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, train_frac=0.8, val_frac=0.1, seed=123):
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        rng = np.random.default_rng(seed)
        all_lin = np.arange(Nx * Ny * Nt)
        rng.shuffle(all_lin)
        n_total = all_lin.size
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        self.train_idx = torch.from_numpy(all_lin[:n_train]).long()
        self.val_idx   = torch.from_numpy(all_lin[n_train:n_train+n_val]).long()
        self.test_idx  = torch.from_numpy(all_lin[n_train+n_val:]).long()
        self.split = "train"
    def set_split(self, split):
        assert split in ["train","val","test"]; self.split = split
    def __len__(self):
        return getattr(self, f"{self.split}_idx").numel()
    def __getitem__(self, i):
        return getattr(self, f"{self.split}_idx")[i]

# ----------------------
# Utils & model
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

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, ard_lengthscale_init=(0.2,0.2,0.1), outputscale_init=1.0):
        M, D = inducing_points.shape
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=M)
        variational_strategy = VariationalStrategy(self, inducing_points,
                                                   variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=D))
        # sensible inits for [0,1]^3 inputs
        with torch.no_grad():
            ls = torch.tensor(list(ard_lengthscale_init), dtype=torch.float32, device=inducing_points.device)
            self.covar_module.base_kernel.lengthscale = ls
            self.covar_module.outputscale = torch.tensor(float(outputscale_init), dtype=torch.float32, device=inducing_points.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

@torch.no_grad()
def eval_rmse_denorm(model, likelihood, dl_idx, coords, y_std_all, y_mean, y_std, jitter=1e-4):
    """RMSE in original (de-standardized) units using batches of linear indices."""
    model.eval(); likelihood.eval()
    se_sum = 0.0; n_sum = 0
    with gpytorch.settings.fast_pred_var(True), gpytorch.settings.cholesky_jitter(jitter):
        for lin in dl_idx:
            xb = coords[lin]
            out = model(xb)
            pred_std = likelihood(out).mean  # standardized space
            pred = pred_std * y_std + y_mean
            tgt  = y_std_all[lin] * y_std + y_mean
            se_sum += F.mse_loss(pred, tgt, reduction="sum").item()
            n_sum  += lin.numel()
    return math.sqrt(se_sum / max(1, n_sum))

# ----------------------
# Main
# ----------------------

def main():
    # device
    if DEVICE == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(DEVICE)

    # seeds
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # ----- Load data -----
    pack = np.load(DATA_PATH)
    U = pack['U']  # (Nx,Ny,Nt)
    x = pack['x']; y = pack['y']; t = pack['t']
    Nx, Ny, Nt = U.shape

    XX, YY, TT = np.meshgrid(x, y, t, indexing='ij')
    coords_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1).astype(np.float32)  # (N,3)
    vals_np   = U.reshape(-1).astype(np.float32)

    coords = torch.from_numpy(coords_np).to(device)
    vals   = torch.from_numpy(vals_np).to(device)

    # Dataset & loaders (unchanged class)
    ds_train = SplitDataset(Nx, Ny, Nt, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED)
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    ds_val = SplitDataset(Nx, Ny, Nt, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED)
    ds_val.set_split("val")
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Standardize targets using TRAIN split only
    y_train_vec = vals[ds_train.train_idx.to(device)]
    y_mean = y_train_vec.mean(); y_std = y_train_vec.std().clamp_min(1e-6)
    y_std_all = (vals - y_mean) / y_std  # (N,)

    # Inducing points from TRAIN split
    M_eff = min(M, ds_train.train_idx.numel())
    perm = torch.randperm(ds_train.train_idx.numel(), device=device)
    Z = coords[ds_train.train_idx.to(device)[perm[:M_eff]]].clone().contiguous()

    # Model & likelihood
    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6)).to(device)
    model = SVGP(inducing_points=Z,
                 ard_lengthscale_init=ARD_LENGTHSCALE_INIT,
                 outputscale_init=OUTPUTSCALE_INIT).to(device)
    with torch.no_grad():
        likelihood.noise = torch.tensor(NOISE_INIT, device=device)

    # Optim
    params = [
        {"params": model.parameters(), "lr": LR},
        {"params": likelihood.parameters(), "lr": LR_NOISE},
    ]
    opt = torch.optim.Adam(params)
    mll = VariationalELBO(likelihood, model, num_data=ds_train.train_idx.numel())

    early = EarlyStopping(patience=PATIENCE)
    best_rmse = float('inf')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # ----------------------
    # Train
    # ----------------------
    for epoch in range(1, EPOCHS + 1):
        model.train(); likelihood.train()
        train_elbo_sum = 0.0

        for lin in tqdm(dl_train, desc=f"Epoch {epoch:03d} [train]", leave=False):
            xb = coords[lin.to(device)]
            yb_std = y_std_all[lin.to(device)]

            opt.zero_grad(set_to_none=True)
            with gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(JITTER):
                out = model(xb)
                loss = -mll(out, yb_std)
            if not torch.isfinite(loss):
                print('[warn] non-finite loss; skipping batch')
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            train_elbo_sum += (-loss.item())

        # Validation RMSE (de-standardized)
        rmse = eval_rmse_denorm(model, likelihood, dl_val, coords, y_std_all, y_mean, y_std, jitter=JITTER)
        print(f"Epoch {epoch:03d} | train ELBO {train_elbo_sum/len(dl_train):.4f} | val RMSE {rmse:.6f}")

        # Checkpoint
        if rmse < best_rmse - 1e-8:
            best_rmse = rmse
            torch.save({
                'epoch': epoch,
                'best_val_rmse': float(best_rmse),
                'model_state_dict': model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'inducing_points': Z.detach().cpu(),
                'config': {
                    'seed': int(SEED),
                    'M': int(M_eff),
                    'batch': int(BATCH),
                    'lr': float(LR),
                    'lr_noise': float(LR_NOISE),
                    'train_frac': float(TRAIN_FRAC),
                    'val_frac': float(VAL_FRAC),
                    'data': os.path.abspath(DATA_PATH),
                    'out': os.path.abspath(OUT_PATH),
                    'y_mean': float(y_mean.detach().cpu()),
                    'y_std': float(y_std.detach().cpu()),
                    'ard_lengthscale_init': tuple(float(x) for x in ARD_LENGTHSCALE_INIT),
                    'outputscale_init': float(OUTPUTSCALE_INIT),
                    'noise_init': float(NOISE_INIT),
                    'epochs': int(EPOCHS),
                    'patience': int(PATIENCE),
                    'clip_norm': float(CLIP_NORM),
                    'jitter': float(JITTER),
                    'device': str(device)
                }
            }, OUT_PATH)
            print(f"✓ Saved new best to {OUT_PATH} (val RMSE {best_rmse:.6f})")

        early.step(rmse)
        if early.stopped:
            print(f"⏹ Early stopping at epoch {epoch} (best RMSE {early.best:.6f})")
            break

    print('Done.')

if __name__ == '__main__':
    main()
