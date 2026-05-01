# FieldFormer

Physics-informed transformers for spatio-temporal field reconstruction from sparse sensors, with synthetic PDE datasets (heat, shallow-water) and a real-world air-pollution case. Includes strong baselines (SIREN, Fourier-MLP, SVGP) and evaluation scripts.

> **Core idea.** FieldFormer uses local transformer inference with learnable anisotropic neighborhoods and differentiable physics regularization to reconstruct full fields from sparse, irregular observations.

---

## Contents

```
fieldformer-main/
├─ README.md                      ← (this file)
├─ data/
│  ├─ heat_periodic.py            # synthetic heat dataset generator (periodic BC)
│  ├─ swe_periodic.py             # synthetic shallow-water generator (periodic BC)
│  ├─ pollution.py                # Delhi pollution case utilities (kriging, drivers)
│  ├─ *.ipynb                     # notebook variants of the above
│  ├─ *.npy / *.csv               # example pollution drivers (intensity grids, wind)
│  └─ heat_periodic.ipynb, swe_periodic.ipynb, pollution.ipynb
├─ model/
│  ├─ ff_fd_heat_train.py         # FieldFormer (FD residuals) – heat
│  ├─ ffag_heat_train.py          # FieldFormer-Autograd – heat
│  ├─ ffag_swe_train.py           # FieldFormer-Autograd – shallow-water (η,u,v)
│  ├─ fmlp_*_train.py             # Fourier-MLP baselines (heat/pol/swe)
│  ├─ siren_*_train.py            # SIREN baselines (heat/pol/swe)
│  ├─ svgp_*_train.py             # SVGP (GPyTorch) baselines (heat/pol/swe)
│  ├─ heat_eval.py                # evaluation—heat (sensor vs full-field, residuals)
│  ├─ swe_eval.py                 # evaluation—SWE (joint metrics)
│  └─ pol_eval.py                 # evaluation—pollution
├─ run_cpu_jupyter.sh             # example cluster launcher (Singularity)
├─ run_gpu_jupyter.sh             # example cluster launcher (GPU + overlay)
└─ torch_jupyter.sh               # example convenience script
```

---

## Setup

### 1) Python & CUDA

* Python ≥ 3.9 (3.10–3.12 tested)
* PyTorch with CUDA if you have a GPU (CPU also works)

### 2) Dependencies

Install with `pip`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick wheel for your CUDA
pip install gpytorch numpy pandas matplotlib tqdm pykrige pytz
```

> If you’re on CPU only, install the CPU PyTorch wheels; if you’re on a cluster, load the appropriate CUDA module or Singularity image first.

### 3) (Optional) Conda environment

```bash
conda create -n fieldformer python=3.11 -y
conda activate fieldformer
# then run the pip commands above
```

---

## Data

### Synthetic PDE datasets

Two generators live in `data/`:

* **Heat (periodic)**: `data/heat_periodic.py`

  * Solves (u_t = \alpha_x u_{xx} + \alpha_y u_{yy} + f(x,y,t)) on a periodic grid.
* **Shallow-Water (periodic)**: `data/swe_periodic.py`

  * Solves the 2D linearized shallow-water system for ((\eta, u, v)) on a periodic grid.

Run either to generate arrays and parameters (grids, spacings, PDE constants). The training scripts expect a **single NPZ** named like:

* `heat_periodic_dataset_sharp.npz` for heat
* `swe_periodic_dataset.npz` for SWE

Each NPZ should contain (by convention used in the training scripts):

* **Heat**: `u`, `x`, `y`, `t`, `params` (with names `["alpha_x","alpha_y","dx","dy","dt",...]`)
* **SWE**: `eta`, `u`, `v`, `x`, `y`, `t`, `params` (with names `["g","H","dx","dy","dt",...]`)

> The code currently **hard-codes absolute paths** to `/scratch/ab9738/fieldformer/data/*.npz` inside some scripts. See **Path configuration** below to point them to your data.

### Pollution (Delhi case)

`data/pollution.py` and companions (`*.csv`, `*_intensity_80x80.npy`) provide utilities and example drivers (traffic/industry/brick-kiln intensity maps, wind). These are used by the `*_pol_*` training/eval scripts.

---

## Path configuration (important)

Several training/eval scripts load datasets or save checkpoints via absolute paths under `/scratch/ab9738/fieldformer/...`.

You have three easy options:

1. **Edit the path constants** near the top of each script (search for `Config` or `np.load("...")`).
   Examples (actual lines exist in your repo):

   * `model/ffag_heat_train.py`: loads `"/scratch/ab9738/fieldformer/data/heat_periodic_dataset_sharp.npz"`
   * `model/ff_fd_heat_train.py`: same as above
   * `model/fmlp_heat_train.py`, `model/siren_heat_train.py`: same as above
   * `model/ffag_swe_train.py`, `model/fmlp_swe_train.py`, `model/siren_swe_train.py`: load `swe_periodic_dataset.npz`
   * `model/heat_eval.py`: checkpoint paths like `"/scratch/ab9738/fieldformer/model/ffag_heat_best.pt"`

2. **Symlink** your data/model directories to those paths:

```bash
mkdir -p /scratch/$USER/fieldformer/data /scratch/$USER/fieldformer/model
ln -s /your/local/path/heat_periodic_dataset_sharp.npz /scratch/$USER/fieldformer/data/heat_periodic_dataset_sharp.npz
```

3. **Export environment variables** and quickly replace in files:

```bash
export FF_BASE=/your/local/path/fieldformer
# then do a quick sed across the scripts if you want to template paths
```

---

## Training

All scripts are pure-Python (no CLI flags); edit the `Config` section at the top to change hyper-params, paths, seeds, etc. Splits default to **80/10/10** via a simple linear-index shuffler.

> **Tip:** the local transformer uses PyTorch’s scaled-dot-product attention. On some GPUs you may want to toggle `torch.backends.cuda.sdp_kernel` for stability/perf (many scripts already do this).

### Heat (periodic)

**FieldFormer-Autograd (physics residual via autograd)**

```bash
python model/ffag_heat_train.py
# checkpoint saved to .../model/ffag_heat_best.pt (path in script)
```

**FieldFormer (finite-difference residual)**

```bash
python model/ff_fd_heat_train.py
```

**Baselines**

```bash
python model/siren_heat_train.py
python model/fmlp_heat_train.py
python model/svgp_heat_train.py   # requires gpytorch; uses mini-batch ELBO
```

### Shallow-Water (periodic, joint 3-output)

```bash
python model/ffag_swe_train.py    # FieldFormer-AG for (η,u,v)
python model/siren_swe_train.py
python model/fmlp_swe_train.py
python model/svgp_swe_train.py
```

### Pollution

```bash
python model/ffag_pol_trainv2.py  # FieldFormer-AG variant for pollution
python model/fmlp_pol_train.py
python model/siren_pol_train.py
python model/svgp_pol_train.py
```

---

## Evaluation

Each evaluation script loads the dataset **exactly** like its training counterpart and reports:

* **RMSE / MAE on sensor test set** (the held-out indices)
* **Full-field RMSE / MAE** (all grid points)
* **Physics residual metrics** (optional), e.g.
  Heat: ( R = u_t - (\alpha_x u_{xx} + \alpha_y u_{yy}) - f ) via autograd
  SWE: residuals for ((\eta,u,v)) system

### Heat

```bash
python model/heat_eval.py
```

Edit in-file config to point to the best checkpoints:

* `ckpt_fieldformer`, `ckpt_svgp`, `ckpt_siren`, `ckpt_fmlp`

### Shallow-Water

```bash
python model/swe_eval.py
```

Reports joint metrics across the three outputs, plus (optionally) relative residuals if enabled in the code section.

### Pollution

```bash
python model/pol_eval.py
```

---

## Results (what to expect)

On the provided synthetic tasks, FieldFormer variants should match or exceed the baselines on both **sensor-set** and **full-field** errors, with **lower physics residuals** when the residual loss is enabled. SWE scripts evaluate **joint metrics** for ((\eta,u,v)).

(Exact numbers depend on seeds, grid sizes, residual weights, and sensor fractions; start with the defaults in each script.)

---

## Cluster notes (Singularity/Slurm)

The repo includes sample launchers (`run_cpu_jupyter.sh`, `run_gpu_jupyter.sh`, `torch_jupyter.sh`) illustrating:

* Using an overlay for a writeable conda/pip env
* Exposing a Jupyter port via reverse SSH
* Enabling `--nv` for GPU

Adapt the image path, overlay, and port for your cluster.
