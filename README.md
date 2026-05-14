# FieldFormer: Locality-Aware Transformers for Spatio-Temporal Modeling on Sparse Sensor Networks

This repository contains the code and experiment artifacts for FieldFormer, a locality-aware transformer for reconstructing spatio-temporal fields from sparse sensor networks. The repository is anonymized for review and includes the model implementation, baseline training scripts, evaluation scripts, generated datasets, checkpoints, and summary table utilities.

FieldFormer is trained only from sparse sensor observations. Evaluation reports held-out sparse sensor metrics and, for datasets with gridded ground truth, full-field reconstruction metrics.

## Repository Layout

```text
.
├── data/                         # Dataset generators and generated NPZ/CSV/NPY data
│   ├── heat_periodic.py          # Synthetic periodic heat generator
│   ├── swe_periodic.py           # Synthetic periodic shallow-water generator
│   ├── pollution.py              # Synthetic pollution utilities
│   ├── pollution_60.py
│   ├── build_gov_sensor_dataset.py
│   ├── build_gov_atm_dataset.py
│   └── *.npz, *.csv, *.npy       # Experiment data used by the scripts
├── fieldformer_core/
│   ├── models/                   # FieldFormer model code
│   ├── scripts/                  # FieldFormer sparse training scripts
│   └── checkpoints/              # FieldFormer checkpoints used by evaluation
├── baselines/
│   ├── models/                   # Baseline model implementations
│   ├── scripts/                  # Baseline sparse training scripts and launchers
│   └── checkpoints/              # Baseline checkpoints used by evaluation
├── original_baseline/            # Reference baseline code adapted by wrappers
├── eval/
│   ├── main/                     # Main evaluation runner and JSON outputs
│   ├── ablations/architecture/   # Ablation evaluation runner and outputs
│   └── make_sparse_summary_tables.py
├── ablations/architecture/       # FieldFormer architecture ablation training scripts
└── run_gpu_shell.sh              # Local cluster helper
```

## Environment

Python 3.10 or newer is recommended. The experiments use PyTorch and run most comfortably on a CUDA GPU.

Install the core dependencies with the PyTorch wheel that matches your system:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install numpy pandas scipy scikit-learn matplotlib tqdm gpytorch pykrige
```

Some baseline scripts also import code under `original_baseline/`; run commands from the repository root so local imports resolve correctly.

## Data

The current repository includes the code to generate data files expected by the training and evaluation scripts:

```text
data/heat_periodic_dataset_sharp.npz
data/heat_periodic_dataset_sharp_64.npz
data/swe_periodic_dataset.npz
data/swe_periodic_dataset_64.npz
data/pollution_dataset.npz
data/pollution_dataset_60.npz
data/gov_sensor_dataset.npz
data/gov_atm_dataset.npz
```

If the generated government sensor datasets need to be rebuilt, use:

```bash
python data/build_gov_sensor_dataset.py
python data/build_gov_atm_dataset.py
```

The synthetic heat and shallow-water generators are also in `data/`. They are script-style generator files with configuration constants near the top, including grid size, number of sensors, noise settings, and save path:

```bash
python data/heat_periodic.py
python data/swe_periodic.py
```

## Training

Training scripts use dataclass configs and accept simple CLI overrides through flags such as `--data`, `--save`, `--epochs`, `--batch_size`, `--val_batch_size`, `--lr`, and `--load_checkpoint`.

Several defaults were used on the original scratch path. For review, the most reliable pattern is to pass explicit local paths from the repository root.

### FieldFormer

```bash
python fieldformer_core/scripts/ffag_heatsparse_train.py \
  --data data/heat_periodic_dataset_sharp.npz \
  --save fieldformer_core/checkpoints/ffag_heatsparse_best.pt

python fieldformer_core/scripts/ffag_swesparse_train.py \
  --data data/swe_periodic_dataset.npz \
  --save fieldformer_core/checkpoints/ffag_swesparse_best.pt

python fieldformer_core/scripts/ffag_polsparse_train.py \
  --data data/pollution_dataset.npz \
  --save fieldformer_core/checkpoints/ffag_polsparse_best.pt
```

For quick smoke tests, reduce the budget, for example:

```bash
python fieldformer_core/scripts/ffag_heatsparse_train.py \
  --data data/heat_periodic_dataset_sharp.npz \
  --save /tmp/ffag_heatsparse_smoke.pt \
  --epochs 1 --batch_size 16 --val_batch_size 64
```

### Baselines

Baseline scripts are named as:

```text
baselines/scripts/<model>_<dataset>sparse_train.py
```

where model is one of `fmlp`, `siren`, `svgp`, `recfno`, `imputeformer`, or `senseiver`, and dataset includes `heat`, `pol`, `swe`, `govpol`, `atm`, `govpolsplit`, and `atmsplit` where supported.

Examples:

```bash
python baselines/scripts/fmlp_heatsparse_train.py \
  --data data/heat_periodic_dataset_sharp_64.npz \
  --save baselines/checkpoints/fmlp_heatsparse_best.pt

python baselines/scripts/siren_swesparse_train.py \
  --data data/swe_periodic_dataset_64.npz \
  --save baselines/checkpoints/siren_swesparse_best.pt

python baselines/scripts/svgp_polsparse_train.py \
  --data data/pollution_dataset_60.npz \
  --save baselines/checkpoints/svgp_polsparse_best.pt
```

The launcher `baselines/scripts/launch_all_sparse.sh` is a Slurm helper for selected baseline sweeps. Inspect it before use because the active dataset list is intentionally editable.

## Evaluation

The main evaluator loads datasets from `data/` and checkpoints from:

```text
fieldformer_core/checkpoints/
baselines/checkpoints/
ablations/architecture/checkpoints/
```

Run one dataset/model pair:

```bash
python eval/main/evaluate_all_sparse.py \
  --datasets heat \
  --models ffag fmlp siren svgp recfno imputeformer senseiver \
  --device cuda \
  --output_dir eval/main/outputs
```

Useful options:

```bash
--device cpu
--max_sparse_test 10000
--max_full_field 10000
--bootstrap_samples 0
```

For a fast CPU check:

```bash
python eval/main/evaluate_all_sparse.py \
  --datasets heat \
  --models ffag \
  --device cpu \
  --max_sparse_test 256 \
  --max_full_field 256 \
  --bootstrap_samples 0 \
  --output_dir /tmp/fieldformer_eval_smoke
```

Main evaluation outputs are JSON files named `<model>-<dataset>.json`.

## Ablations

Architecture ablations live under `ablations/architecture/scripts/`, with evaluation support under `eval/ablations/architecture/`. The included ablation checkpoints cover variants such as no physics loss, no position-guided filtering, and MLP-only variants.

To evaluate selected ablation outputs directly:

```bash
python eval/ablations/architecture/evaluate_all_sparse.py \
  --datasets heat pol swe \
  --models ffag_nophys ffag_npgf ffag_mlp \
  --device cuda \
  --output_dir eval/ablations/architecture/outputs
```

## Summary Tables

After running evaluation, regenerate the LaTeX summary tables with:

```bash
python eval/make_sparse_summary_tables.py
```

The default output is:

```text
eval/sparse_summary_tables.tex
```

## Cluster Notes

The repository includes Slurm/Apptainer helper scripts:

```text
fieldformer_core/scripts/run.sh
baselines/scripts/run.sh
eval/main/launch_all_sparse_eval.sh
eval/main/launch_selected_sparse_eval.sh
```

These scripts contain site-specific settings for partition, account, Singularity image, overlay path, and scratch directory. Reviewers should either edit those values for their cluster or run the Python commands above directly.

## Reproducibility Notes

- Default train/validation/test splits use fixed seeds in each script, usually `seed=123`.
- Sparse sensor observations are selected from the dataset NPZ files; full-field targets are used for evaluation, not for FieldFormer sparse training.
- Checkpoint metadata stores the config and observation key used for the run.
- Existing JSON outputs in `eval/main/outputs/` and `eval/ablations/architecture/outputs/` can be inspected without retraining.
