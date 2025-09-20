#!/bin/bash
#SBATCH --job-name=fmlp-heat-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/ab9738/fieldformer/logs/%x_%j.out
#SBATCH -e /scratch/ab9738/fieldformer/logs/%x_%j.err

set -euo pipefail

# --- Args ---
if [ $# -lt 1 ]; then
  echo "Usage: sbatch $0 <script.py> [script args...]"
  exit 1
fi

PY_SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

# --- Paths & config ---
SIF="/scratch/ab9738/fieldformer/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
OVERLAY="/scratch/ab9738/fieldformer/overlay-25GB-500K.ext3"
WORKDIR="/scratch/ab9738/fieldformer"
SING_BIN="/share/apps/apptainer/bin/singularity"

# Node-local runtime areas for Apptainer
RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}/${USER}_appt_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNTIME_BASE"/{tmp,cache,session}
export APPTAINER_TMPDIR="$RUNTIME_BASE/tmp"
export APPTAINER_CACHEDIR="$RUNTIME_BASE/cache"
export APPTAINER_SESSIONDIR="$RUNTIME_BASE/session"
export TMPDIR="$RUNTIME_BASE/tmp"
export XDG_RUNTIME_DIR="$RUNTIME_BASE/session"

# Helpful CPU/GPU env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[info] job ${SLURM_JOB_ID:-N/A} on node(s):"
scontrol show hostname "$SLURM_JOB_NODELIST"

# Ensure output dirs
mkdir -p "$WORKDIR/logs" "$WORKDIR/checkpoints" "$WORKDIR/runs"

# --- Run inside the same container+overlay setup ---
"$SING_BIN" exec --nv \
  --writable-tmpfs \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    set -e
    cd '${WORKDIR}'
    source /ext3/env.sh
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH

    echo '[info] python:' \$(which python)
    echo '[info] node:' \$(hostname)
    echo '[info] running: python ${PY_SCRIPT} ${SCRIPT_ARGS}'

    python '${PY_SCRIPT}' ${SCRIPT_ARGS}
  "
