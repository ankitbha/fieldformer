#!/bin/bash
#SBATCH --job-name=jupyter-ro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=5:00:00
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_633_general

set -euo pipefail

PORT="${PORT:-8882}"
SIF="/scratch/ab9738/stamp/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
OVERLAY="/scratch/ab9738/stamp/overlay-25GB-500K.ext3"
NOTEBOOK_DIR="/scratch/ab9738"
SING_BIN="/share/apps/apptainer/bin/singularity"

# node-local runtime for Apptainer
RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}/${USER}_appt_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNTIME_BASE"/{tmp,cache,session}
export APPTAINER_TMPDIR="$RUNTIME_BASE/tmp"
export APPTAINER_CACHEDIR="$RUNTIME_BASE/cache"
export APPTAINER_SESSIONDIR="$RUNTIME_BASE/session"
export TMPDIR="$RUNTIME_BASE/tmp"
export XDG_RUNTIME_DIR="$RUNTIME_BASE/session"

echo "[info] job ${SLURM_JOB_ID:-N/A} on node(s):"
scontrol show hostname "$SLURM_JOB_NODELIST"
SUBMIT_HOST="${SLURM_SUBMIT_HOST:-$(hostname)}"
echo "[info] submitting login host: ${SUBMIT_HOST}"
echo "[info] using port: ${PORT}"

# Jupyter with overlay read-only; all in-container writes go to tmpfs
"$SING_BIN" exec --nv \
  --fakeroot \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    set -e
    source /ext3/env.sh
    echo '[info] starting jupyter on compute node:' \$(hostname)
    jupyter lab --allow-root --no-browser --ip=127.0.0.1 --port ${PORT} \
      --notebook-dir ${NOTEBOOK_DIR} \
      --NotebookApp.token='' --NotebookApp.password=''
  "
