#!/bin/bash
srun \
  --nodes=1 \
  --cpus-per-task=4 \
  --mem=32GB \
  --time=6:00:00 \
  --partition=l40s_public \
  --gres=gpu:l40s:1 \
  --account=torch_pr_633_general \
  --pty bash -lc '

set -euo pipefail

SIF="/scratch/ab9738/stamp/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
OVERLAY="/scratch/ab9738/stamp/overlay-25GB-500K.ext3"
SING_BIN="/share/apps/apptainer/bin/singularity"

RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}/${USER}_appt_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNTIME_BASE"/{tmp,cache,session}
export APPTAINER_TMPDIR="$RUNTIME_BASE/tmp"
export APPTAINER_CACHEDIR="$RUNTIME_BASE/cache"
export APPTAINER_SESSIONDIR="$RUNTIME_BASE/session"
export TMPDIR="$RUNTIME_BASE/tmp"
export XDG_RUNTIME_DIR="$RUNTIME_BASE/session"

echo "[info] interactive job ${SLURM_JOB_ID:-N/A} on node:"
hostname
nvidia-smi

"$SING_BIN" exec --nv \
  --fakeroot \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    source /ext3/env.sh
    cd /scratch/ab9738/fieldformer
    echo \"[info] inside container on \$(hostname)\"
    echo \"[info] python: \$(which python)\"
    exec bash
  "
'