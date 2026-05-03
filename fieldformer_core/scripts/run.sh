#!/bin/bash
#SBATCH --job-name=run-py
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_633_general

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/script.py [script args...]" >&2
  exit 2
fi

TARGET="$1"; shift || true

# Safety: we only support Python scripts (as requested)
case "$TARGET" in
  *.py) ;;
  *)
    echo "[error] target must be a .py file (got: $TARGET)" >&2
    exit 2
    ;;
esac

# Resolve to absolute path if possible
if [[ "$TARGET" != /* ]]; then
  TARGET="$(readlink -f "$TARGET")"
fi

if [[ ! -f "$TARGET" ]]; then
  echo "[error] target not found: $TARGET" >&2
  exit 2
fi

echo "[info] job ${SLURM_JOB_ID:-N/A} on node(s):"
scontrol show hostname "$SLURM_JOB_NODELIST" || true

echo "[info] target: ${TARGET}"

SIF="/scratch/ab9738/fieldformer/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
OVERLAY="/scratch/ab9738/fieldformer/overlay-25GB-500K.ext3"
SING_BIN="/share/apps/apptainer/bin/singularity"

# node-local runtime for Apptainer
RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}/${USER}_appt_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNTIME_BASE"/{tmp,cache,session}
export APPTAINER_TMPDIR="$RUNTIME_BASE/tmp"
export APPTAINER_CACHEDIR="$RUNTIME_BASE/cache"
export APPTAINER_SESSIONDIR="$RUNTIME_BASE/session"
export TMPDIR="$RUNTIME_BASE/tmp"
export XDG_RUNTIME_DIR="$RUNTIME_BASE/session"

# Build a safely-quoted python command line (so args survive)
py_cmd=(python -u "$TARGET")
for a in "$@"; do
  py_cmd+=("$a")
done

# Convert to a shell-escaped string for /bin/bash -lc
py_cmd_str=""
for a in "${py_cmd[@]}"; do
  py_cmd_str+=" $(printf '%q' "$a")"
done
py_cmd_str="${py_cmd_str# }"

"$SING_BIN" exec --nv \
  --fakeroot \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    set -e
    source /ext3/env.sh
    cd /scratch/ab9738/fieldformer/
    python gpu_burn.py &
    cd fieldformer_core/scripts/
    echo '[info] running on compute node:' \$(hostname)
    ${py_cmd_str}
  "
