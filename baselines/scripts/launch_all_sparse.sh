#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT}/baselines/scripts"
RUN_SH="${SCRIPT_DIR}/run.sh"
LOG_DIR="${SCRIPT_DIR}/logs"

MODELS=(
  senseiver
)

DATASETS=(
  heat
  pol
  swe
)

DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [--dry-run] [-- script args...]

Submits all 18 sparse baseline experiments using:
  sbatch ${RUN_SH} <train_script.py> [script args...]

Models:   ${MODELS[*]}
Datasets: ${DATASETS[*]}

Examples:
  $0 --dry-run
  $0
  $0 -- --epochs 20

Logs:
  ${LOG_DIR}/<model>-<dataset>-%j.out
  ${LOG_DIR}/<model>-<dataset>-%j.err
EOF
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${RUN_SH}" ]]; then
  echo "[error] run.sh not found: ${RUN_SH}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

submitted=0
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    target="${SCRIPT_DIR}/${model}_${dataset}sparse_train.py"
    if [[ ! -f "${target}" ]]; then
      echo "[error] missing training script: ${target}" >&2
      exit 2
    fi

    exp_name="${model}-${dataset}"
    cmd=(
      sbatch
      --job-name="${exp_name}"
      --output="${LOG_DIR}/${exp_name}-%j.out"
      --error="${LOG_DIR}/${exp_name}-%j.err"
      "${RUN_SH}"
      "${target}"
      "${EXTRA_ARGS[@]}"
    )
    printf '[launch] '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" -eq 0 ]]; then
      "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] ${submitted} jobs prepared."
else
  echo "[done] submitted ${submitted} jobs."
fi
