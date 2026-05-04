#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${ROOT}/ablations/architecture/scripts"
RUN_SH="${ROOT}/ablations/architecture/scripts/run.sh"
LOG_DIR="${SCRIPT_DIR}/logs"

DRY_RUN=0
EXTRA_ARGS=()

EXPERIMENTS=(
  "ffag-nophys-polsparse ffag_polsparse_nophys_train.py"
  "ffag-mlp-heatsparse ffag_mlp_heatsparse_train.py"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [--dry-run] [-- script args...]

Submits all 6 sparse FieldFormer architecture ablation experiments using:
  sbatch ${RUN_SH} <train_script.py> [script args...]

Experiments:
  ffag-nophys-heatsparse
  ffag-nophys-swesparse
  ffag-nophys-polsparse
  ffag-mlp-heatsparse
  ffag-mlp-swesparse
  ffag-mlp-polsparse

Examples:
  $0 --dry-run
  $0
  $0 -- --epochs 20

Logs:
  ${LOG_DIR}/<experiment>-%j.out
  ${LOG_DIR}/<experiment>-%j.err
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
for exp in "${EXPERIMENTS[@]}"; do
  read -r exp_name script_name <<<"${exp}"
  target="${SCRIPT_DIR}/${script_name}"
  if [[ ! -f "${target}" ]]; then
    echo "[error] missing training script: ${target}" >&2
    exit 2
  fi

  cmd=(
    sbatch
    --job-name="${exp_name}"
    --output="${LOG_DIR}/${exp_name}-%j.out"
    --error="${LOG_DIR}/${exp_name}-%j.err"
    "${RUN_SH}"
    "${target}"
    "${EXTRA_ARGS[@]}"
  )

  printf '[launch:%s] ' "${exp_name}"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] ${submitted} jobs prepared."
else
  echo "[done] submitted ${submitted} jobs."
fi
