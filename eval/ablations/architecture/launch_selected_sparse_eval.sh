#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_SH="${ROOT}/ablations/architecture/scripts/run.sh"
LOG_DIR="${ROOT}/eval/ablations/architecture/logs"
OUT_DIR="${ROOT}/eval/ablations/architecture/outputs"
TARGET="${ROOT}/eval/ablations/architecture/evaluate_all_sparse.py"

DRY_RUN=0
BATCH_SIZE=4096
MAX_SPARSE_TEST=0
MAX_FULL_FIELD=0
BOOTSTRAP_SAMPLES=1000
BOOTSTRAP_SEED=123
EXTRA_ARGS=()

EXPERIMENTS=(
  "ffag_nophys heat"
  "ffag_nophys pol"
  "ffag_nophys swe"
  "ffag_mlp heat"
  "ffag_mlp pol"
  "ffag_mlp swe"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --output_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --max_sparse_test)
      MAX_SPARSE_TEST="$2"
      shift 2
      ;;
    --max_full_field)
      MAX_FULL_FIELD="$2"
      shift 2
      ;;
    --bootstrap_samples)
      BOOTSTRAP_SAMPLES="$2"
      shift 2
      ;;
    --bootstrap_seed)
      BOOTSTRAP_SEED="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [--dry-run] [--batch_size N] [--output_dir DIR] [--max_sparse_test N] [--max_full_field N] [--bootstrap_samples N] [--bootstrap_seed N] [-- extra args]

Submits sparse architecture-ablation eval jobs with an 8-hour limit:
  ffag_nophys-heat
  ffag_nophys-pol
  ffag_nophys-swe
  ffag_mlp-heat
  ffag_mlp-pol
  ffag_mlp-swe

Outputs:
  ${OUT_DIR}/<model>-<dataset>.json

Logs:
  ${LOG_DIR}/sparse-arch-eval-selected-<model>-<dataset>-%j.out
  ${LOG_DIR}/sparse-arch-eval-selected-<model>-<dataset>-%j.err
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

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

for exp in "${EXPERIMENTS[@]}"; do
  read -r model dataset <<<"${exp}"
  name="${model}-${dataset}"
  cmd=(
    sbatch
    --job-name="sparse-arch-eval-${name}"
    --time="8:00:00"
    --output="${LOG_DIR}/sparse-arch-eval-selected-${name}-%j.out"
    --error="${LOG_DIR}/sparse-arch-eval-selected-${name}-%j.err"
    "${RUN_SH}"
    "${TARGET}"
    --batch_size "${BATCH_SIZE}"
    --output_dir "${OUT_DIR}"
    --datasets "${dataset}"
    --models "${model}"
    --max_sparse_test "${MAX_SPARSE_TEST}"
    --max_full_field "${MAX_FULL_FIELD}"
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
    --bootstrap_seed "${BOOTSTRAP_SEED}"
    "${EXTRA_ARGS[@]}"
  )

  printf '[launch:%s] ' "${name}"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
done

if [[ "${DRY_RUN}" -ne 0 ]]; then
  echo "[dry-run] no jobs submitted."
fi
