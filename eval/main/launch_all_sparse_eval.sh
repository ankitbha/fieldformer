#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_SH="${ROOT}/baselines/scripts/run.sh"
LOG_DIR="${ROOT}/eval/main/logs"
OUT_DIR="${ROOT}/eval/main/outputs"
TARGET="${ROOT}/eval/main/evaluate_all_sparse.py"

DRY_RUN=0
BATCH_SIZE=4096
SUMMARY_NAME="sparse_eval_all"
MAX_SPARSE_TEST=0
MAX_FULL_FIELD=0
EXTRA_ARGS=()

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
    --summary_name)
      SUMMARY_NAME="$2"
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
    --help|-h)
      cat <<EOF
Usage: $0 [--dry-run] [--batch_size N] [--output_dir DIR] [--summary_name NAME] [--max_sparse_test N] [--max_full_field N] [-- extra args]

Submits one 20-hour GPU job that evaluates:
  ffag, fmlp, fmlp_pinn, siren, siren_pinn, svgp, recfno, imputeformer, senseiver
on:
  heat, pol, swe

Outputs:
  ${OUT_DIR}/<model>-<dataset>.json
  ${OUT_DIR}/${SUMMARY_NAME}.csv
  ${OUT_DIR}/${SUMMARY_NAME}.jsonl

Logs:
  ${LOG_DIR}/sparse-eval-all-%j.out
  ${LOG_DIR}/sparse-eval-all-%j.err
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

cmd=(
  sbatch
  --job-name="sparse-eval-all"
  --time="20:00:00"
  --output="${LOG_DIR}/sparse-eval-all-%j.out"
  --error="${LOG_DIR}/sparse-eval-all-%j.err"
  "${RUN_SH}"
  "${TARGET}"
  --batch_size "${BATCH_SIZE}"
  --output_dir "${OUT_DIR}"
  --summary_name "${SUMMARY_NAME}"
  --max_sparse_test "${MAX_SPARSE_TEST}"
  --max_full_field "${MAX_FULL_FIELD}"
  "${EXTRA_ARGS[@]}"
)

printf '[launch] '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${cmd[@]}"
else
  echo "[dry-run] no job submitted."
fi
