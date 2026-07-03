#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_SH="${ROOT}/baselines/scripts/run.sh"
LOG_DIR="${ROOT}/eval/main/logs"
OUT_DIR="${ROOT}/eval/main/timing_outputs"
TARGET="${ROOT}/eval/main/benchmark_sparse_inference_time.py"

DRY_RUN=0
BATCH_SIZE=4096
MAX_QUERIES=50000
WARMUP_BATCHES=5
TIMED_REPEATS=3
DATASET_FILTERS=()
MODEL_FILTERS=()
EXTRA_ARGS=()

EXPERIMENTS=(
  "ffag heat"
  "fmlp heat"
  "fmlp_ensemble heat"
  "fmlp_pinn heat"
  "siren heat"
  "siren_pinn heat"
  "svgp heat"
  "recfno heat"
  "imputeformer heat"
  "senseiver heat"

  "ffag pol"
  "fmlp pol"
  "fmlp_ensemble pol"
  "fmlp_pinn pol"
  "siren pol"
  "siren_pinn pol"
  "svgp pol"
  "recfno pol"
  "imputeformer pol"
  "senseiver pol"

  "ffag swe"
  "fmlp swe"
  "fmlp_ensemble swe"
  "fmlp_pinn swe"
  "siren swe"
  "siren_pinn swe"
  "svgp swe"
  "recfno swe"
  "imputeformer swe"
  "senseiver swe"

  "ffag atm"
  "fmlp atm"
  "fmlp_ensemble atm"
  "siren atm"
  "svgp atm"
  "recfno atm"
  "imputeformer atm"
  "senseiver atm"

  "ffag govpol"
  "fmlp govpol"
  "fmlp_ensemble govpol"
  "siren govpol"
  "svgp govpol"
  "recfno govpol"
  "imputeformer govpol"
  "senseiver govpol"

  "ffag atmsplit"
  "fmlp atmsplit"
  "fmlp_ensemble atmsplit"
  "siren atmsplit"
  "svgp atmsplit"
  "recfno atmsplit"
  "imputeformer atmsplit"
  "senseiver atmsplit"

  "ffag govpolsplit"
  "fmlp govpolsplit"
  "fmlp_ensemble govpolsplit"
  "siren govpolsplit"
  "svgp govpolsplit"
  "recfno govpolsplit"
  "imputeformer govpolsplit"
  "senseiver govpolsplit"
)

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra benchmark args]

Options:
  --dry-run, -n              Print the sbatch command without submitting.
  --batch_size N             Timed inference batch size. Default: ${BATCH_SIZE}
  --max_queries N            Max sparse test queries per repeat. Default: ${MAX_QUERIES}
  --warmup_batches N         Warmup batches before timing. Default: ${WARMUP_BATCHES}
  --timed_repeats N          Timed full passes over selected queries. Default: ${TIMED_REPEATS}
  --output_dir DIR           Directory for <model>-<dataset>.json outputs. Default: ${OUT_DIR}
  --datasets D1,D2           Restrict to comma-separated datasets.
  --models M1,M2             Restrict to comma-separated models.
  --help, -h                 Show this help.

Submits a Slurm array over valid sparse timing pairs:
  3 synthetic datasets x 10 models + 4 real/split datasets x 8 models = 62 jobs.

Outputs:
  ${OUT_DIR}/<model>-<dataset>.json

Logs:
  ${LOG_DIR}/sparse-timing-%A_%a.out
  ${LOG_DIR}/sparse-timing-%A_%a.err
EOF
}

split_csv_into() {
  local raw="$1"
  local -n dest="$2"
  IFS=',' read -r -a dest <<<"${raw}"
}

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
    --max_queries)
      MAX_QUERIES="$2"
      shift 2
      ;;
    --warmup_batches)
      WARMUP_BATCHES="$2"
      shift 2
      ;;
    --timed_repeats)
      TIMED_REPEATS="$2"
      shift 2
      ;;
    --output_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --datasets)
      split_csv_into "$2" DATASET_FILTERS
      shift 2
      ;;
    --models)
      split_csv_into "$2" MODEL_FILTERS
      shift 2
      ;;
    --help|-h)
      usage
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

TASKS=()
for exp in "${EXPERIMENTS[@]}"; do
  read -r model dataset <<<"${exp}"
  if [[ "${#DATASET_FILTERS[@]}" -gt 0 ]] && ! contains "${dataset}" "${DATASET_FILTERS[@]}"; then
    continue
  fi
  if [[ "${#MODEL_FILTERS[@]}" -gt 0 ]] && ! contains "${model}" "${MODEL_FILTERS[@]}"; then
    continue
  fi
  TASKS+=("${model}:${dataset}")
done

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "[error] no timing tasks selected" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"
ARRAY_MAX=$(("${#TASKS[@]}" - 1))
cmd=(
  sbatch
  --job-name="sparse-timing"
  --time="4:00:00"
  --array="0-${ARRAY_MAX}"
  --output="${LOG_DIR}/sparse-timing-%A_%a.out"
  --error="${LOG_DIR}/sparse-timing-%A_%a.err"
  "${RUN_SH}"
  "${TARGET}"
  --slurm_array
  --tasks "${TASK_CSV}"
  --batch_size "${BATCH_SIZE}"
  --max_queries "${MAX_QUERIES}"
  --warmup_batches "${WARMUP_BATCHES}"
  --timed_repeats "${TIMED_REPEATS}"
  --output_dir "${OUT_DIR}"
  "${EXTRA_ARGS[@]}"
)

echo "[launch] ${#TASKS[@]} timing tasks"
printf '[launch] '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${cmd[@]}"
else
  echo "[dry-run] no jobs submitted."
fi
