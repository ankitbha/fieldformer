#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT}/baselines/scripts"
RUN_SH="${SCRIPT_DIR}/run.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
CKPT_DIR="${ROOT}/baselines/checkpoints/fmlp_ensemble"

DATASETS=(heat pol swe govpol atm govpolsplit atmsplit)
SEEDS=(101 102 103 104 105)
SPLIT_SEED=123
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --datasets)
      shift
      DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --split-seed|--split_seed)
      SPLIT_SEED="$2"
      shift 2
      ;;
    --checkpoint-dir|--checkpoint_dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [--dry-run] [--datasets heat pol swe govpol atm govpolsplit atmsplit] [--seeds 101 102 103 104 105] [--split-seed 123] [--checkpoint-dir DIR] [-- script args...]

Submits one data-only FMLP ensemble member per dataset/seed using:
  sbatch ${RUN_SH} <fmlp_dataset_train.py> --seed <seed> --split_seed <split_seed> --save <member_checkpoint> --pinn false --lambda_phys 0 --lambda_bc 0 --lambda_sponge 0 --lambda_rad 0 [script args...]

Default checkpoint paths:
  ${CKPT_DIR}/fmlp_<dataset>sparse_seed<seed>_best.pt

Logs:
  ${LOG_DIR}/fmlp-ensemble-<dataset>-seed<seed>-%j.out
  ${LOG_DIR}/fmlp-ensemble-<dataset>-seed<seed>-%j.err
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

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "[error] no datasets provided" >&2
  exit 2
fi
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "[error] no seeds provided" >&2
  exit 2
fi
if [[ ! -f "${RUN_SH}" ]]; then
  echo "[error] run.sh not found: ${RUN_SH}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

submitted=0
for dataset in "${DATASETS[@]}"; do
  target="${SCRIPT_DIR}/fmlp_${dataset}sparse_train.py"
  if [[ ! -f "${target}" ]]; then
    echo "[error] missing training script: ${target}" >&2
    exit 2
  fi
  for seed in "${SEEDS[@]}"; do
    exp_name="fmlp-ensemble-${dataset}-seed${seed}"
    save_path="${CKPT_DIR}/fmlp_${dataset}sparse_seed${seed}_best.pt"
    cmd=(
      sbatch
      --job-name="${exp_name}"
      --output="${LOG_DIR}/${exp_name}-%j.out"
      --error="${LOG_DIR}/${exp_name}-%j.err"
      "${RUN_SH}"
      "${target}"
      --seed "${seed}"
      --split_seed "${SPLIT_SEED}"
      --save "${save_path}"
      --pinn false
      --lambda_phys 0
      --lambda_bc 0
      --lambda_sponge 0
      --lambda_rad 0
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
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] ${submitted} jobs prepared."
else
  echo "[done] submitted ${submitted} jobs."
fi
