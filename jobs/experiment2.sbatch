#!/bin/bash
# Experiment 2: intervention-presence detection and false positives.
# One array task runs one decoder layer; block 31 remains unavailable.
# Usage: sbatch jobs/experiment2.sbatch [calibration-config] [run-name] [results-dir]
# Resume: RESUME=1 sbatch jobs/experiment2.sbatch [calibration-config] [run-name]
# Smoke test: LIMIT_TRIALS=8 sbatch --array=0 jobs/experiment2.sbatch \
#   configs/experiment_0_calibration/development_full.yaml smoke
#
#SBATCH --job-name=experiment2
#SBATCH --array=0-30%4
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment2-%A_%a.out
#SBATCH --error=logs/experiment2-%A_%a.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/development_full.yaml}
# Keep raw layer runs under results/experiment_2_presence/layers. A named run
# can still be supplied as the second argument when a separate namespace is wanted.
RUN_NAME=${2:-layers}
CALIBRATION_RESULTS=${3:-results/experiment_0_calibration}
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
LIMIT_ARGS=()
if [[ -n "${LIMIT_TRIALS:-}" ]]; then
  LIMIT_ARGS+=(--limit_trials "${LIMIT_TRIALS}")
fi

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/gpfs/workdir/${USER}/.cache/huggingface}"
export HF_HUB_OFFLINE=1

python code/experiments/experiment2_presence.py \
  --calibration_config "${CONFIG}" \
  --calibration_results "${CALIBRATION_RESULTS}" \
  --device cuda \
  --layers "${SLURM_ARRAY_TASK_ID}" \
  --run_name "${RUN_NAME}/layer_${SLURM_ARRAY_TASK_ID}" \
  --families concept random noise dropout \
  --matchings alpha z_sd z_mad \
  --num_pairs 5 \
  --num_random 3 \
  --num_noise 2 \
  --num_dropout 2 \
  --label_orders AB BA \
  --alpha_doses 0.25 0.5 1 2 4 8 16 32 64 128 \
  --z_doses 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48 \
  --bootstrap_iterations 1000 \
  --no_progress \
  "${LIMIT_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
