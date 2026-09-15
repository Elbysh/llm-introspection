#!/bin/bash

# Requires jobs/vector_save_qwen.sbatch to have completed first.
# GPU sizing is a conservative, unverified assumption -- see
# calibration_qwen.sbatch's comment for the rationale; adjust once known.

#SBATCH --job-name=multi-id-sham-qwen
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/multi-id-sham-qwen-%j.out
#SBATCH --error=logs/multi-id-sham-qwen-%j.err

set -euo pipefail

source ".venv/bin/activate"

export HF_HOME=/gpfs/workdir/dainl/hf_cache
export HF_HUB_OFFLINE=1

cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_identification.py \
  --model qwen \
  --n_real 0 \
  --n_random 0 \
  --condition_name sham \
  --alphas 1 2 3 4 5 6 7 \
  --num_trials 30 \
  --vec_type avg \
  --output_dir ../plots
