#!/bin/bash

# Requires jobs/vector_save_qwen.sbatch to have completed first (needs
# data/saved_vectors/qwen/ populated).
#
# GPU sizing (--gres=gpu:2, --mem=128G) and time limit are conservative,
# unverified assumptions: a 27B dense model + vision tower in bf16 is
# roughly 55GB+ of weights alone, too much for a single 40GB-class GPU, and
# per-token compute is ~3x Llama-3.1-8B's. Adjust once you know this
# cluster's actual gpua100 per-node VRAM and see real Qwen throughput.

#SBATCH --job-name=calibration-qwen
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/calibration-qwen-%j.out
#SBATCH --error=logs/calibration-qwen-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/gpfs/workdir/dainl/hf_cache
export HF_HUB_OFFLINE=1


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/calibration.py \
  --model qwen \
  --alphas 0 1 2 4 6 8 12 16 \
  --num_trials 5 \
  --vec_type avg \
  --output_dir ../plots
