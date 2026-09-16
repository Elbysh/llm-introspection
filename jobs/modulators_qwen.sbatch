#!/bin/bash

# Requires jobs/vector_save_qwen.sbatch to have completed first.
# GPU sizing is a conservative, unverified assumption -- see
# calibration_qwen.sbatch's comment for the rationale; adjust once known.
#
# --distances, --layers, --similarity_reference_layer below are the Llama
# job's values scaled 2x to preserve the same RELATIVE depth in a 64-layer
# network (Qwen) instead of reusing raw 32-layer indices, which would land
# at very different relative positions: distances double (extreme distance
# 63 = L-1 replaces 31), --layers 8/24 (25%/75% of 32) becomes 16/48
# (25%/75% of 64), --similarity_reference_layer 16 (50% of 32) becomes 32.

#SBATCH --job-name=modulators-qwen
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/modulators-qwen-%j.out
#SBATCH --error=logs/modulators-qwen-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/modulators.py \
  --model qwen \
  --experiments e4_distance e5_alpha_ratio e6_similarity \
  --alphas 1 2 3 4 5 6 7 \
  --num_trials 20 \
  --distances 1 2 4 8 16 32 63 \
  --placements early late \
  --layers 16 48 \
  --alpha_ratios 1 2 4 8 \
  --similarity_reference_layer 32 \
  --num_similarity_buckets 4 \
  --vec_type avg \
  --output_dir ../results/multisteering/experiment_11_modulators/raw
