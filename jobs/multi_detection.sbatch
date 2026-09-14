#!/bin/bash


#SBATCH --job-name=multi-detection
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/multi-detection-%j.out
#SBATCH --error=logs/multi-detection-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/gpfs/workdir/dainl/hf_cache
export HF_HUB_OFFLINE=1


# Narrowed to the alpha 2-5 "sweet spot" (coherence stays high and the
# bias-adjusted logit-contrast signal is where it's been found so far),
# with more trials per cell for power; --prompt_styles dropped "scale"
# since it's not used by the digit-logit-contrast measure.
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_detection.py \
  --k_values 0 1 2 3 4 \
  --conditions real random \
  --prompt_styles count \
  --dose_regime individual \
  --alphas 2 3 4 5 \
  --num_trials 40 \
  --vec_type avg \
  --output_dir ../plots
