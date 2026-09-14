#!/bin/bash


#SBATCH --job-name=layer-ordering
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/layer-ordering-%j.out
#SBATCH --error=logs/layer-ordering-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/gpfs/workdir/dainl/hf_cache
export HF_HUB_OFFLINE=1


# Narrowed to the alpha 2-5 "sweet spot" where the bias-adjusted logit
# contrast has been found to peak; --num_trials bumped for power now that
# each cell also generates a matched sham (walltime bumped accordingly).
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/layer_ordering.py \
  --alphas 2 3 4 5 \
  --num_trials 60 \
  --label_assignments shallow_is_a shallow_is_b \
  --vec_type avg \
  --output_dir ../plots
