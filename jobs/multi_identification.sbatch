#!/bin/bash


#SBATCH --job-name=multi-id-single-concept
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/multi-id-single-concept-%j.out
#SBATCH --error=logs/multi-id-single-concept-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1

# One condition per job (paired with multi_identification_two_concepts.sbatch,
# _concept_plus_random.sbatch, _sham.sbatch): --run_all_conditions chains all
# four conditions in a single process, and at real full-scale --num_trials
# this takes ~4h per condition (~16h total), well past any reasonable single
# job's walltime. Running the four conditions as separate, parallel jobs
# instead keeps each one comfortably inside a single walltime budget.
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_identification.py \
  --n_real 1 \
  --n_random 0 \
  --condition_name single_concept \
  --alphas 1 2 3 4 5 6 7 \
  --num_trials 30 \
  --vec_type avg \
  --output_dir ../plots
