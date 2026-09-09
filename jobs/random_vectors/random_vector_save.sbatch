#!/bin/bash

#SBATCH --job-name=vector-save
#SBATCH --partition=prod40
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/vector-save-%j.out
#SBATCH --error=logs/vector-save-%j.err

set -euo pipefail


source ".venv/bin/activate"

cd data && PYTHONPATH=../code/utils uv run python ../code/utils/save_random_vectors.py \
  --save_dir saved_vectors/llama 

