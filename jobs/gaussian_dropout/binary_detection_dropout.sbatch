#!/bin/bash


#SBATCH --job-name=binary-detection-dropout
#SBATCH --partition=prod40
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/binary-detection-dropout-%j.out
#SBATCH --error=logs/binary-detection-dropout-%j.err

set -euo pipefail


source ".venv/bin/activate"

# --strengths are dropout levels, not injection coefficients:
# dropout rate p in [0, 1), rescaled by 1/(1-p)
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/position_detection.py \
  --perturbation_type dropout \
  --layers 0 4 8 12 16 20 24 30 \
  --strengths 0.05 0.1 0.2 0.4 0.6 0.8 \
  --num_trials 10 \
  --output_dir ../plots
