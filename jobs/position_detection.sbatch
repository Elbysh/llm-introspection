#!/bin/bash
#SBATCH --job-name=position-detection
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-position-detection-%j.out
#SBATCH --error=runs/slurm-position-detection-%j.err

# Submit from the canonical repository root on Ruche.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

run_dir="${POSITION_DETECTION_OUTPUT:-runs/position_detection_alpha_1_20}"
python code/experiments/position_detection.py \
    --num-pairs 30 --seed 42 --concepts all \
    --layers 0 3 6 9 12 15 18 21 24 27 30 \
    --alphas {1..20} --output-dir "$run_dir"
python code/analysis/compute_position_detection_accuracy.py --input-dir "$run_dir"
