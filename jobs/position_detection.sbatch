#!/bin/bash
#SBATCH --job-name=position-detection
#SBATCH --partition=prod40
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/position-detection-%j.out
#SBATCH --error=logs/position-detection-%j.err

# Submit from the repository root. Use the cluster's GPU resource options.
set -euo pipefail
source .venv/bin/activate

run_dir="${POSITION_DETECTION_OUTPUT:-results/position_detection}"
resume_args=()
if [[ -f "$run_dir/manifest.json" ]]; then
    resume_args+=(--resume)
fi

python code/experiments/position_detection.py \
    --num-pairs 30 --seed 42 --concepts all \
    --layers 0 3 6 9 12 15 18 21 24 27 30 \
    --alphas 1 2 5 10 --output-dir "$run_dir" "${resume_args[@]}"
python code/analysis/compute_position_detection_accuracy.py --input-dir "$run_dir"
