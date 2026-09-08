#!/bin/bash



#SBATCH --job-name=compute-baseline
#SBATCH --partition=prod40
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/compute-baseline-%j.out
#SBATCH --error=logs/compute-baseline-%j.err

set -euo pipefail


source ".venv/bin/activate"

# Pass the .out file produced by jobs/binary_detection.sbatch, e.g.
#   sbatch jobs/compute_baseline.sbatch logs/binary-detection-1234.out
LOG_FILE="${1:?usage: sbatch jobs/compute_baseline.sbatch <binary-detection log .out>}"
LOG_FILE="$(realpath "$LOG_FILE")"

cd data && PYTHONPATH=../code/utils uv run python \
  ../code/analysis/compute_position_detection_accuracy.py \
  --output_file "$LOG_FILE" \
  --compute_baseline \
  --baseline_trials 30