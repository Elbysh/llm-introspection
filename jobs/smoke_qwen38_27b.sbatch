#!/bin/bash
#SBATCH --job-name=smoke-qwen38
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/smoke-qwen38-%j.out
#SBATCH --error=logs/smoke-qwen38-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1
python "$1"
