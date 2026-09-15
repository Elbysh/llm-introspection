#!/bin/bash
# Run one Python script on a compute node. For checks too heavy for the login node,
# which kills long or fork-hungry processes, and that need no real GPU.
#SBATCH --job-name=py-cpu
#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --output=logs/py-cpu-%j.out
#SBATCH --error=logs/py-cpu-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python "$@"
