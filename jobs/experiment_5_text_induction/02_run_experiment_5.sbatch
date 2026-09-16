#!/bin/bash
# Execute every fresh-context sham and concept-injection trial.
#
# The protocol's model is Llama-3.1-8B-Instruct, whose bf16 weights fit the
# 3g.40gb MIG slice prod40 hands out; jobs/experiment1.sbatch runs the same model
# on the same partition. prod40 caps memory at 60G (MaxMemPerNode=61440) and
# walltime at 24 h, and allocates the GPU implicitly -- a bare --gres=gpu:1 is
# rejected at submission. Switching this experiment to a larger model means
# prod80 plus --gres=gpu:nvidia_a100-sxm4-80gb:1 and --mem=120G.
#SBATCH --job-name=exp5-induction
#SBATCH --partition=prod40
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment_5-run-%j.out
#SBATCH --error=logs/experiment_5-run-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_5_text_induction.run_experiment_5 --config "${CONFIG}"
