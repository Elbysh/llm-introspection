#!/bin/bash

# PREREQUISITE for every other *_qwen.sbatch job: computes and saves Qwen
# concept vectors to data/saved_vectors/qwen/. Must be run (and finish)
# before submitting any of the other Qwen jobs.
#
# Before submitting: Qwen/Qwen3.8-27B must already be in the shared HF
# cache. HF_HUB_OFFLINE=1 below means compute nodes cannot fetch it
# themselves -- pull it down first from a login node with internet access,
# e.g.:
#   HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache huggingface-cli download Qwen/Qwen3.8-27B
#
# Also worth a cheap sanity check before this (expensive) job, in the venv:
#   python -c "from transformers import Qwen3_5ForConditionalGeneration"
# Qwen/Qwen3.8-27B is a native vision-language ConditionalGeneration model
# (confirmed via its config.json), not a plain causal LM -- it needs a
# transformers release with qwen3_5 support merged; see
# code/utils/model_registry.py for the full rationale.

#SBATCH --job-name=vector-save-qwen
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/vector-save-qwen-%j.out
#SBATCH --error=logs/vector-save-qwen-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1


cd data && PYTHONPATH=../code/utils uv run python ../code/utils/save_vectors.py \
  --model qwen \
  --device auto
