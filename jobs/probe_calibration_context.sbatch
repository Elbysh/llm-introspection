#!/bin/bash
#SBATCH --job-name=calib-context-probe
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/calib-context-probe-%j.out
#SBATCH --error=logs/calib-context-probe-%j.err
set -euo pipefail

cd /raid/home/students/gegout_tho/sujet_15/llm-introspection
source .venv/bin/activate
python code/analysis/probe_calibration_context.py
