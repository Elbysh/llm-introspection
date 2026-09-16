#!/bin/bash
# Fetch Qwen/Qwen3.8-27B into the shared Hugging Face cache.
# The login node kills a full-speed `hf download` of this repository (55.6 GB),
# so the transfer runs on a compute node instead. No GPU work happens here; the
# smallest MIG slice is requested only because every partition allocates one.
#SBATCH --job-name=dl-qwen38
#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --output=logs/download-qwen38-%j.out
#SBATCH --error=logs/download-qwen38-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# Conservative worker count: the default parallelism is what the login node killed.
hf download Qwen/Qwen3.8-27B --exclude "*.gguf" --max-workers 4

python - <<'PY'
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3.8-27B")
print("decoder blocks:", config.text_config.num_hidden_layers)
print("hidden size:   ", config.text_config.hidden_size)
PY
