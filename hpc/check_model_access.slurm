#!/bin/bash
#SBATCH -D /users/aczd097/git/vidiq-hpc
#SBATCH --job-name qchk001
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH -e outputs/%x_%j.e
#SBATCH -o outputs/%x_%j.o

set -euo pipefail

# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/users/aczd097/sharedscratch/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
source /users/aczd097/sharedscratch/venvs/main/bin/activate

# Start
start=$(date +%s)
mkdir -p outputs
echo "Job started at $(date)"
echo "Model ref: Qwen/Qwen3-1.7B"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"

python - <<'PY'
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_ref = "Qwen/Qwen3-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python device: {device}")
print(f"Model ref exists locally: {Path(model_ref).expanduser().exists()}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_ref,
    trust_remote_code=True,
    output_hidden_states=True,
)
model.to(device)
model.eval()

inputs = tokenizer(
    ["sanity check for qwen model access"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=32,
)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, return_dict=True)

last_hidden = outputs.hidden_states[-1]
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Model hidden size: {model.config.hidden_size}")
print(f"Last hidden state shape: {tuple(last_hidden.shape)}")
print("Model access check passed.")
PY

# Duration + email
end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) - $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "${SLURM_JOB_NAME} (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
