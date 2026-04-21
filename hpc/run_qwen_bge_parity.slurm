#!/bin/bash
#SBATCH -D /users/aczd097/git/vidiq-hpc
#SBATCH --job-name qpar001
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=0:30:00
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

CONFIG_PATH="${CONFIG_PATH:-experiments/text/multiclass/dair-ai-emotion/configs/qwen-bge-parity.json}"

# Start
start=$(date +%s)
mkdir -p outputs
echo "Job started at $(date)"
echo "Config path: ${CONFIG_PATH}"
echo "HF_HOME: ${HF_HOME}"
echo "This job reuses tracked Qwen fine-tune exports to build Qwen/BGE parity artifacts."

python experiments/text/multiclass/dair-ai-emotion/src/run_qwen_bge_parity.py \
    --config "${CONFIG_PATH}"

python experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_bge_parity.py \
    --config "${CONFIG_PATH}"

# Duration + email
end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) - $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "${SLURM_JOB_NAME} (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
