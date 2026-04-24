#!/bin/bash
#SBATCH -D /users/aczd097/git/vidiq-hpc
#SBATCH --job-name=ie_emo1p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100_80g:1
#SBATCH -e outputs/%x_%j.e
#SBATCH -o outputs/%x_%j.o

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/users/aczd097/sharedscratch/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export VIDIQ_IMAGE_DATA_ROOT="${VIDIQ_IMAGE_DATA_ROOT:-/users/aczd097/archive/vidiq-hpc/data/image/emoset}"
source /users/aczd097/sharedscratch/venvs/main/bin/activate
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Start
start=$(date +%s)
mkdir -p outputs
mkdir -p "${VIDIQ_IMAGE_DATA_ROOT}"
mkdir -p "${HF_DATASETS_CACHE}"
echo "Job started at $(date)"
echo "Experiment: EmoSet Phase 1 (Full Image)"
echo "Node: $(hostname)"
echo "VIDIQ_IMAGE_DATA_ROOT: ${VIDIQ_IMAGE_DATA_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

python3 scripts/run_image_embeddings.py --config configs/emoset_phase1.json

# Duration + email
end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) - $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "${SLURM_JOB_NAME} (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
