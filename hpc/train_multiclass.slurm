#!/bin/bash
#SBATCH -D /users/aczd097/git/vidiq-hpc
#SBATCH --job-name tmq3u001
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -e outputs/%x_%j.e
#SBATCH -o outputs/%x_%j.o

# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
source /users/aczd097/sharedscratch/venvs/main/bin/activate

# Start
start=$(date +%s)
mkdir -p outputs
echo "Job started at $(date)"

python experiments/text_model/train_multiclass.py \
    --data-root experiments/text/multiclass/dair-ai-emotion \
    --model-path models/qwen3-1.7B \
    --run-root experiments/text_model/runs \
    --run-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    --batch-size 8 \
    --num-epochs 3 \
    --max-length 256

# Duration + email
end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) - $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "${SLURM_JOB_NAME} (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
