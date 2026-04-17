Hyperion 2 - SLURM Batch Job Instructions
==========================================

1. SSH into the HPC login node
   ssh aczd097@login2.hyperion2.city.ac.uk

2. Navigate to the project root
   cd /users/aczd097/git/vidiq-hpc

3. Pull latest changes
   git pull

4. Create outputs directory if it doesn't exist
   mkdir -p outputs

5. Submit the job
   sbatch hpc/train_multiclass.slurm

   First validate the default Qwen model access:
   sbatch hpc/check_model_access.slurm

   Then submit any training job:
   sbatch hpc/train_multiclass.slurm

6. Monitor the job
   squeue -u aczd097

7. Watch live output (replace JOBID with actual job ID)
   tail -f outputs/txtmulti_<JOBID>.o

8. Cancel if needed
   scancel <JOBID>

Notes
-----
- Partition: gpu-a100 (A100 80GB)
- Time limit: 4 hours
- Logs: outputs/txtmulti_<JOBID>.o and .e
- Results saved under: experiments/text_model/runs/<run-name>/
