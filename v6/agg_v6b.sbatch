#!/bin/bash
#SBATCH --job-name=agg_v6b
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=agg_v6b_%j.log
# Runs after the 11 real_v6 jobs (submitted with --dependency): summarize results_real_v6b.
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
vllm0112 || exit 1
python agg_real.py results_real_v6b > agg_real_b.txt
