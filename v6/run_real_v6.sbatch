#!/bin/bash
#SBATCH --job-name=real_v6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --nodelist=fc031
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=real_v6_%x_%j.log
# Real-system check of deeptop_v6_best.pkl on the k_max 7 engine. One engine per job.
# Usage (from the deeptop folder):
#   sbatch run_real_v6.sbatch SEED TAG POLICIES RATES [NUM_REQUESTS]
#   e.g. sbatch run_real_v6.sbatch 1 main "module:rl_policy_deeptop_mk.py,turbospec,turbospec_orig,fixed:1,fixed:2,fixed:3,fixed:4" "7.5,15,20,22.5,23.8"
SEED=${1:-1}
TAG=${2:-main}
POLICIES=${3:-"module:rl_policy_deeptop_mk.py,turbospec,turbospec_orig,fixed:1,fixed:2,fixed:3,fixed:4"}
RATES=${4:-"7.5,15,20,22.5,23.8"}
NREQ=${5:-2000}

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
vllm0112 || exit 1

# compile / kernel caches on node-local disk (stall precaution; never shared between jobs)
LOCAL=/tmp/$USER.$SLURM_JOB_ID        # node-local disk ($TMPDIR on this cluster points to /scratch)
mkdir -p "$LOCAL"
export VLLM_CACHE_ROOT=$LOCAL/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=$LOCAL/inductor_cache
export TRITON_CACHE_DIR=$LOCAL/triton_cache
export HF_HUB_OFFLINE=1
echo "node $(hostname)  seed $SEED  tag $TAG  local cache $LOCAL"
nvidia-smi --query-gpu=name,memory.used --format=csv
df -h /tmp | tail -1

trap 'rm -rf "$LOCAL"' EXIT
python dynk_serve.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompts-file humaneval_prompts.jsonl \
    --k-max 7 --shrink-drafter \
    --budget 384 --max-num-seqs 256 \
    --arrival poisson --rates "$RATES" \
    --num-requests "$NREQ" \
    --policies "$POLICIES" \
    --policy-ckpt deeptop_v6_best.pkl \
    --gc-mode freeze \
    --seed "$SEED" \
    --out-root "results_real_v6b/${TAG}_s${SEED}"
