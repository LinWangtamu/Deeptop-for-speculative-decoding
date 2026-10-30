#!/bin/bash
# DeepTOP v6 retraining (CPU). Usage:  bash sweep.sh <parallel_jobs>    e.g. bash sweep.sh 12
# 12 runs = 3 seeds x 2 feature sets x 2 reward modes; each 1.5M steps, checkpoint every 50k.
cd "$(dirname "$0")"
P=${1:-12}
[ -f capacity.json ] || python capacity.py --procs "$P"
mkdir -p runs
BASE="--total_steps 1500000 --ckpt_every 50000 --alpha_low 0.60 --alpha_high 0.92 \
--rho_low 0.4 --rho_high 0.98 --p_over 0.1 --ovh_jitter 0.5 --duration 60 --warmup_s 10 \
--explore 0.1 --actor_start 100000 --logit_cap_hi 0.45 --logit_cap_lo 4.16 --avg_tau 1e-4"
{
for seed in 1 2 3; do
  for feats in all ac; do
    echo "f${feats}_avg_s${seed}|--seed $((400+seed)) --feats $feats --reward_mode average --rho_mode sgd --prate 1e-5 $BASE"
    echo "f${feats}_td5_s${seed}|--seed $((500+seed)) --feats $feats --reward_mode tdiscount --horizon 5 --prate 1e-5 $BASE"
  done
done
} | xargs -P "$P" -I{} bash -c 'line="{}"; tag="${line%%|*}"; args="${line#*|}"; mkdir -p runs/$tag; python main_DeepTOP_multik.py $args --ckpt_dir runs/$tag/ckpt --out runs/$tag/actor_final.pkl > runs/$tag/train.log 2>&1; echo "done $tag"'
echo "all runs finished"
