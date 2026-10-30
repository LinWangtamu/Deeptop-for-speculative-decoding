# DeepTOP v6 retraining (k in 1..7, position-dependent acceptance)

Requirements: Python 3 with torch (CPU), numpy, gymnasium (or gym).

Run everything on one CPU node:   sbatch run_v6.sbatch
or step by step:
1. nohup bash sweep.sh 12 > sweep.log 2>&1 &        (computes capacity.json first if missing)
2. nohup python select_all.py --procs 28 --thr > select_all.log 2>&1 &
3. python deeptop_threshold_sweep.py --ckpt deeptop_v6_best.pkl
Send back: select_all.log, convergence.txt, select_all.csv, capacity.json, deeptop_v6_best.pkl.

Deploy:  python dynk_serve.py ... --k-max 7 --policies module:rl_policy_deeptop_mk.py --policy-ckpt deeptop_v6_best.pkl

Changes from v4
- Simulator: acceptance of draft position i is alpha * 0.96^(i-1) (env_config.ACCEPT_DECAY), fitted on
  real untrimmed decode steps for k = 1..7; the old model assumed a constant alpha and overstated tau(7) by 17%.
  alpha in training and validation is the first-position acceptance.
- Actions k in {1..7} (K_MIN = 1, K_MAX = 7): 6 thresholds, k = 1 + #{j : theta_j > s}; no k = 0.
- k_min is stored in the checkpoint (actor buffer "k_min"); evaluation, the threshold sweep and the
  deployment policy read it. Old v1/v4 checkpoints have no such key and are treated as k in 0..4.
- Thresholds start log-spaced over 10..300 requests and are capped to n in [4, 400].
- Actor lr 1e-5, critic-only warm-up 100k updates, checkpoints save a Polyak-averaged actor (avg_tau 1e-4).
- select_all.py: threshold baselines up to k = 7; alpha-direction check requires a >= +10% geometric-mean
  shift of the switch points (below the cap) from alpha 0.70 to 0.84; writes deeptop_v6_best.pkl.
- Step-time correction unchanged from v4 (+0.6 ms per step, +1.9 ms when the step is not full).
