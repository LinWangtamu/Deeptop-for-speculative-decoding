"""
Multi-level DeepTOP policy for dynk_serve.py.

  python dynk_serve.py ... --k-max 7 --max-num-seqs 256 --budget 384 --shrink-drafter \
      --policies module:rl_policy_deeptop_mk.py --policy-ckpt deeptop_v6_best.pkl
  (old v1/v4 checkpoints: k in 0..4, run with --k-max 4)

State, built exactly as MDP_Env._state():
  scalar = n_in_system / (n_in_system + N_SCALE)          n_in_system = running + waiting
  vector (3-input checkpoints) = [ n_decoding / MAX_NUM_SEQS, alpha, mean decoding context / CTX_SCALE ]
  vector (2-input checkpoints) = [ alpha, mean decoding context / CTX_SCALE ]
  alpha = accepted / verified over the last 20 request-steps.
Read from the checkpoint: input size, number of thresholds T, and k_min (stored as a buffer since v5;
missing in v1/v4 checkpoints, which use k_min = 0).
Action: k = k_min + #{ j : theta_j(vector) > scalar - 1e-2 }, theta_1 >= ... >= theta_T, k in k_min..k_min+T.
Constants must match env_config.py used for training.
"""
import numpy as np
import torch

N_SCALE = 256.0
MAX_NUM_SEQS = 256.0
CTX_SCALE = 240.0


def make_policy(args):
    sd = torch.load(args.policy_ckpt, map_location="cpu")
    n_layers = len([key for key in sd if key.startswith("fc.") and key.endswith(".weight")])
    n_in = sd["fc.0.weight"].shape[1]
    n_thr = sd[f"fc.{n_layers - 1}.weight"].shape[0]     # number of thresholds
    k_min = int(sd["k_min"]) if "k_min" in sd else 0     # v5/v6: 1; v1/v4: 0
    k_top = k_min + n_thr                                 # longest k the policy can choose
    # numpy forward: a 3-layer MLP is ~20 us per call in numpy, versus ~0.3 ms
    # per call plus a one-time ~1-2 s first-call stall in torch on CPU.
    Ws = [sd[f"fc.{i}.weight"].numpy().astype(np.float64) for i in range(n_layers)]
    bs = [sd[f"fc.{i}.bias"].numpy().astype(np.float64) for i in range(n_layers)]

    def thresholds(vec):
        x = vec
        for i in range(n_layers):
            x = Ws[i] @ x + bs[i]
            if i < n_layers - 1:
                x = np.maximum(x, 0.0)
        base, gaps = x[0], np.logaddexp(0.0, x[1:])       # softplus
        logits = np.concatenate([[base], base + np.cumsum(gaps)])[::-1]
        return 1.0 / (1.0 + np.exp(-logits))              # theta_1 >= ... >= theta_T

    def act(s):
        n = s["n_total"]
        scalar = n / (n + N_SCALE)
        feats = [s["alpha_deeptop"], s["ctx_mean_decoding"] / CTX_SCALE]
        if n_in == 3:
            feats = [min(s["n_decoding"], MAX_NUM_SEQS) / MAX_NUM_SEQS] + feats
        thr = thresholds(np.asarray(feats, dtype=np.float64))
        k = k_min + int((thr > scalar - 1e-2).sum())
        return min(k, s["k_max"])

    print(f"DeepTOP policy {args.policy_ckpt}: {n_in} state inputs, {n_thr} thresholds, "
          f"k in {k_min}..{k_top}", flush=True)
    act({"n_total": 0, "n_decoding": 0, "alpha_deeptop": 0.8, "ctx_mean_decoding": 0.0, "k_max": k_top})  # warm-up
    return act