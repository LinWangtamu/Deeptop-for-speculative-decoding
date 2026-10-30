"""
Sweep the learned DeepTOP thresholds over the serving state.

  python deeptop_threshold_sweep.py --ckpt deeptop_v5_best.pkl

Works for any checkpoint: the number of thresholds T and k_min are read from it
(v1/v4: T = 4, k in 0..4; v6: T = 6, k in 1..7).

For each state vector, the actor outputs thresholds theta_1 >= ... >= theta_T in s-space,
s = n / (n + 256). The policy uses k = k_min + #{j : theta_j > s - 0.01}; k_min is read from the
checkpoint (v5/v6: 1; v1/v4 checkpoints have no k_min and use 0). Threshold j is the switch from k = k_min + j to k = k_min + j - 1; it happens
at s_j = theta_j + 0.01, i.e. at queue length n_j = 256 * s_j / (1 - s_j). This script reports
n_j (in requests) for:
  (1) acceptance alpha in [0.60, 0.95], at several decode-batch sizes and contexts;
  (2) the sensitivity of each switch point to alpha, decode-batch size and context;
  (3) the two workloads, alpha = 0.84 (HumanEval) and 0.69 (MT-Bench).
Writes threshold_sweep.csv and prints a summary. No GPU or vLLM needed.
"""
import argparse
import csv

import numpy as np
import torch

N_SCALE, MAX_NUM_SEQS, CTX_SCALE, EPS = 256.0, 256.0, 240.0, 1e-2


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    n_layers = len([k for k in sd if k.startswith("fc.") and k.endswith(".weight")])
    n_in = sd["fc.0.weight"].shape[1]
    n_thr = sd[f"fc.{n_layers - 1}.weight"].shape[0]
    Ws = [sd[f"fc.{i}.weight"].numpy().astype(np.float64) for i in range(n_layers)]
    bs = [sd[f"fc.{i}.bias"].numpy().astype(np.float64) for i in range(n_layers)]

    def thresholds(n_dec, alpha, ctx):
        feats = [alpha, ctx / CTX_SCALE]
        if n_in == 3:
            feats = [min(n_dec, MAX_NUM_SEQS) / MAX_NUM_SEQS] + feats
        x = np.asarray(feats, dtype=np.float64)
        for i in range(n_layers):
            x = Ws[i] @ x + bs[i]
            if i < n_layers - 1:
                x = np.maximum(x, 0.0)
        base, gaps = x[0], np.logaddexp(0.0, x[1:])                # softplus
        logits = np.concatenate([[base], base + np.cumsum(gaps)])[::-1]
        return 1.0 / (1.0 + np.exp(-logits))                       # theta_1 >= ... >= theta_T

    k_min = int(sd["k_min"]) if "k_min" in sd else 0
    return thresholds, n_in, n_thr, k_min


def to_requests(theta):
    s = np.clip(theta + EPS, 0.0, 1.0 - 1e-9)
    return N_SCALE * s / (1.0 - s)                                 # n_j for each threshold j


def fmt(v):
    return "  >1e4" if v > 1e4 else f"{v:7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="deeptop_mk_actor_best.pkl")
    ap.add_argument("--out", default="threshold_sweep.csv")
    a = ap.parse_args()
    th, n_in, n_thr, k_min = load(a.ckpt)
    K_MAX = k_min + n_thr
    # threshold j (1..n_thr) is the switch from k = k_min + j down to k = k_min + j - 1
    key = {j: f"n_switch_{k_min + j}to{k_min + j - 1}" for j in range(1, n_thr + 1)}
    lab = {j: f"{k_min + j}->{k_min + j - 1}" for j in range(1, n_thr + 1)}
    order = list(range(n_thr, 0, -1))                              # longest k first: 4->3, 3->2, ...
    print(f"checkpoint {a.ckpt}: {n_in} state inputs "
          f"({'n_decoding, alpha, ctx' if n_in == 3 else 'alpha, ctx'}), "
          f"{n_thr} thresholds, k in {k_min}..{K_MAX}")

    rows = []
    alphas = np.round(np.arange(0.60, 0.951, 0.025), 3)
    n_decs = [16, 64, 128, 256] if n_in == 3 else [0]
    ctxs = [150, 200, 250, 300]
    for nd in n_decs:
        for ctx in ctxs:
            for al in alphas:
                n = to_requests(th(nd, al, ctx))
                rows.append({"n_decoding": nd, "ctx": ctx, "alpha": float(al),
                             **{key[j]: round(float(n[j - 1]), 2) for j in range(1, n_thr + 1)}})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n(1) switch points n_j (requests in system) vs alpha; "
          f"k = {k_min} + number of switch points above the current queue length")
    for nd in n_decs:
        for ctx in (200,):
            print(f"\n  n_decoding={nd}, ctx={ctx}")
            print("   alpha " + " ".join(f"{lab[j]:>8}" for j in order))
            for r in rows:
                if r["n_decoding"] == nd and r["ctx"] == ctx:
                    print(f"   {r['alpha']:.3f} " + " ".join(f"{fmt(r[key[j]]):>8}" for j in order))

    print("\n(2) sensitivity: range of each switch point over the whole grid")
    for j in order:
        v = np.array([r[key[j]] for r in rows])
        print(f"   {lab[j]}: min {fmt(v.min())}  median {fmt(np.median(v))}  max {fmt(v.max())}")
    for name, col, vals in (("alpha", "alpha", list(alphas)), ("n_decoding", "n_decoding", n_decs),
                            ("ctx", "ctx", ctxs)):
        if len(vals) < 2:
            continue
        lo = [r for r in rows if r[col] == vals[0]]
        hi = [r for r in rows if r[col] == vals[-1]]
        d = [np.median([r[key[j]] for r in hi]) - np.median([r[key[j]] for r in lo]) for j in order]
        rel = [np.median([r[key[j]] for r in hi]) / max(np.median([r[key[j]] for r in lo]), 1e-9) - 1
               for j in order]
        print(f"   {name} {vals[0]} -> {vals[-1]}: median shift of ("
              + ", ".join(lab[j] for j in order) + ") = "
              + ", ".join(f"{x:+.1f} ({y:+.0%})" for x, y in zip(d, rel)))

    print("\n(3) the two workloads (ctx=200), queue length at which k switches:")
    for al, name in ((0.84, "HumanEval"), (0.69, "MT-Bench")):
        for nd in n_decs:
            n = to_requests(th(nd, al, 200))
            print(f"   {name:9s} alpha={al} n_decoding={nd:3d}: "
                  + ", ".join(f"{lab[j]} at {fmt(n[j - 1]).strip()}" for j in order))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
