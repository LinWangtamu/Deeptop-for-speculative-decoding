"""
Evaluate a trained DeepTOP threshold policy against baselines.

Baselines: fixed stationary threshold, SmartSpec (goodput + periodic reset),
Fixed k=0, Fixed k=5. All run on the same SpecDecodingEnv dynamics.

Run after training:  python eval_deeptop_spec.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from MDP_Env import SpecDecodingEnv, spec_t_decode_step, _K_SPEC
from model import Actor

THRESHOLD = 10        # fixed stationary threshold baseline
RESETSTEPS = 20       # SmartSpec periodic reset


def acc_len(a, k):
    if k == 0: return 1.0
    if a >= 1: return float(k + 1)
    if a <= 0: return 1.0
    return (1 - a ** (k + 1)) / (1 - a)


def goodput(batch, k, a):
    lat = spec_t_decode_step(batch, k)
    return len(batch) * acc_len(a, k) / lat if lat > 0 else 0.0


def run_episode(policy, lam, seed=0, actor=None):
    """
    policy: 'deeptop' | 'threshold' | 'smartspec' | 'k0' | 'k5'
    Runs one full episode at a FIXED lambda; returns mean latency.
    """
    env = SpecDecodingEnv(seed=seed, duration=20.0, warmup=5.0,
                          lam_low=lam, lam_high=lam, true_alpha=0.7)
    # Gymnasium reset() -> (obs, info)
    s, _ = env.reset()
    skip_cnt = 0
    done = False
    steps = 0
    info = {}
    while not done and steps < 500000:
        if policy == 'deeptop':
            vector = s[1:]
            scalar = s[0]
            thr = actor.forward(torch.FloatTensor(vector)).item()
            a = 1 if thr > scalar else 0
        elif policy == 'threshold':
            # s[0] is batch normalized by env.max_num_seqs -> recover raw count.
            batch_size = s[0] * env.max_num_seqs
            a = 1 if batch_size < THRESHOLD else 0
        elif policy == 'smartspec':
            # The decode batch is the set of currently-decoding requests.
            # running is already capped at max_num_seqs, so no slicing needed.
            batch = env._decoding_reqs()
            k = max([0, _K_SPEC], key=lambda kk: goodput(batch, kk, env.at.value))
            if k == 0:
                skip_cnt += 1
                if skip_cnt % RESETSTEPS == 0:
                    k = _K_SPEC
            else:
                skip_cnt = 0
            a = 1 if k > 0 else 0
        elif policy == 'k0':
            a = 0
        elif policy == 'k5':
            a = 1
        # Gymnasium step() -> (obs, reward, terminated, truncated, info)
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        steps += 1
    return info.get('mean_latency')


def avg(policy, lam, seeds=5, actor=None):
    vals = []
    for sd in range(seeds):
        v = run_episode(policy, lam, seed=sd, actor=actor)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--actor", default="deeptop_spec_actor_average.pkl",
                    help="path to the trained actor .pkl (per ablation mode)")
    _args = _p.parse_args()

    actor = Actor(3, 1, [128, 128])
    actor.load_state_dict(torch.load(_args.actor,
                                     map_location="cpu"))
    actor.eval()

    # Sweep spans light load (below the ~6-7 stability boundary, where SD helps)
    # through overload (where the queue dominates), so the policies' crossover
    # behaviour is visible. Earlier sweeps started at 10 (all overload) and hid
    # the light-load regime where speculation pays off.
    lambdas = [2, 6, 20, 40, 80, 120, 160, 200]
    SEEDS = 5
    names = ["DeepTOP", "Threshold", "SmartSpec", "k=0", "k=5"]
    keys = ["deeptop", "threshold", "smartspec", "k0", "k5"]
    res = {n: [] for n in names}

    print(f"{'lam':>5}" + "".join(f"{n:>11}" for n in names))
    print("-" * 62)
    for lam in lambdas:
        row = []
        for n, k in zip(names, keys):
            v = avg(k, lam, seeds=SEEDS, actor=actor)
            res[n].append(v)
            row.append(v)
        print(f"{lam:>5}" + "".join(f"{v:>11.3f}" for v in row))

    k0arr = np.array(res["k=0"], dtype=float)
    styles = {
        "DeepTOP":   ("#7c3aed", "o", "-", 2.2),
        "Threshold": ("#16a34a", "^", "-", 2.0),
        "SmartSpec": ("#2563eb", "o", "-", 2.0),
        "k=0":       ("#9ca3af", "s", "--", 1.5),
        "k=5":       ("#dc2626", "D", "--", 1.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DeepTOP learned policy vs baselines  |  alpha=0.7", fontsize=11)

    ax = axes[0]
    for n, (c, m, ls, lw) in styles.items():
        ax.plot(lambdas, res[n], color=c, marker=m, ls=ls, lw=lw,
                ms=6, mfc="white", mew=1.6, label=n)
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("Mean E2E Latency (s)")
    ax.set_title("Mean E2E Latency")
    ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    for n, (c, m, ls, lw) in styles.items():
        if n == "k=0":
            continue
        ax.plot(lambdas, k0arr / np.array(res[n], dtype=float),
                color=c, marker=m, ls=ls, lw=lw, ms=6,
                mfc="white", mew=1.6, label=n)
    ax.axhline(1.0, color="#9ca3af", lw=1.2, ls="--", label="k=0 baseline")
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("Speedup vs k=0")
    ax.set_title("Speedup vs No-SD Baseline")
    ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("deeptop_eval.png", dpi=150, bbox_inches="tight")
    print("\nSaved: deeptop_eval.png")
