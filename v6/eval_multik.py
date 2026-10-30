"""
Evaluate multi-level DeepTOP against baselines in the v2 simulator, or pick the best checkpoint.

  python eval_multik.py --actor deeptop_mk_actor_best.pkl            # evaluation (seeds 0..4)
  python eval_multik.py --select                                     # checkpoint selection (seeds 100, 101)

Policies (all on the same SD engine unless noted):
  DeepTOP-MK   learned multi-level thresholds, k in {0..K_MAX}
  Goodput      TurboSpec-style: argmax_k expected accepted tokens / predicted step time of the
               current decode batch (budget-aware plan, simulator's own cost model)
  k=1..K_MAX   fixed speculation length
  k=0          pure no-SD engine (no drafter)
  Best fixed   min over fixed k at each arrival rate (an oracle that knows the load in advance)
Metric: steady-state mean latency (Little's law over [warmup, duration)), as in eval_deeptop_spec.py.
"""
import argparse
import glob
import re
import shutil
from multiprocessing import Pool

import numpy as np
import torch

import env_config as C
from model_multik import ThresholdActor, load_actor

LAMBDAS = [4, 8, 12, 16, 20, 24, 26, 28, 32, 36, 40]
VAL_LAMBDAS = [8, 20, 24, 28, 32, 36]
FIXED = list(range(0, C.K_MAX + 1))


def acc_len(a, k):
    a = min(max(a, 1e-3), 0.999)
    return 1.0 if k == 0 else (1 - a ** (k + 1)) / (1 - a)


def goodput_k(env):
    batch = env._decoding_reqs()
    if not batch:
        return C.K_MAX
    a = env.at.value
    best_k, best = 0, -1.0
    for k in range(C.K_MAX + 1):
        env.k_spec = max(k, 1)
        plan = env.decode_query_plan(batch, k)
        acc = sum(acc_len(a, n - 1) for n in plan if n > 0)
        lat = env._unified_step_latency(batch, [], k, decode_query=plan, sd_engine=True)
        if acc / lat > best:
            best, best_k = acc / lat, k
    return best_k


def run_episode(job):
    policy, lam, seed, alpha, actor_path = job
    actor = None
    if policy == "deeptop":
        actor, feat = load_actor(actor_path, C.K_MAX)
    env = C.make_env(seed=seed, lam_low=lam, lam_high=lam, alpha=alpha)
    s, _ = env.reset()
    done, info, ks = False, {}, []
    while not done:
        if policy == "deeptop":
            k, sd = actor.act(feat(s), s[0]), True       # k_min is stored in the checkpoint
        elif policy == "goodput":
            k, sd = goodput_k(env), True
        elif policy.startswith("thr:"):         # "thr:klo:khi:m": khi while n_in_system <= m, else klo
            _, klo, khi, m = policy.split(":")
            k, sd = (int(khi) if env._num_in_system() <= float(m) else int(klo)), True
        else:                                   # "fixed:K"
            k = int(policy.split(":")[1])
            sd = k > 0                          # k=0 -> pure no-SD engine
        ks.append(k)
        s, _, term, trunc, info = C.env_step(env, k, sd_engine=sd)
        done = term or trunc
    return policy, lam, seed, info.get("mean_latency_steady"), float(np.mean(ks))


def evaluate(policies, lambdas, seeds, alpha, actor_path, procs):
    jobs = [(p, l, s, alpha, actor_path) for p in policies for l in lambdas for s in seeds]
    with Pool(procs) as pool:
        out = pool.map(run_episode, jobs, chunksize=1)
    res = {}
    for p, l, s, lat, km in out:
        res.setdefault((p, l), []).append((lat, km))
    return {key: (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
            for key, vals in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="deeptop_mk_actor.pkl")
    ap.add_argument("--select", action="store_true", help="choose the best ckpt_mk/*.pkl on validation seeds")
    ap.add_argument("--alpha", type=float, default=C.EVAL_ALPHA)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="deeptop_mk_eval")
    ap.add_argument("--lambdas", default=None, help="comma-separated arrival rates (default: full grid)")
    ap.add_argument("--select_alphas", default=None, help="--select: comma-separated alphas (default: --alpha)")
    ap.add_argument("--ckpt_dir", default="ckpt_mk")
    ap.add_argument("--best_out", default="deeptop_mk_actor_best.pkl")
    a = ap.parse_args()
    global LAMBDAS
    if a.lambdas:
        LAMBDAS = [float(x) for x in a.lambdas.split(",")]
    fixed = [f"fixed:{k}" for k in FIXED]

    if a.select:
        seeds = (100, 101)
        alphas = [float(x) for x in a.select_alphas.split(",")] if a.select_alphas else [a.alpha]
        best_fixed = {}
        for al in alphas:
            base = evaluate(fixed, VAL_LAMBDAS, seeds, al, None, a.procs)
            for l in VAL_LAMBDAS:
                best_fixed[(al, l)] = min(base[(p, l)][0] for p in fixed)
        paths = sorted(glob.glob(f"{a.ckpt_dir}/actor_step*.pkl"), key=lambda p: int(re.search(r"step(\d+)", p)[1]))
        best = None
        cols = [(al, l) for al in alphas for l in VAL_LAMBDAS]
        print(f"{'checkpoint':<30}" + "".join(f"{f'{al:.2f}/{l:g}':>10}" for al, l in cols) + "  score (geomean lat / best fixed)")
        for path in paths:
            lat = {}
            for al in alphas:
                r = evaluate(["deeptop"], VAL_LAMBDAS, seeds, al, path, a.procs)
                for l in VAL_LAMBDAS:
                    lat[(al, l)] = r[("deeptop", l)][0]
            score = float(np.exp(np.mean([np.log(lat[c] / best_fixed[c]) for c in cols])))
            print(f"{path:<30}" + "".join(f"{lat[c] / best_fixed[c]:10.3f}" for c in cols) + f"  {score:.3f}", flush=True)
            if best is None or score < best[1]:
                best = (path, score)
        if best:
            shutil.copy(best[0], a.best_out)
            print(f"\nbest: {best[0]} (score {best[1]:.3f}) -> {a.best_out}")
        return

    seeds = range(a.seeds)
    pols = ["deeptop", "goodput"] + fixed
    r = evaluate(pols, LAMBDAS, seeds, a.alpha, a.actor, a.procs)
    lat = {p: np.array([r[(p, l)][0] for l in LAMBDAS]) for p in pols}
    kmean = {p: np.array([r[(p, l)][1] for l in LAMBDAS]) for p in pols}
    fixed_mat = np.stack([lat[p] for p in fixed])
    best_fixed = fixed_mat.min(0)
    best_k = np.array(FIXED)[fixed_mat.argmin(0)]

    hdr = (f"{'lam':>4} {'DeepTOP':>8} {'(k)':>5} {'Goodput':>8} {'(k)':>5} {'k=3':>7} {'Best':>7} {'k*':>3} "
           f"{'k=0':>7} {'DT/Best':>8} {'GP/Best':>8}")
    lines = [f"v2 simulator, budget={C.BUDGET}, max_num_seqs={C.MAX_NUM_SEQS}, alpha={a.alpha}, "
             f"{len(seeds)} seeds; latency in s, (k) = mean k chosen", hdr, "-" * len(hdr)]
    for i, l in enumerate(LAMBDAS):
        lines.append(f"{l:>4} {lat['deeptop'][i]:8.2f} {kmean['deeptop'][i]:5.2f} {lat['goodput'][i]:8.2f} "
                     f"{kmean['goodput'][i]:5.2f} {lat['fixed:3'][i]:7.2f} {best_fixed[i]:7.2f} {best_k[i]:>3} "
                     f"{lat['fixed:0'][i]:7.2f} {lat['deeptop'][i] / best_fixed[i]:8.3f} "
                     f"{lat['goodput'][i] / best_fixed[i]:8.3f}")
    g = lambda x: float(np.exp(np.mean(np.log(x))))
    lines.append(f"geomean ratio to best fixed:  DeepTOP {g(lat['deeptop'] / best_fixed):.3f}   "
                 f"Goodput {g(lat['goodput'] / best_fixed):.3f}   k=3 {g(lat['fixed:3'] / best_fixed):.3f}")
    print("\n".join(lines))
    with open(f"{a.out}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    import csv
    with open(f"{a.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda"] + [f"lat_{p}" for p in pols] + [f"kmean_{p}" for p in pols] + ["best_fixed", "best_k"])
        for i, l in enumerate(LAMBDAS):
            w.writerow([l] + [lat[p][i] for p in pols] + [kmean[p][i] for p in pols] + [best_fixed[i], best_k[i]])

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, GRID = "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    series = [("DeepTOP (multi-level k)", lat["deeptop"], "#2a78d6", "-", "o"),
              ("Goodput-greedy (TurboSpec)", lat["goodput"], "#eb6834", "-", "s"),
              ("Fixed k=3", lat["fixed:3"], "#1baf7a", "--", "D")]
    ax = axes[0]
    for name, y, c, ls, m in series:
        ax.plot(LAMBDAS, y, color=c, ls=ls, marker=m, ms=5, mfc="white", mew=1.6, lw=2, label=name)
    ax.plot(LAMBDAS, best_fixed, color="#0b0b0b", ls=":", lw=1.6, label="Best fixed k (oracle per rate)")
    ax.plot(LAMBDAS, lat["fixed:0"], color="#9b9a95", ls="--", lw=1.4, label="No SD (k=0)")
    ax.set_xlabel("Arrival rate (req/s)"); ax.set_ylabel("Mean end-to-end latency (s)")
    ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for name, y, c, ls, m in series:
        ax.plot(LAMBDAS, y / best_fixed, color=c, ls=ls, marker=m, ms=5, mfc="white", mew=1.6, lw=2, label=name)
    ax.axhline(1.0, color="#0b0b0b", ls=":", lw=1.6, label="Best fixed k")
    ax.set_xlabel("Arrival rate (req/s)"); ax.set_ylabel("Latency / best fixed k  (lower is better)")
    ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"HumanEval, alpha={a.alpha}, budget={C.BUDGET}, max_num_seqs={C.MAX_NUM_SEQS} (v2 simulator)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}.txt/.csv/.png/.pdf")


if __name__ == "__main__":
    main()
