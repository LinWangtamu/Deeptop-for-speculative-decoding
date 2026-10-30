"""
Rank every checkpoint of the v6 sweep on a fixed validation suite and report convergence.
  python select_all.py --procs 28            # add --thr to also compute the best threshold policy per config

Validation suite (corrected simulator, no jitter): alpha in {0.65, 0.72, 0.78, 0.84, 0.89} x
load rho in {0.3, 0.6, 0.8, 0.9, 0.95, 1.05} (lambda = rho * capacity(alpha), capacity.json) x seeds {100, 101, 102}.
References per config: best fixed k (k = 0..4) and, with --thr, the best two-level threshold policy
(k_hi while n_in_system <= m, else k_lo) found by grid search.
Score = geometric mean over configs of latency / best fixed k (lower is better).

Outputs
  select_all.csv        score of every checkpoint (and the TurboSpec rule, the v1 policy)
  convergence.txt       per run: score by step, plateau test, threshold drift, alpha direction
  deeptop_v6_best.pkl   best checkpoint over all runs that passes the alpha-direction check
"""
import argparse, glob, json, os, re, shutil
from multiprocessing import Pool
import numpy as np
import torch
import eval_multik as E
import env_config as C
from model_multik import load_actor

ALPHAS = [0.65, 0.72, 0.78, 0.84, 0.89]
RHOS = [0.3, 0.6, 0.8, 0.9, 0.95, 1.05]
SEEDS = [100, 101, 102]
THR = [f"thr:{lo}:{hi}:{m}" for lo in range(C.K_MIN, 4) for hi in range(lo + 1, C.K_MAX + 1)
       for m in (8, 16, 24, 32, 48, 64, 96, 128)]
CAP_N = 256.0 * np.exp(0.45) * 0.97       # thresholds at or above this are parked at the upper cap (~400)


def lam_of(alpha, rho):
    with open(C.CAPACITY) as f:
        tab = json.load(f)
    return rho * float(np.interp(alpha, tab["alpha"], tab["req_per_s"]))


def run(job):
    pol, lam, seed, alpha, path, key = job
    torch.set_num_threads(1)
    _, _, _, lat, km = E.run_episode((pol, lam, seed, alpha, path))
    return pol, path, key, seed, lat, km


def cutoffs(path, alpha, n_dec=0.15, ctx=200.0):
    """Queue lengths at which k drops 1->0, 2->1, 3->2, 4->3 for a probe state."""
    actor, feat = load_actor(path, C.K_MAX)
    s = [0.0, n_dec, alpha, ctx / C.CTX_SCALE]
    with torch.no_grad():
        thr = actor(torch.tensor(feat(s), dtype=torch.float32)).numpy()
    return [float(t * C.N_SCALE / (1 - t)) if t < 1 else 1e9 for t in thr]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint of each run")
    ap.add_argument("--thr", action="store_true", help="also grid-search the best threshold policy per config")
    a = ap.parse_args()
    if not os.path.exists(C.CAPACITY):
        raise SystemExit("capacity.json missing: run  python capacity.py  first")
    cfgs = [(al, rho) for al in ALPHAS for rho in RHOS]
    lam = {c: lam_of(*c) for c in cfgs}

    runs = sorted(glob.glob("runs/*/ckpt"))
    ckpts = []
    for r in runs:
        ps = sorted(glob.glob(f"{r}/actor_step*.pkl"), key=lambda p: int(re.search(r"step(\d+)", p)[1]))
        ckpts += ps[a.every - 1::a.every]
    V1 = "deeptop_mk_actor_best_v1.pkl"          # reference policy, scored if the file is present
    if os.path.exists(V1):
        ckpts += [V1]

    base = [f"fixed:{k}" for k in range(C.K_MAX + 1)] + ["goodput"] + (THR if a.thr else [])
    jobs = [(p, lam[c], s, c[0], None, c) for p in base for c in cfgs for s in SEEDS]
    jobs += [("deeptop", lam[c], s, c[0], p, c) for p in ckpts for c in cfgs for s in SEEDS]
    print(f"{len(cfgs)} configs x {len(SEEDS)} seeds; {len(ckpts)} checkpoints; {len(jobs)} episodes "
          f"on {a.procs} processes", flush=True)
    with Pool(a.procs) as pool:
        out = pool.map(run, jobs, chunksize=4)

    lat = {}
    for pol, path, key, seed, l, km in out:
        lat.setdefault((pol, path, key), []).append(l)
    lat = {k: float(np.mean(v)) for k, v in lat.items()}
    best_fixed = {c: min(lat[(f"fixed:{k}", None, c)] for k in range(C.K_MAX + 1)) for c in cfgs}
    best_thr = {c: min(lat[(t, None, c)] for t in THR) for c in cfgs} if a.thr else None

    def score(pol, path, ref=best_fixed, alphas=None):
        cs = [c for c in cfgs if alphas is None or c[0] in alphas]
        return float(np.exp(np.mean([np.log(lat[(pol, path, c)] / ref[c]) for c in cs])))

    rows = [("goodput", None)] + [("deeptop", p) for p in ckpts]
    with open("select_all.csv", "w") as f:
        f.write("score_vs_best_fixed,score_vs_best_thr,score_low_alpha,score_high_alpha,policy,checkpoint\n")
        for pol, p in rows:
            st = f"{score(pol, p, best_thr):.4f}" if a.thr else ""
            f.write(f"{score(pol, p):.4f},{st},{score(pol, p, alphas=[0.65, 0.72]):.4f},"
                    f"{score(pol, p, alphas=[0.84, 0.89]):.4f},{pol},{p}\n")
    if a.thr:
        print(f"best threshold policy vs best fixed k (oracle per config): "
              f"{float(np.exp(np.mean([np.log(best_thr[c] / best_fixed[c]) for c in cfgs]))):.3f}")
    print(f"TurboSpec rule (sim): {score('goodput', None):.3f}"
          + (f"   v1 policy: {score('deeptop', V1):.3f}" if V1 in ckpts else ""))

    # ---- convergence report per run ----
    lines, candidates = [], []
    for r in runs:
        ps = [p for p in ckpts if p.startswith(r + "/")]
        if not ps:
            continue
        steps = [int(re.search(r"step(\d+)", p)[1]) for p in ps]
        sc = [score("deeptop", p) for p in ps]
        cut = [cutoffs(p, 0.70) + cutoffs(p, 0.84) for p in ps]
        half = len(cut[0]) // 2
        best_i = int(np.argmin(sc))
        last = list(range(max(0, len(ps) - 5), len(ps)))
        plateau = all(sc[i] <= sc[best_i] * 1.01 for i in last)
        fin = lambda v: [min(x, 5000.0) for x in v]
        drift = max(np.max(np.abs(np.array(fin(cut[i])) - np.array(fin(cut[last[-1]]))) /
                           np.maximum(np.array(fin(cut[last[-1]])), 1.0)) for i in last)
        c70, c84 = cutoffs(ps[best_i], 0.70), cutoffs(ps[best_i], 0.84)
        # alpha direction: among switch points inside the visited range (below the upper cap), the
        # geometric-mean shift from alpha 0.70 to 0.84 must be at least +10% (later switches at higher alpha)
        inside = [(x, y) for x, y in zip(c70, c84) if x < CAP_N and y < CAP_N]
        shift = float(np.exp(np.mean([np.log(y / x) for x, y in inside]))) - 1 if inside else float("nan")
        direction = bool(inside) and shift >= 0.10
        lines.append(f"\n{r}")
        lines.append("  step     score   cutoffs n (k_min+1->k_min, ..., k_max->k_max-1) at a=0.70 | a=0.84")
        for st, s_, c in zip(steps, sc, cut):
            lines.append(f"  {st:>8} {s_:6.3f}   {[round(x) for x in c[:half]]} | {[round(x) for x in c[half:]]}")
        lines.append(f"  best {ps[best_i]} score {sc[best_i]:.3f}; plateau (last 5 within 1% of best): {plateau}; "
                     f"threshold drift over last 5: {drift:.1%} ({'stable' if drift < 0.05 else 'moving'}); "
                     f"alpha shift 0.70->0.84 of switch points below the cap: {shift:+.0%} -> direction ok: {direction}")
        lines.append(f"  converged: {plateau and drift < 0.05}")
        candidates.append((sc[best_i], ps[best_i], direction, plateau and drift < 0.05))
    with open("convergence.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    ok = sorted([c for c in candidates if c[2]]) or sorted(candidates)
    if ok:
        shutil.copy(ok[0][1], "deeptop_v6_best.pkl")
        print(f"\nselected {ok[0][1]} (score {ok[0][0]:.3f}, alpha direction {ok[0][2]}, converged {ok[0][3]}) "
              f"-> deeptop_v6_best.pkl")
        print("check it with:  python deeptop_threshold_sweep.py --ckpt deeptop_v6_best.pkl")
