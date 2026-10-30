"""
Saturated capacity of the corrected v4 simulator, per acceptance rate.  Run once before training:
  python capacity.py --procs 28
For each alpha, submits 1000 requests at once and runs every fixed k in 0..4 (SD engine; k=0 still
runs the drafter, as in the deployed dynk engine). Capacity = max over k of requests / makespan.
Writes capacity.json, which training (load-based arrival rates) and select_all.py read.
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
import env_config as C

ALPHAS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84, 0.88, 0.92, 0.95]
N = 1000


def run(job):
    alpha, k, seed = job
    env = C.make_env(seed=seed, lam_low=1e7, lam_high=1e7, alpha=alpha, duration=1e9, warmup=0.0,
                     num_requests=N)
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = C.env_step(env, k, sd_engine=True)
        done = term or trunc
    return alpha, k, seed, N / env.clock


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = [(al, k, s) for al in ALPHAS for k in range(C.K_MAX + 1) for s in (1, 2)]
    with Pool(a.procs) as pool:
        out = pool.map(run, jobs)
    cap, best_k = [], []
    for al in ALPHAS:
        per_k = [np.mean([r for (x, k, s, r) in out if x == al and k == kk]) for kk in range(C.K_MAX + 1)]
        cap.append(float(max(per_k))); best_k.append(int(np.argmax(per_k)))
        print(f"alpha {al:.2f}: req/s by k = {np.round(per_k, 2)}  -> capacity {max(per_k):.2f} (k={np.argmax(per_k)})")
    with open(C.CAPACITY, "w") as f:
        json.dump({"alpha": ALPHAS, "req_per_s": cap, "best_k": best_k, "n_requests": N}, f, indent=1)
    print(f"wrote {C.CAPACITY}")
