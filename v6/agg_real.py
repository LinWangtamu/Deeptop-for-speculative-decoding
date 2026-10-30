"""
Collect every summary.csv under results_real_v6/ and print, per (policy, rate):
mean latency of each seed, the median over seeds, and stalled runs.
  python agg_real.py [results_real_v6]
A run counts as stalled if its stall_s column (if present) exceeds 2 s; stalled runs are left out
of the median and listed so they can be rerun.
"""
import csv, glob, os, sys
from collections import defaultdict
from statistics import median

root = sys.argv[1] if len(sys.argv) > 1 else "results_real_v6"
runs = defaultdict(list)          # (policy, rate) -> [(seed_dir, lat, p99, k_mean, stall)]
for f in glob.glob(os.path.join(root, "*", "**", "summary.csv"), recursive=True):
    seed_dir = os.path.relpath(f, root).split(os.sep)[0]
    for r in csv.DictReader(open(f)):
        stall = float(r.get("stall_s") or 0.0)
        runs[(r["policy"], float(r["rate"]))].append(
            (seed_dir, float(r["lat_mean_s"]), float(r["lat_p99_s"]), float(r["k_mean_time_weighted"]), stall))

name = lambda p: "deeptop" if p.startswith("module:") else p
rates = sorted({r for _, r in runs})
pols = sorted({p for p, _ in runs}, key=lambda p: (not p.startswith("module:"), p.startswith("fixed:"), p))
print(f"{'policy':15s}" + "".join(f"{r:>18g}" for r in rates) + "     (median mean latency s, [clean seeds], k_mean)")
for p in pols:
    line = f"{name(p):15s}"
    for r in rates:
        xs = [x for x in runs.get((p, r), []) if x[4] <= 2.0]
        if not xs:
            line += f"{'-':>18s}"
            continue
        line += f"{median(x[1] for x in xs):9.3f} [{len(xs)}] k{median(x[3] for x in xs):4.1f}"
    print(line)

print("\nbest fixed k per rate (median over clean seeds):")
for r in rates:
    fx = {p: median(x[1] for x in runs[(p, r)] if x[4] <= 2.0)
          for p in pols if p.startswith("fixed:") and any(x[4] <= 2.0 for x in runs.get((p, r), []))}
    if not fx:
        continue
    bp = min(fx, key=fx.get)
    extra = "".join(f"  {name(q)} {median(x[1] for x in runs[(q, r)] if x[4] <= 2.0) / fx[bp]:.3f}"
                    for q in pols if not q.startswith("fixed:") and any(x[4] <= 2.0 for x in runs.get((q, r), [])))
    print(f"  rate {r:g}: {bp} {fx[bp]:.3f} s;  ratio to it:{extra}")

bad = [(p, r, x[0], x[4]) for (p, r), xs in runs.items() for x in xs if x[4] > 2.0]
if bad:
    print("\nstalled runs (stall_s > 2 s), rerun these:")
    for p, r, s, st in sorted(bad):
        print(f"  {name(p):15s} rate {r:g}  {s}  stall {st:.1f} s")
