#!/usr/bin/env python3
"""
Per-step dynamic speculation length (k) in vLLM 0.11.2, with online arrivals.

One engine is built with k_max draft tokens (EAGLE-3). Before every scheduler
step a policy picks one k in {0..k_max} for the whole batch, and every running
request's draft list is trimmed to k (the scheduler then verifies only those).
With --shrink-drafter the drafter also proposes only max(k, 1) tokens at the
end of the step, so its cost follows k (an increase of k takes effect one step
later, a decrease immediately).

Policies (several can be run on the same engine, one after another):
  fixed:K                 always k = K
  nthresh:X / bthresh:X   k = k_max iff running+waiting < X / decoding requests < X, else 0
  goodput                 TurboSpec-style greedy: k maximizing expected accepted
                          tokens / predicted step time (measured A100 cost model)
  module:path/to/file.py  your policy: the file defines make_policy(args) that
                          returns act(state: dict) -> int (see rl_policy_template.py)

State passed to the policy (dict):
  n_running, n_waiting, n_total (= running + waiting), ctx_mean, ctx_sum
  (tokens in running requests), acc_rate (EMA of accepted / drafted tokens),
  acc_len (EMA of accepted drafts per request per step), k_prev, budget, k_max

Arrivals: offline (all at t = 0), poisson (rate), gamma (rate, cv; cv > 1 is bursty).
Latency of a request = finish time - its scheduled arrival time.

Outputs, one folder per (policy, rate):
  results/<gpu>/vllm-<ver>/<model>/dynk_<arrival>_<stamp>/<policy>_r<rate>/
      requests.csv  steps.csv  summary.json
  and results/.../dynk_<arrival>_<stamp>/summary.csv with one row per run.

Example (single line):
  python dynk_serve.py --model meta-llama/Llama-3.1-8B-Instruct --prompts-file humaneval_prompts.jsonl --policies fixed:1,fixed:3,goodput,module:rl_policy.py --arrival poisson --rates 4,8,12 --num-requests 500
"""
import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import subprocess
import time
from collections import deque

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process: needed to reach the scheduler

import numpy as np

# --- garbage-collection monitor: records every gen-2 collection and its pause ---
GC_LOG = []
_gc_t0 = [0.0]
def _gc_cb(phase, info):
    if info.get("generation") != 2:
        return
    if phase == "start":
        _gc_t0[0] = time.perf_counter()
    else:
        GC_LOG.append(time.perf_counter() - _gc_t0[0])
gc.callbacks.append(_gc_cb)


# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--spec-method", default="eagle3")
    p.add_argument("--spec-model", default="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3")
    p.add_argument("--k-max", type=int, default=4, help="draft tokens the engine is built with (0 = no SD)")
    p.add_argument("--shrink-drafter", action="store_true", help="drafter proposes only max(k,1) tokens")
    p.add_argument("--policies", default="fixed:3", help="comma-separated: fixed:K, goodput, module:file.py")
    p.add_argument("--policy-ckpt", default=None, help="passed to make_policy(args) of a module policy")
    p.add_argument("--budget", type=int, default=384, help="max_num_batched_tokens")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--arrival", choices=["offline", "poisson", "gamma"], default="poisson")
    p.add_argument("--rates", default="4", help="comma-separated request rates (req/s); ignored for offline")
    p.add_argument("--cv", type=float, default=2.0, help="coefficient of variation for gamma arrivals")
    p.add_argument("--num-requests", type=int, default=500)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--prompts-file", default=None, help=".jsonl (prompt/text) or .txt (one per line)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None, help="comma-separated seeds; runs every (policy, rate) once per seed")
    p.add_argument("--acc-ema", type=float, default=0.1, help="EMA weight for acceptance estimates")
    p.add_argument("--alpha-prior", type=float, default=0.838,
                   help="prior of the DeepTOP-style acceptance tracker (window of 20 request-steps)")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--cudagraph-max", type=int, default=None)
    p.add_argument("--out-root", default="results")
    p.add_argument("--gc-mode", choices=["default", "freeze"], default="default",
                   help="freeze: gc.freeze()+gc.disable() during each recorded run (tests GC pauses)")
    p.add_argument("--no-warmup-random", dest="warmup_random", action="store_false",
                   help="skip the random-k warm-up that compiles every (batch, k) shape")
    return p.parse_args()


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
FALLBACK = ["Explain how a refrigerator works, step by step.",
            "Write a Python function that checks whether a string is a palindrome.",
            "Describe the water cycle in detail."]


def load_prompt_ids(a):
    from transformers import AutoTokenizer

    texts = []
    if a.prompts_file:
        with open(a.prompts_file) as f:
            for line in f:
                if not line.strip():
                    continue
                if a.prompts_file.endswith(".jsonl"):
                    d = json.loads(line)
                    t = d.get("prompt") or d.get("text") or (d.get("turns") or [None])[0]
                else:
                    t = line.strip()
                if t:
                    texts.append(t)
    texts = texts or FALLBACK
    tok = AutoTokenizer.from_pretrained(a.model)
    out = []
    for t in texts:
        try:
            ids = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True)
        except Exception:
            ids = tok.encode(t)
        out.append(list(ids))
    return out


# ----------------------------------------------------------------------------
# controller: state, policy, hooks
# ----------------------------------------------------------------------------
class Ctrl:
    def __init__(self, k_max, budget, ema, alpha_prior=0.838):
        self.k_max, self.budget, self.ema, self.alpha_prior = k_max, budget, ema, alpha_prior
        self.policy = lambda s: k_max
        self.k = k_max            # k used in the current step
        self.k_draft = k_max      # drafts proposed at the end of this step
        self.acc_rate, self.acc_len = 0.7, 2.0
        self.last = {}            # filled by the scheduler hook for the runner
        self._new_tracker()

    def _new_tracker(self):
        # same estimator as MDP_Env._SpecAcceptanceTracker: accepted / verified over the last
        # 20 (request, step) records, seeded with 2 prior records
        self.dt_buf = deque([(self.alpha_prior, 1.0)] * 2, maxlen=20)

    def alpha_deeptop(self):
        ta = sum(a for a, _ in self.dt_buf)
        tv = sum(v for _, v in self.dt_buf)
        return ta / tv if tv > 0 else self.alpha_prior

    def reset(self, policy):
        self.policy = policy
        self.k = self.k_draft = self.k_max
        self.acc_rate, self.acc_len = 0.7, 2.0
        self._new_tracker()

    def state(self, sched):
        run = sched.running
        ctx = [r.num_tokens for r in run]
        dec_ctx = [r.num_computed_tokens for r in run if r.num_computed_tokens >= r.num_prompt_tokens]
        return {"n_running": len(run), "n_waiting": len(sched.waiting), "n_total": len(run) + len(sched.waiting),
                "ctx_mean": float(np.mean(ctx)) if ctx else 0.0, "ctx_sum": int(sum(ctx)),
                "n_decoding": len(dec_ctx), "ctx_mean_decoding": float(np.mean(dec_ctx)) if dec_ctx else 0.0,
                "acc_rate": self.acc_rate, "acc_len": self.acc_len, "alpha_deeptop": self.alpha_deeptop(),
                "k_prev": self.k, "budget": self.budget, "k_max": self.k_max}

    def update_acceptance(self, spec_per_req, new_tokens):
        drafted = accepted = n = 0
        for rid, s in spec_per_req.items():
            if s > 0 and rid in new_tokens:
                m = max(0, min(s, new_tokens[rid] - 1))
                drafted += s
                accepted += m
                n += 1
                self.dt_buf.append((m, m + 1 if m < s else s))  # verified = up to first rejection
        if drafted:
            e = self.ema
            self.acc_rate = (1 - e) * self.acc_rate + e * accepted / drafted
            self.acc_len = (1 - e) * self.acc_len + e * accepted / n


CTRL = None


def patch_inproc_post_step():
    """vLLM 0.11.x in-process bug: InprocClient.get_output() never calls post_step(),
    so drafts never reach the scheduler. Call it as the multiprocess path does."""
    import inspect
    from vllm.v1.engine.core_client import InprocClient

    if "post_step" in inspect.getsource(InprocClient.get_output):
        return
    from vllm.v1.engine import EngineCoreOutputs

    def get_output(self):
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    InprocClient.get_output = get_output


def install_hooks(a):
    from vllm.v1.core.sched.scheduler import Scheduler

    orig = Scheduler.schedule

    def schedule(self):
        st = CTRL.state(self)
        t0 = time.perf_counter()
        k = int(CTRL.policy(st)) if CTRL.k_max > 0 else 0
        t_pol = time.perf_counter() - t0
        k = max(0, min(CTRL.k_max, k))
        for r in self.running:  # trim every running request's drafts to k
            if len(r.spec_token_ids) > k:
                del r.spec_token_ids[k:]
        CTRL.k = CTRL.k_draft = k
        out = orig(self)
        spec = {rid: len(v) for rid, v in out.scheduled_spec_decode_tokens.items()}
        before = {nr.req_id: nr.num_computed_tokens for nr in out.scheduled_new_reqs}
        c = out.scheduled_cached_reqs
        before.update(zip(c.req_ids, c.num_computed_tokens))
        pre = 0
        for rid, n in out.num_scheduled_tokens.items():
            pre += max(0, min(n, self.requests[rid].num_prompt_tokens - before.get(rid, 0)))
        CTRL.last = {"state": st, "k": k, "t_policy": t_pol, "spec": spec, "tokens": out.total_num_scheduled_tokens,
                     "pre_tokens": pre, "reqs": len(out.num_scheduled_tokens)}
        return out

    Scheduler.schedule = schedule

    if a.k_max > 0:
        patch_inproc_post_step()
        import torch
        from vllm.v1.spec_decode.eagle import EagleProposer

        orig_propose = EagleProposer.propose

        def propose(self, *args, **kw):
            if a.shrink_drafter:
                self.num_speculative_tokens = max(1, CTRL.k_draft)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = orig_propose(self, *args, **kw)
            torch.cuda.synchronize()
            CTRL.last["t_draft"] = CTRL.last.get("t_draft", 0.0) + time.perf_counter() - t0
            return r

        EagleProposer.propose = propose


# ----------------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------------
def verify_ms(n_tokens, ctx_sum):
    """Measured A100 / Llama-3.1-8B step cost (ms): hinge at 384 tokens + context term."""
    return 12.87 + 0.0768 * min(n_tokens, 384) + 0.0949 * max(0, n_tokens - 384) + 0.0000858 * ctx_sum


def draft_ms(k, r):
    return k * (0.78 + 0.008 * r)


def goodput_policy(s):
    """TurboSpec-style: maximize expected accepted tokens per unit time for the current batch.
    Requests served this step are capped by the budget; per-position acceptance is geometric."""
    R = s["n_running"]
    if R == 0:
        return s["k_max"]
    alpha = min(max(s["acc_rate"], 0.01), 0.99)
    best, best_g = 0, -1.0
    for k in range(s["k_max"] + 1):
        served = min(R, s["budget"] // (k + 1))
        exp_tokens = sum(alpha ** i for i in range(k + 1))  # 1 + alpha + ... + alpha^k
        ctx = s["ctx_sum"] * served / R
        t = verify_ms(served * (k + 1), ctx) + draft_ms(k, served)
        g = served * exp_tokens / t
        if g > best_g:
            best, best_g = k, g
    return best


V2 = dict(fixed=13.705, dec=0.067367, ctx=1.1094e-4, over512=0.017735,
          d_fixed=1.118, d_tok=0.00257, d_pass=0.536, d_seq=0.01391)


def decode_step_ms_v2(n_dec_tokens, n_seqs, ctx, k, sd_engine=True):
    """v2 cost model (same constants as MDP_Env.py), decode-only step, in ms."""
    v = V2
    t = v["fixed"] + v["dec"] * n_dec_tokens + v["ctx"] * ctx + v["over512"] * max(0, n_dec_tokens - 512)
    if sd_engine:
        t += v["d_fixed"] + v["d_tok"] * n_dec_tokens
        if k > 1:
            t += (k - 1) * (v["d_pass"] + v["d_seq"] * n_seqs)
    return t


def turbospec_goodputs(s):
    """Predicted goodput of the DECODE batch for k = 0..k_max (None if no decode batch)."""
    D = s["n_decoding"]
    if D == 0:
        return None
    a = min(max(s["alpha_deeptop"], 1e-3), 0.999)
    gs = []
    for k in range(s["k_max"] + 1):
        served = min(D, s["budget"] // (k + 1))
        rest = s["budget"] - served * (k + 1)          # a partially trimmed request gets the leftover
        n_tok = served * (k + 1) + (min(rest, k + 1) if served < D else 0)
        n_seq = served + (1 if served < D and rest > 0 else 0)
        acc = served * (1 - a ** (k + 1)) / (1 - a) + (1 if n_seq > served else 0)
        gs.append(acc / decode_step_ms_v2(n_tok, n_seq, s["ctx_mean_decoding"] * n_seq, k))
    return gs


def turbospec_policy(s):
    """TurboSpec (goodput) exactly as in the simulator baseline: only the DECODE batch is considered
    (prefill chunks are unknown before scheduling), the budget trims the plan in running order,
    acceptance is geometric with alpha = accepted/verified, step time from the v2 cost model."""
    gs = turbospec_goodputs(s)
    return s["k_max"] if gs is None else int(np.argmax(gs))


def make_turbospec_orig():
    """TurboSpec with the extra rules of the authors' code (LiuXiaoxuanPKU/vllm, branch dsd,
    vllm/spec_decode/dsd.py and spec_decode_worker.py):
      * k = 0 goodput is multiplied by 1.1 (speculation must beat no speculation by 10%);
      * k is enumerated upward and the search stops at the first drop in goodput;
      * k is recomputed every 20 steps and held in between;
      * after choosing k = 0 at running-queue size R, speculation stays off while the running
        queue is >= R; this sticky disable is reset after 100 disabled steps."""
    st = {"step": 0, "k": None, "disable_at": float("inf"), "disabled": 0}

    def act(s):
        n_run = s["n_running"]
        if n_run >= st["disable_at"]:
            st["disabled"] += 1
            if st["disabled"] % 100 == 0:
                st["disable_at"], st["disabled"] = float("inf"), 0
            if n_run >= st["disable_at"]:
                return 0
        if st["k"] is None or st["step"] % 20 == 0:
            gs = turbospec_goodputs(s)
            if gs is None:
                k = s["k_max"]
            else:
                k, best = 0, -1.0
                for i, g in enumerate(gs):
                    g = g * 1.1 if i == 0 else g
                    if g > best:
                        k, best = i, g
                    else:
                        break
            st["k"] = k
            if k == 0:
                st["disable_at"] = n_run
        st["step"] += 1
        return st["k"]
    return act


def make_policy(spec, a):
    if spec.startswith("fixed:"):
        k = int(spec.split(":")[1])
        return lambda s: k
    if spec == "goodput":       # budget-aware goodput over ALL running requests (our variant)
        return goodput_policy
    if spec == "turbospec":     # goodput over the decode batch only, as TurboSpec / the simulator baseline
        return turbospec_policy
    if spec == "turbospec_orig":  # + the authors' 1.1x bias to k=0, 20-step updates, sticky disable
        return make_turbospec_orig()
    if spec.startswith("nthresh:"):  # speculate (k_max) iff running + waiting < X
        x = float(spec.split(":")[1])
        return lambda s: s["k_max"] if s["n_total"] < x else 0
    if spec.startswith("bthresh:"):  # speculate (k_max) iff decoding requests < X
        x = float(spec.split(":")[1])
        return lambda s: s["k_max"] if s["n_decoding"] < x else 0
    if spec.startswith("module:"):
        path = spec.split(":", 1)[1]
        m = importlib.util.spec_from_file_location("user_policy", path)
        mod = importlib.util.module_from_spec(m)
        m.loader.exec_module(mod)
        return mod.make_policy(a)
    raise ValueError(f"unknown policy {spec}")


# ----------------------------------------------------------------------------
# engine and one run
# ----------------------------------------------------------------------------
def build_llm(a, max_model_len):
    from vllm import LLM

    kw = dict(model=a.model, dtype="bfloat16", enable_prefix_caching=False, enable_chunked_prefill=True,
              max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.budget, max_model_len=max_model_len,
              gpu_memory_utilization=a.gpu_mem, disable_log_stats=True)
    cg = a.cudagraph_max
    if a.k_max > 0:
        kw["speculative_config"] = {"method": a.spec_method, "model": a.spec_model,
                                    "num_speculative_tokens": a.k_max}
        # vLLM 0.11.x bug: with spec decode the target's capture sizes are rounded up to multiples
        # of k+1 (and capped), but the EAGLE drafter keeps the unrounded list, so a step whose token
        # count lies between the two maxima crashes in pad_for_cudagraph (e.g. k=4, budget 384:
        # rounded max 370, drafter max 384). Pass an explicit list that is already a multiple of
        # k+1 and <= budget, so both lists are identical.
        m = a.k_max + 1
        cap = min(a.budget, cg or 512)
        base = [1, 2, 4, 8] + list(range(16, 513, 16)) + [cap]
        sizes = sorted({-(-s // m) * m for s in base if -(-s // m) * m <= cap} | {cap // m * m} - {0})
        if not sizes:
            sizes = [m]
        kw["compilation_config"] = {"cudagraph_capture_sizes": sizes, "max_cudagraph_capture_size": sizes[-1]}
    elif cg:
        kw["compilation_config"] = {"max_cudagraph_capture_size": cg}
    return LLM(**kw)


def arrival_times(a, rate, n, rng):
    if a.arrival == "offline":
        return np.zeros(n)
    if a.arrival == "poisson":
        gaps = rng.exponential(1.0 / rate, n)
    else:
        shape = 1.0 / a.cv ** 2
        gaps = rng.gamma(shape, 1.0 / (rate * shape), n)
    gaps[0] = 0.0
    return np.cumsum(gaps)


def run_one(llm, a, prompts, rate, tag, record=True, n=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    eng = llm.llm_engine
    n = n or a.num_requests
    rng = np.random.default_rng(a.seed)
    arr = arrival_times(a, rate, n, rng)
    order = rng.integers(0, len(prompts), n)
    sp = SamplingParams(temperature=0.0, max_tokens=a.output_len, ignore_eos=True, detokenize=False,
                        output_kind=RequestOutputKind.DELTA)
    reqs = {}
    steps = []
    i = 0
    gc.collect()
    GC_LOG.clear()
    if record and a.gc_mode == "freeze":      # move all existing objects out of the GC's reach
        gc.freeze(); gc.disable()
    t0 = time.perf_counter()
    while i < n or eng.has_unfinished_requests():
        now = time.perf_counter() - t0
        while i < n and arr[i] <= now:
            rid = f"{tag}-{i}"
            ids = prompts[order[i]]
            eng.add_request(rid, {"prompt_token_ids": ids}, sp)
            reqs[rid] = {"id": i, "arrival": float(arr[i]), "prompt_len": len(ids), "out": 0,
                         "first": None, "finish": None}
            i += 1
        if not eng.has_unfinished_requests():
            time.sleep(max(0.0, arr[i] - (time.perf_counter() - t0)))
            continue
        CTRL.last = {}
        ts = time.perf_counter()
        outs = eng.step()
        te = time.perf_counter()
        new = {}
        for o in outs:
            r = reqs.get(o.request_id)
            if r is None:
                continue
            k_new = len(o.outputs[0].token_ids) if o.outputs else 0
            new[o.request_id] = k_new
            r["out"] += k_new
            if k_new and r["first"] is None:
                r["first"] = te - t0
            if o.finished:
                r["finish"] = te - t0
        L = CTRL.last
        if L:
            CTRL.update_acceptance(L["spec"], new)
            if record:
                st = L["state"]
                steps.append({"t": ts - t0, "dt": te - ts, "t_draft": L.get("t_draft", 0.0),
                              "t_policy": L["t_policy"], "k": L["k"], "tokens": L["tokens"],
                              "pre_tokens": L["pre_tokens"], "reqs": L["reqs"],
                              "spec": sum(L["spec"].values()),
                              "accepted": sum(max(0, new.get(rid, 1) - 1) for rid, s in L["spec"].items() if s),
                              "n_running": st["n_running"], "n_waiting": st["n_waiting"],
                              "ctx_mean": st["ctx_mean"], "acc_rate": st["acc_rate"]})
    wall = time.perf_counter() - t0
    if record and a.gc_mode == "freeze":
        gc.enable(); gc.unfreeze()
    return reqs, steps, wall


def summarize(reqs, steps, wall):
    lat = np.array([r["finish"] - r["arrival"] for r in reqs.values() if r["finish"] is not None])
    ttft = np.array([r["first"] - r["arrival"] for r in reqs.values() if r["first"] is not None])
    out_tok = sum(r["out"] for r in reqs.values())
    ks = np.array([s["k"] for s in steps]) if steps else np.array([0])
    w = np.array([s["dt"] for s in steps]) if steps else np.array([1.0])
    return {"n": int(len(lat)), "lat_mean_s": float(lat.mean()), "lat_p50_s": float(np.percentile(lat, 50)),
            "lat_p99_s": float(np.percentile(lat, 99)), "ttft_mean_s": float(ttft.mean()),
            "ttft_p99_s": float(np.percentile(ttft, 99)), "throughput_tok_s": out_tok / wall,
            "makespan_s": wall, "k_mean_time_weighted": float((ks * w).sum() / w.sum()),
            "policy_ms_mean": float(np.mean([s["t_policy"] for s in steps]) * 1e3) if steps else 0.0,
            "n_stalls": int((w > 0.5).sum()) if steps else 0,          # steps longer than 0.5 s
            "stall_s": float(w[w > 0.5].sum()) if steps else 0.0,
            "step_p50_ms": float(np.percentile(w, 50) * 1e3) if steps else 0.0,   # ~ time between tokens
            "step_p99_ms": float(np.percentile(w, 99) * 1e3) if steps else 0.0,
            "gc2_n": len(GC_LOG), "gc2_max_ms": float(max(GC_LOG) * 1e3) if GC_LOG else 0.0,
            "gc2_total_s": float(sum(GC_LOG))}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, mem = [x.strip() for x in out.split(",")]
        return f"{name.replace('NVIDIA ', '').replace(' ', '-')}-{round(int(mem.split()[0]) / 1024)}GB"
    except Exception:
        return "gpu"


def vllm_version():
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return "unknown"


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    global CTRL
    a = parse_args()
    policies = [p.strip() for p in a.policies.split(",") if p.strip()]
    if a.k_max == 0 and any(p != "fixed:0" for p in policies):
        raise SystemExit("--k-max 0 builds an engine without SD; use --policies fixed:0")
    rates = [0.0] if a.arrival == "offline" else [float(x) for x in a.rates.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = os.path.join(a.out_root, gpu_name(), f"vllm-{vllm_version()}", a.model.split("/")[-1],
                        f"dynk_{a.arrival}_kmax{a.k_max}_b{a.budget}_{stamp}")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({**vars(a), "gpu": gpu_name(), "vllm": vllm_version()}, f, indent=2)
    print(f"Run folder: {root}\n", flush=True)

    CTRL = Ctrl(a.k_max, a.budget, a.acc_ema, a.alpha_prior)
    install_hooks(a)
    prompts = load_prompt_ids(a)
    llm = build_llm(a, max(len(p) for p in prompts) + a.output_len + a.k_max + 32)

    # warm-up (compilation, CUDA graphs, every draft length), not recorded
    saved = a.arrival
    a.arrival = "offline"
    for kw in range(a.k_max + 1):
        CTRL.reset(lambda s, kw=kw: kw)
        run_one(llm, a, prompts, 0.0, f"warm{kw}", record=False, n=64)
    # random k per step over many batch sizes, offline and online, so that every
    # (batch size, draft length) shape is compiled before any recorded run
    if a.warmup_random and a.k_max > 0:
        wr = np.random.default_rng(12345)
        CTRL.reset(lambda s: int(wr.integers(0, a.k_max + 1)))
        run_one(llm, a, prompts, 0.0, "warmrand-off", record=False, n=min(a.max_num_seqs, 256))
        a.arrival = "poisson"
        for r in (8.0, 16.0, 24.0, 32.0):
            CTRL.reset(lambda s: int(wr.integers(0, a.k_max + 1)))
            run_one(llm, a, prompts, r, f"warmrand-{r:g}", record=False, n=int(r * 15))
    a.arrival = saved

    rows = []
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    for seed in seeds:
      a.seed = seed
      for pol in policies:
        for rate in rates:
            CTRL.reset(make_policy(pol, a))
            name = pol.replace(":", "_").replace("/", "_").replace(".py", "")
            tag = f"{name}_r{rate:g}_s{seed}"
            reqs, steps, wall = run_one(llm, a, prompts, rate, tag)
            d = os.path.join(root, tag)
            os.makedirs(d, exist_ok=True)
            write_csv(os.path.join(d, "requests.csv"), list(reqs.values()))
            write_csv(os.path.join(d, "steps.csv"), steps)
            s = {"policy": pol, "rate": rate, "seed": seed, **summarize(reqs, steps, wall)}
            with open(os.path.join(d, "summary.json"), "w") as f:
                json.dump(s, f, indent=2)
            rows.append(s)
            print(f"seed={seed} {pol:28s} rate={rate:6.2f}  lat_mean={s['lat_mean_s']:7.3f}s  p99={s['lat_p99_s']:7.3f}s  "
                  f"ttft={s['ttft_mean_s']:6.3f}s  tput={s['throughput_tok_s']:7.0f} tok/s  "
                  f"k_mean={s['k_mean_time_weighted']:.2f}  policy={s['policy_ms_mean']:.3f} ms", flush=True)
            write_csv(os.path.join(root, "summary.csv"), rows)
    print(f"\nSaved to {root}/")


if __name__ == "__main__":
    main()