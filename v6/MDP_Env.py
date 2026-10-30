r"""
Speculative Decoding serving environment for DeepTOP.

Scheduling now mirrors vLLM V1's *synchronous* core from vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py:

  * No prefill / decode "phase" split. Each step forms ONE unified batch that
    can mix decode tokens (1 query token per decoding seq) AND prefill chunks,
    all sharing a single `max_num_batched_tokens` budget -- exactly like
    vllm Scheduler.schedule(): it first packs RUNNING requests, then admits
    WAITING requests until the token budget or the running cap max_num_seqs is hit.

  * `max_num_seqs` is a hard cap on the number of RUNNING requests, enforced at
    admission time (the waiting->running gate). It is NOT a post-hoc slice of
    an unbounded `active` list anymore, so the 257th (max_num_seqs=256) request truly waits instead
    of sitting in the batch holding KV while never decoding.

  * Long prompts are CHUNKED across steps: if a prompt does not fit the
    remaining token budget, only `min(prompt_remaining, budget)` tokens are
    scheduled this step and the rest continues next step (chunked prefill).

  * KV is still reserved UP FRONT (prompt_len + decode_len) when a request is
    first admitted to RUNNING, and there is NO preemption. This is a deliberate
    simplification: with max_num_seqs=256 and a 200k-token pool the KV ceiling is
    effectively never the binding constraint, so incremental KV + preemption
    would be dead code in this regime.
    
    #need verify (avg_prompt = avg_decode =128, total max tokens 65536, much less than 200k)

  * RUNNING requests are scheduled in list (admission) order, exactly like
    vLLM V1 -- no decode-first priority. Each takes min(want, budget), where a
    decode wants 1 + k (vLLM's `1 + num_spec_tokens`) and a prefill wants its
    remaining prompt. If the budget runs short, spec tokens are trimmed and the
    running requests after that point are not scheduled this step.

State (a Python list; state[0] is the SCALAR compared against the learned
threshold, state[1:] is the vector fed to the actor to learn the threshold):
    state = [ n_in_system / (n_in_system + n_in_system_scale),  # scalar [0,1) queue length
              decode_batch / max_num_seqs,           # vector[0]: how full the batch already is
              alpha_estimate,                        # vector[1]
              average_context / avg_ctx_scale ]      # vector[2]
    avg_ctx_scale and n_in_system_scale are __init__ params. Neither is
    derived from max_num_seqs or the workload automatically -- changing
    either (e.g. max_num_seqs=16 + workload_trace_path to match real
    deployment) means these need to be re-picked by hand and passed in, or
    the normalized state the actor sees during training silently stops
    matching what training was tuned against.

  n_in_system : len(running) + len(waiting) -- "queue length" in the current
                 num-of-requests-in-system sense (Little's Law's L, same
                 quantity _window_L integrates for the steady-state latency
                 estimate below), NOT backlog (which explicitly excludes the
                 decode batch). Since n_in_system >= decode_batch always, this
                 is never near-zero just because nothing is queued -- it
                 reads the whole system's occupancy, batch included. Default
                 n_in_system_scale = max_num_seqs, so n_in_system ==
                 max_num_seqs (batch exactly full, nothing queued behind it)
                 lands the scalar at exactly 0.5, a clean "capacity reached,
                 no backlog yet" reference point. Like the old backlog-scalar
                 design, this never reaches exactly 1.0 for finite occupancy,
                 avoiding the old batch-fraction scalar's exact-1.0 boundary
                 issue (see DeepTOP.py select_action).
  decode_batch: number of RUNNING requests currently in the decode phase, a
                 vector input (not the scalar) so the network can use "how
                 full is the batch right now" to modulate how aggressively it
                 should back off under queue-length pressure -- e.g. per the
                 goodput crossover analysis, a large batch should tolerate
                 less queueing before turning speculation off than a small
                 one, since the verify pass's extra query-token cost scales
                 with batch size while the accepted-token benefit is capped
                 at (k+1)x. NOTE: an earlier iteration used backlog (batch
                 EXCLUDED) as the scalar instead and empirically regressed at
                 lam>=8 in the simulator sweep vs. both k=3-always and the
                 batch-fraction-scalar design -- it learned to back off under
                 backlog pressure exactly when lambda exceeds the system's
                 own stability ceiling, i.e. exactly when backing off hurts
                 most (see Finding #4/#7). n_in_system is a related but
                 distinct signal (never reads as empty while the batch itself
                 is full) -- not assumed to fix that regression, needs its
                 own empirical check same as everything else in this file.

Action: 1 -> speculate this step (k=_K_SPEC=3);  0 -> normal decode (k=0)

The learned DeepTOP policy is:
    speculate iff threshold(decode_batch, alpha_est, average_context)
                  > normalized_n_in_system

Reward (per step):
    reward = tokens_this_step / token_norm(diable by default)  -  holding_cost / reward_norm

    The holding cost (sum over the step of step_latency * #-in-system) is the
    classic queueing proxy for latency: by Little's law its long-run average is
    proportional to mean E2E latency. The holding cost of a step covers the
    whole unified forward pass (prefill chunks + decode), since prefill is no
    longer a separate clock advance.

    This task is trained with the AVERAGE-REWARD objective (no discounting): the
    speculate-or-not decision pays off (sd enable faster drain -> lower future holding)
    over thousands of steps, a horizon that any gamma < 1 discounts away, so a
    discounted return inverts the sign of the optimum. Under average reward,
    maximizing mean per-step reward == minimizing mean #-in-system == minimizing
    mean latency. For that objective the token term is left OFF (token_norm
    large, i.e. pure holding cost) and NO reward clipping is applied, so the
    overload region keeps the gradient the load-dependent speculation threshold
    relies on. The token term and reward_clip remain available for discounted-RL
    experiments but are not used by the average-reward setup.

Episode: fixed wall-clock duration with Poisson arrivals; the arrival rate
lambda is re-randomized every episode (generalization across loads). Episode
terminates once the clock has passed `duration` and all requests have drained.
make an action is one step

#our SpecDecodingEnv obeys the gym rule to  
Gym API: this env follows the Gymnasium 5-tuple convention --
    reset() -> (obs, info) #reset the env(resample \lambda, \alpha, generate new arrival, dequeue) 
    step()  -> (obs, reward, terminated, truncated, info)
    obs, reward, terminated, truncated, info = env.step(action)# move one step, agent feed to env, and env return 5-tuple to agent
"""

from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# =============================================================================
# Latency model: A100, fitted from REAL EAGLE3 execution inside vLLM V1
# (target=meta-llama/Llama-3.1-8B-Instruct, draft=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B,
# num_speculative_tokens=3, temperature=0, 600 real requests / 9444 timed steps,
# outlier-filtered least squares -- see
# /scratch/user/linwang0420/results/eagle3_fit/SUMMARY.md for the full writeup).
# Supersedes the previous H100 constants below, which were fitted for a
# standalone 1B draft_model (a full separate causal LM), not a fused EAGLE3
# draft head -- draft cost turned out to behave completely differently (see
# _DRAFT_C_SPEC note below), so those numbers do not transfer.
#
#   target_fwd(n_kv, n_query) = c_fixed + c_kv * N_kv + c_compute * N_query #fixed overhead + load kv cache + compute token
#       big model; N_kv = total context (KV) attended, N_query = query tokens.
#   draft_step(n_kv, n_spec)  = c_fixed + c_kv * N_kv + c_spec * N_spec
#       small model run ONCE to produce n_spec draft tokens (KV is reused across
#       the k autoregressive draft steps, so cost is linear in #spec tokens, NOT
#       k independent full-context passes).
#
#   no speculation (k=0): t = target_fwd(N_kv, bs)            # 1 query / seq
#   speculation   (k>0) : t = draft_step(N_kv, k*bs)          # draft k tokens/seq
#                           + target_fwd(N_kv, bs*(k+1))      # verify k+1 pos/seq
#
# Previous (superseded) H100 fit, for a standalone 1B draft_model:
#   _TARGET_C_KV=3.92645679e-08  _TARGET_C_COMPUTE=2.10103129e-05  _TARGET_C_FIXED=5.88590713e-03
#   _DRAFT_C_KV=2.37768493e-09   _DRAFT_C_SPEC=8.41429603e-07      _DRAFT_C_FIXED=2.13526175e-03
# =============================================================================
# ---------------------------------------------------------------------------
# CURRENT PROFILE (2026-10): replaces the constants listed further below.
#
#   T_verify(N_ctx, N_q) = 12.9 ms + 0.086 us * N_ctx
#                          + 77 us * min(N_q, 384) + 95 us * max(0, N_q - 384)
#       Fitted on the AUTOREGRESSIVE (k=0) target forward, then reused for the
#       verify pass with N_q = N_dec * (k+1) (+ prefill chunk tokens).
#   D_k(R) = k * (0.78 ms + 8.0 us * N)
#       k autoregressive draft passes, each over N = #speculating seqs.
#   T_step(R, k) = T_verify(N_ctx, N*(k+1)) + D_k(R)
#
# Prefill and draft prefill: measured separately, see the constants below.
# MIXED steps (prefill chunk + decodes in one pass) were not profiled; they are
# modelled additively (decode piecewise term + linear prefill term, one fixed
# cost). Pure-prefill steps come out 0.7 ms above the prefill fit (12.9 vs
# 12.2 ms fixed) -- accepted to keep a single fixed cost.
# ---------------------------------------------------------------------------
# T_verify is PIECEWISE in query tokens (memory-bound -> compute-bound knee),
# fitted with no SD at input=output=32:
#   T_verify = 12.9 ms + 0.086 us*N_ctx + 77 us*min(N_q, 384) + 95 us*max(0, N_q-384)
_TARGET_C_FIXED   = 12.9e-3          # target: fixed per-forward overhead
_TARGET_C_KV      = 0.086e-6         # target: per KV (context) token attended
_TARGET_C_COMPUTE = 77e-6            # target: per query token, below the knee
_TARGET_C_COMPUTE_HI = 95e-6         # target: per query token, above the knee
_TARGET_KNEE      = 384              # query tokens at the knee
# Prefill tokens are priced separately and LINEARLY (no knee), from pure-prefill
# steps (step 0 of the main run, 50..12504 prompt tokens in one step):
#   T_prefill ~= 12.2 ms + 74 us * prefill_tokens
# Target prefill time was measured identical with and without SD.
_TARGET_C_PREFILL = 74e-6            # target: per prefill (prompt) token
_DRAFT_C_FIXED    = 0.78e-3          # draft: fixed cost per draft pass
_DRAFT_C_TOK      = 8.0e-6           # draft: per decode seq in a draft pass
# Draft PREFILL, measured (k=1 runs: the drafter makes exactly one pass over
# the prompt): ~0.8 ms + 3.2 us/prompt token, ~4.3% of target prefill.
# Replaces the earlier extrapolation (8 us/token), which was ~2.5x too high.
_DRAFT_C_PREFILL_FIXED = 0.8e-3      # draft: fixed cost of a prefill-only pass
_DRAFT_C_PREFILL_TOK   = 3.2e-6      # draft: per prompt token

# ---- Superseded A100 EAGLE3 fit (kept for reference only) ------------------
# _TARGET_C_KV=1.153933e-07  _TARGET_C_COMPUTE=5.460380e-05  _TARGET_C_FIXED=1.443821e-02
# _DRAFT_C_KV=2.751961e-08   _DRAFT_C_SPEC=0.0               _DRAFT_C_FIXED=3.761316e-03
# (draft was modelled as ONE ~flat call; the new profile shows it is k
#  passes, each linear in batch size)
# [SUPERSEDED, old fit] _DRAFT_C_SPEC note: real EAGLE3's draft head (1-3 layers, much smaller than a
# standalone 1B model) shows draft-call latency that is nearly CONSTANT
# (3.82ms +/- 0.65ms) across the profiled range (n_kv up to ~2000, batch size
# up to 64) -- correlation with both n_kv and batch size is weak (0.14-0.16),
# and a fit including a batch-size/N_spec term came back with a negative (i.e.
# unphysical) coefficient, a symptom of fitting noise rather than a real
# effect. Set to 0.0 rather than reporting a fake-precision unphysical number;
# _DRAFT_C_KV is kept at its (also weak but at least non-negative) fitted
# value. Net effect: draft_step() is ~flat at _DRAFT_C_FIXED regardless of
# load, which is a defensible real finding, not a modeling shortcut.

# Per-speculative-position overhead OUTSIDE the model forward passes.
#
# Why this exists as a separate term rather than folded into draft_step() or
# target_fwd(): a re-profiling run at max_num_seqs=128 (job 19671747,
# results/eagle3_profile_cap128_*/, 9061 clean samples after dropping 20
# warmup/compile artifacts with n_kv==0) showed BOTH forward-pass costs are
# already well described by the existing linear constants at high batch --
# target R2=0.756 with residuals within +/-12% per bs bucket all the way to
# bs~100 (and those samples already carry the speculative n_query=bs*(k+1),
# since that run always drafted K=3), and draft latency flat at ~4.4-5.8ms
# from bs<16 to bs~100. So the forward passes are NOT where real speculation
# gets expensive at high concurrency.
#
# But real deployment at max_num_seqs=128 (job 19669245) shows k=0 beating
# always-K=3 by 35-67% throughput at rate>=16, while the simulator with these
# same constants predicts only a 1-10% gap. The missing cost therefore lives
# outside the timed forward calls -- CPU-side accept/reject sampling,
# rejection resampling, draft-buffer bookkeeping and dispatch, all of which
# scale with the number of speculative positions (k * decode_batch) and none
# of which the [PROF][DRAFT]/[PROF][TARGET] instrumentation measures.
#
# CALIBRATION (2026-09-10): fitted to the real SATURATION-THROUGHPUT ratio
# k0/k3fixed at cap=128 from job 19669245, not the latency ratio -- latency
# ratio is the wrong target here: in sim it is hypersensitive at low rate and
# compresses toward 1.0 under overload (both policies equally queue-bound), and
# on the real side it is a single unseeded run whose tails swing +/-15-34%
# between identical configs. Throughput is stable on both sides. Grid over
# {0, 3e-5, 5e-5, 8e-5, 1.2e-4} on sum-of-squared-log-error of the k0/k3
# throughput ratio across rates {16,32,64,96,128}:
#   0.0    -> 0.589 (sim ratios 1.00-1.09 vs real 1.35-1.67: no crossover)
#   8e-5   -> 0.036 (BEST; e.g. rate=64 sim 1.531 vs real 1.537)
#   1.2e-4 -> 0.249 (overshoots)
# Regression-checked at cap=16 against job 19580641 (no prefix cache): with
# 8e-5 the sim still says k3 wins at every saturation rate (ratio 0.928), which
# is what the real cap=16 data says (0.949 / 0.802 / 0.312) -- so the term does
# not invert the conclusion at the cap this project actually deploys at.
#
# HONESTY NOTE: 8e-5 s per speculative position is ~30ms of the step at
# bs=128,k=3, which is large next to ~56ms of measured GPU forward work. This
# is a fitted lump term standing in for whatever the real non-forward-pass cost
# is; it is NOT a measured per-operation quantity, and the linear-in-(k*n_dec)
# form is an assumption that happens to fit over bs 16-128. Do not read it as a
# mechanistic claim.
# 2026-10: set to 0.0. The new profile defines the WHOLE step as
# T_step = T_verify + D_k, and the 8e-5 above was calibrated against the OLD
# forward-pass constants, so stacking it on the new ones would double-count.
# If the new T_verify / D_k were timed around the forward calls only (not the
# full engine step), this term must be re-calibrated, not restored to 8e-5.
_SPEC_SCHED_OVERHEAD = 0.0      # seconds per speculative position (k * n_dec)

_K_SPEC = 3                     # speculation length when SD is on -- matches
                                 # the real deployment's num_speculative_tokens=3
                                 # (was 5, mismatched with actual EAGLE3 usage)
_MAX_NUM_SEQS = 256             # max RUNNING requests (vLLM max_num_seqs cap)
_MAX_NUM_BATCHED_TOKENS = 2048  # per-step token budget (vLLM max_num_batched_tokens)


def target_fwd(n_kv, n_query, n_prefill=0):
    """Target forward for one unified step.

    n_kv      : KV (context) tokens attended
    n_query   : DECODE query tokens (1 + spec per decoding seq) -> piecewise
    n_prefill : prompt tokens prefilled this step -> linear, no knee
    """
    return (_TARGET_C_FIXED + _TARGET_C_KV * n_kv
            + _TARGET_C_COMPUTE * min(n_query, _TARGET_KNEE)
            + _TARGET_C_COMPUTE_HI * max(0, n_query - _TARGET_KNEE)
            + _TARGET_C_PREFILL * n_prefill)


def draft_fwd(n_tokens):
    """One draft forward pass over n_tokens query tokens."""
    return _DRAFT_C_FIXED + _DRAFT_C_TOK * n_tokens


def draft_cost(k, n_spec, n_prefill_tokens=0):
    """Draft time for one step.

    Decode drafting: k passes over the n_spec speculating seqs,
        D_k = k * (0.78 ms + 8 us * n_spec).
    Draft prefill (prompt tokens run through the drafter to build its KV):
    3.2 us/token. When there is drafting, it rides in the FIRST pass (one
    fixed cost); otherwise it is one prefill-only pass (0.8 ms + 3.2 us/token).
    """
    if k > 0 and n_spec > 0:
        return (draft_fwd(n_spec) + _DRAFT_C_PREFILL_TOK * n_prefill_tokens
                + (k - 1) * draft_fwd(n_spec))
    if n_prefill_tokens > 0:
        return _DRAFT_C_PREFILL_FIXED + _DRAFT_C_PREFILL_TOK * n_prefill_tokens
    return 0.0


# ---------------------------------------------------------------------------
# COST MODEL v2 (2026-10-09): one unified fit over controlled mixed prefill+decode
# sweeps (profile_mixed.py, k=0 and k=3) and every step of the real budget-384 run
# (k=0/1/3). Times in ms. Per-step MAPE on the real run 3-5%; fitting on k=0,1 only
# predicts k=3 with 3.9% MAPE (+2.3% total time).
#   T_target = 13.705 + 0.0522*P + 0.0674*N_d + 1.109e-4*C_d
#              + 0.0752*S*[P>0]                 (mixing penalty, per decode seq)
#              + 0.0177*max(0, P+N_d-512)       (beyond the CUDA-graph capture size)
#              + 1.081e-5*A_p                   (prefill attention, A_p = sum c*(b+c/2))
#   P   = prefill tokens, N_d = decode/verify tokens, S = decode seqs,
#   C_d = KV tokens read by decode seqs, b = tokens already computed before a chunk c.
#   T_draft (SD engine) = 1.118 + 0.00257*(P_draft + N_d) + (k-1)*(0.536 + 0.01391*S)
#   (k = 0 on an SD engine still runs one drafter pass over the step's tokens.)
# ---------------------------------------------------------------------------
V2 = dict(fixed=13.705, pre=0.052243, dec=0.067367, ctx=1.1094e-4, mix=0.07524,
          over512=0.017735, attn=1.0807e-5,
          d_fixed=1.118, d_tok=0.00257, d_pass=0.536, d_seq=0.01391)


def step_latency_v2(P, N_d, S, C_d, A_p, k, draft_tokens, sd_engine):
    """Seconds. P/N_d/S/C_d/A_p as above; draft_tokens = prefill tokens run through the drafter."""
    v = V2
    t = (v["fixed"] + v["pre"] * P + v["dec"] * N_d + v["ctx"] * C_d
         + (v["mix"] * S if (P > 0 and S > 0) else 0.0)
         + v["over512"] * max(0, P + N_d - 512) + v["attn"] * A_p)
    if sd_engine:
        t += v["d_fixed"] + v["d_tok"] * (draft_tokens + N_d)
        if k > 1 and S > 0:
            t += (k - 1) * (v["d_pass"] + v["d_seq"] * S)
    return t / 1e3


# # Backward-compat alias: spec_t_fwd historically meant a target forward pass.
# def spec_t_fwd(nc, nb):
#     return target_fwd(nc, nb)


def spec_t_decode_step(batch, k):
    """Decode step latency. k=0: target only. k>0: one draft call + one verify.

    Kept for reference / the SmartSpec baseline. The env uses
    `_unified_step_latency`, which is a strict generalization (it reduces to this
    when the batch has no prefill chunks).
    """
    n_kv = sum(r.context_len for r in batch)
    bs = len(batch)
    if k == 0:
        return target_fwd(n_kv, bs)
    return draft_cost(k, bs) + target_fwd(n_kv, bs * (k + 1))


# def spec_t_prefill(pl):
#     return target_fwd(0, pl)


# =============================================================================
# Per-request, KV cache, acceptance tracking, stochastic acceptance
# =============================================================================
class _SpecRequest(object):
    """vLLM-style request bookkeeping.

    A request is PREFILLING while num_computed_tokens(current prefill token number) < num_prompt_tokens and
    DECODING once it has caught up. context_len tracks the KV size it attends
    to and stays equal to num_computed_tokens.
    """

    def __init__(self, req_id, prompt_len, decode_len, arrive_t):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.decode_len = decode_len
        self.arrive_t = arrive_t
        self.num_prompt_tokens = prompt_len
        self.num_computed_tokens = 0
        self.num_output_tokens = 0
        self.context_len = 0
        self.prefill_done = False
        self.finish_t = -1.0
        # True while the draft model has run every prefill chunk of this
        # request. If any chunk skips the draft prefill (TurboSpec prefill
        # disabling), the draft KV is missing and the request can never
        # speculate again for the rest of its life.
        self.has_draft_kv = True

    @property
    def is_decoding(self):
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def is_done(self):
        return self.num_output_tokens >= self.decode_len

    @property
    def latency(self):
        return self.finish_t - self.arrive_t if self.finish_t > 0 else None


class _SpecKVCache(object):
    def __init__(self, max_tokens=200000):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def can_allocate(self, n):
        return self.used_tokens + n <= self.max_tokens

    def allocate(self, n):
        self.used_tokens += n

    def free(self, n):
        self.used_tokens = max(0, self.used_tokens - n)

##change the logit to let deeptop adjust to different alpha
# class _SpecAcceptanceTracker(object):
#     """Definition A: alpha = total accepted / total VERIFIED tokens (moving window)."""

#     def __init__(self, window=20, init=0.7):
#         self._buf = [(init, 1.0)] * (window // 2)
#         self._w = window

#     def update(self, acc, ver):
#         if ver > 0:
#             self._buf.append((float(acc), float(ver)))
#             if len(self._buf) > self._w:
#                 self._buf.pop(0)

#     @property
#     def value(self):
#         ta = sum(a for a, _ in self._buf)
#         tv = sum(v for _, v in self._buf)
#         return ta / tv if tv > 0 else 0.7

# new alpha estimator

class _SpecAcceptanceTracker(object):
    """Definition A: alpha = total accepted / total VERIFIED tokens (moving window).

    Exploration-honest version: the prior does NOT leak the episode's true_alpha.
    The agent starts from a fixed, episode-independent prior and can only learn
    the real acceptance rate by actually speculating (k>0). This mirrors a real
    serving system, where acceptance can only be measured by running the drafter.

    `prior` is a fixed optimistic estimate (default 0.9) so that, under
    uncertainty, speculating looks attractive -- this pushes the agent to explore
    (optimism in the face of uncertainty) instead of never speculating and thus
    never observing alpha. Set prior lower for a more neutral start.

    `prior_weight` controls how many synthetic prior samples seed the buffer:
    larger -> the prior persists longer before real data dominates; smaller ->
    real observations take over faster (stronger exploration pressure).
    """

    def __init__(self, window=20, prior=0.9, prior_weight=2):
        self._w = window
        self._prior = float(prior)
        # Seed with `prior_weight` synthetic records, each worth 1 verified
        # token, encoding the prior acceptance rate. These get flushed out of the
        # moving window as real (k>0) observations arrive.
        self._buf = deque(
            [(self._prior, 1.0)] * max(0, prior_weight),
            maxlen=window,
        )

    def update(self, acc, ver):
        # Only real speculative steps (ver>0) carry information. With a bounded
        # deque, append auto-evicts the oldest record once full.
        if ver > 0:
            self._buf.append((float(acc), float(ver)))

    @property
    def value(self):
        ta = sum(a for a, _ in self._buf)
        tv = sum(v for _, v in self._buf)
        return ta / tv if tv > 0 else self._prior
def spec_simulate_acceptance(k, true_alpha, rng, decay=1.0):
    """Returns (m, n_verified). Early-stop at first rejection.
    Position i (1-based) is accepted with probability true_alpha * decay**(i-1); decay = 1 is the
    old constant-acceptance model."""
    if k == 0:
        return 0, 0
    m = 0
    p = true_alpha
    for _ in range(k):
        if rng.random() < p:
            m += 1
            p *= decay
        else:
            return m, m + 1
    return m, k


# =============================================================================
# Gym environment
# =============================================================================
class SpecDecodingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed, duration=120.0, warmup=20.0,
                 lam_low=0.5, lam_high=20.0, true_alpha=0.4744,
                 # true_alpha is the per-position CONDITIONAL Bernoulli
                 # accept probability spec_simulate_acceptance() draws from
                 # each of the k early-stopped trials -- NOT the same
                 # statistic as vLLM's logged "Avg Draft acceptance rate"
                 # (accepted / (k * num_calls), which counts every position
                 # as "drafted" regardless of early stop). Solved so this
                 # sim's implied mean_acceptance_length (1+p+p^2+p^3) matches
                 # real EAGLE3's measured 1.81 (E[m]=0.806 accepted/call,
                 # 43498 accepted / 53960 calls, 161880 drafted / 3):
                 # p + p^2 + p^3 = 0.8061 -> p = 0.4744.
                 # An earlier pass here set true_alpha=0.2687 by directly
                 # copying vLLM's reported ratio into this slot -- that
                 # UNDERSTATED simulated speculation benefit (implied mean
                 # acceptance length 1.36 vs the real 1.81) because it
                 # conflated the two different statistics above. Real
                 # per-position CONDITIONAL rates are [0.44, 0.52, 0.59]
                 # (increasing, not decreasing -- a survivorship effect: a
                 # draft that already survived to position i tends to be on
                 # an "easy" streak) -- a single constant p is a first-order
                 # approximation of that, not an exact per-position model.
                 # See results/eagle3_fit/SUMMARY.md.
                 alpha_low=None, alpha_high=None,
                 # v6: acceptance decays with draft position (see spec_simulate_acceptance)
                 accept_decay=1.0,
                 # v4: per-step overhead missing from the v2 cost model, fitted on real online
                 # step logs (calib_online, 2026-10-20): +overhead_full_ms on every step and
                 # +overhead_part_ms more when the step is not full. overhead_jitter scales both
                 # by U(1-j, 1+j) per episode (training only).
                 overhead_full_ms=0.0, overhead_part_ms=0.0, overhead_jitter=0.0,
                 # v4: load-based arrival rates. If rho_low is set, each episode draws a load
                 # rho ~ U(rho_low, rho_high) (with prob. p_over instead rho ~ U(1.0, 1.15)) and
                 # sets lambda = rho * capacity(alpha), capacity read from capacity_path (json).
                 rho_low=None, rho_high=None, p_over=0.0, capacity_path=None,
                 avg_prompt=128.0, avg_decode=128.0,
                 workload_trace_path=None,
                 # Real (prompt_tokens, decode_tokens) pairs, captured from
                 # actual generation over the same ShareGPT-filtered dataset
                 # used everywhere else in this project (see
                 # capture_workload_trace.py / results/eagle3_fit/). When
                 # given, reset() samples arrivals' (p, d) with replacement
                 # from this real empirical distribution instead of drawing
                 # from Normal(avg_prompt,...)/Normal(avg_decode,...), which
                 # measurably does not match reality: real prompt length is
                 # heavy-tailed (mean 198, median only 57, std 303 -- bigger
                 # than the mean), and real decode length is cap-dominated
                 # (median 200 == max_tokens, most requests hit the cap
                 # rather than stopping naturally) -- neither resembles a
                 # tight symmetric Gaussian(128, 38.4). avg_prompt/avg_decode
                 # remain as a fallback when no trace is given.
                 max_tokens=200000, reward_norm=50.0, token_norm=1e9,
                 reward_clip=None,
                 max_num_seqs=_MAX_NUM_SEQS,
                 max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
                 # When set, an episode is a burst of exactly this many Poisson
                 # arrivals followed by a drain, instead of a continuous stream
                 # for `duration` seconds. This matches offline_delay_bench.py,
                 # which fires --num-prompts requests and then lets the system
                 # empty: most of its scheduling steps happen in that drain
                 # tail, which is why the real run shows a median running batch
                 # of 2 while the stream-mode sim pegs at max_num_seqs. Use
                 # warmup=0 with this so done_latencies keeps every request,
                 # making mean_latency directly comparable to the real
                 # mean_e2el_ms (the Little's-law windowed estimate is not
                 # meaningful for a burst that never reaches steady state).
                 num_requests=None,
                 long_prefill_token_threshold=0,
                 enable_chunked_prefill=True,
                 n_in_system_scale=50.0,
                 avg_ctx_scale=256.0,
                 spec_sched_overhead=_SPEC_SCHED_OVERHEAD,
                 alpha_prior=0.6,
                 cost_model="v1",
                 k_spec=_K_SPEC):
                 # alpha_prior seeds _SpecAcceptanceTracker (reset() below).
                 # Was hardcoded 0.9 ("optimistic" to encourage exploring
                 # speculation) -- but the actor only ever trains on
                 # alpha_estimate values within [alpha_low, alpha_high], so
                 # 0.9 is out-of-distribution input the network never learned
                 # a sensible response for. If the trained actor's arbitrary
                 # extrapolation at alpha=0.9 happens to say "don't
                 # speculate", the tracker (which only updates from real
                 # speculate-and-verify outcomes) never gets a single
                 # observation to correct it -- a self-reinforcing deadlock
                 # that locks the WHOLE episode at k=0 from step 0, with no
                 # escape at eval/deployment time (no random-exploration
                 # override there, unlike during training). Verified directly
                 # on a real collapsed run (deeptop_spec_actor_smdp.pkl,
                 # n_in_system-scalar version): forcing alpha_estimate to the
                 # true 0.4744 instead of the stuck 0.9 prior took the same
                 # actor from action=1 rate 0.0 to 1.0 at lambda=8. Default
                 # 0.6 matches alpha_high in every current training/eval
                 # config in this project, so it stays in-distribution while
                 # keeping the same "optimistic" (upper-end) bias as before.
                 # Both scales were tuned for the OLD regime (max_num_seqs=256,
                 # synthetic Normal(128,128) workload -> avg_prompt+avg_decode
                 # =256). Neither is tied to max_num_seqs or the workload
                 # automatically -- when either changes (e.g. max_num_seqs=16
                 # to match real deployment, or workload_trace_path for the
                 # real prompt/decode distribution), these must be re-picked
                 # by hand and passed in explicitly, or the normalized state
                 # the actor sees stops meaning what it meant during training.
                 # scheduler_policy_deeptop.py's deployment-side state
                 # construction has its own copies of these two numbers and
                 # must be kept in sync by hand -- they are not read from here.
        super(SpecDecodingEnv, self).__init__()
        self.cost_model = cost_model
        self.k_spec = k_spec   # speculation length used when action == 1
        self.seed_val = seed
        self.rng = np.random.RandomState(seed)
        self.duration = duration
        self.warmup = warmup
        self.lam_low = lam_low
        self.lam_high = lam_high
        self.true_alpha = true_alpha
        # Per-episode acceptance randomization. When both bounds are given,
        # reset() draws a fresh true_alpha each episode so the agent sees both
        # the speculate-favorable (high alpha) and speculate-unfavorable (low
        # alpha) regimes; the alpha_estimate state dim is what it keys off. When
        # left None, true_alpha stays fixed (controlled eval / backward compat).
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.alpha_prior = alpha_prior
        self.accept_decay = accept_decay
        self.overhead_full_ms, self.overhead_part_ms = overhead_full_ms, overhead_part_ms
        self.overhead_jitter, self.ovh_scale = overhead_jitter, 1.0
        self.rho_low, self.rho_high, self.p_over = rho_low, rho_high, p_over
        self.rho = None
        self._cap = None
        if capacity_path is not None:
            import json
            with open(capacity_path) as f:
                tab = json.load(f)
            self._cap = (np.array(tab["alpha"], dtype=float), np.array(tab["req_per_s"], dtype=float))
        self.n_in_system_scale = n_in_system_scale
        self.avg_ctx_scale = avg_ctx_scale
        self.spec_sched_overhead = spec_sched_overhead
        self.avg_prompt = avg_prompt
        self.avg_decode = avg_decode
        self.workload_trace = None
        if workload_trace_path is not None:
            import json as _json
            with open(workload_trace_path) as _f:
                pairs = _json.load(_f)
            self.workload_trace = np.array(pairs, dtype=np.int64)  # [N, 2]: (prompt_tokens, decode_tokens)
        self.max_tokens = max_tokens
        self.reward_norm = reward_norm
        self.token_norm = token_norm
        self.reward_clip = reward_clip

        # vLLM-style scheduling knobs.
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_requests = num_requests
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.enable_chunked_prefill = enable_chunked_prefill

        # vLLM's SchedulerConfig check. The old, stricter guard
        # (budget >= max_num_seqs*(1+k)) is gone: when the budget cannot fit
        # every running seq at 1+k tokens, vLLM simply trims spec tokens and
        # leaves the remaining running requests unscheduled this step, and
        # _schedule() now reproduces that instead of refusing the config.
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens ({}) must be >= max_num_seqs ({})".format(
                    self.max_num_batched_tokens, self.max_num_seqs))

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self._holding_cost = 0.0
        self._tokens_this_step = 0
        # Steady-state window accumulators (Little's-law latency estimate):
        # time-integral of n_in_system, elapsed window time, and arrivals
        # counted, all restricted to [warmup, duration) -- see _execute().
        self._window_L = 0.0
        self._window_time = 0.0
        self._window_arrivals = 0
        self._last_prefill_tokens = 0

    # ---------- internal helpers ----------
    def _decoding_reqs(self):
        return [r for r in self.running if r.is_decoding]

    def _num_in_system(self):
        return len(self.running) + len(self.waiting)

    def _state(self):
        decoding = self._decoding_reqs()
        bs = len(decoding)
        n_in_system = self._num_in_system()   # len(running) + len(waiting)
        if decoding:
            avg_ctx = sum(r.context_len for r in decoding) / len(decoding)
        else:
            avg_ctx = 0.0
        return [
            float(n_in_system / (n_in_system + self.n_in_system_scale)),  # scalar: queue length (num in system)
            # forced into [0,1), but it hurts because n_in_system=500 vs 50 are totally different -- should not consider them equal.
            float(min(bs, self.max_num_seqs)) / self.max_num_seqs,  # vector[0]: batch size
            float(self.at.value),                                   # vector[1]: acceptance rate
            float(avg_ctx) / self.avg_ctx_scale,                    # vector[2]: avg ctx
        ]

    def _drain(self):
        """Move all arrivals with arrival_time <= clock into the waiting queue."""
        while self.ai < len(self.arrivals) and self.arrivals[self.ai][0] <= self.clock:
            ta, p, d = self.arrivals[self.ai]
            self.ai += 1
            self.waiting.append(_SpecRequest(self.rid, p, d, ta))
            self.rid += 1
            if self.warmup <= ta < self.duration:
                self._window_arrivals += 1

    # ---------- vLLM-style scheduling ----------
    def _schedule(self, k=0):
        """Form one unified batch the way vLLM V1 Scheduler.schedule() does.

        Returns a list of (request, num_new_tokens): the QUERY tokens the request
        puts into this step's forward pass. For a prefill that is the chunk;
        for a decode it is 1 + the number of spec tokens actually scheduled.

        RUNNING loop, aligned with vllm/v1/core/sched/scheduler.py:
          * self.running is walked IN LIST ORDER (admission order). There is no
            decode-first priority: a request still in chunked prefill that was
            admitted earlier takes its chunk before decodes behind it.
          * want = num_tokens_with_spec - num_computed_tokens
                 = 1 + k for a decode holding draft KV, 1 for one without,
                   remaining prompt for a prefill (capped by
                   long_prefill_token_threshold);
            num_new = min(want, token_budget).
          * A decode that gets num_new < 1 + k has its spec tokens TRIMMED to
            num_new - 1 (vLLM: spec_token_ids[:num_scheduled_spec_tokens]).
          * The loop stops once token_budget hits 0; running requests after
            that point are not scheduled this step (they wait, still holding
            KV and accruing holding cost).
        Not modelled: preemption (KV is reserved up front at admission, see the
        module docstring), prefix caching, max_model_len capping.
        """
        token_budget = self.max_num_batched_tokens
        scheduled = []

        # 1) RUNNING requests, in list order.
        for r in self.running:
            if token_budget <= 0:
                break
            if r.is_decoding:
                want = 1 + (k if r.has_draft_kv else 0)
            else:
                want = r.num_prompt_tokens - r.num_computed_tokens
                thr = self.long_prefill_token_threshold
                if 0 < thr < want:
                    want = thr
            num_new = min(want, token_budget)
            if num_new <= 0:
                continue
            scheduled.append((r, num_new))
            token_budget -= num_new

        # 2) Schedule WAITING requests (admission + new prefills).
        #    Respect the running cap and the token budget. No preemption.
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.max_num_seqs:# if running request large than max_num_seqs, stop
                break
            r = self.waiting[0]# take the first waiting request and check some condition afterwards

            # Up-front full-sequence KV reservation at admission time.
            need = r.prompt_len + r.decode_len
            if need > self.kv.max_tokens:
                raise ValueError(
                    "Request {} needs {} KV tokens, exceeding max_tokens={}".format(
                        r.req_id, need, self.kv.max_tokens
                    )
                )
            if not self.kv.can_allocate(need):
                # KV pool full -> cannot admit. With no preemption we simply
                # stop admitting and let running requests drain first.
                break

            num_new = r.num_prompt_tokens - r.num_computed_tokens  # full prompt (fresh)
            thr = self.long_prefill_token_threshold
            if 0 < thr < num_new:
                num_new = thr
            if not self.enable_chunked_prefill and num_new > token_budget:
                # Cannot fit the whole prompt and chunking is disabled: stop.
                break
            num_new = min(num_new, token_budget)
            if num_new <= 0:
                break
#after checking all the conditions right, we do:
            self.kv.allocate(need) #reserve kv cache
            self.waiting.popleft() #pop off from the waiting queue
            self.running.append(r)#add to running
            scheduled.append((r, num_new))
            token_budget -= num_new

        return scheduled

    def _unified_step_latency(self, decoders, prefills, k,
                              draft_prefill_tokens=0, decode_query=None, sd_engine=True):
        """Latency of one unified engine step.

        decoders             : list of decoding requests in the batch
        prefills             : list of (request, chunk) prefill-chunk entries
        k                    : speculation length chosen this step (0 = none)
        draft_prefill_tokens : prefill tokens also run through the draft model
                               (to build its KV); 0 = draft prefill skipped
        decode_query         : per-decoder query tokens actually scheduled
                               (1 + spec after budget trimming). None = untrimmed
                               (1+k with draft KV, 1 without), used by estimators.

            T_step = T_target(N_ctx, N_q_dec, N_prefill) + D_k + draft-prefill + sched
            N_q_dec = sum(decode_query)   (piecewise, knee 384)
            N_prefill = prefill chunk tokens (linear, 74 us/token)

        The draft runs k passes for every decoder that holds draft KV, even if
        the budget later trims its spec tokens -- that drafting work is wasted,
        as in vLLM.
        """
        c_dec = sum(r.context_len for r in decoders)
        c_pre = sum(r.context_len for r, _ in prefills)
        q_pre = sum(chunk for _, chunk in prefills)
        if self.cost_model == "v2":
            n_spec = sum(1 for r in decoders if r.has_draft_kv) if k > 0 else 0
            if decode_query is None:
                dq = [1 + (k if r.has_draft_kv else 0) for r in decoders]
            else:
                dq = list(decode_query)
            S = sum(1 for n in dq if n > 0)
            a_pre = sum(c * (r.num_computed_tokens + c / 2.0) for r, c in prefills)
            t = step_latency_v2(q_pre, sum(dq), S, c_dec, a_pre, k if n_spec else 0,
                                draft_prefill_tokens, sd_engine)
            if self.overhead_full_ms or self.overhead_part_ms:
                full = (q_pre + sum(dq)) >= self.max_num_batched_tokens
                t += (self.overhead_full_ms + (0.0 if full else self.overhead_part_ms)) * self.ovh_scale / 1000.0
            return t

        n_spec = sum(1 for r in decoders if r.has_draft_kv) if k > 0 else 0
        if decode_query is None:
            q_dec = (len(decoders) - n_spec) + n_spec * (k + 1)
        else:
            q_dec = sum(decode_query)

        t_verify = target_fwd(c_dec + c_pre, q_dec, q_pre)
        t_draft = draft_cost(k, n_spec, draft_prefill_tokens)
        # Non-forward-pass speculative overhead; 0 under the current profile
        # (see _SPEC_SCHED_OVERHEAD).
        t_sched = self.spec_sched_overhead * k * n_spec
        return t_verify + t_draft + t_sched

    def capacity(self, alpha):
        """Saturated throughput in requests/s at acceptance alpha (best fixed k), from capacity.json."""
        a, c = self._cap
        return float(np.interp(alpha, a, c))

    def decode_query_plan(self, decoders, k):
        """Per-decoder query tokens the scheduler WOULD give `decoders` at length
        k, with the budget applied in running order. Prefill chunks interleaved
        in the running list are not known in advance and are ignored, so this
        is an optimistic plan (used by the TurboSpec goodput estimate)."""
        budget = self.max_num_batched_tokens
        plan = []
        for r in decoders:
            want = 1 + (k if r.has_draft_kv else 0)
            n = min(want, budget)
            plan.append(n)          # n == 0: not scheduled this step
            budget -= n
        return plan

    def _retire(self, r):
        r.finish_t = self.clock
        self.kv.free(r.prompt_len + r.decode_len)
        if r.arrive_t >= self.warmup:
            self.done_latencies.append(r.latency)
            self.done_ttft.append(getattr(r, "first_t", r.finish_t) - r.arrive_t)
        self.running.remove(r)

    def _execute(self, scheduled, action, draft_prefill=True):
        """Run the scheduled unified batch: advance clock, accrue holding cost,
        advance tokens, retire finished requests."""
        self._last_prefill_tokens = 0
        if not scheduled:
            return

        n_in_system = self._num_in_system()

        # Partition the batch into decoders and prefill chunks.
        decoders = []
        decode_query = []   # 1 + spec tokens actually scheduled (after trim)
        prefills = []  # (request, chunk)
        for r, num_new in scheduled:
            if r.is_decoding:
                decoders.append(r)
                decode_query.append(num_new)
            else:
                prefills.append((r, num_new))

        k = self.k_spec if action == 1 else 0
        # Only chunks of requests that still have draft KV are worth running
        # through the draft; once a request lost it, it never regains it.
        draft_pre_tokens = (sum(c for r, c in prefills if r.has_draft_kv)
                            if draft_prefill else 0)
        step_t = self._unified_step_latency(decoders, prefills, k,
                                            draft_prefill_tokens=draft_pre_tokens,
                                            decode_query=decode_query,
                                            sd_engine=draft_prefill)
        self._last_prefill_tokens = sum(c for _, c in prefills)
        if not draft_prefill:
            # Draft prefill skipped -> these requests have no draft KV and can
            # never speculate (TurboSpec "prefill disabling").
            for r, _ in prefills:
                r.has_draft_kv = False
        # Steady-state window: [warmup, duration). Reward (holding_cost,
        # tokens_this_step) and the Little's-law window accumulators below
        # only count steps that START inside this window, so the reward
        # driving training excludes both the startup ramp (system filling
        # from empty) and the open-ended post-arrival drain tail -- neither
        # is representative steady-state load, and an unbounded drain tail
        # at high lambda could otherwise dominate the reward signal.
        in_steady_window = self.warmup <= self.clock < self.duration
       # Little's law: N=\lambda*T, when \lambda fix, T is linear with N, we want to minimize \ sum (step_t*n_in_system), the sum of weighted N
        if in_steady_window:
            self._holding_cost += step_t * n_in_system
            self._window_L += step_t * n_in_system
            self._window_time += step_t
        self.clock += step_t

        # --- advance prefill chunks (no speculation on prefill) ---
        for r, chunk in prefills:
            r.num_computed_tokens += chunk
            r.context_len = r.num_computed_tokens
            if r.num_computed_tokens >= r.num_prompt_tokens:
                # Prompt finished this step: the last prefill position samples
                # the first output token (no extra latency -- it's part of this
                # forward pass). It joins the decode set next step.
                r.prefill_done = True
                r.first_t = self.clock
                first = min(1, r.decode_len)
                r.num_output_tokens += first
                r.num_computed_tokens = r.num_prompt_tokens + first
                r.context_len = r.num_computed_tokens
                if in_steady_window:
                    self._tokens_this_step += first
                if r.is_done:
                    self._retire(r)

        # --- advance decoders (apply speculation / acceptance) ---
        for r, nq in zip(decoders, decode_query):
            # spec tokens actually verified: k, fewer if the budget trimmed
            # them, 0 without draft KV or when k == 0
            kr = nq - 1
            m, v = spec_simulate_acceptance(kr, self.true_alpha, self.rng, self.accept_decay)
            adv = min(m + 1, r.decode_len - r.num_output_tokens)
            r.num_output_tokens += adv
            r.num_computed_tokens += adv
            r.context_len = r.num_computed_tokens
            if in_steady_window:
                self._tokens_this_step += adv
            if kr > 0: #only update acc when do sd
                self.at.update(m, v)
            if r.is_done:
                self._retire(r)

    def _advance_to_work(self):
        """Ensure the next step has something to schedule; idle-skip the clock to
        the next arrival if the system is momentarily empty. Returns False once
        the system is fully drained (episode over)."""
        while True:
            self._drain()
            if self.running or self.waiting:
                return True
            if self.ai < len(self.arrivals):
                # System idle: jump to the next arrival (no holding cost while
                # nothing is in the system).
                self.clock = self.arrivals[self.ai][0]
                continue
            return False

    # ---------- gym API ----------
    def step(self, action, draft_prefill=True):
        """One unified scheduler step of the serving system.

        draft_prefill: run this step's prefill chunks through the draft model
            too (default True: a speculation-capable engine keeps draft KV for
            every request). False skips that cost, but the affected requests
            lose draft KV and can never speculate (TurboSpec prefill disabling;
            also used by the pure k=0 / no-SD baseline).

        Gymnasium 5-tuple: returns (obs, reward, terminated, truncated, info).
        """
        action = int(action)  # coerce np.int64 -> int (avoids spaces.contains quirks)
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1, got {!r}".format(action))

        self._holding_cost = 0.0
        self._tokens_this_step = 0
        t_before = self.clock

        k = self.k_spec if action == 1 else 0
        scheduled = self._schedule(k)
        self._execute(scheduled, action, draft_prefill=draft_prefill)

        has_work = self._advance_to_work()
        terminated = not has_work    # natural terminal: episode fully drained
        truncated = False            # no time-limit truncation in this env
        nextState = self._state()
        # Holding-cost reward (Little's-law latency proxy). The token term is
        # off by default (token_norm large) so this is pure -holding/reward_norm;
        # trained with the average-reward objective, NOT discounting. No clip by
        # default -- clipping would flatten the overload gradient the threshold
        # depends on. reward_clip can be set for discounted-RL experiments.
        reward = (self._tokens_this_step / self.token_norm
                  - self._holding_cost / self.reward_norm)
        if self.reward_clip is not None:
            reward = max(reward, self.reward_clip)

        # SMDP step duration (wall-clock advanced by this step, including any
        # idle-skip to the next arrival). Speculation changes how much time/work
        # one step represents, so average-reward training must weight by tau
        # rather than by step count.
        tau = self.clock - t_before

        # Steady-state mean latency via Little's law (L = lambda_hat * W):
        # L is the time-average number in system over the [warmup, duration)
        # window, lambda_hat is the arrival rate actually observed in that
        # same window. This is a window-average estimate, not a per-request
        # average, so it isn't skewed by a handful of stragglers whose
        # completion happens to land far outside the window (the failure
        # mode of averaging done_latencies directly, especially at high
        # lambda where the drain tail can dominate the per-request mean).
        mean_latency_steady = None
        if self._window_time > 0 and self._window_arrivals > 0:
            L = self._window_L / self._window_time
            lam_hat = self._window_arrivals / self._window_time
            mean_latency_steady = L / lam_hat if lam_hat > 0 else None

        info = {
            'lam': self.lam,
            'clock': self.clock,
            'tau': tau,
            'mean_latency': (float(np.mean(self.done_latencies))
                             if self.done_latencies else None),
            'mean_latency_steady': mean_latency_steady,
            # prompt tokens prefilled this step (TurboSpec counts a prefill
            # skip only on steps that actually ran a prefill)
            'prefill_tokens': self._last_prefill_tokens,
        }
        return nextState, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """New episode with a freshly randomized arrival rate.

        Gymnasium signature: reset(seed=None, options=None) -> (obs, info).
        """
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.RandomState(seed)

        self.lam = float(self.rng.uniform(self.lam_low, self.lam_high))
        # Re-randomize acceptance rate per episode when a range is configured.
        if self.alpha_low is not None and self.alpha_high is not None:
            self.true_alpha = float(self.rng.uniform(self.alpha_low, self.alpha_high))
        if self.rho_low is not None:
            if self._cap is None:
                raise ValueError("rho sampling needs capacity_path (run capacity.py first)")
            if self.rng.rand() < self.p_over:
                self.rho = float(self.rng.uniform(1.0, 1.15))
            else:
                self.rho = float(self.rng.uniform(self.rho_low, self.rho_high))
            self.lam = self.rho * self.capacity(self.true_alpha)
        j = self.overhead_jitter
        self.ovh_scale = float(self.rng.uniform(1 - j, 1 + j)) if j > 0 else 1.0
        self.clock = 0.0
        self.kv = _SpecKVCache(self.max_tokens)
        self.at = _SpecAcceptanceTracker(window=20, prior=self.alpha_prior, prior_weight=2)
        self.waiting = deque()
        self.running = []
        self.done_latencies = []
        self.done_ttft = []
        self.rid = 0
        self._holding_cost = 0.0
        self._tokens_this_step = 0
        self._window_L = 0.0
        self._window_time = 0.0
        self._window_arrivals = 0
        self._last_prefill_tokens = 0

        self.arrivals = []
        t = 0.0
        while True:
            t += self.rng.exponential(1.0 / self.lam)
            if self.num_requests is None:
                if t >= self.duration:
                    break
            elif len(self.arrivals) >= self.num_requests:
                break
            if self.workload_trace is not None:
                idx = self.rng.randint(0, len(self.workload_trace))
                p, d = int(self.workload_trace[idx, 0]), int(self.workload_trace[idx, 1])
                p, d = max(1, p), max(1, d)
            else:
                p = max(1, int(self.rng.normal(self.avg_prompt, self.avg_prompt * 0.3)))
                d = max(1, int(self.rng.normal(self.avg_decode, self.avg_decode * 0.3)))
            self.arrivals.append((t, p, d))
        self.ai = 0

        self._advance_to_work()
        return self._state(), {}
