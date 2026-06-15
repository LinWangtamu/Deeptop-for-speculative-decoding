"""
Speculative Decoding serving environment for DeepTOP.

Scheduling now mirrors vLLM V1's *synchronous* core (no async scheduler):

  * No prefill / decode "phase" split. Each step forms ONE unified batch that
    can mix decode tokens (1 query token per decoding seq) AND prefill chunks,
    all sharing a single `max_num_batched_tokens` budget -- exactly like
    vllm Scheduler.schedule(): it first packs RUNNING requests, then admits
    WAITING requests until the budget or the running cap is hit.

  * `max_num_seqs` is a hard cap on the number of RUNNING requests, enforced at
    admission time (the waiting->running gate). It is NOT a post-hoc slice of
    an unbounded `active` list anymore, so the 33rd request truly waits instead
    of sitting in the batch holding KV while never decoding.

  * Long prompts are CHUNKED across steps: if a prompt does not fit the
    remaining token budget, only `min(prompt_remaining, budget)` tokens are
    scheduled this step and the rest continues next step (chunked prefill).

  * KV is still reserved UP FRONT (prompt_len + decode_len) when a request is
    first admitted to RUNNING, and there is NO preemption. This is a deliberate
    simplification: with max_num_seqs=32 and a 200k-token pool the KV ceiling is
    effectively never the binding constraint, so incremental KV + preemption
    would be dead code in this regime.

State (a python list; state[0] is the SCALAR compared against the learned
threshold, state[1:] is the VECTOR fed to the actor):
    state = [ decode_batch / max_num_seqs,   # scalar
              alpha_estimate,                # vector[0]
              average_context / 256,         # vector[1]
              backlog / 50 ]                 # vector[2]

  decode_batch : number of RUNNING requests currently in the decode phase
                 (this is the set the speculate/no-speculate action acts on)
  backlog      : len(waiting) + (#running still prefilling)

Action: 1 -> speculate this step (k=5);  0 -> normal decode (k=0)

The learned DeepTOP policy is:
    speculate iff threshold(alpha_est, average_context, backlog)
                  > normalized_decode_batch

Reward (per step, includes queueing):
    reward = -(holding_cost accumulated during the step) / reward_norm
    clipped to [-1, 0] for stability.
    The holding cost of a step covers the WHOLE unified forward pass (prefill
    chunks + decode), since prefill is no longer a separate clock advance.

Episode: fixed wall-clock duration with Poisson arrivals; the arrival rate
lambda is re-randomized every episode (generalization across loads). Episode
terminates once the clock has passed `duration` and all requests have drained.
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
# Latency model (SmartSpec Eq.7), A100 + LLaMA-7B target + LLaMA-160M draft
# =============================================================================
_ALPHA_T, _GAMMA_T, _DELTA_T = 1.5e-6, 2.5e-4, 0.011
_ALPHA_D, _GAMMA_D, _DELTA_D = 0.3e-6, 1.5e-6, 0.002
_K_SPEC = 5                     # speculation length when SD is on
_MAX_NUM_SEQS = 32              # max RUNNING requests (vLLM max_num_seqs cap)
_MAX_NUM_BATCHED_TOKENS = 2048  # per-step token budget (vLLM max_num_batched_tokens)


def spec_t_fwd(nc, nb, a=_ALPHA_T, g=_GAMMA_T, d=_DELTA_T):
    """Single forward pass latency: T = delta + alpha*N_context + gamma*N_batched."""
    return d + a * nc + g * nb


def spec_t_decode_step(batch, k):
    """Decode step latency. k=0: target only. k>0: k draft passes + 1 verify pass.

    Kept for reference / unit tests. The env now uses `_unified_step_latency`,
    which is a strict generalization (it reduces to this when the batch has no
    prefill chunks).
    """
    nc = sum(r.context_len for r in batch)
    bs = len(batch)
    if k == 0:
        return spec_t_fwd(nc, bs)
    return k * spec_t_fwd(nc, bs, _ALPHA_D, _GAMMA_D, _DELTA_D) + spec_t_fwd(nc, bs * (k + 1))


def spec_t_prefill(pl):
    return spec_t_fwd(0, pl)


# =============================================================================
# Per-request, KV cache, acceptance tracking, stochastic acceptance
# =============================================================================
class _SpecRequest(object):
    """vLLM-style request bookkeeping.

    A request is PREFILLING while num_computed_tokens < num_prompt_tokens and
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


class _SpecAcceptanceTracker(object):
    """Definition A: alpha = total accepted / total VERIFIED tokens (moving window)."""

    def __init__(self, window=20, init=0.7):
        self._buf = [(init, 1.0)] * (window // 2)
        self._w = window

    def update(self, acc, ver):
        if ver > 0:
            self._buf.append((float(acc), float(ver)))
            if len(self._buf) > self._w:
                self._buf.pop(0)

    @property
    def value(self):
        ta = sum(a for a, _ in self._buf)
        tv = sum(v for _, v in self._buf)
        return ta / tv if tv > 0 else 0.7


def spec_simulate_acceptance(k, true_alpha, rng):
    """Returns (m, n_verified). Early-stop at first rejection."""
    if k == 0:
        return 0, 0
    m = 0
    for _ in range(k):
        if rng.random() < true_alpha:
            m += 1
        else:
            return m, m + 1
    return m, k


# =============================================================================
# Gym environment
# =============================================================================
class SpecDecodingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed, duration=120.0, warmup=20.0,
                 lam_low=0.5, lam_high=20.0, true_alpha=0.7,
                 avg_prompt=128.0, avg_decode=128.0,
                 max_tokens=200000, reward_norm=50.0,
                 max_num_seqs=_MAX_NUM_SEQS,
                 max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
                 long_prefill_token_threshold=0,
                 enable_chunked_prefill=True):
        super(SpecDecodingEnv, self).__init__()
        self.seed_val = seed
        self.rng = np.random.RandomState(seed)
        self.duration = duration
        self.warmup = warmup
        self.lam_low = lam_low
        self.lam_high = lam_high
        self.true_alpha = true_alpha
        self.avg_prompt = avg_prompt
        self.avg_decode = avg_decode
        self.max_tokens = max_tokens
        self.reward_norm = reward_norm

        # vLLM-style scheduling knobs.
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.enable_chunked_prefill = enable_chunked_prefill

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._holding_cost = 0.0

    # ---------- internal helpers ----------
    def _decoding_reqs(self):
        return [r for r in self.running if r.is_decoding]

    def _num_in_system(self):
        return len(self.running) + len(self.waiting)

    def _state(self):
        decoding = self._decoding_reqs()
        bs = len(decoding)
        # backlog = queued requests + running requests still prefilling.
        backlog = len(self.waiting) + (len(self.running) - bs)
        if decoding:
            avg_ctx = sum(r.context_len for r in decoding) / len(decoding)
        else:
            avg_ctx = 0.0
        return [
            float(min(bs, self.max_num_seqs)) / self.max_num_seqs,  # scalar
            float(self.at.value),                                   # vector[0]: alpha
            float(avg_ctx) / 256.0,                                 # vector[1]: avg ctx
            float(min(backlog, 50)) / 50.0,                         # vector[2]: backlog
        ]

    def _drain(self):
        """Move all arrivals with arrival_time <= clock into the waiting queue."""
        while self.ai < len(self.arrivals) and self.arrivals[self.ai][0] <= self.clock:
            ta, p, d = self.arrivals[self.ai]
            self.ai += 1
            self.waiting.append(_SpecRequest(self.rid, p, d, ta))
            self.rid += 1

    # ---------- vLLM-style scheduling ----------
    def _schedule(self):
        """Form one unified batch the way vLLM Scheduler.schedule() does.

        Returns a list of (request, num_new_tokens) pairs. `num_new_tokens` is
        the number of QUERY tokens this request contributes to the forward pass
        (1 for a decode, or the prefill-chunk size for a prefill).
        """
        token_budget = self.max_num_batched_tokens
        scheduled = []

        # 1) Schedule RUNNING requests first (decodes + continuing prefill chunks).
        for r in self.running:
            if token_budget <= 0:
                break
            if r.is_decoding:
                # Decode: a single query token (speculation is handled in the
                # latency / token-advance logic, not by inflating num_new here).
                num_new = 1
            else:
                # Continue prefilling this request's prompt.
                num_new = r.num_prompt_tokens - r.num_computed_tokens
                thr = self.long_prefill_token_threshold
                if 0 < thr < num_new:
                    num_new = thr
            num_new = min(num_new, token_budget)
            if num_new <= 0:
                continue
            scheduled.append((r, num_new))
            token_budget -= num_new

        # 2) Schedule WAITING requests (admission + new prefills).
        #    Respect the running cap and the token budget. No preemption.
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.max_num_seqs:
                break
            r = self.waiting[0]

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

            self.kv.allocate(need)
            self.waiting.popleft()
            self.running.append(r)
            scheduled.append((r, num_new))
            token_budget -= num_new

        return scheduled

    def _unified_step_latency(self, decoders, prefills, k):
        """Latency of one unified forward pass.

        decoders : list of decoding requests (1 query token each)
        prefills : list of (request, chunk) prefill-chunk entries
        k        : speculation length for the decode set (0 = no speculation)

        Reduces exactly to spec_t_decode_step when there are no prefills.
        Speculation only applies to the decode set; prefill chunks ride along in
        the single target forward pass.
        """
        c_dec = sum(r.context_len for r in decoders)
        n_dec = len(decoders)
        c_pre = sum(r.context_len for r, _ in prefills)
        q_pre = sum(chunk for _, chunk in prefills)

        if k == 0 or n_dec == 0:
            # One target forward pass over the whole unified batch:
            # decoders contribute 1 query each, prefills contribute their chunk.
            return spec_t_fwd(c_dec + c_pre, n_dec + q_pre)

        # Speculative decode with decoders present:
        #   * k draft forward passes over the decode set (draft model)
        #   * 1 target verify pass over the FULL batch; decoders verify (k+1)
        #     positions each, prefill chunks contribute their query tokens.
        t_draft = k * spec_t_fwd(c_dec, n_dec, _ALPHA_D, _GAMMA_D, _DELTA_D)
        t_verify = spec_t_fwd(c_dec + c_pre, n_dec * (k + 1) + q_pre)
        return t_draft + t_verify

    def _retire(self, r):
        r.finish_t = self.clock
        self.kv.free(r.prompt_len + r.decode_len)
        if r.arrive_t >= self.warmup:
            self.done_latencies.append(r.latency)
        self.running.remove(r)

    def _execute(self, scheduled, action):
        """Run the scheduled unified batch: advance clock, accrue holding cost,
        advance tokens, retire finished requests."""
        if not scheduled:
            return

        n_in_system = self._num_in_system()

        # Partition the batch into decoders and prefill chunks.
        decoders = []
        prefills = []  # (request, chunk)
        for r, num_new in scheduled:
            if r.is_decoding:
                decoders.append(r)
            else:
                prefills.append((r, num_new))

        k = _K_SPEC if action == 1 else 0
        step_t = self._unified_step_latency(decoders, prefills, k)

        self._holding_cost += step_t * n_in_system
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
                first = min(1, r.decode_len)
                r.num_output_tokens += first
                r.num_computed_tokens = r.num_prompt_tokens + first
                r.context_len = r.num_computed_tokens
                if r.is_done:
                    self._retire(r)

        # --- advance decoders (apply speculation / acceptance) ---
        for r in decoders:
            m, v = spec_simulate_acceptance(k, self.true_alpha, self.rng)
            adv = min(m + 1, r.decode_len - r.num_output_tokens)
            r.num_output_tokens += adv
            r.num_computed_tokens += adv
            r.context_len = r.num_computed_tokens
            if k > 0:
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
    def step(self, action):
        """One unified scheduler step of the serving system."""
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 or 1, got {!r}".format(action))

        self._holding_cost = 0.0

        scheduled = self._schedule()
        self._execute(scheduled, action)

        has_work = self._advance_to_work()
        done = not has_work          # episode ends when fully drained
        nextState = self._state()
        reward = max(-self._holding_cost / self.reward_norm, -1.0)

        info = {
            'lam': self.lam,
            'clock': self.clock,
            'mean_latency': (float(np.mean(self.done_latencies))
                             if self.done_latencies else None),
        }
        return nextState, reward, done, info

    def reset(self):
        """New episode with a freshly randomized arrival rate."""
        self.lam = float(self.rng.uniform(self.lam_low, self.lam_high))
        self.clock = 0.0
        self.kv = _SpecKVCache(self.max_tokens)
        self.at = _SpecAcceptanceTracker(init=self.true_alpha)
        self.waiting = deque()
        self.running = []
        self.done_latencies = []
        self.rid = 0
        self._holding_cost = 0.0

        self.arrivals = []
        t = 0.0
        while t < self.duration:
            t += self.rng.exponential(1.0 / self.lam)
            if t >= self.duration:
                break
            p = max(1, int(self.rng.normal(self.avg_prompt, self.avg_prompt * 0.3)))
            d = max(1, int(self.rng.normal(self.avg_decode, self.avg_decode * 0.3)))
            self.arrivals.append((t, p, d))
        self.ai = 0

        self._advance_to_work()
        return self._state()
