"""
Speculative Decoding serving environment for DeepTOP.

State (a python list; state[0] is the SCALAR compared against the
learned threshold, state[1:] is the VECTOR fed to the actor):
    state = [ batch_size / MAX_NUM_SEQS,   # scalar
              alpha_estimate,              # vector[0]
              backlog / 50 ]               # vector[1]

Action: 1 -> speculate this step (k=5);  0 -> normal decode (k=0)

The learned DeepTOP policy is:
    speculate  iff  threshold(alpha_est, backlog) > normalized_batch_size

Reward (per step, includes queueing):
    reward = -(step_latency * num_requests_in_system) / reward_norm
    clipped to [-1, 0] for stability.

Episode: fixed wall-clock duration with Poisson arrivals; the arrival
rate lambda is re-randomized every episode (generalization across loads).
Episode terminates when the clock passes `duration` and all requests drain.
"""

import gym
import numpy as np
from gym import spaces


# =============================================================================
# Latency model (SmartSpec Eq.7), A100 + LLaMA-7B target + LLaMA-160M draft
# =============================================================================
_ALPHA_T, _GAMMA_T, _DELTA_T = 1.5e-6, 2.5e-4, 0.011
_ALPHA_D, _GAMMA_D, _DELTA_D = 0.3e-6, 1.5e-6, 0.002
_K_SPEC = 5              # speculation length when SD is on
_MAX_NUM_SEQS = 32       # max requests per decode step (vLLM-style cap)


def spec_t_fwd(nc, nb, a=_ALPHA_T, g=_GAMMA_T, d=_DELTA_T):
    """Single forward pass latency: T = delta + alpha*N_context + gamma*N_batched."""
    return d + a * nc + g * nb


def spec_t_decode_step(batch, k):
    """Decode step latency. k=0: target only. k>0: k draft passes + 1 verify pass."""
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
    def __init__(self, req_id, prompt_len, decode_len, arrive_t):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.decode_len = decode_len
        self.arrive_t = arrive_t
        self.context_len = 0
        self.decode_done = 0
        self.prefill_done = False
        self.finish_t = -1.0

    @property
    def is_done(self):
        return self.decode_done >= self.decode_len

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
                 lam_low=2.0, lam_high=18.0, true_alpha=0.7,
                 avg_prompt=128.0, avg_decode=128.0,
                 max_tokens=200000, reward_norm=50.0):
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

        self.action_space = spaces.Discrete(2)

    # ---------- internal helpers ----------
    def _state(self):
        bs = len(self.active)
        backlog = len(self.prefill_q) + len(self.waiting)
        return [
            float(min(bs, _MAX_NUM_SEQS)) / _MAX_NUM_SEQS,   # scalar
            float(self.at.value),                            # vector[0]
            float(min(backlog, 50)) / 50.0,                  # vector[1]
        ]

    def _num_in_system(self):
        return len(self.active) + len(self.waiting) + len(self.prefill_q)

    def _drain(self):
        while self.ai < len(self.arrivals) and self.arrivals[self.ai][0] <= self.clock:
            ta, p, d = self.arrivals[self.ai]
            self.ai += 1
            self.prefill_q.append(_SpecRequest(self.rid, p, d, ta))
            self.rid += 1

    def _admit(self):
        i = 0
        while i < len(self.prefill_q):
            r = self.prefill_q[i]
            if self.kv.can_allocate(r.prompt_len + r.decode_len):
                self.kv.allocate(r.prompt_len + r.decode_len)
                self.waiting.append(self.prefill_q.pop(i))
            else:
                i += 1

    def _prefill(self):
        for r in list(self.waiting):
            if not r.prefill_done:
                self.clock += spec_t_prefill(r.prompt_len)
                r.context_len = r.prompt_len
                r.prefill_done = True
                self.waiting.remove(r)
                self.active.append(r)

    def _advance_to_decodable(self):
        """Advance until there is something to decode; returns False if drained."""
        while True:
            self._drain(); self._admit(); self._prefill(); self._drain(); self._admit()
            if self.active:
                return True
            if self.ai < len(self.arrivals):
                self.clock = self.arrivals[self.ai][0]
                continue
            if self.prefill_q or self.waiting:
                self.clock += 1e-4
                continue
            return False

    # ---------- gym API ----------
    def _calRewardAndState(self, action):
        k = _K_SPEC if action == 1 else 0
        batch = self.active[:_MAX_NUM_SEQS]
        n_in_system = self._num_in_system()

        step_t = spec_t_decode_step(batch, k)
        self.clock += step_t

        for r in list(batch):
            m, v = spec_simulate_acceptance(k, self.true_alpha, self.rng)
            adv = min(m + 1, r.decode_len - r.decode_done)
            r.decode_done += adv
            r.context_len += adv
            if k > 0:
                self.at.update(m, v)
            if r.is_done:
                r.finish_t = self.clock
                self.kv.free(r.prompt_len + r.decode_len)
                if r.arrive_t >= self.warmup:
                    self.done_latencies.append(r.latency)
                self.active.remove(r)

        # Clip reward to [-1, 0]: with reward_norm=50 the natural reward is
        # already in this range across all lambdas; the clip is a safety net.
        reward = -(step_t * n_in_system) / self.reward_norm
        reward = max(reward, -1.0)
        nextState = self._state()
        return nextState, reward

    def step(self, action):
        """One decode step of the serving system."""
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)

        has_work = self._advance_to_decodable()
        done = not has_work          # episode ends when fully drained
        nextState = self._state()

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
        self.prefill_q = []
        self.waiting = []
        self.active = []
        self.done_latencies = []
        self.rid = 0

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

        self._advance_to_decodable()
        return self._state()
