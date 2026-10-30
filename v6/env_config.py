"""Single source of truth for the simulator settings used by training AND evaluation
(main_DeepTOP_multik.py, eval_multik.py). Change values here only."""
import os
from MDP_Env import SpecDecodingEnv

K_MAX = 7                    # longest draft length (real engine: --k-max 7)
K_MIN = 1                    # shortest draft length; actions k in {K_MIN, ..., K_MAX} (no k=0)
BUDGET = 384                 # max_num_batched_tokens (real engine: --max-num-batched-tokens 384)
MAX_NUM_SEQS = 256           # real engine: --max-num-seqs 256
COST_MODEL = "v2"            # unified step model with mixing penalty (validated vs real vLLM)
N_SCALE = 256.0              # scalar state = n_in_system / (n_in_system + N_SCALE)
CTX_SCALE = 240.0            # mean decode context on HumanEval
MAX_TOKENS = 147216          # KV pool of the SD engine
DURATION, WARMUP = 120.0, 20.0
TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humaneval_trace.json")
EVAL_ALPHA = 0.838           # HumanEval, acceptance of the first draft position
# v6: acceptance falls with draft position: position i is accepted with prob. alpha * DECAY**(i-1)
# (fit on real untrimmed decode steps, k=1..7: tau(4..7) = 3.42/3.69/3.88/4.00 vs real 3.41/3.64/3.87/4.00)
ACCEPT_DECAY = 0.96
# v4 step-time correction (real online step logs, 2026-10-20): +0.6 ms per step, +1.9 ms more if not full
OVH_FULL_MS, OVH_PART_MS = 0.6, 1.9
CAPACITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capacity.json")


def make_env(seed, lam_low, lam_high, alpha=EVAL_ALPHA, alpha_low=None, alpha_high=None,
             alpha_prior=None, reward_norm=1.33, duration=None, warmup=None,
             rho_low=None, rho_high=None, p_over=0.0, overhead_jitter=0.0, num_requests=None,
             accept_decay=None):
    """Evaluation calls use the defaults (fixed lambda, corrected step time, no jitter)."""
    return SpecDecodingEnv(seed=seed, duration=DURATION if duration is None else duration,
                           warmup=WARMUP if warmup is None else warmup,
                           lam_low=lam_low, lam_high=lam_high, num_requests=num_requests,
                           overhead_full_ms=OVH_FULL_MS, overhead_part_ms=OVH_PART_MS,
                           overhead_jitter=overhead_jitter, rho_low=rho_low, rho_high=rho_high,
                           p_over=p_over, capacity_path=CAPACITY if rho_low is not None else None,
                           accept_decay=ACCEPT_DECAY if accept_decay is None else accept_decay,
                           true_alpha=alpha, alpha_low=alpha_low, alpha_high=alpha_high,
                           alpha_prior=alpha if alpha_prior is None else alpha_prior,
                           workload_trace_path=TRACE, max_num_seqs=MAX_NUM_SEQS,
                           max_num_batched_tokens=BUDGET, max_tokens=MAX_TOKENS,
                           n_in_system_scale=N_SCALE, avg_ctx_scale=CTX_SCALE,
                           spec_sched_overhead=0.0, reward_norm=reward_norm,
                           cost_model=COST_MODEL, k_spec=K_MAX)


def env_step(env, k, sd_engine=True):
    """Run one step with speculation length k (0 = no speculation this step).
    sd_engine=False models a pure no-SD engine (no drafter at all)."""
    env.k_spec = max(int(k), 1)
    return env.step(1 if k > 0 else 0, draft_prefill=sd_engine)
