
import gym
import math
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from gym import spaces
from random_process import OrnsteinUhlenbeckProcess


class chargingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment for charging EVs 
    '''

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, seed, min_charge, max_charge, min_deadline, max_deadline, theta, mu, sigma, dt, x0):
        super(chargingEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_deadline = min_deadline
        self.max_deadline = max_deadline
        self.ou_process = OrnsteinUhlenbeckProcess(theta, mu, sigma, dt, x0)

        self.x = self.ou_process.sample()
        self.charge = self.myRandomPRNG.randint(self.min_charge, self.max_charge)
        self.deadline = self.myRandomPRNG.randint(self.min_deadline, self.max_deadline)

        self.action_space = spaces.Discrete(2)



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        if self.charge <= 0:
            action = 0
        reward = 0
        if action == 1:
            reward = 1 - self.x
            self.deadline = self.deadline - 1
            self.charge = self.charge - 1
        elif action == 0:
            reward = 0
            self.deadline = self.deadline - 1

        if self.deadline <= 0:  # A new vehicle arrives
            reward = reward - 0.2 * self.charge * self.charge
            self.charge = self.myRandomPRNG.randint(self.min_charge, self.max_charge)
            self.deadline = self.myRandomPRNG.randint(self.min_deadline, self.max_deadline)

        self.x = self.ou_process.sample()
        nextState = [self.x, self.charge, self.deadline]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.ou_process.reset_states()
        self.x = self.ou_process.sample()
        self.charge = self.myRandomPRNG.randint(self.min_charge, self.max_charge)
        self.deadline = self.myRandomPRNG.randint(self.min_deadline, self.max_deadline)
        initialState = [self.x, self.charge, self.deadline]

        return initialState

class inventoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment for inventory management
    parameters are: 
    cap: capacity of the warehouse
    order_size: size of an order
    demand_list: a list of seasonal demand
    selling_price: selling price
    '''

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, seed, cap, order_size, demand_list, selling_price):
        super(inventoryEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = np.random.RandomState(self.seed)
        self.cap = cap
        self.order_size = order_size
        self.demand_list = demand_list
        self.selling_price = selling_price

        self.t = 0
        self.inventory = self.order_size
        self.arriving = 0  # 1 if there is an order arriving at the end of this slot

        self.action_space = spaces.Discrete(2)



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''

        # First, determine demand in this slot and calculate reward
        this_demand = self.myRandomPRNG.poisson(self.demand_list[self.t])
        reward = self.selling_price * min(this_demand, self.inventory)
        self.inventory = self.inventory - min(this_demand, self.inventory)

        # Next, incur holding cost, update order arrival and current time
        reward = reward - self.inventory
        if action > 0:
            self.inventory = min(self.inventory + self.order_size, self.cap)
        self.t = (self.t + 1) % len(self.demand_list)

        # Finally, generate state
        nextState = [self.inventory, self.t]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.t = 0
        self.inventory = self.order_size
        self.arriving = 0  # 1 if there is an order arriving at the end of this slot

        initialState = [self.inventory, self.t]

        return initialState


class MakeToStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, seed, customer_classes, m, B, mu):
        super(MakeToStockEnv, self).__init__()
        self.seed = seed 
        self.customer_classes = customer_classes
        self.m = m 
        self.B = B 
        self.mu = mu 
        self.myRandomPRNG = np.random.RandomState(self.seed)
        
        self.k_choices = np.arange(0, self.customer_classes+1)

        self.R = np.linspace(200, 10, num=self.customer_classes)

    def _h_s(self, scalar_state):
        return 0.1*(scalar_state)**2


    def _calRewardAndState(self, action):

        if action == 0:
            reward = -1*self._h_s(self.scalar_state)
            self.scalar_state = self.scalar_state
        elif action == 1:

            if self.scalar_state == (self.m + self.B):
                reward = -1*self._h_s(self.scalar_state)
                self.scalar_state = self.scalar_state

            else:
                reward = self.R[self.vector_state-1] - self._h_s(self.scalar_state) #self.vector_state - 1 for correct indexing
                self.scalar_state = min(self.m + self.B, self.scalar_state + 1)
        else:
            print(f'wrong action value. exiting...')
            exit(2)
    
        k = 0
        while (k <= 1): 
            w = min(self.m, self.scalar_state)
            probabilites = np.append( (self.mu*w)/(self.mu*w + self.customer_classes), np.repeat(1 / (self.mu*w + self.customer_classes), self.customer_classes))
            
            k = self.myRandomPRNG.choice(self.k_choices, p=probabilites)
            
            if k == 0:
                self.scalar_state = max(0, self.scalar_state - 1)

        self.vector_state = k 

        nextState = [self.scalar_state, self.vector_state]

        return nextState, reward 


    def step(self, action):

        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info 


    def reset(self):
        self.scalar_state = self.myRandomPRNG.randint(0, self.m+self.B+1)
        self.vector_state = self.myRandomPRNG.randint(1, self.customer_classes+1)

        initialState = [self.scalar_state, self.vector_state]

        return initialState



# =============================================================================
# Speculative Decoding serving environment
# =============================================================================
#
# Latency model (SmartSpec Eq.7), A100 / LLaMA-7B target / LLaMA-160M draft
_ALPHA_T, _GAMMA_T, _DELTA_T = 1.5e-6, 2.5e-4, 0.011
_ALPHA_D, _GAMMA_D, _DELTA_D = 0.3e-6, 1.5e-6, 0.002
_K_SPEC = 5            # speculation length when SD is on
_MAX_NUM_SEQS = 32     # max requests per decode step (vLLM-style cap)


def spec_t_fwd(nc, nb, a=_ALPHA_T, g=_GAMMA_T, d=_DELTA_T):
    '''Single forward pass latency: T = delta + alpha*N_context + gamma*N_batched'''
    return d + a * nc + g * nb


def spec_t_decode_step(batch, k):
    '''Decode step latency. k=0: target only. k>0: k draft passes + 1 verify pass.'''
    nc = sum(r.context_len for r in batch)
    bs = len(batch)
    if k == 0:
        return spec_t_fwd(nc, bs)
    return k * spec_t_fwd(nc, bs, _ALPHA_D, _GAMMA_D, _DELTA_D) + spec_t_fwd(nc, bs * (k + 1))


def spec_t_prefill(pl):
    return spec_t_fwd(0, pl)


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
    '''Definition A: alpha = total accepted / total VERIFIED tokens (moving window).'''
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
    '''Returns (m, n_verified). Early-stop at first rejection.'''
    if k == 0:
        return 0, 0
    m = 0
    for _ in range(k):
        if rng.random() < true_alpha:
            m += 1
        else:
            return m, m + 1
    return m, k


class SpecDecodingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment for speculative-decoding on/off control in a
    vLLM-style continuous-batching serving system.

    State (a python list; state[0] is the SCALAR compared against the
    learned threshold, state[1:] is the VECTOR fed to the actor):
        state = [ batch_size / MAX_NUM_SEQS,   # scalar
                  alpha_estimate,              # vector[0]
                  backlog / 50 ]               # vector[1]

    Action: 1 -> speculate this step (k=5);  0 -> normal decode (k=0)
    So the learned DeepTOP policy is:
        speculate  iff  threshold(alpha_est, backlog) > normalized_batch_size
    i.e. a state-adaptive batch-size threshold.

    Reward (per step, includes queueing):
        reward = -(step_latency * num_requests_in_system) / reward_norm
    Cumulative reward ~ negative total waiting time across requests.

    Episode: fixed wall-clock duration with Poisson arrivals; the arrival
    rate lambda is re-randomized every episode (generalization across loads).
    Episode terminates when the clock passes `duration` and all admitted
    requests have drained.
    '''

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
        '''Advance until there is something to decode; returns False if drained.'''
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

        reward = -(step_t * n_in_system) / self.reward_norm
        nextState = self._state()
        return nextState, reward

    def step(self, action):
        '''Standard Gym step. One call == one decode step of the serving system.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)

        has_work = self._advance_to_decodable()
        done = not has_work          # episode ends when fully drained past duration
        nextState = self._state()

        info = {
            'lam': self.lam,
            'clock': self.clock,
            'mean_latency': (float(np.mean(self.done_latencies))
                             if self.done_latencies else None),
        }
        return nextState, reward, done, info

    def reset(self):
        '''Standard Gym reset: new episode with a freshly randomized arrival rate.'''
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
