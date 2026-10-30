"""
DeepTOP with a multi-level action k in {0, 1, ..., K} (average-reward SMDP).

Critic  Q(vector, scalar, k)  with k one-hot (K+1 dims), trained with the SMDP differential target
    target = reward - rho * tau + (1 - done) * max_k' Q_target(s', k')
Actor   K ordered thresholds (model_multik.ThresholdActor). DeepTOP's threshold gradient, applied
        at every boundary: threshold j separates action j-1 (above it) from action j (below it), so
            d loss / d L_j = -[ Q(v, theta_j, k=j) - Q(v, theta_j, k=j-1) ]
        raises theta_j (use the longer speculation over a wider load range) while k=j beats k=j-1
        at the boundary, and lowers it otherwise. With K = 1 this is exactly the original DeepTOP.
"""
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from model import Critic
from memory import SequentialMemory
from model_multik import ThresholdActor
from util import hard_update, soft_update

criterion = nn.MSELoss()


class DeepTOP_MK(object):
    def __init__(self, state_dim, K, hidden, args,k_min=0):
        self.k_min = k_min
        self.state_dim, self.K = state_dim, K
        self.n_thr = K - k_min  #num of thresholds (K=4, k_min=1 -> 3 thresholds: theta_1, theta_2, theta_3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ThresholdActor(state_dim, self.n_thr, hidden, k_min=k_min,
                                    init_n=getattr(args, "init_n", (10.0, 300.0))).to(self.device)
        self.actor_avg = deepcopy(self.actor) 
        self.avg_tau = getattr(args, "avg_tau", 1e-4)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)
        self.critic = Critic(state_dim + 1, K + 1, hidden).to(self.device)
        self.critic_target = Critic(state_dim + 1, K + 1, hidden).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)
        hard_update(self.critic_target, self.critic)

        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.s_t, self.a_t = None, None
        self.batch_size, self.tau, self.discount = args.bsize, args.tau, args.discount
        self.rho, self.rho_lr = 0.0, getattr(args, "rho_lr", 0.02)
        # rho estimator: "sgd" = original DeepTOP rule rho += lr*(r - rho*tau);
        # "ratio" = EMA(reward) / EMA(tau), a smoother estimate of the reward rate E[r]/E[tau]
        self.rho_mode = getattr(args, "rho_mode", "sgd")
        self._r_ema, self._t_ema = 0.0, 1e-6
        self.reward_mode = getattr(args, "reward_mode", "average")
        # tdiscount: continuous-time discounting, factor exp(-tau / horizon) per step (horizon in seconds)
        self.horizon = getattr(args, "horizon", 5.0)
        # episode rho: average reward rate of the CURRENT episode (fixed arrival rate), stored per
        # transition, instead of one global rho shared by episodes with very different loads
        self._ep_r, self._ep_t = 0.0, 0.0
        self.smdp = getattr(args, "smdp", True)
        cap = getattr(args, "logit_cap", 2.5)
        # asymmetric caps on the threshold logits. logit(s) = ln(n/256), so a cap L keeps the threshold
        # at n = 256*e^L: keep thresholds inside the region of n the agent actually visits, where the
        # critic is trained (a threshold parked at n ~ 3000 is never corrected).
        self.cap_hi = getattr(args, "logit_cap_hi", None) or cap
        self.cap_lo = getattr(args, "logit_cap_lo", None) or cap
        self.actor_start = getattr(args, "actor_start", 0)
        self.n_updates = 0
        self.is_training = True

    # ---------------------------------------------------------------- helpers
    def _onehot(self, k):
        k = torch.as_tensor(k, device=self.device).long().view(-1)
        return F.one_hot(k, self.K + 1).float()

    # ---------------------------------------------------------------- training
    def update_policy(self):
        s0, a, r, s1, term = self.memory.sample_and_split(self.batch_size)
        dev = self.device
        vec = torch.as_tensor(s0[:, 1:], dtype=torch.float32, device=dev)
        scal = torch.as_tensor(s0[:, :1], dtype=torch.float32, device=dev)
        nvec = torch.as_tensor(s1[:, 1:], dtype=torch.float32, device=dev)
        nscal = torch.as_tensor(s1[:, :1], dtype=torch.float32, device=dev)
        act = torch.as_tensor(a[:, 0], device=dev).long()
        rew = torch.as_tensor(r[:, :1], dtype=torch.float32, device=dev)
        tau = torch.as_tensor(r[:, 1:2], dtype=torch.float32, device=dev)
        rho_t = torch.as_tensor(r[:, 2:3], dtype=torch.float32, device=dev)
        mask = torch.as_tensor(term, dtype=torch.float32, device=dev)   # 0 at episode end
        B = vec.shape[0]

        with torch.no_grad():
            q_next = torch.cat([self.critic_target([nvec, nscal, self._onehot(torch.full((B,), j))])
                                for j in range(self.k_min, self.K + 1)], dim=1)
            next_v = q_next.max(dim=1, keepdim=True).values
            if self.reward_mode == "discounted":
                target = rew + self.discount * mask * next_v
            elif self.reward_mode == "tdiscount":
                target = rew + mask * torch.exp(-tau / self.horizon) * next_v
            elif self.rho_mode == "episode":
                target = rew - rho_t * tau + mask * next_v
            elif self.smdp:
                target = rew - self.rho * tau + mask * next_v
            else:
                target = rew - self.rho + mask * next_v

        self.critic_optim.zero_grad()
        loss_c = criterion(self.critic([vec, scal, self._onehot(act)]), target)
        loss_c.backward()
        self.critic_optim.step()

        # actor: DeepTOP gradient at every threshold boundary
        logits = self.actor.logit(vec)                       # (B, K), decreasing
        thr = torch.sigmoid(logits).detach()
        qd = []
        with torch.no_grad():
            for j in range(1, self.n_thr + 1):  # j = 1, ..., n_thr (thresholds theta_1, ..., theta_n_thr)
                s_j = thr[:, j - 1:j]
                q_hi = self.critic([vec, s_j, self._onehot(torch.full((B,), self.k_min + j))])  # k = j (above threshold)
                q_lo = self.critic([vec, s_j, self._onehot(torch.full((B,), self.k_min + j - 1))])
                qd.append(q_hi - q_lo)
        qd = torch.cat(qd, dim=1)                            # (B, K)
        outward = ((logits > self.cap_hi) & (qd > 0)) | ((logits < -self.cap_lo) & (qd < 0))
        loss_a = -(qd * logits * (~outward).float()).sum(dim=1).mean()
        # the thresholds share one network, so masking one threshold does not stop the others from
        # dragging it out of range; pull any logit that left [-cap_lo, cap_hi] back in
        loss_a = loss_a + (F.relu(logits - self.cap_hi) ** 2 + F.relu(-self.cap_lo - logits) ** 2).sum(dim=1).mean()
        self.n_updates += 1
        if self.n_updates > self.actor_start:      # let the critic learn before moving thresholds
            self.actor_optim.zero_grad()
            loss_a.backward()
            self.actor_optim.step()
            soft_update(self.actor_avg, self.actor, self.avg_tau) 

        soft_update(self.critic_target, self.critic, self.tau)

    # ---------------------------------------------------------------- acting
    def observe(self, r_t, s_t1, done):
        if self.is_training:
            reward, tau = r_t
            if self.reward_mode == "average":
                if self.rho_mode == "ratio":
                    b = self.rho_lr
                    self._r_ema = (1 - b) * self._r_ema + b * reward
                    self._t_ema = (1 - b) * self._t_ema + b * (tau if self.smdp else 1.0)
                    self.rho = self._r_ema / self._t_ema
                elif self.smdp:
                    self.rho += self.rho_lr * (reward - self.rho * tau)
                else:
                    self.rho += self.rho_lr * (reward - self.rho)
            self._ep_r += reward
            self._ep_t += tau
            rho_ep = self._ep_r / self._ep_t if self._ep_t > 1.0 else self.rho
            self.memory.append(self.s_t, self.a_t, (reward, tau, rho_ep), done)
            self.s_t = s_t1

    def random_action(self):
        self.a_t = [random.randint(self.k_min, self.K)]
        return self.a_t

    def select_action(self, s_t):
        k = self.actor.act(s_t[1:], s_t[0])          # k_min is a buffer of the actor
        self.a_t = [k]
        return self.a_t

    def reset(self, obs):
        self.s_t = obs
        self._ep_r, self._ep_t = 0.0, 0.0
