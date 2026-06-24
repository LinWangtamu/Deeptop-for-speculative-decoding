import random

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()


class DeepTOP_MDP(object):
    # state_dim: the dimension of the vector state. IMPORTANT: not including the scalar state
    # action_dim: the dimension of actions. This should be 1
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, state_dim, action_dim, hidden, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Create Actor and Critic Networks
        self.actor = Actor(self.state_dim, 1, hidden)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)
        self.critic = Critic(self.state_dim + 1, 1, hidden)  # Input is both the scalar state and the vector state
        self.critic_target = Critic(self.state_dim + 1, 1, hidden)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount   # NOTE: unused in average-reward SMDP mode (kept for compat)
        self.depsilon = 1.0 / args.epsilon

        # --- Average-reward SMDP setup ----------------------------------------
        # This is a SEMI-MDP: the speculate/no-speculate action changes how much
        # wall-clock time (tau = step_t) one step represents, so per-step
        # discounting OR per-step averaging both distort the true objective
        # (minimize mean latency == minimize time-averaged #-in-system). We
        # therefore optimize the average REWARD RATE and use the SMDP
        # differential TD target
        #     target = reward - rho * tau + (1-done) * Q(s', a')
        # where rho = (running sum of reward) / (running sum of tau) is the
        # time-averaged reward rate (NOT a per-step mean). rho only recenters Q
        # (the actor uses Q-differences, invariant to it) but keeps Q bounded.
        self.rho = 0.0
        self.rho_lr = getattr(args, 'rho_lr', 1e-4)
        # ----------------------------------------------------------------------

        # --- ABLATION: reward objective -------------------------------------
        # 'average'    -> SMDP differential target (default, paper setting):
        #                 target = reward - rho*tau + (1-done)*Q(s',a')
        # 'discounted' -> standard discounted return (per-decision gamma, tau
        #                 ignored, rho disabled):
        #                 target = reward + gamma*(1-done)*Q(s',a')
        self.reward_mode = getattr(args, 'reward_mode', 'average')
        # --- ABLATION: SMDP vs MDP (only meaningful when reward_mode=='average')
        # True  -> SMDP: time-averaged rate, target subtracts rho*tau
        # False -> MDP : per-step average, target subtracts rho (tau ignored)
        self.smdp = getattr(args, 'smdp', True)
        # ----------------------------------------------------------------------

        self.epsilon = 1.0
        self.is_training = True

        
        if self.device == torch.device('cuda'): 
            self.cuda()

    def update_policy(self):
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # reward_batch packs (reward, tau) per transition -> shape (B, 2).
        # Column 0 is the reward, column 1 is the SMDP step duration tau.
        reward_col = reward_batch[:, 0:1]
        tau_col = reward_batch[:, 1:2]

        vector_batch = []  # The batch of vector state
        scalar_batch = []  # The batch of scalar state
        next_vector_batch = []
        next_scalar_batch = []
        
        for i in range(self.batch_size):
            vector_batch.append(list(state_batch[i]))
            next_vector_batch.append(list(next_state_batch[i]))
            scalar_batch.append([vector_batch[i].pop(0)])
            next_scalar_batch.append([next_vector_batch[i].pop(0)])

        # Convert all batches to arrays
        vector_batch = torch.FloatTensor(np.array(vector_batch)).to(self.device)
        scalar_batch = torch.FloatTensor(np.array(scalar_batch)).to(self.device)
        next_vector_batch = torch.FloatTensor(np.array(next_vector_batch)).to(self.device)
        next_scalar_batch = torch.FloatTensor(np.array(next_scalar_batch)).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch_t = torch.FloatTensor(reward_col).to(self.device)
        tau_batch = torch.FloatTensor(tau_col).to(self.device)

        with torch.no_grad():
            critic_plus = self.critic_target([next_vector_batch,
                                              next_scalar_batch,
                                              to_tensor(np.ones((self.batch_size, 1), dtype=int)).to(self.device)]).cpu()
            critic_minus = self.critic_target([next_vector_batch,
                                               next_scalar_batch,
                                               to_tensor(np.zeros((self.batch_size, 1), dtype=int)).to(self.device)]).cpu()
            next_action_batch = torch.FloatTensor(torch.clamp(torch.sign(critic_plus - critic_minus), min=0.0)).to(self.device)
            
            # Prepare for the target Q batch
            next_q_values = self.critic_target([next_vector_batch,
                                                next_scalar_batch,
                                                next_action_batch])

            # SMDP differential TD target: subtract rho*tau (NOT a constant, and
            # NO gamma), bootstrap next differential Q (masked at episode
            # boundaries where terminal_batch == 0).
            if self.reward_mode == 'discounted':
                # ABLATION: standard discounted return. Plain per-decision gamma,
                # tau ignored, rho not subtracted. This is the discounted-RL
                # baseline that the average-reward formulation is compared against.
                target_q_batch = (reward_batch_t
                                  + self.discount
                                  * to_tensor(terminal_batch.astype(np.float32))
                                  * next_q_values)
            else:
                # Average-reward. ABLATION SMDP vs MDP:
                if self.smdp:
                    # SMDP: time-averaged, subtract rho*tau (default).
                    target_q_batch = (reward_batch_t - self.rho * tau_batch
                                      + to_tensor(terminal_batch.astype(np.float32)) * next_q_values)
                else:
                    # MDP: per-step average, subtract rho (tau ignored).
                    target_q_batch = (reward_batch_t - self.rho
                                      + to_tensor(terminal_batch.astype(np.float32)) * next_q_values)

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic([vector_batch, scalar_batch, 
                               action_batch])
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        q_diff_batch = self.critic([vector_batch, self.actor(vector_batch),
                                    to_tensor(np.ones((self.batch_size, 1), dtype=int))]) - \
                       self.critic([vector_batch, self.actor(vector_batch),
                                    to_tensor(np.zeros((self.batch_size, 1), dtype=int))])
        
        q_diff_batch = q_diff_batch.detach().cpu().numpy()

        policy_loss = -to_tensor(q_diff_batch).to(self.device) * self.actor(vector_batch)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        torch.cuda.set_device(0) # specify which gpu to train on 
        self.actor.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        # r_t is a (reward, tau) pair. rho is the time-averaged reward RATE,
        # updated by the standard SMDP average-reward rule
        #     rho <- rho + rho_lr * (reward - rho * tau)
        # which converges to E[reward]/E[tau]. Kept slow (small rho_lr) so rho is
        # a stable long-run estimate that is not whipped around by the large
        # per-episode swings in reward and tau across light vs overloaded loads.
        if self.is_training:
            reward, tau = r_t
            if self.reward_mode == 'average':
                if self.smdp:
                    self.rho += self.rho_lr * (reward - self.rho * tau)  # rate E[r]/E[tau]
                else:
                    self.rho += self.rho_lr * (reward - self.rho)        # per-step mean E[r]
            # (discounted mode: rho stays 0 and is unused)
            self.memory.append(self.s_t, self.a_t, (reward, tau), done)
            self.s_t = s_t1

    def random_action(self):
        action = [random.randint(0, 1)]
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        vector = s_t.copy()
        scalar = vector.pop(0)
        threshold = self.actor.forward(torch.FloatTensor(vector).to(self.device)).cpu().item()

        if threshold > scalar:
            action = [1]
        else:
            action = [0]
        self.a_t = action
        return action
        

    def reset(self, obs):
        self.s_t = obs
