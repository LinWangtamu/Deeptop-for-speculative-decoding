
import os
import torch
import random
import argparse
from copy import deepcopy
import numpy as np
import time
import math
import torch.nn as nn
import torch.nn.functional as F
from MDP_Env import SpecDecodingEnv
from DeepTOP import DeepTOP_MDP


def initializeEnv():
    global env
    # Speculative-decoding serving environment.
    # Episode = 120 s of Poisson arrivals; lambda randomized in [2, 18] each episode.
    env = SpecDecodingEnv(seed=args.seed if args.seed > 0 else 42,
                          duration=120.0, warmup=20.0,
                          lam_low=0.5, lam_high=20.0,
                          true_alpha=0.7)


def resetEnvs():
    global state, env
    state = env.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepTOP speculative-decoding on/off control')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=458472, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--total_steps', default=200000, type=int, help='total training steps')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env = None
    state = None
    # state = [batch_size/MAX(scalar), alpha_est, avg_context_norm, backlog_norm]
    # state_dim counts ONLY the vector part (excludes the scalar).
    state_dim = 3
    action_dim = 1
    initializeEnv()

    # initialize agent
    hidden = [128, 128]
    agent = DeepTOP_MDP(state_dim, action_dim, hidden, args)

    resetEnvs()
    agent.reset(state)

    cumulative_reward = 0

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    num_step = 0
    episode_count = 0
    recent_latencies = []

    for t in range(args.total_steps + 1):
        agent.is_training = True
        num_step = num_step + 1

        # agent picks action
        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
