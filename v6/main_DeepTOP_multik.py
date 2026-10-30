"""
Train multi-level DeepTOP (k in {0..K_MAX}) in the v2 simulator.

  python main_DeepTOP_multik.py --total_steps 400000
Checkpoints: ckpt_mk/actor_step<N>.pkl every --ckpt_every steps; final: deeptop_mk_actor.pkl.
Then pick the best checkpoint on validation seeds:  python eval_multik.py --select
"""
import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch

import env_config as C
from DeepTOP_multik import DeepTOP_MK # how to update policy to learn
from model_multik import feat_fn # network and Feature Processing of deeptop

p = argparse.ArgumentParser()
p.add_argument("--rate", default=1e-3, type=float, help="critic learning rate")
p.add_argument("--prate", default=1e-4, type=float, help="actor learning rate")
p.add_argument("--warmup", default=1000, type=int, help="Initial random-action steps to collect replay data; no actor or critic updates")
p.add_argument("--discount", default=0.99, type=float)
p.add_argument("--bsize", default=64, type=int)#how many experience will sample from buffer for each training.
p.add_argument("--rmsize", default=100000, type=int) #replay buffer memory size, after full, new experience will repalce old ones. one experience:(state, action, reward, next_state, done)
p.add_argument("--window_length", default=1, type=int)#how many observation consist of a state input
p.add_argument("--tau", default=0.001, type=float)#soft-update rate for target networks:\theta_target = (1-\tau)\theta_target + \tau\theta
p.add_argument("--rho_lr", default=0.02, type=float)#Update rate for the average reward-rate estimate rho
"""
        "Average reward-rate estimator: "
        "sgd updates rho += rho_lr * (reward - rho * step_duration); "
        "ratio uses EMA(reward) / EMA(step_duration), with EMA weight rho_lr; "
        "see implementation for episode mode"
"""
p.add_argument("--rho_mode", default="sgd", choices=["sgd", "ratio", "episode"],
               help="sgd: rho += lr*(r - rho*tau) (original); ratio: rho = EMA(r)/EMA(tau) with weight rho_lr")
p.add_argument("--explore", default=0.05, type=float, help="probability of a uniformly random k")
p.add_argument("--total_steps", default=400000, type=int)#total step; update policy step = total step - warm up
p.add_argument("--ckpt_every", default=25000, type=int)# every 25000 step save one checkpoint.
p.add_argument("--seed", default=458472, type=int)
p.add_argument("--lam_low", default=0.5, type=float)# minimum arrival rate
p.add_argument("--lam_high", default=40.0, type=float)#maximum arrival rate
p.add_argument("--alpha_low", default=0.70, type=float)
p.add_argument("--alpha_high", default=0.89, type=float)
p.add_argument("--reward_norm", default=1.33, type=float)#Reward scaling parameter passed to the environment
p.add_argument("--logit_cap", default=2.5, type=float) # the logit limit the range that the thresholds can switch from. -2.5<=z<2.5:256e^(-2.5)<=n_threshold<=256e^2.5: 21~3119 requests
p.add_argument("--logit_cap_hi", default=None, type=float, help="upper cap; threshold n <= 256*e^cap")
p.add_argument("--logit_cap_lo", default=None, type=float, help="lower cap; threshold n >= 256*e^-cap")
p.add_argument("--actor_start", default=0, type=int, help="critic-only updates before the actor moves")#Number of critic-only updates before actor updates begin
p.add_argument("--ckpt_dir", default="ckpt_mk",help="Directory for saving actor checkpoints")
p.add_argument("--init_actor", default=None, help="warm-start the actor from this checkpoint (same feature set)")#e.g.:--init_actor ckpt_mk/actor_step25000.pkl start training from this pkl
p.add_argument("--reward_mode", default="average", choices=["average", "discounted", "tdiscount"])
p.add_argument("--horizon", default=5.0, type=float, help="tdiscount: time constant in seconds")
p.add_argument("--feats", default="all", choices=["all", "ac"],
               help="actor/critic vector: all = [batch, alpha, ctx] (v1); ac = [alpha, ctx] (load-exogenous only)")
p.add_argument("--smdp", default="true", choices=["true", "false"])
# v4: load-based arrival rates, step-time jitter, shorter episodes
p.add_argument("--rho_low", default=None, type=float, help="load rho ~ U(rho_low, rho_high); lambda = rho * capacity(alpha)")
p.add_argument("--rho_high", default=0.98, type=float)
p.add_argument("--p_over", default=0.1, type=float, help="probability of an overloaded episode, rho ~ U(1.0, 1.15)")
p.add_argument("--ovh_jitter", default=0.5, type=float, help="per-episode scale U(1-j, 1+j) on the step overhead")
p.add_argument("--duration", default=None, type=float, help="episode length in s (default env_config.DURATION)")
p.add_argument("--warmup_s", default=None, type=float, help="steady-state window start in s (logging only)")
p.add_argument("--out", default="deeptop_mk_actor.pkl")
p.add_argument("--avg_tau", default=1e-4, type=float, help="Polyak averaging rate of the saved actor") #\theta_(avg) = (1-avg_tau)\theta_(avg) +avg_tau\theta_actor 
args = p.parse_args()
args.smdp = args.smdp == "true"

np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

env = C.make_env(seed=args.seed, lam_low=args.lam_low, lam_high=args.lam_high,
                 alpha=(args.alpha_low + args.alpha_high) / 2, alpha_low=args.alpha_low,
                 alpha_high=args.alpha_high, alpha_prior=(args.alpha_low + args.alpha_high) / 2,
                 reward_norm=args.reward_norm, duration=args.duration, warmup=args.warmup_s,
                 rho_low=args.rho_low, rho_high=args.rho_high, p_over=args.p_over,
                 overhead_jitter=args.ovh_jitter)
NV = 3 if args.feats == "all" else 2 #number of vectors
_feat = feat_fn(NV)# # Get the feature extraction function for the specified number of features
def T(s):  # env state -> agent state [scalar, *vector]
    return [s[0]] + _feat(s)
# Initialize the DeepTOP agent with the feature dimension, action bounds, network architecture, and training arguments.
agent = DeepTOP_MK(state_dim=NV, K=C.K_MAX, hidden=[128, 128], args=args, k_min=C.K_MIN)
if args.init_actor:
    agent.actor.load_state_dict(torch.load(args.init_actor, map_location=agent.device))
    print(f"actor warm-started from {args.init_actor}", flush=True)
_load = (f"rho~U({args.rho_low},{args.rho_high}) (+{args.p_over:.0%} overloaded)" if args.rho_low is not None
         else f"lam~U({args.lam_low},{args.lam_high})")
print(f"multi-level DeepTOP: k in {C.K_MIN}..{C.K_MAX}, budget={C.BUDGET}, cost_model={C.COST_MODEL}, {_load}, "
      f"alpha~U({args.alpha_low},{args.alpha_high}), step overhead jitter +-{args.ovh_jitter:.0%}, "
      f"reward_mode={args.reward_mode}, smdp={args.smdp}", flush=True)

state, _ = env.reset()# Reset the environment and get the initial state
state = T(state)# Convert the environment state to the agent's input format:agent state [scalar, *vector]
agent.reset(state)# Reset the agent's interaction state for the new episode
episodes, recent, cum_r, t0 = 0, [], 0.0, time.time()# Initialize training statistics and start time
os.makedirs(args.ckpt_dir, exist_ok=True)# Create the checkpoint directory if needed


def probe(batch, alpha, ctx):
    return [batch, alpha, ctx] if NV == 3 else [alpha, ctx]

# Given the current feature vector vec = [0.15, 0.80, 1.0] (batch, alpha, ctx), return the request-count thresholds list at which k decreases by one,[θ₁, θ₂, θ₃, θ₄] when k = 0..3
def cutoffs(vec):
    """n_in_system at which k drops 1->0, 2->1, 3->2, 4->3 (theta_1..theta_K), for a probe state."""
    with torch.no_grad():
        thr = agent.actor_avg(torch.tensor(vec, dtype=torch.float32, device=agent.device)).cpu().numpy()
    return [round(float(t * C.N_SCALE / (1 - t))) if t < 1 else float("inf") for t in thr]# # For every normalized threshold t, convert it to the corresponding request-count threshold.


for step in range(1, args.total_steps + 1):
    if step <= args.warmup or random.random() < args.explore:
        action = agent.random_action()
    else:
        action = agent.select_action(state)
    #Execute the selected k in the SD engine
    nxt, reward, term, trunc, info = C.env_step(env, action[0], sd_engine=True)
  # # Convert the next state to the agent's state
    nxt = T(nxt)
    done = term or trunc# Check whether the episode has ended
    agent.observe((reward, info["tau"]), deepcopy(nxt), done)# Record the reward, step duration, next state, and episode-end flag
    if step > args.warmup:
        cum_r += reward # Accumulate reward for logging
        agent.update_policy()# Update the networks after warmup
    state = deepcopy(nxt)# Use the next state as the current state
    if done:
        episodes += 1
          # Record the episode's steady-state mean latency if available
        if info["mean_latency_steady"] is not None:
            recent.append(info["mean_latency_steady"])
        state, _ = env.reset() # Get the initial state of a new episode
        state = T(state) # Convert it to the agent's input format
        agent.reset(state) # Reset the agent's interaction state
    if step % 5000 == 0:
        print(f"step {step}: avg_reward {cum_r / 5000:.4f}  rho {agent.rho:.3f}  episodes {episodes}  "
              f"recent latency {np.mean(recent[-10:]) if recent else float('nan'):.2f}s  "
              f"cutoffs n[{C.K_MIN+1}->{C.K_MIN},...,{C.K_MAX}->{C.K_MAX-1}] a=0.70: {cutoffs(probe(0.15, 0.70, 1.0))} a=0.84: {cutoffs(probe(0.15, 0.84, 1.0))}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        cum_r = 0.0
    if step % args.ckpt_every == 0:
        torch.save(agent.actor_avg.state_dict(), f"{args.ckpt_dir}/actor_step{step}.pkl")

torch.save(agent.actor_avg.state_dict(), args.out)
torch.save(agent.critic.state_dict(), args.out.replace("actor", "critic"))
print(f"\nLearned cutoffs n[{C.K_MIN+1}->{C.K_MIN}, ..., {C.K_MAX}->{C.K_MAX-1}]: k drops from k_min+j to k_min+j-1 once n_in_system exceeds the j-th number")
#Given a feature vector, this function returns the corresponding request-count thresholds for switching between adjacent values of k.
for bf in (0.05, 0.15, 0.3):#nomalize N_batch
    for al in (args.alpha_low, 0.70, 0.838, args.alpha_high):
        print(f"  batch={bf * C.MAX_NUM_SEQS:.0f}  alpha={al:.3f}  ctx=1.0 : {cutoffs(probe(bf, al, 1.0))}")
print(f"saved {args.out}")
