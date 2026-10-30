"""
Multi-level threshold actor for DeepTOP with k in {k_min, ..., k_min + K}.

The actor maps the vector state to K ordered thresholds  theta_1 >= theta_2 >= ... >= theta_K
(in the same [0, 1) space as the scalar state s = n_in_system / (n_in_system + scale)).
Policy:  k = k_min + #{ j : theta_j > s }.
More requests in the system (larger s) -> fewer thresholds above s -> smaller k (monotone).
k_min is stored as a buffer, so it is saved in the checkpoint (old checkpoints without it: k_min = 0).

Ordering is built in: logits L_K = a, L_{K-1} = a + softplus(g_1), ..., L_1 = a + sum softplus(g).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    v = 1.0 / np.sqrt(fanin or size[0])
    return torch.Tensor(size).uniform_(-v, v)


class ThresholdActor(nn.Module):
    def __init__(self, nb_inputs, K, hidden, init_w=5e-1, k_min=0, init_n=None):
        super().__init__()
        self.K = K                                            # number of thresholds
        self.register_buffer("k_min", torch.tensor(int(k_min)))
        dims = [nb_inputs, *hidden, K]
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        for i, layer in enumerate(self.fc):
            if i == len(self.fc) - 1:
                layer.weight.data.uniform_(-init_w, init_w)
            else:
                layer.weight.data = fanin_init(layer.weight.data.size())
        if init_n is not None and K > 0:
            # start with the K thresholds log-spaced over init_n = (n_low, n_high) requests in the
            # system (inside the visited range) instead of wherever a random init puts them
            last = self.fc[-1]
            last.weight.data.uniform_(-0.05, 0.05)
            ns = np.geomspace(init_n[0], init_n[1], K)        # n_K (lowest) ... n_1 (highest)
            lg = np.log(ns / 256.0)                            # logit(s) = ln(n / 256)
            bias = [lg[0]] + [float(np.log(np.expm1(max(d, 1e-3)))) for d in np.diff(lg)]
            last.bias.data = torch.tensor(bias, dtype=torch.float32)

    def logit(self, x):
        """Ordered logits, shape (..., K): column j-1 is L_j, decreasing in j."""
        out = x
        for i, layer in enumerate(self.fc):
            out = layer(out)
            if i < len(self.fc) - 1:
                out = F.relu(out)
        base = out[..., :1]                                   # L_K (lowest threshold)
        gaps = F.softplus(out[..., 1:])                      # >= 0
        low_to_high = torch.cat([base, base + torch.cumsum(gaps, dim=-1)], dim=-1)
        return torch.flip(low_to_high, dims=[-1])            # [L_1, ..., L_K]

    def forward(self, x):
        return torch.sigmoid(self.logit(x))                  # [theta_1, ..., theta_K]

    def act(self, vector, scalar, tol=1e-2):
        """k for one state (vector: list/array of the vector state, scalar: float)."""
        with torch.no_grad():
            thr = self.forward(torch.as_tensor(np.asarray(vector), dtype=torch.float32))
        return int(self.k_min.item()) + int((thr > scalar - tol).sum().item())


def feat_fn(n_inputs):
    """Map an env state [scalar, batch, alpha, ctx] to the actor's vector input.
    3 inputs: [batch, alpha, ctx]. 2 inputs: [alpha, ctx] (load-exogenous features only)."""
    if n_inputs == 3:
        return lambda s: list(s[1:4])
    if n_inputs == 2:
        return lambda s: [s[2], s[3]]
    raise ValueError(n_inputs)


def load_actor(path, K=None, hidden=(128, 128)):
    """Load a ThresholdActor checkpoint; input size, number of thresholds and k_min are read from it.
    Returns (actor, feature_fn). K is ignored (kept for old call sites)."""
    sd = dict(torch.load(path, map_location="cpu"))
    n_in = sd["fc.0.weight"].shape[1]
    n_thr = sd[f"fc.{len(hidden)}.weight"].shape[0]
    if "k_min" not in sd:                     # old checkpoints (v1/v4): k in 0..n_thr
        sd["k_min"] = torch.tensor(0)
    actor = ThresholdActor(n_in, n_thr, list(hidden), k_min=int(sd["k_min"]))
    actor.load_state_dict(sd)
    actor.eval()
    return actor, feat_fn(n_in)
