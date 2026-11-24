"""v2 observation: v1 features + game context (running totals, rounds left, standing)."""
import numpy as np, torch
from .features import encode, mask_of, Net, NetAgent, OBS_DIM

OBS2_DIM = OBS_DIM + 7


def encode2(r, me):
    base = encode(r, me)
    tot = r.totals
    ex = np.zeros(7, np.float32)
    for k in range(1, 4):
        ex[k - 1] = max(-3.0, min(3.0, (tot[(me + k) % 4] - tot[me]) / 300.0))
    ex[3] = max(-3.0, min(3.0, tot[me] / 300.0))
    ex[4] = (15 - r.n) / 15.0
    ex[5] = sum(1 for q in range(4) if tot[q] > tot[me]) / 3.0
    ex[6] = max(0.0, min(3.0, (max(tot) - tot[me]) / 300.0))
    return np.concatenate([base, ex])


def load_net2(path, dev="cpu"):
    n = Net(in_dim=OBS2_DIM)
    n.load_state_dict(torch.load(path, map_location=dev))
    return n.to(dev).eval()


def Agent2(net, greedy=True, seed=0):
    return NetAgent(net, greedy, seed, enc=encode2)

