"""Observation encoding + policy/value net shared by training and play."""
import numpy as np
import torch, torch.nn as nn
from .engine import *
from .pimc import infer_voids

N_ACT = 60 + 16          # 60 cards, then bids 0..15
OBS_DIM = 60 * 7 + 5 + 4 * 4 + 4 + 12 + 6


def encode(r, me):
    o = np.zeros(OBS_DIM, np.float32)
    o[0:60][list(r.hands[me])] = 1
    for t in r.played:
        for _, c in t:
            o[60 + c] = 1
    for q, c in r.trick:
        o[120 + ((q - me) % 4) * 60 + c] = 1
    if r.upcard is not None:
        o[360 + r.upcard] = 1
    i = 420
    o[i + (r.trump if r.trump is not None else 4)] = 1; i += 5
    bids = r.bid_by_player()
    for k in range(4):
        q = (me + k) % 4
        if bids[q] is not None:
            need = bids[q] - r.won[q]
            o[i + k * 4:i + k * 4 + 4] = (1, bids[q] / 15, r.won[q] / 15, need / 15)
    i += 16
    o[i + (r.dealer - me) % 4] = 1; i += 4
    voids = infer_voids(r)
    for k in range(1, 4):
        for s in voids[(me + k) % 4]:
            o[i + (k - 1) * 4 + s] = 1
    i += 12
    o[i:i + 6] = (r.n / 15, len(r.played) / 15, (r.n - len(r.played)) / 15,
                  sum(b for b in r.bids) / 15, 1.0 if r.phase == "bid" else 0.0, len(r.trick) / 3)
    return o


def mask_of(r):
    m = np.zeros(N_ACT, bool)
    if r.phase == "bid":
        m[[60 + b for b in r.legal_bids()]] = True
    else:
        m[r.legal_cards()] = True
    return m


class Net(nn.Module):
    def __init__(self, h=512, in_dim=OBS_DIM):
        super().__init__()
        self.inp = nn.Linear(in_dim, h)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)) for _ in range(3)])
        self.pi = nn.Linear(h, N_ACT)
        self.v = nn.Linear(h, 1)

    def forward(self, x, mask):
        h = torch.relu(self.inp(x))
        for b in self.blocks:
            h = torch.relu(h + b(h))
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


class NetAgent:
    """Plays with a Net on CPU. greedy=True -> argmax."""
    def __init__(self, net, greedy=True, seed=0, enc=None):
        self.net, self.greedy = net, greedy
        self.enc = enc or encode
        self.gen = torch.Generator().manual_seed(seed)

    def _act(self, r, p):
        x = torch.from_numpy(self.enc(r, p))[None]
        m = torch.from_numpy(mask_of(r))[None]
        with torch.no_grad():
            lg, _ = self.net(x, m)
        if self.greedy:
            return int(lg.argmax())
        u = torch.rand(lg.shape, generator=self.gen).clamp_(1e-10, 1.0)
        return int((lg - torch.log(-torch.log(u))).argmax())

    def bid(self, v): return self._act(v.r, v.player) - 60
    def play(self, v): return self._act(v.r, v.player)



