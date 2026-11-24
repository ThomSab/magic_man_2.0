"""Net-guided PIMC with a v2 net, GPU-batched rollouts.
Each candidate bid/card is evaluated on n_samples sampled opponent hands; every rollout is played to the end of
the round by the net in all four seats (sampled). Score = own round score (optionally minus mean of others)."""
import random, numpy as np, torch
from .engine import *
from .pimc import infer_voids, sample_hands
from .features import mask_of
from .features2 import encode2, load_net2


class NetSearch2:
    def __init__(self, net, n_samples=16, seed=0, device="cuda", zero_sum=False, temp=1.0):
        self.dev = torch.device(device)
        self.net = net.to(self.dev).eval()
        self.n, self.rng, self.zero_sum, self.temp = n_samples, random.Random(seed), zero_sum, temp
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    def _rollout(self, rounds):
        live = [r for r in rounds if r.phase != "done"]
        while live:
            x = torch.from_numpy(np.stack([encode2(r, r.to_act) for r in live])).to(self.dev)
            m = torch.from_numpy(np.stack([mask_of(r) for r in live])).to(self.dev)
            with torch.no_grad():
                lg, _ = self.net(x, m)
            u = torch.rand(lg.shape, generator=self.gen, device=self.dev).clamp_(1e-10, 1.0)
            a = (lg / self.temp - torch.log(-torch.log(u))).argmax(-1).tolist()
            for r, ai in zip(live, a):
                if r.phase == "bid": r.bid(ai - 60)
                else: r.play(ai)
            live = [r for r in live if r.phase != "done"]

    def _decide(self, v, options, apply, use_voids):
        r, me = v.r, v.player
        voids = infer_voids(r) if use_voids else [set()] * N_PLAYERS
        sims, keys = [], []
        for _ in range(self.n):
            hands = sample_hands(r, me, self.rng, voids)
            base = r.clone()
            for p, h in hands.items(): base.hands[p] = h
            for o in options:
                s = base.clone(); apply(s, o); sims.append(s); keys.append(o)
        self._rollout(sims)
        tot = {o: 0.0 for o in options}
        for s, o in zip(sims, keys):
            sc = s.scores()
            tot[o] += sc[me] - (np.mean([sc[q] for q in range(4) if q != me]) if self.zero_sum else 0.0)
        return max(options, key=lambda o: tot[o])

    def bid(self, v):
        if len(v.legal_bids) == 1: return v.legal_bids[0]
        return self._decide(v, v.legal_bids, lambda s, o: s.bid(o), False)

    def play(self, v):
        if len(v.legal_cards) == 1: return v.legal_cards[0]
        return self._decide(v, v.legal_cards, lambda s, o: s.play(o), True)
