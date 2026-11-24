"""Net-guided PIMC (v2 net for the searcher) with an optional opponent model mixture.
opp_mix: 'self' -> opponents rolled out with the same net; 'mix' -> each simulated opponent is randomly the
v2 net or the run4 (v1) net, fixed per simulation."""
import random, numpy as np, torch
from .engine import *
from .pimc import infer_voids, sample_hands
from .features import mask_of, encode, Net
from .features2 import encode2


class NetSearch3:
    def __init__(self, net, n_samples=16, seed=0, device="cuda", opp_mix="self",
                 run4_path="checkpoints/run4_final.pt", temp=1.0):
        self.dev = torch.device(device)
        self.models = [(net.to(self.dev).eval(), encode2)]
        if opp_mix == "mix":
            r4 = Net(); r4.load_state_dict(torch.load(run4_path, map_location="cpu"))
            self.models.append((r4.to(self.dev).eval(), encode))
        self.n, self.rng, self.temp = n_samples, random.Random(seed), temp
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    def _rollout(self, rounds, assign):
        """assign[i][seat] -> model index for simulation i."""
        live = list(range(len(rounds)))
        while live:
            groups = {}
            for i in live:
                r = rounds[i]; p = r.to_act
                groups.setdefault(assign[i][p], []).append(i)
            for mi, idxs in groups.items():
                net, enc = self.models[mi]
                x = torch.from_numpy(np.stack([enc(rounds[i], rounds[i].to_act) for i in idxs])).to(self.dev)
                m = torch.from_numpy(np.stack([mask_of(rounds[i]) for i in idxs])).to(self.dev)
                with torch.no_grad():
                    lg, _ = net(x, m)
                u = torch.rand(lg.shape, generator=self.gen, device=self.dev).clamp_(1e-10, 1.0)
                a = (lg / self.temp - torch.log(-torch.log(u))).argmax(-1).tolist()
                for i, ai in zip(idxs, a):
                    r = rounds[i]
                    if r.phase == "bid": r.bid(ai - 60)
                    else: r.play(ai)
            live = [i for i in live if rounds[i].phase != "done"]

    def _decide(self, v, options, apply, use_voids):
        r, me = v.r, v.player
        voids = infer_voids(r) if use_voids else [set()] * N_PLAYERS
        sims, keys, assign = [], [], []
        for _ in range(self.n):
            hands = sample_hands(r, me, self.rng, voids)
            base = r.clone()
            for p, h in hands.items(): base.hands[p] = h
            asg = [0] * 4
            if len(self.models) > 1:
                for q in range(4):
                    if q != me: asg[q] = self.rng.randrange(len(self.models))
            for o in options:
                s = base.clone(); apply(s, o); sims.append(s); keys.append(o); assign.append(asg)
        self._rollout(sims, assign)
        tot = {o: 0.0 for o in options}
        for s, o in zip(sims, keys):
            tot[o] += s.scores()[me]
        return max(options, key=lambda o: tot[o])

    def bid(self, v):
        if len(v.legal_bids) == 1: return v.legal_bids[0]
        return self._decide(v, v.legal_bids, lambda s, o: s.bid(o), False)

    def play(self, v):
        if len(v.legal_cards) == 1: return v.legal_cards[0]
        return self._decide(v, v.legal_cards, lambda s, o: s.play(o), True)
