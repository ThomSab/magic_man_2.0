"""Net-guided PIMC: sample hidden hands, roll out every candidate with the net playing all seats (batched)."""
import random, numpy as np, torch
from .engine import *
from .pimc import infer_voids, sample_hands
from .features import encode, mask_of


class NetSearchAgent:
    def __init__(self, net, n_samples=16, seed=0, greedy_rollout=False):
        self.net, self.n, self.rng = net, n_samples, random.Random(seed)
        self.greedy = greedy_rollout
        self.gen = torch.Generator().manual_seed(seed)

    def _rollout(self, rounds):
        """Advance all rounds to completion with batched net inference."""
        live = [r for r in rounds if r.phase != "done"]
        while live:
            x = torch.from_numpy(np.stack([encode(r, r.to_act) for r in live]))
            m = torch.from_numpy(np.stack([mask_of(r) for r in live]))
            with torch.no_grad():
                lg, _ = self.net(x, m)
            if self.greedy:
                a = lg.argmax(-1)
            else:
                u = torch.rand(lg.shape, generator=self.gen).clamp_(1e-10, 1.0)
                a = (lg - torch.log(-torch.log(u))).argmax(-1)  # Gumbel-max; masked logits stay at -1e9
            for r, ai in zip(live, a.tolist()):
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
            tot[o] += score_round(s.bid_by_player()[me], s.won[me])
        return max(options, key=lambda o: tot[o])

    def bid(self, v):
        if len(v.legal_bids) == 1: return v.legal_bids[0]
        return self._decide(v, v.legal_bids, lambda s, o: s.bid(o), False)

    def play(self, v):
        if len(v.legal_cards) == 1: return v.legal_cards[0]
        return self._decide(v, v.legal_cards, lambda s, o: s.play(o), True)


