"""Perfect-Information Monte Carlo agent: sample hidden hands, roll out with heuristic."""
import random
from .engine import *
from .agents import HeuristicAgent


def infer_voids(r):
    """voids[p] = set of suits player p is known to lack."""
    voids = [set() for _ in range(N_PLAYERS)]
    for trick in r.played + ([r.trick] if r.trick else []):
        ls = None
        for p, c in trick:
            if ls is not None and not is_wizard(c) and not is_jester(c) and suit_of(c) != ls:
                voids[p].add(ls)
            elif ls is not None and (is_wizard(c) or is_jester(c)):
                pass  # wizard/jester can be played anytime: no info
            if ls is None and not is_jester(c):
                ls = suit_of(c)  # None for wizard lead => no suit
                if ls is None:
                    break
    return voids


def sample_hands(r, me, rng, voids):
    seen = set(r.hands[me])
    for t in r.played:
        seen.update(c for _, c in t)
    seen.update(c for _, c in r.trick)
    if r.upcard is not None:
        seen.add(r.upcard)
    unseen = [c for c in range(N_CARDS) if c not in seen]
    sizes = {p: len(r.hands[p]) for p in range(N_PLAYERS) if p != me}
    others = list(sizes)
    for attempt in range(30):
        rng.shuffle(unseen)
        pool = list(unseen)
        hands, ok = {}, True
        # fill most-constrained players first
        for p in sorted(others, key=lambda q: -len(voids[q])):
            ok_cards = [c for c in pool if suit_of(c) not in voids[p]]
            if len(ok_cards) < sizes[p]:
                ok = False; break
            pick = ok_cards[:sizes[p]]
            hands[p] = set(pick)
            ps = set(pick)
            pool = [c for c in pool if c not in ps]
        if ok:
            return hands
    # fallback: ignore voids
    rng.shuffle(unseen)
    i = 0
    hands = {}
    for p in others:
        hands[p] = set(unseen[i:i + sizes[p]]); i += sizes[p]
    return hands


class PIMCAgent:
    def __init__(self, n_samples=24, seed=0, rollout=None):
        self.n = n_samples
        self.rng = random.Random(seed)
        self.rollout = rollout or HeuristicAgent()

    def _finish(self, r):
        a = self.rollout
        while r.phase != "done":
            p = r.to_act
            v = View(r, p)
            if r.phase == "bid":
                r.bid(a.bid(v))
            else:
                r.play(a.play(v))

    def _score(self, r, me):
        b = r.bid_by_player()[me]
        return score_round(b, r.won[me])

    def bid(self, v):
        r = v.r
        if len(v.legal_bids) == 1:
            return v.legal_bids[0]
        me = v.player
        tot = {b: 0.0 for b in v.legal_bids}
        for _ in range(self.n):
            hands = sample_hands(r, me, self.rng, [set()] * N_PLAYERS)
            base = r.clone()
            for p, h in hands.items():
                base.hands[p] = h
            for b in v.legal_bids:
                c = base.clone()
                c.bid(b)
                self._finish(c)
                tot[b] += self._score(c, me)
        return max(v.legal_bids, key=lambda b: tot[b])

    def play(self, v):
        r = v.r
        if len(v.legal_cards) == 1:
            return v.legal_cards[0]
        me = v.player
        voids = infer_voids(r)
        tot = {c: 0.0 for c in v.legal_cards}
        for _ in range(self.n):
            hands = sample_hands(r, me, self.rng, voids)
            base = r.clone()
            for p, h in hands.items():
                base.hands[p] = h
            for c in v.legal_cards:
                s = base.clone()
                s.play(c)
                self._finish(s)
                tot[c] += self._score(s, me)
        return max(v.legal_cards, key=lambda c: tot[c])
