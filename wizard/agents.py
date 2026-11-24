import random
from .engine import *


def strength(c, trump):
    """Ordinal power of a card, ignoring led suit."""
    if is_wizard(c): return 100
    if is_jester(c): return -1
    return rank_of(c) + (20 if suit_of(c) == trump else 0)


class RandomAgent:
    def __init__(self, rng=None): self.rng = rng or random.Random()
    def bid(self, v): return self.rng.choice(v.legal_bids)
    def play(self, v): return self.rng.choice(v.legal_cards)


def _wins_now(v, c):
    return trick_winner(v.trick + [(v.player, c)], v.trump) == v.player


class HeuristicAgent:
    def __init__(self, bid_bias=0.0): self.bid_bias = bid_bias

    def expected_tricks(self, v):
        n, t = v.n, v.trump
        e = 0.0
        for c in v.hand:
            if is_wizard(c): e += 1.0
            elif is_jester(c): continue
            else:
                r = rank_of(c)
                if suit_of(c) == t:
                    e += 0.15 + 0.85 * max(0, r - 5) / 7  # high trumps likely win
                elif r == 12: e += 0.55
                elif r == 11: e += 0.3
                elif r == 10: e += 0.12
        if t is not None and n >= 4:
            pass
        # with many cards, high-card tricks are contested more
        if n > 8: e *= 0.9
        return e + self.bid_bias

    def bid(self, v):
        want = round(self.expected_tricks(v))
        return min(v.legal_bids, key=lambda b: (abs(b - want), b))

    def play(self, v):
        legal = v.legal_cards
        need = v.bids[v.player] - v.won[v.player]
        tr = v.trump
        key = lambda c: strength(c, tr)
        if not v.trick:
            if need > 0:
                return max(legal, key=key)
            jest = [c for c in legal if is_jester(c)]
            if jest: return jest[0]
            return min(legal, key=key)
        winners = [c for c in legal if _wins_now(v, c)]
        losers = [c for c in legal if not _wins_now(v, c)]
        last = len(v.trick) == N_PLAYERS - 1
        if need > 0:
            if winners:
                return min(winners, key=key)  # cheapest winning card
            return min(losers, key=key)  # dump junk, jester first
        if losers:
            return max(losers, key=key)  # shed the highest card that still loses
        return min(winners, key=key)
