"""Wizard rules engine (4 players, 15 rounds). No I/O. Seedable.

Cards: 0..51 suited (suit=c//13, rank=c%13, 0='2' .. 12='A'),
52..55 Wizards, 56..59 Jesters.
"""
import random

N_PLAYERS = 4
N_ROUNDS = 15
SUITS = "GYRB"  # suit index 0..3 = Green, Yellow, Red, Blue
RANKS = "23456789TJQKA"
WIZARD0, JESTER0, N_CARDS = 52, 56, 60


def is_wizard(c): return 52 <= c < 56
def is_jester(c): return c >= 56
def suit_of(c): return c // 13 if c < 52 else None
def rank_of(c): return c % 13


def card_name(c):
    if is_wizard(c): return "Wiz"
    if is_jester(c): return "Jes"
    return RANKS[rank_of(c)] + SUITS[suit_of(c)]


def led_suit(trick):
    """Suit that must be followed: suit of first non-jester card, None if wizard/none."""
    for _, c in trick:
        if is_jester(c):
            continue
        return suit_of(c)  # None for wizard
    return None


def trick_winner(trick, trump):
    """trick: list of (player, card) in play order. Returns player."""
    for p, c in trick:
        if is_wizard(c):
            return p
    if trump is not None:
        best = None
        for p, c in trick:
            if suit_of(c) == trump and (best is None or rank_of(c) > rank_of(best[1])):
                best = (p, c)
        if best:
            return best[0]
    ls = led_suit(trick)
    if ls is not None:
        best = None
        for p, c in trick:
            if suit_of(c) == ls and (best is None or rank_of(c) > rank_of(best[1])):
                best = (p, c)
        return best[0]
    return trick[0][0]  # all jesters


def score_round(bid, won):
    return 20 + 10 * won if bid == won else -10 * abs(bid - won)


class Round:
    """One deal. Phases: 'bid' then 'play' then 'done'."""

    def __init__(self, n, dealer, rng):
        self.n, self.dealer = n, dealer
        self.totals = [0] * N_PLAYERS  # game totals before this round (set by Game)
        deck = list(range(N_CARDS))
        rng.shuffle(deck)
        self.hands = [set(deck[i * n:(i + 1) * n]) for i in range(N_PLAYERS)]
        self.upcard = deck[N_PLAYERS * n] if N_PLAYERS * n < N_CARDS else None
        if self.upcard is None:
            self.trump = None
        elif suit_of(self.upcard) is None:
            self.trump = rng.randrange(4)
        else:
            self.trump = suit_of(self.upcard)
        self.bids = []  # in bidding order, first = left of dealer
        self.bid_order = [(dealer + 1 + i) % N_PLAYERS for i in range(N_PLAYERS)]
        self.won = [0] * N_PLAYERS
        self.trick = []
        self.played = []  # list of completed tricks [(p,c),...]
        self.leader = self.bid_order[0]
        self.phase = "bid"

    # ---- bidding
    @property
    def to_act(self):
        if self.phase == "bid":
            return self.bid_order[len(self.bids)]
        if self.phase == "play":
            return (self.trick[0][0] + len(self.trick)) % N_PLAYERS if self.trick else self.leader
        return None

    def legal_bids(self):
        opts = list(range(self.n + 1))
        if len(self.bids) == N_PLAYERS - 1:  # dealer
            forbidden = self.n - sum(self.bids)
            opts = [b for b in opts if b != forbidden]
        return opts

    def bid(self, b):
        assert self.phase == "bid" and b in self.legal_bids(), "illegal bid"
        self.bids.append(b)
        if len(self.bids) == N_PLAYERS:
            self.phase = "play"

    def bid_by_player(self):
        out = [None] * N_PLAYERS
        for p, b in zip(self.bid_order, self.bids):
            out[p] = b
        return out

    # ---- play
    def legal_cards(self):
        p = self.to_act
        hand = self.hands[p]
        ls = led_suit(self.trick) if self.trick else None
        if ls is None:
            return sorted(hand)
        follow = [c for c in hand if suit_of(c) == ls]
        if not follow:
            return sorted(hand)
        return sorted(follow + [c for c in hand if suit_of(c) is None])

    def play(self, c):
        assert self.phase == "play" and c in self.legal_cards(), "illegal card"
        p = self.to_act
        self.hands[p].remove(c)
        self.trick.append((p, c))
        if len(self.trick) == N_PLAYERS:
            w = trick_winner(self.trick, self.trump)
            self.won[w] += 1
            self.played.append(self.trick)
            self.trick = []
            self.leader = w
            if len(self.played) == self.n:
                self.phase = "done"

    def clone(self):
        c = Round.__new__(Round)
        c.__dict__.update(self.__dict__)
        c.hands = [set(h) for h in self.hands]
        c.bids = list(self.bids)
        c.won = list(self.won)
        c.trick = list(self.trick)
        c.played = list(self.played)
        return c

    def scores(self):
        b = self.bid_by_player()
        return [score_round(b[p], self.won[p]) for p in range(N_PLAYERS)]


class Game:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.round_no = 0
        self.totals = [0] * N_PLAYERS
        self.round = None

    def start_round(self):
        self.round_no += 1
        self.round = Round(self.round_no, (self.round_no - 1) % N_PLAYERS, self.rng)
        self.round.totals = list(self.totals)
        return self.round

    def finish_round(self):
        for p, s in enumerate(self.round.scores()):
            self.totals[p] += s

    @property
    def over(self):
        return self.round_no == N_ROUNDS and self.round.phase == "done"


def play_game(agents, seed=None):
    """agents: 4 objects with .bid(view)->int and .play(view)->card. Returns totals."""
    g = Game(seed)
    for _ in range(N_ROUNDS):
        r = g.start_round()
        while r.phase != "done":
            p = r.to_act
            if r.phase == "bid":
                r.bid(agents[p].bid(View(r, p)))
            else:
                r.play(agents[p].play(View(r, p)))
        g.finish_round()
    return g.totals


class View:
    """What player p can see (no opponents' hands)."""

    def __init__(self, r, p):
        self.r, self.player = r, p
        self.n, self.trump, self.upcard = r.n, r.trump, r.upcard
        self.hand = sorted(r.hands[p])
        self.bids = r.bid_by_player()
        self.won = list(r.won)
        self.trick = list(r.trick)
        self.played = r.played
        self.dealer = r.dealer
        self.totals = list(r.totals)
        self.legal_bids = r.legal_bids() if r.phase == "bid" else []
        self.legal_cards = r.legal_cards() if r.phase == "play" else []


