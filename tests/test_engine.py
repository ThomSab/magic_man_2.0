import random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wizard.engine import *
from wizard.agents import RandomAgent


def test_winner():
    S = lambda s, r: s * 13 + r
    assert trick_winner([(0, S(0, 5)), (1, S(0, 9)), (2, S(1, 12)), (3, S(0, 2))], None) == 1
    assert trick_winner([(0, S(0, 5)), (1, S(0, 9)), (2, S(1, 0)), (3, S(0, 2))], 1) == 2
    assert trick_winner([(0, S(0, 5)), (1, 52), (2, S(1, 0)), (3, 53)], 1) == 1
    assert trick_winner([(0, 56), (1, 57), (2, 58), (3, 59)], 0) == 0
    # jester lead: next suited card sets suit
    assert trick_winner([(0, 56), (1, S(2, 3)), (2, S(2, 8)), (3, S(0, 12))], None) == 2


def test_follow():
    r = Round(3, 0, random.Random(1))
    r.phase = "play"; r.leader = 1
    r.hands = [{0}, {5, 20}, {13, 14, 56}, {30, 52}]
    r.play(5)  # player1 leads C suit (0..12)
    assert r.to_act == 2
    r.hands[2] = {13, 14, 56, 7}
    assert r.legal_cards() == [7, 56]
    # wizard lead: no obligation
    r2 = Round(3, 0, random.Random(1)); r2.phase = "play"; r2.leader = 0
    r2.hands = [{52, 1}, {2, 20}, {3}, {4}]
    r2.play(52)
    assert r2.legal_cards() == [2, 20]


def test_dealer_bid():
    r = Round(3, 3, random.Random(0))
    for b in (1, 1, 0):
        r.bid(b)
    assert 1 not in r.legal_bids()


def test_fuzz():
    for seed in range(300):
        ag = [RandomAgent(random.Random(seed * 4 + i)) for i in range(4)]
        tot = play_game(ag, seed)
        assert len(tot) == 4


def test_round15_no_trump():
    r = Round(15, 0, random.Random(0))
    assert r.trump is None and r.upcard is None and all(len(h) == 15 for h in r.hands)
