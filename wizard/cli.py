"""Console Wizard: you (P0) vs 3 bots.
Run: python -m wizard.cli --bots net net net [--seed N]
"""
import argparse, os, torch
from .engine import *
from .agents import HeuristicAgent, RandomAgent
from .pimc import PIMCAgent
from .features import Net, NetAgent

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "run3_final.pt")
SUIT_NAMES = ["Green", "Yellow", "Red", "Blue"]
RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
BAR = "=" * 64
LINE = "-" * 64


def make_bot(name, seed):
    if name == "search":
        from .features2 import load_net2
        from .netsearch3 import NetSearch3
        net = load_net2(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "v2w_w0.5.pt"), "cuda")
        return NetSearch3(net, 64, seed, "cuda", "self")
    if name == "net2":
        from .features2 import load_net2, Agent2
        return Agent2(load_net2(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "v2w_w0.5.pt")))
    if name == "net":
        net = Net(); net.load_state_dict(torch.load(CKPT, map_location="cpu")); net.eval()
        return NetAgent(net)
    if name == "pimc": return PIMCAgent(8, seed)
    if name == "random": return RandomAgent()
    return HeuristicAgent()


def long_name(c):
    if is_wizard(c): return "WIZARD"
    if is_jester(c): return "JESTER"
    return f"{RANK_NAMES[rank_of(c)]} of {SUIT_NAMES[suit_of(c)]}"


def short(c):
    return card_name(c)


def pname(p, names):
    return "YOU" if p == 0 else names[p]


def trump_str(r):
    if r.trump is None:
        return "NONE (no trump this round)"
    return SUIT_NAMES[r.trump].upper()


def print_hand(hand, trump, legal=None):
    """Hand grouped by suit, with numbered picks for legal cards."""
    legal = set(legal) if legal is not None else set(hand)
    groups = [("Wizards", [c for c in hand if is_wizard(c)]),
              ("Jesters", [c for c in hand if is_jester(c)])]
    for s in range(4):
        label = SUIT_NAMES[s] + (" (TRUMP)" if s == trump else "")
        groups.append((label, sorted((c for c in hand if suit_of(c) == s), key=rank_of, reverse=True)))
    for label, cs in groups:
        if not cs: continue
        items = []
        for c in cs:
            tag = f"[{long_name(c)}]" if c in legal else f"({long_name(c)} - can't play)"
            items.append(tag)
        print(f"   {label:<18} " + "  ".join(items))


def print_table(r, names, totals):
    """Scoreboard: bid / won / still needed for each player."""
    bids = r.bid_by_player()
    print(LINE)
    print(f"   {'Player':<18}{'Bid':>5}{'Won':>5}{'Need':>6}{'Cards left':>12}{'Game total':>12}")
    for p in range(N_PLAYERS):
        b = bids[p]
        bs = "-" if b is None else str(b)
        need = "-" if b is None else f"{b - r.won[p]:+d}"
        dealer = " (dealer)" if p == r.dealer else ""
        print(f"   {(pname(p, names) + dealer):<18}{bs:>5}{r.won[p]:>5}{need:>6}{len(r.hands[p]):>12}{totals[p]:>12}")
    print("   (Need: + = tricks still to win, 0 = on target, - = over your bid)")
    print(LINE)


def ask_bid(v):
    while True:
        s = input(f"YOUR BID - how many tricks will you win? Allowed: {v.legal_bids}  > ").strip()
        if s.isdigit() and int(s) in v.legal_bids:
            return int(s)
        print("   Not allowed. Pick one of:", v.legal_bids)


def ask_card(v):
    legal = v.legal_cards
    print("   Legal cards (type the number or the name):")
    order = sorted(legal, key=lambda c: (suit_of(c) if suit_of(c) is not None else 9, -rank_of(c)))
    for i, c in enumerate(order, 1):
        print(f"      {i:>2}) {long_name(c)}  [{short(c)}]")
    while True:
        s = input("YOUR PLAY > ").strip().lower()
        if s.isdigit() and 1 <= int(s) <= len(order):
            return order[int(s) - 1]
        for c in order:
            if s in (short(c).lower(), long_name(c).lower()):
                return c
        print("   Not a legal card. Use a number from the list.")


def describe_lead_rule(r):
    ls = led_suit(r.trick)
    if not r.trick:
        return "You lead: play anything."
    if ls is None:
        return "Wizard/Jester led: no suit to follow, play anything."
    return f"Suit led: {SUIT_NAMES[ls]}. You must follow it if you can (Wizards/Jesters always allowed)."


def main():
    ap = argparse.ArgumentParser(description="Play Wizard vs 3 bots")
    ap.add_argument("--bots", nargs=3, default=["net", "net", "net"],
                    choices=["net", "net2", "search", "pimc", "heuristic", "random"])
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    bots = [None] + [make_bot(n, i) for i, n in enumerate(a.bots)]
    names = ["YOU"] + [f"P{i + 1}-{n}" for i, n in enumerate(a.bots)]
    print(BAR)
    print(" WIZARD - 4 players, 15 rounds")
    for p in range(4):
        print(f"   Seat {p}: {names[p]}")
    print(" Score: made bid = 20 + 10/trick. Missed = -10 per trick off.")
    print(BAR)
    g = Game(a.seed)
    for _ in range(N_ROUNDS):
        r = g.start_round()
        print("\n" + BAR)
        print(f" ROUND {r.n} of {N_ROUNDS}   ({r.n} card{'s' if r.n > 1 else ''} each, {r.n} trick{'s' if r.n > 1 else ''})")
        print(f" Dealer: {pname(r.dealer, names)}   First to bid/lead: {pname(r.bid_order[0], names)}")
        if r.upcard is None:
            print(" Upcard: none (all cards dealt)   TRUMP: NONE")
        else:
            extra = ""
            if suit_of(r.upcard) is None:
                extra = f"  (Wizard/Jester upcard -> random trump)"
            print(f" Upcard: {long_name(r.upcard)}{extra}")
            print(f" TRUMP SUIT: {trump_str(r)}")
        print(BAR)
        print("--- BIDDING ---")
        trick_no = 0
        while r.phase != "done":
            p = r.to_act
            v = View(r, p)
            if r.phase == "bid":
                if p == 0:
                    print("\nYour hand:")
                    print_hand(v.hand, r.trump)
                    so_far = ", ".join(f"{pname(q, names)}={b}" for q, b in zip(r.bid_order, r.bids)) or "nobody yet"
                    print(f"Bids so far: {so_far}   (total {sum(r.bids)} of {r.n} tricks)")
                    if len(r.bids) == 3:
                        print(f"You are the dealer: you cannot bid {r.n - sum(r.bids)}.")
                    r.bid(ask_bid(v))
                else:
                    b = bots[p].bid(v)
                    r.bid(b)
                    print(f"   {pname(p, names)} bids {b}")
                if r.phase == "play":
                    print("\n--- BIDS ARE IN ---")
                    print(f" Total bid {sum(r.bids)} vs {r.n} tricks available "
                          f"({'under' if sum(r.bids) < r.n else 'over'}-bid table)")
            else:
                if not r.trick:
                    trick_no += 1
                    print("\n" + LINE)
                    print(f" TRICK {trick_no} of {r.n}   (led by {pname(r.to_act, names)})")
                    print_table(r, names, g.totals)
                if p == 0:
                    if r.trick:
                        print(" Played so far this trick: " +
                              ", ".join(f"{pname(q, names)}: {long_name(c)}" for q, c in r.trick))
                    print(" " + describe_lead_rule(r))
                    print(f" Trump is {trump_str(r)}.  Your hand:")
                    print_hand(v.hand, r.trump, v.legal_cards)
                    c = ask_card(v)
                    print(f"   YOU play {long_name(c)}")
                else:
                    c = bots[p].play(v)
                    print(f"   {pname(p, names)} plays {long_name(c)}")
                r.play(c)
                if not r.trick and r.played:
                    t = r.played[-1]
                    w = trick_winner(t, r.trump)
                    wc = [c2 for q, c2 in t if q == w][0]
                    print(f"   >>> {pname(w, names)} {'win' if w == 0 else 'wins'} the trick with {long_name(wc)} "
                          f"(now has {r.won[w]} won, bid {r.bid_by_player()[w]})")
        g.finish_round()
        print("\n" + BAR)
        print(f" ROUND {r.n} RESULT")
        bids = r.bid_by_player()
        print(f"   {'Player':<14}{'Bid':>5}{'Won':>5}{'Result':>10}{'Round pts':>11}{'Total':>8}")
        for p in range(4):
            res = "MADE" if bids[p] == r.won[p] else "MISSED"
            print(f"   {names[p]:<14}{bids[p]:>5}{r.won[p]:>5}{res:>10}{r.scores()[p]:>+11d}{g.totals[p]:>8}")
        print(BAR)
        input("Press Enter for the next round... " if r.n < N_ROUNDS else "Press Enter for final result... ")
    print("\n" + BAR)
    print(" FINAL STANDINGS")
    for rank, p in enumerate(sorted(range(4), key=lambda q: -g.totals[q]), 1):
        print(f"   {rank}. {names[p]:<14}{g.totals[p]:>6}")
    print(BAR)


if __name__ == "__main__":
    main()



