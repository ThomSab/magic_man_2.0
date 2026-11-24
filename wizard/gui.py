"""Mouse-driven Wizard (tkinter). You are P0 (bottom). Run:
    python -m wizard.gui --bots net net net [--seed N] [--ckpt checkpoints/run4.pt] [--speed 1.0]
"""
import argparse, tkinter as tk
from tkinter import font as tkfont
from .engine import *
from . import cli

# suit index 0..3 -> colour (engine order: Green, Yellow, Red, Blue)
SUIT_BG = ["#1e9e3a", "#f2c200", "#d62828", "#1f5fd6"]
SUIT_FG = ["white", "#222222", "white", "white"]
RANKS_SHORT = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
FELT, PANEL = "#2f3b4a", "#222b36"
EMPTY = "#3e4c5e"
GOLD = "#ffd21f"


def face(c):
    """(text, fg, bg) for a card."""
    if is_wizard(c): return "WIZARD\n\u2605", "white", "#6a1fb0"
    if is_jester(c): return "JESTER\n\u263A", "white", "#555555"
    s = suit_of(c)
    return f"{RANKS_SHORT[rank_of(c)]}\n{cli.SUIT_NAMES[s]}", SUIT_FG[s], SUIT_BG[s]


class App:
    def __init__(self, root, bots, names, seed, speed):
        self.root, self.bots, self.names, self.speed = root, bots, names, speed
        self.g = Game(seed)
        self.r = None
        self.shown = []        # cards displayed on the table (p, c)
        self.busy = False
        self.trick_no = 0
        self.log_lines = []
        root.title("Wizard")
        root.configure(bg=FELT)
        self.f_big = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.f_mid = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Segoe UI", size=10)
        self.f_card = tkfont.Font(family="Segoe UI", size=15, weight="bold")
        self.build()
        self.new_round()

    # ---------- layout
    def build(self):
        root = self.root
        self.info = tk.Label(root, text="", font=self.f_big, bg=PANEL, fg="white", anchor="w", padx=10, pady=6)
        self.info.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.trump_bar = tk.Label(root, text="", font=tkfont.Font(family="Segoe UI", size=20, weight="bold"), pady=6)
        self.trump_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.bid_bar = tk.Label(root, text="", font=self.f_big, pady=5)
        self.bid_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        left = tk.Frame(root, bg=FELT)
        left.grid(row=3, column=0, sticky="nsew")
        right = tk.Frame(root, bg=FELT)
        right.grid(row=3, column=1, sticky="ns", padx=6)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)

        # scoreboard
        self.board = tk.Frame(left, bg=PANEL, padx=6, pady=6)
        self.board.pack(fill="x", padx=8, pady=6)
        heads = ["Player", "Bid", "Won", "Need", "Cards", "Total"]
        for j, h in enumerate(heads):
            tk.Label(self.board, text=h, font=self.f_mid, bg=PANEL, fg="#e8e8a0", width=16 if j == 0 else 7).grid(row=0, column=j)
        self.cells = [[tk.Label(self.board, text="", font=self.f_small, bg=PANEL, fg="white",
                                width=16 if j == 0 else 7, anchor="w" if j == 0 else "center")
                       for j in range(6)] for _ in range(4)]
        for i in range(4):
            for j in range(6):
                self.cells[i][j].grid(row=i + 1, column=j)

        # table
        self.table = tk.Frame(left, bg=FELT)
        self.table.pack(pady=4)
        self.slot_name, self.slot_card = {}, {}
        pos = {2: (0, 1), 1: (1, 0), 3: (1, 2), 0: (2, 1)}  # P2 top, P1 left, P3 right, YOU bottom
        for p, (rw, cl) in pos.items():
            fr = tk.Frame(self.table, bg=FELT, padx=14, pady=4)
            fr.grid(row=rw, column=cl)
            self.slot_name[p] = tk.Label(fr, text=self.names[p], font=self.f_mid, bg=FELT, fg="white")
            self.slot_name[p].pack()
            self.slot_card[p] = tk.Label(fr, text="", font=self.f_card, width=8, height=3, bg=EMPTY,
                                         relief="groove", bd=3)
            self.slot_card[p].pack()
        self.center = tk.Label(self.table, text="", font=self.f_mid, bg=FELT, fg="#ffe9a0", width=26, wraplength=220)
        self.center.grid(row=1, column=1)

        # prompt + action area
        self.prompt = tk.Label(left, text="", font=self.f_mid, bg=FELT, fg="white")
        self.prompt.pack(pady=(6, 2))
        self.actions = tk.Frame(left, bg=FELT)
        self.actions.pack(pady=4, padx=8)

        # log
        tk.Label(right, text="Game log", font=self.f_mid, bg=FELT, fg="white").pack()
        self.log = tk.Text(right, width=44, height=34, font=self.f_small, bg="#0f3d20", fg="#e8f5e8",
                           state="disabled", wrap="word")
        self.log.pack(fill="y", expand=True)

    # ---------- helpers
    def say(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pn(self, p):
        return "You" if p == 0 else self.names[p]

    def clear_actions(self):
        for w in self.actions.winfo_children():
            w.destroy()

    def trump_text(self):
        r = self.r
        return "none" if r.trump is None else cli.SUIT_NAMES[r.trump].upper()

    def refresh(self):
        r = self.r
        up = "none" if r.upcard is None else cli.long_name(r.upcard)
        self.info.configure(text=f"Round {r.n}/15   |   Trump: {self.trump_text()}   |   Upcard: {up}   |   Dealer: {self.pn(r.dealer)}")
        bids = r.bid_by_player()
        for p in range(4):
            b = bids[p]
            need = "-" if b is None else f"{b - r.won[p]:+d}"
            row = [self.pn(p) + (" (dealer)" if p == r.dealer else ""), "-" if b is None else b,
                   r.won[p], need, len(r.hands[p]), self.g.totals[p]]
            for j, v in enumerate(row):
                self.cells[p][j].configure(text=str(v), fg="#ffe9a0" if p == 0 else "white")
        for p in range(4):
            txt, col, bg = "", "white", EMPTY
            hl = FELT
            for q, c in self.shown:
                if q == p:
                    txt, col, bg = face(c)
                    if r.trump is not None and suit_of(c) == r.trump: hl = GOLD
            self.slot_card[p].configure(text=txt, fg=col, bg=bg, highlightthickness=4, highlightbackground=hl)
        self.update_bars()

    def update_bars(self):
        r = self.r
        if r.trump is None:
            self.trump_bar.configure(text="TRUMP: NONE (no trump this round)", bg="#444444", fg="white")
        else:
            s = r.trump
            self.trump_bar.configure(
                text=f"TRUMP SUIT:  {cli.SUIT_NAMES[s].upper()}   (every {cli.SUIT_NAMES[s].lower()} card beats all other colours)",
                bg=SUIT_BG[s], fg=SUIT_FG[s])
        total, n, k = sum(r.bids), r.n, len(r.bids)
        if k < 4:
            self.bid_bar.configure(text=f"Bids so far: {total}  |  tricks this round: {n}  |  {k} of 4 players have bid",
                                   bg=PANEL, fg="white")
        elif total > n:
            self.bid_bar.configure(text=f"Total bids {total} vs {n} tricks:  OVER-BID by {total - n}  (more bids than tricks)",
                                   bg="#b5651d", fg="white")
        else:
            self.bid_bar.configure(text=f"Total bids {total} vs {n} tricks:  UNDER-BID by {n - total}  (fewer bids than tricks)",
                                   bg="#1d6fb5", fg="white")

    # ---------- round flow
    def new_round(self):
        self.clear_actions()
        self.r = self.g.start_round()
        self.shown, self.trick_no = [], 0
        r = self.r
        self.say(f"\n=== Round {r.n}: {r.n} card(s) each. Trump: {self.trump_text()} ===")
        self.center.configure(text="Bidding")
        self.refresh()
        self.root.after(200, self.step)

    def delay(self, ms):
        return int(ms * self.speed)

    def step(self):
        r = self.r
        if r.phase == "done":
            return self.round_over()
        p = r.to_act
        if p == 0:
            return self.human_turn()
        v = View(r, p)
        self.prompt.configure(text=f"{self.pn(p)} is thinking...")
        self.root.update_idletasks()
        if r.phase == "bid":
            b = self.bots[p].bid(v)
            r.bid(b)
            self.say(f"{self.pn(p)} bids {b}")
            self.refresh()
            if r.phase == "play":
                self.say(f"Bids total {sum(r.bids)} for {r.n} tricks.")
                self.center.configure(text="")
            self.root.after(self.delay(350), self.step)
        else:
            c = self.bots[p].play(v)
            self.do_play(p, c)

    def human_turn(self):
        r = self.r
        v = View(r, 0)
        self.clear_actions()
        if r.phase == "bid":
            self.show_hand(v.hand, [], hand_only=True)
            so_far = ", ".join(f"{self.pn(q)}={b}" for q, b in zip(r.bid_order, r.bids)) or "nobody yet"
            extra = f"  (dealer: you can't bid {r.n - sum(r.bids)})" if len(r.bids) == 3 else ""
            self.prompt.configure(text=f"Your bid - tricks you expect to win.  Bids so far: {so_far}.{extra}")
            row = tk.Frame(self.actions, bg=FELT)
            row.pack(pady=(8, 0))
            for b in range(r.n + 1):
                ok = b in v.legal_bids
                tk.Button(row, text=str(b), width=4, height=2, font=self.f_mid, state="normal" if ok else "disabled",
                          command=lambda b=b: self.human_bid(b)).pack(side="left", padx=3)
        else:
            if not r.trick:
                self.shown = []
                self.refresh()
            ls = led_suit(r.trick) if r.trick else None
            if not r.trick: hint = "You lead - play any card."
            elif ls is None: hint = "No suit to follow - play any card."
            else: hint = f"{cli.SUIT_NAMES[ls].upper()} was led - follow suit if you can (Wizards/Jesters always OK)."
            self.prompt.configure(text=f"Your turn. {hint}  Click a card.")
            self.show_hand(v.hand, v.legal_cards)

    def show_hand(self, hand, legal, hand_only=False):
        row = tk.Frame(self.actions, bg=FELT)
        row.pack()
        order = sorted(hand, key=lambda c: (100 if is_wizard(c) else 101 if is_jester(c) else suit_of(c) * 20 - rank_of(c)))
        prev = None
        for c in order:
            grp = "W" if is_wizard(c) else "J" if is_jester(c) else suit_of(c)
            if prev is not None and grp != prev:
                tk.Frame(row, width=10, bg=FELT).pack(side="left")
            prev = grp
            txt, col, bg = face(c)
            is_trump = self.r.trump is not None and suit_of(c) == self.r.trump
            playable = (c in legal) and not hand_only
            if is_trump: txt += "\nTRUMP"
            btn = tk.Button(row, text=txt, font=self.f_card, width=7, height=4, fg=col, bg=bg,
                            activebackground=bg, activeforeground=col, disabledforeground=col,
                            relief="raised", bd=3,
                            highlightthickness=5 if is_trump else 0,
                            highlightbackground=GOLD, highlightcolor=GOLD,
                            state="normal" if playable else "disabled",
                            command=lambda c=c: self.human_play(c))
            if not playable and not hand_only:
                btn.configure(bg="#7a7f87", fg="#d0d0d0", disabledforeground="#d0d0d0")
            btn.pack(side="left", padx=2)
        tk.Label(self.actions, text="Gold-bordered cards marked TRUMP are trump. Greyed cards cannot be played now.",
                 font=self.f_small, bg=FELT, fg="#e8e8e8").pack(pady=2)

    def human_bid(self, b):
        self.clear_actions()
        self.r.bid(b)
        self.say(f"You bid {b}")
        self.refresh()
        if self.r.phase == "play":
            self.say(f"Bids total {sum(self.r.bids)} for {self.r.n} tricks.")
            self.center.configure(text="")
        self.root.after(self.delay(250), self.step)

    def human_play(self, c):
        self.clear_actions()
        self.do_play(0, c)

    def do_play(self, p, c):
        r = self.r
        if not r.trick:
            self.trick_no += 1
            self.shown = []
            self.say(f"-- Trick {self.trick_no} (led by {self.pn(p)})")
            self.center.configure(text=f"Trick {self.trick_no}")
        r.play(c)
        self.shown.append((p, c))
        self.say(f"{self.pn(p)} plays {cli.long_name(c)}")
        self.refresh()
        if not r.trick and r.played:  # trick just completed
            t = r.played[-1]
            w = trick_winner(t, r.trump)
            wc = [c2 for q, c2 in t if q == w][0]
            b = r.bid_by_player()[w]
            self.say(f"   {self.pn(w)} win{'' if w == 0 else 's'} the trick with {cli.long_name(wc)} ({r.won[w]} won, bid {b})")
            self.center.configure(text=f"{self.pn(w)} win{'' if w == 0 else 's'} the trick!")
            self.prompt.configure(text="")
            self.root.after(self.delay(1700), self.after_trick)
        else:
            self.root.after(self.delay(450), self.step)

    def after_trick(self):
        self.shown = []
        self.center.configure(text="")
        self.refresh()
        self.step()

    def round_over(self):
        r, g = self.r, self.g
        g.finish_round()
        self.refresh()
        bids, sc = r.bid_by_player(), r.scores()
        self.say(f"--- Round {r.n} result ---")
        for p in range(4):
            self.say(f"{self.pn(p):<14} bid {bids[p]} won {r.won[p]} -> {sc[p]:+d}  (total {g.totals[p]})")
        self.clear_actions()
        self.center.configure(text="")
        self.prompt.configure(text=f"Round {r.n} over.")
        box = tk.Frame(self.actions, bg=PANEL, padx=10, pady=8)
        box.pack()
        for j, h in enumerate(["Player", "Bid", "Won", "Result", "Round pts", "Total"]):
            tk.Label(box, text=h, font=self.f_mid, bg=PANEL, fg="#e8e8a0", width=12).grid(row=0, column=j)
        for p in range(4):
            made = bids[p] == r.won[p]
            vals = [self.pn(p), bids[p], r.won[p], "MADE" if made else "MISSED", f"{sc[p]:+d}", g.totals[p]]
            for j, v in enumerate(vals):
                tk.Label(box, text=str(v), font=self.f_small, bg=PANEL, width=12,
                         fg=("#8dff8d" if made else "#ff9b9b") if j == 3 else "white").grid(row=p + 1, column=j)
        if g.over:
            order = sorted(range(4), key=lambda q: -g.totals[q])
            tk.Label(self.actions, text="GAME OVER - winner: " + self.pn(order[0]) +
                     "   |   " + "   ".join(f"{i + 1}. {self.pn(q)} {g.totals[q]}" for i, q in enumerate(order)),
                     font=self.f_mid, bg=FELT, fg="#ffe9a0").pack(pady=6)
            self.say("=== GAME OVER ===")
            tk.Button(self.actions, text="Close", font=self.f_mid, command=self.root.destroy).pack()
        else:
            tk.Button(self.actions, text="Next round", font=self.f_mid, width=16, height=2,
                      command=self.new_round).pack(pady=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", nargs=3, default=["net", "net", "net"], choices=["net", "net2", "search", "pimc", "heuristic", "random"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="network checkpoint for 'net' bots")
    ap.add_argument("--speed", type=float, default=1.0, help="animation delay multiplier (0.3 = faster)")
    a = ap.parse_args()
    if a.ckpt:
        cli.CKPT = a.ckpt
    bots = [None] + [cli.make_bot(n, i) for i, n in enumerate(a.bots)]
    names = ["You"] + [f"P{i + 1} ({n})" for i, n in enumerate(a.bots)]
    root = tk.Tk()
    App(root, bots, names, a.seed, a.speed)
    root.mainloop()


if __name__ == "__main__":
    main()


