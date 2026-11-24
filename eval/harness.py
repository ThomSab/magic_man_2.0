"""Game-level evaluation harness: score, bid accuracy, win rate / rank, collusion indicators."""
import random, numpy as np, torch
from multiprocessing import Pool
from wizard.engine import *
from wizard.agents import HeuristicAgent, RandomAgent
from wizard.pimc import PIMCAgent
from wizard.features import Net, NetAgent
from wizard.features2 import load_net2, Agent2

_cache = {}


def make_agent(spec, seed=0):
    """spec: 'heur' | 'random' | 'pimc[:samples]' | 'net1:<ckpt>' | 'net2:<ckpt>'"""
    torch.set_num_threads(1)
    kind, _, arg = spec.partition(":")
    if kind == "heur": return HeuristicAgent()
    if kind == "random": return RandomAgent(random.Random(seed))
    if kind == "pimc": return PIMCAgent(int(arg or 8), seed)
    if kind == "net1":
        if spec not in _cache:
            n = Net(); n.load_state_dict(torch.load(arg, map_location="cpu")); _cache[spec] = n.eval()
        return NetAgent(_cache[spec], True, seed)
    if kind == "net2":
        if spec not in _cache: _cache[spec] = load_net2(arg)
        return Agent2(_cache[spec], True, seed)
    if kind == "ns2":  # ns2:<ckpt>:<samples>  (GPU batched search over a v2 net)
        from wizard.netsearch2 import NetSearch2
        ck, _, ns = arg.rpartition(":")
        key = "net:" + ck
        if key not in _cache: _cache[key] = load_net2(ck, "cuda")
        return NetSearch2(_cache[key], int(ns), seed)
    if kind == "ns3":  # ns3:<ckpt>:<self|mix>:<samples>
        from wizard.netsearch3 import NetSearch3
        ck, _, rest = arg.partition("|")
        mix, ns = rest.split(":")
        key = "net:" + ck
        if key not in _cache: _cache[key] = load_net2(ck, "cuda")
        return NetSearch3(_cache[key], int(ns), seed, opp_mix=mix)
    raise ValueError(spec)


def ranks_of(totals):
    """average ranks (1=best), ties averaged."""
    out = []
    for p in range(4):
        better = sum(1 for q in range(4) if totals[q] > totals[p])
        ties = sum(1 for q in range(4) if totals[q] == totals[p])
        out.append(better + (ties + 1) / 2.0)
    return out


def rank_reward(rank):  # rank 1..4 -> +1, +1/3, -1/3, -1
    return (2.5 - rank) * (2.0 / 3.0)


def win_share(totals, p):
    top = max(totals)
    return (1.0 / sum(1 for t in totals if t == top)) if totals[p] == top else 0.0


def run_game(agents, seed):
    g = Game(seed)
    made, nb = [0] * 4, [0] * 4
    opp, gift = [0] * 4, [0] * 4
    dev = []
    for _ in range(N_ROUNDS):
        r = g.start_round()
        while r.phase != "done":
            p = r.to_act
            v = View(r, p)
            if r.phase == "bid":
                r.bid(agents[p].bid(v)); continue
            c = agents[p].play(v)
            last = len(r.trick) == 3
            if last:
                bids = r.bid_by_player()
                need = [bids[q] - r.won[q] for q in range(4)]
                wins = [x for x in v.legal_cards if trick_winner(r.trick + [(p, x)], r.trump) == p]
                can = need[p] > 0 and wins and len(v.legal_cards) > 1
            r.play(c)
            if last and can:
                opp[p] += 1
                w = trick_winner(r.played[-1], r.trump)
                if c not in wins and w != p and need[w] > 0:
                    gift[p] += 1
        b = r.bid_by_player()
        for p in range(4):
            made[p] += int(b[p] == r.won[p]); nb[p] += 1
        dev.append(sum(b) - r.n)
        g.finish_round()
    return dict(totals=list(g.totals), made=made, nb=nb, dev=dev, opp=opp, gift=gift)


def _job(a):
    specs, seed = a
    return run_game([make_agent(s, seed * 11 + i) for i, s in enumerate(specs)], seed)


def run_jobs(jobs, procs=8):
    with Pool(procs) as p:
        return p.map(_job, jobs, chunksize=1)


class Stats:
    def __init__(self):
        self.score, self.made, self.nb, self.rank, self.win = [], 0, 0, [], []
        self.opp, self.gift = 0, 0

    def add(self, res, seat):
        t = res["totals"]
        self.score.append(t[seat]); self.made += res["made"][seat]; self.nb += res["nb"][seat]
        self.rank.append(ranks_of(t)[seat]); self.win.append(win_share(t, seat))
        self.opp += res["opp"][seat]; self.gift += res["gift"][seat]

    def summary(self):
        s = np.array(self.score, float)
        return dict(n=len(s), score=s.mean(), ci=1.96 * s.std() / len(s) ** .5,
                    bid_acc=self.made / max(1, self.nb), rank=float(np.mean(self.rank)),
                    win=float(np.mean(self.win)), gift=self.gift / max(1, self.opp))


def focal_vs(focal, opps, n_games, seed0=100000, procs=8):
    """focal agent in each seat vs 3 given opponents (list of 3 specs). Same deals for every focal."""
    jobs, seats = [], []
    for g in range(n_games):
        for seat in range(4):
            specs = list(opps); specs.insert(seat, focal)
            jobs.append((specs, seed0 + g)); seats.append(seat)
    res = run_jobs(jobs, procs)
    st = Stats()
    for r, s in zip(res, seats): st.add(r, s)
    return st


def mixed_table(specs4, n_games, seed0=200000, procs=8):
    """4 different agents; rotate seats. Returns {spec: Stats}."""
    jobs, labels = [], []
    for g in range(n_games):
        for k in range(4):
            order = specs4[k:] + specs4[:k]
            jobs.append((order, seed0 + g)); labels.append(order)
    res = run_jobs(jobs, procs)
    out = {s: Stats() for s in specs4}
    for r, order in zip(res, labels):
        for seat, s in enumerate(order): out[s].add(r, seat)
    return out


def self_table(spec, n_games, seed0=300000, procs=8):
    """4 copies of the same agent. Returns table-level collusion indicators."""
    res = run_jobs([([spec] * 4, seed0 + g) for g in range(n_games)], procs)
    tot = np.array([r["totals"] for r in res], float)
    made = sum(sum(r["made"]) for r in res); nb = sum(sum(r["nb"]) for r in res)
    dev = np.concatenate([r["dev"] for r in res]).astype(float)
    opp = sum(sum(r["opp"]) for r in res); gift = sum(sum(r["gift"]) for r in res)
    return dict(n=len(res), mean_player_score=tot.mean(), bid_acc=made / nb, mean_bid_dev=dev.mean(),
                gift_rate=gift / max(1, opp), spread=float(tot.std(axis=1).mean()))



