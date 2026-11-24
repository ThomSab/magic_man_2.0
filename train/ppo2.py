"""PPO v2: full 15-round games, zero-sum round reward + rank reward.
usage: python -m train.ppo2 NAME MINUTES W_RANK [BATCH_GAMES] [LR] [INIT]
reward per decision in round j = (s_p - mean(others))/50 + W * (rank_reward_final - rank_reward(totals before round j))
"""
import sys, time, random, copy
import numpy as np, torch
from wizard.engine import *
from wizard.agents import HeuristicAgent
from wizard.features import Net, encode, mask_of
from wizard.features2 import encode2, OBS2_DIM
from eval.harness import ranks_of, rank_reward, run_game
from wizard.features2 import Agent2

dev = torch.device("cuda")


def gumbel_sample(lg):
    u = torch.rand_like(lg).clamp_(1e-10, 1.0)
    return (lg - torch.log(-torch.log(u))).argmax(-1)


def phi(totals, p):
    return rank_reward(ranks_of(totals)[p])


def collect_games(net, G, rng, w, league, p_heur=0.12, p_league=0.4):
    """league: list of (net, enc). Seat code: 0 learner, 1 heuristic, else (net, enc)."""
    heur = HeuristicAgent()
    games = [Game(rng.randrange(1 << 30)) for _ in range(G)]
    seats = []
    for _ in range(G):
        s = []
        for _ in range(4):
            u = rng.random()
            if u < p_heur: s.append(1)
            elif league and u < p_heur + p_league: s.append(rng.choice(league))
            else: s.append(0)
        if 0 not in s: s[rng.randrange(4)] = 0
        seats.append(s)
    cur = [[] for _ in range(G)]
    rounds = [[] for _ in range(G)]  # per game: list of (decisions, scores, totals_before)
    done = [False] * G
    for g in games: g.start_round()
    active = list(range(G))
    while active:
        learner, others = [], {}
        nxt = []
        for i in active:
            g = games[i]; r = g.round
            if r.phase == "done":
                before = list(r.totals)
                g.finish_round()
                rounds[i].append((cur[i], r.scores(), before)); cur[i] = []
                if g.round_no < N_ROUNDS: g.start_round(); r = g.round
                else: done[i] = True; continue
            p = r.to_act
            code = seats[i][p]
            if isinstance(code, int) and code == 0: learner.append((i, p))
            elif isinstance(code, int): 
                v = View(r, p)
                r.bid(heur.bid(v)) if r.phase == "bid" else r.play(heur.play(v))
            else:
                others.setdefault(id(code), (code, []))[1].append((i, p))
            nxt.append(i)
        for (onet, oenc), items in others.values():
            x = torch.from_numpy(np.stack([oenc(games[i].round, p) for i, p in items])).to(dev)
            m = torch.from_numpy(np.stack([mask_of(games[i].round) for i, p in items])).to(dev)
            with torch.no_grad(): a = gumbel_sample(onet(x, m)[0]).cpu().numpy()
            for (i, p), ai in zip(items, a):
                r = games[i].round
                r.bid(int(ai) - 60) if r.phase == "bid" else r.play(int(ai))
        if learner:
            ob = np.stack([encode2(games[i].round, p) for i, p in learner])
            mk = np.stack([mask_of(games[i].round) for i, p in learner])
            with torch.no_grad():
                lg, val = net(torch.from_numpy(ob).to(dev), torch.from_numpy(mk).to(dev))
                a = gumbel_sample(lg)
                lp = torch.distributions.Categorical(logits=lg).log_prob(a)
            a, lp, val = a.cpu().numpy(), lp.cpu().numpy(), val.cpu().numpy()
            for k, (i, p) in enumerate(learner):
                r = games[i].round
                cur[i].append((p, ob[k], mk[k], a[k], lp[k], val[k]))
                r.bid(int(a[k]) - 60) if r.phase == "bid" else r.play(int(a[k]))
        active = [i for i in nxt if not done[i]]
    O, M, A, LP, V, R = [], [], [], [], [], []
    made = nb = 0; ranksum = 0.0; nl = 0
    for i, g in enumerate(games):
        fin = g.totals
        rk = ranks_of(fin)
        for dec, sc, before in rounds[i]:
            for p, o, m, a, lp, v in dec:
                diff = sc[p] - (sum(sc) - sc[p]) / 3.0
                rr = diff / 50.0 + w * (rank_reward(rk[p]) - phi(before, p))
                O.append(o); M.append(m); A.append(a); LP.append(lp); V.append(v); R.append(rr)
        for p in range(4):
            if seats[i][p] == 0: ranksum += rk[p]; nl += 1
    t = lambda z, dt=None: torch.tensor(np.array(z), dtype=dt)
    stats = dict(rank=ranksum / max(1, nl))
    return (t(O), t(M, torch.bool), t(A, torch.long), t(LP, torch.float32), t(V, torch.float32), t(R, torch.float32)), stats


def update(net, opt, data, epochs=3, mb=16384, clip=0.2, ent=0.01):
    O, M, A, LP, V, R = [d.to(dev) for d in data]
    adv = R - V
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(O)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, mb):
            j = perm[s:s + mb]
            lg, val = net(O[j], M[j])
            dist = torch.distributions.Categorical(logits=lg)
            lp = dist.log_prob(A[j])
            ratio = (lp - LP[j]).exp()
            pl = -torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j]).mean()
            vl = ((val - R[j]) ** 2).mean()
            loss = pl + 0.5 * vl - ent * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return pl.item(), vl.item()


def quick_eval(net, games=12):
    cpu = copy.deepcopy(net).cpu().eval()
    sc, rk, made, nb = [], [], 0, 0
    for g in range(games):
        for seat in range(4):
            ag = [HeuristicAgent() for _ in range(4)]
            ag[seat] = Agent2(cpu)
            res = run_game(ag, 5000 + g)
            sc.append(res["totals"][seat]); rk.append(ranks_of(res["totals"])[seat])
            made += res["made"][seat]; nb += res["nb"][seat]
    return float(np.mean(sc)), float(np.mean(rk)), made / nb


if __name__ == "__main__":
    name, minutes, w = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    G = int(sys.argv[4]) if len(sys.argv) > 4 else 192
    LR = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-4
    init = sys.argv[6] if len(sys.argv) > 6 else None
    torch.manual_seed(1); rng = random.Random(1)
    net = Net(in_dim=OBS2_DIM).to(dev)
    if init: net.load_state_dict(torch.load(init, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    fixed = Net(); fixed.load_state_dict(torch.load("checkpoints/run4_final.pt", map_location="cpu"))
    fixed = fixed.to(dev).eval()
    fixed_entry = (fixed, encode)   # run4 (v1 encoder) is a permanent league member
    snaps = []
    t0 = time.time(); it = 0
    out = f"checkpoints/{name}.pt"
    log = open(f"train/{name}.log", "a")
    while time.time() - t0 < minutes * 60:
        it += 1
        tc = time.time()
        league = [fixed_entry, fixed_entry] + [(s, encode2) for s in snaps]
        data, st = collect_games(net, G, rng, w, league)
        tc = time.time() - tc
        pl, vl = update(net, opt, data)
        if it % 12 == 0:
            snaps.append(copy.deepcopy(net).eval()); snaps[:] = snaps[-6:]
        frac = (time.time() - t0) / (minutes * 60)
        for g_ in opt.param_groups: g_["lr"] = LR * max(0.1, 1 - frac)
        if it % 10 == 0:
            sc, rk, ba = quick_eval(net)
            line = (f"it {it} t={time.time()-t0:.0f}s n={len(data[0])} collect={tc:.1f}s vloss={vl:.3f} train_rank={st['rank']:.2f} "
                    f"| vs 3 heur: score {sc:.0f} rank {rk:.2f} bidacc {ba:.2f}")
            print(line, flush=True); log.write(line + "\n"); log.flush()
            torch.save(net.state_dict(), out)
    torch.save(net.state_dict(), out)
