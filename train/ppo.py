"""PPO self-play. Episode = one round (n random in 1..15). Run: python -m train.ppo [minutes]"""
import sys, time, random, copy
import numpy as np, torch
from wizard.engine import *
from wizard.agents import HeuristicAgent
from wizard.features import *

dev = torch.device("cuda")


def collect(net, B, rng, p_heur=0.15, league=(), p_league=0.35):
    heur = HeuristicAgent()
    rounds, seats = [], []
    for _ in range(B):
        r = Round(rng.randint(1, 15), rng.randrange(4), rng)
        rounds.append(r)
        s = []
        for _ in range(4):
            u = rng.random()
            s.append(0 if u >= p_heur + (p_league if league else 0) else (1 if u < p_heur else rng.choice(league)))
        if 0 not in s: s[rng.randrange(4)] = 0
        seats.append(s)
    buf = [[] for _ in range(B)]  # per round: (seat, obs, mask, act, logp, val)
    active = list(range(B))
    while active:
        idx, obs, msk = [], [], []
        lgq = {}
        nxt = []
        for i in active:
            r = rounds[i]
            if r.phase == "done": continue
            p = r.to_act
            if seats[i][p] == 0:
                idx.append(i); obs.append(encode(r, p)); msk.append(mask_of(r))
            elif seats[i][p] == 1:
                v = View(r, p)
                r.bid(heur.bid(v)) if r.phase == "bid" else r.play(heur.play(v))
            else:
                lgq.setdefault(id(seats[i][p]), (seats[i][p], []))[1].append((i, p, encode(r, p), mask_of(r)))
            nxt.append(i)
        for net_, items in lgq.values():
            x = torch.from_numpy(np.stack([z[2] for z in items])).to(dev); m = torch.from_numpy(np.stack([z[3] for z in items])).to(dev)
            with torch.no_grad(): aa = torch.distributions.Categorical(logits=net_(x, m)[0]).sample().cpu().numpy()
            for z, a_ in zip(items, aa):
                rr = rounds[z[0]]
                rr.bid(int(a_) - 60) if rr.phase == "bid" else rr.play(int(a_))
        if idx:
            x = torch.from_numpy(np.stack(obs)).to(dev); m = torch.from_numpy(np.stack(msk)).to(dev)
            with torch.no_grad():
                lg, val = net(x, m)
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample(); lp = dist.log_prob(a)
            a, lp, val = a.cpu().numpy(), lp.cpu().numpy(), val.cpu().numpy()
            for k, i in enumerate(idx):
                r = rounds[i]; p = r.to_act
                buf[i].append((p, obs[k], msk[k], a[k], lp[k], val[k]))
                if r.phase == "bid": r.bid(int(a[k]) - 60)
                else: r.play(int(a[k]))
        active = [i for i in nxt if rounds[i].phase != "done"]
    O, M, A, LP, V, R = [], [], [], [], [], []
    for i in range(B):
        sc = rounds[i].scores()
        for p, o, m, a, lp, v in buf[i]:
            O.append(o); M.append(m); A.append(a); LP.append(lp); V.append(v); R.append(sc[p] / 50.0)
    t = lambda z, dt=None: torch.tensor(np.array(z), dtype=dt)
    return (t(O), t(M, torch.bool), t(A, torch.long), t(LP, torch.float32), t(V, torch.float32), t(R, torch.float32))


def update(net, opt, data, epochs=4, mb=8192, clip=0.2, ent=0.01):
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


def quick_eval(net, games=24):
    cpu = copy.deepcopy(net).cpu().eval()
    tot = []
    for g in range(games):
        for seat in range(4):
            ag = [HeuristicAgent() for _ in range(4)]
            ag[seat] = NetAgent(cpu)
            tot.append(play_game(ag, 5000 + g)[seat])
    return float(np.mean(tot)), 1.96 * float(np.std(tot)) / len(tot) ** .5


if __name__ == "__main__":
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    init = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "none" else None
    OUT = sys.argv[4] if len(sys.argv) > 4 else "checkpoints/ppo_latest.pt"
    LR = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-4
    net = Net().to(dev)
    if init: net.load_state_dict(torch.load(init, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = random.Random(0)
    league = []
    t0 = time.time(); it = 0
    while time.time() - t0 < minutes * 60:
        it += 1
        tc = time.time()
        data = collect(net, B, rng, league=league)
        tc = time.time() - tc
        pl, vl = update(net, opt, data)
        if it % 15 == 0:
            snap = copy.deepcopy(net).eval(); league.append(snap); league[:] = league[-8:]
        for g_ in opt.param_groups: g_['lr'] = LR * max(0.1, 1 - (time.time() - t0) / (minutes * 60))
        if it % 10 == 0:
            m, ci = quick_eval(net)
            print(f"it {it} t={time.time()-t0:.0f}s samples={len(data[0])} collect={tc:.1f}s vloss={vl:.3f} | net vs 3 heur: {m:.0f} +- {ci:.0f}", flush=True)
            torch.save(net.state_dict(), OUT)
    torch.save(net.state_dict(), OUT)


