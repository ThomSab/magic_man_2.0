"""Paired comparison on identical deals. python -m eval.compare2 N_GAMES kind1 kind2 ... (kinds: heur net pimc nsearch)"""
import sys, numpy as np, torch
from multiprocessing import Pool
from wizard.engine import play_game
from wizard.agents import HeuristicAgent
from wizard.pimc import PIMCAgent
from wizard.features import Net, NetAgent
from wizard.netsearch import NetSearchAgent

def load():
    n = Net(); n.load_state_dict(torch.load("checkpoints/run3_final.pt", map_location="cpu")); n.eval(); return n

def make(kind, g):
    torch.set_num_threads(1)
    if kind == "heur": return HeuristicAgent()
    if kind == "pimc": return PIMCAgent(8, g)
    if kind == "net": return NetAgent(load())
    if kind == "nsearch": return NetSearchAgent(load(), 16, g)

def job(a):
    kind, g, seat = a
    ag = [HeuristicAgent() for _ in range(4)]
    ag[seat] = make(kind, g)
    return kind, g, seat, play_game(ag, 9000 + g)[seat]

if __name__ == "__main__":
    N, kinds = int(sys.argv[1]), sys.argv[2:]
    jobs = [(k, g, s) for k in kinds for g in range(N) for s in range(4)]
    with Pool(8) as p:
        res = p.map(job, jobs, chunksize=1)
    d = {k: {(r[1], r[2]): r[3] for r in res if r[0] == k} for k in kinds}
    for k in kinds:
        x = np.array(list(d[k].values()))
        print(f"{k:8s} n={len(x)} mean={x.mean():7.1f} +- {1.96*x.std()/len(x)**.5:.1f}", flush=True)
    for k in kinds[1:]:
        diff = np.array([d[k][key] - d[kinds[0]][key] for key in d[k]])
        print(f"paired {k}-{kinds[0]}: {diff.mean():.1f} +- {1.96*diff.std()/len(diff)**.5:.1f}", flush=True)
