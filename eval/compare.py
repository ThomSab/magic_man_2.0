"""Same seeds, same opponents (3 heuristics), seat-rotated. python -m eval.compare N_GAMES"""
import sys, numpy as np, torch
from multiprocessing import Pool
from wizard.engine import play_game
from wizard.agents import HeuristicAgent
from wizard.pimc import PIMCAgent
from wizard.features import Net, NetAgent

def make(kind, g):
    if kind == "heur": return HeuristicAgent()
    if kind == "pimc": return PIMCAgent(8, g)
    torch.set_num_threads(1)
    n = Net(); n.load_state_dict(torch.load("checkpoints/run3_final.pt", map_location="cpu")); n.eval()
    return NetAgent(n)

def job(a):
    kind, g, seat = a
    ag = [HeuristicAgent() for _ in range(4)]
    ag[seat] = make(kind, g)
    return kind, g, seat, play_game(ag, 9000 + g)[seat]

if __name__ == "__main__":
    N = int(sys.argv[1])
    jobs = [(k, g, s) for k in ("heur", "net", "pimc") for g in range(N) for s in range(4)]
    with Pool(10) as p:
        res = p.map(job, jobs, chunksize=4)
    for k in ("heur", "net", "pimc"):
        x = np.array([r[3] for r in res if r[0] == k])
        print(f"{k:5s} n={len(x)} mean={x.mean():7.1f} +- {1.96*x.std()/len(x)**.5:.1f}")
    d = {k: {(r[1], r[2]): r[3] for r in res if r[0] == k} for k in ("net", "pimc")}
    diff = np.array([d["pimc"][key] - d["net"][key] for key in d["net"]])
    print(f"paired pimc-net: {diff.mean():.1f} +- {1.96*diff.std()/len(diff)**.5:.1f}")
