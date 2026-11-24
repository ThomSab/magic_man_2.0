"""python -m eval.searcheval2 N procs  -> eval/search2.json + log. Same deals for all agents (paired)."""
import sys, json, numpy as np
from eval.harness import *

NET = "checkpoints/v2w_w0.5.pt"
RUN4 = "net1:checkpoints/run4_final.pt"
if __name__ == "__main__":
    N = int(sys.argv[1]); procs = int(sys.argv[2])
    plain = "net2:" + NET
    cells = [("vs 3 run4", [RUN4] * 3, [plain, f"ns3:{NET}|self:16", f"ns3:{NET}|self:64", f"ns3:{NET}|mix:16", f"ns3:{NET}|mix:64"]),
             ("vs 3 heur", ["heur"] * 3, [plain, f"ns3:{NET}|self:64"])]
    res = {}
    for title, opps, specs in cells:
        for s in specs:
            st = focal_vs(s, opps, N, procs=procs)
            sm = st.summary(); res[f"{title}|{s}"] = st.score
            print(f"{title:10s} {s:45s} score {sm['score']:.0f}+-{sm['ci']:.0f} bid {sm['bid_acc']*100:.1f}% win {sm['win']*100:.0f}% rank {sm['rank']:.2f}", flush=True)
            json.dump(res, open("eval/search2.json", "w"))
        base = np.array(res[f"{title}|{specs[0]}"])
        for s in specs[1:]:
            d = np.array(res[f"{title}|{s}"]) - base
            print(f"  paired {s} - plain [{title}]: {d.mean():+.1f} +- {1.96*d.std()/len(d)**.5:.1f}", flush=True)
