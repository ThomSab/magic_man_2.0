"""Search evaluation. python -m eval.searcheval N [procs]
Compares plain net vs net-search at the same deals: vs 3 heuristics, vs 3 run4, mixed table."""
import sys, numpy as np
from eval.harness import *

NET = "checkpoints/v2w_w0.5.pt"
RUN4 = "net1:checkpoints/run4_final.pt"
if __name__ == "__main__":
    N = int(sys.argv[1]); procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    plain = "net2:" + NET
    out = {}
    for s in [plain, f"ns2:{NET}:8", f"ns2:{NET}:16"]:
        for title, opps in [("vs 3 heur", ["heur"] * 3), ("vs 3 run4", [RUN4] * 3)]:
            st = focal_vs(s, opps, N, procs=procs)
            sm = st.summary(); out[(s, title)] = st.score
            print(f"{s:40s} {title:10s} score {sm['score']:.0f}+-{sm['ci']:.0f} bid {sm['bid_acc']*100:.1f}% win {sm['win']*100:.0f}% rank {sm['rank']:.2f}", flush=True)
    for title in ["vs 3 heur", "vs 3 run4"]:
        for s in [f"ns2:{NET}:8", f"ns2:{NET}:16"]:
            d = np.array(out[(s, title)]) - np.array(out[(plain, title)])
            print(f"paired {s} - plain {title}: {d.mean():.1f} +- {1.96*d.std()/len(d)**.5:.1f}", flush=True)
