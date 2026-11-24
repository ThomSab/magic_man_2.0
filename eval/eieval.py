import sys, json, numpy as np
from eval.harness import *
RUN4 = "net1:checkpoints/run4_final.pt"
if __name__ == "__main__":
    N = int(sys.argv[1]); specs = sys.argv[2:]
    res = {}
    for title, opps in [("vs 3 run4", [RUN4]*3), ("vs 3 heur", ["heur"]*3)]:
        for s in specs:
            st = focal_vs(s, opps, N, procs=8); sm = st.summary(); res[f"{title}|{s}"] = st.score
            print(f"{title:10s} {s:42s} score {sm['score']:.0f}+-{sm['ci']:.0f} bid {sm['bid_acc']*100:.1f}% win {sm['win']*100:.0f}% rank {sm['rank']:.2f}", flush=True)
        b = np.array(res[f"{title}|{specs[0]}"])
        for s in specs[1:]:
            d = np.array(res[f"{title}|{s}"]) - b
            print(f"  paired {s} - {specs[0]} [{title}]: {d.mean():+.1f} +- {1.96*d.std()/len(d)**.5:.1f}", flush=True)
    json.dump(res, open("eval/ei_eval.json", "w"))
