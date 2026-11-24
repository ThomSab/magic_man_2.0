"""Report: python -m eval.report OUT_PREFIX N_GAMES spec1 spec2 ...
Fixed opponents (never trained against except run4): heur, net1 run4, pimc.
Writes eval/<OUT>.md and eval/<OUT>_hist.png
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eval.harness import *

RUN4 = "net1:checkpoints/run4_final.pt"
BASE = ["random", "heur", RUN4, "pimc:8"]


def fmt(name, s):
    return f"| {name} | {s['n']} | {s['score']:.0f} +- {s['ci']:.0f} | {s['bid_acc']*100:.1f}% | {s['win']*100:.1f}% | {s['rank']:.2f} |"


HDR = ["| agent | games | mean score | bid accuracy | win rate | mean rank |", "|---|---|---|---|---|---|"]


def hist(ax, data, title):
    """Box plot of final game scores per agent (name kept for call sites)."""
    names = list(data.keys())
    vals = [np.array(data[n], float) for n in names]
    order = np.argsort([np.median(v) for v in vals])
    names = [names[i] for i in order]; vals = [vals[i] for i in order]
    bp = ax.boxplot(vals, vert=False, showmeans=True, patch_artist=True, widths=0.6,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#8fb8e8"); patch.set_alpha(0.8)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels([f"{n.split('/')[-1].replace('checkpoints/', '')} (n={len(v)})" for n, v in zip(names, vals)], fontsize=7)
    ax.set_title(title); ax.set_xlabel("final game score (box = quartiles, line = median, diamond = mean)")
    ax.grid(axis="x", alpha=0.3)

if __name__ == "__main__":
    out, N = sys.argv[1], int(sys.argv[2])
    cands = sys.argv[3:]
    agents = []
    for s in BASE + cands:
        if s not in agents: agents.append(s)
    md = [f"# Evaluation report ({out})", "",
          f"{N} deals x 4 seat rotations per cell (same deals for every agent). Win rate = share of games finishing first (ties split). Bid accuracy = % of rounds where tricks won == bid.", ""]
    raw = {}
    for title, opps, n in [("vs 3 heuristics", ["heur"] * 3, N), ("vs 3 run4 nets", [RUN4] * 3, N),
                           ("vs 3 PIMC searchers", ["pimc:8"] * 3, max(8, N // 4))]:
        md += [f"## {title}", ""] + HDR
        raw[title] = {}
        for a in agents:
            if a.startswith("pimc") and "PIMC" not in title and False: continue
            st = focal_vs(a, opps, n)
            raw[title][a] = st.score
            md.append(fmt(a, st.summary())); print(title, a, st.summary(), flush=True)
        md.append("")
    md += ["## Mixed table: agent vs [heuristic, run4, PIMC] (one seat each; focal replaces the matching seat by playing its own role)", ""]
    # mixed: each candidate sits with heur, run4, pimc. Baselines play the 4-way table heur/run4/pimc/random.
    md += HDR
    raw["mixed"] = {}
    for a in cands:
        others = ["heur", RUN4, "pimc:8"]
        res = mixed_table([a] + others, max(8, N // 2))
        st = res[a]; raw["mixed"][a] = st.score
        md.append(fmt(a, st.summary()))
        for o in others:
            md.append(fmt("&nbsp;&nbsp;(opp) " + o, res[o].summary()))
        print("mixed", a, st.summary(), flush=True)
    md.append("")
    import json; json.dump({k: {a: [float(x) for x in v] for a, v in d.items()} for k, d in raw.items()}, open(f"eval/{out}_raw.json", "w"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    hist(axes[0][0], raw["vs 3 heuristics"], "Final score distribution: agent vs 3 heuristics")
    hist(axes[0][1], raw["vs 3 run4 nets"], "agent vs 3 run4 nets")
    hist(axes[1][0], raw["vs 3 PIMC searchers"], "agent vs 3 PIMC searchers")
    if raw["mixed"]: hist(axes[1][1], raw["mixed"], "candidate in mixed table [cand, heur, run4, PIMC]")
    plt.tight_layout(); plt.savefig(f"eval/{out}_hist.png", dpi=110)
    md += [f"![score box plots]({out}_hist.png)"]
    open(f"eval/{out}.md", "w").write("\n".join(md))



