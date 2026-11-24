"""Collusion check. python -m eval.collusion N  -> writes eval/collusion.md"""
import sys, json
from eval.harness import *

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    specs = sys.argv[2:] or ["heur", "random", "net1:checkpoints/run4.pt"]
    rows = []
    for s in specs:
        t = self_table(s, N)
        f = focal_vs(s, ["heur"] * 3, N // 2)
        fs = f.summary()
        rows.append((s, t, fs))
        print(s, t, fs, flush=True)
    out = ["# Collusion check", "",
           "Self table = 4 copies of one agent. 'vs heur' = same agent as one seat against 3 heuristics.", "",
           "| agent | self-table mean pts/player | self bid acc | vs-heur bid acc | self mean(sum bids - tricks) | self gift rate | vs-heur gift rate | self-table spread (std of 4 totals) |",
           "|---|---|---|---|---|---|---|---|"]
    for s, t, fs in rows:
        out.append(f"| {s} | {t['mean_player_score']:.0f} | {t['bid_acc']:.3f} | {fs['bid_acc']:.3f} | {t['mean_bid_dev']:+.2f} | {t['gift_rate']:.3f} | {fs['gift']:.3f} | {t['spread']:.0f} |")
    out += ["", "Reading: collusion would show as self-table bid accuracy and points per player far above what the agent gets",
            "against independent opponents, a higher gift rate (last player had a winning card while needing tricks, declined, and the trick went to another player who needed tricks), and a small spread.",
            "Gift rate is a noisy proxy (declining can be a legitimate card-saving play), so only compare across agents."]
    open("eval/collusion.md", "w").write("\n".join(out))
