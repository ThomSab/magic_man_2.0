# Collusion check

Self table = 4 copies of one agent. 'vs heur' = same agent as one seat against 3 heuristics.

| agent | self-table mean pts/player | self bid acc | vs-heur bid acc | self mean(sum bids - tricks) | self gift rate | vs-heur gift rate | self-table spread (std of 4 totals) |
|---|---|---|---|---|---|---|---|
| heur | 122 | 0.509 | 0.503 | -2.35 | 0.000 | 0.000 | 78 |
| net1:checkpoints/run4_final.pt | 292 | 0.624 | 0.622 | -0.16 | 0.530 | 0.373 | 87 |
| net2:checkpoints/v2w_w0.5.pt | 278 | 0.615 | 0.623 | -0.26 | 0.484 | 0.364 | 83 |
| net2:checkpoints/v2w_w1.pt | 282 | 0.610 | 0.598 | -0.06 | 0.498 | 0.321 | 88 |

Reading: collusion would show as self-table bid accuracy and points per player far above what the agent gets
against independent opponents, a higher gift rate (last player had a winning card while needing tricks, declined, and the trick went to another player who needed tricks), and a small spread.
Gift rate is a noisy proxy (declining can be a legitimate card-saving play), so only compare across agents.