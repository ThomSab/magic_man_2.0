"""Expert-iteration data generation. python -m train.expert_gen OUT_PREFIX MINUTES PROCS [SAMPLES] [CKPT]
One searcher seat (net-guided PIMC) vs random opponent mix. Records per decision: obs(v2), mask, Q per action (mean round score)."""
import sys, time, random, numpy as np, torch
from multiprocessing import Process
from wizard.engine import *
from wizard.features import mask_of
from wizard.features2 import encode2


def worker(wid, prefix, minutes, samples, ckpt):
    from wizard.netsearch3 import NetSearch3
    from eval.harness import make_agent
    from wizard.features2 import load_net2
    torch.set_num_threads(1)
    net = load_net2(ckpt, "cuda")
    rng = random.Random(1000 + wid)
    O, M, Q = [], [], []
    class Rec(NetSearch3):
        def _decide(self, v, options, apply, use_voids):
            r, me = v.r, v.player
            voids = infer_voids(r) if use_voids else [set()] * N_PLAYERS
            sims, keys, assign = [], [], []
            for _ in range(self.n):
                hands = sample_hands(r, me, self.rng, voids)
                base = r.clone()
                for p, h in hands.items(): base.hands[p] = h
                asg = [0] * 4
                for o in options:
                    s = base.clone(); apply(s, o); sims.append(s); keys.append(o); assign.append(asg)
            self._rollout(sims, assign)
            tot = {o: 0.0 for o in options}
            for s, o in zip(sims, keys): tot[o] += s.scores()[me]
            q = np.full(76, np.nan, np.float32)
            for o in options:
                q[(60 + o) if r.phase == "bid" else o] = tot[o] / self.n
            O.append(encode2(r, me)); M.append(mask_of(r)); Q.append(q)
            return max(options, key=lambda o: tot[o])
    from wizard.pimc import infer_voids, sample_hands
    srch = Rec(net, samples, 77 + wid, "cuda", "self")
    pool = ["heur", "net1:checkpoints/run4_final.pt", "net2:checkpoints/v2w_w0.5.pt", "net2:checkpoints/v2w_w1.pt"]
    t0 = time.time(); g = 0
    while time.time() - t0 < minutes * 60:
        seat = rng.randrange(4)
        ag = [make_agent(rng.choice(pool), rng.randrange(10**6)) for _ in range(4)]
        ag[seat] = srch
        run_game(ag, rng.randrange(1 << 30)) if False else None
        from eval.harness import run_game
        run_game(ag, rng.randrange(1 << 30))
        g += 1
        if g % 3 == 0 and O:
            np.savez_compressed(f"{prefix}_w{wid}.npz", O=np.stack(O), M=np.stack(M), Q=np.stack(Q))
            print(f"w{wid} games={g} decisions={len(O)} t={time.time()-t0:.0f}s", flush=True)
    if O: np.savez_compressed(f"{prefix}_w{wid}.npz", O=np.stack(O), M=np.stack(M), Q=np.stack(Q))


if __name__ == "__main__":
    prefix, minutes, procs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    samples = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    ckpt = sys.argv[5] if len(sys.argv) > 5 else "checkpoints/v2w_w0.5.pt"
    ps = [Process(target=worker, args=(i, prefix, minutes, samples, ckpt)) for i in range(procs)]
    for p in ps: p.start()
    for p in ps: p.join()
