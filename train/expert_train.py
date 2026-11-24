"""Distil search into the net. python -m train.expert_train OUT_CKPT INIT_CKPT TAU EPOCHS shard_glob...
Target policy = softmax(Q / TAU) over legal actions (Q = mean round score from search)."""
import sys, glob, numpy as np, torch
from wizard.features import Net
from wizard.features2 import OBS2_DIM

dev = torch.device("cuda")


def load(globs):
    O, M, Q = [], [], []
    for g in globs:
        for f in sorted(glob.glob(g)):
            d = np.load(f); O.append(d["O"]); M.append(d["M"]); Q.append(d["Q"])
    return np.concatenate(O), np.concatenate(M), np.concatenate(Q)


if __name__ == "__main__":
    out, init, tau, epochs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    O, M, Q = load(sys.argv[5:])
    n = len(O); print("decisions:", n)
    Qf = np.where(np.isnan(Q), -1e9, Q)
    Qt = torch.tensor(Qf / tau); Mt = torch.tensor(M)
    T = torch.softmax(Qt, -1) * Mt
    T = (T / T.sum(-1, keepdim=True)).to(dev)
    best = torch.tensor(np.argmax(Qf, -1)).to(dev)
    O = torch.tensor(O).to(dev); Mt = Mt.to(dev)
    rng = np.random.RandomState(0); perm = rng.permutation(n)
    nv = max(200, n // 10); vi = torch.tensor(perm[:nv]).to(dev); ti = torch.tensor(perm[nv:]).to(dev)
    net = Net(in_dim=OBS2_DIM).to(dev); net.load_state_dict(torch.load(init, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    def evaluate():
        net.eval()
        with torch.no_grad():
            lg, _ = net(O[vi], Mt[vi])
            lp = torch.log_softmax(lg, -1)
            ce = -(T[vi] * lp.clamp(min=-30)).sum(-1).mean().item()
            agree = (lg.argmax(-1) == best[vi]).float().mean().item()
        net.train(); return ce, agree

    print("before: val CE %.3f, top1 agreement with search %.3f" % evaluate())
    for ep in range(epochs):
        p = ti[torch.randperm(len(ti), device=dev)]
        for s in range(0, len(p), 512):
            j = p[s:s + 512]
            lg, _ = net(O[j], Mt[j])
            loss = -(T[j] * torch.log_softmax(lg, -1).clamp(min=-30)).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        print("epoch %d: val CE %.3f, top1 agreement %.3f" % ((ep + 1,) + evaluate()), flush=True)
    torch.save(net.state_dict(), out)
