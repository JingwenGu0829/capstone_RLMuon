# Pilot toy: contextual bandit "LM head". Teacher = pretrained head + rank-k change (k = signal rank).
# RL (GRPO, binary reward) vs SFT (CE on teacher label); optimizers SGD / Adam / Muon(NS5) / Muon(exact polar) / Muon(top-8).
import numpy as np, sys
V, d = 256, 128

def world(seed, k):
    r = np.random.default_rng(seed)
    W0 = r.standard_normal((V, d)) / np.sqrt(d)
    D = r.standard_normal((V, k)) @ r.standard_normal((k, d)); D *= 1.2 * np.linalg.norm(W0) / np.linalg.norm(D)
    scale = np.sqrt(d) * np.arange(1, d + 1) ** -0.5; scale *= 3.0 * np.sqrt(d) / np.linalg.norm(scale)  # anisotropic h
    return W0, W0 + D, scale

def softmax(Z): Z = Z - Z.max(-1, keepdims=True); E = np.exp(Z); return E / E.sum(-1, keepdims=True)

def ns5(X):
    a, b, c = 3.4445, -4.7750, 2.0315; Y = X / (np.linalg.norm(X) + 1e-7); tr = Y.shape[0] > Y.shape[1]
    if tr: Y = Y.T
    for _ in range(5): A = Y @ Y.T; Y = a * Y + (b * A + c * A @ A) @ Y
    return Y.T if tr else Y
def polar(X, topk=None):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if topk: U, Vt = U[:, :topk], Vt[:topk]
    return U @ Vt

class Opt:
    def __init__(s, kind, lr): s.kind, s.lr, s.m, s.v, s.t = kind, lr, 0, 0, 0
    def step(s, W, g):
        s.t += 1
        if s.kind == "sgd": s.m = 0.9 * s.m + g; return W - s.lr * s.m
        if s.kind == "adam":
            s.m = 0.9 * s.m + 0.1 * g; s.v = 0.999 * s.v + 0.001 * g * g
            return W - s.lr * (s.m / (1 - 0.9 ** s.t)) / (np.sqrt(s.v / (1 - 0.999 ** s.t)) + 1e-8)
        s.m = 0.95 * s.m + g; u = g + 0.95 * s.m
        o = {"muon": ns5, "muon_exact": polar, "muon_top8": lambda X: polar(X, 8)}[s.kind](u)
        return W - s.lr * 0.2 * np.sqrt(max(W.shape)) * o                 # Moonlight update-RMS matching

def acc(W, Wt, H, W0=None):
    y = (H @ Wt.T).argmax(-1); P = softmax(H @ W.T); a = P[np.arange(len(y)), y].mean()
    if W0 is None: return a
    keep = y == (H @ W0.T).argmax(-1); P0 = softmax(H[keep] @ W0.T)
    return a, float((P0 * (np.log(P0 + 1e-12) - np.log(P[keep] + 1e-12))).sum(-1).mean())

def run(kind, lr, mode, seed, k, steps=300, B=16, G=8):
    W0, Wt, sc = world(seed, k); W = W0.copy(); r = np.random.default_rng(seed + 7); opt = Opt(kind, lr)
    Hte = np.random.default_rng(seed + 99).standard_normal((2000, d)) * sc
    for _ in range(steps):
        H = r.standard_normal((B, d)) * sc; y = (H @ Wt.T).argmax(-1); P = softmax(H @ W.T)
        if mode == "rl":
            Pg, Hg, yg = np.repeat(P, G, 0), np.repeat(H, G, 0), np.repeat(y, G)
            a = np.minimum((r.random((B * G, 1)) > Pg.cumsum(-1)).sum(-1), V - 1)
            rew = (a == yg).astype(float).reshape(B, G); A = (rew - rew.mean(1, keepdims=True)).reshape(-1)
            R = -Pg; R[np.arange(B * G), a] += 1; g = -(R * A[:, None]).T @ Hg / (B * G)
        else:
            R = -P; R[np.arange(B), y] += 1; g = -R.T @ H / B
        W = opt.step(W, g)
        if not np.isfinite(W).all(): return (0.0, np.nan)
    return acc(W, Wt, Hte, W0)

# usage: python toy_rank_teacher.py {rl|sft} <signal_rank k>
# prints best-LR accuracy and off-task drift KL (KL(pi_0||pi) on contexts the teacher did not change)
if __name__ == "__main__":
    mode, k = sys.argv[1], int(sys.argv[2])
    W0, Wt, sc = world(0, k); Hte = np.random.default_rng(99).standard_normal((2000, d)) * sc
    out = [f"[{mode} k={k}] init acc={acc(W0, Wt, Hte):.3f}"]
    grids = {"sgd": [0.1, 0.3, 1, 3, 10], **{o: [3e-4, 1e-3, 3e-3, 1e-2, 3e-2] for o in ["adam", "muon", "muon_exact", "muon_top8"]}}
    for kind, lrs in grids.items():
        res = {lr: np.mean([run(kind, lr, mode, s, k) for s in range(2)], 0) for lr in lrs}
        lr = max(res, key=lambda l: res[l][0])
        out.append(f"  {kind:10s} best lr={lr:<7g} acc={res[lr][0]:.3f} drift_KL={res[lr][1]:.3f} | (acc,KL) per lr: " + " ".join(f"({v[0]:.2f},{v[1]:.2f})" for v in res.values()))
    print("\n".join(out), flush=True)
