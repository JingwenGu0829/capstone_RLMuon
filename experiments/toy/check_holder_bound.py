# Check: cos(msign(S+N), S) <= ||S||_* / (||S||_F * sqrt(rank(S+N)))  -- holds for any S, N (Holder duality)
import numpy as np
rng = np.random.default_rng(1); worst = 0
for _ in range(2000):
    m, n = rng.integers(8, 64, 2); k = rng.integers(1, min(m, n) + 1)
    S = rng.standard_normal((m, k)) @ rng.standard_normal((k, n)) * rng.exponential()
    N = rng.standard_normal((m, n)) * rng.exponential() * rng.choice([1e-3, 1e-1, 1, 10])
    G = S + N; U, s, Vt = np.linalg.svd(G, full_matrices=False); r = int(np.sum(s > s[0] * 1e-10))
    Q = U[:, :r] @ Vt[:r]
    c = (Q * S).sum() / (np.linalg.norm(Q) * np.linalg.norm(S))
    cap = np.linalg.svd(S, compute_uv=False).sum() / (np.linalg.norm(S) * np.sqrt(r))
    worst = max(worst, c - cap)
print("max violation (should be <= 0):", worst)
