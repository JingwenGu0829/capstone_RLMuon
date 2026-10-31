# Static toy: G = S + N, S rank-k signal, N iid noise. How much of each optimizer's update lies along the true signal?
import numpy as np
rng = np.random.default_rng(0)
m, n = 256, 512

def msign(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt

def ns5(X, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    Y = X / (np.linalg.norm(X) + 1e-7)
    T = Y.shape[0] > Y.shape[1]
    if T: Y = Y.T
    for _ in range(steps):
        A = Y @ Y.T
        Y = a * Y + (b * A + c * A @ A) @ Y
    return Y.T if T else Y

def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"{'k':>4} {'SNR(F)':>7} | cos(update, S):  {'SGD':>6} {'sign':>6} {'Muon':>6} {'NS5':>6} | cos(update, msign S): {'Muon':>6}")
for k in [1, 4, 16, 64, 256]:
    for snr in [0.1, 0.3, 1.0, 3.0]:
        U = np.linalg.qr(rng.standard_normal((m, k)))[0]
        V = np.linalg.qr(rng.standard_normal((n, k)))[0]
        S = U @ np.diag(np.linspace(1, 0.3, k)) @ V.T
        S *= snr / np.linalg.norm(S)                 # ||S||_F = snr
        N = rng.standard_normal((m, n)); N /= np.linalg.norm(N)  # ||N||_F = 1
        G = S + N
        r = [cos(G, S), cos(np.sign(G), S), cos(msign(G), S), cos(ns5(G), S)]
        print(f"{k:>4} {snr:>7.1f} |                  " + " ".join(f"{x:6.3f}" for x in r) + f" |                        {cos(msign(G), msign(S)):6.3f}")
