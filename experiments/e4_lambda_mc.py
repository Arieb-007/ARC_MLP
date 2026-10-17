"""E4: ground-truth kappa4 partial-trace core vs the post-ReLU covariance, per layer (Stage 1).

Question: is the off-diagonal kappa4 core that V29 regenerates, G_ij = c_A sum_k kappa4(h)_{ijkk}
(c_A = 6/(n+4); 504aldo lean_k3_aug.py:1262), proportional to C_post,ij, and with what lambda_l?
Falsifier (stated before running): if the true per-layer LS coefficient
lambda*_l = <G_off, C_off> / ||C_off||^2 has R^2 < 0.5 at most layers, or has no stable value
across seeds (> 20% spread), a one-scalar-per-layer law is not a property of the true kappa4 and
no analytic lambda can be justified from it.

Exact identity used (a = h - E h, C = Cov h, T = tr C):
    sum_k kappa4_{ijkk} = E[a_i a_j (|a|^2 - T)] - 2 (C^2)_ij
(the centred weight |a|^2 - T cuts the MC variance by ~n compared with E[a_i a_j |a|^2] - C_ij T).
Two passes per network: pass 1 means and tr C per layer; pass 2 the centred moments.
Also records per-layer pre-activation stats (mean m, std s from the exact linear push of the MC
covariance) so analytic candidates can be evaluated on the same networks (e4b).

    python experiments/e4_lambda_mc.py --n 128 --L 16 --seeds 0 1 --N 1e7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(n, L, seed, N, B=100_000, mc_seed=None):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]
    WT = [W.T.copy() for W in Ws]
    mrng = np.random.default_rng(10_000 + seed if mc_seed is None else mc_seed)
    nb = int(N // B)
    # pass 1: means and second moments (for tr C and C)
    s1 = np.zeros((L, n)); s2 = np.zeros((L, n, n))
    for _ in range(nb):
        h = mrng.standard_normal((B, n))
        for l in range(L):
            h = np.maximum(h @ WT[l], 0)
            s1[l] += h.sum(0); s2[l] += h.T @ h
    mu = s1 / (nb * B)
    C1 = s2 / (nb * B) - mu[:, :, None] * mu[:, None, :]
    T = np.trace(C1, axis1=1, axis2=2)
    # pass 2: centred moments with fresh samples
    M1 = np.zeros((L, n, n)); M2 = np.zeros((L, n, n)); m1 = np.zeros((L, n))
    for _ in range(nb):
        h = mrng.standard_normal((B, n))
        for l in range(L):
            h = np.maximum(h @ WT[l], 0)
            a = h - mu[l]
            w = (a * a).sum(1) - T[l]
            m1[l] += a.sum(0)
            M1[l] += a.T @ a
            M2[l] += (a * w[:, None]).T @ a
    N2 = nb * B
    abar = m1 / N2                                      # residual mean of a (~1/sqrt(N)), corrected below
    C = M1 / N2 - abar[:, :, None] * abar[:, None, :]
    E3 = M2 / N2                                        # E[a_i a_j (|a|^2 - T)] (bias O(1/N), ignored)
    out = []
    cA = 6.0 / (n + 4)
    for l in range(L):
        PT = E3[l] - 2 * C[l] @ C[l]
        G = cA * PT
        off = ~np.eye(n, dtype=bool)
        g, c = G[off], C[l][off]
        lam = float(g @ c / (c @ c))
        r2 = float(1 - np.sum((g - lam * c) ** 2) / np.sum(g ** 2))
        # pre-activation stats of layer l (z_l = W_l h_{l-1}); h_{-1} = x
        if l == 0:
            m_pre = np.zeros(n); C_pre = Ws[0] @ Ws[0].T
        else:
            m_pre = Ws[l] @ mu[l - 1]; C_pre = Ws[l] @ C[l - 1] @ Ws[l].T
        # pre-activation fit of the NEXT layer, as V29's LAM table was fitted:
        # G_pre(l+1) = W_{l+1} G W_{l+1}^T with G = cA PT + cI tr(PT) I (diag of the r=1 core)
        lam_pre = r2_pre = None
        if l + 1 < L:
            cI = -3.0 / ((n + 2) * (n + 4))
            Gfull = G + cI * np.trace(PT) * np.eye(n)
            Gp = Ws[l + 1] @ Gfull @ Ws[l + 1].T
            Cp = Ws[l + 1] @ C[l] @ Ws[l + 1].T
            gp, cp = Gp[off], Cp[off]
            lam_pre = float(gp @ cp / (cp @ cp))
            r2_pre = float(1 - np.sum((gp - lam_pre * cp) ** 2) / np.sum(gp ** 2))
        out.append(dict(layer=l, lam_post=lam, r2_post=r2, lam_pre_next=lam_pre, r2_pre_next=r2_pre,
                        mean_var=float(np.mean(np.diag(C[l]))), trC=float(T[l]),
                        mean_dG=float(np.mean(np.diag(G))),
                        frac_alive=float(np.mean(mu[l] > 1e-3))))
    return out, dict(Ws=Ws, mu=mu, C=C, PT=[E3[l] - 2 * C[l] @ C[l] for l in range(L)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--N", type=float, default=1e7)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    res = {}
    for sd in a.seeds:
        t0 = time.time()
        rows, arrs = run(a.n, a.L, sd, int(a.N))
        res[sd] = rows
        if a.save:
            np.savez_compressed(f"{a.save}_s{sd}.npz", mu=arrs["mu"], C=arrs["C"], PT=np.array(arrs["PT"]),
                                Ws=np.array(arrs["Ws"]))
        print(f"seed {sd} ({time.time()-t0:.0f}s)  n={a.n} L={a.L} N={a.N:.0e}")
        print(" l | lam_post  R2_post | lam_pre(l+1) R2_pre | mean var | mean dG")
        for r in rows:
            lp = "   -    " if r["lam_pre_next"] is None else f"{r['lam_pre_next']:.4e}"
            rp = "  -  " if r["r2_pre_next"] is None else f"{r['r2_pre_next']:.3f}"
            print(f"{r['layer']:2d} | {r['lam_post']:.4e} {r['r2_post']:6.3f} | {lp} {rp} | {r['mean_var']:.4f} | {r['mean_dG']:.3e}")
    print(json.dumps({"n": a.n, "L": a.L, "N": a.N, "rows": res}))
