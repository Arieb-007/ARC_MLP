"""W7 step 3b: per-layer coherent kappa4 constants at n = 1024 vs V29's fitted lambda_l table.

On the self-baked 1024x16 MLPs (data/bake_1024x16_s0-3_n2e4, seeds 0-3) run the K=2 closure and, per
post-ReLU layer, the newborn (Gaussian-input) quantities from the diagram rule restricted to
covariance blocks (cap 2):
  b_m   = kappa4(h_m)                  ((4,) slice, closed form + c-diagrams on one neuron = none)
  B_ab  = kappa(h_a, h_a, h_b, h_b)    ((2,2) slice: c_ab^2 diagrams, tree level)
  beta  = 4 bbar/n + 4 Bbar            (coherent (2,2) value of the next pre-activation)
and compare with V29's LAM_l * 0.95 (the regenerated-core coefficient; FINDINGS F3.2 / F8 explain
why these are different objects: V29's lambda multiplies C_off in the partial-trace core).
Falsifier for "beta is V29's lambda": the ratio LAM/beta (or LAM/(beta/4)) varies by > 2x across layers.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.closure.relu_moments import relu_cumulants_all  # noqa: E402
from arcmlp.eval.local_eval import provenance  # noqa: E402

LAM = [4.9895e-03, 8.0876e-03, 9.8291e-03, 1.0549e-02, 1.0851e-02, 1.0828e-02, 1.0589e-02, 1.0048e-02,
       9.6483e-03, 9.1720e-03, 8.7730e-03, 8.3938e-03, 8.0555e-03, 7.6770e-03, 7.2588e-03, 7.2588e-03]
LAM = [0.95 * x for x in LAM]


def newborn_b_B(m, C):
    """Per-neuron kappa4 and the (2,2) slice of h = ReLU(z), z Gaussian (m, C), diagram rule with
    covariance blocks only at cap 2."""
    n = m.size
    s = np.sqrt(np.diag(C))
    Ct = C.copy(); np.fill_diagonal(Ct, 0.0)
    w = lambda k, p: dc.wick_vec(m, s, k, p)
    pK = {}
    for p in [(1,), (2,), (3,), (4,), (1, 1), (2, 1), (2, 2)]:
        r = len(p)
        acc = np.zeros((n,) * r)
        for rep, cnt in dc.enumerate_terms(p, 2, 2):
            ops, subs, legs = [], [], [0] * r
            for v in rep:
                verts = [a for a in range(r) if v[a]]
                ops.append(Ct); subs.append("".join(dc.LETTERS[a] for a in verts))
                for a in range(r):
                    legs[a] += v[a]
            for a in range(r):
                ops.append(w(legs[a], p[a])); subs.append(dc.LETTERS[a])
            acc += cnt * dc.vec_part_coef(rep) * np.einsum(",".join(subs) + "->" + dc.LETTERS[:r], *ops)
        pK[p] = dc._symmetrize(acc, p) if r > 1 else acc
    out = {}
    for q in [(4,), (2, 2)]:
        r = len(q)
        acc = np.zeros((n,) * r)
        for coef, mono in dc.pk_to_k_formula(q):
            ops, subs = [], []
            for verts, powv in mono:
                pw = tuple(sorted(powv, reverse=True))
                order = sorted(range(len(verts)), key=lambda i: -powv[i])
                ops.append(pK[pw]); subs.append("".join(dc.LETTERS[verts[i]] for i in order))
            acc += coef * np.einsum(",".join(subs) + "->" + dc.LETTERS[:r], *ops)
        out[q] = acc
    B = out[(2, 2)]; np.fill_diagonal(B, 0.0)
    return out[(4,)], B


def main():
    from whestbench.dataset import load_dataset, iter_mlps
    ds = load_dataset("data/bake_1024x16_s0-3_n2e4", split="dev")
    rows = []
    for i, mlp in enumerate(iter_mlps(ds)):
        Ws = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
        n = Ws[0].shape[0]
        mu, S = np.zeros(n), np.eye(n)
        per = []
        for l, W in enumerate(Ws):
            m = W @ mu; C = W @ S @ W.T
            b, B = newborn_b_B(m, C)
            bbar = float(b.mean()); Bbar = float(B.sum() / n ** 2)
            beta = 4 * bbar / n + 4 * Bbar
            s = np.sqrt(np.diag(C)); t = m / s
            per.append(dict(layer=l, bbar=bbar, Bbar=Bbar, beta=beta, frac_b_neg=float((b < 0).mean()),
                            lam_v29=LAM[l], ratio_lam_over_beta=LAM[l] / beta, ratio_lam_over_quarter_beta=LAM[l] / (beta / 4)))
            # K=2 closure update
            from scipy.special import ndtr
            P = ndtr(t); ph = np.exp(-t * t / 2) / np.sqrt(2 * np.pi)
            mu = m * P + s * ph; e2 = (m * m + s * s) * P + m * s * ph
            S = np.outer(P, P) * C; np.fill_diagonal(S, e2 - mu * mu)
        rows.append(dict(mlp=i, per_layer=per))
        print(f"MLP {i}: " + " ".join(f"{r['ratio_lam_over_beta']:.2f}" for r in per), flush=True)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    json.dump(dict(rows=rows, provenance=provenance(".")), open(f"data/w7_v29_compare.json", "w"), indent=1)
    lay = len(rows[0]["per_layer"])
    mean = lambda k: [float(np.mean([r["per_layer"][l][k] for r in rows])) for l in range(lay)]
    print(json.dumps(dict(bbar=mean("bbar"), Bbar=mean("Bbar"), beta=mean("beta"), frac_b_neg=mean("frac_b_neg"),
                          lam=LAM, ratio=mean("ratio_lam_over_beta"))))


if __name__ == "__main__":
    main()
