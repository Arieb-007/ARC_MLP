"""E4c: memory + newborn recursion for lambda_l vs Monte-Carlo-exact lambda_l (n=128 arrays from E4).

Recursion [derived at leading order, hypothesis until this test]:
  lambda_l = 2 wbar_l * [lambda_{l-1} ||C_{l-1,off}||^2 + sum_m g_{l-1,m} v_{l-1,m}] / ||C_{l-1}||_F^2
             + lambda_newborn,l
  w_k = Phi_k - mu_k phi_k / s_k (Gaussian ReLU stats of z_l), wbar = mean_k w_k,
  g = diagonal of the r=1 core (c_A PT_ii + c_I tr PT), v = diag C_{l-1}.
Two variants: (i) one-step: lambda_{l-1}, g_{l-1} from MC (tests the transport law per layer);
(ii) closed loop: lambda_{l-1} from the recursion itself, g from MC (tests error accumulation).
Falsifier: one-step ratio outside [0.8, 1.25] at more than a third of the layers.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.closure.kappa4_lambda import lambda_tree, relu_stats
for path in sys.argv[1:]:
    z = np.load(path); mu, C, PT, Ws = z["mu"], z["C"], z["PT"], z["Ws"]
    L, n = mu.shape; cA = 6/(n+4); cI = -3/((n+2)*(n+4)); off = ~np.eye(n, dtype=bool)
    lam_mc = [cA*float(PT[l][off] @ C[l][off] / (C[l][off] @ C[l][off])) for l in range(L)]
    g = [cA*np.diag(PT[l]) + cI*np.trace(PT[l]) for l in range(L)]
    print(f"\n{path}\n l | lam_MC   | newborn  | 2wbar | one-step  ratio | closed-loop ratio")
    lam_cl = None; rows = []
    for l in range(L):
        if l == 0:
            m = np.zeros(n); Cp = Ws[0] @ Ws[0].T
        else:
            m = Ws[l] @ mu[l-1]; Cp = Ws[l] @ C[l-1] @ Ws[l].T
        s = np.sqrt(np.diag(Cp)); st = relu_stats(m, s)
        w = st["Phi"] - st["E1"]*st["phi"]/s; wbar = float(w.mean())
        nb = lambda_tree(m, Cp, C[l])["total"]
        if l == 0:
            one = nb; lam_cl = nb
        else:
            Cq = C[l-1]; v = np.diag(Cq); F2 = float(np.sum(Cq*Cq)); off2 = F2 - float(v @ v)
            trans = lambda lp: 2*wbar*(lp*off2 + float(g[l-1] @ v))/F2
            one = trans(lam_mc[l-1]) + nb
            lam_cl = trans(lam_cl) + nb
        rows.append(dict(layer=l, lam_mc=lam_mc[l], newborn=nb, two_wbar=2*wbar, one_step=one, closed_loop=lam_cl))
        print(f"{l:2d} | {lam_mc[l]:.3e} | {nb:+.2e} | {2*wbar:.3f} | {one:.3e} {one/lam_mc[l]:5.2f} | {lam_cl:.3e} {lam_cl/lam_mc[l]:5.2f}")
    print(json.dumps(rows))
