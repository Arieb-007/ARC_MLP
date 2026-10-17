"""Conditional-coherent (CC) proxies for the random-sign kappa4 slices of the pre-activation.

Written by an LLM agent (Claude Code). Research code (round 2 of the GPU program, 2026-10-17).

Principle [hypothesis, tested here]: the MSE-optimal O(n^2) replacement of a dropped random-sign
slice X of kappa4(z) is its conditional expectation given the retained state, which to leading
order is a regression on the covariance of z with a per-neuron coefficient:
    (2,1,1):  kappa4(z)_{aabc} ~ lam211_a * C_bc,
              lam211_a = [sum_m W_am^2 b_m c_m + sum_p (W^2 B)_ap c_p] / sum_m c_m^2
    (3,1):    kappa4(z)_{aaab} ~ lam31_a * C_ab,
              lam31_a = [sum_m W_am^4 b_m c_m + 3 sum_p W_ap^2 c_p (W^2 B)_ap] / sum_m W_am^2 c_m^2
with b = kappa4(h) diagonal, c = var(h), B = the (2,2) slice of kappa4(h) (m != p), W^2 = W o W.
Derivation: the newborn part of the slice is sum_m W_am^2 (W_bm W_cm) b_m (resp. sum_m W_am^3 W_bm
b_m); regressing on C_bc(z) = sum_m W_bm W_cm c_m over the random signs of W_b, W_c gives the
coefficients above (E[W_bm W_bm'] = 2/n delta_mm'); the B terms are the paired-memory part. The
random-sign slices of kappa4(h) (not carried by a light chain) also feed these slices and are the
part a formula cannot see; the ORACLE fit (least squares of the exact slice on C) bounds what any
proxy of this form can recover. V29's lam_l * C_off for (3,1) is the fitted, per-layer-constant
version of lam31.

Transport: a kappa3 birth of the form u_a C_bc maps to (W u)_i (W C W^T)_jk under the next linear
layer, so this proxy is implementable at O(n^2) per layer in a factored chain.

CCProp: the dense Kt=4 reference with slice hooks that replace (2,1,1) and/or (3,1) by the oracle
fit or by the formula, recording per-layer R^2 of the fit; used at n <= 192 against the dense
reference. The light chain (coherent_even.CoherentEvenProp with p31/p211) applies the formula at
n = 1024.
"""
from __future__ import annotations

import numpy as np

from arcmlp.reference import dense_cumulants as dc
from arcmlp.reference.device import asnumpy, get_array_module


def lam_formula_211(W, b, c, B=None):
    xp = get_array_module(W)
    W2 = W * W
    num = W2 @ (b * c)
    if B is not None:
        num = num + (W2 @ B) @ c
    return num / xp.sum(c * c)


def lam_formula_31(W, b, c, B=None):
    xp = get_array_module(W)
    W2 = W * W
    num = (W2 * W2) @ (b * c)
    if B is not None:
        num = num + 3.0 * xp.sum((W2 * c[None, :]) * (W2 @ B), axis=1)
    den = W2 @ (c * c)
    return num / xp.maximum(den, 1e-30)


def _corr(a, b):
    xp = get_array_module(a)
    if float(xp.std(a)) == 0.0 or float(xp.std(b)) == 0.0:
        return None
    return float(xp.corrcoef(xp.stack([a, b]))[0, 1])


def oracle_211(S211, Ct):
    """Least-squares fit S[a,b,c] ~ lam_a Ct[b,c]; returns lam and the mean R^2 over a."""
    xp = get_array_module(S211)
    den = xp.sum(Ct * Ct)
    lam = xp.einsum("abc,bc->a", S211, Ct) / den
    ss = xp.einsum("abc,abc->a", S211, S211)
    r2 = 1.0 - (ss - lam * lam * den) / xp.maximum(ss, 1e-30)
    return lam, float(xp.mean(r2))


def oracle_31(S31, Ct):
    """Least-squares fit S[a,b] ~ lam_a Ct[a,b] (per row a)."""
    xp = get_array_module(S31)
    den = xp.sum(Ct * Ct, axis=1)
    lam = xp.sum(S31 * Ct, axis=1) / xp.maximum(den, 1e-30)
    ss = xp.sum(S31 * S31, axis=1)
    r2 = 1.0 - (ss - lam * lam * den) / xp.maximum(ss, 1e-30)
    return lam, float(xp.mean(r2))


RS = [(3, 1), (2, 1, 1), (1, 1, 1, 1)]


def random_sign_single_terms(reg):
    """Cost-3 terms that are a single random-sign kappa4 block (no extra covariance edges)."""
    return [t for t in reg if t.cost == 3 and len(t.diagram) == 1 and dc.slice_type(t.diagram[0]) in RS]


class CCProp(dc.DenseProp):
    """Dense Kt=4 chain with the random-sign slices replaced by CC proxies.

    mode211 / mode31 in {None, 'exact', 'oracle', 'formula', 'zero'}; mode1111 in {'exact', 'zero'};
    terms='cap3' (all cost<=3 diagrams) or 'singles' (cost<=2 plus single random-sign blocks).
    """

    def __init__(self, Ws, mode211="exact", mode31="exact", mode1111="exact", terms="cap3", device=None):
        super().__init__(Kt=4, cap=3, slice_hook=self._hook, device=device)
        self.Ws = Ws
        self.mode211, self.mode31, self.mode1111 = mode211, mode31, mode1111
        if terms == "singles":
            self.active = [t for t in self.registry if t.cost <= 2] + random_sign_single_terms(self.registry)
        self._h = None
        self.diag = []

    def relu_step(self, Z, layer=0):
        K = super().relu_step(Z, layer)
        self._h = K
        return K

    def _hook(self, layer, Z, sl):
        xp = get_array_module(Z[1])
        n = Z[1].shape[0]
        rec = dict(layer=layer)
        if self._h is None or layer == 0:
            b = xp.zeros(n, dtype=Z[1].dtype); c = xp.ones(n, dtype=Z[1].dtype); B = None
        else:
            K4, K2 = self._h[4], self._h[2]
            b = dc._diag_slice(K4, (4,)); B = dc._diag_slice(K4, (2, 2)).copy(); xp.fill_diagonal(B, 0.0)
            c = xp.diag(K2)
        W = xp.asarray(self.Ws[layer], dtype=Z[1].dtype)
        Ct = sl[(1, 1)]
        if self.mode211 != "exact":
            lam_o, r2 = oracle_211(sl[(2, 1, 1)], Ct)
            lam_f = lam_formula_211(W, b, c, B)
            rec.update(r2_211=r2, lam211_oracle_mean=float(xp.mean(lam_o)), lam211_formula_mean=float(xp.mean(lam_f)),
                       corr211=_corr(lam_o, lam_f))
            if self.mode211 == "zero":
                sl[(2, 1, 1)] = xp.zeros_like(sl[(2, 1, 1)])
            else:
                lam = lam_o if self.mode211 == "oracle" else lam_f
                sl[(2, 1, 1)] = xp.einsum("a,bc->abc", lam, Ct).astype(Z[1].dtype)
        if self.mode31 != "exact":
            lam_o, r2 = oracle_31(sl[(3, 1)], Ct)
            lam_f = lam_formula_31(W, b, c, B)
            rec.update(r2_31=r2, lam31_oracle_mean=float(xp.mean(lam_o)), lam31_formula_mean=float(xp.mean(lam_f)),
                       corr31=_corr(lam_o, lam_f))
            if self.mode31 == "zero":
                sl[(3, 1)] = xp.zeros_like(sl[(3, 1)])
            else:
                lam = lam_o if self.mode31 == "oracle" else lam_f
                sl[(3, 1)] = (lam[:, None] * Ct).astype(Z[1].dtype)
        if self.mode1111 == "zero":
            sl[(1, 1, 1, 1)] = xp.zeros_like(sl[(1, 1, 1, 1)])
        self.diag.append(rec)


CC_CONFIGS = {
    # name: (mode211, mode31, mode1111, terms)
    "K4c3": ("exact", "exact", "exact", "cap3"),
    "rs_singles": ("exact", "exact", "exact", "singles"),       # is the full cap-3 set needed?
    "cc211_oracle": ("oracle", "exact", "exact", "cap3"),
    "cc211_formula": ("formula", "exact", "exact", "cap3"),
    "cc31_oracle": ("exact", "oracle", "exact", "cap3"),
    "cc31_formula": ("exact", "formula", "exact", "cap3"),
    "cc_both_oracle": ("oracle", "oracle", "exact", "cap3"),
    "cc_both_formula": ("formula", "formula", "exact", "cap3"),
    "cc_both_formula_no1111": ("formula", "formula", "zero", "cap3"),
    "cc_both_formula_singles": ("formula", "formula", "zero", "singles"),   # what a light chain can carry
    "paired_only": ("zero", "zero", "zero", "cap3"),
}
