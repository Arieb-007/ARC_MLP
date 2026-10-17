"""Algorithm E (round 2 add-on): spiked-covariance / anchor-neuron closure and its structure
diagnostics. Written by an LLM agent (Claude Code) from Areeb's suggestions of 2026-10-17.

Premise under test [hypothesis]: the post-ReLU covariance C_l splits into a low-rank "spike" part
and a diagonal bulk, C ~ L L^T + D with rank r << n (Johnstone's spiked model), so that (i) the
covariance transport costs 2 r n^2 instead of 2 n^3 and (ii) k anchor neurons predict the rest by
Gaussian conditioning (pivoted Cholesky / Schur complement), the residual being nearly Gaussian.

SpikedCovProp: the light chain (K=3 dense + coherent kappa4) with C_post replaced after every ReLU
step by diag(C) + the top-r eigen-part of its off-diagonal (r = 0 is the independent-neuron
closure). Its MSE against the full chain is the error the cheaper algorithm would make.

StructureDiag: per layer, the fraction of the off-diagonal Frobenius energy of C in its top r
eigenvectors, the median variance explained by k pivoted-Cholesky anchors (Schur complement), and
the same spectral fractions for the kappa3 (2,1) matrix D21 and the kappa4 (2,2) memory B.
"""
from __future__ import annotations

import numpy as np

from arcmlp.estimators.coherent_even import CoherentEvenProp
from arcmlp.reference import dense_cumulants as dc
from arcmlp.reference.device import asnumpy, get_array_module

RANKS = (16, 64, 256)
ANCHORS = (64, 256)


def offdiag(M):
    xp = get_array_module(M)
    O = M.copy()
    xp.fill_diagonal(O, 0.0)
    return O


def spectral_fraction(M, ranks=RANKS):
    """Fraction of ||offdiag(M)||_F^2 carried by the top-r eigenpairs (by |eigenvalue|) of the
    symmetrized off-diagonal part."""
    xp = get_array_module(M)
    O = offdiag(M)
    O = 0.5 * (O + O.T)
    w = xp.linalg.eigvalsh(O.astype(xp.float64))
    w2 = xp.sort(w * w)[::-1]
    tot = float(xp.sum(w2)) + 1e-300
    return {r: float(xp.sum(w2[:r])) / tot for r in ranks if r < w2.size}


def anchor_r2(C, k):
    """Greedy pivoted Cholesky with k anchors on the covariance C: returns the median over the
    non-anchor neurons of the variance fraction explained by the anchors (1 - Schur / var)."""
    xp = get_array_module(C)
    n = C.shape[0]
    C = C.astype(xp.float64)
    d = xp.diag(C).copy()
    var0 = d.copy()
    Lcols = []
    chosen = []
    for _ in range(min(k, n)):
        i = int(xp.argmax(d))
        if float(d[i]) <= 1e-12:
            break
        col = C[:, i].copy()
        for Lc in Lcols:
            col -= Lc[i] * Lc
        col = col / xp.sqrt(d[i])
        Lcols.append(col)
        chosen.append(i)
        d = xp.maximum(d - col * col, 0.0)
        d[i] = 0.0
    mask = xp.ones(n, dtype=bool)
    mask[xp.asarray(chosen)] = False
    explained = xp.clip(1.0 - d[mask] / xp.maximum(var0[mask], 1e-30), 0.0, 1.0)
    return float(xp.median(explained)), float(xp.mean(explained))


def spike_project(C, r):
    """diag(C) + top-r eigen-part (by |eigenvalue|) of the off-diagonal of C."""
    xp = get_array_module(C)
    dt = C.dtype
    if r <= 0:
        return xp.diag(xp.diag(C)).astype(dt)
    O = offdiag(C)
    O = 0.5 * (O + O.T)
    w, V = xp.linalg.eigh(O.astype(xp.float64))
    idx = xp.argsort(-xp.abs(w))[:r]
    Lr = V[:, idx] * w[idx][None, :]
    S = Lr @ V[:, idx].T
    xp.fill_diagonal(S, 0.0)
    S = S + xp.diag(xp.diag(C).astype(xp.float64))
    return S.astype(dt)


class SpikedCovProp(CoherentEvenProp):
    def __init__(self, rank=64, mode="coherent", cap=2, device=None, **kw):
        super().__init__(mode=mode, cap=cap, device=device, **kw)
        self.rank = rank

    def relu_light(self, Z, layer):
        S = super().relu_light(Z, layer)
        S[2] = spike_project(S[2], self.rank)
        return S


class StructureDiag(CoherentEvenProp):
    """Light chain that records the structure diagnostics of its post-ReLU state at every layer."""

    def __init__(self, mode="coherent", cap=2, device=None, ranks=RANKS, anchors=ANCHORS, **kw):
        super().__init__(mode=mode, cap=cap, device=device, **kw)
        self.ranks, self.anchors = ranks, anchors
        self.structure = []

    def relu_light(self, Z, layer):
        S = super().relu_light(Z, layer)
        xp = get_array_module(S[2])
        C = S[2]
        O = offdiag(C)
        rec = dict(layer=layer,
                   offdiag_over_diag_energy=float(xp.sum(O * O) / xp.sum(xp.diag(C) ** 2)),
                   cov_spectral=spectral_fraction(C, self.ranks),
                   anchor_r2={k: anchor_r2(C, k) for k in self.anchors})
        D21 = dc._diag_slice(S[3], (2, 1))
        rec["d21_spectral"] = spectral_fraction(D21, self.ranks)
        rec["B_spectral"] = spectral_fraction(S["B"], self.ranks)
        self.structure.append(rec)
        return S
