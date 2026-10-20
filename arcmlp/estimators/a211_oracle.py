"""Round 5 oracle (docs/notes/r5_cp_memory_theorem.md): the dense K=4 cap-3 chain with the (1,1,1,1) bulk
dropped and the (2,1,1) input slice A[j, k, l] = kappa4(z)[j, j, k, l] projected, after every linear step,
onto the top-r left singular directions of its (j | kl) unfolding. Records per layer the (j | kl) spectrum
of A: energy fractions of the top-1 / top-r directions, k50, k90, and the fraction along the all-ones
j-direction (the partial-trace channel of Proposition B).

Written by an LLM agent (Claude Code) for team Arieb-007, 2026-10-20.
"""
from __future__ import annotations

import numpy as np

from arcmlp.reference import dense_cumulants as dc
from arcmlp.reference.device import asnumpy, free_pool, get_array_module, resolve_device


def jkl_spectrum(A, r):
    """Eigen-decomposition of the n x n Gram matrix of the (j | kl) unfolding of A (n, n, n).
    Returns (eigenvalues descending, eigenvectors as columns, total energy)."""
    xp = get_array_module(A)
    n = A.shape[0]
    M = A.reshape(n, n * n)
    G = M @ M.T
    w, V = xp.linalg.eigh(G)
    w = w[::-1]
    V = V[:, ::-1]
    return w, V, float(xp.sum(w))


class A211Oracle:
    """keep_bulk=False drops the (1,1,1,1) feature (as K4c3-no1111); rank=None keeps A exact (spectra only)."""

    def __init__(self, rank=None, keep_bulk=False, device=None):
        from w5_selection import FEATURES, gates   # experiments/ is on sys.path in the GPU scripts
        self.rank, self.keep_bulk = rank, keep_bulk
        self.device = resolve_device(device)
        feats = set(FEATURES["dense"])
        if not keep_bulk:
            feats = feats - {"K4(1,1,1,1)"}
        self.prop = dc.DenseProp(Kt=4, cap=3, device=self.device, slice_hook=self._hook)
        self.prop.active = [t for t in self.prop.active if gates(t) <= feats]
        self.diag = []

    def _hook(self, layer, Z, sl):
        A = sl.get((2, 1, 1))
        if A is None:
            return
        xp = get_array_module(A)
        n = A.shape[0]
        w, V, tot = jkl_spectrum(A, self.rank)
        wn = asnumpy(w).astype(np.float64)
        e = np.cumsum(np.maximum(wn, 0)) / max(tot, 1e-300)
        ones = xp.ones(n, dtype=A.dtype) / np.sqrt(n)
        P = xp.tensordot(ones, A, axes=([0], [0]))            # partial-trace direction (all-ones j)
        f_ones = float(xp.sum(P * P)) / max(tot, 1e-300)
        rank_at = lambda f: int(np.searchsorted(e, f) + 1) if e[-1] >= f else None
        self.diag.append(dict(layer=layer, f1=float(e[0]), f4=float(e[min(3, n - 1)]), f16=float(e[min(15, n - 1)]),
                              f64=float(e[min(63, n - 1)]), k50=rank_at(0.5), k90=rank_at(0.9), f_ones=f_ones,
                              energy=tot))
        if self.rank is not None and self.rank < n:
            Vr = V[:, :self.rank]
            M = A.reshape(n, n * n)
            Ar = (Vr @ (Vr.T @ M)).reshape(n, n, n)
            # the array also holds the coincident entries (k = j, l = j, k = l: the (3,1), (2,2) and (4,)
            # information); restore them so that only the genuinely (2,1,1) part is truncated
            idx = xp.arange(n)
            Ar[idx, idx, :] = A[idx, idx, :]
            Ar[idx, :, idx] = A[idx, :, idx]
            Ar[:, idx, idx] = A[:, idx, idx]
            sl[(2, 1, 1)] = Ar
            del Ar, M
        del P, V, w
        free_pool()

    def run(self, Ws):
        self.diag = []
        return self.prop.run(Ws)


def parse_a211(cfg):
    """'a211:<r>' | 'a211:<r>:bulk' | 'a211:spec' (no projection) -> kwargs."""
    parts = cfg.split(":")[1:]
    kw = dict(rank=None if parts[0] == "spec" else int(parts[0]), keep_bulk="bulk" in parts[1:])
    return kw


ROUND5 = "K3c2,K4c3,K4c3-no1111,a211:spec,a211:1,a211:4,a211:16,a211:64,a211:256"
