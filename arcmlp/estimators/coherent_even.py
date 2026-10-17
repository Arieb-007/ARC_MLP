"""W7: K=3 dense propagation plus switchable coherent even-order cumulant terms (research code).

Written by an LLM agent (Claude Code). Derivation: docs/notes/w7_coherent_even_orders.md.

Light state per post-ReLU layer: mu (n), C (n,n), K3 (n,n,n) dense, plus the kappa4 channels
  b   (n)    kappa4(h_m)                 (diagonal slice (4,))
  B   (n,n)  kappa(h_m, h_m, h_p, h_p)   ((2,2) slice, m != p)
No n^4 array is ever allocated, so this runs at n = 256.

Pre-activation kappa4 slices z = W h used by the next ReLU (mode):
  'none'      : kappa4(z) = 0                                   (plain K=3)
  'coherent'  : (4,)_a = (W^4 b)_a + 12 Bbar,  (2,2)_ab = 4 bbar/n + 4 Bbar   (rank one, O(n^2))
  'exact22'   : (4,)_a = (W^4 b)_a + 12 Bbar,  (2,2)_ab = ((W^2) diag(b) (W^2)^T)_ab + 4 Bbar
                (one n x n x n matmul: the exact transport of the diagonal source)
  'ex4'       : (4,)_a = (W^4 b)_a + 3 diag(W^2 B W^2^T)_a (one matmul), (2,2) rank one as 'coherent'
                (added after the decomposition in results/*_w7_coherent_even.md showed the (4,) mean-field
                 replacement 12 Bbar is what fails)
  'exall'     : exact paired slices: (4,) as 'ex4', (2,2) = W^2 diag(b) W^2^T + W^2 B W^2^T (two matmuls)
with W^k the entrywise power, bbar = mean_m b_m, Bbar = n^-2 sum_{m != p} B_mp. All other kappa4(z)
slices ((3,1), (2,1,1), (1,1,1,1)) are set to zero: they are random-sign O(n^{-3/2}).
The ReLU step is the dense reference's diagram expansion (arcmlp/reference/dense_cumulants.py) at
Kt = 4 restricted to output slices with at most two distinct neurons for order 4.

Mean-only higher even orders (option k6 / k8, added to the final-layer mean only; see the note):
  k6: coherent kappa6(z)_aaaaaa = 15 gamma6 with gamma6 = 8 cbar6 / n^2 (cbar6 = mean kappa6(h_m),
      Gaussian-ReLU closed form) plus the kappa4 x kappa4 product (3 beta_a)^2 / 2 term:
      dmu_a = kappa6_aaaaaa w(6,1)/720 + (kappa4_aaaa)^2 w(8,1) / (2 * 24^2)
  k8: kappa8_aaaaaaaa = 105 gamma8, gamma8 = 16 cbar8 / n^3, dmu_a = kappa8 w(8,1)/8!
"""
from __future__ import annotations

import math

import numpy as np

from arcmlp.closure.relu_moments import relu_cumulants_all
from arcmlp.reference import dense_cumulants as dc

ALLOWED_K4 = {(4,), (2, 2)}


class CoherentEvenProp(dc.DenseProp):
    def __init__(self, mode="coherent", cap=2, dtype=None, k6=False, k8=False, k3sq=False):
        super().__init__(Kt=4, cap=cap, dtype=dtype)
        self.mode = mode
        self.k6, self.k8, self.k3sq = k6, k8, k3sq
        # drop terms that need unavailable kappa4(z) slices, and order-4 outputs over > 2 neurons
        keep = []
        for t in self.registry:
            if sum(t.p) == 4 and len(t.p) > 2:
                continue
            if any(sum(v) == 4 and dc.slice_type(v) not in ALLOWED_K4 for v in t.diagram):
                continue
            if mode == "none" and any(sum(v) == 4 for v in t.diagram):
                continue
            keep.append(t)
        self.active = keep
        self.diag = []   # per-layer diagnostics

    # -------------------------------------------------- linear step on the light state
    def linear_light(self, S, W):
        n = W.shape[0]
        Z = dc.linear_step({1: S[1], 2: S[2], 3: S[3]}, W, 3)
        bbar = float(np.mean(S["b"]))
        Bm = S["B"]
        Bbar = float((Bm.sum() - np.trace(Bm)) / n ** 2)
        W2 = W * W
        if self.mode in ("ex4", "exall"):
            W2B = W2 @ Bm                                          # one matmul
            z4 = (W2 * W2) @ S["b"] + 3.0 * np.sum(W2B * W2, axis=1)   # exact paired (4,) slice
        else:
            z4 = (W2 * W2) @ S["b"] + 12.0 * Bbar
        if self.mode == "exact22":
            z22 = (W2 * S["b"]) @ W2.T + 4.0 * Bbar
        elif self.mode == "exall":
            z22 = (W2 * S["b"]) @ W2.T + W2B @ W2.T                # exact paired (2,2) slice
        else:
            z22 = np.full((n, n), 4.0 * bbar / n + 4.0 * Bbar, dtype=W.dtype)
        np.fill_diagonal(z22, 0.0)
        Z["s4"], Z["s22"] = z4.astype(W.dtype), z22.astype(W.dtype)
        self.diag.append(dict(bbar=bbar, Bbar=Bbar, beta=4 * bbar / n + 4 * Bbar))
        return Z

    def input_slices(self, Z, layer=0):
        sl = {}
        Ct = Z[2].copy()
        np.fill_diagonal(Ct, 0.0)
        sl[(1, 1)] = Ct
        for pat in dc.int_partitions(3):
            sl[pat] = dc._diag_slice(Z[3], pat)
        sl[(4,)] = Z["s4"]
        sl[(2, 2)] = Z["s22"]
        return sl

    def relu_light(self, Z, layer):
        n = Z[1].shape[0]
        dt = Z[1].dtype
        pK = self.power_cumulants_light(Z, layer)
        Ks = {}
        for q in [(1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1), (4,), (2, 2)]:
            r = len(q)
            form = dc.pk_to_k_formula(q)
            if len(form) == 1 and form[0][0] == 1.0 and form[0][1] == ((tuple(range(r)), q),):
                Ks[q] = pK[q]
                continue
            out = np.zeros((n,) * r, dtype=dt)
            for coef, mono in form:
                ops, subs = [], []
                for verts, powv in mono:
                    pw = tuple(sorted(powv, reverse=True))
                    order = sorted(range(len(verts)), key=lambda i: -powv[i])
                    ops.append(pK[pw])
                    subs.append("".join(dc.LETTERS[verts[i]] for i in order))
                out += coef * np.einsum(",".join(subs) + "->" + dc.LETTERS[:r], *ops)
            Ks[q] = dc._symmetrize(out, q) if r > 1 else out
        S = {1: Ks[(1,)].copy()}
        C = Ks[(1, 1)].copy(); np.fill_diagonal(C, Ks[(2,)]); S[2] = C
        T = Ks[(1, 1, 1)].copy(); idx = np.arange(n); I, J = np.ix_(idx, idx)
        A = Ks[(2, 1)]; T[I, I, J] = A; T[I, J, I] = A; T[J, I, I] = A; T[idx, idx, idx] = Ks[(3,)]
        S[3] = T
        S["b"] = Ks[(4,)].copy()
        B = Ks[(2, 2)].copy(); np.fill_diagonal(B, 0.0); S["B"] = B
        return S

    def power_cumulants_light(self, Z, layer):
        """dc.DenseProp.power_cumulants restricted to the slices the light state needs."""
        n = Z[1].shape[0]
        dt = Z[1].dtype
        m = Z[1].astype(np.float64)
        s = np.sqrt(np.maximum(np.diag(Z[2]).astype(np.float64), 1e-30))
        sl = self.input_slices(Z, layer)
        wc = {}

        def w(k, p):
            if (k, p) not in wc:
                wc[(k, p)] = dc.wick_vec(m, s, k, p).astype(dt)
            return wc[(k, p)]
        need = [(1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1), (4,), (3, 1), (2, 2)]
        by_p = {}
        for t in self.active:
            by_p.setdefault(t.p, []).append(t)
        pK = {}
        for p in need:
            r = len(p)
            acc = np.zeros((n,) * r, dtype=dt)
            for t in by_p.get(p, []):
                ops, subs, legs = [], [], [0] * r
                for v in t.diagram:
                    st = dc.slice_type(v)
                    verts = sorted([a for a in range(r) if v[a]], key=lambda a: -v[a])
                    ops.append(sl[st]); subs.append("".join(dc.LETTERS[a] for a in verts))
                    for a in range(r):
                        legs[a] += v[a]
                for a in range(r):
                    ops.append(w(legs[a], p[a])); subs.append(dc.LETTERS[a])
                acc += t.count * dc.vec_part_coef(t.diagram) * np.einsum(",".join(subs) + "->" + dc.LETTERS[:r], *ops, optimize=True)
            pK[p] = dc._symmetrize(acc, p) if r > 1 else acc
        return pK

    def _mean_corrections(self, Z, S_prev, n):
        """k6 / k8 coherent corrections to the mean of this layer's ReLU output (mean only)."""
        m = Z[1].astype(np.float64)
        s = np.sqrt(np.diag(Z[2]).astype(np.float64))
        dmu = np.zeros(n)
        mp, sp = S_prev.get("m_pre"), S_prev.get("s_pre")
        kap = relu_cumulants_all(mp, sp, 8) if mp is not None else None   # previous layer's Gaussian-ReLU cumulants
        if self.k6 and kap is not None:
            g6 = 8.0 * float(np.mean(kap[6])) / n ** 2
            k4aaaa = Z["s4"].astype(np.float64)
            dmu += 15.0 * g6 * dc.wick_vec(m, s, 6, 1) / 720.0
            dmu += k4aaaa ** 2 * dc.wick_vec(m, s, 8, 1) / (2.0 * 24.0 ** 2)
        if self.k3sq:   # (kappa3_aaa)^2 / 72 * w(6,1): coherent (a square), O(n^-2)
            dmu += Z[3][np.arange(n), np.arange(n), np.arange(n)].astype(np.float64) ** 2 * dc.wick_vec(m, s, 6, 1) / 72.0
        if self.k8 and kap is not None:
            g8 = 16.0 * float(np.mean(kap[8])) / n ** 3
            dmu += 105.0 * g8 * dc.wick_vec(m, s, 8, 1) / math.factorial(8)
        return dmu

    def run(self, Ws, return_all=False):
        n = Ws[0].shape[0]
        dt = self.dtype or (np.float64 if n <= 64 else np.float32)
        S = {1: np.zeros(n, dt), 2: np.eye(n, dtype=dt), 3: np.zeros((n,) * 3, dt),
             "b": np.zeros(n, dt), "B": np.zeros((n, n), dt)}
        mus, dmus = [], []
        for l, W in enumerate(Ws):
            Wd = W.astype(dt)
            Z = self.linear_light(S, Wd)
            dmu = self._mean_corrections(Z, S, n) if (self.k6 or self.k8 or self.k3sq) else np.zeros(n)
            Snew = self.relu_light(Z, l)
            Snew["m_pre"] = Z[1].astype(np.float64)
            Snew["s_pre"] = np.sqrt(np.diag(Z[2]).astype(np.float64))
            S = Snew
            mus.append(S[1].astype(np.float64) + dmu)
            dmus.append(dmu)
        self.dmus = np.array(dmus)
        return np.array(mus), S


def run_variant(Ws, variant, cap=2):
    """Named W7 variants. Returns final-layer mean estimate and per-layer means."""
    if variant == "K3":
        return dc.propagate(Ws, K=3, cap=cap)[0]
    if variant == "K4c":     # all kappa4 slices, dense reference (n <= 128)
        return dc.propagate(Ws, K=4, cap=cap)[0]
    mode = {"k4coh": "coherent", "k4ex22": "exact22", "k4coh_k6": "coherent", "k4coh_k6_k8": "coherent"}[variant]
    pr = CoherentEvenProp(mode=mode, cap=cap, k6="k6" in variant, k8="k8" in variant)
    return pr.run(Ws)[0]
