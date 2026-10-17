"""Dense cumulant propagation through bias-free ReLU MLPs, cumulant orders <= 4 (Stage 2.1 / W5 / W7).

Written by an LLM agent (Claude Code). Transcribed from ARC's reference implementation of the
companion paper (github.com/alignment-research-center/mlp_cumulant_propagation @ 93d091a, MIT;
src/mlp_kprop/kprop_ds.py `nonlin_kprop`, kprop_harmonic.py `get_all_terms`, `get_vec_cond`,
partitions.py `vec_part_coef`, `is_mixed`, `is_connected`), re-implemented densely in numpy.
The paper itself (arXiv:2605.05179) was not reachable from the sandbox; the code is the source.

State: post-activation cumulants of h, dense symmetric tensors {1: mu (n), 2: C, 3: K3, 4: K4},
orders <= Kt (Kt = 2, 3 or 4). Input x ~ N(0, I).

Linear step (exact): kappa_k(z) = W^{(x)k} . kappa_k(h) (mode products).

ReLU step (diagram expansion around the independent Gaussian with each neuron's mean m_i and
variance s_i^2 = C_ii): for an output power-cumulant slice p = (p_1..p_r) over r distinct neurons,
    pK_p[a_1..a_r] = kappa(h_{a_1}^{p_1}, ..., h_{a_r}^{p_r})
                   = sum over connected, 2-mixed multisets of input blocks v (vectors over the r
                     vertices, |v| in {2..Kt}; |v| = 2 needs two distinct vertices)
                     coef(nu) * prod_v kappa_v * prod_a w(k_a, p_a)(a),
    coef(nu) = 1 / prod_{distinct v} (c(v)! * prod_i v_i!^{c(v)})        (vec_part_coef),
    k_a = total legs at vertex a, w(k, p) = E[d^k ReLU(z)^p] (closed form, closure/relu_moments).
Truncation (paper eq. 14 as implemented): keep diagrams with sum_v cost(v) <= cap, cost(v) =
|v| - 1 - [all entries of v even]; |kappa_v| = O(n^{-cost/2}). Default cap = Kt - 1, which makes
Kt = 2 exactly the starter-kit covariance propagation (only the single covariance edge).
Power cumulants are converted to ordinary cumulants symbolically (moments as intermediate).

Every (output slice p, diagram isomorphism class) is a registry Term with metadata and can be
switched off (W5). Input-slice hooks allow zeroing / replacing kappa4 slices after each linear
layer (W7). Note the paper's budget-K algorithm also tracks paired slices of order > K
(e.g. kappa6 (2,2,2) at K = 3); this dense order-Kt version tracks orders <= Kt only.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from arcmlp.closure.relu_moments import hermite_e, relu_raw_moments

LETTERS = "abcdefgh"


# ------------------------------------------------------------------ combinatorics
def int_partitions(d, maxpart=None):
    maxpart = d if maxpart is None else maxpart
    if d == 0:
        yield ()
        return
    for first in range(min(d, maxpart), 0, -1):
        for rest in int_partitions(d - first, first):
            yield (first,) + rest


def block_cost(v):
    return sum(v) - 1 - int(all(x % 2 == 0 for x in v))


def slice_type(v):
    """Sorted nonzero multiplicities of a block vector, e.g. (0,2,1) -> (2,1)."""
    return tuple(sorted((x for x in v if x), reverse=True))


def _connected(vp, r):
    parent = list(range(r))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for v in vp:
        supp = [i for i, x in enumerate(v) if x]
        for j in supp[1:]:
            parent[find(j)] = find(supp[0])
    return len({find(i) for i in range(r)}) == 1


def vec_part_coef(vp):
    cnt = Counter(vp)
    den = 1
    for v, c in cnt.items():
        den *= math.factorial(c) * math.prod(math.factorial(x) for x in v) ** c
    return 1.0 / den


def _canonical(vp, p):
    """Canonical representative of a vector partition under permutations of vertices that preserve
    the multiplicity vector p."""
    r = len(p)
    best = None
    for perm in itertools.permutations(range(r)):
        if any(p[perm[i]] != p[i] for i in range(r)):
            continue
        q = tuple(sorted(tuple(v[perm[i]] for i in range(r)) for v in vp))
        if best is None or q < best:
            best = q
    return best


@lru_cache(maxsize=None)
def enumerate_terms(p, Kt, cap):
    """All diagram classes for output power-cumulant slice p: list of (rep, count)."""
    r = len(p)
    blocks = []
    for size in range(2, Kt + 1):
        for v in itertools.product(range(size + 1), repeat=r):
            if sum(v) != size:
                continue
            nz = sum(1 for x in v if x)
            if size == 2 and nz < 2:
                continue                       # mean/variance are absorbed (2-mixed)
            if block_cost(v) > cap:
                continue
            blocks.append(v)
    classes = Counter()
    if r == 1:
        classes[()] += 1                       # empty diagram: E[h^p] under the reference Gaussian
    maxb = cap  # every allowed block has cost >= 1
    for nb in range(1, maxb + 1):
        for combo in itertools.combinations_with_replacement(range(len(blocks)), nb):
            vp = tuple(sorted(blocks[i] for i in combo))
            if sum(block_cost(v) for v in vp) > cap:
                continue
            if not _connected(vp, r) and r > 1:
                continue
            if r == 1 and False:
                pass
            classes[_canonical(vp, p)] += 1
    return sorted(classes.items())


# ------------------------------------------------------------------ term registry
@dataclass(frozen=True)
class Term:
    p: tuple            # output power-cumulant slice
    diagram: tuple      # canonical vector partition (tuple of block vectors over len(p) vertices)
    count: int

    @property
    def tid(self):
        return f"pK{self.p}|" + "+".join(f"{'CKK'[min(sum(v), 4) - 2] if sum(v) == 2 else 'K' + str(sum(v))}{slice_type(v)}@{v}" for v in self.diagram)

    @property
    def cost(self):
        return sum(block_cost(v) for v in self.diagram)

    @property
    def order_in_1_over_n(self):
        """Exponent e such that the term is O(n^{-e}) per entry (sum of block costs / 2)."""
        return self.cost / 2.0

    @property
    def inputs(self):
        names = []
        for v in self.diagram:
            k = sum(v)
            names.append(("C" if k == 2 else f"K{k}") + str(slice_type(v)).replace(" ", ""))
        return tuple(names)

    @property
    def coherent(self):
        """Coherent iff every random-sign block appears an even number of times with the same
        support (e.g. c_ij^2) and the rest are even (paired) blocks such as K4(2,2), K4(4)."""
        cnt = Counter(self.diagram)
        for v, c in cnt.items():
            if all(x % 2 == 0 for x in v):
                continue
            if c % 2:
                return False
        return True


def build_registry(Kt, cap=None):
    cap = Kt - 1 if cap is None else cap
    reg = []
    for d in range(1, Kt + 1):
        for p in int_partitions(d):
            for rep, cnt in enumerate_terms(p, Kt, cap):
                reg.append(Term(p, rep, cnt))
    return reg


# ------------------------------------------------------------------ pK -> K conversion
def _set_partitions(items):
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


@lru_cache(maxsize=None)
def pk_to_k_formula(q):
    """Ordinary cumulant slice q (multiplicities over r vertices) as a polynomial in power
    cumulants: list of (coef, ((powers over r vertices restricted, vertex subset), ...))."""
    r = len(q)
    occ = [a for a in range(r) for _ in range(q[a])]
    poly = defaultdict(float)
    for pi in _set_partitions(list(range(len(occ)))):
        k = len(pi)
        mu = (-1) ** (k - 1) * math.factorial(k - 1)
        # each block's moment E[prod h_a^{B_a}] expanded into power cumulants over vertex partitions
        moment_expansions = []
        for B in pi:
            powv = [0] * r
            for o in B:
                powv[occ[o]] += 1
            verts = [a for a in range(r) if powv[a]]
            terms = []
            for rho in _set_partitions(verts):
                terms.append(tuple(sorted((tuple(sorted(C)), tuple(powv[a] for a in sorted(C))) for C in rho)))
            moment_expansions.append(terms)
        for combo in itertools.product(*moment_expansions):
            mono = tuple(sorted(f for t in combo for f in t))
            poly[mono] += mu
    return [(c, mono) for mono, c in poly.items() if abs(c) > 1e-12]


# ------------------------------------------------------------------ numerics
def wick_vec(m, s, k, p):
    """w(k, p) = E[d^k ReLU(z)^p] for z ~ N(m, s^2), vectorised (see closure/relu_moments.wick)."""
    if k < p:
        ff = math.prod(range(p - k + 1, p + 1))
        return ff * relu_raw_moments(m, s, p - k)[p - k]
    if p > 1:
        return math.factorial(p) * wick_vec(m, s, k - p + 1, 1)
    t = m / s
    if k == 0:
        return m * ndtr(t) + s * np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
    if k == 1:
        return ndtr(t)
    return (-1) ** k * hermite_e(k - 2, t) * np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi) / s ** (k - 1)


def _diag_slice(T, pattern):
    """Slice of a dense symmetric tensor T (order = sum(pattern)) on a multiplicity pattern, as an
    array over len(pattern) axes, e.g. pattern (2,1) of K3 -> S[x, y] = K3[x, x, y]."""
    labels = []
    for ax, m in enumerate(pattern):
        labels += [LETTERS[ax]] * m
    return np.einsum("".join(labels) + "->" + LETTERS[:len(pattern)], T)


def linear_step(K, W, Kt):
    Z = {1: W @ K[1], 2: W @ K[2] @ W.T}
    if Kt >= 3:
        T = K[3]
        for ax in range(3):
            T = np.tensordot(W, T, axes=([1], [ax]))   # new axis 0; cycles the axes
        Z[3] = np.ascontiguousarray(np.transpose(T, (2, 1, 0)))
    if Kt >= 4:
        T = K[4]
        for ax in range(4):
            T = np.tensordot(W, T, axes=([1], [3]))
        Z[4] = np.ascontiguousarray(T)
    return Z


def _sym_axes(T, axes):
    """Average of T over all permutations of the given axes, via the coset recursion
    Sym_{a0..ak} = (1/(k+1)) sum_j tau_{a0 aj} Sym_{a1..ak}  (k(k+1)/2 transposes instead of (k+1)!-1)."""
    axes = list(axes)
    if len(axes) <= 1:
        return T
    inner = _sym_axes(T, axes[1:])
    acc = inner.copy()
    for j in axes[1:]:
        perm = list(range(T.ndim))
        perm[axes[0]], perm[j] = perm[j], perm[axes[0]]
        acc += np.transpose(inner, perm)
    acc /= len(axes)
    return acc


def _symmetrize(T, p):
    """Average over permutations of axes with equal multiplicity in p."""
    groups = defaultdict(list)
    for i, x in enumerate(p):
        groups[x].append(i)
    for g in groups.values():
        if len(g) > 1:
            T = _sym_axes(T, g)
    return T


class DenseProp:
    """Dense order-Kt cumulant propagation with a switchable term registry and slice hooks."""

    def __init__(self, Kt=3, cap=None, dtype=None, terms_off=(), slice_hook=None, record=False):
        self.Kt, self.cap = Kt, (Kt - 1 if cap is None else cap)
        self.dtype = dtype
        self.registry = build_registry(Kt, self.cap)
        off = set(terms_off)
        self.active = [t for t in self.registry if t.tid not in off]
        self.slice_hook = slice_hook          # f(layer, Z: dict order->dense tensor, slices: dict) -> None
        self.record = record
        self.records = []

    def input_slices(self, Z, layer=0):
        Kt = self.Kt
        sl = {}
        Ct = Z[2].copy()
        np.fill_diagonal(Ct, 0.0)
        sl[(1, 1)] = Ct
        for k in range(3, Kt + 1):
            for pat in int_partitions(k):
                sl[pat] = _diag_slice(Z[k], pat)
        if self.slice_hook is not None:
            self.slice_hook(layer, Z, sl)
        return sl

    def power_cumulants(self, Z, layer=0, terms=None):
        """pK slices (dict p -> array over len(p) distinct-neuron axes) from the active terms
        (or the given list of terms)."""
        n = Z[1].shape[0]
        Kt = self.Kt
        dt = Z[1].dtype
        m = Z[1].astype(np.float64)
        s = np.sqrt(np.maximum(np.diag(Z[2]).astype(np.float64), 1e-30))
        sl = self.input_slices(Z, layer)
        wcache = {}

        def w(k, p):
            if (k, p) not in wcache:
                wcache[(k, p)] = wick_vec(m, s, k, p).astype(dt)
            return wcache[(k, p)]
        pK = {}
        by_p = defaultdict(list)
        for t in (self.active if terms is None else terms):
            by_p[t.p].append(t)
        for d in range(1, Kt + 1):
            for p in int_partitions(d):
                r = len(p)
                acc = np.zeros((n,) * r, dtype=dt)
                for t in by_p.get(p, []):
                    ops, subs = [], []
                    legs = [0] * r
                    for v in t.diagram:
                        st = slice_type(v)
                        verts = sorted([a for a in range(r) if v[a]], key=lambda a: -v[a])
                        ops.append(sl[st])
                        subs.append("".join(LETTERS[a] for a in verts))
                        for a in range(r):
                            legs[a] += v[a]
                    for a in range(r):
                        ops.append(w(legs[a], p[a]))
                        subs.append(LETTERS[a])
                    expr = ",".join(subs) + "->" + LETTERS[:r]
                    coef = t.count * vec_part_coef(t.diagram)
                    acc += coef * np.einsum(expr, *ops, optimize=True)
                    if self.record:
                        self.records.append((layer, t.tid))
                pK[p] = _symmetrize(acc, p) if r > 1 else acc
        return pK

    def relu_step(self, Z, layer=0):
        n = Z[1].shape[0]
        Kt = self.Kt
        dt = Z[1].dtype
        pK = self.power_cumulants(Z, layer)
        # ordinary cumulant slices on distinct vertices
        Kslices = {}
        for d in range(1, Kt + 1):
            for q in int_partitions(d):
                r = len(q)
                form = pk_to_k_formula(q)
                if len(form) == 1 and form[0][0] == 1.0 and form[0][1] == ((tuple(range(r)), q),):
                    Kslices[q] = pK[q]          # all multiplicities 1: cumulant == power cumulant
                    continue
                out = np.zeros((n,) * r, dtype=dt)
                for coef, mono in form:
                    ops, subs = [], []
                    for verts, powv in mono:
                        pw = tuple(sorted(powv, reverse=True))
                        order = sorted(range(len(verts)), key=lambda i: -powv[i])
                        ops.append(pK[pw])
                        subs.append("".join(LETTERS[verts[i]] for i in order))
                    out += coef * np.einsum(",".join(subs) + "->" + LETTERS[:r], *ops)
                Kslices[q] = _symmetrize(out, q) if r > 1 else out
        return self._assemble(Kslices, n, dt)

    def _assemble(self, S, n, dt):
        K = {1: S[(1,)].copy()}
        C = S[(1, 1)].copy()
        np.fill_diagonal(C, S[(2,)])
        K[2] = C
        idx = np.arange(n)
        if self.Kt >= 3:
            T = S[(1, 1, 1)].copy()
            I, J = np.ix_(idx, idx)
            A = S[(2, 1)]
            T[I, I, J] = A; T[I, J, I] = A; T[J, I, I] = A
            T[idx, idx, idx] = S[(3,)]
            K[3] = T
        if self.Kt >= 4:
            T = S[(1, 1, 1, 1)]          # in place (largest array)
            I3 = np.ix_(idx, idx, idx)
            i, j, k = I3
            A = S[(2, 1, 1)]
            for pos in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                ind = [None] * 4
                rest = [x for x in range(4) if x not in pos]
                ind[pos[0]] = i; ind[pos[1]] = i; ind[rest[0]] = j; ind[rest[1]] = k
                T[tuple(ind)] = A
            I2, J2 = np.ix_(idx, idx)
            B22 = S[(2, 2)]
            T[I2, I2, J2, J2] = B22; T[I2, J2, I2, J2] = B22; T[I2, J2, J2, I2] = B22
            B31 = S[(3, 1)]
            T[I2, I2, I2, J2] = B31; T[I2, I2, J2, I2] = B31; T[I2, J2, I2, I2] = B31; T[J2, I2, I2, I2] = B31
            T[idx, idx, idx, idx] = S[(4,)]
            K[4] = T
        return K

    def run(self, Ws, return_all=False):
        n = Ws[0].shape[0]
        dt = self.dtype or (np.float64 if n <= 64 else np.float32)
        K = {1: np.zeros(n, dt), 2: np.eye(n, dtype=dt)}
        if self.Kt >= 3:
            K[3] = np.zeros((n,) * 3, dt)
        if self.Kt >= 4:
            K[4] = np.zeros((n,) * 4, dt)
        mus, states = [], []
        for l, W in enumerate(Ws):
            Z = linear_step(K, W.astype(dt), self.Kt)
            del K
            K = self.relu_step(Z, layer=l)
            del Z
            mus.append(K[1].astype(np.float64))
            if return_all:
                states.append({k: v.copy() for k, v in K.items()})
        return (np.array(mus), K, states) if return_all else (np.array(mus), K)


def propagate(Ws, K=3, terms_off=(), cap=None, slice_hook=None, dtype=None, return_all=False):
    return DenseProp(Kt=K, cap=cap, dtype=dtype, terms_off=terms_off, slice_hook=slice_hook).run(Ws, return_all)
