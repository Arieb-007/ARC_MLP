"""Round 4: linear response of the kappa4 memory (pre-registration: docs/notes/r4_prereg_linear_response.md).

Written by an LLM agent (Claude Code) for team Arieb-007, 2026-10-19.

The dense K=4 cap-3 chain (K4c3) is split term by term into three chains run in lockstep:

  background  S_bar : order <= 3 outputs, no kappa4 block                 (= the dense K=3 cap-3 chain)
  probe       P     : every order-4 output — transmission (reads a kappa4 block) and births (no kappa4
                      block, plus the products of lower cumulants in the power-cumulant -> cumulant
                      conversion); its coefficients, edges and kappa3 come from the driver
  perturbed   S     : background terms on its own state + eps * read-outs (order <= 3 outputs that read
                      a kappa4 block of P's pre-activation)

drive="s", eps=1 reproduces K4c3 (identity of the partition). drive="bg" is the one-way chain: P is affine
in the births, P^{l+1} = T_l(S_bar) P^l + beta_l(S_bar), and the read-outs are added afterwards — the
premise of an adjoint read-out add-on. Schedules switch read-outs and births per layer; "kind" restricts
the read-outs to paired (coh) or random-sign (rs) z-slices; rank_layers records singular-value spectra of
the index-distinct bulk of P (and of the births alone) in the (ab)(cd) unfolding.
"""
from __future__ import annotations

import itertools

import numpy as np

from arcmlp.reference import dense_cumulants as dc
from arcmlp.reference.device import asnumpy, free_pool, get_array_module, resolve_device, xp_for_device

Q3 = [q for d in (1, 2, 3) for q in dc.int_partitions(d)]
Q4 = list(dc.int_partitions(4))


def has_k4_block(t):
    return any(sum(v) == 4 for v in t.diagram)


def k4_block_kind(t):
    """'coh' if the term's kappa4 block is paired ((4,) or (2,2) slice), 'rs' if random-sign, None if none."""
    for v in t.diagram:
        if sum(v) == 4:
            return "coh" if all(x % 2 == 0 for x in v) else "rs"
    return None


def split_terms(Kt=4, cap=3):
    """Partition of the Kt=4 cap-3 registry: background, read-out, transmission, birth."""
    reg = dc.build_registry(Kt, cap)
    bg = [t for t in reg if sum(t.p) <= 3 and not has_k4_block(t)]
    ro = [t for t in reg if sum(t.p) <= 3 and has_k4_block(t)]
    trans = [t for t in reg if sum(t.p) == 4 and has_k4_block(t)]
    birth = [t for t in reg if sum(t.p) == 4 and not has_k4_block(t)]
    return bg, ro, trans, birth


def convert(pK, qs, n, xp, dt, base=None):
    """Power cumulants -> ordinary cumulant slices q (as DenseProp.relu_step). base (births off): a dict of
    the lower power cumulants without the kappa4 read-outs; multi-block monomials then contribute
    prod(pK) - prod(base), i.e. only their part that vanishes with the kappa4 input."""
    out = {}
    for q in qs:
        r = len(q)
        form = dc.pk_to_k_formula(q)
        if len(form) == 1 and form[0][0] == 1.0 and form[0][1] == ((tuple(range(r)), q),):
            out[q] = pK[q]
            continue
        acc = xp.zeros((n,) * r, dtype=dt)
        for coef, mono in form:
            subs, pws = [], []
            for verts, powv in mono:
                pws.append(tuple(sorted(powv, reverse=True)))
                order = sorted(range(len(verts)), key=lambda i: -powv[i])
                subs.append("".join(dc.LETTERS[verts[i]] for i in order))
            expr = ",".join(subs) + "->" + dc.LETTERS[:r]
            tmp = xp.einsum(expr, *[pK[pw] for pw in pws])
            if base is not None and len(mono) > 1:
                tmp = tmp - xp.einsum(expr, *[base[pw] for pw in pws])
            tmp = tmp * coef if tmp.ndim == 0 else tmp.__imul__(coef)
            acc += tmp
            del tmp
        out[q] = dc._symmetrize(acc, q) if r > 1 else acc
    return out


def sq_norm(T):
    """Squared Frobenius norm in first-axis slabs (an n^4 float32 vector at n = 256 exceeds the 32-bit
    length of a single BLAS dot)."""
    xp = get_array_module(T)
    return float(sum(float(xp.vdot(T[i].reshape(-1), T[i].reshape(-1))) for i in range(T.shape[0])))


def masked_spectrum(T, r=256, power_iters=2, seed=0):
    """Singular values (top r, randomized range finder) of the index-distinct part of a symmetric order-4
    tensor in the (ab)(cd) unfolding, and its squared Frobenius norm. Makes one n^4 copy."""
    xp = get_array_module(T)
    n = T.shape[0]
    M = T.copy()
    idx = xp.arange(n)
    for i, j in itertools.combinations(range(4), 2):
        ind = [slice(None)] * 4
        ind[i] = idx
        ind[j] = idx
        M[tuple(ind)] = 0
    tot = sq_norm(M)
    A = M.reshape(n * n, n * n)
    r = min(r, n * n)
    rs = xp.random.RandomState(seed)
    Y = A @ rs.standard_normal((n * n, r)).astype(A.dtype)
    for _ in range(power_iters):
        Q, _ = xp.linalg.qr(Y)
        Y = A @ (A.T @ Q)
    Q, _ = xp.linalg.qr(Y)
    B = Q.T @ A
    s = asnumpy(xp.linalg.svd(B, compute_uv=False)).astype(np.float64)
    del M, A, Y, Q, B
    free_pool()
    return s, tot


def spectrum_summary(s, tot, full_norm2):
    e = np.cumsum(s ** 2) / max(tot, 1e-300)
    def frac(k):
        return float(e[min(k, len(e)) - 1]) if len(e) else float("nan")
    def rank_at(f):
        hit = np.nonzero(e >= f)[0]
        return int(hit[0] + 1) if len(hit) else None
    return dict(frac16=frac(16), frac64=frac(64), frac256=frac(256), k50=rank_at(0.5), k90=rank_at(0.9),
                bulk_frac=float(tot / max(full_norm2, 1e-300)), r=len(s))


class LinearResponseProp:
    """See the module docstring. mode: 'bg' (background only, = K3c3) or 'probe'.
    ro_last / birth_last / birth_first: None = every layer; ro_last=k -> read-outs only at the last k ReLU
    steps; birth_last=g -> births only at the last g+1 steps (g generations reach the final mean);
    birth_first=A -> births only at steps < A. kind: None, 'coh' or 'rs' (read-out z-slices)."""

    def __init__(self, mode="probe", drive="bg", eps=1.0, ro_last=None, birth_last=None, birth_first=None,
                 kind=None, rank_layers=(), rank_r=256, births_rank_max_n=200, dtype=None, device=None):
        self.mode, self.drive, self.eps = mode, drive, float(eps)
        self.ro_last, self.birth_last, self.birth_first, self.kind = ro_last, birth_last, birth_first, kind
        self.rank_layers, self.rank_r, self.births_rank_max_n = tuple(rank_layers), rank_r, births_rank_max_n
        self.dtype = dtype
        self.device = resolve_device(device)
        self.p3 = dc.DenseProp(Kt=3, cap=3, device=self.device)
        self.p4 = dc.DenseProp(Kt=4, cap=3, device=self.device)
        self.bg, ro, self.trans, self.birth = split_terms(4, 3)
        self.ro_all = ro
        self.ro = [t for t in ro if kind is None or k4_block_kind(t) == kind]
        self.extra = {}

    # schedules ---------------------------------------------------------------------------------
    def ro_on(self, l, L):
        return self.ro_last is None or l >= L - self.ro_last

    def birth_on(self, l, L):
        if self.birth_first is not None and l >= self.birth_first:
            return False
        if self.birth_last is not None and l < L - 1 - self.birth_last:
            return False
        return True

    def _spectrum(self, T, layer, what, norm2=None):
        """Spectrum record of T's index-distinct bulk; a failure (non-finite tensor) is recorded, not raised."""
        try:
            norm2 = sq_norm(T) if norm2 is None else norm2
            if not np.isfinite(norm2):
                raise FloatingPointError("non-finite tensor")
            s, tot = masked_spectrum(T, self.rank_r)
            return dict(layer=layer, what=what, **spectrum_summary(s, tot, norm2), s=s[:64].tolist())
        except Exception as e:   # noqa: BLE001 — diagnostics must not void the configuration's MSE
            free_pool()
            return dict(layer=layer, what=what, error=f"{type(e).__name__}: {str(e)[:120]}")

    # one ReLU step of a K=3 state from its power cumulants ----------------------------------------
    def _assemble3(self, pK, n, xp, dt):
        return self.p3._assemble(convert(pK, Q3, n, xp, dt), n, dt)

    def run(self, Ws):
        n, L = Ws[0].shape[0], len(Ws)
        dt = self.dtype or (np.float64 if n <= 64 else np.float32)
        xp = xp_for_device(self.device)
        Kb = {1: xp.zeros(n, dt), 2: xp.eye(n, dtype=dt), 3: xp.zeros((n,) * 3, dt)}
        probe = self.mode != "bg"
        if probe:
            Ks = {k: v.copy() for k, v in Kb.items()}
            P = xp.zeros((n,) * 4, dt)
        mus, mus_bg, pnorm, ranks = [], [], [], []
        for l, W in enumerate(Ws):
            W = xp.asarray(W, dtype=dt)
            Zb = dc.linear_step(Kb, W, 3)
            del Kb
            pKb = self.p3.power_cumulants(Zb, l)
            if not probe:
                del Zb
                Kb = self._assemble3(pKb, n, xp, dt)
                del pKb
                mus.append(asnumpy(Kb[1]).astype(np.float64))
                continue
            Zs = dc.linear_step(Ks, W, 3)
            del Ks
            holder = [P]
            del P                      # mode4 owns the only reference: freed after the first mode product
            Z4 = dc.mode4(holder, W)
            free_pool()
            # perturbed chain: background terms on S + eps * read-outs of the probe's pre-activation
            pKs = self.p3.power_cumulants(Zs, l)
            eps_l = self.eps if self.ro_on(l, L) else 0.0
            if eps_l and self.ro:
                Zr = {1: Zs[1], 2: Zs[2], 3: Zs[3], 4: Z4}
                pKr = self.p4.power_cumulants(Zr, l, terms=self.ro, orders=(1, 2, 3))
                del Zr
                for p, v in pKr.items():
                    pKs[p] = pKs[p] + eps_l * v
                del pKr
            # probe: transmission (+ births) with the driver's coefficients, edges and kappa3. The lower power
            # cumulants of its pK -> K conversion must contain the same-layer read-outs of the probe itself
            # (e.g. kappa4(h_a) = E h^4 - 3 (E h^2)^2 - ..., and E h^2 reads kappa4(z_a)): without them half of
            # the kappa4 transmission is lost at order one.
            b_on = self.birth_on(l, L)
            if self.drive == "bg":
                Zd = Zb
                Zr = {1: Zb[1], 2: Zb[2], 3: Zb[3], 4: Z4}
                pKrb = self.p4.power_cumulants(Zr, l, terms=self.ro_all, orders=(1, 2, 3))
                del Zr
                base = {p: v for p, v in pKb.items() if sum(p) <= 3}
                lower = {p: base[p] + pKrb[p] for p in base}
                del pKrb
            else:
                Zd = Zs
                lower = {p: v for p, v in pKs.items() if sum(p) <= 3}
                base = None
                if not b_on:
                    Zr = {1: Zs[1], 2: Zs[2], 3: Zs[3], 4: Z4}
                    pKrs = self.p4.power_cumulants(Zr, l, terms=self.ro_all, orders=(1, 2, 3))
                    del Zr
                    base = {p: lower[p] - eps_l * pKrs[p] for p in lower}
                    del pKrs
            if l in self.rank_layers and n <= self.births_rank_max_n and b_on:
                pKbirth = self.p4.power_cumulants({1: Zd[1], 2: Zd[2], 3: Zd[3]}, l, terms=self.birth, orders=(4,))
                nolow = base if base is not None else {p: v for p, v in (pKb if self.drive == "bg" else pKs).items() if sum(p) <= 3}
                Bt = dc.assemble4(convert({**nolow, **pKbirth}, Q4, n, xp, dt), n)
                del pKbirth
                ranks.append(self._spectrum(Bt, l, "birth"))
                del Bt
                free_pool()
            Zp = {1: Zd[1], 2: Zd[2], 3: Zd[3], 4: Z4}
            del Z4
            terms4 = self.trans + (self.birth if b_on else [])
            pK4 = self.p4.power_cumulants(Zp, l, terms=terms4, orders=(4,), release=True)
            del Zp
            free_pool()
            S4 = convert({**lower, **pK4}, Q4, n, xp, dt, base=None if b_on else base)
            del pK4, lower, base
            P = dc.assemble4(S4, n)
            del S4
            free_pool()
            pn = sq_norm(P)
            pnorm.append(float(np.sqrt(max(pn, 0.0))))
            if l in self.rank_layers:
                ranks.append(self._spectrum(P, l, "probe", pn))
            Ks = self._assemble3(pKs, n, xp, dt)
            Kb = self._assemble3(pKb, n, xp, dt)
            del pKs, pKb, Zs, Zb, Zd
            free_pool()
            mus.append(asnumpy(Ks[1]).astype(np.float64))
            mus_bg.append(asnumpy(Kb[1]).astype(np.float64))
        self.extra = dict(pnorm=pnorm)
        if mus_bg:
            self.extra["mu_bg_final"] = mus_bg[-1].tolist()
        if ranks:
            self.extra["rank"] = ranks
        if probe:
            del P
            free_pool()
            return np.array(mus), Ks
        return np.array(mus), Kb


def parse_lr(cfg):
    """'lr:bg' | 'lr:twoway' | 'lr:oneway[:key=val...][:rank]' -> LinearResponseProp kwargs."""
    parts = cfg.split(":")[1:]
    head, opts = parts[0], parts[1:]
    if head == "bg":
        return dict(mode="bg")
    kw = dict(mode="probe", drive="s" if head == "twoway" else "bg")
    for o in opts:
        if o == "rank":
            kw["rank_layers"] = (4, 8, 12, 14)
            continue
        k, v = o.split("=")
        if k == "eps":
            kw["eps"] = float(v)
        elif k == "ro":
            kw["ro_last"] = int(v)
        elif k == "b":
            kw["birth_last"] = int(v)
        elif k == "bf":
            kw["birth_first"] = int(v)
        elif k == "kind":
            kw["kind"] = v
        elif k == "drive":
            kw["drive"] = v
        else:
            raise ValueError(f"unknown lr option {o!r} in {cfg!r}")
    return kw


ROUND4 = ("lr:bg,lr:twoway,lr:oneway,lr:oneway:eps=0.5,lr:oneway:ro=1,lr:oneway:ro=2,lr:oneway:ro=4,"
          "lr:oneway:b=1,lr:oneway:b=2,lr:oneway:b=4,lr:oneway:b=8,lr:oneway:ro=1:b=4,"
          "lr:oneway:bf=4,lr:oneway:bf=8,lr:oneway:kind=coh,lr:oneway:kind=rs,"
          # added after the CPU preliminary of 2026-10-19 (post hoc, exploratory): read-out strength sweep
          "lr:oneway:eps=0.25,lr:oneway:eps=0.75")
