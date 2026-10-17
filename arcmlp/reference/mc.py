"""Chunked Monte-Carlo ground truth for bias-free ReLU MLPs (float32 passes, float64 accumulators).

Written by an LLM agent (Claude Code). Reference tool, not part of any submission.

mc_truth(Ws, N, seed, chunk=100_000, keep_layers=None, full_tensors_upto=None) returns
  mu        (L, n)   per-layer means of h_l
  cm        (L, 5, n) per-neuron central moments of h_l, orders 0..4 (cm[:,2] = variance)
  covL      (n, n)   final-layer covariance
  varL      (n,)     final-layer variances
  floor     float    MC noise floor of the final-layer MSE: tr(Cov h_L) / (n N)
  K{2,3,4}  dict layer -> full joint cumulant tensors of h_l (only if full_tensors_upto and n <= 32)
Samples are drawn chunk by chunk with np.random.default_rng([seed, c]), so results do not depend
on the BLAS thread count and runs can be split and merged. Moments are accumulated for
y = h - shift (shift = a 1e4-sample pilot mean, independent stream) and converted exactly to central
moments, which avoids the cancellation of raw-moment accumulation without a second pass.

debiased_mse(est_final, truth): bias-corrected final-layer MSE with standard error (see below).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return None


def _central_from_shifted(S, N):
    """S[p] = sum y^p (p = 1..4) per entry -> central moments (orders 0..4) of h (shift-invariant)."""
    e = [np.ones_like(S[1])] + [S[p] / N for p in (1, 2, 3, 4)]
    d = e[1]
    c2 = e[2] - d * d
    c3 = e[3] - 3 * d * e[2] + 2 * d ** 3
    c4 = e[4] - 4 * d * e[3] + 6 * d * d * e[2] - 3 * d ** 4
    return np.stack([np.ones_like(d), np.zeros_like(d), c2, c3, c4])


def mc_truth(Ws, N, seed=0, chunk=100_000, keep_layers=None, full_tensors_upto=None,
             dtype=np.float32, log_every=0.0, keep_cov=True, moments="full"):
    """moments='full': per-neuron central moments (orders <= 4) of every layer; moments='final':
    per-layer means only plus final-layer covariance (~10x faster; enough for MSE vs truth)."""
    Ws = [np.asarray(W) for W in Ws]
    WT = [np.ascontiguousarray(W.T.astype(dtype)) for W in Ws]
    L, n = len(Ws), Ws[0].shape[0]
    N = int(N)
    keep = set(range(L)) if keep_layers is None else set(keep_layers)
    full_k = full_tensors_upto if (full_tensors_upto and n <= 32) else None
    # pilot shift (independent stream)
    rng0 = np.random.default_rng([seed, 10 ** 9])
    h = rng0.standard_normal((10_000, n)).astype(dtype)
    shift = []
    for l in range(L):
        h = np.maximum(h @ WT[l], 0)
        shift.append(h.mean(0, dtype=np.float64))
    shift = np.array(shift)
    S = [None] + [np.zeros((L, n)) for _ in range(4)]
    sLL = np.zeros((n, n))
    F2 = {l: np.zeros((n, n)) for l in keep} if full_k else {}
    F3 = {l: np.zeros((n, n, n)) for l in keep} if full_k and full_k >= 3 else {}
    F4 = {l: np.zeros((n * n, n * n)) for l in keep} if full_k and full_k >= 4 else {}
    F1 = {l: np.zeros(n) for l in keep} if full_k else {}
    done, c, t0, tl = 0, 0, time.time(), time.time()
    while done < N:
        b = min(chunk, N - done)
        rng = np.random.default_rng([seed, c])
        h = rng.standard_normal((b, n), dtype=np.float32).astype(dtype, copy=False)
        for l in range(L):
            h = h @ WT[l]
            np.maximum(h, 0, out=h)
            if moments == "final" and l < L - 1:
                S[1][l] += h.sum(0, dtype=np.float64) - b * shift[l]
                continue
            y = h.astype(np.float64) - shift[l]
            S[1][l] += y.sum(0)
            if moments == "full" or l == L - 1:
                y2 = y * y
                S[2][l] += y2.sum(0)
                if moments == "full":
                    S[3][l] += (y2 * y).sum(0); S[4][l] += (y2 * y2).sum(0)
            if l == L - 1 and keep_cov:
                sLL += y.T @ y
            if full_k and l in keep:
                F1[l] += y.sum(0)
                F2[l] += y.T @ y
                if full_k >= 3:
                    Y2 = (y[:, :, None] * y[:, None, :]).reshape(b, n * n)
                    if full_k >= 3:
                        F3[l] += (Y2.T @ y).reshape(n, n, n)
                    if full_k >= 4:
                        F4[l] += Y2.T @ Y2
        done += b; c += 1
        if log_every and time.time() - tl > log_every:
            tl = time.time(); r = done / (tl - t0)
            print(f"[mc] {done:,}/{N:,} ({r:,.0f}/s, eta {(N - done) / r / 60:.1f} min)", flush=True)
    cm = np.stack([_central_from_shifted([None] + [S[p][l] for p in (1, 2, 3, 4)], N) for l in range(L)])
    mu = shift + S[1] / N
    out = dict(mu=mu, cm=cm, varL=cm[L - 1, 2], N=N, seed=seed, chunk=chunk, n=n, L=L,
               seconds=time.time() - t0, git=_git_hash())
    if keep_cov:
        d = S[1][L - 1] / N
        out["covL"] = sLL / N - np.outer(d, d)
    out["floor"] = float(np.sum(out["varL"]) / (n * N))
    if full_k:
        out["K2"], out["K3"], out["K4"] = {}, {}, {}
        for l in keep:
            d = F1[l] / N
            E2 = F2[l] / N
            M2 = E2 - np.outer(d, d)
            out["K2"][l] = M2
            if full_k >= 3:
                E3 = F3[l] / N
                M3 = (E3 - np.einsum("ij,k->ijk", E2, d) - np.einsum("ik,j->ijk", E2, d)
                      - np.einsum("jk,i->ijk", E2, d) + 2 * np.einsum("i,j,k->ijk", d, d, d))
                out["K3"][l] = M3
            if full_k >= 4:
                E4 = (F4[l] / N).reshape(n, n, n, n)
                M4 = _central4(E2, E3, E4, d)   # central 4th moment from raw moments of y
                out["K4"][l] = (M4 - np.einsum("ij,kl->ijkl", M2, M2) - np.einsum("ik,jl->ijkl", M2, M2)
                                - np.einsum("il,jk->ijkl", M2, M2))
    return out


def _central4(E2, E3, E4, d):
    """Central 4th joint moment from raw moments E2, E3, E4 of a vector with mean d."""
    n = d.size
    M = E4.copy()
    # - sum over the 4 positions of E3(other three) * d(position)
    M -= np.einsum("jkl,i->ijkl", E3, d) + np.einsum("ikl,j->ijkl", E3, d) \
        + np.einsum("ijl,k->ijkl", E3, d) + np.einsum("ijk,l->ijkl", E3, d)
    # + sum over the 6 pairs of E2(other two) * d d
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        rest = [x for x in range(4) if x not in (a, b)]
        sub = "ijkl"
        expr = f"{sub[rest[0]]}{sub[rest[1]]},{sub[a]},{sub[b]}->ijkl"
        M += np.einsum(expr, E2, d, d)
    # - 3 d d d d
    M -= 3 * np.einsum("i,j,k,l->ijkl", d, d, d, d)
    return M


def debiased_mse(est_final, truth):
    """Bias-corrected final-layer MSE: E[(est - mc)^2] = (est - mu)^2 + Var/N per neuron.
    Returns raw, floor, debiased, se (MC-noise standard error, uses covL when present)."""
    mu, N = truth["mu"][-1], truth["N"]
    e = np.asarray(est_final, dtype=np.float64) - mu
    n = e.size
    raw = float(np.mean(e ** 2))
    floor = float(np.mean(truth["varL"]) / N)
    if "covL" in truth:
        C = truth["covL"]
        var_cross = 4.0 * float(e @ C @ e) / (n * n * N)
        var_chi = 2.0 * float(np.sum(C * C)) / (n * n * N * N)
    else:
        v = truth["varL"]
        var_cross = 4.0 * float(np.sum(e * e * v)) / (n * n * N)
        var_chi = 2.0 * float(np.sum(v * v)) / (n * n * N * N)
    return dict(raw=raw, floor=floor, debiased=raw - floor, se=math.sqrt(var_cross + var_chi), N=N)


def cache_key(n, L, wseed, N, mc_seed):
    return f"n{n}_L{L}_w{wseed}_N{N:.0e}_mc{mc_seed}".replace("+", "")


def cached_truth(Ws, n, L, wseed, N, mc_seed, cache_dir="data/mc", **kw):
    """Load data/mc/<key>.npz if present (verifying a weight hash), else compute and save."""
    os.makedirs(cache_dir, exist_ok=True)
    wh = hashlib.sha256(np.ascontiguousarray(np.asarray(Ws, dtype=np.float32)).tobytes()).hexdigest()[:16]
    path = os.path.join(cache_dir, cache_key(n, L, wseed, N, mc_seed) + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if str(z["whash"]) == wh:
            d = {k: z[k] for k in z.files if k not in ("meta",)}
            d.update(json.loads(str(z["meta"])))
            return d
    out = mc_truth(Ws, N, seed=mc_seed, **kw)
    meta = dict(N=out["N"], seed=out["seed"], n=out["n"], L=out["L"], floor=out["floor"],
                seconds=out["seconds"], git=out["git"], wseed=wseed)
    arrs = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, whash=wh, meta=json.dumps(meta), **arrs)
    out["whash"] = wh
    return out


def save(path, out, meta=None):  # backward compatibility (notebooks/run_mc_truth.py)
    np.savez_compressed(path, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
                        N=out["N"], seed=out["seed"], chunk=out["chunk"])
    if meta is not None:
        with open(str(path).replace(".npz", ".json"), "w") as f:
            json.dump(meta, f, indent=1)


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["N"] = int(d["N"]); d["seed"] = int(d["seed"])
    return d
