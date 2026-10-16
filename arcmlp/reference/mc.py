"""Chunked Monte-Carlo ground truth for bias-free ReLU MLPs (float32 passes, float64 accumulators).

Status: reference tool. Reports, per layer, the sample means; at the final layer also per-neuron
sample variances and (optionally) the full sample covariance, so that downstream code can
(a) state the MC noise floor  floor = mean_i Var(h_{L,i}) / N, and
(b) debias an estimator's MSE against this truth:
        E[(est_i - mc_i)^2] = (est_i - mu_i)^2 + Var(h_{L,i})/N      (exact, mc_i unbiased)
    so  mse_debiased = mean_i (est_i - mc_i)^2 - mean_i s_i^2/N,
    with standard error from the cross term 2 e.eta (Var = 4 e^T C e / (n^2 N)) and the
    chi-square term (Var ~ 2 sum_ij C_ij^2 / (n^2 N^2)); see `debiased_mse`.

The RNG stream is numpy PCG64 seeded per (seed, chunk index) so a run can be split across
machines and merged exactly (`merge`).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np


def mc_truth(weights, n_samples: int, seed: int = 0, batch: int = 8192, chunk: int = 2 ** 20,
             keep_cov: bool = True, dtype=np.float32, log_every: float = 60.0):
    """weights: sequence of (n, n) arrays W_l (z_l = W_l h_{l-1}). Returns dict of float64 arrays.

    Samples are generated in chunks of `chunk` samples; chunk c uses
    np.random.default_rng([seed, c]) so the result is independent of `batch` and of how many
    processes share the chunks (pass chunk ranges via `chunks=`)."""
    Ws = [np.ascontiguousarray(np.asarray(W, dtype=dtype).T) for W in weights]  # h @ W.T
    L, n = len(Ws), Ws[0].shape[0]
    n_chunks = math.ceil(n_samples / chunk)
    s1 = np.zeros((L, n)); s2L = np.zeros(n); sLL = np.zeros((n, n)) if keep_cov else None
    done = 0; t0 = time.time(); tlast = t0
    for c in range(n_chunks):
        rng = np.random.default_rng([seed, c])
        m_c = min(chunk, n_samples - c * chunk)
        for b0 in range(0, m_c, batch):
            b = min(batch, m_c - b0)
            h = rng.standard_normal((b, n), dtype=np.float32).astype(dtype, copy=False)
            for l, WT in enumerate(Ws):
                h = h @ WT
                np.maximum(h, 0, out=h)
                s1[l] += h.sum(0, dtype=np.float64)
            hd = h.astype(np.float64)
            s2L += (hd * hd).sum(0)
            if keep_cov:
                sLL += hd.T @ hd
            done += b
            if log_every and time.time() - tlast > log_every:
                tlast = time.time()
                rate = done / (tlast - t0)
                print(f"[mc] {done:,}/{n_samples:,} samples, {rate:,.0f}/s, eta {(n_samples-done)/rate/60:.1f} min", flush=True)
    N = done
    mu = s1 / N
    varL = s2L / N - mu[-1] ** 2
    out = dict(mu=mu, varL=varL, N=N, seed=seed, chunk=chunk, seconds=time.time() - t0)
    if keep_cov:
        out["covL"] = sLL / N - np.outer(mu[-1], mu[-1])
    return out


def debiased_mse(est_final, truth):
    """Bias-corrected final-layer MSE of `est_final` (n,) against an MC truth dict from `mc_truth`.

    Returns dict(raw=naive MSE vs MC, floor=mean var/N, debiased=raw-floor, se=standard error of
    `debiased` w.r.t. MC noise (uses covL if present, else the independent-neuron approximation))."""
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
    # e here is (est - mc); its MC part inflates var_cross slightly, conservative.
    return dict(raw=raw, floor=floor, debiased=raw - floor, se=math.sqrt(var_cross + var_chi), N=N)


def save(path, out, meta=None):
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
