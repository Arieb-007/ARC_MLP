"""Shared helpers for the GPU pipeline (experiments/gpu/*). Written by an LLM agent (Claude Code)."""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
B_FLOPS = 2 ** 41
DATA = os.environ.get("GPU_DATA_DIR", os.path.join(ROOT, "data", "gpu"))


def parse_idx(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def mlp_path(d, i):
    return os.path.join(d, f"mlp_{i:04d}.npz")


def available_idx(d):
    return sorted(int(os.path.basename(p)[4:8]) for p in glob.glob(os.path.join(d, "mlp_*.npz")))


def load_mlp(d, i):
    z = np.load(mlp_path(d, i), allow_pickle=True)
    return dict(idx=i, W=z["weights"], final_means=z["final_means"], all_layer_means=z["all_layer_means"],
                seed=int(z["mlp_seed"]), name=str(z["mlp_name"]))


def ours_orientation(W_kit):
    """Dataset weights follow the starter kit (z = w.T @ h); our chains use z = W @ h."""
    return [np.ascontiguousarray(w.T) for w in W_kit]


def kit_closure(W_kit):
    """K=2 Gaussian covariance closure (the starter kit's covariance propagation) in float64 numpy
    with the dataset's weight orientation (z = w.T @ h): C_pre = w.T C w, post-ReLU means and
    variances from the univariate ReLU moments, off-diagonal C_post_ij = Phi_i Phi_j C_pre_ij.
    Returns (L, n) means; must equal DenseProp(Kt=2) on ours_orientation(W_kit) to rounding."""
    from scipy.stats import norm
    n = W_kit[0].shape[0]
    mu, C = np.zeros(n), np.eye(n)
    mus = []
    for w in W_kit:
        w = np.asarray(w, dtype=np.float64)
        m = w.T @ mu
        Cp = w.T @ C @ w
        v = np.maximum(np.diag(Cp), 1e-30)
        s = np.sqrt(v)
        a = m / s
        P, ph = norm.cdf(a), norm.pdf(a)
        mu = m * P + s * ph
        e2 = (m * m + v) * P + m * s * ph
        C = np.outer(P, P) * Cp
        np.fill_diagonal(C, e2 - mu * mu)
        mus.append(mu)
    return np.array(mus)


def mse(pred, truth):
    return float(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def boot_mean(x, B=10000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    s = x[rng.integers(0, x.size, (B, x.size))].mean(1)
    return float(x.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def paired_log_ratio(a, b, B=10000, seed=0):
    """Geometric-mean ratio a/b over paired MLPs with a bootstrap CI (log scale)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    d = np.log(a[ok] / b[ok])
    if d.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    m, lo, hi = boot_mean(d, B, seed)
    return float(np.exp(m)), float(np.exp(lo)), float(np.exp(hi)), int(d.size)
