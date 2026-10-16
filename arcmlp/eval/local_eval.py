"""Grader-faithful local evaluation and statistics.

`run_whest(...)` shells out to `whest run --runner subprocess --max-threads 1 --json` (the
subprocess runner enforces RLIMIT_AS = 8 GiB, the 0.4 s residual cap and the 120 s wall cap,
like the grader) and returns a compact per-MLP table. `paired_bootstrap(...)` gives a CI for the
mean of per-MLP differences and the minimum detectable effect (MDE) at 80% power, alpha 0.05.

This module is tooling (not part of the submission): numpy and subprocess are fine here.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

B_FLOPS = 2 ** 41


def provenance(repo_dir: str | os.PathLike = ".") -> dict:
    """Commit hash, dirty flag, package versions, host — attached to every results file."""
    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo_dir), *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    from importlib.metadata import PackageNotFoundError, version
    vers = {}
    for dist in ("numpy", "scipy", "flopscope", "whestbench", "opt-einsum", "threadpoolctl"):
        try:
            vers[dist] = version(dist)
        except PackageNotFoundError:
            vers[dist] = None
    return dict(commit=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")),
                versions=vers, python=sys.version.split()[0], host=platform.node(),
                cpu_count=os.cpu_count(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_whest(estimator: str, dataset: str, out_json: str, n_mlps: int | None = None,
              split: str | None = None, extra_env: dict | None = None, extra_args=()) -> dict:
    cmd = ["whest", "run", "--estimator", os.path.basename(estimator), "--dataset", dataset,
           "--runner", "subprocess", "--max-threads", "1", "--detail", "full", "--json", *extra_args]
    if n_mlps:
        cmd += ["--n-mlps", str(n_mlps)]
    if split:
        cmd += ["--split", split]
    env = dict(os.environ, **(extra_env or {}))
    t0 = time.time()
    with open(out_json, "w") as f:
        p = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(estimator)), stdout=f,
                           stderr=subprocess.PIPE, text=True, env=env)
    rep = json.load(open(out_json))
    rep["_cmd"] = cmd; rep["_returncode"] = p.returncode; rep["_seconds"] = time.time() - t0
    rep["_stderr_tail"] = p.stderr[-2000:]
    return rep


def summarize(rep: dict) -> dict:
    """Compact per-MLP table + aggregates from a `whest run --json` report."""
    rows = []
    for m in rep["results"]["per_mlp"]:
        rows.append(dict(
            idx=m["mlp_index"], name=m["mlp_name"], mse=m["final_layer_mse"],
            cb=m["flops_used"] / B_FLOPS, adjusted=m["adjusted_final_layer_score"],
            wall=m["wall_time_s"], residual=m["residual_wall_time_s"],
            failed=bool(m.get("budget_exhausted") or m.get("time_exhausted")
                        or m.get("residual_wall_time_exhausted") or m.get("traceback")),
            error=(str(m.get("traceback"))[-300:] if m.get("traceback") else None)))
    ok = [r for r in rows if not r["failed"]]
    agg = dict(n=len(rows), n_failed=len(rows) - len(ok))
    if ok:
        for k in ("mse", "cb", "adjusted", "wall", "residual"):
            v = np.array([r[k] for r in ok])
            agg[k + "_mean"] = float(v.mean()); agg[k + "_max"] = float(v.max())
    return dict(rows=rows, agg=agg)


def bootstrap_ci(x, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    s = stat(x[idx], axis=1)
    return float(stat(x)), float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0, log_ratio: bool = True):
    """Paired comparison of per-MLP metrics a (candidate) vs b (baseline), same MLPs.

    log_ratio=True works on d_i = log(a_i/b_i) (scale-free; MSEs vary ~2x across MLPs) and reports
    the geometric-mean ratio exp(mean d) with a bootstrap CI and the MDE (as a ratio) at 80% power,
    two-sided alpha 0.05: MDE = exp((1.96 + 0.84) * sd(d)/sqrt(n))."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = np.log(a / b) if log_ratio else a - b
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    sd = float(d.std(ddof=1)) if d.size > 1 else float("nan")
    mde = (1.96 + 0.84) * sd / math.sqrt(d.size)
    if log_ratio:
        return dict(ratio=math.exp(mean), ci=(math.exp(lo), math.exp(hi)), mde_ratio=math.exp(mde), n=int(d.size))
    return dict(diff=mean, ci=(lo, hi), mde=mde, n=int(d.size))
