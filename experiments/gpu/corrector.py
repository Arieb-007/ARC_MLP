"""Algorithm C: a per-neuron residual corrector on top of the V29 chain, fitted on networks with the
organizers' Monte-Carlo reference and evaluated on held-out networks (never the ones it was fitted
on; CLAUDE.md rule 6). Written by an LLM agent (Claude Code).

Target per final-layer neuron: r = truth - mu_V29. Features (all available inside the metered
estimator at O(n) cost): V29's final-layer mean, the K=2 closure's mean and their difference, the
pre-activation mean m, std s, t = m/s, V29's kappa3 diagonal D3 and kappa4 diagonal dG of the final
pre-activation, the transported diagonal g_prev, var_prev, lambda, the (2,1)-slice diagonal and row
norm, and Edgeworth-type products D3 w3/6, dG w4/24, D3^2 w6/72, dG^2 w8/1152 (so that a linear
model can re-fit the chain's effective constants), plus per-network means of the main features.
Models: ridge (closed form, lambda by inner CV over networks) and a 2x64 tanh MLP (numpy Adam,
early stopping on an inner network split). Evaluation: 5-fold grouped CV over the training networks
and a final fit evaluated on --test-idx.

    python experiments/gpu/corrector.py --v29 data/gpu/v29/baseline --mlps data/gpu/mlps --train-idx 20-99 --test-idx 0-19
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, boot_mean, kit_closure, load_mlp, mse, paired_log_ratio, parse_idx, read_json, write_json  # noqa: E402
sys.path.insert(0, ROOT)
from arcmlp.closure.relu_moments import hermite_e  # noqa: E402

FEATS = ["mu_v29", "mu_k2", "d_k2", "t", "s", "m", "D3", "dG", "g_prev", "var_prev", "lam", "D21_diag", "D21_rownorm",
         "e3", "e4", "e33", "e44", "phi", "Phi", "mean_dG", "mean_D3", "mean_lam", "mean_d_k2"]


def features_for(v29_dir, mlps_dir, i):
    row = read_json(os.path.join(v29_dir, f"mlp_{i:04d}.json"))
    if row is None or row.get("error"):
        return None
    z = np.load(os.path.join(v29_dir, f"mlp_{i:04d}.npz"))
    m = load_mlp(mlps_dir, i)
    p = z["preds"].astype(np.float64)
    L, n = p.shape
    layers = [int(k.split("_l")[-1]) for k in z.files if k.startswith("dG_l")]
    if not layers:
        return None
    lf = max(layers)

    def g(name):
        k = f"{name}_l{lf}"
        return z[k].astype(np.float64) if k in z.files else np.zeros(n)
    dG, D3, var, var_prev, g_prev, lam = g("dG"), g("D3"), g("var"), g("var_prev"), g("g_prev"), g("lam")
    d21d, d21r = g("D21_diag"), g("D21_rownorm")
    mu_k2 = kit_closure(m["W"])[-1]
    mprev = p[L - 2]
    mpre = np.asarray(m["W"][L - 1], dtype=np.float64).T @ mprev
    s = np.sqrt(np.maximum(var, 1e-12)) if np.any(var > 0) else np.sqrt(np.maximum(np.asarray(m["W"][L - 1], dtype=np.float64).T ** 2 @ np.ones(n) * 0.3, 1e-12))
    t = mpre / s
    phi, Phi = norm.pdf(t), norm.cdf(t)
    w3 = -t * phi / s ** 2
    w4 = (t * t - 1) * phi / s ** 3
    w6 = hermite_e(4, t) * phi / s ** 5
    w8 = hermite_e(6, t) * phi / s ** 7
    f = dict(mu_v29=p[-1], mu_k2=mu_k2, d_k2=p[-1] - mu_k2, t=t, s=s, m=mpre, D3=D3, dG=dG, g_prev=g_prev,
             var_prev=var_prev, lam=lam, D21_diag=d21d, D21_rownorm=d21r,
             e3=D3 * w3 / 6, e4=dG * w4 / 24, e33=D3 ** 2 * w6 / 72, e44=dG ** 2 * w8 / 1152, phi=phi, Phi=Phi)
    for k in ("dG", "D3", "lam", "d_k2"):
        f["mean_" + k] = np.full(n, float(np.mean(f[k])))
    X = np.stack([f[k] for k in FEATS], 1)
    r = m["final_means"] - p[-1]
    return dict(idx=i, X=X, r=r, mu=p[-1], truth=m["final_means"], mse0=mse(p[-1], m["final_means"]))


# ------------------------------------------------------------------ models
class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-12
        self.my = y.mean()
        Xs = (X - self.mx) / self.sx
        A = Xs.T @ Xs + self.lam * len(y) * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, Xs.T @ (y - self.my))
        return self

    def predict(self, X):
        return ((X - self.mx) / self.sx) @ self.w + self.my

    def export(self):
        return dict(kind="ridge", lam=self.lam, mean=self.mx.tolist(), std=self.sx.tolist(), w=self.w.tolist(), b=float(self.my))


class MLP:
    def __init__(self, hidden=(64, 64), lr=2e-3, epochs=300, batch=8192, wd=1e-5, seed=0, patience=30):
        self.h, self.lr, self.epochs, self.batch, self.wd, self.seed, self.patience = hidden, lr, epochs, batch, wd, seed, patience

    def _init(self, d):
        rng = np.random.default_rng(self.seed)
        sizes = [d, *self.h, 1]
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.params += [rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def _forward(self, X):
        acts = [X]
        for k in range(0, len(self.params), 2):
            z = acts[-1] @ self.params[k] + self.params[k + 1]
            acts.append(np.tanh(z) if k < len(self.params) - 2 else z)
        return acts

    def _grads(self, X, y):
        acts = self._forward(X)
        out = acts[-1][:, 0]
        g = ((out - y) * 2.0 / len(y))[:, None]
        grads = [None] * len(self.params)
        for k in range(len(self.params) - 2, -1, -2):
            a = acts[k // 2]
            grads[k] = a.T @ g + self.wd * self.params[k]
            grads[k + 1] = g.sum(0)
            if k > 0:
                g = (g @ self.params[k].T) * (1 - a * a)
        return grads, float(np.mean((out - y) ** 2))

    def fit(self, X, y, Xv=None, yv=None):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-12
        self.sy = y.std() + 1e-12
        Xs, ys = (X - self.mx) / self.sx, y / self.sy
        if Xv is not None:
            Xvs, yvs = (Xv - self.mx) / self.sx, yv / self.sy
        self._init(X.shape[1])
        rng = np.random.default_rng(self.seed + 1)
        best, best_params, bad = np.inf, None, 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(self.epochs):
            perm = rng.permutation(len(ys))
            for s0 in range(0, len(ys), self.batch):
                idx = perm[s0:s0 + self.batch]
                grads, _ = self._grads(Xs[idx], ys[idx])
                self.step += 1
                for j, gr in enumerate(grads):
                    self.m[j] = b1 * self.m[j] + (1 - b1) * gr
                    self.v[j] = b2 * self.v[j] + (1 - b2) * gr * gr
                    mh = self.m[j] / (1 - b1 ** self.step); vh = self.v[j] / (1 - b2 ** self.step)
                    self.params[j] -= self.lr * mh / (np.sqrt(vh) + eps)
            if Xv is not None:
                vl = float(np.mean((self._forward(Xvs)[-1][:, 0] - yvs) ** 2))
                if vl < best - 1e-6:
                    best, best_params, bad = vl, [p.copy() for p in self.params], 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_params is not None:
            self.params = best_params
        self.epochs_run = ep + 1
        return self

    def predict(self, X):
        return self._forward((X - self.mx) / self.sx)[-1][:, 0] * self.sy

    def export(self):
        return dict(kind="mlp", hidden=list(self.h), mean=self.mx.tolist(), std=self.sx.tolist(), sy=float(self.sy),
                    params=[p.tolist() for p in self.params], epochs_run=self.epochs_run)


def stack(data, ids):
    X = np.concatenate([data[i]["X"] for i in ids]); y = np.concatenate([data[i]["r"] for i in ids])
    return X, y


def evaluate(model, data, ids):
    rows = []
    for i in ids:
        corr = model.predict(data[i]["X"])
        rows.append(dict(idx=i, mse0=data[i]["mse0"], mse1=mse(data[i]["mu"] + corr, data[i]["truth"])))
    return rows


def fit_ridge_cv(data, ids, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), seed=0):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    if len(ids) < 4:
        X, y = stack(data, ids)
        return Ridge(1e-2).fit(X, y), 1e-2
    folds = np.array_split(rng.permutation(ids), 4)
    best, best_lam = np.inf, None
    for lam in lams:
        tot = []
        for f in folds:
            tr = [i for i in ids if i not in set(f.tolist())]
            X, y = stack(data, tr)
            mdl = Ridge(lam).fit(X, y)
            tot += [r["mse1"] for r in evaluate(mdl, data, f.tolist())]
        if np.mean(tot) < best:
            best, best_lam = np.mean(tot), lam
    X, y = stack(data, ids)
    return Ridge(best_lam).fit(X, y), best_lam


def fit_mlp(data, ids, seed=0, epochs=300):
    rng = np.random.default_rng(seed)
    ids = list(rng.permutation(list(ids)))
    nv = max(1, len(ids) // 5) if len(ids) >= 3 else 0
    val, tr = ids[:nv], ids[nv:]
    X, y = stack(data, tr)
    if val:
        Xv, yv = stack(data, val)
        return MLP(seed=seed, epochs=epochs).fit(X, y, Xv, yv)
    return MLP(seed=seed, epochs=epochs).fit(X, y)


def summarize(rows):
    a = np.array([r["mse1"] for r in rows]); b = np.array([r["mse0"] for r in rows])
    ratio = paired_log_ratio(a, b)
    return dict(n=len(rows), mse_before=boot_mean(b), mse_after=boot_mean(a), ratio_after_over_before=ratio[:3],
                wins=int(np.sum(a < b)), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v29", default="data/gpu/v29/baseline")
    ap.add_argument("--mlps", default="data/gpu/mlps")
    ap.add_argument("--train-idx", default="20-99")
    ap.add_argument("--test-idx", default="0-19")
    ap.add_argument("--out", default="data/gpu/corrector")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    data = {}
    for i in parse_idx(a.train_idx) + parse_idx(a.test_idx):
        d = features_for(a.v29, a.mlps, i)
        if d is not None:
            data[i] = d
    train = [i for i in parse_idx(a.train_idx) if i in data]
    test = [i for i in parse_idx(a.test_idx) if i in data and i not in set(train)]
    out = dict(features=FEATS, train_idx=train, test_idx=test, n_train_neurons=int(sum(len(data[i]["r"]) for i in train)))
    if len(train) < 3:
        out["error"] = f"need >= 3 training networks with V29 features, have {len(train)}"
        write_json(os.path.join(a.out, "summary.json"), out); print(json.dumps(out)); return
    # grouped CV on the training networks
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(train), min(a.folds, len(train)))
    cv = {"ridge": [], "mlp": []}
    for f in folds:
        f = f.tolist()
        tr = [i for i in train if i not in set(f)]
        mdl, lam = fit_ridge_cv(data, tr)
        cv["ridge"] += evaluate(mdl, data, f)
        cv["mlp"] += evaluate(fit_mlp(data, tr, epochs=a.epochs), data, f)
    out["cv"] = {k: summarize(v) for k, v in cv.items()}
    # final fit on all training networks, evaluated on the test networks
    ridge, lam = fit_ridge_cv(data, train)
    mlp = fit_mlp(data, train, epochs=a.epochs)
    out["ridge_lambda"] = lam
    out["test"] = {"ridge": summarize(evaluate(ridge, data, test)) if test else None,
                   "mlp": summarize(evaluate(mlp, data, test)) if test else None}
    # ridge coefficients (standardized) for interpretation
    out["ridge_coef_std"] = dict(zip(FEATS, ridge.w.tolist()))
    out["seconds"] = time.time() - t0
    write_json(os.path.join(a.out, "summary.json"), out)
    write_json(os.path.join(a.out, "model_ridge.json"), dict(features=FEATS, **ridge.export()))
    write_json(os.path.join(a.out, "model_mlp.json"), dict(features=FEATS, **mlp.export()))
    brief = {k: {kk: (vv["ratio_after_over_before"] if vv else None) for kk, vv in out[k].items()} for k in ("cv", "test")}
    print(json.dumps(dict(train=len(train), test=len(test), ratios=brief, lam=lam, seconds=out["seconds"])), flush=True)


if __name__ == "__main__":
    main()
