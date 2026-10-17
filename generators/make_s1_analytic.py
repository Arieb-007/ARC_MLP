"""Generate submission_analytic/estimator.py from submission/estimator.py (V29 + leaf 8) by
asserted textual patches (each must match exactly once).

Change: the kappa4 regeneration coefficient lambda_l (V29: fitted 16-entry LAM table x 0.95 scale,
15-entry REF_R table, exponent BETA, all fitted on public MLPs) is replaced by the tree-level
analytic value of arcmlp/closure/kappa4_lambda.py, computed online from the layer's own
pre-activation statistics (alpha, sigma, Phi, phi, C_pre off-diagonal) and post-ReLU covariance.
Kill-switch: env ARC_LAMBDA=table restores the V29 behaviour bit-for-bit. ARC_LAMBDA_DEBUG=1
records both the analytic value and the V29 table/adapted value per layer in DEBUG.
"""
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC).read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)


# 0) header note
sub("# except one constant: STRASSEN_MIN default 32 -> 8 (smallest Strassen-Winograd leaf block).\n",
    "# except one constant: STRASSEN_MIN default 32 -> 8 (smallest Strassen-Winograd leaf block),\n"
    "# and (ARC_MLP Stage 1) the fitted kappa4 regeneration tables LAM / REF_R / BETA replaced by an\n"
    "# analytic per-layer lambda (see _lambda_analytic; derivation in the ARC_MLP repo,\n"
    "# arcmlp/closure/kappa4_lambda.py and docs/FINDINGS.md F3.2).\n")

# 1) switches
sub('BETA = float(_os.environ.get("V25_BETA", "1.0"))\n',
    'BETA = float(_os.environ.get("V25_BETA", "1.0"))\n'
    '# ARC_MLP Stage 1: analytic lambda (default) or V29 tables ("table")\n'
    'ANALYTIC = _os.environ.get("ARC_LAMBDA", "analytic") != "table"\n'
    'LAM_DEBUG = _os.environ.get("ARC_LAMBDA_DEBUG", "0") == "1"\n'
    'RHO_PROBES = int(_os.environ.get("ARC_RHO_PROBES", "16"))\n')

# 2) adaptive rule only in table mode (it uses the fitted REF_R table)
sub("                    if BETA != 0.0:\n                        # V25: adaptive lambda.",
    "                    if LAM_DEBUG and ANALYTIC is False:\n"
    "                        DEBUG.append(dict(layer=li, lam_table=lam_prev))\n"
    "                    if BETA != 0.0 and not ANALYTIC:\n                        # V25: adaptive lambda.")
sub("                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)\n",
    "                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)\n"
    "                        if LAM_DEBUG:\n"
    "                            DEBUG.append(dict(layer=li, lam_adapted=float(lam_prev)))\n")

# 3) the regeneration coefficient at the end of each layer
sub("                lam_prev = float(LAM[min(li, len(LAM) - 1)])\n",
    "                if ANALYTIC or LAM_DEBUG:\n"
    "                    lam_an, rho3_cur = self._lambda_analytic(alpha, sigma, phi, Phi, C_off, C, K2v, n,\n"
    "                                                             rho3_cur, mlp.weights[(li + 1) % L])\n"
    "                    if LAM_DEBUG:\n"
    "                        DEBUG.append(dict(layer=li, lam_analytic=lam_an, rho3=rho3_cur))\n"
    "                lam_prev = lam_an if ANALYTIC else float(LAM[min(li, len(LAM) - 1)])\n")

# 4) init rho3 (C_prev = I at layer 0 -> rho3 = 1)
sub("        lam_prev = 0.0\n",
    "        lam_prev = 0.0\n        rho3_cur = 1.0  # ARC_MLP: tr(C^3)/tr(C^2) of the previous post-ReLU covariance (I at layer 0)\n")

# 5) the method
sub("    # ------------------------------------------------------------------\n    def _sym_product(",
    '''    # ------------------------------------------------------------------
    @staticmethod
    def _lambda_analytic(alpha, sigma, phi, Phi, C_off, C, K2v, n, rho3_prev, Wnext):
        """ARC_MLP Stage 1: tree-level kappa4 regeneration coefficient (no fitted constants).

        G_ij = c_A sum_k kappa4(h_i,h_j,h_k,h_k), c_A = 6/(n+4), regressed on C_post,off:
          lambda = c_A <A + B + D, C_post,off> / ||C_post,off||^2  with, at leading order in
          the pre-activation correlations c_ij (C_off),
          D_ij = c_ij [Phi_j k3'_i + Phi_i k3'_j]   ((3,1) slice; k3' = d_m kappa3(ReLU))
          B_ij = c_ij [(phi_i/s_i) Phi_j (C_off beta)_i + (i<->j)]
          A_ij ~ 2 mean(gamma) rho3 Phi_i Phi_j c_ij   (coherent k-sum, random-W mean field)
        gamma = 2[Phi(1-Phi) - mu phi/s], beta = 2 mu (1-Phi), k3' = 3(1-Phi)(E h^2 - 2 (E h)^2),
        Gaussian ReLU moments of the pre-activation (m = alpha*s, s). rho3 = tr(C^3)/tr(C^2) of the
        previous post-ReLU covariance; returned for the next layer from a Hutchinson estimate with
        probes = columns of the NEXT layer's weights (independent of C). Cost ~0.06 units/layer.
        """
        if C_off is None:
            return 0.0, rho3_prev
        m = alpha * sigma
        omP = 1.0 - Phi
        E1 = m * Phi + sigma * phi
        E2 = (m * m + sigma * sigma) * Phi + m * sigma * phi
        k3p = (3.0 * omP) * (E2 - 2.0 * E1 * E1)
        gam = 2.0 * (Phi * omP - E1 * phi / sigma)
        beta = 2.0 * E1 * omP
        Y = C * C_off                       # C_off has a zero diagonal -> off-diagonal only
        YPhi = Y @ Phi
        btil = C_off @ beta
        num_d = 2.0 * fnp.sum(k3p * YPhi)
        num_b = 2.0 * fnp.sum((phi / sigma) * btil * YPhi)
        prox = fnp.sum(Phi * YPhi)
        num_a = 2.0 * fnp.mean(gam) * rho3_prev * prox
        tr2 = fnp.sum(C * C)
        den = tr2 - fnp.sum(K2v * K2v)
        cA = 6.0 / (n + 4.0)
        lam = float(cA * (num_a + num_b + num_d) / den)
        if not (lam == lam) or lam < 0.0 or lam > 1.0:   # NaN / absurd -> zero (no regen)
            lam = 0.0
        p = min(RHO_PROBES, n)
        Z = fnp.asarray(Wnext)[:, :p]
        V = C @ Z
        U = C @ V
        tr3 = fnp.sum(V * U) * (n / (2.0 * p))
        rho3 = float(tr3 / tr2)
        if not (rho3 == rho3) or rho3 <= 0.0:
            rho3 = 1.0
        return lam, rho3

    # ------------------------------------------------------------------
    def _sym_product(''')

open(DST, "w").write(s)
print("wrote", DST)
