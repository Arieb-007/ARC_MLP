"""kappa4 slices riding on kappa3's transport legs (Stage 2.3; NOT YET WRITTEN).

Design (hypothesis, docs/FINDINGS.md F3.3): same legs Q as kappa3, extra weights b_m = kappa4(h_m);
(4,) is an n^2 contraction, (3,1) and (2,2) one matmul each per source-layer; (2,1,1) only if the
Stage-2.2 ablation says it matters. Gate: Stage 2.2 must show the slices move the final error.
"""
raise NotImplementedError("k4_on_k3_legs: Stage 2.3, gated on the Stage 2.2 ablation")
