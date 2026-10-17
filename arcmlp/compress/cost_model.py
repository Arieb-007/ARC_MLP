"""W5 cost model: FLOP cost at n = 1024, L = 16 of a selected set of diagram terms in a factorized
representation (research model, see docs/notes/w5_cost_model.md). Written by an LLM agent.

Unit = one dense n x n x n matmul = 2 n^3 FLOPs; budget B = 1024 units; scoring floor ~100 units.
Sources: one third-cumulant (and, if used, fourth-cumulant) source is born per layer, carried as an
n x n leg P (identity at birth, transported P <- T P each later layer). Live source-layers over the
network: S = sum_{l=1}^{L-1} l = L(L-1)/2 = 120 at L = 16.

Costs are attached to FAMILIES (shared machinery), not to individual terms; a term needs the
families of every input slice it reads and of the output slice it writes (if that output is a
factored, multi-index object). Within the cap all per-term work is elementwise / O(n^2) (0 units).

  family            what it buys                                         raw units
  base              mean + dense covariance transport W C W^T              2 / layer = 32
  k3_legs           kappa3 sources: leg transport (1 / source-layer)        S
  k3_21             D21 = sum_s (P o P) diag(a) P^T  (1 / source-layer)     S
  k4_22_exact       (P o P) diag(b) (P o P)^T        (1 / source-layer)     S   (+ k3_legs)
  k4_31             (P o P o P) diag(b) P^T          (1 / source-layer)     S   (+ k3_legs)
  k4_211            three-index slice, two contractions per source-layer   2S  (+ k3_legs)
  k4_top_newborn    new kappa4 leg family for (2,1,1)/(1,1,1,1) outputs     S
  k4_diag           (4,) slice (W o W o W o W) b from per-neuron b            0
  k4_coherent       rank-one (2,2) (4 bbar/n + 4 Bbar)                      0
  k4_22_layer       exact (2,2) of the current layer's diagonal source only  L (one unit per layer)
  (3,)/(4,) diagonals, covariance edges, rank-one terms                     0
Calibration [measured externally, 504aldo]: V29 implements base + k3_legs + k3_21 with an exact
7 -> 4 collapse (4 units per source-layer, 480 raw) and bills 260 units after shared-basis
compression and Strassen: engineering factor ENG = 260 / (32 + 4 * 120) = 0.51. Predicted billed
cost = ENG * raw. This model is a hypothesis; only V29's point calibrates it.
"""
from __future__ import annotations

L_DEFAULT = 16
ENG = 260.0 / (32 + 4 * 120)


def source_layers(L=L_DEFAULT):
    return L * (L - 1) // 2


def family_cost(fam, L=L_DEFAULT):
    S = source_layers(L)
    return {"base": 2 * L, "k3_legs": S, "k3_21": S, "k4_22_exact": S, "k4_31": S, "k4_211": 2 * S,
            "k4_top_newborn": S, "k4_coherent": 0, "k4_diag": 0, "k4_22_layer": L}[fam]


def term_families(term):
    """Families a dense-registry Term needs in a factorized implementation."""
    fams = {"base"}
    for v in term.diagram:
        k = sum(v)
        st = tuple(sorted((x for x in v if x), reverse=True))
        if k == 3:
            fams.add("k3_legs")
            if st == (2, 1):
                fams.add("k3_21")
        elif k == 4:
            if st == (4,):
                fams.add("k4_diag")            # (W o W o W o W) b: O(n^2), no legs
            else:
                fams.add("k3_legs")            # kappa4 sources ride on the kappa3 legs
                fams.add({(2, 2): "k4_22_exact", (3, 1): "k4_31", (2, 1, 1): "k4_211",
                          (1, 1, 1, 1): "k4_211"}[st])
    order = sum(term.p)
    if order == 3 and len(term.p) == 3:
        fams.add("k3_legs")                    # writes a kappa3 source
    if order == 4 and len(term.p) >= 3:
        fams.add("k4_top_newborn")             # writes a factored kappa4 object
    return fams


def selection_cost(terms, L=L_DEFAULT, billed=True):
    fams = set()
    for t in terms:
        fams |= term_families(t)
    raw = sum(family_cost(f, L) for f in fams)
    return (ENG * raw if billed else raw), sorted(fams)


# W5 features (experiments/w5_selection.py): the families a feature needs.  'light' kappa4 features are
# the coherent O(n^2) values; 'dense' kappa4 features read the exact transported kappa4 of z, which
# needs the whole factored kappa4 machinery (an upper-bound model: any exact kappa4 read is billed for
# all kappa4 leg families).
K4_ALL = ["k3_legs", "k4_22_exact", "k4_31", "k4_211", "k4_top_newborn"]
FEATURE_FAMILIES = {
    "light": {"Cpair": [], "cap3": [], "K3(3,)": ["k3_legs"], "K3(1,1,1)": ["k3_legs"],
              "K3(2,1)": ["k3_legs", "k3_21"], "K4(4,)": ["k4_diag"], "K4(2,2)": ["k4_coherent"]},
    "dense": {"Cpair": [], "cap3": [], "K3(3,)": ["k3_legs"], "K3(1,1,1)": ["k3_legs"],
              "K3(2,1)": ["k3_legs", "k3_21"], "K4(4,)": K4_ALL, "K4(2,2)": K4_ALL, "K4(3,1)": K4_ALL,
              "K4(2,1,1)": K4_ALL, "K4(1,1,1,1)": K4_ALL},
}


def feature_cost(feats, tier="light", L=L_DEFAULT, billed=True):
    fams = {"base"}
    for f in feats:
        fams |= set(FEATURE_FAMILIES[tier][f])
    raw = sum(family_cost(f, L) for f in fams)
    return ENG * raw if billed else raw
