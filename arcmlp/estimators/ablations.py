"""Round-3 ablations of the dense Kt=4 cap-3 reference by per-layer term filters (no extra memory).

Written by an LLM agent (Claude Code), 2026-10-18.

GenProp(A):  kappa4 is born only at layers <= A. At layers > A every order-4 output term whose inputs
             contain no kappa4 block (a newborn) is dropped; transmission and memory-modified terms
             stay. Measures how many kappa4 source generations a factored chain must carry.
FamilyProp:  the cost-3 diagrams split into families by block content, each switchable:
               rs   a single random-sign kappa4 block ((3,1), (2,1,1), (1,1,1,1))
               k3e  a kappa3 block plus one covariance edge
               p4e  a paired kappa4 block ((4,) or (2,2)) plus one covariance edge
               c3   three covariance edges
             (cost <= 2 terms are always on). Measures which families the factored chain must carry.
"""
from __future__ import annotations

from arcmlp.reference import dense_cumulants as dc


def block_kind(v):
    k = sum(v)
    if k == 2:
        return "edge"
    if k == 3:
        return "k3"
    return "p4" if all(x % 2 == 0 for x in v) else "rs"


def family(t):
    """Family of a cost-3 term, None for cost <= 2 terms."""
    if t.cost < 3:
        return None
    kinds = sorted(block_kind(v) for v in t.diagram)
    if kinds == ["rs"]:
        return "rs"
    if kinds == ["edge", "k3"]:
        return "k3e"
    if kinds == ["edge", "p4"]:
        return "p4e"
    if kinds == ["edge", "edge", "edge"]:
        return "c3"
    return "other"


RS_OUT = {(3, 1), (2, 1, 1), (1, 1, 1, 1)}


def is_newborn_k4(t):
    """A random-sign kappa4 birth: order-4 output on a random-sign slice with no kappa4 input block.
    (Coherent births, outputs (4,) and (2,2), are kept at every layer: they are cheap in a factored
    chain and dropping them destabilises the dense chain — round 3 of 2026-10-18.)"""
    return sum(t.p) == 4 and t.p in RS_OUT and not any(sum(v) == 4 for v in t.diagram)


def family_out(t):
    """Family refined by output order: '<family>_4' writes kappa4 (a memory birth that a factored
    chain must transport), '<family>_3' writes orders <= 3 (a per-layer contraction)."""
    f = family(t)
    return None if f is None else f"{f}_{'4' if sum(t.p) == 4 else '3'}"


class LayerTermProp(dc.DenseProp):
    """DenseProp whose active term list may depend on the layer."""

    def __init__(self, device=None):
        super().__init__(Kt=4, cap=3, device=device)
        self.all_terms = list(self.active)

    def terms_for_layer(self, layer):
        return self.all_terms

    def relu_step(self, Z, layer=0):
        self.active = self.terms_for_layer(layer)
        return super().relu_step(Z, layer)


class GenProp(LayerTermProp):
    def __init__(self, A, device=None):
        super().__init__(device=device)
        self.A = A
        self._late = [t for t in self.all_terms if not is_newborn_k4(t)]

    def terms_for_layer(self, layer):
        return self.all_terms if layer < self.A else self._late


class FamilyProp(LayerTermProp):
    """families_on: names from ALL_FAMILIES (whole families) and/or ALL_FAMILIES_OUT (by output order)."""

    def __init__(self, families_on, device=None):
        super().__init__(device=device)
        fams = set(families_on)
        self.active = self.all_terms = [t for t in self.all_terms
                                        if family(t) is None or family(t) in fams or family_out(t) in fams]


ALL_FAMILIES = ("rs", "k3e", "p4e", "c3", "other")
ALL_FAMILIES_OUT = ("rs_4", "rs_3", "k3e_4", "k3e_3", "p4e_4", "p4e_3", "c3_4", "c3_3")
