"""Reachability / observability Gramian recursions for goal-oriented ranks (Stage 2.4).

P_{l+1} = T_l P_l T_l^T + B_l B_l^T   (what a source can reach: forward)
O_l     = T_l^T O_{l+1} T_l + C_l^T C_l (what the output can see: backward)
Hankel singular values sqrt(eig(P O)) rank (source, layer) pairs for keep-dense / low-rank / drop.
Two matmuls per layer each (no L^2 interval products). Status: tooling; the selection rule is a
hypothesis until Stage 2.4 measures it. Applies to transports and covariance corrections only —
the entrywise cube of F2 needs full rows of Q.
"""
from __future__ import annotations

import numpy as np


def reachability(Ts, Bs, P0=None):
    n = Ts[0].shape[0]
    P = np.zeros((n, n)) if P0 is None else P0
    out = []
    for T, B in zip(Ts, Bs):
        P = T @ P @ T.T + (B @ B.T if B is not None else 0)
        out.append(P)
    return out


def observability(Ts, Cs, OL=None):
    n = Ts[0].shape[0]
    O = np.zeros((n, n)) if OL is None else OL
    out = []
    for T, C in zip(reversed(Ts), reversed(Cs)):
        O = T.T @ O @ T + (C.T @ C if C is not None else 0)
        out.append(O)
    return out[::-1]


def hankel_values(P, O):
    ev = np.linalg.eigvals(P @ O)
    return np.sqrt(np.clip(np.sort(ev.real)[::-1], 0, None))
