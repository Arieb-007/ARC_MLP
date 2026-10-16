"""Strassen-Winograd reference (numpy) with flopscope-style FLOP accounting (Stage 2.7).

strassen(A, B, levels, min_block) computes A @ B with `levels` recursion levels (7 products of
half-size blocks + 15 block additions per level, Winograd variant) and returns (C, flops) where
flops follows flopscope's costing: leaf matmul (2k-1) m n, additions 1 per element.
Use: estimate the C/B of a family at a given depth before writing it in flopscope ops.
"""
from __future__ import annotations

import numpy as np


def _mm_cost(m, k, n):
    return (2 * k - 1) * m * n


def strassen(A, B, levels=1, min_block=32):
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    if levels <= 0 or min(m, k, n) // 2 < min_block or m % 2 or k % 2 or n % 2:
        return A @ B, _mm_cost(m, k, n)
    h, q, v = m // 2, k // 2, n // 2
    A11, A12, A21, A22 = A[:h, :q], A[:h, q:], A[h:, :q], A[h:, q:]
    B11, B12, B21, B22 = B[:q, :v], B[:q, v:], B[q:, :v], B[q:, v:]
    adds = 0
    S1 = A21 + A22; S2 = S1 - A11; S3 = A11 - A21; S4 = A12 - S2; adds += 4 * h * q
    T1 = B12 - B11; T2 = B22 - T1; T3 = B22 - B12; T4 = T2 - B21; adds += 4 * q * v
    f = 0
    P1, c = strassen(A11, B11, levels - 1, min_block); f += c
    P2, c = strassen(A12, B21, levels - 1, min_block); f += c
    P3, c = strassen(S4, B22, levels - 1, min_block); f += c
    P4, c = strassen(A22, T4, levels - 1, min_block); f += c
    P5, c = strassen(S1, T1, levels - 1, min_block); f += c
    P6, c = strassen(S2, T2, levels - 1, min_block); f += c
    P7, c = strassen(S3, T3, levels - 1, min_block); f += c
    U1 = P1 + P2; U2 = P1 + P6; U3 = U2 + P7; U4 = U2 + P5; U5 = U4 + P3; U6 = U3 - P4; U7 = U3 + P5
    adds += 7 * h * v
    C = np.empty((m, n), dtype=np.result_type(A, B))
    C[:h, :v] = U1; C[:h, v:] = U5; C[h:, :v] = U6; C[h:, v:] = U7
    return C, f + adds
