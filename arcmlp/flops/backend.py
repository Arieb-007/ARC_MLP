"""numpy <-> flopscope switch (`xp`) and a light FLOP counter that mirrors flopscope's costing
for the ops we use in research code, so numpy-speed experiments can report C/B.

Costs measured on flopscope 0.12.3 (float32; float64 = 2x), see `_selftest()`:
    matmul (m,k)@(k,n)          (2k - 1) m n
    elementwise +,-,*,/,max,    1 per output element (copyto too)
    sqrt, abs, where
    exp                         16 per element
    norm.cdf / norm.pdf         96 per element (scipy-compatible float64 promotion included)
    sum over an axis of len N   (N - 1) per output element
The counter is approximate for anything else; the authoritative number is always flopscope's.

Usage:
    from arcmlp.flops.backend import get_xp, Counter
    xp = get_xp("numpy")          # or "flopscope" (returns flopscope.numpy)
    C = Counter(); C.matmul(m, k, n); C.ew(n_elems); C.units  # 1 unit = 2 * 1024**3 FLOPs
"""
from __future__ import annotations

UNIT = 2 * 1024 ** 3
BUDGET = 2 ** 41


def get_xp(name: str = "numpy"):
    if name == "numpy":
        import numpy as np
        return np
    if name == "flopscope":
        import flopscope.numpy as fnp
        return fnp
    raise ValueError(name)


class Counter:
    def __init__(self, dtype_bytes: int = 4):
        self.flops = 0
        self.f = 1 if dtype_bytes == 4 else 2
        self.log = {}

    def _add(self, key, v):
        v = int(v) * self.f
        self.flops += v
        self.log[key] = self.log.get(key, 0) + v

    def matmul(self, m, k, n, batch=1, key="matmul"):
        self._add(key, batch * (2 * k - 1) * m * n)

    def ew(self, n_elems, per=1, key="ew"):
        self._add(key, n_elems * per)

    def exp(self, n_elems, key="exp"):
        self._add(key, 16 * n_elems)

    def normcdf(self, n_elems, key="cdf"):
        self._add(key, 96 * n_elems)

    def reduce(self, n_out, axis_len, key="reduce"):
        self._add(key, n_out * (axis_len - 1))

    @property
    def units(self):
        return self.flops / UNIT

    @property
    def cb(self):
        return self.flops / BUDGET


def flopscope_cost(fn, *args, **kwargs) -> int:
    """FLOPs flopscope bills for fn(*args) (fn must use flopscope.numpy)."""
    import flopscope as flops
    with flops.BudgetContext(flop_budget=10 ** 18, quiet=True) if _has_quiet() else flops.BudgetContext(flop_budget=10 ** 18) as b:
        fn(*args, **kwargs)
        return int(b.flops_used)


def _has_quiet():
    import inspect
    import flopscope as flops
    try:
        return "quiet" in inspect.signature(flops.BudgetContext).parameters
    except (TypeError, ValueError):
        return False


def _selftest():
    import flopscope as flops
    import flopscope.numpy as fnp
    A = fnp.ones((64, 32), dtype=fnp.float32); B = fnp.ones((32, 16), dtype=fnp.float32)
    c = Counter(); c.matmul(64, 32, 16)
    assert flopscope_cost(lambda: A @ B) == c.flops, "matmul cost model changed"
    c = Counter(); c.ew(64 * 32)
    assert flopscope_cost(lambda: A + A) == c.flops
    c = Counter(); c.exp(64 * 32)
    assert flopscope_cost(lambda: fnp.exp(A)) == c.flops
    c = Counter(); c.reduce(32, 64)
    assert flopscope_cost(lambda: fnp.sum(A, axis=0)) == c.flops
    c = Counter(); c.normcdf(64 * 32)
    assert flopscope_cost(lambda: flops.stats.norm.cdf(A)) == c.flops
    return True


if __name__ == "__main__":
    print("selftest", _selftest())
