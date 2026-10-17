"""Array-module dispatch for the reference chains: numpy on the CPU, CuPy on a CUDA GPU.

Written by an LLM agent (Claude Code). Research code only (never imported by submission/).

Every tensor routine in arcmlp/reference and arcmlp/estimators resolves its array module from the
arrays it is given (`get_array_module`), so the same code runs on either device; `run()` methods
take `device="cpu" | "cuda"` (default from ARCMLP_DEVICE) and create the initial state there.
Closed-form per-neuron quantities (Wick coefficients, ReLU cumulants) stay in float64 numpy on the
host and are moved to the device afterwards. TF32 is never enabled (CUPY_TF32 is forced to "0"):
the chains need full float32 matmuls.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("CUPY_TF32", "0")
_cp = None          # None: not probed; False: unavailable; module: available


def cupy():
    """The cupy module if a CUDA device is usable, else None."""
    global _cp
    if _cp is None:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            _cp = cp
        except Exception:
            _cp = False
    return _cp or None


def resolve_device(device=None):
    device = device or os.environ.get("ARCMLP_DEVICE", "cpu")
    if device == "cuda" and cupy() is None:
        raise RuntimeError("ARCMLP device 'cuda' requested but cupy / a CUDA device is not available")
    return device


def xp_for_device(device=None):
    return cupy() if resolve_device(device) == "cuda" else np


def get_array_module(*arrays):
    cp = _cp if _cp else None
    if cp is not None and any(isinstance(a, cp.ndarray) for a in arrays):
        return cp
    return np


def asnumpy(a):
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get") and not isinstance(a, (int, float)):
        return a.get()
    return np.asarray(a)


def to_device(a, device=None, dtype=None):
    xp = xp_for_device(device)
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


_hwm = 0


def free_pool():
    """Return cached GPU blocks to the device (no-op on the CPU); records the pool high-water mark."""
    global _hwm
    cp = _cp if _cp else None
    if cp is not None:
        pool = cp.get_default_memory_pool()
        _hwm = max(_hwm, pool.total_bytes())
        pool.free_all_blocks()


def device_info(device=None):
    device = resolve_device(device)
    if device != "cuda":
        return dict(device="cpu")
    cp = cupy()
    d = cp.cuda.Device()
    free, total = d.mem_info
    props = cp.cuda.runtime.getDeviceProperties(d.id)
    return dict(device="cuda", name=props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
                mem_total_gb=total / 2 ** 30, mem_free_gb=free / 2 ** 30, cupy=cp.__version__,
                compute_capability=str(d.compute_capability), tf32=os.environ.get("CUPY_TF32"))


def peak_mem_gb(reset=True):
    """Largest pool size seen since the last call (GB), or None on the CPU."""
    global _hwm
    cp = _cp if _cp else None
    if cp is None:
        return None
    v = max(_hwm, cp.get_default_memory_pool().total_bytes()) / 2 ** 30
    if reset:
        _hwm = 0
    return v
