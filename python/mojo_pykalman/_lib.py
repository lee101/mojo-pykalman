"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYKALMAN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pykalman.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mpk_filter": ([I] * 26, I),
    "mpk_smooth": ([I] * 12, I),
    "mpk_smooth_pair": ([I] * 5, I),
    "mpk_em_observation_matrix_stats": ([I] * 11, I),
    "mpk_em_transition_matrix_stats": ([I] * 9, I),
    "mpk_em_observation_covariance": ([I] * 12, I),
    "mpk_em_transition_covariance": ([I] * 10, I),
    "mpk_em_transition_offset": ([I] * 6, I),
    "mpk_em_observation_offset": ([I] * 9, I),
}

_LIBRARY: ctypes.CDLL | None = None


def build() -> str:
    if os.path.exists(LIB):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def f64(value) -> np.ndarray:
    source = np.asanyarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be represented by the float64 kernels")
    if np.issubdtype(source.dtype, np.floating) and source.dtype.itemsize > 8:
        raise TypeError(
            f"{source.dtype} values would be silently narrowed to float64"
        )
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError(f"unsupported FFI buffer dtype {value.dtype}")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    address = int(value.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address
