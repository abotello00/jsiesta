"""Loader and registration helper for compiled C++/CUDA XLA FFI kernels.

Theory:
    Bridges JAX XLA compilation with compiled C++/CUDA kernels in
    ``libjsiesta_ffi`` for performance-critical bounded-sphere mesh loops
    and exchange-correlation evaluations.

Algorithm:
    Locates ``libjsiesta_ffi.dylib`` / ``libjsiesta_ffi.so`` under ``cpp/build/``
    (or ``JSIESTA_FFI_LIB``), caches the ``ctypes.CDLL`` handle, and registers
    requested FFI symbols once with ``jax.extend.ffi`` (or ``jax.ffi``).

Invariants:
    - Idempotent target registration across repeated imports or JIT traces.
    - Exposes compile-time ``has_libxc`` and ``ffi_version`` metadata.

SIESTA Mapping:
    Infrastructure bridge for ``rhoofd``, ``vmat``, ``dfscf``, and ``libxc``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import jax

try:
    from jax import ffi as _jax_ffi  # type: ignore[attr-defined]
except ImportError:
    from jax.extend import ffi as _jax_ffi  # type: ignore[no-redef]


_REGISTERED_TARGETS: set[str] = set()
_CACHED_LIB: ctypes.CDLL | None = None


def find_ffi_library() -> Path:
    """Locate the compiled ``jsiesta_ffi`` shared library.

    Theory:
        Resolves the compiled C++ FFI shared library without hardcoding build trees.

    Algorithm:
        Checks ``JSIESTA_FFI_LIB`` env var, then ``cpp/build/libjsiesta_ffi.{dylib,so}``.

    Returns:
        Absolute path to the shared library.

    Invariants:
        Raises ``FileNotFoundError`` if no compiled FFI library is found.

    SIESTA Mapping:
        Analogous to runtime plugin/library discovery for C++ FFI kernels.
    """
    env_path = os.environ.get("JSIESTA_FFI_LIB")
    if env_path:
        candidate = Path(env_path).resolve()
        if candidate.is_file():
            return candidate
    repo_root = Path(__file__).resolve().parent.parent.parent
    for name in ("libjsiesta_ffi.dylib", "libjsiesta_ffi.so"):
        candidate = repo_root / "cpp" / "build" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Compiled jsiesta C++ FFI library not found. Build with: cmake -S cpp -B cpp/build && cmake --build cpp/build"
    )


def load_ffi_library() -> ctypes.CDLL:
    """Load and cache the ``jsiesta_ffi`` shared library via ``ctypes``.

    Theory:
        Provides direct ABI access for version checks and FFI capsule registration.

    Algorithm:
        Caches the loaded ``CDLL`` singleton for the process lifetime.

    Returns:
        Loaded ``ctypes.CDLL`` handle.

    Invariants:
        Returned handle exposes ``jsiesta_ffi_version`` and ``jsiesta_has_libxc``.

    SIESTA Mapping:
        FFI library bootstrapper.
    """
    global _CACHED_LIB
    if _CACHED_LIB is None:
        lib = ctypes.CDLL(str(find_ffi_library()))
        lib.jsiesta_ffi_version.restype = ctypes.c_int32
        lib.jsiesta_has_libxc.restype = ctypes.c_int32
        if hasattr(lib, "jsiesta_has_cuda"):
            lib.jsiesta_has_cuda.restype = ctypes.c_int32
        _CACHED_LIB = lib
    return _CACHED_LIB


def register_target(
    symbol_name: str,
    platform: str = "cpu",
    c_symbol: str | None = None,
) -> None:
    """Idempotently register an exported C/CUDA XLA FFI symbol with JAX.

    Theory:
        Binds a C++ ``XLA_FFI_Handler*`` symbol to an XLA custom call target name.

    Algorithm:
        Extracts the symbol pointer via ``ctypes`` and wraps it in ``ffi.pycapsule``.

    Parameters:
        symbol_name: Registered XLA FFI target name.
        platform: Target XLA platform (``"cpu"`` or ``"CUDA"``).
        c_symbol: Optional exported C symbol name in ``libjsiesta_ffi`` (defaults
            to ``symbol_name``).

    Invariants:
        Safe to call repeatedly across multiple JIT compilations.

    SIESTA Mapping:
        Registers FFI handlers for ``rhoofd``, ``vmat``, ``dfscf_grad``, ``na_grid``,
        and ``xc_eval``.
    """
    c_sym = c_symbol or symbol_name
    key = f"{platform}:{symbol_name}:{c_sym}"
    if key in _REGISTERED_TARGETS:
        return
    lib = load_ffi_library()
    fn = getattr(lib, c_sym)
    _jax_ffi.register_ffi_target(symbol_name, _jax_ffi.pycapsule(fn), platform=platform)
    _REGISTERED_TARGETS.add(key)


def register_grid_xc_target(target_name: str) -> None:
    """Register an FFI handler for CPU and (when compiled with CUDA) CUDA GPU backends.

    Theory:
        Binds both host CPU and device CUDA XLA FFI symbols under identical target names
        so JAX traces and executes seamlessly on either backend without Python branching.

    Algorithm:
        Registers ``target_name`` on ``platform="cpu"`` unconditionally and registers
        ``<target_name>_cuda`` on ``platform="CUDA"`` when ``jsiesta_has_cuda() == 1``.

    Parameters:
        target_name: Canonical XLA FFI target name (e.g. ``"jsiesta_rhoofd_f64"``).

    Invariants:
        Idempotent; never invokes CUDA registration on CPU-only builds.

    SIESTA Mapping:
        Unified CPU/CUDA FFI registration for real-space grid and XC evaluators.
    """
    register_target(target_name, platform="cpu")
    lib = load_ffi_library()
    cuda_sym = f"{target_name}_cuda"
    if hasattr(lib, "jsiesta_has_cuda") and bool(lib.jsiesta_has_cuda()) and hasattr(lib, cuda_sym):
        register_target(target_name, platform="CUDA", c_symbol=cuda_sym)


def ffi_ping(x: jax.Array) -> jax.Array:
    """Execute the C++ XLA FFI ``jsiesta_ffi_ping`` round-trip health kernel.

    Theory:
        Evaluates :math:`y = 2x + 1` in compiled C++ via XLA FFI.

    Algorithm:
        Ensures ``jsiesta_ffi_ping`` is registered and dispatches ``ffi_call``.

    Parameters:
        x: 1D float64 JAX array.

    Returns:
        1D float64 JAX array equal to ``2 * x + 1``.

    Invariants:
        Shape and float64 dtype are preserved.

    SIESTA Mapping:
        Infrastructure verification call for the XLA FFI bridge.
    """
    register_target("jsiesta_ffi_ping", platform="cpu")
    call = _jax_ffi.ffi_call(
        "jsiesta_ffi_ping",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    return call(x)


def ffi_backend_info() -> dict[str, Any]:
    """Return compiled FFI library version and backend capabilities.

    Theory:
        Reports compile-time backend configuration (libxc linkage, FFI ABI version).

    Algorithm:
        Queries ``jsiesta_ffi_version`` and ``jsiesta_has_libxc`` C ABI functions.

    Returns:
        Dictionary with ``ffi_version``, ``has_libxc``, and ``library_path``.

    Invariants:
        Pure read-only metadata query.

    SIESTA Mapping:
        Replaces compile-time feature reporting in ``siesta_version_info``.
    """
    lib = load_ffi_library()
    has_cuda = bool(lib.jsiesta_has_cuda()) if hasattr(lib, "jsiesta_has_cuda") else False
    return {
        "ffi_version": int(lib.jsiesta_ffi_version()),
        "has_libxc": bool(lib.jsiesta_has_libxc()),
        "has_cuda": has_cuda,
        "library_path": str(find_ffi_library()),
    }
