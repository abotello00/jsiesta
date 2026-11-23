r"""Persistent compilation cache for jsiesta's compiled functions.

Theory:
    XLA compiles every distinct (function, array shapes, static settings) once per process. A persistent cache keyed
    on the lowered program lets later processes load the executable instead of compiling it again; the program does
    not depend on the geometry (D-046: geometry enters as arguments), so every run with the same shapes hits.

Algorithm:
    Points ``jax_compilation_cache_dir`` at a directory and lowers the thresholds to zero, so that the many short
    compilations of a run are cached too (JAX's default keeps only compilations longer than one second).
    :func:`enable_compilation_cache` is called on import of :mod:`jax_siesta` when ``JSIESTA_COMPILATION_CACHE`` names
    a directory; it can also be called directly.

Invariants:
    Opt-in: without the environment variable or an explicit call, JAX's configuration is left unchanged. Results do
    not depend on whether an executable came from the cache.

SIESTA Mapping:
    None; SIESTA is compiled ahead of time.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax

#: Environment variable naming the persistent compilation cache directory (created if missing).
CACHE_ENV = "JSIESTA_COMPILATION_CACHE"


def enable_compilation_cache(path: str | os.PathLike[str] | None = None) -> Path | None:
    """Enable JAX's persistent compilation cache for every compilation of the process.

    Theory:
        See the module docstring: compiled executables are reused across processes with the same shapes.

    Algorithm:
        Resolves the directory (argument, else ``$JSIESTA_COMPILATION_CACHE``), creates it, and sets
        ``jax_compilation_cache_dir``, ``jax_persistent_cache_min_compile_time_secs = 0`` and
        ``jax_persistent_cache_min_entry_size_bytes = 0``.

    Parameters:
        path: Cache directory; ``None`` uses ``$JSIESTA_COMPILATION_CACHE``.

    Returns:
        The cache directory, or ``None`` when no directory was given or configured (nothing is changed then).

    Invariants:
        Idempotent; affects only compilations after the call.

    SIESTA Mapping:
        None.
    """
    value = os.fspath(path) if path is not None else os.environ.get(CACHE_ENV)
    if not value:
        return None
    cache_dir = Path(value).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir
