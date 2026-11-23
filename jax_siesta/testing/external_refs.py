r"""Resolution of external reference resources (SIESTA binaries, pseudopotentials, pilot runs).

Theory:
    Cross-code parity needs resources outside this repository: SIESTA executables built from
    different branches, the SIESTA test pseudopotentials, and frozen pilot runs. Their location is
    machine specific, so it must never be hardcoded in tests or library code.

Algorithm:
    Every resource has a key (e.g. ``siesta_bin``). The path is resolved in this order:

    1. environment variable ``JSIESTA_<KEY>`` (upper case);
    2. ``tests/external_refs.local.toml`` (git-ignored, per machine);
    3. ``tests/external_refs.toml`` (committed defaults, ``~`` expanded).

    A key with no value resolves to ``None``; tests then skip, or fail under ``--require-refs``.

Invariants:
    Resolution is pure and deterministic for a fixed environment and file contents. No library
    module may embed an absolute user path.

SIESTA Mapping:
    Plays the role of SIESTA's ``Tests/`` harness variables (``SIESTA``, pseudo directories)
    that the reference makefiles take from the environment.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "tests" / "external_refs.toml"
LOCAL_CONFIG = _REPO_ROOT / "tests" / "external_refs.local.toml"

#: Known keys and what they point to. Kept explicit so typos fail loudly.
KNOWN_KEYS: dict[str, str] = {
    "siesta_bin": "Standard SIESTA executable",
    "siesta_sym_bin": "siesta_spglib_full_impl SIESTA executable",
    "siesta_mgga_bin": "pr_mgga SIESTA executable (meta-GGA / SOC)",
    "siesta_pseudos_dir": "SIESTA Tests/Pseudos directory (Si.tm2.nrl.psml, ...)",
    "pr_mgga_tests_dir": "pr_mgga SIESTA Tests directory",
    "co_pilot_dir": "scf_params collinear hcp Co pilot run directory",
    "pseudodojo_sr_pbe_dir": "PseudoDojo nc-sr-05 PBE standard PSML set (scalar relativistic; H, C, O, Si, Co, ...)",
    "pseudodojo_fr_pbe_dir": "PseudoDojo nc-fr-04 PBE standard PSML set (fully relativistic; used below nspin 8 and with SOC)",
    "siesta_doc_index": "SQLite documentation index of a SIESTA source tree (docs.db from rebuild_index.py)",
    "siesta_bin_or_cache": "Standard SIESTA executable, or a SIESTA reference cache (JSIESTA_SIESTA_REF_CACHE)",
    "siesta_mgga_bin_or_cache": "pr_mgga SIESTA executable, or a SIESTA reference cache (JSIESTA_SIESTA_REF_CACHE)",
    "siesta_sym_bin_or_cache": "siesta_spglib_full_impl executable, or a SIESTA reference cache (JSIESTA_SIESTA_REF_CACHE)",
}

#: Environment variable naming a directory of cached SIESTA reference runs (see ``run_siesta_reference``).
REF_CACHE_ENV = "JSIESTA_SIESTA_REF_CACHE"


def ref_cache_dir() -> Path | None:
    """Return the SIESTA reference-cache directory if ``JSIESTA_SIESTA_REF_CACHE`` names an existing one.

    Theory:
        A cache of real SIESTA runs lets cross-code tests replay reference results on machines that cannot run a SIESTA
        binary (for example a GPU cluster).

    Algorithm:
        Reads the environment variable, expands ``~`` and checks that it is a directory.

    Returns:
        The cache directory, or ``None`` when unset or not an existing directory.

    Invariants:
        Pure function of the environment and the file system; never creates the directory.

    SIESTA Mapping:
        None (test-harness utility).
    """
    value = os.environ.get(REF_CACHE_ENV)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_dir() else None



def _read_toml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    refs = data.get("refs", {})
    return {str(k): str(v) for k, v in refs.items() if v not in (None, "")}


def ref_path(key: str) -> Path | None:
    """Resolve the path of an external reference resource.

    Theory:
        See the module docstring: machine-specific resources are configuration, not code.

    Algorithm:
        Environment variable, then the local TOML, then the committed TOML; ``~`` is expanded.

    Parameters:
        key: One of :data:`KNOWN_KEYS`.

    Returns:
        The configured path (which may not exist), or ``None`` if the key is unset everywhere.

    Raises:
        KeyError: If ``key`` is not a known reference key.

    Invariants:
        Returns the same value for the same environment and configuration files.

    SIESTA Mapping:
        Equivalent of reading ``$SIESTA`` / pseudo paths in SIESTA's test makefiles.
    """
    if key not in KNOWN_KEYS:
        raise KeyError(f"Unknown external reference key {key!r}; known: {sorted(KNOWN_KEYS)}")
    env = os.environ.get(f"JSIESTA_{key.upper()}")
    if env:
        return Path(env).expanduser()
    for cfg in (LOCAL_CONFIG, DEFAULT_CONFIG):
        value = _read_toml(cfg).get(key)
        if value:
            return Path(value).expanduser()
    return None


def ref_available(key: str) -> bool:
    """Return whether an external reference resource is configured and exists on disk.

    Theory:
        Availability gates the cross-code tests; a missing resource means the check cannot run.

    Algorithm:
        Resolve with :func:`ref_path` and test existence.

    Parameters:
        key: One of :data:`KNOWN_KEYS`.

    Returns:
        ``True`` if the path is configured and exists.

    Invariants:
        Never raises for a known key, even on permission errors.

    SIESTA Mapping:
        None (test-harness utility).
    """
    if key.endswith("_or_cache"):
        # Tests that obtain SIESTA results only through run_siesta_reference can replay a cached run on machines
        # without a SIESTA binary (e.g. a GPU cluster).
        return ref_cache_dir() is not None or ref_available(key[: -len("_or_cache")])
    path = ref_path(key)
    try:
        return path is not None and path.exists()
    except OSError:
        return False


def missing_refs(keys: tuple[str, ...] | list[str]) -> list[str]:
    """List the keys among ``keys`` whose resources are unavailable.

    Theory:
        Lets a test state all of its external requirements at once and report every missing one.

    Algorithm:
        Filter ``keys`` with :func:`ref_available`.

    Parameters:
        keys: Reference keys required by a test.

    Returns:
        Missing keys, in input order.

    Invariants:
        ``missing_refs(k) == []`` iff every resource in ``k`` exists.

    SIESTA Mapping:
        None (test-harness utility).
    """
    return [k for k in keys if not ref_available(k)]
