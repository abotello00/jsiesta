"""Shared pytest configuration: external-reference gating and a summary of what did not run.

Tests that need resources outside the repository (SIESTA binaries, pseudopotentials, pilot runs)
declare them with ``@pytest.mark.external_refs("key", ...)``; keys are resolved by
:mod:`jax_siesta.testing.external_refs`. Missing resources skip the test, or fail it under
``--require-refs`` so a CI or cluster run cannot silently pass without any cross-code check.
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
from pathlib import Path
import subprocess
from typing import Callable, Iterator, Mapping

import pytest

from jax_siesta.testing.external_refs import missing_refs, ref_path

_MISSING: dict[str, list[str]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-refs",
        action="store_true",
        default=False,
        help="Fail (instead of skip) tests whose external SIESTA references are unavailable.",
    )


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parent.parent
    res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else ""


@pytest.fixture(scope="session", autouse=True)
def _record_run_identity(record_testsuite_property: Callable[[str, object], None]) -> None:
    """Stamp the JUnit file with commit, dirtiness and platform so the KG can judge whether results are stale."""
    record_testsuite_property("git_commit", _git("rev-parse", "HEAD"))
    record_testsuite_property("git_dirty", "1" if _git("status", "--porcelain") else "0")
    record_testsuite_property(
        "platform", os.environ.get("JSIESTA_PLATFORM") or f"{platform.system().lower()}-{platform.machine()}"
    )


# Resident memory above which the compiled-function caches are dropped after a test (MiB).
_RSS_CLEAR_MIB = float(os.environ.get("JSIESTA_TEST_RSS_CLEAR_MIB", "1500"))


def _rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


@pytest.fixture(autouse=True)
def _bound_process_memory() -> Iterator[None]:
    """Drop JAX's compiled-function caches after a test once the process has grown past ``_RSS_CLEAR_MIB``.

    Every compiled SCF function keeps its executable and the arrays baked into it (grids, symmetry tables) alive
    for the life of the process, so a single-process run of the whole suite otherwise grows by several hundred MiB
    per SCF test and ends in the out-of-memory killer. Small tests stay below the threshold and keep their caches.
    """
    yield
    if _rss_mib() > _RSS_CLEAR_MIB:
        import jax

        jax.clear_caches()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


@pytest.fixture
def record_residuals(record_property: Callable[[str, object], None]) -> Callable[..., None]:
    """Return ``record(residuals, tolerances=None)``: write measured residuals into the JUnit XML.

    Each key becomes ``resid:<key>`` (and ``tol:<key>``) so ``kg/junit_ingest.py`` can attach residuals and
    tolerances to the test in the knowledge graph.
    """

    def record(residuals: Mapping[str, float], tolerances: Mapping[str, float] | None = None) -> None:
        for key, value in residuals.items():
            record_property(f"resid:{key}", f"{float(value):.6e}")
        for key, value in (tolerances or {}).items():
            record_property(f"tol:{key}", f"{float(value):.6e}")

    return record


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: test that runs real SIESTA reference calculations (minutes).")
    config.addinivalue_line(
        "markers",
        "external_refs(*keys): test needs external reference resources (see tests/external_refs.toml).",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    marker = item.get_closest_marker("external_refs")
    if marker is None:
        return
    missing = missing_refs(list(marker.args))
    if not missing:
        return
    _MISSING[item.nodeid] = missing
    detail = ", ".join(f"{k}={ref_path(k)}" for k in missing)
    msg = f"external references unavailable: {detail}"
    if item.config.getoption("--require-refs"):
        pytest.fail(msg + " (--require-refs)", pytrace=False)
    pytest.skip(msg)


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:  # noqa: ANN001
    if not _MISSING:
        return
    terminalreporter.section("cross-code reference tests NOT run")
    for nodeid, keys in sorted(_MISSING.items()):
        terminalreporter.write_line(f"{nodeid}: missing {', '.join(keys)}")
    terminalreporter.write_line(
        "Configure tests/external_refs.local.toml or JSIESTA_<KEY>; use --require-refs to make this fatal."
    )
