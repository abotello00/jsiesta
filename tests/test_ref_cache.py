"""Replay semantics of the SIESTA reference cache for directory runs (``run_siesta_directory``)."""

from __future__ import annotations

from pathlib import Path
import stat

import pytest

from jax_siesta.testing.siesta_adapter import run_siesta_directory


def _fake_siesta(path: Path, counter: Path) -> None:
    path.write_text(f"#!/bin/sh\necho run >> {counter}\ncat > /dev/null\necho 'siesta: fake output'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _prepare(workdir: Path, fdf: str) -> None:
    workdir.mkdir()
    (workdir / "input.fdf").write_text(fdf, encoding="utf-8")
    (workdir / "X.psml").write_text("<psml/>", encoding="utf-8")


def test_directory_run_is_cached_by_input_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical inputs replay the stored output without the executable; changed inputs run it again."""
    counter = tmp_path / "count.txt"
    exe = tmp_path / "fake_siesta"
    _fake_siesta(exe, counter)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("JSIESTA_SIESTA_REF_CACHE", str(cache))
    monkeypatch.setenv("JSIESTA_SIESTA_BIN", str(exe))

    _prepare(tmp_path / "a", "MeshCutoff 100 Ry\n")
    out = run_siesta_directory(tmp_path / "a", "input.fdf", stdout_name="out.txt")
    assert "fake output" in out.read_text()
    assert counter.read_text().count("run") == 1

    # Same inputs, executable gone: replay from the cache.
    monkeypatch.setenv("JSIESTA_SIESTA_BIN", str(tmp_path / "missing"))
    _prepare(tmp_path / "b", "MeshCutoff 100 Ry\n")
    out_b = run_siesta_directory(tmp_path / "b", "input.fdf", stdout_name="out.txt")
    assert out_b.read_text() == out.read_text()
    assert counter.read_text().count("run") == 1

    # Different FDF: a cache miss without an executable is an error, not a silent pass.
    _prepare(tmp_path / "c", "MeshCutoff 200 Ry\n")
    with pytest.raises(FileNotFoundError):
        run_siesta_directory(tmp_path / "c", "input.fdf", stdout_name="out.txt")
