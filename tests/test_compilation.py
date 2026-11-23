"""Compilation cost (D-046): a new geometry with the same array shapes compiles nothing; the persistent cache is opt-in."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import logging
from pathlib import Path

import jax
import pytest

from jax_siesta.core.compilation import CACHE_ENV, enable_compilation_cache
from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parent
PSML_DIR = ROOT / "fixtures" / "psml"


@contextmanager
def _count_compilations() -> Iterator[list[str]]:
    """Collect the names of the XLA compilations done inside the block (``jax_log_compiles`` messages)."""
    names: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if msg.startswith("Finished XLA compilation of "):
                names.append(msg.split(" of ", 1)[1].split(" in ", 1)[0])

    handler = _Handler(level=logging.DEBUG)
    loggers = [logging.getLogger(n) for n in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler")]
    levels = [lg.level for lg in loggers]
    previous = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    try:
        yield names
    finally:
        jax.config.update("jax_log_compiles", previous)
        for lg, lvl in zip(loggers, levels, strict=True):
            lg.removeHandler(handler)
            lg.setLevel(lvl)


def _single_point(structure, config, k_grid):  # type: ignore[no-untyped-def]
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species, include_soc_projectors=False)
    pattern = build_sparse_pattern(structure, basis)
    kmesh = build_monkhorst_pack_mesh(k_grid)
    mesh = choose_mesh_dimensions(structure.cell, config.mesh_cutoff_ry, criterion=config.mesh_criterion)
    tc = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    state, mesh = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh)
    obs = StaticObservableEvaluator().evaluate(structure, basis, pattern, tc, kmesh, state, config, mesh_shape=mesh)
    jax.block_until_ready(obs.forces_ha_per_bohr)
    return pattern, obs


def test_new_geometry_with_same_shapes_compiles_nothing() -> None:
    case = load_case_json(ROOT / "manifests" / "cases" / "si_bulk" / "case.json")
    config = replace(case["config"], use_symmetry=False, max_scf_iterations=4)
    structure = case["structure"]
    pattern1, obs1 = _single_point(structure, config, case["k_grid"])
    moved = replace(structure, positions=structure.positions.at[0, 0].add(0.01))
    with _count_compilations() as names:
        pattern2, obs2 = _single_point(moved, config, case["k_grid"])
    assert pattern2.n_edges == pattern1.n_edges  # same shapes, so nothing may be compiled
    assert names == [], f"compiled again for a new geometry: {names}"
    assert abs(float(obs2.forces_ha_per_bohr[0, 0])) > 1e-4  # the displacement produced a force


def test_compilation_cache_is_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CACHE_ENV, raising=False)
    assert enable_compilation_cache() is None
    keys = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
    )
    saved = {k: getattr(jax.config, k) for k in keys}
    try:
        monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
        cache_dir = enable_compilation_cache()
        assert cache_dir == (tmp_path / "cache").resolve() and cache_dir.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(cache_dir)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    finally:
        for k, v in saved.items():
            jax.config.update(k, v)
