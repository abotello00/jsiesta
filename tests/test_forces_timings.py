"""Stage timers of the force evaluation (``timings_out``): complete, and without effect on the results."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parent
PSML_DIR = ROOT / "fixtures" / "psml"
STAGES = ("fresh_grid", "kb", "dispersion", "grid_forward", "grid_backward", "two_center", "moments", "total")


@pytest.mark.parametrize("xc", ["PBE", "R2SCAN"])
def test_timings_out_reports_every_stage_and_leaves_forces_unchanged(xc: str) -> None:
    c = load_case_json(ROOT / "manifests" / "cases" / "h2o_gamma" / "case.json")
    structure = c["structure"]
    config = replace(
        c["config"], xc_functional=xc, mesh_cutoff_ry=40.0, max_scf_iterations=3, enable_adaptive_controller=False
    )
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species)
    pattern = build_sparse_pattern(structure, basis)
    tc = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    kmesh = build_monkhorst_pack_mesh((1, 1, 1))
    mesh = choose_mesh_dimensions(structure.cell, config.mesh_cutoff_ry)
    state, mesh = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh)

    ev = StaticObservableEvaluator()
    plain = ev.evaluate(structure, basis, pattern, tc, kmesh, state, config, mesh_shape=mesh)
    stages: dict[str, float] = {}
    timed = ev.evaluate(structure, basis, pattern, tc, kmesh, state, config, mesh_shape=mesh, timings_out=stages)

    assert set(STAGES) <= set(stages)
    assert all(v >= 0.0 for v in stages.values())
    assert stages["total"] >= max(v for k, v in stages.items() if k != "total")
    for field in ("forces_ha_per_bohr", "stress_ha_per_bohr3", "total_energy_ha"):
        np.testing.assert_array_equal(np.asarray(getattr(plain, field)), np.asarray(getattr(timed, field)))
