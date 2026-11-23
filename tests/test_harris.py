"""Harris-Foulkes energy (review M3, open issue O-3): independent recomputation and convergence to the KS energy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
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


def _setup(case: str, **overrides):
    c = load_case_json(ROOT / "manifests" / "cases" / case / "case.json")
    structure = c["structure"]
    config = replace(c["config"], mesh_cutoff_ry=40.0, enable_adaptive_controller=False, **overrides)
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species)
    pattern = build_sparse_pattern(structure, basis)
    two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    k = (1, 1, 1) if case == "h2o_gamma" else (2, 2, 2)
    kmesh = build_monkhorst_pack_mesh(k)
    mesh = choose_mesh_dimensions(structure.cell, config.mesh_cutoff_ry)
    return structure, basis, pattern, two_center, kmesh, config, mesh


@pytest.mark.parametrize("mix_target", ["DM", "Hamiltonian"])
@pytest.mark.parametrize("case", ["h2o_gamma", "co_hcp_collinear"])
def test_harris_energy_equals_independent_recomputation(case: str, mix_target: str, record_residuals) -> None:
    """E_Harris = E_KS[D_in] + Tr[H_in (D_out - D_in)] + entropy, with E_KS[D_in] from the post-SCF evaluator."""
    structure, basis, pattern, tc, kmesh, config, mesh = _setup(
        case, mix_target=mix_target, max_scf_iterations=3, mixer_variant="pulay"
    )
    state, mesh = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh)
    assert state.harris_energy_ha is not None
    assert state.harris_provenance == ("exact_dm_mixing" if mix_target == "DM" else "siesta_h_mixing")

    # E_KS[D_in]: evaluate the energy terms on D_in (no entropy), then add Tr[H_in (D_out - D_in)] and -TS.
    at_d_in = replace(state, dm_out=state.dm_in, entropy_ha=jnp.array(0.0))
    e_ks_in = float(
        StaticObservableEvaluator()
        .evaluate(structure, basis, pattern, tc, kmesh, at_d_in, config, mesh_shape=mesh, compute_forces_stress=False)
        .total_energy_ha
    )
    h_in = np.asarray(jnp.real(state.h_sparse))
    d_diff = np.asarray(jnp.real(state.dm_out - state.dm_in))
    trace = float(np.sum(h_in * d_diff))  # nspin 1: one channel; nspin 2: sum over both spin channels
    harris_ref = e_ks_in + trace + float(state.entropy_ha)
    resid = abs(float(state.harris_energy_ha) - harris_ref)
    record_residuals({f"{case}.{mix_target}": resid}, {f"{case}.{mix_target}": 1e-8})
    assert resid < 1e-8, (float(state.harris_energy_ha), harris_ref)
    # The Harris and KS(D_out) energies differ before convergence: the field is not an alias (review M3).
    assert abs(float(state.harris_energy_ha) - float(state.last_scf_free_e_ha)) > 1e-7


@pytest.mark.parametrize("mix_target", ["DM", "Hamiltonian"])
def test_harris_converges_to_the_ks_free_energy(mix_target: str) -> None:
    structure, basis, pattern, tc, kmesh, config, mesh = _setup(
        "h2o_gamma", mix_target=mix_target, max_scf_iterations=100, d_tol=1e-8, h_tol_ha=1e-8, e_tol_ha=1e-10
    )
    state, mesh = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh)
    assert state.converged
    obs = StaticObservableEvaluator().evaluate(
        structure, basis, pattern, tc, kmesh, state, config, mesh_shape=mesh, compute_forces_stress=False
    )
    assert obs.harris_provenance == state.harris_provenance
    assert abs(float(obs.harris_energy_ha) - float(obs.final_post_scf_free_e_ha)) < 1e-7
