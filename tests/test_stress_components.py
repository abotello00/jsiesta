"""Analytic stress terms equal their fixed-(D, E) finite differences (open issue O-2, no external references).

At fixed ``dm_out``/``edm_out`` the stress is the strain derivative of a sum of separate energy terms. Checking each
term against a central finite difference (``jax_siesta.testing.strain_fd``) localises any error in the analytic
derivative: NLCC core-density strain terms, the spin-polarised GGA gradient term, the Pulay overlap term, ...
The identity holds for any ``(D, E)``, so the SCF only needs to run a few iterations.
"""

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
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.testing.strain_fd import (
    STRESS_TERMS,
    analytic_stress_components,
    component_energies_at_fixed_dm,
    fd_stress_components,
)

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parent
PSML_DIR = ROOT / "fixtures" / "psml"

# Ha/Bohr^3. With the default step 1e-5 every term agrees to ~1e-11; larger steps (1e-4 and up) pick up the
# non-smoothness of the discretised energy, not an error of the analytic stress (logs/DECISIONS.md D-021).
TOL: dict[str, float] = {}
DEFAULT_TOL = 1.0e-9


@pytest.mark.parametrize(
    ("case", "nspin", "xc"),
    [("si_bulk", 1, "PBE"), ("co_hcp_collinear", 2, "PBE"), ("co_hcp_collinear", 2, "LDA")],
    ids=["si_nlcc_nspin1_gga", "co_nlcc_nspin2_gga", "co_nlcc_nspin2_lda"],
)
def test_stress_terms_match_fixed_dm_finite_differences(case: str, nspin: int, xc: str, record_residuals) -> None:
    c = load_case_json(ROOT / "manifests" / "cases" / case / "case.json")
    structure = c["structure"]
    config = replace(
        c["config"], mesh_cutoff_ry=40.0, xc_functional=xc, max_scf_iterations=4, enable_adaptive_controller=False
    )
    assert config.nspin == nspin
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species)
    pattern = build_sparse_pattern(structure, basis)
    two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    kmesh = build_monkhorst_pack_mesh((2, 2, 2))
    mesh = choose_mesh_dimensions(structure.cell, config.mesh_cutoff_ry)
    state, mesh = run_scf(structure, basis, pattern, two_center, kmesh, config, mesh_shape=mesh)

    entries = ((0, 0), (2, 2), (0, 1))
    analytic = analytic_stress_components(structure, basis, pattern, two_center, kmesh, state, config, mesh)
    fd = fd_stress_components(structure, basis, kmesh, state, config, mesh, entries=entries)

    resid: dict[str, float] = {}
    for term in STRESS_TERMS:
        for a, b in entries:
            sym = 0.5 * (analytic[term][a, b] + analytic[term][b, a])
            resid[f"{term}[{a}{b}]"] = abs(sym - fd[term][(a, b)])
    record_residuals(resid, {k: TOL.get(k.split("[")[0], DEFAULT_TOL) for k in resid})
    bad = {k: v for k, v in resid.items() if v > TOL.get(k.split("[")[0], DEFAULT_TOL)}
    assert not bad, bad
    # The terms must be non-trivial, otherwise the comparison proves nothing.
    assert max(abs(analytic[t][0, 0]) for t in ("kinetic", "pulay_overlap", "xc", "na_pair")) > 1e-4


@pytest.mark.parametrize(
    ("case", "nspin"), [("si_bulk", 1), ("co_hcp_collinear", 2)], ids=["si_nlcc_nspin1", "co_nlcc_nspin2_gga"]
)
def test_forces_match_fixed_dm_finite_differences(case: str, nspin: int, record_residuals) -> None:
    """Analytic forces on a displaced structure equal the fixed-(D, E) finite difference of the energy.

    The perfect-hcp Co parity run has vanishing forces by symmetry, so it never exercises the two-centre and KB
    derivative tables of Co's semicore orbitals; displacing the atoms does. Si/Co atoms are moved off their sites along
    x and z and the force on atom 0 is compared component by component.
    """
    c = load_case_json(ROOT / "manifests" / "cases" / case / "case.json")
    structure = c["structure"]
    shift = np.array([[0.11, -0.07, 0.13], [-0.05, 0.09, -0.12]])
    structure = replace(structure, positions=jnp.asarray(np.asarray(structure.positions) + shift))
    config = replace(
        c["config"], mesh_cutoff_ry=40.0, xc_functional="PBE", max_scf_iterations=4, enable_adaptive_controller=False
    )
    assert config.nspin == nspin
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species)
    pattern = build_sparse_pattern(structure, basis)
    two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    kmesh = build_monkhorst_pack_mesh((2, 2, 2))
    mesh = choose_mesh_dimensions(structure.cell, config.mesh_cutoff_ry)
    state, mesh = run_scf(structure, basis, pattern, two_center, kmesh, config, mesh_shape=mesh)
    forces = np.asarray(
        StaticObservableEvaluator()
        .evaluate(structure, basis, pattern, two_center, kmesh, state, config, mesh_shape=mesh)
        .forces_ha_per_bohr
    )

    def total_energy(displacement: np.ndarray) -> float:
        moved = replace(structure, positions=jnp.asarray(np.asarray(structure.positions) + displacement))
        return sum(component_energies_at_fixed_dm(moved, basis, kmesh, state, config, mesh, np.zeros((3, 3))).values())

    h = 1.0e-5
    resid: dict[str, float] = {}
    for axis in (0, 2):
        d = np.zeros_like(shift)
        d[0, axis] = h
        d2 = 2.0 * d
        fd = -(-total_energy(d2) + 8.0 * total_energy(d) - 8.0 * total_energy(-d) + total_energy(-d2)) / (12.0 * h)
        resid[f"F0[{'xyz'[axis]}]"] = abs(float(forces[0, axis]) - fd)
    record_residuals(resid, {k: 2.0e-8 for k in resid})
    assert max(resid.values()) < 2.0e-8, resid
    assert float(np.max(np.abs(forces))) > 1e-3  # displaced: not trivially zero

