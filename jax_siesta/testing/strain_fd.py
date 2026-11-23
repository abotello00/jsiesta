r"""Fixed-``(D, E)`` strain finite differences, term by term, for the analytic stress tensor (open issue O-2).

Theory:
    With the density matrix :math:`D` and energy-weighted matrix :math:`E` held fixed, the stress is
    :math:`\sigma_{ab} = \Omega^{-1}\,\partial \mathcal E / \partial \varepsilon_{ab}` of
    :math:`\mathcal E(\varepsilon) = \mathrm{Tr}[D T] + E_{\text{KB}} + E_{\text{NA}} + E_H + E_{xc} + E_{\text{disp}}
    - \mathrm{Tr}[E S]` under a homogeneous strain that scales the cell and all positions. Each term is a separate
    energy, so a disagreement between analytic and finite-difference values points at one term. This avoids the
    SCF-tolerance and smearing noise of finite differences of a re-converged free energy.

Algorithm:
    For a strain :math:`\varepsilon`, rebuilds the sparse pattern and two-centre integrals of the strained structure
    (the edge topology must not change), then evaluates the grid terms with
    :class:`~jax_siesta.observables.forces.StaticObservableEvaluator` on the same mesh shape and the fixed
    ``scf_state``. A fourth-order central stencil gives :math:`\partial \mathcal E_i / \partial \varepsilon_{ab}`.

Invariants:
    - The mesh shape, k-mesh, ``dm_out`` and ``edm_out`` are identical for every strained evaluation.
    - Symmetric strain: off-diagonal entries use :math:`\varepsilon_{ab} = \varepsilon_{ba} = h/2`.
    - Only the non-symmetrised path (``sym_proj=None``) is supported.

SIESTA Mapping:
    Replaces comparing against the finite difference of SIESTA's converged free energy, which mixes in SCF noise.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import BasisData, KPointMesh, SCFConfig, SCFState, Structure
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.species import build_sparse_pattern

#: Names of the stress components, in the order they are reported.
STRESS_TERMS = ("kinetic", "pulay_overlap", "kb", "na_pair", "na_grid", "hartree", "xc", "dispersion")


def strained_structure(structure: Structure, eps: np.ndarray) -> Structure:
    r"""Return ``structure`` with cell and positions deformed by :math:`(1 + \varepsilon)`.

    Theory:
        Homogeneous strain :math:`\mathbf A \to (1+\varepsilon)\mathbf A`, :math:`\mathbf R \to (1+\varepsilon)\mathbf R`.

    Algorithm:
        Multiplies the cell columns and the Cartesian positions by ``1 + eps``.

    Parameters:
        structure: Unstrained structure.
        eps: ``(3, 3)`` strain matrix.

    Returns:
        New ``Structure``; fractional coordinates are unchanged.

    Invariants:
        ``eps = 0`` returns an identical structure.

    SIESTA Mapping:
        None; a jsiesta-only derivative check.
    """
    def_mat = np.eye(3) + np.asarray(eps, dtype=np.float64)
    return replace(
        structure,
        cell=jnp.asarray(def_mat @ np.asarray(structure.cell), dtype=jnp.float64),
        positions=jnp.asarray(np.asarray(structure.positions) @ def_mat.T, dtype=jnp.float64),
    )


def component_energies_at_fixed_dm(
    structure: Structure,
    basis: BasisData,
    kmesh: KPointMesh,
    scf_state: SCFState,
    config: SCFConfig,
    mesh_shape: tuple[int, int, int],
    eps: np.ndarray,
    reference_pattern: Any = None,
) -> dict[str, float]:
    r"""Energy terms (Ha) of the strained structure at the fixed ``dm_out``/``edm_out`` of ``scf_state``.

    Theory:
        See the module docstring: each returned value is one term of :math:`\mathcal E(\varepsilon)`;
        ``pulay_overlap`` is :math:`-\mathrm{Tr}[E S(\varepsilon)]`.

    Algorithm:
        Strains the structure, rebuilds pattern and two-centre matrices, evaluates the energy terms without
        forces or stress.

    Parameters:
        structure: Unstrained structure.
        basis: Basis (strain independent).
        kmesh: K-point mesh (fractional, strain independent).
        scf_state: Converged state whose ``dm_out`` and ``edm_out`` are held fixed.
        config: SCF configuration.
        mesh_shape: Fixed real-space mesh.
        eps: Symmetric ``(3, 3)`` strain.
        reference_pattern: Unstrained pattern; when given, the strained pattern must have the same edges.

    Returns:
        Mapping ``term -> energy`` for the keys in :data:`STRESS_TERMS`.

    Invariants:
        Raises ``RuntimeError`` if the strained edge topology differs from ``reference_pattern``.

    SIESTA Mapping:
        None; a jsiesta-only derivative check.
    """
    s_eps = strained_structure(structure, eps)
    pattern = build_sparse_pattern(s_eps, basis)
    if reference_pattern is not None and not (
        np.array_equal(np.asarray(pattern.src_orb), np.asarray(reference_pattern.src_orb))
        and np.array_equal(np.asarray(pattern.dst_unit_orb), np.asarray(reference_pattern.dst_unit_orb))
    ):
        raise RuntimeError("Strain changed the sparse-pattern edge topology; use a smaller strain.")
    two_center = TwoCenterEvaluator().evaluate(s_eps, basis, pattern, config)
    obs = StaticObservableEvaluator().evaluate(
        s_eps, basis, pattern, two_center, kmesh, scf_state, config, mesh_shape=mesh_shape, compute_forces_stress=False
    )
    edm = np.asarray(jnp.real(scf_state.edm_out))
    edm_tot = edm[0] if config.nspin == 1 else edm[0] + edm[1]
    return {
        "kinetic": float(obs.e_kinetic_ha),
        "pulay_overlap": -float(np.sum(edm_tot * np.asarray(two_center.overlap))),
        "kb": float(obs.e_kb_ha),
        "na_pair": float(two_center.e_na_pair),
        "na_grid": float(obs.e_na_ha) - float(two_center.e_na_pair),
        "hartree": float(obs.e_hartree_ha),
        "xc": float(obs.e_xc_ha),
        "dispersion": float(obs.e_disp_ha),
    }


def fd_stress_components(
    structure: Structure,
    basis: BasisData,
    kmesh: KPointMesh,
    scf_state: SCFState,
    config: SCFConfig,
    mesh_shape: tuple[int, int, int],
    entries: tuple[tuple[int, int], ...] = ((0, 0), (2, 2)),
    h: float = 1.0e-5,
) -> dict[str, dict[tuple[int, int], float]]:
    r"""Finite-difference stress (Ha/Bohr^3) per term: :math:`\Omega^{-1}\,\partial \mathcal E_i/\partial\varepsilon_{ab}`.

    Theory:
        Fourth-order central difference
        :math:`[-E(2h) + 8E(h) - 8E(-h) + E(-2h)]/(12h)` of each energy term at fixed ``(D, E)``.

    Algorithm:
        Four strained evaluations per requested ``(a, b)`` entry.

    Parameters:
        structure: Unstrained structure.
        basis: Basis.
        kmesh: K-point mesh.
        scf_state: Converged state held fixed.
        config: SCF configuration.
        mesh_shape: Fixed real-space mesh.
        entries: Stress components ``(a, b)`` to evaluate.
        h: Strain step. The discretised energy (spline tables, cut-off radii, mesh) is smooth only at small
            steps: with ``h = 1e-4`` the neutral-atom grid term of a coarse-mesh Co cell is already off by
            1.7e-6 Ha/Bohr^3 and with ``h = 3e-3`` by 7e-6, while at ``h = 1e-5`` every term agrees with the
            analytic stress to 1e-12 (logs/DECISIONS.md D-021).

    Returns:
        ``{term: {(a, b): sigma_ab}}`` for the terms in :data:`STRESS_TERMS`.

    Invariants:
        Off-diagonal entries use a symmetric strain, so they equal :math:`\sigma_{ab}` of the symmetrised tensor.
        Roundoff limits ``h`` from below (energies of order 1e2 Ha give ~1e-11 Ha/Bohr^3 at ``h = 1e-5``).

    SIESTA Mapping:
        None; a jsiesta-only derivative check.
    """
    vol = float(abs(np.linalg.det(np.asarray(structure.cell))))
    ref_pattern = build_sparse_pattern(structure, basis)
    out: dict[str, dict[tuple[int, int], float]] = {t: {} for t in STRESS_TERMS}
    for a, b in entries:
        shape = np.zeros((3, 3))
        shape[a, b] += 0.5 if a != b else 1.0
        shape[b, a] += 0.5 if a != b else 0.0
        energies = {
            k: component_energies_at_fixed_dm(
                structure, basis, kmesh, scf_state, config, mesh_shape, k * h * shape, ref_pattern
            )
            for k in (2, 1, -1, -2)
        }
        for term in STRESS_TERMS:
            d = (-energies[2][term] + 8.0 * energies[1][term] - 8.0 * energies[-1][term] + energies[-2][term]) / (12.0 * h)
            out[term][(a, b)] = d / vol
    return out


def analytic_stress_components(
    structure: Structure,
    basis: BasisData,
    pattern: Any,
    two_center: Any,
    kmesh: KPointMesh,
    scf_state: SCFState,
    config: SCFConfig,
    mesh_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    r"""Analytic per-term stress (Ha/Bohr^3, unsymmetrised ``(3, 3)`` arrays) from the evaluator.

    Theory:
        The terms of :math:`\sigma = \Omega^{-1}(\partial \mathcal E/\partial\varepsilon|_D - \mathrm{Tr}[E\,\partial S/\partial\varepsilon])`
        as assembled by ``StaticObservableEvaluator`` (``components_out``).

    Algorithm:
        Runs the evaluator once with ``components_out`` set.

    Parameters:
        structure: Structure.
        basis: Basis.
        pattern: Sparse pattern of ``structure``.
        two_center: Two-centre matrices of ``structure``.
        kmesh: K-point mesh.
        scf_state: Converged state.
        config: SCF configuration.
        mesh_shape: Real-space mesh.

    Returns:
        ``{term: (3, 3) array}`` for the keys in :data:`STRESS_TERMS`.

    Invariants:
        The sum of the terms equals the evaluator's stress before symmetrisation.

    SIESTA Mapping:
        None; a jsiesta-only derivative check.
    """
    comps: dict[str, np.ndarray] = {}
    StaticObservableEvaluator().evaluate(
        structure, basis, pattern, two_center, kmesh, scf_state, config, mesh_shape=mesh_shape, components_out=comps
    )
    return comps
