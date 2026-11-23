"""Tests for Tier B 1D radial PAO generation and Junquera Basis Enthalpy optimization."""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core.types import ANGSTROM_TO_BOHR, HARTREE_TO_EV, SCFConfig, Structure
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import run_scf
from jax_siesta.species.basis_opt import (
    EV_PER_BOHR3_PER_GPA,
    BasisEnthalpyObjective,
    BasisTrainingStructure,
    MappedBasisSpace,
    ShellOptConfig,
    basis_enthalpy_ev,
    basis_volume_bohr3,
    default_opt_space_from_psml,
    format_pao_basis_block,
    optimize_basis_enthalpy,
    to_optimizer_u,
    to_physical_x,
)
from jax_siesta.species.pao_generator import (
    ShellBasisSpec,
    SpeciesBasisSpec,
    generate_species_from_psml,
)
from jax_siesta.species.psml import PSMLBasisLoader
from jax_siesta.symmetry.engine import build_native_symmetry
from jax_siesta.testing.external_refs import ref_path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_SI_PSML = _REPO_ROOT / "tests" / "fixtures" / "psml" / "Si.psml"
_REAL_SI_PSML = (ref_path("siesta_pseudos_dir") or Path("/nonexistent")) / "Si.tm2.nrl.psml"


def _si_diamond_structure(a_ang: float = 5.43) -> Structure:
    a_bohr = a_ang * ANGSTROM_TO_BOHR
    cell = jnp.array(
        [
            [0.0, 0.5 * a_bohr, 0.5 * a_bohr],
            [0.5 * a_bohr, 0.0, 0.5 * a_bohr],
            [0.5 * a_bohr, 0.5 * a_bohr, 0.0],
        ],
        dtype=jnp.float64,
    )
    frac = jnp.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=jnp.float64)
    pos = frac @ cell.T
    return Structure(
        cell=cell,
        positions=pos,
        species_indices=(0, 0),
        atomic_numbers=(14, 14),
        species_labels=("Si",),
        initial_moments=jnp.zeros((2, 3), dtype=jnp.float64),
    )


def test_bijective_mapping_zero_invalid_rate_and_float64_safety() -> None:
    """Arccos bijection satisfies strict radius ordering across arbitrary unconstrained draws."""
    shells = {
        "Si": (
            ShellOptConfig(n=3, l=0, n_zeta=3, rc_guess=6.0, rc_initials=(6.0, 4.4, 3.0)),
            ShellOptConfig(n=3, l=1, n_zeta=2, rc_guess=6.8, rc_initials=(6.8, 4.9)),
            ShellOptConfig(
                n=3,
                l=2,
                n_zeta=1,
                rc_guess=6.2,
                is_polarization=True,
                polarization_mode="eigenstate",
            ),
        )
    }
    space = MappedBasisSpace.from_species_configs(shells)
    # 13 params: q(1) + 3s(5) + 3p(4) + 3d(3) = 13 active params
    assert space.ndim == 13

    # Exact float64 cosine boundary test at multiples of pi
    for mult in (-4, -1, 0, 1, 2, 5):
        val = to_physical_x(mult * math.pi, 1.5, 6.5)
        assert 1.5 <= val <= 6.5

    u0 = space.initial_u()
    p0 = space.decode(u0)
    u_rt = space.encode(p0)
    np.testing.assert_allclose(u_rt, u0, atol=1e-10)

    rng = np.random.default_rng(12345)
    for _ in range(500):
        u_rand = rng.uniform(-25.0, 25.0, size=space.ndim)
        p, sp_specs = space.decode_to_species_specs(u_rand)
        assert -0.75 <= p["Si_q"] <= 1.5
        si = sp_specs["Si"]
        assert si.shells[0].rc_list is not None
        rc1, rc2, rc3 = si.shells[0].rc_list
        assert rc1 > rc2 + 0.349
        assert rc2 > rc3 + 0.349
        assert rc3 > si.shells[0].ri_bohr + 0.299
        assert si.shells[0].ri_bohr >= 0.5


def test_basis_pressure_unit_matches_codata_and_siesta() -> None:
    """1 GPa in eV/Bohr^3: CODATA 2018 (1 Ha/Bohr^3 = 29421.015697 GPa) and SIESTA's BASIS_ENTHALPY (D-040).

    SIESTA's output for 0.2 GPa reads "Plus the pressure : 1.3595723686972976E-005" (Ry/Bohr^3); the p V term in eV
    is that times 13.605693 eV/Ry. The constant held the Ry value until D-040, a factor 13.6 too small.
    """
    assert abs(EV_PER_BOHR3_PER_GPA / (HARTREE_TO_EV / 29421.015697) - 1.0) < 1e-8
    assert abs(0.2 * EV_PER_BOHR3_PER_GPA / (1.3595723686972976e-5 * 0.5 * HARTREE_TO_EV) - 1.0) < 1e-8


def test_pao_generator_explicit_rc_v0_ri_q_invariants() -> None:
    """Explicit (rc per zeta, V0, ri, q) controls orbital shapes, norms, and tail zeros cleanly."""
    spec_neutral = SpeciesBasisSpec(
        symbol="Si",
        shells=(
            ShellBasisSpec(n=3, l=0, n_zeta=2, rc_list=(5.5, 3.8), v0_ha=0.0, ri_bohr=2.5),
            ShellBasisSpec(n=3, l=1, n_zeta=2, rc_list=(6.6, 4.7), v0_ha=0.0, ri_bohr=3.0),
            ShellBasisSpec(
                n=3, l=2, n_zeta=1, rc_list=(6.0,), v0_ha=0.0, ri_bohr=2.5,
                is_polarization=True, polarization_mode="eigenstate",
            ),
        ),
        ionic_charge=0.0,
        xc="PBE",
    )
    sp0, diag0 = generate_species_from_psml(_FIXTURE_SI_PSML, basis_spec=spec_neutral)
    r = np.asarray(sp0.pao_r)
    for k, rc in enumerate(sp0.pao_rcut):
        vals = np.asarray(sp0.pao_values[k])
        nrm = float(np.trapezoid((vals**2) * (r**2), r))
        assert abs(nrm - 1.0) < 5e-3
        tail = vals[r >= rc + 1e-6]
        if len(tail) > 0:
            assert np.max(np.abs(tail)) == 0.0
        assert abs(vals[int(np.argmin(np.abs(r - rc)))]) < 0.05

    # Applying soft confinement V0 and positive ionic charge q shrinks/deepens orbitals physically
    spec_conf = SpeciesBasisSpec(
        symbol="Si",
        shells=(
            ShellBasisSpec(n=3, l=0, n_zeta=2, rc_list=(5.5, 3.8), v0_ha=25.0, ri_bohr=2.5),
            ShellBasisSpec(n=3, l=1, n_zeta=2, rc_list=(6.6, 4.7), v0_ha=25.0, ri_bohr=3.0),
            ShellBasisSpec(
                n=3, l=2, n_zeta=1, rc_list=(6.0,), v0_ha=25.0, ri_bohr=2.5,
                is_polarization=True, polarization_mode="eigenstate",
            ),
        ),
        ionic_charge=+0.6,
        xc="PBE",
    )
    sp1, diag1 = generate_species_from_psml(_FIXTURE_SI_PSML, basis_spec=spec_conf)
    assert diag1["free_energies_ha"][(3, 0)] < diag0["free_energies_ha"][(3, 0)]
    assert sp1.pao_values.shape == (5, 401)
    assert not np.allclose(np.asarray(sp1.pao_values), np.asarray(sp0.pao_values))


def _has_real_si_psml() -> bool:
    try:
        return _REAL_SI_PSML.is_file()
    except OSError:
        return False


@pytest.mark.external_refs("siesta_pseudos_dir")
def test_si_default_sz_dz_dzp_validation_against_siesta() -> None:
    """Pure-Python SZ/DZ/DZP PAOs from real Si.tm2.nrl.psml match SIESTA cutoffs & energy trends."""
    struct = _si_diamond_structure()
    loader = PSMLBasisLoader()
    cfg = SCFConfig(
        nspin=1,
        xc_functional="PBE",
        mesh_cutoff_ry=120.0,
        max_scf_iterations=60,
        d_tol=1e-6,
        h_tol_ha=1e-5,
        mix_weight=0.25,
        mix_history=4,
        use_symmetry=True,
    )
    energies_ev: dict[str, float] = {}
    enthalpies_ev: dict[str, float] = {}
    volumes: dict[str, float] = {}

    for bs in ("SZ", "DZ", "DZP"):
        sp, diag = generate_species_from_psml(_REAL_SI_PSML, basis_spec=bs, xc="PBE")
        if bs == "SZ":
            rc_s = diag["shells"][0]["rc_bohr"]
            rc_p = diag["shells"][1]["rc_bohr"]
            # SIESTA default Si.tm2.nrl.psml gives rc(3s)=5.371, rc(3p)=6.762 Bohr
            assert abs(rc_s - 5.371) < 0.15
            assert abs(rc_p - 6.762) < 0.15
        basis = loader.build_basis(struct, {"Si": sp})
        sym_proj, pat, kmesh, mesh_shape = build_native_symmetry(
            struct, basis, (4, 4, 4), cfg, k_shifts=(0.5, 0.5, 0.5)
        )
        tc = sym_proj.evaluate_irreducible_two_center(struct, basis, pat, cfg)
        st, mesh_shape = run_scf(struct, basis, pat, tc, kmesh, cfg, mesh_shape=mesh_shape, sym_proj=sym_proj)
        obs = StaticObservableEvaluator().evaluate(
            struct,
            basis,
            pat,
            tc,
            kmesh,
            st,
            cfg,
            mesh_shape=mesh_shape,
            sym_proj=sym_proj,
            compute_forces_stress=False,
        )
        energies_ev[bs] = float(obs.final_post_scf_free_e_ha) * HARTREE_TO_EV
        volumes[bs] = basis_volume_bohr3(struct, basis)
        enthalpies_ev[bs] = basis_enthalpy_ev(
            float(obs.final_post_scf_free_e_ha), struct, basis, pressure_gpa=0.2
        )

    # Exact SIESTA reference step drops: DZ - SZ = -0.493 eV, DZP - DZ = -1.025 eV
    de_dz = energies_ev["DZ"] - energies_ev["SZ"]
    de_dzp = energies_ev["DZP"] - energies_ev["DZ"]
    assert -0.80 < de_dz < -0.35
    assert -1.25 < de_dzp < -0.65
    assert enthalpies_ev["DZP"] < enthalpies_ev["DZ"] < enthalpies_ev["SZ"]
    assert abs(volumes["SZ"] - 3916.6) < 100.0


def test_basis_enthalpy_optimization_explicit_params_and_pressure_shrinkage() -> None:
    """Optimizing explicit (rc_z*, V0, ri, q) lowers H_basis and shrinks V_basis at higher pressure."""
    psml = _REAL_SI_PSML if _has_real_si_psml() else _FIXTURE_SI_PSML
    struct = _si_diamond_structure()
    cfg = SCFConfig(
        nspin=1,
        xc_functional="PBE",
        mesh_cutoff_ry=60.0,
        max_scf_iterations=35,
        d_tol=2e-5,
        h_tol_ha=5e-5,
        mix_weight=0.3,
        mix_history=4,
        use_symmetry=True,
    )
    space = MappedBasisSpace.from_species_configs(
        {
            "Si": (
                ShellOptConfig(
                    n=3, l=0, n_zeta=2, rc_guess=5.4, rc_initials=(5.4, 3.8), v0_initial_ha=10.0, ri_initial_bohr=2.5
                ),
                ShellOptConfig(
                    n=3, l=1, n_zeta=2, rc_guess=6.7, rc_initials=(6.7, 4.8), v0_initial_ha=10.0, ri_initial_bohr=3.0
                ),
            )
        },
        xc="PBE",
        optimize_rc=True,
        optimize_v0=True,
        optimize_ri=True,
        optimize_q=True,
    )
    # 1(q) + 4(3s) + 4(3p) = 9 parameters
    assert space.ndim == 9
    target = BasisTrainingStructure(structure=struct, k_grid=(2, 2, 2), k_shifts=(0.5, 0.5, 0.5))

    # Exercise a compact optimization over the full 9D explicit space first
    small_space = MappedBasisSpace.from_species_configs(
        {
            "Si": (
                ShellOptConfig(
                    n=3, l=0, n_zeta=2, rc_guess=5.4, rc_initials=(5.4, 3.8), v0_initial_ha=10.0, ri_initial_bohr=2.5
                ),
                ShellOptConfig(
                    n=3, l=1, n_zeta=1, rc_guess=6.7, rc_initials=(6.7,)
                ),
            )
        },
        xc="PBE",
        optimize_rc=True,
        optimize_v0=True,
        optimize_ri=True,
        optimize_q=True,
    )
    # 1(q) + 4(3s: rc1, rc2, V0, ri) + 3(3p: rc1, V0, ri) = 8 parameters
    assert small_space.ndim == 8

    obj_low = BasisEnthalpyObjective(small_space, {"Si": psml}, [target], cfg, pressure_gpa=0.2)
    h0 = obj_low(small_space.initial_u())
    res_low = optimize_basis_enthalpy(obj_low, method="nm", max_evaluations=12)
    assert res_low.success
    assert res_low.best_enthalpy_ev <= h0 + 1e-9
    expected_h = res_low.best_energy_ev + 0.2 * EV_PER_BOHR3_PER_GPA * res_low.best_volume_bohr3
    assert abs(res_low.best_enthalpy_ev - expected_h) < 1e-8

    # Effective radii (after snapping to the generator's radial grid) are reported next to the requested ones.
    eff = res_low.best_effective_params
    for name, val in res_low.best_params.items():
        if "_rc_z" in name:
            assert abs(eff[name] - val) <= 0.004 * val + 1e-9, (name, val, eff[name])

    # Pressure response (review H4: the former check compared two runs stuck on the same simplex vertex). A 2D
    # space (3s, 3p cut-off radii) with an adequate budget must give a measurably smaller orbital volume at
    # 5 GPa than at 0.2 GPa.
    rc_space = MappedBasisSpace.from_species_configs(
        {
            "Si": (
                ShellOptConfig(n=3, l=0, n_zeta=1, rc_guess=5.4, rc_initials=(5.4,)),
                ShellOptConfig(n=3, l=1, n_zeta=1, rc_guess=6.7, rc_initials=(6.7,)),
            )
        },
        xc="PBE",
        optimize_rc=True,
        optimize_v0=False,
        optimize_ri=False,
        optimize_q=False,
    )
    assert rc_space.ndim == 2
    vols = {}
    for p_gpa in (0.2, 5.0):
        obj = BasisEnthalpyObjective(rc_space, {"Si": psml}, [target], cfg, pressure_gpa=p_gpa)
        res = optimize_basis_enthalpy(obj, method="nm", max_evaluations=20)
        assert res.success and res.n_evaluations >= 10
        vols[p_gpa] = res.best_volume_bohr3
    assert vols[5.0] < 0.99 * vols[0.2], vols

    fdf_block = format_pao_basis_block(res_low.best_species_specs)
    assert "%block PAO.Basis" in fdf_block and "n=3" in fdf_block

    # Smoke check automatic space builder from PSML
    auto_space = default_opt_space_from_psml({"Si": psml}, basis_size="DZP", xc="PBE")
    assert auto_space.ndim > 9
    assert abs(to_optimizer_u(to_physical_x(0.7, 1.0, 5.0), 1.0, 5.0) - 0.7) < 1e-10
