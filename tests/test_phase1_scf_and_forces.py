"""Phase 1 Stage 4 tests: eigensolver, scf_params mixers/controller, SCF convergence, forces/stress vs FD, and validation.py."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core import HARTREE_TO_EV, SCFConfig, SCFState, Structure
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.io import parse_fdf
from jax_siesta.scf import (
    GeneralizedEigenSolver,
    build_h0_sparse,
    build_monkhorst_pack_mesh,
    deterministic_svd_pinv,
    recommend_scf_defaults,
    update_adaptive_state,
)
from jax_siesta.species import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.testing import run_single_point
from validation import run_manifest

jax.config.update("jax_enable_x64", True)
REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = REPO_ROOT / "tests" / "manifests" / "cases"
PSML_DIR = REPO_ROOT / "tests" / "fixtures" / "psml"


def test_scalar_svd_contract_and_descriptors() -> None:
    pinv0, rank0, info0 = deterministic_svd_pinv(jnp.array([[1e-15]]), svd_tol=1e-12)
    assert info0 == 0 and rank0 == 0 and float(pinv0[0, 0]) == 0.0

    pinv1, rank1, info1 = deterministic_svd_pinv(jnp.array([[4.0]]), svd_tol=1e-12)
    assert info1 == 0 and rank1 == 1 and abs(float(pinv1[0, 0]) - 0.25) < 1e-14

    loader = PSMLBasisLoader()
    sp_co = loader.load_species("Co", str(PSML_DIR / "Co.psml"))
    st = Structure(
        cell=jnp.diag(jnp.array([4.7, 4.7, 22.0])),
        positions=jnp.array([[0.0, 0.0, 10.0], [2.35, 2.35, 13.5]]),
        species_indices=(0, 0),
        atomic_numbers=(27, 27),
        species_labels=("Co",),
        initial_moments=jnp.array([[0.0, 0.0, 1.7], [0.0, 0.0, 1.7]]),
    )
    basis = loader.build_basis(st, {"Co": sp_co})
    rec = recommend_scf_defaults(st, basis, SCFConfig(nspin=2))
    assert 0.020 <= rec.w_base <= 0.250
    assert 4 <= rec.n_hist <= 12
    assert rec.is_heuristic_prior
    assert rec.descriptors is not None and rec.descriptors.is_slab_or_molecule


def test_adaptive_controller_and_annealing_guard() -> None:
    cfg = SCFConfig(
        nspin=1,
        d_tol=1e-4,
        h_tol_ha=1e-4,
        e_tol_ha=1e-4,
        electronic_temp_ha=1e-3,
        enable_adaptive_controller=True,
        enable_smearing_annealing=True,
    )
    st0 = SCFState(
        iteration=0,
        h_sparse=jnp.zeros((1, 2)),
        dm_in=jnp.zeros((1, 2)),
        dm_out=jnp.zeros((1, 2)),
        edm_out=jnp.zeros((1, 2)),
        fermi_level_ha=jnp.array(0.0),
        entropy_ha=jnp.array(0.0),
        last_scf_free_e_ha=jnp.array(0.0),
        w_cur=jnp.array(0.1),
        w_base=jnp.array(0.1),
        active_temp_ha=jnp.array(1e-3),
        streak_bad=0,
        prev_d_max=jnp.array(1e-2),
        prev_h_max=jnp.array(1e-2),
        history_x=jnp.zeros((6, 1, 2)),
        history_f=jnp.zeros((6, 1, 2)),
        history_count=5,
    )
    # Iteration-1 post-seed guard: even tiny residuals cannot converge on iteration 1
    s1, conv1, _ = update_adaptive_state(st0, cfg, 1e-8, 1e-8, 1e-8)
    assert not conv1 and s1.iteration == 1

    # Stationary metric preserved for iscf <= 4 even on spike
    s4 = st0
    for _ in range(4):
        s4, _, _ = update_adaptive_state(s4, cfg, 0.05, 0.05, 0.01)
    assert abs(float(s4.w_cur) - 0.1) < 1e-12
    # Annealing boosted temperature during transient (d_max=0.05 > 0.03)
    assert float(s4.active_temp_ha) > cfg.electronic_temp_ha

    # For iscf > 4, 3 consecutive severe spikes (> 3x each step) trigger backoff and history quarantine
    s_spike = s4
    for spike_val in (0.2, 0.8, 3.2):
        s_spike, _, _ = update_adaptive_state(s_spike, cfg, spike_val, spike_val, 0.05)
    assert float(s_spike.w_cur) < 0.1
    assert s_spike.history_count == 2

    # NaN classified as numerical_breakdown
    s_nan, conv_nan, out_nan = update_adaptive_state(s4, cfg, float("nan"), 1e-6, 1e-6)
    assert not conv_nan and out_nan == "numerical_breakdown"


@pytest.mark.parametrize("nspin", [1, 2, 4, 8])
def test_eigensolver_trace_and_band_identity(nspin: int) -> None:
    loader = PSMLBasisLoader()
    sp_co = loader.load_species("Co", str(PSML_DIR / "Co.psml"))
    st = Structure(
        cell=jnp.eye(3) * 15.0,
        positions=jnp.array([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]]),
        species_indices=(0, 0),
        atomic_numbers=(27, 27),
        species_labels=("Co",),
        initial_moments=jnp.zeros((2, 3)),
        pbc=(False, False, False),
    )
    basis = loader.build_basis(st, {"Co": sp_co}, include_soc_projectors=(nspin == 8))
    pattern = build_sparse_pattern(st, basis)
    tc = TwoCenterEvaluator().evaluate(st, basis, pattern, SCFConfig(nspin=nspin))
    h0 = build_h0_sparse(tc, nspin)
    kmesh = build_monkhorst_pack_mesh((1, 1, 1))

    dm, edm, _, e_band, _ = GeneralizedEigenSolver().solve(
        h0, tc.overlap, pattern, kmesh, basis.total_valence_electrons, 1e-3, nspin
    )
    s_vals = tc.overlap
    dm_tot = jnp.real(dm[0] if nspin == 1 else (dm[0] + dm[1]))
    edm_tot = jnp.real(edm[0] if nspin == 1 else (edm[0] + edm[1]))
    assert abs(float(jnp.sum(dm_tot * s_vals)) - basis.total_valence_electrons) < 1e-10
    assert abs(float(jnp.sum(edm_tot * s_vals)) - float(e_band)) < 1e-10


def test_h2o_scf_forces_and_stress_vs_finite_differences(tmp_path: Path) -> None:
    case_path = CASES_DIR / "h2o_gamma" / "case.json"
    obs = run_single_point(case_path, config_overrides={"mesh_cutoff_ry": 60.0, "d_tol": 1e-6, "h_tol_ha": 1e-6})
    assert obs.converged
    assert obs.dm_out_recomputed
    assert abs(float(obs.charge_residual)) < 0.05
    # Verify translational acoustic sum rule decreases cleanly under mesh refinement (eggbox convergence)
    sum_f_60 = float(jnp.max(jnp.abs(jnp.sum(obs.forces_ha_per_bohr, axis=0))))
    assert sum_f_60 < 0.05
    obs_fine = run_single_point(case_path, config_overrides={"mesh_cutoff_ry": 150.0, "d_tol": 1e-6, "h_tol_ha": 1e-6})
    sum_f_150 = float(jnp.max(jnp.abs(jnp.sum(obs_fine.forces_ha_per_bohr, axis=0))))
    assert sum_f_150 < sum_f_60
    assert sum_f_150 < 0.02

    # Central finite difference check on atomic force F[1, 1] (H1 y-axis)
    case_data = json.loads(case_path.read_text())
    eps = 2e-3

    def _run_mod(base_data: dict, mut_fn, overrides: dict):
        d = json.loads(json.dumps(base_data))
        mut_fn(d)
        tmp_p = tmp_path / "_tmp_fd_case.json"
        tmp_p.write_text(json.dumps(d))
        return run_single_point(tmp_p, config_overrides=overrides)

    h2o_ov = {"mesh_cutoff_ry": 60.0, "d_tol": 1e-7, "h_tol_ha": 1e-7}
    obs_p = _run_mod(case_data, lambda d: d["positions_bohr"][1].__setitem__(1, d["positions_bohr"][1][1] + eps), h2o_ov)
    obs_m = _run_mod(case_data, lambda d: d["positions_bohr"][1].__setitem__(1, d["positions_bohr"][1][1] - eps), h2o_ov)
    fd_force = -(float(obs_p.final_post_scf_free_e_ha) - float(obs_m.final_post_scf_free_e_ha)) / (2.0 * eps)
    an_force = float(obs.forces_ha_per_bohr[1, 1])
    assert abs(fd_force - an_force) < 1e-3, f"Force FD={fd_force} vs analytic={an_force}"

    # Central finite difference check on diagonal stress[0, 0]
    def _strain_cell(d, s):
        c = np.array(d["cell_columns_bohr"])
        c[0, :] *= (1.0 + s)
        d["cell_columns_bohr"] = c.tolist()
        for row in d["positions_bohr"]:
            row[0] *= (1.0 + s)

    obs_sp = _run_mod(case_data, lambda d: _strain_cell(d, eps), h2o_ov)
    obs_sm = _run_mod(case_data, lambda d: _strain_cell(d, -eps), h2o_ov)
    vol = float(np.linalg.det(np.array(case_data["cell_columns_bohr"])))
    fd_stress = (float(obs_sp.final_post_scf_free_e_ha) - float(obs_sm.final_post_scf_free_e_ha)) / (2.0 * eps * vol)
    an_stress = float(obs.stress_ha_per_bohr3[0, 0])
    assert abs(fd_stress - an_stress) < 1e-5, f"Stress FD={fd_stress} vs analytic={an_stress}"

    # Displaced-atom shear stress[0, 1] FD check on Si bulk (catches (g_pos.T @ pos)/vol term)
    si_data = json.loads((CASES_DIR / "si_bulk" / "case.json").read_text())
    si_data["positions_bohr"] = [[1.2, 0.5, 0.3], [4.0, 3.3, 3.1]]
    si_data["k_grid"] = [2, 2, 2]
    si_ov = {"mesh_cutoff_ry": 50.0, "d_tol": 1e-7, "h_tol_ha": 1e-7, "e_tol_ha": 1e-9}
    obs_si_disp = _run_mod(si_data, lambda d: None, si_ov)

    eps_s = 2e-4

    def _shear_cell(d, s):
        def_mat = np.eye(3)
        def_mat[0, 1] = s
        c = def_mat @ np.array(d["cell_columns_bohr"])
        p = np.array(d["positions_bohr"]) @ def_mat.T
        d["cell_columns_bohr"] = c.tolist()
        d["positions_bohr"] = p.tolist()

    obs_shp = _run_mod(si_data, lambda d: _shear_cell(d, eps_s), si_ov)
    obs_shm = _run_mod(si_data, lambda d: _shear_cell(d, -eps_s), si_ov)
    vol_si = float(np.linalg.det(np.array(si_data["cell_columns_bohr"])))
    fd_shear = (float(obs_shp.final_post_scf_free_e_ha) - float(obs_shm.final_post_scf_free_e_ha)) / (2.0 * eps_s * vol_si)
    an_shear = float(obs_si_disp.stress_ha_per_bohr3[0, 1])
    assert abs(fd_shear - an_shear) < 1e-5, f"Shear stress FD={fd_shear} vs analytic={an_shear}"


def test_si_bulk_and_co_hcp_spin_modes_and_mixer_endpoint_agreement() -> None:
    # 1. Bulk Si k-sampled semiconductor
    obs_si = run_single_point(
        CASES_DIR / "si_bulk" / "case.json",
        config_overrides={"mesh_cutoff_ry": 80.0, "d_tol": 1e-5, "h_tol_ha": 1e-5},
    )
    assert obs_si.converged

    # 2. Co hcp collinear: DM_out endpoint (< 0.2 meV/cell) must not depend on the SCF trajectory. The five runs
    # differ genuinely (asserted below): Tikhonov-Pulay with the adaptive controller, plain Pulay with a fixed small
    # weight and a short history, Tikhonov-Pulay with coupled spin mixing, modified Broyden and the ADIIS-Pulay hybrid.
    co_case = CASES_DIR / "co_hcp_collinear" / "case.json"
    e_endpoints = {}
    fingerprints = {}
    trajectories = {
        "tikhonov_adaptive": {"mixer_variant": "tikhonov"},
        "pulay_fixed": {"mixer_variant": "pulay", "mix_weight": 0.05, "mix_history": 4,
                        "enable_adaptive_controller": False},
        "tikhonov_coupled": {"mixer_variant": "tikhonov", "mix_spin": "coupled"},
        "broyden": {"mixer_variant": "broyden"},
        "adiis_pulay_hybrid": {"mixer_variant": "adiis"},
    }
    for variant, extra in trajectories.items():
        obs = run_single_point(
            co_case,
            config_overrides={
                "mesh_cutoff_ry": 80.0,
                "d_tol": 1e-5,
                "h_tol_ha": 1e-5,
                "e_tol_ha": 1e-7,
                "max_scf_iterations": 300,
                **extra,
            },
        )
        assert obs.converged, f"Co collinear failed with mixer={variant}: {obs.outcome}"
        assert obs.dm_out_recomputed
        e_endpoints[variant] = float(obs.final_post_scf_free_e_ha) * HARTREE_TO_EV
        # The in-flight energy converges to the shared endpoint, so it cannot tell trajectories apart; the final
        # residuals can.
        fingerprints[variant] = (int(obs.n_iterations), f"{float(obs.d_max):.2e}", f"{float(obs.h_max_ha):.2e}")

    # Four copies of the same DIIS would agree on endpoints and trajectories alike (review M1): require the
    # trajectories (iteration count and final residuals) to be pairwise different.
    assert len(set(fingerprints.values())) == len(fingerprints), fingerprints
    spread_ev = max(e_endpoints.values()) - min(e_endpoints.values())
    assert spread_ev <= 2e-4, f"Mixer endpoint spread {spread_ev:.6f} eV > 0.2 meV/cell: {e_endpoints}"

    # 3. Co hcp non-collinear (nspin=4) and SOC (nspin=8)
    for cname in ("co_hcp_nc", "co_hcp_soc"):
        obs_m = run_single_point(
            CASES_DIR / cname / "case.json",
            config_overrides={"mesh_cutoff_ry": 80.0, "d_tol": 1e-5, "h_tol_ha": 1e-5},
        )
        assert obs_m.converged, f"{cname} failed: {obs_m.outcome}"
        # Verify Im(D_uu)/Im(D_dd) are preserved cleanly across last_scf and post-SCF DM_out passes
        assert abs(float(obs_m.final_post_scf_free_e_ha) - float(obs_m.last_scf_free_e_ha)) < 1e-5

    # 4. Dispersion bilayer (DFT-D3)
    obs_d3 = run_single_point(
        CASES_DIR / "c_bilayer" / "case.json",
        config_overrides={"mesh_cutoff_ry": 80.0, "d_tol": 1e-4, "h_tol_ha": 1e-4},
    )
    assert obs_d3.converged and float(obs_d3.e_disp_ha) < 0.0


def test_fdf_parser_and_validation_harness(tmp_path: Path) -> None:
    fdf = tmp_path / "h2.fdf"
    fdf.write_text(
        """
        SystemName H2
        LatticeConstant 12.0 Bohr
        %block LatticeVectors
          1.0 0.0 0.0
          0.0 1.0 0.0
          0.0 0.0 1.0
        %endblock LatticeVectors
        %block ChemicalSpeciesLabel
          1 1 H
        %endblock ChemicalSpeciesLabel
        AtomicCoordinatesFormat Bohr
        %block AtomicCoordinatesAndAtomicSpecies
          5.0 6.0 6.0 1
          6.4 6.0 6.0 1
        %endblock AtomicCoordinatesAndAtomicSpecies
        Spin none
        MeshCutoff 80.0 Ry
        XC.authors PBE
        """,
        encoding="utf-8",
    )
    parsed = parse_fdf(fdf)
    assert parsed["structure"].n_atoms == 2
    assert parsed["config"].nspin == 1

    # Run validation.py comparing Tikhonov-Pulay vs plain Pulay on h2o_gamma
    manifest = {
        "schema_version": 1,
        "timeout_seconds": 120,
        "environment": {"OMP_NUM_THREADS": "1"},
        "baseline": {
            "command": [
                "{python}",
                str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py"),
                "--mode",
                "mixer-tikhonov",
                "{workdir}/case.json",
            ],
            "artifacts": [str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py")],
            "identity": {"build": "jsiesta-tikhonov", "revision": "0.3.0-phase1"},
        },
        "candidate": {
            "command": [
                "{python}",
                str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py"),
                "--mode",
                "mixer-pulay",
                "{workdir}/case.json",
            ],
            "artifacts": [str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py")],
            "identity": {"build": "jsiesta-pulay", "revision": "0.3.0-phase1"},
        },
        "cases": [
            {
                "name": "h2o_gamma_mixer_parity",
                "input_dir": str(CASES_DIR / "h2o_gamma"),
                "tolerances": {
                    "last_scf_free_e_ev": {"atol": 1e-3, "rtol": 1e-7, "unit": "eV"},
                    "final_post_scf_free_e_ev": {"atol": 2e-4, "rtol": 1e-7, "unit": "eV"},
                    **{f"forces[{i}][{j}]": {"atol": 1e-3, "rtol": 1e-4, "unit": "eV/Ang"} for i in range(3) for j in range(3)},
                    **{f"stress[{a}][{b}]": {"atol": 1e-4, "rtol": 1e-4, "unit": "eV/Ang^3"} for a in range(3) for b in range(3)},
                },
            }
        ],
    }
    mpath = tmp_path / "phase1_mixer_parity.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path, passed = run_manifest(mpath, tmp_path / "reports")
    assert passed, report_path.read_text()


def test_vacuum_fraction_classifies_bulk_slab_and_molecule() -> None:
    """Review M2: the former estimate called every bulk crystal a slab (diamond Si -> 0.875)."""
    from dataclasses import replace

    from jax_siesta.io.fdf import load_case_json
    from jax_siesta.scf.descriptors import estimate_vacuum_fraction

    expected = {
        "si_bulk": (0.0, 0.0),
        "co_hcp_collinear": (0.0, 0.0),
        "si_triclinic": (0.0, 0.0),
        "h2o_gamma": (0.8, 0.95),
        "graphene_slab": (0.5, 0.9),
    }
    for name, (lo, hi) in expected.items():
        vac = estimate_vacuum_fraction(load_case_json(CASES_DIR / name / "case.json")["structure"])
        assert lo - 1e-12 <= vac <= hi, (name, vac)
    s = load_case_json(CASES_DIR / "h2o_gamma" / "case.json")["structure"]
    shifted = replace(s, positions=s.positions + 1.234)
    assert abs(estimate_vacuum_fraction(shifted) - estimate_vacuum_fraction(s)) < 1e-12


def test_no_convergence_on_dm_built_at_annealed_temperature() -> None:
    """Review L1: a DM computed at the annealing temperature must not be accepted as converged."""
    from dataclasses import replace

    cfg = SCFConfig(nspin=1, d_tol=1e-4, h_tol_ha=1e-4, e_tol_ha=1e-4, electronic_temp_ha=1e-3,
                    enable_smearing_annealing=True)
    st = SCFState(
        iteration=4, h_sparse=jnp.zeros((1, 2)), dm_in=jnp.zeros((1, 2)), dm_out=jnp.zeros((1, 2)),
        edm_out=jnp.zeros((1, 2)), fermi_level_ha=jnp.array(0.0), entropy_ha=jnp.array(0.0),
        last_scf_free_e_ha=jnp.array(0.0), w_cur=jnp.array(0.1), w_base=jnp.array(0.1),
        active_temp_ha=jnp.array(0.01),  # this iteration's DM used the annealing temperature
        streak_bad=0, prev_d_max=jnp.array(1e-2), prev_h_max=jnp.array(1e-2),
        history_x=jnp.zeros((6, 1, 2)), history_f=jnp.zeros((6, 1, 2)), history_count=3,
    )
    # d_max below the annealing threshold, so the controller restores the target T for the *next* step...
    s_next, conv, _ = update_adaptive_state(st, cfg, 1e-6, 1e-6, 1e-6)
    assert not conv
    assert abs(float(s_next.active_temp_ha) - 1e-3) < 1e-15
    # ...and only a DM recomputed at the target temperature may converge.
    _, conv2, _ = update_adaptive_state(replace(s_next, iteration=5), cfg, 1e-6, 1e-6, 1e-6)
    assert conv2
