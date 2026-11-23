"""Phase 2 Steps 3 & 4 tests: grid symmetrization, irreducible integrals, forces/stress, invariance, and validation.py."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core import HARTREE_TO_EV, SCFConfig
from jax_siesta.grid import GridOps
from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.io import load_case_json
from jax_siesta.observables import StaticObservableEvaluator
from jax_siesta.scf import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species import PSMLBasisLoader
from jax_siesta.symmetry.engine import NativeSymmetryEngine, build_native_symmetry
from jax_siesta.symmetry.spacegroup import choose_commensurate_mesh_dimensions, extract_space_group
from jax_siesta.testing import run_single_point
from validation import run_manifest

CASES_DIR = Path(__file__).resolve().parent / "manifests" / "cases"
PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(("cname", "nspin"), [("si_bulk", 1), ("co_hcp_soc", 8)])
def test_irreducible_two_center_and_grid_symmetrization(cname: str, nspin: int) -> None:
    case = load_case_json(CASES_DIR / cname / "case.json")
    structure = case["structure"]
    config = SCFConfig(nspin=nspin, mesh_cutoff_ry=60.0, use_symmetry=True)

    loader = PSMLBasisLoader()
    sp_map = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, sp_map, include_soc_projectors=(nspin == 8))

    sym_proj, pattern, _, mesh_shape = build_native_symmetry(structure, basis, case["k_grid"], config)
    tc_irred = sym_proj.evaluate_irreducible_two_center(structure, basis, pattern, config)
    tc_full = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)

    # Orbit count strictly reduced
    assert sym_proj.last_st_irred_blocks < sym_proj.last_st_total_blocks
    if sym_proj.last_kb_total_blocks > 0:
        assert sym_proj.last_kb_irred_blocks < sym_proj.last_kb_total_blocks

    # Exact parity with full direct two-center evaluation
    assert float(jnp.max(jnp.abs(tc_irred.overlap - tc_full.overlap))) < 1e-10
    assert float(jnp.max(jnp.abs(tc_irred.kinetic - tc_full.kinetic))) < 1e-10
    assert float(jnp.max(jnp.abs(tc_irred.d_overlap_dr - tc_full.d_overlap_dr))) < 1e-10
    assert float(jnp.max(jnp.abs(tc_irred.d_kinetic_dr - tc_full.d_kinetic_dr))) < 1e-10
    assert float(jnp.max(jnp.abs(tc_irred.v_kb - tc_full.v_kb))) < 1e-10
    if nspin == 8:
        assert tc_irred.v_kb_so is not None and tc_full.v_kb_so is not None
        assert float(jnp.max(jnp.abs(tc_irred.v_kb_so - tc_full.v_kb_so))) < 1e-10
    assert float(jnp.abs(tc_irred.e_na_pair - tc_full.e_na_pair)) < 1e-10

    # Verify grid symmetrization under jit & grad, charge conservation, and idempotency
    n_ch = 1 if nspin == 1 else (2 if nspin == 2 else 4)
    rng = np.random.default_rng(123 + nspin)
    rand_grid = jnp.asarray(rng.standard_normal((n_ch, *mesh_shape)), dtype=jnp.float64)
    sym_jit = jax.jit(sym_proj.symmetrize_grid)
    g1 = sym_jit(rand_grid)
    g2 = sym_jit(g1)
    assert float(jnp.max(jnp.abs(g2 - g1))) < 1e-12
    assert float(jnp.abs(jnp.sum(g1[0] if n_ch == 1 else (g1[0] + g1[1])) - jnp.sum(rand_grid[0] if n_ch == 1 else (rand_grid[0] + rand_grid[1])))) < 1e-10

    grad_val = jax.grad(lambda x: jnp.sum(sym_proj.symmetrize_grid(x) ** 2))(rand_grid)
    assert jnp.all(jnp.isfinite(grad_val))

    # Neutral atom grid density symmetrization removes raw grid eggbox residual
    rho_na, _, _ = GridOps().neutral_atom_grid(structure, basis, mesh_shape)
    rho_na_sym = sym_jit(rho_na[None, ...])[0]
    assert float(jnp.max(jnp.abs(rho_na_sym - rho_na))) < 1e-4


@pytest.mark.parametrize("cname", ["si_bulk", "h2o_gamma", "co_hcp_collinear", "co_hcp_nc", "co_hcp_soc"])
def test_ibz_symmetrized_vs_full_bz_unsymmetrized_parity(cname: str) -> None:
    case_path = CASES_DIR / cname / "case.json"
    case = load_case_json(case_path)
    structure = case["structure"]
    config_sym = SCFConfig(
        nspin=case["config"].nspin,
        xc_functional=case["config"].xc_functional,
        mesh_cutoff_ry=60.0,
        d_tol=1e-5,
        h_tol_ha=1e-5,
        use_symmetry=True,
    )
    loader = PSMLBasisLoader()
    sp_map = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, sp_map, include_soc_projectors=(config_sym.nspin == 8))

    # Build commensurate pattern/mesh once, then run BOTH unsymmetrized full-BZ and symmetrized IBZ on same mesh!
    sym_proj, pattern_sym, kmesh_ibz, mesh_shape = build_native_symmetry(
        structure, basis, case["k_grid"], config_sym
    )
    tc = sym_proj.evaluate_irreducible_two_center(structure, basis, pattern_sym, config_sym)
    kmesh_full = build_monkhorst_pack_mesh(case["k_grid"])
    config_unsym = SCFConfig(
        nspin=config_sym.nspin,
        xc_functional=config_sym.xc_functional,
        mesh_cutoff_ry=60.0,
        d_tol=1e-5,
        h_tol_ha=1e-5,
        use_symmetry=False,
    )

    st_sym, _ = run_scf(structure, basis, pattern_sym, tc, kmesh_ibz, config_sym, mesh_shape=mesh_shape, sym_proj=sym_proj)
    st_unsym, _ = run_scf(structure, basis, pattern_sym, tc, kmesh_full, config_unsym, mesh_shape=mesh_shape, sym_proj=None)
    assert st_sym.converged and st_unsym.converged

    # Verify [H_sym, U_S] == 0
    assert float(jnp.max(jnp.abs(sym_proj.project_matrix(st_sym.h_sparse, config_sym.nspin) - st_sym.h_sparse))) < 1e-9

    obs_sym = StaticObservableEvaluator().evaluate(
        structure, basis, pattern_sym, tc, kmesh_ibz, st_sym, config_sym, mesh_shape=mesh_shape, sym_proj=sym_proj
    )
    obs_unsym = StaticObservableEvaluator().evaluate(
        structure, basis, pattern_sym, tc, kmesh_full, st_unsym, config_unsym, mesh_shape=mesh_shape, sym_proj=None
    )

    e_diff = abs(float(obs_sym.final_post_scf_free_e_ha) - float(obs_unsym.final_post_scf_free_e_ha))
    assert e_diff < 2e-5, f"{cname}: IBZ={obs_sym.final_post_scf_free_e_ha} vs fullBZ={obs_unsym.final_post_scf_free_e_ha}"
    assert float(jnp.max(jnp.abs(obs_sym.forces_ha_per_bohr - obs_unsym.forces_ha_per_bohr))) < 5e-4
    assert float(jnp.max(jnp.abs(obs_sym.stress_ha_per_bohr3 - obs_unsym.stress_ha_per_bohr3))) < 1e-5


def test_symmetry_forces_stress_fd_rotation_permutation_and_validation(tmp_path: Path) -> None:
    # 1. Symmetric stretch FD force/stress on symmetrized Si bulk & raw symmetry residual check
    si_path = CASES_DIR / "si_bulk" / "case.json"
    si_data = json.loads(si_path.read_text())
    si_ov = {"mesh_cutoff_ry": 100.0, "d_tol": 1e-7, "h_tol_ha": 1e-7, "e_tol_ha": 1e-9, "use_symmetry": True}
    obs_si = run_single_point(si_path, config_overrides=si_ov)
    assert obs_si.converged and obs_si.active_symmetry_ops == 48
    # Cubic crystal at equilibrium geometry -> forces vanish & stress isotropic diagonal
    assert float(jnp.max(jnp.abs(obs_si.forces_ha_per_bohr))) < 1e-4
    diag = np.diag(np.asarray(obs_si.stress_ha_per_bohr3))
    assert np.max(np.abs(diag - np.mean(diag))) < 1e-10
    assert float(jnp.max(jnp.abs(obs_si.stress_ha_per_bohr3 - np.diag(diag)))) < 1e-10

    # FD hydrostatic pressure check with symmetry enabled
    eps = 2e-4
    def _scale(d: dict, s: float) -> None:
        d["cell_columns_bohr"] = ((1.0 + s) * np.array(d["cell_columns_bohr"])).tolist()
        d["positions_bohr"] = ((1.0 + s) * np.array(d["positions_bohr"])).tolist()

    dp = json.loads(json.dumps(si_data))
    _scale(dp, eps)
    dm = json.loads(json.dumps(si_data))
    _scale(dm, -eps)
    pp = tmp_path / "si_p.json"
    pp.write_text(json.dumps(dp))
    pm = tmp_path / "si_m.json"
    pm.write_text(json.dumps(dm))
    op = run_single_point(pp, config_overrides=si_ov)
    om = run_single_point(pm, config_overrides=si_ov)
    vol = float(np.linalg.det(np.array(si_data["cell_columns_bohr"])))
    fd_trace = (float(op.final_post_scf_free_e_ha) - float(om.final_post_scf_free_e_ha)) / (2.0 * eps * vol)
    an_trace = float(jnp.trace(obs_si.stress_ha_per_bohr3))
    assert abs(fd_trace - an_trace) < 1e-5, f"Sym stress trace FD={fd_trace} vs analytic={an_trace}"

    # 2. Rigid 3D rotation invariance & atom-permutation invariance on symmetrized h2o_gamma
    h2o_data = json.loads((CASES_DIR / "h2o_gamma" / "case.json").read_text())
    h2o_ov = {"mesh_cutoff_ry": 100.0, "d_tol": 1e-7, "h_tol_ha": 1e-7, "e_tol_ha": 1e-9, "use_symmetry": True}
    obs_h0 = run_single_point(CASES_DIR / "h2o_gamma" / "case.json", config_overrides=h2o_ov)

    # Arbitrary proper SO(3) rotation around diagonal axis
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    ang = 0.73
    k_mat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(ang) * k_mat + (1.0 - np.cos(ang)) * (k_mat @ k_mat)

    d_rot = json.loads(json.dumps(h2o_data))
    d_rot["cell_columns_bohr"] = (rot @ np.array(h2o_data["cell_columns_bohr"])).tolist()
    d_rot["positions_bohr"] = (np.array(h2o_data["positions_bohr"]) @ rot.T).tolist()
    p_rot = tmp_path / "h2o_rot.json"
    p_rot.write_text(json.dumps(d_rot))
    obs_rot = run_single_point(p_rot, config_overrides=h2o_ov)
    assert abs(float(obs_rot.final_post_scf_free_e_ha) - float(obs_h0.final_post_scf_free_e_ha)) < 2e-3
    f_h0 = np.asarray(obs_h0.forces_ha_per_bohr)
    f_rot = np.asarray(obs_rot.forces_ha_per_bohr)
    assert np.max(np.abs(f_rot - f_h0 @ rot.T)) < 3e-2
    # Wait: rotated cell grid points change discretization slightly, while permutation on exact same mesh is machine-exact:
    d_perm = json.loads(json.dumps(h2o_data))
    perm = [2, 0, 1]
    d_perm["species"] = [h2o_data["species"][i] for i in perm]
    d_perm["positions_bohr"] = [h2o_data["positions_bohr"][i] for i in perm]
    if "initial_moments" in h2o_data:
        d_perm["initial_moments"] = [h2o_data["initial_moments"][i] for i in perm]
    p_perm = tmp_path / "h2o_perm.json"
    p_perm.write_text(json.dumps(d_perm))
    obs_perm = run_single_point(p_perm, config_overrides=h2o_ov)
    assert abs(float(obs_perm.final_post_scf_free_e_ha) - float(obs_h0.final_post_scf_free_e_ha)) < 1e-8
    f_h0 = np.asarray(obs_h0.forces_ha_per_bohr)
    f_perm = np.asarray(obs_perm.forces_ha_per_bohr)
    assert np.max(np.abs(f_perm - f_h0[perm])) < 1e-6

    # Check H2O C2v FD force with symmetry enabled along symmetric z stretch (O moved)
    d_hp = json.loads(json.dumps(h2o_data))
    d_hp["positions_bohr"][0][2] += eps
    d_hm = json.loads(json.dumps(h2o_data))
    d_hm["positions_bohr"][0][2] -= eps
    p_hp = tmp_path / "h2o_p.json"
    p_hp.write_text(json.dumps(d_hp))
    p_hm = tmp_path / "h2o_m.json"
    p_hm.write_text(json.dumps(d_hm))
    ohp = run_single_point(p_hp, config_overrides=h2o_ov)
    ohm = run_single_point(p_hm, config_overrides=h2o_ov)
    fd_fz = -(float(ohp.final_post_scf_free_e_ha) - float(ohm.final_post_scf_free_e_ha)) / (2.0 * eps)
    assert abs(fd_fz - float(obs_h0.forces_ha_per_bohr[0, 2])) < 1e-3

    # Check raw pre-projection forces & stress symmetry residuals on converged symmetric Si state
    si_case_obj = load_case_json(si_path)
    si_cfg_sym = SCFConfig(**{**si_case_obj["config"].__dict__, **si_ov})
    si_loader = PSMLBasisLoader()
    si_sp = {lbl: si_loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in si_case_obj["structure"].species_labels}
    si_basis = si_loader.build_basis(si_case_obj["structure"], si_sp)
    si_sym, si_pat, si_km, si_ms = build_native_symmetry(si_case_obj["structure"], si_basis, si_case_obj["k_grid"], si_cfg_sym)
    si_tc = si_sym.evaluate_irreducible_two_center(si_case_obj["structure"], si_basis, si_pat, si_cfg_sym)
    si_st, _ = run_scf(si_case_obj["structure"], si_basis, si_pat, si_tc, si_km, si_cfg_sym, mesh_shape=si_ms, sym_proj=si_sym)
    obs_raw = StaticObservableEvaluator().evaluate(
        si_case_obj["structure"], si_basis, si_pat, si_tc, si_km, si_st, si_cfg_sym, mesh_shape=si_ms, sym_proj=None
    )
    assert float(jnp.max(jnp.abs(si_sym.project_forces(obs_raw.forces_ha_per_bohr) - obs_raw.forces_ha_per_bohr))) < 1e-4
    assert float(jnp.max(jnp.abs(si_sym.project_stress(obs_raw.stress_ha_per_bohr3) - obs_raw.stress_ha_per_bohr3))) < 1e-5

    # 3. End-to-end validation.py cross-verification: unsymmetrized full-BZ vs symmetrized IBZ
    si_case = json.loads(si_path.read_text())
    si_case_dir = tmp_path / "cases" / "si_bulk"
    si_case_dir.mkdir(parents=True)
    si_case["mesh_cutoff_ry"] = 80.0
    # The cross-check isolates the symmetry treatment, so both runs must use the same mesh. Diamond's (1/4, 1/4, 1/4)
    # translations need N divisible by 4: the SIESTA criterion gives 18^3 unsymmetrised but 20^3 symmetrised at 80 Ry
    # (a 25 meV mesh difference, not a symmetry error); the legacy criterion gives 24^3 for both (D-035).
    si_case["mesh_criterion"] = "jsiesta"
    si_cfg_80 = load_case_json(si_path)["config"]
    si_sg_80 = extract_space_group(si_case_obj["structure"], si_cfg_80)
    mesh_plain = choose_mesh_dimensions(si_case_obj["structure"].cell, 80.0, criterion="jsiesta")
    mesh_sym = choose_commensurate_mesh_dimensions(si_case_obj["structure"], 80.0, si_sg_80, criterion="jsiesta")
    assert mesh_plain == mesh_sym, (mesh_plain, mesh_sym)
    (si_case_dir / "case.json").write_text(json.dumps(si_case))

    tolerances = {
        "last_scf_free_e_ev": {"atol": 2e-3, "rtol": 1e-6, "unit": "eV"},
        "final_post_scf_free_e_ev": {"atol": 5e-4, "rtol": 1e-6, "unit": "eV"},
        **{f"forces[{i}][{j}]": {"atol": 0.005, "rtol": 1e-4, "unit": "eV/Ang"} for i in range(2) for j in range(3)},
        **{f"stress[{a}][{b}]": {"atol": 5e-4, "rtol": 1e-4, "unit": "eV/Ang^3"} for a in range(3) for b in range(3)},
    }

    manifest = {
        "schema_version": 1,
        "timeout_seconds": 180,
        "environment": {"OMP_NUM_THREADS": "1"},
        "baseline": {
            "command": [
                "{python}",
                str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py"),
                "--mode",
                "unsymmetrized",
                "{workdir}/case.json",
            ],
            "artifacts": [str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py")],
            "identity": {"build": "jsiesta-unsymmetrized", "revision": "0.3.0-phase2"},
        },
        "candidate": {
            "command": [
                "{python}",
                str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py"),
                "--mode",
                "symmetrized",
                "{workdir}/case.json",
            ],
            "artifacts": [str(REPO_ROOT / "jax_siesta" / "testing" / "run_case.py")],
            "identity": {"build": "jsiesta-symmetrized-ibz", "revision": "0.3.0-phase2"},
        },
        "cases": [
            {
                "name": "si_bulk_unsym_vs_sym_ibz",
                "input_dir": "cases/si_bulk",
                "tolerances": tolerances,
            }
        ],
    }
    m_path = tmp_path / "phase2_manifest.json"
    m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path, passed = run_manifest(m_path, tmp_path / "reports")
    assert passed, report_path.read_text()
    assert HARTREE_TO_EV > 27.0


def test_nspin8_offdiagonal_su2_hamiltonian_and_dm_mixing() -> None:
    """Verify [H_sym, U_S]==0 on cubic nspin=8 (off-diagonal SU(2)) and DM mixing for nspin=8."""
    from jax_siesta.core import Structure

    a_fcc = 6.6
    cell_fcc = jnp.asarray([[0.0, 0.5 * a_fcc, 0.5 * a_fcc], [0.5 * a_fcc, 0.0, 0.5 * a_fcc], [0.5 * a_fcc, 0.5 * a_fcc, 0.0]])
    st_fcc = Structure(
        cell=cell_fcc,
        positions=jnp.zeros((1, 3), dtype=jnp.float64),
        species_indices=(0,),
        atomic_numbers=(27,),
        species_labels=("Co",),
        initial_moments=jnp.zeros((1, 3), dtype=jnp.float64),
    )
    loader = PSMLBasisLoader()
    sp_co = loader.load_species("Co", str(PSML_DIR / "Co.psml"))
    basis = loader.build_basis(st_fcc, {"Co": sp_co}, include_soc_projectors=True)

    for mix_target in ("Hamiltonian", "DM"):
        cfg = SCFConfig(
            nspin=8,
            mesh_cutoff_ry=50.0,
            d_tol=1e-5,
            h_tol_ha=1e-5,
            mix_target=mix_target,
            use_symmetry=True,
            max_scf_iterations=40,
        )
        sym, pat, km, ms = build_native_symmetry(st_fcc, basis, (2, 2, 2), cfg)
        assert isinstance(sym, NativeSymmetryEngine)
        assert sym.n_ops == 48
        tc = sym.evaluate_irreducible_two_center(st_fcc, basis, pat, cfg)
        scf_s, _ = run_scf(st_fcc, basis, pat, tc, km, cfg, mesh_shape=ms, sym_proj=sym)
        assert scf_s.converged, f"FCC Co SOC failed with mix_target={mix_target}: {scf_s.outcome}"
        assert float(jnp.max(jnp.abs(sym.project_matrix(scf_s.h_sparse, 8) - scf_s.h_sparse))) < 1e-10
