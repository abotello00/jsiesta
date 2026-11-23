r"""Meta-GGA and spin-orbit coupling tests matching ``pr_mgga`` reference physics."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core.ffi import ffi_backend_info
from jax_siesta.grid.mesh import (
    GridOps,
    _pack_basis_grid_buffers,
    _pack_na_buffers,
    choose_mesh_dimensions,
    tau_core_grid_differentiable,
    tauofd_differentiable,
)
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.symmetry.engine import build_native_symmetry
from jax_siesta.testing.run_case import load_case_json
from jax_siesta.testing.siesta_adapter import (
    observables_to_component_validation_dict,
    parse_siesta_outputs,
    run_jsiesta_matched_case,
    run_siesta_directory,
)
from jax_siesta.testing.external_refs import ref_path
from jax_siesta.xc.functional import LibxcFFIEvaluator, resolve_xc_spec, xc_energy_from_grid

_CASES_DIR = Path(__file__).resolve().parent / "manifests" / "cases"
_PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"
_PR_MGGA_TESTS = ref_path("pr_mgga_tests_dir") or Path("/nonexistent")


def _has_libxc() -> bool:
    try:
        return bool(ffi_backend_info().get("has_libxc", False))
    except Exception:
        return False


def _load_fixture(case_name: str, prefer_lj: bool = False):
    case = load_case_json(_CASES_DIR / case_name / "case.json")
    s = case["structure"]
    cfg = case["config"]
    loader = PSMLBasisLoader()
    sp_map = {
        lbl: loader.load_species(lbl, str(_PSML_DIR / f"{lbl}.psml"), prefer_lj_projectors=prefer_lj)
        for lbl in s.species_labels
    }
    basis = loader.build_basis(s, sp_map, include_soc_projectors=prefer_lj)
    pat = build_sparse_pattern(s, basis)
    return s, basis, pat, cfg, case["k_grid"]


@pytest.mark.parametrize("nspin", [1, 2, 4])
def test_tauofd_vmat_tau_adjoint_and_tau_core_fd(nspin: int) -> None:
    """Verify discrete adjointness of ``tauofd``/``vmat_tau`` and orbital-Hessian/``tau_core`` FD."""
    s, basis, pat, _, _ = _load_fixture("si_bulk")
    s = replace(s, positions=s.positions + jnp.array([0.15, 0.11, -0.07], dtype=jnp.float64))
    pat = build_sparse_pattern(s, basis)
    sp0 = basis.species[0]
    r_grid = np.linspace(0.0, float(sp0.r_grid[-1]), len(sp0.rho_na), dtype=np.float64)
    sp0_tc = replace(sp0, tau_core=jnp.asarray(0.05 * np.exp(-0.7 * (r_grid**2))), has_tau_core=True)
    basis = replace(basis, species=(sp0_tc,))
    packed = _pack_basis_grid_buffers(basis, pat)
    na_packed = _pack_na_buffers(s, basis)
    grid_ops = GridOps()
    mesh_shape = (14, 14, 14)

    n_spin_mat = 1 if nspin == 1 else (2 if nspin == 2 else 4)
    rng = np.random.default_rng(100 + nspin)
    dm = jnp.asarray(0.05 * rng.standard_normal((n_spin_mat, pat.n_edges)), dtype=jnp.float64)
    v_tau = jnp.asarray(rng.standard_normal((n_spin_mat, *mesh_shape)), dtype=jnp.float64)

    tau = grid_ops.kinetic_density_on_grid(dm, s, basis, pat, mesh_shape, nspin)
    h_tau = grid_ops.tau_potential_to_matrix(v_tau, s, basis, pat, nspin)
    dvol = float(jnp.abs(jnp.linalg.det(s.cell))) / float(np.prod(mesh_shape))
    assert abs(float(jnp.sum(v_tau * tau)) * dvol - float(jnp.sum(dm * h_tau))) < 1e-12

    def _tau_loss(pos, cell):
        return jnp.sum(v_tau * tauofd_differentiable(dm, pos, cell, packed, mesh_shape, nspin))

    def _tc_loss(pos, cell):
        return jnp.sum(v_tau[0] * tau_core_grid_differentiable(pos, cell, na_packed, mesh_shape))

    for loss_fn, tol in ((_tau_loss, 1e-7), (_tc_loss, 1e-8)):
        g_pos, g_cell = jax.grad(loss_fn, argnums=(0, 1))(s.positions, s.cell)
        delta = 1e-5
        fd_pos = (
            float(loss_fn(s.positions.at[0, 1].add(delta), s.cell))
            - float(loss_fn(s.positions.at[0, 1].add(-delta), s.cell))
        ) / (2.0 * delta)
        fd_cell = (
            float(loss_fn(s.positions, s.cell.at[0, 0].add(delta)))
            - float(loss_fn(s.positions, s.cell.at[0, 0].add(-delta)))
        ) / (2.0 * delta)
        assert abs(float(g_pos[0, 1]) - fd_pos) < tol
        assert abs(float(g_cell[0, 0]) - fd_cell) < tol


@pytest.mark.skipif(not _has_libxc(), reason="External Libxc meta-GGA support unavailable.")
@pytest.mark.parametrize("func", ["R2SCAN01", "R2SCAN", "TPSS"])
def test_mgga_xc_functional_derivative_stress_and_spin_rotation(func: str) -> None:
    """Verify Libxc meta-GGA VJP, stress vs FD, and non-collinear SU(2) spin-rotation invariance."""
    _, xc_id, xc_pair = resolve_xc_spec(func)
    mesh = (12, 12, 12)
    cell = jnp.eye(3, dtype=jnp.float64) * 9.5
    x = jnp.linspace(0.0, 2.0 * jnp.pi, mesh[0], endpoint=False)
    gx, gy, gz = jnp.meshgrid(x, x, x, indexing="ij")
    rho_base = 0.12 + 0.04 * jnp.sin(gx) * jnp.cos(gy) + 0.02 * jnp.cos(gz)
    tau_base = 0.15 + 0.05 * (jnp.sin(gx) ** 2 + jnp.cos(gz) ** 2)
    rho_u = 0.56 * rho_base
    rho_d = 0.44 * rho_base
    tau_u = 0.59 * tau_base
    tau_d = 0.41 * tau_base
    rho_2 = jnp.stack([rho_u, rho_d], axis=0)
    tau_2 = jnp.stack([tau_u, tau_d], axis=0)

    ev = LibxcFFIEvaluator(func)
    e2, vxc2, vtau2, stress2 = ev.evaluate_mgga(rho_2, tau_2, cell, 2)
    assert float(jnp.linalg.norm(stress2 - stress2.T)) < 1e-12

    # Central FD wrt density, tau, and strain
    delta = 1e-6
    dr = jnp.zeros_like(rho_2).at[0, 2, 3, 4].set(1.0)
    dt = jnp.zeros_like(tau_2).at[1, 3, 2, 5].set(1.0)
    dvol = float(jnp.abs(jnp.linalg.det(cell))) / float(np.prod(mesh))
    e_rp = xc_energy_from_grid(rho_2 + delta * dr, cell, 2, xc_id, tau_grid=tau_2, xc_pair=xc_pair)
    e_rm = xc_energy_from_grid(rho_2 - delta * dr, cell, 2, xc_id, tau_grid=tau_2, xc_pair=xc_pair)
    assert abs((float(e_rp) - float(e_rm)) / (2.0 * delta) - float(vxc2[0, 2, 3, 4]) * dvol) < 1e-8

    e_tp = xc_energy_from_grid(rho_2, cell, 2, xc_id, tau_grid=tau_2 + delta * dt, xc_pair=xc_pair)
    e_tm = xc_energy_from_grid(rho_2, cell, 2, xc_id, tau_grid=tau_2 - delta * dt, xc_pair=xc_pair)
    assert abs((float(e_tp) - float(e_tm)) / (2.0 * delta) - float(vtau2[1, 3, 2, 5]) * dvol) < 1e-8

    eps = np.zeros((3, 3))
    eps[0, 1] = eps[1, 0] = delta
    e_sp = xc_energy_from_grid(rho_2, (jnp.eye(3) + eps) @ cell, 2, xc_id, tau_grid=tau_2, xc_pair=xc_pair)
    e_sm = xc_energy_from_grid(rho_2, (jnp.eye(3) - eps) @ cell, 2, xc_id, tau_grid=tau_2, xc_pair=xc_pair)
    vol = float(jnp.abs(jnp.linalg.det(cell)))
    s01_fd = (float(e_sp) - float(e_sm)) / (2.0 * delta) / (2.0 * vol)
    assert abs(float(stress2[0, 1]) - s01_fd) < 1e-8

    # Non-collinear SU(2) spin rotation of (nspin=2) along (theta=60°, phi=30°)
    theta, phi = math.radians(60.0), math.radians(30.0)
    ux = math.sin(theta) * math.cos(phi)
    uy = math.sin(theta) * math.sin(phi)
    uz = math.cos(theta)
    n_tot = rho_u + rho_d
    m_z = rho_u - rho_d
    rho_4 = jnp.stack(
        [0.5 * (n_tot + uz * m_z), 0.5 * (n_tot - uz * m_z), 0.5 * ux * m_z, -0.5 * uy * m_z],
        axis=0,
    )
    t_tot = tau_u + tau_d
    tm_z = tau_u - tau_d
    tau_4 = jnp.stack(
        [0.5 * (t_tot + uz * tm_z), 0.5 * (t_tot - uz * tm_z), 0.5 * ux * tm_z, -0.5 * uy * tm_z],
        axis=0,
    )
    e4, vxc4, vtau4, stress4 = ev.evaluate_mgga(rho_4, tau_4, cell, 4)
    assert abs(float(e4) - float(e2)) < 1e-12
    assert float(jnp.max(jnp.abs(stress4 - stress2))) < 1e-12
    vxc4_rot = 0.5 * (vxc4[0] + vxc4[1]) + 0.5 * (vxc4[0] - vxc4[1]) * uz + ux * vxc4[2] - uy * vxc4[3]
    vtau4_rot = 0.5 * (vtau4[0] + vtau4[1]) + 0.5 * (vtau4[0] - vtau4[1]) * uz + ux * vtau4[2] - uy * vtau4[3]
    assert float(jnp.max(jnp.abs(vxc4_rot - vxc2[0]))) < 1e-9
    assert float(jnp.max(jnp.abs(vtau4_rot - vtau2[0]))) < 1e-9


@pytest.mark.skipif(not _has_libxc(), reason="External Libxc meta-GGA support unavailable.")
@pytest.mark.parametrize(
    ("case_name", "prefer_lj"),
    [
        ("si_bulk", False),
        ("co_hcp_collinear", False),
        ("co_hcp_nc", False),
        ("co_hcp_soc", True),
    ],
)
def test_mgga_ibz_symmetry_vs_full_bz(case_name: str, prefer_lj: bool) -> None:
    """Verify ``R2SCAN01`` native ``spglib`` IBZ symmetry matches full-BZ SCF across all spin modes."""
    s, basis, pat, cfg, k_grid = _load_fixture(case_name, prefer_lj=prefer_lj)
    k_use = (2, 2, 2) if case_name == "si_bulk" else k_grid
    cfg_full = replace(cfg, xc_functional="R2SCAN01", use_symmetry=False, mesh_cutoff_ry=100.0)
    kmesh_full = build_monkhorst_pack_mesh(k_use)
    ms = choose_mesh_dimensions(s.cell, cfg_full.mesh_cutoff_ry)
    tc_full = TwoCenterEvaluator().evaluate(s, basis, pat, cfg_full)
    st_full, ms = run_scf(s, basis, pat, tc_full, kmesh_full, cfg_full, mesh_shape=ms)
    obs_full = StaticObservableEvaluator().evaluate(s, basis, pat, tc_full, kmesh_full, st_full, cfg_full, mesh_shape=ms)

    cfg_sym = replace(cfg_full, use_symmetry=True)
    sym_proj, pat_s, kmesh_s, ms_s = build_native_symmetry(s, basis, k_use, cfg_sym)
    assert kmesh_s.n_k < kmesh_full.n_k
    tc_sym = sym_proj.evaluate_irreducible_two_center(s, basis, pat_s, cfg_sym)
    st_sym, _ = run_scf(s, basis, pat_s, tc_sym, kmesh_s, cfg_sym, mesh_shape=ms_s, sym_proj=sym_proj)
    obs_sym = StaticObservableEvaluator().evaluate(
        s, basis, pat_s, tc_sym, kmesh_s, st_sym, cfg_sym, mesh_shape=ms_s, sym_proj=sym_proj
    )

    assert obs_full.converged and obs_sym.converged
    assert abs(float(obs_sym.total_energy_ha) - float(obs_full.total_energy_ha)) < 1e-6
    assert abs(float(obs_sym.e_xc_ha) - float(obs_full.e_xc_ha)) < 1e-6
    assert np.max(np.abs(np.asarray(obs_sym.forces_ha_per_bohr) - np.asarray(obs_full.forces_ha_per_bohr))) < 5e-4
    assert np.max(np.abs(np.asarray(obs_sym.stress_ha_per_bohr3) - np.asarray(obs_full.stress_ha_per_bohr3))) < 1e-5


@pytest.mark.skipif(not _has_libxc(), reason="External Libxc meta-GGA support unavailable.")
def test_mgga_scf_forces_and_stress_vs_finite_differences() -> None:
    """Verify variational ``R2SCAN01`` forces and stress on displaced Si bulk against central FD."""
    s, basis, _, cfg, _ = _load_fixture("si_bulk")
    pos_disp = s.positions.at[1, 0].add(0.18).at[1, 1].add(-0.11)
    s_disp = replace(s, positions=pos_disp)
    pat_disp = build_sparse_pattern(s_disp, basis)
    cfg_m = replace(cfg, xc_functional="R2SCAN01", d_tol=1e-7, h_tol_ha=1e-7, mesh_cutoff_ry=120.0)
    kmesh = build_monkhorst_pack_mesh((2, 2, 2))
    ms = choose_mesh_dimensions(s_disp.cell, cfg_m.mesh_cutoff_ry)
    tc = TwoCenterEvaluator().evaluate(s_disp, basis, pat_disp, cfg_m)
    st, _ = run_scf(s_disp, basis, pat_disp, tc, kmesh, cfg_m, mesh_shape=ms)
    obs = StaticObservableEvaluator().evaluate(s_disp, basis, pat_disp, tc, kmesh, st, cfg_m, mesh_shape=ms)
    assert obs.converged

    def _free_e(struct):
        pat_i = build_sparse_pattern(struct, basis)
        tc_i = TwoCenterEvaluator().evaluate(struct, basis, pat_i, cfg_m)
        st_i, _ = run_scf(struct, basis, pat_i, tc_i, kmesh, cfg_m, mesh_shape=ms)
        obs_i = StaticObservableEvaluator().evaluate(struct, basis, pat_i, tc_i, kmesh, st_i, cfg_m, mesh_shape=ms)
        return float(obs_i.final_post_scf_free_e_ha)

    delta = 1e-4
    f_fd = -(
        _free_e(replace(s_disp, positions=pos_disp.at[1, 0].add(delta)))
        - _free_e(replace(s_disp, positions=pos_disp.at[1, 0].add(-delta)))
    ) / (2.0 * delta)
    assert abs(float(obs.forces_ha_per_bohr[1, 0]) - f_fd) < 5e-4

    eps = jnp.zeros((3, 3), dtype=jnp.float64).at[0, 0].set(delta)
    s_p = replace(s_disp, cell=(jnp.eye(3) + eps) @ s_disp.cell, positions=s_disp.positions @ (jnp.eye(3) + eps).T)
    s_m = replace(s_disp, cell=(jnp.eye(3) - eps) @ s_disp.cell, positions=s_disp.positions @ (jnp.eye(3) - eps).T)
    vol = float(jnp.abs(jnp.linalg.det(s_disp.cell)))
    s00_fd = (_free_e(s_p) - _free_e(s_m)) / (2.0 * delta) / vol
    assert abs(float(obs.stress_ha_per_bohr3[0, 0]) - s00_fd) < 5e-4


def _run_pr_mgga_fdf(fdf_name: str, pseudo_src: Path, pseudo_dst_name: str, run_dir: Path, label: str, nspin: int) -> dict:
    """Copy a canonical ``pr_mgga`` test FDF, execute SIESTA, and return parsed reference observables."""
    import os

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pseudo_src, run_dir / pseudo_dst_name)
    shutil.copy2(_PR_MGGA_TESTS / "10.Functionals" / fdf_name, run_dir / fdf_name)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    run_siesta_directory(run_dir, fdf_name, executable_key="siesta_mgga_bin", stdout_name="si.out", env=env)
    return parse_siesta_outputs(run_dir, label, nspin)


def _mgga_residuals(prefix: str, val: dict, ref: dict) -> dict[str, float]:
    """Absolute differences of the comparable quantities, keyed ``<prefix>.<name>`` (recorded in the JUnit XML)."""
    out: dict[str, float] = {}
    for key in ("e_kinetic_ev", "e_kb_ev", "e_hartree_ev", "e_xc_ev", "total_spin_moment", "fermi_level_ev"):
        if key in val and key in ref:
            out[f"{prefix}.{key}"] = abs(float(val[key]) - float(ref[key]))
    out[f"{prefix}.forces_max"] = float(np.max(np.abs(np.array(val["forces"]) - np.array(ref["forces"]))))
    out[f"{prefix}.stress_max"] = float(np.max(np.abs(np.array(val["stress"]) - np.array(ref["stress"]))))
    return out


@pytest.mark.skipif(not _has_libxc(), reason="External Libxc meta-GGA support unavailable.")
@pytest.mark.external_refs("siesta_mgga_bin_or_cache", "pr_mgga_tests_dir")
def test_pr_mgga_si_bulk_noncol_and_soc_parity(tmp_path: Path, record_residuals) -> None:
    """Verify full cross-code parity against ``pr_mgga`` across forces/stress, NC, and SOC."""
    # 1. mgga_forces (displaced diamond Si, nspin=1, 4x4x4 shifted k-grid, 14_Si_mgga.psml with tau_core)
    ref_dir = tmp_path / "ref_forces"
    ref_f = _run_pr_mgga_fdf(
        "mgga_forces.fdf",
        _PR_MGGA_TESTS / "Pseudos" / "14_Si_mgga.psml",
        "14_Si_mgga.psml",
        ref_dir,
        "mgga_forces",
        1,
    )
    a_bohr = 5.430 / 0.529177210903
    cell_si = a_bohr * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    pos_si = (cell_si @ np.array([[0.0, 0.0, 0.0], [0.260, 0.245, 0.250]]).T).T
    shutil.copy2(_PR_MGGA_TESTS / "Pseudos" / "14_Si_mgga.psml", ref_dir / "Si.psml")
    case_f = tmp_path / "case_forces.json"
    case_f.write_text(
        json.dumps(
            {
                "name": "mgga_forces",
                "cell_columns_bohr": cell_si.tolist(),
                "positions_bohr": pos_si.tolist(),
                "species": ["Si", "Si"],
                "k_grid": [4, 4, 4],
                "k_shift": [0.5, 0.5, 0.5],
                "mesh_cutoff_ry": 150.0,
                "xc": "R2SCAN01",
                "nspin": 1,
                "mix_weight": 0.3,
                "mix_history": 5,
            }
        ),
        encoding="utf-8",
    )
    obs_f, _ = run_jsiesta_matched_case(
        case_f, ref_dir, ion_dir=ref_dir, use_symmetry=False, reference_mesh_dims=ref_f["_meta"]["mesh_dims"]
    )
    val_f = observables_to_component_validation_dict(obs_f, 1)
    record_residuals(_mgga_residuals("mgga_forces", val_f, ref_f))
    assert abs(val_f["e_xc_ev"] - ref_f["e_xc_ev"]) < 5e-3
    assert abs(val_f["e_kinetic_ev"] - ref_f["e_kinetic_ev"]) < 5e-3
    assert np.max(np.abs(np.array(val_f["forces"]) - np.array(ref_f["forces"]))) < 1e-2
    assert np.max(np.abs(np.array(val_f["stress"]) - np.array(ref_f["stress"]))) < 1e-3

    # 2. mgga_noncol (Si2 magnetic dimer along x, nspin=4, r2SCAN01)
    nc_dir = tmp_path / "ref_noncol"
    ref_nc = _run_pr_mgga_fdf(
        "mgga_noncol.fdf",
        _PR_MGGA_TESTS / "Pseudos" / "14_Si_mgga.psml",
        "14_Si_mgga.psml",
        nc_dir,
        "mgga_noncol",
        4,
    )
    shutil.copy2(_PR_MGGA_TESTS / "Pseudos" / "14_Si_mgga.psml", nc_dir / "Si.psml")
    case_nc = tmp_path / "case_noncol.json"
    case_nc.write_text(
        json.dumps(
            {
                "name": "mgga_noncol",
                "cell_columns_bohr": [[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
                "positions_bohr": [[4.9, 7.5, 7.5], [10.1, 7.8, 7.5]],
                "species": ["Si", "Si"],
                "k_grid": [1, 1, 1],
                "mesh_cutoff_ry": 120.0,
                "xc": "R2SCAN01",
                "nspin": 4,
                "initial_moments": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "electronic_temp_ha": 0.300 / 27.211386245988,
                "mix_weight": 0.05,
                "mix_history": 12,
                "mixer_variant": "pulay",
                "max_scf_iterations": 80,
                "d_tol": 1e-5,
            }
        ),
        encoding="utf-8",
    )
    # mgga_noncol/mgga_soc set an explicit ``Mesh.Sizes`` block (no cutoff criterion), so the mesh is passed through.
    obs_nc, _ = run_jsiesta_matched_case(
        case_nc, nc_dir, ion_dir=nc_dir, use_symmetry=False, mesh_shape_override=ref_nc["_meta"]["mesh_dims"]
    )
    val_nc = observables_to_component_validation_dict(obs_nc, 4)
    record_residuals(_mgga_residuals("mgga_noncol", val_nc, ref_nc))
    assert abs(val_nc["e_xc_ev"] - ref_nc["e_xc_ev"]) < 1e-2
    assert abs(val_nc["total_spin_moment"] - ref_nc["total_spin_moment"]) < 5e-3
    assert np.max(np.abs(np.array(val_nc["forces"]) - np.array(ref_nc["forces"]))) < 5e-2
    assert np.max(np.abs(np.array(val_nc["stress"]) - np.array(ref_nc["stress"]))) < 5e-4

    # 3. mgga_soc (Fe atom spin-orbit r2SCAN01, nspin=8)
    soc_dir = tmp_path / "ref_soc"
    ref_soc = _run_pr_mgga_fdf(
        "mgga_soc.fdf",
        _PR_MGGA_TESTS / "Pseudos" / "Fe_soc.psml",
        "Fe_soc.psml",
        soc_dir,
        "mgga_soc",
        8,
    )
    shutil.copy2(_PR_MGGA_TESTS / "Pseudos" / "Fe_soc.psml", soc_dir / "Fe.psml")
    shutil.copy2(soc_dir / "Fe_soc.ion.xml", soc_dir / "Fe.ion.xml")
    th, ph = math.radians(60.0), math.radians(30.0)
    m0 = [4.0 * math.sin(th) * math.cos(ph), 4.0 * math.sin(th) * math.sin(ph), 4.0 * math.cos(th)]
    case_soc = tmp_path / "case_soc.json"
    case_soc.write_text(
        json.dumps(
            {
                "name": "mgga_soc",
                "cell_columns_bohr": [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]],
                "positions_bohr": [[6.0, 6.3, 5.7]],
                "species": ["Fe"],
                "k_grid": [1, 1, 1],
                "mesh_cutoff_ry": 120.0,
                "xc": "R2SCAN01",
                "nspin": 8,
                "initial_moments": [m0],
                "electronic_temp_ha": 0.300 / 27.211386245988,
                "mix_weight": 0.3,
                "mixer_variant": "pulay",
                "max_scf_iterations": 80,
                "d_tol": 1e-5,
            }
        ),
        encoding="utf-8",
    )
    obs_soc, _ = run_jsiesta_matched_case(
        case_soc, soc_dir, ion_dir=soc_dir, use_symmetry=False, mesh_shape_override=ref_soc["_meta"]["mesh_dims"]
    )
    val_soc = observables_to_component_validation_dict(obs_soc, 8)
    record_residuals(_mgga_residuals("mgga_soc", val_soc, ref_soc))
    assert abs(val_soc["e_xc_ev"] - ref_soc["e_xc_ev"]) < 1e-2
    assert abs(val_soc["fermi_level_ev"] - ref_soc["fermi_level_ev"]) < 1e-2
    assert abs(val_soc["total_spin_moment"] - ref_soc["total_spin_moment"]) < 1e-4
    assert np.max(np.abs(np.array(val_soc["stress"]) - np.array(ref_soc["stress"]))) < 5e-4
