r"""CUDA GPU XLA FFI verification tests for Phase 3 real-space grid, XC, and SCF parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core.ffi import ffi_backend_info
from jax_siesta.grid.mesh import (
    _pack_basis_grid_buffers,
    _pack_na_buffers,
    choose_mesh_dimensions,
    neutral_atom_grid_differentiable,
    rhoofd_differentiable,
    tau_core_grid_differentiable,
    tauofd_differentiable,
    vmat_differentiable,
    vmat_tau_differentiable,
)
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.testing.run_case import load_case_json
from jax_siesta.xc.functional import (
    LibxcFFIEvaluator,
    resolve_xc_id,
    resolve_xc_spec,
    xc_energy_from_grid,
)

_CASES_DIR = Path(__file__).resolve().parent / "manifests" / "cases"
_PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"


def _gpu_available() -> bool:
    try:
        info = ffi_backend_info()
        return bool(info.get("has_cuda", False)) and jax.default_backend() == "gpu"
    except Exception:
        return False


def _load_si_fixture():
    case = load_case_json(_CASES_DIR / "si_bulk" / "case.json")
    s = case["structure"]
    cfg = case["config"]
    loader = PSMLBasisLoader()
    sp_map = {lbl: loader.load_species(lbl, str(_PSML_DIR / f"{lbl}.psml")) for lbl in s.species_labels}
    basis = loader.build_basis(s, sp_map)
    pat = build_sparse_pattern(s, basis)
    return s, basis, pat, cfg, case["k_grid"]


def _set_plan_mode(monkeypatch: pytest.MonkeyPatch, plan_mode: str) -> None:
    """Select the device grid plan mode: resident on the device, or streamed from the host in small chunks (D-043),
    with page-locked host memory and copies overlapped with kernels (D-044) or with both switched off."""
    if plan_mode in ("streamed", "streamed_serial"):
        monkeypatch.setenv("JSIESTA_CUDA_PLAN_STREAM", "1")
        monkeypatch.setenv("JSIESTA_CUDA_PLAN_CHUNK_VALS", "20000")  # many chunks on the 16^3 test mesh
        serial = "0" if plan_mode == "streamed_serial" else "1"
        monkeypatch.setenv("JSIESTA_CUDA_PLAN_PIN", serial)
        monkeypatch.setenv("JSIESTA_CUDA_PLAN_OVERLAP", serial)
    else:
        monkeypatch.setenv("JSIESTA_CUDA_PLAN_STREAM", "0")


@pytest.mark.skipif(not _gpu_available(), reason="CUDA XLA FFI backend or GPU device not active.")
@pytest.mark.parametrize("plan_mode", ["resident", "streamed", "streamed_serial"])
def test_cuda_grid_and_dfscf_parity_and_adjointness(monkeypatch: pytest.MonkeyPatch, plan_mode: str) -> None:
    """Verify CUDA ``rhoofd``, ``vmat``, ``dfscf_grad``, ``na_grid``, and ``na_grid_grad`` vs CPU."""
    _set_plan_mode(monkeypatch, plan_mode)
    s, basis, pat, cfg, _ = _load_si_fixture()
    packed = _pack_basis_grid_buffers(basis, pat)
    na_packed = _pack_na_buffers(s, basis)
    mesh_shape = (16, 16, 16)
    n_spin_mat = 2
    nspin = 2
    rng = np.random.default_rng(7)
    dm = jnp.asarray(0.05 * rng.standard_normal((n_spin_mat, pat.n_edges)), dtype=jnp.float64)
    v_g = jnp.asarray(rng.standard_normal((n_spin_mat, *mesh_shape)), dtype=jnp.float64)

    cpu_dev = jax.devices("cpu")[0]
    gpu_dev = jax.devices("gpu")[0]

    def _run_rho(dm_in, pos_in, cell_in, p_in):
        return rhoofd_differentiable(dm_in, pos_in, cell_in, p_in, mesh_shape, nspin)

    def _run_vmat(vg_in, pos_in, cell_in, p_in):
        return vmat_differentiable(vg_in, pos_in, cell_in, p_in, pat.n_edges, nspin)

    rho_cpu = jax.jit(_run_rho)(
        jax.device_put(dm, cpu_dev),
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), packed),
    )
    rho_gpu = jax.jit(_run_rho)(
        jax.device_put(dm, gpu_dev),
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), packed),
    )
    assert np.max(np.abs(np.asarray(rho_gpu) - np.asarray(rho_cpu))) < 1e-11

    vmat_cpu = jax.jit(_run_vmat)(
        jax.device_put(v_g, cpu_dev),
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), packed),
    )
    vmat_gpu = jax.jit(_run_vmat)(
        jax.device_put(v_g, gpu_dev),
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), packed),
    )
    assert np.max(np.abs(np.asarray(vmat_gpu) - np.asarray(vmat_cpu))) < 1e-11

    # Discrete adjointness directly on GPU: <V | rho(D)> == Tr[D V]
    dvol = float(jnp.abs(jnp.linalg.det(s.cell))) / float(np.prod(mesh_shape))
    lhs = float(jnp.sum(v_g * rho_gpu)) * dvol
    rhs = float(jnp.sum(dm * vmat_gpu))
    assert abs(lhs - rhs) < 1e-12

    # Position and cell gradient parity (dfscf_grad)
    def _loss_rho(dm_in, pos_in, cell_in, p_in, v_in):
        r = rhoofd_differentiable(dm_in, pos_in, cell_in, p_in, mesh_shape, nspin)
        return jnp.sum(r * v_in)

    g_cpu = jax.jit(jax.grad(_loss_rho, argnums=(0, 1, 2)))(
        jax.device_put(dm, cpu_dev),
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), packed),
        jax.device_put(v_g, cpu_dev),
    )
    g_gpu = jax.jit(jax.grad(_loss_rho, argnums=(0, 1, 2)))(
        jax.device_put(dm, gpu_dev),
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), packed),
        jax.device_put(v_g, gpu_dev),
    )
    for gc, gg in zip(g_cpu, g_gpu, strict=True):
        assert np.max(np.abs(np.asarray(gg) - np.asarray(gc))) < 1e-10

    # Neutral atom grid & grad parity
    def _loss_na(pos_in, cell_in, na_in):
        rna, vna, rc = neutral_atom_grid_differentiable(pos_in, cell_in, na_in, mesh_shape)
        return jnp.sum(0.3 * rna + 0.5 * vna + 0.2 * rc)

    na_c = jax.jit(jax.value_and_grad(_loss_na, argnums=(0, 1)))(
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), na_packed),
    )
    na_g = jax.jit(jax.value_and_grad(_loss_na, argnums=(0, 1)))(
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), na_packed),
    )
    assert abs(float(na_g[0]) - float(na_c[0])) < 1e-10
    for gc, gg in zip(na_c[1], na_g[1], strict=True):
        assert np.max(np.abs(np.asarray(gg) - np.asarray(gc))) < 1e-10


@pytest.mark.skipif(not _gpu_available(), reason="CUDA XLA FFI backend or GPU device not active.")
@pytest.mark.parametrize("xc_name", ["LDA", "LDA_PW92", "PBE", "PBEsol"])
@pytest.mark.parametrize("nspin", [1, 2])
def test_cuda_xc_parity(xc_name: str, nspin: int) -> None:
    """Verify CUDA Dual-AD XC energy and potential derivatives match CPU across functionals."""
    xc_id = resolve_xc_id(xc_name)
    mesh = (12, 12, 12)
    cell = jnp.eye(3, dtype=jnp.float64) * 10.0
    x = jnp.linspace(0.0, 2.0 * jnp.pi, mesh[0], endpoint=False)
    gx, gy, gz = jnp.meshgrid(x, x, x, indexing="ij")
    rho0 = 0.08 + 0.03 * jnp.sin(gx) * jnp.cos(gy) + 0.01 * jnp.cos(gz)
    if nspin == 1:
        rho_in = rho0[None, ...]
    else:
        rho_in = jnp.stack([0.55 * rho0, 0.45 * rho0], axis=0)

    cpu_dev = jax.devices("cpu")[0]
    gpu_dev = jax.devices("gpu")[0]

    def _val_grad(r_g, c_m):
        return jax.value_and_grad(lambda r: xc_energy_from_grid(r, c_m, nspin, xc_id))(r_g)

    ec, gc = jax.jit(_val_grad)(jax.device_put(rho_in, cpu_dev), jax.device_put(cell, cpu_dev))
    eg, gg = jax.jit(_val_grad)(jax.device_put(rho_in, gpu_dev), jax.device_put(cell, gpu_dev))
    assert abs(float(eg) - float(ec)) < 1e-9
    assert np.max(np.abs(np.asarray(gg) - np.asarray(gc))) < 1e-9


@pytest.mark.skipif(not _gpu_available(), reason="CUDA XLA FFI backend or GPU device not active.")
@pytest.mark.parametrize("xc_name", ["LDA", "LDA_PW92", "PBE", "PBEsol"])
@pytest.mark.parametrize("nspin", [1, 2])
def test_cuda_builtin_xc_pointwise_matches_cpu(xc_name: str, nspin: int) -> None:
    """The device LDA/GGA kernel (D-039) reproduces the CPU path (Libxc when compiled in) point by point.

    Same inputs and tolerances as ``test_builtin_xc_kernel_matches_libxc``: densities from 1e-11 to 20 a.u., energy
    density and v_rho to relative 1e-12 / 1e-11, v_sigma on the energy scale.
    """
    from test_phase1_grid_xc import _builtin_xc_inputs

    from jax_siesta.core.ffi import register_grid_xc_target
    from jax_siesta.xc.functional import _jax_ffi

    xc_id = resolve_xc_id(xc_name)
    rho, sig = _builtin_xc_inputs(nspin)
    n_pts = rho.shape[1]
    register_grid_xc_target("jsiesta_xc_eval_f64")
    call = _jax_ffi.ffi_call(
        "jsiesta_xc_eval_f64",
        (
            jax.ShapeDtypeStruct((n_pts,), jnp.float64),
            jax.ShapeDtypeStruct((2, n_pts), jnp.float64),
            jax.ShapeDtypeStruct((3, n_pts), jnp.float64),
        ),
    )

    def _eval(dev: jax.Device) -> list[np.ndarray]:
        r_d, s_d = jax.device_put(jnp.asarray(rho), dev), jax.device_put(jnp.asarray(sig), dev)
        out = jax.jit(lambda r, s: call(r, s, xc_id=xc_id, n_spin_pol=nspin))(r_d, s_d)
        return [np.asarray(x) for x in out]

    e_c, vr_c, vs_c = _eval(jax.devices("cpu")[0])
    e_g, vr_g, vs_g = _eval(jax.devices("gpu")[0])
    live = rho.sum(axis=0) > 1e-12
    assert np.all(np.abs(e_g - e_c) <= 1e-12 * np.abs(e_c))
    assert np.all((np.abs(vr_g - vr_c) <= 1e-11 * np.abs(vr_c))[:, live])
    assert np.all(((np.abs(vs_g - vs_c) * np.abs(sig)) <= 1e-12 * np.abs(e_c)[None, :])[:, live])


@pytest.mark.skipif(not _gpu_available(), reason="CUDA XLA FFI backend or GPU device not active.")
def test_cuda_end_to_end_scf_and_observables() -> None:
    """Verify complete GPU SCF + Post-SCF observables (forces & stress) match CPU on Si bulk."""
    s, basis, pat, cfg, k_grid = _load_si_fixture()
    kmesh = build_monkhorst_pack_mesh(k_grid)
    ms = choose_mesh_dimensions(s.cell, cfg.mesh_cutoff_ry)
    tc = TwoCenterEvaluator().evaluate(s, basis, pat, cfg)

    cpu_dev = jax.devices("cpu")[0]
    gpu_dev = jax.devices("gpu")[0]

    with jax.default_device(cpu_dev):
        st_cpu, _ = run_scf(s, basis, pat, tc, kmesh, cfg, mesh_shape=ms)
        obs_cpu = StaticObservableEvaluator().evaluate(s, basis, pat, tc, kmesh, st_cpu, cfg, mesh_shape=ms)

    with jax.default_device(gpu_dev):
        st_gpu, _ = run_scf(s, basis, pat, tc, kmesh, cfg, mesh_shape=ms)
        obs_gpu = StaticObservableEvaluator().evaluate(s, basis, pat, tc, kmesh, st_gpu, cfg, mesh_shape=ms)

    assert obs_gpu.converged
    assert obs_gpu.dm_out_recomputed
    assert abs(float(obs_gpu.total_energy_ha) - float(obs_cpu.total_energy_ha)) < 1e-10
    assert np.max(np.abs(np.asarray(obs_gpu.forces_ha_per_bohr) - np.asarray(obs_cpu.forces_ha_per_bohr))) < 1e-9
    assert np.max(np.abs(np.asarray(obs_gpu.stress_ha_per_bohr3) - np.asarray(obs_cpu.stress_ha_per_bohr3))) < 1e-9
    assert np.max(np.abs(np.sum(np.asarray(obs_gpu.forces_ha_per_bohr), axis=0))) < 1e-4
    assert LibxcFFIEvaluator(cfg.xc_functional).xc_id == 10


def _gpu_libxc_available() -> bool:
    try:
        info = ffi_backend_info()
        return (
            bool(info.get("has_cuda", False))
            and bool(info.get("has_libxc", False))
            and jax.default_backend() == "gpu"
        )
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="CUDA XLA FFI backend or GPU device not active.")
@pytest.mark.parametrize("plan_mode", ["resident", "streamed", "streamed_serial"])
def test_cuda_mgga_grid_and_dfscf_parity(monkeypatch: pytest.MonkeyPatch, plan_mode: str) -> None:
    """Verify CUDA ``tauofd``, ``vmat_tau``, ``tau_dfscf_grad``, ``tau_core``, and ``tau_core_grad``."""
    _set_plan_mode(monkeypatch, plan_mode)
    s, basis, pat, _, _ = _load_si_fixture()
    # Attach deterministic radial tau_core table to verify NLCC tau_core CUDA kernels.
    sp0 = basis.species[0]
    r_grid = np.linspace(0.0, float(sp0.r_grid[-1]), len(sp0.rho_na), dtype=np.float64)
    tc_tab = 0.05 * np.exp(-0.7 * (r_grid**2))
    sp0_tc = replace(sp0, tau_core=jnp.asarray(tc_tab, dtype=jnp.float64), has_tau_core=True)
    basis = replace(basis, species=(sp0_tc,))
    packed = _pack_basis_grid_buffers(basis, pat)
    na_packed = _pack_na_buffers(s, basis)

    mesh_shape = (16, 16, 16)
    nspin = 4
    n_spin_mat = 4
    rng = np.random.default_rng(19)
    dm = jnp.asarray(0.05 * rng.standard_normal((n_spin_mat, pat.n_edges)), dtype=jnp.float64)
    v_tau = jnp.asarray(rng.standard_normal((n_spin_mat, *mesh_shape)), dtype=jnp.float64)

    cpu_dev = jax.devices("cpu")[0]
    gpu_dev = jax.devices("gpu")[0]

    tau_cpu = jax.jit(lambda d, p, c, pk: tauofd_differentiable(d, p, c, pk, mesh_shape, nspin))(
        jax.device_put(dm, cpu_dev),
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), packed),
    )
    tau_gpu = jax.jit(lambda d, p, c, pk: tauofd_differentiable(d, p, c, pk, mesh_shape, nspin))(
        jax.device_put(dm, gpu_dev),
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), packed),
    )
    assert np.max(np.abs(np.asarray(tau_gpu) - np.asarray(tau_cpu))) < 1e-11

    vmt_cpu = jax.jit(lambda v, p, c, pk: vmat_tau_differentiable(v, p, c, pk, pat.n_edges, nspin))(
        jax.device_put(v_tau, cpu_dev),
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), packed),
    )
    vmt_gpu = jax.jit(lambda v, p, c, pk: vmat_tau_differentiable(v, p, c, pk, pat.n_edges, nspin))(
        jax.device_put(v_tau, gpu_dev),
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), packed),
    )
    assert np.max(np.abs(np.asarray(vmt_gpu) - np.asarray(vmt_cpu))) < 1e-11

    dvol = float(jnp.abs(jnp.linalg.det(s.cell))) / float(np.prod(mesh_shape))
    assert abs(float(jnp.sum(v_tau * tau_gpu)) * dvol - float(jnp.sum(dm * vmt_gpu))) < 1e-12

    def _loss_tau(d, p, c, pk, vt):
        return jnp.sum(tauofd_differentiable(d, p, c, pk, mesh_shape, nspin) * vt)

    gc_tau = jax.jit(jax.grad(_loss_tau, argnums=(0, 1, 2)))(
        jax.device_put(dm, cpu_dev),
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), packed),
        jax.device_put(v_tau, cpu_dev),
    )
    gg_tau = jax.jit(jax.grad(_loss_tau, argnums=(0, 1, 2)))(
        jax.device_put(dm, gpu_dev),
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), packed),
        jax.device_put(v_tau, gpu_dev),
    )
    for gc, gg in zip(gc_tau, gg_tau, strict=True):
        np.testing.assert_allclose(np.asarray(gg), np.asarray(gc), rtol=1e-12, atol=1e-10)

    def _loss_tc(p, c, na_p):
        return jnp.sum(0.4 * tau_core_grid_differentiable(p, c, na_p, mesh_shape))

    tc_c = jax.jit(jax.value_and_grad(_loss_tc, argnums=(0, 1)))(
        jax.device_put(s.positions, cpu_dev),
        jax.device_put(s.cell, cpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, cpu_dev), na_packed),
    )
    tc_g = jax.jit(jax.value_and_grad(_loss_tc, argnums=(0, 1)))(
        jax.device_put(s.positions, gpu_dev),
        jax.device_put(s.cell, gpu_dev),
        jax.tree.map(lambda x: jax.device_put(x, gpu_dev), na_packed),
    )
    assert abs(float(tc_g[0]) - float(tc_c[0])) < 1e-11
    for gc, gg in zip(tc_c[1], tc_g[1], strict=True):
        assert np.max(np.abs(np.asarray(gg) - np.asarray(gc))) < 1e-10


@pytest.mark.skipif(not _gpu_libxc_available(), reason="CUDA Libxc meta-GGA backend or GPU device not active.")
def test_cuda_mgga_xc_and_end_to_end_scf_parity() -> None:
    """Verify CUDA Libxc ``r2SCAN01`` grid VJP and end-to-end SCF + observables match CPU."""
    _, xc_id, xc_pair = resolve_xc_spec("R2SCAN01")
    mesh = (12, 12, 12)
    cell = jnp.eye(3, dtype=jnp.float64) * 10.0
    x = jnp.linspace(0.0, 2.0 * jnp.pi, mesh[0], endpoint=False)
    gx, gy, gz = jnp.meshgrid(x, x, x, indexing="ij")
    rho0 = 0.08 + 0.03 * jnp.sin(gx) * jnp.cos(gy) + 0.01 * jnp.cos(gz)
    tau0 = 0.12 + 0.04 * (jnp.sin(gx) ** 2 + jnp.cos(gz) ** 2)
    rho_in = jnp.stack([0.55 * rho0, 0.45 * rho0], axis=0)
    tau_in = jnp.stack([0.58 * tau0, 0.42 * tau0], axis=0)

    cpu_dev = jax.devices("cpu")[0]
    gpu_dev = jax.devices("gpu")[0]

    def _mgga_vg(r_g, t_g, c_m):
        return jax.value_and_grad(
            lambda r, t: xc_energy_from_grid(r, c_m, 2, xc_id, tau_grid=t, xc_pair=xc_pair),
            argnums=(0, 1),
        )(r_g, t_g)

    ec, (gr_c, gt_c) = jax.jit(_mgga_vg)(
        jax.device_put(rho_in, cpu_dev),
        jax.device_put(tau_in, cpu_dev),
        jax.device_put(cell, cpu_dev),
    )
    eg, (gr_g, gt_g) = jax.jit(_mgga_vg)(
        jax.device_put(rho_in, gpu_dev),
        jax.device_put(tau_in, gpu_dev),
        jax.device_put(cell, gpu_dev),
    )
    assert abs(float(eg) - float(ec)) < 1e-10
    assert np.max(np.abs(np.asarray(gr_g) - np.asarray(gr_c))) < 1e-9
    assert np.max(np.abs(np.asarray(gt_g) - np.asarray(gt_c))) < 1e-9

    s, basis, pat, cfg, k_grid = _load_si_fixture()
    cfg_m = replace(cfg, xc_functional="R2SCAN01")
    kmesh = build_monkhorst_pack_mesh(k_grid)
    ms = choose_mesh_dimensions(s.cell, cfg_m.mesh_cutoff_ry)
    tc = TwoCenterEvaluator().evaluate(s, basis, pat, cfg_m)

    with jax.default_device(cpu_dev):
        st_c, _ = run_scf(s, basis, pat, tc, kmesh, cfg_m, mesh_shape=ms)
        obs_c = StaticObservableEvaluator().evaluate(s, basis, pat, tc, kmesh, st_c, cfg_m, mesh_shape=ms)

    with jax.default_device(gpu_dev):
        st_g, _ = run_scf(s, basis, pat, tc, kmesh, cfg_m, mesh_shape=ms)
        obs_g = StaticObservableEvaluator().evaluate(s, basis, pat, tc, kmesh, st_g, cfg_m, mesh_shape=ms)

    assert obs_g.converged
    assert abs(float(obs_g.total_energy_ha) - float(obs_c.total_energy_ha)) < 1e-10
    assert np.max(np.abs(np.asarray(obs_g.forces_ha_per_bohr) - np.asarray(obs_c.forces_ha_per_bohr))) < 1e-9
    assert np.max(np.abs(np.asarray(obs_g.stress_ha_per_bohr3) - np.asarray(obs_c.stress_ha_per_bohr3))) < 1e-9
