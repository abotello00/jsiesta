"""Phase 1 Stage 2 & 3 tests: rhoofd/vmat adjointness, dfscf VJP, Poisson, XC FFI, and DFT-D3/rVV10."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core import SCFConfig, Structure
from jax_siesta.grid import GridOps, choose_mesh_dimensions, reciprocal_grid_vectors
from jax_siesta.grid.mesh import siesta_mesh_cutoff_ry
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.species import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.xc import LibxcFFIEvaluator, RVV10Dispersion

jax.config.update("jax_enable_x64", True)
PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"


def _make_si_system():
    loader = PSMLBasisLoader()
    sp_si = loader.load_species("Si", str(PSML_DIR / "Si.psml"))
    struct = Structure(
        cell=jnp.array([[0.0, 5.13, 5.13], [5.13, 0.0, 5.13], [5.13, 5.13, 0.0]]),
        positions=jnp.array([[0.0, 0.0, 0.0], [2.565, 2.565, 2.565]]),
        species_indices=(0, 0),
        atomic_numbers=(14, 14),
        species_labels=("Si",),
        initial_moments=jnp.zeros((2, 3)),
    )
    basis = loader.build_basis(struct, {"Si": sp_si})
    pattern = build_sparse_pattern(struct, basis)
    return struct, basis, pattern


@pytest.mark.parametrize("nspin", [1, 2, 4])
def test_rhoofd_vmat_adjoint_and_electron_count(nspin: int) -> None:
    struct, basis, pattern = _make_si_system()
    tc = TwoCenterEvaluator().evaluate(struct, basis, pattern, SCFConfig(nspin=nspin))
    grid = GridOps()
    mesh = choose_mesh_dimensions(struct.cell, 180.0)
    vol = float(struct.volume)
    dvol = vol / float(np.prod(mesh))

    rng = np.random.default_rng(42 + nspin)
    n_spin_mat = 1 if nspin == 1 else (2 if nspin == 2 else 4)
    dm = rng.normal(scale=0.1, size=(n_spin_mat, pattern.n_edges))
    # Symmetrize random DM across transpose edges for Hermiticity
    tr = np.asarray(pattern.transpose_edge_idx)
    dm = 0.5 * (dm + dm[:, tr])
    if n_spin_mat == 4:
        dm[3] = 0.5 * (dm[3] - dm[3, tr])  # Im(D_ud) is antisymmetric under transpose
    dm_j = jnp.asarray(dm)

    rho = grid.density_on_grid(dm_j, struct, basis, pattern, mesh, nspin)
    v_rand = jnp.asarray(rng.normal(size=rho.shape))
    v_sp = grid.potential_to_matrix(v_rand, struct, basis, pattern, nspin)

    # Exact discrete adjointness: <V | rho(D)>_grid == Tr[D V_sparse]
    lhs = float(dvol * jnp.sum(v_rand * rho))
    rhs = float(jnp.sum(dm_j * v_sp))
    assert abs(lhs - rhs) < 1e-12, f"Adjoint mismatch for nspin={nspin}: {lhs} vs {rhs}"

    # Electron count integral check vs Tr[D S] (for converged mesh)
    s_vals = np.asarray(tc.overlap)
    tr_ds = float(np.sum(dm[0] * s_vals)) if nspin == 1 else float(np.sum((dm[0] + dm[1]) * s_vals))
    rho_tot = rho[0] if nspin == 1 else (rho[0] + rho[1])
    q_grid = float(dvol * jnp.sum(rho_tot))
    assert abs(q_grid - tr_ds) < 1e-2


def test_dfscf_and_na_grid_vjp_vs_finite_differences() -> None:
    struct, basis, pattern = _make_si_system()
    grid = GridOps()
    mesh = (18, 18, 18)
    rng = np.random.default_rng(7)
    dm = jnp.asarray(rng.normal(scale=0.05, size=(1, pattern.n_edges)))
    v_test = jnp.asarray(np.cos(np.linspace(0, 6.28, int(np.prod(mesh)))).reshape((1, *mesh)))

    def _grid_coupling(pos: jax.Array, cell: jax.Array) -> jax.Array:
        st = Structure(
            cell=cell,
            positions=pos,
            species_indices=struct.species_indices,
            atomic_numbers=struct.atomic_numbers,
            species_labels=struct.species_labels,
            initial_moments=struct.initial_moments,
        )
        rho = grid.density_on_grid(dm, st, basis, pattern, mesh, nspin=1)
        dvol = jnp.abs(jnp.linalg.det(cell)) / float(np.prod(mesh))
        return dvol * jnp.sum(rho * v_test)

    g_pos, g_cell = jax.grad(_grid_coupling, argnums=(0, 1))(struct.positions, struct.cell)
    eps = 1e-5
    fd_pos = float(
        _grid_coupling(struct.positions.at[1, 0].add(eps), struct.cell)
        - _grid_coupling(struct.positions.at[1, 0].add(-eps), struct.cell)
    ) / (2.0 * eps)
    assert abs(fd_pos - float(g_pos[1, 0])) < 1e-6

    fd_cell = float(
        _grid_coupling(struct.positions, struct.cell.at[0, 1].add(eps))
        - _grid_coupling(struct.positions, struct.cell.at[0, 1].add(-eps))
    ) / (2.0 * eps)
    assert abs(fd_cell - float(g_cell[0, 1])) < 1e-6

    # Neutral atom + NLCC grid VJP vs finite difference
    w_r = jnp.asarray(v_test[0])
    def _na_coupling(pos: jax.Array, cell: jax.Array) -> jax.Array:
        st = Structure(
            cell=cell,
            positions=pos,
            species_indices=struct.species_indices,
            atomic_numbers=struct.atomic_numbers,
            species_labels=struct.species_labels,
            initial_moments=struct.initial_moments,
        )
        rna, vna, rcore = grid.neutral_atom_grid(st, basis, mesh)
        return jnp.sum(w_r * (rna + vna + rcore))

    na_gp, na_gc = jax.grad(_na_coupling, argnums=(0, 1))(struct.positions, struct.cell)
    fd_na_p = float(
        _na_coupling(struct.positions.at[1, 2].add(eps), struct.cell)
        - _na_coupling(struct.positions.at[1, 2].add(-eps), struct.cell)
    ) / (2.0 * eps)
    assert abs(fd_na_p - float(na_gp[1, 2])) < 1e-6
    fd_na_c = float(
        _na_coupling(struct.positions, struct.cell.at[2, 2].add(eps))
        - _na_coupling(struct.positions, struct.cell.at[2, 2].add(-eps))
    ) / (2.0 * eps)
    assert abs(fd_na_c - float(na_gc[2, 2])) < 1e-6


def test_poisson_residual_and_hartree_stress() -> None:
    cell = jnp.array([[8.0, 0.4, 0.0], [0.2, 8.5, 0.0], [0.0, 0.1, 9.0]])
    n1, n2, n3 = 16, 16, 16
    x = jnp.linspace(0, 1, n1, endpoint=False)
    y = jnp.linspace(0, 1, n2, endpoint=False)
    z = jnp.linspace(0, 1, n3, endpoint=False)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    drho = jnp.sin(2 * jnp.pi * xx) + 0.5 * jnp.cos(2 * jnp.pi * yy) * jnp.sin(2 * jnp.pi * zz)

    grid = GridOps()
    vh, eh, stress_h = grid.hartree_potential(drho, cell)
    _, g2 = reciprocal_grid_vectors((n1, n2, n3), cell)
    lap_vh = jnp.fft.irfftn(-g2 * jnp.fft.rfftn(vh), s=(n1, n2, n3))
    drho_zm = drho - jnp.mean(drho)
    assert float(jnp.max(jnp.abs(lap_vh + 4.0 * jnp.pi * drho_zm))) < 1e-12

    # Verify Hartree stress vs central finite difference of E_H under affine Q-conserving strain
    eps = 1e-5
    vol = float(jnp.abs(jnp.linalg.det(cell)))
    for a, b in ((0, 0), (0, 1), (2, 2)):
        ep = jnp.zeros((3, 3)).at[a, b].set(eps)
        em = jnp.zeros((3, 3)).at[a, b].set(-eps)
        dp, dm = jnp.eye(3) + ep, jnp.eye(3) + em
        _, e_p, _ = grid.hartree_potential(drho / jnp.linalg.det(dp), dp @ cell)
        _, e_m, _ = grid.hartree_potential(drho / jnp.linalg.det(dm), dm @ cell)
        fd_sig = float(e_p - e_m) / (2.0 * eps * vol)
        assert abs(fd_sig - float(stress_h[a, b])) < 1e-8


@pytest.mark.parametrize("func,nspin", [("LDA", 1), ("PBE", 1), ("PBE", 2), ("PBE", 4), ("PBEsol", 2)])
def test_xc_ffi_functional_derivative_and_stress(func: str, nspin: int) -> None:
    cell = jnp.array([[7.0, 0.3, 0.0], [0.1, 7.5, 0.0], [0.0, 0.0, 8.0]])
    n1, n2, n3 = 12, 12, 12
    x = jnp.linspace(0, 1, n1, endpoint=False)
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing="ij")
    rho_tot = 0.08 + 0.04 * jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy) + 0.02 * jnp.cos(2 * jnp.pi * zz)
    if nspin == 1:
        rho = rho_tot[None, ...]
    elif nspin == 2:
        rho = jnp.stack([0.6 * rho_tot, 0.4 * rho_tot], axis=0)
    else:
        rho = jnp.stack([0.55 * rho_tot, 0.45 * rho_tot, 0.05 * rho_tot, -0.03 * rho_tot], axis=0)

    ev = LibxcFFIEvaluator(func)
    e_xc, v_xc, stress_xc = ev.evaluate(rho, cell, nspin)
    vol = float(jnp.abs(jnp.linalg.det(cell)))
    dvol = vol / float(n1 * n2 * n3)

    # Functional derivative test: dE_xc = <v_xc | delta_rho>
    rng = np.random.default_rng(123)
    delta = jnp.asarray(rng.normal(scale=1e-6, size=rho.shape))
    ep, _, _ = ev.evaluate(rho + delta, cell, nspin)
    em, _, _ = ev.evaluate(rho - delta, cell, nspin)
    fd_de = float(ep - em) / 2.0
    if nspin == 4:
        an_de = float(dvol * (
            jnp.sum(v_xc[0] * delta[0])
            + jnp.sum(v_xc[1] * delta[1])
            + 2.0 * jnp.sum(v_xc[2:4] * delta[2:4])
        ))
    else:
        an_de = float(dvol * jnp.sum(v_xc * delta))
    assert abs(fd_de - an_de) < 1e-11, f"XC derivative mismatch {func}/{nspin}: {fd_de} vs {an_de}"

    # Strain stress check
    eps = 1e-5
    deform_p = jnp.eye(3).at[0, 1].add(eps)
    deform_m = jnp.eye(3).at[0, 1].add(-eps)
    sep, _, _ = ev.evaluate(rho / jnp.linalg.det(deform_p), deform_p @ cell, nspin)
    sem, _, _ = ev.evaluate(rho / jnp.linalg.det(deform_m), deform_m @ cell, nspin)
    fd_sig = float(sep - sem) / (2.0 * eps * vol)
    assert abs(fd_sig - float(stress_xc[0, 1])) < 1e-9


def test_rvv10_potential_is_functional_derivative() -> None:
    struct, _, _ = _make_si_system()
    # rVV10 potential functional derivative test
    rvv = RVV10Dispersion()
    n1, n2, n3 = 10, 10, 10
    x = jnp.linspace(0, 1, n1, endpoint=False)
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing="ij")
    rho = (0.05 + 0.02 * jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy))[None, ...]
    e_r, v_r, s_r = rvv.potential_and_energy(rho[0], struct.cell)
    dvol = float(struct.volume) / float(n1 * n2 * n3)
    delta = jnp.asarray(np.random.default_rng(9).normal(scale=1e-6, size=rho[0].shape))
    ep, _, _ = rvv.potential_and_energy(rho[0] + delta, struct.cell)
    em, _, _ = rvv.potential_and_energy(rho[0] - delta, struct.cell)
    assert abs(float(ep - em) / 2.0 - float(dvol * jnp.sum(v_r * delta))) < 1e-11
    assert jnp.all(jnp.isfinite(s_r))


def test_libxc_spec_rejects_hybrids_and_unknown_ids() -> None:
    """Review L3: hybrids must not run silently without exact exchange."""
    import pytest

    from jax_siesta.core.ffi import ffi_backend_info
    from jax_siesta.xc.functional import resolve_xc_spec

    if not ffi_backend_info().get("has_libxc", False):
        pytest.skip("libjsiesta_ffi built without Libxc")
    assert resolve_xc_spec("LIBXC-101+130")[2] == (101, 130)  # PBE exchange + correlation
    with pytest.raises(ValueError, match="hybrid"):
        resolve_xc_spec("LIBXC-402")  # B3LYP
    with pytest.raises(ValueError, match="Unknown Libxc"):
        resolve_xc_spec("LIBXC-99999")


def test_siesta_mesh_criterion_values_and_invariants() -> None:
    """``criterion="siesta"`` ports ``get_mesh_sizes`` + ``chkgmx``; values frozen from real SIESTA 5.4.2 output."""
    ang = 1.0 / 0.529177210903
    a, c = 2.507100 * ang, 4.069500 * ang
    co_hcp = np.array([[a, -0.5 * a, 0.0], [0.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c]])
    si_fcc = 5.13150248 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    triclinic = np.array([[5.2, 0.35, 0.15], [0.2, 5.05, 0.4], [0.1, 0.25, 5.3]]).T  # columns = lattice vectors
    # SIESTA 5.4.2 "InitMesh: MESH =" for MeshCutoff 120 / 150 / 100 Ry (Co hcp parity run, Si parity runs, triclinic).
    assert choose_mesh_dimensions(co_hcp, 120.0, criterion="siesta") == (16, 16, 30)
    assert choose_mesh_dimensions(si_fcc, 150.0, criterion="siesta") == (24, 24, 24)
    assert choose_mesh_dimensions(triclinic, 100.0, criterion="siesta") == (18, 16, 18)
    # SIESTA's criterion is the default (D-027); the legacy Nyquist criterion stays available and differs from it.
    assert choose_mesh_dimensions(co_hcp, 120.0) == (16, 16, 30)
    assert SCFConfig().mesh_criterion == "siesta"
    assert choose_mesh_dimensions(co_hcp, 120.0, criterion="jsiesta") == (18, 18, 27)

    for cell, cutoff in ((co_hcp, 120.0), (si_fcc, 150.0), (triclinic, 100.0)):
        for nsm in (1, 2, 4):
            dims = choose_mesh_dimensions(cell, cutoff, criterion="siesta", mesh_subdivisions=nsm)
            assert all(n % nsm == 0 for n in dims)
            assert siesta_mesh_cutoff_ry(cell, dims) >= cutoff  # alias free up to the requested cutoff
    # Orthogonal cell: chkgmx reduces to (pi N / L)^2.
    assert abs(siesta_mesh_cutoff_ry(np.eye(3) * 16.0, (54, 54, 54)) - (np.pi * 54 / 16.0) ** 2) < 1e-6
    # Symmetry constraints are honoured: equal counts on equivalent axes, a divisor on axis 2.
    dims = choose_mesh_dimensions(
        co_hcp, 120.0, divisors=(1, 1, 4), equiv_axes=((0, 1),), criterion="siesta"
    )
    assert dims[0] == dims[1] and dims[2] % 4 == 0 and dims[2] % 2 == 0
    with pytest.raises(ValueError):
        choose_mesh_dimensions(co_hcp, 120.0, criterion="nope")
    with pytest.raises(ValueError):
        SCFConfig(mesh_criterion="nope")



@pytest.mark.parametrize(("func", "tol"), [("LDA", 1e-6), ("PBE", 2e-4)])
def test_noncollinear_xc_derivative_with_frozen_spin_axis(func: str, tol: float, record_property) -> None:
    """The frozen-axis (libgridxc ``old_scheme``) XC derivative is exact for LDA and nearly so for GGA (review L2).

    The energy is evaluated in the local spin frame with the axis ``u = m/|m|`` held fixed under differentiation, as
    in libgridxc. The omitted term is ``(grad u) . dm`` inside the density gradient, so a GGA derivative is only exact
    where the axis is uniform. This test measures it on a spin spiral (axis rotating through 2 pi per cell, plus a
    z-component) and pins the size of the error, so a regression that worsens it is visible.
    """
    from jax_siesta.xc.functional import XC_IDS, xc_energy_from_grid

    n_pts, length = 16, 8.0
    cell = jnp.eye(3) * length
    ax = jnp.arange(n_pts) / n_pts
    x, y, z = jnp.meshgrid(ax, ax, ax, indexing="ij")
    rho_tot = 0.6 * (1.2 + 0.3 * jnp.cos(2 * np.pi * z) + 0.2 * jnp.sin(2 * np.pi * x))
    phase = 2 * np.pi * (x + 0.5 * y)
    m0 = 0.1
    mx, my, mz = m0 * jnp.cos(phase), m0 * jnp.sin(phase), 0.3 * m0 * jnp.cos(2 * np.pi * y)
    rho = jnp.stack([0.5 * (rho_tot + mz), 0.5 * (rho_tot - mz), 0.5 * mx, -0.5 * my])

    def energy(r):
        return xc_energy_from_grid(r, cell, 4, XC_IDS[func])

    direction = jnp.asarray(np.random.default_rng(0).standard_normal(rho.shape)) * 0.01
    analytic = float(jnp.sum(jax.grad(energy)(rho) * direction))
    step = 1e-4
    numeric = float((energy(rho + step * direction) - energy(rho - step * direction)) / (2 * step))
    rel = abs(analytic - numeric) / abs(numeric)
    record_property(f"resid:frozen_axis_{func}", rel)
    record_property(f"tol:frozen_axis_{func}", tol)
    assert rel < tol


def _builtin_xc_inputs(nspin: int, n_pts: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Random spin densities over 1e-11..20 a.u. with reduced gradients s in [0, 3] (and |s_ud| <= sqrt(s_uu s_dd))."""
    rng = np.random.default_rng(7)
    n_tot = np.exp(rng.uniform(np.log(1e-11), np.log(20.0), n_pts))
    if nspin == 1:
        ru = rd = 0.5 * n_tot
        sig = (2.0 * (3.0 * np.pi**2 * n_tot) ** (1.0 / 3.0) * n_tot * rng.uniform(0.0, 3.0, n_pts)) ** 2
        suu = sud = sdd = 0.25 * sig
    else:
        z = rng.uniform(-0.95, 0.95, n_pts)
        ru, rd = 0.5 * n_tot * (1.0 + z), 0.5 * n_tot * (1.0 - z)

        def _sig_s(r_s: np.ndarray) -> np.ndarray:
            n2 = 2.0 * r_s
            return 0.25 * (2.0 * (3.0 * np.pi**2 * n2) ** (1.0 / 3.0) * n2 * rng.uniform(0.0, 3.0, n_pts)) ** 2

        suu, sdd = _sig_s(ru), _sig_s(rd)
        sud = rng.uniform(-1.0, 1.0, n_pts) * np.sqrt(suu * sdd)
    return np.ascontiguousarray(np.stack([ru, rd])), np.ascontiguousarray(np.stack([suu, sud, sdd]))


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("func", ["LDA", "LDA_PW92", "PBE", "PBEsol"])
def test_builtin_xc_kernel_matches_libxc(func: str, nspin: int) -> None:
    """The built-in LDA/GGA kernel (the arithmetic of the CUDA XC kernel, D-039) reproduces Libxc point by point.

    Compares ``jsiesta_xc_eval_builtin`` (``cpp/include/jsiesta/xc_semilocal.hpp``) with the Libxc path of
    ``jsiesta_xc_eval_f64`` from vacuum (1e-11) to core densities: energy density and v_rho to relative 1e-12, and the
    v_sigma error on the energy scale, |dv_sigma| * |sigma| <= 1e-12 |e_xc|.
    """
    import ctypes

    from jax_siesta.core.ffi import ffi_backend_info, load_ffi_library, register_grid_xc_target
    from jax_siesta.xc.functional import _jax_ffi, resolve_xc_id

    if not ffi_backend_info().get("has_libxc", False):
        pytest.skip("library built without Libxc: nothing to compare against")
    lib = load_ffi_library()
    builtin = lib.jsiesta_xc_eval_builtin
    dptr = ctypes.POINTER(ctypes.c_double)
    builtin.argtypes = [ctypes.c_int64, dptr, dptr, ctypes.c_int32, ctypes.c_int32, dptr, dptr, dptr]
    builtin.restype = ctypes.c_int32

    xc_id = resolve_xc_id(func)
    rho, sig = _builtin_xc_inputs(nspin)
    n_pts = rho.shape[1]
    e_b, vr_b, vs_b = np.zeros(n_pts), np.zeros((2, n_pts)), np.zeros((3, n_pts))
    rc = builtin(
        n_pts, rho.ctypes.data_as(dptr), sig.ctypes.data_as(dptr), xc_id, nspin,
        e_b.ctypes.data_as(dptr), vr_b.ctypes.data_as(dptr), vs_b.ctypes.data_as(dptr),
    )
    assert rc == 0

    register_grid_xc_target("jsiesta_xc_eval_f64")
    call = _jax_ffi.ffi_call(
        "jsiesta_xc_eval_f64",
        (
            jax.ShapeDtypeStruct((n_pts,), jnp.float64),
            jax.ShapeDtypeStruct((2, n_pts), jnp.float64),
            jax.ShapeDtypeStruct((3, n_pts), jnp.float64),
        ),
    )
    e_l, vr_l, vs_l = (np.asarray(x) for x in call(jnp.asarray(rho), jnp.asarray(sig), xc_id=xc_id, n_spin_pol=nspin))

    live = rho.sum(axis=0) > 1e-12
    assert np.all(np.abs(e_b - e_l) <= 1e-12 * np.abs(e_l))
    assert np.all((np.abs(vr_b - vr_l) <= 1e-11 * np.abs(vr_l))[:, live])
    assert np.all(((np.abs(vs_b - vs_l) * np.abs(sig)) <= 1e-12 * np.abs(e_l)[None, :])[:, live])
