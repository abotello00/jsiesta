"""Phase 2 Step 2 tests: symmetry-closed sparsity, Wigner-D, and sparse matrix projection."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core import SCFConfig
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json
from jax_siesta.scf import GeneralizedEigenSolver, build_h0_sparse, build_monkhorst_pack_mesh
from jax_siesta.species import PSMLBasisLoader
from jax_siesta.symmetry.projection import (
    apply_projection_table,
    build_symmetry_closed_pattern,
    compile_projection_table,
    compute_wigner_d_matrices,
)
from jax_siesta.symmetry.spacegroup import build_ibz_kpoint_mesh, extract_space_group

CASES_DIR = Path(__file__).resolve().parent / "manifests" / "cases"
PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"


def _make_random_hermitian_matrix(pattern, rng: np.random.Generator, nspin: int) -> jnp.ndarray:
    """Create a random Hermitian sparse matrix on ``pattern`` respecting ``transpose_edge_idx``."""
    n_edges = pattern.n_edges
    t_idx = np.asarray(pattern.transpose_edge_idx, dtype=np.int32)
    src = np.asarray(pattern.src_orb, dtype=np.int32)
    dst = np.asarray(pattern.dst_unit_orb, dtype=np.int32)

    def _sym_block(cplx: bool = False) -> np.ndarray:
        a = rng.standard_normal(n_edges) + (1j * rng.standard_normal(n_edges) if cplx else 0.0)
        return 0.5 * (a + np.conj(a[t_idx]))

    if nspin == 1:
        return jnp.asarray(_sym_block(False)[None, :], dtype=jnp.float64)
    if nspin == 2:
        return jnp.asarray(np.stack([_sym_block(False), _sym_block(False)], axis=0), dtype=jnp.float64)

    # nspin in (4, 8): M_uu, M_dd complex Hermitian; M_du = M_ud^H -> M_ud independent across (e, e^T) pairs
    m_uu = _sym_block(True)
    m_dd = _sym_block(True)
    m_ud = rng.standard_normal(n_edges) + 1j * rng.standard_normal(n_edges)
    # Remove diagonal ambiguity on self-transpose edges (e == e^T): fine as-is because M_10(e) = conj(M_01(e^T))
    _ = (src, dst)
    return jnp.asarray(
        np.stack([m_uu, m_dd, np.real(m_ud).astype(np.complex128), np.imag(m_ud).astype(np.complex128)], axis=0),
        dtype=jnp.complex128,
    )


@pytest.mark.parametrize(
    ("cname", "nspin"),
    [
        ("si_bulk", 1),
        ("co_hcp_collinear", 2),
        ("co_afm", 2),
        ("co_hcp_nc", 4),
        ("co_hcp_soc", 8),
    ],
)
def test_symmetry_closed_pattern_wigner_d_and_projection_invariants(cname: str, nspin: int) -> None:
    case = load_case_json(CASES_DIR / cname / "case.json")
    structure = case["structure"]
    config = SCFConfig(nspin=nspin)

    sg = extract_space_group(structure, config)
    loader = PSMLBasisLoader()
    sp_map = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, sp_map, include_soc_projectors=(nspin == 8))

    pattern = build_symmetry_closed_pattern(structure, basis, sg)
    assert pattern.symmetry_closed
    t_idx = np.asarray(pattern.transpose_edge_idx, dtype=np.int32)
    assert np.array_equal(t_idx[t_idx], np.arange(pattern.n_edges, dtype=np.int32))

    l_max = max(basis.orb_l)
    wigner_d = compute_wigner_d_matrices(sg.rotations_cart, l_max=l_max, n_fib_pts=16, orth_tol=1e-10)
    for ang_l, d_arr in wigner_d.items():
        # Verify group multiplication D^l(I) == I and D^l(S) D^l(S^-1) == I
        assert np.allclose(d_arr[0], np.eye(2 * ang_l + 1), atol=1e-12)
        for s in range(sg.n_ops):
            s_inv = int(sg.inv_op[s])
            assert np.allclose(d_arr[s] @ d_arr[s_inv], np.eye(2 * ang_l + 1), atol=1e-11)

    table = compile_projection_table(basis, pattern, sg, wigner_d, nspin=nspin)
    proj_jit = jax.jit(lambda x: apply_projection_table(x, table))

    # 1. Idempotency P_sym^2 == P_sym on random Hermitian sparse matrices
    rng = np.random.default_rng(42 + nspin)
    m_rand = _make_random_hermitian_matrix(pattern, rng, nspin)
    p1 = proj_jit(m_rand)
    p2 = proj_jit(p1)
    assert float(jnp.max(jnp.abs(p2 - p1))) < 1e-11

    # 2. Exact two-center H0 and S invariance: P_sym(S) == S and P_sym(H0) == H0
    tc = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    h0 = build_h0_sparse(tc, nspin)
    p_h0 = proj_jit(h0)
    assert float(jnp.max(jnp.abs(p_h0 - h0))) < 1e-10

    if nspin == 1:
        p_s = proj_jit(tc.overlap[None, :])
        assert float(jnp.max(jnp.abs(p_s[0] - tc.overlap))) < 1e-10

    # 3. Trace / electron-count conservation on DM & IBZ-DM invariance check
    ibz = build_ibz_kpoint_mesh(sg, (2, 2, 2))
    dm, _, _, _, _ = GeneralizedEigenSolver().solve(
        h0, tc.overlap, pattern, ibz, basis.total_valence_electrons, 1e-3, nspin
    )
    tr_before = float(jnp.real(jnp.sum((dm[0] if nspin == 1 else (dm[0] + dm[1])) * tc.overlap)))
    dm_sym = proj_jit(dm)
    tr_after = float(jnp.real(jnp.sum((dm_sym[0] if nspin == 1 else (dm_sym[0] + dm_sym[1])) * tc.overlap)))
    assert abs(tr_before - tr_after) < 1e-10

    # Compare full-BZ DM of H0 vs P_sym(IBZ DM of H0)!
    full_km = build_monkhorst_pack_mesh((2, 2, 2))
    dm_full, _, _, _, _ = GeneralizedEigenSolver().solve(
        h0, tc.overlap, pattern, full_km, basis.total_valence_electrons, 1e-3, nspin
    )
    # Note: zero-moment co_hcp_soc/nc with H0 only can have degenerate Fermi states unless small Zeeman seed,
    # while on non-degenerate si_bulk / h0_soc the symmetrized IBZ DM matches full-BZ DM closely!
    if cname in ("si_bulk", "co_hcp_collinear"):
        assert float(jnp.max(jnp.abs(dm_sym - dm_full))) < 1e-9

    # 4. Spin moment covariance: P_sym preserves magnetic moments aligned with crystal symmetry
    if nspin in (4, 8):
        oa = np.asarray(basis.orb_atom, dtype=np.int32)
        src = np.asarray(pattern.src_orb, dtype=np.int32)
        s_vals = np.asarray(tc.overlap, dtype=np.float64)

        def _atom_moments(d_mat: jnp.ndarray) -> np.ndarray:
            dr = np.asarray(jnp.real(d_mat), dtype=np.float64)
            am = np.zeros((structure.n_atoms, 3), dtype=np.float64)
            np.add.at(am[:, 0], oa[src], 2.0 * dr[2] * s_vals)
            np.add.at(am[:, 1], oa[src], -2.0 * dr[3] * s_vals)
            np.add.at(am[:, 2], oa[src], (dr[0] - dr[1]) * s_vals)
            return am

        # Build Zeeman-polarized Hermitian matrix matching initial_moments
        moments = np.asarray(structure.initial_moments, dtype=np.float64)
        sh = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)]
        diag_mask = (src == np.asarray(pattern.dst_unit_orb)) & np.all(sh == 0, axis=1)
        m_seed = np.zeros((4, pattern.n_edges), dtype=np.complex128)
        for e in np.where(diag_mask)[0]:
            mx, my, mz = moments[oa[src[e]]]
            m_seed[0, e] = 0.5 + 0.1 * mz
            m_seed[1, e] = 0.5 - 0.1 * mz
            m_seed[2, e] = 0.1 * mx
            m_seed[3, e] = -0.1 * my
        m_in_sym = proj_jit(jnp.asarray(m_seed))
        m_pre = _atom_moments(jnp.asarray(m_seed))
        m_post = _atom_moments(m_in_sym)
        assert np.allclose(m_post, m_pre, atol=1e-4)
