"""Phase 1 Step 1 tests: PSML parser, .psf rejection, and C++ two-center integrals (S, T, Vkb, E_NA)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core import SCFConfig, Structure
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.species import PSMLBasisLoader, build_sparse_pattern

jax.config.update("jax_enable_x64", True)
PSML_DIR = Path(__file__).resolve().parent / "fixtures" / "psml"


def _bloch_dense(values: np.ndarray, pattern, k_frac: np.ndarray) -> np.ndarray:
    n = pattern.n_orb
    mat = np.zeros((n, n), dtype=np.complex128)
    src = np.asarray(pattern.src_orb)
    dst = np.asarray(pattern.dst_unit_orb)
    shifts = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)]
    phases = np.exp(2j * np.pi * (shifts @ k_frac))
    for e in range(pattern.n_edges):
        mat[src[e], dst[e]] += values[e] * phases[e]
    return mat


def test_psml_loader_and_psf_rejection(tmp_path: Path) -> None:
    loader = PSMLBasisLoader()
    fake_psf = tmp_path / "Si.psf"
    fake_psf.write_text("legacy psf", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.psf"):
        loader.load_species("Si", str(fake_psf))

    si = loader.load_species("Si", str(PSML_DIR / "Si.psml"))
    assert si.atomic_number == 14
    assert np.isclose(si.valence_charge, 4.0)
    assert si.has_nlcc
    assert len(set(si.pao_iz)) == len(si.pao_iz) or len(si.pao_iz) > 2
    # Check per-l iz uniqueness
    seen_l_iz = set(zip(si.pao_l, si.pao_iz, strict=True))
    assert len(seen_l_iz) == len(si.pao_l)
    # Neutral-atom density integrates to valence_charge
    r = np.asarray(si.r_grid)
    q = float(np.trapezoid(4.0 * np.pi * (r**2) * np.asarray(si.rho_na), r))
    assert np.isclose(q, 4.0, atol=1e-6)
    # Neutral-atom potential vanishes cleanly beyond rcut_na
    assert abs(float(si.v_na[-1])) < 1e-10

    co = loader.load_species("Co", str(PSML_DIR / "Co.psml"))
    assert co.has_spin_orbit
    assert any(j2 > 0 for j2 in co.kb_j2)


def test_two_center_hermiticity_posdef_and_finite_differences() -> None:
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
    ev = TwoCenterEvaluator()
    tc = ev.evaluate(struct, basis, pattern, SCFConfig(nspin=1))

    s_vals = np.asarray(tc.overlap)
    t_vals = np.asarray(tc.kinetic)
    vkb_vals = np.asarray(tc.v_kb)
    tr_idx = np.asarray(pattern.transpose_edge_idx)

    # Diagonal self-overlap == 1.0 (quadrature accuracy), positive-definiteness & Hermiticity
    diag_s = [
        s_vals[e]
        for e in range(pattern.n_edges)
        if int(pattern.src_orb[e]) == int(pattern.dst_unit_orb[e])
        and np.all(np.asarray(pattern.cell_shifts)[int(pattern.edge_cell_idx[e])] == 0)
    ]
    assert np.allclose(diag_s, 1.0, atol=2e-4)
    assert np.allclose(s_vals, s_vals[tr_idx], atol=1e-11)
    assert np.allclose(t_vals, t_vals[tr_idx], atol=1e-11)
    assert np.allclose(vkb_vals, vkb_vals[tr_idx], atol=1e-11)

    for k_frac in ([0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.0, 0.5]):
        sk = _bloch_dense(s_vals, pattern, np.array(k_frac))
        tk = _bloch_dense(t_vals, pattern, np.array(k_frac))
        vk = _bloch_dense(vkb_vals, pattern, np.array(k_frac))
        assert np.allclose(sk, sk.conj().T, atol=1e-11)
        assert np.allclose(tk, tk.conj().T, atol=1e-11)
        assert np.allclose(vk, vk.conj().T, atol=1e-11)
        eigs = np.linalg.eigvalsh(sk)
        assert np.all(eigs > 1e-4), f"S(k) not positive definite: min eig={eigs.min()}"

    # Finite-difference derivative check for dS/dR, dT/dR, and f_na_atoms
    eps = 1e-4
    struct_p = Structure(
        cell=struct.cell,
        positions=struct.positions.at[1, 0].add(eps),
        species_indices=struct.species_indices,
        atomic_numbers=struct.atomic_numbers,
        species_labels=struct.species_labels,
        initial_moments=struct.initial_moments,
    )
    struct_m = Structure(
        cell=struct.cell,
        positions=struct.positions.at[1, 0].add(-eps),
        species_indices=struct.species_indices,
        atomic_numbers=struct.atomic_numbers,
        species_labels=struct.species_labels,
        initial_moments=struct.initial_moments,
    )
    tc_p = ev.evaluate(struct_p, basis, build_sparse_pattern(struct_p, basis), SCFConfig())
    tc_m = ev.evaluate(struct_m, basis, build_sparse_pattern(struct_m, basis), SCFConfig())

    # Match (0, 1, shift=(0,0,0)) edges where d(R_1 - R_0)/dx_1 = +1
    pat_p = build_sparse_pattern(struct_p, basis)
    pat_m = build_sparse_pattern(struct_m, basis)
    def _find_edge_map(pat, vals):
        sh = np.asarray(pat.cell_shifts)[np.asarray(pat.edge_cell_idx)]
        return {
            (int(pat.src_orb[e]), int(pat.dst_unit_orb[e]), tuple(int(x) for x in sh[e])): vals[e]
            for e in range(pat.n_edges)
        }
    sp_map = _find_edge_map(pat_p, np.asarray(tc_p.overlap))
    sm_map = _find_edge_map(pat_m, np.asarray(tc_m.overlap))
    tp_map = _find_edge_map(pat_p, np.asarray(tc_p.kinetic))
    tm_map = _find_edge_map(pat_m, np.asarray(tc_m.kinetic))

    shifts_0 = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)]
    oa = np.asarray(basis.orb_atom)
    max_ds_err = 0.0
    max_dt_err = 0.0
    for e in range(pattern.n_edges):
        mu, nu = int(pattern.src_orb[e]), int(pattern.dst_unit_orb[e])
        t_trip = tuple(int(x) for x in shifts_0[e])
        if oa[mu] == 0 and oa[nu] == 1 and (mu, nu, t_trip) in sp_map and (mu, nu, t_trip) in sm_map:
            fd_s = (sp_map[(mu, nu, t_trip)] - sm_map[(mu, nu, t_trip)]) / (2.0 * eps)
            fd_t = (tp_map[(mu, nu, t_trip)] - tm_map[(mu, nu, t_trip)]) / (2.0 * eps)
            max_ds_err = max(max_ds_err, abs(fd_s - float(tc.d_overlap_dr[e, 0])))
            max_dt_err = max(max_dt_err, abs(fd_t - float(tc.d_kinetic_dr[e, 0])))
    assert max_ds_err < 1e-6, f"dS/dR mismatch: {max_ds_err}"
    assert max_dt_err < 1e-5, f"dT/dR mismatch: {max_dt_err}"

    # f_na_atoms vs -d(e_na_pair)/dR_1
    fd_fna = -(float(tc_p.e_na_pair) - float(tc_m.e_na_pair)) / (2.0 * eps)
    assert abs(fd_fna - float(tc.f_na_atoms[1, 0])) < 1e-6


def test_two_center_rotation_invariance_and_soc_collapse() -> None:
    loader = PSMLBasisLoader()
    sp_co = loader.load_species("Co", str(PSML_DIR / "Co.psml"))
    # Molecule dimer along z vs rotated by arbitrary orthogonal R
    pos0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.2]])
    theta, phi = 0.73, 1.15
    rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
    rot = rz @ rx
    pos1 = pos0 @ rot.T

    ev = TwoCenterEvaluator()
    eigs_list = []
    for pos in (pos0, pos1):
        st = Structure(
            cell=jnp.eye(3) * 25.0,
            positions=jnp.asarray(pos),
            species_indices=(0, 0),
            atomic_numbers=(27, 27),
            species_labels=("Co",),
            initial_moments=jnp.zeros((2, 3)),
            pbc=(False, False, False),
        )
        bas = loader.build_basis(st, {"Co": sp_co}, include_soc_projectors=True)
        pat = build_sparse_pattern(st, bas)
        tc = ev.evaluate(st, bas, pat, SCFConfig(nspin=8))
        s_k = _bloch_dense(np.asarray(tc.overlap), pat, np.zeros(3))
        t_k = _bloch_dense(np.asarray(tc.kinetic), pat, np.zeros(3))
        v_k = _bloch_dense(np.asarray(tc.v_kb), pat, np.zeros(3))
        eigs_list.append((np.linalg.eigvalsh(s_k), np.linalg.eigvalsh(t_k), np.linalg.eigvalsh(v_k)))

    for a, b in zip(eigs_list[0], eigs_list[1], strict=True):
        assert np.allclose(a, b, atol=1e-7)

    # Check SOC Hermiticity & collapse to scalar when j-energies are degenerate
    st = Structure(
        cell=jnp.eye(3) * 25.0,
        positions=jnp.asarray(pos0),
        species_indices=(0, 0),
        atomic_numbers=(27, 27),
        species_labels=("Co",),
        initial_moments=jnp.zeros((2, 3)),
        pbc=(False, False, False),
    )
    sp_deg = type(sp_co)(
        **{
            **{f: getattr(sp_co, f) for f in sp_co.__dataclass_fields__},
            "kb_energies": jnp.array([1.8, 0.75, 0.75, -1.30, -1.30]),
        }
    )
    bas_deg = loader.build_basis(st, {"Co": sp_deg}, include_soc_projectors=True)
    pat_deg = build_sparse_pattern(st, bas_deg)
    tc_deg = ev.evaluate(st, bas_deg, pat_deg, SCFConfig(nspin=8))
    vso = np.asarray(tc_deg.v_kb_so)
    # Spin-flip blocks (up-dn, dn-up) must vanish identically when j=l+-1/2 are degenerate!
    assert np.max(np.abs(vso[2])) < 1e-12
    assert np.max(np.abs(vso[3])) < 1e-12
    assert np.allclose(vso[0], vso[1], atol=1e-12)


def test_species_with_different_table_lengths_are_unified_and_evaluate() -> None:
    """Species whose radial tables differ in length (H and O .ion.xml files, ...) can share one calculation.

    Before D-024 the KB packer stacked tables of unequal length and crashed; the PAO packer resampled linearly onto the
    first species' grid. ``unify_species_tables`` resamples onto the finest grid with a cubic spline.
    """
    from dataclasses import replace

    from jax_siesta.species.psml import unify_species_tables

    loader = PSMLBasisLoader()
    fine = loader.load_species("Si", str(PSML_DIR / "Si.psml"))
    coarse = replace(
        fine,
        r_grid=fine.r_grid[::2], v_local=fine.v_local[::2], v_na=fine.v_na[::2], rho_na=fine.rho_na[::2],
        rho_core=fine.rho_core[::2], tau_core=None if fine.tau_core is None else fine.tau_core[::2],
        pao_r=fine.pao_r[::2], pao_values=fine.pao_values[:, ::2],
        kb_r=fine.kb_r[::2], kb_projectors=fine.kb_projectors[:, ::2],
    )
    assert int(coarse.r_grid.shape[0]) < int(fine.r_grid.shape[0])
    out_coarse, out_fine = unify_species_tables([coarse, fine])
    assert out_fine is fine
    for a, b in (("r_grid", "r_grid"), ("pao_r", "pao_r"), ("kb_r", "kb_r")):
        assert getattr(out_coarse, a).shape == getattr(fine, b).shape
    # Same range, and the cubic-spline resampling reproduces the fine tables of this smooth species.
    assert float(out_coarse.pao_r[-1]) == float(coarse.pao_r[-1])
    scale = float(jnp.max(jnp.abs(fine.pao_values)))
    assert float(jnp.max(jnp.abs(out_coarse.pao_values - fine.pao_values))) < 1e-3 * scale

    # End to end: two species of unequal table length in one cell (this stacked inhomogeneous KB tables before).
    struct = Structure(
        cell=jnp.array([[0.0, 5.13, 5.13], [5.13, 0.0, 5.13], [5.13, 5.13, 0.0]]),
        positions=jnp.array([[0.0, 0.0, 0.0], [2.565, 2.565, 2.565]]),
        species_indices=(0, 1),
        atomic_numbers=(14, 14),
        species_labels=("SiFine", "SiCoarse"),
        initial_moments=jnp.zeros((2, 3)),
    )
    basis = loader.build_basis(struct, {"SiFine": fine, "SiCoarse": coarse})
    pattern = build_sparse_pattern(struct, basis)
    tc = TwoCenterEvaluator().evaluate(struct, basis, pattern, SCFConfig(nspin=1))
    assert bool(jnp.all(jnp.isfinite(tc.overlap))) and bool(jnp.all(jnp.isfinite(tc.kinetic)))


def _brute_force_pattern_edges(structure, basis) -> set[tuple[int, int, tuple[int, int, int]]]:
    """Scalar reference of ``build_sparse_pattern``'s edge set (the pre-vectorisation algorithm)."""
    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    rc_orb = np.asarray(basis.orb_rcut, dtype=np.float64)
    rc_kb = np.asarray(basis.kb_rcut, dtype=np.float64)
    r_int = 2.0 * float(np.max(rc_orb)) + 2.0 * float(np.max(rc_kb))
    b_norms = np.linalg.norm(np.linalg.inv(cell), axis=1)
    half = [int(np.ceil(r_int * b_norms[ax])) if structure.pbc[ax] else 0 for ax in range(3)]
    shifts = [(a, b, c) for a in range(-half[0], half[0] + 1) for b in range(-half[1], half[1] + 1)
              for c in range(-half[2], half[2] + 1)]
    n_at = structure.n_atoms
    offs = basis.atom_orb_offsets
    amo = np.zeros(n_at)
    for mu in range(basis.n_orb):
        amo[basis.orb_atom[mu]] = max(amo[basis.orb_atom[mu]], rc_orb[mu])
    amk = np.zeros(n_at)
    for k, ka in enumerate(basis.kb_atom):
        amk[ka] = max(amk[ka], rc_kb[k])
    links: list[list[tuple[int, tuple[int, int, int]]]] = [[] for _ in range(n_at)]
    edges: set[tuple[int, int, tuple[int, int, int]]] = set()
    for ia in range(n_at):
        for ja in range(n_at):
            for t in shifts:
                d = float(np.linalg.norm(pos[ja] + cell @ np.asarray(t, dtype=np.float64) - pos[ia]))
                if amk[ja] > 0.0 and d <= amo[ia] + amk[ja] + 1e-9:
                    links[ia].append((ja, t))
                if d <= amo[ia] + amo[ja] + 1e-9:
                    for mu in range(offs[ia], offs[ia + 1]):
                        for nu in range(offs[ja], offs[ja + 1]):
                            if d <= float(rc_orb[mu] + rc_orb[nu]) + 1e-9:
                                edges.add((mu, nu, t))
    for ia in range(n_at):
        for ka, t_ik in links[ia]:
            for ja in range(n_at):
                for kb, t_jk in links[ja]:
                    if kb != ka:
                        continue
                    t_ij = (t_ik[0] - t_jk[0], t_ik[1] - t_jk[1], t_ik[2] - t_jk[2])
                    for mu in range(offs[ia], offs[ia + 1]):
                        for nu in range(offs[ja], offs[ja + 1]):
                            edges.add((mu, nu, t_ij))
    closed = set(edges)
    for mu, nu, t in edges:
        closed.add((nu, mu, (-t[0], -t[1], -t[2])))
    return closed


@pytest.mark.parametrize("case_name", ["si_triclinic", "h2o_gamma", "graphene_slab", "co_afm"])
def test_vectorised_neighbour_searches_match_scalar_reference(case_name: str) -> None:
    """The vectorised pattern, KB-link and neutral-atom pair searches equal the scalar loops they replace."""
    from jax_siesta.integrals.two_center import _enumerate_atom_pairs, _enumerate_kb_links
    from jax_siesta.io.fdf import load_case_json

    case = load_case_json(Path(__file__).resolve().parent / "manifests" / "cases" / case_name / "case.json")
    structure = case["structure"]
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species)
    pattern = build_sparse_pattern(structure, basis)

    # 1. Pattern: same edge set, CSR order (src, shift index, dst), exact transposes and edge vectors
    src = np.asarray(pattern.src_orb)
    dst = np.asarray(pattern.dst_unit_orb)
    cidx = np.asarray(pattern.edge_cell_idx)
    shifts = np.asarray(pattern.cell_shifts)
    got = {(int(m), int(n), tuple(int(v) for v in shifts[c])) for m, n, c in zip(src, dst, cidx, strict=True)}
    assert len(got) == pattern.n_edges
    assert got == _brute_force_pattern_edges(structure, basis)
    order_key = (src.astype(np.int64) * len(shifts) + cidx) * basis.n_orb + dst
    assert np.all(np.diff(order_key) > 0)
    tr = np.asarray(pattern.transpose_edge_idx)
    assert np.array_equal(src[tr], dst) and np.array_equal(dst[tr], src)
    assert np.array_equal(shifts[cidx[tr]], -shifts[cidx])
    cell = np.asarray(structure.cell)
    pos = np.asarray(structure.positions)
    orb_atom = np.asarray(basis.orb_atom)
    ref_vec = pos[orb_atom[dst]] + shifts[cidx] @ cell.T - pos[orb_atom[src]]
    np.testing.assert_allclose(np.asarray(pattern.edge_vectors), ref_vec, atol=1e-12)

    # 2. KB links: same rows in the scalar loop's (k_inst, shift, mu, m2) order
    atom_kb_shells = []
    g = 0
    sp_offsets = [0]
    for sp in basis.species:
        sp_offsets.append(sp_offsets[-1] + len(sp.kb_l))
    for ia, sp_idx in enumerate(structure.species_indices):
        sp = basis.species[sp_idx]
        for k in range(len(sp.kb_l)):
            g = sp_offsets[sp_idx] + k
            atom_kb_shells.append((ia, g, float(sp.kb_energies[k]), int(sp.kb_l[k]), int(sp.kb_j2[k]), float(sp.kb_rcut[k])))
    nsc = tuple(max(1, abs(int(s))) for s in pattern.nsc)
    kb_shifts = [(a, b, c) for a in range(-nsc[0], nsc[0] + 1) for b in range(-nsc[1], nsc[1] + 1)
                 for c in range(-nsc[2], nsc[2] + 1)
                 if all(structure.pbc[ax] or (a, b, c)[ax] == 0 for ax in range(3))]
    orb_rc = np.asarray(basis.orb_rcut)
    ref_rows = []
    for k_inst, (ka, _g, ek, l_k, _j2, rc_k) in enumerate(atom_kb_shells):
        if abs(ek) < 1e-16:
            continue
        for s_i, t in enumerate(kb_shifts):
            disp_base = pos[ka] + cell @ np.asarray(t, dtype=np.float64)
            for mu in range(basis.n_orb):
                disp = disp_base - pos[orb_atom[mu]]
                if float(np.linalg.norm(disp)) <= float(orb_rc[mu] + rc_k) + 1e-8:
                    ref_rows.extend((k_inst, s_i, mu, m2) for m2 in range(2 * l_k + 1))
    k_a, s_a, mu_a, d_a, m2_a = _enumerate_kb_links(pos, cell, orb_atom.astype(np.int32), orb_rc, atom_kb_shells, kb_shifts)
    assert list(zip(k_a.tolist(), s_a.tolist(), mu_a.tolist(), m2_a.tolist(), strict=True)) == ref_rows
    ref_d = pos[np.asarray([atom_kb_shells[k][0] for k in k_a], dtype=np.int64)] + np.asarray(kb_shifts)[s_a] @ cell.T
    np.testing.assert_allclose(d_a, ref_d - pos[orb_atom[mu_a]], atol=1e-12)

    # 3. Neutral-atom pairs (i, j, T) with 1e-8 < d < rc_i + rc_j, in (i, j, T) order
    rc_sp = np.asarray([float(sp.rcut_na) for sp in basis.species])
    sp_of = np.asarray(structure.species_indices, dtype=np.int64)
    half = [int(np.ceil(2.0 * rc_sp.max() * np.linalg.norm(np.linalg.inv(cell)[ax]))) if structure.pbc[ax] else 0
            for ax in range(3)]
    ref_pairs = []
    for ia in range(structure.n_atoms):
        for ja in range(structure.n_atoms):
            for a in range(-half[0], half[0] + 1):
                for b in range(-half[1], half[1] + 1):
                    for c in range(-half[2], half[2] + 1):
                        if ia == ja and (a, b, c) == (0, 0, 0):
                            continue
                        d = float(np.linalg.norm(pos[ja] + cell @ np.array([a, b, c], dtype=np.float64) - pos[ia]))
                        if 1e-8 < d < rc_sp[sp_of[ia]] + rc_sp[sp_of[ja]]:
                            ref_pairs.append((ia, ja, d))
    ij, _vec, dist = _enumerate_atom_pairs(pos, cell, sp_of, rc_sp, half)
    assert [(int(i), int(j)) for i, j in ij] == [(i, j) for i, j, _ in ref_pairs]
    np.testing.assert_allclose(dist, [d for _, _, d in ref_pairs], rtol=1e-14)


@pytest.mark.parametrize("nspin", [1, 2])
def test_native_kb_kernel_matches_numpy_reference(nspin: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The C++ KB kernel (D-041) gives the NumPy loop's V_KB, energy, forces and stress.

    Displaced two-species cell (Si and Co fixtures, so shells of l = 0, 1, 2 and unequal tables) with a random
    density matrix: V_KB bitwise up to summation order, forces and stress to 1e-12.
    """
    import jax_siesta.integrals.two_center as tcm
    import jax_siesta.observables.forces as fm

    loader = PSMLBasisLoader()
    species = {
        "Si": loader.load_species("Si", str(PSML_DIR / "Si.psml")),
        "Co": loader.load_species("Co", str(PSML_DIR / "Co.psml")),
    }
    a = 9.0
    struct = Structure(
        cell=jnp.array([[a, 0.3, 0.0], [0.0, a, 0.4], [0.2, 0.0, a]]).T,
        positions=jnp.array([[0.1, 0.0, 0.2], [2.3, 2.1, 2.6], [4.6, 0.3, 4.4], [0.4, 4.7, 4.3]]),
        species_indices=(0, 0, 1, 1),
        atomic_numbers=(14, 14, 27, 27),
        species_labels=("Si", "Co"),
        initial_moments=jnp.zeros((4, 3)),
    )
    cfg = SCFConfig(nspin=nspin, mesh_cutoff_ry=60.0)
    basis = loader.build_basis(struct, species)
    pattern = build_sparse_pattern(struct, basis)
    tc_native = TwoCenterEvaluator().evaluate(struct, basis, pattern, cfg)
    dm = jnp.asarray(
        np.random.default_rng(3).normal(size=(nspin, int(pattern.n_edges))) * 0.02
        + np.asarray(tc_native.overlap)[None] * 0.1
    )
    e_n, f_n, s_n = fm._kb_forces_and_stress(dm, struct, basis, pattern, tc_native, nspin)

    monkeypatch.setattr(tcm, "kb_three_center_native", lambda *args, **kwargs: None)
    monkeypatch.setattr(fm, "kb_three_center_native", lambda *args, **kwargs: None)
    tc_ref = TwoCenterEvaluator().evaluate(struct, basis, pattern, cfg)
    e_r, f_r, s_r = fm._kb_forces_and_stress(dm, struct, basis, pattern, tc_ref, nspin)

    assert np.max(np.abs(f_r)) > 1e-4  # the displaced cell has KB forces to compare
    np.testing.assert_allclose(np.asarray(tc_native.v_kb), np.asarray(tc_ref.v_kb), rtol=0, atol=1e-14)
    assert abs(float(e_n) - float(e_r)) <= 1e-12 * max(1.0, abs(float(e_r)))
    np.testing.assert_allclose(f_n, f_r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s_n, s_r, rtol=0, atol=1e-12)
