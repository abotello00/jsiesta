"""Phase 2 Step 1 tests: crystal/magnetic space groups (Shubnikov Type I-IV), IBZ, and grid permutations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_siesta.core import SCFConfig
from jax_siesta.io.fdf import load_case_json
from jax_siesta.scf.driver import build_monkhorst_pack_mesh
from jax_siesta.symmetry.spacegroup import (
    build_ibz_kpoint_mesh,
    choose_commensurate_mesh_dimensions,
    extract_space_group,
    precompute_grid_permutations,
)

CASES_DIR = Path(__file__).resolve().parent / "manifests" / "cases"


@pytest.mark.parametrize(
    ("cname", "expected_min_ops", "expected_msg_types"),
    [
        ("si_bulk", 48, (1,)),
        ("h2o_gamma", 2, (1,)),
        ("graphene_slab", 4, (1,)),
        ("co_hcp_collinear", 4, (1, 3)),
        ("co_afm", 16, (4,)),
        ("co_hcp_nc", 2, (3,)),
        ("co_hcp_soc", 4, (2, 3)),
        ("si_triclinic", 1, (1,)),
    ],
)
def test_spacegroup_atom_maps_and_magnetic_covariance(
    cname: str, expected_min_ops: int, expected_msg_types: tuple[int, ...]
) -> None:
    case = load_case_json(CASES_DIR / cname / "case.json")
    structure = case["structure"]
    config: SCFConfig = case["config"]

    sg = extract_space_group(structure, config)
    assert sg.n_ops >= expected_min_ops
    assert sg.msg_type in expected_msg_types
    assert np.array_equal(sg.inv_op[sg.inv_op], np.arange(sg.n_ops))
    assert np.allclose(sg.rotations_frac[0], np.eye(3))
    assert not bool(sg.time_reversals[0])

    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    moms = np.asarray(structure.initial_moments, dtype=np.float64)

    for s in range(sg.n_ops):
        c_s = sg.rotations_cart[s]
        assert np.allclose(c_s.T @ c_s, np.eye(3), atol=1e-7)
        t_c = sg.translations_cart[s]
        for ia in range(structure.n_atoms):
            ja = int(sg.atom_map[s, ia])
            n_shift = sg.atom_translation[s, ia].astype(np.float64)
            lhs = c_s @ pos[ia] + t_c
            rhs = pos[ja] + cell @ n_shift
            assert np.allclose(lhs, rhs, atol=1e-4)

            if sg.is_magnetic:
                sign_t = -1.0 if bool(sg.time_reversals[s]) else 1.0
                if config.nspin == 2 and np.allclose(moms[:, :2], 0.0):
                    # Collinear z-moment flips only under time reversal
                    assert np.isclose(sign_t * moms[ia, 2], moms[ja, 2], atol=1e-4)
                else:
                    # 3D axial magnetic pseudovector transforms with det(C_S) * (-1)^theta * C_S
                    m_rot = sign_t * float(sg.det_cart[s]) * (c_s @ moms[ia])
                    assert np.allclose(m_rot, moms[ja], atol=1e-4)


def test_ibz_kmesh_and_commensurate_grid_permutations() -> None:
    # 1. Diamond Si (Fd-3m, 48 ops, nonsymmorphic (1/4, 1/4, 1/4))
    si = load_case_json(CASES_DIR / "si_bulk" / "case.json")
    sg_si = extract_space_group(si["structure"], si["config"])
    ibz_si = build_ibz_kpoint_mesh(sg_si, (4, 4, 4))
    assert ibz_si.is_ibz
    assert ibz_si.n_k == 8
    assert np.isclose(float(np.sum(ibz_si.weights)), 1.0, atol=1e-12)

    # Compare full-BZ vs IBZ average of point-group-invariant crystal star sum
    full_si = build_monkhorst_pack_mesh((4, 4, 4))
    r_shell = np.unique(
        np.einsum("sij,j->si", sg_si.rotations_frac, np.array([1, 0, 0], dtype=np.int32)),
        axis=0,
    ).astype(np.float64)
    val_full = np.sum(
        np.asarray(full_si.weights)[:, None] * np.cos(2.0 * np.pi * (np.asarray(full_si.kpoints_frac) @ r_shell.T))
    )
    val_ibz = np.sum(
        np.asarray(ibz_si.weights)[:, None] * np.cos(2.0 * np.pi * (np.asarray(ibz_si.kpoints_frac) @ r_shell.T))
    )
    assert np.isclose(val_full, val_ibz, atol=1e-12)

    mesh_si = choose_commensurate_mesh_dimensions(si["structure"], 80.0, sg_si)
    assert mesh_si[0] == mesh_si[1] == mesh_si[2]
    assert mesh_si[0] % 4 == 0
    perm_fwd, perm_inv = precompute_grid_permutations(sg_si, mesh_si)
    n_grid = int(np.prod(mesh_si))
    assert perm_fwd.shape == (sg_si.n_ops, n_grid)
    ar = np.arange(n_grid, dtype=np.int32)
    for s in range(sg_si.n_ops):
        assert np.array_equal(perm_inv[s, perm_fwd[s]], ar)
        assert np.array_equal(perm_fwd[s, perm_inv[s]], ar)

    # Verify scalar Gaussian radial field placed on Si atoms is exactly invariant under pi_S
    cell = np.asarray(si["structure"].cell, dtype=np.float64)
    pos = np.asarray(si["structure"].positions, dtype=np.float64)
    n1, n2, n3 = mesh_si
    g0, g1, g2 = np.meshgrid(np.arange(n1)/n1, np.arange(n2)/n2, np.arange(n3)/n3, indexing="ij")
    f_grid = np.stack([g0.ravel(), g1.ravel(), g2.ravel()], axis=1)
    rho = np.zeros(n_grid, dtype=np.float64)
    for ia in range(si["structure"].n_atoms):
        f_a = np.linalg.solve(cell, pos[ia])
        for s0 in (-1, 0, 1):
            for s1 in (-1, 0, 1):
                for s2 in (-1, 0, 1):
                    df = f_grid - (f_a[None, :] + np.array([[s0, s1, s2]]))
                    dr = df @ cell.T
                    rho += np.exp(-np.sum(dr**2, axis=1) / 1.5)
    rho_sym = np.mean(rho[perm_inv], axis=0)
    assert np.max(np.abs(rho_sym - rho)) < 1e-10

    # 2. Magnetic Co SOC IBZ + shifted k-mesh check
    co = load_case_json(CASES_DIR / "co_hcp_soc" / "case.json")
    sg_co = extract_space_group(co["structure"], co["config"])
    ibz_co = build_ibz_kpoint_mesh(sg_co, (4, 4, 3), shifts=(0.5, 0.5, 0.5))
    assert ibz_co.n_k < 48
    assert np.isclose(float(np.sum(ibz_co.weights)), 1.0, atol=1e-12)
    mesh_co = choose_commensurate_mesh_dimensions(co["structure"], 80.0, sg_co)
    pf_co, pi_co = precompute_grid_permutations(sg_co, mesh_co)
    assert pf_co.shape == (sg_co.n_ops, int(np.prod(mesh_co)))
    assert np.array_equal(pi_co[0], np.arange(int(np.prod(mesh_co))))

    # 3. Ideal hexagonal P6_3/mmc (24 ops with 1/2 screw axis along c)
    from jax_siesta.core import Structure
    import jax.numpy as jnp

    a_hex = 4.73
    c_hex = 7.69
    cell_hex = np.array([[a_hex, -0.5 * a_hex, 0.0], [0.0, 0.5 * np.sqrt(3.0) * a_hex, 0.0], [0.0, 0.0, c_hex]])
    frac_hex = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]])
    pos_hex = (cell_hex @ frac_hex.T).T
    st_hex = Structure(
        cell=jnp.asarray(cell_hex),
        positions=jnp.asarray(pos_hex),
        species_indices=(0, 0),
        atomic_numbers=(27, 27),
        species_labels=("Co",),
        initial_moments=jnp.zeros((2, 3)),
    )
    sg_hex = extract_space_group(st_hex, SCFConfig(nspin=1))
    assert sg_hex.n_ops == 24
    mesh_hex = choose_commensurate_mesh_dimensions(st_hex, 80.0, sg_hex)
    assert mesh_hex[0] == mesh_hex[1]
    assert mesh_hex[2] % 2 == 0
    pf_hex, pi_hex = precompute_grid_permutations(sg_hex, mesh_hex)
    assert pf_hex.shape == (24, int(np.prod(mesh_hex)))
    assert np.array_equal(pi_hex[0], np.arange(int(np.prod(mesh_hex))))

    # 4. Open-boundary molecule centered at [0.5, 0.5, 0.5] with pbc=(False, False, False) retains Pmm2 (4 ops)
    h2o = load_case_json(CASES_DIR / "h2o_gamma" / "case.json")
    h2o_open = Structure(
        cell=h2o["structure"].cell,
        positions=h2o["structure"].positions,
        species_indices=h2o["structure"].species_indices,
        atomic_numbers=h2o["structure"].atomic_numbers,
        species_labels=h2o["structure"].species_labels,
        initial_moments=h2o["structure"].initial_moments,
        pbc=(False, False, False),
    )
    sg_h2o_open = extract_space_group(h2o_open, SCFConfig(nspin=1))
    assert sg_h2o_open.n_ops == 4
