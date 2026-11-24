"""SIESTA restart files (``.XV``, ``.DM``, ``.nc``): formats, sparsity/spin conventions and warm starts.

The format tests need no SIESTA. Two tests use files from a SIESTA source tree (``siesta_pseudos_dir``) and one runs
SIESTA itself (``siesta_bin``); they are skipped when those references are not configured.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io import (
    SiestaSparsity,
    convert_spin,
    dm_from_siesta,
    dm_to_siesta,
    h_from_siesta,
    h_to_siesta,
    map_siesta_to_edges,
    parse_fdf,
    pattern_to_siesta,
    read_dm,
    read_dm_file,
    read_nc_restart,
    read_siesta_nc,
    read_xv,
    siesta_supercell_offsets,
    structure_from_nc,
    structure_from_xv,
    write_dm,
    write_dm_file,
    write_restart_files,
    write_siesta_nc,
    write_xv,
)
from jax_siesta.io.fdf import load_case_json
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.scf.eigensolver import GeneralizedEigenSolver
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.testing.external_refs import ref_path

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parent
PSML_DIR = ROOT / "fixtures" / "psml"

try:
    import netCDF4  # noqa: F401

    HAVE_NETCDF = True
except ImportError:
    HAVE_NETCDF = False
needs_netcdf = pytest.mark.skipif(not HAVE_NETCDF, reason="netCDF4 is not installed")


def _setup(case: str, k_grid: tuple[int, int, int] = (1, 1, 1), **overrides):
    c = load_case_json(ROOT / "manifests" / "cases" / case / "case.json")
    structure = c["structure"]
    config = replace(c["config"], mesh_cutoff_ry=40.0, enable_adaptive_controller=False, **overrides)
    loader = PSMLBasisLoader()
    species = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in structure.species_labels}
    basis = loader.build_basis(structure, species, include_soc_projectors=(config.nspin == 8))
    pattern = build_sparse_pattern(structure, basis)
    two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    kmesh = build_monkhorst_pack_mesh(k_grid)
    mesh = choose_mesh_dimensions(structure.cell, config.mesh_cutoff_ry)
    return structure, basis, pattern, two_center, kmesh, config, mesh


@pytest.fixture(scope="module")
def si():
    return _setup("si_bulk", k_grid=(2, 2, 2))


def _random_matrix(pattern, nspin: int, kind: str, seed: int) -> np.ndarray:
    """A random Hermitian matrix in the jsiesta channel layout the SCF driver uses for ``nspin``."""
    rng = np.random.default_rng(seed)
    n = pattern.n_edges
    tr = np.asarray(pattern.transpose_edge_idx)

    def sym(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x + np.conj(x[tr]))

    if nspin <= 2:
        return np.stack([sym(rng.normal(size=n)) for _ in range(nspin)])
    complex_diag = nspin == 8 or kind == "dm"
    diag = [sym(rng.normal(size=n) + (1j * rng.normal(size=n) if complex_diag else 0.0)) for _ in range(2)]
    off = [rng.normal(size=n), rng.normal(size=n)]
    if nspin == 4 and kind == "h":
        # SIESTA's non-collinear H has one off-diagonal spin block, H^du = (H^ud)^* element by element: it can
        # only hold an orbital-symmetric H^ud (what a local exchange-correlation field gives).
        off = [np.real(sym(x)) for x in off]
    out = np.stack(diag + off)
    return out if (nspin == 8 or kind == "dm") else np.real(out)


# ---------------------------------------------------------------------------------------------------------------
# Sparsity layout
# ---------------------------------------------------------------------------------------------------------------


def test_supercell_offsets_follow_siesta_order() -> None:
    """First axis fastest; offsets above nsc/2 wrap to negative (linear2pm), also for an even nsc."""
    off = siesta_supercell_offsets((3, 1, 5))
    assert off.shape == (15, 3)
    assert off[:4].tolist() == [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1]]
    assert off[:, 2].reshape(5, 3)[:, 0].tolist() == [0, 1, 2, -2, -1]
    assert siesta_supercell_offsets((4, 1, 1))[:, 0].tolist() == [0, 1, 2, -1]


def test_pattern_to_siesta_is_a_sorted_relabelling(si) -> None:
    _, _, pattern, _, _, _, _ = si
    sparsity, order = pattern_to_siesta(pattern)
    assert sorted(order.tolist()) == list(range(pattern.n_edges))
    assert all(n % 2 == 1 for n in sparsity.nsc) and max(sparsity.nsc) > 1
    rows = sparsity.rows
    assert np.array_equal(rows, np.asarray(pattern.src_orb)[order])
    same_row = rows[1:] == rows[:-1]
    assert np.all(np.diff(sparsity.list_col)[same_row] > 0)  # columns increase within a row: no duplicates
    # Column -> (orbital, offset) recovers the edge.
    shifts = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)][order]
    assert np.array_equal(sparsity.list_col % pattern.n_orb, np.asarray(pattern.dst_unit_orb)[order])
    assert np.array_equal(sparsity.isc_off[sparsity.list_col // pattern.n_orb], shifts)
    # The mapping back is the inverse permutation and uses every element.
    vals = np.arange(pattern.n_edges, dtype=np.float64)[None, :]
    back, info = map_siesta_to_edges(sparsity, pattern, vals[:, order])
    assert np.array_equal(back, vals)
    assert (info.n_edges_matched, info.n_file_unused, info.folded_axes) == (pattern.n_edges, 0, (False, False, False))


def test_mapping_between_different_supercells_and_patterns(si) -> None:
    """A file with a larger nsc and extra / missing elements maps like SIESTA's restructuring: by (io, juo, offset)."""
    _, _, pattern, _, _, _, _ = si
    sparsity, order = pattern_to_siesta(pattern)
    vals = np.random.default_rng(0).normal(size=(1, pattern.n_edges))
    # Re-index the same elements into a larger supercell.
    nsc_big = tuple(n + 2 for n in sparsity.nsc)
    off = sparsity.isc_off[sparsity.list_col // pattern.n_orb]
    idx = np.mod(off, np.asarray(nsc_big)[None, :])
    col_big = (idx[:, 0] + nsc_big[0] * (idx[:, 1] + nsc_big[1] * idx[:, 2])) * pattern.n_orb + sparsity.list_col % pattern.n_orb
    big = SiestaSparsity(pattern.n_orb, nsc_big, sparsity.n_col, col_big)
    back, _ = map_siesta_to_edges(big, pattern, vals[:, order])
    assert np.array_equal(back, vals)
    # Drop every third file element: those edges come back zero, the rest are untouched.
    keep = np.ones(sparsity.nnzs, dtype=bool)
    keep[::3] = False
    part = SiestaSparsity(
        pattern.n_orb, sparsity.nsc, np.bincount(sparsity.rows[keep], minlength=pattern.n_orb), sparsity.list_col[keep]
    )
    back, info = map_siesta_to_edges(part, pattern, vals[:, order][:, keep])
    expect = vals.copy()
    expect[:, order[~keep]] = 0.0
    assert np.array_equal(back, expect)
    assert info.n_edges_matched == int(keep.sum())
    with pytest.raises(ValueError, match="different basis sets"):
        map_siesta_to_edges(SiestaSparsity(3, (1, 1, 1), np.zeros(3, dtype=np.int32), np.zeros(0, dtype=np.int64)), pattern, np.zeros((1, 0)))


def test_folded_gamma_file_is_unfolded(si) -> None:
    """A file without auxiliary supercell: 'replicate' copies the folded value to all images, 'first' keeps the sum."""
    _, _, pattern, _, _, _, _ = si
    n_orb = pattern.n_orb
    src = np.asarray(pattern.src_orb)
    dst = np.asarray(pattern.dst_unit_orb)
    pairs = np.unique(src * n_orb + dst)
    folded = SiestaSparsity(n_orb, (1, 1, 1), np.bincount(pairs // n_orb, minlength=n_orb), pairs % n_orb)
    vals = np.random.default_rng(1).normal(size=(1, pairs.shape[0]))
    pair_of_edge = np.searchsorted(pairs, src * n_orb + dst)

    rep, info = map_siesta_to_edges(folded, pattern, vals, folded="replicate")
    assert info.folded_axes == (True, True, True)
    assert np.array_equal(rep[0], vals[0][pair_of_edge])

    first, _ = map_siesta_to_edges(folded, pattern, vals, folded="first")
    summed = np.zeros(pairs.shape[0])
    np.add.at(summed, pair_of_edge, first[0])
    assert np.allclose(summed, vals[0], atol=0.0)
    assert np.count_nonzero(first[0]) == pairs.shape[0]


# ---------------------------------------------------------------------------------------------------------------
# Spin components
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nspin", [1, 2, 4, 8])
def test_spin_component_round_trip(si, nspin: int) -> None:
    _, _, pattern, _, _, _, _ = si
    dm = _random_matrix(pattern, nspin, "dm", 2)
    h = _random_matrix(pattern, nspin, "h", 3)
    dm_s = dm_to_siesta(dm, pattern, nspin)
    h_s = h_to_siesta(h, pattern, nspin)
    assert dm_s.shape == h_s.shape == (nspin, pattern.n_edges) and dm_s.dtype == np.float64
    dm_back, n_dm = dm_from_siesta(dm_s)
    h_back, n_h = h_from_siesta(h_s, pattern)
    assert n_dm == n_h == nspin
    # SIESTA's non-collinear mode has no Im(D^uu), Im(D^dd); everything else returns exactly.
    dm_ref = np.stack([np.real(dm[0]), np.real(dm[1]), dm[2], dm[3]]) if nspin == 4 else dm
    assert np.allclose(dm_back, dm_ref, atol=1e-15)
    assert np.allclose(h_back, h, atol=1e-15)
    assert np.array_equal(convert_spin(dm_back, nspin, nspin, "dm"), dm_back)


def test_convert_spin_between_modes() -> None:
    d1 = np.asarray([[2.0, 4.0]])
    assert np.array_equal(convert_spin(d1, 1, 2, "dm"), [[1.0, 2.0], [1.0, 2.0]])
    assert np.array_equal(convert_spin(d1, 1, 2, "h"), [[2.0, 4.0], [2.0, 4.0]])
    d2 = np.asarray([[1.0, 2.0], [3.0, 5.0]])
    assert np.array_equal(convert_spin(d2, 2, 1, "dm"), [[4.0, 7.0]])
    assert np.array_equal(convert_spin(d2, 2, 1, "h"), [[2.0, 3.5]])
    d4 = convert_spin(d2, 2, 4, "dm")
    assert d4.dtype == np.complex128 and np.array_equal(d4, [[1, 2], [3, 5], [0, 0], [0, 0]])
    assert convert_spin(d2, 2, 4, "h").dtype == np.float64
    d8 = np.asarray([[1 + 1j, 2], [3, 5 - 2j], [7, 8], [9, 10]])
    assert np.array_equal(convert_spin(d8, 8, 2, "dm"), d2)
    assert np.array_equal(convert_spin(d8, 8, 4, "h"), np.real(d8))


def _siesta_solve_like_diag3k(h_s, s_edge, pattern, kpts, wts, mu, temp, nspin):
    """Literal NumPy transcription of SIESTA's ``diag3k`` (lattice gauge): H(k) and S(k) from the SIESTA spin
    components, eigenvectors, then ``Dnew`` in SIESTA components (Ry, edge order)."""
    n = pattern.n_orb
    src = np.asarray(pattern.src_orb)
    dst = np.asarray(pattern.dst_unit_orb)
    t = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)].astype(float)
    hc = np.zeros((8, h_s.shape[1]))
    hc[: h_s.shape[0]] = h_s
    if nspin == 4:  # non-collinear: spin-box Hermitian, H^du = (H^ud)^*
        hc[6], hc[7] = hc[2], hc[3]
    d_new = np.zeros((8, h_s.shape[1]))
    for k, w in zip(kpts, wts):
        kphs = np.exp(-2j * np.pi * (t @ k))
        haux = np.zeros((2, n, 2, n), dtype=complex)  # (spin_j, juo, spin_i, iuo)
        saux = np.zeros((2, n, 2, n), dtype=complex)
        np.add.at(saux, (0, dst, 0, src), s_edge * kphs)
        np.add.at(saux, (1, dst, 1, src), s_edge * kphs)
        np.add.at(haux, (0, dst, 0, src), (hc[0] + 1j * hc[4]) * kphs)
        np.add.at(haux, (1, dst, 1, src), (hc[1] + 1j * hc[5]) * kphs)
        np.add.at(haux, (0, dst, 1, src), (hc[2] - 1j * hc[3]) * kphs)
        np.add.at(haux, (1, dst, 0, src), (hc[6] + 1j * hc[7]) * kphs)
        # Fortran (2, n, 2, n) -> matrix with combined index (spin, orbital), spin fastest.
        hm = haux.transpose(1, 0, 3, 2).reshape(2 * n, 2 * n)
        sm = saux.transpose(1, 0, 3, 2).reshape(2 * n, 2 * n)
        assert np.allclose(hm, hm.conj().T, atol=1e-12)
        eps, psi = scipy.linalg.eigh(hm, sm)
        occ = 1.0 / (1.0 + np.exp(np.clip((eps - mu) / temp, -700, 700)))
        c = psi.reshape(n, 2, 2 * n)  # c[orbital, spin, state]
        # Dk(s, juo, s', iuo) = sum_e q_e c(s, iuo) conj(c(s', juo))
        dk = np.einsum("e,ise,jte->sjti", w * occ, c, np.conj(c))
        d11 = dk[0, dst, 0, src] * kphs
        d22 = dk[1, dst, 1, src] * kphs
        d12 = dk[0, dst, 1, src] * kphs
        d21 = dk[1, dst, 0, src] * kphs
        d_new += np.stack([d11.real, d22.real, d12.real, -d12.imag, d11.imag, d22.imag, d21.real, d21.imag])
    return d_new


@pytest.mark.parametrize("nspin", [4, 8])
def test_spin_conventions_reproduce_siesta_diag3k(si, nspin: int) -> None:
    """H in SIESTA components, diagonalised the way SIESTA's ``diag3k`` does, gives the SIESTA components of the
    density matrix jsiesta's own eigensolver produces from the jsiesta H: checks transposes, conjugations and signs."""
    _, basis, pattern, tc, _, config, _ = si
    kmesh = build_monkhorst_pack_mesh((2, 1, 1), shifts=(0.25, 0.1, 0.3))  # complex k-points
    h0 = np.asarray(tc.kinetic + tc.v_kb)
    pert = 0.05 * _random_matrix(pattern, nspin, "h", 4)
    h = pert.astype(np.complex128 if nspin == 8 else np.float64)
    h[0] += h0
    h[1] += h0
    temp = 0.01
    n_el = float(basis.total_valence_electrons)
    dm, _, mu, _, _ = GeneralizedEigenSolver().solve(jnp.asarray(h), tc.overlap, pattern, kmesh, n_el, temp, nspin)

    h_s = h_to_siesta(h, pattern, nspin)
    d_siesta = _siesta_solve_like_diag3k(
        h_s, np.asarray(tc.overlap), pattern, np.asarray(kmesh.kpoints_frac), np.asarray(kmesh.weights),
        2.0 * float(mu), 2.0 * temp, nspin,
    )
    ours = dm_to_siesta(np.asarray(dm), pattern, 8)
    assert np.max(np.abs(ours)) > 0.1
    assert np.max(np.abs(ours[2:4])) > 1e-3 and (nspin == 4 or np.max(np.abs(ours[4:6])) > 1e-4)
    assert np.allclose(ours, d_siesta, atol=1e-10), np.max(np.abs(ours - d_siesta), axis=1)
    # And back: SIESTA's components give the jsiesta density matrix.
    back, _ = dm_from_siesta(d_siesta)
    assert np.allclose(back, np.asarray(dm), atol=1e-10)


# ---------------------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------------------


def test_xv_round_trip_and_template(tmp_path: Path, si) -> None:
    structure = si[0]
    moved = replace(structure, positions=structure.positions + 0.0123, cell=structure.cell * 1.01)
    write_xv(tmp_path / "a.XV", moved)
    xv = read_xv(tmp_path / "a.XV")
    assert np.allclose(xv.cell, np.asarray(moved.cell), atol=5e-10)
    assert np.allclose(xv.positions, np.asarray(moved.positions), atol=5e-10)
    assert xv.species_index.tolist() == [i + 1 for i in structure.species_indices]
    assert xv.atomic_numbers.tolist() == list(structure.atomic_numbers) and not xv.velocities.any()

    from_tpl = structure_from_xv(tmp_path / "a.XV", template=structure)
    assert from_tpl.species_labels == structure.species_labels and from_tpl.pbc == structure.pbc
    assert np.allclose(from_tpl.positions, moved.positions, atol=5e-10)
    alone = structure_from_xv(tmp_path / "a.XV")
    assert alone.species_labels == ("Si",) and alone.species_indices == structure.species_indices

    wrong = replace(structure, atomic_numbers=(6,) * structure.n_atoms)
    with pytest.raises(ValueError, match="atomic numbers"):
        structure_from_xv(tmp_path / "a.XV", template=wrong)
    (tmp_path / "bad.XV").write_text("1 0 0 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an XV file"):
        read_xv(tmp_path / "bad.XV")


@pytest.mark.parametrize("nspin", [1, 2, 4, 8])
def test_dm_file_round_trip(tmp_path: Path, si, nspin: int) -> None:
    _, _, pattern, _, _, _, _ = si
    dm = _random_matrix(pattern, nspin, "dm", 5)
    if nspin == 4:
        dm = np.stack([np.real(dm[0]), np.real(dm[1]), dm[2], dm[3]])
    path = tmp_path / "x.DM"
    write_dm(path, pattern, dm, nspin)
    sparsity, values = read_dm_file(path)
    assert (sparsity.no_u, values.shape) == (pattern.n_orb, (nspin, pattern.n_edges))
    # Record structure: header, n_col, one record per row for the columns and per row and spin for the values.
    n_rec = 2 + pattern.n_orb * (1 + nspin)
    assert path.stat().st_size == 8 * n_rec + 4 * 5 + 4 * pattern.n_orb + 4 * pattern.n_edges + 8 * nspin * pattern.n_edges
    back, info = read_dm(path, pattern, nspin, return_info=True)
    assert info.n_edges_matched == pattern.n_edges
    assert np.allclose(np.asarray(back), dm, atol=1e-15)
    # Another spin mode on reading.
    other = np.asarray(read_dm(path, pattern, 1))
    total = np.real(dm[0]) if nspin == 1 else np.real(dm[0] + dm[1])
    assert np.allclose(other[0], total, atol=1e-14)


def test_dm_file_errors(tmp_path: Path, si) -> None:
    _, _, pattern, _, _, _, _ = si
    write_dm(tmp_path / "x.DM", pattern, np.ones((1, pattern.n_edges)), 1)
    raw = (tmp_path / "x.DM").read_bytes()
    (tmp_path / "cut.DM").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError):
        read_dm_file(tmp_path / "cut.DM")
    h2o = _setup("h2o_gamma")[2]
    with pytest.raises(ValueError, match="different basis sets"):
        read_dm(tmp_path / "x.DM", h2o, 1)


@needs_netcdf
@pytest.mark.parametrize("nspin", [1, 2, 8])
def test_nc_round_trip_and_layout(tmp_path: Path, si, nspin: int) -> None:
    import netCDF4

    structure, basis, pattern, tc, _, config, _ = si
    dm = _random_matrix(pattern, nspin, "dm", 6)
    edm = _random_matrix(pattern, nspin, "dm", 7)
    h = _random_matrix(pattern, nspin, "h", 8)
    forces = np.arange(3.0 * structure.n_atoms).reshape(-1, 3)
    path = tmp_path / "x.nc"
    write_siesta_nc(
        path, structure, basis, pattern, nspin, tc.overlap, h=h, dm=dm, edm=edm, fermi_level_ha=-0.15,
        forces_ha_per_bohr=forces, stress_ha_per_bohr3=np.eye(3), config=replace(config, nspin=nspin), k_grid=(2, 2, 2),
    )
    with netCDF4.Dataset(path) as ds:  # SIESTA's dimension and variable names, index order and units
        sp = ds.groups["SPARSE"]
        nnz = pattern.n_edges
        assert ds.data_model == "NETCDF4"
        assert {k: len(v) for k, v in ds.dimensions.items()} == {
            "na_u": structure.n_atoms, "no_u": pattern.n_orb, "no_s": pattern.n_orb * len(ds.dimensions["n_s"]),
            "spin": nspin, "xyz": 3, "n_s": int(np.prod(ds.variables["nsc"][:])), "one": 1,
        }
        assert sp.variables["DM"].shape == sp.variables["H"].shape == (nspin, nnz)
        assert sp.variables["EDM"].shape == (min(nspin, 4), nnz) and sp.variables["S"].shape == (nnz,)
        assert sp.variables["isc_off"].shape == (len(ds.dimensions["n_s"]), 3)
        assert int(np.min(sp.variables["list_col"][:])) >= 1
        assert int(np.sum(sp.variables["n_col"][:])) == nnz
        assert ds.variables["lasto"][:].tolist() == list(basis.atom_orb_offsets[1:])
        assert ds.variables["Ef"].unit == "Ry" and np.isclose(ds.variables["Ef"][0], -0.30)
        assert np.allclose(ds.variables["fa"][:], 2.0 * forces)
        assert np.allclose(ds.variables["cell"][:], np.asarray(structure.cell).T)  # rows are lattice vectors
        assert np.isclose(ds.variables["Qtot"][0], basis.total_valence_electrons)
        assert ds.groups["BASIS"].variables["basis"][:].tolist() == [i + 1 for i in structure.species_indices]
        assert int(ds.groups["BASIS"].groups["Si"].Number_of_orbitals) == pattern.n_orb // structure.n_atoms
    data = read_siesta_nc(path)
    assert np.allclose(data.cell, np.asarray(structure.cell)) and np.allclose(data.positions, np.asarray(structure.positions))
    assert np.allclose(map_siesta_to_edges(data.sparsity, pattern, data.overlap)[0][0], np.asarray(tc.overlap))
    dm_back, h_back, ef = read_nc_restart(path, pattern, nspin)
    assert np.allclose(np.asarray(dm_back), dm, atol=1e-15) and np.allclose(np.asarray(h_back), h, atol=1e-15)
    assert np.isclose(ef, -0.15)
    moved = structure_from_nc(path, replace(structure, positions=structure.positions + 1.0))
    assert np.allclose(moved.positions, structure.positions)


def test_fdf_restart_keywords(tmp_path: Path) -> None:
    fdf = tmp_path / "in.fdf"
    fdf.write_text(
        "SystemLabel water\nDM.UseSaveDM T\nMD.UseSaveXV\nCDF.Save true\nSaveHS .false.\n"
        "%block ChemicalSpeciesLabel\n1 8 O\n%endblock ChemicalSpeciesLabel\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n",
        encoding="utf-8",
    )
    io = parse_fdf(fdf)["io"]
    assert io == {
        "system_label": "water", "use_save_xv": True, "use_save_dm": True, "dm_init_file": "water.DM",
        "cdf_save": True, "save_hs": False, "must_converge": True,
    }


# ---------------------------------------------------------------------------------------------------------------
# Warm starts
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def converged_si(si, tmp_path_factory):
    structure, basis, pattern, tc, kmesh, config, mesh = si
    config = replace(config, max_scf_iterations=60, d_tol=1e-7, h_tol_ha=1e-7, e_tol_ha=1e-9)
    state, mesh = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh)
    assert state.converged
    out = tmp_path_factory.mktemp("restart")
    formats = ("XV", "DM", "nc") if HAVE_NETCDF else ("XV", "DM")
    files = write_restart_files(out, "si", structure, basis, pattern, tc.overlap, state, config, k_grid=(2, 2, 2), formats=formats)
    return config, state, files


def _energy(setup, config, state) -> float:
    structure, basis, pattern, tc, kmesh, _, mesh = setup
    obs = StaticObservableEvaluator().evaluate(
        structure, basis, pattern, tc, kmesh, state, config, mesh_shape=mesh, compute_forces_stress=False
    )
    return float(obs.final_post_scf_free_e_ha)


def test_warm_start_from_dm_file(si, converged_si, record_residuals) -> None:
    structure, basis, pattern, tc, kmesh, _, mesh = si
    config, cold, files = converged_si
    dm0 = read_dm(files["DM"], pattern, config.nspin)
    assert np.allclose(np.asarray(dm0), np.asarray(cold.dm_out), atol=1e-14)
    warm, _ = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh, initial_dm=dm0)
    de = abs(_energy(si, config, warm) - _energy(si, config, cold))
    record_residuals({"iterations_cold": cold.iteration, "iterations_warm": warm.iteration, "dE_ha": de}, {"dE_ha": 1e-7})
    assert warm.converged and warm.iteration <= 3 < cold.iteration
    assert de < 1e-7
    assert float(jnp.max(jnp.abs(warm.dm_out - cold.dm_out))) < 1e-5


@needs_netcdf
@pytest.mark.parametrize("use", ["h", "h+dm"])
@pytest.mark.parametrize("mix_target", ["Hamiltonian", "DM"])
def test_warm_start_from_nc_file(si, converged_si, use: str, mix_target: str) -> None:
    structure, basis, pattern, tc, kmesh, _, mesh = si
    config, cold, files = converged_si
    config = replace(config, mix_target=mix_target)
    dm0, h0, ef = read_nc_restart(files["nc"], pattern, config.nspin)
    assert np.allclose(np.asarray(h0), np.asarray(cold.h_sparse), atol=1e-14)
    assert np.isclose(ef, float(cold.fermi_level_ha))
    warm, _ = run_scf(
        structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh,
        initial_h=h0, initial_dm=dm0 if use == "h+dm" else None,
    )
    assert warm.converged and warm.iteration <= 3 < cold.iteration
    assert abs(_energy(si, config, warm) - _energy(si, config, cold)) < 1e-7


def test_warm_start_rescales_and_validates(si, converged_si) -> None:
    structure, basis, pattern, tc, kmesh, _, mesh = si
    config, cold, _ = converged_si
    warm, _ = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh, initial_dm=0.9 * cold.dm_out)
    assert warm.converged and warm.iteration <= 3  # the charge is restored before the first Hamiltonian
    with pytest.raises(ValueError, match="Warm-start dm"):
        run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh, initial_dm=cold.dm_out[:, :-1])


@needs_netcdf
def test_run_single_point_reads_and_writes_restart_files(tmp_path: Path) -> None:
    """The runner: cold run writes XV/DM/nc; a second run from a displaced case file reads them back."""
    from jax_siesta.testing.run_case import run_single_point

    case = ROOT / "manifests" / "cases" / "h2o_gamma" / "case.json"
    over = {"mesh_cutoff_ry": 40.0, "d_tol": 1e-7, "h_tol_ha": 1e-7, "e_tol_ha": 1e-9}
    cold = run_single_point(case, config_overrides=over, write_restart_dir=tmp_path, system_label="h2o")
    for ext in ("XV", "DM", "nc"):
        assert (tmp_path / f"h2o.{ext}").stat().st_size > 0
    # Same geometry from the XV file although the case file is displaced; matrices from DM and nc.
    import json

    data = json.loads(case.read_text(encoding="utf-8"))
    data["positions_bohr"] = (np.asarray(data["positions_bohr"]) + [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]]).tolist()
    (tmp_path / "case.json").write_text(json.dumps(data), encoding="utf-8")
    warm = run_single_point(
        tmp_path / "case.json", config_overrides=over, read_xv_path=tmp_path / "h2o.XV",
        read_dm_path=tmp_path / "h2o.DM", read_nc_path=tmp_path / "h2o.nc",
    )
    assert warm.converged and warm.n_iterations <= 3 < cold.n_iterations
    assert abs(float(warm.final_post_scf_free_e_ha) - float(cold.final_post_scf_free_e_ha)) < 1e-6  # XV has 9 decimals
    displaced = run_single_point(tmp_path / "case.json", config_overrides=over)
    assert abs(float(displaced.final_post_scf_free_e_ha) - float(cold.final_post_scf_free_e_ha)) > 1e-5


# ---------------------------------------------------------------------------------------------------------------
# Against SIESTA
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.external_refs("siesta_pseudos_dir")
def test_siesta_written_dm_and_xv_are_reproduced_byte_for_byte(tmp_path: Path) -> None:
    """Files written by SIESTA (its own test suite) are read and written back identically."""
    tests_dir = ref_path("siesta_pseudos_dir").parent
    dm_files = [tests_dir / "14.FileIO" / "save_dm.DM", tests_dir / "15.TDDFT" / "TDDFT_h2o.DM"]
    xv_file = tests_dir / "15.TDDFT" / "TDDFT_h2o.XV"
    if not all(f.is_file() for f in [*dm_files, xv_file]):
        pytest.skip("this SIESTA tree has no Tests/14.FileIO/save_dm.DM or Tests/15.TDDFT/TDDFT_h2o.{DM,XV}")
    for f in dm_files:
        sparsity, values = read_dm_file(f)
        assert sparsity.nsc == (1, 1, 1) and values.shape == (1, sparsity.nnzs)
        dense = np.zeros((sparsity.no_u, sparsity.no_u))
        dense[sparsity.rows, sparsity.list_col] = values[0]  # SIESTA's columns are not sorted within a row
        assert np.allclose(dense, dense.T, atol=1e-8) and np.trace(dense) > 1.0  # a symmetric density matrix
        write_dm_file(tmp_path / f.name, sparsity, values)
        assert (tmp_path / f.name).read_bytes() == f.read_bytes()
    structure = structure_from_xv(xv_file)
    assert structure.species_labels == ("O", "H") and structure.atomic_numbers == (8, 1, 1)
    write_xv(tmp_path / "out.XV", structure)
    assert (tmp_path / "out.XV").read_text() == xv_file.read_text()


@pytest.mark.external_refs("siesta_bin", "pseudodojo_sr_pbe_dir")
def test_siesta_reads_jsiesta_files_and_jsiesta_reads_siesta_files(tmp_path: Path, record_residuals) -> None:
    """Both directions with a real SIESTA run on bulk Si (same PSML and ``.ion.xml`` tables in both codes).

    1. SIESTA runs cold and leaves ``.DM``, ``.XV`` (and ``.nc`` when built with NetCDF). jsiesta started from them
       converges in a few iterations to the energy of its own cold run.
    2. jsiesta writes ``.DM`` and ``.XV``; SIESTA, restarted with ``DM.UseSaveDM`` and ``MD.UseSaveXV``, reads them
       and needs fewer SCF iterations than its cold run.
    """
    import subprocess

    from jax_siesta.testing.siesta_adapter import resolve_siesta_executable, write_siesta_fdf

    case_json = ROOT / "manifests" / "cases" / "si_bulk" / "case.json"
    psml_dir = ref_path("pseudodojo_sr_pbe_dir")
    label = "jsiesta_ref"

    def _n_scf(text: str) -> int:
        return len(re.findall(r"^\s*scf:\s+\d+", text, flags=re.MULTILINE))

    case = load_case_json(case_json)

    def _run_siesta(workdir: Path, extra: str) -> str:
        workdir.mkdir(exist_ok=True)
        for lbl in case["structure"].species_labels:
            shutil.copy(psml_dir / f"{lbl}.psml", workdir / f"{lbl}.psml")
        fdf = write_siesta_fdf(workdir, case, system_label=label)
        # FDF keeps the first definition of a key: the extra lines go in front.
        fdf.write_text(extra + "\n" + fdf.read_text(encoding="utf-8"), encoding="utf-8")
        with fdf.open("rb") as fin, (workdir / "restart.out").open("wb") as fout:
            subprocess.run([str(resolve_siesta_executable("siesta"))], cwd=workdir, stdin=fin, stdout=fout,
                           stderr=subprocess.STDOUT, timeout=600, check=True)
        return (workdir / "restart.out").read_text(encoding="utf-8", errors="replace")

    # SIESTA cold run (also provides the .ion.xml tables) with DM, XV and, if available, nc output.
    cold_dir = tmp_path / "siesta_cold"
    cold_text = _run_siesta(cold_dir, "DM.UseSaveDM F\nCDF.Save T\nSaveHS T")
    n_cold = _n_scf(cold_text)
    assert (cold_dir / f"{label}.DM").is_file() and (cold_dir / f"{label}.XV").is_file()

    # jsiesta with the same tables: cold, then started from SIESTA's files.
    loader = PSMLBasisLoader()
    species = {
        lbl: loader.load_species(lbl, str(psml_dir / f"{lbl}.psml"), ion_path=str(cold_dir / f"{lbl}.ion.xml"))
        for lbl in case["structure"].species_labels
    }
    structure = structure_from_xv(cold_dir / f"{label}.XV", template=case["structure"])
    assert np.allclose(structure.positions, case["structure"].positions, atol=1e-8)
    config = replace(case["config"], d_tol=1e-6, h_tol_ha=1e-6)
    basis = loader.build_basis(structure, species)
    pattern = build_sparse_pattern(structure, basis)
    tc = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    kmesh = build_monkhorst_pack_mesh(case["k_grid"])
    js_cold, mesh = run_scf(structure, basis, pattern, tc, kmesh, config)
    dm_siesta, info = read_dm(cold_dir / f"{label}.DM", pattern, config.nspin, return_info=True)
    d_dm = float(jnp.max(jnp.abs(dm_siesta - js_cold.dm_out)))
    js_warm, _ = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh, initial_dm=dm_siesta)
    resid = {"max_dDM_siesta_vs_jsiesta": d_dm, "jsiesta_iterations_cold": js_cold.iteration,
             "jsiesta_iterations_from_siesta_dm": js_warm.iteration}
    assert info.n_edges_matched > 0.9 * info.n_file
    assert d_dm < 5e-3, d_dm  # same orbitals, order and sign conventions
    assert js_warm.converged and js_warm.iteration < js_cold.iteration
    if HAVE_NETCDF and (cold_dir / f"{label}.nc").is_file():
        dm_nc, h_nc, _ = read_nc_restart(cold_dir / f"{label}.nc", pattern, config.nspin)
        assert dm_nc is not None and float(jnp.max(jnp.abs(dm_nc - dm_siesta))) < 1e-10
        if h_nc is not None:
            resid["max_dH_ha_siesta_vs_jsiesta"] = float(jnp.max(jnp.abs(h_nc - js_cold.h_sparse)))
            js_h, _ = run_scf(structure, basis, pattern, tc, kmesh, config, mesh_shape=mesh, initial_h=h_nc)
            resid["jsiesta_iterations_from_siesta_h"] = js_h.iteration
            assert js_h.converged and js_h.iteration < js_cold.iteration

    # SIESTA restarted from jsiesta's files.
    warm_dir = tmp_path / "siesta_warm"
    write_restart_files(warm_dir, label, structure, basis, pattern, tc.overlap, js_cold, config, formats=("XV", "DM"))
    warm_text = _run_siesta(warm_dir, "DM.UseSaveDM T\nMD.UseSaveXV T")
    n_warm = _n_scf(warm_text)
    resid.update({"siesta_iterations_cold": n_cold, "siesta_iterations_from_jsiesta_dm": n_warm})
    record_residuals(resid)
    assert "Reading coordinates and velocities from file" in warm_text
    assert re.search(r"Attempting to read DM from file\.\.\.\s*Succeeded", warm_text)
    assert 0 < n_warm < n_cold
