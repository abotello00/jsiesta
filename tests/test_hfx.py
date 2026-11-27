"""Exact-exchange matrix on the sparse pattern (Stages 3 and 5 of ``docs/design/HYBRID_FUNCTIONALS_PLAN.md``).

External reference: PySCF for an isolated molecule, with the same contracted Gaussians and the same density matrix.
Identities: supercell consistency of the lattice sums, symmetry, linearity, ``E = Tr[P H]/2``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_siesta.app.calculation import Calculation, make_structure
from jax_siesta.hfx import Nao2GtoExchange, fit_species
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.xc.functional import ExactExchangeSpec

PSEUDOS = Path(__file__).resolve().parents[1] / "examples" / "pseudos"
TIGHT = {"screen_tol": 1e-15, "pair_tol": 0.0, "prim_tol": 0.0}


def _setup(cell_ang, symbols, positions_ang, basis: str):
    structure = make_structure(np.asarray(cell_ang), symbols, np.asarray(positions_ang))
    calc = Calculation.from_structure(structure, pseudo_dir=PSEUDOS, basis=basis)
    orbitals = PSMLBasisLoader().build_basis(structure, calc.species)
    return structure, orbitals, build_sparse_pattern(structure, orbitals)


def _symmetric_dm(pattern, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    raw = rng.standard_normal((n, int(pattern.src_orb.shape[0])))
    return 0.5 * (raw + raw[:, np.asarray(pattern.transpose_edge_idx)])


def _dense(pattern, values: np.ndarray, n_orb: int) -> np.ndarray:
    out = np.zeros((n_orb, n_orb))
    np.add.at(out, (np.asarray(pattern.src_orb), np.asarray(pattern.dst_unit_orb)), values)
    return out


def test_gaussian_fit_of_numerical_orbitals() -> None:
    """Fit quality for the generated Si DZP basis: residual norm, overlap with the original, ordered exponents.

    No external reference (the fit is a definition); the bounds pin the measured quality (residuals of 1 to 2.5 %
    of the orbital norm with five Gaussians) so that a regression is visible.
    """
    structure, orbitals, _ = _setup(np.eye(3) * 6.0, ["Si"], [[0.0, 0.0, 0.0]], "DZP")
    species = orbitals.species[0]
    r = np.asarray(species.pao_r)
    for shell, values in zip(fit_species(species, 5), np.asarray(species.pao_values)):
        assert np.all(np.diff(shell.exponents) > 0) and shell.exponents[0] >= 0.05
        assert np.all(shell.exponents[1:] / shell.exponents[:-1] >= 1.5 - 1e-9)
        assert shell.residual < 0.03
        fit = r**shell.l * np.sum(shell.coefficients[None, :] * np.exp(-shell.exponents[None, :] * r[:, None] ** 2), axis=1)
        overlap = np.trapezoid(r**2 * fit * values, r)
        assert abs(overlap - 1.0) < 2e-3


@pytest.mark.parametrize("spec", [ExactExchangeSpec(0.0, 0.25, 0.11), ExactExchangeSpec(0.2, 0.3, 0.3)], ids=["hse06", "mixed"])
def test_exchange_matrix_of_water_against_pyscf(spec: ExactExchangeSpec) -> None:
    """``K[P]`` of a water molecule in a box equals PySCF's for the same Gaussians and density matrix.

    The box is large enough that no periodic image contributes, so the reference is the molecular exchange matrix
    ``sum_qs P_qs (pq|rs)`` with the kernel ``alpha/r + beta erfc(omega r)/r`` from libcint. Shell functions are
    mapped between the two codes by a fitted linear map (no assumption on PySCF's harmonics).
    """
    gto = pytest.importorskip("pyscf.gto")
    from test_eri import _shell_values

    cell = np.eye(3) * 8.0
    pos = [[4.0, 4.0, 4.0], [4.757, 4.586, 4.0], [3.243, 4.586, 4.0]]
    structure, orbitals, pattern = _setup(cell, ["O", "H", "H"], pos, "DZP")
    n_orb = int(orbitals.n_orb)
    assert int(pattern.src_orb.shape[0]) == n_orb * n_orb  # every pair of the molecule, no image
    stage = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=4, **TIGHT)
    dm = _symmetric_dm(pattern, np.random.default_rng(3), 2)
    # A third matrix without symmetry: the spin blocks of a non-collinear density matrix have none.
    dm_all = np.concatenate([dm, np.random.default_rng(5).standard_normal((1, dm.shape[1]))])
    k_mine = np.stack([_dense(pattern, row, n_orb) for row in stage.exchange_matrices(dm_all)])

    shells = []
    for ia, sp_idx in enumerate(structure.species_indices):
        for gs in stage.species_shells[sp_idx]:
            shells.append((gs.l, gs.exponents, gs.coefficients, np.asarray(structure.positions[ia], dtype=float)))
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[f"X{i}", tuple(s[3])] for i, s in enumerate(shells)]
    mol.basis = {f"X{i}": [[s[0]] + [[float(a), float(c / gto.gto_norm(s[0], a))] for a, c in zip(s[1], s[2])]] for i, s in enumerate(shells)}
    mol.charge, mol.spin, mol.verbose = 0, 0, 0
    mol._atom  # noqa: B018
    mol.build(parse_arg=False, dump_input=False)
    eri = spec.alpha * mol.intor("int2e_sph")
    with mol.with_range_coulomb(-spec.omega):
        eri = eri + spec.beta * mol.intor("int2e_sph")
    centre = np.asarray(structure.positions).mean(axis=0)
    pts = centre + np.random.default_rng(7).uniform(-3.0, 3.0, (600, 3))
    ao = mol.eval_gto("GTOval_sph", pts)
    u = np.zeros((n_orb, n_orb))  # mine = ao @ u
    off = 0
    for s in shells:
        n = 2 * s[0] + 1
        mine = _shell_values(s, pts)
        blk, *_ = np.linalg.lstsq(ao[:, off : off + n], mine, rcond=None)
        assert np.max(np.abs(ao[:, off : off + n] @ blk - mine)) <= 1e-9 * np.max(np.abs(mine))
        u[off : off + n, off : off + n] = blk
        off += n
    for p, k in zip(dm_all, k_mine):
        d = u @ _dense(pattern, p, n_orb) @ u.T
        ref = u.T @ np.einsum("pqrs,qs->pr", eri, d) @ u
        assert np.max(np.abs(k - ref)) <= 1e-9 * np.max(np.abs(ref))

    h_x, e_x = stage.matrix_and_energy(dm, 2)
    assert np.allclose(h_x, -stage.exchange_matrices(dm), rtol=0, atol=1e-13)
    assert e_x == pytest.approx(0.5 * float(np.sum(dm * h_x)), rel=1e-13)
    h_1, e_1 = stage.matrix_and_energy(dm[:1] * 2.0, 1)  # closed shell: total density matrix = 2 P
    assert np.allclose(h_1[0], h_x[0], rtol=0, atol=1e-12)
    assert e_1 == pytest.approx(float(np.sum(dm[0] * h_x[0])), rel=1e-12)


def _spinor_dm(pattern, rng: np.random.Generator) -> np.ndarray:
    """Random spinor density matrix ``[uu, dd, Re ud, Im ud]``: Hermitian diagonal blocks, general ``ud``."""
    t_idx = np.asarray(pattern.transpose_edge_idx)
    raw = rng.standard_normal((6, int(pattern.src_orb.shape[0])))
    sym, anti = 0.5 * (raw[:2] + raw[:2, t_idx]), 0.5 * (raw[2:4] - raw[2:4, t_idx])
    return np.stack([sym[0] + 1j * anti[0], sym[1] + 1j * anti[1], raw[4] + 0j, raw[5] + 0j])


def _spin_blocks(m: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Blocks ``[[uu, ud], [du, dd]]`` of a spinor matrix in the channel layout; ``du`` is the conjugate transpose."""
    ud = m[2] + 1j * m[3]
    return np.array([[m[0], ud], [np.conj(ud[t_idx]), m[1]]])


def _spin_channels(blocks: np.ndarray) -> np.ndarray:
    return np.stack([blocks[0, 0], blocks[1, 1], blocks[0, 1].real + 0j, blocks[0, 1].imag + 0j])


def _spinor_trace(dm: np.ndarray, h: np.ndarray) -> float:
    weight = np.array([1.0, 1.0, 2.0, 2.0])[:, None]
    return float(np.sum(weight * np.real(np.conj(dm) * h)))


def test_exchange_of_spinor_density_matrices() -> None:
    """Identities of the exchange matrix and energy for ``nspin`` 4 and 8 (complex spin blocks).

    No external reference of its own: the kernel acting on real matrices without symmetry is compared with PySCF
    in ``test_exchange_matrix_of_water_against_pyscf``. Checked here are the assembly of the spin blocks:
    a collinear density matrix in the spinor layout gives the ``nspin = 2`` result; a rotation of the spin axes
    leaves the energy unchanged and rotates the matrix; the matrix is the derivative of the energy, with the
    trace convention of the SCF driver.
    """
    spec = ExactExchangeSpec(0.0, 0.25, 0.11)
    cell = np.array([[1.6, 0.1, 0.0], [0.0, 4.8, 0.2], [0.1, 0.0, 4.8]])
    structure, orbitals, pattern = _setup(cell, ["H", "H"], [[0.1, 0.2, 0.3], [0.75, 0.5, 0.45]], "SZ")
    stage = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=3, **TIGHT)
    t_idx = np.asarray(pattern.transpose_edge_idx)
    rng = np.random.default_rng(21)

    dm2 = _symmetric_dm(pattern, rng, 2)
    h2, e2 = stage.matrix_and_energy(dm2, 2)
    zero = np.zeros_like(dm2[0])
    h4, e4 = stage.matrix_and_energy(np.stack([dm2[0], dm2[1], zero, zero]) + 0j, 4)
    assert e4 == pytest.approx(e2, rel=1e-13)
    assert np.allclose(h4[:2], h2, rtol=0, atol=1e-13) and np.max(np.abs(h4[2:])) == 0.0

    dm = _spinor_dm(pattern, rng)
    h_x, e_x = stage.matrix_and_energy(dm, 8)
    assert e_x == pytest.approx(0.5 * _spinor_trace(dm, h_x), rel=1e-13)
    assert np.max(np.abs(h_x[:2] - np.conj(h_x[:2, t_idx]))) <= 1e-12 * np.max(np.abs(h_x))  # Hermitian blocks
    assert np.max(np.abs(h_x[:2].imag)) > 1e-3 * np.max(np.abs(h_x))  # the imaginary part is not negligible

    # Rotation of the spin axes by a general SU(2) matrix.
    angle, axis = 0.7, np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    rot = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * np.einsum("a,aij->ij", axis, pauli)
    dm_rot = _spin_channels(np.einsum("sa,abe,tb->ste", rot, _spin_blocks(dm, t_idx), rot.conj()))
    h_rot, e_rot = stage.matrix_and_energy(dm_rot, 8)
    assert e_rot == pytest.approx(e_x, rel=1e-12)
    expected = _spin_channels(np.einsum("sa,abe,tb->ste", rot, _spin_blocks(h_x, t_idx), rot.conj()))
    assert np.max(np.abs(h_rot - expected)) <= 1e-12 * np.max(np.abs(h_x))

    # dE = Re Tr[dD^dagger H_x]; E is quadratic, so the central difference is exact up to rounding.
    delta = _spinor_dm(pattern, rng)
    step = 1e-3
    fd = (stage.matrix_and_energy(dm + step * delta, 8)[1] - stage.matrix_and_energy(dm - step * delta, 8)[1]) / (2 * step)
    assert fd == pytest.approx(_spinor_trace(delta, h_x), rel=1e-9)


def test_exchange_matrix_lattice_sums_are_consistent_with_a_supercell() -> None:
    """A translation-invariant density matrix gives the same ``K`` in the unit cell and in a 2x1x1 supercell.

    Independent identity for the bookkeeping of the lattice translations (cells of the bra, of the ket and of the
    density matrix). A hydrogen chain-like crystal with a short cell so that several images contribute; also checks
    symmetry under edge transposition and linearity.
    """
    spec = ExactExchangeSpec(0.0, 0.25, 0.11)
    a = 1.6  # Angstrom
    cell = np.diag([a, 3.0 * a, 3.0 * a])
    structure, orbitals, pattern = _setup(cell, ["H"], [[0.1, 0.2, 0.3]], "SZ")
    stage = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=3, **TIGHT)
    rng = np.random.default_rng(11)
    dm = _symmetric_dm(pattern, rng, 2)
    k_unit = stage.exchange_matrices(dm)
    assert stage.last_stats["quartets_computed"] > 100  # several images contribute
    t_idx = np.asarray(pattern.transpose_edge_idx)
    assert np.max(np.abs(k_unit - k_unit[:, t_idx])) <= 1e-12 * np.max(np.abs(k_unit))
    k_sum = stage.exchange_matrices((dm[0] + 0.3 * dm[1])[None, :])[0]
    assert np.max(np.abs(k_sum - (k_unit[0] + 0.3 * k_unit[1]))) <= 1e-12 * np.max(np.abs(k_unit))

    cell2 = np.diag([2.0 * a, 3.0 * a, 3.0 * a])
    structure2, orbitals2, pattern2 = _setup(cell2, ["H", "H"], [[0.1, 0.2, 0.3], [0.1 + a, 0.2, 0.3]], "SZ")
    stage2 = Nao2GtoExchange(structure2, orbitals2, pattern2, spec, n_gauss=3, **TIGHT)
    # Edge (mu in copy c, lambda in copy c', cell G') of the supercell <-> unit-cell edge with shift 2 G' + c' - c.
    shifts = np.rint(np.asarray(pattern.cell_shifts)).astype(int)
    unit_edge = {(int(s), int(d), *shifts[int(c)]): e for e, (s, d, c) in enumerate(zip(np.asarray(pattern.src_orb), np.asarray(pattern.dst_unit_orb), np.asarray(pattern.edge_cell_idx)))}
    shifts2 = np.rint(np.asarray(pattern2.cell_shifts)).astype(int)
    src2, dst2, cel2 = (np.asarray(x) for x in (pattern2.src_orb, pattern2.dst_unit_orb, pattern2.edge_cell_idx))
    n_per = int(orbitals.n_orb)
    mapping = np.empty(src2.shape[0], dtype=int)
    for e in range(src2.shape[0]):
        g = shifts2[int(cel2[e])]
        c_src, c_dst = int(src2[e]) // n_per, int(dst2[e]) // n_per
        mapping[e] = unit_edge[(int(src2[e]) % n_per, int(dst2[e]) % n_per, 2 * g[0] + c_dst - c_src, g[1], g[2])]
    k_super = stage2.exchange_matrices(dm[:, mapping])
    assert np.max(np.abs(k_super - k_unit[:, mapping])) <= 1e-11 * np.max(np.abs(k_unit))


def test_exchange_forces_and_strain_derivative_against_finite_differences() -> None:
    """Derivatives of ``E_x`` at a fixed density matrix equal central differences of ``E_x``.

    Independent identity for the derivative integrals, for the use of the permutation symmetry (one derivative per
    quartet stands for four) and for the virial form of the strain derivative. A triclinic periodic cell with
    several images, the short-range kernel, collinear and spinor density matrices.
    """
    bohr = 0.529177210903
    spec = ExactExchangeSpec(0.0, 0.25, 0.11)
    cell = np.array([[1.6, 0.1, 0.0], [0.0, 4.8, 0.2], [0.1, 0.0, 4.8]])
    pos = np.array([[0.1, 0.2, 0.3], [0.75, 0.5, 0.45]])
    structure, orbitals, pattern = _setup(cell, ["H", "H"], pos, "SZ")
    stage = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=3, **TIGHT)

    def energy(cell_now: np.ndarray, pos_now: np.ndarray, dm: np.ndarray, nspin: int) -> float:
        s, o, p = _setup(cell_now, ["H", "H"], pos_now, "SZ")
        assert np.array_equal(np.asarray(p.src_orb), np.asarray(pattern.src_orb))
        assert np.array_equal(np.asarray(p.edge_cell_idx), np.asarray(pattern.edge_cell_idx))
        return Nao2GtoExchange(s, o, p, spec, n_gauss=3, **TIGHT).matrix_and_energy(dm, nspin)[1]

    for nspin in (1, 2, 4):
        rng = np.random.default_rng(nspin)
        dm = _spinor_dm(pattern, rng) if nspin == 4 else _symmetric_dm(pattern, rng, nspin)
        forces, de_deps = stage.forces_and_strain_derivative(dm, nspin)
        assert np.max(np.abs(forces.sum(axis=0))) <= 1e-12
        step = 2.5e-4  # Angstrom; the error of the central difference is 2e-8 here (3e-7 with 1e-3)
        for ia, d in ((0, 0), (1, 2)):
            shift = np.zeros_like(pos)
            shift[ia, d] = step
            fd = -(energy(cell, pos + shift, dm, nspin) - energy(cell, pos - shift, dm, nspin)) / (2 * step / bohr)
            assert forces[ia, d] == pytest.approx(fd, abs=1e-7)
        eps_step = 1e-4
        for a, b in ((0, 0), (1, 2), (2, 0)):
            eps = np.zeros((3, 3))
            eps[a, b] = eps_step
            plus, minus = np.eye(3) + eps, np.eye(3) - eps  # rows of the cell are lattice vectors: r -> (1 + eps) r
            fd = (energy(cell @ plus.T, pos @ plus.T, dm, nspin) - energy(cell @ minus.T, pos @ minus.T, dm, nspin)) / (2 * eps_step)
            assert de_deps[a, b] == pytest.approx(fd, abs=1e-7)



def test_shared_integrals_and_incremental_builds() -> None:
    """The build that computes each integral once, and the incremental build, in a crystal with s and p shells.

    Each integral serves up to eight contributions to ``K`` (Stage 9). Independent of the symmetry of the density
    matrix, ``K[P^T] = K[P]^T``; a wrong assignment of the cells of the density-matrix or of the result blocks of
    one of the eight breaks it for a matrix that is not symmetric (the reference values themselves are checked
    against PySCF and against a supercell in the tests above). The incremental build must agree with a full build
    of the new density matrix within the screening tolerance while computing fewer integrals, and return to a
    full build after ``max_incremental`` steps.
    """
    spec = ExactExchangeSpec(0.0, 0.25, 0.11)
    cell = np.array([[2.6, 0.0, 0.0], [0.4, 2.8, 0.0], [0.0, 0.3, 7.0]])
    structure, orbitals, pattern = _setup(cell, ["O", "H"], [[0.1, 0.2, 0.3], [0.9, 0.7, 0.8]], "SZ")
    n_edges = int(pattern.src_orb.shape[0])
    t_idx = np.asarray(pattern.transpose_edge_idx)
    rng = np.random.default_rng(5)

    stage = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=2, **TIGHT)
    raw = rng.standard_normal((1, n_edges))
    k_raw = stage.exchange_matrices(raw)
    k_t = stage.exchange_matrices(raw[:, t_idx])
    assert np.max(np.abs(raw - raw[:, t_idx])) > 0.1
    assert np.max(np.abs(k_t - k_raw[:, t_idx])) <= 1e-12 * np.max(np.abs(k_raw))

    tol = 1e-8
    dm = _symmetric_dm(pattern, rng, 2)
    delta = _symmetric_dm(pattern, rng, 2)
    full = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=2, screen_tol=tol)
    steps = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=2, screen_tol=tol, incremental=True, max_incremental=2)
    k_first = steps.exchange_matrices(dm)
    n_full = steps.last_stats["quartets_computed"]
    assert steps.last_stats["incremental"] == 0
    assert np.max(np.abs(k_first - full.exchange_matrices(dm))) < 1e-13  # the order of the sums depends on the threads
    for n, scale in enumerate((1e-4, 1e-6), start=1):
        dm_n = dm + scale * n * delta
        k_n = steps.exchange_matrices(dm_n)
        assert steps.last_stats["incremental"] == 1
        assert steps.last_stats["quartets_computed"] < n_full
        assert np.max(np.abs(k_n - full.exchange_matrices(dm_n))) < 1e-6
    steps.exchange_matrices(dm)  # third call after a full build: full again
    assert steps.last_stats["incremental"] == 0
    assert steps.last_stats["quartets_computed"] == n_full


def test_distance_bound_of_the_screened_kernel() -> None:
    """The bound from the distance between two pair densities removes integrals without changing the result.

    On a pattern of three times the orbital range (where the bound matters), the exchange matrix with and without the
    bound agrees within the screening tolerance while fewer integrals are computed; with a tolerance of zero the
    bound must not remove anything that matters at all. A kernel with a full-range part is not affected.
    """
    a = 1.6  # Angstrom
    structure, orbitals, _ = _setup(np.diag([a, 3.0 * a, 3.0 * a]), ["H"], [[0.1, 0.2, 0.3]], "SZ")
    pattern = build_sparse_pattern(structure, orbitals, orbital_range_scale=3.0)
    dm = _symmetric_dm(pattern, np.random.default_rng(3), 1)
    counts = {}
    for name, spec in (("screened", ExactExchangeSpec(0.0, 0.25, 0.11)), ("full", ExactExchangeSpec(0.25, 0.0, 0.0))):
        results = []
        for bound in (True, False):
            stage = Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=3, screen_tol=1e-9, distance_screening=bound)
            results.append(stage.exchange_matrices(dm))
            counts[name, bound] = stage.last_stats["quartets_computed"]
        assert np.max(np.abs(results[0] - results[1])) < 1e-7
    assert counts["screened", True] < 0.8 * counts["screened", False]  # measured: 3446 of 5454
    assert counts["full", True] == counts["full", False]
    spec = ExactExchangeSpec(0.0, 0.25, 0.11)
    exact = [Nao2GtoExchange(structure, orbitals, pattern, spec, n_gauss=3, distance_screening=b, **TIGHT).exchange_matrices(dm) for b in (True, False)]
    assert np.max(np.abs(exact[0] - exact[1])) < 1e-12
