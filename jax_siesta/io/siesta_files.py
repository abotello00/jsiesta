r"""SIESTA restart files: ``.XV`` geometries, ``.DM`` density matrices and ``.nc`` sparse-matrix files.

Theory:
    Three SIESTA files carry the state needed to continue a calculation:

    - ``<label>.XV`` (formatted): lattice vectors, species index, atomic number, positions (Bohr) and
      velocities (Bohr/fs) of every atom.
    - ``<label>.DM`` (Fortran unformatted): the sparse density matrix :math:`D_{\mu\nu}(\mathbf{T})` on SIESTA's
      own sparsity pattern, with the auxiliary supercell size ``nsc``.
    - ``<label>.nc`` (NetCDF-4, ``CDF.Save T``): geometry, sparsity pattern, overlap :math:`S`, density matrix,
      energy-density matrix and (with ``SaveHS T``) the Hamiltonian :math:`H` in Ry, plus the Fermi level.

    SIESTA stores row ``io`` (a unit-cell orbital) and column ``jo = isc * no_u + juo`` (orbital ``juo`` in
    supercell ``isc``). Supercell ``isc = i1 + nsc1 * (i2 + nsc2 * i3)`` with ``i_a`` in ``[0, nsc_a)`` has the
    lattice offset ``i_a`` if ``i_a <= nsc_a / 2`` and ``i_a - nsc_a`` otherwise (``superx`` in ``atomlist.f``,
    ``linear2pm`` in ``m_handle_sparse.F90``). A jsiesta edge ``(src, dst, T)`` is the SIESTA element
    ``(io = src, juo = dst, offset = T)``.

    Spin components. Let :math:`D^{\sigma\sigma'}_{\mu\nu} = \sum f\, c_{\mu\sigma} c^*_{\nu\sigma'}` and
    :math:`h^{\sigma\sigma'}_{\mu\nu} = \langle \mu\sigma | H | \nu\sigma' \rangle` be the jsiesta edge values
    (channels ``[uu, dd, Re(ud), Im(ud)]``, with complex ``uu`` and ``dd`` when needed). SIESTA stores
    (``diag3k.F``)

    .. math::
        D_{1..8} &= \big(\mathrm{Re} D^{uu}, \mathrm{Re} D^{dd}, \mathrm{Re} D^{ud}, -\mathrm{Im} D^{ud},
                         \mathrm{Im} D^{uu}, \mathrm{Im} D^{dd}, \mathrm{Re} D^{du}, \mathrm{Im} D^{du}\big), \\
        H^{\sigma\sigma'}_{\text{SIESTA}}(io, jo) &= \langle jo\,\sigma | H | io\,\sigma' \rangle
            = \big(h^{\sigma'\sigma}_{io,jo}\big)^*,

    with :math:`H^{uu} = H_1 + iH_5`, :math:`H^{dd} = H_2 + iH_6`, :math:`H^{ud} = H_3 - iH_4`,
    :math:`H^{du} = H_7 + iH_8`. The ``du`` blocks follow from Hermiticity:
    :math:`X^{du}(\text{edge}) = (X^{ud}(\text{transposed edge}))^*`. With one spin component SIESTA's ``D`` is
    the sum over both spins, as in jsiesta. SIESTA energies are in Ry, jsiesta's in Hartree.

Algorithm:
    Pure NumPy readers and writers. A file's sparsity pattern is matched to a ``SparsePattern`` through integer
    keys ``(row, column orbital, offset)``; elements absent on either side are dropped or left zero, as SIESTA
    does when it reads a ``.DM`` file written for another pattern. A file from a run without auxiliary supercell
    (``nsc = 1`` along an axis, the :math:`\Gamma`-only case) holds elements folded over the lattice images; see
    ``map_siesta_to_edges`` for how they are unfolded.

Invariants:
    - Writers produce one element per pattern edge, in rows ordered by column index.
    - ``write_dm_file(read_dm_file(f))`` reproduces a SIESTA-written file byte for byte.
    - Readers raise ``ValueError`` when the number of orbitals differs from the pattern's.

SIESTA Mapping:
    ``ioxv.F`` (``.XV``), ``m_iodm.F90`` + ``io_sparse.F90`` (``.DM``), ``m_ncdf_siesta.F90`` + ``ncdf_io.F90``
    (``.nc``), ``m_new_dm.F90`` / ``m_handle_sparse.F90`` (reading a density matrix into another pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import (
    HARTREE_TO_RY,
    RY_TO_HARTREE,
    BasisData,
    SCFConfig,
    SCFState,
    SparsePattern,
    Structure,
)

ELEMENT_SYMBOLS: tuple[str, ...] = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
)  # fmt: skip
"""Element symbols indexed by atomic number (``"X"`` at 0), used to label species read from an ``.XV`` file."""


# --------------------------------------------------------------------------------------------------------------
# Sparsity pattern in SIESTA's layout
# --------------------------------------------------------------------------------------------------------------


def siesta_supercell_offsets(nsc: tuple[int, int, int]) -> np.ndarray:
    r"""Lattice offsets of SIESTA's auxiliary supercells in SIESTA's order.

    Theory:
        Supercell ``isc = i1 + nsc1 * (i2 + nsc2 * i3)`` has offset ``linear2pm(i_a, nsc_a)`` along axis ``a``:
        ``i_a`` when ``i_a <= nsc_a / 2`` (integer division), ``i_a - nsc_a`` otherwise.

    Algorithm:
        Three nested index ranges, first axis fastest.

    Parameters:
        nsc: Number of supercells along each lattice vector.

    Returns:
        Integer array ``(prod(nsc), 3)``; row 0 is ``(0, 0, 0)``.

    Invariants:
        Rows are distinct.

    SIESTA Mapping:
        ``isc_off`` in ``siesta_geom.F90``; ``generate_isc`` and ``linear2pm`` in ``m_handle_sparse.F90``.
    """
    axes = []
    for n in nsc:
        i = np.arange(int(n), dtype=np.int64)
        axes.append(np.where(i > int(n) // 2, i - int(n), i))
    g3, g2, g1 = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)


@dataclass(frozen=True)
class SiestaSparsity:
    r"""A sparsity pattern in SIESTA's row / supercell-column layout.

    Theory:
        Row ``io`` holds ``n_col[io]`` elements; element ``ind`` couples ``io`` to the supercell orbital
        ``list_col[ind] = isc * no_u + juo`` (0-based here, 1-based in the files).

    Algorithm:
        Plain NumPy container; ``isc_off`` defaults to ``siesta_supercell_offsets(nsc)``.

    Invariants:
        - ``sum(n_col) == len(list_col)``.
        - ``isc_off`` has shape ``(prod(nsc), 3)``.

    SIESTA Mapping:
        ``numh``, ``listh`` (``sparse_matrices.F90``) and ``nsc``, ``isc_off`` (``siesta_geom.F90``).
    """

    no_u: int
    nsc: tuple[int, int, int]
    n_col: np.ndarray
    list_col: np.ndarray
    isc_off: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))

    def __post_init__(self) -> None:
        if self.isc_off.shape[0] == 0:
            object.__setattr__(self, "isc_off", siesta_supercell_offsets(self.nsc))
        if int(np.sum(self.n_col)) != int(self.list_col.shape[0]):
            raise ValueError("SiestaSparsity: sum(n_col) differs from the number of column indices.")
        if self.isc_off.shape != (int(np.prod(self.nsc)), 3):
            raise ValueError("SiestaSparsity: isc_off does not have one row per supercell.")

    @property
    def nnzs(self) -> int:
        """Number of stored elements."""
        return int(self.list_col.shape[0])

    @property
    def rows(self) -> np.ndarray:
        """Row (unit-cell orbital) of every stored element."""
        return np.repeat(np.arange(self.no_u, dtype=np.int64), self.n_col)


@dataclass(frozen=True)
class MappingInfo:
    """Bookkeeping of one file-to-pattern mapping (``map_siesta_to_edges``).

    Theory:
        Counts how much of a file's matrix landed on the jsiesta pattern.

    Algorithm:
        Filled from the key match.

    Invariants:
        ``n_edges_matched <= n_edges`` and ``n_file_unused <= n_file``.

    SIESTA Mapping:
        The diagnostics SIESTA prints in ``restruct_dSpData2D`` when patterns differ.
    """

    n_file: int
    n_edges: int
    n_edges_matched: int
    n_file_unused: int
    folded_axes: tuple[bool, bool, bool]


def _edge_table(pattern: SparsePattern) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(pattern.src_orb, dtype=np.int64)
    dst = np.asarray(pattern.dst_unit_orb, dtype=np.int64)
    shifts = np.asarray(pattern.cell_shifts, dtype=np.int64)[np.asarray(pattern.edge_cell_idx, dtype=np.int64)]
    return src, dst, shifts.reshape(-1, 3)


def pattern_to_siesta(pattern: SparsePattern) -> tuple[SiestaSparsity, np.ndarray]:
    r"""Express a jsiesta ``SparsePattern`` in SIESTA's layout.

    Theory:
        The auxiliary supercell must contain every shift of the pattern: ``nsc_a = 2 max|T_a| + 1`` (odd, as
        SIESTA's own). SIESTA accepts any such ``nsc`` on reading and re-indexes the columns to its own.

    Algorithm:
        Column index ``(T_a mod nsc_a)`` combined as ``i1 + nsc1 * (i2 + nsc2 * i3)``, then a sort by
        ``(row, column)``.

    Parameters:
        pattern: Static sparsity pattern.

    Returns:
        ``(sparsity, order)``: ``order[ind]`` is the edge stored as SIESTA element ``ind``.

    Invariants:
        ``order`` is a permutation of ``range(n_edges)``.

    SIESTA Mapping:
        ``sparse_pattern`` as written by ``io_write_Sp`` / ``cdf_w_Sp``.
    """
    src, dst, shifts = _edge_table(pattern)
    n_orb = int(pattern.n_orb)
    half = np.max(np.abs(shifts), axis=0) if shifts.shape[0] else np.zeros(3, dtype=np.int64)
    nsc = (int(2 * half[0] + 1), int(2 * half[1] + 1), int(2 * half[2] + 1))
    idx = np.mod(shifts, np.asarray(nsc, dtype=np.int64)[None, :])
    isc = idx[:, 0] + nsc[0] * (idx[:, 1] + nsc[1] * idx[:, 2])
    col = isc * n_orb + dst
    order = np.lexsort((col, src))
    n_col = np.bincount(src, minlength=n_orb).astype(np.int32)
    return SiestaSparsity(n_orb, nsc, n_col, col[order].astype(np.int64)), order


def map_siesta_to_edges(
    sparsity: SiestaSparsity,
    pattern: SparsePattern,
    values: np.ndarray,
    folded: str = "replicate",
) -> tuple[np.ndarray, MappingInfo]:
    r"""Move a matrix from a file's sparsity pattern onto the edges of a jsiesta ``SparsePattern``.

    Theory:
        Elements are identified by ``(io, juo, offset)``. An edge without partner in the file gets zero; a file
        element without edge is dropped. Along an axis where the file has ``nsc = 1`` but the pattern has
        shifted edges, the file was written without auxiliary supercell (:math:`\Gamma`-only run) and its
        element ``(io, juo)`` is the sum over all images. Two ways to unfold it:

        - ``"replicate"``: every image receives the folded value. Exact for a density matrix of a
          :math:`\Gamma`-only calculation, where :math:`D_{\mu\nu}(\mathbf{T})` does not depend on
          :math:`\mathbf{T}`.
        - ``"first"``: one image (the unshifted one when the pattern has it) receives the folded value, the
          others zero. Keeps :math:`\sum_{\mathbf{T}} M_{\mu\nu}(\mathbf{T})`, hence the :math:`\Gamma`-point
          Hamiltonian, but is not the real-space matrix at other k-points.

    Algorithm:
        Integer keys on both sides and one ``searchsorted``.

    Parameters:
        sparsity: The file's pattern.
        pattern: Target pattern.
        values: File values, shape ``(n_comp, nnzs)``.
        folded: ``"replicate"`` or ``"first"``.

    Returns:
        ``(edge_values, info)`` with ``edge_values`` of shape ``(n_comp, n_edges)``.

    Invariants:
        Raises ``ValueError`` when the numbers of orbitals differ.

    SIESTA Mapping:
        ``correct_supercell_SpD``, ``unfold_noauxiliary_supercell_Sp2D`` and ``restruct_dSpData2D``.
    """
    if folded not in ("replicate", "first"):
        raise ValueError(f"folded must be 'replicate' or 'first', not {folded!r}.")
    n_orb = int(pattern.n_orb)
    if sparsity.no_u != n_orb:
        raise ValueError(f"The file has {sparsity.no_u} orbitals, the calculation has {n_orb}: different basis sets.")
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != sparsity.nnzs:
        raise ValueError("values must have shape (n_comp, nnzs).")
    src, dst, shifts = _edge_table(pattern)
    n_edges = src.shape[0]

    f_rows = sparsity.rows
    f_juo = sparsity.list_col % n_orb
    f_off = sparsity.isc_off[sparsity.list_col // n_orb]

    used_axes = np.any(shifts != 0, axis=0) if n_edges else np.zeros(3, dtype=bool)
    folded_axes = np.asarray([sparsity.nsc[a] == 1 and bool(used_axes[a]) for a in range(3)])
    e_off = np.where(folded_axes[None, :], 0, shifts)

    half = 1 + int(max(np.max(np.abs(f_off)) if f_off.size else 0, np.max(np.abs(e_off)) if e_off.size else 0))
    stride = 2 * half + 1

    def _key(rows: np.ndarray, cols: np.ndarray, off: np.ndarray) -> np.ndarray:
        t_code = ((off[:, 0] + half) * stride + (off[:, 1] + half)) * stride + (off[:, 2] + half)
        return (rows * n_orb + cols) * stride**3 + t_code

    f_key = _key(f_rows, f_juo, f_off)
    e_key = _key(src, dst, e_off)
    f_order = np.argsort(f_key, kind="stable")
    f_sorted = f_key[f_order]
    pos = np.clip(np.searchsorted(f_sorted, e_key), 0, max(f_sorted.shape[0] - 1, 0))
    matched = (f_sorted[pos] == e_key) if f_sorted.shape[0] else np.zeros(n_edges, dtype=bool)

    if folded == "first" and bool(np.any(folded_axes)):
        # Keep one edge per folded key: the one with the smallest shift along the folded axes.
        dist = np.sum(np.where(folded_axes[None, :], shifts, 0) ** 2, axis=1)
        e_order = np.lexsort((dist, e_key))
        first = np.ones(n_edges, dtype=bool)
        first[1:] = e_key[e_order][1:] != e_key[e_order][:-1]
        keep = np.zeros(n_edges, dtype=bool)
        keep[e_order[first]] = True
        matched &= keep

    out = np.zeros((values.shape[0], n_edges), dtype=values.dtype)
    if f_sorted.shape[0]:
        out[:, matched] = values[:, f_order[pos[matched]]]
    n_unused = int(np.count_nonzero(~np.isin(f_key, e_key[matched])))
    info = MappingInfo(
        n_file=sparsity.nnzs,
        n_edges=int(n_edges),
        n_edges_matched=int(np.count_nonzero(matched)),
        n_file_unused=n_unused,
        folded_axes=(bool(folded_axes[0]), bool(folded_axes[1]), bool(folded_axes[2])),
    )
    return out, info


# --------------------------------------------------------------------------------------------------------------
# Spin-component conventions
# --------------------------------------------------------------------------------------------------------------


def _n_spin_file(nspin: int) -> int:
    if nspin not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported number of spin components: {nspin}.")
    return nspin


def dm_to_siesta(dm: Any, pattern: SparsePattern, nspin: int, n_comp: int | None = None) -> np.ndarray:
    r"""Density (or energy-density) matrix in SIESTA's spin components, edge order.

    Theory:
        See the module docstring: ``(Re D^uu, Re D^dd, Re D^ud, -Im D^ud, Im D^uu, Im D^dd, Re D^du, Im D^du)``
        with :math:`D^{du}(\text{edge}) = (D^{ud}(\text{transposed edge}))^*`.

    Algorithm:
        Channel arithmetic on the jsiesta array ``(n_spin_mat, n_edges)``.

    Parameters:
        dm: jsiesta matrix, real or complex.
        pattern: Its pattern (for the transposed edges).
        nspin: Spin mode of the calculation.
        n_comp: Components to return (default ``nspin``); 4 gives the non-collinear subset SIESTA uses for the
            energy-density matrix of a spin-orbit run.

    Returns:
        Real array ``(n_comp, n_edges)``.

    Invariants:
        ``dm_from_siesta`` inverts this for a Hermitian matrix.

    SIESTA Mapping:
        ``Dscf`` / ``Escf`` as filled in ``diagg.F``, ``diagk.F``, ``diag2k.F`` and ``diag3k.F``.
    """
    _n_spin_file(nspin)
    d = np.asarray(dm)
    if nspin == 1:
        return np.real(d[:1]).astype(np.float64)
    if nspin == 2:
        return np.real(d[:2]).astype(np.float64)
    tr = np.asarray(pattern.transpose_edge_idx, dtype=np.int64)
    comps = [np.real(d[0]), np.real(d[1]), np.real(d[2]), -np.real(d[3])]
    if nspin == 8 and (n_comp is None or n_comp == 8):
        comps += [np.imag(d[0]), np.imag(d[1]), np.real(d[2])[tr], -np.real(d[3])[tr]]
    return np.asarray(comps, dtype=np.float64)


def dm_from_siesta(values: np.ndarray) -> tuple[np.ndarray, int]:
    r"""jsiesta density-matrix channels from SIESTA spin components (edge order).

    Theory:
        Inverse of ``dm_to_siesta``; components 7 and 8 are redundant for a Hermitian matrix and not used.

    Algorithm:
        Channel arithmetic.

    Parameters:
        values: SIESTA components ``(n_comp, n_edges)`` with ``n_comp`` in ``(1, 2, 4, 8)``.

    Returns:
        ``(dm, nspin)``: real ``(n_comp, n_edges)`` for one or two components, complex ``(4, n_edges)`` otherwise.

    Invariants:
        ``nspin == values.shape[0]``.

    SIESTA Mapping:
        ``Dscf`` in ``sparse_matrices.F90``.
    """
    v = np.asarray(values, dtype=np.float64)
    n = _n_spin_file(int(v.shape[0]))
    if n <= 2:
        return v.copy(), n
    uu = v[0].astype(np.complex128)
    dd = v[1].astype(np.complex128)
    if n == 8:
        uu = uu + 1j * v[4]
        dd = dd + 1j * v[5]
    return np.stack([uu, dd, v[2].astype(np.complex128), (-v[3]).astype(np.complex128)], axis=0), n


def h_to_siesta(h: Any, pattern: SparsePattern, nspin: int) -> np.ndarray:
    r"""Hamiltonian in SIESTA's spin components and in Ry, edge order.

    Theory:
        :math:`H^{\sigma\sigma'}_{\text{SIESTA}} = (h^{\sigma'\sigma})^*` per edge (module docstring), hence
        ``H1 = Re h^uu``, ``H2 = Re h^dd``, ``H5 = -Im h^uu``, ``H6 = -Im h^dd``, ``H7 + iH8 = (h^ud)^*`` and
        ``H3 - iH4 = h^ud(transposed edge)``.

    Algorithm:
        Channel arithmetic, then the factor 2 from Hartree to Ry.

    Parameters:
        h: jsiesta Hamiltonian ``(n_spin_mat, n_edges)`` in Hartree.
        pattern: Its pattern.
        nspin: Spin mode.

    Returns:
        Real array ``(nspin, n_edges)`` in Ry.

    Invariants:
        ``h_from_siesta`` inverts this.

    SIESTA Mapping:
        ``H`` in ``sparse_matrices.F90`` as read by ``diag3k.F``.
    """
    _n_spin_file(nspin)
    x = np.asarray(h)
    if nspin <= 2:
        return HARTREE_TO_RY * np.real(x[:nspin]).astype(np.float64)
    tr = np.asarray(pattern.transpose_edge_idx, dtype=np.int64)
    h2 = np.real(x[2])
    h3 = np.real(x[3])
    comps = [np.real(x[0]), np.real(x[1]), h2[tr], -h3[tr]]
    if nspin == 8:
        comps += [-np.imag(x[0]), -np.imag(x[1]), h2, -h3]
    return HARTREE_TO_RY * np.asarray(comps, dtype=np.float64)


def h_from_siesta(values: np.ndarray, pattern: SparsePattern) -> tuple[np.ndarray, int]:
    r"""jsiesta Hamiltonian channels (Hartree) from SIESTA spin components (Ry), edge order.

    Theory:
        Inverse of ``h_to_siesta``. With four components the ``ud`` block is taken from the transposed edge;
        with eight, from components 7 and 8 of the edge itself.

    Algorithm:
        Channel arithmetic, then the factor 1/2 from Ry to Hartree.

    Parameters:
        values: SIESTA components ``(n_comp, n_edges)`` in Ry.
        pattern: Target pattern.

    Returns:
        ``(h, nspin)``: real for 1, 2 and 4 components, complex ``(4, n_edges)`` for 8.

    Invariants:
        ``nspin == values.shape[0]``.

    SIESTA Mapping:
        ``H`` in ``sparse_matrices.F90``.
    """
    v = RY_TO_HARTREE * np.asarray(values, dtype=np.float64)
    n = _n_spin_file(int(v.shape[0]))
    if n <= 2:
        return v, n
    if n == 4:
        tr = np.asarray(pattern.transpose_edge_idx, dtype=np.int64)
        return np.stack([v[0], v[1], v[2][tr], -v[3][tr]], axis=0), n
    return (
        np.stack(
            [v[0] - 1j * v[4], v[1] - 1j * v[5], v[6].astype(np.complex128), (-v[7]).astype(np.complex128)], axis=0
        ),
        n,
    )


def convert_spin(matrix: np.ndarray, nspin_from: int, nspin_to: int, kind: str) -> np.ndarray:
    r"""Carry a jsiesta matrix from one spin mode to another (for restarts across spin modes).

    Theory:
        A density matrix is extensive in spin: an unpolarised ``D`` (sum over spins) splits into two halves,
        and two spin channels add up to it. A Hamiltonian is intensive: both spin channels get the unpolarised
        ``H``, and the unpolarised one is their mean. Off-diagonal spin blocks start from zero; the imaginary
        parts of a spin-orbit matrix are dropped when the target has no room for them.

    Algorithm:
        Case analysis on ``(nspin_from, nspin_to)``.

    Parameters:
        matrix: jsiesta channels for ``nspin_from``.
        nspin_from: Spin mode the matrix was built for.
        nspin_to: Spin mode wanted.
        kind: ``"dm"`` (also for the energy-density matrix) or ``"h"``.

    Returns:
        Array in the layout the SCF driver uses for ``nspin_to``.

    Invariants:
        The identity when both modes agree, except for the dtype (complex only where the driver uses it).

    SIESTA Mapping:
        The spin handling of ``init_DM_file`` in ``m_new_dm.F90``.
    """
    if kind not in ("dm", "h"):
        raise ValueError(f"kind must be 'dm' or 'h', not {kind!r}.")
    _n_spin_file(nspin_from)
    _n_spin_file(nspin_to)
    m = np.asarray(matrix)
    # Target dtype: DM is complex for nspin 4 and 8, H only for nspin 8 (as in scf/driver.py).
    want_complex = nspin_to == 8 or (nspin_to == 4 and kind == "dm")

    def _finish(chs: list[np.ndarray]) -> np.ndarray:
        out = np.stack(chs, axis=0)
        return out.astype(np.complex128) if want_complex else np.real(out).astype(np.float64)

    if nspin_from >= 4:
        if nspin_to >= 4:
            return _finish([m[0], m[1], m[2], m[3]])
        up, dn = np.real(m[0]), np.real(m[1])
    elif nspin_from == 2:
        up, dn = np.real(m[0]), np.real(m[1])
    else:
        up = dn = (0.5 if kind == "dm" else 1.0) * np.real(m[0])
    if nspin_to == 1:
        return _finish([(up + dn) if kind == "dm" else 0.5 * (up + dn)])
    if nspin_to == 2:
        return _finish([up, dn])
    zero = np.zeros_like(up)
    return _finish([up, dn, zero, zero])


# --------------------------------------------------------------------------------------------------------------
# .XV
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class XVData:
    """Contents of a SIESTA ``.XV`` file in SIESTA's units.

    Theory:
        Lattice vectors and positions in Bohr, velocities in Bohr/fs.

    Algorithm:
        Plain NumPy container. ``cell`` holds the lattice vectors as columns (the jsiesta convention).

    Invariants:
        ``positions``, ``velocities`` are ``(n_atoms, 3)``; ``species_index`` is 1-based as in the file.

    SIESTA Mapping:
        ``ucell``, ``vcell``, ``isa``, ``iza``, ``xa``, ``va`` of ``ioxv``.
    """

    cell: np.ndarray
    cell_velocity: np.ndarray
    species_index: np.ndarray
    atomic_numbers: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray


def _floats(tokens: list[str]) -> list[float]:
    return [float(t.replace("D", "E").replace("d", "e")) for t in tokens]


def read_xv(path: str | Path) -> XVData:
    """Read a SIESTA ``.XV`` file.

    Theory:
        Three lines with a lattice vector and its velocity, the number of atoms, then one line per atom:
        species index, atomic number, position, velocity.

    Algorithm:
        Whitespace tokenisation (the file is read list-directed by SIESTA too).

    Parameters:
        path: File name.

    Returns:
        ``XVData``.

    Invariants:
        Raises ``ValueError`` on a truncated file.

    SIESTA Mapping:
        ``ioxv('read', ...)`` in ``ioxv.F``.
    """
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 4:
        raise ValueError(f"{path}: not an XV file (fewer than four lines).")
    cell_rows = np.zeros((3, 3))
    vcell_rows = np.zeros((3, 3))
    for iv in range(3):
        vals = _floats(lines[iv].split())
        cell_rows[iv] = vals[:3]
        if len(vals) >= 6:
            vcell_rows[iv] = vals[3:6]
    na = int(lines[3].split()[0])
    if len(lines) < 4 + na:
        raise ValueError(f"{path}: {na} atoms announced, {len(lines) - 4} found.")
    isa = np.zeros(na, dtype=np.int64)
    iza = np.zeros(na, dtype=np.int64)
    xa = np.zeros((na, 3))
    va = np.zeros((na, 3))
    for ia in range(na):
        toks = lines[4 + ia].split()
        isa[ia] = int(toks[0])
        iza[ia] = int(toks[1])
        vals = _floats(toks[2:8])
        xa[ia] = vals[:3]
        if len(vals) >= 6:
            va[ia] = vals[3:6]
    return XVData(cell_rows.T.copy(), vcell_rows.T.copy(), isa, iza, xa, va)


def structure_from_xv(path: str | Path, template: Structure | None = None) -> Structure:
    """Build a ``Structure`` from an ``.XV`` file, optionally on top of an input structure.

    Theory:
        The ``.XV`` file fixes cell and positions; species labels, initial moments and periodicity are not in
        it. With ``template`` (the structure of the input file) those are kept and only the geometry is
        replaced, which is what SIESTA does with ``MD.UseSaveXV``. Without it, species are labelled by element
        symbol in order of species index.

    Algorithm:
        ``read_xv`` followed by consistency checks against the template.

    Parameters:
        path: ``.XV`` file.
        template: Structure whose species data are kept.

    Returns:
        ``Structure`` with the file's cell and positions (Bohr).

    Invariants:
        Raises ``ValueError`` if the number of atoms or the atomic numbers differ from the template's.

    SIESTA Mapping:
        The ``.XV`` branch of ``coor`` / ``siesta_init`` (``MD.UseSaveXV``).
    """
    xv = read_xv(path)
    na = xv.positions.shape[0]
    if template is not None:
        if template.n_atoms != na:
            raise ValueError(f"{path}: {na} atoms, the input structure has {template.n_atoms}.")
        if tuple(int(abs(z)) for z in xv.atomic_numbers) != tuple(int(z) for z in template.atomic_numbers):
            raise ValueError(f"{path}: atomic numbers differ from the input structure.")
        return Structure(
            cell=jnp.asarray(xv.cell, dtype=jnp.float64),
            positions=jnp.asarray(xv.positions, dtype=jnp.float64),
            species_indices=template.species_indices,
            atomic_numbers=template.atomic_numbers,
            species_labels=template.species_labels,
            initial_moments=template.initial_moments,
            pbc=template.pbc,
        )
    uniq = sorted({int(s) for s in xv.species_index})
    dense = {s: i for i, s in enumerate(uniq)}
    labels: list[str] = []
    for s in uniq:
        z = int(abs(xv.atomic_numbers[np.nonzero(xv.species_index == s)[0][0]]))
        if z >= len(ELEMENT_SYMBOLS):
            raise ValueError(f"{path}: no element symbol for Z = {z}; pass a template structure.")
        sym = ELEMENT_SYMBOLS[z]
        labels.append(sym if sym not in labels else f"{sym}{s}")
    return Structure(
        cell=jnp.asarray(xv.cell, dtype=jnp.float64),
        positions=jnp.asarray(xv.positions, dtype=jnp.float64),
        species_indices=tuple(dense[int(s)] for s in xv.species_index),
        atomic_numbers=tuple(int(abs(z)) for z in xv.atomic_numbers),
        species_labels=tuple(labels),
        initial_moments=jnp.zeros((na, 3), dtype=jnp.float64),
    )


def write_xv(path: str | Path, structure: Structure, velocities: Any | None = None) -> None:
    """Write a SIESTA ``.XV`` file.

    Theory:
        Positions and lattice vectors in Bohr; velocities (Bohr/fs) are zero for a static calculation. The
        species index written is ``species_indices + 1``, so the reading SIESTA input must list its
        ``ChemicalSpeciesLabel`` entries in the order of ``structure.species_labels``.

    Algorithm:
        SIESTA's formats ``(2(3x,3f18.9))`` and ``(i3,i6,3f18.9,3x,3f18.9)``.

    Parameters:
        path: File name.
        structure: Geometry to write.
        velocities: Optional ``(n_atoms, 3)`` velocities in Bohr/fs.

    Returns:
        ``None``.

    Invariants:
        ``read_xv`` recovers the geometry to the nine decimals of the format.

    SIESTA Mapping:
        ``ioxv('write', ...)`` in ``ioxv.F``.
    """
    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    vel = np.zeros_like(pos) if velocities is None else np.asarray(velocities, dtype=np.float64)
    if vel.shape != pos.shape:
        raise ValueError("velocities must have shape (n_atoms, 3).")
    out: list[str] = []
    for iv in range(3):
        a = cell[:, iv]
        out.append("   " + "".join(f"{x:18.9f}" for x in a) + "   " + "".join(f"{0.0:18.9f}" for _ in range(3)))
    out.append(f"{structure.n_atoms:12d}")
    for ia in range(structure.n_atoms):
        out.append(
            f"{structure.species_indices[ia] + 1:3d}{structure.atomic_numbers[ia]:6d}"
            + "".join(f"{x:18.9f}" for x in pos[ia])
            + "   "
            + "".join(f"{x:18.9f}" for x in vel[ia])
        )
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------
# .DM (Fortran unformatted sequential)
# --------------------------------------------------------------------------------------------------------------


class _RecordReader:
    """Walks the records of a Fortran unformatted sequential file (4-byte little-endian record markers)."""

    def __init__(self, raw: bytes, name: str) -> None:
        self._raw = raw
        self._pos = 0
        self._name = name

    def read(self, dtype: str) -> np.ndarray:
        if self._pos + 4 > len(self._raw):
            raise ValueError(f"{self._name}: unexpected end of file.")
        n = int(np.frombuffer(self._raw, dtype="<i4", count=1, offset=self._pos)[0])
        start = self._pos + 4
        if n < 0 or start + n + 4 > len(self._raw):
            raise ValueError(f"{self._name}: corrupt record at byte {self._pos}.")
        tail = int(np.frombuffer(self._raw, dtype="<i4", count=1, offset=start + n)[0])
        if tail != n:
            raise ValueError(f"{self._name}: record markers disagree at byte {self._pos}.")
        self._pos = start + n + 4
        return np.frombuffer(self._raw, dtype=dtype, count=n // np.dtype(dtype).itemsize, offset=start)


def _record(arr: np.ndarray) -> bytes:
    payload = arr.tobytes()
    marker = np.asarray([len(payload)], dtype="<i4").tobytes()
    return marker + payload + marker


def read_dm_file(path: str | Path) -> tuple[SiestaSparsity, np.ndarray]:
    """Read a SIESTA ``.DM`` file into its own sparsity pattern.

    Theory:
        Records: ``no_u, nspin, nsc(3)``; ``n_col(no_u)``; one record of column indices per row; then, per
        spin component, one record of values per row.

    Algorithm:
        The file is read at once and the records are walked with NumPy views. The pre-4.1 header without
        ``nsc`` is accepted when no column lies outside the unit cell.

    Parameters:
        path: File name.

    Returns:
        ``(sparsity, values)`` with ``values`` of shape ``(nspin, nnzs)`` in SIESTA's spin components.

    Invariants:
        Raises ``ValueError`` on inconsistent record lengths.

    SIESTA Mapping:
        ``read_dm`` in ``m_iodm.F90`` (``io_read_Sp``, ``io_read_d2D``).
    """
    rd = _RecordReader(Path(path).read_bytes(), str(path))
    head = rd.read("<i4")
    if head.shape[0] not in (2, 5):
        raise ValueError(f"{path}: unrecognised DM header ({head.shape[0]} integers).")
    no_u, nspin = int(head[0]), int(head[1])
    n_col = rd.read("<i4").astype(np.int32)
    if n_col.shape[0] != no_u:
        raise ValueError(f"{path}: n_col has {n_col.shape[0]} entries for {no_u} orbitals.")
    cols = [rd.read("<i4") for _ in range(no_u)]
    list_col = (np.concatenate(cols) if cols else np.zeros(0, dtype=np.int32)).astype(np.int64) - 1
    if head.shape[0] == 5:
        nsc = (int(head[2]), int(head[3]), int(head[4]))
    else:
        if list_col.size and int(np.max(list_col)) >= no_u:
            raise ValueError(f"{path}: old DM format without nsc and with supercell columns; cannot be mapped.")
        nsc = (1, 1, 1)
    values = np.zeros((nspin, list_col.shape[0]), dtype=np.float64)
    ptr = np.concatenate(([0], np.cumsum(n_col))).astype(np.int64)
    for s in range(nspin):
        for io in range(no_u):
            rec = rd.read("<f8")
            if rec.shape[0] != n_col[io]:
                raise ValueError(f"{path}: row {io + 1} of spin {s + 1} has {rec.shape[0]} values, not {n_col[io]}.")
            values[s, ptr[io] : ptr[io + 1]] = rec
    return SiestaSparsity(no_u, nsc, n_col, list_col), values


def write_dm_file(path: str | Path, sparsity: SiestaSparsity, values: np.ndarray) -> None:
    """Write a SIESTA ``.DM`` file from a matrix on a SIESTA-layout pattern.

    Theory:
        The record sequence of ``read_dm_file``.

    Algorithm:
        Records are assembled in memory and written at once.

    Parameters:
        path: File name.
        sparsity: Pattern in SIESTA's layout.
        values: ``(n_comp, nnzs)`` SIESTA spin components.

    Returns:
        ``None``.

    Invariants:
        Byte-identical to SIESTA's own file for the same pattern and values.

    SIESTA Mapping:
        ``write_dm`` in ``m_iodm.F90``.
    """
    vals = np.asarray(values, dtype="<f8")
    if vals.ndim != 2 or vals.shape[1] != sparsity.nnzs:
        raise ValueError("values must have shape (n_comp, nnzs).")
    n_col = np.asarray(sparsity.n_col, dtype="<i4")
    ptr = np.concatenate(([0], np.cumsum(n_col))).astype(np.int64)
    cols = (np.asarray(sparsity.list_col, dtype=np.int64) + 1).astype("<i4")
    chunks = [_record(np.asarray([sparsity.no_u, vals.shape[0], *sparsity.nsc], dtype="<i4")), _record(n_col)]
    chunks.extend(_record(cols[ptr[io] : ptr[io + 1]]) for io in range(sparsity.no_u))
    for s in range(vals.shape[0]):
        chunks.extend(_record(vals[s, ptr[io] : ptr[io + 1]]) for io in range(sparsity.no_u))
    Path(path).write_bytes(b"".join(chunks))


def write_dm(path: str | Path, pattern: SparsePattern, dm: Any, nspin: int) -> None:
    """Write a jsiesta density matrix as a SIESTA ``.DM`` file.

    Theory:
        The file SIESTA reads with ``DM.UseSaveDM T`` (or ``File.DM.Init``). SIESTA re-indexes ``nsc`` and
        restructures the matrix to its own sparsity pattern, so the patterns need not coincide; the orbitals
        (number and order per atom) must.

    Algorithm:
        ``pattern_to_siesta`` + ``dm_to_siesta`` + ``write_dm_file``.

    Parameters:
        path: File name (``<SystemLabel>.DM`` for SIESTA to find it).
        pattern: Pattern of ``dm``.
        dm: jsiesta density matrix ``(n_spin_mat, n_edges)``.
        nspin: Spin mode ``1, 2, 4, 8``.

    Returns:
        ``None``.

    Invariants:
        ``read_dm`` on the same pattern returns ``dm`` (Hermitian part for the spin-orbit blocks).

    SIESTA Mapping:
        ``write_dm`` in ``m_iodm.F90``.
    """
    sparsity, order = pattern_to_siesta(pattern)
    write_dm_file(path, sparsity, dm_to_siesta(dm, pattern, nspin)[:, order])


@overload
def read_dm(path: str | Path, pattern: SparsePattern, nspin: int, return_info: Literal[False] = False) -> jax.Array: ...


@overload
def read_dm(
    path: str | Path, pattern: SparsePattern, nspin: int, return_info: Literal[True]
) -> tuple[jax.Array, MappingInfo]: ...


def read_dm(
    path: str | Path, pattern: SparsePattern, nspin: int, return_info: bool = False
) -> jax.Array | tuple[jax.Array, MappingInfo]:
    """Read a SIESTA ``.DM`` file onto a jsiesta pattern, as an initial density matrix.

    Theory:
        Elements are matched by ``(row orbital, column orbital, lattice offset)``; a file without auxiliary
        supercell is unfolded by replication (``map_siesta_to_edges``). A file with another number of spin
        components is converted with ``convert_spin``. The result need not integrate to the electron count if
        the patterns differ; ``run_scf`` rescales it.

    Algorithm:
        ``read_dm_file`` + ``map_siesta_to_edges`` + ``dm_from_siesta`` + ``convert_spin``.

    Parameters:
        path: ``.DM`` file.
        pattern: Pattern of the calculation.
        nspin: Spin mode of the calculation.
        return_info: Also return the ``MappingInfo`` of the match.

    Returns:
        Density matrix ``(n_spin_mat, n_edges)`` (complex for ``nspin`` 4 and 8), optionally with the info.

    Invariants:
        Raises ``ValueError`` if the number of orbitals differs.

    SIESTA Mapping:
        ``init_DM_file`` in ``m_new_dm.F90``.
    """
    sparsity, values = read_dm_file(path)
    edge_vals, info = map_siesta_to_edges(sparsity, pattern, values, folded="replicate")
    dm_file, nspin_file = dm_from_siesta(edge_vals)
    dm = jnp.asarray(convert_spin(dm_file, nspin_file, nspin, "dm"))
    return (dm, info) if return_info else dm


# --------------------------------------------------------------------------------------------------------------
# .nc (NetCDF-4, CDF.Save)
# --------------------------------------------------------------------------------------------------------------


def _netcdf4() -> Any:
    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading or writing SIESTA .nc files needs the 'netCDF4' package (pip install 'jsiesta[netcdf]')."
        ) from exc
    return netCDF4


@dataclass(frozen=True)
class SiestaNCData:
    """Contents of a SIESTA ``<label>.nc`` file, in the file's units and spin components.

    Theory:
        Geometry in Bohr, ``H``, ``EDM`` and the Fermi level in Ry; matrices on the file's own pattern.

    Algorithm:
        Plain NumPy container. ``cell`` holds the lattice vectors as columns. Matrices absent from the file
        are ``None``.

    Invariants:
        Every matrix has ``sparsity.nnzs`` columns.

    SIESTA Mapping:
        The root variables and the ``SPARSE`` group written by ``cdf_init_file`` / ``cdf_save_state``.
    """

    cell: np.ndarray
    positions: np.ndarray
    lasto: np.ndarray
    sparsity: SiestaSparsity
    overlap: np.ndarray | None
    dm: np.ndarray | None
    edm: np.ndarray | None
    h: np.ndarray | None
    fermi_level_ry: float | None
    q_tot: float | None


def read_siesta_nc(path: str | Path) -> SiestaNCData:
    """Read geometry, sparsity pattern and sparse matrices from a SIESTA ``<label>.nc`` file.

    Theory:
        NetCDF stores SIESTA's Fortran arrays with reversed index order: ``DM(nnzs, spin)`` appears as
        ``(spin, nnzs)``, ``xa(xyz, na_u)`` as ``(na_u, xyz)`` and ``cell(xyz, ivec)`` as ``(ivec, xyz)``.

    Algorithm:
        Reads the root variables and the ``SPARSE`` group; ``isc_off`` is taken from the file when present.

    Parameters:
        path: File name.

    Returns:
        ``SiestaNCData``.

    Invariants:
        Raises ``ValueError`` if the file has no ``SPARSE`` group.

    SIESTA Mapping:
        ``cdf_r_Sp`` and ``cdf_r_d2D`` in ``ncdf_io.F90``.
    """
    nc4 = _netcdf4()
    with nc4.Dataset(str(path), "r") as ds:
        ds.set_auto_mask(False)
        if "SPARSE" not in ds.groups:
            raise ValueError(f"{path}: no SPARSE group; not a SIESTA CDF.Save file.")
        sp = ds.groups["SPARSE"]
        nsc_arr = np.asarray(ds.variables["nsc"][:], dtype=np.int64)
        nsc = (int(nsc_arr[0]), int(nsc_arr[1]), int(nsc_arr[2]))
        n_col = np.asarray(sp.variables["n_col"][:], dtype=np.int32)
        list_col = np.asarray(sp.variables["list_col"][:], dtype=np.int64) - 1
        kwargs: dict[str, Any] = {}
        if "isc_off" in sp.variables:
            isc_off = np.asarray(sp.variables["isc_off"][:], dtype=np.int64)
            # A file whose isc_off was never filled holds zeros (or the fill value); fall back to the standard order.
            if isc_off.shape == (int(np.prod(nsc)), 3) and len({tuple(r) for r in isc_off.tolist()}) == isc_off.shape[0]:
                kwargs["isc_off"] = isc_off
        sparsity = SiestaSparsity(int(n_col.shape[0]), nsc, n_col, list_col, **kwargs)

        def _mat(name: str) -> np.ndarray | None:
            if name not in sp.variables:
                return None
            arr = np.asarray(sp.variables[name][:], dtype=np.float64)
            return arr.reshape(-1, sparsity.nnzs)

        ef = float(np.asarray(ds.variables["Ef"][:]).ravel()[0]) if "Ef" in ds.variables else None
        q_tot = float(np.asarray(ds.variables["Qtot"][:]).ravel()[0]) if "Qtot" in ds.variables else None
        return SiestaNCData(
            cell=np.asarray(ds.variables["cell"][:], dtype=np.float64).T.copy(),
            positions=np.asarray(ds.variables["xa"][:], dtype=np.float64),
            lasto=np.asarray(ds.variables["lasto"][:], dtype=np.int64),
            sparsity=sparsity,
            overlap=_mat("S"),
            dm=_mat("DM"),
            edm=_mat("EDM"),
            h=_mat("H"),
            fermi_level_ry=ef,
            q_tot=q_tot,
        )


def structure_from_nc(path: str | Path, template: Structure) -> Structure:
    """Take cell and positions from a SIESTA ``.nc`` file, species data from ``template``.

    Theory:
        The file has the geometry the matrices belong to, but no species labels outside its optional
        ``BASIS`` group.

    Algorithm:
        ``read_siesta_nc`` and a copy of the template with the geometry replaced.

    Parameters:
        path: ``.nc`` file.
        template: Structure whose species, moments and periodicity are kept.

    Returns:
        ``Structure`` at the file's geometry.

    Invariants:
        Raises ``ValueError`` if the number of atoms differs.

    SIESTA Mapping:
        ``xa`` and ``cell`` of ``cdf_save_state``.
    """
    data = read_siesta_nc(path)
    if data.positions.shape[0] != template.n_atoms:
        raise ValueError(f"{path}: {data.positions.shape[0]} atoms, the input structure has {template.n_atoms}.")
    return Structure(
        cell=jnp.asarray(data.cell, dtype=jnp.float64),
        positions=jnp.asarray(data.positions, dtype=jnp.float64),
        species_indices=template.species_indices,
        atomic_numbers=template.atomic_numbers,
        species_labels=template.species_labels,
        initial_moments=template.initial_moments,
        pbc=template.pbc,
    )


def read_nc_restart(
    path: str | Path, pattern: SparsePattern, nspin: int
) -> tuple[jax.Array | None, jax.Array | None, float | None]:
    r"""Initial density matrix and Hamiltonian for ``run_scf`` from a SIESTA ``.nc`` file.

    Theory:
        ``H`` (Ry, SIESTA spin components) becomes the jsiesta Hamiltonian in Hartree, ``DM`` the density
        matrix. Matrices folded over lattice images (file written without auxiliary supercell) are unfolded
        with ``"replicate"`` for ``DM`` and ``"first"`` for ``H``; the latter reproduces :math:`H(\Gamma)`
        only, so such a Hamiltonian is a valid start for a :math:`\Gamma`-only calculation.

    Algorithm:
        ``read_siesta_nc`` + ``map_siesta_to_edges`` + spin conversion.

    Parameters:
        path: ``.nc`` file.
        pattern: Pattern of the calculation.
        nspin: Spin mode of the calculation.

    Returns:
        ``(dm, h, fermi_level_ha)``; each is ``None`` if the file does not have it.

    Invariants:
        Raises ``ValueError`` if the number of orbitals differs.

    SIESTA Mapping:
        No counterpart: SIESTA restarts from the density matrix only.
    """
    data = read_siesta_nc(path)
    dm: jax.Array | None = None
    h: jax.Array | None = None
    if data.dm is not None:
        vals, _ = map_siesta_to_edges(data.sparsity, pattern, data.dm, folded="replicate")
        dm_file, n_file = dm_from_siesta(vals)
        dm = jnp.asarray(convert_spin(dm_file, n_file, nspin, "dm"))
    if data.h is not None:
        vals, _ = map_siesta_to_edges(data.sparsity, pattern, data.h, folded="first")
        h_file, n_file = h_from_siesta(vals, pattern)
        h = jnp.asarray(convert_spin(h_file, n_file, nspin, "h"))
    ef = None if data.fermi_level_ry is None else data.fermi_level_ry * RY_TO_HARTREE
    return dm, h, ef


def write_siesta_nc(
    path: str | Path,
    structure: Structure,
    basis: BasisData,
    pattern: SparsePattern,
    nspin: int,
    overlap: Any,
    h: Any | None = None,
    dm: Any | None = None,
    edm: Any | None = None,
    fermi_level_ha: float | None = None,
    forces_ha_per_bohr: Any | None = None,
    stress_ha_per_bohr3: Any | None = None,
    config: SCFConfig | None = None,
    k_grid: tuple[int, int, int] | None = None,
    label: str = "jsiesta",
) -> None:
    """Write a SIESTA-layout ``<label>.nc`` file (the ``CDF.Save T`` file) with ``S``, ``H``, ``DM`` and ``EDM``.

    Theory:
        Same dimensions, variable names, index order and units as SIESTA: geometry in Bohr, ``H``, ``EDM`` and
        ``Ef`` in Ry, forces in Ry/Bohr, stress in Ry/Bohr^3, 1-based column indices, ``isc_off`` in SIESTA's
        supercell order. The ``BASIS`` group has the species of every atom and each species' header
        (label, atomic number, mass, number of orbitals), which is what readers such as sisl need to build
        the orbital list; SIESTA's radial tables and its ``GRID`` group are not written.

    Algorithm:
        ``pattern_to_siesta`` for the layout; ``h_to_siesta`` / ``dm_to_siesta`` for the spin components.

    Parameters:
        path: File name.
        structure: Geometry.
        basis: Basis (for ``lasto`` and the electron count).
        pattern: Pattern of the matrices.
        nspin: Spin mode.
        overlap: Overlap matrix ``(n_edges,)``.
        h: Hamiltonian in Hartree, optional.
        dm: Density matrix, optional.
        edm: Energy-density matrix in Hartree, optional.
        fermi_level_ha: Fermi level in Hartree, optional.
        forces_ha_per_bohr: Forces ``(n_atoms, 3)``, optional.
        stress_ha_per_bohr3: Stress ``(3, 3)``, optional.
        config: SCF configuration, for the ``SETTINGS`` group.
        k_grid: Monkhorst-Pack divisions, for ``SETTINGS/BZ``.
        label: Value of the ``label`` attribute.

    Returns:
        ``None``.

    Invariants:
        ``read_siesta_nc`` + ``read_nc_restart`` recover ``h`` and ``dm`` on the same pattern.

    SIESTA Mapping:
        ``cdf_init_file`` and ``cdf_save_state`` in ``m_ncdf_siesta.F90``.
    """
    nc4 = _netcdf4()
    sparsity, order = pattern_to_siesta(pattern)
    n_edm = min(nspin, 4)
    q_tot = float(basis.total_valence_electrons) - (float(config.net_charge) if config is not None else 0.0)
    with nc4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("na_u", structure.n_atoms)
        ds.createDimension("no_u", sparsity.no_u)
        ds.createDimension("no_s", sparsity.no_u * int(np.prod(sparsity.nsc)))
        ds.createDimension("spin", nspin)
        ds.createDimension("xyz", 3)
        ds.createDimension("n_s", int(np.prod(sparsity.nsc)))
        ds.createDimension("one", 1)

        def _var(grp: Any, name: str, kind: str, dims: tuple[str, ...], info: str, unit: str | None = None) -> Any:
            v = grp.createVariable(name, kind, dims)
            v.info = info
            if unit is not None:
                v.unit = unit
            return v

        _var(ds, "lasto", "i4", ("na_u",), "Last orbital of equivalent atom")[:] = np.asarray(
            basis.atom_orb_offsets[1:], dtype=np.int32
        )
        _var(ds, "Qtot", "f8", ("one",), "Total charge")[:] = q_tot
        _var(ds, "nsc", "i4", ("xyz",), "Number of supercells in each unit-cell direction")[:] = np.asarray(
            sparsity.nsc, dtype=np.int32
        )
        v_ef = _var(ds, "Ef", "f8", ("one",), "Fermi level", "Ry")
        if fermi_level_ha is not None:
            v_ef[:] = float(fermi_level_ha) * HARTREE_TO_RY
        _var(ds, "xa", "f8", ("na_u", "xyz"), "Atomic coordinates", "Bohr")[:] = np.asarray(
            structure.positions, dtype=np.float64
        )
        _var(ds, "cell", "f8", ("xyz", "xyz"), "Unit cell", "Bohr")[:] = np.asarray(structure.cell, dtype=np.float64).T
        v_fa = _var(ds, "fa", "f8", ("na_u", "xyz"), "Atomic forces", "Ry/Bohr")
        if forces_ha_per_bohr is not None:
            v_fa[:] = HARTREE_TO_RY * np.asarray(forces_ha_per_bohr, dtype=np.float64)
        v_st = _var(ds, "stress", "f8", ("xyz", "xyz"), "Cell stress", "Ry/Bohr**3")
        if stress_ha_per_bohr3 is not None:
            v_st[:] = HARTREE_TO_RY * np.asarray(stress_ha_per_bohr3, dtype=np.float64).T

        sp = ds.createGroup("SPARSE")
        sp.createDimension("nnzs", sparsity.nnzs)
        sp.createDimension("spin_EDM", n_edm)
        _var(sp, "isc_off", "i4", ("n_s", "xyz"), "Index of supercell coordinates")[:] = sparsity.isc_off.astype(np.int32)
        _var(sp, "n_col", "i4", ("no_u",), "Number of non-zero elements per row")[:] = sparsity.n_col
        _var(sp, "list_col", "i4", ("nnzs",), "Supercell column indices in the sparse format")[:] = (
            sparsity.list_col + 1
        ).astype(np.int32)
        _var(sp, "S", "f8", ("nnzs",), "Overlap matrix")[:] = np.asarray(overlap, dtype=np.float64)[order]
        if dm is not None:
            _var(sp, "DM", "f8", ("spin", "nnzs"), "Density matrix")[:] = dm_to_siesta(dm, pattern, nspin)[:, order]
        if edm is not None:
            _var(sp, "EDM", "f8", ("spin_EDM", "nnzs"), "Energy density matrix", "Ry")[:] = (
                HARTREE_TO_RY * dm_to_siesta(edm, pattern, nspin, n_comp=n_edm)[:, order]
            )
        if h is not None:
            _var(sp, "H", "f8", ("spin", "nnzs"), "Hamiltonian", "Ry")[:] = h_to_siesta(h, pattern, nspin)[:, order]

        st = ds.createGroup("SETTINGS")
        if config is not None:
            _var(st, "DMTolerance", "f8", ("one",), "Tolerance for converging the density matrix")[:] = config.d_tol
            _var(st, "HTolerance", "f8", ("one",), "Tolerance for converging the Hamiltonian", "Ry")[:] = (
                config.h_tol_ha * HARTREE_TO_RY
            )
            _var(st, "NetCharge", "f8", ("one",), "Net charge of the system")[:] = config.net_charge
            _var(st, "ElectronicTemperature", "f8", ("one",), "Temperature for electrons", "Ry")[:] = (
                config.electronic_temp_ha * HARTREE_TO_RY
            )
            _var(st, "MeshCutoff", "f8", ("one",), "Mesh cutoff for real space grid", "Ry")[:] = config.mesh_cutoff_ry
        if k_grid is not None:
            _var(st, "BZ", "i4", ("xyz", "xyz"), "Grid used for the Brillouin zone integration")[:] = np.diag(
                np.asarray(k_grid, dtype=np.int32)
            )
            _var(st, "BZ_displ", "f8", ("xyz",), "Grid displacement used in Brillouin zone", "b**-1")[:] = 0.0

        bs = ds.createGroup("BASIS")
        _var(bs, "basis", "i4", ("na_u",), "Basis of each atom by ID")[:] = (
            np.asarray(structure.species_indices, dtype=np.int32) + 1
        )
        offsets = np.asarray(basis.atom_orb_offsets, dtype=np.int64)
        for isp, lbl in enumerate(structure.species_labels):
            ia = structure.species_indices.index(isp)
            spp = basis.species[isp]
            grp = bs.createGroup(lbl)
            grp.Element = spp.symbol
            grp.Label = lbl
            grp.Atomic_number = np.int32(spp.atomic_number)
            grp.Valence_charge = float(spp.valence_charge)
            grp.Mass = float(spp.mass_amu)
            grp.Number_of_orbitals = np.int32(offsets[ia + 1] - offsets[ia])
            grp.ID = np.int32(isp + 1)

        ds.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.setncattr("name", "jsiesta")
        ds.label = label
        ds.method = "diagon"
        ds.generator = "jsiesta"


# --------------------------------------------------------------------------------------------------------------
# One call for a finished SCF
# --------------------------------------------------------------------------------------------------------------


def normalize_dm(dm: jax.Array, overlap: jax.Array, n_electrons: float, nspin: int) -> jax.Array:
    r"""Scale a density matrix so that :math:`\mathrm{Tr}[D S] = N_e`.

    Theory:
        A density matrix read on another sparsity pattern, or for a slightly different geometry, no longer
        holds the right charge. A uniform scale restores it, as SIESTA does on reading (``normalize_dm``).

    Algorithm:
        One edge sum; no change if the trace is not positive.

    Parameters:
        dm: Density matrix ``(n_spin_mat, n_edges)``.
        overlap: Overlap ``(n_edges,)``.
        n_electrons: Target electron count.
        nspin: Spin mode.

    Returns:
        Scaled density matrix.

    Invariants:
        ``Tr[D S] == n_electrons`` afterwards (when the input trace is positive).

    SIESTA Mapping:
        ``normalize_dm`` in ``normalize_dm.F90``.
    """
    diag = jnp.real(dm[0]) if nspin == 1 else jnp.real(dm[0]) + jnp.real(dm[1])
    q = jnp.sum(diag * overlap)
    scale = jnp.where(q > 0.0, n_electrons / jnp.where(q > 0.0, q, 1.0), 1.0)
    return dm * scale.astype(dm.dtype)


def write_restart_files(
    directory: str | Path,
    label: str,
    structure: Structure,
    basis: BasisData,
    pattern: SparsePattern,
    overlap: Any,
    state: SCFState,
    config: SCFConfig,
    forces_ha_per_bohr: Any | None = None,
    stress_ha_per_bohr3: Any | None = None,
    k_grid: tuple[int, int, int] | None = None,
    formats: tuple[str, ...] = ("XV", "DM", "nc"),
) -> dict[str, Path]:
    """Write ``<label>.XV``, ``<label>.DM`` and ``<label>.nc`` for a finished SCF.

    Theory:
        The density matrix written is ``DM_out`` of the last iteration and the Hamiltonian the one that
        produced it (``state.h_sparse``), the pair SIESTA leaves on disk.

    Algorithm:
        Calls ``write_xv``, ``write_dm`` and ``write_siesta_nc``.

    Parameters:
        directory: Output directory (created if missing).
        label: SIESTA ``SystemLabel``.
        structure: Geometry.
        basis: Basis.
        pattern: Pattern of the matrices.
        overlap: Overlap ``(n_edges,)``.
        state: Final ``SCFState``.
        config: SCF configuration.
        forces_ha_per_bohr: Optional forces for the ``.nc`` file.
        stress_ha_per_bohr3: Optional stress for the ``.nc`` file.
        k_grid: Optional Monkhorst-Pack divisions for the ``.nc`` file.
        formats: Subset of ``("XV", "DM", "nc")``.

    Returns:
        Mapping from format to the path written.

    Invariants:
        Raises ``ValueError`` for an unknown format.

    SIESTA Mapping:
        The end-of-run output of ``siesta_end`` / ``state_analysis`` (``ioxv``, ``write_dm``, ``cdf_save_state``).
    """
    unknown = set(formats) - {"XV", "DM", "nc"}
    if unknown:
        raise ValueError(f"Unknown restart formats: {sorted(unknown)}.")
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "XV" in formats:
        written["XV"] = out_dir / f"{label}.XV"
        write_xv(written["XV"], structure)
    if "DM" in formats:
        written["DM"] = out_dir / f"{label}.DM"
        write_dm(written["DM"], pattern, state.dm_out, config.nspin)
    if "nc" in formats:
        written["nc"] = out_dir / f"{label}.nc"
        write_siesta_nc(
            written["nc"],
            structure,
            basis,
            pattern,
            config.nspin,
            overlap,
            h=state.h_sparse,
            dm=state.dm_out,
            edm=state.edm_out,
            fermi_level_ha=float(state.fermi_level_ha),
            forces_ha_per_bohr=forces_ha_per_bohr,
            stress_ha_per_bohr3=stress_ha_per_bohr3,
            config=config,
            k_grid=k_grid,
            label=label,
        )
    return written
