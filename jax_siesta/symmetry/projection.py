r"""Symmetry-closed sparsity, Fibonacci-sphere Wigner-D, SU(2) spinors, and matrix projection.

Theory:
    Closes the sparse NAO CSR graph under all space/magnetic group operations :math:`S \in G`
    and precompiles the exact linear matrix projection operator

    .. math::
        \mathcal{P}_{\text{sym}}(M)_{\mu'\nu'}(\mathbf{T}')
        = \frac{1}{|G|} \sum_{S \in G} \sum_{\mu\nu}
          D^{l(\mu)}_{m(\mu'), m(\mu)}(S)\, D^{l(\nu)}_{m(\nu'), m(\nu)}(S)\,
          \mathcal{S}_S\!\left[M_{\mu\nu}(\mathbf{T})\right],

    where :math:`D^l(S)` are real spherical-harmonic Wigner rotation matrices and
    :math:`\mathcal{S}_S` handles collinear spin swaps (``nspin=2``) or full :math:`SU(2)`
    spinor + time-reversal conjugation (``nspin=4, 8``) using ``transpose_edge_idx``.

Algorithm:
    1. Computes real-harmonic Wigner-:math:`D^l` matrices on a 16-point Fibonacci sphere
       via least squares and asserts :math:`\|D^l (D^l)^T - I\|_\infty \le 10^{-10}` at startup.
    2. Expands ``build_sparse_pattern`` edges and supercell shifts under all :math:`S \in G`
       so every orbit is closed with zero missing partners.
    3. Aggregates all operator contributions into a single compacted sparse table (rows padded to
       equal length) executed in JAX as one gather and row sum.

Invariants:
    - Idempotency: :math:`\mathcal{P}_{\text{sym}}^2 = \mathcal{P}_{\text{sym}}`.
    - Trace / electron-count conservation: :math:`\mathrm{Tr}[\mathcal{P}_{\text{sym}}(D) S] = \mathrm{Tr}[D S]`.
    - Sequential ``(atom, l, m, iz)`` shell indexing verified before table compilation.

SIESTA Mapping:
    Replaces and generalizes ``m_dm_symmetrize.F90`` and symmetry edge closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import BasisData, SparsePattern, Structure
from jax_siesta.species.psml import build_sparse_pattern
from jax_siesta.symmetry.spacegroup import SpaceGroupData


def _fibonacci_sphere_points(n_pts: int = 16) -> np.ndarray:
    """Generate ``n_pts`` quasi-uniform unit vectors on :math:`S^2` via golden-angle spiral."""
    idx = np.arange(n_pts, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * idx / float(n_pts)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = math.pi * (3.0 - math.sqrt(5.0)) * idx
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _real_solid_harmonics_unit(pts: np.ndarray, l_max: int) -> np.ndarray:
    """Evaluate SIESTA-convention orthonormal real spherical harmonics up to ``l_max`` on unit vectors."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r2 = x * x + y * y + z * z
    n_p = pts.shape[0]
    a_tbl = np.zeros((n_p, l_max + 1, l_max + 1), dtype=np.float64)
    b_tbl = np.zeros((n_p, l_max + 1, l_max + 1), dtype=np.float64)
    a_tbl[:, 0, 0] = 1.0
    for ang_l in range(l_max):
        f = float(2 * ang_l + 1)
        a_tbl[:, ang_l + 1, ang_l + 1] = f * (x * a_tbl[:, ang_l, ang_l] - y * b_tbl[:, ang_l, ang_l])
        b_tbl[:, ang_l + 1, ang_l + 1] = f * (x * b_tbl[:, ang_l, ang_l] + y * a_tbl[:, ang_l, ang_l])
        a_tbl[:, ang_l + 1, ang_l] = f * z * a_tbl[:, ang_l, ang_l]
        b_tbl[:, ang_l + 1, ang_l] = f * z * b_tbl[:, ang_l, ang_l]
        for m in range(ang_l):
            c1 = float(2 * ang_l + 1) / float(ang_l - m + 1)
            c2 = float(ang_l + m) / float(ang_l - m + 1)
            a_tbl[:, ang_l + 1, m] = c1 * z * a_tbl[:, ang_l, m] - c2 * r2 * a_tbl[:, ang_l - 1, m]
            b_tbl[:, ang_l + 1, m] = c1 * z * b_tbl[:, ang_l, m] - c2 * r2 * b_tbl[:, ang_l - 1, m]

    out = np.zeros((n_p, (l_max + 1) ** 2), dtype=np.float64)
    for ang_l in range(l_max + 1):
        for m in range(-ang_l, ang_l + 1):
            am = abs(m)
            norm = math.sqrt(
                ((2 * ang_l + 1) / (4.0 * math.pi))
                * (math.factorial(ang_l - am) / math.factorial(ang_l + am))
                * (1.0 if m == 0 else 2.0)
            )
            out[:, ang_l * ang_l + m + ang_l] = norm * (b_tbl[:, ang_l, am] if m < 0 else a_tbl[:, ang_l, am])
    return out


def compute_wigner_d_matrices(
    rotations_cart: np.ndarray,
    l_max: int,
    n_fib_pts: int = 16,
    orth_tol: float = 1e-10,
) -> dict[int, np.ndarray]:
    r"""Compute real-harmonic Wigner-:math:`D^l(S)` matrices via 16-pt Fibonacci least squares.

    Theory:
        For each Cartesian operation :math:`\mathbf{C}_S \in O(3)`, rotated orbitals satisfy
        :math:`Y_{lm}(\mathbf{C}_S^T \hat{\mathbf{u}}) = \sum_{m'} Y_{lm'}(\hat{\mathbf{u}}) D^l_{m',m}(S)`,
        so right-hand matrix multiplication guarantees :math:`D^l(S_1 S_2) = D^l(S_1) D^l(S_2)`.

    Algorithm:
        Solves ``np.linalg.lstsq(Y_src, Y_rot)`` on ``n_fib_pts`` unit-sphere nodes and
        verifies orthogonality :math:`\|D^l (D^l)^T - I\|_\infty \le 10^{-10}` at startup.

    Parameters:
        rotations_cart: Cartesian orthogonal matrices of shape ``(n_ops, 3, 3)``.
        l_max: Maximum orbital angular momentum quantum number.
        n_fib_pts: Number of Fibonacci-sphere sample points (default ``16``).
        orth_tol: Startup orthogonality assertion threshold (default ``1e-10``).

    Returns:
        Dictionary mapping angular momentum ``l`` to array of shape ``(n_ops, 2*l+1, 2*l+1)``.

    Invariants:
        - Asserts ``max |D^l (D^l)^T - I| <= orth_tol`` for every ``(s, l)``.

    SIESTA Mapping:
        Replaces analytical Euler-angle Wigner-D construction in symmetry projection.
    """
    n_ops = rotations_cart.shape[0]
    pts = _fibonacci_sphere_points(n_fib_pts)
    y_all = _real_solid_harmonics_unit(pts, l_max)

    d_mats: dict[int, np.ndarray] = {}
    for ang_l in range(l_max + 1):
        dim = 2 * ang_l + 1
        d_arr = np.empty((n_ops, dim, dim), dtype=np.float64)
        ys = y_all[:, ang_l * ang_l : (ang_l + 1) ** 2]
        eye_dim = np.eye(dim, dtype=np.float64)
        for s in range(n_ops):
            # Active orbital transform: Y_lm(C_S^T u) = (pts @ C_S) evaluated in Y_lm
            yr = _real_solid_harmonics_unit(pts @ rotations_cart[s], ang_l)[:, ang_l * ang_l : (ang_l + 1) ** 2]
            d_l, *_ = np.linalg.lstsq(ys, yr, rcond=None)
            err = float(np.max(np.abs(d_l @ d_l.T - eye_dim)))
            if err > orth_tol:
                raise AssertionError(
                    f"Wigner-D orthogonality check failed at op={s}, l={ang_l}: err={err:.3e} > {orth_tol:.3e}"
                )
            d_arr[s] = d_l
        d_mats[ang_l] = d_arr
    return d_mats


def su2_spinor_matrix(rot_cart: np.ndarray, time_reversal: bool) -> np.ndarray:
    r"""Compute :math:`2\times 2` spin-1/2 :math:`SU(2)` matrix for :math:`(\mathbf{C}_S, \theta_S)`.

    Theory:
        Extracts the proper rotation :math:`\mathbf{R} = (\det \mathbf{C}_S) \mathbf{C}_S \in SO(3)`
        via Shepperd's singularity-free quaternion method, forms
        :math:`U(\mathbf{R}) = w I_2 - i (x\sigma_x + y\sigma_y + z\sigma_z)`, and multiplies
        by :math:`i\sigma_y` when time reversal (:math:`\theta_S=1`) is active (leaving complex
        conjugation :math:`M \mapsto M^*` explicit in the projection weights).

    Algorithm:
        Branch-safe Shepperd 4-case trace/diagonal quaternion extraction.

    Parameters:
        rot_cart: Orthogonal :math:`3\times 3` Cartesian rotation matrix.
        time_reversal: Whether operation includes time reversal :math:`\mathcal{T}`.

    Returns:
        Complex ``(2, 2)`` unitary spin transformation matrix.

    Invariants:
        - Unitary: :math:`U U^\dagger = I_2` to ``1e-13``.
        - Covariance: :math:`U (\mathbf{m}\cdot\boldsymbol{\sigma}) U^\dagger = ((\det C) C \mathbf{m})\cdot\boldsymbol{\sigma}`.

    SIESTA Mapping:
        Provides spin-1/2 representation for NC/SOC symmetry projection.
    """
    det_c = float(np.linalg.det(rot_cart))
    r = (1.0 if det_c >= 0.0 else -1.0) * rot_cart
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + tr))
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] >= r[2, 2]:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    u = w * np.eye(2, dtype=np.complex128) - 1j * (x * sx + y * sy + z * sz)
    if time_reversal:
        # T = i * sigma_y * K -> return u @ (i * sigma_y), caller conjugates M
        u = u @ (1j * sy)
    return u


def _validate_sequential_shells(
    basis: BasisData,
) -> tuple[list[tuple[int, int, int, int]], dict[tuple[int, int, int], int]]:
    """Verify sequential ``(atom, l, m, iz)`` layout and return ``(shells, atom_shell_lookup)``.

    Each entry in ``shells`` is ``(atom, l, iz, orb_start)`` of contiguous length ``2*l + 1``.
    """
    shells: list[tuple[int, int, int, int]] = []
    lookup: dict[tuple[int, int, int], int] = {}
    mu = 0
    while mu < basis.n_orb:
        ia = int(basis.orb_atom[mu])
        ang_l = int(basis.orb_l[mu])
        iz = int(basis.orb_iz[mu])
        dim = 2 * ang_l + 1
        if mu + dim > basis.n_orb:
            raise ValueError(f"Truncated angular shell at orbital {mu}.")
        for off, m_exp in enumerate(range(-ang_l, ang_l + 1)):
            if (
                int(basis.orb_atom[mu + off]) != ia
                or int(basis.orb_l[mu + off]) != ang_l
                or int(basis.orb_iz[mu + off]) != iz
                or int(basis.orb_m[mu + off]) != m_exp
            ):
                raise ValueError(
                    f"Non-sequential (atom, l, m, iz) shell ordering around orbital {mu + off}."
                )
        key = (ia, ang_l, iz)
        if key in lookup:
            raise ValueError(f"Duplicate (atom, l, iz) key {key} in basis!")
        sh_idx = len(shells)
        lookup[key] = sh_idx
        shells.append((ia, ang_l, iz, mu))
        mu += dim
    return shells, lookup


def _shell_arrays(
    basis: BasisData, sg: SpaceGroupData
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shells of ``basis`` and, per shell, ``(atom, l, first orbital)`` plus the orbital-to-shell map and, per operation
    and shell, the image shell (same ``l`` and ``iz`` on the image atom): ``(shells, sh_atom, sh_l, sh_mu0, orb_to_sh,
    sh_map)``."""
    shells, lookup = _validate_sequential_shells(basis)
    sh_atom = np.asarray([sh[0] for sh in shells], dtype=np.int64)
    sh_l = np.asarray([sh[1] for sh in shells], dtype=np.int64)
    sh_mu0 = np.asarray([sh[3] for sh in shells], dtype=np.int64)
    orb_to_sh = np.empty(basis.n_orb, dtype=np.int64)
    for sh_i, (_, ang_l, _, mu0) in enumerate(shells):
        orb_to_sh[mu0 : mu0 + 2 * ang_l + 1] = sh_i
    sh_map = np.empty((sg.n_ops, len(shells)), dtype=np.int64)
    for s in range(sg.n_ops):
        for sh_i, (ia, ang_l, iz, _) in enumerate(shells):
            sh_map[s, sh_i] = lookup[(int(sg.atom_map[s, ia]), ang_l, iz)]
    return shells, sh_atom, sh_l, sh_mu0, orb_to_sh, sh_map


def _image_shifts(sg: SpaceGroupData, s: int, t: np.ndarray, atom1: np.ndarray, atom2: np.ndarray) -> np.ndarray:
    """Cell shifts :math:`W_S \\mathbf{T} + \\mathbf{n}_{S,J} - \\mathbf{n}_{S,I}` of shell pairs (one row each) under
    operation ``s`` (integer arithmetic)."""
    rot = np.asarray(sg.rotations_frac[s], dtype=np.int64)
    atr = np.asarray(sg.atom_translation[s], dtype=np.int64)
    return t @ rot.T + atr[atom2] - atr[atom1]


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    """``np.unique(rows, axis=0)`` for small-range integer rows, through one int64 code per row (much faster)."""
    if rows.shape[0] == 0:
        return rows
    lo = rows.min(axis=0)
    span = rows.max(axis=0) - lo + 1
    if float(np.prod(span.astype(np.float64))) >= 2.0**62:
        return np.unique(rows, axis=0)
    code = np.zeros(rows.shape[0], dtype=np.int64)
    for col in range(rows.shape[1]):
        code = code * int(span[col]) + (rows[:, col] - lo[col])
    _, first = np.unique(code, return_index=True)
    return rows[first]


def _encode_rows(rows: np.ndarray, base: int) -> np.ndarray:
    """Encode non-negative integer rows as one int64 per row, lexicographically ordered (``base`` > every entry)."""
    code = np.zeros(rows.shape[0], dtype=np.int64)
    for col in range(rows.shape[1]):
        code = code * base + rows[:, col]
    return code


def build_symmetry_closed_pattern(
    structure: Structure,
    basis: BasisData,
    sg: SpaceGroupData,
    base: SparsePattern | None = None,
) -> SparsePattern:
    r"""Build a CSR ``SparsePattern`` closed under all space/magnetic group operations.

    Theory:
        Starting from distance-cutoff neighbor pairs :math:`(\mu, \nu, \mathbf{T})`, adds
        every symmetry image :math:`(\mu', \nu', \mathbf{T}')` where :math:`\mu'` runs over
        the same :math:`(n, l, \text{iz})` shell on atom :math:`I' = \text{atom\_map}[s, I]`,
        and :math:`\mathbf{T}' = W_S \mathbf{T} + \mathbf{n}_{S, J} - \mathbf{n}_{S, I}`.

    Algorithm:
        Closes both the pair set and its Hermitian transposes :math:`(\nu', \mu', -\mathbf{T}')`,
        sorts edges into canonical CSR order, recomputes ``row_ptr``, ``edge_vectors``, and
        ``transpose_edge_idx``, and asserts zero missing symmetry partners.

    Parameters:
        structure: ``Structure`` PyTree.
        basis: ``BasisData`` with sequential ``(atom, l, m, iz)`` orbitals.
        sg: ``SpaceGroupData`` from ``extract_space_group``.
        base: The distance-cutoff pattern of ``structure`` and ``basis`` if the caller already built it.

    Returns:
        ``SparsePattern`` with ``symmetry_closed=True``.

    Invariants:
        - ``pattern.symmetry_closed == True``.
        - Zero missing partner edges across all ``s`` in ``range(sg.n_ops)``.

    SIESTA Mapping:
        Replaces sparse-pattern closure around ``hsparse.F``.
    """
    _validate_sequential_shells(basis)
    if base is None:
        base = build_sparse_pattern(structure, basis)

    if sg.n_ops == 1:
        return SparsePattern(
            n_orb=base.n_orb,
            n_super_cells=base.n_super_cells,
            nsc=base.nsc,
            cell_shifts=base.cell_shifts,
            src_orb=base.src_orb,
            dst_unit_orb=base.dst_unit_orb,
            edge_cell_idx=base.edge_cell_idx,
            edge_vectors=base.edge_vectors,
            row_ptr=base.row_ptr,
            transpose_edge_idx=base.transpose_edge_idx,
            symmetry_closed=True,
        )

    _, sh_atom, sh_l, sh_mu0, orb_to_sh, sh_map = _shell_arrays(basis, sg)

    # Distinct seed shell-pair edges (sh1, sh2, T) and their transposes, then their images under every operation
    base_src = np.asarray(base.src_orb, dtype=np.int64)
    base_dst = np.asarray(base.dst_unit_orb, dtype=np.int64)
    base_t = np.asarray(base.cell_shifts, dtype=np.int64)[np.asarray(base.edge_cell_idx, dtype=np.int64)]
    sh1 = orb_to_sh[base_src]
    sh2 = orb_to_sh[base_dst]
    seeds = _unique_rows(np.concatenate([np.column_stack([sh1, sh2, base_t]), np.column_stack([sh2, sh1, -base_t])]))
    images = []
    for s in range(sg.n_ops):
        a1, a2 = sh_map[s, seeds[:, 0]], sh_map[s, seeds[:, 1]]
        tp = _image_shifts(sg, s, seeds[:, 2:5], sh_atom[seeds[:, 0]], sh_atom[seeds[:, 1]])
        images.append(np.column_stack([a1, a2, tp]))
        images.append(np.column_stack([a2, a1, -tp]))
    closed = _unique_rows(np.concatenate(images))  # (n_blocks, 5): sh1, sh2, T, sorted

    # Canonical sorted cell_shifts with (0,0,0) at index 0
    uniq_t = _unique_rows(np.concatenate([closed[:, 2:5], np.zeros((1, 3), dtype=np.int64)]))
    nonzero = np.any(uniq_t != 0, axis=1)
    cell_shifts_np = np.concatenate([np.zeros((1, 3), dtype=np.int64), uniq_t[nonzero]]).astype(np.int32)
    n_c = cell_shifts_np.shape[0]
    t_off = int(np.max(np.abs(cell_shifts_np))) if n_c else 0
    t_base = 2 * t_off + 1
    t_codes = _encode_rows(cell_shifts_np.astype(np.int64) + t_off, t_base)
    t_order = np.argsort(t_codes)

    def _shift_index(t: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(t_codes[t_order], _encode_rows(t + t_off, t_base))
        return t_order[pos]

    block_c = _shift_index(closed[:, 2:5])

    # Expand shell-pair blocks into orbital pairs (mu, nu, c), one group of blocks per (l1, l2)
    mus, nus, cs = [], [], []
    l1_b, l2_b = sh_l[closed[:, 0]], sh_l[closed[:, 1]]
    for l1 in np.unique(l1_b):
        for l2 in np.unique(l2_b):
            sel = np.nonzero((l1_b == l1) & (l2_b == l2))[0]
            if sel.size == 0:
                continue
            m1, m2 = np.meshgrid(np.arange(2 * l1 + 1), np.arange(2 * l2 + 1), indexing="ij")
            mus.append((sh_mu0[closed[sel, 0]][:, None] + m1.ravel()[None, :]).ravel())
            nus.append((sh_mu0[closed[sel, 1]][:, None] + m2.ravel()[None, :]).ravel())
            cs.append(np.repeat(block_c[sel], m1.size))
    mu_all, nu_all, c_all = np.concatenate(mus), np.concatenate(nus), np.concatenate(cs)
    # rows by mu, within a row by (c_idx, nu); duplicates removed
    n_orb = basis.n_orb
    keys = np.unique((mu_all * n_c + c_all) * n_orb + nu_all)
    src_arr = (keys // (n_c * n_orb)).astype(np.int32)
    eci_arr = ((keys // n_orb) % n_c).astype(np.int32)
    dst_arr = (keys % n_orb).astype(np.int32)
    row_ptr = np.zeros(n_orb + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(np.bincount(src_arr, minlength=n_orb))

    # Transpose lookup: (nu, mu, -T)
    neg_c = _shift_index(-cell_shifts_np.astype(np.int64))
    t_keys = (dst_arr.astype(np.int64) * n_c + neg_c[eci_arr]) * n_orb + src_arr
    trans_pos = np.searchsorted(keys, t_keys)
    if np.any(trans_pos >= keys.size) or np.any(keys[np.minimum(trans_pos, keys.size - 1)] != t_keys):
        raise AssertionError("Symmetry-closed pattern is not closed under transposition.")
    trans_idx = trans_pos.astype(np.int32)

    cell_np = np.asarray(structure.cell, dtype=np.float64)
    pos_np = np.asarray(structure.positions, dtype=np.float64)
    orb_atom_np = np.asarray(basis.orb_atom, dtype=np.int32)
    shift_cart = cell_shifts_np[eci_arr].astype(np.float64) @ cell_np.T
    edge_vecs = pos_np[orb_atom_np[dst_arr]] + shift_cart - pos_np[orb_atom_np[src_arr]]
    nsc = (
        int(np.max(np.abs(cell_shifts_np[:, 0]))),
        int(np.max(np.abs(cell_shifts_np[:, 1]))),
        int(np.max(np.abs(cell_shifts_np[:, 2]))),
    )

    return SparsePattern(
        n_orb=basis.n_orb,
        n_super_cells=n_c,
        nsc=nsc,
        cell_shifts=jnp.asarray(cell_shifts_np, dtype=jnp.int32),
        src_orb=jnp.asarray(src_arr, dtype=jnp.int32),
        dst_unit_orb=jnp.asarray(dst_arr, dtype=jnp.int32),
        edge_cell_idx=jnp.asarray(eci_arr, dtype=jnp.int32),
        edge_vectors=jnp.asarray(edge_vecs, dtype=jnp.float64),
        row_ptr=jnp.asarray(row_ptr, dtype=jnp.int32),
        transpose_edge_idx=jnp.asarray(trans_idx, dtype=jnp.int32),
        symmetry_closed=True,
    )


@dataclass(frozen=True)
class ProjectionTable:
    r"""Precompiled sparse table executing :math:`\mathcal{P}_{\text{sym}}` as gathers and row sums.

    Theory:
        Row ``d`` of the projection (one flattened spin-edge index) is
        :math:`\sum_j w_{dj}\, x_{s_{dj}}` over the aggregated source entries of that row.

    Algorithm:
        Built once per geometry and ``nspin`` mode by ``compile_projection_table``. The rows are sorted by their
        number of entries and stored in a few buckets of rows of similar length, each padded to its own longest
        row with weight 0 (``src_idx[b]`` and ``weights[b]`` of shape ``(n_rows_b, width_b)``). Applying the table
        is one gather and one sum along rows per bucket, then a gather with ``row_pos`` that puts the rows back in
        their order; there is no scatter. When all rows have about the same length there is a single bucket (the
        plain row-padded layout, about 20x faster than ``segment_sum`` on the CPU for a 48-operation group). When
        they do not (spin-orbit coupling with a cubic group: 73 % padding in one bucket) the buckets keep the stored
        entries within about 15 % of the non-zero ones.

    Invariants:
        - ``row_pos.shape == (n_segments,)``; ``src_idx[b].shape == weights[b].shape``.
        - Rows without entries are not stored: ``row_pos`` sends them to the zero tail.
        - Zero missing partners on any symmetry-closed ``SparsePattern``.

    SIESTA Mapping:
        Precompiled sparse matrix representation of ``m_dm_symmetrize.F90``.
    """

    nspin: int
    n_edges: int
    n_segments: int
    row_pos: jax.Array
    src_idx: tuple[jax.Array, ...]
    weights: tuple[jax.Array, ...]

    @property
    def n_stored(self) -> int:
        """Number of stored entries, padding included."""
        return int(sum(int(w.size) for w in self.weights))

    def arrays(self) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """The arrays of the table as a pytree, to pass to a compiled function as an argument.

        Theory:
            The table is the sparse matrix of the projection; its arrays are data, not program.

        Algorithm:
            Returns the stored arrays without copying.

        Returns:
            Tuple ``(row_pos, src_idx, weights)``.

        Invariants:
            ``self.with_arrays(self.arrays())`` equals ``self``.

        SIESTA Mapping:
            None.
        """
        return (self.row_pos, self.src_idx, self.weights)

    def with_arrays(self, arrays: tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]) -> "ProjectionTable":
        """The same table on other (for example traced) copies of its arrays.

        Theory:
            Same projection; only the storage of the arrays differs.

        Algorithm:
            Builds a new frozen table with the scalar fields of this one.

        Parameters:
            arrays: Tuple ``(row_pos, src_idx, weights)`` as returned by ``arrays``.

        Returns:
            ``ProjectionTable`` with ``nspin``, ``n_edges`` and ``n_segments`` of this table.

        Invariants:
            Applies exactly as this table when the arrays hold the same values.

        SIESTA Mapping:
            None.
        """
        row_pos, src_idx, weights = arrays
        return ProjectionTable(self.nspin, self.n_edges, self.n_segments, row_pos, tuple(src_idx), tuple(weights))


def _bucket_bounds(lengths_desc: np.ndarray, max_buckets: int, max_padding: float) -> list[int]:
    """Split rows sorted by decreasing length into at most ``max_buckets`` contiguous buckets.

    Returns the start position of each bucket. The split minimises the stored entries (every row padded to the
    longest row of its bucket) by dynamic programming over the distinct lengths, with the fewest buckets that keep
    the padding below ``max_padding`` times the non-zero entries (or ``max_buckets`` when none does).
    """
    vals, first, counts = np.unique(-lengths_desc, return_index=True, return_counts=True)
    vals = -vals  # distinct lengths, decreasing; first[i] is the position of the first row of length vals[i]
    n_d = int(vals.size)
    cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.float64)
    nnz = float(np.sum(vals * counts))
    # best[k][j]: minimal stored entries for the first j distinct lengths in k buckets; a bucket i..j-1 costs
    # vals[i] * (number of rows in it)
    inf = np.inf
    best = np.full((max_buckets + 1, n_d + 1), inf)
    back = np.zeros((max_buckets + 1, n_d + 1), dtype=np.int64)
    best[0, 0] = 0.0
    k_used = max_buckets
    for k in range(1, max_buckets + 1):
        for j in range(k, n_d + 1):
            cand = best[k - 1, :j] + vals[:j] * (cum[j] - cum[:j])
            i = int(np.argmin(cand))
            best[k, j] = cand[i]
            back[k, j] = i
        if best[k, n_d] <= (1.0 + max_padding) * nnz or k == n_d:
            k_used = k
            break
    k_used = min(k_used, n_d)
    cuts: list[int] = []
    j = n_d
    for k in range(k_used, 0, -1):
        i = int(back[k, j])
        cuts.append(int(first[i]))
        j = i
    return cuts[::-1]


class _CooAccumulator:
    """Running sum of duplicate ``(dst, src)`` weights, kept compact so the set-up memory stays bounded.

    The entries of a projection table are generated operation by operation; a cubic group with spin-orbit coupling
    generates far more entries than the table finally holds, because most of them fall on the same ``(dst, src)``
    pair. Appended entries are buffered and merged (sorted by key, duplicates summed) whenever the buffer exceeds
    ``flush_at`` entries, so the peak memory is set by the size of the final table, not by the number generated.
    """

    def __init__(self, n_segments: int, flush_at: int = 2_000_000, tol: float = 1e-13) -> None:
        self.n_segments = int(n_segments)
        self.flush_at = int(flush_at)
        self.tol = float(tol)
        self._keys: list[np.ndarray] = []
        self._w: list[np.ndarray] = []
        self._pending = 0
        self._has_merged = False

    def add(self, dst: np.ndarray, src: np.ndarray, w: np.ndarray) -> None:
        """Append entries ``w`` at ``(dst, src)``; single entries below the tolerance are dropped."""
        w = np.asarray(w, dtype=np.float64)
        mask = np.abs(w) > self.tol
        if not np.any(mask):
            return
        keys = np.asarray(dst, dtype=np.int64)[mask] * np.int64(self.n_segments) + np.asarray(src, dtype=np.int64)[mask]
        self._keys.append(keys)
        self._w.append(w[mask])
        self._pending += int(keys.size)
        # merge when the new entries outweigh the merged ones: the total sorting work stays O(n log n)
        merged = int(self._keys[0].size) if self._has_merged else 0
        if self._pending > max(self.flush_at, merged):
            self._merge()

    def _merge(self) -> None:
        if not self._keys:
            return
        keys = np.concatenate(self._keys)
        w = np.concatenate(self._w)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        w = w[order]
        first = np.empty(keys.shape[0], dtype=bool)
        first[0] = True
        first[1:] = keys[1:] != keys[:-1]
        starts = np.nonzero(first)[0]
        self._keys = [keys[starts]]
        self._w = [np.add.reduceat(w, starts)]
        # the merged entries are not counted as pending: only new entries trigger the next merge
        self._pending = 0
        self._has_merged = True

    def to_buckets(
        self, max_buckets: int = 8, max_padding: float = 0.15
    ) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Summed weights as ``(row_pos, src_idx, weights)`` in row-length buckets (see ``ProjectionTable``)."""
        n_segments = self.n_segments
        self._merge()
        keys = np.zeros((0,), dtype=np.int64)
        w_c = np.zeros((0,), dtype=np.float64)
        if self._keys:
            keep = np.abs(self._w[0]) > self.tol
            keys = self._keys[0][keep]
            w_c = self._w[0][keep]
        if keys.size == 0:
            # no entries: every row is in the zero tail
            return jnp.zeros((n_segments,), dtype=jnp.int32), (), ()
        dst_c = keys // n_segments
        src_c = keys % n_segments
        # entries are sorted by (dst, src): the position of an entry within its row is its offset from the row start
        row_len = np.bincount(dst_c, minlength=n_segments)
        row_start = np.concatenate([[0], np.cumsum(row_len)[:-1]])
        col = np.arange(dst_c.size) - row_start[dst_c]

        order = np.argsort(-row_len, kind="stable")  # rows by decreasing length; empty rows last
        row_pos = np.empty(n_segments, dtype=np.int64)
        row_pos[order] = np.arange(n_segments)
        n_rows = int(np.count_nonzero(row_len))
        len_sorted = row_len[order][:n_rows]
        starts = _bucket_bounds(len_sorted, max_buckets, max_padding)
        ends = starts[1:] + [n_rows]
        ent_pos = row_pos[dst_c]
        src_b: list[jax.Array] = []
        w_b: list[jax.Array] = []
        for lo, hi in zip(starts, ends):
            width = int(len_sorted[lo])
            sel = (ent_pos >= lo) & (ent_pos < hi)
            src_ell = np.zeros((hi - lo, width), dtype=np.int32)
            w_ell = np.zeros((hi - lo, width), dtype=np.float64)
            src_ell[ent_pos[sel] - lo, col[sel]] = src_c[sel]
            w_ell[ent_pos[sel] - lo, col[sel]] = w_c[sel]
            src_b.append(jnp.asarray(src_ell))
            w_b.append(jnp.asarray(w_ell))
        return jnp.asarray(row_pos.astype(np.int32)), tuple(src_b), tuple(w_b)


def compile_projection_table(
    basis: BasisData,
    pattern: SparsePattern,
    sg: SpaceGroupData,
    wigner_d: dict[int, np.ndarray],
    nspin: int,
) -> ProjectionTable:
    r"""Precompile the compacted sparse table for :math:`\mathcal{P}_{\text{sym}}` on ``nspin=1,2,4,8``.

    Theory:
        For each shell pair :math:`(\text{sh}_1, \text{sh}_2, \mathbf{T})` and operation
        :math:`S \in G`, maps source edges :math:`(\mu, \nu, \mathbf{T})` onto partner edges
        :math:`(\mu', \nu', \mathbf{T}')` with weights
        :math:`\frac{1}{|G|} D^{l_1}_{m',m}(S) D^{l_2}_{n',n}(S)` coupled to the exact
        collinear flip (``nspin=2``) or :math:`SU(2)` spinor-plus-:math:`\mathcal{T}`
        block transformation (``nspin=4, 8``).

    Algorithm:
        Assembles the entries of all shell-pair blocks of one :math:`(l_1, l_2)` class at once per operation
        (array operations, no loop over blocks) and compacts duplicate entries via
        sorting and ``np.add.reduceat``, and pads the rows to equal length (gather and row sum at run time).

    Parameters:
        basis: ``BasisData`` with sequential shells.
        pattern: Symmetry-closed ``SparsePattern``.
        sg: ``SpaceGroupData``.
        wigner_d: Precomputed Wigner-:math:`D^l` dictionary.
        nspin: Spin mode ``1, 2, 4, 8``.

    Returns:
        Frozen ``ProjectionTable`` ready for ``apply_projection_table``.

    Invariants:
        - Raises ``AssertionError`` if any rotated partner edge is missing from ``pattern``.

    SIESTA Mapping:
        Replaces inner edge loops in ``m_dm_symmetrize.F90`` with static sparse compilation.
    """
    _validate_sequential_shells(basis)
    n_edges = pattern.n_edges
    if not pattern.symmetry_closed and sg.n_ops > 1:
        raise ValueError("compile_projection_table requires a symmetry_closed SparsePattern.")

    if sg.n_ops == 1:
        n_ch = 1 if nspin == 1 else (2 if nspin == 2 else 6)
        n_seg = n_ch * n_edges
        return ProjectionTable(
            nspin=nspin,
            n_edges=n_edges,
            n_segments=n_seg,
            row_pos=jnp.arange(n_seg, dtype=jnp.int32),
            src_idx=(jnp.arange(n_seg, dtype=jnp.int32)[:, None],),
            weights=(jnp.ones((n_seg, 1), dtype=jnp.float64),),
        )

    _, sh_atom, sh_l, sh_mu0, orb_to_sh, sh_map = _shell_arrays(basis, sg)
    n_sh = int(sh_atom.shape[0])
    src_np = np.asarray(pattern.src_orb, dtype=np.int64)
    dst_np = np.asarray(pattern.dst_unit_orb, dtype=np.int64)
    eci_np = np.asarray(pattern.edge_cell_idx, dtype=np.int64)
    shifts_np = np.asarray(pattern.cell_shifts, dtype=np.int64)
    trans_np = np.asarray(pattern.transpose_edge_idx, dtype=np.int64)
    n_c = shifts_np.shape[0]

    # Shell-pair blocks (sh1, sh2, c_idx): one id per block, sorted by key; per edge its block and (m1, m2) slot
    sh1_e, sh2_e = orb_to_sh[src_np], orb_to_sh[dst_np]
    e_key = (sh1_e * n_sh + sh2_e) * n_c + eci_np
    blk_keys, e_blk = np.unique(e_key, return_inverse=True)
    n_blk = blk_keys.shape[0]
    b_sh1 = blk_keys // (n_sh * n_c)
    b_sh2 = (blk_keys // n_c) % n_sh
    b_c = blk_keys % n_c
    b_l1, b_l2 = sh_l[b_sh1], sh_l[b_sh2]
    max_d = int(2 * sh_l.max() + 1) if n_sh else 1
    # blk_src[b, m1, m2] = edge index (-1 if missing), padded to the largest shell dimension
    blk_src = np.full((n_blk, max_d, max_d), -1, dtype=np.int64)
    blk_src[e_blk, src_np - sh_mu0[sh1_e], dst_np - sh_mu0[sh2_e]] = np.arange(n_edges)
    for l1 in np.unique(b_l1):
        for l2 in np.unique(b_l2):
            sel = (b_l1 == l1) & (b_l2 == l2)
            if np.any(blk_src[sel, : 2 * l1 + 1, : 2 * l2 + 1] < 0):
                bad = int(np.nonzero(sel & np.any(blk_src[:, : 2 * l1 + 1, : 2 * l2 + 1] < 0, axis=(1, 2)))[0][0])
                raise ValueError(
                    f"Incomplete orbital block in sparse pattern for shells {(int(b_sh1[bad]), int(b_sh2[bad]), int(b_c[bad]))}."
                )

    # cell-shift lookup by integer code
    t_off = int(np.max(np.abs(shifts_np))) + 1
    t_base = 2 * t_off + 1
    t_codes = _encode_rows(shifts_np + t_off, t_base)
    t_order = np.argsort(t_codes)
    t_sorted = t_codes[t_order]

    # Precompute SU(2) 6x8-source real coefficients per op for nspin in (4, 8):
    # Source vector per edge e has 6 channels [0:uu_r, 1:dd_r, 2:ud_r, 3:ud_i, 4:uu_i, 5:dd_i]
    # plus M_{10}(e) = ud_r(e^T) - i ud_i(e^T).
    # Each op s yields spin transitions: list of (dst_ch, src_ch, use_transpose_edge, coeff)
    spin_rules: list[list[tuple[int, int, bool, float]]] = []
    inv_g = 1.0 / float(sg.n_ops)

    for s in range(sg.n_ops):
        if nspin in (4, 8):
            tr = bool(sg.time_reversals[s])
            a_mat = su2_spinor_matrix(sg.rotations_cart[s], tr)
            eta = -1.0 if tr else 1.0
            # Destination (a, b) in {(0,0)->uu, (1,1)->dd, (0,1)->ud}:
            # (A X A^H)_{ab} = sum_{c,d} T_{ab,cd} X_{cd}, where T_{ab,cd} = A_{a,c} * conj(A_{b,d})
            # X_{00} = (ch0) + i*(eta*ch4); X_{11} = (ch1) + i*(eta*ch5)
            # X_{01} = (ch2) + i*(eta*ch3); X_{10} = (ch2_T) + i*(-eta*ch3_T)
            src_specs = [
                (0, 0, 0, False, 1.0, 4, False, eta),
                (1, 1, 1, False, 1.0, 5, False, eta),
                (0, 1, 2, False, 1.0, 3, False, eta),
                (1, 0, 2, True, 1.0, 3, True, -eta),
            ]
            rules_s: list[tuple[int, int, bool, float]] = []
            # Map (a,b) to output channels: (0,0)->(ch0, ch4); (1,1)->(ch1, ch5); (0,1)->(ch2, ch3)
            dst_pairs = [(0, 0, 0, 4), (1, 1, 1, 5), (0, 1, 2, 3)]
            for a, b, ch_r, ch_i in dst_pairs:
                for c, d, sr, sr_t, wr, si, si_t, wi in src_specs:
                    t_c = a_mat[a, c] * np.conj(a_mat[b, d])
                    tr_val = float(np.real(t_c))
                    ti_val = float(np.imag(t_c))
                    # Re(t_c * (x_r + i*x_i)) = tr_val*x_r - ti_val*x_i
                    if abs(tr_val * wr) > 1e-14:
                        rules_s.append((ch_r, sr, sr_t, tr_val * wr))
                    if abs(ti_val * wi) > 1e-14:
                        rules_s.append((ch_r, si, si_t, -ti_val * wi))
                    # Im(t_c * (x_r + i*x_i)) = ti_val*x_r + tr_val*x_i
                    if abs(ti_val * wr) > 1e-14:
                        rules_s.append((ch_i, sr, sr_t, ti_val * wr))
                    if abs(tr_val * wi) > 1e-14:
                        rules_s.append((ch_i, si, si_t, tr_val * wi))
            spin_rules.append(rules_s)

    n_ch = 1 if nspin == 1 else (2 if nspin == 2 else 6)
    n_seg = n_ch * n_edges
    acc = _CooAccumulator(n_seg)
    time_rev = np.asarray(sg.time_reversals, dtype=bool)

    for l1 in np.unique(b_l1):
        for l2 in np.unique(b_l2):
            cls = np.nonzero((b_l1 == l1) & (b_l2 == l2))[0]
            if cls.size == 0:
                continue
            dim1, dim2 = 2 * int(l1) + 1, 2 * int(l2) + 1
            src_cls = blk_src[cls, :dim1, :dim2].reshape(cls.size, dim1 * dim2)
            src_cls_t = trans_np[src_cls]
            t_cls = shifts_np[b_c[cls]]
            a1, a2 = sh_atom[b_sh1[cls]], sh_atom[b_sh2[cls]]
            for s in range(sg.n_ops):
                # partner blocks of the whole class under operation s
                sh1_p, sh2_p = sh_map[s, b_sh1[cls]], sh_map[s, b_sh2[cls]]
                tp = _image_shifts(sg, s, t_cls, a1, a2)
                in_range = np.all(np.abs(tp) < t_off, axis=1)
                tp_code = _encode_rows(np.where(in_range[:, None], tp, 0) + t_off, t_base)
                pos = np.minimum(np.searchsorted(t_sorted, tp_code), n_c - 1)
                found_t = in_range & (t_sorted[pos] == tp_code)
                if not np.all(found_t):
                    bad = int(np.nonzero(~found_t)[0][0])
                    raise AssertionError(f"Missing rotated cell shift {tp[bad].tolist()} in symmetry_closed pattern.")
                c_p = t_order[pos]
                key_p = (sh1_p * n_sh + sh2_p) * n_c + c_p
                bpos = np.minimum(np.searchsorted(blk_keys, key_p), n_blk - 1)
                if not np.all(blk_keys[bpos] == key_p):
                    bad = int(np.nonzero(blk_keys[bpos] != key_p)[0][0])
                    raise AssertionError(
                        f"Missing rotated orbital block {(int(sh1_p[bad]), int(sh2_p[bad]), tuple(tp[bad].tolist()))} in pattern."
                    )
                dst_cls = blk_src[bpos, :dim1, :dim2].reshape(cls.size, dim1 * dim2)

                # Weight tensor W[m', m, n', n] = (1/|G|) * D1[m', m] * D2[n', n], flattened dst (a, b) x src (m, n)
                w4 = inv_g * np.einsum("am,bn->abmn", wigner_d[int(l1)][s], wigner_d[int(l2)][s])
                w_mat = w4.reshape(dim1 * dim2, dim1 * dim2)
                ii, jj = np.nonzero(np.abs(w_mat) > 1e-13)
                if ii.size == 0:
                    continue
                d_edges = dst_cls[:, ii].ravel()
                s_edges = src_cls[:, jj].ravel()
                w_vals = np.broadcast_to(w_mat[ii, jj], (cls.size, ii.size)).ravel()

                if nspin == 1:
                    acc.add(d_edges, s_edges, w_vals)
                elif nspin == 2:
                    for ch_s in (0, 1):
                        ch_d = (1 - ch_s) if time_rev[s] else ch_s
                        acc.add(ch_d * n_edges + d_edges, ch_s * n_edges + s_edges, w_vals)
                else:
                    s_edges_t = src_cls_t[:, jj].ravel()
                    for ch_d, ch_s, use_t, sp_w in spin_rules[s]:
                        acc.add(ch_d * n_edges + d_edges, ch_s * n_edges + (s_edges_t if use_t else s_edges), sp_w * w_vals)

    row_pos, src_b, w_b = acc.to_buckets()
    return ProjectionTable(
        nspin=nspin,
        n_edges=n_edges,
        n_segments=n_seg,
        row_pos=row_pos,
        src_idx=src_b,
        weights=w_b,
    )


def apply_projection_table(
    matrix_edges: jax.Array,
    table: ProjectionTable,
) -> jax.Array:
    r"""Execute the precompiled sparse symmetry projection (gather and row sum).

    Theory:
        Evaluates :math:`\mathcal{P}_{\text{sym}}(M)` using precompiled row weights over
        real channels (or 6 real subchannels encoding complex ``[uu, dd, Re(ud), Im(ud)]``).

    Algorithm:
        Pure JAX vectorized gather-multiply-row sum per row-length bucket, compiled once per
        shape (calls outside ``jax.jit`` would otherwise dispatch each operation separately).

    Parameters:
        matrix_edges: Sparse matrix of shape ``(n_spin_mat, n_edges)``.
        table: Precompiled ``ProjectionTable`` matching ``matrix_edges.shape[1]``.

    Returns:
        Projected matrix with the same shape and dtype as ``matrix_edges``.

    Invariants:
        - Preserves input dtype (``float64`` vs ``complex128``).
        - Satisfies :math:`\mathcal{P}_{\text{sym}}^2 = \mathcal{P}_{\text{sym}}` on compatible inputs.

    SIESTA Mapping:
        Replaces runtime projection loop in ``m_dm_symmetrize.F90``.
    """
    return _apply_projection(
        matrix_edges, table.row_pos, table.src_idx, table.weights, nspin=table.nspin, n_edges=table.n_edges
    )


@partial(jax.jit, static_argnames=("nspin", "n_edges"))
def _apply_projection(
    matrix_edges: jax.Array,
    row_pos: jax.Array,
    src_idx: tuple[jax.Array, ...],
    weights: tuple[jax.Array, ...],
    *,
    nspin: int,
    n_edges: int,
) -> jax.Array:
    """Compiled body of :func:`apply_projection_table` (one compilation per shape, also when called eagerly)."""

    def _rows(flat: jax.Array) -> jax.Array:
        # one gather and row sum per bucket, in sorted-row order; rows without entries are the zero tail
        parts = [jnp.sum(flat[s_b] * w_b, axis=1) for s_b, w_b in zip(src_idx, weights)]
        n_stored = sum(int(s_b.shape[0]) for s_b in src_idx)
        if n_stored < row_pos.shape[0]:
            parts.append(jnp.zeros((row_pos.shape[0] - n_stored,), dtype=jnp.float64))
        return jnp.concatenate(parts)[row_pos]

    if nspin in (1, 2):
        flat_in = jnp.real(matrix_edges).reshape(-1)
        out = _rows(flat_in)
        res = out.reshape((nspin, n_edges))
        return res.astype(matrix_edges.dtype)

    # nspin in (4, 8): pack into 6 float64 channels [uu_r, dd_r, ud_r, ud_i, uu_i, dd_i]
    is_cplx = jnp.iscomplexobj(matrix_edges)
    z_zeros = jnp.zeros((n_edges,), dtype=jnp.float64)
    uu_i = jnp.imag(matrix_edges[0]) if is_cplx else z_zeros
    dd_i = jnp.imag(matrix_edges[1]) if is_cplx else z_zeros
    flat6 = jnp.concatenate(
        [
            jnp.real(matrix_edges[0]),
            jnp.real(matrix_edges[1]),
            jnp.real(matrix_edges[2]),
            jnp.real(matrix_edges[3]),
            uu_i,
            dd_i,
        ],
        axis=0,
    )
    out6 = _rows(flat6).reshape((6, n_edges))

    if is_cplx:
        return jnp.stack(
            [
                out6[0] + 1j * out6[4],
                out6[1] + 1j * out6[5],
                out6[2].astype(jnp.complex128),
                out6[3].astype(jnp.complex128),
            ],
            axis=0,
        )
    return out6[:4]
