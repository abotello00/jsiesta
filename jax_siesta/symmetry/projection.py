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
    shells, lookup = _validate_sequential_shells(basis)
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

    orb_to_sh = np.empty(basis.n_orb, dtype=np.int32)
    for sh_i, (_, ang_l, _, mu0) in enumerate(shells):
        orb_to_sh[mu0 : mu0 + 2 * ang_l + 1] = sh_i

    base_src = np.asarray(base.src_orb, dtype=np.int32)
    base_dst = np.asarray(base.dst_unit_orb, dtype=np.int32)
    base_eci = np.asarray(base.edge_cell_idx, dtype=np.int32)
    base_sh = np.asarray(base.cell_shifts, dtype=np.int32)

    # Deduplicate initial shell-pair edges (sh1, sh2, t0, t1, t2)
    seed_shell_edges: set[tuple[int, int, int, int, int]] = set()
    for e_i in range(base.n_edges):
        sh1 = int(orb_to_sh[base_src[e_i]])
        sh2 = int(orb_to_sh[base_dst[e_i]])
        t = (int(base_sh[base_eci[e_i], 0]), int(base_sh[base_eci[e_i], 1]), int(base_sh[base_eci[e_i], 2]))
        seed_shell_edges.add((sh1, sh2, t[0], t[1], t[2]))
        seed_shell_edges.add((sh2, sh1, -t[0], -t[1], -t[2]))

    closed_shell_edges: set[tuple[int, int, int, int, int]] = set()
    for sh1, sh2, t0, t1, t2 in seed_shell_edges:
        ia1, l1, iz1, _ = shells[sh1]
        ia2, l2, iz2, _ = shells[sh2]
        t_vec = np.array([t0, t1, t2], dtype=np.int32)
        for s in range(sg.n_ops):
            ja1 = int(sg.atom_map[s, ia1])
            ja2 = int(sg.atom_map[s, ia2])
            sh1_p = lookup[(ja1, l1, iz1)]
            sh2_p = lookup[(ja2, l2, iz2)]
            tp = sg.rotations_frac[s] @ t_vec + sg.atom_translation[s, ia2] - sg.atom_translation[s, ia1]
            tp0, tp1, tp2 = int(tp[0]), int(tp[1]), int(tp[2])
            closed_shell_edges.add((sh1_p, sh2_p, tp0, tp1, tp2))
            closed_shell_edges.add((sh2_p, sh1_p, -tp0, -tp1, -tp2))

    # Build canonical sorted cell_shifts with (0,0,0) at index 0
    shift_set = {(0, 0, 0)} | {(t0, t1, t2) for (_, _, t0, t1, t2) in closed_shell_edges}
    other_shifts = sorted(s for s in shift_set if s != (0, 0, 0))
    cell_shifts_list = [(0, 0, 0)] + other_shifts
    shift_to_idx = {s: idx for idx, s in enumerate(cell_shifts_list)}
    cell_shifts_np = np.asarray(cell_shifts_list, dtype=np.int32)

    # Expand shell-pair blocks into individual orbital pairs grouped per row src_orb
    edges_per_row: list[list[tuple[int, int]]] = [[] for _ in range(basis.n_orb)]
    for sh1, sh2, t0, t1, t2 in closed_shell_edges:
        _, l1, _, mu0 = shells[sh1]
        _, l2, _, nu0 = shells[sh2]
        c_idx = shift_to_idx[(t0, t1, t2)]
        for mu in range(mu0, mu0 + 2 * l1 + 1):
            for nu in range(nu0, nu0 + 2 * l2 + 1):
                edges_per_row[mu].append((nu, c_idx))

    row_ptr = np.zeros(basis.n_orb + 1, dtype=np.int32)
    src_list: list[int] = []
    dst_list: list[int] = []
    eci_list: list[int] = []
    edge_lookup: dict[tuple[int, int, int], int] = {}

    for mu in range(basis.n_orb):
        row_items = sorted(set(edges_per_row[mu]), key=lambda pair: (pair[1], pair[0]))
        for nu, c_idx in row_items:
            e_idx = len(src_list)
            src_list.append(mu)
            dst_list.append(nu)
            eci_list.append(c_idx)
            edge_lookup[(mu, nu, c_idx)] = e_idx
        row_ptr[mu + 1] = len(src_list)

    src_arr = np.asarray(src_list, dtype=np.int32)
    dst_arr = np.asarray(dst_list, dtype=np.int32)
    eci_arr = np.asarray(eci_list, dtype=np.int32)

    # Transpose lookup: (nu, mu, -T)
    trans_idx = np.empty(len(src_list), dtype=np.int32)
    for e_i in range(len(src_list)):
        t = cell_shifts_list[eci_arr[e_i]]
        neg_c = shift_to_idx[(-t[0], -t[1], -t[2])]
        trans_idx[e_i] = edge_lookup[(int(dst_arr[e_i]), int(src_arr[e_i]), neg_c)]

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
        n_super_cells=len(cell_shifts_list),
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
    r"""Precompiled sparse table executing :math:`\mathcal{P}_{\text{sym}}` as one gather and row sum.

    Theory:
        Row ``d`` of the projection (one flattened spin-edge index) is
        :math:`\sum_j w_{dj}\, x_{s_{dj}}` over the aggregated source entries of that row.

    Algorithm:
        Built once per geometry and ``nspin`` mode by ``compile_projection_table``. The rows are stored padded to the
        longest row (ELL layout: ``src_idx`` and ``weights`` of shape ``(n_segments, row_width)``, padding with
        weight 0), so applying the table is a gather and a sum along rows, with no scatter: about 20x faster than
        ``segment_sum`` on the CPU for a 48-operation group, at under 25 % padding.

    Invariants:
        - ``src_idx.shape == weights.shape == (n_segments, row_width)``.
        - Zero missing partners on any symmetry-closed ``SparsePattern``.

    SIESTA Mapping:
        Precompiled sparse matrix representation of ``m_dm_symmetrize.F90``.
    """

    nspin: int
    n_edges: int
    n_segments: int
    src_idx: jax.Array
    weights: jax.Array


def _aggregate_ell(
    dst_list: list[np.ndarray],
    src_list: list[np.ndarray],
    w_list: list[np.ndarray],
    n_segments: int,
    tol: float = 1e-13,
) -> tuple[jax.Array, jax.Array]:
    """Sum duplicate ``(dst, src)`` entries, drop sub-tolerance weights, and pad the rows (``ProjectionTable``)."""
    dst = np.concatenate(dst_list).astype(np.int64)
    src = np.concatenate(src_list).astype(np.int64)
    w = np.concatenate(w_list).astype(np.float64)

    mask = np.abs(w) > tol
    dst = dst[mask]
    src = src[mask]
    w = w[mask]
    if dst.size == 0:
        return jnp.zeros((n_segments, 1), dtype=jnp.int32), jnp.zeros((n_segments, 1), dtype=jnp.float64)

    keys = dst * np.int64(n_segments) + src
    order = np.argsort(keys)
    keys_s = keys[order]
    dst_s = dst[order]
    src_s = src[order]
    w_s = w[order]

    uniq_mask = np.empty(keys_s.shape[0], dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.where(uniq_mask)[0]
    w_agg = np.add.reduceat(w_s, starts)
    keep = np.abs(w_agg) > tol

    src_c = src_s[starts][keep]
    dst_c = dst_s[starts][keep]
    w_c = w_agg[keep]
    # entries are sorted by (dst, src): the position of an entry within its row is its offset from the row start
    row_len = np.bincount(dst_c, minlength=n_segments)
    width = max(1, int(row_len.max()))
    row_start = np.concatenate([[0], np.cumsum(row_len)[:-1]])
    col = np.arange(dst_c.size) - row_start[dst_c]
    src_ell = np.zeros((n_segments, width), dtype=np.int32)
    w_ell = np.zeros((n_segments, width), dtype=np.float64)
    src_ell[dst_c, col] = src_c
    w_ell[dst_c, col] = w_c
    return jnp.asarray(src_ell), jnp.asarray(w_ell)


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
        Assembles vectorized blocks per shell pair and compacts duplicate entries via
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
    shells, lookup = _validate_sequential_shells(basis)
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
            src_idx=jnp.arange(n_seg, dtype=jnp.int32)[:, None],
            weights=jnp.ones((n_seg, 1), dtype=jnp.float64),
        )

    src_np = np.asarray(pattern.src_orb, dtype=np.int32)
    dst_np = np.asarray(pattern.dst_unit_orb, dtype=np.int32)
    eci_np = np.asarray(pattern.edge_cell_idx, dtype=np.int32)
    shifts_np = np.asarray(pattern.cell_shifts, dtype=np.int32)
    trans_np = np.asarray(pattern.transpose_edge_idx, dtype=np.int32)

    shift_to_idx = {(int(s[0]), int(s[1]), int(s[2])): i for i, s in enumerate(shifts_np)}

    orb_to_sh = np.empty(basis.n_orb, dtype=np.int32)
    for sh_i, (_, ang_l, _, mu0) in enumerate(shells):
        orb_to_sh[mu0 : mu0 + 2 * ang_l + 1] = sh_i

    # Group edges by shell-pair block (sh1, sh2, c_idx) -> (dim1, dim2) matrix of edge indices
    block_keys: dict[tuple[int, int, int], np.ndarray] = {}
    for e_i in range(n_edges):
        mu = int(src_np[e_i])
        nu = int(dst_np[e_i])
        c_idx = int(eci_np[e_i])
        sh1 = int(orb_to_sh[mu])
        sh2 = int(orb_to_sh[nu])
        _, l1, _, mu0 = shells[sh1]
        _, l2, _, nu0 = shells[sh2]
        blk = block_keys.get((sh1, sh2, c_idx))
        if blk is None:
            blk = np.full((2 * l1 + 1, 2 * l2 + 1), -1, dtype=np.int32)
            block_keys[(sh1, sh2, c_idx)] = blk
        blk[mu - mu0, nu - nu0] = e_i

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

    dst_acc: list[np.ndarray] = []
    src_acc: list[np.ndarray] = []
    w_acc: list[np.ndarray] = []

    for (sh1, sh2, c_idx), blk_src in block_keys.items():
        if np.any(blk_src < 0):
            raise ValueError(f"Incomplete orbital block in sparse pattern for shells {(sh1, sh2, c_idx)}.")
        ia1, l1, iz1, _ = shells[sh1]
        ia2, l2, iz2, _ = shells[sh2]
        t_vec = shifts_np[c_idx]
        d1_all = wigner_d[l1]
        d2_all = wigner_d[l2]
        dim1 = 2 * l1 + 1
        dim2 = 2 * l2 + 1
        blk_src_t = trans_np[blk_src]  # transpose edge index for each entry in blk_src

        for s in range(sg.n_ops):
            ja1 = int(sg.atom_map[s, ia1])
            ja2 = int(sg.atom_map[s, ia2])
            sh1_p = lookup[(ja1, l1, iz1)]
            sh2_p = lookup[(ja2, l2, iz2)]
            tp = sg.rotations_frac[s] @ t_vec + sg.atom_translation[s, ia2] - sg.atom_translation[s, ia1]
            c_idx_p = shift_to_idx.get((int(tp[0]), int(tp[1]), int(tp[2])))
            if c_idx_p is None:
                raise AssertionError(f"Missing rotated cell shift {tp.tolist()} in symmetry_closed pattern.")
            blk_dst = block_keys.get((sh1_p, sh2_p, c_idx_p))
            if blk_dst is None:
                raise AssertionError(
                    f"Missing rotated orbital block {(sh1_p, sh2_p, tuple(tp.tolist()))} in pattern."
                )

            # Weight tensor W[m', m, n', n] = (1/|G|) * D1[m', m] * D2[n', n]
            w4 = inv_g * np.einsum("am,bn->abmn", d1_all[s], d2_all[s])  # (dim1, dim2, dim1, dim2)
            # Flatten dst (a, b) -> n_pairs_dst, src (m, n) -> n_pairs_src
            w_mat = w4.reshape(dim1 * dim2, dim1 * dim2)
            mask = np.abs(w_mat) > 1e-13
            ii, jj = np.where(mask)
            if ii.size == 0:
                continue
            d_edges = blk_dst.ravel()[ii]
            s_edges = blk_src.ravel()[jj]
            s_edges_t = blk_src_t.ravel()[jj]
            w_vals = w_mat[ii, jj]

            if nspin == 1:
                dst_acc.append(d_edges)
                src_acc.append(s_edges)
                w_acc.append(w_vals)
            elif nspin == 2:
                tr = bool(sg.time_reversals[s])
                for ch_s in (0, 1):
                    ch_d = (1 - ch_s) if tr else ch_s
                    dst_acc.append(ch_d * n_edges + d_edges)
                    src_acc.append(ch_s * n_edges + s_edges)
                    w_acc.append(w_vals)
            else:
                for ch_d, ch_s, use_t, sp_w in spin_rules[s]:
                    dst_acc.append(ch_d * n_edges + d_edges)
                    src_acc.append(ch_s * n_edges + (s_edges_t if use_t else s_edges))
                    w_acc.append(sp_w * w_vals)

    n_ch = 1 if nspin == 1 else (2 if nspin == 2 else 6)
    n_seg = n_ch * n_edges
    src_ell, w_ell = _aggregate_ell(dst_acc, src_acc, w_acc, n_segments=n_seg)
    return ProjectionTable(
        nspin=nspin,
        n_edges=n_edges,
        n_segments=n_seg,
        src_idx=src_ell,
        weights=w_ell,
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
        Pure JAX vectorized gather-multiply-row sum without Python loops, compiled once per
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
    return _apply_projection(matrix_edges, table.src_idx, table.weights, nspin=table.nspin, n_edges=table.n_edges)


@partial(jax.jit, static_argnames=("nspin", "n_edges"))
def _apply_projection(
    matrix_edges: jax.Array,
    src_idx: jax.Array,
    weights: jax.Array,
    *,
    nspin: int,
    n_edges: int,
) -> jax.Array:
    """Compiled body of :func:`apply_projection_table` (one compilation per shape, also when called eagerly)."""
    if nspin in (1, 2):
        flat_in = jnp.real(matrix_edges).reshape(-1)
        out = jnp.sum(flat_in[src_idx] * weights, axis=1)
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
    out6 = jnp.sum(flat6[src_idx] * weights, axis=1).reshape((6, n_edges))

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
