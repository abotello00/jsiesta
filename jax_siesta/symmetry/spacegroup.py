r"""Crystal and magnetic space-group engine, IBZ k-mesh, and commensurate mesh maps.

Theory:
    Extracts crystallographic and Shubnikov Type I--IV magnetic symmetry operations
    :math:`S = (W_S, \mathbf{t}_S, \theta_S)` from ``spglib``, enforces finite-group
    closure and real-space mesh commensurability, and constructs:

    1. Atomic permutation tables ``atom_map[s, I]`` and lattice wraps ``atom_translation[s, I]``
       satisfying :math:`\mathbf{C}_S \mathbf{R}_I + \mathbf{A}\mathbf{t}_S = \mathbf{R}_{I'} + \mathbf{A}\mathbf{n}_{S,I}`.
    2. Symmetry-reduced irreducible Brillouin-zone (IBZ) k-point meshes aware of
       Shubnikov time-reversal breaking.
    3. Symmetry-commensurate grid counts :math:`(N_1, N_2, N_3)` and bijective flat
       permutation indices :math:`\pi_S`.

Algorithm:
    Wraps ``spglib.get_symmetry_dataset``, ``spglib.get_magnetic_symmetry_dataset``,
    and ``spglib.get_stabilized_reciprocal_mesh``; filters out non-PBC-compatible or
    non-rational-shift operations and closes the surviving set under group multiplication.

Invariants:
    - Operation index ``0`` is always the identity :math:`(I_3, \mathbf{0}, 0)`.
    - Group operations are strictly closed under composition and inversion.
    - Grid permutation :math:`\pi_S` is a bijection on ``[0, N1 * N2 * N3)`` for every ``s``.

SIESTA Mapping:
    Replaces and generalizes ``m_symmetry.F90``, ``kpoint_scf.F90``, and symmetry mesh
    commensurability checks in ``dhscf.F``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import spglib

from jax_siesta.core.types import KPointMesh, SCFConfig, Structure
from jax_siesta.grid.mesh import _smooth_grid_dim, choose_mesh_dimensions
from jax_siesta.scf.driver import build_monkhorst_pack_mesh


@dataclass(frozen=True)
class SpaceGroupData:
    r"""Immutable crystal/magnetic space-group representation for native symmetry.

    Theory:
        Stores the closed group of commensurate spatial and time-reversed operations
        acting on fractional and Cartesian coordinates, atomic sites, and lattice vectors.

    Algorithm:
        Populated once per geometry by ``extract_space_group``.

    Invariants:
        - ``rotations_frac.shape == (n_ops, 3, 3)`` integer unimodular matrices.
        - ``inv_op[s]`` indexes the exact group inverse :math:`S^{-1}`.

    SIESTA Mapping:
        Replaces static symmetry tables in ``m_symmetry.F90``.
    """

    n_ops: int
    subgroup_label: str
    msg_type: int
    spacegroup_number: int
    rotations_frac: np.ndarray        # (n_ops, 3, 3) int32
    translations_frac: np.ndarray     # (n_ops, 3) float64 in [0, 1)
    rotations_cart: np.ndarray        # (n_ops, 3, 3) float64 orthogonal O(3)
    translations_cart: np.ndarray     # (n_ops, 3) float64
    det_cart: np.ndarray              # (n_ops,) float64 (+1 or -1)
    time_reversals: np.ndarray        # (n_ops,) bool
    inv_op: np.ndarray                # (n_ops,) int32
    atom_map: np.ndarray              # (n_ops, n_atoms) int32
    atom_translation: np.ndarray      # (n_ops, n_atoms, 3) int32
    inv_atom_map: np.ndarray          # (n_ops, n_atoms) int32
    inv_atom_translation: np.ndarray  # (n_ops, n_atoms, 3) int32
    has_time_reversal_k: bool
    is_magnetic: bool


def _rational_denom(val: float, max_denom: int = 12, tol: float = 1e-5) -> int | None:
    """Return small denominator ``q <= max_denom`` if ``val`` is rational within ``tol``, else ``None``."""
    v = val - math.floor(val)
    if v < tol or 1.0 - v < tol:
        return 1
    frac = Fraction(v).limit_denominator(max_denom)
    if abs(v - float(frac)) <= tol:
        return int(frac.denominator)
    return None


def _wrap_frac(vec: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates cleanly into ``[0, 1)`` avoiding ``1.0`` edge jitter."""
    w = vec - np.floor(vec + 1e-9)
    return np.where(np.abs(w - 1.0) < 1e-9, 0.0, w)


def _op_key(w: np.ndarray, t: np.ndarray, tr: bool) -> tuple:
    """Hashable canonical key for space-time group operation ``(W, t mod 1, time_rev)``."""
    t_w = _wrap_frac(t)
    t_q = tuple(int(round(float(x) * 1440.0)) % 1440 for x in t_w)
    w_q = tuple(int(x) for x in np.asarray(w, dtype=np.int32).ravel())
    return (w_q, t_q, bool(tr))


def _close_group(
    rots: list[np.ndarray],
    trans: list[np.ndarray],
    trevs: list[bool],
) -> tuple[list[np.ndarray], list[np.ndarray], list[bool]]:
    """Close a list of ``(W, t, theta)`` operations under group composition, keeping identity first."""
    ident_w = np.eye(3, dtype=np.int32)
    ident_t = np.zeros(3, dtype=np.float64)
    ops_dict: dict[tuple, tuple[np.ndarray, np.ndarray, bool]] = {
        _op_key(ident_w, ident_t, False): (ident_w, ident_t, False)
    }
    ordered_keys = [_op_key(ident_w, ident_t, False)]

    for w, t, tr in zip(rots, trans, trevs, strict=True):
        tw = _wrap_frac(t)
        k = _op_key(w, tw, tr)
        if k not in ops_dict:
            ops_dict[k] = (np.asarray(w, dtype=np.int32), tw, bool(tr))
            ordered_keys.append(k)

    idx = 0
    while idx < len(ordered_keys):
        w1, t1, tr1 = ops_dict[ordered_keys[idx]]
        for k2 in list(ordered_keys):
            w2, t2, tr2 = ops_dict[k2]
            wc = np.asarray(w1 @ w2, dtype=np.int32)
            tc = _wrap_frac(w1 @ t2 + t1)
            trc = bool(tr1 ^ tr2)
            kc = _op_key(wc, tc, trc)
            if kc not in ops_dict:
                ops_dict[kc] = (wc, tc, trc)
                ordered_keys.append(kc)
                if len(ordered_keys) > 256:
                    break
        idx += 1

    return (
        [ops_dict[k][0] for k in ordered_keys],
        [ops_dict[k][1] for k in ordered_keys],
        [ops_dict[k][2] for k in ordered_keys],
    )


def _compute_atom_map_for_op(
    w: np.ndarray,
    t: np.ndarray,
    frac_pos: np.ndarray,
    species_idx: tuple[int, ...],
    pbc: tuple[bool, bool, bool],
    symprec_frac: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Verify and compute ``(atom_map, atom_translation)`` for a single operation."""
    n_at = frac_pos.shape[0]
    amap = np.empty(n_at, dtype=np.int32)
    atrans = np.empty((n_at, 3), dtype=np.int32)

    for ia in range(n_at):
        mapped = w @ frac_pos[ia] + t
        found = -1
        best_shift = np.zeros(3, dtype=np.int32)
        for ja in range(n_at):
            if species_idx[ja] != species_idx[ia]:
                continue
            diff = mapped - frac_pos[ja]
            shift = np.round(diff).astype(np.int32)
            rem = diff - shift
            if np.max(np.abs(rem)) <= symprec_frac:
                found = ja
                best_shift = shift
                break
        if found < 0:
            return None
        amap[ia] = found
        atrans[ia] = best_shift

    # For open-boundary (non-PBC) axes, all atoms must share one common cell wrap
    # (prevents spurious periodic image pairing while permitting box-centered reflections).
    for ax in range(3):
        if not pbc[ax] and not np.all(atrans[:, ax] == atrans[0, ax]):
            return None
    return amap, atrans


def extract_space_group(
    structure: Structure,
    config: SCFConfig,
    symprec: float = 1e-5,
    mag_symprec: float | None = None,
) -> SpaceGroupData:
    r"""Extract crystal or Shubnikov Type I--IV magnetic space group via ``spglib``.

    Theory:
        Queries crystallographic symmetry for spin-unpolarized / non-magnetic runs or
        magnetic symmetry (Shubnikov Type I, III, IV; Type II for grey non-magnetic SOC)
        when initial atomic magnetic moments break time-reversal or spin symmetry.
        Filters operations to respect open boundary conditions (``structure.pbc``) and
        rational translation commensurability (denominators :math:`\le 12`).

    Algorithm:
        Converts column-major ``structure.cell`` to ``spglib`` row-major lattice,
        queries ``get_symmetry_dataset`` or ``get_magnetic_symmetry_dataset``, filters
        PBC/commensurate operations, enforces group closure, and precomputes forward and
        inverse ``atom_map`` and ``atom_translation`` tables.

    Parameters:
        structure: ``Structure`` PyTree with columns as lattice vectors.
        config: ``SCFConfig`` controlling spin mode ``nspin``.
        symprec: Cartesian symmetry tolerance in Bohr (default ``1e-5``).

    Returns:
        Frozen ``SpaceGroupData`` descriptor with closed group operations.

    Invariants:
        - ``inv_op[inv_op[s]] == s``.
        - ``rotations_cart[s]`` is orthogonal (:math:`C^T C = I_3`) to ``1e-9``.

    SIESTA Mapping:
        Replaces ``m_symmetry.F90`` crystal and magnetic symmetry detection.
    """
    cell = np.asarray(structure.cell, dtype=np.float64)
    cell_rows = cell.T
    pos = np.asarray(structure.positions, dtype=np.float64)
    frac_raw = np.linalg.solve(cell, pos.T).T
    frac_wrapped = _wrap_frac(frac_raw)
    numbers = np.asarray(structure.atomic_numbers, dtype=np.int32)
    moments = np.asarray(structure.initial_moments, dtype=np.float64)

    has_nonzero_mom = config.nspin >= 2 and float(np.max(np.abs(moments))) > 1e-8
    if has_nonzero_mom:
        is_collinear_z = config.nspin == 2 and bool(np.allclose(moments[:, :2], 0.0, atol=1e-8))
        mag_in = moments[:, 2] if is_collinear_z else moments
        mag_tol = mag_symprec if mag_symprec is not None else max(1e-4, symprec * 10.0)
        mds = spglib.get_magnetic_symmetry_dataset(
            cast(Any, (cell_rows, frac_wrapped, numbers, mag_in)),
            symprec=symprec,
            mag_symprec=mag_tol,
        )
        if mds is None:
            raw_rots = [np.eye(3, dtype=np.int32)]
            raw_trans: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
            raw_trevs = [False]
            msg_type = 1
            sg_num = 1
            label = "Type-I (P1 fallback)"
        else:
            raw_rots = [np.asarray(r, dtype=np.int32) for r in mds.rotations]
            raw_trans = [np.asarray(t, dtype=np.float64) for t in mds.translations]
            raw_trevs = [bool(x) for x in mds.time_reversals]
            msg_type = int(mds.msg_type)
            sg_num = int(mds.uni_number)
            type_names = {1: "Type-I", 2: "Type-II", 3: "Type-III", 4: "Type-IV"}
            label = f"{type_names.get(msg_type, 'Magnetic')} (UNI {sg_num})"
        # Note: all time-reversal symmetries are explicitly represented by raw_trevs == True
        has_tr_k = False
    else:
        ds = spglib.get_symmetry_dataset(cast(Any, (cell_rows, frac_wrapped, numbers)), symprec=symprec)
        if ds is None:
            raw_rots = [np.eye(3, dtype=np.int32)]
            raw_trans = [np.zeros(3, dtype=np.float64)]
            sg_num = 1
            intl = "P1"
        else:
            raw_rots = [np.asarray(r, dtype=np.int32) for r in ds.rotations]
            raw_trans = [np.asarray(t, dtype=np.float64) for t in ds.translations]
            sg_num = int(ds.number)
            intl = str(ds.international)
        raw_trevs = [False] * len(raw_rots)
        msg_type = 2 if config.nspin >= 2 else 1
        label = f"{'Type-II (Grey)' if config.nspin >= 2 else 'Type-I'} ({intl}, #{sg_num})"
        has_tr_k = True

    # Filter operations for open boundaries (pbc), orthogonality, and rational mesh-commensurate shifts
    inv_cell = np.linalg.inv(cell)
    kept_r: list[np.ndarray] = []
    kept_t: list[np.ndarray] = []
    kept_tr: list[bool] = []
    pbc = structure.pbc

    for w, t, tr in zip(raw_rots, raw_trans, raw_trevs, strict=True):
        # 1. Check Cartesian orthogonality
        c_mat = cell @ w.astype(np.float64) @ inv_cell
        if np.max(np.abs(c_mat.T @ c_mat - np.eye(3))) > 1e-6:
            continue
        # 2. Check PBC-axis decoupling
        pbc_ok = True
        for ax in range(3):
            if not pbc[ax]:
                for other in range(3):
                    if pbc[other] and (w[ax, other] != 0 or w[other, ax] != 0):
                        pbc_ok = False
        if not pbc_ok:
            continue
        # 3. Check rational fractional translation denominator <= 12 (commensurate with real-space grid)
        tw = _wrap_frac(t)
        if any(_rational_denom(float(tw[ax]), max_denom=12, tol=1e-5) is None for ax in range(3)):
            continue
        # 4. Check atomic mapping preserves PBC boundaries (note: compare on raw frac_pos for non-PBC safety)
        amap_res = _compute_atom_map_for_op(w, tw, frac_raw, structure.species_indices, pbc)
        if amap_res is None:
            continue
        kept_r.append(w)
        kept_t.append(tw)
        kept_tr.append(tr)

    # Close surviving operations under group multiplication
    closed_r, closed_t, closed_tr = _close_group(kept_r, kept_t, kept_tr)

    # Verify that every closed operation still maps the crystal onto itself (rational translation and atom map).
    def _filter(rs: list[np.ndarray], ts: list[np.ndarray], trs: list[bool]) -> list[tuple]:
        out = []
        for w, t, tr in zip(rs, ts, trs, strict=True):
            if any(_rational_denom(float(t[ax]), max_denom=12, tol=1e-5) is None for ax in range(3)):
                continue
            amap_res = _compute_atom_map_for_op(w, t, frac_raw, structure.species_indices, pbc)
            if amap_res is not None:
                out.append((w, t, tr, amap_res[0], amap_res[1]))
        return out

    kept = _filter(closed_r, closed_t, closed_tr)
    if len(kept) < len(closed_r):
        # Dropping elements of a closed group generally leaves a set that is not a group, and the projector
        # (1/|G|) sum_S would then not be idempotent (review L5). Rebuild the largest subgroup greedily: add an
        # operation only if the closure of the enlarged set consists entirely of valid operations.
        sub_r = [np.eye(3, dtype=np.int32)]
        sub_t = [np.zeros(3, dtype=np.float64)]
        sub_tr = [False]
        for w, t, tr, _, _ in kept:
            cr, ct, ctr = _close_group(sub_r + [w], sub_t + [t], sub_tr + [tr])
            if len(_filter(cr, ct, ctr)) == len(cr):
                sub_r, sub_t, sub_tr = cr, ct, ctr
        kept = _filter(sub_r, sub_t, sub_tr)
    valid_r = [k[0] for k in kept]
    valid_t = [k[1] for k in kept]
    valid_tr = [k[2] for k in kept]
    valid_amap = [k[3] for k in kept]
    valid_atrans = [k[4] for k in kept]

    n_ops = len(valid_r)
    rots_frac = np.stack(valid_r, axis=0).astype(np.int32)
    trans_frac = np.stack(valid_t, axis=0).astype(np.float64)
    trevs = np.asarray(valid_tr, dtype=bool)
    rots_cart = np.einsum("ai,sij,jb->sab", cell, rots_frac.astype(np.float64), inv_cell)
    trans_cart = (cell @ trans_frac.T).T
    det_cart = np.linalg.det(rots_cart)
    amap_arr = np.stack(valid_amap, axis=0).astype(np.int32)
    atrans_arr = np.stack(valid_atrans, axis=0).astype(np.int32)

    # Precompute inverse operation indices and inverse atom maps
    key_to_idx = {
        _op_key(rots_frac[s], trans_frac[s], bool(trevs[s])): s for s in range(n_ops)
    }
    inv_op = np.zeros(n_ops, dtype=np.int32)
    for s in range(n_ops):
        w_inv = np.round(np.linalg.inv(rots_frac[s].astype(np.float64))).astype(np.int32)
        t_inv = _wrap_frac(-(w_inv @ trans_frac[s]))
        inv_idx = key_to_idx.get(_op_key(w_inv, t_inv, bool(trevs[s])))
        if inv_idx is None:
            raise RuntimeError(f"Symmetry group is not closed under inversion at op {s}.")
        inv_op[s] = inv_idx

    inv_amap_arr = amap_arr[inv_op]
    inv_atrans_arr = atrans_arr[inv_op]

    if n_ops == 1 and len(raw_rots) > 1:
        label = f"{label} -> P1 (mesh-commensurate)"

    return SpaceGroupData(
        n_ops=n_ops,
        subgroup_label=label,
        msg_type=msg_type,
        spacegroup_number=sg_num,
        rotations_frac=rots_frac,
        translations_frac=trans_frac,
        rotations_cart=rots_cart,
        translations_cart=trans_cart,
        det_cart=det_cart,
        time_reversals=trevs,
        inv_op=inv_op,
        atom_map=amap_arr,
        atom_translation=atrans_arr,
        inv_atom_map=inv_amap_arr,
        inv_atom_translation=inv_atrans_arr,
        has_time_reversal_k=has_tr_k,
        is_magnetic=has_nonzero_mom,
    )


def build_ibz_kpoint_mesh(
    sg: SpaceGroupData,
    grid_shape: tuple[int, int, int],
    shifts: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> KPointMesh:
    r"""Construct an irreducible Brillouin-zone (IBZ) k-point mesh via ``spglib``.

    Theory:
        Reduces the uniform Monkhorst-Pack grid under the crystal/magnetic point group.
        Unprimed operations map :math:`\mathbf{k} \to W_S^T \mathbf{k}` (dual orbit), while
        primed magnetic operations containing time reversal :math:`\mathcal{T}` map
        :math:`\mathbf{k} \to -W_S^T \mathbf{k}`.

    Algorithm:
        When shifts are half-integer (``0.0`` or ``0.5``), invokes
        ``spglib.get_stabilized_reciprocal_mesh`` directly. For arbitrary shifts, performs
        exact orbit reduction over the generated grid using ``-W_S`` for primed ops.

    Parameters:
        sg: ``SpaceGroupData`` from ``extract_space_group``.
        grid_shape: Uniform Monkhorst-Pack integers ``(nk1, nk2, nk3)``.
        shifts: Fractional grid offsets in ``[0, 1)``.

    Returns:
        ``KPointMesh`` with ``is_ibz=True`` and normalized ``weights`` summing to ``1.0``.

    Invariants:
        - ``sum(weights) == 1.0`` to machine precision.
        - Matches full-BZ uniform sampling for observables when weighted by ``weights``.

    SIESTA Mapping:
        Replaces ``kpoint_scf.F90`` and magnetic IBZ reduction.
    """
    is_half_step = all(abs(2.0 * s - round(2.0 * s)) < 1e-8 for s in shifts)
    k_rots = np.where(
        sg.time_reversals[:, None, None],
        -sg.rotations_frac,
        sg.rotations_frac,
    ).astype(np.intc)

    if is_half_step:
        is_shift = [int(round(2.0 * s)) % 2 for s in shifts]
        mesh_result = spglib.get_stabilized_reciprocal_mesh(
            list(grid_shape),
            k_rots,
            is_shift=is_shift,
            is_time_reversal=bool(sg.has_time_reversal_k),
        )
        if mesh_result is None:
            raise RuntimeError(f"spglib failed to build the stabilised reciprocal mesh {grid_shape}")
        mapping, grid = mesh_result
        uniq_idx, counts = np.unique(mapping, return_counts=True)
        mesh_arr = np.asarray(grid_shape, dtype=np.float64)
        shift_arr = np.asarray(is_shift, dtype=np.float64) * 0.5
        k_raw = (grid[uniq_idx].astype(np.float64) + shift_arr[None, :]) / mesh_arr[None, :]
        k_pts = k_raw - np.round(k_raw)
        weights = counts.astype(np.float64) / float(grid.shape[0])
        return KPointMesh(
            kpoints_frac=jnp.asarray(k_pts, dtype=jnp.float64),
            weights=jnp.asarray(weights, dtype=jnp.float64),
            grid_shape=grid_shape,
            shifts=shifts,
            is_ibz=True,
            used_time_reversal=bool(sg.has_time_reversal_k or np.any(sg.time_reversals)),
        )

    full = build_monkhorst_pack_mesh(grid_shape, shifts)
    k_all = np.asarray(full.kpoints_frac, dtype=np.float64)
    # Orbit reduction under k' = +/- W_S^T k
    r_mats = np.transpose(k_rots.astype(np.float64), (0, 2, 1))
    if sg.has_time_reversal_k:
        r_mats = np.concatenate([r_mats, -r_mats], axis=0)

    n_all = k_all.shape[0]
    visited = np.zeros(n_all, dtype=bool)
    ibz_pts: list[np.ndarray] = []
    ibz_cnts: list[int] = []
    k_quant = np.round(k_all * 1000000.0).astype(np.int64)
    for i in range(n_all):
        if visited[i]:
            continue
        orbit_k = (r_mats @ k_all[i])
        orbit_k = orbit_k - np.round(orbit_k)
        visited[i] = True
        cnt = 1
        for ko in orbit_k:
            diff = np.abs(k_all - ko[None, :])
            diff = np.minimum(diff, 1.0 - diff)
            matches = np.where(np.max(diff, axis=1) < 1e-6)[0]
            for m in matches:
                if not visited[m]:
                    visited[m] = True
                    cnt += 1
        ibz_pts.append(k_all[i])
        ibz_cnts.append(cnt)
    _ = k_quant
    weights = np.asarray(ibz_cnts, dtype=np.float64) / float(n_all)
    return KPointMesh(
        kpoints_frac=jnp.asarray(np.stack(ibz_pts, axis=0), dtype=np.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
        grid_shape=grid_shape,
        shifts=shifts,
        is_ibz=True,
        used_time_reversal=bool(sg.has_time_reversal_k or np.any(sg.time_reversals)),
    )


def choose_commensurate_mesh_dimensions(
    structure: Structure,
    mesh_cutoff_ry: float,
    sg: SpaceGroupData,
    criterion: str = "siesta",
    mesh_subdivisions: int = 2,
) -> tuple[int, int, int]:
    r"""Choose FFT-smooth ``(N1, N2, N3)`` strictly commensurate with all ``sg`` operations.

    Theory:
        A real-space mesh is invariant under :math:`(W_S, \mathbf{t}_S)` iff:
        1. Rotation-coupled axes share equal grid subdivisions (:math:`W_S[b, a] \ne 0 \implies N_b = N_a`),
        2. Every fractional shift satisfies :math:`t_{S,a} N_a \in \mathbb{Z}`.

    Algorithm:
        Starts from ``choose_mesh_dimensions(structure.cell, mesh_cutoff_ry)``, computes
        connected components of axes coupled by ``sg.rotations_frac``, takes the LCM of
        rational translation denominators per component, and rounds the component max up to a
        smooth multiple of ``2, 3, 5`` divisible by that LCM.

    Parameters:
        structure: ``Structure`` PyTree.
        mesh_cutoff_ry: Plane-wave kinetic energy cutoff in Ry.
        sg: ``SpaceGroupData`` from ``extract_space_group``.
        criterion: Base mesh criterion of :func:`jax_siesta.grid.mesh.choose_mesh_dimensions`. With
            ``"siesta"`` every axis is also a multiple of ``mesh_subdivisions``.
        mesh_subdivisions: SIESTA ``MeshSubDivisions`` (used by the ``"siesta"`` criterion).

    Returns:
        Symmetry-commensurate ``(N1, N2, N3)`` tuple.

    Invariants:
        - Every axis count is at least the unsymmetrized ``mesh_cutoff_ry`` baseline.
        - All translations ``t_frac[s, a] * N_a`` are integers to ``1e-5``.

    SIESTA Mapping:
        Upgrades ``dhscf.F`` FFT grid dimension selection with exact group divisibility.
    """
    base = choose_mesh_dimensions(
        structure.cell, mesh_cutoff_ry, divisors=(1, 1, 1), criterion=criterion, mesh_subdivisions=mesh_subdivisions
    )
    nsm = int(mesh_subdivisions) if criterion == "siesta" else 1

    # Connected components among axes {0, 1, 2} linked by non-zero off-diagonal W_S
    parent = [0, 1, 2]

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for w in sg.rotations_frac:
        for a in range(3):
            for b in range(3):
                if a != b and int(w[a, b]) != 0:
                    _union(a, b)

    # Required denominator divisor per axis from rational fractional translations
    ax_denom = [1, 1, 1]
    for t in sg.translations_frac:
        for a in range(3):
            d = _rational_denom(float(t[a]), max_denom=12, tol=1e-5)
            if d is None:
                raise ValueError(f"Translation {t[a]} is not rational within denom <= 12.")
            ax_denom[a] = math.lcm(ax_denom[a], d)

    # Aggregate denominator & baseline target across each coupled component
    comp_denom: dict[int, int] = {}
    comp_target: dict[int, int] = {}
    for a in range(3):
        r = _find(a)
        comp_denom[r] = math.lcm(comp_denom.get(r, 1), ax_denom[a])
        comp_target[r] = max(comp_target.get(r, 2), base[a])

    # Round each component target to a smooth integer divisible by its component LCM
    out = [0, 0, 0]
    for a in range(3):
        r = _find(a)
        lcm_d = math.lcm(comp_denom[r], nsm)
        n = _smooth_grid_dim(comp_target[r], divisor=lcm_d)
        while any(
            abs(float(sg.translations_frac[s, a]) * n - round(float(sg.translations_frac[s, a]) * n)) > 1e-4
            for s in range(sg.n_ops)
        ):
            n += lcm_d
        out[a] = n
    return (int(out[0]), int(out[1]), int(out[2]))


def check_mesh_commensurate(sg: SpaceGroupData, mesh_shape: tuple[int, int, int]) -> None:
    r"""Raise ``ValueError`` unless ``mesh_shape`` maps onto itself under every operation of ``sg``.

    Theory:
        A grid point :math:`\mathbf f = \mathbf i / \mathbf N` maps to :math:`W_S \mathbf f + \mathbf t_S`.
        This is again a grid point for all :math:`\mathbf i` iff axes coupled by an off-diagonal
        :math:`W_S` have equal :math:`N` and every :math:`t_{S,a} N_a` is an integer.

    Algorithm:
        Checks both conditions for every operation (tolerance 1e-4 on :math:`t N`).

    Parameters:
        sg: ``SpaceGroupData``.
        mesh_shape: Candidate ``(N1, N2, N3)``.

    Returns:
        None.

    Invariants:
        Accepts every mesh produced by ``choose_commensurate_mesh_dimensions``.

    SIESTA Mapping:
        Same condition that ``siesta_spglib_full_impl`` imposes on its mesh for grid symmetrisation.
    """
    n = [int(x) for x in mesh_shape]
    for s in range(sg.n_ops):
        w = np.asarray(sg.rotations_frac[s])
        for a in range(3):
            for b in range(3):
                if a != b and int(w[a, b]) != 0 and n[a] != n[b]:
                    raise ValueError(f"Mesh {tuple(n)} not commensurate: op {s} couples axes {a},{b}.")
            t = float(sg.translations_frac[s, a]) * n[a]
            if abs(t - round(t)) > 1e-4:
                raise ValueError(f"Mesh {tuple(n)} not commensurate: translation {sg.translations_frac[s, a]} on axis {a}.")


def precompute_grid_permutations(
    sg: SpaceGroupData,
    mesh_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Precompute exact bijective flat grid index permutations ``(perm_fwd, perm_inv)``.

    Theory:
        For operation :math:`S` mapping fractional coordinate
        :math:`\mathbf{f}' = (W_S \mathbf{f} + \mathbf{t}_S) \bmod 1`, symmetrizing a scalar
        density uses :math:`\rho_{\text{sym}}(\mathbf{r}_p) = \frac{1}{|G|} \sum_S \rho(\pi_S^{-1}(\mathbf{r}_p))`,
        where ``perm_inv[s, p]`` gives the source flat mesh index :math:`q = \pi_S^{-1}(p)`
        such that :math:`S \mathbf{r}_q = \mathbf{r}_p \pmod{\mathbf{A}}`.

    Algorithm:
        Computes exact integer modular arithmetic :math:`\mathbf{j} = (W_S \mathbf{i} + \mathbf{t}_{\text{int}}) \bmod \mathbf{N}`
        across the vectorized mesh coordinate array and verifies bijectivity on ``[0, N_grid)``.

    Parameters:
        sg: ``SpaceGroupData`` from ``extract_space_group``.
        mesh_shape: Commensurate ``(N1, N2, N3)`` from ``choose_commensurate_mesh_dimensions``.

    Returns:
        Tuple ``(perm_fwd, perm_inv)`` each of shape ``(n_ops, N1 * N2 * N3)`` ``int32``,
        where ``perm_inv[s, perm_fwd[s, q]] == q``.

    Invariants:
        - Raises ``ValueError`` if any operation fails to permute the mesh bijectively.

    SIESTA Mapping:
        Replaces real-space grid symmetrization index tables in ``dhscf.F``.
    """
    n1, n2, n3 = mesh_shape
    n_grid = n1 * n2 * n3
    i0, i1, i2 = np.meshgrid(
        np.arange(n1, dtype=np.int64),
        np.arange(n2, dtype=np.int64),
        np.arange(n3, dtype=np.int64),
        indexing="ij",
    )
    coords = np.stack([i0.ravel(), i1.ravel(), i2.ravel()], axis=1)  # (N_grid, 3)
    strides = np.array([n2 * n3, n3, 1], dtype=np.int64)
    n_vec = np.array([n1, n2, n3], dtype=np.int64)

    perm_fwd = np.empty((sg.n_ops, n_grid), dtype=np.int32)
    for s in range(sg.n_ops):
        w = sg.rotations_frac[s].astype(np.int64)
        t_int_f = sg.translations_frac[s] * n_vec.astype(np.float64)
        t_int = np.round(t_int_f).astype(np.int64)
        if np.max(np.abs(t_int_f - t_int)) > 1e-4:
            raise ValueError(
                f"Mesh {mesh_shape} is not commensurate with op {s} translation {sg.translations_frac[s]}."
            )
        mapped = (coords @ w.T + t_int[None, :]) % n_vec[None, :]
        dst_flat = (mapped @ strides).astype(np.int32)
        if np.unique(dst_flat).shape[0] != n_grid:
            raise ValueError(
                f"Operation {s} with W={w.tolist()} did not bijectively permute mesh {mesh_shape}."
            )
        perm_fwd[s] = dst_flat

    # Inverse permutation: if perm_fwd[s, q] = p, then perm_inv[s, p] = q
    perm_inv = np.empty_like(perm_fwd)
    arange_g = np.arange(n_grid, dtype=np.int32)
    for s in range(sg.n_ops):
        perm_inv[s, perm_fwd[s]] = arange_g
    return perm_fwd, perm_inv
