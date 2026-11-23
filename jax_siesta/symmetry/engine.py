r"""Native space/magnetic-group symmetry engine for grid averaging, irreducible integrals, forces, and stress.

Theory:
    Integrates Steps 1--4 of the native ``spglib`` symmetry architecture:
    1. Space/magnetic group maps, IBZ k-mesh, and commensurate grid permutations ``pi_S``.
    2. Symmetry-closed CSR sparsity and 16-pt Fibonacci Wigner-D + SU(2) ``segment_sum`` projection.
    3. Real-space density and potential symmetrization:

       .. math::
           n_{\text{sym}}(\mathbf{r}) = \frac{1}{|G|} \sum_{S \in G} n(\pi_S^{-1}\mathbf{r}), \qquad
           \mathbf{m}_{\text{sym}}(\mathbf{r}) = \frac{1}{|G|} \sum_{S \in G} (-1)^{\theta_S} (\det \mathbf{C}_S) \mathbf{C}_S \mathbf{m}(\pi_S^{-1}\mathbf{r}),

       together with irreducible-orbit C++ two-center ``S, T`` and ``KB`` evaluation rotated to
       all partner edges via :math:`(D^{l_1}, D^{l_2}, \mathbf{C}_S)`.
    4. Exact crystal invariant projection on atomic forces and stress tensors.

Algorithm:
    ``symmetrize_grid`` uses ``jax.lax.scan`` over precomputed ``perm_inv`` and axial
    rotation tensors so both ``jax.jit`` and reverse-mode ``jax.grad`` (variational
    grid forces/stress) execute without per-op peak memory blowup. ``evaluate_irreducible_two_center``
    evaluates C++ radial Bessel integrals once per space-group orbit representative.

Invariants:
    - ``NativeSymmetryEngine`` satisfies the ``SymmetryProjector`` protocol.
    - :math:`[H_0, U_S] = 0`, :math:`[S, U_S] = 0`, and :math:`[H_{\text{SCF}}, U_S] = 0`.
    - Grid symmetrization preserves total charge and magnetic moment covariance.

SIESTA Mapping:
    Replaces and unifies ``m_symmetry.F90``, ``m_dm_symmetrize.F90``, ``kpoint_scf.F90``,
    and symmetry force/stress cleanup in ``siesta_forces.F``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.ffi import load_ffi_library
from jax_siesta.core.types import (
    BasisData,
    KPointMesh,
    SCFConfig,
    SparsePattern,
    Structure,
    TwoCenterMatrices,
)
from jax_siesta.integrals.two_center import (
    _eval_kb_and_na_pair,
    _eval_st_subset,
    _pack_pao_tables,
    _setup_c_signatures,
)
from jax_siesta.symmetry.projection import (
    ProjectionTable,
    _validate_sequential_shells,
    apply_projection_table,
    build_symmetry_closed_pattern,
    compile_projection_table,
    compute_wigner_d_matrices,
)
from jax_siesta.symmetry.spacegroup import (
    SpaceGroupData,
    build_ibz_kpoint_mesh,
    check_mesh_commensurate,
    choose_commensurate_mesh_dimensions,
    extract_space_group,
    precompute_grid_permutations,
)


@dataclass
class NativeSymmetryEngine:
    r"""End-to-end native ``spglib`` symmetry engine implementing ``SymmetryProjector``.

    Theory:
        Enforces exact group equivariance across IBZ diagonalization, sparse matrix
        projection, real-space grid averaging, irreducible two-center integrals, and
        atomic force / stress projection.

    Algorithm:
        Precompiles all lookup tables once at initialization and stores pure JAX arrays
        for JIT-safe ``project_matrix``, ``symmetrize_grid``, ``project_forces``, and
        ``project_stress``.

    Invariants:
        - ``n_ops >= 1``.
        - ``symmetrize_grid`` and ``project_matrix`` support ``jax.jit`` and ``jax.grad``.

    SIESTA Mapping:
        Implements ``SymmetryProjector`` across SCF and observable evaluation.
    """

    sg: SpaceGroupData
    basis: BasisData
    pattern: SparsePattern
    mesh_shape: tuple[int, int, int]
    perm_fwd: np.ndarray
    perm_inv: np.ndarray
    wigner_d: dict[int, np.ndarray]
    proj_table: ProjectionTable
    nspin: int
    _perm_inv_jax: jax.Array = field(repr=False)
    _rot_cart_jax: jax.Array = field(repr=False)
    _axial_rot_jax: jax.Array = field(repr=False)
    _trev_sign_jax: jax.Array = field(repr=False)
    _atom_map_jax: jax.Array = field(repr=False)
    grid_symmetrization: bool = True
    _extra_tables: dict[int, ProjectionTable] = field(default_factory=dict, repr=False)
    last_st_irred_blocks: int = 0
    last_st_total_blocks: int = 0
    last_kb_irred_blocks: int = 0
    last_kb_total_blocks: int = 0

    @property
    def n_ops(self) -> int:
        """Number of symmetry operations in the closed commensurate group."""
        return self.sg.n_ops

    @property
    def subgroup_label(self) -> str:
        """Human-readable crystallographic / Shubnikov magnetic subgroup label."""
        return self.sg.subgroup_label

    def project_matrix(self, matrix_edges: jax.Array, nspin: int) -> jax.Array:
        r"""Apply precompiled sparse symmetry projection :math:`\mathcal{P}_{\text{sym}}` to ``D``, ``E``, or ``H``.

        Theory:
            Symmetrizes sparse edge arrays across all :math:`S \in G` using precompiled
            Wigner-D orbital rotations and spinor/collinear transformations.

        Algorithm:
            Dispatches to ``apply_projection_table`` using ``jax.ops.segment_sum``.

        Parameters:
            matrix_edges: Sparse matrix of shape ``(n_spin_mat, n_edges)``.
            nspin: Active spin mode ``1, 2, 4, 8``.

        Returns:
            Projected sparse matrix with matching shape and dtype.

        Invariants:
            Idempotent and electron-count preserving.

        SIESTA Mapping:
            Replaces ``m_dm_symmetrize.F90``.
        """
        if self.sg.n_ops == 1:
            return matrix_edges
        if nspin == self.proj_table.nspin:
            table = self.proj_table
        else:
            cached = self._extra_tables.get(nspin)
            if cached is None:
                cached = compile_projection_table(self.basis, self.pattern, self.sg, self.wigner_d, nspin)
                self._extra_tables[nspin] = cached
            table = cached
        return apply_projection_table(matrix_edges, table)

    def symmetrize_grid(self, grid_field: jax.Array) -> jax.Array:
        r"""Apply exact group averaging over commensurate 3D real-space permutations :math:`\pi_S`.

        Theory:
            Averages scalar densities (:math:`n_{\text{ch}}=1`), collinear spin densities
            (:math:`n_{\text{ch}}=2`), or 4-component non-collinear/SOC spinor fields
            (:math:`n_{\text{ch}}=4`, converting :math:`(\rho_{uu}, \rho_{dd}, \mathrm{Re}\rho_{ud}, \mathrm{Im}\rho_{ud})`
            into scalar density :math:`n` + axial magnetization :math:`\mathbf{m}`).

        Algorithm:
            Executes a memory-bounded ``jax.lax.scan`` over operations ``0 .. n_ops - 1``
            gathering ``flat[:, perm_inv[s]]`` and rotating axial spin vectors cleanly so
            both ``jax.jit`` and ``jax.grad`` are supported.

        Parameters:
            grid_field: Array of shape ``(n_ch, N1, N2, N3)`` with ``n_ch in (1, 2, 4)``.

        Returns:
            Symmetrized grid field of identical shape ``(n_ch, N1, N2, N3)``.

        Invariants:
            - Conserves discrete grid integral :math:`\sum_p n(\mathbf{r}_p)`.
            - Exact fixed point for any symmetry-invariant field.

        SIESTA Mapping:
            Replaces real-space FFT grid symmetrization in ``dhscf.F``. With
            ``SCFConfig.symmetrize_grid = False`` it is the identity, which reproduces
            ``siesta_spglib_full_impl`` (density-matrix symmetrisation only) for parity runs.
        """
        if self.sg.n_ops == 1 or not self.grid_symmetrization:
            return grid_field

        n_ch, n1, n2, n3 = grid_field.shape
        n_grid = n1 * n2 * n3
        flat = grid_field.reshape((n_ch, n_grid))
        inv_g = 1.0 / float(self.sg.n_ops)

        if n_ch == 1:
            def _scan_scalar(acc: jax.Array, p_inv: jax.Array) -> tuple[jax.Array, None]:
                return acc + flat[0, p_inv], None

            acc0 = jnp.zeros((n_grid,), dtype=grid_field.dtype)
            total, _ = jax.lax.scan(_scan_scalar, acc0, self._perm_inv_jax)
            return (inv_g * total).reshape((1, n1, n2, n3))

        if n_ch == 2:
            n_tot = flat[0] + flat[1]
            mz = flat[0] - flat[1]

            def _scan_col(
                acc: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, jax.Array]
            ) -> tuple[tuple[jax.Array, jax.Array], None]:
                p_inv, tr_s = xs
                return (acc[0] + n_tot[p_inv], acc[1] + tr_s * mz[p_inv]), None

            z = jnp.zeros((n_grid,), dtype=grid_field.dtype)
            (n_sum, mz_sum), _ = jax.lax.scan(_scan_col, (z, z), (self._perm_inv_jax, self._trev_sign_jax))
            n_avg = inv_g * n_sum
            mz_avg = inv_g * mz_sum
            return jnp.stack([0.5 * (n_avg + mz_avg), 0.5 * (n_avg - mz_avg)], axis=0).reshape((2, n1, n2, n3))

        if n_ch == 4:
            n_tot = flat[0] + flat[1]
            m_vec = jnp.stack([2.0 * flat[2], -2.0 * flat[3], flat[0] - flat[1]], axis=0)  # (3, n_grid)

            def _scan_nc(
                acc: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, jax.Array]
            ) -> tuple[tuple[jax.Array, jax.Array], None]:
                p_inv, a_s = xs
                m_rot = a_s @ m_vec[:, p_inv]
                return (acc[0] + n_tot[p_inv], acc[1] + m_rot), None

            z_n = jnp.zeros((n_grid,), dtype=grid_field.dtype)
            z_m = jnp.zeros((3, n_grid), dtype=grid_field.dtype)
            (n_sum, m_sum), _ = jax.lax.scan(_scan_nc, (z_n, z_m), (self._perm_inv_jax, self._axial_rot_jax))
            n_avg = inv_g * n_sum
            m_avg = inv_g * m_sum
            return jnp.stack(
                [
                    0.5 * (n_avg + m_avg[2]),
                    0.5 * (n_avg - m_avg[2]),
                    0.5 * m_avg[0],
                    -0.5 * m_avg[1],
                ],
                axis=0,
            ).reshape((4, n1, n2, n3))

        raise ValueError(f"Unsupported grid_field channel count {n_ch} (expected 1, 2, or 4).")

    def project_forces(self, forces: jax.Array) -> jax.Array:
        r"""Project atomic forces onto the crystal/magnetic symmetry-invariant subspace.

        Theory:
            Computes :math:`\mathbf{F}^{\text{sym}}_I = \frac{1}{|G|} \sum_{S \in G} \mathbf{C}_S^T \mathbf{F}_{\text{atom\_map}[S, I]}`,
            preserving translational acoustic sum rule and equivariance.

        Algorithm:
            Vectorized gather along ``atom_map`` followed by ``jnp.einsum`` contraction.

        Parameters:
            forces: Atomic force array of shape ``(n_atoms, 3)``.

        Returns:
            Symmetry-projected forces of shape ``(n_atoms, 3)``.

        Invariants:
            Idempotent and zero-sum-preserving.

        SIESTA Mapping:
            Replaces ``symmetrize_forces`` in ``siesta_forces.F``.
        """
        if self.sg.n_ops == 1:
            return forces
        gathered = forces[self._atom_map_jax]  # (n_ops, n_atoms, 3)
        rotated = jnp.einsum("sba,sib->sia", self._rot_cart_jax, gathered)
        return jnp.mean(rotated, axis=0)

    def project_stress(self, stress: jax.Array) -> jax.Array:
        r"""Project symmetric rank-2 stress tensor onto the point-group invariant subspace.

        Theory:
            Evaluates :math:`\sigma^{\text{sym}} = \frac{1}{|G|} \sum_{S \in G} \mathbf{C}_S \sigma \mathbf{C}_S^T`.

        Algorithm:
            Batched ``jnp.einsum`` similarity transform averaged over all operations and symmetrized.

        Parameters:
            stress: Symmetric Cartesian stress tensor of shape ``(3, 3)``.

        Returns:
            Projected stress tensor of shape ``(3, 3)``.

        Invariants:
            Symmetric and invariant under every :math:`\mathbf{C}_S`.

        SIESTA Mapping:
            Replaces stress tensor symmetrization in ``siesta_forces.F``.
        """
        if self.sg.n_ops == 1:
            return stress
        rotated = jnp.einsum("sac,cd,sbd->sab", self._rot_cart_jax, stress, self._rot_cart_jax)
        s_avg = jnp.mean(rotated, axis=0)
        return 0.5 * (s_avg + s_avg.T)

    def evaluate_irreducible_two_center(
        self,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        config: SCFConfig,
    ) -> TwoCenterMatrices:
        r"""Evaluate C++ two-center integrals once per orbit representative and rotate partners.

        Theory:
            Partitions shell pairs :math:`(\text{sh}_1, \text{sh}_2, \mathbf{T})` and
            Kleinman-Bylander overlap blocks :math:`(\text{sh}_\mu, k_{\text{inst}}, \mathbf{T}_k)`
            into space-group orbits. For each representative :math:`b_0`, evaluates C++
            Fourier-Bessel integrals once and constructs every partner :math:`b'` via

            .. math::
                M(b') = D^{l_1}(S)\, M(b_0)\, D^{l_2}(S)^T, \qquad
                \nabla_{\mathbf{r}'} M(b') = \sum_{a,b} D^{l_1}(S)\, \left(\mathbf{C}_S \nabla_{\mathbf{r}} M(b_0)\right)\, D^{l_2}(S)^T.

        Algorithm:
            Identifies orbit representatives in Python over shell-pair indices, calls
            ``_eval_st_subset`` and ``jsiesta_eval_kb_overlaps`` only on representative sets,
            rotates partner values and Cartesian gradients, and shares ``_eval_kb_and_na_pair``
            for contraction and neutral-atom pairs.

        Parameters:
            structure: ``Structure`` PyTree.
            basis: ``BasisData`` PyTree.
            pattern: Symmetry-closed ``SparsePattern``.
            config: ``SCFConfig``.

        Returns:
            ``TwoCenterMatrices`` numerically matching direct full evaluation.

        Invariants:
            - Records ``last_st_irred_blocks <= last_st_total_blocks``.
            - Preserves :math:`[S, U_S] = 0` and :math:`[H_0, U_S] = 0`.

        SIESTA Mapping:
            Replaces redundant full-edge calls to ``matel`` and ``nlefsm`` with irreducible orbit evaluation.
        """
        shells, lookup = _validate_sequential_shells(basis)
        n_pao_pts, pao_r_max, pao_rcut, pao_l, pao_vals, orb_g_shell = _pack_pao_tables(basis)
        n_edges = pattern.n_edges

        src_np = np.asarray(pattern.src_orb, dtype=np.int32)
        dst_np = np.asarray(pattern.dst_unit_orb, dtype=np.int32)
        eci_np = np.asarray(pattern.edge_cell_idx, dtype=np.int32)
        shifts_np = np.asarray(pattern.cell_shifts, dtype=np.int32)
        vec_np = np.asarray(pattern.edge_vectors, dtype=np.float64)
        orb_m_np = np.asarray(basis.orb_m, dtype=np.int32)
        shift_to_idx = {(int(s[0]), int(s[1]), int(s[2])): i for i, s in enumerate(shifts_np)}

        orb_to_sh = np.empty(basis.n_orb, dtype=np.int32)
        for sh_i, (_, ang_l, _, mu0) in enumerate(shells):
            orb_to_sh[mu0 : mu0 + 2 * ang_l + 1] = sh_i

        # 1. Group all edges into shell-pair blocks (sh1, sh2, c_idx)
        blocks: dict[tuple[int, int, int], np.ndarray] = {}
        for e_i in range(n_edges):
            sh1 = int(orb_to_sh[src_np[e_i]])
            sh2 = int(orb_to_sh[dst_np[e_i]])
            c_idx = int(eci_np[e_i])
            _, l1, _, mu0 = shells[sh1]
            _, l2, _, nu0 = shells[sh2]
            blk = blocks.get((sh1, sh2, c_idx))
            if blk is None:
                blk = np.full((2 * l1 + 1, 2 * l2 + 1), -1, dtype=np.int32)
                blocks[(sh1, sh2, c_idx)] = blk
            blk[src_np[e_i] - mu0, dst_np[e_i] - nu0] = e_i

        self.last_st_total_blocks = len(blocks)

        if self.sg.n_ops == 1:
            self.last_st_irred_blocks = len(blocks)
            out_s, out_t, out_ds, out_dt = _eval_st_subset(
                n_pao_pts,
                pao_r_max,
                pao_rcut,
                pao_l,
                pao_vals,
                orb_g_shell[src_np],
                orb_m_np[src_np],
                orb_g_shell[dst_np],
                orb_m_np[dst_np],
                vec_np,
            )
            return _eval_kb_and_na_pair(
                structure,
                basis,
                pattern,
                config,
                n_pao_pts,
                pao_r_max,
                pao_rcut,
                pao_l,
                pao_vals,
                orb_g_shell,
                out_s,
                out_t,
                out_ds,
                out_dt,
            )

        # 2. Partition (sh1, sh2, c_idx) into space-group orbits: (rep_key -> list[(dst_key, op_s)])
        visited: set[tuple[int, int, int]] = set()
        rep_edges: list[int] = []
        rep_slices: list[tuple[tuple[int, int, int], int, int, int, int]] = []
        orbit_members: list[list[tuple[tuple[int, int, int], int]]] = []

        for b_key, blk in blocks.items():
            if b_key in visited:
                continue
            sh1, sh2, c_idx = b_key
            ia1, l1, iz1, _ = shells[sh1]
            ia2, l2, iz2, _ = shells[sh2]
            t_vec = shifts_np[c_idx]
            start_e = len(rep_edges)
            rep_edges.extend(blk.ravel().tolist())
            end_e = len(rep_edges)
            rep_slices.append((b_key, start_e, end_e, 2 * l1 + 1, 2 * l2 + 1))

            members: list[tuple[tuple[int, int, int], int]] = []
            for s in range(self.sg.n_ops):
                ja1 = int(self.sg.atom_map[s, ia1])
                ja2 = int(self.sg.atom_map[s, ia2])
                sh1_p = lookup[(ja1, l1, iz1)]
                sh2_p = lookup[(ja2, l2, iz2)]
                tp = self.sg.rotations_frac[s] @ t_vec + self.sg.atom_translation[s, ia2] - self.sg.atom_translation[s, ia1]
                c_idx_p = shift_to_idx[(int(tp[0]), int(tp[1]), int(tp[2]))]
                dst_key = (sh1_p, sh2_p, c_idx_p)
                if dst_key not in visited:
                    visited.add(dst_key)
                    members.append((dst_key, s))
            orbit_members.append(members)

        self.last_st_irred_blocks = len(rep_slices)
        rep_idx = np.asarray(rep_edges, dtype=np.int32)
        rep_s, rep_t, rep_ds, rep_dt = _eval_st_subset(
            n_pao_pts,
            pao_r_max,
            pao_rcut,
            pao_l,
            pao_vals,
            orb_g_shell[src_np[rep_idx]],
            orb_m_np[src_np[rep_idx]],
            orb_g_shell[dst_np[rep_idx]],
            orb_m_np[dst_np[rep_idx]],
            vec_np[rep_idx],
        )

        # Rotate representative matrices & gradients to every orbit partner
        out_s = np.zeros(n_edges, dtype=np.float64)
        out_t = np.zeros(n_edges, dtype=np.float64)
        out_ds = np.zeros((n_edges, 3), dtype=np.float64)
        out_dt = np.zeros((n_edges, 3), dtype=np.float64)

        for (b_key, st, en, d1, d2), members in zip(rep_slices, orbit_members, strict=True):
            sh1, sh2, _ = b_key
            l1 = shells[sh1][1]
            l2 = shells[sh2][1]
            s_mat = rep_s[st:en].reshape(d1, d2)
            t_mat = rep_t[st:en].reshape(d1, d2)
            ds_mat = rep_ds[st:en].reshape(d1, d2, 3)
            dt_mat = rep_dt[st:en].reshape(d1, d2, 3)

            for dst_key, s in members:
                blk_dst = blocks[dst_key]
                e_flat = blk_dst.ravel()
                if s == 0:
                    out_s[e_flat] = s_mat.ravel()
                    out_t[e_flat] = t_mat.ravel()
                    out_ds[e_flat] = ds_mat.reshape(d1 * d2, 3)
                    out_dt[e_flat] = dt_mat.reshape(d1 * d2, 3)
                else:
                    dm1 = self.wigner_d[l1][s]
                    dm2 = self.wigner_d[l2][s]
                    c_s = self.sg.rotations_cart[s]
                    out_s[e_flat] = (dm1 @ s_mat @ dm2.T).ravel()
                    out_t[e_flat] = (dm1 @ t_mat @ dm2.T).ravel()
                    # Rotate both (m1, m2) via D^l and spatial derivative vector via C_s
                    out_ds[e_flat] = np.einsum("am,bn,cd,mnd->abc", dm1, dm2, c_s, ds_mat).reshape(d1 * d2, 3)
                    out_dt[e_flat] = np.einsum("am,bn,cd,mnd->abc", dm1, dm2, c_s, dt_mat).reshape(d1 * d2, 3)

        # 3. Irreducible KB projector orbit evaluator callback
        def _eval_kb_irred(
            n_p_pts: int,
            p_rmax: np.ndarray,
            p_rcut: np.ndarray,
            p_l: np.ndarray,
            p_vals: np.ndarray,
            n_k_pts: int,
            k_rmax: np.ndarray,
            k_rcut: np.ndarray,
            k_l: np.ndarray,
            k_vals: np.ndarray,
            atom_kb_shells: list[tuple[int, int, float, int, int, float]],
            link_mu: list[int],
            link_k_inst: list[int],
            link_tk: list[tuple[int, int, int]],
            link_m2_idx: list[int],
            l_pao_sh: np.ndarray,
            l_m1: np.ndarray,
            l_kb_sh: np.ndarray,
            l_m2: np.ndarray,
            l_vec: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            n_links = len(link_mu)
            # Map (atom_K, global_kb_shell_id) -> k_inst_idx
            ak_to_inst = {(ka, g_ksh): idx for idx, (ka, g_ksh, _, _, _, _) in enumerate(atom_kb_shells)}

            # Group links into blocks by (sh_mu, k_inst, tk) -> matrix (2*l_mu+1, 2*l_k+1) of link indices
            kb_blocks: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
            for i in range(n_links):
                mu = link_mu[i]
                sh_mu = int(orb_to_sh[mu])
                k_i = link_k_inst[i]
                tk = link_tk[i]
                l_mu = shells[sh_mu][1]
                mu0 = shells[sh_mu][3]
                l_k_val = atom_kb_shells[k_i][3]
                key = (sh_mu, k_i, tk)
                blk = kb_blocks.get(key)
                if blk is None:
                    blk = np.full((2 * l_mu + 1, 2 * l_k_val + 1), -1, dtype=np.int32)
                    kb_blocks[key] = blk
                blk[mu - mu0, link_m2_idx[i]] = i

            self.last_kb_total_blocks = len(kb_blocks)
            kb_visited: set[tuple[int, int, tuple[int, int, int]]] = set()
            rep_link_idx: list[int] = []
            kb_rep_slices: list[tuple[int, int, int, int, int, int]] = []
            kb_orbits: list[list[tuple[tuple[int, int, tuple[int, int, int]], int]]] = []

            for kkey, blk in kb_blocks.items():
                if kkey in kb_visited or np.any(blk < 0):
                    continue
                sh_mu, k_i, tk = kkey
                ia_mu, l_mu, iz_mu, _ = shells[sh_mu]
                ka, g_ksh, _, l_k_val, _, _ = atom_kb_shells[k_i]
                tk_arr = np.asarray(tk, dtype=np.int32)
                st_i = len(rep_link_idx)
                rep_link_idx.extend(blk.ravel().tolist())
                en_i = len(rep_link_idx)
                kb_rep_slices.append((l_mu, l_k_val, st_i, en_i, 2 * l_mu + 1, 2 * l_k_val + 1))

                mems: list[tuple[tuple[int, int, tuple[int, int, int]], int]] = []
                for s in range(self.sg.n_ops):
                    ja_mu = int(self.sg.atom_map[s, ia_mu])
                    ja_k = int(self.sg.atom_map[s, ka])
                    sh_mu_p = lookup[(ja_mu, l_mu, iz_mu)]
                    k_i_p = ak_to_inst.get((ja_k, g_ksh))
                    if k_i_p is None:
                        continue
                    tk_p = self.sg.rotations_frac[s] @ tk_arr + self.sg.atom_translation[s, ka] - self.sg.atom_translation[s, ia_mu]
                    dkey = (sh_mu_p, k_i_p, (int(tk_p[0]), int(tk_p[1]), int(tk_p[2])))
                    if dkey in kb_blocks and dkey not in kb_visited and not np.any(kb_blocks[dkey] < 0):
                        kb_visited.add(dkey)
                        mems.append((dkey, s))
                kb_orbits.append(mems)

            # Include any partial/edge-clipped links that weren't full shell blocks safely in rep_link_idx
            covered_links = np.zeros(n_links, dtype=bool)
            for mems in kb_orbits:
                for dkey, _ in mems:
                    covered_links[kb_blocks[dkey].ravel()] = True
            missing_links = np.where(~covered_links)[0].tolist()
            n_rep_main = len(rep_link_idx)
            all_eval_idx = np.asarray(rep_link_idx + missing_links, dtype=np.int32)
            self.last_kb_irred_blocks = len(kb_rep_slices)

            n_eval = int(all_eval_idx.shape[0])
            eval_p = np.zeros(n_eval, dtype=np.float64)
            eval_dp = np.zeros((n_eval, 3), dtype=np.float64)
            lib = load_ffi_library()
            _setup_c_signatures(lib)
            if n_eval > 0:
                rc = lib.jsiesta_eval_kb_overlaps(
                    ctypes.c_int32(int(p_vals.shape[0])),
                    ctypes.c_int32(n_p_pts),
                    p_rmax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    p_rcut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    p_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    p_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_int32(int(k_vals.shape[0])),
                    ctypes.c_int32(n_k_pts),
                    k_rmax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    k_rcut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    k_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    k_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_int32(n_eval),
                    np.ascontiguousarray(l_pao_sh[all_eval_idx], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    np.ascontiguousarray(l_m1[all_eval_idx], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    np.ascontiguousarray(l_kb_sh[all_eval_idx], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    np.ascontiguousarray(l_m2[all_eval_idx], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    np.ascontiguousarray(l_vec[all_eval_idx], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    eval_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    eval_dp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
                if rc != 0:
                    raise RuntimeError(f"jsiesta_eval_kb_overlaps failed with code {rc}")

            out_p = np.zeros(n_links, dtype=np.float64)
            out_dp = np.zeros((n_links, 3), dtype=np.float64)
            for (l_mu, l_k_val, st_i, en_i, d1, d2), mems in zip(kb_rep_slices, kb_orbits, strict=True):
                p_mat = eval_p[st_i:en_i].reshape(d1, d2)
                dp_mat = eval_dp[st_i:en_i].reshape(d1, d2, 3)
                for dkey, s in mems:
                    l_dst = kb_blocks[dkey].ravel()
                    if s == 0:
                        out_p[l_dst] = p_mat.ravel()
                        out_dp[l_dst] = dp_mat.reshape(d1 * d2, 3)
                    else:
                        dm1 = self.wigner_d[l_mu][s]
                        dmk = self.wigner_d[l_k_val][s]
                        c_s = self.sg.rotations_cart[s]
                        out_p[l_dst] = (dm1 @ p_mat @ dmk.T).ravel()
                        out_dp[l_dst] = np.einsum("am,bn,cd,mnd->abc", dm1, dmk, c_s, dp_mat).reshape(d1 * d2, 3)

            if missing_links:
                m_idx = np.asarray(missing_links, dtype=np.int32)
                out_p[m_idx] = eval_p[n_rep_main:]
                out_dp[m_idx] = eval_dp[n_rep_main:]
            return out_p, out_dp

        return _eval_kb_and_na_pair(
            structure,
            basis,
            pattern,
            config,
            n_pao_pts,
            pao_r_max,
            pao_rcut,
            pao_l,
            pao_vals,
            orb_g_shell,
            out_s,
            out_t,
            out_ds,
            out_dt,
            eval_kb_links_fn=_eval_kb_irred,
        )


def build_native_symmetry(
    structure: Structure,
    basis: BasisData,
    k_grid: tuple[int, int, int],
    config: SCFConfig,
    k_shifts: tuple[float, float, float] = (0.0, 0.0, 0.0),
    symprec: float | None = None,
    mag_symprec: float | None = None,
    mesh_shape_override: tuple[int, int, int] | None = None,
) -> tuple[NativeSymmetryEngine, SparsePattern, KPointMesh, tuple[int, int, int]]:
    r"""Construct complete native symmetry suite ``(engine, pattern, ibz_kmesh, mesh_shape)``.

    Theory:
        Orchestrates Steps 1--2 at geometry startup: extracts Shubnikov/space group,
        reduces the Brillouin zone to the IBZ, chooses symmetry-commensurate FFT mesh
        dimensions, precomputes grid permutations :math:`\pi_S`, closes the sparse pattern,
        validates 16-pt Fibonacci Wigner-D orthogonality, and compiles the sparse COO
        projection table for ``jax.ops.segment_sum``.

    Algorithm:
        Executes Steps 1--2 sequentially with zero mutable global state.

    Parameters:
        structure: ``Structure`` PyTree.
        basis: ``BasisData`` PyTree.
        k_grid: Monkhorst-Pack grid counts ``(nk1, nk2, nk3)``.
        config: ``SCFConfig`` hyperparameters.
        k_shifts: Fractional k-mesh offsets (default ``(0.0, 0.0, 0.0)``).
        symprec: Optional symmetry tolerance in Bohr (defaults to ``config.symprec``).
        mag_symprec: Optional magnetic tolerance in :math:`\mu_B` (defaults to ``config.mag_symprec``).
        mesh_shape_override: Explicit mesh (e.g. SIESTA's, for matched parity runs). It must be
            commensurate with the group (``check_mesh_commensurate``); otherwise ``ValueError``.

    Returns:
        Tuple ``(engine, pattern, ibz_kmesh, commensurate_mesh_shape)``.

    Invariants:
        - ``pattern.symmetry_closed == True``.
        - ``ibz_kmesh.is_ibz == True`` and ``sum(ibz_kmesh.weights) == 1.0``.

    SIESTA Mapping:
        Unified native symmetry setup replacing ``m_symmetry``, ``kpoint_scf``, and ``m_dm_symmetrize``.
    """
    eff_symprec = config.symprec if symprec is None else symprec
    eff_magsym = config.mag_symprec if mag_symprec is None else mag_symprec
    sg = extract_space_group(structure, config, symprec=eff_symprec, mag_symprec=eff_magsym)
    kmesh = build_ibz_kpoint_mesh(sg, k_grid, shifts=k_shifts)
    if mesh_shape_override is not None:
        mesh_shape = (int(mesh_shape_override[0]), int(mesh_shape_override[1]), int(mesh_shape_override[2]))
        check_mesh_commensurate(sg, mesh_shape)
    else:
        mesh_shape = choose_commensurate_mesh_dimensions(
            structure,
            config.mesh_cutoff_ry,
            sg,
            criterion=config.mesh_criterion,
            mesh_subdivisions=config.mesh_subdivisions,
        )
    perm_fwd, perm_inv = precompute_grid_permutations(sg, mesh_shape)
    pattern = build_symmetry_closed_pattern(structure, basis, sg)

    l_max = max(basis.orb_l) if basis.orb_l else 0
    for sp in basis.species:
        if sp.kb_l:
            l_max = max(l_max, max(sp.kb_l))

    wigner_d = compute_wigner_d_matrices(sg.rotations_cart, l_max=l_max, n_fib_pts=16, orth_tol=1e-10)
    proj_table = compile_projection_table(basis, pattern, sg, wigner_d, nspin=config.nspin)

    # Precompute static JAX arrays for JIT/grad-compatible grid/force/stress symmetrization
    axial_rot = np.where(sg.time_reversals[:, None, None], -1.0, 1.0) * sg.det_cart[:, None, None] * sg.rotations_cart
    trev_sign = np.where(sg.time_reversals, -1.0, 1.0)

    engine = NativeSymmetryEngine(
        sg=sg,
        basis=basis,
        pattern=pattern,
        mesh_shape=mesh_shape,
        perm_fwd=perm_fwd,
        perm_inv=perm_inv,
        wigner_d=wigner_d,
        proj_table=proj_table,
        nspin=config.nspin,
        _perm_inv_jax=jnp.asarray(perm_inv, dtype=jnp.int32),
        _rot_cart_jax=jnp.asarray(sg.rotations_cart, dtype=jnp.float64),
        _axial_rot_jax=jnp.asarray(axial_rot, dtype=jnp.float64),
        _trev_sign_jax=jnp.asarray(trev_sign, dtype=jnp.float64),
        _atom_map_jax=jnp.asarray(sg.atom_map, dtype=jnp.int32),
        grid_symmetrization=bool(getattr(config, "symmetrize_grid", True)),
    )
    return engine, pattern, kmesh, mesh_shape
