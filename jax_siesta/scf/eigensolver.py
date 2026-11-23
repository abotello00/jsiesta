r"""Batched generalized Kohn-Sham eigensolver and density/energy-density matrix assembly.

Theory:
    Solves the generalized eigenvalue problem at each Brillouin-zone wavevector
    :math:`\mathbf{k}_q`:

    .. math::
        H(\mathbf{k}_q) C(\mathbf{k}_q) = S(\mathbf{k}_q) C(\mathbf{k}_q) \varepsilon(\mathbf{k}_q),

    across all four spin modes (``nspin = 1, 2, 4, 8``), determines the chemical
    potential :math:`\mu_F` via Fermi-Dirac smearing so total electron count
    :math:`\sum_{q,n} w_q g_s f(\varepsilon_{qn} - \mu_F) = N_e`, and transforms
    back to real-space sparse density matrix :math:`D` and energy-density matrix :math:`E`:

    .. math::
        D_{\mu\nu}(\mathbf{T}) = \sum_q w_q e^{-i\mathbf{k}_q\cdot\mathbf{T}} \sum_n g_s f_{qn} C_{\mu n}(\mathbf{k}_q) C_{\nu n}^*(\mathbf{k}_q),
        \qquad
        E_{\mu\nu}(\mathbf{T}) = \sum_q w_q e^{-i\mathbf{k}_q\cdot\mathbf{T}} \sum_n g_s f_{qn} \varepsilon_{qn} C_{\mu n}(\mathbf{k}_q) C_{\nu n}^*(\mathbf{k}_q).

Algorithm:
    Uses ``jax.vmap`` over k-points, Cholesky orthogonalization :math:`S = L L^\dagger`
    with symmetric Löwdin fallback (computed once per SCF, ``overlap_transform``),
    ``jnp.linalg.eigh`` of :math:`L^{-1} H L^{-\dagger}` (real arithmetic when all ``2k`` are
    integer and ``nspin`` is 1 or 2), and 64-step vectorized bisection for :math:`\mu_F`.

Invariants:
    - :math:`C(\mathbf{k})^\dagger S(\mathbf{k}) C(\mathbf{k}) = I`.
    - :math:`\mathrm{Tr}[D S] = N_e` and :math:`\mathrm{Tr}[E S] = E_{\text{band}}`.

SIESTA Mapping:
    Implements ``EigenSolver`` replacing ``diagon.F`` and ``compute_dm.F``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import KPointMesh, SparsePattern


def _assembly_plan(src_orb: jax.Array, dst_orb: jax.Array, n_orb: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Host index tables that turn the edge-to-dense scatter-add into two gathers; ``None`` for traced patterns.

    ``groups[u, j]`` lists the edges of the ``u``-th distinct ``(src, dst)`` pair in edge order (padded with
    ``n_edges``, a zero slot); ``dense[mu * n_orb + nu]`` is that pair's ``u`` (or ``n_pairs``, a zero slot).
    """
    try:
        src = np.asarray(src_orb, dtype=np.int64)
        dst = np.asarray(dst_orb, dtype=np.int64)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return None
    n_edges = int(src.shape[0])
    flat = src * n_orb + dst
    order = np.argsort(flat, kind="stable")
    uniq, start, counts = np.unique(flat[order], return_index=True, return_counts=True)
    m_max = int(counts.max()) if counts.size else 1
    groups = np.full((uniq.shape[0], m_max), n_edges, dtype=np.int64)
    rows = np.repeat(np.arange(uniq.shape[0], dtype=np.int64), counts)
    cols = np.arange(n_edges, dtype=np.int64) - np.repeat(start, counts)
    groups[rows, cols] = order
    dense = np.full(n_orb * n_orb, uniq.shape[0], dtype=np.int64)
    dense[uniq] = np.arange(uniq.shape[0], dtype=np.int64)
    return groups, dense


def _bloch_assemble(
    edge_vals: jax.Array,
    src_orb: jax.Array,
    dst_orb: jax.Array,
    edge_shifts: jax.Array,
    kpoints_frac: jax.Array,
    n_orb: int,
) -> jax.Array:
    """Assemble ``(n_k, n_orb, n_orb)`` complex matrices from sparse real/complex edges.

    With a concrete pattern the periodic images of each ``(mu, nu)`` pair are summed through a padded gather and placed
    by a second gather (``_assembly_plan``); the XLA CPU scatter-add with repeated indices costs microseconds per edge.
    """
    phases = jnp.exp(2j * jnp.pi * (kpoints_frac @ edge_shifts.T))  # (n_k, n_edges)
    weighted = phases * edge_vals[None, :]  # (n_k, n_edges)
    n_k = kpoints_frac.shape[0]
    plan = _assembly_plan(src_orb, dst_orb, n_orb)
    if plan is None:
        mats = jnp.zeros((n_k, n_orb, n_orb), dtype=jnp.complex128)
        return mats.at[:, src_orb, dst_orb].add(weighted)
    groups, dense = plan
    zero = jnp.zeros((n_k, 1), dtype=weighted.dtype)
    pair_sums = jnp.sum(jnp.concatenate([weighted, zero], axis=1)[:, groups], axis=2)  # (n_k, n_pairs)
    return jnp.concatenate([pair_sums, zero], axis=1)[:, dense].reshape(n_k, n_orb, n_orb).astype(jnp.complex128)


def _overlap_transform_single(s_k: jax.Array) -> jax.Array:
    """Orthogonalising transform ``X`` with ``X^dagger S X = I``: ``X = L^-dagger`` (Cholesky) or ``S^-1/2`` (Löwdin).

    The Löwdin branch (eigenvalue floor ``1e-10``) is taken when ``min eig(S) <= 1e-8``.
    """
    s_sym = 0.5 * (s_k + s_k.conj().T)
    s_eigs, s_vecs = jnp.linalg.eigh(s_sym)
    min_s = jnp.min(s_eigs)

    def _cholesky_path() -> jax.Array:
        l_mat = jnp.linalg.cholesky(s_sym)
        l_inv = jnp.linalg.inv(l_mat)
        return l_inv.conj().T

    def _lowdin_path() -> jax.Array:
        inv_sqrt = jnp.where(s_eigs > 1e-10, 1.0 / jnp.sqrt(jnp.maximum(s_eigs, 1e-10)), 0.0)
        return (s_vecs * inv_sqrt[None, :]) @ s_vecs.conj().T

    return jax.lax.cond(min_s > 1e-8, _cholesky_path, _lowdin_path)


def _solve_with_transform_single(h_k: jax.Array, x_k: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve ``H C = S C eps`` given the orthogonalising transform ``X`` of ``S``: ``eigh(X^dagger H X)``, ``C = X U``."""
    h_sym = 0.5 * (h_k + h_k.conj().T)
    h_ortho = x_k.conj().T @ h_sym @ x_k
    h_ortho = 0.5 * (h_ortho + h_ortho.conj().T)
    eps, u = jnp.linalg.eigh(h_ortho)
    return eps, x_k @ u


def _solve_generalized_eigh_single(h_k: jax.Array, s_k: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve ``H C = S C eps`` via Cholesky orthogonalization (with Löwdin eigendecomp fallback)."""
    return _solve_with_transform_single(h_k, _overlap_transform_single(s_k))


def _map_k(fn, args: tuple[jax.Array, ...], k_batch_size: int | None):  # type: ignore[no-untyped-def]
    n_k = int(args[0].shape[0])
    if k_batch_size is None or k_batch_size <= 0 or n_k <= k_batch_size:
        return jax.vmap(fn)(*args)
    return jax.lax.map(lambda a: fn(*a), args, batch_size=int(k_batch_size))


def batched_overlap_transform(s_k: jax.Array, k_batch_size: int | None = None) -> jax.Array:
    r"""Orthogonalising transforms :math:`X(\mathbf{k})` with :math:`X^\dagger S X = I` for a batch of overlaps.

    Theory:
        For a Hermitian positive-definite :math:`S`, :math:`X = L^{-\dagger}` with :math:`S = L L^\dagger` turns
        :math:`H C = S C \varepsilon` into the standard problem :math:`(X^\dagger H X) U = U \varepsilon`,
        :math:`C = X U`. For a near-singular :math:`S` the symmetric Löwdin form :math:`X = S^{-1/2}` (eigenvalue floor
        :math:`10^{-10}`) is used instead.

    Algorithm:
        Per k-point: ``eigh(S)`` for the smallest eigenvalue, then the Cholesky branch when it exceeds ``1e-8`` and the
        Löwdin branch otherwise (``jax.lax.cond``). ``S`` does not change during an SCF, so the driver computes this
        once and passes it to every ``GeneralizedEigenSolver.solve`` call.

    Parameters:
        s_k: Hermitian overlaps of shape ``(n_k, m, m)`` (real symmetric or complex Hermitian).
        k_batch_size: Optional maximum k-points per vectorized batch.

    Returns:
        ``X`` of shape ``(n_k, m, m)`` and the dtype of ``s_k``.

    Invariants:
        - :math:`X^\dagger S X = I` on the Cholesky branch; :math:`X^\dagger S X` is the projector on the retained
          eigenspace on the Löwdin branch.
        - The result equals the transform built inside ``batched_generalized_eigh`` for the same ``s_k``.

    SIESTA Mapping:
        Corresponds to the overlap factorisation step of ``rdiag``/``cdiag`` (``Src/diag``), done there on every call.
    """
    return _map_k(_overlap_transform_single, (s_k,), k_batch_size)


def _fermi_dirac(eps: jax.Array, mu: jax.Array, temp_ha: float) -> jax.Array:
    """Numerically stable Fermi-Dirac occupation in ``[0, 1]``."""
    t_eff = jnp.maximum(temp_ha, 1e-8)
    x = jnp.clip((eps - mu) / t_eff, -60.0, 60.0)
    return 1.0 / (1.0 + jnp.exp(x))


def _solve_fermi_level(
    evals: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    temp_ha: float,
    g_spin: float,
) -> jax.Array:
    """Determine chemical potential ``mu_F`` via 64-step bisection."""
    e_min = jnp.min(evals) - 20.0 * jnp.maximum(temp_ha, 1e-3) - 1.0
    e_max = jnp.max(evals) + 20.0 * jnp.maximum(temp_ha, 1e-3) + 1.0

    def _body(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        occ = _fermi_dirac(evals, mid, temp_ha)
        n_est = jnp.sum(weights[:, None] * g_spin * occ)
        lo_next = jnp.where(n_est < n_electrons, mid, lo)
        hi_next = jnp.where(n_est < n_electrons, hi, mid)
        return (lo_next, hi_next)

    lo_f, hi_f = jax.lax.fori_loop(0, 64, _body, (e_min, e_max))
    return 0.5 * (lo_f + hi_f)


def _entropy_term(occ: jax.Array, weights: jax.Array, temp_ha: float, g_spin: float) -> jax.Array:
    """Compute electronic entropy contribution ``-T * S_el`` (Hartree, negative)."""
    occ_clip = jnp.clip(occ, 1e-15, 1.0 - 1e-15)
    s_per_state = -(occ_clip * jnp.log(occ_clip) + (1.0 - occ_clip) * jnp.log(1.0 - occ_clip))
    return -temp_ha * jnp.sum(weights[:, None] * g_spin * s_per_state)


def batched_generalized_eigh(
    h_k: jax.Array,
    s_k: jax.Array,
    k_batch_size: int | None = None,
    x_k: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Solve batched :math:`H(\mathbf{k}) C = S(\mathbf{k}) C \varepsilon` with configurable k-batching.

    Theory:
        Diagonalizes independent Hermitian pencils across Brillouin-zone wavevectors
        :math:`\mathbf{k}_1, \dots, \mathbf{k}_{N_k}`.

    Algorithm:
        Executes full ``jax.vmap`` when ``k_batch_size is None`` or ``n_k <= k_batch_size``,
        otherwise dispatches bounded batches via ``jax.lax.map(..., batch_size=k_batch_size)``
        to cap peak workspace memory on GPU/CPU. The orthogonalising transform of ``S`` is
        ``x_k`` when given (``batched_overlap_transform``), otherwise it is computed here; either way
        one ``eigh`` of :math:`X^\dagger H X` follows per k-point.

    Parameters:
        h_k: Complex Hermitian Hamiltonians of shape ``(n_k, m, m)``.
        s_k: Complex Hermitian positive-definite overlaps of shape ``(n_k, m, m)``.
        k_batch_size: Optional maximum k-points per vectorized batch.
        x_k: Optional precomputed ``batched_overlap_transform(s_k)``; ``s_k`` is then not used.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` of shapes ``(n_k, m)`` and ``(n_k, m, m)``.

    Invariants:
        - Output eigenvalues and eigenvectors agree across batch sizes to machine precision.
        - Preserves :math:`C(\mathbf{k})^\dagger S(\mathbf{k}) C(\mathbf{k}) = I`.

    SIESTA Mapping:
        Batched k-point diagonalization driver replacing ``diagon.F`` k-point loops.
    """
    if x_k is None:
        x_k = batched_overlap_transform(s_k, k_batch_size)
    return _map_k(_solve_with_transform_single, (h_k, x_k), k_batch_size)  # type: ignore[no-any-return]


def _real_bloch_problem(kpts: jax.Array, nspin: int) -> bool:
    """True when every k-point has ``2k`` integer and ``nspin`` is 1 or 2: then ``H(k)`` and ``S(k)`` are real.

    Needs concrete k-points (a closed-over mesh); for traced k-points the complex path is kept.
    """
    if nspin not in (1, 2):
        return False
    try:
        k_np = np.asarray(kpts, dtype=np.float64)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return False
    return bool(np.all(2.0 * k_np == np.round(2.0 * k_np)))


class GeneralizedEigenSolver:
    r"""Batched JAX generalized Kohn-Sham eigensolver implementing ``EigenSolver``.

    Theory:
        Solves :math:`H(\mathbf{k})C = S(\mathbf{k})C\varepsilon` across all
        ``nspin in (1, 2, 4, 8)`` sectors using exact Bloch phases and Fermi-Dirac smearing.

    Algorithm:
        Batches over k-points via ``jax.vmap``, builds spin-1/2 or 2-spinor matrices,
        diagonalizes, solves for :math:`\mu_F`, and contracts occupied eigenstates back
        onto the sparse real-space edge list.

    Invariants:
        - :math:`C^\dagger S C = I` and :math:`\mathrm{Tr}[D S] = N_e`.
        - Output ``dm_out`` and ``edm_out`` preserve exact Hermitian symmetry.

    SIESTA Mapping:
        Implements ``EigenSolver`` replacing ``diagon.F`` and ``compute_dm.F``.
    """

    def __init__(self, k_batch_size: int | None = None) -> None:
        self.k_batch_size = k_batch_size

    def _overlap_blocks(
        self, s_sparse: jax.Array, pattern: SparsePattern, kmesh: KPointMesh, nspin: int, real_k: bool
    ) -> jax.Array:
        """Bloch overlaps ``S(k)`` (``(n_k, N, N)``), or ``diag(S, S)`` (``(n_k, 2N, 2N)``) for ``nspin = 4, 8``."""
        n_orb = pattern.n_orb
        e_shifts = pattern.cell_shifts[pattern.edge_cell_idx].astype(jnp.float64)
        s_k = _bloch_assemble(
            s_sparse.astype(jnp.complex128), pattern.src_orb, pattern.dst_unit_orb, e_shifts, kmesh.kpoints_frac, n_orb
        )
        if nspin in (4, 8):
            z_k = jnp.zeros_like(s_k)
            return jnp.concatenate(
                [jnp.concatenate([s_k, z_k], axis=2), jnp.concatenate([z_k, s_k], axis=2)],
                axis=1,
            )
        return jnp.real(s_k) if real_k else s_k

    def overlap_transform(
        self,
        s_sparse: jax.Array,
        pattern: SparsePattern,
        kmesh: KPointMesh,
        nspin: int,
        k_batch_size: int | None = None,
    ) -> jax.Array:
        r"""Orthogonalising transforms of the Bloch overlaps, for reuse across the calls of ``solve``.

        Theory:
            :math:`X(\mathbf{k})` with :math:`X^\dagger S(\mathbf{k}) X = I` (see ``batched_overlap_transform``);
            :math:`S` is fixed during an SCF, so :math:`X` is too.

        Algorithm:
            Assembles :math:`S(\mathbf{k})` (block-diagonal :math:`2N \times 2N` for ``nspin = 4, 8``; real when
            every ``2k`` is integer and ``nspin`` is 1 or 2, as in ``solve``) and calls ``batched_overlap_transform``.

        Parameters:
            s_sparse: Sparse overlap ``(n_edges,)``.
            pattern: Static ``SparsePattern``.
            kmesh: ``KPointMesh`` PyTree.
            nspin: Spin mode ``1, 2, 4, 8``.
            k_batch_size: Optional maximum k-points per vectorized batch.

        Returns:
            ``(n_k, M, M)`` transforms, ``M = N`` or ``2N``.

        Invariants:
            - ``solve(..., s_transform=overlap_transform(...))`` equals ``solve(...)`` without it.

        SIESTA Mapping:
            The overlap factorisation of ``rdiag``/``cdiag``, hoisted out of the SCF loop.
        """
        real_k = _real_bloch_problem(kmesh.kpoints_frac, nspin)
        eff_batch = self.k_batch_size if k_batch_size is None else k_batch_size
        return batched_overlap_transform(self._overlap_blocks(s_sparse, pattern, kmesh, nspin, real_k), eff_batch)

    def solve(
        self,
        h_sparse: jax.Array,
        s_sparse: jax.Array,
        pattern: SparsePattern,
        kmesh: KPointMesh,
        n_electrons: float,
        electronic_temp_ha: float,
        nspin: int,
        k_batch_size: int | None = None,
        s_transform: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Solve generalized KS equations and return ``(dm_out, edm_out, fermi_ha, e_band_ha, entropy_ha)``.

        Theory:
            Constructs :math:`H(\mathbf{k}), S(\mathbf{k})` for ``nspin = 1, 2, 4, 8``,
            finds occupations at ``electronic_temp_ha``, and evaluates sparse real-space
            density and energy-density matrices.

        Algorithm:
            Executes compiled ``jax.vmap`` eigensolution and edge back-projection. With
            ``s_transform`` (from ``overlap_transform``, computed once per SCF) each call costs one
            ``eigh`` per k-point and spin; without it the transform of ``S`` is rebuilt here. When every
            ``2k`` is integer and ``nspin`` is 1 or 2, :math:`H(\mathbf{k})` and :math:`S(\mathbf{k})` are real
            and the problem is solved in real arithmetic.

        Parameters:
            h_sparse: Sparse Hamiltonian ``(n_spin_mat, n_edges)`` (for ``nspin=8``,
                contains real/imag blocks ``[Re H_uu, Re H_dd, Re H_ud, Im H_ud]`` or
                with explicit imaginary diagonal handled via ``h_so_imag``).
            s_sparse: Sparse overlap matrix ``(n_edges,)``.
            pattern: Static ``SparsePattern``.
            kmesh: ``KPointMesh`` PyTree.
            n_electrons: Target total electron count :math:`N_e`.
            electronic_temp_ha: Electronic smearing temperature :math:`k_B T_e` (Hartree).
            nspin: Spin mode ``1, 2, 4, 8``.
            k_batch_size: Optional maximum k-points per vectorized batch.
            s_transform: Optional ``overlap_transform(s_sparse, pattern, kmesh, nspin)``.

        Returns:
            Tuple ``(dm_out, edm_out, fermi_level_ha, e_band_ha, entropy_ha)``.

        Invariants:
            - ``dm_out.shape == h_sparse.shape``.
            - ``Tr[dm_out S] == n_electrons``.

        SIESTA Mapping:
            Replaces ``diagon.F`` and ``compute_dm.F``.
        """
        n_orb = pattern.n_orb
        src = pattern.src_orb
        dst = pattern.dst_unit_orb
        e_shifts = pattern.cell_shifts[pattern.edge_cell_idx].astype(jnp.float64)
        kpts = kmesh.kpoints_frac
        wts = kmesh.weights

        phases_conj = jnp.exp(-2j * jnp.pi * (kpts @ e_shifts.T))  # (n_k, n_edges)

        eff_batch = self.k_batch_size if k_batch_size is None else k_batch_size
        real_k = _real_bloch_problem(kpts, nspin)
        x_k = s_transform
        if x_k is None:
            x_k = batched_overlap_transform(self._overlap_blocks(s_sparse, pattern, kmesh, nspin, real_k), eff_batch)

        def _h(edges: jax.Array) -> jax.Array:
            h_mat = _bloch_assemble(edges.astype(jnp.complex128), src, dst, e_shifts, kpts, n_orb)
            return jnp.real(h_mat) if real_k else h_mat

        if nspin == 1:
            h_k = _h(h_sparse[0])
            evals, evecs = batched_generalized_eigh(h_k, x_k, k_batch_size=eff_batch, x_k=x_k)  # (n_k, N), (n_k, N, N)
            mu_f = _solve_fermi_level(evals, wts, n_electrons, electronic_temp_ha, g_spin=2.0)
            occ = _fermi_dirac(evals, mu_f, electronic_temp_ha)  # (n_k, N)
            e_band = jnp.sum(wts[:, None] * 2.0 * occ * evals)
            entropy = _entropy_term(occ, wts, electronic_temp_ha, g_spin=2.0)

            w_occ = 2.0 * wts[:, None] * occ  # (n_k, N)
            w_eocc = w_occ * evals
            # D_k[mu, nu] = sum_n w_occ[n] C[mu, n] C^*[nu, n]
            d_k = jnp.einsum("kn,kmn,kpn->kmp", w_occ, evecs, jnp.conj(evecs))
            e_k = jnp.einsum("kn,kmn,kpn->kmp", w_eocc, evecs, jnp.conj(evecs))
            dm_edge = jnp.real(jnp.sum(phases_conj * d_k[:, src, dst], axis=0))[None, :]
            edm_edge = jnp.real(jnp.sum(phases_conj * e_k[:, src, dst], axis=0))[None, :]
            return dm_edge, edm_edge, mu_f, e_band, entropy

        if nspin == 2:
            h_u = _h(h_sparse[0])
            h_d = _h(h_sparse[1])
            evals_u, evecs_u = batched_generalized_eigh(h_u, x_k, k_batch_size=eff_batch, x_k=x_k)
            evals_d, evecs_d = batched_generalized_eigh(h_d, x_k, k_batch_size=eff_batch, x_k=x_k)
            evals_both = jnp.concatenate([evals_u, evals_d], axis=1)
            mu_f = _solve_fermi_level(evals_both, wts, n_electrons, electronic_temp_ha, g_spin=1.0)
            occ_u = _fermi_dirac(evals_u, mu_f, electronic_temp_ha)
            occ_d = _fermi_dirac(evals_d, mu_f, electronic_temp_ha)
            e_band = jnp.sum(wts[:, None] * (occ_u * evals_u + occ_d * evals_d))
            entropy = _entropy_term(occ_u, wts, electronic_temp_ha, 1.0) + _entropy_term(
                occ_d, wts, electronic_temp_ha, 1.0
            )
            du_k = jnp.einsum("kn,kmn,kpn->kmp", wts[:, None] * occ_u, evecs_u, jnp.conj(evecs_u))
            dd_k = jnp.einsum("kn,kmn,kpn->kmp", wts[:, None] * occ_d, evecs_d, jnp.conj(evecs_d))
            eu_k = jnp.einsum("kn,kmn,kpn->kmp", wts[:, None] * occ_u * evals_u, evecs_u, jnp.conj(evecs_u))
            ed_k = jnp.einsum("kn,kmn,kpn->kmp", wts[:, None] * occ_d * evals_d, evecs_d, jnp.conj(evecs_d))
            dm_edge = jnp.stack([
                jnp.real(jnp.sum(phases_conj * du_k[:, src, dst], axis=0)),
                jnp.real(jnp.sum(phases_conj * dd_k[:, src, dst], axis=0)),
            ], axis=0)
            edm_edge = jnp.stack([
                jnp.real(jnp.sum(phases_conj * eu_k[:, src, dst], axis=0)),
                jnp.real(jnp.sum(phases_conj * ed_k[:, src, dst], axis=0)),
            ], axis=0)
            return dm_edge, edm_edge, mu_f, e_band, entropy

        # nspin == 4 (NC) or nspin == 8 (SOC): 2N x 2N complex 2-spinor Hamiltonian
        # h_sparse may be float64 (4, n_edges) or complex128 (4, n_edges) [H_uu, H_dd, H_ud_re, H_ud_im]
        h_uu = _bloch_assemble(h_sparse[0].astype(jnp.complex128), src, dst, e_shifts, kpts, n_orb)
        h_dd = _bloch_assemble(h_sparse[1].astype(jnp.complex128), src, dst, e_shifts, kpts, n_orb)
        h_ud_edges = (h_sparse[2] + 1j * h_sparse[3]).astype(jnp.complex128)
        h_ud = _bloch_assemble(h_ud_edges, src, dst, e_shifts, kpts, n_orb)
        h_du = jnp.swapaxes(jnp.conj(h_ud), -1, -2)

        top = jnp.concatenate([h_uu, h_ud], axis=2)
        bot = jnp.concatenate([h_du, h_dd], axis=2)
        h_2n = jnp.concatenate([top, bot], axis=1)

        evals, evecs = batched_generalized_eigh(h_2n, x_k, k_batch_size=eff_batch, x_k=x_k)
        mu_f = _solve_fermi_level(evals, wts, n_electrons, electronic_temp_ha, g_spin=1.0)
        occ = _fermi_dirac(evals, mu_f, electronic_temp_ha)
        e_band = jnp.sum(wts[:, None] * occ * evals)
        entropy = _entropy_term(occ, wts, electronic_temp_ha, g_spin=1.0)

        w_occ = wts[:, None] * occ
        w_eocc = w_occ * evals
        d_2n = jnp.einsum("kn,kmn,kpn->kmp", w_occ, evecs, jnp.conj(evecs))
        e_2n = jnp.einsum("kn,kmn,kpn->kmp", w_eocc, evecs, jnp.conj(evecs))

        def _extract_4ch(mat_2n: jax.Array) -> jax.Array:
            m_uu = jnp.sum(phases_conj * mat_2n[:, :n_orb, :n_orb][:, src, dst], axis=0)
            m_dd = jnp.sum(phases_conj * mat_2n[:, n_orb:, n_orb:][:, src, dst], axis=0)
            m_ud = jnp.sum(phases_conj * mat_2n[:, :n_orb, n_orb:][:, src, dst], axis=0)
            if nspin in (4, 8):
                # Preserve complex Im(D_uu), Im(D_dd) needed for NC off-diagonal Hermiticity and SOC <L_z S_z>
                return jnp.stack(
                    [
                        m_uu.astype(jnp.complex128),
                        m_dd.astype(jnp.complex128),
                        jnp.real(m_ud).astype(jnp.complex128),
                        jnp.imag(m_ud).astype(jnp.complex128),
                    ],
                    axis=0,
                )
            return jnp.stack([jnp.real(m_uu), jnp.real(m_dd), jnp.real(m_ud), jnp.imag(m_ud)], axis=0)

        return _extract_4ch(d_2n), _extract_4ch(e_2n), mu_f, e_band, entropy
