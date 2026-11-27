r"""Static KS-DFT SCF loop driver with ``scf_params`` adaptive control and symmetry hooks.

Theory:
    Executes the self-consistent Kohn-Sham fixed-point loop on a uniform 3D
    real-space mesh and sparse NAO basis:

    1. Assemble :math:`H(\mathbf{R}) = H_0 + H_{\text{grid}}[D_{\text{in}}]`
    2. Solve :math:`H(\mathbf{k})C = S(\mathbf{k})C\varepsilon` across k-points
    3. Build output density matrix :math:`D_{\text{out}}` and energy-density matrix :math:`E_{\text{out}}`
    4. Optional native symmetry projection :math:`D_{\text{out}} \leftarrow \mathcal{P}_{\text{sym}}(D_{\text{out}})`
    5. Evaluate residuals, adaptive trust-region / smearing controller, and mix
    6. Scatter :math:`\rho(\mathbf{r}) = \text{rhoofd}(D)` and optional grid symmetrization
    7. Solve Poisson (:math:`V_H`) + XC (:math:`V_{\text{xc}}`) + rVV10 + :math:`V^{\text{NA}}` and gather :math:`H_{\text{grid}}`

Algorithm:
    Purely functional state loop over ``SCFState`` with zero global state. Supports
    both Hamiltonian mixing (default) and density-matrix mixing, plus optional
    ``SymmetryProjector`` hooks for Phase 2.

Invariants:
    - Records ``last_scf_free_e_ha`` separately from the mandatory post-SCF ``DM_out`` pass.
    - Guarantees exact user target ``electronic_temp_ha`` upon convergence.

SIESTA Mapping:
    Replaces ``siesta_forces.F``, ``scfconvergence_test.F``, and ``m_compute_dm.F90``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.protocols import ExactExchange, SymmetryProjector, bind_symmetry, symmetry_jit_arrays
from jax_siesta.core.types import (
    BasisData,
    KPointMesh,
    SCFConfig,
    SCFState,
    SparsePattern,
    Structure,
    TwoCenterMatrices,
)
from jax_siesta.grid.mesh import (
    GridOps,
    _pack_basis_grid_buffers,
    _pack_na_buffers,
    choose_mesh_dimensions,
    neutral_atom_grid_differentiable,
    rhoofd_differentiable,
    tau_core_grid_differentiable,
    tauofd_differentiable,
    vmat_differentiable,
    vmat_tau_differentiable,
)
from jax_siesta.io.siesta_files import normalize_dm
from jax_siesta.scf.controller import update_adaptive_state
from jax_siesta.scf.descriptors import recommend_scf_defaults
from jax_siesta.scf.eigensolver import AssemblyPlan, GeneralizedEigenSolver, bloch_assembly_plan
from jax_siesta.scf.mixer import TikhonovPulayMixer
from jax_siesta.xc.dispersion import RVV10Dispersion
from jax_siesta.xc.functional import LibxcFFIEvaluator, exact_exchange_spec


def build_monkhorst_pack_mesh(
    grid_shape: tuple[int, int, int],
    shifts: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> KPointMesh:
    r"""Construct an unreduced uniform Monkhorst-Pack Brillouin-zone mesh.

    Theory:
        Generates fractional wavevectors :math:`k_\alpha = (i_\alpha + s_\alpha)/N_\alpha`
        folded into :math:`(-1/2, 1/2]` with uniform weights :math:`w_q = 1 / (N_1 N_2 N_3)`.

    Algorithm:
        Cartesian product of 1D uniform fractional grids.

    Parameters:
        grid_shape: Tuple ``(nk1, nk2, nk3)`` of positive integers.
        shifts: Fractional grid offsets in ``[0, 1)``.

    Returns:
        ``KPointMesh`` with ``is_ibz=False``.

    Invariants:
        ``sum(weights) == 1.0``.

    SIESTA Mapping:
        Replaces unreduced Monkhorst-Pack setup in ``kpoint_grid.F90``.
    """
    axes = []
    for n, sh in zip(grid_shape, shifts, strict=True):
        pts = (np.arange(n, dtype=np.float64) + sh) / float(n)
        pts = pts - np.round(pts)
        axes.append(pts)
    g0, g1, g2 = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    kpts = np.stack([g0.ravel(), g1.ravel(), g2.ravel()], axis=-1)
    n_k = kpts.shape[0]
    weights = np.full((n_k,), 1.0 / float(n_k), dtype=np.float64)
    return KPointMesh(
        kpoints_frac=jnp.asarray(kpts, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
        grid_shape=grid_shape,
        shifts=shifts,
        is_ibz=False,
        used_time_reversal=False,
    )


def _trace_dm_h(dm: jax.Array, h: jax.Array, nspin: int) -> jax.Array:
    """Spin-resolved ``Re Tr[D^dagger H]`` over the sparse edges (off-diagonal spin blocks count twice)."""
    if nspin in (4, 8):
        return jnp.sum(
            jnp.real(dm[0]) * jnp.real(h[0])
            + jnp.real(dm[1]) * jnp.real(h[1])
            + jnp.imag(dm[0]) * jnp.imag(h[0])
            + jnp.imag(dm[1]) * jnp.imag(h[1])
            + 2.0 * (jnp.real(dm[2]) * jnp.real(h[2]) + jnp.real(dm[3]) * jnp.real(h[3]))
        )
    if nspin == 2:
        return jnp.sum(jnp.real(dm) * jnp.real(h))
    return jnp.sum(jnp.real(dm[0]) * jnp.real(h[0]))


def _pack_spinor6(m: jax.Array) -> jax.Array:
    """Real channels ``[Re uu, Re dd, Re ud, Im ud, Im uu, Im dd]`` of a spinor matrix ``(4, n_edges)``."""
    return jnp.stack(
        [jnp.real(m[0]), jnp.real(m[1]), jnp.real(m[2]), jnp.real(m[3]), jnp.imag(m[0]), jnp.imag(m[1])], axis=0
    )


def _unpack_spinor6(x: jax.Array) -> jax.Array:
    """Inverse of ``_pack_spinor6``: complex ``(4, n_edges)`` with real off-diagonal channels."""
    return jnp.stack(
        [x[0] + 1j * x[4], x[1] + 1j * x[5], x[2].astype(jnp.complex128), x[3].astype(jnp.complex128)], axis=0
    )


def build_h0_sparse(two_center: TwoCenterMatrices, nspin: int) -> jax.Array:
    r"""Assemble the geometry-dependent core Hamiltonian :math:`H_0 = T + V^{\text{KB}}`.

    Theory:
        Combines kinetic energy :math:`T_{\mu\nu}` and Kleinman-Bylander nonlocal
        projectors :math:`V^{\text{KB}}_{\mu\nu}` across the spin channels
        (including :math:`V^{\text{KB}}_{\text{SO}}` for ``nspin=8``).

    Algorithm:
        Broadcasts scalar ``T + V_kb`` to diagonal spin channels ``(uu, dd)`` and
        adds complex spin-orbit couplings when ``nspin == 8``.

    Parameters:
        two_center: ``TwoCenterMatrices`` PyTree.
        nspin: Spin mode ``1, 2, 4, 8``.

    Returns:
        Sparse array of shape ``(n_spin_mat, n_edges)`` (float64 for ``nspin<=4``,
        float64 real/imag components packed into 4 channels for ``nspin=8`` when
        ``v_kb_so`` real/imag are used — wait: ``(4, n_edges)`` complex128 if SOC
        imaginary diagonal exists).

    Invariants:
        Preserves Hermiticity across ``transpose_edge_idx``.

    SIESTA Mapping:
        Replaces ``H0`` construction in ``setup_H0.F``.
    """
    h0_scalar = two_center.kinetic + two_center.v_kb
    if nspin == 1:
        return h0_scalar[None, :]
    if nspin == 2:
        return jnp.stack([h0_scalar, h0_scalar], axis=0)
    if nspin == 4 or two_center.v_kb_so is None:
        zeros = jnp.zeros_like(h0_scalar)
        return jnp.stack([h0_scalar, h0_scalar, zeros, zeros], axis=0)

    # nspin == 8: v_kb_so has shape (4, n_edges) complex128 [uu, dd, ud, du]
    # Pack as complex128 (4, n_edges) [H_uu, H_dd, Re(H_ud), Im(H_ud)] where H_uu, H_dd can be complex
    t_c = two_center.kinetic.astype(jnp.complex128)
    vso = two_center.v_kb_so
    return jnp.stack([t_c + vso[0], t_c + vso[1], jnp.real(vso[2]).astype(jnp.complex128), jnp.imag(vso[2]).astype(jnp.complex128)], axis=0)


def _as_restart_matrix(matrix: jax.Array, pattern: SparsePattern, nspin: int, kind: str) -> jax.Array:
    """Check the shape of a warm-start matrix and give it the dtype the SCF loop uses for this spin mode."""
    n_ch = 1 if nspin == 1 else (2 if nspin == 2 else 4)
    arr = jnp.asarray(matrix)
    if arr.shape != (n_ch, pattern.n_edges):
        raise ValueError(
            f"Warm-start {kind} has shape {arr.shape}; nspin={nspin} on this pattern needs {(n_ch, pattern.n_edges)}."
        )
    want_complex = nspin == 8 or (nspin == 4 and kind == "dm")
    if want_complex:
        return arr.astype(jnp.complex128)
    return jnp.real(arr).astype(jnp.float64)


@partial(jax.jit, static_argnames=("mesh_shape", "nspin", "seed_functional", "with_tau_core", "n_edges", "sym"))
def _seed_grids_and_hamiltonian(
    positions: jax.Array,
    cell: jax.Array,
    packed_basis: dict[str, jax.Array],
    na_packed: dict[str, jax.Array],
    h0: jax.Array,
    *,
    mesh_shape: tuple[int, int, int],
    nspin: int,
    seed_functional: str,
    with_tau_core: bool,
    n_edges: int,
    sym: _StaticByIdentity | None,
    sym_arrays: Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Neutral-atom grids, core densities and the seed Hamiltonian ``H0 + V[rho_NA]``, in one compiled call.

    Returns ``(h_seed, rho_na, v_na, rho_core, tau_core)``; ``tau_core`` is ``None`` unless ``with_tau_core``.
    """
    sym_proj = bind_symmetry(sym.obj if sym is not None else None, sym_arrays, nspin)
    rho_na, v_na, rho_core = neutral_atom_grid_differentiable(positions, cell, na_packed, mesh_shape)
    tau_core = tau_core_grid_differentiable(positions, cell, na_packed, mesh_shape) if with_tau_core else None
    if sym_proj is not None:
        rho_na = sym_proj.symmetrize_grid(rho_na[None, ...])[0]
        v_na = sym_proj.symmetrize_grid(v_na[None, ...])[0]
        rho_core = sym_proj.symmetrize_grid(rho_core[None, ...])[0]
        if tau_core is not None:
            tau_core = sym_proj.symmetrize_grid(tau_core[None, ...])[0]

    # Seed neutral-atom density for initial V_xc[rho_NA] + V_NA
    if nspin == 1:
        rho_seed = rho_na[None, ...]
    elif nspin == 2:
        rho_seed = jnp.stack([0.5 * rho_na, 0.5 * rho_na], axis=0)
    else:
        rho_seed = jnp.stack([0.5 * rho_na, 0.5 * rho_na, jnp.zeros_like(rho_na), jnp.zeros_like(rho_na)], axis=0)
    _, v_xc_0, _ = LibxcFFIEvaluator(seed_functional).evaluate(rho_seed, cell, nspin, rho_core, compute_stress=False)
    v_eff_0 = v_xc_0.at[0].add(v_na)
    if nspin >= 2:
        v_eff_0 = v_eff_0.at[1].add(v_na)
    h_seed = vmat_differentiable(v_eff_0, positions, cell, packed_basis, n_edges, nspin)
    return h0 + h_seed, rho_na, v_na, rho_core, tau_core


def _zeeman_seed(
    structure: Structure, basis: BasisData, pattern: SparsePattern, overlap: jax.Array, nspin: int
) -> np.ndarray | None:
    """On-site spin splitting ``-b m_I S_mumu`` (rows z, x, y) that seeds a magnetic SCF, or ``None`` without moments."""
    moments = np.asarray(structure.initial_moments, dtype=np.float64)
    if nspin < 2 or np.max(np.abs(moments)) <= 1e-8:
        return None
    orb_atom = np.asarray(basis.orb_atom, dtype=np.int32)
    src = np.asarray(pattern.src_orb)
    dst = np.asarray(pattern.dst_unit_orb)
    sh = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)]
    is_diag0 = (src == dst) & np.all(sh == 0, axis=1)
    s_vals = np.asarray(overlap)
    b_scale = 0.04  # Ha/mu_B initial Zeeman seed
    out = np.zeros((3, pattern.n_edges), dtype=np.float64)
    diag_edges = np.where(is_diag0)[0]
    m_e = moments[orb_atom[src[diag_edges]]]  # (n_diag, 3): (mx, my, mz) of each diagonal edge's atom
    out[0, diag_edges] = -b_scale * m_e[:, 2] * s_vals[diag_edges]
    out[1, diag_edges] = -b_scale * m_e[:, 0] * s_vals[diag_edges]
    out[2, diag_edges] = -b_scale * m_e[:, 1] * s_vals[diag_edges]
    return out


class _StaticByIdentity:
    """Hashable-by-identity wrapper so an unhashable object (the symmetry engine) can be a static ``jit`` argument."""

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticByIdentity) and other.obj is self.obj


# The SCF's compiled functions live at module level and take everything that depends on the geometry as arguments,
# so a new geometry with the same array shapes reuses the compiled code (and the persistent compilation cache can hit).
# Only shapes, spin mode, functional, dispersion choice and the symmetry engine (by identity) are static; the engine's
# large arrays are passed as the argument ``sym_arrays`` so that they are not baked into every compiled function.


@partial(jax.jit, static_argnames=("nspin", "real_k"))
def _overlap_transform_compiled(
    s_sparse: jax.Array,
    pattern: SparsePattern,
    kmesh: KPointMesh,
    plan: AssemblyPlan | None,
    nspin: int,
    real_k: bool,
) -> jax.Array:
    return GeneralizedEigenSolver().overlap_transform(
        s_sparse, pattern, kmesh, nspin, assembly_plan=plan, real_k=real_k
    )


@partial(jax.jit, static_argnames=("nspin", "real_k"))
def _solve_compiled(
    h_mat: jax.Array,
    temp_val: jax.Array,
    s_transform: jax.Array,
    s_sparse: jax.Array,
    pattern: SparsePattern,
    kmesh: KPointMesh,
    plan: AssemblyPlan | None,
    n_electrons: jax.Array,
    nspin: int,
    real_k: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return GeneralizedEigenSolver().solve(
        h_mat,
        s_sparse,
        pattern,
        kmesh,
        n_electrons,
        temp_val,
        nspin,
        s_transform=s_transform,
        assembly_plan=plan,
        real_k=real_k,
    )


@partial(jax.jit, static_argnames=("mesh_shape", "nspin", "xc_functional", "use_rvv10", "n_edges", "sym"))
def _grid_hamiltonian_and_energies(
    dm_mat: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed_basis: dict[str, jax.Array],
    rho_na: jax.Array,
    v_na: jax.Array,
    rho_core: jax.Array,
    tau_core: jax.Array | None,
    dvol: jax.Array,
    *,
    mesh_shape: tuple[int, int, int],
    nspin: int,
    xc_functional: str,
    use_rvv10: bool,
    n_edges: int,
    sym: _StaticByIdentity | None,
    sym_arrays: Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Grid Hamiltonian and grid energies of a density matrix: ``(h_grid, e_h, e_xc, e_disp, e_v_na, Tr[rho V_eff])``."""
    grid_ops = GridOps()
    xc_eval = LibxcFFIEvaluator(xc_functional)
    rvv10 = RVV10Dispersion() if use_rvv10 else None
    sym_proj = bind_symmetry(sym.obj if sym is not None else None, sym_arrays, nspin)
    dm_real = jnp.real(dm_mat) if jnp.iscomplexobj(dm_mat) else dm_mat
    rho = rhoofd_differentiable(dm_real, positions, cell, packed_basis, mesh_shape, nspin)
    if sym_proj is not None:
        rho = sym_proj.symmetrize_grid(rho)
    rho_tot = rho[0] if nspin == 1 else (rho[0] + rho[1])
    delta_rho = rho_tot - rho_na
    v_h, e_h, _ = grid_ops.hartree_potential(delta_rho, cell)
    h_tau = None
    if xc_eval.is_mgga:
        tau = tauofd_differentiable(dm_real, positions, cell, packed_basis, mesh_shape, nspin)
        if sym_proj is not None:
            tau = sym_proj.symmetrize_grid(tau)
        e_xc, v_xc, v_tau, _ = xc_eval.evaluate_mgga(
            rho, tau, cell, nspin, rho_core, tau_core_grid=tau_core, compute_stress=False
        )
        if sym_proj is not None:
            v_tau = sym_proj.symmetrize_grid(v_tau)
        h_tau = vmat_tau_differentiable(v_tau, positions, cell, packed_basis, n_edges, nspin)
    else:
        e_xc, v_xc, _ = xc_eval.evaluate(rho, cell, nspin, rho_core, compute_stress=False)
    v_eff = v_xc.at[0].add(v_na + v_h)
    if nspin >= 2:
        v_eff = v_eff.at[1].add(v_na + v_h)
    e_disp = jnp.array(0.0, dtype=jnp.float64)
    if rvv10 is not None:
        e_disp, v_rvv, _ = rvv10.potential_and_energy(rho_tot + rho_core, cell, compute_stress=False)
        v_eff = v_eff.at[0].add(v_rvv)
        if nspin >= 2:
            v_eff = v_eff.at[1].add(v_rvv)
    if sym_proj is not None:
        v_eff = sym_proj.symmetrize_grid(v_eff)
    h_grid = vmat_differentiable(v_eff, positions, cell, packed_basis, n_edges, nspin)
    if h_tau is not None:
        h_grid = h_grid + h_tau
    if sym_proj is not None:
        h_grid = sym_proj.project_matrix(h_grid, nspin)
    # Atomic part sum_IJ int V_NA,I n_J is exact in TwoCenterMatrices.e_na_pair (D-017).
    e_v_na = dvol * jnp.sum((rho_tot - rho_na) * v_na)
    e_v_eff_rho = dvol * (
        jnp.sum(rho[0] * v_eff[0])
        if nspin == 1
        else (
            jnp.sum(rho[0] * v_eff[0] + rho[1] * v_eff[1])
            + (2.0 * jnp.sum(rho[2:4] * v_eff[2:4]) if nspin in (4, 8) else 0.0)
        )
    )
    return h_grid, e_h, e_xc, e_disp, e_v_na, e_v_eff_rho


@partial(jax.jit, static_argnames=("nspin", "sym"))
def _iteration_update(
    h0: jax.Array,
    h_cur: jax.Array,
    h_grid_out: jax.Array,
    dm_out: jax.Array,
    dm_in: jax.Array,
    e_band: jax.Array,
    entropy: jax.Array,
    e_v_na: jax.Array,
    e_h: jax.Array,
    e_xc: jax.Array,
    e_disp: jax.Array,
    e_grid_in: jax.Array,
    e_na_pair: jax.Array,
    *,
    nspin: int,
    sym: _StaticByIdentity | None,
    sym_arrays: Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Bookkeeping of one SCF iteration: ``(h_out, free_e_scf, harris_e, d_max, h_max)``.

    ``free_e_scf`` is the in-flight KS free energy ``E_band - Tr[D_out H_grid,in] + E_grid[D_out] + E_NA,pair - TS``;
    ``harris_e`` the Harris-Foulkes free energy ``E_band[H_in] - Tr[D_in H_grid,in] + E_grid[D_in] + E_NA,pair - TS``
    (SIESTA ``compute_energies``; exact when ``H_in = H[D_in]``, i.e. DM mixing, D-023).
    """
    h_out = h0 + h_grid_out.astype(h0.dtype)
    if sym is not None:
        h_out = bind_symmetry(sym.obj if sym is not None else None, sym_arrays, nspin).project_matrix(h_out, nspin)
    h_grid_in = h_cur - h0
    tr_in = _trace_dm_h(dm_out, h_grid_in, nspin)
    free_e_scf = e_band - tr_in + e_v_na + e_h + e_xc + e_disp + e_na_pair + entropy
    harris_e = e_band - _trace_dm_h(dm_in, h_grid_in, nspin) + e_grid_in + e_na_pair + entropy
    d_max = jnp.max(jnp.abs(dm_out - dm_in))
    h_max = jnp.max(jnp.abs(h_out - h_cur))
    return h_out, free_e_scf, harris_e, d_max, h_max


def run_scf(
    structure: Structure,
    basis: BasisData,
    pattern: SparsePattern,
    two_center: TwoCenterMatrices,
    kmesh: KPointMesh,
    config: SCFConfig,
    mesh_shape: tuple[int, int, int] | None = None,
    sym_proj: SymmetryProjector | None = None,
    initial_dm: jax.Array | None = None,
    initial_h: jax.Array | None = None,
    callback: Callable[[dict[str, float]], None] | None = None,
    exact_exchange: ExactExchange | None = None,
) -> tuple[SCFState, tuple[int, int, int]]:
    r"""Execute the adaptive static SCF cycle until convergence or iteration cap.

    Theory:
        Solves the self-consistent Kohn-Sham fixed-point equations using
        physics-informed ``scf_params`` conditioning defaults, Tikhonov-Pulay / Pulay
        mixing with decoupled charge/spin preconditioning, native adaptive trust-region
        backoff, and optional symmetry projection on ``D``, ``E``, and :math:`\rho(\mathbf{r})`.

    Algorithm:
        Initializes ``SCFState`` from a neutral-atom + Zeeman seed, iterates
        ``solve -> symmetrize -> grid_potential -> residual -> controller -> mix``,
        and returns the final ``SCFState`` holding unmixed ``dm_out`` and ``edm_out``.

        Warm start. ``initial_dm`` (e.g. from ``jax_siesta.io.read_dm``) replaces the cold-start density matrix;
        it is rescaled to the electron count and symmetry-projected, and the first Hamiltonian is ``H[D]`` as in
        a SIESTA restart. ``initial_h`` (e.g. from ``jax_siesta.io.read_nc_restart``) is used as the first input
        Hamiltonian when ``mix_target == "Hamiltonian"``; without ``initial_dm`` the starting density matrix is
        the one it produces (one extra diagonalisation). Under density-matrix mixing the first Hamiltonian is
        always ``H[D_in]``, so ``initial_h`` only serves to build ``D_in`` when no ``initial_dm`` is given.

    Parameters:
        structure: ``Structure`` PyTree.
        basis: ``BasisData`` PyTree.
        pattern: Static ``SparsePattern`` PyTree.
        two_center: Precomputed ``TwoCenterMatrices``.
        kmesh: ``KPointMesh`` PyTree.
        config: ``SCFConfig`` hyperparameters.
        mesh_shape: Optional explicit 3D mesh shape (defaults to ``choose_mesh_dimensions``).
        sym_proj: Optional ``SymmetryProjector`` (used in Phase 2).
        initial_dm: Optional warm-start density matrix ``(n_spin_mat, n_edges)`` on ``pattern``.
        initial_h: Optional warm-start Hamiltonian ``(n_spin_mat, n_edges)`` in Hartree on ``pattern``.
        callback: Optional function called once per iteration with a dictionary of floats: ``iteration``
            (from 1), ``free_energy_ha``, ``harris_energy_ha``, ``fermi_level_ha``, ``d_max``, ``h_max_ha`` and
            ``energy_change_ha``. It is for progress reports and convergence histories; it cannot change the run.
        exact_exchange: Optional exact-exchange stage for a hybrid functional, built for this geometry and
            pattern. When it is omitted and ``config.xc_functional`` is a hybrid, one is built here
            (``jax_siesta.hfx.build_exact_exchange``). Its matrix is added to the density-dependent Hamiltonian
            and its energy to the exchange-correlation energy of every iteration.

    Returns:
        Tuple ``(final_scf_state, mesh_shape)``.

    Invariants:
        - Never returns ``converged=True`` with ``active_temp_ha != config.electronic_temp_ha``.
        - Enforces iteration >= 2 before convergence.

    SIESTA Mapping:
        Replaces the main SCF loop in ``siesta_forces.F``.
    """
    n_electrons = basis.total_valence_electrons - config.net_charge
    if n_electrons <= 0.0 or structure.n_atoms == 0:
        raise ValueError("Invalid electron count or empty structure.")
    if exact_exchange is None and exact_exchange_spec(config.xc_functional).is_hybrid:
        from jax_siesta.hfx import build_exact_exchange

        exact_exchange = build_exact_exchange(structure, basis, pattern, config, k_grid=kmesh.grid_shape)

    if mesh_shape is None:
        mesh_shape = choose_mesh_dimensions(
            structure.cell, config.mesh_cutoff_ry, criterion=config.mesh_criterion, mesh_subdivisions=config.mesh_subdivisions
        )

    rec = recommend_scf_defaults(structure, basis, config)
    # The recommended Tikhonov lambda was computed but never reached the mixer (review M2).
    mix_config = replace(config, tikhonov_reg=rec.tikhonov_reg)
    xc_eval = LibxcFFIEvaluator(config.xc_functional)
    rvv10 = RVV10Dispersion() if config.dispersion.upper() == "RVV10" else None
    mixer = TikhonovPulayMixer()

    # S is fixed during the SCF, so its orthogonalising transform is computed once and passed to every solve. The
    # assembly tables and the real-arithmetic flag come from the concrete pattern, once (see bloch_assembly_plan).
    assembly_plan, real_k = bloch_assembly_plan(pattern, kmesh, config.nspin)
    n_electrons_arr = jnp.asarray(n_electrons, dtype=jnp.float64)
    s_transform = _overlap_transform_compiled(
        two_center.overlap, pattern, kmesh, assembly_plan, nspin=config.nspin, real_k=real_k
    )

    def solve_jit(h_mat: jax.Array, temp_val: float, s_x: jax.Array):  # type: ignore[no-untyped-def]
        return _solve_compiled(
            h_mat,
            jnp.asarray(temp_val, dtype=jnp.float64),
            s_x,
            two_center.overlap,
            pattern,
            kmesh,
            assembly_plan,
            n_electrons_arr,
            nspin=config.nspin,
            real_k=real_k,
        )

    h0 = build_h0_sparse(two_center, config.nspin)
    if exact_exchange is not None and config.nspin in (4, 8):
        # The exchange matrix of the diagonal spin blocks is complex, so the Hamiltonian is too.
        h0 = h0.astype(jnp.complex128)
    n_spin_mat = h0.shape[0]
    n_edges = pattern.n_edges
    packed_basis = _pack_basis_grid_buffers(basis, pattern)
    sym_static = _StaticByIdentity(sym_proj) if sym_proj is not None else None
    sym_arrays = symmetry_jit_arrays(sym_proj, config.nspin)

    # Neutral-atom grids and the cold-start seed H0 + V_xc[rho_NA] + V_NA (the grids are needed by warm starts too).
    h_init, rho_na, v_na, rho_core, tau_core = _seed_grids_and_hamiltonian(
        structure.positions,
        structure.cell,
        packed_basis,
        _pack_na_buffers(structure, basis),
        h0,
        mesh_shape=tuple(int(x) for x in mesh_shape),
        nspin=config.nspin,
        seed_functional="PBE" if xc_eval.is_mgga else config.xc_functional,
        with_tau_core=xc_eval.is_mgga,
        n_edges=n_edges,
        sym=sym_static,
        sym_arrays=sym_arrays,
    )
    if initial_h is not None:
        initial_h = _as_restart_matrix(initial_h, pattern, config.nspin, "h")
        if sym_proj is not None:
            initial_h = sym_proj.project_matrix(initial_h, config.nspin)
    if initial_dm is not None:
        # Warm start from a stored density matrix, rescaled to the electron count.
        dm_init = normalize_dm(
            _as_restart_matrix(initial_dm, pattern, config.nspin, "dm"), two_center.overlap, n_electrons, config.nspin
        )
    elif initial_h is not None:
        # Warm start from a stored Hamiltonian alone: the starting density matrix is the one it produces.
        dm_init = solve_jit(initial_h.astype(h0.dtype), config.electronic_temp_ha, s_transform)[0]
    else:
        # Cold start: diagonalise the seed (+ an on-site Zeeman term for magnetic runs).
        zeeman = _zeeman_seed(structure, basis, pattern, two_center.overlap, config.nspin)
        if zeeman is not None:
            h_init = h_init.at[0].add(jnp.asarray(zeeman[0]))
            h_init = h_init.at[1].add(jnp.asarray(-zeeman[0]))
            if config.nspin in (4, 8):
                h_init = h_init.at[2].add(jnp.asarray(zeeman[1]))
                h_init = h_init.at[3].add(jnp.asarray(-zeeman[2]))
        dm_init = solve_jit(h_init, config.electronic_temp_ha, s_transform)[0]
    if sym_proj is not None:
        dm_init = sym_proj.project_matrix(dm_init, config.nspin)

    n1, n2, n3 = mesh_shape
    dvol = float(structure.volume) / float(n1 * n2 * n3)
    positions_static = structure.positions
    cell_static = structure.cell

    dvol_arr = jnp.asarray(dvol, dtype=jnp.float64)

    def _eval_h_grid_and_energy(dm_mat: jax.Array):  # type: ignore[no-untyped-def]
        h_g, e_h_g, e_xc_g, e_disp_g, e_v_na_g, e_veff_g = _eval_grid_only(dm_mat)
        if exact_exchange is not None:
            # Exact exchange of a hybrid: a host kernel outside the compiled step. Its matrix joins the
            # density-dependent part of H and its energy the exchange-correlation energy, so that the Kohn-Sham and
            # Harris bookkeeping of _iteration_update holds unchanged (dE_x/dP = H_x, E_x = Tr[P H_x]/2).
            # For nspin 4 and 8 the exchange matrix of the diagonal spin blocks is complex (K of Im D_uu, Im D_dd).
            h_x_np, e_x = exact_exchange.matrix_and_energy(np.asarray(dm_mat), config.nspin)
            h_x = jnp.asarray(h_x_np)
            if sym_proj is not None:
                h_x = sym_proj.project_matrix(h_x, config.nspin)
            h_g = h_g + h_x
            e_xc_g = e_xc_g + e_x
        return h_g, e_h_g, e_xc_g, e_disp_g, e_v_na_g, e_veff_g

    def _eval_grid_only(dm_mat: jax.Array):  # type: ignore[no-untyped-def]
        return _grid_hamiltonian_and_energies(
            dm_mat,
            positions_static,
            cell_static,
            packed_basis,
            rho_na,
            v_na,
            rho_core,
            tau_core,
            dvol_arr,
            mesh_shape=tuple(int(x) for x in mesh_shape),
            nspin=config.nspin,
            xc_functional=config.xc_functional,
            use_rvv10=rvv10 is not None,
            n_edges=n_edges,
            sym=sym_static,
            sym_arrays=sym_arrays,
        )

    h_grid_0, e_h_0, e_xc_0, e_disp_0, e_v_na_0, _ = _eval_h_grid_and_energy(dm_init)
    # Grid energy terms at the density D_in that built H_in: they enter the Harris-Foulkes energy (D-023).
    e_grid_in = e_h_0 + e_xc_0 + e_disp_0 + e_v_na_0
    harris_provenance = "exact_dm_mixing" if config.mix_target == "DM" else "siesta_h_mixing"
    h_in = h0 + h_grid_0.astype(h0.dtype)
    if sym_proj is not None:
        h_in = sym_proj.project_matrix(h_in, config.nspin)
    if initial_h is not None and config.mix_target == "Hamiltonian":
        # Warm start from a stored Hamiltonian: it is the first H_in (D_in stays the density paired with it).
        h_in = initial_h.astype(h0.dtype)

    # Note for nspin=8, mixer works on real float64 channels while preserving imag(H_uu, H_dd) from H0_SO.
    # With exact exchange Im(H_uu, H_dd) depends on the density matrix and is mixed as two more real channels.
    mix_h_imag = exact_exchange is not None and config.nspin in (4, 8)
    h_so_imag = jnp.imag(h0[:2]) if (jnp.iscomplexobj(h0) and not mix_h_imag) else None

    n_mix_ch = 6 if (config.nspin in (4, 8) and (config.mix_target == "DM" or mix_h_imag)) else n_spin_mat
    state = SCFState(
        iteration=0,
        h_sparse=h_in,
        dm_in=dm_init,
        dm_out=dm_init,
        edm_out=jnp.zeros_like(dm_init),
        fermi_level_ha=jnp.array(0.0, dtype=jnp.float64),
        entropy_ha=jnp.array(0.0, dtype=jnp.float64),
        last_scf_free_e_ha=jnp.array(0.0, dtype=jnp.float64),
        w_cur=jnp.asarray(rec.w_base, dtype=jnp.float64),
        w_base=jnp.asarray(rec.w_base, dtype=jnp.float64),
        active_temp_ha=jnp.asarray(config.electronic_temp_ha, dtype=jnp.float64),
        streak_bad=0,
        prev_d_max=jnp.array(1.0, dtype=jnp.float64),
        prev_h_max=jnp.array(1.0, dtype=jnp.float64),
        history_x=jnp.zeros((rec.n_hist, n_mix_ch, n_edges), dtype=jnp.float64),
        history_f=jnp.zeros((rec.n_hist, n_mix_ch, n_edges), dtype=jnp.float64),
        history_count=0,
        converged=False,
        outcome="running",
        history_d=jnp.zeros((rec.n_hist, n_mix_ch, n_edges), dtype=jnp.float64) if config.mixer_variant == "adiis" else None,
        history_e=jnp.zeros((rec.n_hist,), dtype=jnp.float64) if config.mixer_variant == "adiis" else None,
    )

    prev_free_e = 0.0
    dm_in = dm_init
    h_cur = h_in

    for _ in range(config.max_scf_iterations):
        temp_cur = float(state.active_temp_ha)
        dm_out, edm_out, mu_f, e_band, entropy = solve_jit(h_cur, temp_cur, s_transform)
        if sym_proj is not None:
            dm_out = sym_proj.project_matrix(dm_out, config.nspin)
            edm_out = sym_proj.project_matrix(edm_out, config.nspin)

        h_grid_out, e_h, e_xc, e_disp, e_v_na, e_v_eff_rho = _eval_h_grid_and_energy(dm_out)
        # In-flight KS and Harris-Foulkes free energies. Under Hamiltonian mixing D_in is the previous D_out and H_in
        # the mixed H, as in SIESTA; that Harris value is not variational (harris_provenance records which one it is).
        h_out, free_e_scf, harris_e, d_max_arr, h_max_arr = _iteration_update(
            h0,
            h_cur,
            h_grid_out,
            dm_out,
            dm_in,
            e_band,
            entropy,
            e_v_na,
            e_h,
            e_xc,
            e_disp,
            e_grid_in,
            two_center.e_na_pair,
            nspin=config.nspin,
            sym=sym_static,
            sym_arrays=sym_arrays,
        )
        d_max = float(d_max_arr)
        h_max = float(h_max_arr)
        e_diff = abs(float(free_e_scf) - prev_free_e) if state.iteration > 0 else 1.0
        prev_free_e = float(free_e_scf)

        state, converged, outcome = update_adaptive_state(state, config, d_max, h_max, e_diff)
        state = replace(
            state,
            h_sparse=h_cur,
            dm_in=dm_in,
            dm_out=dm_out,
            edm_out=edm_out,
            fermi_level_ha=mu_f,
            entropy_ha=entropy,
            last_scf_free_e_ha=free_e_scf,
            harris_energy_ha=harris_e,
            harris_provenance=harris_provenance,
        )
        if callback is not None:
            callback(
                {
                    "iteration": float(state.iteration),
                    "free_energy_ha": float(free_e_scf),
                    "harris_energy_ha": float(harris_e),
                    "fermi_level_ha": float(mu_f),
                    "d_max": d_max,
                    "h_max_ha": h_max,
                    "energy_change_ha": e_diff,
                }
            )
        if outcome == "numerical_breakdown":
            return state, mesh_shape
        if converged:
            return state, mesh_shape

        if config.mix_target == "Hamiltonian" and mix_h_imag:
            adiis_aux = (_pack_spinor6(dm_out), free_e_scf) if config.mixer_variant == "adiis" else None
            x_next6, state = mixer.step(_pack_spinor6(h_cur), _pack_spinor6(h_out), state, mix_config, aux=adiis_aux)
            h_next = _unpack_spinor6(x_next6)
            if sym_proj is not None:
                h_next = sym_proj.project_matrix(h_next, config.nspin)
            h_cur = h_next
            dm_in = dm_out
            e_grid_in = e_h + e_xc + e_disp + e_v_na
        elif config.mix_target == "Hamiltonian":
            x_in_real = jnp.real(h_cur)
            x_out_real = jnp.real(h_out)
            adiis_aux = (jnp.real(dm_out), free_e_scf) if config.mixer_variant == "adiis" else None
            x_next_real, state = mixer.step(x_in_real, x_out_real, state, mix_config, aux=adiis_aux)
            if h_so_imag is not None:
                h_next = x_next_real.astype(jnp.complex128)
                h_next = h_next.at[:2].add(1j * h_so_imag)
            else:
                h_next = x_next_real
            if sym_proj is not None:
                h_next = sym_proj.project_matrix(h_next, config.nspin)
            h_cur = h_next
            dm_in = dm_out
            e_grid_in = e_h + e_xc + e_disp + e_v_na
        else:
            if config.nspin in (4, 8):
                # Pack [Re(uu), Re(dd), Re(ud), Im(ud), Im(uu), Im(dd)] so complex DM is mixed consistently
                x_next6, state = mixer.step(_pack_spinor6(dm_in), _pack_spinor6(dm_out), state, mix_config)
                x_next_dm = _unpack_spinor6(x_next6)
            else:
                x_next_dm, state = mixer.step(jnp.real(dm_in), jnp.real(dm_out), state, mix_config)
            if sym_proj is not None:
                x_next_dm = sym_proj.project_matrix(x_next_dm, config.nspin)
            dm_in = x_next_dm.astype(dm_out.dtype)
            h_grid_next, e_h_n, e_xc_n, e_disp_n, e_v_na_n, _ = _eval_h_grid_and_energy(dm_in)
            e_grid_in = e_h_n + e_xc_n + e_disp_n + e_v_na_n
            h_cur = h0 + h_grid_next.astype(h0.dtype)
            if sym_proj is not None:
                h_cur = sym_proj.project_matrix(h_cur, config.nspin)

    return replace(state, converged=False, outcome="iteration_cap_exhausted"), mesh_shape
