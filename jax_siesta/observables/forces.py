r"""Post-SCF ``DM_out`` evaluation, Harris/KS energies, and fixed-(D,E) variational forces & stress.

Theory:
    Enforces the mandatory post-SCF ``DM_out`` contract (``docs/design/IMPLEMENTATION_PLAN.md`` §4.4):
    after SCF convergence at the exact target ``electronic_temp_ha``, performs one
    clean non-mixed grid/potential pass directly from ``dm_out`` to record
    ``final_post_scf_free_e_ha`` (with ``dm_out_recomputed=True``) separately from
    ``last_scf_free_e_ha``.

    Evaluates atomic forces and stress tensor at fixed converged :math:`(D, E)` via
    Kohn-Sham variational stationarity:

    .. math::
        \mathbf{F}_I = -\left.\frac{\partial E_{\text{KS}}}{\partial \mathbf{R}_I}\right|_{D}
                       + \mathrm{Tr}\!\left[E \frac{\partial S}{\partial \mathbf{R}_I}\right], \qquad
        \sigma_{\alpha\beta} = \frac{1}{\Omega}\left(\left.\frac{\partial E_{\text{KS}}}{\partial \epsilon_{\alpha\beta}}\right|_{D}
                       - \mathrm{Tr}\!\left[E \frac{\partial S}{\partial \epsilon_{\alpha\beta}}\right]\right).

Algorithm:
    Combines analytical two-center kinetic/Pulay derivatives, exact 3-center KB
    projector link derivatives, short-range neutral-atom pair terms, ``jax.grad`` over
    the differentiable C++/JAX FFI grid energy (capturing ``dfscf``, Hartree, XC/GGA,
    NLCC, and rVV10), and DFT-D3 dispersion gradients. Optional symmetry cleanup
    projections are applied in Phase 2.

Invariants:
    - ``dm_out_recomputed == True`` for every evaluated ``Observables``.
    - Forces satisfy the acoustic sum rule :math:`\sum_I \mathbf{F}_I = \mathbf{0}` and
      match central finite differences of ``final_post_scf_free_e_ha``.

SIESTA Mapping:
    Implements ``ObservableEvaluator`` replacing ``post_scf_work.F``,
    ``compute_energies.F90``, ``dfscf.F``, and ``siesta_forces.F``.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.protocols import ExactExchange, SymmetryProjector, bind_symmetry, symmetry_jit_arrays
from jax_siesta.core.types import (
    BasisData,
    KPointMesh,
    Observables,
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
)
from jax_siesta.integrals.two_center import _iter_kb_shell_pairs, _kb_soc_shell_spin_matrix, kb_three_center_native
from jax_siesta.xc.dispersion import DFTD3Dispersion, RVV10Dispersion
from jax_siesta.xc.functional import LibxcFFIEvaluator, exact_exchange_spec, resolve_xc_id, xc_energy_from_grid


def _stage(timings: dict[str, float] | None, key: str, t_start: float, *results: object) -> float:
    """Add the wall time since ``t_start`` to ``timings[key]`` after the device work of ``results`` (no-op if None)."""
    if timings is None:
        return t_start
    jax.block_until_ready(results)
    now = time.perf_counter()
    timings[key] = timings.get(key, 0.0) + (now - t_start)
    return now


class _StaticByIdentity:
    """Hashable-by-identity wrapper so an unhashable object (the symmetry engine) can be a static ``jit`` argument."""

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticByIdentity) and other.obj is self.obj


@partial(jax.jit, static_argnames=("mesh_shape", "nspin", "xc_functional", "sym"))
def _fresh_grid_energies(
    dm_real: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    na_packed: dict[str, jax.Array],
    dvol: jax.Array,
    *,
    mesh_shape: tuple[int, int, int],
    nspin: int,
    xc_functional: str,
    sym: _StaticByIdentity | None,
    sym_arrays: Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fresh grid pass from ``DM_out``: ``(rho_tot, rho_core, sum rho dV, E_H, E_xc, E_V_NA,grid)``."""
    sym_proj = bind_symmetry(sym.obj if sym is not None else None, sym_arrays, nspin)
    xc_eval = LibxcFFIEvaluator(xc_functional)
    rho = rhoofd_differentiable(dm_real, positions, cell, packed, mesh_shape, nspin)
    rho_na, v_na, rho_core = neutral_atom_grid_differentiable(positions, cell, na_packed, mesh_shape)
    tau = tauofd_differentiable(dm_real, positions, cell, packed, mesh_shape, nspin) if xc_eval.is_mgga else None
    tau_core = tau_core_grid_differentiable(positions, cell, na_packed, mesh_shape) if xc_eval.is_mgga else None
    if sym_proj is not None:
        rho = sym_proj.symmetrize_grid(rho)
        rho_na = sym_proj.symmetrize_grid(rho_na[None, ...])[0]
        v_na = sym_proj.symmetrize_grid(v_na[None, ...])[0]
        rho_core = sym_proj.symmetrize_grid(rho_core[None, ...])[0]
        if tau is not None:
            tau = sym_proj.symmetrize_grid(tau)
        if tau_core is not None:
            tau_core = sym_proj.symmetrize_grid(tau_core[None, ...])[0]
    rho_tot = rho[0] if nspin == 1 else (rho[0] + rho[1])
    q_grid = dvol * jnp.sum(rho_tot)
    _, e_hartree, _ = GridOps().hartree_potential(rho_tot - rho_na, cell)
    if xc_eval.is_mgga:
        assert tau is not None
        e_xc, _, _, _ = xc_eval.evaluate_mgga(
            rho, tau, cell, nspin, rho_core, tau_core_grid=tau_core, compute_stress=False
        )
    else:
        e_xc, _, _ = xc_eval.evaluate(rho, cell, nspin, rho_core, compute_stress=False)
    # Atomic part sum_IJ int V_NA,I n_J is exact in two_center.e_na_pair (D-017).
    e_v_na_grid = dvol * jnp.sum((rho_tot - rho_na) * v_na)
    return rho_tot, rho_core, q_grid, e_hartree, e_xc, e_v_na_grid


def _deformed_grid_terms(
    disp: jax.Array,
    eps: jax.Array,
    dm_real: jax.Array,
    positions: jax.Array,
    cell0: jax.Array,
    packed: dict[str, jax.Array],
    na_packed: dict[str, jax.Array],
    mesh_shape: tuple[int, int, int],
    nspin: int,
    xc_functional: str,
    use_rvv10: bool,
    sym_proj: SymmetryProjector | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Grid energy terms ``(E_V_NA, E_H, E_xc, E_rVV10)`` at fixed DM, after a displacement and a homogeneous strain.

    Their derivatives at ``disp = 0, eps = 0`` are the grid forces and the grid stress (times the volume).
    """
    xc_eval = LibxcFFIEvaluator(xc_functional)
    xc_id = resolve_xc_id(xc_functional)
    n1, n2, n3 = mesh_shape
    def_mat = jnp.eye(3, dtype=jnp.float64) + eps
    pos = positions @ def_mat.T + disp
    cell = def_mat @ cell0
    r_g = rhoofd_differentiable(dm_real, pos, cell, packed, mesh_shape, nspin)
    rna_g, vna_g, rc_g = neutral_atom_grid_differentiable(pos, cell, na_packed, mesh_shape)
    t_g = tauofd_differentiable(dm_real, pos, cell, packed, mesh_shape, nspin) if xc_eval.is_mgga else None
    tc_g = tau_core_grid_differentiable(pos, cell, na_packed, mesh_shape) if xc_eval.is_mgga else None
    if sym_proj is not None:
        r_g = sym_proj.symmetrize_grid(r_g)
        rna_g = sym_proj.symmetrize_grid(rna_g[None, ...])[0]
        vna_g = sym_proj.symmetrize_grid(vna_g[None, ...])[0]
        rc_g = sym_proj.symmetrize_grid(rc_g[None, ...])[0]
        if t_g is not None:
            t_g = sym_proj.symmetrize_grid(t_g)
        if tc_g is not None:
            tc_g = sym_proj.symmetrize_grid(tc_g[None, ...])[0]
    rtot = r_g[0] if nspin == 1 else (r_g[0] + r_g[1])
    dv = jnp.abs(jnp.linalg.det(cell)) / float(n1 * n2 * n3)
    e_vna_l = dv * jnp.sum((rtot - rna_g) * vna_g)
    _, e_h_l, _ = GridOps().hartree_potential(rtot - rna_g, cell)
    e_xc_l = xc_energy_from_grid(
        r_g,
        cell,
        nspin,
        xc_id,
        rc_g,
        tau_grid=t_g,
        tau_core_grid=tc_g,
        xc_pair=xc_eval.xc_pair if xc_eval.is_mgga else None,
    )
    e_rvv_l = RVV10Dispersion().energy_from_density(rtot + rc_g, cell) if use_rvv10 else jnp.zeros(())
    return e_vna_l, e_h_l, e_xc_l, e_rvv_l


@partial(jax.jit, static_argnames=("mesh_shape", "nspin", "xc_functional", "use_rvv10", "sym"))
def _grid_energy_and_pullback(
    dm_real: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    na_packed: dict[str, jax.Array],
    *,
    mesh_shape: tuple[int, int, int],
    nspin: int,
    xc_functional: str,
    use_rvv10: bool,
    sym: _StaticByIdentity | None,
    sym_arrays: Any = None,
) -> tuple[jax.Array, Any]:
    """Forward pass of the fixed-DM grid energy and its pullback (a ``jax.tree_util.Partial``) for the backward pass."""
    sym_proj = bind_symmetry(sym.obj if sym is not None else None, sym_arrays, nspin)

    def _energy(disp: jax.Array, eps: jax.Array) -> jax.Array:
        t_vna, t_h, t_xc, t_rvv = _deformed_grid_terms(
            disp, eps, dm_real, positions, cell, packed, na_packed, mesh_shape, nspin, xc_functional, use_rvv10, sym_proj
        )
        return t_vna + t_h + t_xc + t_rvv

    return jax.vjp(_energy, jnp.zeros_like(positions), jnp.zeros((3, 3), dtype=jnp.float64))


@jax.jit
def _apply_pullback(pullback: Any, cotangent: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Backward pass of ``_grid_energy_and_pullback``: gradients with respect to displacement and strain."""
    return pullback(cotangent)  # type: ignore[no-any-return]


def _kb_forces_and_stress(
    dm_out: jax.Array,
    structure: Structure,
    basis: BasisData,
    pattern: SparsePattern,
    two_center: TwoCenterMatrices,
    nspin: int,
) -> tuple[jax.Array, np.ndarray, np.ndarray]:
    """Compute exact 3-center KB energy, atomic forces, and stress from ``kb_link_p``/``dp``."""
    n_at = structure.n_atoms
    vol = float(structure.volume)
    f_kb = np.zeros((n_at, 3), dtype=np.float64)
    stress_kb = np.zeros((3, 3), dtype=np.float64)
    e_kb = float(jnp.real(jnp.sum(
        (dm_out[0] if nspin == 1 else (dm_out[0] + dm_out[1])) * two_center.v_kb
    )))

    if two_center.kb_link_p is None or two_center.kb_link_p.shape[0] == 0:
        return jnp.asarray(e_kb, dtype=jnp.float64), f_kb, stress_kb

    # Reconstruct atom_kb_shells metadata matching TwoCenterEvaluator
    atom_kb_shells: list[tuple[int, float, int, int]] = []
    for ia, sp_idx in enumerate(structure.species_indices):
        sp = basis.species[sp_idx]
        ek_arr = np.asarray(sp.kb_energies, dtype=np.float64)
        for k_sh in range(len(sp.kb_l)):
            l_k = int(sp.kb_l[k_sh])
            j2_k = int(sp.kb_j2[k_sh])
            ek_eff = float(ek_arr[k_sh])
            if nspin != 8 and j2_k > 0 and l_k > 0:
                ek_eff *= (j2_k + 1.0) / (2.0 * (2 * l_k + 1.0))
            atom_kb_shells.append((ia, ek_eff, l_k, j2_k))

    p_vals = np.asarray(two_center.kb_link_p, dtype=np.float64)
    dp_vals = np.asarray(two_center.kb_link_dp, dtype=np.float64)
    l_mu = two_center.kb_link_orb
    l_ki = two_center.kb_link_kb
    l_tk = np.asarray(two_center.kb_link_cell_shift, dtype=np.int32)

    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    orb_atom = np.asarray(basis.orb_atom, dtype=np.int32)
    dm_np = np.asarray(dm_out)
    dm_tot = np.real(dm_np[0]) if nspin == 1 else np.real(dm_np[0] + dm_np[1])

    if nspin != 8:
        # Threaded C++ kernel (D-041): the same pairs and sums as the NumPy loop below, which stays for spin-orbit
        # shells and for libraries built without the kernel.
        nat = kb_three_center_native(
            pattern,
            cell,
            pos,
            orb_atom,
            np.asarray(l_mu),
            np.asarray(l_ki),
            l_tk,
            np.arange(int(np.asarray(l_mu).shape[0])),
            p_vals,
            dp_vals,
            np.asarray([item[0] for item in atom_kb_shells], dtype=np.int32),
            np.asarray([item[1] for item in atom_kb_shells], dtype=np.float64),
            np.asarray([2 * item[2] + 1 for item in atom_kb_shells], dtype=np.int32),
            np.ascontiguousarray(dm_tot, dtype=np.float64),
            False,
        )
        if nat is not None and nat[2] is not None and nat[3] is not None:
            return jnp.asarray(nat[1], dtype=jnp.float64), nat[2], nat[3] / vol
    soc_cache: dict[tuple[int, int], np.ndarray] = {}
    e_kb_exact = 0.0
    shell_dims = [2 * item[2] + 1 for item in atom_kb_shells]

    for k_i, mu_arr, tk_arr, r_idx, c_idx, e_idx, p_mat, dp_mat in _iter_kb_shell_pairs(
        pattern=pattern,
        n_orb=basis.n_orb,
        n_kb_shells=len(atom_kb_shells),
        shell_dims=shell_dims,
        link_mu=np.asarray(l_mu, dtype=np.int64),
        link_k_inst=np.asarray(l_ki, dtype=np.int64),
        link_tk=np.asarray(l_tk, dtype=np.int64),
        p_vals=p_vals,
        dp_vals=dp_vals,
        link_m2_idx=None,
    ):
        assert dp_mat is not None
        ka, ek_eff, l_k, j2_k = atom_kb_shells[k_i]
        use_soc = nspin == 8 and j2_k > 0 and l_k > 0

        ia_r = orb_atom[mu_arr[r_idx]]
        rk_r = pos[ka] + tk_arr[r_idx] @ cell.T
        d_vec = rk_r - pos[ia_r]  # (n_valid, 3)

        if not use_soc:
            d_val = dm_tot[e_idx]
            p_dot = np.sum(p_mat[r_idx] * p_mat[c_idx], axis=1)
            e_kb_exact += ek_eff * float(np.sum(d_val * p_dot))
            # Both (mu, nu, t_ij) and (nu, mu, -t_ij) contribute; differentiating both
            # pm and pn over the symmetric pair table doubles the single-factor dpm term.
            dpm_pn = np.einsum("rmx,rm->rx", dp_mat[r_idx], p_mat[c_idx])
            grad_d = (2.0 * ek_eff * d_val)[:, None] * dpm_pn
        else:
            s_mat = soc_cache.get((l_k, j2_k))
            if s_mat is None:
                s_mat = _kb_soc_shell_spin_matrix(l_k, j2_k)
                soc_cache[(l_k, j2_k)] = s_mat
            # D = [D_uu, D_dd, Re(D_ud), Im(D_ud)], V_so = [uu, dd, ud, du]
            # Tr[D V] = Re(D_uu^* V_uu + D_dd^* V_dd + 2 D_ud^* V_ud)
            d_uu = dm_np[0, e_idx]
            d_dd = dm_np[1, e_idx]
            d_ud_r = np.real(dm_np[2, e_idx])
            d_ud_i = np.real(dm_np[3, e_idx])

            p_spin = np.einsum("rm,mnc->rnc", p_mat, s_mat)
            v_c = ek_eff * np.einsum("rnc,rn->rc", p_spin[r_idx], p_mat[c_idx])
            e_kb_exact += float(
                np.sum(
                    np.real(np.conj(d_uu) * v_c[:, 0])
                    + np.real(np.conj(d_dd) * v_c[:, 1])
                    + 2.0 * (d_ud_r * np.real(v_c[:, 2]) + d_ud_i * np.imag(v_c[:, 2]))
                )
            )
            d_uu_c = np.conj(d_uu)[:, None]
            d_dd_c = np.conj(d_dd)[:, None]
            d_ud_r_col = d_ud_r[:, None]
            d_ud_i_col = d_ud_i[:, None]

            def _soc_weights(dv: np.ndarray) -> np.ndarray:
                return ek_eff * (
                    np.real(d_uu_c * dv[:, :, 0])
                    + np.real(d_dd_c * dv[:, :, 1])
                    + 2.0 * (d_ud_r_col * np.real(dv[:, :, 2]) + d_ud_i_col * np.imag(dv[:, :, 2]))
                )

            # For nspin == 8, neither S_ud nor D_ud is symmetric under transpose,
            # so dP_r (left projector) and dP_c (right projector) must be evaluated explicitly.
            s_pc = np.einsum("mnc,rn->rmc", s_mat, p_mat)  # (M, dim_m, 4)
            grad_r = np.einsum("rmx,rm->rx", dp_mat[r_idx], _soc_weights(s_pc[c_idx]))
            grad_c = np.einsum("cnx,cn->cx", dp_mat[c_idx], _soc_weights(p_spin[r_idx]))

            ia_c = orb_atom[mu_arr[c_idx]]
            rk_c = pos[ka] + tk_arr[c_idx] @ cell.T
            d_vec_c = rk_c - pos[ia_c]

            np.add.at(f_kb, ia_r, grad_r)
            np.add.at(f_kb, ia_c, grad_c)
            f_kb[ka] -= np.sum(grad_r, axis=0) + np.sum(grad_c, axis=0)
            stress_kb += (grad_r.T @ d_vec + grad_c.T @ d_vec_c) / vol
            continue

        np.add.at(f_kb, ia_r, grad_d)
        f_kb[ka] -= np.sum(grad_d, axis=0)
        stress_kb += (grad_d.T @ d_vec) / vol

    return jnp.asarray(e_kb_exact, dtype=jnp.float64), f_kb, stress_kb


class StaticObservableEvaluator:
    r"""Post-SCF ``DM_out`` evaluator and variational force/stress engine implementing ``ObservableEvaluator``.

    Theory:
        Enforces the explicit post-SCF ``DM_out`` contract and computes atomic forces
        and stress tensor at fixed converged ``(dm_out, edm_out)``.

    Algorithm:
        1. Re-evaluates real-space density, Hartree, XC, dispersion, and Harris/KS
           energies directly from unmixed ``scf_state.dm_out``.
        2. Evaluates two-center kinetic/Pulay, 3-center KB, and neutral-atom pair
           forces/stress analytically.
        3. Evaluates grid forces/stress (``dfscf``, ``V_NA``, ``Hartree``, ``XC/GGA``,
           ``NLCC``, ``rVV10``, meta-GGA ``tau``) by reverse-mode differentiation wrt ``(positions, cell)``.
        4. Computes atomic and total spin moments and optionally applies final symmetry
           projection cleanup in Phase 2.

    Invariants:
        - ``dm_out_recomputed`` is always ``True``.
        - Zero mixing fallback between ``last_scf_free_e_ha`` and ``final_post_scf_free_e_ha``.

    SIESTA Mapping:
        Implements ``ObservableEvaluator`` replacing ``post_scf_work.F``,
        ``compute_energies.F90``, ``dfscf.F``, and ``siesta_forces.F``.
    """

    def evaluate(
        self,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        two_center: TwoCenterMatrices,
        kmesh: KPointMesh,
        scf_state: SCFState,
        config: SCFConfig,
        mesh_shape: tuple[int, int, int] | None = None,
        sym_proj: SymmetryProjector | None = None,
        compute_forces_stress: bool = True,
        components_out: dict[str, np.ndarray] | None = None,
        timings_out: dict[str, float] | None = None,
        exact_exchange: ExactExchange | None = None,
    ) -> Observables:
        r"""Execute mandatory post-SCF ``DM_out`` pass and compute variational forces & stress.

        Theory:
            Recomputes all grid-dependent KS energy terms from unmixed ``dm_out``
            and differentiates fixed-``(D, E)`` energy contributions wrt coordinates
            and lattice strain.

        Algorithm:
            Runs ``rhoofd(dm_out)``, Hartree FFT, XC FFI, optional dispersion,
            the reverse-mode derivative (``jax.vjp`` with unit cotangent, identical to ``jax.grad``) of the fixed-DM
            grid energy, and two-center/KB/Pulay sums (skipping gradient evaluation when
            ``compute_forces_stress=False``).

        Parameters:
            structure: ``Structure`` PyTree.
            basis: ``BasisData`` PyTree.
            pattern: Static ``SparsePattern``.
            two_center: Precomputed ``TwoCenterMatrices``.
            kmesh: ``KPointMesh`` PyTree.
            scf_state: Final ``SCFState`` from ``run_scf``.
            config: ``SCFConfig``.
            mesh_shape: Optional 3D mesh shape matching the SCF run.
            sym_proj: Optional ``SymmetryProjector`` (Phase 2).
            compute_forces_stress: When ``False``, skips analytic and ``jax.grad``
                force/stress differentiation and returns zero forces and stress.
            components_out: Optional dictionary filled with the unsymmetrised per-term stress (Ha/Bohr^3, ``(3, 3)``
                arrays) for the terms ``kinetic``, ``pulay_overlap``, ``kb``, ``na_pair``, ``na_grid``, ``hartree``,
                ``xc`` and ``dispersion`` (see ``jax_siesta.testing.strain_fd``). It costs one extra ``jax.grad``
                per grid term and requires ``compute_forces_stress=True``.
            timings_out: Optional dictionary filled with wall times in seconds per stage: ``fresh_grid`` (density,
                kinetic density, neutral atom, Hartree, XC at ``dm_out``), ``kb``, ``dispersion``, ``grid_forward``
                and ``grid_backward`` (the two halves of the reverse-mode derivative of the grid energy),
                ``grid_components`` (only with ``components_out``), ``two_center``, ``moments`` and ``total``. Each
                stage waits for its device work, so timing adds synchronisation points; ``None`` adds none.
            exact_exchange: Optional exact-exchange stage of a hybrid functional (built here when omitted and the
                functional is a hybrid). Its energy enters the total energy as ``e_exact_exchange_ha`` and its
                derivatives at fixed density matrix enter the forces and the stress (``components_out`` then gets
                an ``exact_exchange`` entry).

        Returns:
            Fully provenanced ``Observables`` PyTree.

        Invariants:
            - ``Observables.dm_out_recomputed == True``.
            - Forces sum to zero (within numerical noise) under periodic translation.

        SIESTA Mapping:
            Replaces ``post_scf_work``, ``compute_energies``, and ``siesta_forces``.
        """
        if mesh_shape is None:
            mesh_shape = choose_mesh_dimensions(
                structure.cell, config.mesh_cutoff_ry, criterion=config.mesh_criterion, mesh_subdivisions=config.mesh_subdivisions
            )
        t_total = time.perf_counter()
        t_mark = t_total
        n1, n2, n3 = mesh_shape
        vol = float(structure.volume)
        dvol = vol / float(n1 * n2 * n3)

        dm_out = jax.lax.stop_gradient(scf_state.dm_out)
        edm_out = jax.lax.stop_gradient(scf_state.edm_out)
        dm_real = jnp.real(dm_out) if jnp.iscomplexobj(dm_out) else dm_out
        edm_real = jnp.real(edm_out) if jnp.iscomplexobj(edm_out) else edm_out

        xc_eval = LibxcFFIEvaluator(config.xc_functional)
        disp_mode = config.dispersion.upper()
        rvv10 = RVV10Dispersion() if disp_mode == "RVV10" else None
        d3 = DFTD3Dispersion(config.xc_functional) if disp_mode in ("DFT-D3", "DFTD3", "D3") else None

        packed = _pack_basis_grid_buffers(basis, pattern)
        na_packed = _pack_na_buffers(structure, basis)

        # 1. Mandatory fresh non-mixed post-SCF evaluation from DM_out
        sym_static = _StaticByIdentity(sym_proj) if sym_proj is not None else None
        sym_arrays = symmetry_jit_arrays(sym_proj, config.nspin)
        mesh_static = (int(mesh_shape[0]), int(mesh_shape[1]), int(mesh_shape[2]))
        rho_tot, rho_core, q_grid_arr, e_hartree, e_xc, e_v_na_grid = _fresh_grid_energies(
            dm_real,
            structure.positions,
            structure.cell,
            packed,
            na_packed,
            jnp.asarray(dvol, dtype=jnp.float64),
            mesh_shape=mesh_static,
            nspin=config.nspin,
            xc_functional=config.xc_functional,
            sym=sym_static,
            sym_arrays=sym_arrays,
        )
        q_grid = float(q_grid_arr)
        n_electrons = basis.total_valence_electrons - config.net_charge
        charge_residual = jnp.asarray(q_grid - n_electrons, dtype=jnp.float64)
        t_mark = _stage(timings_out, "fresh_grid", t_mark, e_hartree, e_xc, e_v_na_grid, charge_residual)

        dm_tot = dm_real[0] if config.nspin == 1 else (dm_real[0] + dm_real[1])
        edm_tot = edm_real[0] if config.nspin == 1 else (edm_real[0] + edm_real[1])
        e_kinetic = jnp.sum(dm_tot * two_center.kinetic)
        e_kb, f_kb, stress_kb = _kb_forces_and_stress(
            dm_out, structure, basis, pattern, two_center, config.nspin
        )
        t_mark = _stage(timings_out, "kb", t_mark, e_kb, f_kb, stress_kb)
        e_na = e_v_na_grid + two_center.e_na_pair

        if rvv10 is not None:
            e_disp = rvv10.energy_from_density(rho_tot + rho_core, structure.cell)
            f_disp = jnp.zeros((structure.n_atoms, 3), dtype=jnp.float64)
            stress_disp_extra = jnp.zeros((3, 3), dtype=jnp.float64)
        elif d3 is not None:
            e_disp, f_disp, stress_disp_extra = d3.evaluate(structure)
        else:
            e_disp = jnp.array(0.0, dtype=jnp.float64)
            f_disp = jnp.zeros((structure.n_atoms, 3), dtype=jnp.float64)
            stress_disp_extra = jnp.zeros((3, 3), dtype=jnp.float64)

        t_mark = _stage(timings_out, "dispersion", t_mark, e_disp, f_disp, stress_disp_extra)
        # Exact exchange of a hybrid functional, from the same unmixed density matrix.
        if exact_exchange is None and exact_exchange_spec(config.xc_functional).is_hybrid:
            from jax_siesta.hfx import build_exact_exchange

            exact_exchange = build_exact_exchange(structure, basis, pattern, config, k_grid=kmesh.grid_shape)
        e_exx = 0.0
        if exact_exchange is not None:
            e_exx = float(exact_exchange.matrix_and_energy(np.asarray(dm_out), config.nspin)[1])
            t_mark = _stage(timings_out, "exact_exchange", t_mark)
        e_entropy = scf_state.entropy_ha
        total_energy = e_kinetic + e_kb + e_na + e_hartree + e_xc + e_disp + e_exx
        final_post_scf_free_e = total_energy + e_entropy

        # Harris-Foulkes free energy of the last SCF iteration (scf/driver.py, D-023). NaN (never an alias of the KS
        # energy, review M3) when the state does not carry one.
        harris_energy = (
            jnp.asarray(scf_state.harris_energy_ha, dtype=jnp.float64)
            if scf_state.harris_energy_ha is not None
            else jnp.array(jnp.nan, dtype=jnp.float64)
        )

        # 2. Fixed-(D, E) variational grid forces & stress: derivatives of the grid energy with respect to
        # (displacement, homogeneous strain eps), so both the cell deformation and the scaled coordinates contribute.
        def _grid_terms_here(disp: jax.Array, eps: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return _deformed_grid_terms(
                disp,
                eps,
                dm_real,
                structure.positions,
                structure.cell,
                packed,
                na_packed,
                mesh_static,
                config.nspin,
                config.xc_functional,
                rvv10 is not None,
                sym_proj,
            )

        oa = np.asarray(basis.orb_atom, dtype=np.int32)
        if not compute_forces_stress:
            forces = jnp.zeros((structure.n_atoms, 3), dtype=jnp.float64)
            stress = jnp.zeros((3, 3), dtype=jnp.float64)
        else:
            # jax.vjp with a unit cotangent is what jax.grad does; the compiled forward pass returns the pullback so
            # the forward pass and the backward kernels can be timed separately.
            e_grid, grid_pullback = _grid_energy_and_pullback(
                dm_real,
                structure.positions,
                structure.cell,
                packed,
                na_packed,
                mesh_shape=mesh_static,
                nspin=config.nspin,
                xc_functional=config.xc_functional,
                use_rvv10=rvv10 is not None,
                sym=sym_static,
                sym_arrays=sym_arrays,
            )
            t_mark = _stage(timings_out, "grid_forward", t_mark, e_grid)
            g_grid_pos, g_grid_eps = _apply_pullback(grid_pullback, jnp.ones_like(e_grid))
            t_mark = _stage(timings_out, "grid_backward", t_mark, g_grid_pos, g_grid_eps)
            f_grid = -g_grid_pos
            stress_grid = g_grid_eps / vol
            grid_term_stress: dict[str, np.ndarray] = {}
            if components_out is not None:
                zero_d, zero_e = jnp.zeros_like(structure.positions), jnp.zeros((3, 3), dtype=jnp.float64)
                for idx, name in enumerate(("na_grid", "hartree", "xc", "dispersion")):
                    g_idx = jax.grad(lambda d, e, i=idx: _grid_terms_here(d, e)[i], argnums=1)(zero_d, zero_e)
                    grid_term_stress[name] = np.asarray(g_idx, dtype=np.float64) / vol
                t_mark = _stage(timings_out, "grid_components", t_mark)

            # 3. Two-center Kinetic + Pulay forces & stress
            src = np.asarray(pattern.src_orb, dtype=np.int32)
            dst = np.asarray(pattern.dst_unit_orb, dtype=np.int32)
            dm_t_np = np.asarray(dm_tot, dtype=np.float64)
            edm_t_np = np.asarray(edm_tot, dtype=np.float64)
            ds_dr = np.asarray(two_center.d_overlap_dr, dtype=np.float64)
            dt_dr = np.asarray(two_center.d_kinetic_dr, dtype=np.float64)
            r_vec = np.asarray(pattern.edge_vectors, dtype=np.float64)

            # F_I = -Tr[D dT/dR_I] + Tr[E dS/dR_I]; edge vector r = R_{I(dst)} + T - R_{I(src)}
            g_pair = (-dm_t_np[:, None] * dt_dr) + (edm_t_np[:, None] * ds_dr)  # (n_edges, 3)
            f_two = np.zeros((structure.n_atoms, 3), dtype=np.float64)
            np.add.at(f_two, oa[dst], g_pair)
            np.add.at(f_two, oa[src], -g_pair)
            stress_two = np.einsum("ea,eb->ab", -g_pair, r_vec) / vol
            if components_out is not None:
                g_kin = -dm_t_np[:, None] * dt_dr
                g_pul = edm_t_np[:, None] * ds_dr
                components_out["kinetic"] = np.einsum("ea,eb->ab", -g_kin, r_vec) / vol
                components_out["pulay_overlap"] = np.einsum("ea,eb->ab", -g_pul, r_vec) / vol
                components_out["kb"] = np.asarray(stress_kb, dtype=np.float64)
                components_out["na_pair"] = np.asarray(two_center.stress_na, dtype=np.float64)
                components_out["na_grid"] = grid_term_stress["na_grid"]
                components_out["hartree"] = grid_term_stress["hartree"]
                components_out["xc"] = grid_term_stress["xc"]
                components_out["dispersion"] = np.asarray(stress_disp_extra, dtype=np.float64) + grid_term_stress[
                    "dispersion"
                ]

            forces = jnp.asarray(f_two + f_kb, dtype=jnp.float64) + two_center.f_na_atoms + f_grid + f_disp
            stress = (
                jnp.asarray(stress_two + stress_kb, dtype=jnp.float64)
                + two_center.stress_na
                + stress_grid
                + stress_disp_extra
            )
            if exact_exchange is not None:
                # Exact exchange at fixed density matrix; its Pulay part is in the energy-density-matrix term above,
                # because the eigenvalues that build E are those of the Hamiltonian with the exchange matrix.
                f_exx, de_exx = exact_exchange.forces_and_strain_derivative(np.asarray(dm_out), config.nspin)
                forces = forces + jnp.asarray(f_exx, dtype=jnp.float64)
                stress = stress + jnp.asarray(de_exx, dtype=jnp.float64) / vol
                if components_out is not None:
                    components_out["exact_exchange"] = np.asarray(de_exx, dtype=np.float64) / vol
                t_mark = _stage(timings_out, "exact_exchange", t_mark)
            # Symmetrize stress tensor (numerical noise cleanup)
            stress = 0.5 * (stress + stress.T)

            if sym_proj is not None:
                forces = sym_proj.project_forces(forces)
                stress = sym_proj.project_stress(stress)
            t_mark = _stage(timings_out, "two_center", t_mark, forces, stress)

        # 4. Atomic & total spin moments (Mulliken overlap weighted)
        src = np.asarray(pattern.src_orb, dtype=np.int32)
        s_vals = np.asarray(two_center.overlap, dtype=np.float64)
        dm_r_np = np.asarray(dm_real, dtype=np.float64)
        atom_m = np.zeros((structure.n_atoms, 3), dtype=np.float64)
        if config.nspin == 2:
            mz_edges = (dm_r_np[0] - dm_r_np[1]) * s_vals
            np.add.at(atom_m[:, 2], oa[src], mz_edges)
        elif config.nspin in (4, 8):
            mz_edges = (dm_r_np[0] - dm_r_np[1]) * s_vals
            mx_edges = (2.0 * dm_r_np[2]) * s_vals
            my_edges = (-2.0 * dm_r_np[3]) * s_vals
            np.add.at(atom_m[:, 0], oa[src], mx_edges)
            np.add.at(atom_m[:, 1], oa[src], my_edges)
            np.add.at(atom_m[:, 2], oa[src], mz_edges)
        tot_m = jnp.asarray(np.sum(atom_m, axis=0), dtype=jnp.float64)

        d_max = jnp.max(jnp.abs(scf_state.dm_out - scf_state.dm_in))
        h_max = scf_state.prev_h_max
        t_mark = _stage(timings_out, "moments", t_mark, tot_m, d_max)
        if timings_out is not None:
            timings_out["total"] = t_mark - t_total

        return Observables(
            outcome=scf_state.outcome,
            converged=scf_state.converged,
            n_iterations=scf_state.iteration,
            dm_out_recomputed=True,
            last_scf_free_e_ha=scf_state.last_scf_free_e_ha,
            last_scf_cycle=scf_state.iteration,
            final_post_scf_free_e_ha=final_post_scf_free_e,
            total_energy_ha=total_energy,
            harris_energy_ha=harris_energy,
            harris_provenance=scf_state.harris_provenance,
            e_kinetic_ha=e_kinetic,
            e_na_ha=e_na,
            e_kb_ha=e_kb,
            e_hartree_ha=e_hartree,
            e_xc_ha=e_xc,
            e_disp_ha=e_disp,
            e_entropy_ha=e_entropy,
            fermi_level_ha=scf_state.fermi_level_ha,
            forces_ha_per_bohr=forces,
            stress_ha_per_bohr3=stress,
            total_spin_moment=tot_m,
            atomic_spin_moments=jnp.asarray(atom_m, dtype=jnp.float64),
            charge_residual=charge_residual,
            d_max=d_max,
            h_max_ha=h_max,
            xc_backend=xc_eval.backend_name,
            e_exact_exchange_ha=e_exx,
            forces_available=bool(compute_forces_stress),
        )
