r"""C++ Fourier-Bessel two-center integral wrapper (:math:`S, T, V^{\text{KB}}, E^{\text{NA-pair}}`).

Theory:
    Evaluates sparse overlap :math:`S_{\mu\nu}`, kinetic :math:`T_{\mu\nu}`,
    Kleinman-Bylander nonlocal :math:`V^{\text{KB}}_{\mu\nu}` (including scalar and
    full :math:`2\times 2` spin-orbit coupling for ``nspin=8``), and short-range
    neutral-atom electrostatic pair corrections :math:`E_{\text{NA-pair}}` via
    spherical Bessel transforms and real Gaunt coefficients in compiled C++.

Algorithm:
    Invokes ``jsiesta_eval_two_center_st``, ``jsiesta_eval_kb_overlaps``, and
    ``jsiesta_eval_na_pair`` in ``libjsiesta_ffi``, then contracts KB overlaps
    with Clebsch-Gordan spin-orbit projectors to form Hermitian :math:`V^{\text{KB}}`
    and :math:`V^{\text{KB}}_{\text{SO}}`.

Invariants:
    - :math:`S_{\mu\nu}(\mathbf{T}) = S_{\nu\mu}(-\mathbf{T})` and
      :math:`T_{\mu\nu}(\mathbf{T}) = T_{\nu\mu}(-\mathbf{T})`.
    - When :math:`j=l+1/2` and :math:`j=l-1/2` KB energies are degenerate,
      the spin-orbit difference vanishes identically.

SIESTA Mapping:
    Implements ``TwoCenterIntegrals`` replacing ``matel.F90``, ``kinefsm.F``,
    ``naefs.F``, and ``nlefsm.f``.
"""

from __future__ import annotations

import ctypes
import hashlib
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from jax_siesta.core.ffi import load_ffi_library
from jax_siesta.core.types import BasisData, SCFConfig, SparsePattern, Structure, TwoCenterMatrices


def _radial_ft(r: np.ndarray, f: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3D Fourier transform ``4 pi int r^2 f(r) j0(k r) dr`` of a spherical function."""
    kr = np.outer(k, r)
    j0 = np.where(kr > 1e-12, np.sin(kr) / np.where(kr > 1e-12, kr, 1.0), 1.0)
    return 4.0 * np.pi * np.trapezoid(j0 * (r**2 * f)[None, :], r, axis=1)


_W_CACHE: dict[tuple[str, ...], Any] = {}


def neutral_atom_density_two_center(species_v: Any, species_n: Any) -> tuple[Any, float]:
    r"""Spline of :math:`W(R) = \int V_{NA,v}(\mathbf r)\, n_{\text{at},n}(\mathbf r - \mathbf R)\, d^3r`.

    Theory:
        The neutral-atom grid energy :math:`\int V_{NA}\,\rho` is split as
        :math:`\int V_{NA}\,\delta\rho` (mesh) :math:`+ \sum_{I,J}\int V_{NA,I}\, n_J` (exact). The second
        part is a sum of spherical two-centre integrals,
        :math:`W(R) = \frac{1}{2\pi^2}\int_0^\infty k^2\,\hat V(k)\,\hat n(k)\, j_0(kR)\, dk`, which removes
        its mesh (egg-box) error from energies, forces and stress (review H2/M5, D-017).

    Algorithm:
        Radial Fourier transforms on a dense k grid (0..80 Bohr⁻¹), a cubic spline of :math:`W` on
        :math:`R\in[0, r_{c,v} + r_{c,n}]`, cached per species pair.

    Parameters:
        species_v: Species owning :math:`V_{NA}`.
        species_n: Species owning the atomic density :math:`n_{\text{at}}` (``rho_na``).

    Returns:
        ``(spline, r_cut)``; ``spline(R, 1)`` gives :math:`dW/dR`; :math:`W = 0` for :math:`R \ge r_{cut}`.

    Invariants:
        ``spline(0) == 4 pi int r^2 V_NA n_at dr`` when both are the same species.

    SIESTA Mapping:
        SIESTA's ``Ena`` two-centre term (``naefs``) with ``DEna`` on the mesh (``dhscf``).
    """
    r_v = np.asarray(species_v.r_grid, dtype=np.float64)
    r_n = np.asarray(species_n.r_grid, dtype=np.float64)
    v = np.asarray(species_v.v_na, dtype=np.float64)
    n = np.asarray(species_n.rho_na, dtype=np.float64)
    key = tuple(hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest() for a in (r_v, v, r_n, n))
    if key in _W_CACHE:
        return _W_CACHE[key]
    rc_v = float(r_v[np.nonzero(np.abs(v) > 0.0)[0][-1]]) if np.any(v) else float(r_v[-1])
    rc_n = float(r_n[np.nonzero(n > 0.0)[0][-1]]) if np.any(n) else float(r_n[-1])
    k = np.linspace(0.0, 80.0, 8001)
    integrand = k**2 * _radial_ft(r_v, v, k) * _radial_ft(r_n, n, k)
    r_cut = rc_v + rc_n
    big_r = np.linspace(0.0, r_cut, 1201)
    kr = np.outer(big_r, k)
    j0 = np.where(kr > 1e-12, np.sin(kr) / np.where(kr > 1e-12, kr, 1.0), 1.0)
    w_vals = np.trapezoid(j0 * integrand[None, :], k, axis=1) / (2.0 * np.pi**2)
    w_vals[-1] = 0.0
    out = (CubicSpline(big_r, w_vals, bc_type=((1, 0.0), (1, 0.0))), r_cut)
    _W_CACHE[key] = out
    return out


def neutral_atom_self_energy(species: Any) -> float:
    r"""One-centre constant that turns the neutral-atom energy into the absolute Kohn–Sham energy.

    Theory:
        With :math:`\rho_L = -\nabla^2 V_{\text{loc}}/4\pi` (charge :math:`-Z`), the neutral-atom charge
        :math:`\rho_{NA} = \rho_L + n_{\text{at}}` and :math:`V_{NA} = V_H[\rho_{NA}]`, the point-ion
        electrostatic energy :math:`\int V_{\text{loc}} n + E_H[n] + E_{ii}` equals
        :math:`\int V_{NA} n + E_H[\delta n] + \sum_{I<J} U_{IJ}` (what the grid and pair terms compute)
        plus, per atom,

        .. math:: C_I = \tfrac12 J[\rho_{NA},\rho_{NA}] - \tfrac12 J[\rho_L,\rho_L]
                  - \int V_{NA}\, n_{\text{at}}\, d^3r .

        When the atomic part :math:`\sum_{IJ}\int V_{NA,I} n_J` is evaluated exactly
        (:func:`neutral_atom_density_two_center`, off-site pairs only) its one-centre term
        :math:`\int V_{NA}\,n_{\text{at}}` cancels the last term, so this function returns
        :math:`\tfrac12 J[\rho_{NA},\rho_{NA}] - \tfrac12 J[\rho_L,\rho_L]`.

        For a spherical charge with potential :math:`V`, :math:`\tfrac12 J = \tfrac{1}{8\pi}\int|\nabla V|^2
        = \tfrac12\int_0^\infty r^2 V'(r)^2 dr`; the :math:`-Z/r` tail of :math:`V_{\text{loc}}` beyond the
        table adds :math:`Z^2/(2 r_{\max})`.

    Algorithm:
        Cubic-spline derivatives of the radial tables and trapezoidal quadrature on the species grid.

    Parameters:
        species: ``SpeciesData`` with ``r_grid``, ``v_na``, ``v_local``, ``rho_na`` (valence density of the
            neutral pseudo-atom, integrating to :math:`Z`) and ``valence_charge``.

    Returns:
        :math:`C_I` in Hartree.

    Invariants:
        Independent of geometry, so it changes energies only (no forces or stress). Adding it makes
        ``total_energy`` equal SIESTA's ``Etot`` (``Ena + DEna - Eions`` convention).

    SIESTA Mapping:
        Equals the one-centre part of SIESTA's ``Ena`` (``naefs`` self term) minus the atom's ``Eions``
        share (``m_energies``), i.e. the constant missing from ``Ena + DEna - Eions`` otherwise.
    """
    r = np.asarray(species.r_grid, dtype=np.float64)
    v_na = np.asarray(species.v_na, dtype=np.float64)
    v_loc = np.asarray(species.v_local, dtype=np.float64)
    n_at = np.asarray(species.rho_na, dtype=np.float64)
    z_val = float(species.valence_charge)
    dv_na = CubicSpline(r, v_na)(r, 1)
    dv_loc = CubicSpline(r, v_loc)(r, 1)
    half_j_na = 0.5 * np.trapezoid(r**2 * dv_na**2, r)
    half_j_loc = 0.5 * np.trapezoid(r**2 * dv_loc**2, r) + 0.5 * z_val**2 / r[-1]
    del n_at  # its one-centre term -int V_NA n_at cancels W_II(0) of the exact atomic part (see Theory)
    return float(half_j_na - half_j_loc)


def _setup_c_signatures(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_two_center_configured", False):
        return
    c_i32 = ctypes.c_int32
    c_i32_p = ctypes.POINTER(ctypes.c_int32)
    c_f64_p = ctypes.POINTER(ctypes.c_double)

    lib.jsiesta_eval_two_center_st.argtypes = [
        c_i32, c_i32, c_f64_p, c_f64_p, c_i32_p, c_f64_p,
        c_i32, c_i32_p, c_i32_p, c_i32_p, c_i32_p, c_f64_p,
        c_f64_p, c_f64_p, c_f64_p, c_f64_p,
    ]
    lib.jsiesta_eval_two_center_st.restype = c_i32

    lib.jsiesta_eval_kb_overlaps.argtypes = [
        c_i32, c_i32, c_f64_p, c_f64_p, c_i32_p, c_f64_p,
        c_i32, c_i32, c_f64_p, c_f64_p, c_i32_p, c_f64_p,
        c_i32, c_i32_p, c_i32_p, c_i32_p, c_i32_p, c_f64_p,
        c_f64_p, c_f64_p,
    ]
    lib.jsiesta_eval_kb_overlaps.restype = c_i32

    lib.jsiesta_eval_na_pair.argtypes = [
        c_i32, c_i32, c_f64_p, c_f64_p, c_f64_p, c_f64_p,
        c_i32, c_i32_p, c_i32_p, c_f64_p, c_f64_p, c_f64_p,
    ]
    lib.jsiesta_eval_na_pair.restype = c_i32
    lib._two_center_configured = True  # type: ignore[attr-defined]


def _real_to_complex_sh_row(l_val: int, m_real: int) -> dict[int, complex]:
    """Express real spherical harmonic :math:`Y^{\\text{real}}_{l, m}` in complex :math:`Y^c_{l, m_c}` basis."""
    if m_real == 0:
        return {0: 1.0 + 0.0j}
    am = abs(m_real)
    phase = (-1) ** am
    inv_s2 = 1.0 / np.sqrt(2.0)
    if m_real > 0:
        # Y_real(m>0) = (phase * Y^c_{+m} + Y^c_{-m}) / sqrt(2)
        return {am: phase * inv_s2, -am: inv_s2}
    # Y_real(m<0) = (phase * Y^c_{+m} - Y^c_{-m}) / (i sqrt(2))
    return {am: -1j * phase * inv_s2, -am: 1j * inv_s2}


def _clebsch_gordan_spin_half(l_val: int, m_c: int, j2: int, mj2: int) -> tuple[complex, complex]:
    """Return ``(c_up, c_down)`` coupling ``|l, m_c> x {|up>, |down>}`` into ``|j=j2/2, mj=mj2/2>``."""
    denom = float(2 * l_val + 1)
    c_up = 0.0j
    c_dn = 0.0j
    is_plus = j2 == 2 * l_val + 1
    if 2 * m_c + 1 == mj2:
        num = l_val + 0.5 * mj2 + 0.5 if is_plus else l_val - 0.5 * mj2 + 0.5
        val = np.sqrt(max(0.0, num) / denom)
        c_up = val if is_plus else -val
    if 2 * m_c - 1 == mj2:
        num = l_val - 0.5 * mj2 + 0.5 if is_plus else l_val + 0.5 * mj2 + 0.5
        c_dn = np.sqrt(max(0.0, num) / denom)
    return complex(c_up), complex(c_dn)


def _kb_soc_shell_spin_matrix(l_val: int, j2: int) -> np.ndarray:
    """Build ``(2l+1, 2l+1, 4)`` projector matrix in ``(up-up, dn-dn, up-dn, dn-up)`` for shell ``(l, j)``."""
    dim = 2 * l_val + 1
    if j2 <= 0 or l_val == 0:
        # Scalar projector: identity on up-up and dn-dn
        mat = np.zeros((dim, dim, 4), dtype=np.complex128)
        for i in range(dim):
            mat[i, i, 0] = 1.0
            mat[i, i, 1] = 1.0
        return mat

    # C[m_real_idx, alpha (0=up,1=dn), mj_idx] = <l m_real, alpha | j m_j>
    n_mj = j2 + 1
    coeffs = np.zeros((dim, 2, n_mj), dtype=np.complex128)
    for m_idx, m_real in enumerate(range(-l_val, l_val + 1)):
        expansion = _real_to_complex_sh_row(l_val, m_real)
        for mj_idx in range(n_mj):
            mj2 = -j2 + 2 * mj_idx
            cu_tot = 0.0j
            cd_tot = 0.0j
            for m_c, w_mc in expansion.items():
                cu, cd = _clebsch_gordan_spin_half(l_val, m_c, j2, mj2)
                cu_tot += np.conj(w_mc) * cu
                cd_tot += np.conj(w_mc) * cd
            coeffs[m_idx, 0, mj_idx] = cu_tot
            coeffs[m_idx, 1, mj_idx] = cd_tot

    # Sum over m_j: M_{m, m', alpha, beta} = sum_{m_j} C[m, alpha, mj] * conj(C[m', beta, mj])
    full = np.einsum("iam,jbm->ijab", coeffs, np.conj(coeffs))
    out = np.zeros((dim, dim, 4), dtype=np.complex128)
    out[:, :, 0] = full[:, :, 0, 0]
    out[:, :, 1] = full[:, :, 1, 1]
    out[:, :, 2] = full[:, :, 0, 1]
    out[:, :, 3] = full[:, :, 1, 0]
    return out


def _pack_pao_tables(
    basis: BasisData,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack global PAO radial tables and per-orbital global shell indices."""
    pao_tables: list[np.ndarray] = []
    pao_r_max: list[float] = []
    pao_rcut: list[float] = []
    pao_l: list[int] = []
    sp_shell_offsets: list[int] = [0]
    n_pao_pts = int(basis.species[0].pao_r.shape[0])
    for sp in basis.species:
        r_max = float(sp.pao_r[-1])
        vals = np.asarray(sp.pao_values, dtype=np.float64)
        if vals.shape[1] != n_pao_pts:
            r_unif = np.linspace(0.0, r_max, n_pao_pts)
            vals = np.stack([np.interp(r_unif, np.asarray(sp.pao_r), row) for row in vals], axis=0)
        for sh_i in range(vals.shape[0]):
            pao_tables.append(vals[sh_i])
            pao_r_max.append(r_max)
            pao_rcut.append(float(sp.pao_rcut[sh_i]))
            pao_l.append(int(sp.pao_l[sh_i]))
        sp_shell_offsets.append(len(pao_tables))

    orb_g_shell = np.array(
        [sp_shell_offsets[basis.orb_species[mu]] + basis.orb_shell[mu] for mu in range(basis.n_orb)],
        dtype=np.int32,
    )
    return (
        n_pao_pts,
        np.asarray(pao_r_max, dtype=np.float64),
        np.asarray(pao_rcut, dtype=np.float64),
        np.asarray(pao_l, dtype=np.int32),
        np.ascontiguousarray(np.stack(pao_tables, axis=0), dtype=np.float64),
        orb_g_shell,
    )


def _eval_st_subset(
    n_pao_pts: int,
    pao_r_max: np.ndarray,
    pao_rcut: np.ndarray,
    pao_l: np.ndarray,
    pao_vals: np.ndarray,
    edge_sh1: np.ndarray,
    edge_m1: np.ndarray,
    edge_sh2: np.ndarray,
    edge_m2: np.ndarray,
    edge_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call C++ ``jsiesta_eval_two_center_st`` on an arbitrary edge subset."""
    lib = load_ffi_library()
    _setup_c_signatures(lib)
    n_shells = int(pao_vals.shape[0])
    n_edges = int(edge_sh1.shape[0])
    out_s = np.zeros(n_edges, dtype=np.float64)
    out_t = np.zeros(n_edges, dtype=np.float64)
    out_ds = np.zeros((n_edges, 3), dtype=np.float64)
    out_dt = np.zeros((n_edges, 3), dtype=np.float64)
    if n_edges == 0:
        return out_s, out_t, out_ds, out_dt

    rc = lib.jsiesta_eval_two_center_st(
        ctypes.c_int32(n_shells),
        ctypes.c_int32(n_pao_pts),
        pao_r_max.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pao_rcut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pao_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        pao_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(n_edges),
        np.ascontiguousarray(edge_sh1, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(edge_m1, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(edge_sh2, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(edge_m2, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(edge_vec, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_ds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_dt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise RuntimeError(f"jsiesta_eval_two_center_st failed with code {rc}")
    return out_s, out_t, out_ds, out_dt


class TwoCenterEvaluator:
    """C++ Fourier-Bessel two-center integral evaluator implementing ``TwoCenterIntegrals``.

    Theory:
        Computes :math:`S, T, V^{\\text{KB}}` (scalar and SOC), and short-range
        neutral-atom pair energy :math:`E_{\\text{NA-pair}}` with exact pair derivatives.

    Algorithm:
        Packs species PAO and KB radial tables into contiguous NumPy arrays,
        dispatches compiled C++ ``libjsiesta_ffi`` Fourier-Bessel routines, and
        assembles sparse edge arrays and KB spin-orbit couplings.

    Invariants:
        - Hermiticity across ``pattern.transpose_edge_idx``.
        - Zero NaNs or non-finite values.

    SIESTA Mapping:
        Implements ``TwoCenterIntegrals`` replacing ``matel.F90``, ``kinefsm.F``,
        ``naefs.F``, and ``nlefsm.f``.
    """

    def evaluate(
        self,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        config: SCFConfig,
    ) -> TwoCenterMatrices:
        """Evaluate sparse two-center matrices :math:`S, T, V^{\\text{KB}}, E_{\\text{NA-pair}}` and derivatives.

        Theory:
            Uses Fourier-Bessel convolution for :math:`S, T, P_{\\mu k}` and short-range
            neutral-atom density overlap for :math:`E_{\\text{NA-pair}}`.

        Algorithm:
            Calls ``jsiesta_eval_two_center_st``, ``jsiesta_eval_kb_overlaps``, and
            ``jsiesta_eval_na_pair`` in ``libjsiesta_ffi``. The orbital/projector links and the
            neutral-atom pairs are found by vectorised per-atom distance tables over all images
            (``_enumerate_kb_links``, ``_enumerate_atom_pairs``), in the order of the original scalar
            loops, so the cost of the search is linear in the number of atoms times the images.

        Parameters:
            structure: Periodic or molecular ``Structure``.
            basis: ``BasisData`` for unit-cell orbitals and KB projectors.
            pattern: Static ``SparsePattern`` of interacting orbital pairs.
            config: ``SCFConfig`` controlling spin mode (``nspin=8`` enables SOC KB).

        Returns:
            ``TwoCenterMatrices`` PyTree.

        Invariants:
            Returned arrays match ``pattern.n_edges`` along the sparse dimension.

        SIESTA Mapping:
            Replaces ``overlap``, ``kinefsm``, ``nlefsm``, and ``naefs`` initialization.
        """
        lib = load_ffi_library()
        _setup_c_signatures(lib)

        # 1. Pack global unique PAO shells across species and evaluate S, T + derivatives
        n_pao_pts, pao_r_max_arr, pao_rcut_arr, pao_l_arr, pao_vals_arr, orb_g_shell = _pack_pao_tables(basis)

        src_orb = np.asarray(pattern.src_orb, dtype=np.int32)
        dst_orb = np.asarray(pattern.dst_unit_orb, dtype=np.int32)
        edge_vec = np.ascontiguousarray(np.asarray(pattern.edge_vectors, dtype=np.float64))

        orb_m = np.array(basis.orb_m, dtype=np.int32)
        edge_sh1 = np.ascontiguousarray(orb_g_shell[src_orb], dtype=np.int32)
        edge_m1 = np.ascontiguousarray(orb_m[src_orb], dtype=np.int32)
        edge_sh2 = np.ascontiguousarray(orb_g_shell[dst_orb], dtype=np.int32)
        edge_m2 = np.ascontiguousarray(orb_m[dst_orb], dtype=np.int32)

        out_s, out_t, out_ds, out_dt = _eval_st_subset(
            n_pao_pts,
            pao_r_max_arr,
            pao_rcut_arr,
            pao_l_arr,
            pao_vals_arr,
            edge_sh1,
            edge_m1,
            edge_sh2,
            edge_m2,
            edge_vec,
        )
        return _eval_kb_and_na_pair(
            structure,
            basis,
            pattern,
            config,
            n_pao_pts,
            pao_r_max_arr,
            pao_rcut_arr,
            pao_l_arr,
            pao_vals_arr,
            orb_g_shell,
            out_s,
            out_t,
            out_ds,
            out_dt,
        )


def _build_kb_pattern_index(
    pattern: SparsePattern,
    n_orb: int,
    max_link_shift: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray]:
    """Encode ``(src, dst, t_ij)`` pattern edges into sorted ``int64`` lookup keys.

    Returns ``(sorted_keys, sorted_edge_idx, b_stride, b_half, bias, shifts_i64)`` so
    any batch of candidate ``(mu_r, mu_c, tk_r - tk_c)`` pairs can be matched via
    additive ``row_key[:, None] + col_key[None, :] + bias`` and ``np.searchsorted``.
    """
    shifts = np.asarray(pattern.cell_shifts, dtype=np.int64)
    src = np.asarray(pattern.src_orb, dtype=np.int64)
    dst = np.asarray(pattern.dst_unit_orb, dtype=np.int64)
    eci = np.asarray(pattern.edge_cell_idx, dtype=np.int64)

    max_pat = int(np.max(np.abs(shifts))) if shifts.size > 0 else 0
    b_half = max(max_pat, 2 * max_link_shift) + 2
    b_stride = 2 * b_half + 1
    b3 = b_stride * b_stride * b_stride
    n_o = max(int(n_orb), 1)
    max_possible = (n_o * n_o) * b3
    if max_possible >= np.iinfo(np.int64).max // 4:
        raise OverflowError("Edge key encoding exceeds int64 capacity")

    t_edge = shifts[eci]
    edge_keys = (
        (src * n_o + dst) * b3
        + (t_edge[:, 0] + b_half) * (b_stride * b_stride)
        + (t_edge[:, 1] + b_half) * b_stride
        + (t_edge[:, 2] + b_half)
    )
    order = np.argsort(edge_keys)
    sorted_keys = np.ascontiguousarray(edge_keys[order], dtype=np.int64)
    sorted_edge_idx = np.ascontiguousarray(order, dtype=np.int32)
    bias = b_half * (b_stride * b_stride + b_stride + 1)
    return sorted_keys, sorted_edge_idx, b_stride, b_half, bias, shifts


def _iter_kb_shell_pairs(
    pattern: SparsePattern,
    n_orb: int,
    n_kb_shells: int,
    shell_dims: list[int],
    link_mu: np.ndarray,
    link_k_inst: np.ndarray,
    link_tk: np.ndarray,
    p_vals: np.ndarray,
    dp_vals: np.ndarray | None = None,
    link_m2_idx: np.ndarray | None = None,
):
    """Yield vectorized ``(k_i, mu_arr, tk_arr, r_idx, c_idx, e_idx, P_mat, dP_mat)`` per KB shell."""
    n_edges = int(pattern.n_edges)
    n_links = int(link_mu.shape[0])
    if n_edges == 0 or n_links == 0 or n_kb_shells == 0:
        return

    l_mu = np.asarray(link_mu, dtype=np.int64)
    l_ki = np.asarray(link_k_inst, dtype=np.int64)
    l_tk = np.asarray(link_tk, dtype=np.int64)
    max_link_t = int(np.max(np.abs(l_tk))) if l_tk.size > 0 else 0

    sorted_keys, sorted_edge_idx, b_stride, _b_half, bias, _ = _build_kb_pattern_index(
        pattern, n_orb, max_link_t
    )
    b2 = b_stride * b_stride
    b3 = b_stride * b2
    n_o = max(int(n_orb), 1)
    lt_half = max_link_t + 2
    lt_stride = 2 * lt_half + 1

    # Stable sort by k_inst preserves caller's m2 order inside each (k_inst, tk, mu) group
    ki_order = np.argsort(l_ki, kind="stable")
    ki_sorted = l_ki[ki_order]
    ki_bounds = np.searchsorted(ki_sorted, np.arange(n_kb_shells + 1))

    for k_i in range(n_kb_shells):
        s_lo = int(ki_bounds[k_i])
        s_hi = int(ki_bounds[k_i + 1])
        if s_hi <= s_lo:
            continue
        dim_m = int(shell_dims[k_i])
        sub_idx = ki_order[s_lo:s_hi]
        sub_mu = l_mu[sub_idx]
        sub_tk = l_tk[sub_idx]

        ktm_key = (
            ((sub_mu * lt_stride + (sub_tk[:, 0] + lt_half)) * lt_stride + (sub_tk[:, 1] + lt_half))
            * lt_stride
            + (sub_tk[:, 2] + lt_half)
        )
        if link_m2_idx is not None:
            grp_order = np.lexsort((link_m2_idx[sub_idx], ktm_key))
        else:
            grp_order = np.argsort(ktm_key, kind="stable")

        sorted_sub = sub_idx[grp_order]
        sorted_ktm = ktm_key[grp_order]
        _, start_pos, counts = np.unique(sorted_ktm, return_index=True, return_counts=True)
        valid_grp = counts == dim_m
        if not np.any(valid_grp):
            continue

        if np.all(valid_grp):
            idx_2d = sorted_sub.reshape(-1, dim_m)
        else:
            starts = start_pos[valid_grp]
            idx_2d = sorted_sub[starts[:, None] + np.arange(dim_m)[None, :]]

        grp_rep = start_pos[valid_grp]
        mu_arr = l_mu[sorted_sub[grp_rep]]
        tk_arr = l_tk[sorted_sub[grp_rep]]
        p_mat = np.ascontiguousarray(p_vals[idx_2d], dtype=np.float64)
        dp_mat = np.ascontiguousarray(dp_vals[idx_2d], dtype=np.float64) if dp_vals is not None else None

        tk_lin = tk_arr[:, 0] * b2 + tk_arr[:, 1] * b_stride + tk_arr[:, 2]
        row_keys = (mu_arr * n_o) * b3 + tk_lin
        col_keys = mu_arr * b3 - tk_lin + bias
        cand_keys = row_keys[:, None] + col_keys[None, :]

        pos = np.searchsorted(sorted_keys, cand_keys)
        pos_clamped = np.minimum(pos, len(sorted_keys) - 1)
        mask = sorted_keys[pos_clamped] == cand_keys
        r_idx, c_idx = np.nonzero(mask)
        if r_idx.size == 0:
            continue
        e_idx = sorted_edge_idx[pos_clamped[r_idx, c_idx]]
        yield k_i, mu_arr, tk_arr, r_idx, c_idx, e_idx, p_mat, dp_mat


def kb_three_center_native(
    pattern: SparsePattern,
    structure_cell: np.ndarray,
    positions: np.ndarray,
    orb_atom: np.ndarray,
    link_mu: np.ndarray,
    link_k_inst: np.ndarray,
    link_tk: np.ndarray,
    link_order_key: np.ndarray,
    p_vals: np.ndarray,
    dp_vals: np.ndarray | None,
    shell_atom: np.ndarray,
    shell_ek: np.ndarray,
    shell_dim: np.ndarray,
    dm_total: np.ndarray | None,
    assemble_v: bool,
) -> tuple[np.ndarray | None, float, np.ndarray | None, np.ndarray | None] | None:
    r"""Three-centre KB terms in C++ (``jsiesta_kb_three_center``): :math:`V^{KB}` on the edges and/or energy, forces, stress.

    Theory:
        :math:`V_{\mu\nu}(\mathbf{T}) = \sum_{k\mathbf{t}} E_k \sum_m P_{\mu,km}(\mathbf{t}) P_{\nu,km}(\mathbf{t} -
        \mathbf{T})`, :math:`E_{KB} = \sum D V`, and its derivative through the projector overlaps, as in
        ``_iter_kb_shell_pairs`` and ``_kb_forces_and_stress`` (scalar-relativistic shells only).

    Algorithm:
        Sorts the links by ``(k, mu, t, link_order_key)`` and calls the threaded C++ kernel, which pairs the link groups
        of each shell and finds each pair's pattern edge by binary search in the row.

    Parameters:
        pattern: Sparse pattern (CSR ``row_ptr``, ``dst_unit_orb``, cell shifts of the edges).
        structure_cell: Lattice vectors as columns, (3, 3), Bohr.
        positions: Atomic positions, (n_atoms, 3), Bohr.
        orb_atom: Atom of each orbital.
        link_mu: Orbital of each link.
        link_k_inst: KB shell instance of each link.
        link_tk: Projector cell shift of each link, (n_links, 3).
        link_order_key: Order of the m components inside a group (the m index, or the original position).
        p_vals: Projector overlaps, (n_links,).
        dp_vals: Overlap gradients, (n_links, 3), or ``None`` without forces.
        shell_atom: Atom of each shell instance.
        shell_ek: Effective KB energy of each shell instance (Ha).
        shell_dim: 2l+1 of each shell instance.
        dm_total: Spin-summed density matrix on the edges, or ``None`` for assembly only.
        assemble_v: Return :math:`V^{KB}` on the edges.

    Returns:
        ``(v_kb or None, e_kb, gradient (n_atoms, 3) or None, sum g x d (3, 3) or None)``, or ``None`` when the library
        has no such kernel (older build) so the caller keeps the NumPy path.

    Invariants:
        Same pairs and contributions as the NumPy reference; deterministic for a fixed thread count.

    SIESTA Mapping:
        ``nlefsm.f``.
    """
    from jax_siesta.core.ffi import load_ffi_library

    lib = load_ffi_library()
    fn = getattr(lib, "jsiesta_kb_three_center", None)
    if fn is None:
        return None
    if not getattr(fn, "_configured", False):
        i32p, i64p, f64p = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double))
        fn.argtypes = [
            ctypes.c_int64, i32p, i32p, i32p, f64p, f64p, ctypes.c_int32, i32p, f64p, i32p,
            ctypes.c_int32, i32p, ctypes.c_int32, f64p, f64p, ctypes.c_int64, i64p, i32p, i32p,
            f64p, f64p, f64p, f64p, f64p,
        ]
        fn.restype = ctypes.c_int32
        fn._configured = True
    mu = np.asarray(link_mu, dtype=np.int64)
    kk = np.asarray(link_k_inst, dtype=np.int64)
    tk = np.asarray(link_tk, dtype=np.int64).reshape(-1, 3)
    order = np.lexsort((np.asarray(link_order_key), tk[:, 2], tk[:, 1], tk[:, 0], mu, kk))
    c_mu = np.ascontiguousarray(mu[order], dtype=np.int32)
    c_k = np.ascontiguousarray(kk[order], dtype=np.int32)
    c_t = np.ascontiguousarray(tk[order], dtype=np.int32)
    c_p = np.ascontiguousarray(np.asarray(p_vals, dtype=np.float64)[order])
    c_dp = None if dp_vals is None else np.ascontiguousarray(np.asarray(dp_vals, dtype=np.float64).reshape(-1, 3)[order])
    n_edges = int(pattern.n_edges)
    row_ptr = np.ascontiguousarray(np.asarray(pattern.row_ptr), dtype=np.int64)
    dst = np.ascontiguousarray(np.asarray(pattern.dst_unit_orb), dtype=np.int32)
    e_t = np.ascontiguousarray(
        np.asarray(pattern.cell_shifts, dtype=np.int64)[np.asarray(pattern.edge_cell_idx, dtype=np.int64)],
        dtype=np.int32,
    )
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    cell = np.ascontiguousarray(structure_cell, dtype=np.float64)
    o_at = np.ascontiguousarray(orb_atom, dtype=np.int32)
    s_at = np.ascontiguousarray(shell_atom, dtype=np.int32)
    s_ek = np.ascontiguousarray(shell_ek, dtype=np.float64)
    s_dim = np.ascontiguousarray(shell_dim, dtype=np.int32)
    v = np.zeros(n_edges, dtype=np.float64) if assemble_v else None
    d = None if dm_total is None else np.ascontiguousarray(dm_total, dtype=np.float64)
    want_f = d is not None and c_dp is not None
    n_at = int(pos.shape[0])
    f = np.zeros((n_at, 3), dtype=np.float64) if want_f else None
    st = np.zeros((3, 3), dtype=np.float64) if want_f else None
    e = np.zeros(1, dtype=np.float64)

    def ptr(a: np.ndarray | None, ct: type) -> Any:
        return None if a is None else a.ctypes.data_as(ctypes.POINTER(ct))

    rc = fn(
        ctypes.c_int64(int(c_mu.shape[0])), ptr(c_mu, ctypes.c_int32), ptr(c_k, ctypes.c_int32),
        ptr(c_t, ctypes.c_int32), ptr(c_p, ctypes.c_double), ptr(c_dp, ctypes.c_double),
        ctypes.c_int32(int(s_at.shape[0])), ptr(s_at, ctypes.c_int32), ptr(s_ek, ctypes.c_double),
        ptr(s_dim, ctypes.c_int32), ctypes.c_int32(int(o_at.shape[0])), ptr(o_at, ctypes.c_int32),
        ctypes.c_int32(n_at), ptr(pos, ctypes.c_double), ptr(cell, ctypes.c_double), ctypes.c_int64(n_edges),
        ptr(row_ptr, ctypes.c_int64), ptr(dst, ctypes.c_int32), ptr(e_t, ctypes.c_int32), ptr(d, ctypes.c_double),
        ptr(v, ctypes.c_double), ptr(e, ctypes.c_double), ptr(f, ctypes.c_double), ptr(st, ctypes.c_double),
    )
    if rc != 0:
        raise RuntimeError(f"jsiesta_kb_three_center failed with code {rc}")
    return v, float(e[0]), f, st


def _shift_vectors(cell: np.ndarray, shifts: list[tuple[int, int, int]]) -> np.ndarray:
    """Cartesian lattice vectors ``A @ T`` for integer shifts, one matrix-vector product each (shape ``(n, 3)``)."""
    if not shifts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([cell @ np.asarray(t, dtype=np.float64) for t in shifts], axis=0)


def _atom_orbital_lists(orb_atom: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(orb_sorted, offsets, counts)``: orbital indices grouped by atom, in increasing orbital order."""
    orb_sorted = np.argsort(orb_atom, kind="stable").astype(np.int64)
    counts = np.bincount(orb_atom.astype(np.int64), minlength=n_atoms).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    return orb_sorted, offsets, counts


def _expand_groups(offsets: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For groups ``g`` with ``counts[g]`` members starting at ``offsets[g]``, return ``(group_of_item, item)``."""
    total = int(np.sum(counts))
    group = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    item = np.repeat(offsets, counts) + (np.arange(total, dtype=np.int64) - starts)
    return group, item


def _enumerate_kb_links(
    pos: np.ndarray,
    cell: np.ndarray,
    orb_atom: np.ndarray,
    orb_rc: np.ndarray,
    atom_kb_shells: list[tuple[int, int, float, int, int, float]],
    kb_shifts: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Enumerate overlapping orbital / KB-projector pairs :math:`(\phi_\mu, \chi_{k}(\mathbf{r}-\mathbf{R}_K-\mathbf{A}\mathbf{T}))`.

    A link exists when :math:`|\mathbf{R}_K + \mathbf{A}\mathbf{T} - \mathbf{R}_{I(\mu)}| \le r_c(\mu) + r_c(k) + 10^{-8}`
    and the shell's effective KB energy is non-zero. Each link is expanded over the projector's
    :math:`2l_k+1` magnetic components.

    Algorithm: per projector atom, an atom-level distance test over all images and atoms (bound
    :math:`\max_{\mu \in I} r_c(\mu) + \max_k r_c(k)`) selects candidate atoms; only their orbitals are tested.
    The result is sorted by ``(k_inst, shift, mu, m2)``, the order of the original scalar loop, so downstream
    link indices are unchanged.

    Returns:
        ``(k_inst, shift_idx, mu, disp, m2_idx)`` with one row per (link, m2); ``disp = R_K + A T - R_I(mu)``.
    """
    n_atoms = int(pos.shape[0])
    t_kb = _shift_vectors(cell, kb_shifts)
    orb_sorted, offsets, counts = _atom_orbital_lists(orb_atom, n_atoms)
    atom_rc_max = np.zeros(n_atoms, dtype=np.float64)
    if orb_rc.size:
        np.maximum.at(atom_rc_max, orb_atom.astype(np.int64), orb_rc)

    by_atom: dict[int, list[int]] = {}
    for k_inst_idx, (ka, _g, ek_eff, _l, _j2, _rc) in enumerate(atom_kb_shells):
        if abs(ek_eff) >= 1e-16:
            by_atom.setdefault(int(ka), []).append(k_inst_idx)

    k_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    mu_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    for ka, k_list in by_atom.items():
        rk = pos[ka][None, :] + t_kb  # (n_sh, 3), as R_K + A T in the scalar loop
        rc_k_max = max(float(atom_kb_shells[k][5]) for k in k_list)
        disp_at = rk[:, None, :] - pos[None, :, :]  # (n_sh, n_atoms, 3)
        d_at = np.sqrt(np.einsum("sja,sja->sj", disp_at, disp_at))
        cand_s, cand_a = np.nonzero(d_at <= atom_rc_max[None, :] + rc_k_max + 1e-8)
        if cand_s.size == 0:
            continue
        grp, item = _expand_groups(offsets[cand_a], counts[cand_a])
        mu = orb_sorted[item]
        sh = cand_s[grp]
        disp = disp_at[sh, cand_a[grp]]
        dist = np.sqrt(np.einsum("pa,pa->p", disp, disp))
        for k_inst_idx in k_list:
            rc_k = float(atom_kb_shells[k_inst_idx][5])
            keep = dist <= (orb_rc[mu] + rc_k) + 1e-8
            if not np.any(keep):
                continue
            k_parts.append(np.full(int(np.count_nonzero(keep)), k_inst_idx, dtype=np.int64))
            s_parts.append(sh[keep])
            mu_parts.append(mu[keep])
            d_parts.append(disp[keep])

    if not k_parts:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy(), np.zeros((0, 3), dtype=np.float64), z.copy()
    k_all = np.concatenate(k_parts)
    s_all = np.concatenate(s_parts)
    mu_all = np.concatenate(mu_parts)
    d_all = np.concatenate(d_parts, axis=0)
    order = np.lexsort((mu_all, s_all, k_all))
    k_all, s_all, mu_all, d_all = k_all[order], s_all[order], mu_all[order], d_all[order]

    dims = 2 * np.asarray([atom_kb_shells[int(k)][3] for k in range(len(atom_kb_shells))], dtype=np.int64) + 1
    rep = dims[k_all]
    m2_grp = np.repeat(np.arange(rep.shape[0], dtype=np.int64), rep)
    m2_idx = np.arange(int(rep.sum()), dtype=np.int64) - np.repeat(np.cumsum(rep) - rep, rep)
    return k_all[m2_grp], s_all[m2_grp], mu_all[m2_grp], d_all[m2_grp], m2_idx


def _enumerate_atom_pairs(
    pos: np.ndarray,
    cell: np.ndarray,
    species_of_atom: np.ndarray,
    rcut_species: np.ndarray,
    half: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Enumerate ordered atom pairs :math:`(i, j, \mathbf{T})` with :math:`10^{-8} < |\mathbf{R}_j + \mathbf{A}\mathbf{T} - \mathbf{R}_i| < r_c(i) + r_c(j)`.

    Images run over :math:`T_\alpha \in [-h_\alpha, h_\alpha]`; the self term ``i = j, T = 0`` is excluded. Rows are
    ordered by ``(i, j, T)`` with ``T`` in lexicographic order, as in the original scalar loop.

    Returns:
        ``(ij, disp, dist)``: ``(n_pairs, 2)`` int64 atom indices, ``(n_pairs, 3)`` displacements
        ``R_j + A T - R_i`` and ``(n_pairs,)`` distances.
    """
    shifts = [
        (s0, s1, s2)
        for s0 in range(-half[0], half[0] + 1)
        for s1 in range(-half[1], half[1] + 1)
        for s2 in range(-half[2], half[2] + 1)
    ]
    t_vecs = _shift_vectors(cell, shifts)
    zero_idx = shifts.index((0, 0, 0))
    n_atoms = int(pos.shape[0])
    rc_atom = rcut_species[species_of_atom]
    ij_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    r_parts: list[np.ndarray] = []
    shifted = pos[:, None, :] + t_vecs[None, :, :]  # (n_atoms, n_sh, 3): R_j + A T
    for ia in range(n_atoms):
        disp = shifted - pos[ia][None, None, :]
        dist = np.sqrt(np.einsum("jsa,jsa->js", disp, disp))
        mask = (dist > 1e-8) & (dist < (rc_atom[ia] + rc_atom)[:, None])
        mask[ia, zero_idx] = False
        ja, _s = np.nonzero(mask)
        if ja.size == 0:
            continue
        ij_parts.append(np.stack([np.full(ja.size, ia, dtype=np.int64), ja.astype(np.int64)], axis=1))
        d_parts.append(dist[mask])
        r_parts.append(disp[mask])
    if not ij_parts:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.concatenate(ij_parts, axis=0), np.concatenate(r_parts, axis=0), np.concatenate(d_parts)


def _eval_kb_and_na_pair(
    structure: Structure,
    basis: BasisData,
    pattern: SparsePattern,
    config: SCFConfig,
    n_pao_pts: int,
    pao_r_max_arr: np.ndarray,
    pao_rcut_arr: np.ndarray,
    pao_l_arr: np.ndarray,
    pao_vals_arr: np.ndarray,
    orb_g_shell: np.ndarray,
    out_s: np.ndarray,
    out_t: np.ndarray,
    out_ds: np.ndarray,
    out_dt: np.ndarray,
    eval_kb_links_fn: Any = None,
) -> TwoCenterMatrices:
    """Shared KB overlap/contraction and neutral-atom pair evaluator for Phase 1 & Phase 2.

    The link and pair searches are vectorised (``_enumerate_kb_links``, ``_enumerate_atom_pairs``); the
    accumulation of the pair forces keeps the per-pair update order of the scalar loop.
    """
    lib = load_ffi_library()
    _setup_c_signatures(lib)
    n_shells = int(pao_vals_arr.shape[0])
    n_edges = pattern.n_edges
    orb_m = np.array(basis.orb_m, dtype=np.int32)

    # 2. Pack KB projector shells and compute <phi_mu | chi_k> links
    kb_tables: list[np.ndarray] = []
    kb_r_max: list[float] = []
    kb_rcut_sh: list[float] = []
    kb_l_sh: list[int] = []
    kb_j2_sh: list[int] = []
    kb_e_sh: list[float] = []
    sp_kb_sh_offsets: list[int] = [0]
    n_kb_pts = int(basis.species[0].kb_r.shape[0])
    for sp in basis.species:
        r_max = float(sp.kb_r[-1])
        vals = np.asarray(sp.kb_projectors, dtype=np.float64)
        ek_arr = np.asarray(sp.kb_energies, dtype=np.float64)
        for k_sh in range(vals.shape[0]):
            kb_tables.append(vals[k_sh])
            kb_r_max.append(r_max)
            kb_rcut_sh.append(float(sp.kb_rcut[k_sh]))
            kb_l_sh.append(int(sp.kb_l[k_sh]))
            kb_j2_sh.append(int(sp.kb_j2[k_sh]))
            kb_e_sh.append(float(ek_arr[k_sh]))
        sp_kb_sh_offsets.append(len(kb_tables))

    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    orb_atom = np.asarray(basis.orb_atom, dtype=np.int32)
    orb_rc = np.asarray(basis.orb_rcut, dtype=np.float64)

    # Build unit-cell KB shell instances per atom: (atom_K, global_kb_shell, ekb_effective)
    atom_kb_shells: list[tuple[int, int, float, int, int, float]] = []
    for ia, sp_idx in enumerate(structure.species_indices):
        for loc_k in range(sp_kb_sh_offsets[sp_idx], sp_kb_sh_offsets[sp_idx + 1]):
            l_k = kb_l_sh[loc_k]
            j2_k = kb_j2_sh[loc_k]
            ek_eff = kb_e_sh[loc_k]
            if config.nspin != 8 and j2_k > 0 and l_k > 0:
                ek_eff *= (j2_k + 1.0) / (2.0 * (2 * l_k + 1.0))
            atom_kb_shells.append((ia, loc_k, ek_eff, l_k, j2_k, kb_rcut_sh[loc_k]))

    # Search extended shift box for projector overlaps
    nsc_kb = tuple(max(1, abs(int(s))) for s in pattern.nsc)
    kb_shifts: list[tuple[int, int, int]] = []
    for s0 in range(-nsc_kb[0], nsc_kb[0] + 1):
        for s1 in range(-nsc_kb[1], nsc_kb[1] + 1):
            for s2 in range(-nsc_kb[2], nsc_kb[2] + 1):
                if all(structure.pbc[ax] or (s0, s1, s2)[ax] == 0 for ax in range(3)):
                    kb_shifts.append((s0, s1, s2))

    # Enumerate interacting (orb mu, atom_kb_shell_instance k_inst, shift T_k) -> expanded over m2
    links = _enumerate_kb_links(pos, cell, orb_atom, orb_rc, atom_kb_shells, kb_shifts)
    l_k_inst_arr, l_shift_arr, l_mu_arr, l_vec_base, l_m2_idx_arr = links
    link_mu: list[int] = l_mu_arr.tolist()
    link_k_inst: list[int] = l_k_inst_arr.tolist()
    link_m2_idx: list[int] = l_m2_idx_arr.tolist()
    kb_shift_arr = np.asarray(kb_shifts, dtype=np.int64).reshape(-1, 3)
    link_tk_arr = kb_shift_arr[l_shift_arr]
    # The symmetry callback indexes links by Python tuples; build them only when it is used.
    link_tk: list[tuple[int, int, int]] = (
        [(t[0], t[1], t[2]) for t in link_tk_arr.tolist()] if eval_kb_links_fn is not None else []
    )
    kb_l_inst = np.asarray([item[3] for item in atom_kb_shells], dtype=np.int32)
    kb_g_inst = np.asarray([item[1] for item in atom_kb_shells], dtype=np.int32)

    n_links = len(link_mu)
    if n_links > 0:
        l_pao_sh = np.ascontiguousarray(orb_g_shell[l_mu_arr], dtype=np.int32)
        l_m1 = np.ascontiguousarray(orb_m[l_mu_arr], dtype=np.int32)
        l_kb_sh = np.ascontiguousarray(kb_g_inst[l_k_inst_arr], dtype=np.int32)
        l_m2 = np.ascontiguousarray(l_m2_idx_arr - kb_l_inst[l_k_inst_arr], dtype=np.int32)
        l_vec = np.ascontiguousarray(l_vec_base, dtype=np.float64)
        kb_rmax_arr = np.asarray(kb_r_max, dtype=np.float64)
        kb_rcut_arr = np.asarray(kb_rcut_sh, dtype=np.float64)
        kb_l_arr = np.asarray(kb_l_sh, dtype=np.int32)
        kb_vals_arr = np.ascontiguousarray(np.stack(kb_tables, axis=0), dtype=np.float64)

        if eval_kb_links_fn is not None:
            out_p, out_dp = eval_kb_links_fn(
                n_pao_pts,
                pao_r_max_arr,
                pao_rcut_arr,
                pao_l_arr,
                pao_vals_arr,
                n_kb_pts,
                kb_rmax_arr,
                kb_rcut_arr,
                kb_l_arr,
                kb_vals_arr,
                atom_kb_shells,
                link_mu,
                link_k_inst,
                link_tk,
                link_m2_idx,
                l_pao_sh,
                l_m1,
                l_kb_sh,
                l_m2,
                l_vec,
            )
        else:
            out_p = np.zeros(n_links, dtype=np.float64)
            out_dp = np.zeros((n_links, 3), dtype=np.float64)
            rc = lib.jsiesta_eval_kb_overlaps(
                ctypes.c_int32(n_shells),
                ctypes.c_int32(n_pao_pts),
                pao_r_max_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pao_rcut_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pao_l_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                pao_vals_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int32(len(kb_tables)),
                ctypes.c_int32(n_kb_pts),
                kb_rmax_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                kb_rcut_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                kb_l_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                kb_vals_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int32(n_links),
                l_pao_sh.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                l_m1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                l_kb_sh.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                l_m2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                l_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_dp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if rc != 0:
                raise RuntimeError(f"jsiesta_eval_kb_overlaps failed with code {rc}")
    else:
        out_p = np.zeros(0, dtype=np.float64)
        out_dp = np.zeros((0, 3), dtype=np.float64)

    # Assemble sparse V_kb and (if nspin==8) V_kb_so on pattern edges (vectorized)
    v_kb = np.zeros(n_edges, dtype=np.float64)
    v_kb_so = np.zeros((4, n_edges), dtype=np.complex128) if config.nspin == 8 else None
    soc_cache: dict[tuple[int, int], np.ndarray] = {}

    native_done = False
    if n_links > 0 and n_edges > 0 and config.nspin != 8:
        # C++ assembly (same pairs and sums as the NumPy loop below, threaded; D-041).
        nat = kb_three_center_native(
            pattern,
            np.asarray(structure.cell, dtype=np.float64),
            np.asarray(structure.positions, dtype=np.float64),
            np.asarray(basis.orb_atom),
            l_mu_arr,
            l_k_inst_arr,
            link_tk_arr,
            l_m2_idx_arr,
            out_p,
            None,
            np.asarray([item[0] for item in atom_kb_shells], dtype=np.int32),
            np.asarray([item[2] for item in atom_kb_shells], dtype=np.float64),
            np.asarray([2 * item[3] + 1 for item in atom_kb_shells], dtype=np.int32),
            None,
            True,
        )
        if nat is not None and nat[0] is not None:
            v_kb = nat[0]
            native_done = True
    if n_links > 0 and n_edges > 0 and not native_done:
        shell_dims = [2 * item[3] + 1 for item in atom_kb_shells]
        for k_inst_idx, _mu_arr, _tk_arr, r_idx, c_idx, e_idx, p_mat, _dp_mat in _iter_kb_shell_pairs(
            pattern=pattern,
            n_orb=basis.n_orb,
            n_kb_shells=len(atom_kb_shells),
            shell_dims=shell_dims,
            link_mu=l_mu_arr,
            link_k_inst=l_k_inst_arr,
            link_tk=link_tk_arr,
            p_vals=out_p,
            dp_vals=None,
            link_m2_idx=l_m2_idx_arr,
        ):
            _, _, ek_eff, l_k, j2_k, _ = atom_kb_shells[k_inst_idx]
            if config.nspin == 8 and j2_k > 0 and l_k > 0:
                spin_mat = soc_cache.get((l_k, j2_k))
                if spin_mat is None:
                    spin_mat = _kb_soc_shell_spin_matrix(l_k, j2_k)
                    soc_cache[(l_k, j2_k)] = spin_mat
                p_spin = np.einsum("rm,mnc->rnc", p_mat, spin_mat)
                contrib = ek_eff * np.einsum("rnc,rn->rc", p_spin[r_idx], p_mat[c_idx])
                assert v_kb_so is not None
                for ch in range(4):
                    np.add.at(v_kb_so[ch], e_idx, contrib[:, ch])
                np.add.at(v_kb, e_idx, 0.5 * np.real(contrib[:, 0] + contrib[:, 1]))
            else:
                vals = ek_eff * np.sum(p_mat[r_idx] * p_mat[c_idx], axis=1)
                np.add.at(v_kb, e_idx, vals)
                if v_kb_so is not None:
                    np.add.at(v_kb_so[0], e_idx, vals)
                    np.add.at(v_kb_so[1], e_idx, vals)

    # 3. Short-range Neutral-Atom electrostatic pair energy, forces, and stress
    n_sp = len(basis.species)
    n_rho_pts = int(basis.species[0].r_grid.shape[0])
    sp_rmax = np.array([float(sp.r_grid[-1]) for sp in basis.species], dtype=np.float64)
    sp_rcut_na = np.array([float(sp.rcut_na) for sp in basis.species], dtype=np.float64)
    sp_z = np.array([float(sp.valence_charge) for sp in basis.species], dtype=np.float64)
    sp_rho_na = np.ascontiguousarray(
        np.stack([np.asarray(sp.rho_na, dtype=np.float64) for sp in basis.species], axis=0),
        dtype=np.float64,
    )

    max_na_rc = float(np.max(sp_rcut_na)) if n_sp else 0.0
    inv_cell = np.linalg.inv(cell)
    na_half = [
        int(np.ceil(2.0 * max_na_rc * np.linalg.norm(inv_cell[ax]))) if structure.pbc[ax] else 0
        for ax in range(3)
    ]
    pair_ij_arr, pair_vec_arr, pair_dist_arr = _enumerate_atom_pairs(
        pos, cell, np.asarray(structure.species_indices, dtype=np.int64), sp_rcut_na, na_half
    )

    e_na_pair = 0.0
    f_na_atoms = np.zeros((structure.n_atoms, 3), dtype=np.float64)
    stress_na = np.zeros((3, 3), dtype=np.float64)
    vol = float(abs(np.linalg.det(cell)))

    if pair_dist_arr.size > 0:
        n_p = int(pair_dist_arr.shape[0])
        sp_idx_arr = np.asarray(structure.species_indices, dtype=np.int32)
        p_sp1 = np.ascontiguousarray(sp_idx_arr[pair_ij_arr[:, 0]], dtype=np.int32)
        p_sp2 = np.ascontiguousarray(sp_idx_arr[pair_ij_arr[:, 1]], dtype=np.int32)
        p_d = np.ascontiguousarray(pair_dist_arr, dtype=np.float64)
        out_e = np.zeros(n_p, dtype=np.float64)
        out_de = np.zeros(n_p, dtype=np.float64)
        rc = lib.jsiesta_eval_na_pair(
            ctypes.c_int32(n_sp),
            ctypes.c_int32(n_rho_pts),
            sp_rmax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sp_rcut_na.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sp_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sp_rho_na.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(n_p),
            p_sp1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            p_sp2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            p_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_de.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"jsiesta_eval_na_pair failed with code {rc}")
        e_na_pair = 0.5 * float(np.sum(out_e))
        # d(0.5 * U(R))/dR_i = -0.5 * (dU/dR) * r_ij / d => F_i = -dE/dR_i = +0.5 * (dU/dR) * r_ij / d,
        # with r_ij = R_j + T - R_i. Updates are applied in the pair order (i then j), as a sequential loop would.
        grad_vec = (0.5 * out_de / p_d)[:, None] * pair_vec_arr
        upd_idx = pair_ij_arr.reshape(-1)
        upd_val = np.stack([grad_vec, -grad_vec], axis=1).reshape(-1, 3)
        np.add.at(f_na_atoms, upd_idx, upd_val)
        # Strain derivative: d(E)/d(eps_{ab}) = 0.5 * (dU/dR) * r_ij[a] * r_ij[b] / d
        stress_na += np.einsum("p,pa,pb->ab", 0.5 * out_de / (p_d * vol), pair_vec_arr, pair_vec_arr)

    # One-centre constants: make E_NA (and the total energy) absolute rather than defined up to a
    # per-species constant (review finding H1; logs/DECISIONS.md D-016).
    self_terms = [neutral_atom_self_energy(sp) for sp in basis.species]
    e_na_pair += float(sum(self_terms[int(sp_i)] for sp_i in structure.species_indices))

    # Exact atomic part of the neutral-atom grid energy, sum_{I,J} int V_NA,I n_J; the mesh integrates only
    # V_NA * (rho - rho_atm), which removes the egg-box error of this term (D-017). The one-centre I = J terms
    # cancel the -int V_NA n_at part of the self-energy constant exactly, so both are omitted.
    w_tab = {
        (a, b): neutral_atom_density_two_center(basis.species[a], basis.species[b])
        for a in range(n_sp)
        for b in range(n_sp)
    }
    w_rmax = max(rc for _, rc in w_tab.values()) if w_tab else 0.0
    w_half = [
        int(np.ceil(w_rmax * np.linalg.norm(inv_cell[ax]))) + 1 if structure.pbc[ax] else 0 for ax in range(3)
    ]
    shifts = np.array(
        [
            (s0, s1, s2)
            for s0 in range(-w_half[0], w_half[0] + 1)
            for s1 in range(-w_half[1], w_half[1] + 1)
            for s2 in range(-w_half[2], w_half[2] + 1)
        ],
        dtype=np.float64,
    )
    t_vecs = shifts @ cell.T
    sp_of_atom = np.asarray(structure.species_indices, dtype=np.int64)
    for ia in range(structure.n_atoms):
        sp_i = int(sp_of_atom[ia])
        # disp[ja, s] = R_j + T_s - R_i for every atom j and image s
        disp_all = (pos[:, None, :] + t_vecs[None, :, :]) - pos[ia][None, None, :]
        dist_all = np.sqrt(np.einsum("jsa,jsa->js", disp_all, disp_all))
        for sp_j in range(n_sp):
            spl, rc_w = w_tab[(sp_i, sp_j)]
            sel = (dist_all > 1e-8) & (dist_all < rc_w) & (sp_of_atom == sp_j)[:, None]
            if not np.any(sel):
                continue
            ja_sel = np.nonzero(sel)[0]
            d_sel, r_sel = dist_all[sel], disp_all[sel]
            e_na_pair += float(np.sum(spl(d_sel)))
            dw = spl(d_sel, 1)
            grad = (dw / d_sel)[:, None] * r_sel  # dE/dR_j for each image
            np.add.at(f_na_atoms, ja_sel, -grad)
            f_na_atoms[ia] += grad.sum(axis=0)
            stress_na += np.einsum("p,pa,pb->ab", dw / (d_sel * vol), r_sel, r_sel)

    return TwoCenterMatrices(
        overlap=jnp.asarray(out_s, dtype=jnp.float64),
        kinetic=jnp.asarray(out_t, dtype=jnp.float64),
        v_na=jnp.zeros(n_edges, dtype=jnp.float64),  # V_NA applied on 3D real-space grid
        v_kb=jnp.asarray(v_kb, dtype=jnp.float64),
        v_kb_so=jnp.asarray(v_kb_so, dtype=jnp.complex128) if v_kb_so is not None else None,
        d_overlap_dr=jnp.asarray(out_ds, dtype=jnp.float64),
        d_kinetic_dr=jnp.asarray(out_dt, dtype=jnp.float64),
        d_v_na_dr=jnp.zeros((n_edges, 3), dtype=jnp.float64),
        d_v_kb_dr=jnp.zeros((n_edges, 3), dtype=jnp.float64),
        e_na_pair=jnp.asarray(e_na_pair, dtype=jnp.float64),
        f_na_atoms=jnp.asarray(f_na_atoms, dtype=jnp.float64),
        stress_na=jnp.asarray(stress_na, dtype=jnp.float64),
        kb_link_p=jnp.asarray(out_p, dtype=jnp.float64) if n_links > 0 else jnp.zeros((0,), dtype=jnp.float64),
        kb_link_dp=jnp.asarray(out_dp, dtype=jnp.float64) if n_links > 0 else jnp.zeros((0, 3), dtype=jnp.float64),
        kb_link_orb=tuple(link_mu),
        kb_link_kb=tuple(link_k_inst),
        kb_link_cell_shift=jnp.asarray(link_tk_arr, dtype=jnp.int32) if n_links > 0 else jnp.zeros((0, 3), dtype=jnp.int32),
    )
