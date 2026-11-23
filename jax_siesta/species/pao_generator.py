r"""1D radial Confined Numerical Atomic Orbital (NAO / PAO) and pseudopotential generator.

Theory:
    Constructs first-principles Sankey-Niklewski / Junquera-Artacho Pseudo-Atomic Orbitals
    (PAOs) :math:`\phi_{nl\zeta}(r)` directly from a ``.psml`` pseudopotential without
    invoking external ``siesta`` executables or reading pre-generated ``.ion`` files:

    1. **1D Radial Confined Kohn-Sham Solver:** For a species with ionic net charge
       :math:`q` (adjusting valence occupations :math:`f_{nl}` from the highest occupied shell),
       solves the self-consistent spherical atomic Kohn-Sham equations on a logarithmic radial
       grid :math:`r_i = b(e^{a i} - 1)` using exact variational Sturm-Liouville
       discretization of

       .. math::
           \left[ -\frac{1}{2}\frac{d^2}{dr^2} + \frac{l(l+1)}{2r^2} + V_l^{\text{ps}}(r)
           + V_H[\rho](r) + v_{\text{xc}}[\rho + \rho^{\text{core}}](r)
           + V_{\text{conf}}(r; r_i, r_c, V_0) \right] u_{nl}(r) = \varepsilon_{nl} u_{nl}(r),

       with :math:`u_{nl}(0) = u_{nl}(r_c) = 0` and Junquera soft-confinement potential

       .. math::
           V_{\text{conf}}(r; r_i, r_c, V_0) =
           \begin{cases}
               0, & r \le r_i, \\
               V_0 \dfrac{\exp\!\left(-\dfrac{r_c - r_i}{r - r_i}\right)}{r_c - r}, & r_i < r < r_c.
           \end{cases}

    2. **Multiple-\zeta Parabolic Splitting (Explicit :math:`r_c^{(z)}` or ``split_norm``):**
       Given outer orbital :math:`R_{1\zeta}(r) = u_{nl}(r)/r` and explicit inner cutoff
       :math:`r_m = r_c^{(z)} < r_c^{(z-1)}`, subtracts a regular polynomial
       :math:`(c_1 r^2 + c_2) r^l` matched in value and slope at :math:`r_m` (`fit_parabola`
       in SIESTA) so :math:`R_{z\zeta}(r) = 0` smoothly for :math:`r \ge r_m`.

    3. **Polarization Shells:** Supports both SIESTA perturbative electric-field linear
       response :math:`(H_{l+1} + V_{\text{conf}} - \varepsilon_{nl}) u_{\text{pol}} = -r u_{nl}`
       and explicit soft-confined :math:`(n, l+1)` shells.

    4. **Semilocal PSML Support:** When a PSML contains ``<semilocal-potentials>`` without
       pre-split ``<local-potential>`` / ``<nonlocal-projectors>``, builds smooth
       :math:`V_{\text{loc}}(r)` and normalized Kleinman-Bylander projectors matching SIESTA.

Algorithm:
    Discretizes the 1D radial Sturm-Liouville kinetic operator as a symmetric positive-definite
    tridiagonal matrix :math:`T = B^T B` in Euclidean measure :math:`\sum_i \psi_i^2 = \int u^2 dr`,
    supporting both semilocal :math:`V_l(r)` and rank-:math:`k` separable KB projectors cleanly.
    Discrete radial XC potentials are obtained by exact autodiff of spherical quadrature
    :math:`E_{\text{xc}} = 4\pi \sum_i w_i r_i^2 e_{\text{xc}}(\rho_i, |\rho'_i|^2)`.

Invariants:
    - Every radial PAO vanishes identically at and beyond its explicit ``rcut`` and satisfies
      :math:`\int_0^{r_c} r^2 R_{nl\zeta}(r)^2 \, dr = 1`.
    - Neutral-atom density :math:`\rho^{\text{NA}}(r)` and screening :math:`V^{\text{NA}}(r)` are
      recomputed consistently from the generated :math:`1\zeta` occupied orbitals.
    - Explicit inner-zeta cutoffs satisfy strict ordering :math:`r_c^{(1)} > r_c^{(2)} > \dots > r_i`.

SIESTA Mapping:
    Replaces SIESTA ``Src/atom.F`` (``SPLIT``, ``build_vsoft``, ``fit_parabola``, ``POLgen``)
    and ``Src/psoplib`` (``compute_vlocal_chlocal``, ``kbgen``) inside pure Python/JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.optimize import brentq

from jax_siesta.core.types import HARTREE_TO_EV, RY_TO_HARTREE, SpeciesData

_L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_BOHR_PER_UNIT = {
    "bohr": 1.0,
    "a0": 1.0,
    "angstrom": 1.0 / 0.529177210903,
    "ang": 1.0 / 0.529177210903,
}
_HA_PER_UNIT = {
    "hartree": 1.0,
    "ha": 1.0,
    "rydberg": 0.5,
    "ry": 0.5,
    "ev": 1.0 / HARTREE_TO_EV,
}


@dataclass(frozen=True)
class ShellBasisSpec:
    r"""Radial PAO configuration for a single :math:`(n, l)` angular-momentum shell.

    Theory:
        Specifies explicit per-:math:`\zeta` cutoff radii :math:`r_c^{(1)} > r_c^{(2)} > \dots`
        and Junquera soft-confinement parameters :math:`(V_0, r_i)` for either a valence
        shell or a polarization shell.

    Algorithm:
        When ``rc_list`` is explicitly supplied, uses ``rc_list[0]`` for :math:`1\zeta`
        confinement and ``rc_list[z-1]`` directly as the SIESTA ``fit_parabola`` split
        cutoff for :math:`z \ge 2`. When omitted, determines ``rc_z1`` from ``energy_shift_ha``
        and inner zetas from ``split_norm`` (for reproducing default SIESTA presets).

    Invariants:
        - Explicit ``rc_list`` must be strictly decreasing when provided.
        - Soft confinement inner radius satisfies ``0 <= ri < rc_list[-1]``.

    SIESTA Mapping:
        Mirrors one shell block in SIESTA ``%block PAO.Basis`` (``n=n l nzeta E V0 ri``).
    """

    n: int
    l: int  # noqa: E741
    n_zeta: int = 2
    rc_list: tuple[float, ...] | None = None
    v0_ha: float = 0.0
    ri_bohr: float = 0.0
    is_polarization: bool = False
    polarization_mode: Literal["perturbative", "eigenstate"] = "perturbative"
    parent_l: int | None = None
    energy_shift_ha: float | None = None
    split_norm: float | None = None


@dataclass(frozen=True)
class SpeciesBasisSpec:
    r"""Complete PAO generation specification for one chemical species.

    Theory:
        Bundles per-shell :class:`ShellBasisSpec` entries with the atomic net ionic charge
        :math:`q` (`deltaQ` adjusting reference valence occupations) and reference XC
        functional used during radial Kohn-Sham orbital generation.

    Algorithm:
        Passed directly to :func:`generate_species_from_psml` to construct an immutable
        ``SpeciesData`` PyTree in memory without intermediate files.

    Invariants:
        - ``len(shells) >= 1``.
        - ``valence_charge - ionic_charge > 0``.

    SIESTA Mapping:
        Mirrors one species block inside ``%block PAO.Basis`` plus ``XC.functional``.
    """

    symbol: str
    shells: tuple[ShellBasisSpec, ...]
    ionic_charge: float = 0.0
    xc: str = "PBE"
    energy_shift_ha: float = 0.01 * RY_TO_HARTREE
    split_norm: float = 0.15
    #: Logarithmic grid step ``a`` (r_i = b (e^{a i} - 1)). Cut-off radii snap to grid points, so the PAO
    #: energy is piecewise constant in ``rc`` on plateaus of width ``a * rc``; 0.022 (SIESTA-like default)
    #: gives 0.12-0.15 Bohr plateaus, the basis optimiser uses 0.004 (review H4).
    log_grid_step: float = 0.022


# ---------------------------------------------------------------------------
# 1D Radial XC (unpolarized spin-averaged LDA CA/PZ81 + PW92 + GGA PBE96)
# ---------------------------------------------------------------------------


def _exc_pz81(rho: jax.Array) -> jax.Array:
    """Spin-unpolarized Perdew-Zunger 81 LDA correlation + Slater exchange energy per particle."""
    rho_c = jnp.maximum(rho, 1e-24)
    rs = jnp.cbrt(3.0 / (4.0 * jnp.pi * rho_c))
    ex = -0.75 * jnp.cbrt(3.0 * rho_c / jnp.pi)

    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    a_c = 0.0311
    b_c = -0.0480
    c_c = 0.0020
    d_c = -0.0116
    ec_high = a_c * jnp.log(rs) + b_c + c_c * rs * jnp.log(rs) + d_c * rs
    ec_low = gamma / (1.0 + beta1 * jnp.sqrt(rs + 1e-30) + beta2 * rs)
    ec = jnp.where(rs < 1.0, ec_high, ec_low)
    return jnp.where(rho > 1e-22, ex + ec, 0.0)


def _exc_pw92(rho: jax.Array) -> jax.Array:
    """Spin-unpolarized PW92 LDA correlation + Slater exchange energy per particle."""
    rho_c = jnp.maximum(rho, 1e-24)
    rs = jnp.cbrt(3.0 / (4.0 * jnp.pi * rho_c))
    ex = -0.75 * jnp.cbrt(3.0 * rho_c / jnp.pi)

    a, a1 = 0.0310907, 0.21370
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    s_rs = jnp.sqrt(rs + 1e-30)
    q0 = -2.0 * a * (1.0 + a1 * rs)
    q1 = 2.0 * a * (b1 * s_rs + b2 * rs + b3 * rs * s_rs + b4 * rs * rs)
    ec = q0 * jnp.log1p(1.0 / jnp.maximum(q1, 1e-30))
    return jnp.where(rho > 1e-22, ex + ec, 0.0)


def _exc_pbe96(rho: jax.Array, sigma: jax.Array) -> jax.Array:
    """Spin-unpolarized PBE96 GGA exchange-correlation energy per particle."""
    rho_c = jnp.maximum(rho, 1e-24)
    kf = jnp.cbrt(3.0 * (jnp.pi**2) * rho_c)
    ex_unif = -0.75 * kf / jnp.pi
    g_norm = jnp.sqrt(sigma + 1e-50)
    s = g_norm / (2.0 * kf * rho_c)
    kappa = 0.804
    mu = 0.2195149727645171
    fx = 1.0 + kappa - kappa / (1.0 + (mu / kappa) * (s * s))
    ex = ex_unif * fx

    rs = jnp.cbrt(3.0 / (4.0 * jnp.pi * rho_c))
    a, a1 = 0.0310907, 0.21370
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    s_rs = jnp.sqrt(rs + 1e-30)
    q0 = -2.0 * a * (1.0 + a1 * rs)
    q1 = 2.0 * a * (b1 * s_rs + b2 * rs + b3 * rs * s_rs + b4 * rs * rs)
    ec_unif = q0 * jnp.log1p(1.0 / jnp.maximum(q1, 1e-30))

    ks = jnp.sqrt(4.0 * kf / jnp.pi + 1e-30)
    t = g_norm / (2.0 * ks * rho_c)
    gamma_pbe = (1.0 - jnp.log(2.0)) / (jnp.pi**2)
    beta_pbe = 0.06672455060314922
    arg = jnp.clip(-ec_unif / gamma_pbe, 1e-12, 50.0)
    a_t = (beta_pbe / gamma_pbe) / jnp.expm1(arg)
    t2 = t * t
    at2 = a_t * t2
    num = 1.0 + at2
    den = 1.0 + at2 + at2 * at2
    h_pbe = gamma_pbe * jnp.log1p((beta_pbe / gamma_pbe) * t2 * (num / jnp.maximum(den, 1e-30)))
    return jnp.where(rho > 1e-22, ex + ec_unif + h_pbe, 0.0)


@jax.jit
def _radial_xc_jit(
    r: jax.Array,
    drdi: jax.Array,
    rho_tot: jax.Array,
    xc_code: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Jitted discrete spherical XC energy and raw gradient ``dE_xc/d_rho``."""

    def _energy_fn(rho_arr: jax.Array) -> jax.Array:
        rho_pos = jnp.maximum(rho_arr, 0.0)
        drho = jnp.zeros_like(rho_pos)
        dr = r[2:] - r[:-2]
        drho = drho.at[1:-1].set((rho_pos[2:] - rho_pos[:-2]) / jnp.maximum(dr, 1e-18))
        sigma = drho * drho
        e_pz = _exc_pz81(rho_pos)
        e_pw = _exc_pw92(rho_pos)
        e_pbe = _exc_pbe96(rho_pos, sigma)
        exc = jnp.where(xc_code == 0, e_pz, jnp.where(xc_code == 1, e_pw, e_pbe))
        vol_w = 4.0 * np.pi * (r**2) * drdi
        vol_w = vol_w.at[0].set(0.0)
        vol_w = vol_w.at[-1].multiply(0.5)
        return jnp.sum(vol_w * rho_pos * exc)

    return jax.value_and_grad(_energy_fn)(jnp.maximum(rho_tot, 0.0))


def _radial_xc_potential(
    r: np.ndarray,
    drdi: np.ndarray,
    rho_tot: np.ndarray,
    xc: str,
) -> tuple[float, np.ndarray]:
    """Compute exact discrete spherical XC energy and variational potential :math:`v_{xc}(r)`."""
    xc_key = xc.strip().upper().replace("-", "_")
    if xc_key in ("LDA", "LDA_PZ", "CA", "PZ"):
        xc_code = 0
    elif xc_key in ("LDA_PW92", "PW92"):
        xc_code = 1
    else:
        xc_code = 2
    e_xc, d_e = _radial_xc_jit(
        jnp.asarray(r, dtype=jnp.float64),
        jnp.asarray(drdi, dtype=jnp.float64),
        jnp.asarray(rho_tot, dtype=jnp.float64),
        jnp.asarray(xc_code, dtype=jnp.int32),
    )
    vol_w = 4.0 * np.pi * (r**2) * drdi
    vol_w[0] = 1.0
    vol_w[-1] *= 0.5
    v_xc = np.asarray(d_e, dtype=np.float64) / np.maximum(vol_w, 1e-30)
    v_xc[0] = v_xc[1]
    return float(e_xc), v_xc


# ---------------------------------------------------------------------------
# Radial Grid, Hartree, Soft Confinement, Split-Parabola, and Eigen-Solver
# ---------------------------------------------------------------------------


def _make_siesta_log_grid(
    r_max: float = 22.0, n_pts: int = 441, a: float = 0.022
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Construct SIESTA exponential grid :math:`r_i = b (e^{a i} - 1)`."""
    idx = np.arange(n_pts, dtype=np.float64)
    b = r_max / (np.exp(a * (n_pts - 1)) - 1.0)
    r = b * (np.exp(a * idx) - 1.0)
    drdi = a * (r + b)
    return r, drdi, a, b


def _radial_hartree_log(r: np.ndarray, drdi: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r"""1D spherical Hartree potential :math:`V_H(r)` integrated with grid weights ``drdi``."""
    f_in = 4.0 * np.pi * (r**2) * rho * drdi
    f_out = 4.0 * np.pi * r * rho * drdi

    q_in = np.zeros_like(r)
    q_in[1:] = np.cumsum(0.5 * (f_in[:-1] + f_in[1:]))

    tail_out = np.zeros_like(r)
    seg_out = 0.5 * (f_out[:-1] + f_out[1:])
    tail_out[:-1] = np.cumsum(seg_out[::-1])[::-1]

    vh = np.empty_like(r)
    vh[1:] = q_in[1:] / r[1:] + tail_out[1:]
    vh[0] = tail_out[0]
    return vh


def _build_vsoft(r: np.ndarray, ri: float, rc: float, v0_ha: float) -> np.ndarray:
    """SIESTA ``build_vsoft`` soft-confinement potential (Hartree)."""
    vsoft = np.zeros_like(r)
    if v0_ha == 0.0:
        return vsoft
    mask = (r > ri) & (r < rc - 1e-8)
    exponent = -(rc - ri) / np.maximum(r[mask] - ri, 1e-12)
    safe = np.abs(exponent) <= 500.0
    out = np.zeros_like(r[mask])
    out[safe] = v0_ha / (rc - r[mask][safe]) * np.exp(exponent[safe])
    vsoft[mask] = out
    return vsoft


def _radial_tridiag(
    r: np.ndarray,
    drdi: np.ndarray,
    a: float,
    l_qn: int,
    v_eff: np.ndarray,
    n_active: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build vectorized symmetric tridiagonal ``(diag, off_diag)`` Sturm-Liouville radial H."""
    _ = a
    w = drdi[: n_active + 1]
    r_act = r[1:n_active]
    inv_sqrt_w = 1.0 / np.sqrt(w)
    dr = np.diff(r[: n_active + 1])
    c = 1.0 / (2.0 * np.maximum(dr, 1e-18))

    diag = 0.5 * l_qn * (l_qn + 1) / (r_act**2) + v_eff[1:n_active]
    diag += c[:-1] * (inv_sqrt_w[1:n_active] ** 2)
    diag += c[1:] * (inv_sqrt_w[1:n_active] ** 2)
    off = -c[1:-1] * inv_sqrt_w[1 : n_active - 1] * inv_sqrt_w[2:n_active]
    return diag, off


def _solve_radial_state(
    r: np.ndarray,
    drdi: np.ndarray,
    a: float,
    l_qn: int,
    n_radial_nodes: int,
    v_eff: np.ndarray,
    rc_idx: int,
    kb_projs: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[float, np.ndarray]:
    """Solve for the ``n_radial_nodes``-th radial eigenstate (0=nodeless) with ``R(r>=r[rc_idx])=0``."""
    diag, off = _radial_tridiag(r, drdi, a, l_qn, v_eff, rc_idx)
    m = len(diag)
    idx = min(max(n_radial_nodes, 0), m - 1)
    has_kb = bool(kb_projs and any(ekb != 0.0 for _, ekb in kb_projs))
    if not has_kb:
        evals, evecs = eigh_tridiagonal(diag, off, select="i", select_range=(idx, idx))
        eps = float(evals[0])
        psi = evecs[:, 0]
    else:
        h_mat = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
        r_act = r[1:rc_idx]
        sqrt_w_act = np.sqrt(drdi[1:rc_idx])
        for chi_on_r, ekb in kb_projs or []:
            if ekb == 0.0:
                continue
            vec = r_act * chi_on_r[1:rc_idx] * sqrt_w_act
            h_mat += ekb * np.outer(vec, vec)
        evals, evecs = np.linalg.eigh(h_mat)
        eps = float(evals[idx])
        psi = evecs[:, idx]

    w = drdi[: rc_idx + 1]
    u_full = np.zeros_like(r)
    u_full[1:rc_idx] = psi / np.sqrt(w[1:rc_idx])
    rad = np.zeros_like(r)
    rad[1:rc_idx] = u_full[1:rc_idx] / r[1:rc_idx]
    if l_qn == 0:
        r1sq, r2sq = r[1] ** 2, r[2] ** 2
        rad[0] = (r2sq * rad[1] - r1sq * rad[2]) / max(r2sq - r1sq, 1e-18)
    peak = int(np.argmax(np.abs(u_full)))
    if rad[peak] < 0.0:
        rad = -rad
    norm = float(np.sqrt(np.sum((u_full[1:rc_idx] ** 2) * drdi[1:rc_idx])))
    if norm > 0.0:
        rad /= norm
    return eps, rad


def _split_parabola_explicit_rc(
    r: np.ndarray,
    drdi: np.ndarray,
    rad_1z: np.ndarray,
    l_qn: int,
    rc_z: float,
) -> tuple[np.ndarray, float]:
    r"""Construct inner-zeta radial orbital matching SIESTA ``fit_parabola`` at explicit ``rc_z``."""
    nrc = int(np.argmin(np.abs(r - rc_z)))
    nrc = max(3, min(nrc, len(r) - 2))
    rsp = float(r[nrc])
    # Fit c1, c2 so (c1 r^2 + c2) r^l matches R_{1z}(r) in value & slope at r = rsp
    r_m1, r_p1 = float(r[nrc - 1]), float(r[nrc + 1])
    frsp = float(rad_1z[nrc])
    dfrsp = float((rad_1z[nrc + 1] - rad_1z[nrc - 1]) / (r_p1 - r_m1))
    c1 = 0.5 * (dfrsp * rsp - l_qn * frsp) / (rsp ** (l_qn + 2))
    c2 = frsp / (rsp**l_qn) - c1 * (rsp**2)

    rad_z = np.zeros_like(r)
    mask = np.arange(len(r)) < nrc
    rm = r[mask]
    rad_z[mask] = rad_1z[mask] - (c1 * (rm**2) + c2) * (rm**l_qn)
    rad_z[nrc] = 0.0
    norm2 = float(np.sum((rad_z[1:nrc] ** 2) * (r[1:nrc] ** 2) * drdi[1:nrc]))
    if norm2 > 1e-24:
        rad_z /= np.sqrt(norm2)
    return rad_z, rsp


def _find_split_norm_rc(
    r: np.ndarray,
    drdi: np.ndarray,
    rad_1z: np.ndarray,
    rc_1z: float,
    l_qn: int,
    split_norm: float,
) -> float:
    """Determine inner-zeta matching radius via SIESTA's tail-norm criterion."""
    nrc1 = int(np.argmin(np.abs(r - rc_1z)))
    integrand = (rad_1z**2) * (r**2) * drdi
    cum = np.zeros_like(r)
    cum[1:] = np.cumsum(integrand[1:])
    tot = max(float(cum[nrc1]), 1e-12)
    tail_norm = 1.0 - cum / tot
    best_r = 0.72 * rc_1z
    best_err = 1e9
    for idx in range(10, nrc1 - 2):
        rsp = float(r[idx])
        if rsp < 0.35 * rc_1z:
            continue
        r_m1, r_p1 = float(r[idx - 1]), float(r[idx + 1])
        frsp = float(rad_1z[idx])
        dfrsp = float((rad_1z[idx + 1] - rad_1z[idx - 1]) / (r_p1 - r_m1))
        c1 = 0.5 * (dfrsp * rsp - l_qn * frsp) / (rsp ** (l_qn + 2))
        c2 = frsp / (rsp**l_qn) - c1 * (rsp**2)
        p_norm = (
            (c1 * c1) / (2 * l_qn + 7) * (rsp ** (2 * l_qn + 7))
            + (2.0 * c1 * c2) / (2 * l_qn + 5) * (rsp ** (2 * l_qn + 5))
            + (c2 * c2) / (2 * l_qn + 3) * (rsp ** (2 * l_qn + 3))
        )
        # Note SIESTA split_table compares total remainder norm vs split_norm
        diff_sq = tail_norm[idx] + p_norm
        err = abs(diff_sq - split_norm)
        if err < best_err:
            best_err = err
            best_r = rsp
    return best_r


# ---------------------------------------------------------------------------
# PSML Semilocal -> Local + KB Projector Converter
# ---------------------------------------------------------------------------


def _vander(a: float, x: np.ndarray) -> np.ndarray:
    """Vanderbilt generalized Gaussian used by SIESTA ``vlocal_from_chlocal``."""
    s = np.sinh(a * x) / np.sinh(a)
    gexp = s * s
    return np.where(gexp < 700.0, np.exp(-gexp), 0.0)


def _build_vlocal_and_kbs_from_semilocal(
    r: np.ndarray,
    drdi: np.ndarray,
    a: float,
    z_val: float,
    v_sl_map: dict[int, np.ndarray],
    pswf_map: dict[int, tuple[int, np.ndarray]],
    rho_val_psml: np.ndarray,
    rho_core: np.ndarray,
    occ_shells: list[tuple[int, int, float]],
    atom_xc: str,
) -> tuple[np.ndarray, list[np.ndarray], list[int], list[float], list[float]]:
    """Construct smooth :math:`V_{\text{loc}}(r)` and normalized KB projectors from semilocal PSML."""
    # Determine rgauss where semilocal potentials coincide and reach -z_val/r (Ha)
    l_keys = sorted(v_sl_map.keys())
    v_s = v_sl_map[l_keys[0]]
    asympt = -z_val / np.maximum(r, 1e-12)
    rgauss2_idx = np.where((r > 0.5) & (np.abs(v_s - asympt) > 1e-4))[0]
    rgauss2 = float(r[min(int(rgauss2_idx[-1]) + 1, len(r) - 1)]) if len(rgauss2_idx) else 2.5

    if len(l_keys) > 1:
        max_diff = np.max(np.stack([np.abs(v_sl_map[lk] - v_s) for lk in l_keys[1:]], axis=0), axis=0)
        rg_idx = np.where((r > 0.5) & (max_diff > 1e-4))[0]
        rgauss = float(r[min(int(rg_idx[-1]) + 1, len(r) - 1)]) if len(rg_idx) else rgauss2
    else:
        rgauss = rgauss2

    if rgauss2 > 1.3 * rgauss and rgauss > 0.4:
        # Large-core smooth C2 fit matching SIESTA vlocal_as_fit (ndevfit=2)
        nr_g = min(int(np.argmin(np.abs(r - rgauss))) + 2, len(r) - 3)
        rg = float(r[nr_g])
        rv_arr = r * v_s
        dev = (rv_arr[nr_g + 1] - rv_arr[nr_g - 1]) * 0.5
        dev2 = rv_arr[nr_g + 1] + rv_arr[nr_g - 1] - 2.0 * rv_arr[nr_g]
        dev2 = (dev2 - a * dev) / (drdi[nr_g] ** 2)
        dev = dev / drdi[nr_g]
        vlc = float(rv_arr[nr_g])
        var1 = dev / vlc - 1.0 / rg
        var2 = dev2 / vlc - 2.0 * var1 / rg - var1 * var1
        v1 = (6.0 * rg * var1 - 3.0 * rg * rg * var2) / (6.0 * rg * rg)
        v2 = (2.0 * rg * var2 - 2.0 * var1) / (6.0 * rg * rg)
        v3 = vlc / (rg * np.exp((v1 + v2 * rg) * rg * rg))
        v_loc = np.where(r <= rg, v3 * np.exp((v1 + v2 * r) * r * r), v_s)
    else:
        # Small-core Vanderbilt smooth local charge potential (vlocal_from_chlocal)
        alp = 1.815 / max(rgauss, 0.5)
        rho_ch = _vander(1.0, alp * r)
        q_ch = float(np.sum(4.0 * np.pi * (r[1:] ** 2) * rho_ch[1:] * drdi[1:]))
        rho_ch = -z_val * rho_ch / max(q_ch, 1e-12)
        v_loc = _radial_hartree_log(r, drdi, rho_ch)

    # Atomic self-consistent density screening for KB reference wavefunctions
    occ_l = {l_q: occ for _, l_q, occ in occ_shells if occ > 0.0}
    vh = _radial_hartree_log(r, drdi, np.maximum(rho_val_psml, 0.0))
    _, vxc = _radial_xc_potential(r, drdi, np.maximum(rho_val_psml, 0.0) + rho_core, atom_xc)
    v_scr = vh + vxc

    # Short SCF refresh of free-atom screening density using semilocal v_l
    if occ_l:
        rho_cur = np.maximum(rho_val_psml.copy(), 0.0)
        for _ in range(25):
            vh_c = _radial_hartree_log(r, drdi, rho_cur)
            _, vxc_c = _radial_xc_potential(r, drdi, rho_cur + rho_core, atom_xc)
            v_scr_c = vh_c + vxc_c
            rho_new = np.zeros_like(r)
            for l_q, occ in occ_l.items():
                vp = v_sl_map.get(l_q, v_loc)
                _, r_orb = _solve_radial_state(r, drdi, a, l_q, 0, vp + v_scr_c, len(r) - 1)
                rho_new += (occ / (4.0 * np.pi)) * (r_orb**2)
            q_n = float(np.sum(4.0 * np.pi * (r[1:] ** 2) * rho_new[1:] * drdi[1:]))
            if q_n > 1e-12:
                rho_new *= z_val / q_n
            if float(np.max(np.abs(rho_new - rho_cur))) < 1e-6:
                rho_cur = rho_new
                v_scr = vh_c + vxc_c
                break
            rho_cur = 0.4 * rho_new + 0.6 * rho_cur
        else:
            v_scr = _radial_hartree_log(r, drdi, rho_cur) + _radial_xc_potential(
                r, drdi, rho_cur + rho_core, atom_xc
            )[1]

    kb_projs: list[np.ndarray] = []
    kb_l: list[int] = []
    kb_energies: list[float] = []
    kb_rcut: list[float] = []
    for l_q in sorted(v_sl_map.keys()):
        vp = v_sl_map[l_q]
        dv = vp - v_loc
        # Reference orbital for l_q in semilocal V_l + atomic screening
        _, r_ref = _solve_radial_state(r, drdi, a, l_q, 0, vp + v_scr, len(r) - 1)
        if np.max(np.abs(dv)) < 1e-7:
            continue
        chi_raw = dv * r_ref
        # Clip cleanly where chi_raw vanishes
        nz = np.where((r > 0.2) & (np.abs(chi_raw) > 1e-7))[0]
        if len(nz) == 0:
            continue
        nrc = min(int(nz[-1]) + 2, len(r) - 1)
        rc_kb = float(r[nrc])
        chi_raw[nrc:] = 0.0
        nrm2 = float(np.sum((chi_raw[1:nrc] ** 2) * (r[1:nrc] ** 2) * drdi[1:nrc]))
        denom = float(np.sum((r_ref[1:nrc] * chi_raw[1:nrc]) * (r[1:nrc] ** 2) * drdi[1:nrc]))
        if nrm2 <= 1e-14 or abs(denom) <= 1e-14:
            continue
        chi_norm = chi_raw / np.sqrt(nrm2)
        ekb = nrm2 / denom
        kb_projs.append(chi_norm)
        kb_l.append(l_q)
        kb_energies.append(ekb)
        kb_rcut.append(rc_kb)
    return v_loc, kb_projs, kb_l, kb_energies, kb_rcut


# ---------------------------------------------------------------------------
# High-Level PAO Generator API
# ---------------------------------------------------------------------------


def _parse_psml_for_atom(psml_path: str | Path) -> dict[str, Any]:
    """Parse raw PSML XML fields on its native grid for 1D atomic PAO generation."""
    p = Path(psml_path)
    if p.suffix.lower() == ".psf":
        raise ValueError(f"Legacy .psf format is unsupported ({p}); provide a .psml file.")
    raw = p.read_bytes()
    root = ET.fromstring(raw)
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    e_scale = _HA_PER_UNIT.get(root.get("energy_unit", "hartree").lower(), 1.0)
    l_scale = _BOHR_PER_UNIT.get(root.get("length_unit", "bohr").lower(), 1.0)
    spec = root.find("pseudo-atom-spec")
    symbol = spec.get("atomic-label", "X").strip() if spec is not None else "X"
    atomic_number = int(spec.get("atomic-number", "1")) if spec is not None else 1
    z_val = float(spec.get("z-pseudo", "1.0")) if spec is not None else 1.0
    mass = float(spec.get("atomic-mass", "28.085")) if spec is not None else 28.085

    atom_xc = "PBE"
    xc_ann = root.find("pseudo-atom-spec/exchange-correlation/annotation")
    if xc_ann is not None:
        stype = xc_ann.get("siesta-xc-type", "").upper()
        sauth = xc_ann.get("siesta-xc-authors", "").upper()
        if stype == "LDA":
            atom_xc = "LDA_PW92" if "PW" in sauth else "LDA_PZ"

    shells_occ: list[tuple[int, int, float]] = []
    vc = root.find("pseudo-atom-spec/valence-configuration")
    if vc is not None:
        for sh in vc.findall("shell"):
            nq = int(sh.get("n", "1"))
            lq_raw = sh.get("l", "s").lower()
            lq = _L_MAP[lq_raw] if lq_raw in _L_MAP else int(lq_raw)
            occ = float(sh.get("occupation", "0.0"))
            shells_occ.append((nq, lq, occ))
    if not shells_occ:
        shells_occ = [(1, 0, z_val)]

    r_raw = np.fromstring(root.findtext("grid/grid-data", "").strip(), sep=" ", dtype=np.float64) * l_scale

    rho_el = root.find("valence-charge/radfunc/data")
    if rho_el is not None and rho_el.text:
        rho_na_raw = np.fromstring(rho_el.text.strip(), sep=" ", dtype=np.float64) / (l_scale**3)
        # Official PSML 1.1 valence-charge stores 4*pi*r^2*rho or 4*pi*rho? Wait: integrate to verify!
        r_rho = r_raw[: len(rho_na_raw)]
        q_4pi = float(np.trapezoid(rho_na_raw * (r_rho**2), r_rho))  # if 4*pi*rho
        q_plain = float(np.trapezoid(4.0 * np.pi * rho_na_raw * (r_rho**2), r_rho))
        if abs(q_4pi - z_val) < abs(q_plain - z_val):
            rho_na_raw = rho_na_raw / (4.0 * np.pi)
    else:
        sigma = 1.2
        r_rho = r_raw
        rho_na_raw = (z_val / (np.pi**1.5 * sigma**3)) * np.exp(-((r_raw / sigma) ** 2))

    # Pseudo-core charge
    pc_el = root.find("pseudocore-charge/radfunc/data")
    pc_div_4pi = True
    if pc_el is None:
        pc_el = root.find("pseudo-core-charge/radfunc/data")
        pc_div_4pi = False
    if pc_el is not None and pc_el.text:
        rho_core_raw = np.fromstring(pc_el.text.strip(), sep=" ", dtype=np.float64) / (l_scale**3)
        if pc_div_4pi:
            rho_core_raw = rho_core_raw / (4.0 * np.pi)
        r_core = r_raw[: len(rho_core_raw)]
    else:
        rho_core_raw = np.zeros_like(r_raw)
        r_core = r_raw

    # Semilocal potentials (if present)
    v_sl_raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sl_block = root.find("semilocal-potentials")
    if sl_block is not None:
        for sl in sl_block.findall("slps"):
            lr = sl.get("l", "s").lower()
            lq = _L_MAP[lr] if lr in _L_MAP else int(lr)
            vals = np.fromstring((sl.findtext("radfunc/data") or "").strip(), sep=" ", dtype=np.float64) * e_scale
            v_sl_raw[lq] = (r_raw[: len(vals)], vals)

    # Local potential & KB projectors (if already split)
    v_loc_raw: tuple[np.ndarray, np.ndarray] | None = None
    vl_el = root.find("local-potential/radfunc/data")
    if vl_el is not None and vl_el.text:
        vals = np.fromstring(vl_el.text.strip(), sep=" ", dtype=np.float64) * e_scale
        v_loc_raw = (r_raw[: len(vals)], vals)

    kb_raw: list[tuple[int, int, float, float, np.ndarray, np.ndarray]] = []
    for blk in root.findall("nonlocal-projectors"):
        for pr in blk.findall("proj"):
            lr = pr.get("l", "s").lower()
            lq = _L_MAP[lr] if lr in _L_MAP else int(lr)
            ekb = float(pr.get("ekb", "0.0")) * e_scale
            j_attr = pr.get("j")
            j2 = int(round(2.0 * float(j_attr))) if j_attr is not None else 0
            rc_kb = float(pr.get("rcut", "2.0")) * l_scale
            pdata = np.fromstring((pr.findtext("radfunc/data") or "").strip(), sep=" ", dtype=np.float64)
            kb_raw.append((lq, j2, ekb, rc_kb, r_raw[: len(pdata)], pdata))
        break  # take canonical block first

    return {
        "symbol": symbol,
        "atomic_number": atomic_number,
        "z_val": z_val,
        "mass": mass,
        "atom_xc": atom_xc,
        "shells_occ": shells_occ,
        "r_raw": r_raw,
        "rho_na": (r_rho, rho_na_raw),
        "rho_core": (r_core, rho_core_raw),
        "v_sl_raw": v_sl_raw,
        "v_loc_raw": v_loc_raw,
        "kb_raw": kb_raw,
    }


def _adjust_occupations_for_charge(
    shells_occ: list[tuple[int, int, float]],
    ionic_charge: float,
) -> list[tuple[int, int, float]]:
    """Adjust valence occupations for species net charge ``ionic_charge`` (``deltaQ = Zval - Nel``)."""
    total_occ = sum(o for _, _, o in shells_occ)
    target_nel = max(total_occ - ionic_charge, 0.2)
    delta = target_nel - total_occ
    # Sort shells from highest (n, l) downwards so electrons are removed/added on frontier shell first
    ordered = sorted(range(len(shells_occ)), key=lambda idx: (shells_occ[idx][0], shells_occ[idx][1]), reverse=True)
    occ_out = [o for _, _, o in shells_occ]
    rem = delta
    for idx in ordered:
        n_q, l_q, o_cur = shells_occ[idx]
        cap = 2.0 * (2 * l_q + 1)
        if rem < 0.0:
            take = min(o_cur, -rem)
            occ_out[idx] = o_cur - take
            rem += take
        elif rem > 0.0:
            add = min(cap - o_cur, rem)
            occ_out[idx] = o_cur + add
            rem -= add
        _ = n_q
    return [(n_q, l_q, float(occ_out[i])) for i, (n_q, l_q, _) in enumerate(shells_occ)]


def default_basis_spec(
    symbol: str,
    shells_occ: list[tuple[int, int, float]],
    basis_size: Literal["SZ", "DZ", "DZP"] = "DZP",
    xc: str = "PBE",
    energy_shift_ha: float = 0.01 * RY_TO_HARTREE,
    split_norm: float = 0.15,
) -> SpeciesBasisSpec:
    """Build a standard SIESTA-like ``SZ``, ``DZ``, or ``DZP`` :class:`SpeciesBasisSpec`.

    Theory:
        Maps SIESTA standard basis sizes (single-zeta ``SZ``, double-zeta ``DZ``, and
        double-zeta polarized ``DZP``) onto explicit per-shell specifications using
        ``PAO.EnergyShift`` and perturbative polarization of the highest angular momentum
        occupied valence shell.

    Algorithm:
        Constructs valence shells with ``n_zeta = 1`` (``SZ``) or ``2`` (``DZ``/``DZP``)
        and appends a perturbative polarization shell ``l_pol = max(l) + 1`` when
        ``basis_size == "DZP"``.

    Parameters:
        symbol: Element symbol.
        shells_occ: Occupied valence ``(n, l, occ)`` tuples.
        basis_size: Preset tier (``"SZ"``, ``"DZ"``, or ``"DZP"``).
        xc: XC functional identifier for atomic screening.
        energy_shift_ha: Confinement energy shift (default ``0.005 Ha = 0.01 Ry``).
        split_norm: Tail-norm parameter for default inner-zeta determination.

    Returns:
        Configured :class:`SpeciesBasisSpec`.

    Invariants:
        - ``DZP`` adds one perturbative polarization shell with ``l = max(l) + 1``.

    SIESTA Mapping:
        Mirrors ``PAO.BasisSize SZ/DZ/DZP`` defaults in SIESTA ``basis_specs.f``.
    """
    n_z = 1 if basis_size.upper() == "SZ" else 2
    shell_specs = [
        ShellBasisSpec(
            n=n_q,
            l=l_q,
            n_zeta=n_z,
            rc_list=None,
            energy_shift_ha=energy_shift_ha,
            split_norm=split_norm,
        )
        for n_q, l_q, occ in shells_occ
        if occ > 0.0 or (n_q, l_q) in [(s[0], s[1]) for s in shells_occ]
    ]
    if basis_size.upper() == "DZP":
        parent = max(shell_specs, key=lambda s: (s.l, s.n))
        if all(s.l != parent.l + 1 for s in shell_specs):
            shell_specs.append(
                ShellBasisSpec(
                    n=parent.n,
                    l=parent.l + 1,
                    n_zeta=1,
                    rc_list=None,
                    is_polarization=True,
                    polarization_mode="perturbative",
                    parent_l=parent.l,
                )
            )
    return SpeciesBasisSpec(
        symbol=symbol,
        shells=tuple(shell_specs),
        ionic_charge=0.0,
        xc=xc,
        energy_shift_ha=energy_shift_ha,
        split_norm=split_norm,
    )


def generate_species_from_psml(
    psml_path: str | Path,
    basis_spec: SpeciesBasisSpec | Literal["SZ", "DZ", "DZP"] = "DZP",
    xc: str = "PBE",
    n_pao_grid: int = 401,
) -> tuple[SpeciesData, dict[str, Any]]:
    r"""Generate complete confined NAO ``SpeciesData`` directly from a ``.psml`` file.

    Theory:
        Solves the self-consistent 1D radial Kohn-Sham equation (neutral or charged :math:`q`),
        applies explicit cutoffs :math:`r_c^{(1)}` (or ``energy_shift_ha``) and soft
        confinement :math:`V_{\text{conf}}(r; r_i, r_c, V_0)` to obtain :math:`1\zeta` orbitals,
        constructs higher-:math:`\zeta` orbitals at explicit cutoffs :math:`r_c^{(z)}` via
        SIESTA ``fit_parabola``, adds polarization orbitals, and recomputes :math:`\rho^{\text{NA}}(r)`
        and :math:`V^{\text{NA}}(r)` consistently.

    Algorithm:
        Parses semilocal or KB-split ``.psml`` XML onto a SIESTA logarithmic grid, solves the
        1D Sturm-Liouville radial matrix eigenvalue problem per shell, builds inner zetas,
        resamples onto uniform spline tables of length ``n_pao_grid``, and returns
        ``(species_data, diagnostics)``.

    Parameters:
        psml_path: Path to ``.psml`` XML file.
        basis_spec: Either a full :class:`SpeciesBasisSpec` (explicit ``rc``, ``V0``, ``ri``, ``q``)
            or preset string ``"SZ"``, ``"DZ"``, ``"DZP"``.
        xc: Reference XC functional (used when ``basis_spec`` is a preset string).
        n_pao_grid: Uniform radial grid size for ``SpeciesData.pao_r`` spline tables.

    Returns:
        Tuple ``(species_data, diagnostics)`` with orbital eigenvalues and cutoff metadata.

    Invariants:
        - Every generated orbital is normalized and strictly zero for :math:`r \ge r_c`.
        - Recomputed :math:`4\pi \int r^2 \rho^{\text{NA}}(r) dr = Z_{\text{val}}`.

    SIESTA Mapping:
        Replaces external ``siesta`` execution of ``atom.F`` during basis generation and
        optimization, feeding ``SpeciesData`` directly into ``TwoCenterEvaluator`` and ``GridOps``.
    """
    parsed = _parse_psml_for_atom(psml_path)
    z_val = float(parsed["z_val"])
    shells_occ = parsed["shells_occ"]
    if isinstance(basis_spec, str):
        spec = default_basis_spec(parsed["symbol"], shells_occ, basis_size=basis_spec, xc=xc)
    else:
        spec = basis_spec

    # Determine max radial span
    explicit_rcs = [rc for sh in spec.shells if sh.rc_list for rc in sh.rc_list]
    r_max = max([20.0] + [rc + 4.0 for rc in explicit_rcs])
    a_step = float(spec.log_grid_step)
    b_ref = 22.0 / (np.exp(0.022 * 440) - 1.0)  # keep the default first-point spacing near the nucleus
    n_log = max(441, int(np.ceil(np.log(r_max / b_ref + 1.0) / a_step)) + 1)
    r, drdi, a, _ = _make_siesta_log_grid(r_max=r_max, n_pts=n_log, a=a_step)

    r_rho, rho_na_raw = parsed["rho_na"]
    rho_psml = np.maximum(np.interp(r, r_rho, rho_na_raw, right=0.0), 0.0)
    r_core, rho_core_raw = parsed["rho_core"]
    rho_core = np.maximum(np.interp(r, r_core, rho_core_raw, right=0.0), 0.0)
    has_nlcc = bool(np.max(rho_core) > 1e-12)

    # Semilocal channels or Local + KB
    v_sl_map: dict[int, np.ndarray] = {}
    for lq, (r_sl, vals_sl) in parsed["v_sl_raw"].items():
        v_sl_map[lq] = np.where(
            r <= float(r_sl[-1]),
            np.interp(r, r_sl, vals_sl),
            -z_val / np.maximum(r, 1e-12),
        )

    if parsed["v_loc_raw"] is not None:
        r_vl, vals_vl = parsed["v_loc_raw"]
        v_loc_r = np.where(
            r <= float(r_vl[-1]),
            np.interp(r, r_vl, vals_vl),
            -z_val / np.maximum(r, 1e-12),
        )
        kb_projs_r: list[np.ndarray] = []
        kb_l: list[int] = []
        kb_j2: list[int] = []
        kb_energies: list[float] = []
        kb_rcut: list[float] = []
        for lq, j2, ekb, rc_kb, r_p, pdata in parsed["kb_raw"]:
            chi = np.where(r <= rc_kb, np.interp(r, r_p, pdata, right=0.0), 0.0)
            kb_projs_r.append(chi)
            kb_l.append(lq)
            kb_j2.append(j2)
            kb_energies.append(ekb)
            kb_rcut.append(rc_kb)
    elif v_sl_map:
        v_loc_r, kb_projs_r, kb_l, kb_energies, kb_rcut = _build_vlocal_and_kbs_from_semilocal(
            r,
            drdi,
            a,
            z_val,
            v_sl_map,
            {},
            rho_psml,
            rho_core,
            shells_occ,
            parsed["atom_xc"],
        )
        kb_j2 = [0] * len(kb_l)
    else:
        raise ValueError(f"PSML {psml_path} contains neither <local-potential> nor <semilocal-potentials>.")

    # Build lookup of per-l radial potentials for the atomic KS solver
    kb_by_l: dict[int, list[tuple[np.ndarray, float]]] = {}
    for chi, lq, ekb, j2 in zip(kb_projs_r, kb_l, kb_energies, kb_j2, strict=True):
        if j2 == 0:
            kb_by_l.setdefault(lq, []).append((chi, ekb))
    if not kb_by_l:
        for chi, lq, ekb in zip(kb_projs_r, kb_l, kb_energies, strict=True):
            kb_by_l.setdefault(lq, []).append((chi, ekb))

    def _channel_pot(lq: int) -> tuple[np.ndarray, list[tuple[np.ndarray, float]] | None]:
        if lq in v_sl_map:
            return v_sl_map[lq], None
        return v_loc_r, kb_by_l.get(lq)

    # Free-atom (ionic_charge q) SCF radial screening potential
    q_occ_shells = _adjust_occupations_for_charge(shells_occ, spec.ionic_charge)
    nel_atom = sum(occ for _, _, occ in q_occ_shells)
    rho_cur = rho_psml * (nel_atom / max(z_val, 1e-12))
    v_scr = _radial_hartree_log(r, drdi, rho_cur) + _radial_xc_potential(
        r, drdi, rho_cur + rho_core, spec.xc
    )[1]

    # Determine minimum radial node order (usually 0 in pseudopotential valence; 1 if semicore shares l)
    l_min_n: dict[int, int] = {}
    for n_q, l_q, _ in sorted(q_occ_shells, key=lambda x: (x[1], x[0])):
        l_min_n.setdefault(l_q, n_q)

    free_energies: dict[tuple[int, int], float] = {}
    for it in range(45):
        vh_c = _radial_hartree_log(r, drdi, rho_cur)
        _, vxc_c = _radial_xc_potential(r, drdi, rho_cur + rho_core, spec.xc)
        v_scr = vh_c + vxc_c
        rho_new = np.zeros_like(r)
        for n_q, l_q, occ in q_occ_shells:
            vp, kbl = _channel_pot(l_q)
            n_node = max(n_q - l_min_n.get(l_q, n_q), 0)
            eps_nl, rad_nl = _solve_radial_state(r, drdi, a, l_q, n_node, vp + v_scr, len(r) - 1, kb_projs=kbl)
            free_energies[(n_q, l_q)] = eps_nl
            if occ > 0.0:
                rho_new += (occ / (4.0 * np.pi)) * (rad_nl**2)
        q_sum = float(np.sum(4.0 * np.pi * (r[1:] ** 2) * rho_new[1:] * drdi[1:]))
        if q_sum > 1e-12:
            rho_new *= nel_atom / q_sum
        if float(np.max(np.abs(rho_new - rho_cur))) < 1e-7 and it >= 3:
            rho_cur = rho_new
            vh_f = _radial_hartree_log(r, drdi, rho_cur)
            _, vxc_f = _radial_xc_potential(r, drdi, rho_cur + rho_core, spec.xc)
            v_scr = vh_f + vxc_f
            break
        rho_cur = 0.4 * rho_new + 0.6 * rho_cur

    # Generate confined 1zeta, split higher-zeta, and polarization orbitals
    orb_records: list[tuple[int, int, int, float, np.ndarray]] = []
    diag_shells: list[dict[str, Any]] = []
    first_zeta_by_l: dict[tuple[int, int], tuple[float, float, np.ndarray]] = {}

    valence_specs = [sh for sh in spec.shells if not (sh.is_polarization and sh.polarization_mode == "perturbative")]
    pol_pert_specs = [sh for sh in spec.shells if sh.is_polarization and sh.polarization_mode == "perturbative"]

    for sh in valence_specs:
        vp, kbl = _channel_pot(sh.l)
        n_node = max(sh.n - l_min_n.get(sh.l, sh.n), 0)
        if sh.rc_list is not None and len(sh.rc_list) > 0:
            rc_1 = float(sh.rc_list[0])
        else:
            e_shift = sh.energy_shift_ha if sh.energy_shift_ha is not None else spec.energy_shift_ha
            if (sh.n, sh.l) in free_energies:
                eps_free = free_energies[(sh.n, sh.l)]
            else:
                eps_free, _ = _solve_radial_state(r, drdi, a, sh.l, n_node, vp + v_scr, len(r) - 1, kb_projs=kbl)

            def _shift_err(rc_try: float) -> float:
                idx = int(np.argmin(np.abs(r - rc_try)))
                vs = _build_vsoft(r, sh.ri_bohr, float(r[idx]), sh.v0_ha)
                e_c, _ = _solve_radial_state(r, drdi, a, sh.l, n_node, vp + v_scr + vs, idx, kb_projs=kbl)
                return (e_c - eps_free) - e_shift

            lo_rc, hi_rc = 2.0, 12.0
            f_lo, f_hi = _shift_err(lo_rc), _shift_err(hi_rc)
            if f_lo * f_hi < 0.0:
                rc_1 = float(brentq(_shift_err, lo_rc, hi_rc, xtol=1e-4))
            else:
                rc_1 = 6.0

        nrc1 = int(np.argmin(np.abs(r - rc_1)))
        nrc1 = max(nrc1, 5)
        rc_1_eff = float(r[nrc1])
        v_soft = _build_vsoft(r, sh.ri_bohr, rc_1_eff, sh.v0_ha)
        eps_1z, rad_1z = _solve_radial_state(
            r, drdi, a, sh.l, n_node, vp + v_scr + v_soft, nrc1, kb_projs=kbl
        )
        orb_records.append((sh.n, sh.l, 1, rc_1_eff, rad_1z))
        first_zeta_by_l[(sh.n, sh.l)] = (eps_1z, rc_1_eff, rad_1z)
        diag_shells.append(
            {
                "n": sh.n,
                "l": sh.l,
                "zeta": 1,
                "rc_bohr": rc_1_eff,
                "energy_ha": eps_1z,
                "energy_ry": 2.0 * eps_1z,
                "v0_ha": sh.v0_ha,
                "ri_bohr": sh.ri_bohr,
                "polarization": sh.is_polarization,
            }
        )

        # Split-zeta z >= 2 using explicit rc_list[z-1] (or split_norm fallback for presets)
        for z_idx in range(2, sh.n_zeta + 1):
            if sh.rc_list is not None and len(sh.rc_list) >= z_idx:
                rc_z_target = float(sh.rc_list[z_idx - 1])
            else:
                sp_n = sh.split_norm if sh.split_norm is not None else spec.split_norm
                if z_idx > 2:
                    sp_n = sp_n / (2.0 * (z_idx - 2))
                rc_z_target = _find_split_norm_rc(r, drdi, rad_1z, rc_1_eff, sh.l, sp_n)
            rad_z, rc_z_eff = _split_parabola_explicit_rc(r, drdi, rad_1z, sh.l, rc_z_target)
            orb_records.append((sh.n, sh.l, z_idx, rc_z_eff, rad_z))
            diag_shells.append(
                {
                    "n": sh.n,
                    "l": sh.l,
                    "zeta": z_idx,
                    "rc_bohr": rc_z_eff,
                    "polarization": sh.is_polarization,
                }
            )

    # Perturbative polarization orbitals (SIESTA POLgen default for DZP)
    for sh in pol_pert_specs:
        par_l = sh.parent_l if sh.parent_l is not None else sh.l - 1
        par_key = (sh.n, par_l)
        if par_key not in first_zeta_by_l:
            candidates = [k for k in first_zeta_by_l if k[1] == par_l]
            par_key = candidates[-1]
        eps_par, rc_par, rad_par = first_zeta_by_l[par_key]
        rc_1_pol = float(sh.rc_list[0]) if (sh.rc_list and len(sh.rc_list) > 0) else rc_par
        nrc_p = max(int(np.argmin(np.abs(r - rc_1_pol))), 5)
        rc_1_eff = float(r[nrc_p])

        vp_pol, kbl_pol = _channel_pot(sh.l)
        v_soft_pol = _build_vsoft(r, sh.ri_bohr, rc_1_eff, sh.v0_ha)
        diag_p, off_p = _radial_tridiag(
            r, drdi, a, sh.l, vp_pol + v_scr + v_soft_pol, nrc_p
        )
        m = nrc_p - 1
        rhs = -r[1:nrc_p] * (r[1:nrc_p] * rad_par[1:nrc_p] * np.sqrt(drdi[1:nrc_p]))
        has_kb_p = bool(kbl_pol and any(ekb != 0.0 for _, ekb in kbl_pol))
        if not has_kb_p:
            ab = np.zeros((3, m), dtype=np.float64)
            ab[0, 1:] = off_p
            ab[1, :] = diag_p - eps_par
            ab[2, :-1] = off_p
            psi_pol = solve_banded((1, 1), ab, rhs)
        else:
            h_pol = np.diag(diag_p - eps_par) + np.diag(off_p, k=1) + np.diag(off_p, k=-1)
            r_act = r[1:nrc_p]
            sqrt_w_act = np.sqrt(drdi[1:nrc_p])
            for chi_on_r, ekb in kbl_pol or []:
                if ekb == 0.0:
                    continue
                vec = r_act * chi_on_r[1:nrc_p] * sqrt_w_act
                h_pol += ekb * np.outer(vec, vec)
            psi_pol = np.linalg.solve(h_pol, rhs)
        rad_pol = np.zeros_like(r)
        rad_pol[1:nrc_p] = psi_pol / (r[1:nrc_p] * np.sqrt(drdi[1:nrc_p]))
        peak = int(np.argmax(np.abs(rad_pol)))
        if rad_pol[peak] < 0.0:
            rad_pol = -rad_pol
        norm2 = float(np.sum((rad_pol[1:nrc_p] ** 2) * (r[1:nrc_p] ** 2) * drdi[1:nrc_p]))
        if norm2 > 1e-24:
            rad_pol /= np.sqrt(norm2)
        orb_records.append((sh.n, sh.l, 1, rc_1_eff, rad_pol))
        diag_shells.append(
            {
                "n": sh.n,
                "l": sh.l,
                "zeta": 1,
                "rc_bohr": rc_1_eff,
                "v0_ha": sh.v0_ha,
                "ri_bohr": sh.ri_bohr,
                "polarization": True,
                "polarization_mode": "perturbative",
            }
        )
        for z_idx in range(2, sh.n_zeta + 1):
            if sh.rc_list is not None and len(sh.rc_list) >= z_idx:
                rc_z_t = float(sh.rc_list[z_idx - 1])
            else:
                sp_n = sh.split_norm if sh.split_norm is not None else spec.split_norm
                rc_z_t = _find_split_norm_rc(r, drdi, rad_pol, rc_1_eff, sh.l, sp_n)
            rad_z, rc_z_eff = _split_parabola_explicit_rc(r, drdi, rad_pol, sh.l, rc_z_t)
            orb_records.append((sh.n, sh.l, z_idx, rc_z_eff, rad_z))
            diag_shells.append(
                {
                    "n": sh.n,
                    "l": sh.l,
                    "zeta": z_idx,
                    "rc_bohr": rc_z_eff,
                    "polarization": True,
                    "polarization_mode": "perturbative",
                }
            )

    # Resample PAOs, V_NA, rho_NA, V_loc, KB projectors onto clean uniform grid tables
    max_rc = max(rec[3] for rec in orb_records)
    pao_r = np.linspace(0.0, max_rc, n_pao_grid, dtype=np.float64)
    pao_vals_list: list[np.ndarray] = []
    pao_l_list: list[int] = []
    pao_n_list: list[int] = []
    pao_zeta_list: list[int] = []
    pao_rcut_list: list[float] = []
    for n_o, l_o, z_o, rc_o, rad_o in orb_records:
        vals_u = np.where(pao_r <= rc_o, np.interp(pao_r, r, rad_o, right=0.0), 0.0)
        vals_u[-1 if abs(rc_o - max_rc) < 1e-8 else np.argmin(np.abs(pao_r - rc_o))] = 0.0
        pao_vals_list.append(vals_u)
        pao_l_list.append(l_o)
        pao_n_list.append(n_o)
        pao_zeta_list.append(z_o)
        pao_rcut_list.append(rc_o)

    iz_counter: dict[int, int] = {}
    pao_iz_list: list[int] = []
    for l_val in pao_l_list:
        cnt = iz_counter.get(l_val, 0)
        pao_iz_list.append(cnt)
        iz_counter[l_val] = cnt + 1

    # Recompute neutral-atom rho_NA and V_NA on uniform grid from 1zeta PAOs (matching SIESTA atom.F)
    rcut_na = max(max_rc, max(kb_rcut, default=4.0))
    n_unif = 401
    r_unif = np.linspace(0.0, max(rcut_na + 0.5, 8.0), n_unif, dtype=np.float64)
    rho_atm = np.zeros_like(r_unif)
    for (n_o, l_o, z_o, rc_o, rad_o) in orb_records:
        if z_o == 1:
            occ = float(dict(((s[0], s[1]), s[2]) for s in shells_occ).get((n_o, l_o), 0.0))
            if occ > 0.0:
                rad_u = np.where(r_unif <= rc_o, np.interp(r_unif, r, rad_o, right=0.0), 0.0)
                rho_atm += (occ / (4.0 * np.pi)) * (rad_u**2)
    q_atm = float(np.trapezoid(4.0 * np.pi * (r_unif**2) * rho_atm, r_unif))
    if q_atm > 1e-12:
        rho_na = rho_atm * (z_val / q_atm)
    else:
        rho_na = np.maximum(np.interp(r_unif, r, rho_psml, right=0.0), 0.0)
        q_fb = float(np.trapezoid(4.0 * np.pi * (r_unif**2) * rho_na, r_unif))
        if q_fb > 1e-12:
            rho_na *= z_val / q_fb

    v_loc_u = np.where(r_unif <= r[-1], np.interp(r_unif, r, v_loc_r), -z_val / np.maximum(r_unif, 1e-12))
    # Radial Hartree on uniform grid
    dr_u = np.diff(r_unif)
    f_in = 4.0 * np.pi * (r_unif**2) * rho_na
    f_out = 4.0 * np.pi * r_unif * rho_na
    q_in_u = np.zeros_like(r_unif)
    q_in_u[1:] = np.cumsum(0.5 * (f_in[:-1] + f_in[1:]) * dr_u)
    t_out_u = np.zeros_like(r_unif)
    t_out_u[:-1] = np.cumsum((0.5 * (f_out[:-1] + f_out[1:]) * dr_u)[::-1])[::-1]
    vh_u = np.empty_like(r_unif)
    vh_u[1:] = q_in_u[1:] / r_unif[1:] + t_out_u[1:]
    vh_u[0] = t_out_u[0]
    v_na_u = np.where(r_unif < max_rc, v_loc_u + vh_u, 0.0)
    rho_core_u = np.maximum(np.interp(r_unif, r, rho_core, right=0.0), 0.0)

    max_kb_rc = max(kb_rcut, default=4.0)
    kb_r_u = np.linspace(0.0, max_kb_rc, 201, dtype=np.float64)
    kb_projs_u = [
        np.where(kb_r_u <= rc_k, np.interp(kb_r_u, r, chi_k, right=0.0), 0.0)
        for chi_k, rc_k in zip(kb_projs_r, kb_rcut, strict=True)
    ]
    if not kb_projs_u:
        kb_r_u = np.linspace(0.0, 4.0, 161, dtype=np.float64)
        kb_projs_u = [np.zeros_like(kb_r_u)]
        kb_l = [0]
        kb_j2 = [0]
        kb_energies = [0.0]
        kb_rcut = [1.0]

    sp_data = SpeciesData(
        symbol=spec.symbol,
        atomic_number=int(parsed["atomic_number"]),
        valence_charge=z_val,
        mass_amu=float(parsed["mass"]),
        r_grid=jnp.asarray(r_unif, dtype=jnp.float64),
        v_local=jnp.asarray(v_loc_u, dtype=jnp.float64),
        v_na=jnp.asarray(v_na_u, dtype=jnp.float64),
        rho_na=jnp.asarray(rho_na, dtype=jnp.float64),
        rho_core=jnp.asarray(rho_core_u, dtype=jnp.float64),
        pao_r=jnp.asarray(pao_r, dtype=jnp.float64),
        pao_values=jnp.asarray(np.stack(pao_vals_list, axis=0), dtype=jnp.float64),
        pao_l=tuple(pao_l_list),
        pao_n=tuple(pao_n_list),
        pao_zeta=tuple(pao_zeta_list),
        pao_iz=tuple(pao_iz_list),
        pao_rcut=tuple(pao_rcut_list),
        kb_r=jnp.asarray(kb_r_u, dtype=jnp.float64),
        kb_projectors=jnp.asarray(np.stack(kb_projs_u, axis=0), dtype=jnp.float64),
        kb_l=tuple(kb_l),
        kb_j2=tuple(kb_j2),
        kb_energies=jnp.asarray(kb_energies, dtype=jnp.float64),
        kb_rcut=tuple(kb_rcut),
        has_nlcc=has_nlcc,
        has_spin_orbit=any(j2 > 0 for j2 in kb_j2),
        rcut_na=float(max_rc),
        psml_sha256="",
        tau_core=jnp.zeros_like(jnp.asarray(r_unif, dtype=jnp.float64)),
        has_tau_core=False,
    )
    diagnostics = {
        "free_energies_ha": free_energies,
        "shells": diag_shells,
        "ionic_charge": spec.ionic_charge,
        "kb_energies_ha": kb_energies,
        "kb_rcut_bohr": kb_rcut,
    }
    return sp_data, diagnostics


__all__ = [
    "ShellBasisSpec",
    "SpeciesBasisSpec",
    "default_basis_spec",
    "generate_species_from_psml",
]
