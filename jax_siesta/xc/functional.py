r"""Exchange-correlation evaluation via C++ XLA FFI (``libxc`` or analytical C++ LDA/PBE) + JAX FFTs.

Theory:
    Evaluates LDA (Slater + PZ81 / PW92) and GGA (PBE, PBEsol) exchange-correlation
    energies, grid potentials :math:`V_{\text{xc}}(\mathbf{r})`, and stress tensors.
    Density gradients :math:`\nabla\rho_\sigma` are computed on the uniform 3D mesh via
    exact Fourier differentiation in ``jax.numpy.fft``, and pointwise energy densities
    and derivatives are evaluated in compiled C++ via ``jsiesta_xc_eval_f64``.

    For ``nspin in (4, 8)``, non-collinear magnetization
    :math:`\mathbf{m}(\mathbf{r}) = (2\Re\rho_{\uparrow\downarrow}, -2\Im\rho_{\uparrow\downarrow}, \rho_{\uparrow\uparrow}-\rho_{\downarrow\downarrow})`
    is locally diagonalized into :math:`n_{\uparrow,\downarrow} = \frac{1}{2}(n \pm |\mathbf{m}|)`.

Algorithm:
    Wraps ``jsiesta_xc_eval_f64`` in ``@jax.custom_vjp`` so exact C++ dual-number
    derivatives flow seamlessly through ``jax.grad`` for both :math:`V_{\text{xc}} = \delta E_{\text{xc}}/\delta\rho`
    and analytical GGA stress tensors.

Invariants:
    - Functional derivative :math:`V_{\text{xc}}` matches finite perturbations of :math:`\rho(\mathbf{r})`.
    - Zero host-device synchronization inside SCF.

SIESTA Mapping:
    Implements ``ExchangeCorrelation`` replacing ``libgridxc`` and ``cellxc.F``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import jax
import jax.numpy as jnp

try:
    from jax import ffi as _jax_ffi  # type: ignore[attr-defined]
except ImportError:
    from jax.extend import ffi as _jax_ffi  # type: ignore[no-redef]

from jax_siesta.core.ffi import ffi_backend_info, register_grid_xc_target

XC_IDS: dict[str, int] = {
    "LDA": 1,
    "LDA_PZ": 1,
    "PZ": 1,
    "LDA_PW92": 2,
    "PW92": 2,
    "PBE": 10,
    "GGA_PBE": 10,
    "PBESOL": 11,
    "GGA_PBESOL": 11,
    "R2SCAN01": 20,
    "MGGA_R2SCAN01": 20,
    "R2SCAN": 21,
    "MGGA_R2SCAN": 21,
    "SCAN": 22,
    "MGGA_SCAN": 22,
    "TPSS": 23,
    "MGGA_TPSS": 23,
    "PBE_R2SCAN01": 24,
}

# Built-in named mGGA pairs -> (libxc x_id, libxc c_id)
_MGGA_LIBXC_PAIRS: dict[int, tuple[int, int]] = {
    20: (645, 642),  # r2SCAN01 exchange + correlation
    21: (497, 498),  # r2SCAN exchange + correlation
    22: (263, 267),  # SCAN exchange + correlation
    23: (202, 231),  # TPSS exchange + correlation
    24: (101, 642),  # PBE exchange + r2SCAN01 correlation
}


_LIBXC_FAMILY_MGGA = 4  # semilocal XC_FAMILY_MGGA (LDA = 1, GGA = 2); hybrids are reported by their semilocal family

# Named hybrids -> Libxc id of the complete functional (semilocal part plus exact-exchange parameters).
_HYBRID_LIBXC_IDS: dict[str, int] = {
    "HSE06": 428,
    "PBE0": 406,
    "PBEH": 406,
    "B3LYP": 402,
}


@dataclass(frozen=True)
class ExactExchangeSpec:
    r"""Exact-exchange part of an exchange-correlation functional.

    Theory:
        A hybrid is :math:`E_{xc} = E^{sl}_{xc} + \alpha E_x^{HF}[1/r] + \beta E_x^{HF}[\mathrm{erfc}(\omega r)/r]`
        with :math:`E^{sl}_{xc}` the semilocal remainder that Libxc evaluates for the hybrid id. This is the
        Libxc convention, in which :math:`\alpha` is the fraction of exact exchange at long range and
        :math:`\alpha + \beta` the fraction at short range.

    Algorithm:
        Plain record; built by ``exact_exchange_spec``.

    Parameters:
        alpha: Fraction of full-range exact exchange.
        beta: Fraction of short-range exact exchange.
        omega: Range-separation parameter in Bohr^-1.

    Invariants:
        ``is_hybrid`` is false exactly when ``alpha`` and ``beta`` are both zero.

    SIESTA Mapping:
        None in SIESTA. HONPAS hard-codes HSE06 (``beta = 0.25``, ``omega = 0.11``).
    """

    alpha: float = 0.0
    beta: float = 0.0
    omega: float = 0.0

    @property
    def is_hybrid(self) -> bool:
        """True when the functional contains exact exchange."""
        return self.alpha != 0.0 or self.beta != 0.0


def _libxc_ids_need_tau(ids: tuple[int, int], name: str) -> bool:
    """Classify Libxc ids via the FFI (semilocal family); reject VV10 functionals, which belong to ``dispersion``."""
    from jax_siesta.core.ffi import load_ffi_library

    lib = load_ffi_library()
    query = getattr(lib, "jsiesta_libxc_info", None)
    if query is None:  # older library build: keep the previous (meta-GGA) assumption
        return True
    is_mgga = False
    for fid in ids:
        if fid <= 0:
            continue
        fam, hyb, vv10 = ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0)
        rc = query(ctypes.c_int32(fid), ctypes.byref(fam), ctypes.byref(hyb), ctypes.byref(vv10))
        if rc == 1:
            raise RuntimeError(f"{name!r} needs Libxc, but libjsiesta_ffi was built without it.")
        if rc != 0:
            raise ValueError(f"Unknown Libxc functional id {fid} in {name!r}.")
        if vv10.value:
            raise ValueError(f"Libxc id {fid} in {name!r} contains VV10; use dispersion='RVV10' instead.")
        is_mgga = is_mgga or fam.value == _LIBXC_FAMILY_MGGA
    return is_mgga


def _libxc_spec_ids(name: str) -> tuple[int, int] | None:
    """Libxc ids of a functional name that is evaluated through Libxc ids, else ``None`` (built-in names)."""
    key = name.strip().upper().replace("-", "_")
    if key in _HYBRID_LIBXC_IDS:
        return _HYBRID_LIBXC_IDS[key], 0
    if key.startswith("LIBXC_"):
        parts = [int(p) for p in key[len("LIBXC_") :].split("+") if p]
        if len(parts) not in (1, 2):
            raise ValueError(f"Malformed Libxc spec {name!r}; expected 'LIBXC-<x>' or 'LIBXC-<x>+<c>'.")
        return parts[0], parts[1] if len(parts) == 2 else 0
    return None


def exact_exchange_spec(name: str) -> ExactExchangeSpec:
    r"""Exact-exchange parameters ``(alpha, beta, omega)`` of a functional name.

    Theory:
        See ``ExactExchangeSpec``. PBE0 gives ``alpha = 0.25``; HSE06 gives ``beta = 0.25`` and
        ``omega = 0.11`` Bohr^-1; a semilocal functional gives zeros.

    Algorithm:
        Built-in names have no exact exchange. For ``HSE06``, ``PBE0``, ``B3LYP`` and ``LIBXC-<ids>`` the
        parameters are read from Libxc (``jsiesta_libxc_hybrid_coef``) and summed over the ids.

    Parameters:
        name: Functional name, as accepted by ``resolve_xc_spec``.

    Returns:
        The exact-exchange parameters of the functional.

    Invariants:
        Raises ``ValueError`` for a hybrid with a Yukawa kernel and for an unknown id. Independent of the
        Libxc version (5.1 to 7.x).

    SIESTA Mapping:
        None in SIESTA; HONPAS reads ``XC.functional HSE06``.
    """
    ids = _libxc_spec_ids(name)
    if ids is None:
        resolve_xc_spec(name)  # validates the name
        return ExactExchangeSpec()
    from jax_siesta.core.ffi import load_ffi_library

    lib = load_ffi_library()
    query = getattr(lib, "jsiesta_libxc_hybrid_coef", None)
    if query is None:
        raise RuntimeError("libjsiesta_ffi is too old for hybrid functionals; rebuild it (make ffi).")
    alpha = beta = omega = 0.0
    for fid in ids:
        if fid <= 0:
            continue
        a, b, w = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
        rc = query(ctypes.c_int32(fid), ctypes.byref(a), ctypes.byref(b), ctypes.byref(w))
        if rc == 1:
            raise RuntimeError(f"{name!r} needs Libxc, but libjsiesta_ffi was built without it.")
        if rc == 2:
            raise ValueError(f"Unknown Libxc functional id {fid} in {name!r}.")
        if rc == 3:
            raise ValueError(f"Libxc id {fid} in {name!r} is a Yukawa-screened hybrid, which is not supported.")
        if b.value != 0.0 and beta != 0.0 and w.value != omega:
            raise ValueError(f"{name!r} combines two range-separation parameters.")
        alpha += a.value
        beta += b.value
        omega = w.value if b.value != 0.0 else omega
    return ExactExchangeSpec(alpha=alpha, beta=beta, omega=omega)


def libxc_version() -> tuple[int, int, int] | None:
    """Version ``(major, minor, micro)`` of the Libxc library the kernels are linked against, or ``None``.

    Theory:
        Functional ids, default parameters and low-density numerics differ between Libxc releases, so results
        are reported with the version that produced them.

    Algorithm:
        Calls ``jsiesta_libxc_version`` (``xc_version``).

    Returns:
        ``(major, minor, micro)``, or ``None`` when the library was built without Libxc.

    Invariants:
        Supported releases are 5.1 to 7.x; the tests run against 6.2.2, 7.0.0 and 7.1.2.

    SIESTA Mapping:
        The Libxc version line of SIESTA's output header.
    """
    from jax_siesta.core.ffi import load_ffi_library

    query = getattr(load_ffi_library(), "jsiesta_libxc_version", None)
    if query is None:
        return None
    a, b, c = ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0)
    if query(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c)) != 0:
        return None
    return a.value, b.value, c.value


def resolve_xc_spec(name: str) -> tuple[bool, int, tuple[int, int]]:
    """Resolve functional string into ``(is_mgga, legacy_xc_id, (libxc_x_id, libxc_c_id))``.

    Theory:
        Normalizes user-facing LDA, GGA, and meta-GGA functional identifiers into either
        built-in kernel IDs or explicit Libxc functional ID pairs.

    Algorithm:
        Checks named functional tables (``XC_IDS``, ``_MGGA_LIBXC_PAIRS``, the named hybrids) and parses
        explicit ``LIBXC-<x>+<c>`` or ``LIBXC-<id>`` syntax. For a hybrid the ids describe the semilocal
        remainder; ``exact_exchange_spec`` gives the exact-exchange part.

    Parameters:
        name: Functional name string (e.g., ``"PBE"``, ``"R2SCAN01"``, ``"LIBXC-645+642"``).

    Returns:
        Tuple ``(is_mgga, legacy_xc_id, (libxc_x_id, libxc_c_id))``.

    Invariants:
        Raises ``ValueError`` for unknown functional identifiers.

    SIESTA Mapping:
        Mirrors ``XC.functional`` / ``XC.authors`` parsing in SIESTA ``readxcinfo.F``.
    """
    key = name.strip().upper().replace("-", "_")
    if key in XC_IDS:
        xc_id = XC_IDS[key]
        if xc_id in _MGGA_LIBXC_PAIRS:
            return True, xc_id, _MGGA_LIBXC_PAIRS[xc_id]
        return False, xc_id, (0, 0)
    ids = _libxc_spec_ids(name)
    if ids is not None:
        # Validates the ids and rejects VV10 (which would otherwise run silently without non-local correlation).
        # All Libxc specs use the meta-GGA evaluation route, whose C++ kernel dispatches LDA/GGA/MGGA by the
        # semilocal family; tau is simply unused for non-meta-GGA ids. For a hybrid this is the semilocal
        # remainder only: the exact exchange is described by ``exact_exchange_spec`` and added by the SCF.
        _libxc_ids_need_tau(ids, name)
        return True, 20, ids
    raise ValueError(
        f"Unsupported XC functional {name!r}; supported: {sorted(XC_IDS)}, {sorted(_HYBRID_LIBXC_IDS)} "
        "or 'LIBXC-<x>+<c>'"
    )


def resolve_xc_id(name: str) -> int:
    """Map functional name string to compiled C++ FFI functional ID.

    Theory:
        Provides the integer functional selector consumed by LDA/GGA FFI handlers.

    Algorithm:
        Delegates to ``resolve_xc_spec`` and returns its legacy functional ID.

    Parameters:
        name: Functional name string.

    Returns:
        Integer C++ functional ID.

    Invariants:
        Consistent with ``XC_IDS`` and ``resolve_xc_spec``.

    SIESTA Mapping:
        Maps SIESTA functional identifiers to internal kernel switches.
    """
    _, xc_id, _ = resolve_xc_spec(name)
    return xc_id


def _xc_pointwise_with_vjp(
    rho_spin: jax.Array,
    sigma_spin: jax.Array,
    xc_id: int,
    n_spin_pol: int,
) -> jax.Array:
    """Evaluate pointwise XC energy density ``e_xc`` (shape ``(n_grid,)``) with exact C++ VJP."""
    register_grid_xc_target("jsiesta_xc_eval_f64")
    n_grid = int(rho_spin.shape[1])
    ffi_fn = _jax_ffi.ffi_call(
        "jsiesta_xc_eval_f64",
        (
            jax.ShapeDtypeStruct((n_grid,), jnp.float64),
            jax.ShapeDtypeStruct((2, n_grid), jnp.float64),
            jax.ShapeDtypeStruct((3, n_grid), jnp.float64),
        ),
    )

    @jax.custom_vjp
    def _eval(rs: jax.Array, ss: jax.Array) -> jax.Array:
        e_xc, _, _ = ffi_fn(rs, ss, xc_id=int(xc_id), n_spin_pol=int(n_spin_pol))
        return e_xc

    def _fwd(rs: jax.Array, ss: jax.Array):
        e_xc, v_rho, v_sig = ffi_fn(rs, ss, xc_id=int(xc_id), n_spin_pol=int(n_spin_pol))
        return e_xc, (v_rho, v_sig)

    def _bwd(res: tuple[jax.Array, jax.Array], cot: jax.Array) -> tuple[jax.Array, jax.Array]:
        v_rho, v_sig = res
        return v_rho * cot[None, :], v_sig * cot[None, :]

    _eval.defvjp(_fwd, _bwd)
    return _eval(rho_spin, sigma_spin)


def _xc_pointwise_mgga_with_vjp(
    rho_spin: jax.Array,
    sigma_spin: jax.Array,
    tau_spin: jax.Array,
    x_id: int,
    c_id: int,
    n_spin_pol: int,
) -> jax.Array:
    """Evaluate pointwise meta-GGA energy density ``e_xc`` with exact C++/Libxc VJP wrt ``(rho, sigma, tau)``."""
    register_grid_xc_target("jsiesta_xc_eval_mgga_f64")
    n_grid = int(rho_spin.shape[1])
    ffi_fn = _jax_ffi.ffi_call(
        "jsiesta_xc_eval_mgga_f64",
        (
            jax.ShapeDtypeStruct((n_grid,), jnp.float64),
            jax.ShapeDtypeStruct((2, n_grid), jnp.float64),
            jax.ShapeDtypeStruct((3, n_grid), jnp.float64),
            jax.ShapeDtypeStruct((2, n_grid), jnp.float64),
        ),
    )

    @jax.custom_vjp
    def _eval(rs: jax.Array, ss: jax.Array, ts: jax.Array) -> jax.Array:
        e_xc, _, _, _ = ffi_fn(rs, ss, ts, x_id=int(x_id), c_id=int(c_id), n_spin_pol=int(n_spin_pol))
        return e_xc

    def _fwd(rs: jax.Array, ss: jax.Array, ts: jax.Array):
        e_xc, v_rho, v_sig, v_tau = ffi_fn(
            rs, ss, ts, x_id=int(x_id), c_id=int(c_id), n_spin_pol=int(n_spin_pol)
        )
        return e_xc, (v_rho, v_sig, v_tau)

    def _bwd(
        res: tuple[jax.Array, jax.Array, jax.Array], cot: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        v_rho, v_sig, v_tau = res
        return v_rho * cot[None, :], v_sig * cot[None, :], v_tau * cot[None, :]

    _eval.defvjp(_fwd, _bwd)
    return _eval(rho_spin, sigma_spin, tau_spin)


_FD7_WEIGHTS: tuple[tuple[int, float], ...] = (
    (1, 0.75),
    (2, -0.15),
    (3, 1.0 / 60.0),
)


def _fft_gradient_3d(field: jax.Array, g_vecs: jax.Array) -> jax.Array:
    """Compute periodic Fourier gradient ``(3, N1, N2, N3)`` of scalar grid ``field``."""
    n1, n2, n3 = field.shape
    fk = jnp.fft.rfftn(field)
    dfk = 1j * g_vecs * fk[..., None]
    grads = [jnp.fft.irfftn(dfk[..., ax], s=(n1, n2, n3)) for ax in range(3)]
    return jnp.stack(grads, axis=0)


def _fd7_gradient_3d(field: jax.Array, cell: jax.Array) -> jax.Array:
    """Compute SIESTA 6th-order (7-point) local finite-difference gradient ``(3, N1, N2, N3)``."""
    n1, n2, n3 = field.shape
    mesh = (n1, n2, n3)
    # Fractional derivative dF/df_ic = N_ic * dF/di_ic; r = cell @ f => grad_r F = (cell^{-1})^T grad_f F.
    inv_cell = jnp.linalg.inv(cell)
    out = jnp.zeros((3, n1, n2, n3), dtype=jnp.float64)
    for ic in range(3):
        df = jnp.zeros_like(field)
        for m, w in _FD7_WEIGHTS:
            df = df + w * (jnp.roll(field, -m, axis=ic) - jnp.roll(field, m, axis=ic))
        d_scale = float(mesh[ic]) * inv_cell[ic, :]
        out = out + d_scale[:, None, None, None] * df[None, ...]
    return out


def xc_energy_from_grid(
    rho_grid: jax.Array,
    cell: jax.Array,
    nspin: int,
    xc_id: int,
    rho_core_grid: jax.Array | None = None,
    tau_grid: jax.Array | None = None,
    tau_core_grid: jax.Array | None = None,
    xc_pair: tuple[int, int] | None = None,
) -> jax.Array:
    r"""Differentiable total XC energy :math:`E_{\text{xc}}[\rho, \tau, \mathbf{A}]` on the 3D mesh.

    Theory:
        Computes :math:`E_{\text{xc}} = d\Omega \sum_p e_{\text{xc}}(\rho_p, \nabla\rho_p, \tau_p)`
        for LDA, GGA, and meta-GGA functionals (including NLCC :math:`\rho^{\text{core}}` and
        :math:`\tau^{\text{core}}`, and local spin-frame rotation of non-collinear :math:`\tau`).

    Algorithm:
        Rotates spin density and kinetic energy density into the local collinear frame,
        computes real-space gradients using 7-point central finite differences along fractional
        mesh axes, and evaluates pointwise energy densities via XLA FFI.

    Parameters:
        rho_grid: Real-space spin density tensor ``(n_rho_comp, N1, N2, N3)``.
        cell: Lattice vectors as rows ``(3, 3)``.
        nspin: Spin channel mode (1, 2, 4, or 8).
        xc_id: Legacy C++ functional identifier.
        rho_core_grid: Optional NLCC core charge density ``(N1, N2, N3)``.
        tau_grid: Optional orbital kinetic energy density ``(n_tau_comp, N1, N2, N3)``.
        tau_core_grid: Optional NLCC pseudo-core kinetic energy density ``(N1, N2, N3)``.
        xc_pair: Optional Libxc ``(x_id, c_id)`` pair for meta-GGA evaluation.

    Returns:
        Scalar total XC energy in Hartree.

    Invariants:
        SO(3) spin-rotation invariant for ``nspin=4/8`` and gauge-free for semilocal gradients.

    SIESTA Mapping:
        Mirrors ``cellxc.F``, ``ggaxc.F90``, and ``mggaxc.F90`` in ``libgridxc``.
    """
    n1, n2, n3 = int(rho_grid.shape[1]), int(rho_grid.shape[2]), int(rho_grid.shape[3])
    n_grid = n1 * n2 * n3
    vol = jnp.abs(jnp.linalg.det(cell))
    dvol = vol / float(n_grid)

    core_half = 0.5 * rho_core_grid if rho_core_grid is not None else 0.0
    ux = uy = uz = None

    if nspin == 1:
        rho_u = 0.5 * rho_grid[0] + core_half
        rho_d = rho_u
        n_pol = 1
    elif nspin == 2:
        rho_u = rho_grid[0] + core_half
        rho_d = rho_grid[1] + core_half
        n_pol = 2
    else:
        n_tot = rho_grid[0] + rho_grid[1]
        mz = rho_grid[0] - rho_grid[1]
        mx = 2.0 * rho_grid[2]
        my = -2.0 * rho_grid[3]
        m_norm = jnp.sqrt(mx * mx + my * my + mz * mz)
        m_pos = m_norm > 1e-12
        m_denom = jnp.where(m_pos, m_norm, 1.0)
        # Match libgridxc's local-spin-frame projection with fixed local spin axis u
        # (old_scheme = .true. in ggaxc.F90 / mggaxc.F90): avoids 1/|m| singularities
        # and C^0 |m| cusp ringing at spin nodal surfaces.
        ux = jax.lax.stop_gradient(jnp.where(m_pos, mx / m_denom, 0.0))
        uy = jax.lax.stop_gradient(jnp.where(m_pos, my / m_denom, 0.0))
        uz = jax.lax.stop_gradient(jnp.where(m_pos, mz / m_denom, 0.0))
        m_proj = ux * mx + uy * my + uz * mz
        rho_u = 0.5 * (n_tot + m_proj) + core_half
        rho_d = 0.5 * (n_tot - m_proj) + core_half
        n_pol = 2

    is_mgga = (xc_pair is not None and (xc_pair[0] > 0 or xc_pair[1] > 0)) or (
        xc_id in _MGGA_LIBXC_PAIRS
    )
    is_gga = (xc_id >= 10) or is_mgga
    if is_gga:
        if nspin == 1:
            grad_u = _fd7_gradient_3d(rho_u, cell)
            grad_d = grad_u
        elif nspin == 2:
            grad_u = _fd7_gradient_3d(rho_u, cell)
            grad_d = _fd7_gradient_3d(rho_d, cell)
        else:
            assert ux is not None and uy is not None and uz is not None
            g_0 = _fd7_gradient_3d(rho_grid[0] + core_half, cell)
            g_1 = _fd7_gradient_3d(rho_grid[1] + core_half, cell)
            g_2 = _fd7_gradient_3d(rho_grid[2], cell)
            g_3 = _fd7_gradient_3d(rho_grid[3], cell)
            g_tot = g_0 + g_1
            g_mz = g_0 - g_1
            g_mx = 2.0 * g_2
            g_my = -2.0 * g_3
            g_pol = ux[None, ...] * g_mx + uy[None, ...] * g_my + uz[None, ...] * g_mz
            grad_u = 0.5 * (g_tot + g_pol)
            grad_d = 0.5 * (g_tot - g_pol)
        sig_uu = jnp.sum(grad_u * grad_u, axis=0)
        sig_ud = jnp.sum(grad_u * grad_d, axis=0)
        sig_dd = jnp.sum(grad_d * grad_d, axis=0)
    else:
        z = jnp.zeros((n1, n2, n3), dtype=jnp.float64)
        sig_uu = sig_ud = sig_dd = z

    rho_flat = jnp.stack([rho_u.reshape(-1), rho_d.reshape(-1)], axis=0)
    sig_flat = jnp.stack([sig_uu.reshape(-1), sig_ud.reshape(-1), sig_dd.reshape(-1)], axis=0)

    if is_mgga:
        if tau_grid is None:
            raise ValueError("meta-GGA functional requires tau_grid.")
        x_id, c_id = xc_pair if xc_pair is not None else _MGGA_LIBXC_PAIRS[xc_id]
        tcore_half = 0.5 * tau_core_grid if tau_core_grid is not None else 0.0
        if nspin == 1:
            tau_u = 0.5 * tau_grid[0] + tcore_half
            tau_d = tau_u
        elif nspin == 2:
            tau_u = tau_grid[0] + tcore_half
            tau_d = tau_grid[1] + tcore_half
        else:
            assert ux is not None and uy is not None and uz is not None
            tau_tot = tau_grid[0] + tau_grid[1]
            tx = 2.0 * tau_grid[2]
            ty = -2.0 * tau_grid[3]
            tz = tau_grid[0] - tau_grid[1]
            tau_pol = ux * tx + uy * ty + uz * tz
            tau_u = 0.5 * (tau_tot + tau_pol) + tcore_half
            tau_d = 0.5 * (tau_tot - tau_pol) + tcore_half
        tau_flat = jnp.stack([tau_u.reshape(-1), tau_d.reshape(-1)], axis=0)
        e_xc_pts = _xc_pointwise_mgga_with_vjp(
            rho_flat, sig_flat, tau_flat, x_id=x_id, c_id=c_id, n_spin_pol=n_pol
        )
    else:
        e_xc_pts = _xc_pointwise_with_vjp(rho_flat, sig_flat, xc_id=xc_id, n_spin_pol=n_pol)
    return dvol * jnp.sum(e_xc_pts)


class LibxcFFIEvaluator:
    r"""Semilocal XC evaluator implementing ``ExchangeCorrelation`` via C++ XLA FFI.

    Theory:
        Computes :math:`E_{\text{xc}}`, :math:`V_{\text{xc}}(\mathbf{r}) = \delta E_{\text{xc}}/\delta\rho(\mathbf{r})`,
        meta-GGA :math:`v_\tau(\mathbf{r}) = \delta E_{\text{xc}}/\delta\tau(\mathbf{r})`,
        and physical XC stress tensors.

    Algorithm:
        Wraps ``xc_energy_from_grid`` with ``jax.value_and_grad`` with respect to density,
        kinetic energy density, and lattice strain transformations.

    Parameters:
        functional: XC functional specifier (e.g., ``"PBE"``, ``"R2SCAN01"``).

    Invariants:
        Real-space potentials satisfy exact variational derivatives of ``E_xc``; non-collinear
        cross terms are scaled by ``0.5`` to match the Pauli-matrix density convention.

    SIESTA Mapping:
        Replaces SIESTA ``dhscf`` calls to ``cellxc`` for LDA, GGA, and meta-GGA functionals.
    """

    def __init__(self, functional: str = "PBE") -> None:
        self.functional = functional
        self.is_mgga, self.xc_id, self.xc_pair = resolve_xc_spec(functional)
        self.backend_name = "libxc" if ffi_backend_info()["has_libxc"] else "builtin_cpp"

    def evaluate(
        self,
        rho_grid: jax.Array,
        cell: jax.Array,
        nspin: int,
        rho_core_grid: jax.Array | None = None,
        compute_stress: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Evaluate ``(e_xc, v_xc_grid, stress_xc)`` from real-space spin density (LDA/GGA).

        Theory:
            Evaluates LDA/GGA energy :math:`E_{\text{xc}}[\rho]`, functional derivative
            :math:`V_{\text{xc}} = \delta E_{\text{xc}}/\delta\rho`, and metric stress tensor.

        Algorithm:
            Differentiates ``xc_energy_from_grid`` via ``jax.value_and_grad`` wrt ``rho_grid``
            and infinitesimal homogeneous lattice strain.

        Parameters:
            rho_grid: Real-space spin density tensor ``(n_rho_comp, N1, N2, N3)``.
            cell: Lattice vectors as rows ``(3, 3)``.
            nspin: Spin channel mode (1, 2, 4, or 8).
            rho_core_grid: Optional NLCC core charge density ``(N1, N2, N3)``.
            compute_stress: Whether to evaluate the XC virial stress tensor.

        Returns:
            Tuple ``(e_xc, v_xc_grid, stress_xc)``.

        Invariants:
            Rejects meta-GGA functionals unless ``evaluate_mgga`` is called.
            The ``compute_stress=True`` virial is the derivative under a charge-conserving scaling (``rho`` and
            ``rho_core`` divided by ``det(1+eps)``). It is *not* the NLCC stress of an SCF calculation, where the core
            density moves with the atoms; ``StaticObservableEvaluator`` differentiates the grid energy through
            ``neutral_atom_grid_differentiable`` instead (logs/DECISIONS.md D-021).

        SIESTA Mapping:
            Corresponds to LDA/GGA ``cellxc`` evaluation in SIESTA.
        """
        if self.is_mgga:
            raise ValueError(
                f"XC functional {self.functional!r} is meta-GGA and requires evaluate_mgga(..., tau_grid=...)."
            )
        n1, n2, n3 = int(rho_grid.shape[1]), int(rho_grid.shape[2]), int(rho_grid.shape[3])
        vol = jnp.abs(jnp.linalg.det(cell))
        dvol = vol / float(n1 * n2 * n3)

        def _e_fn(r_g: jax.Array) -> jax.Array:
            return xc_energy_from_grid(r_g, cell, nspin, self.xc_id, rho_core_grid)

        e_xc, de_drho = jax.value_and_grad(_e_fn)(rho_grid)
        v_xc = de_drho / dvol
        if nspin in (4, 8):
            v_xc = v_xc.at[2:4].multiply(0.5)

        if not compute_stress:
            return e_xc, v_xc, jnp.zeros((3, 3), dtype=jnp.float64)

        def _strain_energy(eps: jax.Array) -> jax.Array:
            deform = jnp.eye(3) + eps
            det_d = jnp.linalg.det(deform)
            scaled_cell = deform @ cell
            scaled_rho = rho_grid / det_d
            scaled_core = (rho_core_grid / det_d) if rho_core_grid is not None else None
            return xc_energy_from_grid(scaled_rho, scaled_cell, nspin, self.xc_id, scaled_core)

        stress_xc = jax.grad(_strain_energy)(jnp.zeros((3, 3), dtype=jnp.float64)) / vol
        return e_xc, v_xc, stress_xc

    def evaluate_mgga(
        self,
        rho_grid: jax.Array,
        tau_grid: jax.Array,
        cell: jax.Array,
        nspin: int,
        rho_core_grid: jax.Array | None = None,
        tau_core_grid: jax.Array | None = None,
        compute_stress: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Evaluate ``(e_xc, v_xc_grid, v_tau_grid, stress_xc)`` for meta-GGA.

        Theory:
            Evaluates generalized Kohn-Sham meta-GGA functional derivatives
            :math:`V_{\text{xc}} = \delta E_{\text{xc}}/\delta\rho` and
            :math:`v_\tau = \delta E_{\text{xc}}/\delta\tau` plus strain stress tensor.

        Algorithm:
            Differentiates ``xc_energy_from_grid`` wrt ``(rho_grid, tau_grid)`` and metric
            strain preserving integral normalization (:math:`\tau \propto \det(I+\epsilon)^{-5/3}` is handled externally; density scales as :math:`\det(I+\epsilon)^{-1}`).

        Parameters:
            rho_grid: Real-space spin density tensor ``(n_rho_comp, N1, N2, N3)``.
            tau_grid: Real-space kinetic energy density tensor ``(n_tau_comp, N1, N2, N3)``.
            cell: Lattice vectors as rows ``(3, 3)``.
            nspin: Spin channel mode (1, 2, 4, or 8).
            rho_core_grid: Optional NLCC core charge density ``(N1, N2, N3)``.
            tau_core_grid: Optional NLCC core kinetic energy density ``(N1, N2, N3)``.
            compute_stress: Whether to evaluate the XC virial stress tensor.

        Returns:
            Tuple ``(e_xc, v_xc_grid, v_tau_grid, stress_xc)``.

        Invariants:
            Includes exact strain derivatives wrt ``cell``, ``rho``, and ``tau``.
            Like ``evaluate``, the standalone virial scales the core densities by ``1/det(1+eps)``; the SCF stress does not
            (logs/DECISIONS.md D-021).

        SIESTA Mapping:
            Corresponds to meta-GGA ``cellxc`` / ``mggaxc`` evaluation in SIESTA.
        """
        n1, n2, n3 = int(rho_grid.shape[1]), int(rho_grid.shape[2]), int(rho_grid.shape[3])
        vol = jnp.abs(jnp.linalg.det(cell))
        dvol = vol / float(n1 * n2 * n3)

        def _e_fn(r_g: jax.Array, t_g: jax.Array) -> jax.Array:
            return xc_energy_from_grid(
                r_g,
                cell,
                nspin,
                self.xc_id,
                rho_core_grid,
                tau_grid=t_g,
                tau_core_grid=tau_core_grid,
                xc_pair=self.xc_pair if self.is_mgga else None,
            )

        e_xc, (de_drho, de_dtau) = jax.value_and_grad(_e_fn, argnums=(0, 1))(rho_grid, tau_grid)
        v_xc = de_drho / dvol
        v_tau = de_dtau / dvol
        if nspin in (4, 8):
            v_xc = v_xc.at[2:4].multiply(0.5)
            v_tau = v_tau.at[2:4].multiply(0.5)

        if not compute_stress:
            return e_xc, v_xc, v_tau, jnp.zeros((3, 3), dtype=jnp.float64)

        def _strain_energy(eps: jax.Array) -> jax.Array:
            deform = jnp.eye(3) + eps
            det_d = jnp.linalg.det(deform)
            scaled_cell = deform @ cell
            scaled_rho = rho_grid / det_d
            scaled_tau = tau_grid / det_d
            scaled_core = (rho_core_grid / det_d) if rho_core_grid is not None else None
            scaled_tcore = (tau_core_grid / det_d) if tau_core_grid is not None else None
            return xc_energy_from_grid(
                scaled_rho,
                scaled_cell,
                nspin,
                self.xc_id,
                scaled_core,
                tau_grid=scaled_tau,
                tau_core_grid=scaled_tcore,
                xc_pair=self.xc_pair if self.is_mgga else None,
            )

        stress_xc = jax.grad(_strain_energy)(jnp.zeros((3, 3), dtype=jnp.float64)) / vol
        return e_xc, v_xc, v_tau, stress_xc
