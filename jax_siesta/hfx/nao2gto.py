r"""Gaussian fits of numerical atomic orbitals (NAO2GTO).

Theory:
    The four-centre integrals of the exact exchange are analytic for Gaussians. Each numerical radial function
    :math:`R(r)` (zero beyond its cut-off radius) is therefore replaced, in the exchange term only, by
    :math:`\tilde R(r) = r^l \sum_i c_i e^{-a_i r^2}`, so that the orbital is
    :math:`C_{lm}(\mathbf{r}) \sum_i c_i e^{-a_i r^2}` with the real solid harmonics :math:`C_{lm} = r^l Y_{lm}`.

Algorithm:
    Variable projection. For given exponents the coefficients are the linear least-squares solution in the norm
    :math:`\int_0^{2 r_c} r^2 (R - \tilde R)^2 dr` (the target is zero between :math:`r_c` and :math:`2 r_c`, which
    penalises tails); the exponents are optimised with Nelder-Mead from several even-tempered starts, in a
    parametrisation that keeps them above a floor and apart by a minimum ratio (near-equal exponents give large
    coefficients of opposite sign and lose digits in the integrals).

Invariants:
    Exponents are increasing, at least ``a_min`` and separated by at least ``ratio_min``. The residual is the
    norm of the difference; the numerical orbital has norm one.

SIESTA Mapping:
    None in SIESTA. HONPAS: the NAO2GTO scheme of Shang, Li and Yang, J. Chem. Phys. 135, 034110 (2011).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from jax_siesta.core.types import SpeciesData

__all__ = ["GaussianShell", "fit_radial_function", "fit_species"]


@dataclass(frozen=True)
class GaussianShell:
    r"""Gaussian fit :math:`r^l \sum_i c_i e^{-a_i r^2}` of one numerical radial function.

    Theory:
        See the module description.

    Algorithm:
        Plain record produced by ``fit_radial_function``.

    Parameters:
        l: Angular momentum.
        exponents: Exponents :math:`a_i` in Bohr^-2, increasing.
        coefficients: Coefficients :math:`c_i` of :math:`r^l e^{-a_i r^2}`.
        rcut: Cut-off radius of the numerical function in Bohr.
        residual: Norm of the difference to the numerical function (which has norm one).

    Invariants:
        ``len(exponents) == len(coefficients)``.

    SIESTA Mapping:
        None in SIESTA; a contracted Gaussian shell of HONPAS.
    """

    l: int  # noqa: E741
    exponents: np.ndarray
    coefficients: np.ndarray
    rcut: float
    residual: float


def fit_radial_function(
    r: np.ndarray,
    values: np.ndarray,
    l: int,  # noqa: E741
    rcut: float,
    n_gauss: int = 5,
    a_min: float = 0.05,
    ratio_min: float = 1.5,
    a_max: float = 200.0,
) -> GaussianShell:
    r"""Fit a confined radial function with ``n_gauss`` Gaussians.

    Theory:
        Least squares in the norm of the orbital, :math:`\int r^2 (R - \tilde R)^2 dr`, over :math:`[0, 2 r_c]`.

    Algorithm:
        See the module description. The exponents are :math:`\log a_1 = \log a_{min} + e^{x_1}`,
        :math:`\log a_{i+1} = \log a_i + \log q_{min} + e^{x_{i+1}}`.

    Parameters:
        r: Radial grid in Bohr, starting at zero.
        values: Radial function :math:`R(r)` on the grid (normalised, :math:`\int r^2 R^2 dr = 1`).
        l: Angular momentum.
        rcut: Cut-off radius in Bohr.
        n_gauss: Number of Gaussians.
        a_min: Smallest exponent allowed, Bohr^-2.
        ratio_min: Smallest ratio of consecutive exponents.
        a_max: Largest exponent allowed, Bohr^-2.

    Returns:
        The fitted shell with its residual.

    Invariants:
        Deterministic for given arguments.

    SIESTA Mapping:
        None in SIESTA; HONPAS ``nao2gto`` fitting.
    """
    rr = np.linspace(0.0, 2.0 * rcut, 1601)
    target = np.where(rr <= rcut, np.interp(rr, r, values, right=0.0), 0.0)
    weight = rr**2 * (rr[1] - rr[0])
    weight[-1] *= 0.5
    sqrt_w = np.sqrt(weight)
    rl = rr**l
    r2 = rr**2

    def exponents_of(x: np.ndarray) -> np.ndarray:
        log_a = np.empty(n_gauss)
        log_a[0] = np.log(a_min) + np.exp(x[0])
        for i in range(1, n_gauss):
            log_a[i] = log_a[i - 1] + np.log(ratio_min) + np.exp(x[i])
        return np.exp(log_a)

    def solve(a: np.ndarray) -> tuple[np.ndarray, float]:
        basis = rl[:, None] * np.exp(-a[None, :] * r2[:, None])
        coef, *_ = np.linalg.lstsq(basis * sqrt_w[:, None], target * sqrt_w, rcond=None)
        res = basis @ coef - target
        return coef, float(np.sum(weight * res * res))

    def objective(x: np.ndarray) -> float:
        a = exponents_of(np.clip(x, -30.0, 5.0))
        if a[-1] > a_max:
            return 1.0 + float(a[-1])
        return solve(a)[1]

    best_x, best_f = None, np.inf
    for a0, q in ((2.4 / rcut**2, 2.2), (0.08, 2.5), (0.15, 3.0), (0.1, 1.9)):
        x0 = np.empty(n_gauss)
        x0[0] = np.log(max(np.log(max(a0, a_min * 1.001)) - np.log(a_min), 1e-3))
        x0[1:] = np.log(np.log(q) - np.log(ratio_min))
        opt = minimize(objective, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 6000, "maxfev": 6000})
        if opt.fun < best_f:
            best_x, best_f = opt.x, float(opt.fun)
    assert best_x is not None
    a = exponents_of(np.clip(best_x, -30.0, 5.0))
    coef, err = solve(a)
    return GaussianShell(l=int(l), exponents=a, coefficients=coef, rcut=float(rcut), residual=float(np.sqrt(err)))


_FIT_CACHE: dict[str, tuple[GaussianShell, ...]] = {}


def fit_species(species: SpeciesData, n_gauss: int = 5) -> tuple[GaussianShell, ...]:
    """Gaussian fits of all orbital shells of a species, in the order of ``species.pao_l``.

    Theory:
        See the module description.

    Algorithm:
        ``fit_radial_function`` per shell; the result is cached on the content of the radial tables.

    Parameters:
        species: Species with its numerical orbitals.
        n_gauss: Number of Gaussians per shell.

    Returns:
        One ``GaussianShell`` per shell.

    Invariants:
        Two species with identical radial tables share the cached fit.

    SIESTA Mapping:
        None in SIESTA; HONPAS fits each species once at start-up.
    """
    r = np.asarray(species.pao_r, dtype=np.float64)
    vals = np.asarray(species.pao_values, dtype=np.float64)
    digest = hashlib.sha256()
    for part in (r.tobytes(), vals.tobytes(), repr((tuple(species.pao_l), tuple(species.pao_rcut), n_gauss)).encode()):
        digest.update(part)
    key = digest.hexdigest()
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = tuple(
            fit_radial_function(r, vals[i], int(species.pao_l[i]), float(species.pao_rcut[i]), n_gauss=n_gauss)
            for i in range(vals.shape[0])
        )
    return _FIT_CACHE[key]
