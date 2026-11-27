"""Hybrid functionals: Libxc description of the functional (Stage 2 of ``docs/design/HYBRID_FUNCTIONALS_PLAN.md``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core.ffi import ffi_backend_info
from jax_siesta.xc.functional import exact_exchange_spec, libxc_version, resolve_xc_spec, xc_energy_from_grid

jax.config.update("jax_enable_x64", True)

needs_libxc = pytest.mark.skipif(not ffi_backend_info().get("has_libxc", False), reason="library built without Libxc")


@needs_libxc
def test_exact_exchange_parameters_from_libxc() -> None:
    """Named hybrids carry the published exact-exchange parameters; semilocal functionals carry none.

    The reference values are the definitions of the functionals (PBE0: 1/4 full range; HSE06: 1/4 short range with
    omega = 0.11 Bohr^-1; B3LYP: 0.2 full range), independent of jsiesta and of the Libxc version in use.
    """
    version = libxc_version()
    assert version is not None and version[0] in (5, 6, 7)
    pbe0, hse, b3lyp = exact_exchange_spec("PBE0"), exact_exchange_spec("HSE06"), exact_exchange_spec("B3LYP")
    assert (pbe0.alpha, pbe0.beta) == (0.25, 0.0)
    assert (hse.alpha, hse.beta, hse.omega) == (0.0, 0.25, 0.11)
    assert (b3lyp.alpha, b3lyp.beta) == (0.2, 0.0)
    assert exact_exchange_spec("LIBXC-428") == hse
    for name in ("PBE", "LDA", "R2SCAN", "LIBXC-101+130"):
        assert not exact_exchange_spec(name).is_hybrid
    # CAM-B3LYP: 0.19 at short range, 0.65 at long range, omega = 0.33.
    cam = exact_exchange_spec("LIBXC-433")
    assert cam.alpha == pytest.approx(0.65) and cam.alpha + cam.beta == pytest.approx(0.19) and cam.omega == 0.33
    assert resolve_xc_spec("HSE06") == (True, 20, (428, 0))


def _density(nspin: int) -> tuple[jax.Array, jax.Array]:
    n_pts = 12
    cell = jnp.eye(3) * 7.0
    ax = jnp.arange(n_pts) / n_pts
    x, y, z = jnp.meshgrid(ax, ax, ax, indexing="ij")
    rho = 0.3 * (1.2 + 0.4 * jnp.cos(2 * np.pi * z) + 0.3 * jnp.sin(2 * np.pi * x) * jnp.cos(2 * np.pi * y))
    if nspin == 1:
        return rho[None], cell
    pol = 0.25 * jnp.cos(2 * np.pi * (x + y))
    return jnp.stack([0.5 * rho * (1 + pol), 0.5 * rho * (1 - pol)]), cell


@needs_libxc
@pytest.mark.parametrize("nspin", [1, 2])
def test_semilocal_part_of_pbe0_is_three_quarters_pbe_exchange_plus_pbe_correlation(nspin: int) -> None:
    """The semilocal remainder evaluated for the hybrid id equals the sum of its published parts.

    Independent identity: PBE0 = 1/4 exact exchange + 3/4 PBE exchange + PBE correlation, so the Libxc hybrid id 406
    evaluated through the jsiesta route must equal 0.75 E[GGA_X_PBE] + E[GGA_C_PBE] from the separate ids. This checks
    that a hybrid family is sent to the GGA kernel (it was sent to the LDA kernel before this stage).
    """
    rho, cell = _density(nspin)
    tau = jnp.zeros_like(rho)

    def energy(pair: tuple[int, int]) -> float:
        return float(xc_energy_from_grid(rho, cell, nspin, 20, tau_grid=tau, xc_pair=pair))

    e_hyb = energy((406, 0))
    e_parts = 0.75 * energy((101, 0)) + energy((130, 0))
    assert abs(e_hyb - e_parts) <= 1e-12 * abs(e_parts)
    # and the potential of the remainder is the derivative of its energy
    grad = jax.grad(lambda r: xc_energy_from_grid(r, cell, nspin, 20, tau_grid=tau, xc_pair=(406, 0)))(rho)
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(rho.shape)) * 1e-2
    step = 1e-4
    fd = (
        float(xc_energy_from_grid(rho + step * direction, cell, nspin, 20, tau_grid=tau, xc_pair=(406, 0)))
        - float(xc_energy_from_grid(rho - step * direction, cell, nspin, 20, tau_grid=tau, xc_pair=(406, 0)))
    ) / (2 * step)
    assert abs(float(jnp.sum(grad * direction)) - fd) <= 1e-7 * abs(fd)
