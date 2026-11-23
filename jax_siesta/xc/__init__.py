r"""Exchange-correlation (libxc/C++ FFI) and dispersion models (DFT-D3 / rVV10).

Theory:
    Evaluates semilocal LDA/GGA exchange-correlation energy, potential,
    and stress directly via ``libxc`` / C++ analytical XC FFI without ``libgridxc``,
    combined with differentiable JAX DFT-D3 (Becke-Johnson) and 3D-FFT rVV10.

Algorithm:
    Computes grid gradients :math:`\nabla\rho` with ``jax.numpy.fft`` and invokes
    the C++ ``xc_eval`` XLA FFI kernel pointwise, keeping grid arrays resident on device.

Invariants:
    - Functional derivative :math:`V_{\text{xc}}(\mathbf{r}) = \delta E_{\text{xc}}/\delta \rho(\mathbf{r})`
      matches finite perturbations of :math:`\rho(\mathbf{r})`.

SIESTA Mapping:
    Replaces ``libgridxc`` and ``cellxc`` semilocal/vdW drivers.
"""

from jax_siesta.xc.dispersion import DFTD3Dispersion, RVV10Dispersion
from jax_siesta.xc.functional import LibxcFFIEvaluator, resolve_xc_id, xc_energy_from_grid

__all__ = [
    "DFTD3Dispersion",
    "LibxcFFIEvaluator",
    "RVV10Dispersion",
    "resolve_xc_id",
    "xc_energy_from_grid",
]
