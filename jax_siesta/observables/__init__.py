r"""Post-SCF ``DM_out`` evaluation, total energies, and fixed-(D,E) variational forces/stress.

Theory:
    Enforces the explicit post-SCF ``DM_out`` contract and computes atomic forces
    :math:`\mathbf{F}_I = -\partial E_{\text{KS}}/\partial \mathbf{R}_I` and stress tensor
    :math:`\sigma_{\alpha\beta} = \Omega^{-1} \partial E_{\text{KS}}/\partial \epsilon_{\alpha\beta}`
    at fixed converged :math:`(D, E)` via variational stationarity, including
    Pulay terms (:math:`-\mathrm{Tr}[E\,\partial S/\partial \mathbf{R}_I]`) and discrete
    ``dfscf`` orbital-grid eggbox derivatives.

Algorithm:
    Executes one fresh non-mixed density/potential pass from ``DM_out`` and
    accumulates analytical two-center, reciprocal Hartree, XC, dispersion, and
    custom-VJP grid force/stress terms.

Invariants:
    - ``dm_out_recomputed == True`` for converged runs; no silent fallback to
      in-flight ``last_scf_free_e``.
    - Acoustic sum rule :math:`\sum_I \mathbf{F}_I = \mathbf{0}`.

SIESTA Mapping:
    Replaces ``post_scf_work.F``, ``compute_energies.F90``, ``dfscf.F``, and ``siesta_forces.F``.
"""

from jax_siesta.observables.bands import band_energies
from jax_siesta.observables.forces import StaticObservableEvaluator

__all__ = ["StaticObservableEvaluator", "band_energies"]
