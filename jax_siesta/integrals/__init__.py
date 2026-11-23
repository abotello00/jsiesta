r"""Irreducible two-center integral evaluation (:math:`S, T, V^{\text{NA}}, V^{\text{KB}}`).

Theory:
    Evaluates overlap, kinetic energy, neutral-atom screening potential, and
    Kleinman-Bylander nonlocal projector matrices alongside exact Cartesian
    derivatives via spherical Bessel transforms and Gaunt coefficients.

Algorithm:
    Calls compiled C++ two-center spline/Gaunt kernels on symmetry-closed
    neighbor lists (evaluated on space-group orbit representatives when symmetry
    is enabled and rotated to partners via Wigner-D matrices).

Invariants:
    - Matrix elements are Hermitian across supercell pair reversals.
    - Analytical derivatives match central finite differences.

SIESTA Mapping:
    Replaces ``matel.F90``, ``kinefsm.F``, ``naefs.F``, and ``nlefsm.f``.
"""

from jax_siesta.integrals.two_center import TwoCenterEvaluator

__all__ = ["TwoCenterEvaluator"]
