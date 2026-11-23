r"""SIESTA parity adapters and independent physical-identity verification helpers.

Theory:
    Provides head-to-head comparison harnesses against ``validation.py`` manifests,
    central finite-difference derivative checkers for forces and stress tensors,
    and independent physical conservation/commutation verifiers.

Algorithm:
    Wraps ``validation.py`` and executes isolated case checks asserting
    Hermiticity, electron-count conservation, adjoint identities, symmetry
    commutation, and derivative consistency.

Invariants:
    - Physical-contract checks exit non-zero on bound violations per ``docs/design/EVIDENCE.md``.

SIESTA Mapping:
    Interfaces directly with ``validation.py`` and ``docs/design/EVIDENCE.md`` contracts.
"""

from jax_siesta.testing.run_case import run_single_point

__all__ = ["run_single_point"]
