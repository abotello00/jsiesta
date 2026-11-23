r"""Physics-informed SCF eigensolver, Tikhonov-Pulay mixers, and adaptive controller.

Theory:
    Implements generalized Kohn-Sham diagonalization (:math:`H(\mathbf{k})C = S(\mathbf{k})C\varepsilon`)
    across ``nspin in {1, 2, 4, 8}``, physical conditioning descriptors
    (:math:`\kappa_\varepsilon`, :math:`d/f` stiffness), Tikhonov-Pulay / Pulay
    mixers with decoupled charge/spin preconditioning, deterministic scalar SVD
    regularization, and closed-loop trust-region adaptive SCF control.

Algorithm:
    Purely functional JAX state machine updating ``SCFState`` without global
    side effects, followed by a mandatory post-SCF ``DM_out`` evaluation.

Invariants:
    - Orthonormality :math:`C^\dagger S C = I` and electron count :math:`\mathrm{Tr}[DS] = N_e`.
    - Deterministic scalar (:math:`1\times 1`) Gram matrix pseudoinverse.
    - Exact target electronic temperature restored strictly prior to convergence.

SIESTA Mapping:
    Replaces ``diagon.F``, ``compute_dm.F``, ``mixer.F``, ``m_mixing.F90``, and ``scf_adaptive.lua``.
"""

from jax_siesta.scf.controller import VALID_OUTCOMES, update_adaptive_state
from jax_siesta.scf.descriptors import (
    ConditioningDescriptors,
    RecommendedSCFParams,
    estimate_conditioning_descriptors,
    recommend_scf_defaults,
)
from jax_siesta.scf.driver import build_h0_sparse, build_monkhorst_pack_mesh, run_scf
from jax_siesta.scf.eigensolver import GeneralizedEigenSolver
from jax_siesta.scf.mixer import TikhonovPulayMixer, deterministic_svd_pinv

__all__ = [
    "VALID_OUTCOMES",
    "ConditioningDescriptors",
    "GeneralizedEigenSolver",
    "RecommendedSCFParams",
    "TikhonovPulayMixer",
    "build_h0_sparse",
    "build_monkhorst_pack_mesh",
    "deterministic_svd_pinv",
    "estimate_conditioning_descriptors",
    "recommend_scf_defaults",
    "run_scf",
    "update_adaptive_state",
]
