"""Core immutable PyTree data contracts and extension protocols for jsiesta.

Theory:
    Defines the canonical state containers (crystal structure, PSML species,
    NAO radial basis, static sparse orbital-pair pattern, Brillouin-zone mesh,
    SCF state, and provenanced observables) and the Python ``Protocol`` seams
    decoupling physical subsystems.

Algorithm:
    Every array-carrying container registers with ``jax.tree_util`` so JAX
    transformations (``jit``, ``vmap``, ``grad``) trace numeric leaves while
    treating topological indices and configuration flags as static auxiliary data.

Invariants:
    - All dataclasses are frozen (immutable).
    - Coordinate, lattice, and Hamiltonian units are strictly atomic units
      (Bohr and Hartree) unless explicitly suffixed otherwise.

SIESTA Mapping:
    Consolidates legacy Fortran module globals from ``atm_types``, ``atomlist``,
    ``sparse_matrices``, ``m_spin``, and ``m_energies`` into explicit immutable
    PyTrees.
"""

from jax_siesta.core.protocols import (
    BasisAndPseudos,
    DispersionModel,
    EigenSolver,
    ExchangeCorrelation,
    ObservableEvaluator,
    RealSpaceGridOps,
    SCFMixer,
    SymmetryProjector,
    TwoCenterIntegrals,
)
from jax_siesta.core.types import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    HARTREE_TO_RY,
    KB_HARTREE_PER_KELVIN,
    RY_TO_HARTREE,
    BasisData,
    KPointMesh,
    Observables,
    SCFConfig,
    SCFState,
    SparsePattern,
    SpeciesData,
    Structure,
    TwoCenterMatrices,
)

__all__ = [
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "EV_TO_HARTREE",
    "HARTREE_TO_EV",
    "HARTREE_TO_RY",
    "KB_HARTREE_PER_KELVIN",
    "RY_TO_HARTREE",
    "BasisAndPseudos",
    "BasisData",
    "DispersionModel",
    "EigenSolver",
    "ExchangeCorrelation",
    "KPointMesh",
    "ObservableEvaluator",
    "Observables",
    "RealSpaceGridOps",
    "SCFConfig",
    "SCFMixer",
    "SCFState",
    "SparsePattern",
    "SpeciesData",
    "Structure",
    "SymmetryProjector",
    "TwoCenterIntegrals",
    "TwoCenterMatrices",
]
