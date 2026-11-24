"""Clean-room JAX + C++ static SIESTA Kohn-Sham DFT engine.

Theory:
    Implements Kohn-Sham density functional theory (KS-DFT) using confined
    Numerical Atomic Orbitals (NAOs), norm-conserving PSML pseudopotentials
    in Kleinman-Bylander (KB) separable form, neutral-atom (:math:`V^{\\text{NA}}`)
    electrostatic screening cancellation, and a uniform 3D periodic real-space
    mesh for Hartree, exchange-correlation, and grid dispersion terms.

Algorithm:
    Organizes the static single-point SCF calculation into pure, protocol-based
    subsystems over immutable JAX PyTrees and compiled C++/CUDA XLA FFI kernels.

Invariants:
    - All public state is represented explicitly via immutable PyTrees; zero
      module-level mutable simulation globals exist.
    - Internal atomic units throughout: lengths in Bohr (:math:`a_0`),
      energies in Hartree (:math:`E_h`, with explicit eV / Ry conversions at I/O).

SIESTA Mapping:
    Replaces legacy global-state Fortran SIESTA modules (``m_siesta_init``,
    ``m_siesta_forces``, ``dhscf``, ``diagon``) with functional PyTree pipelines.
"""

import jax as _jax

if hasattr(_jax, "config"):
    _jax.config.update("jax_enable_x64", True)

from jax_siesta.core.compilation import enable_compilation_cache
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
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_RY,
    BasisData,
    KPointMesh,
    Observables,
    SCFConfig,
    SCFState,
    SparsePattern,
    SpeciesData,
    Structure,
)

__version__ = "0.1.0"

enable_compilation_cache()  # no-op unless JSIESTA_COMPILATION_CACHE names a directory

# High-level interface (imported last: it uses the stage modules above).
from jax_siesta.app.calculation import BandStructure, Calculation, Results, make_structure  # noqa: E402

__all__ = [
    "BOHR_TO_ANGSTROM",
    "BandStructure",
    "Calculation",
    "Results",
    "make_structure",
    "HARTREE_TO_EV",
    "HARTREE_TO_RY",
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
]
