"""PSML-only pseudopotential parser and confined NAO radial basis builders.

Theory:
    Parses norm-conserving ``.psml`` v1.1 XML files (semilocal potentials,
    Kleinman-Bylander projectors including scalar-relativistic and :math:`j`-dependent
    spin-orbit channels, pseudo-core charge for NLCC, and valence configuration)
    and builds confined Numerical Atomic Orbitals (NAOs) and neutral-atom screening
    potentials :math:`V^{\\text{NA}}(r)`.

Algorithm:
    Uses ``xml.etree`` for strict PSML schema parsing (explicitly
    rejecting legacy ``.psf`` files) and constructs cubic radial spline tables
    plus ``.ion.json`` table I/O for 1-to-1 SIESTA comparisons.

Invariants:
    - Legacy ``.psf`` files raise ``ValueError`` explicitly.
    - Sequential ``pao_iz`` orbital counters prevent semicore/valence collisions.

SIESTA Mapping:
    Replaces ``atom.F``, ``basis_specs.f``, ``psml_wrappers``, and ``atmfuncs.F``.
"""

from jax_siesta.species.basis_opt import (
    EV_PER_BOHR3_PER_GPA,
    BasisEnthalpyObjective,
    BasisOptimizationResult,
    BasisParamSpec,
    BasisTrainingStructure,
    MappedBasisSpace,
    ShellOptConfig,
    basis_enthalpy_ev,
    basis_volume_bohr3,
    default_opt_space_from_psml,
    format_pao_basis_block,
    optimize_basis_enthalpy,
    to_optimizer_u,
    to_physical_x,
)
from jax_siesta.species.pao_generator import (
    ShellBasisSpec,
    SpeciesBasisSpec,
    default_basis_spec,
    generate_species_from_psml,
)
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern

__all__ = [
    "EV_PER_BOHR3_PER_GPA",
    "BasisEnthalpyObjective",
    "BasisOptimizationResult",
    "BasisParamSpec",
    "BasisTrainingStructure",
    "MappedBasisSpace",
    "PSMLBasisLoader",
    "ShellBasisSpec",
    "ShellOptConfig",
    "SpeciesBasisSpec",
    "basis_enthalpy_ev",
    "basis_volume_bohr3",
    "build_sparse_pattern",
    "default_basis_spec",
    "default_opt_space_from_psml",
    "format_pao_basis_block",
    "generate_species_from_psml",
    "optimize_basis_enthalpy",
    "to_optimizer_u",
    "to_physical_x",
]
