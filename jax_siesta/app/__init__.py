r"""User interface: SIESTA-style runs from ``.fdf`` files and a compact Python API.

Theory:
    The physics modules of jsiesta are separate stages (species, integrals, grid, SCF, observables) that a
    developer composes by hand. This package composes them for the common case, a single-point calculation, in
    the two forms users know: an input file given to a command, as with SIESTA, and a few Python objects for
    scripts and notebooks.

Algorithm:
    ``Calculation`` (inputs) -> ``run()`` -> ``Results`` (outputs in eV and Angstrom). ``jax_siesta.app.cli`` is
    the ``jsiesta`` command; ``report`` formats the log; ``outputs`` writes SIESTA-format files; ``basis`` finds
    pseudopotentials and builds basis sets.

Invariants:
    - Nothing here adds physics: every number comes from the stage modules, which remain usable directly.
    - Input that is not understood is reported, never ignored silently.

SIESTA Mapping:
    ``siesta.F`` (program flow), ``siesta_init``, ``siesta_analysis`` and the output routines.
"""

from jax_siesta.app.basis import (
    atomic_xc_name,
    build_species,
    describe_species,
    find_pseudopotential,
    parse_pao_basis_block,
)
from jax_siesta.app.calculation import BandStructure, Calculation, Results, make_structure
from jax_siesta.app.outputs import band_lines_path, write_bands, write_eigenvalues, write_forces, write_outputs

__all__ = [
    "BandStructure",
    "Calculation",
    "Results",
    "atomic_xc_name",
    "band_lines_path",
    "build_species",
    "describe_species",
    "find_pseudopotential",
    "make_structure",
    "parse_pao_basis_block",
    "write_bands",
    "write_eigenvalues",
    "write_forces",
    "write_outputs",
]
