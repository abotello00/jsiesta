Species: pseudopotentials and basis sets
========================================

Everything that depends on one chemical species and not on the structure. A species is read from a PSML
pseudopotential (semilocal potentials or Kleinman-Bylander projectors, core charge, valence configuration) and
completed with a basis of confined numerical atomic orbitals and the neutral-atom potential built from them.

There are two sources for the basis: ``PSMLBasisLoader`` takes the orbitals of a SIESTA run (``.ion.xml``),
which gives identical basis functions in both codes, and ``generate_species_from_psml`` solves the confined
pseudo-atom itself (energy shift, split norm, soft confinement, perturbative polarisation), which needs no
SIESTA run. ``basis_opt`` builds on the generator: it optimises cut-off radii and confinement parameters by
minimising the basis enthalpy over a set of training structures and writes the result as a ``PAO.Basis`` block.

``build_sparse_pattern`` also lives here: the sparsity pattern follows from the cut-off radii of the species.

User-level description: :doc:`../user_guide/pseudos_and_basis`. Theory: :doc:`../theory/index`.

.. include:: ../_generated/api/species.inc
