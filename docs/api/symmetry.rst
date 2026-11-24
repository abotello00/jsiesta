Symmetry
========

Use of the space group (and of the magnetic group for spin-polarised systems) to reduce work and to keep
results exactly symmetric.

``build_native_symmetry`` is the entry point: from a structure and a basis it returns the symmetry engine, a
sparsity pattern closed under the group, the irreducible k-points and a mesh commensurate with the operations.
The engine (``NativeSymmetryEngine``) projects sparse matrices onto their symmetric part with precomputed tables
(``projection``), using Wigner rotation matrices for the orbitals (``spacegroup``), symmetrises mesh functions,
forces and stress, and evaluates two-centre integrals on symmetry-inequivalent pairs only.

Identities the tests rely on: the projection is idempotent, the projected Hamiltonian commutes with every
operation, and a symmetrised run agrees with the full unsymmetrised one.

.. include:: ../_generated/api/symmetry.inc
