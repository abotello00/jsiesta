Data model and protocols
========================

The objects every stage exchanges. All state is explicit and immutable: frozen dataclasses registered as JAX
PyTrees, so that functions can be compiled, vectorised and differentiated, and so that the result of a function
depends only on its arguments.

.. list-table::
   :widths: 26 74

   * - ``Structure``
     - Cell (columns are lattice vectors), positions, species, initial moments
   * - ``SpeciesData``
     - Pseudopotential projectors, neutral-atom potential and radial orbitals of one species
   * - ``BasisData``
     - Orbital and projector index tables of a structure
   * - ``SparsePattern``
     - Which orbital pairs interact; all sparse matrices are arrays over its edges
   * - ``KPointMesh``
     - k-points and weights
   * - ``TwoCenterMatrices``
     - Overlap, kinetic, neutral-atom and non-local matrices with their derivatives
   * - ``SCFConfig``
     - Settings: spin, functional, mesh, tolerances, mixing, symmetry
   * - ``SCFState``
     - Hamiltonian, density matrices and mixer history of an SCF iteration
   * - ``Observables``
     - Energies, forces, stress and moments of the final evaluation

Units are Hartree atomic units throughout (Bohr, Hartree); conversion constants live in
``jax_siesta.core.types``. Sparse matrices have shape ``(n_spin_mat, n_edges)`` with 1, 2 or 4 spin channels.

``jax_siesta.core.protocols`` defines one ``typing.Protocol`` per stage (basis, two-centre integrals, grid
operations, exchange-correlation, dispersion, eigensolver, mixer, symmetry, observables). A new method is added
by implementing a protocol, not by editing another stage. ``core.ffi`` loads the compiled library and
``core.compilation`` controls JAX's persistent compilation cache.

.. include:: ../_generated/api/core.inc
