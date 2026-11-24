API reference
=============

The reference is organised by the stage of a calculation, in the order in which a calculation uses them. Each
page starts with what the stage does and how its pieces fit, followed by the documentation of every public
function and class, generated from the source.

.. code-block:: text

   Structure + PSML files
         |  species      pseudopotential and basis tables of each species
         v
     BasisData  -->  SparsePattern            (symmetry: irreducible k-points, closed pattern)
         |  integrals    S, T, V_KB, V_NA on the pattern
         v
      run_scf   <--  grid (density, Hartree, potential matrix elements), xc
         |  scf          diagonalise, mix, repeat
         v
     SCFState
         |  observables  energy, forces, stress, bands
         v
     Observables

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Section
     - Contents
   * - :doc:`interface`
     - ``Calculation``, ``Results``, the ``jsiesta`` command, output writers: start here
   * - :doc:`core`
     - Data classes exchanged by all stages, stage protocols, library loader
   * - :doc:`species`
     - PSML reader, basis generation, basis optimisation, sparsity pattern
   * - :doc:`integrals`
     - Two-centre matrix elements and their derivatives
   * - :doc:`grid`
     - Mesh selection, density and potential on the mesh
   * - :doc:`xc`
     - Exchange-correlation functionals, DFT-D3, rVV10
   * - :doc:`symmetry`
     - Space-group engine
   * - :doc:`scf`
     - SCF driver, eigensolver, mixers
   * - :doc:`observables`
     - Final energy, forces, stress, band energies
   * - :doc:`io`
     - FDF input, SIESTA restart files
   * - :doc:`testing`
     - SIESTA adapters and validation helpers
   * - :doc:`cpp`
     - Compiled kernels

Conventions used throughout
---------------------------

* Hartree atomic units inside the package; eV and Angstrom only in :doc:`interface` and in files.
* Every docstring has the sections *Theory* (the physics), *Algorithm* (how it is computed), *Invariants*
  (what always holds, and what the tests check) and *SIESTA Mapping* (the SIESTA routine with the same role).
* Names starting with an underscore are internal and may change.

.. toctree::
   :hidden:
   :maxdepth: 1

   interface
   core
   species
   integrals
   grid
   xc
   symmetry
   scf
   observables
   io
   testing
   cpp
