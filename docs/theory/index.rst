Theory
======

These notes state the equations that jsiesta implements and fix the conventions the code follows. They are
short on purpose: each one is the reference for a group of modules, and the docstrings of those modules point
back to them (*Theory* and *SIESTA Mapping* sections).

The method in one paragraph: the Kohn-Sham orbitals are expanded in strictly confined numerical atomic orbitals;
core electrons are replaced by norm-conserving pseudopotentials in separable Kleinman-Bylander form; the
Hamiltonian is split into two-centre terms, computed once per geometry from radial tables, and terms evaluated
on a uniform real-space mesh (Hartree, exchange-correlation and the neutral-atom potential); the density is made
self-consistent by diagonalisation at each k-point and mixing; forces and stress are derivatives of the total
energy at the converged density matrix. The original description is J. M. Soler *et al.*, J. Phys.: Condens.
Matter **14**, 2745 (2002).

.. list-table::
   :widths: 34 66

   * - :doc:`../_generated/theory_conventions_and_units`
     - Units, cell and reciprocal-lattice conventions, spin channels, sign conventions. Read this first if you
       use the internal objects.
   * - :doc:`../_generated/theory_naos_and_psml`
     - Numerical atomic orbitals, PSML pseudopotentials, the neutral-atom potential
   * - :doc:`../_generated/theory_symmetry_and_scf`
     - Symmetry projection of matrices and mesh functions, the SCF cycle and its mixing
   * - :doc:`../_generated/theory_variational_forces_stress`
     - Forces and stress as derivatives of the energy at fixed density and energy-density matrices

.. toctree::
   :hidden:
   :maxdepth: 1

   ../_generated/theory_conventions_and_units
   ../_generated/theory_naos_and_psml
   ../_generated/theory_symmetry_and_scf
   ../_generated/theory_variational_forces_stress
