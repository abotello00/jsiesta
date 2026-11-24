User guide
==========

This section is the reference for running calculations. A calculation is defined by three things:

* the **structure** (lattice vectors, atoms),
* the **species**: for each one a PSML pseudopotential and a basis of numerical atomic orbitals,
* the **settings**: functional, spin, real-space mesh, k-points, electronic temperature, SCF options.

They can come from a SIESTA input file or from Python objects; both routes build the same
:class:`~jax_siesta.app.calculation.Calculation` and return the same
:class:`~jax_siesta.app.calculation.Results`.

.. list-table::
   :widths: 30 70

   * - :doc:`running`
     - The ``jsiesta`` command, its output and the files it writes
   * - :doc:`fdf_reference`
     - Input keywords that are understood, their defaults, and what happens with the rest
   * - :doc:`python_interface`
     - ``Calculation``, ``Results``, band structures, loops over structures and settings
   * - :doc:`pseudos_and_basis`
     - Pseudopotential files, basis sizes, ``PAO.Basis``, using SIESTA's basis, optimisation
   * - :doc:`restart_files`
     - Exchanging ``.XV``, ``.DM`` and ``.nc`` files with SIESTA
   * - :doc:`performance`
     - Symmetry, threads, GPU, compilation time, memory
   * - :doc:`siesta_differences`
     - What is not implemented and where defaults or numbers differ from SIESTA
   * - :doc:`troubleshooting`
     - Common errors and their fixes

Units: inputs follow SIESTA (units are written in the ``.fdf`` file); the text output and ``Results`` use eV and
Angstrom; the internal objects of the :doc:`../api/index` use Hartree atomic units (Bohr, Hartree).

.. toctree::
   :maxdepth: 1

   running
   fdf_reference
   python_interface
   pseudos_and_basis
   restart_files
   performance
   siesta_differences
   troubleshooting
