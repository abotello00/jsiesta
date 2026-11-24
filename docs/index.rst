jsiesta
=======

**jsiesta** is an implementation of the SIESTA method for Kohn-Sham density-functional theory, written in
JAX with C++ and CUDA kernels. It computes total energies, atomic forces and the stress tensor of molecules and
solids with numerical atomic orbitals and norm-conserving pseudopotentials, on CPUs and GPUs.

You can use it in two ways:

.. code-block:: bash

   jsiesta si.fdf > si.out          # like SIESTA: an .fdf input file and <Label>.psml pseudopotentials

.. code-block:: python

   import jax_siesta as js          # or from Python, in scripts and notebooks

   results = js.Calculation.from_fdf("si.fdf").run()
   print(results.energy_ev, results.forces_ev_ang, results.pressure_gpa)

Where to start
--------------

.. list-table::
   :widths: 28 72

   * - :doc:`getting_started/index`
     - Install the package, build the compiled library, run a first calculation in five minutes.
   * - :doc:`user_guide/index`
     - How to run calculations: the ``jsiesta`` command and its input keywords, the Python interface,
       pseudopotentials and basis sets, restart files, performance, differences from SIESTA.
   * - :doc:`tutorials/index`
     - Worked examples with exercises, as notebooks and Python scripts: convergence, lattice constants, band
       structures, molecular geometry, magnetism, basis sets, and the internals of a calculation.
   * - :doc:`api/index`
     - Reference of every public function and class, organised by the stage of the calculation it belongs to.
   * - :doc:`theory/index`
     - The equations that are implemented, with the conventions and units used in the code.
   * - :doc:`development/index`
     - Architecture, validation status against SIESTA, decision log and how to contribute.

What it can do
--------------

.. list-table::
   :widths: 25 75

   * - Calculations
     - Static single points: total energy, forces, stress, band energies, Mulliken spin moments
   * - Spin
     - Unpolarised, collinear, non-collinear and spin-orbit
   * - Functionals
     - LDA, GGA and meta-GGA (r2SCAN, SCAN, TPSS) through Libxc; DFT-D3(BJ) and rVV10 dispersion
   * - Basis sets
     - Generated from the pseudopotential (SZ to TZ2P, or ``PAO.Basis``), read from SIESTA ``.ion.xml`` files,
       and a basis-enthalpy optimiser
   * - Symmetry
     - Space and magnetic groups (spglib): irreducible k-points, symmetrised matrices, forces and stress
   * - Files
     - SIESTA ``.fdf`` input; ``.XV``, ``.DM``, ``.nc`` restart files read and written; ``.FA``, ``.EIG``,
       ``.bands`` written
   * - Hardware
     - Multi-threaded CPU; CUDA kernels for the mesh operations and semilocal functionals

It does **not** do geometry relaxation or molecular dynamics by itself, and it reads PSML pseudopotentials only;
see :doc:`user_guide/siesta_differences`.

.. toctree::
   :hidden:

   getting_started/index
   user_guide/index
   tutorials/index
   api/index
   theory/index
   development/index
