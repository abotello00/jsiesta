Differences from SIESTA
=======================

jsiesta implements the SIESTA *method* for a deliberately small scope. This page lists what a SIESTA user will
find different.

Not implemented
---------------

* **Geometry relaxation, molecular dynamics, phonons.** Only single points. Forces and stress are available, so
  relaxations can be driven from Python (tutorial on molecular geometry).
* **Pseudopotential formats other than PSML** (``.psf``, ``.vps``).
* **Solvers other than diagonalisation** (order-N, PEXSI, ELSI, TranSIESTA).
* **DFT+U, van der Waals density functionals other than rVV10**, external electric fields,
  dipole corrections, charged-slab corrections.
* **Occupation functions other than Fermi-Dirac**.
* **Non-diagonal k-point grids, ``kgrid.Cutoff``, ``BandPoints``**.
* **Ghost atoms, synthetic atoms, Z-matrix input, constraints**.
* **Outputs:** charge densities and potentials on the mesh, wavefunctions, PDOS, COOP, the ``.HSX`` file (the
  ``.nc`` file holds H and S).
* **MPI.** Parallelism is by threads and GPU.

Beyond SIESTA
-------------

* **Hybrid functionals** (``XC.Authors HSE06``, ``PBE0``, ``B3LYP``), with the method of
  HONPAS: the exact exchange is computed from Gaussian fits of the orbitals. Energies, eigenvalues, forces and
  stress, for every ``Spin`` mode, at a cost far above a GGA: 7 s for a full build of the exchange matrix
  for bulk Si with a single-zeta basis and three Gaussians per orbital on an eight-thread laptop, less in the
  later SCF steps. Compared with HONPAS for bulk Si with the same orbitals and Gaussians: the exchange energy
  agrees to 0.3 meV and the band energies to 4 meV. The stress HONPAS prints for a hybrid does not contain the
  exchange term; the one of jsiesta is the strain derivative of the energy.

Different defaults
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Setting
     - jsiesta
     - SIESTA 5
   * - SCF tolerances
     - dDmax 1e-5, dHmax 1e-5 Ha (0.27 meV), and an energy criterion 1e-6 Ha
     - 1e-4 and 1 meV
   * - Mixer
     - regularised (Tikhonov) Pulay on the Hamiltonian with an adaptive step controller; weight and history
       chosen from the system
     - Pulay on the Hamiltonian, weight 0.25, history 2
   * - Max. SCF iterations
     - 100
     - 1000
   * - k-point reduction
     - none unless ``JSiesta.UseSymmetry T`` (then the full space group)
     - time reversal

The keywords for all of these work as in SIESTA when given explicitly.

Numerical differences
---------------------

* **Basis generation.** Independent implementation; see :doc:`pseudos_and_basis`. With SIESTA's own basis
  (``User.Basis``) the total energy of bulk Si agrees within 2 meV per cell and the forces of hcp Co within
  1e-4 eV/Ang; the validation status page under :doc:`../development/index` gives the tolerance of each test.
* **Neutral-atom energy.** jsiesta integrates the neutral-atom pair term exactly, whereas SIESTA interpolates
  it from a table; this is the largest term in the remaining energy differences (decision D-025).
* **Forces and stress** are derivatives of the discrete energy, including the egg-box ripple of the mesh.

Output
------

The text output is similar in layout, with the prefix ``jsiesta:``; it is not a drop-in replacement for parsers
of SIESTA's output. The ``.XV``, ``.DM``, ``.FA``, ``.EIG``, ``.bands`` and ``.nc`` files use SIESTA's formats.

Extensions
----------

``JSiesta.UseSymmetry``, ``JSiesta.Dispersion`` (``RVV10``), meta-GGA functionals by name, restart from a
Hamiltonian in a ``.nc`` file, the Python interface, automatic differentiation of the energy, and GPU
execution.
