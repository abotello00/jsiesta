Tutorials
=========

Worked examples that take you from a first calculation to the internals of the code. Each tutorial is a
notebook shown here with its output, and ends with exercises; solutions to selected exercises are in the last
notebook.

How to use them
---------------

The tutorials live in ``docs/tutorials`` of the repository, each in two forms with the same content:

* ``NN_name.ipynb``: a Jupyter notebook. Start ``jupyter lab`` in ``docs/tutorials`` (``pip install jupyterlab``
  in the environment) and run the cells.
* ``NN_name.py``: a plain Python script with the same cells, for those who prefer an editor and a terminal:
  ``python 01_first_calculation.py``.

They need jsiesta installed with its compiled library (:doc:`../getting_started/installation`) and
``matplotlib``. The pseudopotentials are the PseudoDojo PBE files in ``examples/pseudos``. All tutorials run on a
laptop CPU in one to four minutes each and use less than 1.5 GB of memory.

.. list-table::
   :header-rows: 1
   :widths: 6 34 44 16

   * -
     - Tutorial
     - What you learn
     - Script
   * - 1
     - :doc:`01_first_calculation`
     - Input file, the ``jsiesta`` command, reading the output, the same from Python
     - :download:`py <01_first_calculation.py>`
   * - 2
     - :doc:`02_convergence`
     - Mesh cutoff, k-points, the egg-box effect, series of calculations
     - :download:`py <02_convergence.py>`
   * - 3
     - :doc:`03_lattice_constant`
     - Equation of state, bulk modulus, the stress tensor as a check
     - :download:`py <03_lattice_constant.py>`
   * - 4
     - :doc:`04_bands_and_dos`
     - Band structure, band gaps, ``.bands`` files, density of states
     - :download:`py <04_bands_and_dos.py>`
   * - 5
     - :doc:`05_molecular_geometry`
     - Molecules in a box, forces, relaxation with an external optimiser, vibrations
     - :download:`py <05_molecular_geometry.py>`
   * - 6
     - :doc:`06_magnetism`
     - Collinear and non-collinear spin, starting moments, checking magnetic solutions
     - :download:`py <06_magnetism.py>`
   * - 7
     - :doc:`07_basis_sets`
     - What a basis is made of, sizes, energy shift, split norm, ``PAO.Basis``
     - :download:`py <07_basis_sets.py>`
   * - 8
     - :doc:`08_under_the_hood`
     - The stages behind ``Calculation.run``, sparse matrices, H(k), restarts
     - :download:`py <08_under_the_hood.py>`
   * -
     - :doc:`09_solutions`
     - Worked solutions: basis and lattice constant, graphene bands, H\ :sub:`2`, O\ :sub:`2` binding, Mulliken
       charges
     - :download:`py <09_solutions.py>`

Suggested paths
---------------

* **New to SIESTA-type codes:** 1, 2, 3, 4 in order, then 7.
* **SIESTA user:** 1 for the differences in running and output, then 8, and
  :doc:`../user_guide/siesta_differences`.
* **Developer:** 8, then the :doc:`../api/index`.

A note on the numbers
---------------------

The outputs shown were produced on a laptop CPU with the pseudopotentials of ``examples/pseudos``. Your numbers
should agree to the digits shown, up to small differences in the last digits between machines. The tutorials use
parameters chosen to run quickly; tutorial 2 shows how to check whether they are converged enough for your
purpose.

.. toctree::
   :hidden:
   :maxdepth: 1

   01_first_calculation
   02_convergence
   03_lattice_constant
   04_bands_and_dos
   05_molecular_geometry
   06_magnetism
   07_basis_sets
   08_under_the_hood
   09_solutions
