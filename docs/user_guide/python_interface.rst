The Python interface
====================

Everything the command does is available as three objects in the top-level package:

.. code-block:: python

   import jax_siesta as js

   js.make_structure(...)     # geometry from cell, symbols, positions
   js.Calculation             # inputs: structure, species, settings, k-grid
   js.Results                 # outputs: energies, forces, stress, bands

Building a calculation
----------------------

From an input file (all keywords of :doc:`fdf_reference`):

.. code-block:: python

   calc = js.Calculation.from_fdf("si.fdf", pseudo_dir="pseudos")

From a structure:

.. code-block:: python

   import numpy as np

   water = js.make_structure(
       cell=np.eye(3) * 8.0,                      # rows are lattice vectors, Angstrom
       symbols=["O", "H", "H"],
       positions=[[4.0, 4.0, 4.0], [4.757, 4.586, 4.0], [3.243, 4.586, 4.0]],
   )
   calc = js.Calculation.from_structure(
       water, pseudo_dir="pseudos", basis="DZP",
       xc_functional="PBE", mesh_cutoff_ry=150.0,
   )

Keyword arguments after the basis are fields of :class:`~jax_siesta.core.types.SCFConfig`: ``xc_functional``,
``nspin`` (1, 2, 4, 8), ``mesh_cutoff_ry``, ``electronic_temp_ha``, ``use_symmetry``, ``max_scf_iterations``,
``d_tol``, ``h_tol_ha``, ``dispersion``, ``net_charge``, ``mix_target``, ``mixer_variant``...

``basis`` is a size name for all species or a dictionary per label; a value may also be a
:class:`~jax_siesta.species.pao_generator.SpeciesBasisSpec` with explicit radii.

Running and reading results
---------------------------

.. code-block:: python

   results = calc.run()             # silent; calc.run(log=print) shows the report

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Attribute
     - Meaning
   * - ``converged``, ``n_iterations``
     - SCF outcome
   * - ``energy_ev``, ``free_energy_ev``
     - Kohn-Sham total energy and electronic free energy
   * - ``energies_ev``
     - Dictionary of energy terms
   * - ``forces_ev_ang``
     - ``(n_atoms, 3)`` array
   * - ``stress_ev_ang3``, ``pressure_gpa``
     - Stress tensor with SIESTA's sign; pressure is ``-trace/3``
   * - ``fermi_level_ev``
     -
   * - ``spin_moment``, ``atomic_spin_moments``
     - Bohr magnetons
   * - ``eigenvalues_ev(kpoints_frac=None)``
     - Eigenvalues at the SCF k-points or at any k-points
   * - ``band_structure(path, n)``
     - :class:`~jax_siesta.app.calculation.BandStructure` with ``plot()`` and ``band_gap_ev``
   * - ``scf_history``, ``timings``
     - Per-iteration records and wall times
   * - ``to_dict()``
     - Plain numbers for JSON
   * - ``observables``, ``scf_state``, ``basis``, ``pattern``, ``overlap``
     - The underlying objects in atomic units, for use with the :doc:`../api/index`

A run that does not converge returns normally with ``converged == False``: check it.

Writing SIESTA files from Python:

.. code-block:: python

   from jax_siesta.app import write_outputs

   write_outputs(results, "out_dir")        # .XV .DM .FA .EIG .json

Series of calculations
----------------------

``Calculation`` is immutable. ``with_config`` and ``with_structure`` return modified copies, which keeps loops
short:

.. code-block:: python

   for cutoff in (100, 150, 200, 300):
       r = calc.with_config(mesh_cutoff_ry=cutoff).run()
       print(cutoff, r.energy_ev)

   for scale in np.linspace(0.96, 1.04, 7):
       s = js.make_structure(cell0 * scale, symbols, frac, fractional=True)
       print(scale, calc.with_structure(s).run().energy_ev)

Within one Python process, functions are compiled once per set of array shapes. Changing the geometry without
changing the number of neighbours, or changing tolerances, reuses compiled code; changing the mesh, the basis
or the k-grid compiles again (a few seconds for small systems).

Going below the interface
-------------------------

``Calculation.run`` is a short function that chains the stages of the :doc:`../api/index` (basis, sparsity
pattern, two-centre integrals, SCF, observables). When you need access between the stages, for example to
modify the Hamiltonian or to differentiate through a stage, call them directly; the last tutorial shows how.
