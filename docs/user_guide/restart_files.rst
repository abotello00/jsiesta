Restart files and exchange with SIESTA
======================================

``jax_siesta.io`` reads and writes the three files SIESTA uses to continue a calculation, converting the
sparsity layout (supercell column indices), the spin components and the units (Ry) in both directions.

.. list-table::
   :header-rows: 1
   :widths: 14 28 28 30

   * - File
     - jsiesta reads
     - jsiesta writes
     - SIESTA reads it with
   * - ``<label>.XV``
     - cell and positions (species data from the input)
     - yes, zero velocities
     - ``MD.UseSaveXV T``
   * - ``<label>.DM``
     - initial density matrix, rescaled to the electron count
     - yes
     - ``DM.UseSaveDM T`` / ``File.DM.Init``
   * - ``<label>.nc``
     - ``H`` (first input Hamiltonian), ``DM``, geometry
     - geometry, ``S``, ``H``, ``DM``, ``EDM``, ``Ef``, forces, stress
     - not as a restart (SIESTA has no reader for ``H``); ``sisl`` and other tools read it

With the command
----------------

Every run writes ``<label>.XV`` and ``<label>.DM`` (``<label>.nc`` too with ``CDF.Save T``). In the next run:

.. code-block:: text

   DM.UseSaveDM  T      # start from <label>.DM
   MD.UseSaveXV  T      # geometry from <label>.XV

From Python
-----------

.. code-block:: python

   from jax_siesta.io import read_dm, read_nc_restart, structure_from_xv, write_restart_files

   structure = structure_from_xv("si.XV", template=structure)       # cell and positions from SIESTA
   dm0 = read_dm("si.DM", pattern, config.nspin)                    # any nsc, any sparsity pattern, any spin mode
   dm0, h0, ef = read_nc_restart("si.nc", pattern, config.nspin)    # DM and H (Hartree) from a CDF.Save file
   results = calc.run(initial_dm=dm0, initial_h=h0)

``pattern`` is the sparsity pattern of the calculation (``Results.pattern`` of an earlier run of the same
structure and basis, or ``build_sparse_pattern``).

Conditions for exchanging files with SIESTA
-------------------------------------------

* Both codes must use the **same basis**: the same orbitals in the same order per atom. Use ``User.Basis T`` in
  jsiesta with SIESTA's ``.ion.xml`` files, or the same explicit ``PAO.Basis`` block.
* The species index in the ``.XV`` file is the position of the species in ``ChemicalSpeciesLabel``; list the
  species in the same order in both inputs.
* The sparsity patterns need not match: matrix elements are mapped by orbital pair and lattice vector.
* The ``.nc`` file needs the optional ``netCDF4`` package (``pip install netCDF4``).

Status: the formats are verified byte for byte against SIESTA-written ``.DM`` and ``.XV`` files; the full
exchange with a running SIESTA is covered by a test that needs a SIESTA binary
(``test_siesta_reads_jsiesta_files_and_jsiesta_reads_siesta_files``).
