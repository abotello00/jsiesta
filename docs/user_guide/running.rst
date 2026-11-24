Running the ``jsiesta`` command
===============================

.. code-block:: text

   jsiesta [input.fdf] [--pseudo-dir DIR] [--out-dir DIR] [--check] [--version]

.. list-table::
   :widths: 25 75

   * - ``input.fdf``
     - SIESTA input file. When omitted the input is read from standard input: ``jsiesta < in.fdf > out``.
   * - ``--pseudo-dir DIR``
     - Directory searched first for ``<Label>.psml`` (and ``<Label>.ion.xml`` with ``User.Basis``). The
       directory of the input file and ``SIESTA_PS_PATH`` are searched next.
   * - ``--out-dir DIR``
     - Where output files go (default: the current directory)
   * - ``--check``
     - Read the input, generate the basis, print the report, and stop before the SCF. Use it to see which
       keywords are not used and what basis will be generated.

``python -m jax_siesta`` is the same command.

Exit status: ``0`` converged; ``1`` the SCF did not converge and ``SCF.MustConverge`` is true (the default, as in
SIESTA); ``2`` error in the input.

What the output shows
---------------------

1. **Input report.** The input file and, importantly, a ``WARNING`` line with every label of the input that
   jsiesta did not use, either because the feature is not implemented or because the label is misspelt. Keywords
   that only control SIESTA's printing are accepted silently.
2. **Species and basis.** For each species, the pseudopotential file and every radial function of the basis with
   its cut-off radius.
3. **System.** Cell, coordinates, number of orbitals and electrons, functional, spin, k-points, the real-space
   mesh and the cutoff it actually gives.
4. **SCF table.** One line per iteration: Harris energy, Kohn-Sham energy, free energy, the largest change of the
   density matrix (``dDmax``) and of the Hamiltonian (``dHmax``), and the Fermi level.
5. **Results.** Energy terms, forces, stress, pressure, spin moments (for spin-polarised runs) and timings.

Lines are prefixed ``jsiesta:`` where SIESTA writes ``siesta:``. The layout resembles SIESTA's but is not
identical, so scripts that parse SIESTA's standard output should read ``<label>.json`` or the files below instead.

Output files
------------

All are named after ``SystemLabel`` (default ``siesta``).

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - File
     - Contents
     - Format
   * - ``.XV``
     - Cell and atomic positions
     - SIESTA's; read by SIESTA with ``MD.UseSaveXV T``
   * - ``.DM``
     - Density matrix of the last iteration
     - SIESTA's; read by SIESTA with ``DM.UseSaveDM T``
   * - ``.FA``
     - Forces (eV/Ang)
     - SIESTA's
   * - ``.EIG``
     - Fermi level and eigenvalues at the SCF k-points (eV)
     - SIESTA's
   * - ``.bands``
     - Band energies along ``%block BandLines`` (only with that block)
     - SIESTA's; plot with ``gnubands`` or ``sisl``
   * - ``.nc``
     - Geometry, S, H, DM, forces, stress (only with ``CDF.Save T``; needs ``netCDF4``)
     - SIESTA's ``CDF.Save`` layout
   * - ``.json``
     - Everything in the results section, the SCF history and timings
     - JSON, in eV and Ang

The ``.XV`` and ``.DM`` writers are tested byte for byte against files written by SIESTA. The ``.FA``, ``.EIG``
and ``.bands`` writers follow the record layout of the SIESTA sources and are tested against their own content,
not yet against a SIESTA run.

Restarting
----------

``DM.UseSaveDM T`` starts from ``<label>.DM`` if it exists (a file from jsiesta or from SIESTA with the same
basis); ``MD.UseSaveXV T`` takes the geometry from ``<label>.XV``. See :doc:`restart_files`.

Band structures
---------------

.. code-block:: text

   BandLinesScale  ReciprocalLatticeVectors
   %block BandLines
      1  0.5  0.5  0.5  L
     30  0.0  0.0  0.0  \Gamma
     30  0.5  0.0  0.5  X
   %endblock BandLines

Both ``BandLinesScale`` values of SIESTA are supported (``pi/a``, the default, and
``ReciprocalLatticeVectors``). ``BandPoints`` is not.
