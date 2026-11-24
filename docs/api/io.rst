Input and output files
======================

Readers and writers for files shared with SIESTA and with the validation tools.

``fdf`` parses SIESTA input files (``tokenize_fdf``, ``parse_fdf``) and the JSON case files used by the tests.
``siesta_files`` reads and writes SIESTA's restart files: ``.XV`` (geometry), ``.DM`` (density matrix) and the
NetCDF file of ``CDF.Save`` (overlap, Hamiltonian, density matrices), including the conversions between the
sparsity layouts and spin conventions of the two codes.

User-level description: :doc:`../user_guide/restart_files` and :doc:`../user_guide/fdf_reference`.

.. include:: ../_generated/api/io.inc
