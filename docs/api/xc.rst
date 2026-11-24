Exchange-correlation and dispersion
===================================

Density functionals on the mesh and the dispersion corrections.

``LibxcFFIEvaluator`` evaluates LDA, GGA and meta-GGA functionals (through Libxc, or built-in kernels for LDA and
PBE) and ``xc_energy_from_grid`` turns mesh densities into the energy whose derivative is the potential.
Functional names are resolved by ``resolve_xc_spec`` (``PBE``, ``R2SCAN``, ``LIBXC-<x>+<c>``...).

Two dispersion models implement the ``DispersionModel`` protocol: ``DFTD3Dispersion`` (DFT-D3 with Becke-Johnson
damping, an atom-pairwise correction through the ``dftd3`` package) and ``RVV10Dispersion`` (the rVV10 non-local
density functional, evaluated on the mesh with a kernel interpolation).

.. include:: ../_generated/api/xc.inc
