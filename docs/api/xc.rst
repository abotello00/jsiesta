Exchange-correlation and dispersion
===================================

Density functionals on the mesh and the dispersion corrections.

``LibxcFFIEvaluator`` evaluates LDA, GGA and meta-GGA functionals (through Libxc, or built-in kernels for LDA and
PBE) and ``xc_energy_from_grid`` turns mesh densities into the energy whose derivative is the potential.
Functional names are resolved by ``resolve_xc_spec`` (``PBE``, ``R2SCAN``, ``LIBXC-<x>+<c>``...).

Hybrid functionals (``HSE06``, ``PBE0``, ``B3LYP`` or a Libxc hybrid id) are described by ``exact_exchange_spec``:
the semilocal remainder goes through the same mesh evaluation, and the exact exchange is added by the package
``jax_siesta.hfx``. There ``fit_species`` fits the numerical orbitals with Gaussians and ``Nao2GtoExchange`` builds
the exchange matrix and energy on the sparse pattern (:doc:`../_generated/theory_hybrid_functionals`).

Two dispersion models implement the ``DispersionModel`` protocol: ``DFTD3Dispersion`` (DFT-D3 with Becke-Johnson
damping, an atom-pairwise correction through the ``dftd3`` package) and ``RVV10Dispersion`` (the rVV10 non-local
density functional, evaluated on the mesh with a kernel interpolation).

.. include:: ../_generated/api/xc.inc
