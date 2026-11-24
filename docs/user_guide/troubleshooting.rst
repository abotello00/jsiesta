Troubleshooting
===============

``Compiled jsiesta C++ FFI library not found``
   The kernels are not built, or the package was installed outside the source tree. Run ``make ffi``, or set
   ``JSIESTA_FFI_LIB`` to the path of ``libjsiesta_ffi.so``.

``No pseudopotential X.psml in ...``
   The file name must be the species label of ``ChemicalSpeciesLabel`` plus ``.psml``. Use ``--pseudo-dir`` or
   ``SIESTA_PS_PATH``. If the message mentions a ``.psf`` file: only PSML is supported; PseudoDojo distributes
   PSML files.

``WARNING: input labels not used by jsiesta``
   The listed keywords had no effect. Check the spelling against :doc:`fdf_reference`; if the keyword is a
   SIESTA feature outside the scope (:doc:`siesta_differences`), the result may not be what the input intended.

``Unsupported XC functional``
   Names accepted: ``LDA``/``CA``/``PZ``, ``PW92``, ``PBE``, ``PBEsol``, ``R2SCAN``, ``SCAN``, ``TPSS``, or
   ``LIBXC-<x>+<c>``. Meta-GGAs need a build with Libxc.

The SCF does not converge
   Look at the SCF table. For metals and magnetic systems raise ``ElectronicTemperature`` (for example 25 meV),
   give sensible starting moments in ``DM.InitSpin``, or try ``SCF.Mix density``. Raising ``MaxSCFIterations``
   only helps if ``dDmax`` is still decreasing.

The total force is not zero, or forces look noisy
   The mesh is too coarse (egg-box effect): raise ``MeshCutoff``. With shifted k-grids on fcc or hexagonal
   primitive cells, use an unshifted grid or ``JSiesta.UseSymmetry T``.

Results differ from SIESTA by tens of meV
   Expected when each code generates its own basis. Use ``User.Basis T`` with SIESTA's ``.ion.xml`` files for a
   like-for-like comparison, and the same mesh (``Mesh.Sizes``) and k-points.

The process is killed
   Out of memory: reduce the k-grid (use symmetry), the mesh cutoff or the basis, and see the memory section of
   :doc:`performance`.

The first calculation is slow
   Compilation. See :doc:`performance`.

Where to ask
   Open an issue at https://github.com/abotello00/jsiesta/issues with the input file and the output.
