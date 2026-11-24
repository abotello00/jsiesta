Performance
===========

Where the time goes
-------------------

The timings at the end of every run separate *setup* (basis, neighbour lists, two-centre integrals), *scf* and
*forces_and_stress*. For small systems a large part of the first run is compilation: JAX compiles each
function the first time it sees a new set of array shapes. Three consequences:

* A single small calculation takes seconds even when the arithmetic takes milliseconds.
* In one Python process, repeated runs with the same shapes are much faster than the first; loops over
  geometries or settings should live in one script rather than in many invocations of the command.
* ``JSIESTA_COMPILATION_CACHE=/some/dir`` keeps compiled functions on disk between processes.

Use symmetry for crystals
-------------------------

``JSiesta.UseSymmetry T`` (``use_symmetry=True``) reduces the k-points to the irreducible wedge and symmetrises
the density matrix, forces and stress. For bulk silicon with a 4x4x4 grid this is 8 k-points instead of 64, with
energies equal to 2e-9 eV. It is off by default so that a run uses the same k-points and mesh as SIESTA would.

Shifted Monkhorst-Pack grids on primitive fcc or hexagonal cells break the lattice symmetry; without symmetry
this shows up as small spurious forces and off-diagonal stress. Prefer unshifted grids for those lattices.

Mesh cutoff
-----------

The cost of the mesh operations grows with ``MeshCutoff`` to the power 3/2. Hard pseudopotentials (the ONCV
PseudoDojo files, especially with core corrections) need 150 to 300 Ry for forces free of the *egg-box*
ripple; check the convergence of the quantity you care about (first tutorial on convergence). A sign of a mesh
that is too coarse is a net force on the cell that is not close to zero.

Threads
-------

The C++ kernels use ``OMP_NUM_THREADS`` threads (at most 8 by default); the linear algebra uses the threads of
the JAX runtime.

GPU
---

Build with ``-DJSIESTA_ENABLE_CUDA=ON`` and install a CUDA ``jaxlib``. The mesh operations, the semilocal
functionals and the JAX linear algebra then run on the device; meta-GGA evaluation through Libxc and parts of
the set-up remain on the host. Measured on one NVIDIA A100 for bulk silicon with PBE (8 to 512 atoms), an SCF
iteration is 2.4 to 7.7 times faster than SIESTA on 18 CPU ranks; set-up and compilation dominate below about
64 atoms. The measurements and their conditions are in the project status page of
:doc:`../development/index`.

Memory
------

Memory is dominated by the dense k-point matrices (orbitals squared times k-points) and by the mesh plan
(orbital values on mesh points). The examples of this documentation stay below 1 GB. On a workstation, run
exploratory jobs under a memory limit, for example
``systemd-run --user --scope -p MemoryMax=8G jsiesta in.fdf``.
