Real-space grid
===============

The part of the Hamiltonian evaluated on a uniform mesh: the electron density from the density matrix
(``rhoofd``), the Hartree potential by FFT, and the matrix elements of the mesh potential (``vmat``).

``choose_mesh_dimensions`` selects the mesh for a cutoff with SIESTA's criterion, and ``siesta_mesh_cutoff_ry``
gives the cutoff a mesh provides. The orbital-mesh operations are C++/CUDA kernels wrapped as differentiable JAX
functions (``rhoofd_differentiable``, ``vmat_differentiable``): their derivatives with respect to atomic
positions and strain are what makes forces and stress exact derivatives of the discrete energy. ``GridOps``
bundles them behind the ``RealSpaceGridOps`` protocol.

.. include:: ../_generated/api/grid.inc
