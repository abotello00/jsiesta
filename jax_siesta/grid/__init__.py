r"""Uniform 3D real-space mesh operators, Poisson solver, and grid FFI bridges.

Theory:
    Scatters the sparse density matrix onto the periodic 3D real-space mesh
    (:math:`\rho(\mathbf{r})`), solves the periodic Poisson equation for
    :math:`\delta\rho = \rho - \rho^{\text{NA}}` via 3D FFT, and gathers
    :math:`V_{\text{eff}}(\mathbf{r})` back into sparse matrix elements :math:`H^{\text{grid}}_{\mu\nu}`.

Algorithm:
    Uses compiled C++/CUDA XLA FFI custom calls (``rhoofd``, ``vmat``, ``na_grid``,
    and custom VJP backward passes for ``dfscf``) paired with ``jax.numpy.fft``.

Invariants:
    - Exact discrete adjoint identity: :math:`\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]`.
    - Total integrated density equals :math:`N_e`.

SIESTA Mapping:
    Replaces ``rhoofd.F90``, ``vmat.F90``, ``poison.F``, and ``dhscf.F``.
"""

from jax_siesta.grid.mesh import (
    GridOps,
    choose_mesh_dimensions,
    neutral_atom_grid_differentiable,
    reciprocal_grid_vectors,
    rhoofd_differentiable,
)

__all__ = [
    "GridOps",
    "choose_mesh_dimensions",
    "neutral_atom_grid_differentiable",
    "reciprocal_grid_vectors",
    "rhoofd_differentiable",
]
