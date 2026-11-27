r"""Exact exchange for hybrid functionals.

Theory:
    A hybrid functional adds a fraction of the exact (Hartree-Fock) exchange energy of the Kohn-Sham density
    matrix to a semilocal functional. In a basis of strictly localised numerical orbitals the exchange matrix is
    sparse and is built here from four-centre integrals over Gaussian fits of the orbitals (NAO2GTO).

Algorithm:
    ``nao2gto`` fits the orbitals; ``exchange.Nao2GtoExchange`` builds the matrix and the energy with the C++
    kernels ``eri.cpp`` and ``hfx.cpp``.

Invariants:
    The stage adds nothing for a functional without exact exchange.

SIESTA Mapping:
    None in SIESTA; the method is that of HONPAS.
"""

import numpy as np

from jax_siesta.core.types import BasisData, SCFConfig, SparsePattern, Structure
from jax_siesta.hfx.exchange import Nao2GtoExchange
from jax_siesta.hfx.nao2gto import GaussianShell, fit_radial_function, fit_species
from jax_siesta.xc.functional import exact_exchange_spec

__all__ = ["GaussianShell", "Nao2GtoExchange", "build_exact_exchange", "fit_radial_function", "fit_species"]


def build_exact_exchange(
    structure: Structure,
    basis: BasisData,
    pattern: SparsePattern,
    config: SCFConfig,
    k_grid: tuple[int, int, int] | None = None,
) -> Nao2GtoExchange | None:
    r"""Exact-exchange stage of a calculation, or ``None`` for a functional without exact exchange.

    Theory:
        The functional name decides the kernel and the fractions (``exact_exchange_spec``). The lattice sums of
        the exchange matrix run over the density matrix on the sparse pattern, :math:`P^{\mathbf{R}}` for the
        translations :math:`\mathbf{R}` of the pattern. With an :math:`N_1 N_2 N_3` k-point mesh the density
        matrix is periodic in the supercell of :math:`N_d` cells per direction, so two edges whose translations
        differ by :math:`N_d` cells carry the same value. A sum over such repeated values is not an approximation
        of the exchange of the crystal. For the screened kernel it is the usual finite-size error and decays
        with the size of the supercell; for the full-range kernel :math:`1/r` it grows with the range of the
        pattern, so it is refused.

    Algorithm:
        Builds ``Nao2GtoExchange`` with ``config.hfx_n_gauss`` and ``config.hfx_screen_tol``, with incremental
        builds (the stage serves one self-consistent cycle). For a full-range
        hybrid, the translations used by the edges are compared with ``k_grid``.

    Parameters:
        structure: Geometry.
        basis: Orbital basis.
        pattern: Sparse pattern of the density matrix and the Hamiltonian.
        config: SCF configuration (functional, spin mode, exchange settings).
        k_grid: Divisions of the k-point mesh; needed for a full-range hybrid in a periodic system.

    Returns:
        The stage, or ``None``.

    Invariants:
        Raises ``NotImplementedError`` for a full-range hybrid when the translations of the pattern along some
        direction span as many cells as the k-point mesh has divisions (or more), or when ``k_grid`` is unknown
        and the pattern connects different cells.

    SIESTA Mapping:
        None in SIESTA; the HFX set-up of HONPAS.
    """
    spec = exact_exchange_spec(config.xc_functional)
    if not spec.is_hybrid:
        return None
    used_shifts = np.rint(np.asarray(pattern.cell_shifts)).astype(int)[np.asarray(pattern.edge_cell_idx)]
    if spec.alpha != 0.0 and bool(np.any(used_shifts != 0)):
        span = used_shifts.max(axis=0) - used_shifts.min(axis=0)
        if k_grid is None or any(int(span[d]) >= int(k_grid[d]) for d in range(3)):
            need = tuple(int(x) + 1 for x in span)
            raise NotImplementedError(
                f"{config.xc_functional!r} has full-range exact exchange. Its lattice sum needs a k-point mesh "
                f"with at least {need} divisions for this cell and basis (the orbital pairs of the sparse pattern "
                f"span {tuple(int(x) for x in span)} cells); the mesh is {k_grid}. Use a denser mesh, a larger "
                "cell for an isolated system, or a screened hybrid such as HSE06."
            )
    return Nao2GtoExchange(
        structure, basis, pattern, spec, n_gauss=config.hfx_n_gauss, screen_tol=config.hfx_screen_tol, incremental=True
    )
