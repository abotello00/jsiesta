r"""Band energies of a converged Hamiltonian at arbitrary k-points.

Theory:
    With the self-consistent sparse Hamiltonian :math:`H_{\mu\nu}(\mathbf{T})` and overlap
    :math:`S_{\mu\nu}(\mathbf{T})` fixed, the Kohn-Sham eigenvalues at any wavevector follow from the
    generalised eigenproblem :math:`H(\mathbf{k}) C = S(\mathbf{k}) C \varepsilon` of the Bloch sums
    :math:`M(\mathbf{k}) = \sum_{\mathbf{T}} M(\mathbf{T}) e^{2\pi i \mathbf{k}\cdot\mathbf{T}}`.
    No new self-consistency is needed: this is how band structures along symmetry lines are obtained.

Algorithm:
    Bloch assembly and batched diagonalisation shared with the SCF eigensolver
    (``jax_siesta.scf.eigensolver``), in batches of k-points to bound memory.

Invariants:
    - Eigenvalues are real and sorted in ascending order for every k-point and spin channel.
    - At the k-points of the SCF mesh, Fermi-Dirac occupation at the SCF Fermi level sums to the electron count.

SIESTA Mapping:
    Replaces ``bands.F`` (``BandLines`` / ``BandPoints``) and the eigenvalue output of ``diagon``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import SparsePattern
from jax_siesta.scf.eigensolver import _assembly_plan, _bloch_assemble, batched_generalized_eigh


def band_energies(
    h_sparse: Any,
    overlap: Any,
    pattern: SparsePattern,
    kpoints_frac: Any,
    nspin: int,
    k_batch_size: int = 64,
) -> np.ndarray:
    r"""Kohn-Sham eigenvalues of a sparse Hamiltonian at the given k-points.

    Theory:
        Solves :math:`H^\sigma(\mathbf{k}) C = S(\mathbf{k}) C \varepsilon` for each k-point. For ``nspin`` 1 and
        2 there is one :math:`N \times N` problem per spin channel; for ``nspin`` 4 and 8 the
        :math:`2N \times 2N` spinor Hamiltonian built from the channels ``[uu, dd, Re(ud), Im(ud)]`` has a single
        set of :math:`2N` bands.

    Algorithm:
        ``_bloch_assemble`` for :math:`H(\mathbf{k})` and :math:`S(\mathbf{k})`, then
        ``batched_generalized_eigh``, ``k_batch_size`` k-points at a time.

    Parameters:
        h_sparse: Hamiltonian ``(n_spin_mat, n_edges)`` in Hartree (e.g. ``SCFState.h_sparse``).
        overlap: Overlap ``(n_edges,)`` on the same pattern.
        pattern: ``SparsePattern`` of both matrices.
        kpoints_frac: k-points ``(n_k, 3)`` in fractional reciprocal coordinates.
        nspin: Spin mode ``1, 2, 4, 8``.
        k_batch_size: Number of k-points diagonalised at once.

    Returns:
        Eigenvalues in Hartree, shape ``(n_channels, n_k, n_bands)``: ``n_channels`` is 2 for ``nspin`` 2 and 1
        otherwise; ``n_bands`` is ``n_orb``, or ``2 n_orb`` for ``nspin`` 4 and 8.

    Invariants:
        - Ascending eigenvalues for each k-point and channel.
        - Independent of ``k_batch_size``.

    SIESTA Mapping:
        The diagonalisation part of ``bands.F``.
    """
    kpts = np.atleast_2d(np.asarray(kpoints_frac, dtype=np.float64))
    n_orb = pattern.n_orb
    src, dst = pattern.src_orb, pattern.dst_unit_orb
    shifts = pattern.cell_shifts[pattern.edge_cell_idx].astype(jnp.float64)
    plan = _assembly_plan(src, dst, n_orb)
    h_edges = jnp.asarray(h_sparse)
    s_edges = jnp.asarray(overlap).astype(jnp.complex128)

    def _bloch(edges: Any, k_block: Any) -> Any:
        return _bloch_assemble(edges.astype(jnp.complex128), src, dst, shifts, k_block, n_orb, plan)

    out: list[np.ndarray] = []
    for start in range(0, kpts.shape[0], max(int(k_batch_size), 1)):
        k_block = jnp.asarray(kpts[start : start + max(int(k_batch_size), 1)])
        s_k = _bloch(s_edges, k_block)
        if nspin in (1, 2):
            evals = [batched_generalized_eigh(_bloch(h_edges[i], k_block), s_k)[0] for i in range(nspin)]
        else:
            h_ud = _bloch(h_edges[2] + 1j * h_edges[3], k_block)
            h_2n = jnp.concatenate(
                [
                    jnp.concatenate([_bloch(h_edges[0], k_block), h_ud], axis=2),
                    jnp.concatenate([jnp.swapaxes(jnp.conj(h_ud), -1, -2), _bloch(h_edges[1], k_block)], axis=2),
                ],
                axis=1,
            )
            z_k = jnp.zeros_like(s_k)
            s_2n = jnp.concatenate(
                [jnp.concatenate([s_k, z_k], axis=2), jnp.concatenate([z_k, s_k], axis=2)], axis=1
            )
            evals = [batched_generalized_eigh(h_2n, s_2n)[0]]
        out.append(np.stack([np.asarray(e, dtype=np.float64) for e in evals], axis=0))
    return np.concatenate(out, axis=1)


__all__ = ["band_energies"]
