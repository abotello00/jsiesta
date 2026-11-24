Self-consistent field
=====================

The self-consistent cycle. ``run_scf`` takes the structure, basis, pattern, two-centre matrices, k-points and
configuration and returns the converged ``SCFState``. Each iteration diagonalises the Hamiltonian at every
k-point (``GeneralizedEigenSolver``), builds the output density matrix, evaluates the mesh Hamiltonian for it,
measures the residual, and mixes.

The mixers (``TikhonovPulayMixer``: regularised Pulay, Pulay, Broyden, ADIIS) implement the ``SCFMixer``
protocol. ``descriptors`` estimates how difficult the system is and recommends the mixing weight and history,
and ``controller`` adapts the step during the run and decides convergence and the outcome label.

``run_scf`` accepts an initial density matrix or Hamiltonian (restart) and a per-iteration ``callback`` for
progress reports.

.. include:: ../_generated/api/scf.inc
