Energies, forces, stress and bands
==================================

What is computed after the SCF. ``StaticObservableEvaluator`` evaluates the energy once more at the output
density matrix (no mixing involved) and returns an ``Observables`` object with the energy terms, the forces and
the stress tensor. Forces and stress are derivatives of the discrete energy at fixed density and energy-density
matrices: analytic derivatives of the two-centre terms plus automatic differentiation of the mesh terms.

``band_energies`` gives the eigenvalues of the converged Hamiltonian at arbitrary k-points, the basis of band
structures and densities of states.

.. include:: ../_generated/api/observables.inc
