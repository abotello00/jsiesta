Two-centre integrals
====================

The part of the Hamiltonian that depends only on pairs of atoms and is computed once per geometry: overlap
:math:`S`, kinetic energy :math:`T`, the Kleinman-Bylander non-local pseudopotential :math:`V^{KB}` (with its
spin-orbit part for ``nspin = 8``) and the neutral-atom terms, together with their derivatives with respect to
the interatomic vector, which give the corresponding forces and stress.

``TwoCenterEvaluator`` returns a ``TwoCenterMatrices`` object on the edges of the ``SparsePattern``. The
integrals are evaluated in C++ from radial tables in reciprocal space (SIESTA's ``matel`` approach); with
symmetry on, only symmetry-inequivalent pairs are computed.

.. include:: ../_generated/api/integrals.inc
