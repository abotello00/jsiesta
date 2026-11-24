Testing and validation utilities
================================

Tools used by the test suite to compare jsiesta with SIESTA and to check physical identities. They are part of
the package so that validation runs can be reproduced, but they are not needed for calculations.

``siesta_adapter`` writes SIESTA inputs for a jsiesta structure, runs a SIESTA binary (or replays a cached run)
and parses its output; ``external_refs`` resolves the paths of binaries and pseudopotential sets;
``strain_fd`` computes finite-difference stresses; ``run_case`` is the case runner used by the validation
manifests; ``profile_phase3`` and ``parity_html`` produce timing and comparison reports.

.. include:: ../_generated/api/testing.inc
