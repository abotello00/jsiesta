Development
===========

jsiesta is developed so that anyone can pick it up without the history of how it got here: the design is
written down, every decision is logged with its reason and its measured effect, and every physics term is tied
to a test against an external reference. This section collects that material.

.. list-table::
   :widths: 30 70

   * - :doc:`contributing`
     - Set-up, checks to run, conventions for code, docstrings and tests
   * - :doc:`../_generated/dev_status`
     - Current state: what is validated and by which test, measured performance, open issues
   * - :doc:`../_generated/dev_decisions`
     - Decision log: why each choice was made and what it changed
   * - :doc:`../_generated/dev_work_log`
     - Chronological work log
   * - :doc:`../_generated/dev_architecture`
     - Dependency, lineage and validation-coverage diagrams generated from the knowledge graph

Design documents (specification, evidence contract, GPU performance study) are in
`docs/design <https://github.com/abotello00/jsiesta/tree/main/docs/design>`_ of the repository.

How results are validated
-------------------------

A term counts as *validated* only when a test compares it with SIESTA (or another independent code) **and** an
independent identity holds: hermiticity, trace rules, finite differences of the energy, idempotence of
projections. A test of the code against itself is called a self-consistency test and is reported as such. The
status page lists every validated item with its test and tolerance, and what has not been checked.

Cross-code tests need SIESTA binaries and pseudopotential sets outside the repository; without them those tests
are skipped and the test run prints the list of what did not run.

.. toctree::
   :hidden:
   :maxdepth: 1

   contributing
   ../_generated/dev_status
   ../_generated/dev_decisions
   ../_generated/dev_work_log
   ../_generated/dev_architecture
