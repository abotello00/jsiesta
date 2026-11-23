# Refactoring support implementation plan

The objective is to support an incremental SIESTA modernization into a modular,
GPU-capable library while preserving its physical models and validated numerical
behavior. This repository supplies evidence to reviewers and agents; it does not
certify physical equivalence or perform the SIESTA refactoring itself.

## 1. Reliable navigation (implement now)

- One reusable MCP server, separate immutable index generation per worktree.
- Correct signatures and qualified names; resolve symbols by ID or qualified name.
- Prefer XML reference IDs, retain ambiguous/unresolved references explicitly,
  label Doxygen relationships as references rather than proven calls.
- Record source root, revision, dirty status, source hashes, XML hashes, tool
  versions and extraction coverage. External XML has unverified source provenance.
- Rebuild in a temporary generation, publish through an atomic `current` symlink;
  never delete the last working database before a successful build.
- Reject stale source slices. Expose index health and relationship limitations.
- Acceptance: tests for duplicate names, unresolved edges, stale files, bounded
  paths, failed builds preserving the active generation, and MCP transport.

## 2. Numerical evidence harness (implement generic mechanism now)

- Manifest-driven baseline/candidate commands with isolated working directories,
  input-tree hashes, executable hashes, revision/build identity supplied explicitly,
  environment, timing, logs, and finite numeric observable JSON outputs.
- Explicit per-observable absolute/relative tolerances and units; exact comparison
  is available by setting both tolerances to zero. Missing/extra/nonfinite values
  fail; process failures cannot produce a passing comparison.
- Store structured reports outside the index and expose bounded report retrieval
  over MCP. Reports are evidence, never a blanket equivalence claim.
- Acceptance: synthetic passing/failing executables, timeout and malformed output.

## 3. Architecture and performance evidence (interfaces now; analysis later)

- Implement index-to-index structural comparison and a component dependency report
  including incoming/outgoing references and explicit unknown analyses.
- Define a versioned evidence format for compiler/static-analysis, profiler,
  physical-identity and finite-difference results, tied to revision and file hashes.
- Do not infer complete side effects, ownership or GPU suitability from regexes.
- Next implementation: select a compiler/parser adapter against actual build
  configuration; capture module USE, reads/writes, allocation, I/O, MPI, pointer
  uncertainty and preprocessor variants. Validate on manually reviewed fixtures.
- Next implementation: collect measured CPU/GPU kernel time, allocation, transfer,
  synchronization and communication using the project's actual profiling tools.

## 4. First physical pilot (requires a selected component and runnable builds)

1. Choose a bounded component and its supported physical/configuration cases.
2. Freeze reference executable/build/input manifests and numerical contracts.
3. Add SIESTA observable extraction adapters and tests of independent identities:
   energies, forces/stresses, derivative checks, symmetry and serial/MPI cases as
   appropriate. Define tolerances with the scientific owner.
4. Review dependencies and global state; introduce an explicit context/interface.
5. Preserve original computational behavior, compare every transformation, then
   benchmark. Attempt GPU migration only after data lifetime/ownership is explicit.
6. Repeat for larger components. Keep original and candidate available until the
   relevant numerical and physical contracts pass.

## Delivery boundary

Steps 1–3 deliver usable infrastructure in this repository. Whole-program semantic
analysis, a production SIESTA test matrix, GPU profiling, and the physical pilot
need iterative work with real builds and scientific acceptance criteria. Their
absence must be visible in tool output, rather than represented as zero effects.

## Execution status

Implemented reliable index generations, navigation/health/structural comparisons,
the generic numerical harness, evidence retrieval, and adapter contracts. A real
RECLAT pilot additionally compares -O0/-O2 builds across six cases and enforces the
reciprocal-lattice identity. Regression and real MCP transport checks pass.

Full-tree fresh Doxygen extraction still crashes with installed 1.18.0. The
historical import is explicitly unverified and cannot supply source-range reads;
a fresh one-routine subset is available. Compiler semantic analysis, profiling
adapters and a production SIESTA validation matrix remain follow-up work rather
than inferred capabilities. See IMPLEMENTATION.md for exact artifacts/results.
