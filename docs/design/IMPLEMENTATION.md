# Implementation and verification — 12 November 2026

## Delivered

- `knowledge.py`: read-only index API with schema checks, exact IDs/qualified
  names, ambiguity reporting, source hashes, bounded paths, component boundary
  reports, and index-to-index source/symbol/reference comparison.
- `index_docs.py`: corrected signatures, actual compound entities, reference-ID
  resolution, explicit unresolved/ambiguous references, relative paths, Git/source/
  XML/indexer provenance, and enriched JSON exports. Builds publish whole
  generations atomically through `current`; legacy root artifacts are untouched.
- `rebuild_index.py`: fresh, temporary Doxygen generation outside the worktree;
  source hashes checked across generation/indexing; effective configuration saved
  and hashed. Inputs/exclusions and reduced reference coverage are explicit.
- `siesta_mcp.py`: twelve structured read-only tools, including extraction reports,
  index comparison, and evidence discovery/retrieval. Each request pins one index
  generation. A later request sees a newly published generation.
- `validation.py`: manifest-driven baseline/candidate execution, isolated input
  copies, executable/artifact/input hashes, logs, runtime controls, finite numeric
  outputs, explicit units/tolerances, timeout/process-error handling and saved
  reports. It invokes no shell and is separate from the MCP read interface.
- `pilots/reclat.py`: actual external RECLAT builds and independent reciprocal-cell
  checks. Optional candidate worktree supports future implementation comparisons.
- Portable examples, dependency metadata, regression/transport checks and contracts
  for subsequent compiler-analysis and profiling adapters.

## Local artifacts and evidence

`indexes/default/current/docs.db` is an explicit historical XML import with
`namespacem__diag.xml` excluded because it has mismatched XML tags. It contains
8,127 entities, 14,118 resolved references and 19 unresolved/ambiguous references.
The higher entity count than the legacy index includes compound entities.
The import has **unverified XML/source alignment**. Its source ranges cannot be
used through `get_source`; provenance remains visible on navigation results.

`indexes/reclat/current/docs.db` was freshly generated from `Src/reclat.f` with
Doxygen 1.18.0. This is a one-routine subset, not a full-code graph. Source-range
reads work against the recorded source hashes.

Full-tree fresh Doxygen generation crashed with signal 11 in four configurations:
references enabled/disabled, with/without the BdG development's explicit exclusion
of `Src/extrae_eventllist.F90`. This remains unresolved. Diagnostic output is in
`indexes/default/doxygen-last.log`. No successful index was replaced by a failed
build. External XML was not relabelled as fresh to bypass this problem.

The physical pilot report is `reports/comparison-k3jv32ai/report.json`, with build
sources/logs under `reports/reclat-build-kl82kkwv`. All six cases passed: orthogonal,
triclinic and left-handed cells, each with IOPT=0 and IOPT=1. It compares -O0 and
-O2 builds of the same current `Src/reclat.f`; **no modernization has been applied**.
Each executable also passes an independent `A^T B = c I` residual bound. The saved
report identifies actual executables, copied source, adapter, compiler command,
inputs and measured differences. These tolerances are pilot-specific.

## Verification

- Regression suite covers symbol ambiguity, reference-ID precedence, unresolved
  references, correct signatures/FTS, source confinement/freshness, duplicate
  basenames, new/missing files, bounded paths and isolated nodes, failed-build
  publication, export integrity, legacy rejection, unverified XML rejection,
  comparison contracts and executable failure/timeout/missing-output handling.
- Real stdio MCP checks pass against both the historical import and fresh subset,
  including the intended source-read refusal for unverified XML.
- Synthetic comparison example passes; six actual Fortran pilot cases pass.

## Remaining project work

The database is not a semantic Fortran analyzer. Global state, aliasing, allocation,
MPI/I/O side effects, initialization dependencies and GPU data lifetime remain
unknown in extraction reports. The versioned adapter envelope is specified in
`EVIDENCE.md`, but semantic/profiling adapter ingestion is not implemented.

The next technical priorities are resolving/replacing full-tree Doxygen
extraction, adding compiler/parser-derived analysis for actual build variants,
and choosing the first production library component with a reviewed physical
validation matrix. Full SIESTA SCF/forces/stress/spectral/MPI comparisons and GPU
measurements have not been run. The harness is ready for adapters and executable
identities; scientific coverage and acceptance tolerances must be established for
each component. Whole-program physical equivalence is not established by this work.
