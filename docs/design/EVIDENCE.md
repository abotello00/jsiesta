# Evidence contracts

Numerical comparisons use schema version 1 in `validation.py`. An adapter runs in
an isolated copy of the case inputs and writes `observables.json`: an object of
finite numbers, nested objects or nonempty numeric arrays. Each flattened leaf
must have a named tolerance and unit, e.g. `forces[0][1]`. The comparison is
`abs(candidate-baseline) <= atol + rtol*abs(baseline)`. Extra, missing, boolean,
nonfinite, or empty values fail. Units are declarations, not automatic conversions.
Adapters must perform any needed conversion explicitly. Commands use argument
arrays, never a shell. Add source scripts, shared libraries and other relevant
build artifacts to `artifacts`; the executable itself is always hashed.

Identity labels are supplied by the operator. Executable hashes, input hashes,
exit status, runtime controls, logs and observations are measured. For interpreted
adapters the interpreter is not enough: list the adapter in `artifacts`.
A future SIESTA adapter should record compiler flags, linked library versions,
MPI rank/thread layout, executable provenance and physical configuration in the
variant's `identity` object. The current runner inherits the environment and is
not a hermetic build/execution system. It times whole commands, not GPU kernels.

Scientific acceptance must include independent identities and derivative tests,
not just agreement with one old executable. An adapter can export residuals as
observables, but comparison to a baseline alone does not enforce a physical bound.
A dedicated physical-contract adapter must check that bound and exit unsuccessfully
on violation; retain its logs. Baseline and candidate must both pass it.

## Future static-analysis and profiler adapter envelope

Do not put unsupported state or GPU conclusions into the documentation graph.
Adapters should produce the following versioned JSON envelope before integration:

```json
{
  "schema_version": 1,
  "kind": "static_analysis",
  "source": {
    "root": "/absolute/worktree",
    "commit": "full commit hash",
    "files_sha256": {"Src/example.F90": "sha256"}
  },
  "producer": {"name": "adapter name", "version": "version"},
  "configuration": {"build_options": [], "preprocessor_defines": []},
  "coverage": {"status": "partial", "unknowns": ["procedure pointer targets"]},
  "observations": []
}
```

Static observations should identify the symbol, source range, read/write/call/
allocation/I/O/MPI relation, resolution method and uncertainty. A profiler envelope
uses `kind: profiling`, records executable/input hashes and rank/thread/device
configuration, and provides measurements with units and repetition counts. GPU
transfer, synchronization and residency claims require actual measurements or
reviewed dataflow analysis. These envelopes are a design contract; automatic
import and semantic analysis are not implemented yet.
