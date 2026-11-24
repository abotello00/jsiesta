# jsiesta

[![CI](https://github.com/abotello00/jsiesta/actions/workflows/ci.yml/badge.svg)](https://github.com/abotello00/jsiesta/actions/workflows/ci.yml)
[Documentation](https://abotello00.github.io/jsiesta/)

A clean-room JAX + C++/CUDA implementation of SIESTA's static Kohn–Sham method: total energies, atomic forces and the
stress tensor with confined numerical atomic orbitals, Kleinman–Bylander pseudopotentials from PSML files, the
neutral-atom potential and a uniform real-space grid. Every physics term is checked against SIESTA.

- Specification: [docs/design/IMPLEMENTATION_PLAN.md](docs/design/IMPLEMENTATION_PLAN.md)
- Project state, validation status and open issues: [MEMORY.md](MEMORY.md)
- Decisions: [logs/DECISIONS.md](logs/DECISIONS.md); history: [logs/WORK_LOG.md](logs/WORK_LOG.md)
- Authorship: [AUTHORS_CONTRIBUTORS.md](AUTHORS_CONTRIBUTORS.md)

## What it does

| Area | Capability |
|---|---|
| Spin | `nspin` 1, 2, 4 (non-collinear) and 8 (spin–orbit) |
| Exchange–correlation | LDA, GGA and meta-GGA (r2SCAN) through Libxc; DFT-D3(BJ) and rVV10 dispersion |
| Symmetry | spglib space and magnetic groups, irreducible k-points, symmetrised density and energy-density matrices, symmetry-commensurate mesh |
| SCF | Tikhonov–Pulay, Broyden and ADIIS mixers, adaptive trust-region controller, mandatory post-SCF `DM_out` evaluation, Harris energy |
| Observables | Forces and stress at fixed (D, E) by automatic differentiation of the discrete energy, Mulliken moments |
| Basis | SIESTA `.ion` import, PAO generator, basis-enthalpy optimiser |
| Restart files | SIESTA `.XV`, `.DM` and `.nc` (`CDF.Save`) read and written: start from a SIESTA geometry, density matrix or Hamiltonian, and leave files SIESTA can continue from |
| GPU | CUDA kernels for the orbital–grid operations and their adjoints, device LDA/GGA XC, grid plans streamed from host memory when they do not fit |

The validation status of each item, with the test that checks it against an external reference, is kept in
[MEMORY.md](MEMORY.md).

## Layout

```
jax_siesta/          Python/JAX package
  core/              immutable PyTrees (Structure, BasisData, SparsePattern, SCFState, Observables), protocols, FFI loader
  species/           PSML reader, .ion basis, PAO generator, basis-enthalpy optimiser
  integrals/         two-centre integrals (S, T, V_KB, V_NA) and their derivatives
  grid/              mesh selection, FFT Poisson, orbital-grid FFI wrappers
  xc/                Libxc evaluation, DFT-D3, rVV10
  symmetry/          spglib wrapper, Wigner-D, density-matrix projection, grid symmetrisation
  scf/               eigensolver, mixers, adaptive controller, SCF driver
  observables/       post-SCF pass, energies, forces and stress
  io/                FDF subset, case.json, observables.json, SIESTA restart files (.XV, .DM, .nc)
  testing/           SIESTA adapters, external references, physical-identity checks, profiling
cpp/                 C++ (include/, src/) and CUDA (cuda/) kernels behind the XLA FFI
kg/                  knowledge-graph builder and read-only MCP server
docs/                Sphinx + Doxygen sources, theory notes, design documents, generated diagrams
tests/               unit, identity, SIESTA-parity and GPU tests; fixtures and case manifests
logs/, MEMORY.md     decision log, work log, live project state
```

Every pipeline stage (`BasisAndPseudos`, `TwoCenterIntegrals`, `SymmetryProjector`, `RealSpaceGridOps`,
`ExchangeCorrelation`, `DispersionModel`, `EigenSolver`, `SCFMixer`, `ObservableEvaluator`) sits behind a
`typing.Protocol` in `jax_siesta/core/protocols.py`. New methods are added as new implementations of a protocol, not
by editing the core kernels.

## Quick start

```bash
make venv          # .venv from requirements-dev.txt (uv, Python 3.11)
make ffi           # cpp/build/libjsiesta_ffi with Libxc (from /opt/etsf or LIBXC_ROOT)
make check         # ruff, mypy, Doxygen + Sphinx -W, knowledge graph, pytest with JUnit XML in reports/

# one single point; energies, forces and stress go to observables.json in the current directory
JAX_ENABLE_X64=True .venv/bin/python jax_siesta/testing/run_case.py \
    tests/manifests/cases/si_bulk/case.json --psml-dir tests/fixtures/psml
```

## Restart files (SIESTA `.XV`, `.DM`, `.nc`)

`jax_siesta.io` reads and writes the three files SIESTA uses to continue a calculation, converting the sparsity
layout (supercell column indices), the spin components and the units (Ry) in both directions.

```bash
# write <label>.XV, <label>.DM and <label>.nc from the final state
JAX_ENABLE_X64=True .venv/bin/python jax_siesta/testing/run_case.py case.json --write-restart out --label si

# start from them: geometry from the .XV, density matrix from the .DM, Hamiltonian from the .nc
JAX_ENABLE_X64=True .venv/bin/python jax_siesta/testing/run_case.py case.json \
    --read-xv out/si.XV --read-dm out/si.DM --read-nc out/si.nc
```

```python
from jax_siesta.io import read_dm, read_nc_restart, structure_from_xv, write_restart_files

structure = structure_from_xv("si.XV", template=structure)       # cell and positions from SIESTA
dm0 = read_dm("si.DM", pattern, config.nspin)                    # any nsc, any sparsity pattern, any spin mode
dm0, h0, ef = read_nc_restart("si.nc", pattern, config.nspin)    # DM and H (Hartree) from a CDF.Save file
state, mesh = run_scf(structure, basis, pattern, two_center, kmesh, config, initial_dm=dm0, initial_h=h0)
write_restart_files("out", "si", structure, basis, pattern, two_center.overlap, state, config)
```

| File | jsiesta reads | jsiesta writes | SIESTA reads it with |
|---|---|---|---|
| `<label>.XV` | cell and positions (species data from the input structure) | yes, zero velocities | `MD.UseSaveXV T` |
| `<label>.DM` | initial density matrix, rescaled to the electron count | yes | `DM.UseSaveDM T` / `File.DM.Init` |
| `<label>.nc` | `H` (first input Hamiltonian), `DM`, geometry | yes: geometry, `S`, `H`, `DM`, `EDM`, `Ef`, forces, stress | not as a restart: SIESTA has no reader for `H`; sisl and other tools read it |

The species index in the `.XV` file is the position of the species in `structure.species_labels`, so the SIESTA input
must list `ChemicalSpeciesLabel` in that order; both codes must use the same basis (SIESTA's `.ion` tables). The
`.nc` file needs the optional `netCDF4` package (`pip install .[netcdf]`).

For the GPU build, configure with `cmake -S cpp -B cpp/build -DJSIESTA_USE_LIBXC=ON -DJSIESTA_ENABLE_CUDA=ON`
(CUDA 12). It runs the same algorithms as the CPU build; `tests/test_phase3_cuda.py` checks the two against each
other.

## Validation against SIESTA

Cross-code tests need SIESTA builds and pseudopotential sets outside this repository. Their keys are listed, empty,
in `tests/external_refs.toml`; set them per machine in `tests/external_refs.local.toml` (git-ignored) or with
`JSIESTA_<KEY>` environment variables (e.g. `JSIESTA_SIESTA_BIN`). Tests whose references are missing are skipped;
`make test-refs` makes them fail instead. `make refs-cache` stores the SIESTA runs in `refs_cache/`, so the same tests
can replay them on machines without SIESTA (`JSIESTA_SIESTA_REF_CACHE`). DFT-D3 needs the optional `dftd3` package
(`pip install .[dispersion]`).

## Documentation and knowledge graph

`make docs` builds the API reference from the docstrings (Sphinx) and the C++/CUDA headers (Doxygen). Every public
entity documents its theory, algorithm, invariants and SIESTA mapping, and the build fails on warnings. The CI
workflow (`.github/workflows/ci.yml`) runs ruff, mypy and this build on every push and pull request, and publishes
the result from `main` at <https://abotello00.github.io/jsiesta/>. The physics tests need the C++ library, Libxc
and the SIESTA references, so they run with `make check` outside the CI. `make kg`
rebuilds the knowledge graph in `kg/current/` (code entities, theory notes, decisions, validation edges) and the
diagrams in `docs/architecture/`; `make kg-results` adds the pass/fail status of the last test run.
`python kg/server_mcp.py` serves the graph to MCP clients (`resume_session_context`, `trace_physics_lineage`,
`check_staleness`).

## Vendored tools

`knowledge.py`, `index_docs.py`, `rebuild_index.py`, `siesta_mcp.py` and `validation.py` come from the `siesta-mcp`
project. They index a SIESTA source tree for code navigation (`rebuild_index.py <siesta-source> indexes/default`,
then `siesta_mcp.py indexes/default/current/docs.db`) and run manifest-driven numerical comparisons
(`validation.py <manifest.json>`); `test_mcp.py` is a stdio smoke test of the MCP server. The evidence contract is
[docs/design/EVIDENCE.md](docs/design/EVIDENCE.md).

## Design documents

[docs/design/](docs/design/) holds the specification (`IMPLEMENTATION_PLAN.md`), the plan and report of the vendored
tools (`PLAN.md`, `IMPLEMENTATION.md`), the evidence contract (`EVIDENCE.md`) and the GPU performance study with its
assessment plan (`known_bottlenecks.md`, `bottleneck_assessment_plan.md`).
# jsiesta
