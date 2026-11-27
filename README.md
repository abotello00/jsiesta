# jsiesta

[![CI](https://github.com/abotello00/jsiesta/actions/workflows/ci.yml/badge.svg)](https://github.com/abotello00/jsiesta/actions/workflows/ci.yml)
[Documentation](https://abotello00.github.io/jsiesta/)

A clean-room JAX + C++/CUDA implementation of SIESTA's static Kohn–Sham method: total energies, atomic forces and the
stress tensor with confined numerical atomic orbitals, Kleinman–Bylander pseudopotentials from PSML files, the
neutral-atom potential and a uniform real-space grid. Every physics term is checked against SIESTA.

## Use it like SIESTA, or from Python

```bash
make venv                              # .venv with pinned dependencies; installs the `jsiesta` command
LIBXC_ROOT=/path/to/libxc make ffi     # compiled kernels (cpp/build/libjsiesta_ffi)
source .venv/bin/activate

cd examples/si_bulk
jsiesta si.fdf --pseudo-dir ../pseudos > si.out      # an .fdf input and <Label>.psml pseudopotentials
```

The run prints a SIESTA-style report (basis, SCF table, energies, forces, stress) and writes `si.XV`, `si.DM`,
`si.FA`, `si.EIG`, `si.bands` and `si.json`. Input keywords that jsiesta does not use are listed in a warning, never
dropped silently. The same calculation from Python:

```python
import jax_siesta as js

results = js.Calculation.from_fdf("si.fdf", pseudo_dir="../pseudos").run()
print(results.energy_ev, results.forces_ev_ang, results.pressure_gpa)
results.band_structure([("L", (.5, .5, .5)), ("Γ", (0, 0, 0)), ("X", (.5, 0, .5))]).plot()
```

| | |
|---|---|
| [Getting started](https://abotello00.github.io/jsiesta/getting_started/index.html) | Installation and a first calculation |
| [User guide](https://abotello00.github.io/jsiesta/user_guide/index.html) | The command, supported FDF keywords, Python interface, basis sets, restart files, differences from SIESTA |
| [Tutorials](https://abotello00.github.io/jsiesta/tutorials/index.html) | Eight notebooks with exercises and solutions (`docs/tutorials`), also as plain Python scripts |
| [API reference](https://abotello00.github.io/jsiesta/api/index.html) | Organised by stage of the calculation, each with an introduction |
| [`examples/`](examples/) | Ready-to-run inputs and PseudoDojo pseudopotentials for H, C, O, Si |

Scope: static single points only (no relaxation or dynamics in the code itself; forces and stress are available to
external optimisers), PSML pseudopotentials only.

## For developers

- Specification: [docs/design/IMPLEMENTATION_PLAN.md](docs/design/IMPLEMENTATION_PLAN.md)
- Project state, validation status and open issues: [MEMORY.md](MEMORY.md)
- Decisions: [logs/DECISIONS.md](logs/DECISIONS.md); history: [logs/WORK_LOG.md](logs/WORK_LOG.md)
- Authorship: [AUTHORS_CONTRIBUTORS.md](AUTHORS_CONTRIBUTORS.md)

## What it does

| Area | Capability |
|---|---|
| Spin | `nspin` 1, 2, 4 (non-collinear) and 8 (spin–orbit) |
| Exchange–correlation | LDA, GGA and meta-GGA (r2SCAN) through Libxc; hybrid functionals (HSE06, PBE0, B3LYP) for all spin modes, with forces and stress; DFT-D3(BJ) and rVV10 dispersion |
| Symmetry | spglib space and magnetic groups, irreducible k-points, symmetrised density and energy-density matrices, symmetry-commensurate mesh |
| SCF | Tikhonov–Pulay, Broyden and ADIIS mixers, adaptive trust-region controller, mandatory post-SCF `DM_out` evaluation, Harris energy |
| Observables | Forces and stress at fixed (D, E) by automatic differentiation of the discrete energy, Mulliken moments |
| Basis | SIESTA `.ion` import, PAO generator (`PAO.BasisSize` SZ to TZ2P, `PAO.Basis`), basis-enthalpy optimiser |
| Interface | `jsiesta` command for SIESTA `.fdf` inputs, `Calculation` / `Results` Python API, band structures, `.FA` / `.EIG` / `.bands` output |
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
  hfx/               exact exchange of hybrid functionals (Gaussian fits, four-centre integrals)
  symmetry/          spglib wrapper, Wigner-D, density-matrix projection, grid symmetrisation
  scf/               eigensolver, mixers, adaptive controller, SCF driver
  observables/       post-SCF pass, energies, forces and stress
  io/                FDF reader, case.json, observables.json, SIESTA restart files (.XV, .DM, .nc)
  app/               user interface: Calculation/Results, the jsiesta command, report and output files
  testing/           SIESTA adapters, external references, physical-identity checks, profiling
cpp/                 C++ (include/, src/) and CUDA (cuda/) kernels behind the XLA FFI
kg/                  knowledge-graph builder and read-only MCP server
docs/                Sphinx + Doxygen sources: user guide, tutorials (notebooks), API sections, theory, design
examples/            SIESTA-style inputs and PSML pseudopotentials
tests/               unit, identity, SIESTA-parity and GPU tests; fixtures and case manifests
logs/, MEMORY.md     decision log, work log, live project state
```

Every pipeline stage (`BasisAndPseudos`, `TwoCenterIntegrals`, `SymmetryProjector`, `RealSpaceGridOps`,
`ExchangeCorrelation`, `DispersionModel`, `EigenSolver`, `SCFMixer`, `ObservableEvaluator`) sits behind a
`typing.Protocol` in `jax_siesta/core/protocols.py`. New methods are added as new implementations of a protocol, not
by editing the core kernels.

## Development quick start

```bash
make venv          # .venv from requirements-dev.txt (uv, Python 3.11), editable install
make ffi           # cpp/build/libjsiesta_ffi with Libxc (from /opt/etsf or LIBXC_ROOT)
make check         # ruff, mypy, Doxygen + Sphinx -W, knowledge graph, pytest with JUnit XML in reports/
make tutorials     # execute docs/tutorials/*.py into notebooks (about 12 minutes on a laptop)

# the case runner used by the validation manifests: energies, forces and stress go to observables.json
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

`make docs` builds the documentation site: hand-written guides, the tutorials (notebooks stored with their output,
rendered by myst-nb), and the API reference generated from the docstrings (Sphinx) and the C++/CUDA headers
(Doxygen), split into one section per stage of the calculation. Every public
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
