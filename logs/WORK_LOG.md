# Append-Only Engineering Work Log

## 2026-11-18 — Phase 0 Initialization
- **Environment Check:** Verified Python 3.11 `.venv` (`jax 0.4.38`, `jaxlib 0.4.38`, `numpy 2.2.6`, `scipy 1.14.1`, `spglib 2.8.0`, `lxml 6.1.3`, `pytest 9.0.2`, `mcp 1.27.0`, `ruff 0.15.7`), CMake 4.4.4, AppleClang, Doxygen 1.18.0, Sphinx 9.1.0.
- **Existing Harness Check:** Ran `.venv/bin/python -m pytest tests/test_tools.py -q` -> 11 passed.
- **Scaffolding Created:** `pyproject.toml`, `jax_siesta/core/{types,protocols,ffi}.py`, subpackage `__init__.py` modules, `cpp/{CMakeLists.txt,include/jsiesta/*.hpp,src/ffi_registry.cpp,cuda/kernels.cuh}`.
- **C++ FFI Verification:** Built `cpp/build/libjsiesta_ffi.dylib` cleanly and verified `jax.extend.ffi` round-trip (`jsiesta_ffi_ping`).
- **Phase 0 Gate Passed:** Built Doxygen XML (`docs/Doxyfile`), Sphinx HTML with `-W` (`docs/_build/html`), progressive KG (`kg/current/docs_kg.db`), 5 PSML v1.1 + `.ion.json` fixtures (`H, C, O, Si, Co`), frozen 9-case `tests/manifests/phase0_matrix.json`, `ruff check` (0 issues), and `pytest` (`17 passed` including `kg/server_mcp.py` stdio MCP tests).
- **Next:** Phase 0 review by checker agent, then proceed to Phase 1 (Core Non-Symmetrized Static SCF Engine).

## 2026-11-18 — Phase 1 Complete (Core Non-Symmetrized Static SCF Engine)
- **Species & Two-Center (`tests/test_phase1_species_and_two_center.py` -> 3 passed):** Implemented PSML v1.1-only `PSMLBasisLoader` (rejecting `.psf`, sequential `pao_iz`, SOC $j$-projectors, NLCC, $V^{\text{NA}}$), `build_sparse_pattern`, and C++ Fourier-Bessel `two_center.cpp` (`S, T, P_kb, E_NA_pair` + analytical gradients, SOC Clebsch-Gordan `V_kb_so`).
- **Grid & XC (`tests/test_phase1_grid_xc.py` -> 11 passed):** Implemented C++ XLA FFI kernels (`jsiesta_rhoofd_f64`, `jsiesta_vmat_f64`, `jsiesta_dfscf_grad_f64`, `jsiesta_na_grid_f64`, `jsiesta_na_grid_grad_f64`, `jsiesta_xc_eval_f64` with `JSIESTA_HAS_LIBXC` switch + `Dual<5>` analytical LDA/PBE/PBEsol), 3D FFT Poisson solver & virial stress, `DFTD3Dispersion`, and `RVV10Dispersion`. Refactored `build_active_grid_cache` to flat vectors + prefix counts and added `vmat_differentiable` for zero-packing JITted SCF evaluation.
- **SCF, Observables, IO & Validation (`tests/test_phase1_scf_and_forces.py` -> 9 passed):** Implemented `eigensolver` (`nspin=1,2,4,8`), `descriptors`, `mixer` (decoupled spin preconditioning, deterministic 1x1 SVD, Tikhonov-Pulay/ADIIS/Broyden), `controller` (adaptive trust region + `anneal_smearing` guard), `driver` (`jit` + `SymmetryProjector` hooks), `observables.forces` (mandatory post-SCF `DM_out` pass, fixed-`(D, E)` variational forces & stress), `io.fdf`, and `testing.run_case`. Fixed factor-of-2 symmetry in `_kb_forces_and_stress` verified against central FD (`diff = -7.4e-5` Ha/Bohr on H2O force, `-6.3e-6` Ha/Bohr^3 on stress). Verified mixer-endpoint parity (`h2o_gamma`), Si bulk, and Co HCP (`nspin=2,4,8`).
- **Next:** Rebuild Doxygen + KG + Sphinx `-W`, pass Phase 1 checker audit, and start Phase 2 (4-Step Native Symmetry Roadmap).

## 2026-11-18 — Phase 1 Audit Fixes & Phase 2 Complete (4-Step Native Crystal & Magnetic Symmetry)
- **Phase 1 Audit Fixes:** Fixed `stress_grid` position-scaling term in `StaticObservableEvaluator` via direct differentiation wrt `(displacement, homogeneous strain)`, preserved complex `Im(D_uu)/Im(D_dd)` for `nspin=8` in `GeneralizedEigenSolver`, completed `JSIESTA_HAS_LIBXC` evaluation path in `cpp/src/xc_ffi.cpp`, made `anneal_smearing` monotonically non-increasing, and fixed `jsiesta_eval_na_pair` in `cpp/src/two_center.cpp` by separating analytic $Z_1 Z_2 / R$ from the smooth Hankel density-overlap integral to eliminate $k_{\max}=30$ Gibbs ringing. Verified 40/40 tests passing.
- **Phase 2 Step 1 (`jax_siesta/symmetry/spacegroup.py`):** Implemented `extract_space_group` supporting crystallographic & Shubnikov Type I--IV groups (`spglib 2.8.0`), open-boundary & rational-translation filtering + group closure, `build_ibz_kpoint_mesh`, `choose_commensurate_mesh_dimensions`, and bijective `precompute_grid_permutations` (`tests/test_phase2_step1_spacegroup.py`: 9 passed).
- **Phase 2 Step 2 (`jax_siesta/symmetry/projection.py`):** Implemented `build_symmetry_closed_pattern`, 16-point Fibonacci least-squares Wigner-$D^l$ (`max ||D D^T - I||_inf ~ 5e-15`), Shepperd $SU(2)$ spinor maps, and compacted COO `ProjectionTable` executed via `jax.ops.segment_sum` for `nspin=1,2,4,8` (`tests/test_phase2_step2_projection.py`: 5 passed verifying idempotency, trace conservation, `[H_0, U_S]=0`, and spin-moment covariance).
- **Phase 2 Steps 3 & 4 (`jax_siesta/symmetry/engine.py`):** Implemented `NativeSymmetryEngine` (`symmetrize_grid` via `jax.lax.scan`, `evaluate_irreducible_two_center` reducing `S, T` & `KB` evaluations to orbit representatives, `project_forces`, `project_stress`, `build_native_symmetry`), verifying IBZ vs full-BZ parity across all spin modes, FD stress/force, rigid 3D rotation & permutation invariance, and `validation.py` cross-verification (`tests/test_phase2_step3_and_4_symmetry.py`). Also normalized the Pulay residual Gram matrix by the active residual norm in `jax_siesta/scf/mixer.py` so `deterministic_svd_pinv` remains scale-invariant below `||F|| < 1e-5`.
- **Next:** Pass Phase 2 verification + checker audit, then execute Phase 3 (Performance Profiling & Optimization).

## 2026-11-18 — Phase 2 Audit Verification & Phase 3 Performance Profiling / Optimization
- **Phase 2 Checker Audit:** Verified all Phase 2 checker findings fixed (analytic $Z_1 Z_2 / R$ separation in `two_center.cpp`, scale-invariant normalized Pulay Gram matrix in `mixer.py`, `h_so_imag` addition before `sym_proj.project_matrix` and 6-channel complex DM packing in `driver.py`, shared `_pack_pao_tables`/`_eval_st_subset`/`_eval_kb_and_na_pair` helpers, `symprec`/`mag_symprec` forwarding).
- **Phase 3 Kernel & SCF Optimizations:**
  - Vectorized 3-center KB assembly and KB forces/stress via shared sorted `int64` edge lookup helper `_iter_kb_shell_pairs` (`two_center.py` + `forces.py`), fixing `nspin=8` non-symmetric complex SOC $dP_r + dP_c$ force/stress differentiation (`tests/test_phase3_profiling.py::test_nspin8_kb_forces_and_stress_against_finite_differences` `< 1e-9` vs central FD) and cutting `co_hcp_soc` `two_c` from `14.12s` to `1.75s` and `obs` from `52.10s` to `8.76s`.
  - Implemented cached `GridKernelPlan` + dense atom-image/atom-pair blocks + `std::thread` parallelism (`OMP_NUM_THREADS` aware, signed-overflow & TLS memory cap guards) in `cpp/src/grid_ffi.cpp`, accelerating `co_hcp_collinear` warm `rhoofd` (`2.5383s -> 0.0299s` mean) and `vmat` (`2.6733s -> 0.0311s` mean) with identical SCF free energy (`17.47638918 Ha`).
  - Moved `TikhonovPulayMixer.step` into fixed-shape `@jax.jit` `_mixer_step_jit` (`jax_siesta/scf/mixer.py`) using array-only `_deterministic_svd_pinv_arrays` (zero host syncs inside JIT), cutting `co_hcp_collinear` `run_scf` to `12.64s` cold / `6.86s` warm (`reports/phase3_profiling.json`).
  - Added configurable `batched_generalized_eigh` k-batch sizing (`eigensolver.py`), CUDA tile selector & `rhoofd_tiled_kernel` scaffold (`cpp/cuda/{kernels.cuh,rhoofd_vmat.cu}`, `cpp/CMakeLists.txt` `JSIESTA_ENABLE_CUDA` with CXX-gated `-Wall -Wextra`), and compact Snellius `gpu_a100` backfill batch script (`scripts/snellius_a100_phase3.sbatch`).
- **Phase 3 Profiling & Audit Fixes (`reports/phase3_profiling.json`):** Fixed Phase 3 reviewer findings (unbiased warmup-matched mixer comparison, full JAXPR transfer audit across mixer/solver/grid, case JSON SHA-256 input hashes, unit/repetition fields, `nspin=8` `complex128` byte accounting), re-ran `jax_siesta/testing/profile_phase3.py --mode full` across 7 workloads, k-batch sizing (`max_dm_diff = 2.48e-15`), and 12 adaptive Tikhonov-Pulay vs fixed-weight Pulay (`w=0.05`) benchmarks (including 3 unseen systems), all converging with `endpoint_diff_ev <= 9.02e-8 eV`.
- **Next:** Rebuild `doxygen`, `kg/build_kg.py`, `sphinx-build -E -W`, `ruff`, and run final full `pytest`.

## 2026-11-19 — Static Libxc, Real SIESTA Parity (+ HTML Report & Timings), Sphinx Immaterial, & Snellius A100 CUDA Execution
- **Static External `libxc` Linkage (`cpp/CMakeLists.txt`, `cpp/src/xc_ffi.cpp`):** Linked `/opt/etsf/lib/libxc.a` cleanly (`has_libxc=True`, zero compiler warnings).
- **Real PSML v1.1 & `.ion.xml` Support (`jax_siesta/species/psml.py`):** Added namespace-tolerant PSML v1.1 parsing ($4\pi \rho_{\text{core}}$ normalization, `<grid>` prefix slicing, tail $-Z_{\text{val}}/r$ local potential, `scalar_relativistic` / `lj` projector selection) and SIESTA `.ion.xml` radial table loader ($R(r) = y_{\text{xml}} r^l$, Ry->Ha unit conversions, atomic charge initialization).
- **Real Reference SIESTA Parity & Mobile HTML Report (`jax_siesta/testing/{siesta_adapter.py,parity_html.py}`, `tests/test_siesta_parity.py`):**
  - Verified against the standard SIESTA build and the `siesta_spglib_full_impl` build on Si diamond bulk (`4x4x4`), strained/displaced Si (`3x3x3` via `validation.py`: 20/20 metrics PASSED, `5.64 s` SIESTA vs `28.64 s` `jax_siesta` cold JIT subprocess), and collinear HCP Co (`4x4x2`, `nspin=2`, spin moment `3.3750 mu_B` exact match).
  - Recorded per-variant wall times (`setup`, `scf`, `observables`, `wall_seconds`) into `reports/siesta_parity_benchmarks.json` and generated self-contained mobile HTML report `reports/siesta_parity_report.html`.
- **CUDA XLA FFI Implementation & Snellius A100 Deployment (`cpp/cuda/{kernels.cuh,rhoofd_vmat.cu}`, `jax_siesta/core/ffi.py`, `tests/test_phase3_cuda.py`):**
  - Implemented all 6 stream-ordered CUDA XLA FFI handlers (`rhoofd`, `vmat`, `dfscf_grad`, `na_grid`, `na_grid_grad`, `xc_eval`) registered simultaneously for `"cpu"` and `"CUDA"`.
  - Executed on Snellius `gpu_a100`: job `29450910` (`gcn10`, NVIDIA A100-SXM4-40GB) passed `14/14` (`tests/test_phase3_cuda.py` + `tests/test_phase3_profiling.py` in `387.96s`); split `scripts/snellius_a100_profile.sbatch` and completed job `29452210` (`gcn25`, `18:05` elapsed), retrieving `reports/phase3_profiling_a100.json`.
- **Sphinx Theme Update (`docs/conf.py`):** Switched `.venv` documentation build (`sphinx 8.2.3`) to `sphinx_immaterial 0.13.12` (`font=False`, dynamic `autodoc_mock_imports`) and verified `tests/test_phase0.py` passes with `-E -W`.


### 2026-11-19 — Meta-GGA + Non-Collinear + Spin-Orbit Coupling (SOC) Implementation & Docs Verification
- **Grid / XC / FFI / CUDA Implementation:**
  - `jax_siesta/io/fdf.py`, `jax_siesta/testing/siesta_adapter.py`: Extended element map (atomic numbers up to Bi), SCF/mixer/temperature/shift parsing, `initial_moments` fallback, and `Enl(+so)` / `Eso(nil)` output parsing for spin-orbit SIESTA references.
  - `jax_siesta/xc/functional.py`: Replaced global 3D FFT semilocal XC density gradients with SIESTA's 6th-order `nn=3` 7-point central finite-difference stencil (`_fd7_gradient_3d`) while keeping `_fft_gradient_3d` for `RVV10`; switched non-collinear/SOC local spin-frame rotation to differentiate smooth components and project onto frozen unit magnetization `ux, uy, uz = jax.lax.stop_gradient(...)`.
  - `cpp/src/xc_ffi.cpp`, `cpp/include/jsiesta/grid_xc_ffi.hpp`: Extracted shared `jsiesta::evaluate_mgga_libxc_host` with `xc_func_set_dens_threshold(&func, 1e-10)` and explicit `n_tot <= 1e-9` vacuum masking.
  - `cpp/cuda/kernels.cuh`, `cpp/cuda/rhoofd_vmat.cu`: Extended spline/solid-harmonic device helpers with second spatial derivatives, unified forward/adjoint/Hessian orbital-grid kernels across density and meta-GGA `tau` modes (`tauofd`, `vmat_tau`, `tau_dfscf_grad`), added `tau_core`/`tau_core_grad`, and added `jsiesta_xc_eval_mgga_f64_cuda` via async staging into `jsiesta::evaluate_mgga_libxc_host`. Built cleanly on Snellius (`CUDA/12.6.0` + `libxc-7.1.2`) in the `cpp/build` directory on Snellius.
- **Tests & Documentation Verification:**
  - `tests/test_mgga_soc.py`: 12/12 passed locally (CPU), covering `tauofd`/`vmat_tau`/`tau_core` FD, `R2SCAN01`/`R2SCAN`/`TPSS` derivatives + spin-rotation invariance, IBZ vs full-BZ parity for bulk Si and collinear/NC/SOC hcp Co, meta-GGA SCF forces/stress vs FD, and `pr_mgga` SIESTA parity (`mgga_forces`, `mgga_noncol`, `mgga_soc`).
  - Full local `pytest tests/ -v`: `test_phase0.py` (`6 passed`), `tests/test_mgga_soc.py` (`12 passed`), plus all Phase 1/2/3 CPU & `test_siesta_parity.py` & `test_tools.py` tests passed (`82 CPU tests passed`, `12 CUDA tests skipped` on macOS).
  - Documentation & KG: `doxygen docs/Doxyfile`, `kg/build_kg.py`, `sphinx-build -E -W -b html docs docs/_build/html`, and `ruff check jax_siesta tests` all passed with zero warnings/errors.
  - Snellius `gpu_a100`: Slurm job `29458646` submitted for `tests/test_phase3_cuda.py tests/test_mgga_soc.py -k "not pr_mgga" -v` (queued in `gpu_a100`).

### 2026-11-19 — Tier B 1D Radial PAO Generator, Basis Enthalpy Optimizer & CUDA Parity Fixes
- **Basis Generation & Optimization (`jax_siesta/species/{pao_generator,basis_opt,__init__}.py`, `jax_siesta/observables/forces.py`):**
  - Implemented self-contained `.psml` v1.1 1D logarithmic-grid radial PAO generator (`generate_species_from_psml`) with explicit per-zeta `rc` parabola fitting, Junquera soft confinement (`V0`, `ri`), ionic charge (`q`), and perturbative/eigenstate polarization without external `siesta` calls.
  - Implemented `BasisEnthalpyObjective`, `MappedBasisSpace`, float64-clipped `arccos` bijective parameter transforms with cascading `chain_lo`/`depends_on` bounds, `optimize_basis_enthalpy`, `default_opt_space_from_psml`, and `format_pao_basis_block`.
  - Added `compute_forces_stress: bool = True` fast path in `StaticObservableEvaluator.evaluate` (and fixed `oa` scope in `forces.py`).
- **CPU vs. CUDA XC & PW92 Parity Fixes (`cpp/include/jsiesta/grid_xc_ffi.hpp`, `cpp/src/xc_ffi.cpp`, `cpp/cuda/rhoofd_vmat.cu`, `tests/test_phase3_cuda.py`):**
  - Fixed PW92 spin correlation `fdd0` prefactor (`4.0 -> 8.0`) in both C++ CPU Dual-AD and CUDA kernels.
  - Shared `jsiesta::evaluate_xc_libxc_host` between CPU `XcEvalImpl` and CUDA `XcEvalCudaImpl` when `JSIESTA_HAS_LIBXC=1`; relaxed `tauofd` gradient parity check to `assert_allclose(..., rtol=1e-12, atol=1e-10)`.
- **Local Verification Results (all run post-edit):**
  - `cmake --build cpp/build -j 8`: clean build with 0 compiler warnings.
  - `.venv/bin/ruff check jax_siesta tests`: All checks passed.
  - `doxygen docs/Doxyfile && .venv/bin/python kg/build_kg.py && .venv/bin/sphinx-build -E -W -b html docs docs/_build/html`: built cleanly with zero warnings/errors.
  - Full local pytest (`task-3496` + `task-3447` + `task-3562`): all `test_phase0.py` (`6/6`), `test_tools.py` (`11/11`), `test_phase1_*` (`23/23`), `test_phase2_*` (`21/21`), `test_phase3_profiling.py` (`4/4`), `test_mgga_soc.py` (`12/12` incl. `pr_mgga`), and `test_basis_enthalpy.py` (`4/4` incl. real `Si.tm2.nrl.psml` SZ/DZ/DZP validation) PASSED (81 local tests passed; `test_phase3_cuda.py` CUDA tests skipped on macOS).
  - Snellius `gpu_a100`: rebuilt `libjsiesta_ffi.so` cleanly (`CUDA/12.6.0` + `libxc-7.1.2`); submitted verification jobs `29470466` (`test_phase3_cuda.py` + `test_phase3_profiling.py`) and `29470467` (`test_mgga_soc.py` + `test_basis_enthalpy.py`).

### 2026-11-20 — v0.1 repository and review remediation (Claude Code)
- **Repository:** created `jsiesta_v0.1` from the original tree (caches, builds, SIESTA leftovers and `review/`
  excluded), `git init`, neutral commit identity, one-line commit messages; authorship in `AUTHORS_CONTRIBUTORS.md`.
  Baseline of the imported tree: 83 passed, 3 failed (Phase 0 tests needing generated Doxygen XML / KG), 12 skipped.
- **Phase A (process):** `make check` (ruff, mypy, Doxygen + Sphinx -W, KG rebuild, pytest + JUnit);
  `requirements-dev.txt`; mypy clean on `jax_siesta/` and `kg/`; KG staleness separates handoff docs from code;
  tests no longer write into the source tree; external references from `tests/external_refs.toml` with
  `--require-refs`.
- **Physics:** DFT-D3 via simple-dftd3 (D-014), rVV10 rewritten and checked against QE (D-015), absolute total energy
  (D-016), radial-table fidelity + exact atomic neutral-atom term (D-017), SCF/XC/symmetry honesty fixes (D-018).
- **Measurements (all vs SIESTA 5.4.2 on SIESTA's mesh):** Si E_total 0.06 meV/cell and σ_xx ≤ 2.3e-5 eV/Å³ at
  100–400 Ry; Co hcp ΔE_kin −8 meV, ΔF 4e-6 eV/Å, ΔE_total ≈ 5 meV/atom; D3 energy < 2e-6 eV; rVV10 vs QE 3.5e-11 Ry.
- **Checks run:** `ruff check .`, `mypy`, `make docs`, KG rebuild (no undocumented public entities), targeted pytest
  (dispersion, parity, FD, phase 0/1/2 unit tests); full `make check` at the end of the session (see final entry).
- **Next:** see MEMORY.md open issues O-1 … O-7.

### 2026-11-20 — v0.1 final gate
- `make check`: ruff and mypy clean; Doxygen + Sphinx -W clean (after replacing a Markdown table in MEMORY.md that the
  RST converter cannot render); KG rebuilt with no undocumented public entities.
- Full pytest: 96 passed, 1 failed, 12 skipped (all `test_phase3_cuda.py`; no SIESTA/QE reference test skipped).
  The failure, `test_fdf_parser_and_validation_harness`, still requested the removed `mixer-adiis`; it now compares
  `mixer-tikhonov` with `mixer-pulay` and passes on its own (40 s). The full suite was not re-run after this test-only fix.
- Summary slides: `presentations/v0_1_changes.{tex,pdf}`.

### 2026-11-20 — Open issues O-1…O-7 plan, step 1: O-5 KG verification status (Claude Code)
- `kg/junit_ingest.py`, schema version 2, MCP `verification` states and `results_stale` (D-019). Parity/dispersion
  tests record residuals into the JUnit XML. Real check: `tests/test_tools.py` + `tests/test_dispersion.py` ingested;
  D3 vs SIESTA energy residual 1.1e-8 eV, rVV10 vs QE 3.5e-11 Ry appear in the graph.
- Checks: ruff, mypy, `make kg` (Doxygen + Sphinx -W + KG), `tests/test_phase0.py -k "knowledge_graph or kg_mcp"` (4 passed).

### 2026-11-20 — Open issues plan, step 2: O-6 SIESTA mesh criterion (Claude Code)
- `choose_mesh_dimensions(criterion="siesta")`, `siesta_mesh_cutoff_ry`, `SCFConfig.mesh_criterion/mesh_subdivisions`,
  FDF/case/adapter plumbing, `reference_mesh_dims` in the matched runner (D-020). All 15 test cells give SIESTA's mesh.

### 2026-11-20 — Open issues plan, step 3: O-2 Co stress (Claude Code)
- Term-by-term fixed-(D, E) FD harness (`jax_siesta/testing/strain_fd.py`, `tests/test_stress_components.py`),
  fixed-mesh SCF-level FD vs SIESTA (`scripts/co_strain_*.py`). Result (D-021): the analytic stress is exact; the 2.8 %
  σ_zz gap is in the energy's strain dependence (neutral-atom pair term). New open issue O-8.

### 2026-11-20 — Open issues plan, step 4: O-1 Broyden and ADIIS mixers (Claude Code)
- `scf/mixer.py` (`_broyden_update`, `adiis_coefficients`, `_adiis_step_jit`), `SCFState.history_d/history_e`, controller
  quarantine, FDF mixer keywords, extended mixer benchmark (D-022). Broyden 9-11 iterations vs adaptive Tikhonov-Pulay
  10-14 on three systems; the ADIIS hybrid is slower on all of them (reported, not tuned away).

### 2026-11-20 — Open issues plan, step 5: O-3 Harris energy (Claude Code)
- Per-iteration Harris-Foulkes energy with provenance (D-023); `tests/test_harris.py` 6 passed. O-4 and O-7 not started.

### 2026-11-20 — Open issues plan, steps 6-7: O-4 parity matrix and O-7 Snellius (Claude Code)
- Matrix parity with PseudoDojo SR PBE (D-024); fixed multi-species table handling; SIESTA reference cache + `make
  refs-cache`; synced to Snellius (`scripts/snellius_sync.sh`) and submitted job 29519222 (`scripts/snellius_a100_refs.sbatch`).

### 2026-11-20 — O-8 resolved (Claude Code)
- Read SIESTA's `naefs`/`dnaefs`/`electrostatic.f`/`matel_params`; evaluated its `Ena` formula exactly from the ion tables:
  jsiesta equals it to 0.05-0.07 %, SIESTA's printed value deviates smoothly (D-025). `siesta_na_reference.py`,
  `tests/test_na_pair_exact.py`. No change to the physics code.
- O-8 Co follow-up: exact SIESTA formula for the Co strain runs: absolute 14 micro-eV; derivative jsiesta -0.075 %, SIESTA +0.097 % (opposite); gap 43 % / 57 % (D-025).


### 2026-11-21 — Remaining items, repository incident, scaling study (Claude Code)
- Committed D-026 (frozen-axis XC test), D-027 (SIESTA mesh criterion is the default), D-028 (FR PSML below nspin 8 equals SIESTA's projectors), D-029 (`run_siesta_directory`: all SIESTA-calling tests replay from the reference cache; the `external_refs` marks of `test_pr_mgga_si_bulk_noncol_and_soc_parity` had been attached to a helper function and are fixed).
- `make check` and `make test-refs` (HEAD e8ea531): 127 passed, 12 skipped (all CUDA), both. Not re-run after D-026 to D-029.
- Incident: the symlinks `.venv` and `refs_cache` created in a temporary worktree were committed (the ignore patterns did not match symlinks) and the fast-forward merge replaced the real directories. Venv rebuilt from `requirements-dev.txt`, cache restored from the Snellius copy and refilled by `make refs-cache` (17 passed), commits rewritten without the symlinks, `.gitignore` fixed.
- Snellius job 29519222 failed at the build (Libxc 6 has no `xc_hyb_type`); fixed and verified by a login-node CUDA build.
- D-030: mixer comparison on three Co AFM starts with SIESTA's tables (all mixers reach the same endpoint; ADIIS no faster than Tikhonov-Pulay).
- Scaling study set up (`scripts/scaling_*.py`, sbatch files, `scripts/plot_scaling.py`) and submitted (jobs 29542773 GPU; 29542774, -79, -80 CPU).

### 2026-11-21 — Bottleneck report, assessment plan, branch `perf/bottlenecks` (Claude Code)
- `known_bottlenecks.md` (scaling results, routines, diagnosis) and `bottleneck_assessment_plan.md` (hypotheses and experiments).
- Branch `perf/bottlenecks`: D-031 vectorised set-up searches (64 atoms 331-400 s -> 8.6 s), D-032 overlap transform once per
  SCF + real Gamma path, D-033 gather assembly (the CPU "eigensolver" cost was a scatter-add), D-034 threaded cached CPU `dfscf`
  (forces 9x faster), D-035 symmetry cross-check pinned to one mesh (pre-existing failure since D-027), D-036 CUDA `rhoofd`/`vmat`
  on the cached host plan (compiled on Snellius, not yet run on a GPU). Physics checked against the previous code on the
  manifest cases (bitwise or rounding-level), new tests `test_vectorised_neighbour_searches_match_scalar_reference`,
  `tests/test_eigensolver.py`.
- Scaling job 29542773 ended at its time limit in the 216-atom set-up; A100 128 atoms: 35.5 s per iteration.

### 2026-11-21 — A100 validation of `perf/bottlenecks` (Claude Code)
- Snellius job 29555653 (one A100, 33 min, revision `4b72090`): `tests/test_phase3_cuda.py` passes on the plan path and on
  the legacy kernel (11 passed, 1 Libxc skip each). `test_one_eigh_per_solve_with_precomputed_transform` failed on
  jax 0.12.1 only because `jax.core.ClosedJaxpr` is gone; the jaxpr walk is now duck-typed.
- A100 iteration 5.18 s -> 0.051 s (16 atoms), 35.5 s -> 0.293 s (128); 0.487 s at 216 and 1.87 s at 512 atoms. Set-up
  1583 s -> 13 s at 128 atoms. GPU forces + stress 150 s at 128 atoms (edge-kernel backward), now the main GPU cost.
- `scripts/scaling_jsiesta.py --kernels` called the solve with two arguments instead of three (si008, si064 and the legacy
  A/B failed); fixed and checked on si008 locally.
- Recorded in D-036 and `known_bottlenecks.md` Section 9.1; deck `presentations/perf_bottlenecks.tex`.

### 2026-11-21 — Follow-up A100 job: kernel breakdown and legacy A/B (Claude Code)
- Snellius job 29559809 (15 min, revision `5af4442`): CUDA parity 17 passed on the plan path (the jax 0.12 test fix
  works), 11 on the legacy path.
- 64 atoms, same job: `rhoofd` 9.44 s -> 0.047 s, `vmat` 8.20 s -> 0.068 s, iteration 17.7 s -> 0.117 s; 8 atoms 0.0128 s
  per iteration (SIESTA-18: 0.042 s). `vmat` is now the largest kernel; GPU forces + stress 75 s at 64 atoms.
- `known_bottlenecks.md` Section 9.1 and status note, outcome of `bottleneck_assessment_plan.md`, D-036, deck updated.

### 2026-11-22 — GPU forces: profiling, meta-GGA SCF and forces backward on the plan (Claude Code)
- Branch `perf/gpu-forces`. Stage timers in `StaticObservableEvaluator.evaluate` (`timings_out`, results bitwise unchanged),
  `scripts/scaling_jsiesta.py --xc/--kernels` with backward kernels, `scripts/snellius_gpu_forces_profile.sbatch` (Libxc build).
- Job 29584724 lost its profile to a jax-version pytree mismatch in the script (fixed); job 29586936: 12 CUDA tests pass
  (first Libxc meta-GGA GPU run), profile in `known_bottlenecks.md` Section 9.2: the backward edge kernels are 89-97 % of
  the GPU force time (r2SCAN 265 s at 64 atoms), KB host forces about 0.11 s per atom, host Libxc XC 44 % of the r2SCAN
  iteration.
- D-037 (A100-validated): `tauofd`/`vmat_tau` on the plan, r2SCAN iteration 43.2 s -> 0.132 s at 16 atoms.
- D-038: host plan with orbital Hessians; CPU meta-GGA backward on the plan (r2SCAN `tauofd` bwd 17.0 s -> 1.36 s at 8
  atoms, forces within 2.6e-13 Ha/Bohr of the old code); GPU backward kernel for Modes 2 and 5 (compiled, A100 pending).

### 2026-11-22 — A100 validation of D-038; LDA/GGA XC on the device (Claude Code)
- Snellius job 29589348 (one A100, 25 min, revision `81890c1`): all 12 tests of `tests/test_phase3_cuda.py` pass, so the
  GPU forces backward on the plan (D-038) matches the CPU for PBE and meta-GGA. Forces + stress at 64 Si atoms: PBE
  76.1 s -> 7.98 s, r2SCAN 265.5 s -> 9.05 s; kernel A/B at 16 atoms `rhoofd` bwd 620x, `tauofd` bwd 900x. The KB host
  forces (7.0 s at 64 atoms) are now the largest force stage. Recorded in D-038 and `known_bottlenecks.md` Section 9.2.
- D-039: the dual-number LDA/GGA kernel is one host/device header (`cpp/include/jsiesta/xc_semilocal.hpp`) with Libxc's
  parameters (`PW_MOD` in PBE correlation; the old device and no-Libxc CPU code used the original PW92 digits, about
  1e-5 relative in the correlation energy) and Libxc's thresholds. New `test_builtin_xc_kernel_matches_libxc`: energy
  density within 1.3e-14 relative of Libxc, v_rho 5e-13. `XcEvalCudaImpl` uses the device kernel for LDA/PBE/PBEsol
  with Libxc compiled in (`JSIESTA_CUDA_XC_LIBXC=1` restores the host route for A/B). New GPU test
  `test_cuda_builtin_xc_pointwise_matches_cpu`; the profiling script times the host-Libxc A/B at 64 atoms and runs the
  edge-kernel A/B only with `LEGACY_AB=1`.

### 2026-11-22 — Basis optimiser vs SIESTA; pressure unit bug (D-040); refreshed scaling runs (Claude Code)
- `scripts/basis_opt_vs_siesta.py`: Si DZ basis-enthalpy optimisation in jsiesta, then both codes on the same explicit
  `PAO.Basis` blocks (start, optimum, line scans). Found the basis pressure 13.6x too small (Ry/Bohr^3 used as eV/Bohr^3),
  fixed with a unit test. After the fix the two codes agree on the enthalpy landscape (line minima within 0.05 Bohr,
  curvatures within 3 %, constant 28 meV energy offset per cell). Details in D-040; data `reports/basis_opt/`.
- Snellius: jsiesta CPU scaling on 18 cores with the current code and forces at every size (job 29591326, 12 min): one
  iteration 0.14 s (8 atoms) to 4.25 s (216), forces + stress 1.3-39 s (KB host forces 60 %). A100 scaling 8-512 with
  forces (job 29591325) queued; scripts now take `OUT` and make the edge-kernel A/B optional (`LEGACY_AB=1`).

### 2026-11-22 — KB projector forces in C++ (D-041) (Claude Code)
- Profiled `_kb_forces_and_stress` at 64 Si atoms: 7.2 s, spread over NumPy temporaries, `np.add.at` and `searchsorted`
  in a loop over 384 KB shell instances (about 24 M orbital pairs), not Python overhead.
- New threaded kernel `jsiesta_kb_three_center` (`cpp/src/kb_three_center.cpp`) for both the V_KB assembly and the KB
  energy, forces and stress; NumPy kept for spin-orbit. 7.3 s -> 0.30 s at 64 atoms, V_KB bitwise identical, forces
  within 1e-14 of the NumPy path; new parity test. Expected A100 effect: forces + stress at 64 atoms about 8.0 s -> 1.3 s
  (not yet measured).

### 2026-11-22 — A100 job 29590887: D-039 validated (Claude Code)
- Profiling job on `ef44c1a` (9 min): 20 of 20 CUDA tests pass, including the new
  `test_cuda_builtin_xc_pointwise_matches_cpu` (LDA, LDA_PW92, PBE, PBEsol; nspin 1 and 2).
- Device LDA/GGA XC vs host Libxc at 64 Si atoms (PBE, same job): Hartree + XC 0.096 s -> 0.0006 s, one SCF iteration
  0.199 s -> 0.103 s. Forces + stress unchanged at 8.3 s (KB host forces 7.5 s; D-041 is not in this build).
  r2SCAN iteration 0.437 s, of which host Libxc meta-GGA 0.21 s. Data `reports/forces_prof/job29590887/`.
- The 8-atom PBE iteration time comes out slightly negative (-0.002 s): the timer is the difference of two short SCF
  calls and is below its noise at that size.

### 2026-11-22 — A100 scaling job 29594601 and the plan memory fallback (D-042) (Claude Code)
- Scaling 8-512 atoms on `f7313a9` (31 min): 25 CUDA tests pass. SCF iteration 0.013-1.81 s; set-up (basis, pattern,
  two-centre) 4.4-42.5 s; forces + stress 0.31-2.62 s up to 216 atoms (64 atoms 0.96 s, was 8.3 s; KB stage 0.24 s).
- 512 atoms: forces + stress 546 s, all in the grid backward. The gradient plan (about 10 GB) did not fit next to the
  cached value plan in the memory JAX leaves free, and the code fell back to the edge kernel without a message. D-042:
  reuse a richer cached plan, evict and retry, warn on fallback. Data `reports/scale2/`.
- Deck updated (version 9): iteration, forces, set-up, whole-calculation, summary and next slides.

### 2026-11-23 — A100 job 29607477: 512-atom forces with D-042 (Claude Code)
- 216 and 512 atoms on `18cb66a` (9 min, plan sizes logged): 25 CUDA tests pass; 512-atom PBE forces + stress 546 s ->
  10.1 s. The gradient plan is 13.0 GB (385 M orbital values) and fitted after evicting the 3.8 GB value plan.
- The log showed the value plan uploaded again after the gradient plan: the plan cache key included the gradient flag.
  The inputs are now hashed once and the flags folded in last (`grid_plan_input_hash`, `grid_plan_cache_key`), so the
  device finds the richer plan before building the host plan. Grid tests pass on the CPU; A100 check pending.
- Deck version 10: 512-atom forces, single point 157 s and MD step 73 s at 512 atoms. Data `reports/scale3/`.

### 2026-11-23 — GPU grid plan memory: streaming and host plan size (D-043) (Claude Code)
- Streamed device plan: when a plan does not fit, only its index arrays stay on the device and each call copies the
  images and the value arrays it reads, chunk by chunk of grid points, through the same kernels (which now take a point
  range and base offsets). New GPU test variants run the grid tests with the plan streamed in many chunks.
- A100 job 29607988: 22 CUDA tests pass (Libxc build, no skips); 512-atom PBE forces + stress 6.46 s resident, 9.5 s
  forced to stream (iteration 1.79 -> 3.37 s); r2SCAN 216 atoms resident, 5.6 s. r2SCAN 512 atoms ran out of host
  memory (97 GB) building the Hessian plan.
- Host plan: two-sweep entry build without a scattered copy, exactly sized image arrays, per-entry records freed once
  the atom-block plan exists, richer cached plans serve lighter requests and superseded ones are dropped. 64-atom r2SCAN
  on the laptop: host plan cache 3.9 GB at the end (about 16 GB before). New `JSIESTA_GRID_PLAN_VERBOSE`.
- A100 job 29608548: r2SCAN 512 atoms with forces in 20.1 s, peak host memory 36.5 GB, Hessian plan (31.5 GB) streamed in
  4 chunks. Data `reports/plan_mem/`, `reports/plan_mem2/`. Deck slides edited locally (publish pending).

### 2026-11-23 — Pinned, overlapped plan streaming (D-044); beamer deck updated (Claude Code)
- Streamed plans page-lock their host arrays (`cudaHostRegister`) and copy chunks on a non-blocking copy stream into two
  buffer sets, ordered by events, overlapping the copy of one chunk with the kernels of the previous one. New GPU test
  mode: streamed without pinning or overlap.
- A100 job 29610921: 24 of 24 CUDA tests. 512 atoms, PBE forced to stream, same job: forces + stress 8.25 -> 7.01 s,
  iteration 2.50 -> 2.39 s (one-chunk value plan, copy-bound). r2SCAN 512-atom forces 14.8 s (20.1 s in job 29608548).
  Data `reports/plan_mem3/`.
- `presentations/perf_bottlenecks.tex` updated to D-031 to D-044 (23 November): perf/gpu-forces changes, physics check,
  A100 iteration, forces + stress, 64-atom force stages, GPU plan memory, whole calculations, merge plan.

### 2026-11-23 — Consolidated release repository (D-045) (Claude Code)
- Development repository: `main` fast-forwarded to `perf/gpu-forces` (`ea4b882`).
- Release repository: exported `main`, removed scripts, presentations, pilots and examples, moved the design documents
  to `docs/design/`, emptied the per-machine defaults of `tests/external_refs.toml`, removed user paths, rewrote the
  README and `MEMORY.md`, dropped `matplotlib`. Dead-code scan (AST, every top-level function and class referenced
  somewhere) found nothing to remove.
- Testing the fresh tree found two portability bugs, fixed: the KG test depended on a git-ignored local SIESTA
  documentation index (now the external reference `siesta_doc_index`), and results staleness ignored a missing HEAD.
  `make check` in the release tree; one initial commit.

### 2026-11-23 — Fixed costs: compilation and threaded host set-up (D-046, D-047) (Claude Code)
- Measured on the laptop: a new geometry with unchanged shapes recompiled three SCF functions (closures over the
  geometry); the first call made about 400 small eager compilations; the grid plan, two-centre integrals and host
  Libxc were single-threaded.
- D-046: module-level compiled SCF and force stages with the geometry as arguments, precomputed Bloch gather tables,
  custom-derivative residuals instead of closures, compiled initial density matrix, opt-in persistent compilation cache.
- D-047: shared threading header; threaded grid plan build (order-preserving), two-centre and KB-overlap integrals,
  host Libxc; cheaper shift lookups in the pattern builder.
- 64-atom single point on the laptop: 84.3 s -> 38.9 s (second geometry 65.9 s -> 37.8 s), energies equal to 12
  decimals; component outputs bitwise identical; 57 SCF/forces/symmetry tests pass; CUDA + Libxc build compiles on
  Snellius. A100 measurement pending.

