# Architectural & Physical Decisions Log

## D-001: PSML-Only Pseudopotential Support (2026-11-18)
- **Decision:** Exclusively parse and support the standardized norm-conserving `.psml` v1.1 XML format (semilocal channels, KB projectors, scalar-relativistic and $j$-dependent SOC channels, NLCC pseudo-core charge, valence configuration) alongside `.ion` reference basis tables. Drop legacy `.psf` support completely.
- **Rationale:** Avoids maintaining unstandardized radial-grid conventions while preserving exact norm-conserving SIESTA parity (`IMPLEMENTATION_PLAN.md` §1.1).

## D-002: Drop `libgridxc`; Use Direct C++ XC FFI + JAX FFTs + DFT-D3 / rVV10 (2026-11-18)
- **Decision:** Replace `libgridxc` with `jax.numpy.fft` for $\nabla\rho$, Poisson electrostatics, GGA stress, and Sabatini rVV10, paired with a C++ XLA FFI kernel (`xc_eval`) supporting compile-time `libxc` linkage (`JSIESTA_HAS_LIBXC`) and exact built-in analytical C++ LDA/PBE/PBEsol evaluation when `libxc` is absent.
- **Rationale:** Keeps grid arrays resident on-device with zero host round-trips and eliminates Fortran MPI grid-decomposition boilerplate (`IMPLEMENTATION_PLAN.md` §1.2).

## D-003: Purely Functional PyTrees & Protocol Extension Seams (2026-11-18)
- **Decision:** All simulation state is carried in frozen dataclasses registered with `jax.tree_util` (`Structure`, `SpeciesData`, `BasisData`, `SparsePattern`, `KPointMesh`, `SCFConfig`, `SCFState`, `Observables`) with zero hidden globals, and every subsystem is behind a `typing.Protocol`.
- **Rationale:** Enables `jax.jit`/`vmap`/`grad` transformations and modular extensibility (DFT+U, hybrids, BdG) (`IMPLEMENTATION_PLAN.md` §1.4).

## D-004: Native `spglib` Symmetry & Commensurate Mesh Invariants (2026-11-18)
- **Decision:** Use `spglib` for spatial and Shubnikov Type I–IV magnetic space groups, 16-point Fibonacci Wigner-$D$ rotation matrices, symmetry-closed CSR/COO sparsity, precompiled sparse `segment_sum` projection $\mathcal{P}_{\text{sym}}$, and symmetry-commensurate `(N1, N2, N3)` grid sizing.
- **Rationale:** Eliminates runtime binary search misses and eggbox symmetry-breaking noise (`IMPLEMENTATION_PLAN.md` §3).

## D-005: `scf_params` Adaptive Controller & Post-SCF `DM_out` Contract (2026-11-18)
- **Decision:** Embed conditioning descriptors, Tikhonov–Pulay / ADIIS / Broyden mixers, decoupled charge/spin preconditioning, deterministic $1\times 1$ Gram SVD, closed-loop trust-region adaptive controller, and mandatory post-SCF `DM_out` energy recomputation (`final_post_scf_free_e` distinct from `last_scf_free_e`).
- **Rationale:** Guarantees $\le 0.2\text{ meV/cell}$ endpoint reproducibility across mixers and eliminates premature or oscillating SCF stalls (`IMPLEMENTATION_PLAN.md` §4).

## D-006: Zero-Warning Sphinx + Doxygen Auto-Docs & Progressive Knowledge Graph (2026-11-18)
- **Decision:** Configure `doxygen` XML generation for `cpp/` and `sphinx-build -W` for Python/theory/architecture docs (converting Markdown/Mermaid/Doxygen XML cleanly using Sphinx built-ins when optional third-party extensions like `myst_parser`/`breathe` are absent), and index AST + Doxygen + theory + validation + decision nodes atomically into `kg/current/docs_kg.db`.
- **Rationale:** Ensures documentation builds cleanly in offline environments while remaining compatible with full `myst_parser`/`breathe` setups (`IMPLEMENTATION_PLAN.md` §5.3–5.4).

## D-007: Cached Geometry-Invariant Grid Plans & Fixed-Shape JIT Mixing (2026-11-18)
- **Decision:** Cache geometry- and sparsity-invariant orbital-grid caches and dense atom-pair blocks (`GridKernelPlan`, verified by hash + exact byte compare) inside `cpp/src/grid_ffi.cpp`, and execute SCF mixing via fixed-shape padded `@jax.jit` kernels (`_mixer_step_jit`) without host transfers inside Krylov iterations.
- **Rationale:** Eliminates redundant per-SCF cubic-spline evaluations, open-addressing hash probes, and Python-side JAX kernel dispatches while preserving exact numerical identity and bitwise invariance under atomic perturbations (`IMPLEMENTATION_PLAN.md` §6 Phase 3).

## D-008: Dual-Target CPU/CUDA XLA FFI Registration & Stream-Ordered Device Kernels (2026-11-19)
- **Decision:** Register all 6 grid and XC XLA FFI handlers for both `"cpu"` and `"CUDA"` platforms (`jax_siesta.core.ffi.register_grid_xc_target`), gated by `jsiesta_has_cuda()`, and execute device kernels on XLA's assigned `cudaStream_t` using stream-ordered `cudaMallocAsync`/`cudaFreeAsync` (`cpp/cuda/rhoofd_vmat.cu`).
- **Rationale:** Enables identical pure-JAX host/device graphs (`mesh.py`, `functional.py`, `scf/driver.py`, `observables/forces.py`) to run bit-for-bit on macOS/Linux CPU and NVIDIA A100 GPUs without changing caller APIs.

## D-009: Real PSML v1.1 + SIESTA `.ion.xml` Twin Mode & Mobile HTML Parity Reporting (2026-11-19)
- **Decision:** Support optional `.ion.xml` PAO/KB/VNA radial tables in `PSMLBasisLoader` alongside standardized PSML v1.1 (`$4\pi r^2$` pseudocore normalization and tail $-Z_{\text{val}}/r$ completion), validate against real `siesta` and `siesta_spglib_full_impl` binaries via `validation.py`, and render self-contained responsive HTML reports (`reports/siesta_parity_report.html`) including measured timings.
- **Rationale:** Separates PAO basis-generation choices from Kohn-Sham solver/symmetry parity so component energies, forces, stress, spin moments, and wall-clock runtimes can be inspected reproducibly offline.

## D-010: Sphinx Immaterial Theme for Offline API & Theory Docs (2026-11-19)
- **Decision:** Configure `docs/conf.py` to use `sphinx_immaterial` (`"font": False`, `sphinx==8.2.3`) while keeping `-E -W` strict warning checking and `kg/build_kg.py` dictionary guards.
- **Rationale:** Provides responsive mobile/desktop HTML navigation for Sphinx + Doxygen documentation without requiring external font network fetches.


## D-011: Meta-GGA + Non-Collinear + SOC Architecture (FD7, Frozen Spin Axis, Vacuum Mask, CUDA Staging)
- **Status:** Accepted
- **Decision:**
  1. Use SIESTA's 6th-order ``nn=3`` 7-point central finite-difference stencil (``_fd7_gradient_3d``) in ``jax_siesta/xc/functional.py`` for semilocal XC density gradients while keeping ``_fft_gradient_3d`` for ``RVV10`` dispersion.
  2. Rotate non-collinear/SOC spin densities and kinetic energy densities using smooth component gradients projected onto a frozen local spin axis (``jax.lax.stop_gradient`` on ``ux, uy, uz``, matching ``libgridxc`` ``old_scheme = .true.``).
  3. Extract ``jsiesta::evaluate_mgga_libxc_host`` in ``cpp/src/xc_ffi.cpp`` with ``xc_func_set_dens_threshold(&func, 1e-10)`` and explicit ``n_tot <= 1e-9`` vacuum masking, shared between CPU and CUDA XLA FFI handlers.
  4. Extend ``cpp/cuda/{kernels.cuh,rhoofd_vmat.cu}`` with unified orbital-grid/Hessian kernels (``tauofd``, ``vmat_tau``, ``tau_dfscf_grad``, ``tau_core``, ``tau_core_grad``) and host-staged ``jsiesta_xc_eval_mgga_f64_cuda``.
- **Rationale:** Global FFT gradients caused ``1e6-2.57e9`` ``v_sigma`` vacuum Gibbs spikes on atomic core boundaries, and differentiating ``|m|`` directly introduced ``C^0`` cusps in SOC calculations. FD7 + frozen axis + vacuum threshold eliminates both artifacts and achieves parity with ``pr_mgga`` SIESTA.

## D-012: Tier B In-Python 1D Radial PAO Generator & Junquera Basis Enthalpy Optimizer (2026-11-19)
- **Status:** Accepted
- **Decision:**
  1. Implement an internal 1D logarithmic-grid Sturm–Liouville atomic KS/PAO solver (``jax_siesta/species/pao_generator.py``) directly from ``.psml`` v1.1 semilocal or Vanderbilt local/KB potentials, spherical JAX LDA/PBE XC, soft confinement $V_0 / (r_c - r)$ cutoff divergence, and explicit per-zeta $r_c$ split-norm parabola fitting (replacing split-norm percentages with explicit ``rc_list`` plus ``v0_ha``, ``ri_bohr``, and ``ionic_charge``).
  2. Implement Junquera Basis Enthalpy optimization (``jax_siesta/species/basis_opt.py``) matching SIESTA's ``Src/basis_enthalpy.f90`` radial-shell volume convention ($V_{\text{orb}} = \sum_{\text{atoms}}\sum_{n l \zeta} \frac{4}{3}\pi r_c^3$, $1\text{ GPa} = 6.797861843486488\times 10^{-5}\text{ eV}/\text{Bohr}^3$), float64-guarded arccos bijective parameter mapping, cascading lower/upper bounds (`chain_lo` / `depends_on`), Nelder–Mead / Powell / DE optimizers, and `compute_forces_stress=False` fast energy evaluation in ``StaticObservableEvaluator``.
- **Rationale:** Eliminates external `siesta` binary dependencies inside basis optimization loops while enforcing exact radial-shell ordering and SIESTA basis-volume definitions.
- **Correction (D-040):** the value quoted above, 6.797861843486488e-5, is 1 GPa in Ry/Bohr^3 (SIESTA's internal unit), not eV/Bohr^3; 1 GPa = 9.2489e-4 eV/Bohr^3.

## D-013: PW92 Second-Derivative Pre-factor & CUDA Libxc LDA/GGA Parity Fix (2026-11-19)
- **Status:** Accepted
- **Decision:**
  1. Fix ``fdd0 = 8.0 / (9.0 * (cbrt(2.0) * 2.0 - 2.0))`` in both ``cpp/src/xc_ffi.cpp`` and ``cpp/cuda/rhoofd_vmat.cu`` (replacing ``4.0``).
  2. Extract ``jsiesta::evaluate_xc_libxc_host`` in ``cpp/include/jsiesta/grid_xc_ffi.hpp`` and ``cpp/src/xc_ffi.cpp`` and invoke it under ``#ifdef JSIESTA_HAS_LIBXC`` inside ``XcEvalCudaImpl``, matching the host Libxc bridge used by ``XcEvalMggaCudaImpl``.
  3. Use mixed ``rtol=1e-12, atol=1e-10`` for ``tauofd`` VJP gradient parity assertions in ``tests/test_phase3_cuda.py``.
- **Rationale:** Ensures exact CPU==CUDA parity across spin-polarized PW92 and Libxc LDA/GGA functionals without diverging built-in Dual-AD correlation constants ($A = 0.0310907$ vs $0.031091$).

## D-014: DFT-D3 via simple-dftd3; three upstream SIESTA D3 issues documented (2026-11-20)
- **Status:** Accepted
- **Decision:** Replace the improvised pairwise model in ``jax_siesta/xc/dispersion.py`` with a thin wrapper around
  the reference implementation ``simple-dftd3`` (PyPI ``dftd3`` ≥ 1.7, optional dependency ``[dispersion]``).
  Published damping parameters are loaded per functional (``d3_method_for_functional``); ATM three-body is on
  (``s9 = 1``, SIESTA's default); cut-offs default to the library values (60/40/40 Bohr) and are configurable.
  The functional now reaches the model (``DFTD3Dispersion(config.xc_functional)``).
- **Validation:** ``tests/test_dispersion.py`` — identity with the library, FD forces (1e-9) and FD strain stress
  (1e-10), and cross-code parity with SIESTA 5.4.2 (energy < 2e-6 eV, forces < 3e-5 eV/Å, σ_xx vs FD of SIESTA's
  D3 energies < 2e-6 eV/Å³).
- **Upstream issues found (SIESTA 5.4.2-10-g6080da615, ``Src/grimme_dispersion_m.F90``):**

  1. With ``DFTD3.UseXCDefaults T`` the loaded functional parameters are overwritten by the generic
     ``siesta_d3_dat`` defaults (s8 = 1, a1 = 0.4, a2 = 5.0) at lines 236–251. Si test cell: SIESTA −0.580247 eV,
     which equals s-dftd3 with the generic parameters; PBE-D3(BJ) gives −0.602710 eV.
  2. The D3 stress is added with the wrong sign (``stress = stress - Gstress*2/vol`` while ``Gstress`` is
     +∂E/∂ε). Converged FD of SIESTA's own total energy: −0.0336 eV/Å³; SIESTA prints −0.0619 eV/Å³.
  3. SIESTA bundles s-dftd3 1.1.0, whose analytic gradient is ~0.3 % inconsistent with its energy
     (F₁ₓ −0.006882 vs FD −0.006901 eV/Å); s-dftd3 1.7.0 agrees with FD.
- **Rationale:** Review finding H3. Re-implementing D3 tables is error-prone; the reference library is the
  specification. Parity with SIESTA requires explicit ``DFTD3.UseXCDefaults F`` parameters until issue 1 is fixed.

## D-015: rVV10 re-implemented from Sabatini et al. and validated against Quantum ESPRESSO (2026-11-20)
- **Status:** Accepted
- **Decision:** Replace the improvised k-space kernel in ``RVV10Dispersion`` with the published functional:
  θ = n/κ^{3/2}, κ = b(3π/2)(n/9π)^{1/6}, q = ω₀/κ with smooth saturation (q_cut = 0.5, q_min = 1e-4, 12 terms),
  kernel φ = −(3/2)/[(qR²+1)(q′R²+1)(qR²+q′R²+2)], natural cubic-spline cardinal functions on a 20-point
  q-mesh, the kernel's radial Fourier transform tabulated on R ≤ 100 Bohr (1024 intervals) and spline-interpolated
  in the modulus of G, and β = (1/32)(3/b²)^{3/4}. Numerical parameters (q-mesh, q_cut, r_max, n_r) equal QE's ``xc_rVV10`` so
  the two codes are comparable term by term; the implementation itself is written from the paper. The functional
  now acts on valence + core density, like the semilocal XC and QE. The SCF loop no longer computes the rVV10
  strain derivative every iteration (``compute_stress=False``).
- **Validation:** ``tests/test_dispersion.py`` — analytic k = 0 kernel (< 1e-4 rel.), E_nl on QE's converged
  carbon-bilayer density equals QE's printed value to 3.5e-11 Ry (fixture ``tests/fixtures/qe_rvv10``), stress vs
  FD (< 1e-9), potential = functional derivative (``tests/test_phase1_grid_xc.py``).
- **Consequence:** the SIESTA adapter now refuses matched runs with dispersion: SIESTA has no rVV10
  (``XC.authors VV`` is VV10 with rPW86 exchange) and its D3 defaults are wrong (D-014).

## D-016: Absolute total energy — one-centre neutral-atom constants (2026-11-20)
- **Status:** Accepted
- **Finding (review H1):** ``total_energy`` / ``final_post_scf_free_e`` were defined only up to a per-species
  constant (Si: −84.60 eV/atom vs SIESTA ``Etot``) because ``E_NA`` omitted all one-centre terms; no test compared
  total energies, so it went unnoticed.
- **Decision:** ``neutral_atom_self_energy`` (``jax_siesta/integrals/two_center.py``) adds, per atom,
  C = ½J[ρ_NA,ρ_NA] − ½J[ρ_L,ρ_L] − ∫V_NA n_at, evaluated as ½∫r²V′²dr (plus the −Z/r tail for V_loc) with
  cubic-spline derivatives of the radial tables. It is geometry independent, so forces and stress are unchanged.
  ``e_na`` now equals SIESTA's ``Ena + DEna − Eions``. The SIESTA adapter exports ``e_total_ev``,
  ``final_post_scf_free_e_ev`` and ``e_na_ev`` for both codes.
- **Validation:** ``tests/test_siesta_parity.py::test_siesta_total_and_free_energy_parity`` — three Si geometries on
  SIESTA's mesh: absolute ΔE ≈ 6.4 meV/cell (< 10 meV), geometry-independent to < 0.3 meV (observed 0.11 meV).
  ½J[ρ_L,ρ_L] matches SIESTA's ``Eions`` to 0.14 meV/atom; the isolated-atom one-centre ``Ena`` differs by
  1.8 meV/atom (SIESTA evaluates it with its reciprocal-space two-centre machinery), which accounts for most of the
  residual absolute offset.

## D-017: Radial-table fidelity and exact atomic neutral-atom term (2026-11-20)
- **Status:** Accepted
- **Findings (review H2, M5):** Co hcp disagreed with SIESTA by 0.44–0.85 eV in E_kin; Si stress oscillated with the
  mesh (σ_xx 3.6e-3 … 5.4e-3 eV/Å³ for 100–400 Ry while SIESTA stays at 4.8–4.9e-3).
- **Root causes and decisions:**

  1. ``.ion.xml`` tables (500 points per function, Co 3s spacing 0.0046 Bohr) were linearly interpolated onto a shared
     401-point grid. Now each species' grid is at least as fine as its finest source table and functions are
     cubic-spline resampled (PAOs and KB projectors through SIESTA's smooth R/r^l). With SIESTA's own density matrix,
     jsiesta's S now matches SIESTA's to 2.3e-6 (was 1.7e-5), Tr[D T] to 0.011 eV (was 0.164 eV), Tr[D V_KB] to 0.5 meV.
  2. Two-centre quadrature raised to k_max = 60 Bohr⁻¹, 1024 points, 601-point pair splines (on-site T within 0.5 meV
     of exact for Co semicore; Tr[D T] within 3 meV of SIESTA).
  3. With an ``.ion.xml`` the local potential is SIESTA's ``reduced_vlocal`` (for Co SIESTA regenerates it from
     ``chlocal``; the PSML one shifted the ion self-energy by 24 meV/atom). Now ½J[ρ_L] = SIESTA ``Eions`` to 0.2 meV/atom.
  4. The atomic part Σ_IJ ∫V_NA,I n_J is evaluated exactly as two-centre integrals
     (``neutral_atom_density_two_center``) and only ∫V_NA δρ is integrated on the mesh, as SIESTA does
     (Ena two-centre, DEna on the mesh). Si σ_xx now agrees with SIESTA to ≤ 2.3e-5 eV/Å³ at 100/150/250/400 Ry and
     E_total to 0.06 meV/cell at every mesh; the one-centre constant becomes ½J[ρ_NA] − ½J[ρ_L].
  5. ``build_native_symmetry(..., mesh_shape_override=...)`` accepts SIESTA's mesh after a commensurability check
     (``check_mesh_commensurate``); ``SCFConfig.symmetrize_grid`` can switch grid symmetrisation off.
  6. The adapter reads SIESTA's free energy from ``FreeEng`` (there is no ``Free =`` line).
- **Result (Co hcp, matched 16×16×30 mesh, no symmetry):** ΔE_kin −8 meV, ΔE_KB +1.3 meV, ΔE_xc +1.6 meV,
  ΔE_F 0.07 meV, ΔF 4e-6 eV/Å, ΔE_total ≈ +5 meV/atom; Co σ_zz still differs by ~4 % (SIESTA's own analytic σ_zz
  differs by 4 % from the FD of its free energy; open issue O-2 in MEMORY.md).

## D-018: Honest SCF/XC/symmetry contracts — review items M1, M2, M3, H4, L1, L3, L4, L5 (2026-11-20)
- **Status:** Accepted
- **M1:** ``mixer_variant`` ``"adiis"`` and ``"broyden"`` raise ``NotImplementedError`` (they ran Pulay with a different
  diagonal shift). The mixer-endpoint test now compares three genuinely different trajectories (adaptive
  Tikhonov–Pulay, fixed-weight short-history Pulay, coupled-spin Tikhonov–Pulay). Real ADIIS/Broyden remain planned.
- **M2:** ``estimate_vacuum_fraction`` uses the largest periodic empty gap per lattice direction minus an atomic
  diameter (bulk → 0, H2O box → 0.91, graphene slab → 0.73); the recommended Tikhonov λ now reaches the mixer.
- **M3:** ``harris_energy_ha`` is NaN (not computed) instead of an alias of the KS free energy.
- **H4:** PAO cut-off radii snap to the generator's log grid; ``SpeciesBasisSpec.log_grid_step`` (default 0.022, basis
  optimiser 0.004) shrinks the plateaus 5×; ``BasisOptimizationResult.best_effective_params`` reports the radii
  actually used; the pressure test optimises a 2D (rc 3s, rc 3p) space with a real budget and requires
  V(5 GPa) < 0.99 V(0.2 GPa).
- **L1:** convergence requires that the temperature used for the current DM equals the target.
- **L3:** ``LIBXC-<id>`` specs are classified through ``jsiesta_libxc_info``; hybrids and VV10 functionals are rejected.
- **L4:** removed a no-op conditional; documented which outcome labels are reserved.
- **L5:** when filtering after group closure drops operations, the largest valid subgroup is rebuilt greedily so the
  projector stays idempotent.

## D-019: Knowledge-graph verification status comes from test results (review M7, open issue O-5) (2026-11-20)
- **Status:** Accepted
- **Finding:** all ``verified_by`` edges were derived from imports and every suite was ``registered``; a module
  imported by a failing, skipped or never-run test still counted as verified. A package import
  (``from jax_siesta.species import X``) additionally marked every submodule verified.
- **Decision:**

  1. ``kg/junit_ingest.py`` parses pytest JUnit XML. Suites get ``pass | fail | partial | skip | never-run``;
     ``verified_by`` edges carry that status (schema version 2, new ``edges.status``/``details_json`` and
     ``test_results`` table). ``kg/build_kg.py`` takes ``--junit [PLATFORM=]PATH`` (repeatable); the default is
     ``reports/pytest.junit.xml``.

  2. A suite is an *external-reference* check only if it contains ``@pytest.mark.external_refs`` tests; a module
     is ``verified`` only when those tests passed. Other passing suites give ``self_consistent_only``.
     ``resume_session_context`` returns the state of every module (``verification``) next to the old lists.

  3. Parity and dispersion tests record residuals and tolerances through the ``record_residuals`` fixture
     (``tests/conftest.py``); they travel in the JUnit XML into the graph (``trace_physics_lineage``).

  4. The JUnit file is stamped with commit, dirtiness and platform. ``check_staleness`` reports
     ``results_stale`` (no results, other commit than HEAD, or a tracked file newer than the results) without
     touching ``up_to_date``: running pytest must not make the code graph look stale.

  5. ``verified_by`` edges now spread only to entities in the same file, not to submodules of an imported package.

  6. ``make check`` ends with ``make kg-results``, which re-ingests the JUnit file written by the tests.

- **Consequence:** until the full suite has run with the reference binaries, most modules show
  ``self_consistent_only`` or ``never_run``; that is the intended honest state.

## D-020: SIESTA mesh-selection criterion as an option (review L6, open issue O-6) (2026-11-20)
- **Status:** Accepted
- **Finding:** ``choose_mesh_dimensions`` used ``ceil(|a_i| sqrt(E)/pi)`` along each lattice vector. SIESTA
  (``get_mesh_sizes`` in ``Src/meshsubs.F``) uses the reciprocal vectors, i.e. the interplanar spacing, makes every
  count a 5-smooth multiple of ``MeshSubDivisions`` and increases all counts until ``chkgmx`` reports a mesh cutoff
  of at least the requested one. The two differ for non-orthogonal cells (Co hcp 18x18x27 vs SIESTA 16x16x30,
  Si fcc 30^3 vs 24^3), so parity runs had to pass SIESTA's mesh explicitly.
- **Decision:**

  1. ``choose_mesh_dimensions(..., criterion="siesta", mesh_subdivisions=nsm)`` ports that algorithm, with
     ``siesta_mesh_cutoff_ry`` as the port of ``chkgmx`` (basis-reduced shortest lattice vector, scanning
     coefficients in [-2, 2]). The default ``"jsiesta"`` is unchanged, so no baseline moves.

  2. ``SCFConfig.mesh_criterion`` and ``SCFConfig.mesh_subdivisions`` carry the choice to every caller
     (``run_scf``, ``StaticObservableEvaluator``, ``build_native_symmetry`` via
     ``choose_commensurate_mesh_dimensions``, the basis optimiser, ``run_case``). Symmetry divisors and
     equivalent axes are combined with ``nsm`` by least common multiple.

  3. ``parse_fdf`` selects ``"siesta"`` (an FDF file has SIESTA semantics), reads ``MeshSubDivisions`` and returns
     ``Mesh.Sizes`` as ``mesh_sizes``. ``load_case_json`` reads ``mesh_criterion`` and ``mesh_subdivisions``. The
     SIESTA adapter writes ``MeshSubDivisions`` explicitly.

  4. ``run_jsiesta_matched_case`` always uses the ``"siesta"`` criterion and takes ``reference_mesh_dims``: it
     raises unless the criterion selects exactly SIESTA's ``InitMesh`` dims. ``mesh_shape_override`` stays as an
     escape hatch, but no parity test uses it any more.

- **Validation:** ``tests/test_siesta_parity.py::test_siesta_mesh_selection_matches_siesta`` runs SIESTA's setup on
  15 cells (the nine Phase 0 cells, Co and Si parity cells at two cutoffs, a skewed hexagonal slab, a triclinic cell
  at ``MeshSubDivisions`` 1, 2, 4): every mesh equals SIESTA's. ``tests/test_phase1_grid_xc.py`` freezes three of
  them and the invariants (alias-free cutoff, divisibility, symmetry constraints) without a SIESTA binary.
- **Not done:** the default stays ``"jsiesta"``; switching it would re-baseline the frozen Phase 0 manifest.

## D-021: Co hcp stress — the analytic stress is exact; the 2.8 % gap to SIESTA is in the energy (review H2, open issue O-2) (2026-11-20)
- **Status:** Accepted
- **Question:** MEMORY.md recorded for Co hcp (nspin = 2, NLCC, 16x16x30 mesh) jsiesta sigma_zz 0.06512 vs SIESTA
  0.06699 eV/A^3 (-2.8 %), with *neither* code equal to the finite difference (FD) of its own free energy
  (jsiesta analytic 1.3 % above its FD, SIESTA 4 % below), and suspected the NLCC or spin-polarised GGA strain term.
- **Method:**

  1. ``jax_siesta/testing/strain_fd.py`` and ``StaticObservableEvaluator.evaluate(components_out=...)`` split the
     stress into kinetic, Pulay overlap, KB, neutral-atom pair, neutral-atom grid, Hartree, XC and dispersion terms
     and compare each with a fourth-order FD of the corresponding energy at fixed (D, E).

  2. ``scripts/co_strain_fd.py`` strains real Co along z (+-1e-3, +-2e-3) on a *fixed* 16x16x30 mesh with
     ``SCF.DM.Tolerance`` 1e-8 for both codes and differences the re-converged free energies.

  3. ``scripts/co_strain_components.py`` and ``scripts/co_strain_na_pair.py`` give the strain derivative of every
     energy component and of the neutral-atom pair energy.

- **Findings:**

  1. At fixed (D, E) every analytic term equals its FD: kinetic, Pulay, KB, pair, Hartree and XC to <= 2e-11
     Ha/Bohr^3 at step 1e-5 (Si nspin 1, Co nspin 2 GGA and LDA, all with NLCC). The NLCC and spin-polarised GGA
     strain derivatives are correct. The earlier mismatches came from the step: the discretised energy (tables,
     cut-off radii, mesh) is smooth only at small steps; at 1e-4 the neutral-atom grid term is off by 1.7e-6
     Ha/Bohr^3, at 3e-3 by 7e-6.

  2. At fixed mesh and tight tolerance both codes are self-consistent: SIESTA analytic 0.066991 vs 4-point FD
     0.066999; jsiesta analytic 0.065116 vs 0.065028 (0.14 %). The "SIESTA 4 % below its own FD" and "jsiesta 1.3 %
     above" numbers were artefacts of changing meshes and loose tolerances between strained runs.

  3. The 2.8 % gap is a real difference in how the energy depends on strain, d(E_jsiesta - E_SIESTA)/d(eps) =
     -0.044 eV per unit strain: neutral-atom term -0.062, kinetic +0.027, KB -0.005, XC -0.004. Of the neutral-atom
     term, the pair energy alone is -0.049 (jsiesta 28.841 vs SIESTA ``Ena`` 28.890 eV per unit strain, 0.17 %); the
     value is converged in jsiesta (unchanged with 2x k points, k_max 80 and 3x pair-spline points).

- **Decision:** no change to the stress code. The Co parity test now records the stress residuals and its tolerance
  moves from 5e-3 to 3e-3 eV/A^3 (measured 2.3e-3 without and 2.5e-3 with symmetry).
- **Open (not resolved here):** which side's neutral-atom pair table is more accurate. SIESTA's ``Ena`` comes from its
  own radial tables and interpolation; jsiesta's is a converged two-centre integral of the same ``.ion.xml`` tables.
  The kinetic derivative differs by 0.4 %. A single-atom-pair test against SIESTA's ``Ena`` as a function of distance
  would settle it. Tracked as O-8 in MEMORY.md.

## D-022: Real Broyden and ADIIS mixers (review M1, open issue O-1) (2026-11-20)
- **Status:** Accepted
- **Decision:**

  1. ``mixer_variant="broyden"`` is Johnson's modified Broyden (equivalent to Anderson/type-II mixing): with secant
     pairs ``(dx_i, dF_i)`` and the newest residual ``F_n``, ``x_next = x_n + W F_n - sum_i gamma_i (dx_i + W dF_i)``,
     ``gamma = (G + w0^2 diag(G))^-1 dF F_n`` (``w0 = 0.01``, multiplicative so it is scale invariant; an absolute
     regulariser turned the method into linear mixing once the residual differences shrank).

  2. ``mixer_variant="adiis"`` is the ADIIS-to-Tikhonov-Pulay hybrid of plan section 4.2.1 for Hamiltonian mixing. It
     minimises the augmented-Roothaan-Hall model ``f(c) = E_n + 2 b.c + c.A.c`` on the simplex (projected gradient,
     fixed iteration count, never worse than the newest entry) from the stored output density matrices, the
     Hamiltonians built from them and the energies at those densities, and blends the coefficients with the
     Tikhonov-Pulay ones as the largest residual falls from 0.1 to 1e-4 Ha (Hu and Yang's window).
     ``SCFState`` gets ``history_d`` and ``history_e`` (``None`` for the other mixers); the controller's quarantine
     rolls them with the Krylov history. ``mix_target="DM"`` is rejected for ``"adiis"``.

  3. ``SCF.Mixer.Method`` (pulay, broyden), ``SCF.Mix`` (hamiltonian, density), ``SCF.Mixer.Weight`` and
     ``SCF.Mixer.History`` are read from FDF files; anything else raises. ADIIS is selected through ``case.json``.
     GR-Pulay is not implemented.

- **Validation:** ``tests/test_mixers.py`` (Broyden finite termination: residual 1e-12 after n + 4 steps on an
  n-dimensional affine map, 17 vs 29 iterations for Pulay on a stiff model; ADIIS coefficients feasible, never worse
  than the newest entry, exact energy relation on a quadratic model). ``test_phase1_scf_and_forces.py`` runs Co hcp
  with five trajectories (adaptive Tikhonov-Pulay, fixed Pulay, coupled spin, Broyden, ADIIS hybrid): endpoints agree
  within 0.2 meV and the trajectories are pairwise different. The JIT audit in ``profile_phase3.py`` covers the new
  compiled paths (no host callbacks).
- **Benchmark (quick suite, iterations to ``d_tol`` = ``h_tol`` = 1e-5; adaptive Tikhonov-Pulay / fixed Pulay w = 0.05 /
  Broyden / ADIIS hybrid; endpoint differences to the adaptive run <= 2e-9 eV):**

  - H2O: 10 / 11 / 9 / 28.
  - Si bulk: 5 / 4 / 4 / 9.
  - Co hcp collinear: 14 / 13 / 11 / 18.
  - SiC zincblende: 7 / 10 / 9 / 13.

- **Reading the numbers:** Broyden is at least as fast as the adaptive Tikhonov-Pulay mixer on three of four systems
  and slower on SiC. The ADIIS hybrid is slower everywhere; none of the blend windows tried (0.1-1.0 down to
  1e-4-1e-2) made it faster than Tikhonov-Pulay on these well-conditioned systems, so the published window is kept.
  ADIIS targets poor initial guesses and near-degenerate cases; no such system is in the benchmark set yet. No claim of
  a speed-up is made for it.

## D-023: Harris-Foulkes energy with provenance (review M3, open issue O-3) (2026-11-20)
- **Status:** Accepted
- **Decision:**

  1. Each SCF iteration computes E_Harris = E_band[H_in] - Tr[D_in H_grid,in] + E_grid[D_in] + E_NA,pair - TS, which equals
     E_KS[D_in] + Tr[H_in (D_out - D_in)] - TS (SIESTA ``compute_energies``). The grid terms at D_in come from the
     evaluation that built H_in, so no extra grid pass is needed under Hamiltonian mixing and one is reused under DM mixing.

  2. Under DM mixing H_in = H[D_in], so the value is the exact Harris-Foulkes energy (``harris_provenance`` =
     ``exact_dm_mixing``). Under Hamiltonian mixing no such D_in exists; as in SIESTA, D_in is the previous D_out and
     H_in the mixed H (``siesta_h_mixing``), which is not variational.

  3. ``SCFState.harris_energy_ha``/``harris_provenance`` carry it; ``Observables.harris_energy_ha`` reports the value of the
     last iteration (not recomputed after convergence) with ``Observables.harris_provenance``. NaN only when a state has none.

- **Validation:** ``tests/test_harris.py`` recomputes E_KS[D_in] independently through the post-SCF evaluator and adds
  Tr[H_in (D_out - D_in)] and the entropy: agreement within 1e-8 Ha for H2O (nspin 1) and Co hcp (nspin 2), both mixing
  modes; at convergence Harris equals the KS free energy within 1e-7 Ha.
- **Not comparable with SIESTA:** SIESTA's per-iteration ``Eharris`` cannot be compared, because the trajectories differ.
  Even the first iteration is not comparable: jsiesta's cold start diagonalises ``H_0 + V_NA + V_xc[rho_NA]`` (Si first
  Harris energy -215.749 eV, already within 4 meV of the converged -215.745), SIESTA starts from the atomic density
  matrix (-215.431 eV), so the two ``D_in`` differ. At convergence Harris equals the KS free energy, which the existing
  parity tests cover.

## D-024: Phase 0 parity matrix, multi-species table fix and the SIESTA reference cache (open issues O-4, O-7) (2026-11-20)
- **Status:** Accepted
- **Findings and decisions:**

  1. Real pseudopotentials: the manifest cases run with the PseudoDojo nc-sr-05 PBE scalar-relativistic PSML set (key
     ``pseudodojo_sr_pbe_dir``), whose single scalar-relativistic projector set is unambiguous for both codes. The FR
     set also loads (the loader takes its ``scalar_relativistic`` set below nspin 8), but whether SIESTA reads the same
     projectors from an FR file without SOC was not established. The bilayer runs without D3 (validated separately).

  2. Bug found by the first multi-species parity run (H2O): the KB tables of species with different point counts were
     stacked without resampling (crash), the PAO tables were resampled *linearly* onto the first species' grid, and the
     neutral-atom tables raised. ``unify_species_tables`` (``species/psml.py``, applied in ``build_basis``) now puts all
     species on the finest grid with natural cubic splines. All earlier parity cases were single-species.

  3. Tolerances are per case and measured (about 1.3x), see ``_MATRIX_TOL`` in ``tests/test_siesta_parity.py``. Measured
     residuals against SIESTA (absolute dE_total in meV, max force in eV/Ang, max stress in eV/Ang^3):

     - H2O (box): 3.1, 0.063, 1.0e-4.
     - graphene slab: 2.7, 6.4e-5, 5.2e-5.
     - triclinic Si: 52, 0.097, 5.3e-3.
     - C bilayer (no D3): 5.3, 2.8e-4, 8.0e-5.
     - Co AFM (nspin = 2): 1.8, 1.7e-10 (zero by symmetry), 6.6e-4, spin moment 5.8e-6 muB.

     Kinetic, KB, Hartree and XC agree to <= 4 meV in every case; the Fermi level is compared only for the metals
     (bilayer, Co AFM; in a gap it is arbitrary, and graphene's 4x4 grid misses the K point).

  4. The remaining differences sit in one term, the neutral-atom pair energy (open issue O-8, narrowed down here):
     SIESTA's ``Ena`` and jsiesta's exact two-centre integral differ in their displacement derivative by 2.0 % (H2O, O/H
     pseudos), 0.3 % (triclinic Si; the pair forces are large, so this is 0.1 eV/Ang), < 0.5 % (C) and 0.17 % (Co strain), and
     by a constant 26 meV/atom for PseudoDojo Si. Each code's analytic forces equal the finite difference of its own energy
     (H2O: SIESTA -0.864 vs -0.868, jsiesta -0.927 vs -0.931 eV/Ang). SIESTA's ``Ena`` does not depend on the mesh cutoff
     (60-400 Ry) and jsiesta's value is converged in its quadrature. Rewriting the pair energy as SIESTA's gradient form
     ``sum T[V_NA,I, V_NA,J] / (16 pi)`` does not reproduce ``Ena`` either (triclinic Si 0.538 vs SIESTA 0.607 and
     jsiesta 0.604), so SIESTA's table construction would have to be replicated to close the gap.

  5. SIESTA reference cache (``JSIESTA_SIESTA_REF_CACHE``): ``run_siesta_reference`` identifies a run by the mode, the
     FDF text and the PSML bytes, replays a cached run (inputs, outputs, ion tables) when present and stores new runs.
     Registry keys ``siesta_bin_or_cache`` / ``siesta_sym_bin_or_cache`` are satisfied by a binary or a cache, so
     machines without a SIESTA binary can run the parity tests that go through ``run_siesta_reference``. ``make
     refs-cache`` fills ``refs_cache/`` (12 entries, 124 MB, git-ignored). A replay without any binary passed locally.
     Tests that call SIESTA directly (mesh selection, D3, pr_mgga, the validation-harness strain test) are not cache
     capable and are not run on the cluster.

  6. ``scripts/snellius_sync.sh`` copies the repository, the cache and the pseudopotentials to Snellius;
     ``scripts/snellius_a100_refs.sbatch`` builds the CUDA library with Libxc enabled (so the meta-GGA GPU test no longer
     skips), runs ``test_phase3_cuda.py`` and the cache-capable parity tests with ``--require-refs`` and writes
     ``reports/pytest_a100_refs.junit.xml``.

- **Validation:** all eight cache-capable parity tests pass locally with the cache filled (10.5 minutes);
  ``tests/test_phase1_species_and_two_center.py`` covers the unequal-table case.
- **Forces at fixed (D, E), displaced structures (addendum to D-021):** ``tests/test_stress_components.py::
  test_forces_match_fixed_dm_finite_differences`` moves the atoms of Si (nspin 1) and Co hcp (nspin 2, GGA, NLCC, semicore)
  off their sites and compares the force on atom 0 with a fourth-order finite difference of the fixed-(D, E) energy:
  agreement within 2e-8 Ha/Bohr. The Co parity run itself only sees forces that vanish by symmetry.

## D-025: Neutral-atom pair energy: jsiesta equals the exact value of SIESTA's formula; SIESTA's tables carry the error (open issue O-8) (2026-11-20)
- **Status:** Accepted. No change to the jsiesta energy, forces or stress.
- **Question:** the displacement/strain derivative of the neutral-atom pair energy differed from SIESTA's ``Ena`` by 0.2-2 %
  (H2O 2.0 %, triclinic Si 0.3 %, Co strain 0.17 %), which is the whole force and stress difference of D-021 and D-024.
- **What SIESTA computes** (``Src/naefs.F`` and ``Src/dnaefs.F``, summed in ``setup_H0``): ``Ena = sum_ordered T[vna_I, vna_J] /
  (16 pi) + 1/2 sum_ordered psover(R)``. The first term is the gradient overlap of the screened neutral-atom potentials and
  *includes the self term* ``R = 0`` (no distance exclusion); the second is the point-charge minus smeared-charge
  electrostatic energy of the normalised ``chlocal`` pseudo-charges (``elec_corr`` tables, ``electrostatic.f``).
- **Method:** ``jax_siesta/testing/siesta_na_reference.py`` evaluates both terms exactly (k-space integrals to 200
  inverse Bohr) from the ``vna`` and ``chlocal`` tables of SIESTA's own ``.ion.xml``.
- **Findings:**

  1. The formula and units are right: the exact value reproduces SIESTA's printed ``Ena`` in absolute value to 5 meV
     (H2O, 116.0177 vs 116.0127 eV) and 23 meV (triclinic Si, 118.8326 vs 118.8556 eV), i.e. 0.02 %.

  2. jsiesta's two-centre pair energy equals the exact value of that formula: derivative for a 0.02 Bohr step of atom 0 in z,
     H2O 0.06734 (exact 0.06737, SIESTA 0.06601 eV); triclinic Si 0.60447 (exact 0.60488, SIESTA 0.60652 eV). jsiesta differs
     from the exact value by 0.05-0.07 %, SIESTA by 2.0 % and 0.3 %.

  3. SIESTA's deviation from its own formula is smooth and systematic: for H2O, SIESTA minus exact is -4.3 meV at the
     reference geometry with a slope of -32.9 meV/Bohr (-62 meV/Ang) over a +-0.1 Bohr scan (rms 0.25 meV about the linear
     fit). That slope is the whole 0.063 eV/Ang force difference of the H2O parity case.

  4. Cause: SIESTA builds these tables at a fixed resolution. ``matel_params``: q cut-off 100 inverse Bohr, 1024 points;
     ``electrostatic.f``: the ``chlocal`` charge is integrated on its q grid and accepted if within ``cherr = 0.05`` of Z,
     then renormalised. ``Debug.Elec.Corr true`` shows the integrated charges: O 5.9854 of 6 (-0.24 %), H 0.9915 of 1
     (-0.85 %), Si 4.0108 of 4 (+0.27 %). SIESTA's ``Ena`` does not depend on ``MeshCutoff`` (60-400 Ry), as expected for
     fixed tables; harder pseudopotentials (PseudoDojo O, Si) show the largest deviation, the older tm2 Si and C almost none.

- **Decision:** jsiesta keeps its converged evaluation. The H2O and triclinic-Si force/stress differences in D-024 are
  SIESTA's table error, not a jsiesta defect, and the per-case tolerances stay as measured. Emulating SIESTA's table
  numerics was not attempted: it would make jsiesta less accurate to match a known numerical error.
- **Validation:** ``tests/test_na_pair_exact.py`` (H2O and triclinic Si, through the cache-capable SIESTA runner): the
  exact formula reproduces the printed ``Ena`` to 0.05 eV absolute; jsiesta's derivative is within 0.3 % of the exact one
  and SIESTA's is at least twice as far from it.
- **Co hcp strain case (D-021), checked with the exact formula:** the pilot Co tables are not the same story. Absolute
  ``Ena`` agrees with SIESTA's printed value to 14 micro-eV (689.796823 vs 689.796809 eV), so the formula is right, but
  the strain derivative (eV per unit strain, +-1e-3 and +-2e-3, 4-point) is exact 28.8623, jsiesta 28.8407 (-0.075 %) and
  SIESTA 28.8902 (+0.097 %): both codes deviate by about 0.1 %, in opposite directions. The -0.0495 gap of D-021 therefore
  splits into +0.0280 from SIESTA's tables (57 %) and -0.0216 (43 %) from a difference of *definition*: jsiesta's pair term
  is ``Z1 Z2/R`` minus the Hartree interaction of the PAO-based ``rho_NA``, SIESTA's is the gradient form of ``vna``
  plus the ``chlocal`` correction. The two are the same quantity only if ``vna`` is the Poisson potential of the
  density jsiesta uses, which holds to 0.05-0.07 % for the PseudoDojo tables (H2O, Si) and 0.075 % for this Co
  pseudopotential. The other component differences of D-021 (kinetic +0.027, KB, XC) are the self-consistent density
  response to that term, not separate errors: the total, which is variational, differs by -0.044 and the pair term
  alone by -0.0495. Covered by ``test_co_strain_pair_energy_against_exact_siesta_formula``.
- **Possible follow-up, not done:** replace jsiesta's pair term by SIESTA's formula (``naefs`` + ``dnaefs``, with the
  one-centre constants re-derived). That would remove the 43 % definitional part (sigma_zz gap about -2.8 % to -1.5 %) but
  leaves SIESTA's own table error, and would give up a self-consistent definition. Judged not worth the C++ and
  force/stress rework for a 0.075 % effect on one term.

- **Addendum to D-021 and D-024:** the open point about the neutral-atom pair term is closed by D-025 (SIESTA's tables, not jsiesta).


## D-026: Non-collinear GGA derivative with the frozen spin axis is accepted and bounded (review L2) (2026-11-20)
- **Status:** Accepted. No code change; the error is now measured and pinned by a test.
- **Question:** for ``nspin=4/8`` the XC energy is evaluated in the local spin frame with the axis ``u = m/|m|`` frozen
  under differentiation (libgridxc ``old_scheme``). The potential and the forces/stress that come from ``jax.grad``
  therefore omit the ``(grad u) . dm`` term of the density-gradient projection. How large is the error?
- **Measurement** (``tests/test_phase1_grid_xc.py::test_noncollinear_xc_derivative_with_frozen_spin_axis``): directional
  derivative of ``xc_energy_from_grid`` against a central finite difference on a 16^3 grid with a spin spiral (axis turning
  through 2 pi per cell plus a z-component), three spiral wave numbers: LDA 1e-10 relative (exact, no gradients), PBE
  3e-6 to 7e-5 relative. For a uniform axis (ferromagnetic canted Co) the omitted term is zero.
- **Decision:** keep the frozen axis. It is SIESTA's own scheme, so SIESTA's potential carries the same omission, the
  energy value is exact, and the effect on forces is below the parity tolerances for any realistic texture. Test bounds:
  LDA 1e-6, PBE 2e-4 relative.

## D-027: SIESTA's mesh criterion is the default (2026-11-20)
- **Status:** Accepted. Supersedes the "default stays ``jsiesta``" part of D-020.
- **Change:** ``SCFConfig.mesh_criterion``, ``choose_mesh_dimensions``, ``choose_commensurate_mesh_dimensions`` and the
  ``case.json`` reader now default to ``"siesta"`` (SIESTA's ``get_mesh_sizes`` + ``chkgmx``, ``MeshSubDivisions = 2``).
  The legacy Nyquist criterion stays available as ``mesh_criterion="jsiesta"``.
- **Why:** the criterion reproduces SIESTA's ``InitMesh`` dims in all 15 cells tested (D-020), so every cross-code
  comparison and every FDF-driven run is now on the same mesh without a per-call override, and a default run no longer
  differs from SIESTA by an arbitrary mesh choice. Co hcp at 120 Ry changes from 18x18x27 to 16x16x30.
- **Re-baselining:** the frozen Phase 0 manifest (``tests/manifests/phase0_matrix.json``) stores tolerances, not
  reference numbers (its baseline and candidate both run this code), so nothing there needed regenerating. Tests that
  pinned the old default (``test_siesta_mesh_criterion_values_and_invariants``) were updated; results that depend on the
  mesh shift within the tolerances of their tests (see the full-suite run recorded in the work log).

## D-028: Fully relativistic PSML below nspin 8 reads the same projectors in SIESTA and jsiesta (2026-11-20)
- **Status:** Accepted. No code change; covered by a new test.
- **Question:** the PseudoDojo nc-fr PSML (``relativity="dirac"``) holds three projector sets (``scalar_relativistic``,
  ``spin_orbit``, ``lj``). jsiesta uses ``scalar_relativistic`` for ``nspin < 8`` and ``lj`` only for ``nspin == 8``. Does
  SIESTA, run without spin-orbit, use the same set?
- **Check:** ``test_fully_relativistic_psml_below_nspin8_matches_siesta`` runs H2O and triclinic Si with the nc-fr-04 PBE
  files in both codes (mesh criterion matched). Kinetic and KB energies agree to 0.6 and 0.2 meV (H2O) and 0.2 meV
  (Si), Hartree, XC to 0.2 meV; the residuals of the neutral-atom pair term, forces and stress equal those of the
  scalar-relativistic matrix of D-024 (SIESTA's table error, D-025). A different projector set would move the KB energy by
  tens of meV. Hence the two codes select the same channels.
- **Added:** reference key ``pseudodojo_fr_pbe_dir`` (``tests/external_refs.toml``) and its sync to Snellius.

## D-029: Every SIESTA-calling cross-code test can replay from the reference cache (2026-11-20)
- **Status:** Accepted.
- **Change:** ``run_siesta_directory`` (``jax_siesta/testing/siesta_adapter.py``) runs a SIESTA binary on a prepared
  directory (FDF on stdin) and caches the run in ``JSIESTA_SIESTA_REF_CACHE`` keyed by executable key + FDF + PSML bytes,
  like ``run_siesta_reference``; both share ``_cache_replay`` / ``_cache_store``. The four tests that called SIESTA
  directly now use the cache: ``test_siesta_mesh_selection_matches_siesta`` (15 set-up runs), ``test_d3_matches_siesta``,
  ``test_pr_mgga_si_bulk_noncol_and_soc_parity`` and, through the adapter subprocesses,
  ``test_siesta_strained_displaced_via_validation_harness``. Their markers are ``*_or_cache``.
- **Bug found on the way:** the ``external_refs`` and ``skipif`` marks of ``test_mgga_soc.py`` were attached to the helper
  ``_mgga_residuals`` instead of ``test_pr_mgga_si_bulk_noncol_and_soc_parity``, so that test had no reference gating at all.
- **Verified:** the four tests pass with the binaries pointed at ``/nonexistent`` and only the cache (``tests/test_ref_cache.py``
  covers hit/miss semantics with a fake executable). ``make refs-cache`` and ``scripts/snellius_a100_refs.sbatch`` now run all
  of ``REFS_CACHE_TESTS``; ``snellius_sync.sh`` also sends the nc-fr PseudoDojo set and the ``pr_mgga`` ``Pseudos``/``10.Functionals`` inputs.

## D-030: Mixers on hard starts (matched basis, Co AFM cell, 120 Ry) (2026-11-21)
- **Status:** Accepted. No code change; measurement only.
- **Setup:** the Phase 0 ``co_afm`` cell with SIESTA's ``Co.ion.xml`` tables (so the basis equals SIESTA's), three initial
  moment patterns (AFM +2/-2, ferromagnetic +2/+2, weak AFM +0.1/-0.1) and the four mixers, matrix tolerances
  (``d_tol = h_tol = 1e-6``).
- **Result:** all 12 runs converge and every mixer reaches the same endpoint for a given start. From the AFM and weak-AFM
  starts that is the SIESTA state (moment +-0.743 mu_B per atom, total energy 1.77 meV from SIESTA's -8325.866 eV); from the
  ferromagnetic start it is a different, genuine stationary state at -8326.728 eV (862 meV lower, 3.41 mu_B per cell). Iterations
  (AFM / ferro / weak AFM): Tikhonov-Pulay 37 / 21 / 21, Broyden 42 / 26 / 20, ADIIS 42 / 35 / 29, fixed Pulay 120 / 78 / 40.
- **Reading:** on these starts ADIIS gives no advantage over the adaptive Tikhonov-Pulay default, and Broyden is on par; the
  fixed-weight Pulay is the slowest by a factor 2-3. The earlier 60 Ry scratch runs that appeared to give different endpoints
  (28 eV apart) used jsiesta's own PAO generation, not SIESTA's tables, at a cutoff too low for this pseudopotential, and are
  not evidence about the mixers.

## D-031: Vectorised neighbour searches in the set-up (P2 of `known_bottlenecks.md`) (2026-11-21)
- **Status:** Accepted (branch `perf/bottlenecks`).
- **Change:** the scalar Python loops of `build_sparse_pattern` (`species/psml.py`) and of `_eval_kb_and_na_pair`
  (`integrals/two_center.py`: KB orbital/projector links, neutral-atom pairs, the exact atomic pair-density term) are replaced
  by per-atom distance tables over all atoms and images (`_enumerate_kb_links`, `_enumerate_atom_pairs`), with an atom-level
  cut-off test before the per-orbital one. Same criteria (including the `1e-8`/`1e-9` margins), same output order.
- **Physics unchanged:** on the nine manifest cases (Si bulk and triclinic, H2O, graphene, C bilayer, Co hcp
  collinear/AFM/NC/SOC) and the 8- and 16-atom Si supercells, the pattern (sources, destinations, cell shifts, edge vectors,
  transposes, `row_ptr`), S, T, their derivatives, V_KB, V_KB^SO and the KB link values are bitwise identical to the previous
  code. E_NA pair, the NA pair forces and stress differ only in rounding (summation order): at most 2.3e-13 Ha on E_NA at 16 atoms
  and 2e-16 Ha/Bohr on forces. New test `test_vectorised_neighbour_searches_match_scalar_reference` (4 cases) compares the three
  searches with the scalar algorithms.
- **Effect (laptop, CPU):** pattern + two-centre set-up 26.5 s -> 3.9 s at 8 atoms, 60.7 s -> 3.7 s at 16, 8.6 s at 64 and
  15.3 s at 128 atoms (the scaling study measured 331 s and 1309 s on Snellius for 64 and 128 atoms). Growth is now roughly
  linear; the remainder is the compiled S/T engine and the per-species-pair tables.
- **Minor behaviour change:** shifts outside the search box (possible only through symmetry `extra_edges`) are appended to the
  shift table in sorted order instead of set-iteration order. Results do not depend on the shift index order.

## D-032: Eigensolver: overlap transform once per SCF, real arithmetic for real Bloch problems (F7, B4) (2026-11-21)
- **Status:** Accepted (branch `perf/bottlenecks`).
- **Finding:** every SCF iteration rebuilt the orthogonalising transform of the constant overlap (`eigh(S)`, Cholesky,
  `inv(L)`); under `jax.vmap` the batched `lax.cond` became a `select`, so the Löwdin branch (with its own `eigh` of the
  transformed H) ran as well; and the problem was always solved in complex arithmetic, also at Gamma.
- **Change (`scf/eigensolver.py`):** `batched_overlap_transform` / `GeneralizedEigenSolver.overlap_transform` compute
  `X = L^-dagger` (Cholesky, `min eig S > 1e-8`) or `S^-1/2` (Löwdin) with the previous criterion; `solve(...,
  s_transform=X)` then costs one `eigh(X^dagger H X)` per k-point and spin. `run_scf` computes `X` once and passes it to
  the jitted solve as an argument. When every `2k` is integer and nspin is 1 or 2, `H(k)` and `S(k)` are real and the
  problem is solved in real arithmetic. The Phase 3 profiler times the solve the same way as the driver.
- **Physics unchanged:** on the complex path (Si 3x3x3, graphene 4x4x1, Co hcp nspin 2/4/8) full SCF runs give bitwise
  identical iteration counts, energies, forces, stress and moments; on the real path (H2O at Gamma, Si 2x2x2) the
  differences are rounding: at most 3.6e-15 Ha in energies and 8e-15 Ha/Bohr in forces. New `tests/test_eigensolver.py`
  (pencil identity, Löwdin branch, real vs complex DM to 1e-11, one `eigh` in the jaxpr with the transform).
- **Effect (laptop CPU, one Gamma solve, random pencil):** 0.38 s -> 0.019 s at 416 orbitals, 2.98 s -> 0.12 s at 832,
  25.0 s -> 0.86 s at 1664 (29x); a single real `eigh` is 0.57 s at 1664. The A100 in-SCF solve was about 4x the plain
  `eigh(h, s)` benchmark for the same reasons (`known_bottlenecks.md` 3.3); to be re-measured on Snellius.

## D-033: Bloch assembly by gathers instead of a scatter-add (2026-11-21)
- **Status:** Accepted (branch `perf/bottlenecks`).
- **Finding:** `_bloch_assemble` (`scf/eigensolver.py`) placed the edge values into the dense `H(k)`/`S(k)` with
  `zeros.at[:, src, dst].add(...)`. On the XLA CPU backend a scatter-add with repeated indices costs about 6 us per edge:
  2.37 s at 16 Si atoms (397 488 edges), against 3.6 ms for the `eigh` itself. It ran for H every iteration (and for S too
  before D-032), so it was most of jsiesta's CPU "eigensolver" time in the scaling study, not XLA's `eigh` (B4).
- **Change:** `_assembly_plan` builds, on the host and once per trace, a padded table of the edges of each distinct
  `(mu, nu)` pair and the dense position of each pair; the assembly is then two gathers and a sum over the (few) periodic
  images. A traced pattern falls back to the scatter-add.
- **Physics unchanged:** same SCF comparison as D-032 (seven cases, against the code before D-032): identical iteration
  counts; differences at rounding level (at most 3.9e-14 Ha in energies, 1.5e-14 Ha/Bohr in forces, 1.8e-13 in moments).
  `test_gather_assembly_equals_scatter_assembly`.
- **Effect (laptop CPU, 16 atoms, Gamma):** assembly 2.37 s -> 0.4 ms; one full `solve` 4.78 s -> 5 ms with the
  transform of D-032.

## D-034: CPU `dfscf` backward on the cached atom-block path, threaded (B3 on the CPU) (2026-11-21)
- **Status:** Accepted (branch `perf/bottlenecks`).
- **Finding:** `DfscfGradCoreImpl` (`cpp/src/grid_ffi.cpp`), the backward of `rhoofd` used by `jax.grad` for forces and
  stress, rebuilt the active-orbital cache on every call and ran a single-threaded loop over all pairs of active orbitals
  at every grid point with an edge hash lookup per pair, while the forward kernels used the cached, threaded atom-block
  path. Laptop: 34 s for forces + stress at 16 Si atoms, 170 s at 64.
- **Change:** for non-meta-GGA calls the backward uses `get_or_build_grid_plan(..., need_grad=true)` (values and
  gradients per atom image) and dense DM blocks, with per-thread accumulators for d_DM, d_positions and d_cell. The algebra
  is the same per (orbital pair, point) term; pairs without an edge have a zero DM entry and their d_DM is not mapped back.
  The meta-GGA backward keeps the generic loop. `resolve_num_threads` now honours `OMP_NUM_THREADS` up to the hardware
  thread count (it was capped at 16, so 16 of the 18 cores in the Snellius CPU scaling job).
- **Physics unchanged:** SCF + forces + stress for H2O, Si (3x3x3), graphene, Co hcp nspin 2/4/8: identical energies and
  iteration counts, forces within 3e-13 Ha/Bohr and stress within 2e-13 Ha/Bohr^3 of the previous code (summation order).
  `tests/test_stress_components.py` (analytic terms = fixed-(D, E) finite differences to 1e-9) passes.
- **Effect (laptop, 16 atoms):** forces + stress 34.2 s -> 3.8 s.

## D-035: Symmetrised-vs-unsymmetrised cross-check pinned to one mesh (regression from D-027) (2026-11-21)
- **Status:** Accepted (branch `perf/bottlenecks`); the failure is on `main` as well (not caused by D-031 to D-034).
- **Finding:** the first full `make check` after D-027 failed `test_symmetry_forces_stress_fd_rotation_permutation_and_validation`:
  at 80 Ry the unsymmetrised and symmetrised (IBZ) Si runs differ by 24.6 meV in free energy (tolerance 0.65 meV); forces
  and stress agree. Diamond's non-symmorphic translations (1/4, 1/4, 1/4) require N divisible by 4: with the SIESTA mesh
  criterion (default since D-027) the plain run gets 18^3 and the symmetry-commensurate one 20^3, so the two runs compare
  different meshes. With the legacy criterion both get 24^3, which is why the test passed before D-027. Same numbers on
  `main` and on the branch (-147.10680 / -147.13136 eV).
- **Change:** the test sets `mesh_criterion = "jsiesta"` for this cross-check and asserts that the plain and commensurate
  meshes are equal. No production code change.
- **Consequence for users:** with the default criterion, a symmetrised run of a non-symmorphic crystal can use a finer
  mesh than the same unsymmetrised run (SIESTA itself does not symmetrise the mesh); compare such runs on one mesh.

## D-036: CUDA rhoofd/vmat on the cached host plan (P1 of `known_bottlenecks.md`) (2026-11-21)
- **Status:** Accepted (branch `perf/bottlenecks`); validated on one A100 (Snellius job 29555653, revision `4b72090`,
  jax 0.12.1, CUDA 12.6): `tests/test_phase3_cuda.py` (GPU vs CPU to 1e-11, discrete adjoint to 1e-12, end-to-end SCF
  and observables) passes on the plan path and on the legacy kernel (`JSIESTA_CUDA_GRID_LEGACY=1`), 11 passed each; the
  Libxc meta-GGA test skips because that build has no Libxc. JUnit files and log: `reports/perf/`. Confirmed by job
  29559809 (revision `5af4442`): 17 passed on the plan path (with `tests/test_eigensolver.py`), 11 on the legacy path.
- **Change:** `cpp/include/jsiesta/grid_plan.hpp` exposes the CPU plan cache of `grid_ffi.cpp` (orbital values per
  atom image and grid point, dense atom-pair DM blocks, block-to-edge maps) as `get_flat_grid_plan`. The CUDA `rhoofd`
  and `vmat` handlers copy their (small) inputs to the host, get the cached plan, upload it once per geometry (device LRU of
  two) and contract with one warp per grid point: `rhoofd` without atomics (lanes over the block entries, shuffle
  reduction), `vmat` with `atomicAdd` on consecutive block entries, then a gather to the edges. The edge-parallel kernel
  remains the fallback (meta-GGA, backward modes, plans without the atom-block form, device memory short) and can be forced
  with `JSIESTA_CUDA_GRID_LEGACY=1` for A/B runs.
- **Effect (A100, Si supercells, `reports/perf/jsiesta_a100_perf_*.json`):** one SCF iteration 5.18 s -> 0.051 s at
  16 atoms, 9.45 s -> 0.072 s at 32 and 35.5 s -> 0.293 s at 128 (100-130x); 0.487 s at 216 and 1.87 s at 512 atoms (of
  which the dense solve 0.97 s). Against SIESTA as measured in the scaling study (non-exclusive nodes, assessment A6
  still open) the A100 iteration is now 1.6x (16 atoms) to 7.5x (512) faster than 18 ranks and 1.4-1.8x faster than
  192 ranks. Forces + stress on the GPU are still slow (150 s at 128 atoms; the backward uses the edge kernel). Same-job
  A/B at 64 atoms (job 29559809): `rhoofd` 9.44 s -> 0.047 s, `vmat` 8.20 s -> 0.068 s, iteration 17.7 s -> 0.117 s
  (152x); at 8 atoms the iteration is 0.0128 s. `vmat` (atomics) is now the largest kernel.
- **Effect (A100, job 29589348, Si supercells):** forces + stress at 64 atoms 76.1 s -> 7.98 s (PBE) and 265.5 s ->
  9.05 s (r2SCAN); at 8 atoms 10.4 s -> 1.16 s and 35.0 s -> 1.31 s. Grid backward stage at 64 atoms 68.0 s -> 0.14 s
  (PBE) and 257.2 s -> 0.76 s (r2SCAN). Kernels at 16 atoms, edge kernel vs plan in the same job: `rhoofd` backward
  17.5 s -> 0.028 s (620x), `tauofd` backward 47.7 s -> 0.053 s (900x). The KB projector forces on the host (7.0 s at
  64 atoms) are now 88 % (PBE) and 77 % (r2SCAN) of the force time.
- **Compile-time evidence (ptxas, sm_80):** edge kernel Modes 0/1 148/162 registers and a 1712-byte stack frame per thread,
  Mode 2 236 registers and 6.4 kB, Mode 5 255 registers, 18 kB and spills (assessment F3); new plan kernels 56-90 registers,
  no stack.
- **Known limits:** `vmat` sums with atomics, so its last bits vary from run to run (the old `vmat` was deterministic, the
  old `rhoofd` was not); the GPU backward (forces) still uses the edge kernel; the plan path needs the host plan's atom-block
  form (`n_atoms^2 * n_shifts <= 6.4e7`, raised from 4e6 so that the CPU and GPU fast paths reach about 1500 atoms with
  27 shifts; the dense pair table costs 4 bytes per slot, at most 256 MB) and fits within the memory XLA leaves
  free (`XLA_PYTHON_CLIENT_MEM_FRACTION`).

## D-037: CUDA tauofd/vmat_tau (meta-GGA SCF kernels) on the cached host plan (2026-11-22)
- **Status:** Accepted (branch `perf/gpu-forces`); validated on one A100 (Snellius job 29586936, revision `bf77aa8`,
  Libxc 6.2.2): `test_cuda_mgga_grid_and_dfscf_parity` (GPU `tauofd` and `vmat_tau` against the CPU to 1e-11) and
  `test_cuda_mgga_xc_and_end_to_end_scf_parity` pass, with the rest of `tests/test_phase3_cuda.py` (12 passed).
- **Effect (A100, same job, 16 Si atoms, r2SCAN):** `tauofd` 21.1 s -> 0.015 s, `vmat_tau` 17.5 s -> 0.021 s against the
  edge kernel (`JSIESTA_CUDA_GRID_LEGACY=1`); one SCF iteration 43.2 s -> 0.132 s. 64 atoms: 0.473 s per iteration.
- **Why:** for meta-GGA production runs the SCF iteration also calls `tauofd` and `vmat_tau` (Modes 3 and 4), which
  still used the edge kernel of D-036's diagnosis (191-202 registers, 6.4 kB stack per thread). On the laptop CPU they
  cost 2.5-3x `rhoofd` each at 8 Si atoms.
- **Change (`cpp/cuda/rhoofd_vmat.cu`):** the device plan can carry the orbital gradients of the host plan
  (`get_flat_grid_plan(..., need_grad=true)`, the arrays the CPU `tauofd`/`vmat_tau` already use). The D-036 kernels are
  templated on the number of components: `tau_s(p) = 1/2 sum_ab sum_c d_c phi_a^T D_s,ab d_c phi_b` and
  `V_tau,s,ab += dvol/2 V_tau,s(p) sum_c d_c phi_a d_c phi_b^T`, the same algebra and factors as the CPU plan path.
  The device LRU keeps two plans, so the value plan (`rhoofd`, `vmat`) and the gradient plan (`tauofd`, `vmat_tau`)
  of a meta-GGA SCF step stay resident together. Fallback to the edge kernel as before.
- **Compile-time evidence (ptxas, sm_80):** the twelve plan kernels (`rhoofd`/`vmat` x 1, 2, 4 spin components x 1 or 3
  orbital components) use 48-80 registers and no stack.
- **Known limits:** the gradient plan stores four arrays per orbital value instead of one, and a meta-GGA run keeps
  both plans on the device (five units of value storage). A value request could reuse a resident gradient plan; not
  done yet. The meta-GGA backward (Mode 5) is still the edge kernel.

## D-038: Forces backward (rhoofd and tauofd) on the cached plan, CPU meta-GGA and GPU (2026-11-22)
- **Status:** accepted (branch `perf/gpu-forces`). CPU verified locally; GPU validated on an A100 in Snellius job
  29589348 (revision `81890c1`): all 12 tests of `tests/test_phase3_cuda.py` pass, including
  `test_cuda_grid_and_dfscf_parity_and_adjointness`, `test_cuda_mgga_grid_and_dfscf_parity` (GPU vs CPU to 1e-11) and
  both end-to-end observables tests (PBE and Libxc r2SCAN01).
- **Why:** GPU forces + stress were 75 s at 64 Si atoms against 0.12 s per SCF iteration, because both backward modes
  (Mode 2 `dfscf_grad`, Mode 5 `tau_dfscf_grad`) still used the edge kernel. On the CPU the meta-GGA backward was the
  single-threaded generic loop with an edge-hash lookup per orbital pair: 17.0 s of the 21.4 s r2SCAN forces at 8 atoms.
- **Change (host plan and CPU, `cpp/src/grid_ffi.cpp`):** the host plan can store orbital Hessians (`need_hess`,
  implies gradients; part of the cache key). The CPU backward fast path of D-034 now covers the meta-GGA too: per atom
  pair `R_a = sum_b w_ab grad phi_b`, `f_a = 1/2 H_a R_a` and the mirrored b-side term, the same terms as the generic
  loop. `get_flat_grid_plan` exports the Hessians (`FlatGridPlan::has_hess`, `img_h**`).
- **Change (GPU, `cpp/cuda/rhoofd_vmat.cu`):** `plan_dfscf_kernel<NS, TAU>`, one warp per grid point. The pair terms
  are regrouped per atom image X (a-side of the blocks (X, Y) plus b-side of the blocks (Y, X)), so each image's force
  is reduced once per grid point; `d_dm` by atomics on the block entries (as `vmat`), `d_pos` by atomics per image,
  `d_cell` reduced in shared memory per thread block. Mode 2 uses the gradient plan (shared with `tauofd`), Mode 5 the
  Hessian plan.
- **Physics unchanged (CPU):** SCF + forces + stress against the previous library: Si PBE bitwise identical; Si r2SCAN
  forces within 5.6e-15 Ha/Bohr, stress 3.3e-16 Ha/Bohr^3; spin-polarised Co hcp r2SCAN forces within 2.6e-13 Ha/Bohr,
  stress 3.3e-14 Ha/Bohr^3 (summation order). `test_tauofd_vmat_tau_adjoint_and_tau_core_fd` (backward vs finite
  differences, nspin 1/2/4) and `test_mgga_scf_forces_and_stress_vs_finite_differences` pass.
- **Effect (laptop CPU, 8 Si atoms, r2SCAN):** `tauofd` backward 17.0 s -> 1.36 s; forces + stress 21.4 s -> 4.5 s
  (warm plan; the first call per geometry also builds the Hessian plan).
- **Compile-time evidence (ptxas, sm_80):** new backward kernels 121-134 registers, no stack; old Mode 2 236 registers and
  6.4 kB stack, Mode 5 255 registers, 18 kB stack and spills.
- **Known limits:** atomics make the last bits of GPU forces vary between runs (as `vmat`); the Hessian plan stores ten
  arrays per orbital value, so a meta-GGA force call on the GPU holds up to two large plans (device LRU of two).

## D-039: LDA/GGA exchange-correlation on the device when Libxc is compiled in (2026-11-22)
- **Status:** accepted and A100-validated (branch `perf/gpu-forces`, job 29590887 on `ef44c1a`): all 20 CUDA tests pass,
  including the 8 cases of `test_cuda_builtin_xc_pointwise_matches_cpu` (device kernel vs CPU Libxc), `test_cuda_xc_parity`
  and the end-to-end observables tests.
- **Why:** with Libxc compiled in, `XcEvalCudaImpl` copied the grid to the host and called Libxc for every functional,
  PBE included: Hartree + XC took 0.094 s of the 0.200 s PBE iteration at 64 Si atoms on the A100 (job 29589348), and the
  device kernel (`xc_eval_cuda_kernel`) was only used without Libxc. The device kernel could not simply be switched on:
  its PBE correlation used the original PW92 digits, while Libxc's `GGA_C_PBE` uses `PW_MOD` (A = 0.0310907, 0.01554535,
  0.0168869), a relative difference of about 1e-5 in the correlation energy, and its density floors differed from Libxc's.
- **Change:** one host/device header, `cpp/include/jsiesta/xc_semilocal.hpp` (`jsiesta::xc::eval_semilocal_xc_point`),
  replaces the two copies of the dual-number kernel in `cpp/src/xc_ffi.cpp` (CPU build without Libxc) and
  `cpp/cuda/rhoofd_vmat.cu`. Parameters follow the Libxc ids that `xc_ffi.cpp` calls: `LDA_C_PW` (12, original digits and
  fz20 = 1.709921) for LDA_PW92, `PW_MOD` for PBE and PBEsol correlation. The thresholds follow Libxc too: each part
  floors its inputs at its own density threshold (1e-15 exchange and LDA correlation, 1e-12 PBE correlation), clamps
  `sigma_ud`, screens empty spin channels in exchange, and unpolarised input is evaluated in total n and sigma with the
  wrapper's output mapping. `XcEvalCudaImpl` now evaluates ids 1, 2, 10 and 11 on the device; the host Libxc copy remains
  for A/B runs (`JSIESTA_CUDA_XC_LIBXC=1`). New C entry point `jsiesta_xc_eval_builtin` runs the same arithmetic on the
  host for testing.
- **Physics:** `test_builtin_xc_kernel_matches_libxc` (LDA, LDA_PW92, PBE, PBEsol; nspin 1 and 2; 20000 points from
  1e-11 to 20 a.u., reduced gradients up to 3): energy density within 1.3e-14 relative, v_rho within 5e-13 relative, and
  the v_sigma error times sigma within 6e-15 of e_xc. A CPU build without Libxc now gives the Libxc PBE energies (before,
  about 1e-5 relative in the correlation energy).
- **Effect (A100, job 29590887, PBE, 64 Si atoms, A/B in the same job):** Hartree + XC 0.096 s (host Libxc,
  `JSIESTA_CUDA_XC_LIBXC=1`) -> 0.0006 s (device kernel); one SCF iteration 0.199 s -> 0.103 s. At 8 and 16 atoms the
  Hartree + XC stage is now 0.3 ms (was 12 and 24 ms in job 29589348).
- **Known limits:** meta-GGA XC (`XcEvalMggaCudaImpl`) still evaluates Libxc on the host; custom `LIBXC-<x>+<c>` specs use
  that route as well.

## D-040: Basis pressure in eV, and the basis optimiser checked against SIESTA (2026-11-22)
- **Status:** accepted (branch `perf/gpu-forces`).
- **Bug:** `EV_PER_BOHR3_PER_GPA` in `jax_siesta/species/basis_opt.py` held 6.7979e-5, which is 1 GPa in Ry/Bohr^3 (SIESTA's
  `units_codata2018.f90`: `GPa = 6.79786e-5`, energies in Ry), used as eV/Bohr^3. The p V term of the basis enthalpy was
  therefore 13.6x too small and every optimisation ran at 1/13.6 of the requested pressure (0.2 GPa acted as 0.015 GPa).
  Found by comparing with SIESTA's `BASIS_ENTHALPY` for the same basis: SIESTA's p V = 0.963 eV, jsiesta's 0.071 eV. The
  existing test (smaller volume at 5 GPa than at 0.2 GPa) holds with either constant.
- **Fix:** the constant is SIESTA's `GPa / eV` (9.2489e-4 eV/Bohr^3); `test_basis_pressure_unit_matches_codata_and_siesta`
  checks it against CODATA 2018 and SIESTA's printed pressure.
- **Effect (Si bulk, Si.tm2.nrl.psml, PBE, DZ 3s/3p, four cut-off radii, 0.2 GPa, Nelder-Mead 60 evaluations,
  `scripts/basis_opt_vs_siesta.py`):** with the old constant the radii ran to 7.5/3.6 (3s) and 9.5/4.4 Bohr (3p); with the
  fix the optimum is 6.03/3.35 (3s) and 7.14/3.81 Bohr (3p).
- **Check against SIESTA (same explicit `PAO.Basis` block in both codes, start, optimum and +-0.25/+-0.5 Bohr line scans of
  each radius, 18 bases):** the free energies differ by a near-constant 16-41 meV per 2-atom cell (mean 28 meV). The
  enthalpy landscapes agree: curvature along each radius within 3 %, position of the line minimum within 0.05 Bohr
  (SIESTA's minima lie 0.135, 0.136, 0.004 and 0.127 Bohr from jsiesta's optimum, jsiesta's own 0.137, 0.124, 0.056 and
  0.121), and the enthalpy gain start -> optimum is -99 meV (jsiesta) and -85 meV (SIESTA). So a basis optimised with
  jsiesta is near-optimal in SIESTA; the 60-evaluation optimum itself is about 0.13 Bohr from the line minima (budget).
- **Cost:** 12-24 s per jsiesta evaluation (laptop CPU, PAO generation + two-centre + SCF) against 1.5-1.7 s for SIESTA on
  this 2-atom cell.
- **Not covered:** polarisation shells, soft-confinement and charge parameters, multi-species bases; a comparison with
  SIESTA's own simplex optimiser.

## D-041: KB projector terms (V_KB assembly, energy, forces, stress) in a threaded C++ kernel (2026-11-22)
- **Status:** accepted and A100-measured (branch `perf/gpu-forces`, scaling job 29594601 on `f7313a9`: 25 CUDA tests pass, 1 meta-GGA Libxc test skipped because that build has no Libxc).
- **Why:** the KB projector forces (`_kb_forces_and_stress`) took 7.0 s of the 8.0 s A100 forces + stress at 64 Si atoms
  (88 %, job 29589348) and about 0.11 s per atom on every machine, because they ran on the host as a NumPy loop over the
  384 KB shell instances with large fancy-index temporaries, `np.add.at` and a `searchsorted` per orbital pair (about 24 M
  pairs at 64 atoms). The V_KB assembly in the two-centre set-up used the same loop.
- **Change:** `cpp/src/kb_three_center.cpp`, `jsiesta_kb_three_center` (C ABI, called through
  `jax_siesta.integrals.two_center.kb_three_center_native`). Links are sorted by (shell, orbital, shift, m); for each link
  group of a row it forms `W_m = sum_c D(mu, nu_c, t_r - t_c) P_c,m` over the groups of the same shell (pattern edge by
  binary search in the row's sorted keys), then the energy `E_k sum P W`, the force `2 E_k sum dP W` on the orbital's atom
  with the reaction on the projector atom, and the stress (force times bond vector). Rows are split into contiguous
  ranges per thread (`OMP_NUM_THREADS`), each thread writes only its own rows' V_KB entries and keeps private force and
  stress sums added in thread order, so the result is deterministic. The NumPy loop stays for spin-orbit shells
  (nspin 8) and for libraries built without the kernel.
- **Physics unchanged:** V_KB bitwise identical to the NumPy assembly (Si supercells); KB energy, forces and stress within
  1.4e-13 Ha, 8e-15 Ha/Bohr and 1.3e-17 Ha/Bohr^3 at 64 atoms (nspin 1 and 2); full observables against the previous code:
  energies identical, forces within 2.6e-13 Ha/Bohr, stress within 3.3e-14 Ha/Bohr^3 (Si PBE, Si r2SCAN, Co r2SCAN).
  New test `test_native_kb_kernel_matches_numpy_reference` (displaced Si + Co cell, l = 0-2, unequal tables, nspin 1 and 2).
- **Effect (laptop CPU, 8 threads, 64 Si atoms):** KB forces + stress + energy 7.3 s -> 0.30 s (25x); two-centre set-up
  7.1 s -> 5.9 s (the rest of the set-up is the overlap integrals); 8 atoms 1.1 s -> 0.035 s. Co r2SCAN observables case
  50.6 s -> 36.9 s.
- **Effect (A100 node host, PBE, job 29594601 against job 29590887):** KB stage at 64 atoms 7.54 s -> 0.24 s; forces +
  stress 0.31, 0.40, 0.59, 0.96, 1.71 and 2.62 s at 8, 16, 32, 64, 128 and 216 atoms (64 atoms: 8.3 s before; SIESTA
  PostSCF 1.06 s on 18 ranks, 0.29 s on 192). The largest stage is now the two-centre derivatives (0.32 s at 64 atoms).

## D-042: Device grid plan reuses a richer cached plan and evicts before giving up (2026-11-22)
- **Status:** accepted and A100-validated for the evict-and-retry part (job 29607477 on `18cb66a`: 25 CUDA tests pass; 512-atom PBE forces + stress 546 s -> 10.1 s). The cache-key part (below) is CPU-verified and awaits an A100 run.
- **Why:** in scaling job 29594601 the 512-atom PBE forces took 546 s, of which 540 s in the grid backward, while 216
  atoms took 0.34 s. The backward needs the plan with orbital gradients (4 arrays of 385 M values for 512 Si atoms,
  13 GB); it was uploaded next to the cached value plan of the same geometry (3.8 GB) in the 40 % of device
  memory that JAX leaves free (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`). The upload failed and `get_device_grid_plan`
  returned no plan without a message, so the edge kernel ran.
- **Change:** in `cpp/cuda/rhoofd_vmat.cu`, a cached plan with gradients (or Hessians) now serves requests that need
  less, so a value request after the backward does not upload a second copy. When an upload fails, the cached plans are
  dropped and the upload is retried once. If it still fails, one warning on stderr gives the plan size and the free
  device memory. `JSIESTA_CUDA_PLAN_VERBOSE=1` prints the size of every upload.
- **Cache key:** job 29607477 showed the value plan being uploaded again after the gradient plan, because the cache key
  of `GridKernelPlan` included the gradient and Hessian flags, so the "richer plan" check never matched. The inputs are
  now hashed once (`grid_plan_input_hash`) and the flags folded in last (`grid_plan_cache_key`). The device looks for the
  requested variant and then the richer ones, before building the host plan.
- **Effect (A100, job 29607477, PBE):** 512 atoms: forces + stress 10.1 s (grid backward 3.6 s including the 13.0 GB
  plan upload, two-centre 2.8 s, KB 1.9 s), against SIESTA PostSCF 15.8 s (18 ranks) and 1.5 s (192). Plan sizes logged
  with `JSIESTA_CUDA_PLAN_VERBOSE=1`: 3.81 GB values only, 13.04 GB with gradients (385 M orbital values); 216 atoms
  1.28 and 4.39 GB.
- **Physics:** none; the plan arrays and kernels are unchanged, and a value-only request reads the same `vals` array from
  a gradient plan.
- **Known limits:** if the gradient plan alone does not fit (meta-GGA backward with Hessians at 512 atoms, about 23 GB),
  the edge kernel still runs, now with the warning. A streamed or chunked plan would remove that limit.

## D-043: Stream the device grid plan from the host when it does not fit (2026-11-23)
- **Status:** accepted and A100-validated (streaming: job 29607988; host memory: job 29608548).
- **Why:** the cached grid plan stores every orbital value (and gradient, and for the meta-GGA backward Hessian) per
  grid point. At 512 Si atoms that is 3.8 GB for values, 13.0 GB with gradients and about 23 GB with Hessians, against
  the 16 GB JAX leaves free on a 40 GB A100. When the plan did not fit, the code fell back to the edge kernel, which is
  two orders of magnitude slower (546 s against 10 s for the PBE backward in jobs 29594601 and 29607477). The r2SCAN
  backward at 512 atoms and anything much larger than 512 atoms would always fall back.
- **Change:** a third mode between the resident plan and the edge kernel. Images and values are stored in grid-point
  order, so a range of grid points is one contiguous slice of every array. A streamed plan keeps only the index arrays on
  the device (image offsets, atom-pair block table, block-edge maps) and the host plan alive; each call copies, chunk by
  chunk, the images and only the value arrays that kernel reads, into one set of buffers, and launches the same kernels
  on the chunk. The kernels (`plan_rhoofd_kernel`, `plan_vmat_kernel`, `plan_dfscf_kernel`) take a grid-point range and
  base offsets for the image and value arrays; the resident plan is the one-chunk case. Order of attempts: resident,
  resident after evicting the cache, streamed, edge kernel (with a warning). The chunk budget is half the free device
  memory. Environment: `JSIESTA_CUDA_PLAN_STREAM` (unset automatic, 1 always streamed, 0 never),
  `JSIESTA_CUDA_PLAN_CHUNK_MB` (budget) and `JSIESTA_CUDA_PLAN_CHUNK_VALS` (values per chunk, for tests).
- **Physics:** none; the same kernels on the same arrays. Results differ from the resident plan only through the order
  of atomic additions. GPU gate: `test_cuda_grid_and_dfscf_parity_and_adjointness` and
  `test_cuda_mgga_grid_and_dfscf_parity` now run with the plan resident and streamed (20000 values per chunk, so the
  16^3 test mesh is split into many chunks), against the CPU at 1e-11 (values) and 1e-10 (gradients).
- **A100 job 29607988 (`18cb66a` + this change):** all 22 CUDA tests pass with Libxc built in (no skips), including
  both grid tests with the plan streamed in many chunks. 512 Si atoms, PBE: plan resident, SCF iteration 1.79 s and
  forces + stress 6.46 s (10.1 s in job 29607477; the flag-free cache key of D-042 removed the second value-plan upload:
  grid backward 3.6 -> 1.2 s). Plan forced to stream (1 chunk of 3.7 GB for values, 2 of up to 8.2 GB with
  gradients): iteration 3.37 s, forces + stress 9.5 s, so about 0.8 s per 3.8 GB copied (about 5 GB/s from pageable
  memory). r2SCAN at 216 atoms: all three plans resident (1.28, 4.39 and 10.59 GB), forces + stress 5.6 s. r2SCAN at
  512 atoms was killed by the host out-of-memory handler (97 GB of the 120 GB of the job) while building the host plan
  with Hessians.
- **Host memory (same change):** the host plan kept every per-entry record (96 bytes per orbital value at a point) next
  to the per-image arrays, the build held a second copy while scattering them, the per-image arrays grew one image at
  a time (up to twice their size while growing), and the cache kept the value and gradient plans of a geometry next to
  its Hessian plan. Now the entries are counted in a first sweep and written in place in a second (same order, results
  identical), the per-image arrays are sized exactly before filling, the per-entry records are freed once the atom-block
  plan exists (only the generic path without atom blocks reads them), a cached plan that stores more arrays serves a
  request that needs fewer, and cached plans that store fewer arrays are dropped before a richer one is built. Measured
  at 64 Si atoms, r2SCAN with forces (`JSIESTA_GRID_PLAN_VERBOSE=1`): plans of 0.46, 1.61 and 3.92 GB kept, 3.48 GB of
  per-entry records during each build; the cache ends at 3.92 GB in one plan, against about 16 GB in three plans
  before, and the build peak is about 7.4 GB.
- **A100 job 29608548 (r2SCAN, 512 Si atoms, forces):** 22 CUDA tests pass. Host plans of 3.67, 12.90 and 31.36 GB kept
  (27.86 GB of per-entry records during each build), the cache holding one plan at a time; peak resident memory of the
  job 36.5 GB (97 GB and killed in job 29607988). The value and gradient plans are resident on the device; the plan with
  Hessians (31.5 GB) streams in 4 chunks of up to 8.2 GB. Forces + stress 20.1 s (grid backward 7.4 s, grid forward
  4.4 s, fresh grid 4.1 s, two-centre 2.7 s, KB 1.6 s); SCF iteration 5.6 s; first SCF call with compilation 165 s.
- **Known limits:** each streamed call copies its part of the plan over PCIe (pageable host memory, one buffer set, no
  overlap of copy and kernel), about 5 GB/s on the A100 node. The host plan must fit in host memory; the per-entry
  records are 96 bytes whatever the plan stores, so even a value plan needs them briefly. Value offsets are `int32`, so a plan is limited to 2^31 orbital values (about
  2800 Si atoms with this basis at 100 Ry).

## D-044: Page-locked host plan and copy/kernel overlap for streamed grid plans (2026-11-23)
- **Status:** accepted and A100-validated (job 29610921: 24 of 24 CUDA tests).
- **Why:** a streamed plan (D-043) copied each chunk from pageable host memory with the kernels waiting, about 5 GB/s on
  the A100 node: forcing the 512-atom PBE plan to stream raised the SCF iteration from 1.79 s to 3.37 s and the forces
  from 6.5 s to 9.5 s (job 29607988), and the r2SCAN 512-atom backward streams 31.5 GB per call.
- **Change:** when a streamed plan is created, its host arrays (images and the value arrays it serves) are page-locked
  in place with `cudaHostRegister` and released when the plan is dropped, so the chunk copies are asynchronous DMA with
  no staging copy. The chunks go through two device buffer sets: the copies run on one non-blocking copy stream per
  process, the kernels on the call stream, and events order them (chunk i waits for its copy; the copy into a buffer set
  waits for the kernels of the chunk that used it two steps earlier), so the copy of chunk i + 1 overlaps the kernels of
  chunk i. The chunk budget (half the free device memory) is split over the two sets. A range another plan already
  registered counts as pinned. `JSIESTA_CUDA_PLAN_PIN=0` and `JSIESTA_CUDA_PLAN_OVERLAP=0` switch the two parts off;
  if page-locking fails the copies stay pageable (still correct). Sub-buffers are 256-byte aligned.
- **Physics:** none; same kernels and chunks as D-043. GPU gate: the two grid parity tests now run resident, streamed
  (pinned, overlapped) and streamed without either, each against the CPU.
- **Effect (A100 job 29610921, 512 Si atoms, same job, plan forced to stream):** PBE without pinning or overlap: SCF
  iteration 2.50 s, forces + stress 8.25 s (grid backward 2.23 s); with both: 2.39 s and 7.01 s (grid backward 1.43 s,
  fresh grid and grid forward 0.84 -> 0.61 s and 0.81 -> 0.56 s). r2SCAN (Hessian plan streamed in 8 chunks of up to
  4.1 GB): forces + stress 14.8 s, grid backward 3.9 s, against 20.1 s and 7.4 s in job 29608548 (other node). The
  resident PBE iteration is 1.79 s, so streaming the 3.8 GB value plan twice per iteration still costs about 0.6 s: it
  is one chunk, so nothing overlaps, and the copy itself is the limit. Peak host memory 37.6 GB with page-locking.
  The pageable run here (2.50 s per iteration) is faster than in job 29607988 (3.37 s) on another node.
- **Known limits:** page-locking costs time once per plan (once per geometry), and the pinned memory is not swappable.
  Streaming stays copy-bound: the kernels at 512 atoms are shorter than the copies they overlap.

## D-045: Consolidated release repository (2026-11-23)
- **Status:** accepted.
- **Why:** the development repository grew study scripts, cluster job scripts, presentations and per-machine paths
  next to the engine. A pushable repository should hold the engine, its tests, its documentation and the decision
  record, and nothing tied to one person's machines.
- **Change:** `main` of the development repository was fast-forwarded to `perf/gpu-forces` (D-031 to D-044) and
  exported into a new repository with one initial commit. Removed from the release: `scripts/` (scaling, profiling and
  Snellius job scripts, the O-2 strain studies of D-021, the basis comparison of D-040), `presentations/`,
  `pilots/reclat*` and `examples/`; they remain in the development repository at commit `ea4b882`, where the log
  entries that name them can be followed. Moved to `docs/design/`: `IMPLEMENTATION_PLAN.md`, `PLAN.md`,
  `IMPLEMENTATION.md`, `EVIDENCE.md`, `known_bottlenecks.md`, `bottleneck_assessment_plan.md` (references updated;
  excluded from the Sphinx build). `tests/external_refs.toml` now lists every key with an empty default; per-machine
  paths go in `tests/external_refs.local.toml` or `JSIESTA_<KEY>`. Local file links and user paths were removed from
  the design documents and the work log. `matplotlib` and its dependencies were dropped from `requirements-dev.txt`
  (only the removed plotting script used them). README rewritten for the engine.
- **Portability fixes found by testing the fresh tree:** the knowledge-graph test required the SIESTA documentation
  index (`indexes/default/current/docs.db`), a git-ignored local build, so a fresh clone failed it. The index is now
  the external reference `siesta_doc_index` (`JSIESTA_SIESTA_DOC_INDEX`, also read by `kg/build_kg.py`, which takes
  `--legacy-db`); the legacy-provenance assertions moved to `test_knowledge_graph_links_legacy_siesta_routines`,
  skipped without the index. `check_staleness` counted test results recorded at a commit as current when the tree has
  no HEAD (no git, or no commit yet); it now reports them as stale.
- **CI:** `.github/workflows/ci.yml` runs ruff, mypy and the documentation build (Doxygen 1.18.0, pinned because the
  Doxyfile treats warnings as errors; Sphinx `-W`) on pushes and pull requests, and publishes the HTML on GitHub Pages
  from `main`. The physics tests stay local (`make check`): they need the compiled library, Libxc and the SIESTA
  references. Checked locally: the documentation builds from a clean tree without the compiled library.
- **Physics:** none; the engine and kernels are byte-identical to `ea4b882` except for docstring paths to the moved
  design documents.

## D-046: Compile once per array shape, not once per geometry (2026-11-23)
- **Status:** accepted (branch `perf/fixed-costs`); measured on the A100 (job 29629429: the 26 CUDA and compilation
  tests pass).
- **Why:** `run_scf` built its compiled functions as closures over the geometry (positions, neutral-atom density,
  packed basis, overlap), so every new geometry re-traced and recompiled them even with unchanged shapes: 2.2-2.6 s
  for a 2-atom cell on the laptop, about 2 s at 64 atoms and 9 s at 512 on the A100 (from the scaling jobs). A
  persistent cache could not hit either. The first call of a process also made about 400 small compilations of
  one-off array operations run outside any compiled function (initial density matrix, seed potential, the whole force
  evaluation): 13 of the 16 s of compilation for that cell. (The geometry-step rows of the decks assumed the
  compilation was paid once; it was not.)
- **Change:** the SCF's compiled functions (overlap transform, solve, grid Hamiltonian and energies, the seed
  Hamiltonian, the per-iteration bookkeeping) and the force evaluation's grid stages (fresh grid pass, grid-energy
  forward pass with its pullback, backward pass) are module-level `jax.jit` functions that take every geometry-dependent
  array as an argument; only shapes, spin mode, functional, dispersion choice and the symmetry engine (static by
  identity) are static. The eigensolver accepts the Bloch gather tables and the real-arithmetic flag precomputed once
  from the concrete pattern (`bloch_assembly_plan`), so the pattern can be a traced argument without losing the gather
  assembly or the real path. The custom derivatives of the grid kernels carry their packed buffers as residuals instead
  of closures, so the compiled forward pass can return its pullback. The initial density matrix goes through the
  compiled solve; the on-site Zeeman seed is vectorised NumPy. `jax_siesta.core.compilation.enable_compilation_cache`
  (opt-in, `JSIESTA_COMPILATION_CACHE`) points JAX's persistent cache at a directory with zero thresholds, so the many
  short compilations are cached too.
- **Effect (laptop, Si 2 atoms, SCF + forces):** first call 397 compilations, 16.6 s of compile time -> 43, 3.6 s;
  with a warm persistent cache 0.5 s. A new geometry with the same shapes: 4 compilations and 3.7 s -> none and 0.9 s.
  64 atoms (6 SCF iterations + forces, together with D-047): 84.3 s -> 38.9 s for the first geometry, 65.9 s -> 37.8 s
  for the second.
- **Effect (A100, job 29629429, with D-047; 6 SCF iterations + forces):** Si 64 PBE 39.5 s -> 17.3 s for the first
  geometry (9.7 s with a warm persistent cache), 19.7 s -> 6.7 s for a displaced second geometry; Si 64 r2SCAN
  50.3 s -> 15.7 s and 32.2 s -> 10.3 s; Si 512 PBE 182 s -> 98 s (69 s warm), 139 s -> 65 s. With a warm cache the
  first and second geometries cost the same, so no compilation is left per geometry. At 512 atoms the remaining
  second-geometry cost (pattern 16 s, SCF 23 s, forces 19 s) is set-up and work, not compilation; it is not broken
  down yet.
- **Physics:** none. Energies equal to 6e-14 eV (the initial solve is now compiled instead of eager), forces to 1e-17
  Ha/Bohr; SCF, forces, symmetry, mixer, Harris, meta-GGA/SOC and stress-component tests unchanged (57 passed).
  New `tests/test_compilation.py`: a second geometry with the same shapes compiles nothing; the cache is opt-in.

## D-047: Threaded host set-up and threaded host Libxc (2026-11-23)
- **Status:** accepted (branch `perf/fixed-costs`); measured on the A100 (job 29629429).
- **Why:** after D-046 the fixed cost of a geometry is host work, all single-threaded: the grid plan build, the
  overlap/kinetic and KB-overlap integrals (one loop over orbital pairs that recomputed the solid harmonics of the same
  atom-pair vector for every orbital pair, and spline tables built one shell pair at a time), the sparse pattern
  (NumPy sorts over all edges and their transposes), and for meta-GGA the host Libxc call (one thread over all grid
  points).
- **Change:** `cpp/include/jsiesta/threading.hpp` (thread count from `OMP_NUM_THREADS`, `parallel_ranges`,
  `run_threads`) replaces the private helpers of the grid and KB kernels. Grid plan: the per-entry sweep runs on atom
  ranges with per-thread counters whose prefix sums keep the serial order; image grouping runs on point ranges with
  exact per-point offsets; a plan beyond the `int32` offsets keeps the generic representation instead of overflowing.
  Two-centre S/T and KB overlaps: spline tables built in parallel over shell pairs, orbital pairs (links) in parallel
  ranges, solid harmonics recomputed only when the vector changes. Pattern: shift indices by a lookup table instead of
  `searchsorted`, transposed shift codes by arithmetic, used shifts by `bincount` (no copy of all transposed shifts).
  Libxc (LDA/GGA and meta-GGA host paths, used by the CPU build and by the GPU's meta-GGA route): grid-point ranges per
  thread, each with its own Libxc handles.
- **Effect (laptop, 8 threads, 64 Si atoms):** two-centre 6.35 s -> 2.52 s; first value-plan build and `rhoofd`
  5.60 s -> 3.82 s; pattern 1.89 s -> 1.30 s (512 atoms: 17.9 s -> 13.3 s); r2SCAN Libxc evaluation on 72^3 points
  0.197 s -> 0.076 s (spin-unpolarised), 0.366 s -> 0.147 s (polarised).
- **Effect (A100 node, job 29629429, first geometry):** two-centre 6.23 s -> 1.96 s (64 atoms), 24.7 s -> 7.95 s (512
  atoms); pattern 1.68 s -> 1.42 s and 18.9 s -> 16.5 s. Energies identical to the previous build to 1e-12 Ha.
- **Physics:** none. Overlap, kinetic and their derivatives, V_KB, the neutral-atom pair energy, the grid density and
  the plan gradients are bitwise identical to the previous build at 64 atoms; the sparse patterns are identical at 8, 64
  and 512 atoms; the r2SCAN energy density and potentials are bitwise identical.
- **Known limits:** the pattern builder is still NumPy (13 s at 512 atoms on the laptop, 16 s on the A100 node: sorting and the transpose
  search over 12.7 M edges); a C++ builder would remove most of it.


## D-048: One host grid plan per geometry, built without a serial zero fill (2026-11-23)
- **Status:** accepted (branch `perf/fixed-costs`); measured on the A100 (jobs 29634256, 29635520, 29635599).
- **Why:** a stage breakdown of a second Si 512 geometry (job 29634256, 64.8 s for pattern, two-centre, 6 SCF
  iterations and forces) found 21.3 s in the host grid plans: the SCF built the plan of orbital values (7.9 s), the
  forces then built the plan with gradients of the same geometry (10.0 s), plus 3.4 s of uploads. Timing the phases of
  a build (job 29635520) put 4.8 s of either build in one step: `std::vector::resize` zero-filling the per-entry array
  (27.9 GB at 512 atoms) on one thread, just before the threaded fill sweep overwrites every element. The image arrays
  were zero-filled serially in the same way.
- **Change:** the per-entry array and the image arrays use an allocator that leaves trivially constructible elements
  uninitialised (`PlanVec` in `cpp/src/grid_ffi.cpp`); each thread of the image fill zeroes its own value range before
  writing it. The host plan cache records the richest variant (gradients, Hessians) requested per system and grid (atom,
  orbital, shell and radial-point counts, mesh shape); a plainer request for a geometry not yet cached builds that
  variant, so in a loop over geometries the SCF builds the plan the forces will need and the forces find it in the
  cache (a cached plan that stores more arrays already matched a plainer request). The first geometry of a process
  still builds both. The verbose plan logs (`JSIESTA_GRID_PLAN_VERBOSE`, `JSIESTA_CUDA_PLAN_VERBOSE`) report build and
  upload times.
- **Effect (A100 node, 18 threads):** value-plan build at 512 atoms 8.29 s -> 3.29 s, gradient plan 10.31 s -> 3.89 s;
  at 64 atoms 1.00 s -> 0.39 s and 1.26 s -> 0.48 s. Second geometry (6 SCF iterations + forces): Si 512 PBE 64.8 s ->
  52.6 s (SCF 23.0 s -> 19.1 s, forces 19.1 s -> 9.2 s); Si 64 PBE 6.7 s -> 5.1 s; Si 64 r2SCAN builds one plan with
  Hessians instead of two plans (7.1 s).
- **Physics:** none. Densities and plan gradients are bitwise identical to the previous build; energies identical to
  1e-12 Ha; 26 CUDA and compilation tests (A100) and 55 grid, SCF, force, stress and meta-GGA tests pass.
- **Known limits:** in a process that alternates systems of the same shapes with and without forces, the plainer runs
  build the richer plan (more host memory, about 20 % more build time). At 512 atoms the sparse pattern (17 s) is now
  the largest per-geometry item, then the host two-centre set-up and force terms (11 s) and the 13 GB gradient-plan
  upload (2.7 s).

## D-049: Basis optimisation: reuse the symmetry set-up and the radial tables, faster projection, k-shift fix (2026-11-24)
- **Status:** accepted (branch `perf/basis-opt`); measured on the laptop and on the A100 (jobs 29639602, 29639684).
- **Why:** a profile of `BasisEnthalpyObjective` (Si, 2 atoms, 8 parameters, Nelder-Mead) gave 6.0 s per evaluation, of
  which 3.0 s rebuilt the symmetry set-up of an unchanged structure (space group, closed pattern, projection table in
  Python), 1.0 s went to the irreducible two-centre path (Python orbit bookkeeping), 0.5 s to the neutral-atom pair
  table (NumPy, 16 M `sin` per species pair), 0.2 s to symmetry projections run as a scatter (`segment_sum`, 15 ms each
  on the CPU), and 8 recompilations per evaluation, because each new engine object changed the static key of the
  compiled SCF. With several training structures (SiC, DZP, 3 lattice constants) the radial spline tables were rebuilt
  for each structure although they depend only on the radial functions. Without symmetry the objective (and
  `run_case`) dropped `k_shifts` and sampled an unshifted mesh: 2.1 eV off the symmetric result at 2x2x2 for Si.
- **Change:** `NativeSymmetrySetupCache` (`symmetry/engine.py`) keeps the structure part (space group, IBZ k-mesh,
  commensurate mesh, permutations) and the last 8 engines keyed by orbital layout and distance-cutoff pattern
  (`same_edges`); a reused engine is the same object, so nothing recompiles. The projection table is stored as padded
  rows (ELL) and applied as a gather and a row sum in a compiled function (20x faster than the scatter). The
  neutral-atom density table is the threaded C++ `jsiesta_na_density_table` (same trapezoid rules; 2e-14 Ha from the
  NumPy version), its Python cache capped at 64 entries. The C++ S/T and KB-overlap entry points keep the radial tables
  of the last 4 sets of radial functions and build pair tables on first use; `R^L` and the k weights are hoisted out of
  the inner loop of `build_pair_splines` (bitwise identical). The objective evaluates the two-centre integrals of the
  symmetric path with the full threaded evaluator on the closed pattern (faster than the irreducible path: 0.68 s vs
  1.13 s for Si DZP, 0.95 s vs 5.6 s for 16 atoms; equal to 1e-14). `k_shifts` reach the full-zone mesh in the
  objective and in `run_case`.
- **Effect:** Si 2 atoms (symmetric, laptop): 6.0 s -> 0.51 s per evaluation (0.36 s on the A100). SiC DZP, 3 lattice
  constants, 100 Ry, 4x4x4 shifted k, per evaluation on one A100 node share (18 cores): jsiesta 1.86 s on the GPU
  (symmetric; 1.94 s full zone), 5.0 s on the 18 CPU threads; SIESTA with the same `PAO.Basis` (3 runs) 23.8 s on 1 rank,
  26.1 s on 18 ranks. On the laptop the same evaluation went from 6.8 s to 4.5 s. The 64-atom two-centre set-up went
  from 2.27 s to 1.37 s.
- **Physics:** none beyond the k-shift fix. Enthalpies of the symmetric objective identical to 6 decimals; S, T, their
  derivatives, V_KB, E_NA and the densities of the 64-atom check bitwise identical; 66 symmetry, two-centre, SCF, force
  and basis tests and `make check` pass (158 passed; one earlier `make check` run crashed once in `numpy.linalg.solve`
  inside the rVV10 test and did not reproduce in two reruns).
- **Tried and dropped:** starting each evaluation's SCF from the previous converged density matrix: 8.0 -> 7.4
  iterations at `d_tol` 1e-5, none at 1e-6 (the neutral-atom start is already as good).
- **Known limits:** with symmetry, a new neighbour set still costs the Python set-up (about 15 s for SiC DZP, 54756
  edges) and a recompilation; for basis optimisation on the GPU the full-zone path is as fast and its first evaluation
  is 7 s instead of 33 s. The host two-centre integrals are now 58 % of a GPU evaluation (1.08 s of 1.86 s).
  `format_pao_basis_block` writes a perturbative polarisation shell as an explicit shell (`n=2 2` for C), which SIESTA
  rejects; SIESTA's form is the `P` flag on the parent shell, which also ties the polarisation cut-off to the parent's
  (the default optimisation space varies it independently). PAO generation from a PSML without semilocal potentials
  solves each radial state by a dense `eigh` (2.1 s per state on the 2398-point grid of the test fixture).

## D-050: Basis optimisation, second round: radial tables, vectorised symmetry set-up, SIESTA polarisation convention (2026-11-24)
- **Status:** accepted (branch `perf/basis-opt`); A100 timing in job 29646494.
- **Why:** after D-049 the radial spline tables of each new basis were most of the two-centre cost (one table per
  shell pair and variant, 615 k spherical Bessel values each, S and T built separately, the Hankel transforms and the
  neutral-atom pair splines serial); a new neighbour set with symmetry cost 6.5 s of Python loops (closed pattern and
  projection table, one iteration per block and operation; 306 s for 16 SiC atoms). Exporting an optimised DZP basis
  to SIESTA failed: `format_pao_basis_block` wrote the perturbative polarisation shell as an explicit `n=2 l=2` shell
  (SIESTA: "PAO 2d with n<n_valence should not be in PAO.Basis"), and the default optimisation space varied the
  polarisation cut-off and confinement, which SIESTA ties to the parent shell (`atom.F`, `POLgen`). Running SIESTA on
  a corrected block showed that the polarisation orbitals themselves differed (overlap 0.95-0.98 with SIESTA's; all
  other orbitals above 0.9998).
- **Change:** radial tables: pairs with the same `r_c1 + r_c2` and `l1 + l2` (so the same R grid and Bessel values),
  including S and T of a pair, are built in one task that computes each Bessel value once; tasks are handed out
  dynamically; the Hankel transforms and the neutral-atom pair splines are threaded (all bitwise identical).
  Symmetry set-up: `build_symmetry_closed_pattern` and `compile_projection_table` work on arrays (all blocks of one
  `(l1, l2)` class per operation, shifts and blocks looked up by integer codes); the closed pattern is identical, the
  projection weights agree to 1e-16. Polarisation, SIESTA convention: a perturbative polarisation shell has no
  optimisation parameters; the generator gives it the parent's first-zeta cut-off and the parent's soft confinement
  (an explicit different value raises `ValueError`); its source term is SIESTA's, `r u_parent / (dr/di)^(3/2)` on the
  logarithmic grid (SIESTA's Numerov recursion takes `-r u_parent` as the source of `g = u / sqrt(dr/di)`); the
  export writes it as the `P n_zeta` flag of the parent shell.
- **Effect:** SiC DZP, 3 lattice constants: two-centre per evaluation 1.28 s -> 0.64 s on the laptop; symmetry set-up
  6.55 s -> 0.59 s (16 atoms: closed pattern 54 s -> 10.5 s, projection table 306 s -> 46 s). On the A100 a steady
  evaluation 1.86 s -> 1.32 s (SIESTA 23.8 s for the same 3 runs), the first one 32.9 s -> 15.5 s. The DZP space of
  SiC has 18 parameters instead of 22. Polarisation orbitals now overlap SIESTA's by 1.00000 with soft confinement
  and 0.9995 without (grid snapping of r_c), C and Si.
- **Physics:** the perturbative polarisation orbitals change (to SIESTA's); everything else unchanged (densities,
  S, T, V_KB, E_NA of the 64-atom check bitwise identical; symmetric objective enthalpies identical to 6 decimals
  before the polarisation change). New test `test_exported_polarised_basis_matches_siesta` (external: SIESTA and
  PseudoDojo) runs SIESTA on the exported block and compares every orbital's cut-off and shape.
- **Known limits:** larger symmetric cells still spend tens of seconds in the projection table (aggregation of many
  entries); the textbook-source polarisation is no longer available.

## D-051: SIESTA restart files: `.XV`, `.DM` and `.nc` in and out (2026-11-23)
- **Status:** accepted; branch `feature/siesta-restart-io`. The exchange with a running SIESTA is not yet measured
  (no SIESTA binary on the development machine); the test that does it is in place and skipped there.
- **Why:** every run started cold, and nothing jsiesta computed could be handed to SIESTA or to tools that read
  SIESTA files. Geometries, density matrices and Hamiltonians now move in both directions.
- **Change:** new `jax_siesta/io/siesta_files.py`. `run_scf` takes `initial_dm` and `initial_h`;
  `run_single_point` and its CLI take `--read-xv`, `--read-dm`, `--read-nc`, `--write-restart`, `--label`;
  `parse_fdf` returns the restart keywords (`SystemLabel`, `MD.UseSaveXV`, `DM.UseSaveDM`, `UseSaveData`,
  `File.DM.Init`, `CDF.Save`, `SaveHS`) under `"io"`. `netCDF4` is an optional dependency (`[netcdf]`), imported
  only when a `.nc` file is touched.
- **Sparsity layout:** a jsiesta edge `(src, dst, T)` is SIESTA's element `(io = src, column = isc * no_u + dst)`
  with `isc = i1 + nsc1 (i2 + nsc2 i3)` and offset `i` or `i - nsc` above `nsc / 2` (`superx`, `linear2pm`). Files are
  written with the smallest odd `nsc` that holds the pattern; SIESTA re-indexes any `nsc` on reading. Reading matches
  elements by `(row, column orbital, offset)`, so another `nsc`, unsorted columns (SIESTA's own files) and a
  different pattern are all handled; unmatched elements are dropped or left zero, as in SIESTA.
- **Files without auxiliary supercell** (Gamma-only SIESTA runs, `nsc = 1`): the stored element is the sum over
  images. The density matrix is replicated to every image (exact at Gamma, where `D(T)` does not depend on `T`;
  SIESTA does the same). The Hamiltonian is put on one image, which keeps `H(Gamma)` only: valid for a Gamma-only
  restart, not for k-point sampling.
- **Spin components** (from `diag3k.F` and its notes): SIESTA's `D^{ss'}(io, jo) = c_{io,s} c*_{jo,s'}` is the jsiesta
  edge value, stored as `(Re uu, Re dd, Re ud, -Im ud, Im uu, Im dd, Re du, Im du)`. SIESTA's
  `H^{ss'}(io, jo) = <jo s|H|io s'>` is the conjugate of the jsiesta edge value with the spins exchanged, so
  `H5, H6 = -Im h^uu, -Im h^dd`, `H7 + iH8 = (h^ud)^*` and `H3 - iH4 = h^ud` of the transposed edge. `H`, `EDM` and `Ef`
  are in Ry in the files.
- **Warm start semantics:** `initial_dm` replaces the cold-start density matrix, is rescaled to the electron count
  (as SIESTA's `normalize_dm`) and symmetry-projected; the first Hamiltonian is `H[D]`, as in a SIESTA restart.
  `initial_h` is the first input Hamiltonian under Hamiltonian mixing; alone, the starting density matrix is the one
  it produces. Under density-matrix mixing `H_in = H[D_in]` always, so `initial_h` only builds `D_in` when no density
  matrix is given. A file with another number of spin components is converted (`convert_spin`).
- **What SIESTA can read:** `.XV` (`MD.UseSaveXV`) and `.DM` (`DM.UseSaveDM`, `File.DM.Init`). SIESTA has no reader
  that restarts from a Hamiltonian, in any format, so the `.nc` file is a restart file for jsiesta and an exchange
  file for tools (sisl, TBtrans-style post-processing), not an input to SIESTA's SCF.
- **Checked (`tests/test_siesta_files.py`):** files written by SIESTA (`Tests/14.FileIO/save_dm.DM`,
  `Tests/15.TDDFT/TDDFT_h2o.{DM,XV}`) are read and written back byte for byte. The spin conventions reproduce a
  literal transcription of `diag3k`'s Hamiltonian assembly and density-matrix extraction for `nspin` 4 and 8 to 1e-10
  (density matrix from jsiesta's own eigensolver). Round trips are exact for `nspin` 1, 2, 4, 8. Bulk Si, 2x2x2
  k-points: a restart from the `.DM` file, from `H`, or from `H` and `DM` of the `.nc` file converges in at most 3
  iterations (cold: more) to the same energy within 1e-7 Ha, under both mixing targets. Outside the suite: sisl 0.20
  reads the `.nc` file and gives `S(k)` to 4e-16, `H(k)` to 2e-12 eV, `Tr[D S] = 8` and the Fermi level.
- **Not checked:** SIESTA reading the jsiesta `.DM`/`.XV`, and jsiesta starting from a SIESTA `.DM`/`.nc` of the same
  system (`test_siesta_reads_jsiesta_files_and_jsiesta_reads_siesta_files`, needs `siesta_bin` and
  `pseudodojo_sr_pbe_dir`; it was written without a SIESTA binary to run it against). The `nspin` 4 and 8 conventions
  rest on the `diag3k` transcription, not on a SIESTA run.
- **Not written:** velocities (zero), the `GRID` group and the radial tables of the `BASIS` group of the `.nc` file,
  `.HSX`, `.TSHS` and `DMHS.nc`.

## D-052: Projection table in row-length buckets, passed to compiled code as an argument; test memory bounded (2026-11-24)

- **Why:** on a 15 GB laptop the test suite was killed by the out-of-memory daemon. FCC Co with spin-orbit coupling
  (1 atom, 48 operations, `test_nspin8_offdiagonal_su2_hamiltonian_and_dm_mixing`) needed more than 8 GB: (i) the
  projection-table set-up of D-050 held every generated `(dst, src, weight)` entry before summing duplicates; (ii) the
  row-padded table of D-049 was 215136 x 528 with 73 % padding (1.3 GB), because with spin-orbit coupling and a cubic
  group the row lengths differ by a factor of six; (iii) the compiled SCF and force functions take the symmetry engine
  as a static argument and read its arrays directly, so each of them baked in a private copy of the table and of the
  grid permutations. Independently, a single-process run of the suite grew by several hundred MiB per SCF test,
  because JAX keeps every compiled function and its constants.
- **Decision:** (i) `_CooAccumulator` sums duplicates while the entries are generated. (ii) `ProjectionTable` stores
  the rows sorted by length in at most 8 buckets of similar length (dynamic programming over the distinct lengths,
  fewest buckets with at most 15 % padding); applying it is one gather and row sum per bucket and a final gather
  `row_pos`, still without a scatter. Uniform tables keep a single bucket. (iii) `NativeSymmetryEngine.jit_arrays`
  and `bound`, with `symmetry_jit_arrays` and `bind_symmetry` in `core/protocols.py`: the five compiled functions
  receive the table and the grid permutations as the argument `sym_arrays`. (iv) `tests/conftest.py` drops JAX's
  caches after a test once the process is above 1.5 GB (`JSIESTA_TEST_RSS_CLEAR_MIB`).
  `test_mgga_ibz_symmetry_vs_full_bz` keeps the Si case only: the agreement across spin modes is the subject of
  `test_ibz_symmetrized_vs_full_bz_unsymmetrized_parity`, and the meta-GGA part (symmetrising tau) does not depend on
  the spin mode.
- **Effect (laptop CPU):** projection, single bucket -> buckets, same result to 2e-16: Si bulk (48 operations)
  1.17 -> 0.87 ms; Co hcp collinear 1.22 -> 1.11 ms; Co hcp spin-orbit 3.04 -> 2.84 ms; FCC Co spin-orbit
  190 -> 54 ms with 113.6 M -> 34.6 M stored entries. FCC Co spin-orbit: resident memory after the set-up
  1.6 -> 0.8 GB, SCF from more than 5 GB (killed) to about 1 GB. The R2SCAN test: 236 s -> about 9 s.
- **Not measured:** the GPU, compilation time and end-to-end SCF time.

## D-053: User interface: `jsiesta` command for FDF inputs, `Calculation`/`Results`, sectioned documentation with tutorials (2026-11-24)

- **Why:** the code could only be driven through `jax_siesta/testing/run_case.py` and JSON case files, by composing
  five stage calls by hand, with a basis taken from SIESTA `.ion` files. The documentation site was one page of
  theory notes, one of logs and one flat list of every module. A SIESTA user had no way to run an input file, and a
  new user no place to start.
- **Decision:**
  (i) *Package `jax_siesta/app`* (no new physics): `Calculation` (inputs; `from_fdf`, `from_structure`,
  `with_structure`, `with_config`, `run`), `Results` (eV / Ang properties, `eigenvalues_ev`, `band_structure`,
  `to_dict`), `BandStructure`, `make_structure`; `report` (SIESTA-style log with prefix `jsiesta:`); `outputs`
  (`.FA`, `.EIG`, `.bands` in SIESTA's record layouts, `.XV`/`.DM`/`.nc` through `io.siesta_files`, `.json`);
  `basis` (pseudopotential lookup incl. `SIESTA_PS_PATH`, `%block PAO.Basis` reader, `build_species`); `cli`
  (`jsiesta [in.fdf] [--pseudo-dir] [--out-dir] [--check]`, also `python -m jax_siesta` and standard input; exit
  status 0 / 1 not converged with `SCF.MustConverge` / 2 input error). `pyproject.toml` installs the `jsiesta`
  script; `make venv` installs the package in editable mode.
  (ii) *`parse_fdf` rewritten on a tokenizer* (`tokenize_fdf`: `%include`, `%block X < file`, `;` comments, first
  definition wins as in SIESTA). New keywords: `LatticeParameters`, coordinate-format aliases, `DM.InitSpin`
  (default: every atom +1 when spin-polarised, as SIESTA starts polarised), legacy spin flags, k-grid
  displacement, `SCF.DM.Tolerance`/`DM.Tolerance`, `SCF.H.Tolerance`, `SCF.FreeE.Tolerance`, `MaxSCFIterations`,
  `NetCharge`, `DFTD3`, `PAO.BasisSize`/`EnergyShift`/`SplitNorm`/`SplitNormH`, `PAO.Basis`, `User.Basis`,
  `BandLines`, `SCF.MustConverge`, and the extensions `JSiesta.UseSymmetry`, `JSiesta.Dispersion`. Labels that are
  neither interpreted nor known print options are returned in `"unused"` and reported. **Behaviour changes of
  `parse_fdf`:** defaults now follow SIESTA where the keyword means the same (`XC` LDA/CA instead of PBE,
  `MeshCutoff` 300 Ry instead of 150, `PAO.EnergyShift` 0.02 Ry); the first definition of a label wins (it was the
  last); a non-diagonal k-grid, an unknown `Spin`, coordinate format or occupation function raise instead of
  being ignored. SCF tolerances and the mixer keep jsiesta's defaults.
  (iii) *Small additions to the stages:* `run_scf(callback=)` (one record per iteration, read-only);
  `observables.band_energies` (eigenvalues at arbitrary k-points, all spin modes); `default_basis_spec` accepts
  `SZ`..`QZ` with `P`/`nP` (was SZ, DZ, DZP) and `generate_species_from_psml` takes `energy_shift_ha` and
  `split_norm` for size names; `jax_siesta.__version__`.
  (iv) *Documentation:* Sphinx site in six sections with hand-written introductions: Getting started, User guide (8
  pages incl. an FDF keyword reference and a page of differences from SIESTA), Tutorials, API reference (11 Python
  sections + C++ grouped by purpose, fragments generated by `docs/conf.py` from `API_SECTIONS`; a package without
  a section fails the build), Theory, Development (status, decisions, work log and diagrams as separate pages).
  Tutorials are "percent" scripts in `docs/tutorials` converted and executed by `make tutorials` (jupytext +
  nbclient) and rendered by myst-nb without re-execution, since the CI has no compiled library. `examples/` holds
  three inputs and the PseudoDojo `nc-sr-04_pbe_standard` PSML files for H, C, O, Si (CC BY 4.0, attribution in
  `examples/pseudos/README.md`); the repository's own PSML fixtures are synthetic and unsuitable for tutorials.
- **Effect (laptop CPU, 8 threads, PseudoDojo PBE, basis generated from the PSML files):**
  `jsiesta` on `examples/`: Si bulk DZP 4x4x4 150 Ry 9.5 s wall (5.1 s inside `run`), H2O 4.4 s, O2 triplet 5.4 s
  (total moment 2.000); peak memory below 0.9 GB.
  Si: symmetry on/off energies equal to 2e-9 eV (8 vs 64 k-points); indirect gap 0.57 eV at 0.85 Gamma-X;
  a0 = 5.461 Ang and B0 = 93.9 GPa from a Birch-Murnaghan fit (32^3 mesh, 6x6x6), zero of the stress pressure at
  5.457 Ang (5.4621 vs 5.4617 with a 48^3 mesh).
  Identities in `tests/test_app_interface.py` (6 pass, 14 s, 0.8 GB): eigenvalues vs a dense SciPy solution of the
  Bloch problem 1e-8 eV; Fermi occupations sum to N; force vs central difference of the energy 1.1e-5 eV/Ang (H2,
  200 Ry); file round trips; restart from `.DM` in fewer iterations; exit codes.
  Basis generation from PseudoDojo files takes under 1 s per species (15 s or more for the synthetic fixtures,
  which lack semilocal potentials and take the dense-eigensolver path).
- **Found on the way:**
  *Hydrogen split norm.* H2 with DZP and the default split norm 0.15 has its minimum at 0.777 Ang (TZ2P 0.771),
  against 0.750 for converged PBE. With `PAO.SplitNormH 0.5` (SIESTA's recommendation) it is 0.748 Ang and the
  energy 0.24 eV lower. The generator route and the loader route with the same orbitals agree (0.7780 vs
  0.7782 Ang, total energy within 4 meV), so this is the basis, not the generator's potentials. The keyword is
  supported; the default is unchanged (SIESTA's is also `PAO.SplitNorm`).
  *Mesh noise with ONCV pseudopotentials.* Si with one atom displaced: the force changes by 0.02 eV/Ang between
  150 and 200 Ry and the pressure by 0.2 GPa between 300 and 400 Ry; egg-box amplitude 3.7 meV (60 Ry) and
  0.9 meV (150 Ry) per cell; H2O at 250 Ry has a net force of 0.08 eV/Ang. For H2 at 80 Ry the force differs from
  the finite difference of the energy by about 1e-3 eV/Ang for any step (1e-5 at 200 Ry).
  *Self-overlap of generated orbitals* is 1 + 1e-4 (Si DZP): the radial functions are normalised on the
  logarithmic grid and then tabulated on a uniform one.
  A shifted Monkhorst-Pack grid on the fcc primitive cell breaks the symmetry (0.017 eV/Ang forces in perfect Si
  without symmetry); the examples use unshifted grids.
- **Not checked:** no SIESTA binary on this machine. `test_same_fdf_input_in_siesta_and_jsiesta` (same `.fdf` and
  PSML in both codes, each generating its DZP basis; tolerances 0.1 eV and 0.03 eV/Ang are estimates) has never
  run. The `.FA`, `.EIG` and `.bands` layouts follow the SIESTA sources (`iofa`, `ioeig`, `bands.F`) and have not
  been compared with SIESTA-written files or read by `gnubands`/sisl. Spin-orbit runs with a natively generated
  basis, meta-GGA runs through the new interface, `User.Basis` with a real `.ion.xml`, `CDF.Save` through the
  command, and the GPU were not run. The automatic polarisation shell follows the generator's rule (highest `l`
  of the valence configuration), which may differ from SIESTA's for elements with d or semicore valence states.
  The O2 binding energy of the solutions notebook (7.67 eV, converged PBE about 6.2 eV) is reported there with
  the two effects known to raise it (superposition error, spherical atom), not separated.

