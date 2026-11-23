# Implementation Plan: Clean-Room JAX + C++ Static SIESTA Engine

## 1. Objective and Scope

This document defines the architecture, mathematical standards, software engineering contracts, and phased implementation roadmap for a clean-room **JAX + C++** implementation of the **SIESTA** Kohn–Sham DFT method, scoped strictly to **static single-point total energies, atomic forces, and stress tensors**.

Restricting scope to the static SCF cycle—omitting legacy Fortran global module state, MPI domain decomposition, MD/relaxation drivers, NEGF (`TranSIESTA`), Berry phase, $O(N)$ solvers, and post-processing (`siesta_analysis`)—reduces the problem to a compact, GPU-capable numerical pipeline that can be validated directly against SIESTA using the `validation.py` harness and `EVIDENCE.md` contracts.

### Core Architectural Decisions

1. **Method Formulation (Path 1: PSML Pseudopotentials + Uniform 3D Real-Space Mesh)**
   - Retain SIESTA’s confined Numerical Atomic Orbitals (NAOs), Kleinman–Bylander (KB) nonlocal pseudopotentials, neutral-atom ($V^{\text{NA}}$) electrostatic screening cancellation, and uniform 3D periodic real-space grid (`dhscf` / `poison`).
   - **PSML-only pseudopotential support:** Exclusively support the curated **PSML (`.psml`)** XML format (semilocal channels, precomputed or on-the-fly Kleinman–Bylander projectors, scalar-relativistic and $j$-dependent spin–orbit projectors, pseudo-core charge for non-linear core corrections, and valence configuration). Drop legacy `.psf` format support completely to avoid maintaining outdated unstandardized parsers and radial-grid conventions.
   - *All-electron note:* Confined NAOs themselves can represent all-electron states, but SIESTA’s uniform 3D FFT grid cannot practically resolve $1s$ nuclear cusps (which scale roughly as $Z^2$ in cutoff and $Z^3$ in grid points). True all-electron NAOs would require switching to atom-centered Becke radial $\times$ Lebedev grids and multipole/Ewald Poisson (an FHI-aims-style solver rather than SIESTA). Staying with norm-conserving `.psml` pseudopotentials (including semicore shells when needed) preserves SIESTA’s numerical structure and exact validation parity.

2. **Drop `libgridxc` Completely; Use `libxc` + JAX FFTs + DFT-D3 / rVV10**
   - **Rationale:** Porting `libgridxc` to C is unnecessary and counterproductive for a GPU-first JAX + C++ architecture. Most of `libgridxc`’s code manages Fortran MPI slab/pencil grid decomposition, finite-difference/FFT grid derivatives, and wrapping `libxc`.
   - **Semilocal XC (LDA, GGA, meta-GGA):** Call **`libxc`** directly (C/CUDA API via a thin C++ JAX FFI wrapper) and compute $\nabla\rho$, GGA potential divergence, and GGA stress contributions directly on the GPU via `jax.numpy.fft` (or periodic central finite differences). This keeps $\rho(\mathbf{r})$ and $V_{\text{eff}}(\mathbf{r})$ resident on the GPU with zero host-device round-trips.
   - **Dispersion / van der Waals:**
     - **DFT-D3 / DFT-D4:** Implement as a standalone, JAX-differentiable pairwise + coordination-number energy/force/stress term.
     - **rVV10:** Implement Sabatini’s grid-based rVV10 directly in `jax.numpy.fft` (~150 lines, GPU-native), using its single local parameter $\omega_0(\rho, |\nabla\rho|)$ rather than 2D $(q_0, q_0')$ tables.
     - *(Optional legacy parity)* If exact comparison against SIESTA’s vdW-DF functionals (`DRSLL`, `KBM`, `BH`) is later required, implement only the standalone Román–Pérez–Soler 2D-spline FFT convolution kernel in JAX/C++ rather than porting `libgridxc`.

3. **Native Crystal & Magnetic Symmetry (`spglib`)**
   - Make space-group and magnetic-group symmetry a first-class invariant of the data structures from initialization, reusing the validated mathematical formulations from `siesta_spglib_full_impl` (`Src/m_symmetry.F90` and `Src/m_dm_symmetrize.F90`) while eliminating the compromises imposed by retrofitting onto legacy Fortran MPI SIESTA.

4. **Built for Maintainability and Modular Expansion from Day One**
   - **Purely functional state & zero hidden globals:** All simulation state is represented as explicit, typed, immutable JAX PyTrees (`NamedTuple` / `flax`/`equinox`-free standard `dataclasses` registered as JAX PyTrees) and explicit C++ opaque context handles passed through FFI—eliminating the Fortran module-level `save` globals that complicate legacy SIESTA testing and GPU migration.
   - **Protocol-based extension seams:** Every major stage (`BasisAndPseudos`, `TwoCenterIntegrals`, `SymmetryProjector`, `RealSpaceGridOps`, `ExchangeCorrelation`, `DispersionModel`, `EigenSolver`, `SCFMixer`, `ObservableEvaluator`) sits behind a strict Python `Protocol` and typed data contract. Future capabilities—such as DFT+$U$, hybrid functionals, Bogoliubov–de Gennes (BdG) Nambu pairing channels, or linear-scaling solvers—plug in as new implementations of existing protocols without rewriting core kernels.

5. **Every Physics Term Tested Directly Against SIESTA**
   - No physics term or numerical kernel is accepted without isolated unit verification **and** head-to-head parity tests against reference SIESTA executables via `validation.py` and `EVIDENCE.md`.
   - Verification spans individual Hamiltonian/grid components ($S, T, V^{\text{NA}}, V^{\text{KB}}, \rho(\mathbf{r}), V_H, E_{\text{xc}}/V_{\text{xc}}$, DFT-D3/rVV10, $D, E$), all spin modes (`nspin=1,2,4,8`), symmetry operators, and independent physical identities (Hermiticity, trace/electron-count conservation, space-group commutation, and finite-difference energy derivatives vs. analytical forces and stresses).

6. **Comprehensive Multi-Layer Documentation & Automatic Documentation Pipeline**
   - Every Python/JAX module and C++/CUDA header/kernel must be written as living scientific documentation covering **three required layers**:
     1. **Physics Theory:** Governing equations, sign/gauge conventions (e.g., real spherical harmonics, SU(2) spinors, axial vectors), and explicit atomic units (Bohr, Ry, Hartree, eV).
     2. **Algorithmic Implementation:** Discretization, quadrature/spline order, sparsity invariants, complexity, GPU memory layout, and custom `jax.custom_vjp` differentiation rules.
     3. **Architectural & Legacy Mapping:** PyTree/buffer schemas, invariants, and cross-references to legacy SIESTA modules/routines (as indexed in `indexes/default/current/docs.db`).
   - **Ready for automatic documentation from Commit 1:** Configure **Sphinx** (`autodoc`, `napoleon`, `mathjax`, `myst-parser`) for Python/JAX and **Doxygen + Breathe/Exhale** for C++/CUDA FFI kernels, enforced in CI so undocumented public symbols or broken LaTeX/API builds fail immediately.

7. **Physics-Informed, Adaptive, and Contract-Verified SCF Procedure (`scf_params`)**
   - Incorporate the physical conditioning descriptors, mixer algorithms (Tikhonov–Pulay, ADIIS, decoupled spin preconditioning, BdG readiness), scalar SVD corner-case fixes, closed-loop adaptive trust-region controller, and explicit `DM_out` post-SCF energy contracts developed in `scf_params`.

8. **Progressive Dual-Audience (Agent + Human) Knowledge Graph & Resumable Memory**
   - Build a revision-bound, progressively updated knowledge graph from Commit 1—extending the SQLite/FTS5 + MCP architecture of `knowledge.py` and `siesta_mcp.py` alongside the structured memory and append-only log discipline of `Plan.md` and `MEMORY.md`—so both AI agents and human developers can inspect, audit, and resume development seamlessly across sessions.

---

## 2. System Architecture: JAX vs. C++ Division of Labor

```
  Inputs (Structure, Species/.psml + .ion reference basis, k-mesh, MeshCutoff)
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Initialization Layer (Python + C++ + spglib + scf_params)            │
  │  • Space/magnetic group & atom/shift maps (spglib)                   │
  │  • Symmetry-commensurate 3D mesh dimensions (N1, N2, N3)             │
  │  • IBZ k-points & weights (spg_get_stabilized_reciprocal_mesh)       │
  │  • Symmetry-closed neighbor list & static sparse CSR/COO pattern     │
  │  • Wigner-D (16-pt Fibonacci) & SU(2) spinor tables                  │
  │  • Precompiled static DM/EDM symmetry projection operator P_sym      │
  │  • Irreducible two-center integral evaluation -> S, T, Vkb, Vna      │
  │  • Physics descriptors (kappa_eps, d/f stiffness) -> initial mixer   │
  └──────────────────────────────────────────────────────────────────────┘
         │
         ▼  Static immutable PyTrees transferred once to device (GPU/CPU)
  ┌──────────────────────────────────────────────────────────────────────┐
  │ SCF Loop (Compiled JAX + C++/CUDA FFI + Adaptive Controller)         │
  │                                                                      │
  │  1. [JAX] Assemble H(k) from sparse H_loc + H_nl + H_grid(R)         │
  │  2. [JAX] Batched generalized eigh across IBZ k-points               │
  │  3. [JAX] Occupations (optional annealing) + build raw D_ibz, E_ibz  │
  │  4. [JAX] Native symmetry projection: D = P_sym(D_ibz), E = ...      │
  │  5. [JAX] Mixer (H or D): Tikhonov-Pulay / decoupled spin + adaptive │
  │           trust-region controller (stationary G, spike backoff)      │
  │  6. [C++ FFI] rhoofd: sparse orbital-to-grid scatter -> rho(r)       │
  │  7. [JAX] Exact grid symmetry: rho(r) = (1/|G|) sum_S rho(pi_S r)    │
  │  8. [JAX] Poisson (3D FFT) + GGA grad + rVV10 (3D FFT)               │
  │  9. [C++ FFI] libxc pointwise evaluation -> eps_xc, V_xc             │
  │ 10. [C++ FFI] vmat: sparse grid-to-orbital gather -> H_grid          │
  │ 11. [JAX] Trajectory diagnostics -> record last_scf_free_e           │
  └──────────────────────────────────────────────────────────────────────┘
         │  Converged DM_out (at exact target ElectronicTemperature)
         ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Explicit Post-SCF DM_out Pass & Observables (Fixed-(D,E) Gradients)  │
  │  • Mandatory fresh evaluation from converged DM_out:                 │
  │    rhoofd(DM_out) -> Poisson + libxc + dispersion -> V_eff(DM_out)   │
  │  • Record final_post_scf_free_e & Harris energy with provenance      │
  │  • Forces & Stress via variational stationarity at fixed (D,E)       │
  │    (explicit geometry/cell derivatives + -Tr[E dS/dR] Pulay)         │
  │  • Grid eggbox/dfscf stress + DFT-D3 / rVV10 dispersion terms        │
  │  • Symmetry verification, final projection & outcome taxonomy        │
  └──────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Legacy SIESTA Routine | Responsibility | JAX + C++ Implementation |
|---|---|---|
| `atom`, `basis_specs` | Radial PAOs, KB projectors, $V^{\text{NA}}(r)$ | Python/C++ startup module (`.psml` pseudopotential + basis/.ion loader), run once per species |
| `matel`, `kinefsm`, `nlefsm`, `naefs` | Two-center integrals ($S, T, V^{\text{KB}}, V^{\text{NA}}$) & derivatives | C++ startup/geometry builder using radial splines + spherical harmonics; computed on irreducible atom-pair orbits only |
| `siesta_init`, `kpoint_scf` | Symmetry detection, IBZ $k$-mesh | Python/C++ wrapper around `spglib` |
| `diagon`, `compute_dm` | Generalized $H(k)C = S(k)C\varepsilon$, smearing, $D_{\mu\nu}, E_{\mu\nu}$ | Pure JAX (`jax.vmap` over IBZ $k$-points, Cholesky/Löwdin + `jnp.linalg.eigh`) |
| `m_dm_symmetrize` | Symmetrizing `Dscf` and `Escf` | Precompiled static JAX sparse gather-scatter operator (`segment_sum` / `BCOO`) |
| `mixer`, `m_mixing`, `scf_adaptive.lua` | SCF fixed-point acceleration & adaptive control | Pure JAX Tikhonov–Pulay / ADIIS / decoupled spin mixers + native adaptive trust-region state machine |
| `rhoofd`, `vmat` | Sparse orbital-grid density scatter & potential gather | **C++ / CUDA JAX FFI custom calls** looping only over grid points inside each orbital's $r_c$ sphere ($O(N_{\text{grid}})$ memory & compute) |
| `poison` | Hartree potential & energy of $\delta\rho = \rho - \rho^{\text{NA}}$ | Pure JAX 3D FFT (`jnp.fft.fftn` / `rfftn`) |
| `dhscf` (`cellxc`) + `libgridxc` | XC energy, potential, GGA stress, vdW | **Dropped `libgridxc`:** JAX FFT gradients + **C++/CUDA `libxc` FFI** + JAX `rVV10` / `DFT-D3` |
| `compute_energies`, `dfscf`, `post_scf_work` | Total energy, post-SCF `DM_out` pass, grid/Pulay forces & stress | Pure JAX fixed-$(D,E)$ variational autodiff + analytical two-center/grid stress |

### Critical Design Note: Variational Differentiation for Forces and Stresses
Do **not** differentiate through the SCF iterations (`jax.grad` unrolled over `diagon`/`mixer`). By the Hellmann–Feynman / variational theorem, at SCF convergence $\partial E_{\text{KS}}/\partial D = 0$ subject to the orthonormality constraint $\mathrm{Tr}[D S] = N_e$:
- Hold the converged $(D, E)$ constant (`jax.lax.stop_gradient`).
- Compute the explicit geometric and strain derivatives of $S, T, V^{\text{KB}}, V^{\text{NA}}$, $\rho^{\text{NA}}(\mathbf{r})$, and the orbital-on-grid evaluations $\phi_\mu(\mathbf{r} - \mathbf{R}_I)$, including the Pulay term $-\mathrm{Tr}[E \, \partial S / \partial \mathbf{R}_I]$ and the analogous lattice strain term for the stress tensor.
- Differentiating the discrete mesh evaluations of $\phi_\mu(\mathbf{r} - \mathbf{R}_I)$ at fixed $D$ (using `jax.custom_vjp` against the C++ `rhoofd`/`vmat` FFI kernels) automatically captures the discrete `dfscf` eggbox contribution consistent with the total energy.

---

## 3. Native Symmetry Architecture (Reusing `siesta_spglib_full_impl` + Upgrades)

### Proven Logic to Reuse from `siesta_spglib_full_impl`
The following mathematical patterns already tested in `Src/m_symmetry.F90` and `Src/m_dm_symmetrize.F90` must be carried over verbatim in logic:

1. **Atom & Cell-Shift Maps:** Forward and inverse mappings `atom_map(a, S)`, `atom_translation(3, a, S)`, `inverse_atom_map`, `inverse_atom_translation` (`Lainv`, `Lbinv`) with Born–von Karman folding modulo `nsc` for supercell columns.
2. **Unique Sequential Orbital Keys:** Re-index each `(atom, l, m)` shell with a sequential radial counter (`iz_counter` in `m_dm_symmetrize.F90`) rather than raw zeta index, preventing key collisions when semicore and valence shells share `zeta = 1`.
3. **Well-Conditioned Real-Harmonic Wigner-$D$ Matrices:** Compute $D^l(S)$ via least-squares fit over **16 Fibonacci-sphere probe directions** (`NPROBES = 16`, avoiding rank deficiency for $l=3$ $f$-orbitals) and assert orthogonality ($\|D^l (D^l)^T - I\|_\infty \le 10^{-10}$) at startup.
4. **Joint `D` and `E` Symmetrization Across All Spin Modes:** Symmetrize both density matrix $D$ and energy-density matrix $E$, supporting scalar (`nspin=1`), collinear (`nspin=2`), non-collinear axial-vector (`nspin=4`, multiplying Cartesian spin rotation by $\det(R_S)$), and spin-orbit SU(2) spinor (`nspin=8`, applying $U(R_S) \in \mathrm{SU}(2)$ after orbital $m$-summation).

### Four Native Upgrades Over the Legacy Fortran Patch

1. **Symmetry-Closed Sparsity & Precompiled Sparse Projection Operator**
   - *Legacy issue:* `dm_sym_apply` performs `MPI_Allgatherv` and runtime binary/linear searches (`find_ind_g`, `find_isc`) every SCF step, and silently drops terms (`cnt_miss_ind`, `cnt_miss_isc`) if the cutoff list misses a symmetry partner.
   - *Native upgrade:* At geometry initialization, close the orbital-pair neighbor graph under all $S \in G$ so missing partners are impossible by construction. Precompute the full symmetry-averaging map as a static sparse COO table `(dst_edge, src_edge, weight)` combining Wigner-$D$ and spin weights. Inside SCF, symmetrizing $D$ and $E$ is a single compiled JAX `segment_sum` on GPU.

2. **Symmetry-Commensurate 3D Real-Space Grid & Exact Grid Symmetrization**
   - *Legacy issue:* `siesta_spglib_full_impl` projects forces and stress *after* `dhscf`, leaving asymmetric eggbox noise in $\rho(\mathbf{r})$, $V_{\text{eff}}(\mathbf{r})$, and $E_{\text{KS}}$ whenever the grid dimensions don't match space-group translations or lattice permutations.
   - *Native upgrade:* Constrain $(N_1, N_2, N_3)$ at mesh initialization to be divisible by the fractional-translation denominators of all space-group operations (e.g., $2, 3, 4, 6$) and equal along symmetry-equivalent lattice vectors. Every $S \in G$ then induces an exact permutation $\pi_S$ of grid indices, allowing $\rho(\mathbf{r})$ to be symmetrized directly on the grid before Poisson/`libxc`/rVV10 (or evaluated only on irreducible grid points).

3. **Irreducible Two-Center Integral Evaluation**
   - Partition the symmetry-closed atom-pair list $(a, b, \mathbf{R})$ into space-group orbits. Evaluate radial/spherical-harmonic two-center integrals (`matel` for $S, T, V^{\text{KB}}, V^{\text{NA}}$ and their $\mathbf{R}_I$/strain derivatives) **only once per orbit representative**, then generate the remaining pairs via Wigner-$D$ rotation.

4. **Full Magnetic Space Groups (Shubnikov Type I–IV) via `spglib`**
   - Replace the spatial-only filter (`restrict_to_declared_magnetic_sector`) with `spglib`’s native magnetic space-group API (`spg_get_magnetic_dataset`), capturing anti-unitary time-reversal-combined operations ($\mathcal{T} \cdot S$) for antiferromagnets and properly setting `use_time_reversal` in `spg_get_stabilized_reciprocal_mesh` for non-collinear/SOC systems.

---

## 4. SCF Procedure & Mixing Architecture (Incorporating `scf_params`)

The SCF engine directly embeds the physical insight, mixer stabilization fixes, adaptive controller, and energy-measurement contracts established in `scf_params`.

### 4.1 Physics-Informed Initialization & Conditioning Descriptors
Reuse the physical descriptor model from `features.py` and `recommender.py` to compute safe cold-start SCF defaults automatically whenever explicit user mixer overrides are absent:
- **Conditioning descriptors:** Estimate the Thomas–Fermi long-wavelength Coulomb condition number $\kappa_\varepsilon \sim 1 + (k_{\text{TF}} L_{\text{eff}} / 2\pi)^2$, open $d/f$-shell stiffness ($f_d + 1.5 f_f$), cell dimensionality / vacuum fraction (surface dipole sloshing in 2D slabs), electronegativity variance, and spin/Nambu mode.
- **Parameter priors:**
  - Default to **Hamiltonian mixing** (`mix_target="Hamiltonian"`) for LCAO cold starts because $V_H + V_{\text{xc}}$ is pre-smoothed on the real-space grid, while retaining density-matrix mixing (`"DM"`) as an explicit option.
  - Scale base damping weight $w_{\text{base}}$ inversely with $\ln \kappa_\varepsilon$ and $d/f$-shell fraction ($w \in [0.020, 0.250]$), and set Krylov history depth $N_{\text{hist}} \in [4, 12]$ to span coupled charge-sloshing, $d/f$-multipole, and spin eigenmodes (avoiding legacy `history=2` 1D secant stalls).
  - Support optional refinement from a calibrated surrogate (`SurrogateSCFModel`) only within validated held-out coverage; flag uncalibrated heuristic defaults explicitly in run metadata.

### 4.2 Mixer Variants, Decoupled Spin Preconditioning, & Scalar SVD Contract
1. **Scale-Invariant Multiplicative Tikhonov–Pulay (Default):**
   - Implement `variant="tikhonov"` ($\lambda \sim 1.0\times 10^{-3}$ for bulk, $1.5\times 10^{-3}$ for slabs/SOC/BdG) to regularize near-collinear Krylov directions in the Pulay Gram matrix without altering the stationary preconditioner metric $G = wI$ or clipping valid multi-step secant extrapolations.
   - Also implement standard/GR Pulay, Broyden, and ADIIS-to-Pulay hybrids under the `SCFMixer` protocol.
2. **Synchronized Block-Preconditioned Charge/Spin Mixing (`mix_spin="decoupled"`):**
   - Total charge responds with $1/q^2$ Coulomb amplification ($w_{\text{charge}}$), whereas magnetization/spinor channels ($m_z, m_x, m_y$, SOC spinor) are governed by short-range Stoner exchange.
   - Apply separate preconditioner weights ($w_{\text{spin}} \approx 1.6\,w_{\text{charge}}$ for collinear polarized, $1.4\,w_{\text{charge}}$ for non-collinear/SOC) while keeping the Pulay secant coefficients $\alpha_j$ phase-locked across charge and spin channels.
3. **Deterministic Scalar ($N=1$) SVD / Pseudoinverse Contract:**
   - Avoid the uninitialized `info`/`rank` defect diagnosed in `MIXER_FIX_AND_CANDIDATE_CORE.md` and `0001-mixing-initialize-scalar-svd.patch`: for $1\times 1$ Pulay Gram matrices, explicitly enforce `info=0, rank=0, B=0` when $|A_{11}| \le \epsilon_{\text{svd}}$ and `info=0, rank=1, B=1/A_{11}` otherwise, with deterministic zeroing below the singular threshold for all $N \ge 1$.
4. **Extension Hook for BdG (Spin-Nambu) Anomalous Mixing:**
   - Keep `SCFMixer` extensible to coupled normal + anomalous pairing channels (`SCF.BdG.Mixer.*`, `bdg_mixer_reset=False` to preserve normal-state Krylov history during self-consistent $\Delta$ updates) and enforce a mandatory post-seed update guard so iteration-1 seeds cannot pass convergence checks prematurely.

### 4.3 Native Closed-Loop Adaptive SCF Controller
Compile the three cooperative feedback loops from `scf_adaptive.lua` directly into the JAX SCF loop (eliminating external Lua/Flook coupling):
1. **Stationary Metric Preservation:** Keep $w_{\text{cur}} = w_{\text{base}}$ fixed during initial Krylov subspace construction (`iscf <= 4`) and during normal monotone contraction so all secant pairs $(\Delta x_j + G\,\Delta F_j)$ share a consistent preconditioner metric $G$.
2. **Trust-Region Spike Backoff & Selective History Quarantine:** Upon severe residual spikes (`ratio_max > 3.0` with `dH > 0.25` eV, or `ratio_D > 2.2` and `ratio_H > 2.2` with `dH > 0.15` eV), apply multiplicative backoff $w_{\text{cur}} \leftarrow \max(w_{\min}, 0.65\,w_{\text{cur}})$. Flush oldest Krylov history vectors only if severe divergence persists for 3 consecutive steps (`streak_bad >= 3`), and relax $w_{\text{cur}}$ smoothly back toward $w_{\text{base}}$ once contraction resumes.
3. **Optional Fermi–Surface Smearing Annealing:** When enabled, temporarily boost `ElectronicTemperature` during early cold-start transients (`dDmax > 0.03` and `iscf <= 6`) to damp discontinuous $k$-point occupancy jumps, restoring the exact user target temperature as soon as `dDmax` enters the linear basin and strictly prior to declaring convergence.

### 4.4 Explicit Trajectory vs. Post-SCF `DM_out` Energy Contract & Outcome Taxonomy
As established in `MATCHED_SPIN_PILOT.md` and `ENERGY_CONTRACT.md`, the last SCF loop energy (`last_scf_free_e`, computed from mixed/in-flight inputs) can differ by $>1\text{ meV/cell}$ across mixers even when the converged electronic state is identical, whereas recomputing the total energy explicitly from the converged output density matrix `DM_out` (`final_post_scf_free_e`) agrees within $\le 0.107\text{ meV/cell}$ across mixers and spin paths.

Therefore, the JAX + C++ engine must enforce the following contract:
- **Mandatory Post-SCF `DM_out` Pass:** After SCF convergence at the exact target electronic temperature, run one clean non-mixed evaluation from `DM_out` (`rhoofd` $\to$ `poison` + `libxc` + dispersion $\to$ `compute_energies` + `dfscf`/variational gradients).
- **Separate Provenanced Observables:** Record both `last_scf_free_e` (with cycle index) and `final_post_scf_free_e` (marked with `dm_out_recomputed=True` and valid convergence status) as distinct fields in `observables.json` with **no silent fallback** between them.
- **Outcome & Physical-State Taxonomy (per `Plan.md`):**
  - Classify every run explicitly as `converged`, `iteration_cap_exhausted` (treated as right-censored—never assigned artificial iteration penalties), `numerical_breakdown`, `invalid_input`, or `normal_state_collapse`.
  - Always attach physical-state labels (total and atomic spin moments $\mathbf{m}_I$, charge residual, and active symmetry/magnetic subgroup).

---

## 5. Maintainability, Testing-Against-SIESTA, Auto-Documentation, & Progressive Knowledge Graph

### 5.1 Repository & Package Structure for Maintainability
Organize the codebase cleanly into decoupled Python/JAX packages, C++/CUDA FFI sources, automated SIESTA parity suites, auto-generated documentation, and a progressive knowledge-graph/memory layer:

```
jax_siesta/
├── core/          # Immutable PyTrees (Structure, BasisData, SparsePattern, SCFState, Observables)
├── species/       # PSML-only (.psml) reader (libpsml/XML schema) + .ion basis reference, splines
├── symmetry/      # spglib wrapper, Shubnikov magnetic groups, Wigner-D, SU(2), P_sym, grid maps
├── integrals/     # Irreducible two-center integral evaluation (S, T, Vkb, Vna) & derivatives
├── grid/          # JAX 3D FFT Poisson, GGA gradients/divergence, grid symmetry, FFI wrappers
├── xc/            # Direct libxc FFI bindings, DFT-D3/D4, JAX FFT rVV10 (and optional vdW-DF)
├── scf/           # Physics descriptors, Tikhonov-Pulay/decoupled-spin mixers, adaptive controller
├── observables/   # Post-SCF DM_out pass, Harris/KS energies, fixed-(D,E) variational forces/stress
├── io/            # FDF parser subset, JSON/observables.json writers compatible with validation.py
└── testing/       # SIESTA comparison adapters & physical-identity contract checks
cpp/
├── include/       # Doxygen-documented C++ headers & JAX FFI ABI contracts
├── src/           # Neighbor list, radial spline tables, CPU rhoofd/vmat, libxc & psml bridges
└── cuda/          # CUDA kernels for rhoofd/vmat and GPU libxc bridge
kg/
├── build_kg.py    # Deterministic Python AST + Doxygen XML + validation/decision graph builder
├── server_mcp.py  # Read-only MCP tools extending knowledge.py / siesta_mcp.py
└── current/       # Atomically published SQLite FTS5 + JSON snapshot (docs_kg.db)
logs/
├── WORK_LOG.md    # Append-only session history (commands, checks, blockers, next step)
└── DECISIONS.md   # Timestamped architectural & physical decisions with rationale and links
MEMORY.md          # Compact live project handoff state (milestone, verified modules, active blockers)
docs/
├── conf.py        # Sphinx (autodoc, napoleon, mathjax, myst-parser, breathe/exhale)
├── Doxyfile       # C++/CUDA XML generation feeding Breathe
├── theory/        # Mathematical derivations (NAOs, KB, neutral-atom, Wigner-D, variational forces)
├── architecture/  # Auto-rendered Mermaid dependency, validation-status, & lineage graphs
└── api/           # Auto-generated module & kernel reference
```

### 5.2 Physics-Against-SIESTA Verification Matrix
Every physical subsystem must pass both **isolated component parity** against reference SIESTA data (and `siesta_spglib_full_impl`) and **independent mathematical identity checks** via `validation.py`:

| Physics / Numerical Term | SIESTA Reference Target | Independent Mathematical / Physical Check |
|---|---|---|
| **Overlap & Kinetic ($S_{\mu\nu}, T_{\mu\nu}$)** | SIESTA `.HSX` / `matel` + `kinefsm` | Symmetry ($S = S^\dagger$), positive-definiteness of $S(\mathbf{k})$, $\partial S/\partial \mathbf{R}_I$ vs. finite difference |
| **Nonlocal KB & Neutral-Atom ($V^{\text{KB}}, V^{\text{NA}}$)** | SIESTA `nlefsm`, `naefs`, `H0` | Rotational invariance, trace/sum rules, finite-difference derivative consistency |
| **Density & Grid Integrals ($\rho(\mathbf{r}), V_{\text{grid}}$)** | SIESTA `rhoofd`, `vmat` | $\int \rho(\mathbf{r})\,d\mathbf{r} = N_e$, Hermitian $H_{\text{grid}}$, adjoint identity $\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]$ |
| **Hartree Potential & Energy ($V_H, E_H$)** | SIESTA `poison` | $\nabla^2 V_H = -4\pi\,\delta\rho$ residual in reciprocal space, viral/strain scaling under uniform dilation |
| **XC & Dispersion ($E_{\text{xc}}, V_{\text{xc}}, \sigma_{\text{xc}}, E_{\text{disp}}$)** | SIESTA `libgridxc` (`LDA`/`PBE`), `DFT-D3`, `rVV10` | Functional derivative identity $\delta E_{\text{xc}}/\delta \rho$ vs. finite perturbation of $\rho(\mathbf{r})$ |
| **Diagonalization, $D$, $E$, & Spin (`nspin=1,2,4,8`)** | SIESTA `diagon`, `compute_dm` | Orthonormality $C^\dagger S C = I$, $\mathrm{Tr}[D S] = N_e$, $\mathrm{Tr}[E S] = E_{\text{band}}$ |
| **Crystal & Magnetic Symmetry** | `siesta_spglib_full_impl` (`m_symmetry`, `m_dm_symmetrize`, `kpoint_scf`) | $\mathcal{P}_{\text{sym}}^2 = \mathcal{P}_{\text{sym}}$, $[H, U_S] = 0$, IBZ vs. full-BZ invariance |
| **SCF Convergence & `DM_out` Contract** | `scf_params` Co hcp pilot (`collinear`, `NC`, `SOC`) | Mixer-endpoint agreement on `final_post_scf_free_e` ($\le 0.2\text{ meV/cell}$), cold-start repeatability |
| **Total Energy, Forces, & Stress** | SIESTA `final_post_scf_free_e`, `fa`, `stress` | Central finite-difference checks of $F_I = -\partial E/\partial \mathbf{R}_I$ and $\sigma_{\alpha\beta} = \frac{1}{\Omega}\partial E/\partial \epsilon_{\alpha\beta}$, acoustic sum rule $\sum_I \mathbf{F}_I = \mathbf{0}$ |

### 5.3 Multi-Layer Documentation & Automatic Documentation Standards
1. **Docstring Contract (Python/JAX & C++/CUDA):**
   - Every public class, function, protocol, and FFI kernel must include structured **NumPy/Google-style docstrings** (Python) or **Doxygen `/** ... */` blocks** (C++/CUDA) containing explicit `Theory`, `Algorithm`, `Parameters`, `Returns`, `Invariants`, and `SIESTA Mapping` sections.
   - Equations use standard LaTeX (`.. math::` / `$$...$$` / `\f[...\f]`) with explicit unit annotations.
2. **Automated Build & Quality Gates in CI:**
   - Run `doxygen docs/Doxyfile` followed by `sphinx-build -W -b html docs/ docs/_build/html` (warnings treated as errors so missing references, undocumented public symbols, or malformed math blocks fail CI).
   - Auto-generate API pages directly from docstrings (`autosummary`/`autodoc`) and C++ headers (`breathe`), plus include executable `doctest` / validation report summaries linking directly to `validation.py` artifacts.

### 5.4 Progressive Knowledge Graph for Agents and Humans
To prevent context loss across sessions and make the codebase immediately navigable for both human contributors and autonomous agents, the project maintains a **revision-bound, incrementally updated knowledge graph** built on top of the indexing/provenance model in `knowledge.py` and `siesta_mcp.py`:

1. **Graph Schema (SQLite/FTS5 + Exported JSON):**
   - **`CodeEntity`:** Every Python/JAX module, PyTree, `Protocol`, and function (extracted deterministically via Python `ast` rather than fragile regexes) plus every C++/CUDA FFI kernel (extracted via Doxygen XML), pinned to file SHA256 hashes and line ranges.
   - **`TheoryNode`:** Named mathematical derivations, physical conventions, and unit contracts in `docs/theory/`.
   - **`LegacySIESTANode`:** Explicit provenance links (`implements_equivalent_of`, `replaces_routine`) into `indexes/default/current/docs.db`.
   - **`ValidationNode`:** Links (`verified_by`) to `validation.py` manifests, independent physical-identity tests, pass/fail status, measured residuals, and tolerances.
   - **`DecisionAndTaskNode`:** Architectural decisions (`logs/DECISIONS.md`), append-only session logs (`logs/WORK_LOG.md`), and live milestone/blocker status (`MEMORY.md`).
2. **Agent-Facing Interface (MCP + CLI):**
   - Extend `siesta_mcp.py` with tools tailored for zero-ambiguity session resumption:
     - `resume_session_context`: Returns current phase/step, verified vs. unverified modules, active blockers, recent `DECISIONS.md` entries, and the concrete next task.
     - `trace_physics_lineage`: Walks `TheoryNode` $\leftrightarrow$ `CodeEntity` $\leftrightarrow$ `LegacySIESTANode` $\leftrightarrow$ `ValidationNode` for any physical term.
     - `check_staleness`: Compares live source SHA256 hashes against recorded graph/test/doc hashes (extending `KnowledgeBase.health()` in `knowledge.py`).
3. **Human-Facing Interface (Sphinx + Mermaid + Readable Logs):**
   - Automatically emit Mermaid architecture, module-dependency, and validation-coverage diagrams into `docs/architecture/` on every graph rebuild, rendered directly in Sphinx alongside `MEMORY.md`, `logs/DECISIONS.md`, and `logs/WORK_LOG.md`.
4. **Progressive Update & CI Gate:**
   - Every milestone or PR atomically rebuilds `kg/current/docs_kg.db` via a staging directory and atomic symlink swap (preserving the previous valid snapshot on failure, matching `rebuild_index.py`).
   - CI checks that every newly added public physics/algorithmic `CodeEntity` carries valid edges to its documentation theory section and at least one unit or SIESTA-parity `ValidationNode`.

---

## 6. Phased Implementation Roadmap (with 4-Step Symmetry Rollout)

### Phase 0: Project Scaffolding, Auto-Docs, Knowledge Graph, & Reference Freeze
- Initialize package layout (`jax_siesta/`, `cpp/`, `kg/`, `logs/`, `MEMORY.md`, `docs/`, `tests/`) with strict typing (`mypy`/`pyright`), formatting, and CI auto-doc build (`Doxygen` + `Sphinx -W`).
- Stand up `kg/build_kg.py` (Python AST + Doxygen XML + validation/decision graph indexer) and MCP resumption endpoints (`resume_session_context`, `trace_physics_lineage`, `check_staleness`), seeded with `MEMORY.md`, `logs/DECISIONS.md`, and `logs/WORK_LOG.md`.
- Freeze manifest-driven reference test cases in `validation.py` following `EVIDENCE.md` using **PSML-only (`.psml`)** pseudopotentials across all cases:
  - Reference binaries: standard SIESTA and `siesta_spglib_full_impl`.
  - Test matrix: $\Gamma$-only molecule/2D slab, $k$-sampled bulk semiconductor and metal (including hcp Co collinear, aligned/canted NC, and SOC cases from `scf_params`), antiferromagnetic crystal, strained triclinic cell, and dispersion-bound bilayer.
  - Contracts: explicit `last_scf_free_e` and `final_post_scf_free_e`, atomic forces, stress tensor, plus finite-difference energy derivative checks.

### Phase 1: Core Non-Symmetrized Static SCF Engine & Component Parity
1. **PSML species & two-center C++ module:** Parse `.psml` pseudopotentials exclusively (semilocal channels, KB projectors, NLCC pseudo-core charge, scalar/SOC $j$-dependent terms; no `.psf`) alongside PAO radial basis (matching SIESTA’s `.ion` tables for exact 1-to-1 comparison first), compute $S, T, V^{\text{KB}}, V^{\text{NA}}$ and position/strain derivatives on a C++ neighbor list. *Gate: component parity against SIESTA `.HSX`/`H0`; update Knowledge Graph.*
2. **Grid FFI kernels:** Implement C++ (and CUDA) JAX FFI custom calls for `rhoofd`, `vmat`, and their custom VJP backward passes (`dfscf` orbital-grid gradients), plus `libxc` pointwise evaluation. *Gate: adjoint identity and density/potential parity against SIESTA `rhoofd`/`vmat`; update Knowledge Graph.*
3. **JAX eigensolver, FFT grid ops, & dispersion:** Implement batched `eigh` (`nspin=1,2,4,8`), $D$/$E$ assembly, 3D FFT Poisson, 3D FFT GGA gradients/stress, and DFT-D3 / rVV10. *Gate: isolated XC/Poisson/dispersion parity; update Knowledge Graph.*
4. **SCF mixing & adaptive controller (`scf_params`):** Implement descriptor-driven initialization, Tikhonov–Pulay, ADIIS, decoupled charge/spin mixing, exact scalar SVD contract, native trust-region/annealing adaptive loop, and explicit `DM_out` post-SCF pass. *Gate: match non-symmetrized baseline SIESTA across the full Phase 0 matrix on `final_post_scf_free_e`, forces, stresses, and finite-difference derivative checks, match hcp Co mixer-endpoint convergence within `scf_params` tolerances, and atomically update `kg/current/docs_kg.db` + `MEMORY.md`.*

### Phase 2: 4-Step Native Symmetry Roadmap

#### Step 1 — Space/Magnetic Group Engine, IBZ Reduction, & Commensurate Mesh
- Wrap `spglib` (`spg_get_dataset` + `spg_get_magnetic_dataset` + `spg_get_stabilized_reciprocal_mesh`) in Python/C++.
- Build forward/inverse atom maps, lattice translations, and magnetic subgroup/Shubnikov time-reversal flags.
- Enforce symmetry-commensurate `(N1, N2, N3)` mesh sizing and construct the grid index permutation map $\pi_S$.
- **Acceptance test:** Verify space-group invariance of all generated atom/shift/grid maps, confirm exact agreement of IBZ $k$-points and weights against `Src/kpoint_scf.F90` on cubic, hexagonal, nonsymmorphic, and magnetic test cells, and publish updated Knowledge Graph validation edges.

#### Step 2 — Symmetry-Closed Sparsity & Precompiled `D`/`E` Projection Operator
- Close the CSR/COO orbital-pair sparsity pattern under all $g \in G$ during graph construction (assert zero missing symmetry partners).
- Implement sequential `(atom, l, m, iz)` indexing, 16-point Fibonacci-sphere Wigner-$D$ generation with orthogonality assertion ($\le 10^{-10}$), and SU(2) spinor maps.
- Precompile the sparse symmetry projection $\mathcal{P}_{\text{sym}}$ into static COO/CSR index + weight buffers and execute it in JAX on GPU for `nspin=1, 2, 4, 8`.
- **Acceptance test:** Check idempotency ($\mathcal{P}_{\text{sym}}^2 = \mathcal{P}_{\text{sym}}$), trace/electron-count conservation, and element-wise parity of projected $D$ and $E$ against `Src/m_dm_symmetrize.F90`, and update Knowledge Graph edges.

#### Step 3 — Native Grid Symmetrization & Irreducible Two-Center Integrals
- Apply exact grid permutation symmetrization $\rho(\mathbf{r}) \leftarrow \frac{1}{|G|} \sum_{S} \rho(\pi_S^{-1}(\mathbf{r}))$ inside the SCF loop prior to Poisson/`libxc`/rVV10 (and symmetrize $V_{\text{eff}}(\mathbf{r})$ equivalently, or evaluate on irreducible grid points).
- Partition $(a, b, \mathbf{R})$ pairs into symmetry orbits and evaluate `matel` / $S, T, V^{\text{KB}}, V^{\text{NA}}$ (and derivatives) only on orbit representatives, rotating into symmetry partners via Wigner-$D$.
- **Acceptance test:** Verify that $H$ and $S$ commute with all symmetry operators to machine precision, eggbox force/stress symmetry breaking vanishes *before* post-projection, IBZ SCF energies match full-BZ unsymmetrized SCF energies, and update Knowledge Graph edges.

#### Step 4 — Full Variational Force/Stress Symmetry & End-to-End Validation
- Couple fixed-$(D, E)$ variational force/stress evaluation (and the post-SCF `DM_out` pass) with final symmetry projection (`symmetry_project_forces`, `symmetry_project_stress`) as a numerical cleanup pass and verify that raw pre-projection residuals are already at numerical noise level.
- Run the full `validation.py` suite comparing:
  1. Unsymmetrized full-BZ vs. native-symmetry IBZ (self-consistency check).
  2. Native-symmetry JAX+C++ vs. `siesta_spglib_full_impl` (cross-code parity check on energies, forces, stresses).
  3. Independent physical contracts: finite-difference energy derivatives vs. analytical forces/stresses, and symmetry invariance under arbitrary cell/atom permutations and rigid rotations.
- Atomically publish the validated Knowledge Graph snapshot, Sphinx auto-docs, and milestone entry in `MEMORY.md` / `logs/WORK_LOG.md`.

### Phase 3: Performance Profiling & Optimization
- Profile GPU kernel occupancy, host-device transfers (verifying zero per-SCF CPU transfers), and memory footprint per the profiling envelope in `EVIDENCE.md`.
- Tune orbital-grid bounding-box tiling in the CUDA `rhoofd`/`vmat` kernels and batched $k$-point `eigh` sizing.
- Benchmark SCF iteration counts and wall-clock speedups of the physics-informed Tikhonov–Pulay + decoupled-spin + adaptive controller vs. standard fixed-weight Pulay across unseen material families.
