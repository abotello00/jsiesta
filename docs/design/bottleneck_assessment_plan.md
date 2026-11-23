# Plan: assessing the bottlenecks of `known_bottlenecks.md`

Written 2026-11-21. Companion to `known_bottlenecks.md`.

**Goal.** Turn each diagnosis in `known_bottlenecks.md` (B1-B4 and Section 5.5) into a hypothesis with a measurement
that can confirm or refute it, *before* the fixes P1-P5 are designed. Each work package below ends with a decision
and the number that justifies it.

**Ground rules.**
- Assessment only: no production code changes.
  - Instrumentation (timers, counters, NVTX ranges) may be committed.
  - Ablations that break results (timing-only kernels) live on a throwaway branch `assess/*` and are never merged.
- Every number gets the git commit, device, jax version, grid, and edge/orbital counts. It is the median of at least 5
  repeats, with the spread reported. Run 18-core CPU jobs with `--exclusive`; the scaling study ran on shared nodes.
- Correctness gates stay green on every branch: `tests/test_phase3_cuda.py`, `test_phase1_species_and_two_center.py`,
  `test_na_pair_exact.py`, `test_stress_components.py`.
- Results go into a new Section 9 "Assessment results" of `known_bottlenecks.md`, one decision per package in
  `logs/DECISIONS.md` (D-031 ff.), and the work log.

**Outcome (2026-11-21).** The facts of Section 0 were strong enough to go straight to fixes on branch `perf/bottlenecks`
instead of running the full assessment first. The decisions were therefore numbered by change, not by package: D-031
set-up (A2), D-032/D-033 eigensolver (A4), D-034 CPU forces (A3, CPU part), D-035 a test regression, D-036 CUDA grid
kernels (A1). The section "Status of the packages" at the end records what was measured, what was skipped and the
verdict on each hypothesis.

---

## 0. Facts found while preparing this plan

These come from reading the code while writing this plan. They already correct or sharpen the report, and the work
packages test them.

| # | Fact (file) | Consequence for the report |
|---|---|---|
| F1 | `vmat` (Mode 1 of `edge_orbital_grid_kernel`, `cpp/cuda/rhoofd_vmat.cu:740-748`) accumulates into a per-thread register and writes once per edge, **without atomics**. It costs 83-87 % of `rhoofd` (1.37 s vs 1.65 s at 8 atoms). | The global `atomicAdd` (B1 consequence 3) cannot be the main cost. The per-(edge, point) work is. Section 5.1 needs this caveat. |
| F2 | The CPU kernel (`cpp/src/grid_ffi.cpp:95-299, 522-570`) builds an `ActiveGridCache` (orbital values per grid point). It keeps the cache in an LRU keyed by the geometry, basis and pattern bytes. During an SCF at fixed geometry, only the first call evaluates orbitals; later calls only contract. SIESTA does the same by default: `DirectPhi` defaults to `.false.` (`Src/meshsubs.F:1083`), so it stores phi on the mesh. The CUDA path has no cache and evaluates every call. | The 13-16x "CPU vs GPU" and 50-75x "SIESTA vs GPU" ratios compare *cached* contraction with *uncached* evaluation plus contraction. This is the most likely root cause of B1 and points P1 at "cache phi on the device", not only "tile better". |
| F3 | Every thread of the CUDA kernel holds `c_a[49]`, `c_b[49]`, `dc_a[147]`, `dc_b[147]` (`kShCount = (kMaxL+1)^2` with `kMaxL = 6`), and `dev_real_solid_harmonics` adds `A[49]`, `B[49]` (+ derivatives). This is several kB per thread. | Probably register spills to local memory. This is checkable at compile time (`-Xptxas -v`) at no cost. The Si DZP basis needs l <= 2 (9 harmonics), not 49. |
| F4 | `dev_real_solid_harmonics` computes all (l, m) up to `l_req` for each of the two orbitals at every (edge, point). Only one entry `c[lm]` is used. | Extra work on top of the redundancy in B1 consequence 1. |
| F5 | The pattern stores both directed edges (mu, nu, T) and (nu, mu, -T) (`species/psml.py:827`). It also stores KB-bridged pairs whose orbitals do not overlap: these threads exit at the distance test, but still occupy warps. | The `rhoofd` work is done twice for each symmetric pair: the expected gain from half-edges is up to 2x. The fraction of idle threads is unmeasured. |
| F6 | Each kernel call does `cudaMallocAsync` and re-prepares the splines with `prepare_geom_and_pao_splines_kernel<<<n_shells, 1>>>`: one serial thread per shell running a tridiagonal solve over all spline points. | Probably small, but it is serial and runs on every call, so it must be measured. |
| F7 | The in-SCF solver `_solve_generalized_eigh_single` (`scf/eigensolver.py:56-84`) runs `eigh(S)`, then Cholesky, `inv(L)`, three GEMMs and `eigh(H')`. It runs under `jax.vmap` (`:158`) with a `lax.cond` on a batched predicate, which XLA lowers to a `select`: **both branches run**, including the Löwdin branch's GEMMs. S is constant during the SCF but is factorised every iteration. The measured in-SCF time is 22 / 53 ms at 416 / 832 orbitals against 5.8 / 12.8 ms for one `eigh(h, s)`: about 4x. | The report's sentence "the solver inside the SCF agrees with the benchmark" (Section 3.3) is wrong and must be corrected. B4 (CPU 7x slower than ScaLAPACK) is probably mostly this redundancy, not XLA's `eigh`. |
| F8 | The forces path (`observables/forces.py:299-345`) recomputes rho, rho_NA, Hartree and XC at D_out. It then runs `jax.grad` through Mode 2 (gradients and Hessians per (edge, point)) and the host-side `_kb_forces_and_stress`, which has Python loops over atoms and KB shells (`:90-116`). | B3 has three candidate parts; none is measured. |

---

## A0. Infrastructure (prerequisite, about 1 day)

1. **Close the running study.**
   - Collect jobs 29542773 (A100) and 29543065 (jscpu18).
   - Fill the 128/216-atom rows of Section 3.1/3.4 and regenerate `reports/scaling/scaling.png`.
   - Correct the Section 3.3 sentence (F7).
2. **Kernel micro-benchmark `scripts/bench_grid_kernels.py`.**
   - Load a scaling case once (si008/si016/si032/si064).
   - Call the FFI ops directly: `rhoofd`, `vmat`, the Mode 2 backward, `neutral_atom_grid`, `density_on_grid` cold vs warm. Use `block_until_ready`, 5 repeats, with optional NVTX ranges.
   - Write JSON in the `reports/scaling/` schema.
   - This decouples kernel timing from the SCF driver.
3. **Pattern statistics `scripts/pattern_stats.py`** (pure Python, local).
   - Per case: n_edges; overlapping vs KB-bridge-only edges; symmetric pairs; per-edge bounding-box points and points inside both spheres (min/mean/max/histogram).
   - Totals: Σ(edge, point) pairs against Σ(point, orbital) pairs, i.e. the CPU cache entries.
   - These turn B1 consequence 1 and F5 from estimates into numbers.
4. **Profiler availability on Snellius** (read-only `ssh`):
   - check `module avail` for CUDA 12.6 `nsys`/`ncu`;
   - check whether `ncu` may read performance counters on gpu_a100 nodes (they may require `--cap-add`; if refused, fall back to `nsys` + cudaEvent timing + ptxas statistics).

## A1. B1: the CUDA grid kernels (main package, about 4-5 days, about 6 GPU-hours)

Hypotheses (one may dominate, several may combine):
- H1a: no phi cache (F2);
- H1b: redundant evaluation per (edge, point) and per direction (B1.1, F4, F5);
- H1c: register spills / local memory (F3);
- H1d: warp divergence and load imbalance from per-edge boxes (B1.2, B1.4);
- H1e: atomics (B1.3, already weakened by F1);
- H1f: per-call overhead (F6).

| Step | Experiment | Metric | Refutes / confirms |
|---|---|---|---|
| A1.1 | Compile `rhoofd_vmat.cu` with `-Xptxas -v` and `--resource-usage` for `sm_80`; record registers, spill stores and loads, and stack frame per Mode | bytes spilled per thread | H1c (cost: minutes) |
| A1.2 | `pattern_stats.py` on si008-si064: redundancy factor R = Σ(edge, point) / Σ(point, orbital); fraction of idle edges; box-size spread | R, idle %, max/mean box | H1b, H1d (quantitatively) |
| A1.3 | `nsys` timeline of one SCF iteration at si016: kernel durations, `prepare_*` kernels, malloc/free, gaps between kernels | % of iteration in `edge_orbital_grid_kernel` vs everything else | H1f |
| A1.4 | `ncu --set full` on `edge_orbital_grid_kernel<0>` and `<1>` at si016: achieved occupancy, warp execution efficiency, local-memory traffic, L1/L2 hit rates, atomic throughput, FP64 pipe utilisation, stall reasons | dominant stall reason | H1c/H1d/H1e |
| A1.5 | Ablations on `assess/cuda-ablation`, each timed with A0.2 at si016 and si064: (a) `kMaxL` as a template parameter = actual l_max (2); (b) evaluate only the needed `lm`; (c) `atomicAdd` replaced by a plain store (wrong result, timing only); (d) upper-triangle edges only (rhoofd x2 of weight); (e) edges sorted by box size; (f) splines prepared once and reused | speed-up per ablation, alone and combined | isolates each factor |
| A1.6 | Cache feasibility: CPU `ActiveGridCache` entry count and bytes per case from A1.2 (12-24 B per entry for value and index, more with gradients); extrapolate to 512/1000 atoms against 40 GB | GB at 512/1000 atoms | decides whether a device phi cache (H1a fix) fits or needs tiling/recompute |
| A1.7 | CPU reference split: CPU `rhoofd` cold (cache build) vs warm (contraction only) on 18 cores, si008-si128 | cold/warm ratio | measures how much of the CPU advantage is the cache (F2) |
| A1.8 | Lower-bound model: (A1.2 Σ(point, orbital)) x (cost of one orbital evaluation from A1.4) + contraction FLOPs at A100 FP64 and bandwidth | predicted optimal ms per iteration | sets the P1 target and checks the Section 5.5 estimate |

**Decision D-031 (P1 design).**
- Choose between:
  - (i) a device phi cache plus per-point or per-tile contraction (the SIESTA/CPU design);
  - (ii) on-the-fly evaluation in grid tiles, evaluating each orbital once per point;
  - (iii) quick fixes only (A1.5 a/b/d/f), if they reach the CPU path.
- Record the measured speed-up of each ablation and the A1.8 target.
- Acceptance for "B1 understood": the ablations plus the model explain at least 80 % of the gap between the current GPU time and the A1.8 bound.

## A2. B2: the two-centre set-up (about 2 days, local)

| Step | Experiment | Metric |
|---|---|---|
| A2.1 | Time each block of `_eval_kb_and_na_pair` (KB value tables `:600-620`, KB link search `:633-665`, KB block assembly `:750-770`, NA pair loop `:786-812`, the w-loop `:860-890`) with `line_profiler` or explicit timers at 8, 16, 32 and 64 atoms | seconds per block, fitted exponent per block |
| A2.2 | Count candidates tested vs pairs accepted in each loop | waste ratio (the N^2 part) |
| A2.3 | Check where `TwoCenterEvaluator.evaluate` is called: SCF set-up, forces, `basis_opt`, symmetry paths. Check whether a geometry step or the forces call re-runs it | number of calls per workflow |
| A2.4 | Prototype on `assess/two-center`: replace the link and pair searches by a periodic cell list (NumPy, or `scipy.spatial.cKDTree` with `boxsize` for orthorhombic cells and a general cell list otherwise). Check that the sorted (mu, k, shift) and (ia, ja, shift) index arrays are *bitwise identical* to the current ones; time 8-128 atoms | identical sets; new time and exponent |
| A2.5 | Same for `build_sparse_pattern` and `build_basis` (2-41 s, superlinear) | seconds per block and exponent |

**Decision D-032 (P2 design).**
- NumPy vectorisation is enough if A2.4 reaches < 10 s at 64 atoms; otherwise move the work into `cpp/src/two_center.cpp`.
- Acceptance: the measured exponent is ≤ 1.2 and the index sets are identical.

## A3. B3: forces and stress (about 2 days, about 2 GPU-hours)

| Step | Experiment | Metric |
|---|---|---|
| A3.1 | Timers with `block_until_ready` around each stage of `StaticObservableEvaluator.evaluate`: fresh rho/NA/Hartree/XC at D_out; `_kb_forces_and_stress` (host); `jax.grad` of `_deformed_grid_energy` (forward + Mode 2 backward); two-centre force/stress assembly; symmetrisation. Both A100 and CPU-18, si008-si064 | seconds per stage, per atom |
| A3.2 | Count FFI custom calls in the jaxpr/HLO of the gradient (`jax.make_jaxpr`), to confirm no repeated forward passes | calls per evaluation |
| A3.3 | Mode 2 vs Mode 0 kernel time ratio (A0.2); register and spill numbers from A1.1 for Mode 2/5 | ratio |
| A3.4 | Check that `return_stress_components=False` skips the four extra `jax.grad` calls (`forces.py:430`) | yes/no |

**Decision D-033.** Rank the B3 parts. If Mode 2 dominates, B3 folds into P1. If `_kb_forces_and_stress` dominates, it
joins P2.

## A4. B4 and the eigensolver path (about 1 day, about 0.5 GPU-hour)

| Step | Experiment | Metric |
|---|---|---|
| A4.1 | Dump the optimised HLO of `batched_generalized_eigh` for n_k = 1 and count `eigh`/`cholesky`/`trsm` custom calls; confirm or refute "both branches run" (F7) | number of eigensolves per k-point |
| A4.2 | Benchmark at 416-6656 orbitals on A100 and CPU-18 (extend `scripts/scaling_eigh.py`): (a) current function; (b) `jax.scipy.linalg.eigh(h, s)`; (c) Cholesky of S cached once per SCF + `eigh(L^-1 H L^-T)` with triangular solves instead of `inv`; (d) a scalar predicate (`lax.cond` outside the `vmap`) | time per variant |
| A4.3 | CPU only: thread scaling of the XLA CPU `eigh` (1/4/18 threads); compare with LAPACK `dsygvd` through SciPy | is the XLA kernel itself the problem? |
| A4.4 | Check that the k-point and spin loops (`eigensolver.py:245-305`) do not solve the same S factorisation more than once per iteration | count |

**Decision D-034.** If variant (c) or (d) recovers about 4x, record it as an independent quick win (P5') that benefits
CPU and GPU alike. B4 then narrows to whatever remains.

## A5. Other costs and the transfer audit (about 1 day)

- JIT compile time of the first SCF call (23-73 s on the A100, growing with N):
  - measure it with `jax.log_compiles` and the persistent compilation cache on and off;
  - check whether shape-polymorphic padding (as in `profile_phase3`) removes recompiles between sizes.
- Driver overhead: the iteration time minus (kernels + eigensolver + mixer) from A0.2 and A4, on CPU and GPU.
- Repeat the host-transfer audit (`audit_jit_host_transfers`) at si064. The current audit is from 2-3-atom cells.

## A6. Validity of the comparison (about 1 day, about 2 node-hours)

- SIESTA:
  - re-run si064, si128 and si512 on 18 ranks with `--exclusive`, at 3 repeats;
  - one run with `DirectPhi .true.` (which shows the cost SIESTA *would* have without the phi cache: the like-for-like comparison with the current GPU kernel);
  - one run with ELPA (`Diag.Algorithm ELPA-2stage` or the build's equivalent).
- jsiesta CPU-18: 3 repeats `--exclusive` at si064 and si128.
- Second system class (optional, after the user agrees): a spin-polarised metal with k points (for example a Co or Fe supercell). There the diagonalisation is multiplied by n_k and n_spin, which shifts the grid/eigensolver balance. This affects where the GPU pays.

---

## Order, dependencies and budget

```
A0 ──┬─> A1.1, A1.2, A1.7, A2, A4.1          (cheap; local or one short job)
     ├─> A1.3-A1.6, A3, A4.2-A4.4             (Snellius GPU; needs A0.2, A0.4)
     └─> A6                                    (Snellius CPU; independent)
A1.8 needs A1.2 + A1.4; D-031 needs A1.5-A1.8; P4 re-measurement after P1/P2 reuses all scripts.
```

| Package | Effort | Snellius | Output |
|---|---|---|---|
| A0 | 1 d | collect jobs, 1 short probe | final plot; bench and stats scripts |
| A1 | 4-5 d | about 6 GPU-h | D-031, design of P1 with target |
| A2 | 2 d | none | D-032, design of P2 |
| A3 | 2 d | about 2 GPU-h | D-033 |
| A4 | 1 d | about 0.5 GPU-h | D-034, possible quick win |
| A5 | 1 d | about 1 GPU-h | JIT and overhead numbers |
| A6 | 1 d | about 2 CPU node-h | corrected comparison baseline |

Snellius submissions are outward-facing: each `sbatch` batch is confirmed with the user before it is sent.

## Risks

- `ncu` may be blocked on the shared GPU nodes (no counter access). In that case A1.4 relies on `nsys`, the ptxas
  statistics and the ablations; the conclusions are then weaker and labelled as such.
- The ablation kernels give wrong physics on purpose. They must never feed a correctness test or be merged.
- The local machine has no CUDA. All GPU steps go through Snellius, with queue delays, so bundle them into few jobs
  (one sbatch per package).
- A device phi cache (H1a) can exceed 40 GB at 1000 atoms; A1.6 decides this before any design is fixed.

## What "done" means

- Every row of the `known_bottlenecks.md` summary table carries a verdict: confirmed, refined or refuted, with the
  measurement behind it.
- The P1-P5 table is replaced by designs with measured targets (D-031 to D-034).
- The Section 3.3 sentence (F7) and the atomics claim (F1) are corrected.
- The scripts and sbatch files of A0-A6 are committed, so P4 can re-run everything after the fixes.

## Status of the packages (2026-11-21)

Measured on the laptop (Intel i7-1068NG7) and on Snellius A100 jobs 29555653 and 29559809; details in
`known_bottlenecks.md` Section 9 and `logs/DECISIONS.md` D-031 to D-036.

| Package | Done | Skipped | Verdict |
|---|---|---|---|
| A0 | final scaling plot; `scripts/scaling_jsiesta.py --kernels`; `scripts/snellius_perf_gpu.sbatch` | NVTX ranges, median-of-5 harness | sufficient for the A/B below |
| A1 | A1.1 (ptxas); fix (i) of the D-031 choice: host plan with stored phi uploaded to the device (D-036); same-job A/B at 64 atoms | A1.2-A1.5 (`nsys`, `ncu`, ablations), A1.8 model | **H1a confirmed** as the dominant cause: `rhoofd` 9.44 s -> 0.047 s, `vmat` 8.20 s -> 0.068 s. **H1c confirmed** at compile time (148-255 registers, up to 18 kB stack, against 56-90 and none). H1b/H1d/H1f not separated (the fix removes them together). **H1e (atomics):** refuted for the old kernel (F1); relevant again for the new `vmat`, now 58 % of the iteration |
| A2 | A2.1-A2.2 by profiling; A2.4-A2.5 as NumPy vectorisation (D-031), index sets bitwise identical | cell list / cKDTree, C++ port | **Confirmed:** Python loops over orbitals x images. 64 atoms: 400 s -> 8.7 s on the A100 host (target < 10 s met); growth roughly linear |
| A3 | CPU: cause found and fixed (D-034: single-threaded backward rebuilding its cache), 34.2 s -> 3.8 s at 16 atoms | A3.1 stage timers on the GPU, A3.2-A3.4 | **Open on the GPU:** forces + stress 75 s at 64 atoms (was 100 s), now the main GPU cost; the backward modes still use the edge kernel |
| A4 | A4.1 (jaxpr count), A4.2 (c) and (d) as D-032; assembly found and fixed (D-033) | A4.3 thread scaling of XLA `eigh` | **F7 confirmed** and B4 refined: the CPU cost was mostly the scatter-add assembly (about 6 us per edge), then the repeated S transform. In-SCF solve on the A100 0.178 s -> 0.039 s at 1664 orbitals (plain `eigh` 0.033 s) |
| A5 | - | all | open (JIT time 15-95 s on the A100 for the first call, measured in passing) |
| A6 | - | all | open: the SIESTA comparison still uses shared nodes, `DirectPhi .false.` and ScaLAPACK |

Next measurements: A3 on the GPU (port the backward to the plan, then stage timers), `vmat` without atomics, and A6.

