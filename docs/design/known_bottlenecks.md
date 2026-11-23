# Known performance bottlenecks of jsiesta (GPU and CPU) against SIESTA

Written 2026-11-21 from the GPU-vs-SIESTA scaling study (`scripts/scaling_*.py`, `scripts/snellius_scaling_*.sbatch`,
`scripts/plot_scaling.py`). Raw data: `reports/scaling/*.json`; plot: `reports/scaling/scaling.png` (six panels).
Statements are labelled **measured** (a timing in the data), **read from code** (source inspection or a profile of one
function) or **estimate** (arithmetic on measured numbers, not a measurement).

## 1. Summary

> **Status 2026-11-21:** Sections 1-8 describe the code before branch `perf/bottlenecks`. After D-031 to D-036 the A100
> iteration is 100-240x faster and faster than SIESTA at every size measured (0.117 s at 64 atoms against 0.357 s on 18
> ranks); the set-up is 6-120x faster (more for larger cells); GPU forces + stress are now the main cost. See Section 9.

1. **On the A100, the full SCF iteration is 50-75 times slower than SIESTA and 13-16 times slower than the same
   jsiesta code on 18 CPU cores** (measured, 8-64 atoms). There is no system size, in the range tested, where the GPU
   pays for a complete iteration. Cause: the two CUDA grid kernels `rhoofd` and `vmat` (99 % of the GPU iteration at
   64 atoms), which are a different, much less efficient algorithm than the CPU kernels of the same library.
2. **The dense eigensolver is where the GPU wins, and it wins clearly**: the A100 solves the generalised eigenproblem
   5-15 times faster than SIESTA's `diagon` on 18 ranks and 2.8-18 times faster than on a full 192-core node, for
   every size from about 400 orbitals up (measured: 0.76 s against 2.3 s at 6656 orbitals, 4.4 s against 12.5 s at 13000
   orbitals on 192 ranks; the margin over a full node shrinks from 18x at 416 orbitals to 2.8x at 13000).
3. **The set-up before the first SCF step is a second, independent bottleneck**: the two-centre integrals take 400 s
   at 64 atoms (SIESTA: 0.5 s) and grow as N^2. 98 % of that time is a Python loop (read from code, profiled), not the
   compiled integral engine. It runs on the host, so it costs the same with or without a GPU.
4. **Forces and stress cost 1.5 s per atom on the A100** (100 s at 64 atoms; SIESTA 1.1 s on 18 ranks). Not profiled
   separately; it uses the backward modes of the same CUDA kernels plus the Python two-centre derivative.
5. What is *not* a bottleneck: the Hartree solve and the XC evaluation (0.3-0.6 ms on the A100), the GPU eigensolver,
   and the mixer.

| Stage | Where jsiesta stands | Dominant cause | Fix (Section 6) |
|---|---|---|---|
| Grid: density from DM (`rhoofd`) and potential matrix (`vmat`) | 50-75x slower than SIESTA on the A100, 13-16x slower than its own CPU path | edge-parallel CUDA kernel recomputes basis functions per (edge, grid point); atomics | P1 |
| Two-centre set-up | 800x slower than SIESTA at 64 atoms, N^2 | Python loops over atoms/orbitals x periodic images | P2 |
| Forces + stress | 94x slower than SIESTA on 18 ranks, 350x than on 192 ranks, at 64 atoms | backward CUDA kernels (as P1) + P2 | P1, P2 |
| Dense eigensolver | 5-15x **faster** than SIESTA 18 ranks and 2.8-18x than 192 ranks on the A100; 7x slower on CPU | XLA CPU `eigh` against ScaLAPACK | none needed on GPU |

## 2. What was measured

**System.** Bulk diamond Si supercells of the 8-atom cubic cell (a = 5.43 A): 8, 16, 32, 64, 128, 216, 512, 1000 atoms.
PBE, nspin = 1, Gamma point only, MeshCutoff 100 Ry, SIESTA's default DZP basis (13 orbitals per atom) from the
PseudoDojo nc-sr-05 PBE Si PSML. Both codes read the same `case.json`; SIESTA's `Si.ion.xml` tables are used on the
jsiesta side, and the mesh dimensions of the two codes are identical (checked in the output of every size: for example
72x36x36 at 16 atoms, 144x144x144 at 512 atoms).

**Codes and hardware (Snellius).**
- jsiesta on one NVIDIA A100-SXM4-40GB (gpu_a100, 18 host cores, jax 0.12.1, CUDA 12.6, double precision).
- jsiesta on 18 CPU cores of a genoa node (JAX CPU, CPU-only build of the FFI library).
- SIESTA on 18 MPI ranks (same core count as the GPU's host share) and on 192 ranks (a full genoa node). The binary is
  the `siesta_ndsets` build: a Release MPI build with ScaLAPACK/ELPA support, from a development
  branch (`ndatasets`, 9fd0248a0) whose SCF core is standard SIESTA. No plain upstream SIESTA build was available on
  the cluster. The default solver (`diagon`, ScaLAPACK) was used.
- The 18-core jobs share a node with other jobs (Slurm default), so their timings carry some noise.

**Method.** Each code runs a fixed number of SCF iterations (it cannot converge: tolerances 1e-14).
- SIESTA: its own timer table (`IterSCF`, `setup_H`, `diagon`, `PostSCF`, `Setup_H0`), time per call; 6 iterations.
- jsiesta: the marginal cost of one iteration is the slope of the wall time of `run_scf` between 3 and 3+n
  iterations after a warm-up call (JIT excluded); eigensolver and kernels are timed separately (best of 3,
  `block_until_ready`); set-up and forces+stress are timed once.
- The kernel breakdown (`--kernels`) was run for 8, 32 and 64 atoms only.
- The dense-eigensolver benchmark (`scripts/scaling_eigh.py`) uses random symmetric H and a positive-definite S; it fixes
  the cost, not the physics.

## 3. Results

### 3.1 One SCF iteration (seconds, measured)

| atoms | orbitals | SIESTA 18 ranks | SIESTA 192 ranks | jsiesta 18 CPU cores | jsiesta A100 | A100 / SIESTA-18 | A100 / jsiesta-CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 104 | 0.042 | (fails, see 3.5) | 0.19 | 3.07 | 73x | 16x |
| 16 | 208 | 0.080 | 0.077 | 0.34 | 5.18 | 65x | 15x |
| 32 | 416 | 0.155 | 0.130 | 0.63 | 9.45 | 61x | 15x |
| 64 | 832 | 0.357 | 0.262 | 1.40 | 18.15 | 51x | 13x |
| 128 | 1664 | 0.887 | 0.407 | 4.51 | 35.47 | 40x | 7.9x |
| 216 | 2808 | 1.96 | 0.834 | 14.60 | (not reached) | | |
| 512 | 6656 | 13.98 | 2.72 | | | | |
| 1000 | 13000 | | 13.15 | | | | |

The A100 time grows roughly linearly with the number of atoms (x1.7-1.9 per doubling). SIESTA's grows more slowly at
first (the grid part is linear and small) and then follows its diagonalisation (O(N^3)).

### 3.2 Where the time goes inside an iteration

**jsiesta on the A100** (measured, `diag_a100_*.json`):

| atoms | `rhoofd` | `vmat` | Hartree + XC | eigensolver | whole iteration |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.653 s | 1.371 s | 0.27 ms | 8 ms | 3.06 s |
| 32 | 5.011 s | 4.344 s | 0.39 ms | 22 ms | 9.55 s |
| 64 | 9.655 s | 8.333 s | 0.56 ms | 53 ms | 18.15 s |

`rhoofd` + `vmat` are 99 % of the iteration, at about 0.28-0.38 s per atom, on a grid of only 46 656 points (36^3) at
8 atoms. The Hartree solve (FFT) and the XC evaluation take well under a millisecond: the grid *size* is not the problem.

**SIESTA** (measured, per iteration; `setup_H` contains `rhoofd`, `POISON`/`fft`, `XC` and `vmat`):

| atoms | 18 ranks: `setup_H` / `diagon` | 192 ranks: `setup_H` / `diagon` |
|---:|---|---|
| 64 | 0.26 s / 0.09 s | 0.045 s / 0.21 s |
| 128 | 0.51 s / 0.36 s | 0.083 s / 0.31 s |
| 512 | 2.1 s / 11.7 s | 0.26 s / 2.3 s |
| 1000 | | 0.50 s / 12.5 s |

SIESTA's whole grid stage costs 0.26 s at 64 atoms on 18 ranks, while the A100 needs 18 s for the two kernels: a factor
of about 70. From 512 atoms upwards SIESTA is dominated by the diagonalisation (84 % at 512 atoms on 18 ranks, 95 % at
1000 atoms on 192 ranks).

### 3.3 The dense eigenproblem alone (seconds, measured)

(Correction, 2026-11-21: the in-SCF solve was not equal to the benchmark below. It rebuilt the factorisation of S every
iteration, ran the unused Löwdin branch under `vmap`, worked in complex arithmetic at Gamma and assembled H and S with a slow
scatter-add; see Section 9.)

| orbitals | A100 (`jax.scipy.linalg.eigh(h, s)`) | SIESTA 18 ranks `diagon` | SIESTA 192 ranks `diagon` | jsiesta 18 CPU cores (in SCF) |
|---:|---:|---:|---:|---:|
| 416 | 0.0058 | 0.032 | 0.104 | 0.14 |
| 832 | 0.0128 | 0.091 | 0.210 | 0.59 |
| 1664 | 0.0333 | 0.362 | 0.314 | 2.51 |
| 2808 | 0.102 | 1.131 | 0.627 | |
| 6656 | 0.763 | 11.73 | 2.335 | |
| 13000 | 4.398 | | 12.46 | |

The A100 is ahead of SIESTA on 18 ranks by 5.5-15x (416 to 6656 orbitals) and of a full 192-core node by 18x at 416
orbitals, falling to 9x at 1664, 6x at 2808, 3.1x at 6656 and 2.8x at 13000 (the 192 ranks are oversubscribed at the
small sizes, and ScaLAPACK scales well at the large ones). Below about 200 orbitals nothing is gained (SIESTA 18 ranks 9 ms at 104 orbitals against 8 ms). The solver
inside the SCF (`eigensolver_s` in the A100 runs: 8, 22, 53 ms at 104, 416, 832 orbitals) agrees with the benchmark.
On the CPU, jsiesta's eigensolver is 7x slower than SIESTA's ScaLAPACK at 1664 orbitals (2.51 s against 0.36 s).

### 3.4 Set-up and forces (seconds, measured)

| atoms | jsiesta two-centre (A100 job) | jsiesta two-centre (CPU job) | SIESTA `Setup_H0` (18 / 192) | jsiesta forces+stress (A100) | SIESTA `PostSCF` (18 / 192) |
|---:|---:|---:|---|---:|---|
| 8 | 28 | 24 | 0.09 / - | 14 | 0.20 / - |
| 16 | 65 | 55 | 0.15 / 0.14 | 26 | 0.40 / 0.09 |
| 32 | 160 | 133 | 0.24 / 0.18 | 51 | 0.53 / 0.15 |
| 64 | 400 | 331 | 0.47 / 0.27 | 100 | 1.06 / 0.29 |
| 128 | 1583 | 1309 | 0.90 / 0.36 | 199 | 2.2 / 0.47 |
| 216 | | 3884 | | | |
| 512 | | | 3.2 / 0.69 | | 15.8 / 1.5 |

The two-centre set-up doubles the atom count and quadruples the time between 64 and 128 atoms (331 s to 1309 s):
**N^2**. Extrapolated (estimate): about 5.8 h at 512 atoms and about 22 h at 1000 atoms, which is why the 512- and
1000-atom jsiesta points were not run. The first call of the SCF (JIT compilation plus two iterations) takes 23-73 s on the A100 (8-64 atoms), 8-17 s on
CPU; the basis and sparse-pattern construction (`build_basis`, `build_sparse_pattern`) takes 2-41 s (8-128 atoms), also
superlinear.

### 3.5 Remarks on the data

- SIESTA on 192 ranks fails for the 8-atom cell (its output ends with a `max_size` error; presumably 104 orbitals are too
  few for 192 ranks, not investigated); the point is absent from the plot.
- For jsiesta on the CPU, the iteration minus the eigensolver is the grid stage plus driver overhead: 2.0 s at 128
  atoms against SIESTA-18's 0.51 s, so the CPU grid path is also about 4x slower than SIESTA's, far less than the GPU's
  70x.
- The A100 job (29542773) reached its 3-hour limit during the 216-atom two-centre set-up (about 1.6 h at that size), so the
  measured A100 sizes end at 128 atoms (35.5 s per iteration, 40x SIESTA on 18 ranks, 7.9x jsiesta on 18 CPU cores).

## 4. Routines involved

### 4.1 SIESTA (Src/)

| Stage | Routines | Timer | Algorithm |
|---|---|---|---|
| One-off matrix elements | `setup_H0`: `overfsm` (S), `kinefsm` (T), `nlefsm` (KB), `naefs` + `dnaefs` (neutral-atom pair) | `Setup_H0` | tabulated two-centre integrals, interpolated per neighbour pair from a neighbour list (linear in N) |
| Density from DM | `rhoofd` (inside `dhscf`) | `rhoofd` | per mesh point, evaluates the orbitals active at that point once, contracts with the DM block, on a mesh distributed over the ranks |
| Hartree | `poison` / `fft` | `POISON`, `fft` | FFT Poisson solve |
| XC | `cellxc` (libgridxc) | `XC`, `GXC-CellXC` | pointwise functional on the mesh |
| Potential matrix | `vmat` | `vmat` | same active-orbital structure as `rhoofd`, adjoint operation |
| (sum of the four) | `dhscf` -> `DHSCF3` | `setup_H` | |
| Diagonalisation | `compute_dm` -> `diagon` -> `rdiag` (Gamma) / `cdiag` (k points) -> ScaLAPACK (here), ELPA available | `diagon` | dense generalised eigenproblem, O(N^3), distributed |
| Forces | `dfscf` and the `PostSCF` step | `PostSCF` | one extra pass over the mesh with the derivative of the orbitals |

### 4.2 jsiesta (jax_siesta/, cpp/)

| Stage | Routine | File | Where it runs |
|---|---|---|---|
| Basis and neighbour pattern | `PSMLBasisLoader.build_basis`, `build_sparse_pattern` | `species/psml.py` | host (Python/NumPy) |
| S and T integrals | `TwoCenterEvaluator._eval_st_subset` -> C++ Fourier-Bessel engine | `integrals/two_center.py`, `cpp/src/two_center.cpp` | host, compiled; 1.75 s at 16 atoms |
| KB projectors and neutral-atom pair term | `_eval_kb_and_na_pair`, `_build_kb_pattern_index` | `integrals/two_center.py:563` ff. | **host, Python loops** |
| Density from DM | `rhoofd_differentiable` -> XLA FFI | `grid/mesh.py` -> `cpp/src/grid_ffi.cpp` (CPU), `cpp/cuda/rhoofd_vmat.cu` (GPU) | CPU or GPU |
| Hartree, XC | `GridOps.hartree_potential` (FFT), `LibxcFFIEvaluator.evaluate` | `grid/mesh.py`, `xc/functional.py`, `cpp/src/xc_ffi.cpp` | JAX / CPU / GPU |
| Potential matrix | `vmat_differentiable` -> XLA FFI | same as `rhoofd` | CPU or GPU |
| Diagonalisation | `GeneralizedEigenSolver.solve` (Cholesky + `eigh`) | `scf/eigensolver.py` | JAX (cuSOLVER on GPU, XLA CPU kernels otherwise) |
| Mixing | `_mixer_step_jit` and relatives | `scf/mixer.py` | JAX |
| Forces + stress | `StaticObservableEvaluator.evaluate` (`jax.grad` through the custom-VJP grid kernels, Python two-centre derivatives) | `observables/forces.py` | CPU/GPU + host |

## 5. Diagnosis

### 5.1 B1: the CUDA `rhoofd`/`vmat` kernels (dominant cost of an iteration)

**Measured.** 1.65 s + 1.37 s on a 46 656-point grid at 8 atoms; 0.3-0.4 s per atom per iteration, linear in N; 13-16x
slower than the CPU implementation of the same operation on 18 cores (CPU iteration 0.19 s at 8 atoms, A100 3.07 s);
Hartree + XC and the eigensolver are negligible in comparison.

**Read from code.** The two implementations are different algorithms.
- The CPU kernel (`cpp/src/grid_ffi.cpp`, header comment, steps 1-3) builds, for every grid point, a compact list of
  the orbitals active there (each orbital value and gradient is evaluated *once* per point) and then contracts the active
  pairs against the sparse density matrix. The header documents O(N_grid) complexity; the pair contraction at each point is quadratic in the number of active orbitals, with cheap operands.
- The CUDA kernel `edge_orbital_grid_kernel` (`cpp/cuda/rhoofd_vmat.cu`, launched with `<<<ceil(n_edges/128), 128>>>`)
  has **one thread per orbital-pair edge** `(mu, nu, image)`. Each thread loops over the bounding box of its pair,
  tests the two cut-off spheres at every point, and at every surviving point **re-evaluates both orbitals**
  (`dev_real_solid_harmonics` and `dev_eval_spline` for each of the two orbitals; in the backward mode also gradients
  and Hessians) and does a **global `atomicAdd`** into the density grid.

**Consequences (inferred from the code and the timings, not from a GPU profiler).**
1. The number of orbital evaluations is the number of (edge, point) pairs, about n_active times larger than the number
   of (point, orbital) pairs the CPU design needs. At 16 atoms there are 397 488 edges (about 25 000 per atom, measured
   from the pattern), each scanning a box of thousands of grid points.
2. Threads of a warp handle unrelated edges: divergent loop bounds and cut-off tests, uncoalesced access to the grid.
3. All threads scatter with `atomicAdd` into one array of N_grid doubles, so the throughput is limited by memory
   atomics, not by the A100's 9.7 TFLOP/s of double precision.
4. The thread count equals the number of edges, so the kernel is tail-bound by its slowest (largest-box) edges.
The linear growth with N is consistent with this (work per edge is constant, the number of edges grows with N).

**What it is not.** Not the grid size (Hartree+XC is 0.3 ms), not the transfer of data (the transfer audit in
`reports/phase3_profiling_a100.json` shows no full-array host transfers per iteration), not the eigensolver.

**Validation gap.** The CUDA kernels were validated against the CPU ones for correctness (`tests/test_phase3_cuda.py`),
not for performance; the earlier A100 profile used 2-3-atom cells (about 0.3 s per iteration), where this cost is hidden.

### 5.2 B2: the two-centre set-up (dominant cost of the workflow at 64 atoms and above)

**Measured and profiled** (cProfile of `TwoCenterEvaluator.evaluate` at 16 atoms, 397 488 edges, 208 orbitals, 288 KB
orbitals): 99.6 s in total, of which 97.7 s in `_eval_kb_and_na_pair` (35 s in the function body, 58 s in 16.9 million
calls of `numpy.linalg.norm`, 11 s in `ndarray.dot`), against 1.75 s for the compiled S/T engine
(`_eval_st_subset`). The time grows as N^2 (331 s to 1309 s between 64 and 128 atoms).

**Read from code** (`integrals/two_center.py`):
- KB projector links (lines 633-665): triple Python loop over KB projector shells x periodic shifts x **all orbitals**
  (`for mu in range(basis.n_orb)`), with one `np.linalg.norm` per iteration. O(N_atoms x N_orb) distance checks per shift,
  in pure Python.
- Neutral-atom pair term (lines 786-812): `for ia in range(n_atoms): for ja in range(n_atoms): for shift in images`,
  again with a Python-level norm per candidate. O(N^2 x n_shift).
- SIESTA does the same physics with a neighbour list (linear in N): 0.47 s at 64 atoms.

This part runs on the host in every configuration, so it is the same on the CPU and GPU paths.

### 5.3 B3: forces and stress

**Measured.** 1.5-1.8 s per atom on the A100 (14 s at 8 atoms, 100 s at 64 atoms), 94x slower than SIESTA on 18 ranks
and 350x slower than on 192 ranks at 64 atoms. **Not profiled separately.** Expected causes (estimate): the backward kernels of B1 (Modes 2 and 5 of the same
CUDA kernel, which evaluate gradients and Hessians per (edge, point)), plus the host-side derivative terms of B2.

### 5.4 B4: the CPU eigensolver (only matters when no GPU is used)

**Measured.** On 18 CPU cores jsiesta's dense solve takes 2.51 s at 1664 orbitals against 0.36 s for SIESTA's
ScaLAPACK `diagon` (7x). Likely cause (estimate): the XLA CPU `eigh` is not an optimised, parallel
divide-and-conquer implementation. On the A100 this is irrelevant: the cuSOLVER path is the fastest component.

### 5.5 Where the GPU pays

- **Diagonalisation: from about 400 orbitals** (about 30 Si atoms in DZP): 5-15x against SIESTA on 18 ranks, and 2.8-18x
  against a full 192-core node (measured, Section 3.3); the margin over a full node is smallest, about 3x, at the
  largest sizes.
- **Full SCF iteration: nowhere in the range tested**, because of B1.
- **What the numbers allow (estimate, arithmetic on measured components, not a measurement).** If the grid stage cost
  what SIESTA's does and the eigensolver were the measured A100 one:
  - 512 atoms: 0.76 s + 2.1 s = 2.9 s per iteration against 14.0 s (SIESTA 18 ranks) and 2.7 s (SIESTA 192 ranks):
    5x faster than the same CPU share, on par with a full node;
  - 1000 atoms: 4.4 s + 0.5 s (SIESTA 192 ranks' `setup_H`) = 4.9 s against 13.2 s on a full node: about 2.7x.
  This is the upside that fixing B1 would unlock; it is a bound, not a promise.

## 6. Recommended work, in order of payoff

| | Item | Expected effect | Effort |
|---|---|---|---|
| P1 | Rewrite the CUDA `rhoofd`/`vmat` (and their backward modes) after the CPU design: tile the grid, evaluate each active orbital once per grid point into shared memory or a staging buffer, contract with per-atom-pair DM blocks as small batched matrix products, reduce within a tile instead of `atomicAdd` on the global grid | the grid stage from 0.3-0.4 s per atom to what the CPU path reaches or better (the CPU path itself is 13-16x faster than the current GPU kernel); also speeds up forces and stress | large; the CPU kernel is the specification, `tests/test_phase3_cuda.py` the correctness gate |
| P2 | Replace the Python loops of `_eval_kb_and_na_pair` by a cell-list / neighbour-list build, vectorised with NumPy or moved into the C++ engine that already computes S and T | 400 s to well under 10 s at 64 atoms (SIESTA: 0.5 s); removes the N^2 growth; benefits CPU and GPU | small to medium; correctness gate `test_phase1_species_and_two_center.py`, `test_na_pair_exact.py` |
| P3 | Profile forces and stress with a proper GPU profiler (Nsight Systems) after P1; check `build_sparse_pattern` growth | confirms or corrects B3 | small |
| P4 | Re-measure the whole study after P1 and P2: A100 against SIESTA with the 512- and 1000-atom jsiesta points | turns Section 5.5 from an estimate into a measurement | small (scripts and jobs exist) |
| P5 | Faster CPU eigensolver (optional) | only for CPU-only runs | medium |

P2 should be done first despite its lower payoff in the GPU iteration: it is cheap, it makes the larger study points
reachable, and it is a prerequisite for measuring anything at 512 atoms and above.

## 7. Caveats

- One element, one basis, one functional, Gamma point, nspin = 1. A metal with k points, spin or a different basis size
  changes the balance between grid and diagonalisation work.
- The SIESTA binary is a development-branch Release build with the default ScaLAPACK solver; ELPA and other solver
  settings were not tried, and SIESTA timings could be better with them. The 18-core jobs ran on shared nodes.
- jsiesta timings are slopes over a few iterations after warm-up (JIT excluded); SIESTA's are per-call averages from
  its own timers, including its first iteration.
- The kernel breakdown exists for 8, 32 and 64 atoms; B1's scaling beyond that is extrapolated from the linear trend
  of the whole iteration (measured up to the largest A100 size run).
- The dense-eigensolver benchmark uses random matrices; real H and S have the same cost for a dense solver.
- The cluster ran jax 0.12.1, the local pin is 0.4.38; the cluster stack was used for all numbers here.

## 8. Reproduction

```
python scripts/scaling_cases.py <dir>                       # Si supercell case.json files
sbatch scripts/snellius_scaling_gpu.sbatch                  # jsiesta on one A100 (jobs 29542773)
sbatch --ntasks=192 --exclusive --export=ALL,MODE=siesta192 scripts/snellius_scaling_cpu.sbatch   # 29542774
sbatch --ntasks=18  --export=ALL,MODE=siesta18  scripts/snellius_scaling_cpu.sbatch                # 29542779
sbatch --ntasks=1 --cpus-per-task=18 --export=ALL,MODE=jscpu18 scripts/snellius_scaling_cpu.sbatch # 29543065
sbatch scripts/snellius_scaling_diag.sbatch                 # per-kernel breakdown on the A100 (29543077)
sbatch scripts/snellius_scaling_eigh.sbatch                 # dense eigensolver only (29543711)
python scripts/plot_scaling.py                              # reports/scaling/scaling.png
```

The two-centre profile of Section 5.2 is a `cProfile` run of `TwoCenterEvaluator.evaluate` on the 16-atom case on the
development laptop while other jobs were running (99.6 s there, against 55-65 s measured on the cluster at that size;
the split between the Python function and the compiled engine is what matters).

## 9. Status on branch `perf/bottlenecks` (2026-11-21)

Work started from `bottleneck_assessment_plan.md`. Every change was compared with the code before it on the manifest cases
(bitwise identical, or differences at rounding level: at most 4e-14 Ha in energies, 3e-13 Ha/Bohr in forces); the decisions
record the evidence.

| Item | Decision | Change | Measured effect (laptop CPU unless noted) |
|---|---|---|---|
| B2 / P2 | D-031 | vectorised neighbour searches in `build_sparse_pattern` and `_eval_kb_and_na_pair` | pattern + two-centre 26.5 s -> 3.9 s at 8 atoms, 8.6 s at 64, 15.3 s at 128 (Snellius before: 331 s and 1309 s) |
| B4 (part) | D-032 | overlap transform once per SCF; real arithmetic when every 2k is integer (nspin 1, 2) | one Gamma solve at 1664 orbitals 25.0 s -> 0.86 s |
| B4 (main cause) | D-033 | Bloch assembly by gathers instead of an XLA CPU scatter-add | 2.37 s -> 0.4 ms per assembly at 16 atoms; whole solve 4.78 s -> 5 ms |
| B3 on the CPU | D-034 | `dfscf` backward on the cached atom-block path, threaded; `OMP_NUM_THREADS` honoured up to the core count | forces + stress at 16 atoms 34.2 s -> 3.8 s |
| (regression) | D-035 | the symmetrised-vs-unsymmetrised cross-check compared an 18^3 with a 20^3 mesh since D-027; pinned to one mesh | test passes; no production change |
| B1 / P1 | D-036 | CUDA `rhoofd`/`vmat` on the cached host plan, warp per grid point; legacy kernel as fallback | A100 (jobs 29555653, 29559809): CUDA parity tests pass; at 64 atoms `rhoofd` 9.44 s -> 0.047 s, `vmat` 8.20 s -> 0.068 s, iteration 17.7 s -> 0.117 s (same job, legacy vs plan) |

Corrections to the diagnosis above:
- B1: the global `atomicAdd` is not the main cost (the old `vmat` has no atomics and costs 83-87 % of `rhoofd`); the CPU kernel
  is fast because it keeps the orbital values per grid point between calls, as SIESTA does (`DirectPhi .false.`), and the old
  CUDA kernel recomputes them for every (edge, point) with large per-thread arrays (F2, F3 of the plan).
- B4: most of the CPU "eigensolver" time was the dense assembly (a scatter-add with repeated indices, about 6 us per edge on
  XLA CPU), not XLA's `eigh`.
- B3: on the CPU the forces were slow because the backward kernel was single-threaded and rebuilt its cache on every call.

Laptop CPU after D-031 to D-034, 64 atoms: one SCF iteration 2.3 s, of which `rhoofd` 0.82 s, `vmat` 0.77 s, solve 0.26 s.
### 9.1 A100 after D-031 to D-036 (Snellius jobs 29555653 and 29559809, measured)

Same cases and method as Section 3; raw data `reports/perf/jsiesta_a100_perf_*.json` (8 and 64 atoms from the second job,
`reports/perf/job29559809/`). SIESTA columns are the scaling-study
numbers of Section 3.1 (shared nodes, `DirectPhi .false.`, ScaLAPACK; assessment A6 still open).

| atoms | orbitals | A100 before | A100 after | of which solve | SIESTA 18 ranks | SIESTA 192 ranks | set-up (two-centre) before -> after | forces + stress before -> after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 104 | 3.07 | 0.0128 | 0.0025 | 0.042 | (fails) | 28 -> 4.5 s | 14 -> 10 s |
| 16 | 208 | 5.18 | 0.051 | 0.004 | 0.080 | 0.077 | 65 -> 5.1 s | 26 -> 20 s |
| 32 | 416 | 9.45 | 0.072 | 0.006 | 0.155 | 0.130 | 160 -> 6.2 s | 51 -> 38 s |
| 64 | 832 | 18.15 | 0.117 | 0.013 | 0.357 | 0.262 | 400 -> 8.7 s | 100 -> 75 s |
| 128 | 1664 | 35.47 | 0.293 | 0.039 | 0.887 | 0.407 | 1583 -> 13 s | 199 -> 150 s |
| 216 | 2808 | (not reached) | 0.487 | 0.128 | 1.96 | 0.834 | -> 20 s | -> 204 s |
| 512 | 6656 | | 1.87 | 0.97 | 13.98 | 2.72 | -> 44 s | (not run) |

- The A100 iteration is 100-240x faster than before and now faster than SIESTA at every size measured: 1.6x to 7.5x
  against 18 ranks (7.5x at 512 atoms), 1.4-2.2x against a full 192-rank node. At 512 atoms half the iteration is the
  dense solve. The 16- and 32-atom points (first job) sit below the trend of 8 and 64 atoms (second job); the jobs ran on
  shared nodes, so part of the spread is noise.
- The in-SCF solve is now close to the plain `eigh` benchmark (0.039 s against 0.033 s at 1664 orbitals; it was 0.178 s).
- **Forces + stress are now the GPU bottleneck** (150 s at 128 atoms against an iteration of 0.3 s): the GPU backward
  still uses the edge kernel (B3 on the GPU).

**Kernel breakdown on the A100 (seconds, best of 3; job 29559809).** The 64-atom A/B ran both paths in the same job
(`JSIESTA_CUDA_GRID_LEGACY=1` for the old kernel); the 8-atom "before" column is the scaling study (`diag_a100_si008.json`).

| atoms | path | `rhoofd` | Hartree + XC | `vmat` | dense solve | iteration |
|---:|---|---:|---:|---:|---:|---:|
| 8 | old edge kernel | 1.65 | 0.0003 | 1.37 | 0.0079 | 3.06 |
| 8 | plan (D-036) | 0.0067 | 0.0003 | 0.0106 | 0.0025 | 0.0128 |
| 64 | old edge kernel | 9.44 | 0.0006 | 8.20 | 0.0135 | 17.68 |
| 64 | plan (D-036) | 0.047 | 0.0005 | 0.068 | 0.013 | 0.117 |

- `rhoofd` is 200-250x and `vmat` 120-130x faster. This confirms F2: the cost was re-evaluating the orbitals for every
  (edge, point); with the stored values the GPU contracts like the CPU path and SIESTA do.
- `vmat` is now the most expensive kernel (58 % of the 64-atom iteration): it is the only one that uses atomics
  (`atomicAdd` on the block entries). That makes atomics (B1 consequence 3, H1e) a live question again for the new
  kernel, though not for the old one.
- Hartree + XC (0.5 ms) and the solve (13 ms at 832 orbitals) are small up to 128 atoms.

Open: port the GPU backward (forces) to the plan (forces + stress are now 75 s at 64 atoms against an iteration of
0.12 s); `vmat` without atomics (for example a per-block reduction over points); re-measure SIESTA with `--exclusive`,
`DirectPhi .true.` and ELPA (assessment A6).

### 9.2 GPU forces and the meta-GGA (branch `perf/gpu-forces`, Snellius jobs 29586936 and 29589348, measured)

One A100, revision `bf77aa8` (D-037 in, D-038 not yet), library built with Libxc; PBE and r2SCAN (timing only: PBE
pseudopotential) on the Si supercells. Raw data: `reports/forces_prof/job29586936/`. All 12 CUDA tests pass, including the
Libxc meta-GGA end-to-end parity test, which ran for the first time.

**Forces + stress (seconds), by stage of `StaticObservableEvaluator.evaluate` (`timings_out`):**

| atoms | functional | total | grid backward | of which `rhoofd` bwd | of which `tauofd` bwd | KB (host) | other |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | PBE | 10.4 | 9.3 | 9.3 | | 0.87 | 0.3 |
| 16 | PBE | 19.7 | 17.6 | 17.5 | | 1.72 | 0.4 |
| 64 | PBE | 76.1 | 68.0 | 67.7 | | 7.2 | 0.8 |
| 8 | r2SCAN | 35.0 | 33.8 | 9.3 | 24.4 | 0.86 | 0.4 |
| 16 | r2SCAN | 66.4 | 64.2 | 17.2 | 46.7 | 1.72 | 0.5 |
| 64 | r2SCAN | 265.5 | 257.2 | 67.7 | 189.4 | 7.0 | 1.2 |

- The backward kernels (edge kernel, Modes 2 and 5) are 89-97 % of the force time; the meta-GGA backward costs 2.8x the
  density backward. Both are ported to the plan in D-038 (measured below).
- The KB projector forces (`_kb_forces_and_stress`, Python on the host) grow linearly, about 0.11 s per atom, and will be
  the largest remaining force stage after D-038.

**Meta-GGA SCF kernels (D-037), same job, 16 atoms, legacy edge kernel vs plan:** `tauofd` 21.1 s -> 0.015 s, `vmat_tau`
17.5 s -> 0.021 s; one r2SCAN iteration 43.2 s -> 0.132 s. At 64 atoms an r2SCAN iteration takes 0.473 s (PBE 0.220 s).

**Libxc XC on the host:** with Libxc compiled in, `XcEvalCudaImpl` and the meta-GGA handler copy the grid to the host and
call Libxc there, also for PBE, which otherwise runs on the device (`xc_eval_cuda_kernel`). Hartree + XC at 64 atoms:
0.094 s (PBE) and 0.206 s (r2SCAN), against 0.0005 s for PBE without Libxc (job 29559809). It is now the largest single
step of the r2SCAN iteration (44 %) and doubles the PBE iteration (0.117 s -> 0.220 s).

**After D-038 (Snellius job 29589348, revision `81890c1`, same set-up):** all 12 CUDA tests pass, including the GPU vs
CPU backward parity for PBE and meta-GGA. Raw data: `reports/forces_prof/job29589348/`.

| atoms | functional | total | grid backward | `rhoofd` bwd | `tauofd` bwd | KB (host) | other |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | PBE | 1.16 (10.4) | 0.047 | 0.015 | | 0.86 | 0.25 |
| 16 | PBE | 2.09 (19.7) | 0.060 | 0.028 | | 1.71 | 0.32 |
| 64 | PBE | 7.98 (76.1) | 0.14 | 0.11 | | 7.05 | 0.79 |
| 8 | r2SCAN | 1.31 (35.0) | 0.13 | 0.015 | 0.027 | 0.85 | 0.33 |
| 16 | r2SCAN | 2.44 (66.4) | 0.23 | 0.028 | 0.053 | 1.75 | 0.46 |
| 64 | r2SCAN | 9.05 (265.5) | 0.76 | 0.11 | 0.21 | 6.98 | 1.31 |

(Seconds; in brackets the total before D-038. "Other" is the fresh grid, the grid forward pass, the two-centre terms and
dispersion.)

- Forces + stress at 64 atoms are 9.5x (PBE) and 29x (r2SCAN) faster. Kernel A/B at 16 atoms in the same job: `rhoofd`
  backward 17.5 s -> 0.028 s (620x), `tauofd` backward 47.7 s -> 0.053 s (900x).
- The KB projector forces on the host are now 88 % (PBE) and 77 % (r2SCAN) of the force time at 64 atoms. Next in line:
  the grid forward + backward of r2SCAN (1.2 s), then the two-centre terms (0.33 s).
- SCF iteration at 64 atoms in this job: 0.200 s (PBE), 0.459 s (r2SCAN), of which Hartree + XC 0.095 s and 0.205 s
  (host Libxc).

**D-039, LDA/GGA XC on the device (A100-validated, job 29590887):** the device kernel now uses Libxc's parameters
(`PW_MOD` in PBE correlation) and thresholds and matches Libxc point by point (energy density 1e-14 relative, CPU test
`test_builtin_xc_kernel_matches_libxc`; GPU test `test_cuda_builtin_xc_pointwise_matches_cpu`, 20 of 20 CUDA tests
pass); `XcEvalCudaImpl` uses it for LDA, PBE and PBEsol also when Libxc is compiled in. Measured at 64 atoms (PBE), A/B
in the same job: Hartree + XC 0.096 s (host Libxc) -> 0.0006 s, SCF iteration 0.199 s -> 0.103 s.

**D-041, KB projector terms in C++ (implemented, A100 pending):** the KB forces were a NumPy loop over the 384 KB shell
instances of the 64-atom cell (about 24 M orbital pairs, spent in fancy-index temporaries, `np.add.at` and
`searchsorted`). The threaded kernel `jsiesta_kb_three_center` computes V_KB, the energy, forces and stress with the
same pairs: 7.3 s -> 0.30 s at 64 atoms on the laptop (8 threads), results within 1e-14 of the NumPy path. Expected on
the A100 host: forces + stress at 64 atoms about 8.0 s -> 1.3 s, against SIESTA's 1.06 s (18 ranks).

**D-041 on the A100 (job 29594601):** KB stage at 64 atoms 7.54 s -> 0.24 s; forces + stress 0.31-2.62 s from 8 to 216
atoms, 0.96 s at 64 (SIESTA PostSCF: 1.06 s on 18 ranks, 0.29 s on 192). At 512 atoms the gradient plan did not fit next
to the cached value plan and the backward fell back to the edge kernel (540 s); D-042 reuses, evicts and retries, with a
warning: job 29607477 gives 10.1 s at 512 atoms (SIESTA PostSCF 15.8 s on 18 ranks, 1.5 s on 192), and 6.5 s once
the device cache key ignores the plan flags (job 29607988).

**D-043, plans that do not fit (A100 jobs 29607988, 29608548):** a plan larger than the free device memory streams from
the host in chunks of grid points (about 5 GB/s from pageable memory); the host plan no longer keeps per-entry records
or superseded variants. r2SCAN forces + stress at 512 atoms: 20.1 s with the 31.5 GB Hessian plan streamed in 4 chunks
(before: host out of memory). D-044 (job 29610921): page-locked host plan, two buffer sets and copies overlapped
with kernels: r2SCAN 512-atom forces 14.8 s; PBE forced to stream 8.25 -> 7.01 s (forces), 2.50 -> 2.39 s (iteration,
copy-bound). `int32` value offsets cap a plan at 2^31 orbital values (about 2800 Si atoms).

Next: thread or offload the meta-GGA Libxc call (still on the host, 0.2 s of the 0.46 s
r2SCAN iteration at 64 atoms); the remaining two-centre set-up (5.9 s at 64 atoms on the laptop, mostly the overlap
integrals).

