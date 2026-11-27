# Plan: hybrid functionals in jsiesta

Branch `feature/hybrid-functionals`, started 2026-11-24. Status of each stage is kept in the table of Section 6;
decisions go to `logs/DECISIONS.md` as usual.

## 1. Goal and scope

Total energy, Hamiltonian and eigenvalues with a hybrid exchange-correlation functional, first for the screened
hybrid HSE06, with k-points and `nspin` 1 and 2. Then forces and stress, the non-collinear spin modes, and
full-range hybrids (PBE0, B3LYP).

Two prerequisites are part of this plan: the library must build and run with Libxc 7.x (the hybrid parameters are
read from Libxc), and the Libxc route must evaluate the semilocal part of a hybrid.

Out of scope: MPI, a GPU integral engine (Section 7), range separation with a Yukawa kernel, double hybrids.

## 2. Assessment of the HONPAS paper

Shang et al., Comput. Phys. Commun. 254, 107204 (2020) describes two things.

1. **The method (their Section 2, from the earlier NAO2GTO papers).** Each numerical radial function is fitted by a
   few Gaussians. The four-centre electron repulsion integrals (ERIs) are then analytic. The exchange matrix is built
   in real space, in SIESTA's sparse supercell format, with Schwarz, distance and density-matrix screening:

   `K^G_{mu lambda} = sum_{nu sigma} sum_{N H} P^{H-N}_{nu sigma} (mu0 nuN | lambdaG sigmaH)`.

   This fits jsiesta well: `K` has the same shape as every other sparse matrix here, k-points come for free through
   the existing Bloch assembly, and the real-space density matrix already exists. **Adopted.**
2. **The contribution of that paper: a two-level master/worker MPI distribution of shell quartets.** jsiesta has no
   MPI. **Not adopted**; the quartet loop is threaded with a dynamic schedule over bra shell pairs instead.

Points the paper leaves implicit and this plan makes explicit:

- **Truncation.** HONPAS keeps the density matrix and `K` on SIESTA's sparse pattern (orbital pairs that overlap or
  share a projector). The exact `K` and the exact density matrix extend further. This is an approximation of the
  method, controlled by the orbital radii; jsiesta starts with the same convention (so that HONPAS is a meaningful
  reference) and records it. An extended pattern for `K` is a later option (Stage 9).
- **Short range only.** The supercell of the paper is sized by the orbital overlap, which is adequate only for the
  short-range kernel `erfc(omega r)/r`. Full-range exchange in a periodic system needs a treatment of the long-range
  lattice sum; it is deferred (Stage 8) and until then allowed only where it is well defined. Stage 8 (D-058):
  the sum over the density matrix on the pattern is finite and is accepted when the k-point mesh is fine enough
  that the density matrix does not repeat within the pattern; the truncation error was measured (2 meV per cell
  for Si with PBE0). A correction for the tail of the density matrix is not implemented.
- **Two representations.** Everything except `K` uses the numerical orbitals; `K` uses their Gaussian fits. The
  exchange energy is still a well-defined function of the density matrix and of the atomic positions, so
  `dE_x/dP = K` and forces as derivatives hold exactly for the fitted functional. The fit error is a basis-set
  error of the exchange term and is reported per shell.

Alternatives considered: resolution of the identity with local fitting (needs an auxiliary basis and its
validation), and exchange on the real-space grid with one FFT Poisson solve per orbital pair (no fit error and
GPU-friendly, but cubic cost with a large prefactor: of the order of 5e5 FFTs per build at 512 Si atoms). NAO2GTO
is chosen because it is sparse, linear-scaling with screening, and can be checked against two independent codes.

## 3. Design

### 3.1 Functional description (Libxc)

Libxc convention: `E_xc = E_semilocal + alpha E_x^HF(1/r) + beta E_x^HF(erfc(omega r)/r)`. PBE0: `alpha = 0.25`.
HSE06: `alpha = 0`, `beta = 0.25`, `omega = 0.11` Bohr^-1. `jsiesta_libxc_hybrid_coef` returns the three numbers;
the semilocal part is what Libxc evaluates for the hybrid id through the kernel of its semilocal family.
Names `HSE06`, `PBE0`, `B3LYP` map to Libxc ids 428, 406, 402; `LIBXC-<id>` accepts any hybrid id.

### 3.2 Exchange energy and matrix

Per spin channel `s`, with kernel `w(r) = alpha/r + beta erfc(omega r)/r`:

- `K^s_{mu lambda} = sum_{nu sigma} P^s_{nu sigma} (mu nu | lambda sigma)_w`
- `H^s += -K^s`, `E_x^HF = -(1/2) sum_s Tr[P^s K^s]`

For `nspin = 1` the stored density matrix is the sum of both spins: `H += -(1/2) K[P]`, `E = -(1/4) Tr[P K[P]]`.
For `nspin` 4 and 8, `H^{ss'} = -K[P^{ss'}]` for the four complex spin blocks; the Hamiltonian is then complex
also for `nspin = 4` (Stage 8, D-058).

### 3.3 Components

| Component | Where | Notes |
|---|---|---|
| Gaussian fit of the radial functions | `jax_siesta/hfx/nao2gto.py` | `R(r) ~ r^l sum_i c_i exp(-a_i r^2)`; linear coefficients by least squares, exponents by nonlinear optimisation in `log a`; result cached per species; reports the residual norm |
| ERI engine | `cpp/src/eri.cpp`, `cpp/include/jsiesta/eri.hpp` | McMurchie-Davidson, contracted shells of real solid harmonics in jsiesta's `m` order, `l <= 3`; kernels `1/r`, `erfc(omega r)/r` through the Boys function; own code, no new run-time dependency |
| Exchange build | `cpp/src/hfx.cpp` | shell-pair lists from the orbital radii, Schwarz bounds, density screening with the largest `|P|` per shell pair, threads over bra pairs with thread-local accumulators; output on the edges of the `SparsePattern` |
| Stage interface | `ExactExchange` protocol in `core/protocols.py`, implementation in `jax_siesta/hfx/` | `prepare(structure, basis, pattern)` once per geometry; `matrix_and_energy(dm)` per SCF step |
| SCF | `scf/driver.py` | `h_out = h0 + h_grid + h_x`; energy bookkeeping of Section 3.4; symmetry projection applied to `h_x` |
| Forces and stress | `observables/forces.py`, derivative ERIs in the engine | Stage 7 |
| Input | `XC.functional` / `XC.authors` in `io/fdf.py`, `SCFConfig` | plus `HFX.*` tolerances |

### 3.4 Energy bookkeeping

The exchange matrix joins the density-dependent part of the Hamiltonian (`h_grid` in `_iteration_update`):
the Kohn-Sham energy gets `-Tr[D_out H_x,in] + E_x^HF[D_out]` and the Harris energy the same with `D_in`.
`E_x^HF` is reported as its own line next to `E_xc`.

### 3.5 Cost control inside the SCF

Start from a converged semilocal density matrix (PBE). Build `K` once per SCF step at first; then add incremental
builds from the change of the density matrix, which makes density screening far more effective near convergence.

## 4. Validation

The project rule holds: a term is validated when an external reference and an independent identity both pass.

| What | External reference | Identity |
|---|---|---|
| Libxc 7.x | same results as Libxc 6.2.2 for the existing parity tests | metadata identical across 6.2.2, 7.0.0, 7.1.2 |
| Semilocal part of a hybrid | PBE0 remainder equals `0.75 E_x^PBE + E_c^PBE` evaluated from the separate Libxc ids | potential is the derivative of the energy (finite difference) |
| Gaussian fit | none needed | residual norm, overlap of fit and original, normalisation |
| ERIs | PySCF/libcint `int2e_sph`, with and without `omega`, all `l` combinations up to 3 | 8-fold permutation symmetry, positivity of `(ij|ij)`, translation and rotation invariance |
| `K` and `E_x`, molecule | PySCF `get_k` for the same contracted Gaussians and density matrix (isolated molecule in a large cell) | `K` Hermitian, `dE_x/dP = K` by finite difference, one-orbital self-interaction `K = J` |
| `K`, periodic | HONPAS (gated on a binary: total energy and band gap of Si, HSE06, same basis and pseudopotential) | a supercell at Gamma equals the unit cell with the matching k-mesh; invariance under the space group |
| SCF | HONPAS as above; literature gap of Si (HSE06, about 1.1-1.2 eV) as a sanity bound only | Kohn-Sham and Harris energies agree at convergence; energy stationary with respect to the density matrix |
| Forces, stress | HONPAS forces where available | central finite differences of the total energy |

PySCF is a test-only dependency (`requirements-dev.txt`); tests that need it skip when it is absent.

## 5. Risks

- **Cost.** A plain McMurchie-Davidson engine is several times slower than libint or libcint. Mitigation: screening
  first, permutation symmetry second, incremental builds third; the engine sits behind one function so that it can
  be replaced (libcint, or a GPU engine) without touching the rest.
- **Fit quality for confined orbitals.** Numerical orbitals end at a finite radius with a kink in the derivative;
  Gaussians cannot follow it exactly. The residual is measured and its effect on `E_x` is quantified against a
  larger number of Gaussians.
- **Truncation on the sparse pattern** (Section 2). Measured by widening the pattern on a small system.
- **Metals and small gaps.** The density matrix decays slowly; the truncated `K` is unreliable. Documented limit.
- **No HONPAS binary on the development machines.** The gated test is written and recorded as not run until one
  is available.

## 6. Stages

Each stage ends with its tests passing, a decision entry and updated `MEMORY.md`.

| Stage | Content | Exit criterion | Status |
|---|---|---|---|
| 1 | Libxc 5.1 to 7.x: feature-based hybrid detection, `LIBXC_ROOT` honoured by CMake, version query | builds and XC tests pass with 6.2.2, 7.0.0 and 7.1.2 | done (D-055) |
| 2 | Hybrid description: `jsiesta_libxc_hybrid_coef`, semilocal family dispatch, names `HSE06`/`PBE0`/`B3LYP`, `XCFunctionalSpec` with `alpha`, `beta`, `omega`; a hybrid without the exchange stage is an error | PBE0-remainder identity test; non-hybrids unchanged | done (D-056) |
| 3 | NAO to GTO fit | fit tests; residual table for the PseudoDojo H, C, O, Si bases | done; residual 1 to 2.5 %, not converged with the number of Gaussians (D-056) |
| 4 | ERI engine | agreement with PySCF to 1e-10 for all shell types, both kernels | done |
| 5 | Exchange matrix and energy on the sparse pattern, `nspin` 1 and 2, threaded, screened | molecular `K` equals PySCF; identities of Section 4 | done |
| 6 | SCF with HSE06: driver, energies, symmetry, FDF input, report | converged Si and a molecule; supercell/k-mesh identity; HONPAS test written | done; HONPAS comparison in D-059 |
| 7 | Forces and stress: derivative ERIs | finite-difference tests | done (D-057) |
| 8 | `nspin` 4 and 8; full-range hybrids for isolated systems, then periodic | identities; PySCF for molecules | done (D-058); periodic full-range by real-space truncation with a k-mesh criterion, no long-range correction |
| 9 | Performance: permutation symmetry, incremental builds, extended pattern, measurements against HONPAS | measured table in `known_bottlenecks.md` | done: permutation symmetry, shared derivative tables, incremental builds (D-060); extended pattern and distance bound of the screened kernel (D-061) |

## 7. Later, not planned here

A GPU integral engine; resolution of the identity as a second `ExactExchange` implementation; adaptively
compressed exchange to reduce the number of builds.
