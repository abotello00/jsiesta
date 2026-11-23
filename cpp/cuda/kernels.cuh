/**
 * @file kernels.cuh
 * @brief CUDA FFI kernel declarations for GPU-resident rhoofd, vmat, dfscf, neutral-atom, and XC evaluation.
 *
 * @par Theory
 * Mirrors the CPU XLA FFI contracts in \c grid_xc_ffi.hpp for NVIDIA CUDA execution,
 * evaluating orbital-grid density scatter (\c rhoofd), potential gather (\c vmat),
 * orbital-grid and neutral-atom gradients (\c dfscf / \c na_grid_grad), and pointwise
 * semilocal XC (\c xc_eval) directly on device-resident mesh arrays without host round-trips.
 *
 * @par Algorithm
 * Organizes mesh grids with \c GridTileConfig and dispatches stream-ordered asynchronous
 * CUDA kernels via \c xla::ffi::PlatformStream<cudaStream_t>.
 *
 * @par Invariants
 * - Zero host-device synchronization inside SCF iterations.
 * - Bitwise-compatible mathematical definitions with CPU FFI kernels.
 *
 * @par SIESTA Mapping
 * GPU counterpart to \c rhoofd.F90, \c vmat.F90, \c dfscf.F, \c dhscf.F, and \c cellxc.F.
 */

#pragma once

#include "xla/ffi/api/c_api.h"

namespace jsiesta::cuda {

/**
 * @brief Launch specification for bounding-box orbital-grid tiling on CUDA devices.
 *
 * @par Theory
 * Tunes thread-block dimensions for the sparse orbital-to-grid bounding-box loop.
 *
 * @par Algorithm
 * Configurable tile extents \c (tx, ty, tz) tuned during Phase 3 profiling.
 *
 * @par Invariants
 * Positive tile sizes; total threads per block <= 1024.
 *
 * @par SIESTA Mapping
 * Replaces sub-mesh blocking parameters (\c nsm) in legacy \c mesh.F.
 */
struct GridTileConfig {
    int tx = 8;   /**< Grid tile extent along axis 1. */
    int ty = 8;   /**< Grid tile extent along axis 2. */
    int tz = 4;   /**< Grid tile extent along axis 3. */
};

/**
 * @brief Select a bounded-sphere GPU thread-block tile shape for a target mesh.
 *
 * @par Theory
 * Balances active-orbital register pressure against SM warp occupancy on A100/Ampere
 * GPUs by matching tile extents to the periodic grid dimensions while keeping threads
 * per block within 256--512 threads.
 *
 * @par Algorithm
 * Chooses power-of-two/factorized extents clamped by \c (n1, n2, n3) from candidate
 * Phase 3 tuned profiles \c (8,8,4), \c (16,4,4), and \c (4,4,8).
 *
 * @param n1 Mesh points along axis 1.
 * @param n2 Mesh points along axis 2.
 * @param n3 Mesh points along axis 3.
 * @param n_orb_avg Estimated active orbitals per mesh point.
 * @return Selected \c GridTileConfig satisfying \c tx*ty*tz <= 512.
 *
 * @par Invariants
 * Positive tile sizes; total threads per block <= 1024 and warp-multiple when grid permits.
 *
 * @par SIESTA Mapping
 * GPU counterpart to sub-mesh block tuning (\c nsm) in \c mesh.F.
 */
GridTileConfig select_optimal_tile_config(int n1, int n2, int n3, int n_orb_avg);

}  // namespace jsiesta::cuda

extern "C" {

/**
 * @brief CUDA XLA FFI handler scattering sparse density matrix onto a device 3D mesh (\c rhoofd).
 *
 * @par Theory
 * Evaluates \f$\rho_s(\mathbf{r}_p) = \sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T})\phi_\mu(\mathbf{r}_p)\phi_\nu(\mathbf{r}_p - \mathbf{A}\mathbf{T})\f$
 * directly inside CUDA device memory.
 *
 * @par Algorithm
 * Computes spline coefficients asynchronously on stream and accumulates overlapping edge
 * intersection boxes onto the device mesh via native FP64 atomics.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Zero host-device synchronization inside SCF iterations.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c rhoofd.F90.
 */
XLA_FFI_Error* jsiesta_rhoofd_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI handler gathering device grid potential into sparse matrix elements (\c vmat).
 *
 * @par Theory
 * Evaluates \f$V^{\text{grid}}_{s,\mu\nu}(\mathbf{T}) = d\Omega \sum_p V_s(\mathbf{r}_p)\phi_\mu(\mathbf{r}_p)\phi_\nu(\mathbf{r}_p - \mathbf{A}\mathbf{T})\f$
 * in device memory.
 *
 * @par Algorithm
 * Assigns one CUDA thread per sparse edge to integrate the sphere-intersection bounding box
 * without inter-thread atomics.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Discrete adjoint of \c jsiesta_rhoofd_f64_cuda to machine precision.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c vmat.F90.
 */
XLA_FFI_Error* jsiesta_vmat_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI backward VJP handler for orbital-grid forces and cell gradients (\c dfscf).
 *
 * @par Theory
 * Differentiates \f$\langle V | \rho(D) \rangle\f$ wrt density matrix, atomic positions, and
 * lattice matrix directly on CUDA device buffers.
 *
 * @par Algorithm
 * Accumulates per-edge matrix cotangents directly and reduces position/cell derivatives via
 * stream-ordered FP64 hardware atomics.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches central finite differences of GPU \c rhoofd wrt coordinates and cell.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c dfscf.F.
 */
XLA_FFI_Error* jsiesta_dfscf_grad_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI handler assembling neutral-atom density, potential, and NLCC core on GPU.
 *
 * @par Theory
 * Scatters atomic spherical radial tables \f$(\rho^{\text{NA}}, V^{\text{NA}}, \rho^{\text{core}})\f$
 * onto the 3D device mesh.
 *
 * @par Algorithm
 * Assigns one CUDA block per atom striding over its wrapped fractional bounding box on stream.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches \c jsiesta_na_grid_f64 to IEEE-754 precision.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c dhscf.F neutral-atom setup.
 */
XLA_FFI_Error* jsiesta_na_grid_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI backward VJP handler for neutral-atom and NLCC position/cell gradients.
 *
 * @par Theory
 * Accumulates radial derivative contractions \f$\partial \langle w | \text{na\_grid} \rangle / \partial (\mathbf{R}, \mathbf{A})\f$
 * directly on GPU.
 *
 * @par Algorithm
 * Block-level shared-memory reduction per atom minimizing global atomic traffic.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches finite differences of \c jsiesta_na_grid_f64_cuda.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c dfscf.F neutral-atom gradients.
 */
XLA_FFI_Error* jsiesta_na_grid_grad_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI pointwise semilocal exchange-correlation evaluator on GPU.
 *
 * @par Theory
 * Evaluates LDA (PZ81, PW92) and GGA (PBE, PBEsol) energy density and exact functional
 * derivatives wrt spin densities and gradient invariants via 5-variable Dual automatic
 * differentiation on device threads.
 *
 * @par Algorithm
 * Embarrassingly parallel 1D grid-stride CUDA kernel using device Dual arithmetic.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches \c libxc and built-in CPU Dual-AD XC kernels across all supported functionals.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c libgridxc and \c cellxc.F.
 */
XLA_FFI_Error* jsiesta_xc_eval_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI handler scattering kinetic energy density \f$\tau_s\f$ onto a device 3D mesh.
 *
 * @par Theory
 * Evaluates \f$\tau_s(\mathbf{r}_p) = \frac{1}{2}\sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T})\nabla\phi_\mu(\mathbf{r}_p)\cdot\nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})\f$
 * directly inside device memory.
 *
 * @par Algorithm
 * Computes orbital gradients using device splines and real solid harmonic recurrences and
 * accumulates onto the 3D mesh via stream-ordered FP64 atomics.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Bitwise-compatible mathematical definition with CPU \c jsiesta_tauofd_f64.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c rhoofd.F90 meta-GGA \f$\tau\f$ scatter.
 */
XLA_FFI_Error* jsiesta_tauofd_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI handler gathering meta-GGA \f$v_{\tau,s}\f$ into sparse Hamiltonian edges.
 *
 * @par Theory
 * Evaluates \f$H^\tau_{s,\mu\nu}(\mathbf{T}) = \frac{1}{2} d\Omega \sum_p v_{\tau,s}(\mathbf{r}_p)\nabla\phi_\mu(\mathbf{r}_p)\cdot\nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})\f$.
 *
 * @par Algorithm
 * Assigns one CUDA thread per sparse edge to integrate intersecting atomic boxes without atomics.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Discrete adjoint of \c jsiesta_tauofd_f64_cuda to machine precision.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c vmat.F90 meta-GGA potential gather.
 */
XLA_FFI_Error* jsiesta_vmat_tau_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI backward VJP handler for meta-GGA \f$\tau\f$ orbital-Hessian forces and stress.
 *
 * @par Theory
 * Differentiates \f$\langle w | \tau(D, \mathbf{R}, \mathbf{A}) \rangle\f$ via analytic orbital
 * Hessians directly on CUDA device buffers.
 *
 * @par Algorithm
 * Computes \f$\partial_\alpha\partial_\beta \phi_\mu\f$ on device threads and reduces atomic/cell cotangents via FP64 atomics.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches CPU \c jsiesta_tau_dfscf_grad_f64 and central finite differences of \c tauofd.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c m_tau_force_stress.F90.
 */
XLA_FFI_Error* jsiesta_tau_dfscf_grad_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI handler assembling NLCC \f$\tau^{\text{core}}\f$ on GPU.
 *
 * @par Theory
 * Scatters atomic spherical radial \f$\tau^{\text{core}}\f$ splines onto the 3D device mesh.
 *
 * @par Algorithm
 * Assigns one CUDA block per atom striding over its wrapped fractional bounding box on stream.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches CPU \c jsiesta_tau_core_f64 to IEEE-754 precision.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to PSML \c tau_core evaluation inside \c dhscf.F.
 */
XLA_FFI_Error* jsiesta_tau_core_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI backward VJP handler for NLCC \f$\tau^{\text{core}}\f$ position/cell gradients.
 *
 * @par Theory
 * Accumulates radial derivative contractions \f$\partial \langle w | \tau^{\text{core}} \rangle / \partial (\mathbf{R}, \mathbf{A})\f$ on GPU.
 *
 * @par Algorithm
 * Uses shared-memory block reductions per atom before global atomic accumulation.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches central finite differences of \c jsiesta_tau_core_f64_cuda.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c tau_core Pulay gradient terms.
 */
XLA_FFI_Error* jsiesta_tau_core_grad_f64_cuda(XLA_FFI_CallFrame* call);

/**
 * @brief CUDA XLA FFI pointwise meta-GGA Libxc evaluator on GPU device buffers.
 *
 * @par Theory
 * Evaluates semilocal meta-GGA energy density and functional derivatives wrt spin density,
 * gradient invariants, and kinetic energy density via compiled Libxc.
 *
 * @par Algorithm
 * Stages device arrays asynchronously across \c cudaStream_t to invoke shared host \c evaluate_mgga_libxc_host.
 *
 * @param call XLA FFI call frame pointer executing on \c PlatformStream<cudaStream_t>.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Exact bitwise parity with CPU \c jsiesta_xc_eval_mgga_f64 across all Libxc IDs.
 *
 * @par SIESTA Mapping
 * GPU XLA FFI counterpart to \c mggaxc.F90.
 */
XLA_FFI_Error* jsiesta_xc_eval_mgga_f64_cuda(XLA_FFI_CallFrame* call);

}  // extern "C"
