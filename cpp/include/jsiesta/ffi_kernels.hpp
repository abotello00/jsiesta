/**
 * @file ffi_kernels.hpp
 * @brief XLA/JAX FFI C ABI entrypoints for jsiesta C++ and CUDA kernels.
 *
 * @par Theory
 * Exposes typed XLA Foreign Function Interface (FFI) handlers for sparse orbital-grid
 * density scatter (\c rhoofd), potential gather (\c vmat), custom VJP backward passes
 * (\c dfscf), two-center Fourier-Bessel integrals (\c two_center_eval), and pointwise
 * exchange-correlation evaluation (\c xc_eval).
 *
 * @par Algorithm
 * Each symbol follows XLA's \c XLA_FFI_Handler signature and is exported with C linkage
 * so \c ctypes and \c jax.ffi.register_ffi_target bind directly to \c libjsiesta_ffi.
 *
 * @par Invariants
 * - All handlers return \c XLA_FFI_Error* (nullptr on success).
 * - Handlers perform zero hidden global mutable state updates.
 *
 * @par SIESTA Mapping
 * Provides the FFI ABI boundary replacing \c rhoofd.F90, \c vmat.F90, \c dfscf.F,
 * \c matel.F90, and \c libgridxc/cellxc.F.
 */

#pragma once

#include <cstdint>
#include "xla/ffi/api/c_api.h"

extern "C" {

/**
 * @brief Return compiled jsiesta FFI ABI version integer.
 *
 * @par Theory
 * Identifies ABI compatibility and compile-time backend flags (e.g. libxc availability).
 *
 * @par Algorithm
 * Pure constant accessor.
 *
 * @return Integer ABI version (currently 300).
 *
 * @par Invariants
 * Thread-safe, noexcept.
 *
 * @par SIESTA Mapping
 * Replaces compile-time feature queries in \c siesta_version_info.
 */
int32_t jsiesta_ffi_version();

/**
 * @brief Return 1 if compiled against external libxc, 0 if using built-in C++ XC kernels.
 *
 * @par Theory
 * Allows Python callers to record exact XC backend provenance in \c Observables.
 *
 * @par Algorithm
 * Compile-time preprocessor constant check.
 *
 * @return 1 if external libxc is linked, else 0.
 *
 * @par Invariants
 * Thread-safe, noexcept.
 *
 * @par SIESTA Mapping
 * Replaces \c libgridxc version/backend metadata reporting.
 */
int32_t jsiesta_has_libxc();

/**
 * @brief Query Libxc metadata for one functional id.
 *
 * @par Theory
 * The family (LDA/GGA/MGGA), hybrid type and non-local (VV10) flag decide which density
 * ingredients a functional needs and whether jsiesta can evaluate it (no exact exchange, no VV10).
 *
 * @par Algorithm
 * ``xc_func_init`` (unpolarised), ``xc_func_info_get_family``, ``xc_hyb_type``,
 * ``xc_func_info_get_flags``, ``xc_func_end``.
 *
 * @par Invariants
 * Returns 0 on success, 1 when Libxc is not linked, 2 for an unknown id.
 *
 * @par SIESTA Mapping
 * ``libgridxc`` functional classification in ``gridxc_setXC``.
 *
 * @param id Libxc functional id.
 * @param family Output: Libxc ``XC_FAMILY_*`` value.
 * @param is_hybrid Output: 1 if the functional mixes exact exchange.
 * @param is_vv10 Output: 1 if it carries a VV10 non-local correlation.
 */
int32_t jsiesta_libxc_info(int32_t id, int32_t* family, int32_t* is_hybrid, int32_t* is_vv10);

/**
 * @brief Return 1 if compiled with CUDA XLA FFI device kernels enabled, else 0.
 *
 * @par Theory
 * Exposes compile-time GPU support so Python FFI registration conditionally
 * binds CUDA handlers under the \c "CUDA" XLA platform.
 *
 * @par Algorithm
 * Compile-time preprocessor constant check against \c JSIESTA_HAS_CUDA.
 *
 * @return 1 when built with CUDA device handlers, else 0.
 *
 * @par Invariants
 * Thread-safe, noexcept.
 *
 * @par SIESTA Mapping
 * Replaces GPU build feature query in runtime diagnostics.
 */
int32_t jsiesta_has_cuda();

/**
 * @brief XLA FFI health-check handler verifying buffer round-trip through XLA CPU FFI.
 *
 * @par Theory
 * Computes \f$y_i = 2 x_i + 1\f$ to validate XLA FFI registration and ABI alignment.
 *
 * @par Algorithm
 * Element-wise loop over contiguous double buffer via \c xla::ffi::Ffi::Bind().
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Input and output buffers must share identical length.
 *
 * @par SIESTA Mapping
 * Infrastructure sanity kernel for the XLA FFI bridge.
 */
XLA_FFI_Error* jsiesta_ffi_ping(XLA_FFI_CallFrame* call);

}  // extern "C"
