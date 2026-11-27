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
 * The semilocal family (LDA/GGA/MGGA), the hybrid character and the non-local (VV10) flag decide which density
 * ingredients a functional needs and which further terms (exact exchange, VV10) belong to it.
 *
 * @par Algorithm
 * ``xc_func_init`` (unpolarised), ``xc_func_info_get_family``, ``libxc_is_hybrid``,
 * ``xc_func_info_get_flags``, ``xc_func_end``. The hybrid families of Libxc 5 to 7 (``XC_FAMILY_HYB_LDA``,
 * ``XC_FAMILY_HYB_GGA``, ``XC_FAMILY_HYB_MGGA``) are reported as their semilocal family.
 *
 * @par Invariants
 * Returns 0 on success, 1 when Libxc is not linked, 2 for an unknown id. The same answer for every Libxc
 * release from 5.1 to 7.x: the test is on the features of the header, not on the version number.
 *
 * @par SIESTA Mapping
 * ``libgridxc`` functional classification in ``gridxc_setXC``.
 *
 * @param id Libxc functional id.
 * @param family Output: semilocal Libxc family (1 LDA, 2 GGA, 4 MGGA).
 * @param is_hybrid Output: 1 if the functional mixes exact exchange.
 * @param is_vv10 Output: 1 if it carries a VV10 non-local correlation.
 */
int32_t jsiesta_libxc_info(int32_t id, int32_t* family, int32_t* is_hybrid, int32_t* is_vv10);

/**
 * @brief Version of the Libxc library the kernels were linked against.
 *
 * @par Theory
 * Functional ids, default parameters and thresholds differ between Libxc releases; results are reported with
 * the version that produced them.
 *
 * @par Algorithm
 * ``xc_version``.
 *
 * @par Invariants
 * Returns 0 on success, 1 when Libxc is not linked.
 *
 * @par SIESTA Mapping
 * The Libxc version line of SIESTA's output header.
 *
 * @param major Output: major version.
 * @param minor Output: minor version.
 * @param micro Output: micro version.
 */
int32_t jsiesta_libxc_version(int32_t* major, int32_t* minor, int32_t* micro);

/**
 * @brief Exact-exchange parameters of a Libxc functional.
 *
 * @par Theory
 * A hybrid adds \f$E_x^{HF} = \alpha E_x^{HF,full} + \beta E_x^{HF,SR}(\omega)\f$ to the semilocal part
 * Libxc evaluates, where the short-range kernel is \f$\mathrm{erfc}(\omega r)/r\f$ (Libxc convention:
 * \f$\alpha\f$ is the fraction at long range, \f$\alpha + \beta\f$ at short range). PBE0 has
 * \f$\alpha = 0.25\f$; HSE06 has \f$\alpha = 0\f$, \f$\beta = 0.25\f$, \f$\omega = 0.11\f$ Bohr\f$^{-1}\f$.
 *
 * @par Algorithm
 * ``xc_hyb_cam_coef`` for range-separated hybrids (``XC_FLAGS_HYB_CAM``), ``xc_hyb_exx_coef`` otherwise;
 * zeros for a functional without exact exchange.
 *
 * @par Invariants
 * Returns 0 on success, 1 when Libxc is not linked, 2 for an unknown id, 3 for a Yukawa-screened hybrid
 * (``XC_FLAGS_HYB_CAMY``), which is not supported.
 *
 * @par SIESTA Mapping
 * No counterpart in SIESTA; HONPAS hard-codes the HSE06 values.
 *
 * @param id Libxc functional id.
 * @param alpha Output: fraction of full-range exact exchange.
 * @param beta Output: fraction of short-range exact exchange.
 * @param omega Output: range-separation parameter in Bohr^-1.
 */
int32_t jsiesta_libxc_hybrid_coef(int32_t id, double* alpha, double* beta, double* omega);

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
