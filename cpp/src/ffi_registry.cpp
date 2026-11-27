/**
 * @file ffi_registry.cpp
 * @brief Core XLA FFI registry and Phase 0 ABI verification kernel.
 *
 * @par Theory
 * Implements ABI version queries and the \c jsiesta_ffi_ping XLA FFI kernel
 * validating zero-copy buffer marshaling between JAX and C++.
 *
 * @par Algorithm
 * Uses \c xla::ffi::Ffi::Bind() with typed \c Buffer<F64> arguments and returns
 * \c Error::Success() upon completion.
 *
 * @par Invariants
 * Pure function of input buffers; no static mutable state.
 *
 * @par SIESTA Mapping
 * Establishes the C++/XLA FFI bridge used by \c rhoofd, \c vmat, and \c xc_eval.
 */

#include "jsiesta/ffi_kernels.hpp"
#include "jsiesta/libxc_compat.hpp"

#ifdef JSIESTA_HAS_LIBXC
#include <xc.h>
#endif
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

ffi::Error PingImpl(ffi::Buffer<ffi::DataType::F64> x,
                    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out) {
    const std::size_t n = x.element_count();
    if (out->element_count() != n) {
        return ffi::Error::InvalidArgument("Output buffer length must match input buffer length.");
    }
    const double* in_ptr = x.typed_data();
    double* out_ptr = out->typed_data();
    for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = 2.0 * in_ptr[i] + 1.0;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PingHandler,
    PingImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

}  // namespace

extern "C" {

int32_t jsiesta_ffi_version() {
    return 300;
}

int32_t jsiesta_has_libxc() {
#ifdef JSIESTA_HAS_LIBXC
    return 1;
#else
    return 0;
#endif
}

int32_t jsiesta_libxc_info(int32_t id, int32_t* family, int32_t* is_hybrid, int32_t* is_vv10) {
#ifdef JSIESTA_HAS_LIBXC
    xc_func_type func;
    if (xc_func_init(&func, static_cast<int>(id), XC_UNPOLARIZED) != 0) {
        return 2;
    }
    const int fam = xc_func_info_get_family(func.info);
    *family = static_cast<int32_t>(jsiesta::libxc_semilocal_family(fam));
    *is_hybrid = jsiesta::libxc_is_hybrid(&func) ? 1 : 0;
    *is_vv10 = (xc_func_info_get_flags(func.info) & XC_FLAGS_VV10) ? 1 : 0;
    xc_func_end(&func);
    return 0;
#else
    (void)id;
    (void)family;
    (void)is_hybrid;
    (void)is_vv10;
    return 1;
#endif
}

int32_t jsiesta_libxc_version(int32_t* major, int32_t* minor, int32_t* micro) {
#ifdef JSIESTA_HAS_LIBXC
    int a = 0, b = 0, c = 0;
    xc_version(&a, &b, &c);
    *major = a;
    *minor = b;
    *micro = c;
    return 0;
#else
    (void)major;
    (void)minor;
    (void)micro;
    return 1;
#endif
}

int32_t jsiesta_libxc_hybrid_coef(int32_t id, double* alpha, double* beta, double* omega) {
#ifdef JSIESTA_HAS_LIBXC
    xc_func_type func;
    if (xc_func_init(&func, static_cast<int>(id), XC_UNPOLARIZED) != 0) {
        return 2;
    }
    *alpha = 0.0;
    *beta = 0.0;
    *omega = 0.0;
    int rc = 0;
    if (jsiesta::libxc_is_hybrid(&func)) {
        const int flags = xc_func_info_get_flags(func.info);
        if (flags & XC_FLAGS_HYB_CAMY) {
            rc = 3;  // Yukawa range separation: no integral kernel for it
        } else if (flags & XC_FLAGS_HYB_CAM) {
            xc_hyb_cam_coef(&func, omega, alpha, beta);
        } else {
            *alpha = xc_hyb_exx_coef(&func);
        }
    }
    xc_func_end(&func);
    return rc;
#else
    (void)id;
    (void)alpha;
    (void)beta;
    (void)omega;
    return 1;
#endif
}

int32_t jsiesta_has_cuda() {
#ifdef JSIESTA_HAS_CUDA
    return 1;
#else
    return 0;
#endif
}

XLA_FFI_Error* jsiesta_ffi_ping(XLA_FFI_CallFrame* call) {
    return PingHandler(call);
}

}  // extern "C"
