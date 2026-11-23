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
    *family = static_cast<int32_t>(xc_func_info_get_family(func.info));
#if defined(XC_MAJOR_VERSION) && XC_MAJOR_VERSION >= 7
    *is_hybrid = (xc_hyb_type(&func) != XC_HYB_NONE) ? 1 : 0;
#else
    // Libxc 5/6 mark hybrids by family; Libxc 7 replaced the hybrid families by xc_hyb_type().
    const int fam = xc_func_info_get_family(func.info);
    *is_hybrid = (fam == XC_FAMILY_HYB_GGA || fam == XC_FAMILY_HYB_MGGA) ? 1 : 0;
#endif
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
