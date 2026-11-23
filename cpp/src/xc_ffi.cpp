/**
 * @file xc_ffi.cpp
 * @brief Pointwise exchange-correlation XLA FFI kernel (\c libxc or exact C++ Dual-AD LDA/PBE/PBEsol).
 *
 * @par Theory
 * Evaluates semilocal exchange-correlation energy density \f$e_{\text{xc}} = n\varepsilon_{\text{xc}}\f$
 * and functional derivatives \f$v_{\rho,\sigma} = \partial e_{\text{xc}} / \partial n_\sigma\f$,
 * \f$v_{\gamma,\alpha\beta} = \partial e_{\text{xc}} / \partial \gamma_{\alpha\beta}\f$
 * (where \f$\gamma_{\alpha\beta} = \nabla n_\alpha \cdot \nabla n_\beta\f$) for LDA (Slater + PZ81 / PW92)
 * and GGA (PBE, PBEsol).
 *
 * @par Algorithm
 * When compiled with \c JSIESTA_HAS_LIBXC=1, dispatches directly to \c libxc C API.
 * Otherwise, evaluates the exact analytical textbook functionals using templated
 * forward-mode dual numbers (\c Dual<5>) so derivatives match energy variations to
 * machine precision (\f$\sim 10^{-15}\f$).
 *
 * @par Invariants
 * - Zero NaNs at \f$\rho \to 0\f$ or \f$\nabla\rho \to 0\f$ (density floor \f$10^{-14}\f$).
 * - Thread-safe and stateless per grid point.
 *
 * @par SIESTA Mapping
 * Replaces \c libgridxc and \c cellxc.F pointwise XC evaluation.
 */

#include "jsiesta/grid_xc_ffi.hpp"
#include "jsiesta/xc_semilocal.hpp"
#include "xla/ffi/api/ffi.h"

#ifdef JSIESTA_HAS_LIBXC
#include <xc.h>
#endif

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

namespace ffi = xla::ffi;

namespace {

// Functional IDs matching Python xc module
constexpr int32_t kXcLdaPz = 1;
constexpr int32_t kXcLdaPw92 = 2;
constexpr int32_t kXcGgaPbe = 10;
constexpr int32_t kXcGgaPbesol = 11;


int evaluate_xc_libxc_impl(
    int64_t n_grid,
    const double* r_ptr,
    const double* s_ptr,
    int64_t xc_id,
    int64_t n_spin_pol,
    double* e_ptr,
    double* vr_ptr,
    double* vs_ptr);

ffi::Error XcEvalImpl(
    ffi::Buffer<ffi::DataType::F64> rho_spin,   // (2, n_grid) [rho_up, rho_dn] (for unpol: [rho/2, rho/2])
    ffi::Buffer<ffi::DataType::F64> sigma_spin, // (3, n_grid) [sig_uu, sig_ud, sig_dd]
    int64_t xc_id,
    int64_t n_spin_pol,  // 1 or 2
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_exc_density,  // (n_grid,) n * eps_xc
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_rho,        // (2, n_grid)
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sigma) {    // (3, n_grid)

    const int64_t n_grid = static_cast<int64_t>(out_exc_density->element_count());
    const double* r_ptr = rho_spin.typed_data();
    const double* s_ptr = sigma_spin.typed_data();
    double* e_ptr = out_exc_density->typed_data();
    double* vr_ptr = out_v_rho->typed_data();
    double* vs_ptr = out_v_sigma->typed_data();

#ifdef JSIESTA_HAS_LIBXC
    if (evaluate_xc_libxc_impl(n_grid, r_ptr, s_ptr, xc_id, n_spin_pol, e_ptr, vr_ptr, vs_ptr) == 0) {
        return ffi::Error::Success();
    }
    return ffi::Error::InvalidArgument("Libxc XC evaluation failed in XcEvalImpl.");
#else
    for (int64_t p = 0; p < n_grid; ++p) {
        double vr[2], vs[3];
        jsiesta::xc::eval_semilocal_xc_point(
            static_cast<int32_t>(xc_id), static_cast<int32_t>(n_spin_pol), r_ptr[p], r_ptr[n_grid + p], s_ptr[p], s_ptr[n_grid + p],
            s_ptr[2 * n_grid + p], &e_ptr[p], vr, vs);
        vr_ptr[p] = vr[0];
        vr_ptr[n_grid + p] = vr[1];
        vs_ptr[p] = vs[0];
        vs_ptr[n_grid + p] = vs[1];
        vs_ptr[2 * n_grid + p] = vs[2];
    }
    return ffi::Error::Success();
#endif
}

    XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XcEvalHandler, XcEvalImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // rho_spin (2, n_grid)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sigma_spin (3, n_grid)
        .Attr<int64_t>("xc_id")
        .Attr<int64_t>("n_spin_pol")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // exc_density (n_grid,)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // v_rho (2, n_grid)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // v_sigma (3, n_grid)

ffi::Error XcEvalMggaImpl(
    ffi::Buffer<ffi::DataType::F64> rho_spin,    // (2, n_grid)
    ffi::Buffer<ffi::DataType::F64> sigma_spin,  // (3, n_grid)
    ffi::Buffer<ffi::DataType::F64> tau_spin,    // (2, n_grid)
    int64_t x_id,
    int64_t c_id,
    int64_t n_spin_pol,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_exc_density,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_rho,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sigma,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_tau) {
    const int64_t n_grid = static_cast<int64_t>(out_exc_density->element_count());
    const int rc = jsiesta::evaluate_mgga_libxc_host(
        n_grid,
        rho_spin.typed_data(),
        sigma_spin.typed_data(),
        tau_spin.typed_data(),
        x_id,
        c_id,
        n_spin_pol,
        out_exc_density->typed_data(),
        out_v_rho->typed_data(),
        out_v_sigma->typed_data(),
        out_v_tau->typed_data());
    if (rc == 1) {
        return ffi::Error::InvalidArgument(
            "meta-GGA XC evaluation requires jsiesta compiled with external libxc (JSIESTA_HAS_LIBXC=1).");
    }
    if (rc == 2) {
        return ffi::Error::InvalidArgument("Unknown libxc functional ID in evaluate_mgga_libxc_host.");
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XcEvalMggaHandler, XcEvalMggaImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // rho_spin (2, n_grid)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sigma_spin (3, n_grid)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // tau_spin (2, n_grid)
        .Attr<int64_t>("x_id")
        .Attr<int64_t>("c_id")
        .Attr<int64_t>("n_spin_pol")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // exc_density (n_grid,)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // v_rho (2, n_grid)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // v_sigma (3, n_grid)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // v_tau (2, n_grid)

[[maybe_unused]] int evaluate_xc_libxc_impl(
    int64_t n_grid,
    const double* r_ptr,
    const double* s_ptr,
    int64_t xc_id,
    int64_t n_spin_pol,
    double* e_ptr,
    double* vr_ptr,
    double* vs_ptr) {
#ifndef JSIESTA_HAS_LIBXC
    (void)n_grid; (void)r_ptr; (void)s_ptr;
    (void)xc_id; (void)n_spin_pol;
    (void)e_ptr; (void)vr_ptr; (void)vs_ptr;
    return 1;
#else
    xc_func_type func_x, func_c;
    const int x_id = (xc_id == kXcGgaPbesol) ? 116 : ((xc_id == kXcGgaPbe) ? 101 : 1);
    const int c_id = (xc_id == kXcGgaPbesol) ? 133 : ((xc_id == kXcGgaPbe) ? 130 : (xc_id == kXcLdaPz ? 9 : 12));
    const int spin_mode = (n_spin_pol == 2) ? XC_POLARIZED : XC_UNPOLARIZED;
    const bool is_gga = (xc_id == kXcGgaPbe || xc_id == kXcGgaPbesol);
    xc_func_init(&func_x, x_id, spin_mode);
    xc_func_init(&func_c, c_id, spin_mode);

    const int n_sp = (spin_mode == XC_POLARIZED) ? 2 : 1;
    const int n_sg = (spin_mode == XC_POLARIZED) ? 3 : 1;
    std::vector<double> r_lib(static_cast<size_t>(n_grid) * n_sp, 0.0);
    std::vector<double> s_lib(static_cast<size_t>(n_grid) * n_sg, 0.0);
    for (int64_t p = 0; p < n_grid; ++p) {
        if (n_sp == 1) {
            r_lib[p] = std::max(0.0, r_ptr[p] + r_ptr[n_grid + p]);
            s_lib[p] = std::max(0.0, s_ptr[p] + 2.0 * s_ptr[n_grid + p] + s_ptr[2 * n_grid + p]);
        } else {
            r_lib[2 * p] = std::max(0.0, r_ptr[p]);
            r_lib[2 * p + 1] = std::max(0.0, r_ptr[n_grid + p]);
            s_lib[3 * p] = std::max(0.0, s_ptr[p]);
            s_lib[3 * p + 1] = s_ptr[n_grid + p];
            s_lib[3 * p + 2] = std::max(0.0, s_ptr[2 * n_grid + p]);
        }
    }

    std::vector<double> ex(n_grid, 0.0), ec(n_grid, 0.0);
    std::vector<double> vrx(static_cast<size_t>(n_grid) * n_sp, 0.0), vrc(static_cast<size_t>(n_grid) * n_sp, 0.0);
    std::vector<double> vsx(static_cast<size_t>(n_grid) * n_sg, 0.0), vsc(static_cast<size_t>(n_grid) * n_sg, 0.0);
    const size_t np_sz = static_cast<size_t>(n_grid);
    if (is_gga) {
        xc_gga_exc_vxc(&func_x, np_sz, r_lib.data(), s_lib.data(), ex.data(), vrx.data(), vsx.data());
        xc_gga_exc_vxc(&func_c, np_sz, r_lib.data(), s_lib.data(), ec.data(), vrc.data(), vsc.data());
    } else {
        xc_lda_exc_vxc(&func_x, np_sz, r_lib.data(), ex.data(), vrx.data());
        xc_lda_exc_vxc(&func_c, np_sz, r_lib.data(), ec.data(), vrc.data());
    }
    xc_func_end(&func_x);
    xc_func_end(&func_c);

    for (int64_t p = 0; p < n_grid; ++p) {
        const double n_tot = r_ptr[p] + r_ptr[n_grid + p];
        if (n_tot <= 1e-14) {
            e_ptr[p] = 0.0;
            vr_ptr[p] = vr_ptr[n_grid + p] = 0.0;
            vs_ptr[p] = vs_ptr[n_grid + p] = vs_ptr[2 * n_grid + p] = 0.0;
        } else {
            e_ptr[p] = n_tot * (ex[p] + ec[p]);
            if (n_sp == 1) {
                const double vr = vrx[p] + vrc[p];
                const double vs = vsx[p] + vsc[p];
                vr_ptr[p] = vr;
                vr_ptr[n_grid + p] = vr;
                vs_ptr[p] = vs;
                vs_ptr[n_grid + p] = 2.0 * vs;
                vs_ptr[2 * n_grid + p] = vs;
            } else {
                vr_ptr[p] = vrx[2 * p] + vrc[2 * p];
                vr_ptr[n_grid + p] = vrx[2 * p + 1] + vrc[2 * p + 1];
                vs_ptr[p] = vsx[3 * p] + vsc[3 * p];
                vs_ptr[n_grid + p] = vsx[3 * p + 1] + vsc[3 * p + 1];
                vs_ptr[2 * n_grid + p] = vsx[3 * p + 2] + vsc[3 * p + 2];
            }
        }
    }
    return 0;
#endif
}

}  // namespace

namespace jsiesta {

int evaluate_xc_libxc_host(
    int64_t n_grid,
    const double* r_ptr,
    const double* s_ptr,
    int64_t xc_id,
    int64_t n_spin_pol,
    double* e_ptr,
    double* vr_ptr,
    double* vs_ptr) {
    return evaluate_xc_libxc_impl(n_grid, r_ptr, s_ptr, xc_id, n_spin_pol, e_ptr, vr_ptr, vs_ptr);
}

int evaluate_mgga_libxc_host(
    int64_t n_grid,
    const double* r_ptr,
    const double* s_ptr,
    const double* t_ptr,
    int64_t x_id,
    int64_t c_id,
    int64_t n_spin_pol,
    double* e_ptr,
    double* vr_ptr,
    double* vs_ptr,
    double* vt_ptr) {
#ifndef JSIESTA_HAS_LIBXC
    (void)n_grid; (void)r_ptr; (void)s_ptr; (void)t_ptr;
    (void)x_id; (void)c_id; (void)n_spin_pol;
    (void)e_ptr; (void)vr_ptr; (void)vs_ptr; (void)vt_ptr;
    return 1;
#else
    const int spin_mode = (n_spin_pol == 2) ? XC_POLARIZED : XC_UNPOLARIZED;
    const int n_sp = (spin_mode == XC_POLARIZED) ? 2 : 1;
    const int n_sg = (spin_mode == XC_POLARIZED) ? 3 : 1;
    const size_t np_sz = static_cast<size_t>(n_grid);

    std::vector<double> r_lib(np_sz * n_sp, 0.0);
    std::vector<double> s_lib(np_sz * n_sg, 0.0);
    std::vector<double> t_lib(np_sz * n_sp, 0.0);
    std::vector<double> lapl_zero(np_sz * n_sp, 0.0);
    for (int64_t p = 0; p < n_grid; ++p) {
        if (n_sp == 1) {
            r_lib[p] = std::max(0.0, r_ptr[p] + r_ptr[n_grid + p]);
            s_lib[p] = std::max(0.0, s_ptr[p] + 2.0 * s_ptr[n_grid + p] + s_ptr[2 * n_grid + p]);
            t_lib[p] = std::max(0.0, t_ptr[p] + t_ptr[n_grid + p]);
        } else {
            r_lib[2 * p] = std::max(0.0, r_ptr[p]);
            r_lib[2 * p + 1] = std::max(0.0, r_ptr[n_grid + p]);
            s_lib[3 * p] = std::max(0.0, s_ptr[p]);
            s_lib[3 * p + 1] = s_ptr[n_grid + p];
            s_lib[3 * p + 2] = std::max(0.0, s_ptr[2 * n_grid + p]);
            t_lib[2 * p] = std::max(0.0, t_ptr[p]);
            t_lib[2 * p + 1] = std::max(0.0, t_ptr[n_grid + p]);
        }
    }

    std::vector<double> e_acc(np_sz, 0.0);
    std::vector<double> vr_acc(np_sz * n_sp, 0.0);
    std::vector<double> vs_acc(np_sz * n_sg, 0.0);
    std::vector<double> vt_acc(np_sz * n_sp, 0.0);
    std::vector<double> e_tmp(np_sz, 0.0);
    std::vector<double> vr_tmp(np_sz * n_sp, 0.0);
    std::vector<double> vs_tmp(np_sz * n_sg, 0.0);
    std::vector<double> vl_tmp(np_sz * n_sp, 0.0);
    std::vector<double> vt_tmp(np_sz * n_sp, 0.0);

    const int64_t ids[2] = {x_id, c_id};
    for (int k = 0; k < 2; ++k) {
        const int fid = static_cast<int>(ids[k]);
        if (fid <= 0) continue;
        xc_func_type func;
        if (xc_func_init(&func, fid, spin_mode) != 0) {
            return 2;
        }
        xc_func_set_dens_threshold(&func, 1e-10);
        const int fam = xc_func_info_get_family(func.info);
        std::fill(e_tmp.begin(), e_tmp.end(), 0.0);
        std::fill(vr_tmp.begin(), vr_tmp.end(), 0.0);
        std::fill(vs_tmp.begin(), vs_tmp.end(), 0.0);
        std::fill(vt_tmp.begin(), vt_tmp.end(), 0.0);
        if (fam == XC_FAMILY_MGGA) {
            xc_mgga_exc_vxc(
                &func, np_sz, r_lib.data(), s_lib.data(), lapl_zero.data(), t_lib.data(),
                e_tmp.data(), vr_tmp.data(), vs_tmp.data(), vl_tmp.data(), vt_tmp.data());
        } else if (fam == XC_FAMILY_GGA) {
            xc_gga_exc_vxc(&func, np_sz, r_lib.data(), s_lib.data(), e_tmp.data(), vr_tmp.data(), vs_tmp.data());
        } else {
            xc_lda_exc_vxc(&func, np_sz, r_lib.data(), e_tmp.data(), vr_tmp.data());
        }
        xc_func_end(&func);

        for (size_t i = 0; i < np_sz; ++i) e_acc[i] += e_tmp[i];
        for (size_t i = 0; i < np_sz * n_sp; ++i) {
            vr_acc[i] += vr_tmp[i];
            vt_acc[i] += vt_tmp[i];
        }
        for (size_t i = 0; i < np_sz * n_sg; ++i) vs_acc[i] += vs_tmp[i];
    }

    for (int64_t p = 0; p < n_grid; ++p) {
        const double n_tot = r_ptr[p] + r_ptr[n_grid + p];
        if (n_tot <= 1e-9) {
            e_ptr[p] = 0.0;
            vr_ptr[p] = vr_ptr[n_grid + p] = 0.0;
            vs_ptr[p] = vs_ptr[n_grid + p] = vs_ptr[2 * n_grid + p] = 0.0;
            vt_ptr[p] = vt_ptr[n_grid + p] = 0.0;
        } else {
            e_ptr[p] = n_tot * e_acc[p];
            if (n_sp == 1) {
                vr_ptr[p] = vr_ptr[n_grid + p] = vr_acc[p];
                vs_ptr[p] = vs_acc[p];
                vs_ptr[n_grid + p] = 2.0 * vs_acc[p];
                vs_ptr[2 * n_grid + p] = vs_acc[p];
                vt_ptr[p] = vt_ptr[n_grid + p] = vt_acc[p];
            } else {
                vr_ptr[p] = vr_acc[2 * p];
                vr_ptr[n_grid + p] = vr_acc[2 * p + 1];
                vs_ptr[p] = vs_acc[3 * p];
                vs_ptr[n_grid + p] = vs_acc[3 * p + 1];
                vs_ptr[2 * n_grid + p] = vs_acc[3 * p + 2];
                vt_ptr[p] = vt_acc[2 * p];
                vt_ptr[n_grid + p] = vt_acc[2 * p + 1];
            }
        }
    }
    return 0;
#endif
}

}  // namespace jsiesta

extern "C" {

/**
 * @brief Built-in LDA/GGA kernel on host arrays, bypassing Libxc (the code the CUDA XC kernel runs).
 *
 * @par Theory
 * Same outputs as \c jsiesta_xc_eval_f64 in the spin-resolved layout: \f$e_{\text{xc}}\f$, \f$v_\rho\f$ (2, n),
 * \f$v_\sigma\f$ (3, n).
 *
 * @par Algorithm
 * Loops \c jsiesta::xc::eval_semilocal_xc_point over the points; used to test the device kernel's arithmetic against
 * Libxc on the CPU.
 *
 * @par Invariants
 * Returns 2 for an unknown functional id and writes nothing.
 *
 * @par SIESTA Mapping
 * \c libgridxc \c ldaxc / \c ggaxc.
 *
 * @param n_grid Number of grid points.
 * @param rho_spin Spin densities, (2, n_grid) row-major.
 * @param sigma_spin Gradient invariants, (3, n_grid) row-major.
 * @param xc_id jsiesta functional id (1, 2, 10, 11).
 * @param n_spin_pol 1 (unpolarised) or 2 (polarised), as for \c jsiesta_xc_eval_f64.
 * @param exc Output energy density, (n_grid,).
 * @param v_rho Output density derivatives, (2, n_grid).
 * @param v_sigma Output gradient-invariant derivatives, (3, n_grid).
 * @return 0 on success, 2 for an unsupported id.
 */
int32_t jsiesta_xc_eval_builtin(int64_t n_grid, const double* rho_spin, const double* sigma_spin, int32_t xc_id,
                                int32_t n_spin_pol, double* exc, double* v_rho, double* v_sigma) {
    if (xc_id != kXcLdaPz && xc_id != kXcLdaPw92 && xc_id != kXcGgaPbe && xc_id != kXcGgaPbesol) return 2;
    for (int64_t p = 0; p < n_grid; ++p) {
        double vr[2], vs[3];
        jsiesta::xc::eval_semilocal_xc_point(xc_id, n_spin_pol, rho_spin[p], rho_spin[n_grid + p], sigma_spin[p],
                                             sigma_spin[n_grid + p], sigma_spin[2 * n_grid + p], &exc[p], vr, vs);
        v_rho[p] = vr[0];
        v_rho[n_grid + p] = vr[1];
        v_sigma[p] = vs[0];
        v_sigma[n_grid + p] = vs[1];
        v_sigma[2 * n_grid + p] = vs[2];
    }
    return 0;
}

XLA_FFI_Error* jsiesta_xc_eval_f64(XLA_FFI_CallFrame* call) {
    return XcEvalHandler(call);
}

XLA_FFI_Error* jsiesta_xc_eval_mgga_f64(XLA_FFI_CallFrame* call) {
    return XcEvalMggaHandler(call);
}

}  // extern "C"
