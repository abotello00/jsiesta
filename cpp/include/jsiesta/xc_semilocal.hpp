/**
 * @file xc_semilocal.hpp
 * @brief Built-in pointwise LDA/GGA exchange-correlation (Slater, PZ81, PW92, PBE, PBEsol), shared by the CPU and CUDA
 *        kernels and matched to the Libxc parametrisations jsiesta selects.
 *
 * @par Theory
 * Evaluates \f$e_{\text{xc}}(n_\uparrow, n_\downarrow, \sigma_{\uparrow\uparrow}, \sigma_{\uparrow\downarrow},
 * \sigma_{\downarrow\downarrow}) = n\,\varepsilon_{\text{xc}}\f$ and its five first derivatives
 * \f$v_{\rho,s} = \partial e_{\text{xc}}/\partial n_s\f$, \f$v_{\sigma,ss'} = \partial e_{\text{xc}}/\partial
 * \sigma_{ss'}\f$. Exchange uses spin scaling, \f$E_x[n_\uparrow,n_\downarrow] = \tfrac12(E_x[2n_\uparrow] +
 * E_x[2n_\downarrow])\f$, with the PBE enhancement factor \f$F_x(s) = 1 + \kappa - \kappa/(1 + \mu s^2/\kappa)\f$.
 * Correlation is PZ81, PW92, or PBE \f$\varepsilon_c^{\text{PW92}} + H(r_s, \zeta, t)\f$.
 *
 * @par Algorithm
 * Forward-mode dual numbers with five tangents give the energy density and all derivatives in one pass, exact to
 * rounding. The parameter sets follow the Libxc ids used by \c xc_ffi.cpp: \c LDA_X (1), \c LDA_C_PZ (9),
 * \c LDA_C_PW (12, original PW92 digits, \f$f''(0) = 1.709921\f$), \c GGA_X_PBE (101) / \c GGA_X_PBE_SOL (116), and
 * \c GGA_C_PBE (130) / \c GGA_C_PBE_SOL (133), whose PW92 part is Libxc's \c PW_MOD (\f$A = 0.0310907, 0.01554535,
 * 0.0168869\f$ and exact \f$f''(0)\f$). The thresholds follow Libxc as well: each part floors its inputs at its own
 * density threshold (\f$10^{-15}\f$ for exchange and the LDA correlations, \f$10^{-12}\f$ for PBE/PBEsol correlation)
 * and is dropped below it; points with \f$n \le 10^{-14}\f$ return zero, as the Libxc wrapper does.
 *
 * @par Invariants
 * - The same code runs on host and device (\c JSIESTA_HD), so the CPU build without Libxc, the CUDA kernel and the CPU
 *   parity test against Libxc evaluate identical arithmetic.
 * - No NaN for \f$n \to 0\f$ or \f$\nabla n \to 0\f$.
 *
 * @par SIESTA Mapping
 * The semilocal functionals of \c libgridxc (\c ldaxc, \c ggaxc) as called from \c cellxc.F.
 */

#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define JSIESTA_HD __host__ __device__
#else
#define JSIESTA_HD
#endif

namespace jsiesta::xc {

/// @cond INTERNAL
namespace semilocal_detail {

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr int32_t kLdaPz = 1;
constexpr int32_t kLdaPw92 = 2;
constexpr int32_t kGgaPbe = 10;
constexpr int32_t kGgaPbesol = 11;

template <int N>
struct Dual {
    double v;
    double d[N];
    JSIESTA_HD Dual() : v(0.0) {
        for (int i = 0; i < N; ++i) d[i] = 0.0;
    }
    JSIESTA_HD explicit Dual(double val) : v(val) {
        for (int i = 0; i < N; ++i) d[i] = 0.0;
    }
    JSIESTA_HD static Dual var(double val, int idx) {
        Dual r(val);
        r.d[idx] = 1.0;
        return r;
    }
};

template <int N> JSIESTA_HD inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r(a.v + b.v);
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] + b.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator+(const Dual<N>& a, double b) {
    Dual<N> r = a;
    r.v += b;
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator+(double a, const Dual<N>& b) { return b + a; }
template <int N> JSIESTA_HD inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r(a.v - b.v);
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] - b.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator-(const Dual<N>& a, double b) {
    Dual<N> r = a;
    r.v -= b;
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator-(double a, const Dual<N>& b) {
    Dual<N> r(a - b.v);
    for (int i = 0; i < N; ++i) r.d[i] = -b.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator-(const Dual<N>& a) {
    Dual<N> r(-a.v);
    for (int i = 0; i < N; ++i) r.d[i] = -a.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r(a.v * b.v);
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator*(const Dual<N>& a, double b) {
    Dual<N> r(a.v * b);
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b;
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator*(double a, const Dual<N>& b) { return b * a; }
template <int N> JSIESTA_HD inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r(a.v / b.v);
    const double inv_b2 = 1.0 / (b.v * b.v);
    for (int i = 0; i < N; ++i) r.d[i] = (a.d[i] * b.v - a.v * b.d[i]) * inv_b2;
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> operator/(const Dual<N>& a, double b) { return a * (1.0 / b); }
template <int N> JSIESTA_HD inline Dual<N> operator/(double a, const Dual<N>& b) {
    Dual<N> r(a / b.v);
    const double coeff = -a / (b.v * b.v);
    for (int i = 0; i < N; ++i) r.d[i] = coeff * b.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> d_sqrt(const Dual<N>& a) {
    const double sv = sqrt(a.v);
    Dual<N> r(sv);
    const double f = 0.5 / sv;
    for (int i = 0; i < N; ++i) r.d[i] = f * a.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> d_cbrt(const Dual<N>& a) {
    const double cv = cbrt(a.v);
    Dual<N> r(cv);
    const double f = 1.0 / (3.0 * cv * cv);
    for (int i = 0; i < N; ++i) r.d[i] = f * a.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> d_pow(const Dual<N>& a, double p) {
    const double pv = pow(a.v, p);
    Dual<N> r(pv);
    const double f = (a.v > 0.0) ? (p * pow(a.v, p - 1.0)) : 0.0;
    for (int i = 0; i < N; ++i) r.d[i] = f * a.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> d_log(const Dual<N>& a) {
    Dual<N> r(log(a.v));
    const double f = 1.0 / a.v;
    for (int i = 0; i < N; ++i) r.d[i] = f * a.d[i];
    return r;
}
template <int N> JSIESTA_HD inline Dual<N> d_exp(const Dual<N>& a) {
    const double ev = exp(a.v);
    Dual<N> r(ev);
    for (int i = 0; i < N; ++i) r.d[i] = ev * a.d[i];
    return r;
}

/// PW92 parameter set (Libxc lda_c_pw.c): a, alpha1, beta1..beta4 for (paramagnetic, ferromagnetic, -alpha_c), fz20.
struct Pw92Params {
    double a[3], alpha1[3], beta1[3], beta2[3], beta3[3], beta4[3];
    double fz20;
};

JSIESTA_HD inline Pw92Params pw92_params(bool modified) {
    Pw92Params p = {
        {0.031091, 0.015545, 0.016887},
        {0.21370, 0.20548, 0.11125},
        {7.5957, 14.1189, 10.357},
        {3.5876, 6.1977, 3.6231},
        {1.6382, 3.3662, 0.88026},
        {0.49294, 0.62517, 0.49671},
        1.709921};
    if (modified) {  // PW_MOD: the digits of the PBE reference code
        p.a[0] = 0.0310907;
        p.a[1] = 0.01554535;
        p.a[2] = 0.0168869;
        p.fz20 = 1.709920934161365617563962776245;
    }
    return p;
}

template <int N>
JSIESTA_HD inline Dual<N> pw92_g(const Dual<N>& rs, const Pw92Params& p, int k) {
    const Dual<N> sq_rs = d_sqrt(rs);
    const Dual<N> aux = p.beta1[k] * sq_rs + p.beta2[k] * rs + p.beta3[k] * (rs * sq_rs) + p.beta4[k] * (rs * rs);
    return -2.0 * p.a[k] * (1.0 + p.alpha1[k] * rs) * d_log(1.0 + 1.0 / (2.0 * p.a[k] * aux));
}

template <int N>
JSIESTA_HD inline Dual<N> f_zeta(const Dual<N>& zeta) {
    const Dual<N> zp = d_pow(1.0 + zeta + 1e-30, 4.0 / 3.0);
    const Dual<N> zm = d_pow(1.0 - zeta + 1e-30, 4.0 / 3.0);
    return (zp + zm - 2.0) / (pow(2.0, 4.0 / 3.0) - 2.0);
}

template <int N>
JSIESTA_HD inline Dual<N> pw92_ec(const Dual<N>& rs, const Dual<N>& zeta, bool modified) {
    const Pw92Params p = pw92_params(modified);
    const Dual<N> g1 = pw92_g(rs, p, 0);
    const Dual<N> g2 = pw92_g(rs, p, 1);
    const Dual<N> g3 = pw92_g(rs, p, 2);  // parametrises -alpha_c
    const Dual<N> fz = f_zeta(zeta);
    const Dual<N> z4 = zeta * zeta * zeta * zeta;
    return g1 + z4 * fz * (g2 - g1 + g3 / p.fz20) - fz * g3 / p.fz20;
}

template <int N>
JSIESTA_HD inline Dual<N> pz81_single(const Dual<N>& rs, double gamma, double b1, double b2, double A, double B,
                                      double C, double D) {
    if (rs.v >= 1.0) return gamma / (1.0 + b1 * d_sqrt(rs) + b2 * rs);
    const Dual<N> lrs = d_log(rs);
    return A * lrs + B + C * rs * lrs + D * rs;
}

template <int N>
JSIESTA_HD inline Dual<N> pz81_ec(const Dual<N>& rs, const Dual<N>& zeta) {
    const Dual<N> ec_u = pz81_single(rs, -0.1423, 1.0529, 0.3334, 0.0311, -0.048, 0.0020, -0.0116);
    const Dual<N> ec_p = pz81_single(rs, -0.0843, 1.3981, 0.2611, 0.01555, -0.0269, 0.0007, -0.0048);
    return ec_u + (ec_p - ec_u) * f_zeta(zeta);
}

/// Libxc density thresholds (\c xc_func_info::dens_threshold) of the exchange and correlation parts.
JSIESTA_HD inline double x_threshold(int32_t) { return 1e-15; }  // LDA_X, GGA_X_PBE, GGA_X_PBE_SOL
JSIESTA_HD inline double c_threshold(int32_t xc_id) {
    return (xc_id == kGgaPbe || xc_id == kGgaPbesol) ? 1e-12 : 1e-15;  // GGA_C_PBE*, LDA_C_PZ / LDA_C_PW
}

/// Spin-scaled exchange; a spin channel at or below the threshold is screened out (Libxc \c screen_dens).
template <int N>
JSIESTA_HD inline Dual<N> exchange(int32_t xc_id, const Dual<N>& n_up, const Dual<N>& n_dn, const Dual<N>& sig_uu,
                                   const Dual<N>& sig_dd) {
    const bool is_gga = (xc_id == kGgaPbe || xc_id == kGgaPbesol);
    const double thr = x_threshold(xc_id);
    const double cx_unif = -0.75 * cbrt(3.0 / kPi);
    auto ex_spin = [&](const Dual<N>& n_s, const Dual<N>& sig_s) -> Dual<N> {
        if (n_s.v <= thr) return Dual<N>(0.0);
        const Dual<N> ns = 2.0 * n_s;
        const Dual<N> ex_unif = cx_unif * d_cbrt(ns);
        if (!is_gga) return ns * ex_unif;
        const double kappa = 0.804;
        const double mu = (xc_id == kGgaPbesol) ? (10.0 / 81.0) : 0.2195149727645171;
        const Dual<N> kf = d_cbrt((3.0 * kPi * kPi) * ns);
        const Dual<N> denom = 2.0 * kf * ns;
        const Dual<N> s2 = (4.0 * sig_s) / (denom * denom);
        return ns * ex_unif * (1.0 + kappa - kappa / (1.0 + (mu / kappa) * s2));
    };
    return 0.5 * (ex_spin(n_up, sig_uu) + ex_spin(n_dn, sig_dd));
}

/// Correlation energy density \f$n\varepsilon_c\f$ (PZ81, PW92, or PBE/PBEsol on PW_MOD).
template <int N>
JSIESTA_HD inline Dual<N> correlation(int32_t xc_id, const Dual<N>& n_up, const Dual<N>& n_dn, const Dual<N>& sig_uu,
                                      const Dual<N>& sig_ud, const Dual<N>& sig_dd) {
    const Dual<N> n_tot = n_up + n_dn;
    const Dual<N> rs = d_cbrt(0.75 / (kPi * n_tot));
    const Dual<N> zeta = (n_up - n_dn) / n_tot;
    if (xc_id == kLdaPz) return n_tot * pz81_ec(rs, zeta);
    if (xc_id == kLdaPw92) return n_tot * pw92_ec(rs, zeta, false);

    const Dual<N> ec_u = pw92_ec(rs, zeta, true);
    const double gamma_c = (1.0 - log(2.0)) / (kPi * kPi);
    const double beta_c = (xc_id == kGgaPbesol) ? 0.046 : 0.06672455060314922;
    const Dual<N> phi = 0.5 * (d_pow(1.0 + zeta + 1e-30, 2.0 / 3.0) + d_pow(1.0 - zeta + 1e-30, 2.0 / 3.0));
    const Dual<N> phi3 = phi * phi * phi;
    const Dual<N> kf_tot = d_cbrt((3.0 * kPi * kPi) * n_tot);
    const Dual<N> ks = d_sqrt((4.0 / kPi) * kf_tot);
    const Dual<N> sig_tot = sig_uu + 2.0 * sig_ud + sig_dd;
    const Dual<N> denom_t = 2.0 * phi * ks * n_tot;
    const Dual<N> t2 = sig_tot / (denom_t * denom_t);
    const Dual<N> A = (beta_c / gamma_c) / (d_exp((-ec_u) / (gamma_c * phi3)) - 1.0 + 1e-30);
    const Dual<N> at2 = A * t2;
    const Dual<N> h_frac = (1.0 + at2) / (1.0 + at2 + at2 * at2);
    const Dual<N> H = gamma_c * phi3 * d_log(1.0 + (beta_c / gamma_c) * t2 * h_frac);
    return n_tot * (ec_u + H);
}

/// Spin-polarised inputs after Libxc's per-functional flooring (\c work_gga: \f$n_s \ge\f$ thr,
/// \f$\sigma_{ss} \ge \f$ thr\f$^{8/3}\f$, \f$|\sigma_{\uparrow\downarrow}| \le
/// (\sigma_{\uparrow\uparrow} + \sigma_{\downarrow\downarrow})/2\f$); tangents 0..4 = (n_up, n_dn, s_uu, s_ud, s_dd).
struct Floored {
    Dual<5> n_up, n_dn, s_uu, s_ud, s_dd;
};

JSIESTA_HD inline Floored floor_polarized(double thr, double ru, double rd, double suu, double sud, double sdd) {
    const double s_thr = pow(thr, 8.0 / 3.0);
    const double fuu = fmax(s_thr, suu), fdd = fmax(s_thr, sdd);
    const double s_ave = 0.5 * (fuu + fdd);
    const double fud = fmin(fmax(sud, -s_ave), s_ave);
    return {Dual<5>::var(fmax(thr, ru), 0), Dual<5>::var(fmax(thr, rd), 1), Dual<5>::var(fuu, 2),
            Dual<5>::var(fud, 3), Dual<5>::var(fdd, 4)};
}

/// Unpolarised inputs as Libxc sees them: total \f$n\f$ (tangent 0) and \f$\sigma\f$ (tangent 2), floored, split as
/// \f$n_s = n/2\f$, \f$\sigma_{ss'} = \sigma/4\f$.
JSIESTA_HD inline Floored floor_unpolarized(double thr, double n, double sig) {
    const Dual<5> nt = Dual<5>::var(fmax(thr, n), 0);
    const Dual<5> st = Dual<5>::var(fmax(pow(thr, 8.0 / 3.0), sig), 2);
    const Dual<5> h = 0.5 * nt, q = 0.25 * st;
    return {h, h, q, q, q};
}

}  // namespace semilocal_detail
/// @endcond

/**
 * @brief Evaluate one grid point of the built-in LDA/GGA functional.
 *
 * @par Theory
 * Returns \f$e_{\text{xc}} = n\varepsilon_{\text{xc}}\f$, \f$v_{\rho,s}\f$ and \f$v_{\sigma,ss'}\f$ in jsiesta's
 * spin-resolved layout. Unpolarised input (\c n_spin_pol = 1) is evaluated in the total \f$n\f$ and
 * \f$\sigma = \sigma_{\uparrow\uparrow} + 2\sigma_{\uparrow\downarrow} + \sigma_{\downarrow\downarrow}\f$ and mapped as
 * \f$v_\rho = (v, v)\f$, \f$v_\sigma = (w, 2w, w)\f$, which is what the Libxc wrapper of \c xc_ffi.cpp returns.
 *
 * @par Algorithm
 * Exchange and correlation are evaluated separately, each on inputs floored with its own Libxc density threshold
 * (\c work_gga / \c work_lda), and dropped where the total density is below that threshold. As in the wrapper, the
 * energy density is \f$n\,\varepsilon(\text{floored})\f$ with the unfloored \f$n\f$; derivatives are those at the floored
 * point.
 *
 * @par Invariants
 * Points with \f$n_\uparrow + n_\downarrow \le 10^{-14}\f$ return all zeros.
 *
 * @par SIESTA Mapping
 * One point of \c libgridxc \c ldaxc / \c ggaxc.
 *
 * @param xc_id jsiesta functional id: 1 (LDA, PZ81), 2 (LDA, PW92), 10 (PBE), 11 (PBEsol).
 * @param n_spin_pol 1 (unpolarised) or 2 (polarised).
 * @param ru Spin-up density.
 * @param rd Spin-down density.
 * @param suu Gradient invariant \c sigma_uu = grad(n_up).grad(n_up).
 * @param sud Gradient invariant \c sigma_ud = grad(n_up).grad(n_dn).
 * @param sdd Gradient invariant \c sigma_dd = grad(n_dn).grad(n_dn).
 * @param e Output energy density \c e_xc.
 * @param vr Output \c (v_rho_up, v_rho_dn).
 * @param vs Output \c (v_sigma_uu, v_sigma_ud, v_sigma_dd).
 */
JSIESTA_HD inline void eval_semilocal_xc_point(int32_t xc_id, int32_t n_spin_pol, double ru, double rd, double suu,
                                               double sud, double sdd, double* e, double* vr, double* vs) {
    namespace sd = semilocal_detail;
    *e = 0.0;
    vr[0] = vr[1] = 0.0;
    vs[0] = vs[1] = vs[2] = 0.0;
    const double n_tot = ru + rd;
    if (n_tot <= 1e-14) return;
    const bool unpol = (n_spin_pol != 2);
    const double sig_tot = suu + 2.0 * sud + sdd;
    // Libxc returns eps_xc (per particle) at the floored inputs and the wrapper multiplies by the unfloored n, so each
    // part's energy density is rescaled by n / n_floored; the derivatives are those at the floored point.
    sd::Dual<5> val(0.0);
    double e_val = 0.0;
    const double thr_x = sd::x_threshold(xc_id), thr_c = sd::c_threshold(xc_id);
    if (n_tot >= thr_x) {
        const sd::Floored f = unpol ? sd::floor_unpolarized(thr_x, n_tot, sig_tot)
                                    : sd::floor_polarized(thr_x, ru, rd, suu, sud, sdd);
        const sd::Dual<5> part = sd::exchange(xc_id, f.n_up, f.n_dn, f.s_uu, f.s_dd);
        val = val + part;
        e_val += part.v * (n_tot / (f.n_up.v + f.n_dn.v));
    }
    if (n_tot >= thr_c) {
        const sd::Floored f = unpol ? sd::floor_unpolarized(thr_c, n_tot, sig_tot)
                                    : sd::floor_polarized(thr_c, ru, rd, suu, sud, sdd);
        const sd::Dual<5> part = sd::correlation(xc_id, f.n_up, f.n_dn, f.s_uu, f.s_ud, f.s_dd);
        val = val + part;
        e_val += part.v * (n_tot / (f.n_up.v + f.n_dn.v));
    }
    *e = e_val;
    if (unpol) {
        vr[0] = vr[1] = val.d[0];
        vs[0] = vs[2] = val.d[2];
        vs[1] = 2.0 * val.d[2];
    } else {
        vr[0] = val.d[0];
        vr[1] = val.d[1];
        vs[0] = val.d[2];
        vs[1] = val.d[3];
        vs[2] = val.d[4];
    }
}

}  // namespace jsiesta::xc
