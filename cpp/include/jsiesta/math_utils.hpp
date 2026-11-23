/**
 * @file math_utils.hpp
 * @brief Real spherical harmonics, spherical Bessel functions, cubic splines, and Gaunt tensors.
 *
 * @par Theory
 * Provides canonical orthonormal real spherical harmonics \f$Y_{lm}(\hat{\mathbf{r}})\f$
 * matching SIESTA's \c spher_harm convention (\f$m<0 \propto \sin|m|\phi\f$,
 * \f$m=0\f$, \f$m>0 \propto \cos m\phi\f$), solid harmonics \f$C_{lm}(\mathbf{r}) = r^l Y_{lm}\f$
 * with exact Cartesian gradients, spherical Bessel functions \f$j_l(x)\f$, and
 * natural cubic spline evaluation.
 *
 * @par Algorithm
 * Evaluates associated Legendre polynomials via stable upward recurrence, spherical
 * Bessel functions via upward/downward Miller recurrence and small-argument Taylor
 * series, and natural cubic splines via tridiagonal Thomas solve.
 *
 * @par Invariants
 * - \f$\int_{S^2} Y_{lm} Y_{l'm'} d\Omega = \delta_{ll'}\delta_{mm'}\f$.
 * - \f$C_{lm}(\mathbf{r})\f$ and \f$\nabla C_{lm}(\mathbf{r})\f$ are singularity-free at \f$r=0\f$.
 *
 * @par SIESTA Mapping
 * Replaces \c spher_harm.f, \c bessel.f, and \c radial.f utilities.
 */

#pragma once

#include <cmath>
#include <vector>

namespace jsiesta::math {

/**
 * @brief Mathematical constant \f$\pi\f$ in IEEE-754 double precision.
 *
 * @par Theory
 * Ratio of circle circumference to diameter used across radial FFTs and spherical harmonics.
 *
 * @par Algorithm
 * Compile-time \c constexpr scalar constant.
 *
 * @par Invariants
 * Positive constant matching \c M_PI to double precision.
 *
 * @par SIESTA Mapping
 * Replaces \c pi in SIESTA \c precision / \c units modules.
 */
constexpr double kPi = 3.1415926535897932384626433832795;

/**
 * @brief Evaluate spherical Bessel function \f$j_l(x)\f$ for \f$0 \le l \le l_{\max}\f$.
 *
 * @par Theory
 * Computes \f$j_l(x) = \sqrt{\pi/(2x)} J_{l+1/2}(x)\f$.
 *
 * @par Algorithm
 * Uses Maclaurin series for small \f$x\f$ and downward Miller recurrence when \f$x < l + 2\f$,
 * or stable upward recurrence when \f$x \ge l + 2\f$.
 *
 * @param l_max Maximum angular momentum (\f$0 \le l_{\max} \le 10\f$).
 * @param x Non-negative dimensionless argument \f$kr\f$.
 * @param out Output buffer of size \c l_max + 1.
 *
 * @par Invariants
 * Smooth and finite at \f$x = 0\f$ (\f$j_0(0)=1\f$, \f$j_{l>0}(0)=0\f$).
 *
 * @par SIESTA Mapping
 * Replaces \c bessph in \c matel.F90.
 */
inline void spherical_bessel_jl(int l_max, double x, double* out) {
    if (x < 1e-6) {
        double x_pow = 1.0;
        double denom = 1.0;
        for (int l = 0; l <= l_max; ++l) {
            if (l > 0) {
                x_pow *= x;
                denom *= static_cast<double>(2 * l + 1);
            }
            const double x2 = x * x;
            out[l] = (x_pow / denom) * (1.0 - x2 / (2.0 * (2 * l + 3)));
        }
        return;
    }
    const double j0 = std::sin(x) / x;
    out[0] = j0;
    if (l_max == 0) return;
    const double j1 = (std::sin(x) / x - std::cos(x)) / x;
    if (l_max == 1) {
        out[1] = j1;
        return;
    }
    if (x >= static_cast<double>(l_max) + 2.0) {
        out[1] = j1;
        for (int l = 1; l < l_max; ++l) {
            out[l + 1] = ((2 * l + 1) / x) * out[l] - out[l - 1];
        }
        return;
    }
    // Downward Miller recurrence for stability when x < l_max + 2
    const int l_start = std::min(l_max + 24, 60);
    double jm1 = 0.0;
    double jcurr = 1e-30;
    double tmp[64] = {0.0};
    tmp[l_start] = jcurr;
    for (int l = l_start; l >= 1; --l) {
        const double jprev = ((2 * l + 1) / x) * jcurr - jm1;
        tmp[l - 1] = jprev;
        jm1 = jcurr;
        jcurr = jprev;
    }
    const double scale = (std::abs(tmp[0]) > std::abs(tmp[1])) ? (j0 / tmp[0]) : (j1 / tmp[1]);
    for (int l = 1; l <= l_max; ++l) {
        out[l] = tmp[l] * scale;
    }
}

/**
 * @brief Evaluate real solid harmonics \f$C_{lm}(\mathbf{r}) = r^l Y_{lm}(\hat{\mathbf{r}})\f$, gradients, and optional Hessians.
 *
 * @par Theory
 * Evaluates orthonormal real spherical harmonics multiplied by \f$r^l\f$ so every
 * function is a homogeneous polynomial of degree \f$l\f$ in \f$(x, y, z)\f$,
 * indexed sequentially as \f$l^2 + (m + l)\f$ for \f$m \in [-l, l]\f$.
 *
 * @par Algorithm
 * Computes the unnormalized Cartesian solid harmonics \f$S_l^m(x,y,z)\f$ together with
 * exact first and optional second Cartesian derivatives via recurrence, then applies
 * orthonormalization \f$N_{lm} = \sqrt{\frac{2l+1}{4\pi} \frac{(l-|m|)!}{(l+|m|)!} (2 - \delta_{m0})}\f$.
 *
 * @param x Cartesian x coordinate.
 * @param y Cartesian y coordinate.
 * @param z Cartesian z coordinate.
 * @param l_max Maximum angular momentum (\f$0 \le l_{\max} \le 6\f$).
 * @param c_lm Output solid harmonic values, length \f$(l_{\max}+1)^2\f$.
 * @param dc_lm Optional output Cartesian gradients \f$(\partial_x, \partial_y, \partial_z) C_{lm}\f$, length \f$3(l_{\max}+1)^2\f$.
 * @param hc_lm Optional output row-major \f$3\times 3\f$ Cartesian Hessians \f$\partial_\alpha\partial_\beta C_{lm}\f$, length \f$9(l_{\max}+1)^2\f$.
 *
 * @par Invariants
 * Polynomial exactness at \f$\mathbf{r} = \mathbf{0}\f$ with zero division by \f$r\f$.
 *
 * @par SIESTA Mapping
 * Replaces \c spher_harm.f solid harmonic, gradient, and Hessian evaluation.
 */
inline void real_solid_harmonics_and_grad(
    double x, double y, double z, int l_max, double* c_lm, double* dc_lm, double* hc_lm = nullptr) {
    const int stride = l_max + 1;
    const int n_tbl = stride * stride;
    const bool need_grad = (dc_lm != nullptr) || (hc_lm != nullptr);
    const bool need_hess = (hc_lm != nullptr);
    std::vector<double> A(n_tbl, 0.0), B(n_tbl, 0.0);
    std::vector<double> dAx, dAy, dAz, dBx, dBy, dBz;
    std::vector<double> hAxx, hAxy, hAxz, hAyy, hAyz, hAzz;
    std::vector<double> hBxx, hBxy, hBxz, hByy, hByz, hBzz;
    if (need_grad) {
        dAx.assign(n_tbl, 0.0); dAy.assign(n_tbl, 0.0); dAz.assign(n_tbl, 0.0);
        dBx.assign(n_tbl, 0.0); dBy.assign(n_tbl, 0.0); dBz.assign(n_tbl, 0.0);
    }
    if (need_hess) {
        hAxx.assign(n_tbl, 0.0); hAxy.assign(n_tbl, 0.0); hAxz.assign(n_tbl, 0.0);
        hAyy.assign(n_tbl, 0.0); hAyz.assign(n_tbl, 0.0); hAzz.assign(n_tbl, 0.0);
        hBxx.assign(n_tbl, 0.0); hBxy.assign(n_tbl, 0.0); hBxz.assign(n_tbl, 0.0);
        hByy.assign(n_tbl, 0.0); hByz.assign(n_tbl, 0.0); hBzz.assign(n_tbl, 0.0);
    }

    auto idx = [stride](int l, int m) { return l * stride + m; };
    const double r2 = x * x + y * y + z * z;

    A[idx(0, 0)] = 1.0;
    for (int l = 0; l < l_max; ++l) {
        const double f = static_cast<double>(2 * l + 1);
        const int cur = idx(l, l);
        const int nxt = idx(l + 1, l + 1);
        A[nxt] = f * (x * A[cur] - y * B[cur]);
        B[nxt] = f * (x * B[cur] + y * A[cur]);
        if (need_grad) {
            dAx[nxt] = f * (A[cur] + x * dAx[cur] - y * dBx[cur]);
            dAy[nxt] = f * (x * dAy[cur] - B[cur] - y * dBy[cur]);
            dAz[nxt] = f * (x * dAz[cur] - y * dBz[cur]);
            dBx[nxt] = f * (B[cur] + x * dBx[cur] + y * dAx[cur]);
            dBy[nxt] = f * (x * dBy[cur] + A[cur] + y * dAy[cur]);
            dBz[nxt] = f * (x * dBz[cur] + y * dAz[cur]);
        }
        if (need_hess) {
            hAxx[nxt] = f * (2.0 * dAx[cur] + x * hAxx[cur] - y * hBxx[cur]);
            hAxy[nxt] = f * (dAy[cur] - dBx[cur] + x * hAxy[cur] - y * hBxy[cur]);
            hAxz[nxt] = f * (dAz[cur] + x * hAxz[cur] - y * hBxz[cur]);
            hAyy[nxt] = f * (-2.0 * dBy[cur] + x * hAyy[cur] - y * hByy[cur]);
            hAyz[nxt] = f * (-dBz[cur] + x * hAyz[cur] - y * hByz[cur]);
            hAzz[nxt] = f * (x * hAzz[cur] - y * hBzz[cur]);

            hBxx[nxt] = f * (2.0 * dBx[cur] + x * hBxx[cur] + y * hAxx[cur]);
            hBxy[nxt] = f * (dBy[cur] + dAx[cur] + x * hBxy[cur] + y * hAxy[cur]);
            hBxz[nxt] = f * (dBz[cur] + x * hBxz[cur] + y * hAxz[cur]);
            hByy[nxt] = f * (2.0 * dAy[cur] + x * hByy[cur] + y * hAyy[cur]);
            hByz[nxt] = f * (dAz[cur] + x * hByz[cur] + y * hAyz[cur]);
            hBzz[nxt] = f * (x * hBzz[cur] + y * hAzz[cur]);
        }
    }
    for (int m = 0; m < l_max; ++m) {
        const double f = static_cast<double>(2 * m + 1);
        const int cur = idx(m, m);
        const int nxt = idx(m + 1, m);
        A[nxt] = f * z * A[cur];
        B[nxt] = f * z * B[cur];
        if (need_grad) {
            dAx[nxt] = f * z * dAx[cur];
            dAy[nxt] = f * z * dAy[cur];
            dAz[nxt] = f * (A[cur] + z * dAz[cur]);
            dBx[nxt] = f * z * dBx[cur];
            dBy[nxt] = f * z * dBy[cur];
            dBz[nxt] = f * (B[cur] + z * dBz[cur]);
        }
        if (need_hess) {
            hAxx[nxt] = f * z * hAxx[cur];
            hAxy[nxt] = f * z * hAxy[cur];
            hAxz[nxt] = f * (dAx[cur] + z * hAxz[cur]);
            hAyy[nxt] = f * z * hAyy[cur];
            hAyz[nxt] = f * (dAy[cur] + z * hAyz[cur]);
            hAzz[nxt] = f * (2.0 * dAz[cur] + z * hAzz[cur]);

            hBxx[nxt] = f * z * hBxx[cur];
            hBxy[nxt] = f * z * hBxy[cur];
            hBxz[nxt] = f * (dBx[cur] + z * hBxz[cur]);
            hByy[nxt] = f * z * hByy[cur];
            hByz[nxt] = f * (dBy[cur] + z * hByz[cur]);
            hBzz[nxt] = f * (2.0 * dBz[cur] + z * hBzz[cur]);
        }

        for (int l = m + 1; l < l_max; ++l) {
            const double c1 = static_cast<double>(2 * l + 1) / static_cast<double>(l + 1 - m);
            const double c2 = static_cast<double>(l + m) / static_cast<double>(l + 1 - m);
            const int i_cur = idx(l, m);
            const int i_prev = idx(l - 1, m);
            const int i_nxt = idx(l + 1, m);
            A[i_nxt] = c1 * z * A[i_cur] - c2 * r2 * A[i_prev];
            B[i_nxt] = c1 * z * B[i_cur] - c2 * r2 * B[i_prev];
            if (need_grad) {
                dAx[i_nxt] = c1 * z * dAx[i_cur] - c2 * (2.0 * x * A[i_prev] + r2 * dAx[i_prev]);
                dAy[i_nxt] = c1 * z * dAy[i_cur] - c2 * (2.0 * y * A[i_prev] + r2 * dAy[i_prev]);
                dAz[i_nxt] = c1 * (A[i_cur] + z * dAz[i_cur]) - c2 * (2.0 * z * A[i_prev] + r2 * dAz[i_prev]);
                dBx[i_nxt] = c1 * z * dBx[i_cur] - c2 * (2.0 * x * B[i_prev] + r2 * dBx[i_prev]);
                dBy[i_nxt] = c1 * z * dBy[i_cur] - c2 * (2.0 * y * B[i_prev] + r2 * dBy[i_prev]);
                dBz[i_nxt] = c1 * (B[i_cur] + z * dBz[i_cur]) - c2 * (2.0 * z * B[i_prev] + r2 * dBz[i_prev]);
            }
            if (need_hess) {
                hAxx[i_nxt] = c1 * z * hAxx[i_cur] - c2 * (2.0 * A[i_prev] + 4.0 * x * dAx[i_prev] + r2 * hAxx[i_prev]);
                hAxy[i_nxt] = c1 * z * hAxy[i_cur] - c2 * (2.0 * x * dAy[i_prev] + 2.0 * y * dAx[i_prev] + r2 * hAxy[i_prev]);
                hAxz[i_nxt] = c1 * (dAx[i_cur] + z * hAxz[i_cur]) - c2 * (2.0 * x * dAz[i_prev] + 2.0 * z * dAx[i_prev] + r2 * hAxz[i_prev]);
                hAyy[i_nxt] = c1 * z * hAyy[i_cur] - c2 * (2.0 * A[i_prev] + 4.0 * y * dAy[i_prev] + r2 * hAyy[i_prev]);
                hAyz[i_nxt] = c1 * (dAy[i_cur] + z * hAyz[i_cur]) - c2 * (2.0 * y * dAz[i_prev] + 2.0 * z * dAy[i_prev] + r2 * hAyz[i_prev]);
                hAzz[i_nxt] = c1 * (2.0 * dAz[i_cur] + z * hAzz[i_cur]) - c2 * (2.0 * A[i_prev] + 4.0 * z * dAz[i_prev] + r2 * hAzz[i_prev]);

                hBxx[i_nxt] = c1 * z * hBxx[i_cur] - c2 * (2.0 * B[i_prev] + 4.0 * x * dBx[i_prev] + r2 * hBxx[i_prev]);
                hBxy[i_nxt] = c1 * z * hBxy[i_cur] - c2 * (2.0 * x * dBy[i_prev] + 2.0 * y * dBx[i_prev] + r2 * hBxy[i_prev]);
                hBxz[i_nxt] = c1 * (dBx[i_cur] + z * hBxz[i_cur]) - c2 * (2.0 * x * dBz[i_prev] + 2.0 * z * dBx[i_prev] + r2 * hBxz[i_prev]);
                hByy[i_nxt] = c1 * z * hByy[i_cur] - c2 * (2.0 * B[i_prev] + 4.0 * y * dBy[i_prev] + r2 * hByy[i_prev]);
                hByz[i_nxt] = c1 * (dBy[i_cur] + z * hByz[i_cur]) - c2 * (2.0 * y * dBz[i_prev] + 2.0 * z * dBy[i_prev] + r2 * hByz[i_prev]);
                hBzz[i_nxt] = c1 * (2.0 * dBz[i_cur] + z * hBzz[i_cur]) - c2 * (2.0 * B[i_prev] + 4.0 * z * dBz[i_prev] + r2 * hBzz[i_prev]);
            }
        }
    }

    std::vector<double> fact(2 * l_max + 2, 1.0);
    for (int k = 1; k < static_cast<int>(fact.size()); ++k) {
        fact[k] = fact[k - 1] * static_cast<double>(k);
    }

    for (int l = 0; l <= l_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int am = std::abs(m);
            const double norm = std::sqrt(
                (static_cast<double>(2 * l + 1) / (4.0 * kPi)) *
                (fact[l - am] / fact[l + am]) *
                (m == 0 ? 1.0 : 2.0));
            const int out_idx = l * l + (m + l);
            const int tbl_idx = idx(l, am);
            const bool neg = (m < 0);
            c_lm[out_idx] = norm * (neg ? B[tbl_idx] : A[tbl_idx]);
            if (dc_lm) {
                dc_lm[3 * out_idx + 0] = norm * (neg ? dBx[tbl_idx] : dAx[tbl_idx]);
                dc_lm[3 * out_idx + 1] = norm * (neg ? dBy[tbl_idx] : dAy[tbl_idx]);
                dc_lm[3 * out_idx + 2] = norm * (neg ? dBz[tbl_idx] : dAz[tbl_idx]);
            }
            if (hc_lm) {
                const double hxx = norm * (neg ? hBxx[tbl_idx] : hAxx[tbl_idx]);
                const double hxy = norm * (neg ? hBxy[tbl_idx] : hAxy[tbl_idx]);
                const double hxz = norm * (neg ? hBxz[tbl_idx] : hAxz[tbl_idx]);
                const double hyy = norm * (neg ? hByy[tbl_idx] : hAyy[tbl_idx]);
                const double hyz = norm * (neg ? hByz[tbl_idx] : hAyz[tbl_idx]);
                const double hzz = norm * (neg ? hBzz[tbl_idx] : hAzz[tbl_idx]);
                hc_lm[9 * out_idx + 0] = hxx; hc_lm[9 * out_idx + 1] = hxy; hc_lm[9 * out_idx + 2] = hxz;
                hc_lm[9 * out_idx + 3] = hxy; hc_lm[9 * out_idx + 4] = hyy; hc_lm[9 * out_idx + 5] = hyz;
                hc_lm[9 * out_idx + 6] = hxz; hc_lm[9 * out_idx + 7] = hyz; hc_lm[9 * out_idx + 8] = hzz;
            }
        }
    }
}

/**
 * @brief Uniform-grid natural cubic spline interpolant with \f$O(1)\f$ lookup.
 *
 * @par Theory
 * Evaluates piecewise cubic polynomials \f$f(r)\f$ and \f$f'(r)\f$ on uniform
 * spacing \f$\Delta r\f$ with natural boundary conditions \f$f''(r_0)=f''(r_{N-1})=0\f$.
 *
 * @par Algorithm
 * Thomas tridiagonal factorization on construction, direct index clamping on evaluation.
 *
 * @par Invariants
 * Returns 0 and 0 derivative for \f$r \ge r_{\text{cut}}\f$.
 *
 * @par SIESTA Mapping
 * Replaces \c splift / \c splint in \c radial.f.
 */
struct UniformCubicSpline {
    double dr = 1.0;        /**< Uniform radial grid spacing. */
    double rcut = 0.0;      /**< Cutoff radius beyond which value is zero. */
    std::vector<double> y;  /**< Function values at grid nodes. */
    std::vector<double> y2; /**< Second-derivative spline coefficients. */

    /**
     * @brief Construct natural cubic spline from uniform samples on \f$[0, r_{\max}]\f$.
     *
     * @par Theory
     * Solves natural boundary conditions \f$f''(0)=f''(r_{\max})=0\f$ for \f$C^2\f$ radial interpolation.
     *
     * @par Algorithm
     * Executes the linear-time tridiagonal Thomas algorithm over uniform step \f$\Delta r\f$.
     *
     * @param vals Uniformly spaced radial samples.
     * @param n Number of grid points.
     * @param r_max Maximum grid radius in Bohr.
     * @param r_cut Cutoff radius beyond which the interpolated value vanishes.
     *
     * @par Invariants
     * Exact reproduction of tabulated values at all interior grid nodes.
     *
     * @par SIESTA Mapping
     * Replaces \c spline / \c radial setup in \c radial.f.
     */
    void build(const double* vals, int n, double r_max, double r_cut) {
        y.assign(vals, vals + n);
        y2.assign(n, 0.0);
        rcut = r_cut;
        if (n < 3 || r_max <= 0.0) {
            dr = 1.0;
            return;
        }
        dr = r_max / static_cast<double>(n - 1);
        std::vector<double> u(n, 0.0);
        for (int i = 1; i < n - 1; ++i) {
            const double p = 0.5 * y2[i - 1] + 2.0;
            y2[i] = -0.5 / p;
            const double rhs = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / (dr * dr);
            u[i] = (3.0 * rhs - 0.5 * u[i - 1]) / p;
        }
        for (int k = n - 2; k >= 0; --k) {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }
    }

    /**
     * @brief Evaluate \f$f(r)\f$ and optionally \f$f'(r)\f$ at radius \f$r \ge 0\f$.
     *
     * @par Theory
     * Piecewise cubic Hermite interpolation using precomputed second-derivative coefficients.
     *
     * @par Algorithm
     * Computes fractional bin coordinates in \f$O(1)\f$ time with analytic first derivative.
     *
     * @param r Radial distance in Bohr.
     * @param dfdr Optional output pointer receiving \f$f'(r)\f$.
     * @param d2fdr2 Optional output pointer receiving \f$f''(r)\f$.
     * @return Interpolated function value \f$f(r)\f$.
     *
     * @par Invariants
     * Strictly vanishes for \f$r \ge r_{\text{cut}}\f$.
     *
     * @par SIESTA Mapping
     * Replaces \c splint in SIESTA \c radial.f.
     */
    double eval(double r, double* dfdr = nullptr, double* d2fdr2 = nullptr) const {
        if (r >= rcut || y.size() < 2) {
            if (dfdr) *dfdr = 0.0;
            if (d2fdr2) *d2fdr2 = 0.0;
            return 0.0;
        }
        const int n = static_cast<int>(y.size());
        const double pos = std::max(0.0, r) / dr;
        int klo = static_cast<int>(pos);
        if (klo >= n - 1) {
            if (dfdr) *dfdr = 0.0;
            if (d2fdr2) *d2fdr2 = 0.0;
            return 0.0;
        }
        const int khi = klo + 1;
        const double b = pos - static_cast<double>(klo);
        const double a = 1.0 - b;
        const double dr2_6 = (dr * dr) / 6.0;
        const double val =
            a * y[klo] + b * y[khi] +
            ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * dr2_6;
        if (dfdr) {
            *dfdr = (y[khi] - y[klo]) / dr +
                    (-(3.0 * a * a - 1.0) * y2[klo] + (3.0 * b * b - 1.0) * y2[khi]) * (dr / 6.0);
        }
        if (d2fdr2) {
            *d2fdr2 = a * y2[klo] + b * y2[khi];
        }
        return val;
    }
};

}  // namespace jsiesta::math
