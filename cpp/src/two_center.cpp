/**
 * @file two_center.cpp
 * @brief Compiled C++ Fourier-Bessel two-center integral engine (\c S, \c T, \c Vkb, \c E_NA_pair).
 *
 * @par Theory
 * Implements the spherical Bessel convolution theorem with real Gaunt coefficients
 * and singularity-free solid-harmonic factorization \f$C_{LM}(\mathbf{R}) \cdot [F_{l_1 l_2 L}(R)/R^L]\f$.
 *
 * @par Algorithm
 * Uses high-order Gauss-Legendre quadrature on \f$[-1, 1]\f$ for spherical Gaunt integrals,
 * dense \f$k\f$-space grids (\f$k_{\max} = 40\text{ Bohr}^{-1}\f$) for Hankel transforms,
 * and uniform radial cubic splines for \f$O(1)\f$ pair evaluation.
 *
 * @par Invariants
 * Thread-safe, zero global mutable state; identical radial quadrature for \f$S, T, P_{\mu k}\f$.
 *
 * @par SIESTA Mapping
 * Replaces \c matel.F90, \c kinefsm.F, \c naefs.F, and \c nlefsm.f.
 */

#include "jsiesta/two_center.hpp"
#include "jsiesta/math_utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kMaxL = 3;
constexpr int kMaxBigL = 2 * kMaxL;
constexpr int kNumK = 1024;
constexpr double kMaxK = 60.0;  // Bohr^-1; converged for 3s/3p semicore (review H2, D-017)
constexpr int kNumRSpline = 601;

struct RadialTransform {
    int l = 0;
    double rcut = 0.0;
    std::vector<double> f_k;  // f~(k) = int_0^rc r^2 f(r) j_l(k r) dr
};

void compute_k_grid(std::vector<double>& k_pts, std::vector<double>& k_wts) {
    k_pts.resize(kNumK);
    k_wts.resize(kNumK);
    // Slightly stretched / uniform Simpson-weighted grid on [0, kMaxK]
    const double dk = kMaxK / static_cast<double>(kNumK - 1);
    for (int i = 0; i < kNumK; ++i) {
        k_pts[i] = static_cast<double>(i) * dk;
        if (i == 0 || i == kNumK - 1) {
            k_wts[i] = 0.5 * dk;
        } else {
            k_wts[i] = dk;
        }
    }
}

RadialTransform hankel_transform(
    const double* f_r, int n_r, double r_max, double r_cut, int l,
    const std::vector<double>& k_pts) {
    RadialTransform rt;
    rt.l = l;
    rt.rcut = r_cut;
    rt.f_k.assign(k_pts.size(), 0.0);
    if (n_r < 2 || r_max <= 0.0) return rt;

    const double dr = r_max / static_cast<double>(n_r - 1);
    std::vector<double> jl(l + 1, 0.0);
    for (size_t ik = 0; ik < k_pts.size(); ++ik) {
        const double k = k_pts[ik];
        double sum = 0.0;
        for (int ir = 0; ir < n_r; ++ir) {
            const double r = static_cast<double>(ir) * dr;
            if (r > r_cut) break;
            const double w = (ir == 0 || ir == n_r - 1) ? 0.5 * dr : dr;
            jsiesta::math::spherical_bessel_jl(l, k * r, jl.data());
            sum += w * r * r * f_r[ir] * jl[l];
        }
        rt.f_k[ik] = sum;
    }
    return rt;
}

class RealGauntTable {
public:
    RealGauntTable() {
        const int n_orb_all = (kMaxL + 1) * (kMaxL + 1);
        const int n_big = (kMaxBigL + 1) * (kMaxBigL + 1);
        table_.assign(n_orb_all * n_orb_all * n_big, 0.0);

        // 16-point Gauss-Legendre in mu = cos(theta) in [-1, 1] (exact up to deg 31)
        const int n_mu = 16;
        std::vector<double> mu_pts(n_mu), mu_wts(n_mu);
        gauss_legendre(n_mu, mu_pts, mu_wts);

        const int n_phi = 32;
        const double dphi = 2.0 * jsiesta::math::kPi / static_cast<double>(n_phi);

        std::vector<double> c_lm(n_big);
        for (int imu = 0; imu < n_mu; ++imu) {
            const double z = mu_pts[imu];
            const double sin_t = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double w_mu = mu_wts[imu];
            for (int iphi = 0; iphi < n_phi; ++iphi) {
                const double phi = static_cast<double>(iphi) * dphi;
                const double x = sin_t * std::cos(phi);
                const double y = sin_t * std::sin(phi);
                jsiesta::math::real_solid_harmonics_and_grad(x, y, z, kMaxBigL, c_lm.data(), nullptr);
                const double w = w_mu * dphi;
                for (int a = 0; a < n_orb_all; ++a) {
                    const double ya = c_lm[a];
                    for (int b = 0; b < n_orb_all; ++b) {
                        const double yab = w * ya * c_lm[b];
                        double* dst = &table_[(a * n_orb_all + b) * n_big];
                        for (int c = 0; c < n_big; ++c) {
                            dst[c] += yab * c_lm[c];
                        }
                    }
                }
            }
        }
    }

    double get(int l1, int m1, int l2, int m2, int L, int M) const {
        const int n_orb_all = (kMaxL + 1) * (kMaxL + 1);
        const int n_big = (kMaxBigL + 1) * (kMaxBigL + 1);
        const int a = l1 * l1 + (m1 + l1);
        const int b = l2 * l2 + (m2 + l2);
        const int c = L * L + (M + L);
        return table_[(a * n_orb_all + b) * n_big + c];
    }

private:
    std::vector<double> table_;

    static void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
        const int m = (n + 1) / 2;
        for (int i = 0; i < m; ++i) {
            double z = std::cos(jsiesta::math::kPi * (i + 0.75) / (n + 0.5));
            double pp = 0.0;
            for (int iter = 0; iter < 50; ++iter) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 1; j <= n; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / static_cast<double>(j);
                }
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                const double z1 = z;
                z = z1 - p1 / pp;
                if (std::abs(z - z1) < 1e-15) break;
            }
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[n - 1 - i] = w[i];
        }
    }
};

const RealGauntTable& gaunt_table() {
    static const RealGauntTable tbl;
    return tbl;
}

struct PairRadialSplines {
    int l1 = 0;
    int l2 = 0;
    double rcut_pair = 0.0;
    // Splines for F_{l1 l2 L}(R) / R^L indexed by L in [|l1-l2|, l1+l2] (step 2)
    std::vector<int> L_vals;
    std::vector<jsiesta::math::UniformCubicSpline> splines;
};

PairRadialSplines build_pair_splines(
    const RadialTransform& rt1, const RadialTransform& rt2,
    const std::vector<double>& k_pts, const std::vector<double>& k_wts,
    bool is_kinetic) {
    PairRadialSplines prs;
    prs.l1 = rt1.l;
    prs.l2 = rt2.l;
    prs.rcut_pair = rt1.rcut + rt2.rcut;
    const int l_min = std::abs(rt1.l - rt2.l);
    const int l_max = rt1.l + rt2.l;
    const double r_max = std::max(prs.rcut_pair + 0.5, 1.0);
    const double dr = r_max / static_cast<double>(kNumRSpline - 1);

    for (int L = l_min; L <= l_max; L += 2) {
        prs.L_vals.push_back(L);
    }
    const int n_L = static_cast<int>(prs.L_vals.size());
    std::vector<std::vector<double>> curves(n_L, std::vector<double>(kNumRSpline, 0.0));
    std::vector<double> jl(l_max + 1, 0.0);

    // Precompute double factorial (2L + 1)!! for R -> 0 limit: j_L(kR) / R^L -> k^L / (2L+1)!!
    std::vector<double> dfact(l_max + 1, 1.0);
    for (int L = 1; L <= l_max; ++L) {
        dfact[L] = dfact[L - 1] * static_cast<double>(2 * L + 1);
    }

    for (int ir = 0; ir < kNumRSpline; ++ir) {
        const double R = static_cast<double>(ir) * dr;
        for (size_t ik = 0; ik < k_pts.size(); ++ik) {
            const double k = k_pts[ik];
            const double k_factor = is_kinetic ? (0.5 * k * k * k * k) : (k * k);
            const double w_prod = k_wts[ik] * k_factor * rt1.f_k[ik] * rt2.f_k[ik];
            if (R < 1e-6) {
                double k_pow = 1.0;
                for (int iL = 0; iL < n_L; ++iL) {
                    const int L = prs.L_vals[iL];
                    const double j_over_rL = std::pow(k, L) / dfact[L];
                    curves[iL][ir] += w_prod * j_over_rL;
                    (void)k_pow;
                }
            } else {
                jsiesta::math::spherical_bessel_jl(l_max, k * R, jl.data());
                for (int iL = 0; iL < n_L; ++iL) {
                    const int L = prs.L_vals[iL];
                    curves[iL][ir] += w_prod * (jl[L] / std::pow(R, L));
                }
            }
        }
    }

    prs.splines.resize(n_L);
    for (int iL = 0; iL < n_L; ++iL) {
        prs.splines[iL].build(curves[iL].data(), kNumRSpline, r_max, prs.rcut_pair);
    }
    return prs;
}

void eval_pair_element(
    const PairRadialSplines& prs,
    int m1, int m2,
    double rx, double ry, double rz,
    const double* c_lm, const double* dc_lm,
    double& out_val, double* out_grad) {
    const double R = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (R >= prs.rcut_pair) {
        out_val = 0.0;
        if (out_grad) {
            out_grad[0] = out_grad[1] = out_grad[2] = 0.0;
        }
        return;
    }
    const RealGauntTable& gt = gaunt_table();
    double val = 0.0;
    double gx = 0.0, gy = 0.0, gz = 0.0;
    const double inv_R = (R > 1e-12) ? (1.0 / R) : 0.0;
    const double ux = rx * inv_R, uy = ry * inv_R, uz = rz * inv_R;

    for (size_t iL = 0; iL < prs.L_vals.size(); ++iL) {
        const int L = prs.L_vals[iL];
        const int sign = (((prs.l2 - prs.l1 + L) / 2) & 1) ? -1 : 1;
        const double prefactor = 8.0 * static_cast<double>(sign);
        double dg_dr = 0.0;
        const double g_val = prs.splines[iL].eval(R, out_grad ? &dg_dr : nullptr);

        double ang_sum = 0.0;
        double d_ang_x = 0.0, d_ang_y = 0.0, d_ang_z = 0.0;
        for (int M = -L; M <= L; ++M) {
            const double G = gt.get(prs.l1, m1, prs.l2, m2, L, M);
            if (std::abs(G) < 1e-15) continue;
            const int c_idx = L * L + (M + L);
            ang_sum += G * c_lm[c_idx];
            if (out_grad) {
                d_ang_x += G * dc_lm[3 * c_idx + 0];
                d_ang_y += G * dc_lm[3 * c_idx + 1];
                d_ang_z += G * dc_lm[3 * c_idx + 2];
            }
        }
        val += prefactor * ang_sum * g_val;
        if (out_grad) {
            gx += prefactor * (d_ang_x * g_val + ang_sum * dg_dr * ux);
            gy += prefactor * (d_ang_y * g_val + ang_sum * dg_dr * uy);
            gz += prefactor * (d_ang_z * g_val + ang_sum * dg_dr * uz);
        }
    }
    out_val = val;
    if (out_grad) {
        out_grad[0] = gx;
        out_grad[1] = gy;
        out_grad[2] = gz;
    }
}

}  // namespace

extern "C" {

int32_t jsiesta_eval_two_center_st(
    int32_t n_shells,
    int32_t n_pao_pts,
    const double* pao_r_max,
    const double* pao_rcut,
    const int32_t* pao_l,
    const double* pao_values,
    int32_t n_edges,
    const int32_t* edge_shell1,
    const int32_t* edge_m1,
    const int32_t* edge_shell2,
    const int32_t* edge_m2,
    const double* edge_vectors,
    double* out_s,
    double* out_t,
    double* out_ds,
    double* out_dt) {
    if (n_shells <= 0 || n_pao_pts < 2 || n_edges < 0) return 1;

    std::vector<double> k_pts, k_wts;
    compute_k_grid(k_pts, k_wts);

    std::vector<RadialTransform> rts(n_shells);
    for (int s = 0; s < n_shells; ++s) {
        if (pao_l[s] > kMaxL) return 2;
        rts[s] = hankel_transform(
            pao_values + static_cast<size_t>(s) * n_pao_pts,
            n_pao_pts, pao_r_max[s], pao_rcut[s], pao_l[s], k_pts);
    }

    const int n_pairs = n_shells * n_shells;
    std::vector<PairRadialSplines> s_splines(n_pairs), t_splines(n_pairs);
    std::vector<bool> used_pairs(n_pairs, false);
    for (int e = 0; e < n_edges; ++e) {
        used_pairs[edge_shell1[e] * n_shells + edge_shell2[e]] = true;
    }
    for (int s1 = 0; s1 < n_shells; ++s1) {
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const int pid = s1 * n_shells + s2;
            if (!used_pairs[pid]) continue;
            s_splines[pid] = build_pair_splines(rts[s1], rts[s2], k_pts, k_wts, false);
            t_splines[pid] = build_pair_splines(rts[s1], rts[s2], k_pts, k_wts, true);
        }
    }

    const int n_big = (kMaxBigL + 1) * (kMaxBigL + 1);
    std::vector<double> c_lm(n_big), dc_lm(3 * n_big);
    for (int e = 0; e < n_edges; ++e) {
        const int s1 = edge_shell1[e];
        const int s2 = edge_shell2[e];
        const int pid = s1 * n_shells + s2;
        const double rx = edge_vectors[3 * e + 0];
        const double ry = edge_vectors[3 * e + 1];
        const double rz = edge_vectors[3 * e + 2];
        jsiesta::math::real_solid_harmonics_and_grad(rx, ry, rz, kMaxBigL, c_lm.data(), dc_lm.data());
        eval_pair_element(
            s_splines[pid], edge_m1[e], edge_m2[e], rx, ry, rz,
            c_lm.data(), dc_lm.data(), out_s[e], out_ds + 3 * e);
        eval_pair_element(
            t_splines[pid], edge_m1[e], edge_m2[e], rx, ry, rz,
            c_lm.data(), dc_lm.data(), out_t[e], out_dt + 3 * e);
    }
    return 0;
}

int32_t jsiesta_eval_kb_overlaps(
    int32_t n_pao_shells,
    int32_t n_pao_pts,
    const double* pao_r_max,
    const double* pao_rcut,
    const int32_t* pao_l,
    const double* pao_values,
    int32_t n_kb_shells,
    int32_t n_kb_pts,
    const double* kb_r_max,
    const double* kb_rcut,
    const int32_t* kb_l,
    const double* kb_values,
    int32_t n_links,
    const int32_t* link_pao_shell,
    const int32_t* link_m1,
    const int32_t* link_kb_shell,
    const int32_t* link_m2,
    const double* link_vectors,
    double* out_p,
    double* out_dp) {
    if (n_pao_shells <= 0 || n_kb_shells <= 0 || n_links < 0) return 1;

    std::vector<double> k_pts, k_wts;
    compute_k_grid(k_pts, k_wts);

    std::vector<RadialTransform> rts_pao(n_pao_shells), rts_kb(n_kb_shells);
    for (int s = 0; s < n_pao_shells; ++s) {
        if (pao_l[s] > kMaxL) return 2;
        rts_pao[s] = hankel_transform(
            pao_values + static_cast<size_t>(s) * n_pao_pts,
            n_pao_pts, pao_r_max[s], pao_rcut[s], pao_l[s], k_pts);
    }
    for (int k = 0; k < n_kb_shells; ++k) {
        if (kb_l[k] > kMaxL) return 2;
        rts_kb[k] = hankel_transform(
            kb_values + static_cast<size_t>(k) * n_kb_pts,
            n_kb_pts, kb_r_max[k], kb_rcut[k], kb_l[k], k_pts);
    }

    std::vector<PairRadialSplines> p_splines(n_pao_shells * n_kb_shells);
    std::vector<bool> used_links(n_pao_shells * n_kb_shells, false);
    for (int i = 0; i < n_links; ++i) {
        used_links[link_pao_shell[i] * n_kb_shells + link_kb_shell[i]] = true;
    }
    for (int s = 0; s < n_pao_shells; ++s) {
        for (int k = 0; k < n_kb_shells; ++k) {
            const int pid = s * n_kb_shells + k;
            if (!used_links[pid]) continue;
            p_splines[pid] = build_pair_splines(rts_pao[s], rts_kb[k], k_pts, k_wts, false);
        }
    }

    const int n_big = (kMaxBigL + 1) * (kMaxBigL + 1);
    std::vector<double> c_lm(n_big), dc_lm(3 * n_big);
    for (int i = 0; i < n_links; ++i) {
        const int s = link_pao_shell[i];
        const int k = link_kb_shell[i];
        const double rx = link_vectors[3 * i + 0];
        const double ry = link_vectors[3 * i + 1];
        const double rz = link_vectors[3 * i + 2];
        jsiesta::math::real_solid_harmonics_and_grad(rx, ry, rz, kMaxBigL, c_lm.data(), dc_lm.data());
        eval_pair_element(
            p_splines[s * n_kb_shells + k],
            link_m1[i], link_m2[i], rx, ry, rz,
            c_lm.data(), dc_lm.data(), out_p[i], out_dp + 3 * i);
    }
    return 0;
}

int32_t jsiesta_eval_na_pair(
    int32_t n_species,
    int32_t n_pts,
    const double* r_max,
    const double* rcut_na,
    const double* z_val,
    const double* rho_na,
    int32_t n_pairs,
    const int32_t* pair_sp1,
    const int32_t* pair_sp2,
    const double* pair_dist,
    double* out_e,
    double* out_de_dr) {
    if (n_species <= 0 || n_pts < 2 || n_pairs < 0) return 1;
    std::vector<double> k_pts, k_wts;
    compute_k_grid(k_pts, k_wts);

    std::vector<RadialTransform> rts(n_species);
    for (int s = 0; s < n_species; ++s) {
        rts[s] = hankel_transform(
            rho_na + static_cast<size_t>(s) * n_pts,
            n_pts, r_max[s], rcut_na[s], 0, k_pts);
    }

    std::vector<jsiesta::math::UniformCubicSpline> na_splines(n_species * n_species);
    for (int s1 = 0; s1 < n_species; ++s1) {
        for (int s2 = 0; s2 < n_species; ++s2) {
            const double rc_pair = rcut_na[s1] + rcut_na[s2];
            const double r_span = std::max(rc_pair + 0.5, 1.0);
            const double dr = r_span / static_cast<double>(kNumRSpline - 1);
            std::vector<double> curve(kNumRSpline, 0.0);
            for (int ir = 0; ir < kNumRSpline; ++ir) {
                const double R = static_cast<double>(ir) * dr;
                double sum = 0.0;
                for (size_t ik = 0; ik < k_pts.size(); ++ik) {
                    const double k = k_pts[ik];
                    double j0 = 1.0;
                    if (k * R >= 1e-6) {
                        j0 = std::sin(k * R) / (k * R);
                    } else {
                        const double kr2 = (k * R) * (k * R);
                        j0 = 1.0 - kr2 / 6.0;
                    }
                    // Smooth overlap term -32*pi * int f1(k) f2(k) j0(k R) dk; Z1*Z2/R is added analytically
                    sum += k_wts[ik] * (-rts[s1].f_k[ik] * rts[s2].f_k[ik]) * j0;
                }
                curve[ir] = 32.0 * jsiesta::math::kPi * sum;
            }
            na_splines[s1 * n_species + s2].build(curve.data(), kNumRSpline, r_span, rc_pair);
        }
    }

    for (int p = 0; p < n_pairs; ++p) {
        const int s1 = pair_sp1[p];
        const int s2 = pair_sp2[p];
        const double R = pair_dist[p];
        const double rc_pair = rcut_na[s1] + rcut_na[s2];
        if (R >= rc_pair || R <= 1e-12) {
            out_e[p] = 0.0;
            out_de_dr[p] = 0.0;
        } else {
            double dov_dr = 0.0;
            const double e_ov = na_splines[s1 * n_species + s2].eval(R, &dov_dr);
            const double z12 = z_val[s1] * z_val[s2];
            out_e[p] = (z12 / R) + e_ov;
            out_de_dr[p] = (-z12 / (R * R)) + dov_dr;
        }
    }
    return 0;
}

}  // extern "C"
