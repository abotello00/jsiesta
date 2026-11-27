/**
 * @file eri.cpp
 * @brief McMurchie-Davidson electron repulsion integrals over contracted real solid-harmonic Gaussian shells.
 *
 * @par Theory
 * See \c eri.hpp: Hermite expansion of shell pairs, Boys function, recursion for the Hermite integrals.
 *
 * @par Algorithm
 * The Cartesian expansion of the solid harmonics is derived at start-up from \c real_solid_harmonics_and_grad, so the
 * angular convention is the one of the rest of the library by construction. The Boys function is tabulated.
 *
 * @par Invariants
 * No global state other than tables built once, on first use, in a thread-safe way.
 *
 * @par SIESTA Mapping
 * None in SIESTA; libint in HONPAS.
 */

#include "jsiesta/eri.hpp"

#include "jsiesta/math_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace jsiesta::eri {

namespace {

constexpr double kPiEri = 3.14159265358979323846;
constexpr int kMaxPairL = 2 * kMaxL + 1;     // highest Hermite order of a pair (one more for a derivative)
constexpr int kMaxQuartetL = 4 * kMaxL + 1;  // highest Hermite order of a quartet (one derivative pair)

/// Cartesian form of the real solid harmonics of one \f$l\f$: \f$C_{lm} = \sum_k c_{mk} x^{a_k} y^{b_k} z^{c_k}\f$.
struct SolidHarmonic {
    int n_cart = 0;
    std::vector<std::array<int, 3>> powers;  // (a, b, c) with a + b + c = l
    std::vector<double> coef;                // (2l+1, n_cart)
};

std::array<SolidHarmonic, kMaxL + 1> build_solid_harmonics() {
    std::array<SolidHarmonic, kMaxL + 1> tables;
    const int n_lm = (kMaxL + 1) * (kMaxL + 1);
    std::vector<double> c_lm(n_lm);
    for (int l = 0; l <= kMaxL; ++l) {
        SolidHarmonic& tab = tables[l];
        for (int a = l; a >= 0; --a) {
            for (int b = l - a; b >= 0; --b) tab.powers.push_back({a, b, l - a - b});
        }
        const int n = static_cast<int>(tab.powers.size());
        tab.n_cart = n;
        // Solve M c = C_lm at n points for the monomial coefficients (Gaussian elimination, partial pivoting).
        std::vector<double> mat(n * n), rhs(n * (2 * l + 1));
        for (int ip = 0; ip < n; ++ip) {
            const double x = std::sin(1.0 + 2.3 * ip), y = std::cos(0.7 + 1.9 * ip), z = std::sin(2.1 + 3.1 * ip) + 0.3;
            for (int k = 0; k < n; ++k) {
                mat[ip * n + k] = std::pow(x, tab.powers[k][0]) * std::pow(y, tab.powers[k][1]) * std::pow(z, tab.powers[k][2]);
            }
            jsiesta::math::real_solid_harmonics_and_grad(x, y, z, kMaxL, c_lm.data(), nullptr);
            for (int m = 0; m < 2 * l + 1; ++m) rhs[ip * (2 * l + 1) + m] = c_lm[l * l + m];
        }
        const int nm = 2 * l + 1;
        for (int col = 0; col < n; ++col) {
            int piv = col;
            for (int r = col + 1; r < n; ++r) {
                if (std::abs(mat[r * n + col]) > std::abs(mat[piv * n + col])) piv = r;
            }
            if (piv != col) {
                for (int k = 0; k < n; ++k) std::swap(mat[col * n + k], mat[piv * n + k]);
                for (int m = 0; m < nm; ++m) std::swap(rhs[col * nm + m], rhs[piv * nm + m]);
            }
            const double inv = 1.0 / mat[col * n + col];
            for (int r = 0; r < n; ++r) {
                if (r == col) continue;
                const double f = mat[r * n + col] * inv;
                if (f == 0.0) continue;
                for (int k = col; k < n; ++k) mat[r * n + k] -= f * mat[col * n + k];
                for (int m = 0; m < nm; ++m) rhs[r * nm + m] -= f * rhs[col * nm + m];
            }
        }
        tab.coef.assign(nm * n, 0.0);
        for (int k = 0; k < n; ++k) {
            for (int m = 0; m < nm; ++m) {
                const double c = rhs[k * nm + m] / mat[k * n + k];
                tab.coef[m * n + k] = (std::abs(c) < 1e-11) ? 0.0 : c;
            }
        }
    }
    return tables;
}

const std::array<SolidHarmonic, kMaxL + 1>& solid_harmonics() {
    static const std::array<SolidHarmonic, kMaxL + 1> tables = build_solid_harmonics();
    return tables;
}

/// Hermite triples \f$(t,u,v)\f$, \f$t+u+v \le L\f$, in the storage order of \c ShellPair::e.
struct HermiteList {
    std::vector<std::array<int, 3>> tuv;
};

std::array<HermiteList, kMaxPairL + 1> build_hermite_lists() {
    std::array<HermiteList, kMaxPairL + 1> lists;
    for (int big_l = 0; big_l <= kMaxPairL; ++big_l) {
        for (int t = 0; t <= big_l; ++t) {
            for (int u = 0; u + t <= big_l; ++u) {
                for (int v = 0; v + u + t <= big_l; ++v) lists[big_l].tuv.push_back({t, u, v});
            }
        }
    }
    return lists;
}

const std::array<HermiteList, kMaxPairL + 1>& hermite_lists() {
    static const std::array<HermiteList, kMaxPairL + 1> lists = build_hermite_lists();
    return lists;
}

/// One-dimensional Hermite coefficients \f$E^{ij}_t\f$, \f$i \le l_a\f$, \f$j \le l_b\f$, \f$t \le i+j\f$.
void hermite_1d(int la, int lb, double p, double xpa, double xpb, double e00, double* e) {
    const int nt = la + lb + 1;
    const int sj = lb + 1;
    auto at = [&](int i, int j, int t) -> double& { return e[(i * sj + j) * nt + t]; };
    std::fill(e, e + (la + 1) * sj * nt, 0.0);
    const double half_inv_p = 0.5 / p;
    at(0, 0, 0) = e00;
    for (int i = 0; i < la; ++i) {
        for (int t = 0; t <= i + 1; ++t) {
            double val = xpa * at(i, 0, t);
            if (t > 0) val += half_inv_p * at(i, 0, t - 1);
            if (t + 1 <= i) val += (t + 1) * at(i, 0, t + 1);
            at(i + 1, 0, t) = val;
        }
    }
    for (int i = 0; i <= la; ++i) {
        for (int j = 0; j < lb; ++j) {
            for (int t = 0; t <= i + j + 1; ++t) {
                double val = xpb * at(i, j, t);
                if (t > 0) val += half_inv_p * at(i, j, t - 1);
                if (t + 1 <= i + j) val += (t + 1) * at(i, j, t + 1);
                at(i, j + 1, t) = val;
            }
        }
    }
}

}  // namespace

namespace {

/// Reference Boys function: series and downward recursion (\f$T \le 30\f$) or closed form and upward recursion.
void boys_reference(int n_max, double t, double* f) {
    if (t < 1e-13) {
        for (int n = 0; n <= n_max; ++n) f[n] = 1.0 / (2 * n + 1) - t / (2 * n + 3);
        return;
    }
    const double et = std::exp(-t);
    if (t > 30.0) {
        // Upward recursion is stable here: the amplification factor (2n+1)/(2T) is below one.
        f[0] = 0.5 * std::sqrt(kPiEri / t) * std::erf(std::sqrt(t));
        for (int n = 0; n < n_max; ++n) f[n + 1] = ((2 * n + 1) * f[n] - et) / (2.0 * t);
        return;
    }
    // Series for the highest order (all terms positive), then downward recursion.
    double term = 1.0 / (2 * n_max + 1);
    double sum = term;
    for (int k = 1; k < 400; ++k) {
        term *= 2.0 * t / (2 * n_max + 2 * k + 1);
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    f[n_max] = et * sum;
    for (int n = n_max; n > 0; --n) f[n - 1] = (2.0 * t * f[n] + et) / (2 * n - 1);
}

constexpr double kBoysStep = 0.05;
constexpr int kBoysPoints = 601;                    // T = 0 .. 30
constexpr int kBoysTaylor = 7;                      // terms beyond the value: error below (0.025)^8 / 8!
constexpr int kBoysOrders = kMaxQuartetL + 1 + kBoysTaylor;

/// \f$F_n(T_i)\f$ on the grid \f$T_i = 0.05\,i\f$, for the Taylor expansion \f$F_n(T_i+\delta) = \sum_k F_{n+k}(T_i)(-\delta)^k/k!\f$.
const std::vector<double>& boys_table() {
    static const std::vector<double> table = [] {
        std::vector<double> tab(static_cast<std::size_t>(kBoysPoints) * kBoysOrders);
        for (int i = 0; i < kBoysPoints; ++i) boys_reference(kBoysOrders - 1, i * kBoysStep, tab.data() + static_cast<std::size_t>(i) * kBoysOrders);
        return tab;
    }();
    return table;
}

}  // namespace

void boys(int n_max, double t, double* f) {
    if (t >= 30.0) {
        boys_reference(n_max, t, f);
        return;
    }
    static const double* table = boys_table().data();
    const int i = static_cast<int>(t / kBoysStep + 0.5);
    const double d = i * kBoysStep - t;  // minus delta
    const double* row = table + static_cast<std::size_t>(i) * kBoysOrders;
    static const double inv_k[kBoysTaylor + 1] = {0.0, 1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6, 1.0 / 7};
    for (int n = 0; n <= n_max; ++n) {
        double acc = row[n + kBoysTaylor];
        for (int k = kBoysTaylor; k > 0; --k) acc = row[n + k - 1] + acc * d * inv_k[k];
        f[n] = acc;
    }
}

void make_shell_pair(int la, int npa, const double* exp_a, const double* coef_a, const double* ca,
                     int lb, int npb, const double* exp_b, const double* coef_b, const double* cb,
                     double prim_tol, ShellPair& out) {
    make_shell_pair_mode(la, npa, exp_a, coef_a, ca, lb, npb, exp_b, coef_b, cb, prim_tol, PairMode::kValue, out);
}

void make_shell_pair_mode(int la, int npa, const double* exp_a, const double* coef_a, const double* ca,
                          int lb, int npb, const double* exp_b, const double* coef_b, const double* cb,
                          double prim_tol, PairMode mode, ShellPair& out) {
    const int imode = static_cast<int>(mode);
    const int d_first = (imode >= 1 && imode <= 3) ? imode - 1 : -1;   // derivative with respect to the first centre
    const int d_pair = (imode >= 4) ? imode - 4 : -1;                  // derivative with respect to a pair translation
    const int l_herm = la + lb + (imode == 0 ? 0 : 1);
    const int la_tab = la + (d_first >= 0 ? 1 : 0);
    const SolidHarmonic& sa = solid_harmonics()[la];
    const SolidHarmonic& sb = solid_harmonics()[lb];
    const auto& herm = hermite_lists()[l_herm].tuv;
    const int na = 2 * la + 1, nb = 2 * lb + 1;
    const int n_herm = static_cast<int>(herm.size());
    out.la = la;
    out.lb = lb;
    out.l_herm = l_herm;
    out.n_ab = na * nb;
    out.n_herm = n_herm;
    out.n_prim = 0;
    out.p.clear();
    out.centre.clear();
    out.e.clear();
    out.e_max.clear();

    const double ab[3] = {ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]};
    const double r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    const int n1d = (la_tab + 1) * (lb + 1) * (la_tab + lb + 1);
    std::vector<double> e1(3 * n1d);
    std::vector<double> cart(sa.n_cart * sb.n_cart * n_herm);
    std::vector<double> half(na * sb.n_cart * n_herm);
    std::vector<double> tmp;
    const int nt = la_tab + lb + 1, sj = lb + 1;

    for (int i = 0; i < npa; ++i) {
        for (int j = 0; j < npb; ++j) {
            const double a = exp_a[i], b = exp_b[j];
            const double p = a + b;
            const double mu = a * b / p;
            const double cc = coef_a[i] * coef_b[j];
            const double pref = std::exp(-mu * r2);
            if (std::abs(cc) * pref < prim_tol) continue;
            double pc[3];
            for (int d = 0; d < 3; ++d) {
                pc[d] = (a * ca[d] + b * cb[d]) / p;
                // The Gaussian prefactor is put into the x coefficients only.
                hermite_1d(la_tab, lb, p, pc[d] - ca[d], pc[d] - cb[d], d == 0 ? cc * pref : 1.0, e1.data() + d * n1d);
            }
            // One-dimensional factor of the Hermite coefficient in direction d for the powers (ia, ib):
            // the plain coefficient, its derivative with respect to the first centre
            // (d/dA x_A^i e^{-a x_A^2} = 2a x_A^{i+1} e^{..} - i x_A^{i-1} e^{..}), or the shift t -> t + 1 that
            // differentiates a Hermite Gaussian with respect to its centre.
            auto factor = [&](int d, int ia, int ib, int t) -> double {
                const double* tab = e1.data() + d * n1d;
                if (d == d_first) {
                    double val = (t <= ia + 1 + ib) ? 2.0 * a * tab[((ia + 1) * sj + ib) * nt + t] : 0.0;
                    if (ia > 0 && t <= ia - 1 + ib) val -= ia * tab[((ia - 1) * sj + ib) * nt + t];
                    return val;
                }
                if (d == d_pair) return (t >= 1 && t - 1 <= ia + ib) ? tab[(ia * sj + ib) * nt + t - 1] : 0.0;
                return (t <= ia + ib) ? tab[(ia * sj + ib) * nt + t] : 0.0;
            };
            for (int ka = 0; ka < sa.n_cart; ++ka) {
                const auto& pa = sa.powers[ka];
                for (int kb = 0; kb < sb.n_cart; ++kb) {
                    const auto& pb = sb.powers[kb];
                    double* dst = cart.data() + (ka * sb.n_cart + kb) * n_herm;
                    for (int h = 0; h < n_herm; ++h) {
                        const double fx = factor(0, pa[0], pb[0], herm[h][0]);
                        dst[h] = (fx == 0.0) ? 0.0
                                             : fx * factor(1, pa[1], pb[1], herm[h][1]) * factor(2, pa[2], pb[2], herm[h][2]);
                    }
                }
            }
            // Cartesian -> solid harmonics, first index then second.
            std::fill(half.begin(), half.end(), 0.0);
            for (int ma = 0; ma < na; ++ma) {
                for (int ka = 0; ka < sa.n_cart; ++ka) {
                    const double c = sa.coef[ma * sa.n_cart + ka];
                    if (c == 0.0) continue;
                    const double* src = cart.data() + ka * sb.n_cart * n_herm;
                    double* dst = half.data() + ma * sb.n_cart * n_herm;
                    for (int q = 0; q < sb.n_cart * n_herm; ++q) dst[q] += c * src[q];
                }
            }
            const std::size_t base = out.e.size();
            out.e.resize(base + static_cast<std::size_t>(na) * nb * n_herm, 0.0);
            for (int ma = 0; ma < na; ++ma) {
                for (int mb = 0; mb < nb; ++mb) {
                    double* dst = out.e.data() + base + (ma * nb + mb) * n_herm;
                    for (int kb = 0; kb < sb.n_cart; ++kb) {
                        const double c = sb.coef[mb * sb.n_cart + kb];
                        if (c == 0.0) continue;
                        const double* src = half.data() + (ma * sb.n_cart + kb) * n_herm;
                        for (int h = 0; h < n_herm; ++h) dst[h] += c * src[h];
                    }
                }
            }
            // Store as (n_herm, n_ab): the contractions of eri_quartet then run over contiguous memory.
            tmp.assign(out.e.begin() + static_cast<std::ptrdiff_t>(base), out.e.end());
            for (int ab = 0; ab < na * nb; ++ab) {
                for (int h = 0; h < n_herm; ++h) out.e[base + static_cast<std::size_t>(h) * na * nb + ab] = tmp[static_cast<std::size_t>(ab) * n_herm + h];
            }
            double e_max = 0.0;
            for (std::size_t q = base; q < out.e.size(); ++q) e_max = std::max(e_max, std::abs(out.e[q]));
            out.e_max.push_back(e_max);
            out.p.push_back(p);
            out.centre.insert(out.centre.end(), {pc[0], pc[1], pc[2]});
            ++out.n_prim;
        }
    }
}

// The contractions are short dense loops: a second copy of the function for AVX2 + FMA processors is selected at
// load time where the compiler supports it, the baseline copy runs elsewhere.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
__attribute__((target_clones("arch=x86-64-v3", "default")))
#endif
void eri_quartet_multi(const ShellPair* bras, int n_bra, const ShellPair& ket, const double* ket_shift,
                       const Kernel& kernel, double* out, std::vector<double>& work) {
    const ShellPair& bra = bras[0];
    const int lb_tot = bra.l_herm, lk_tot = ket.l_herm;
    const int big_l = lb_tot + lk_tot;
    const auto& hb = hermite_lists()[lb_tot].tuv;
    const auto& hk = hermite_lists()[lk_tot].tuv;
    const int nhb = bra.n_herm, nhk = ket.n_herm;
    const int n_ab = bra.n_ab, n_cd = ket.n_ab;
    const int dim = big_l + 1;
    const int cube = dim * dim * dim;

    // work = [ r: (big_l+1) cubes | v: (nhb, n_cd) | ket sign-folded offsets are computed on the fly ]
    const std::size_t need = static_cast<std::size_t>(dim) * cube + static_cast<std::size_t>(nhb) * n_cd;
    if (work.size() < need) work.resize(need);
    double* r = work.data();
    double* v = work.data() + static_cast<std::size_t>(dim) * cube;
    auto ridx = [dim](int t, int u, int w) { return (t * dim + u) * dim + w; };

    const std::size_t n_out = static_cast<std::size_t>(n_ab) * n_cd;
    std::fill(out, out + n_bra * n_out, 0.0);
    double fb[kMaxQuartetL + 1], fs[kMaxQuartetL + 1];
    const bool use_sr = (kernel.c_sr != 0.0) && (kernel.omega > 0.0);
    // w = (c_full + c_sr)/r - c_sr erf(omega r)/r
    const double w_coul = kernel.c_full + (use_sr ? kernel.c_sr : 0.0);
    const double two_pi_52 = 2.0 * std::pow(kPiEri, 2.5);

    for (int ip = 0; ip < bra.n_prim; ++ip) {
        const double p = bra.p[ip];
        const double* pc = bra.centre.data() + 3 * ip;
        std::fill(v, v + static_cast<std::size_t>(nhb) * n_cd, 0.0);
        for (int iq = 0; iq < ket.n_prim; ++iq) {
            const double q = ket.p[iq];
            const double* qc = ket.centre.data() + 3 * iq;
            const double x = pc[0] - (qc[0] + ket_shift[0]);
            const double y = pc[1] - (qc[1] + ket_shift[1]);
            const double z = pc[2] - (qc[2] + ket_shift[2]);
            const double alpha = p * q / (p + q);
            const double rr = x * x + y * y + z * z;
            const double pref = two_pi_52 / (p * q * std::sqrt(p + q));
            double e_bra = bra.e_max[ip];
            for (int m = 1; m < n_bra; ++m) e_bra = std::max(e_bra, bras[m].e_max[ip]);
            if (e_bra * ket.e_max[iq] * pref < kernel.prim_cut) continue;

            // Starting values R^n_000 for the combined kernel.
            if (w_coul != 0.0) {
                boys(big_l, alpha * rr, fb);
            } else {
                std::fill(fb, fb + big_l + 1, 0.0);
            }
            double pow_a = pref * w_coul;
            for (int n = 0; n <= big_l; ++n) {
                r[n * cube] = pow_a * fb[n];
                pow_a *= -2.0 * alpha;
            }
            if (use_sr) {
                const double w2 = kernel.omega * kernel.omega;
                const double alpha_s = alpha * w2 / (alpha + w2);
                boys(big_l, alpha_s * rr, fs);
                double pow_s = -pref * kernel.c_sr * std::sqrt(alpha_s / alpha);
                for (int n = 0; n <= big_l; ++n) {
                    r[n * cube] += pow_s * fs[n];
                    pow_s *= -2.0 * alpha_s;
                }
            }
            // R^n_{tuv} for t+u+v = d from R^{n+1} of lower degree.
            for (int d = 1; d <= big_l; ++d) {
                for (int n = 0; n <= big_l - d; ++n) {
                    const double* up = r + (n + 1) * cube;
                    double* cur = r + n * cube;
                    for (int t = 0; t <= d; ++t) {
                        for (int u = 0; u + t <= d; ++u) {
                            const int w = d - t - u;
                            double val;
                            if (t > 0) {
                                val = x * up[ridx(t - 1, u, w)];
                                if (t > 1) val += (t - 1) * up[ridx(t - 2, u, w)];
                            } else if (u > 0) {
                                val = y * up[ridx(t, u - 1, w)];
                                if (u > 1) val += (u - 1) * up[ridx(t, u - 2, w)];
                            } else {
                                val = z * up[ridx(t, u, w - 1)];
                                if (w > 1) val += (w - 1) * up[ridx(t, u, w - 2)];
                            }
                            cur[ridx(t, u, w)] = val;
                        }
                    }
                }
            }
            // v[h_bra, cd] += sum_{h_ket} (-1)^{tau+nu+phi} E^{cd}_{h_ket} R_{h_bra + h_ket}
            const double* ek = ket.e.data() + static_cast<std::size_t>(iq) * n_cd * nhk;
            for (int h1 = 0; h1 < nhb; ++h1) {
                const int t1 = hb[h1][0], u1 = hb[h1][1], w1 = hb[h1][2];
                double* vrow = v + static_cast<std::size_t>(h1) * n_cd;
                for (int h2 = 0; h2 < nhk; ++h2) {
                    const int t2 = hk[h2][0], u2 = hk[h2][1], w2 = hk[h2][2];
                    const double rv = (((t2 + u2 + w2) & 1) ? -1.0 : 1.0) * r[ridx(t1 + t2, u1 + u2, w1 + w2)];
                    const double* erow = ek + static_cast<std::size_t>(h2) * n_cd;
                    for (int cd = 0; cd < n_cd; ++cd) vrow[cd] += erow[cd] * rv;
                }
            }
        }
        // out[ab, cd] += sum_{h_bra} E^{ab}_{h_bra} v[h_bra, cd], for every bra expansion
        for (int m = 0; m < n_bra; ++m) {
            if (bras[m].n_prim == 0) continue;
            const double* eb = bras[m].e.data() + static_cast<std::size_t>(ip) * n_ab * nhb;
            double* out_m = out + m * n_out;
            for (int h1 = 0; h1 < nhb; ++h1) {
                const double* vrow = v + static_cast<std::size_t>(h1) * n_cd;
                const double* erow = eb + static_cast<std::size_t>(h1) * n_ab;
                for (int ab = 0; ab < n_ab; ++ab) {
                    const double c = erow[ab];
                    if (c == 0.0) continue;
                    double* orow = out_m + static_cast<std::size_t>(ab) * n_cd;
                    for (int cd = 0; cd < n_cd; ++cd) orow[cd] += c * vrow[cd];
                }
            }
        }
    }
}

void eri_quartet(const ShellPair& bra, const ShellPair& ket, const double* ket_shift, const Kernel& kernel,
                 double* out, std::vector<double>& work) {
    eri_quartet_multi(&bra, 1, ket, ket_shift, kernel, out, work);
}

}  // namespace jsiesta::eri

extern "C" {

int32_t jsiesta_eri_quartet(const int32_t* l, const int32_t* n_prim, const double* exps, const double* coefs,
                            const double* centres, double c_full, double c_sr, double omega, double* out) {
    return jsiesta_eri_quartet_mode(l, n_prim, exps, coefs, centres, c_full, c_sr, omega, 0, out);
}

int32_t jsiesta_eri_quartet_mode(const int32_t* l, const int32_t* n_prim, const double* exps, const double* coefs,
                                 const double* centres, double c_full, double c_sr, double omega, int32_t bra_mode,
                                 double* out) {
    if (bra_mode < 0 || bra_mode > 6) return 2;
    int off[4];
    int acc = 0;
    for (int s = 0; s < 4; ++s) {
        if (l[s] < 0 || l[s] > jsiesta::eri::kMaxL || n_prim[s] <= 0) return 2;
        off[s] = acc;
        acc += n_prim[s];
    }
    jsiesta::eri::ShellPair bra, ket;
    jsiesta::eri::make_shell_pair_mode(l[0], n_prim[0], exps + off[0], coefs + off[0], centres,
                                       l[1], n_prim[1], exps + off[1], coefs + off[1], centres + 3, 0.0,
                                       static_cast<jsiesta::eri::PairMode>(bra_mode), bra);
    jsiesta::eri::make_shell_pair(l[2], n_prim[2], exps + off[2], coefs + off[2], centres + 6,
                                  l[3], n_prim[3], exps + off[3], coefs + off[3], centres + 9, 0.0, ket);
    jsiesta::eri::Kernel kernel;
    kernel.c_full = c_full;
    kernel.c_sr = c_sr;
    kernel.omega = omega;
    const double zero[3] = {0.0, 0.0, 0.0};
    std::vector<double> work;
    jsiesta::eri::eri_quartet(bra, ket, zero, kernel, out, work);
    return 0;
}

}  // extern "C"
