/**
 * @file grid_ffi.cpp
 * @brief Compiled C++ XLA FFI kernels for \c rhoofd, \c vmat, \c dfscf, and neutral-atom grid scatter.
 *
 * @par Theory
 * Implements uniform 3D periodic real-space orbital evaluations inside finite spherical
 * support \f$|\mathbf{r}_p - \mathbf{R}_I - \mathbf{A}\mathbf{T}| < r_c(\mu)\f$, ensuring
 * \f$O(N_{\text{grid}})\f$ complexity and exact discrete adjointness between density scatter
 * and potential gather.
 *
 * @par Algorithm
 * 1. Invert \f$\mathbf{A}\f$ once per call to determine fractional coordinates and bounding
 *    box half-widths \f$\Delta f_\alpha = r_c \|\mathbf{A}^{-1}_{\alpha,:}\|_2\f$.
 * 2. Populate compact per-grid-point active orbital lists \f$(p \to \{(\mu, \mathbf{T}_\mu, \phi, \nabla\phi)\})\f$
 *    by looping only inside each atom's periodic spherical boxes.
 * 3. Contract active pairs against CSR \c SparsePattern for \c rhoofd, \c vmat, and \c dfscf. When the dense
 *    atom-pair block table fits, the active orbitals are grouped per (atom, image) and contracted with dense DM
 *    blocks on worker threads; this path serves \c rhoofd, \c vmat and (since D-034) the \c dfscf backward with
 *    orbital gradients. The meta-GGA backward (Hessians) uses the per-orbital-pair loop with edge hashing.
 *
 * @par Invariants
 * - Thread-safe internal LRU cache keyed by exact geometry/basis/sparsity bytes; purely functional outputs.
 * - Exact discrete identity: \f$\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]\f$.
 *
 * @par SIESTA Mapping
 * Replaces \c rhoofd.F90, \c vmat.F90, \c dfscf.F, and \c dhscf.F orbital-grid loops.
 */

#include "jsiesta/grid_xc_ffi.hpp"
#include "jsiesta/grid_plan.hpp"
#include "jsiesta/math_utils.hpp"
#include "jsiesta/threading.hpp"
#include "xla/ffi/api/ffi.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ffi = xla::ffi;

namespace {

using jsiesta::threading::run_threads;

inline int resolve_num_threads(int64_t n_grid) { return jsiesta::threading::thread_count(n_grid, 512); }

inline void mat3_inv(const double* a_col, double* inv_row_major, double& det_out) {
    // a_col is stored row-major in memory? Wait: Python/JAX (3, 3) array cell[alpha, j]
    // where column j is lattice vector a_j: A_{alpha, j} = cell[3 * alpha + j].
    const double a00 = a_col[0], a01 = a_col[1], a02 = a_col[2];
    const double a10 = a_col[3], a11 = a_col[4], a12 = a_col[5];
    const double a20 = a_col[6], a21 = a_col[7], a22 = a_col[8];
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a12 * a20 - a10 * a22;
    const double c02 = a10 * a21 - a11 * a20;
    const double c10 = a02 * a21 - a01 * a22;
    const double c11 = a00 * a22 - a02 * a20;
    const double c12 = a01 * a20 - a00 * a21;
    const double c20 = a01 * a12 - a02 * a11;
    const double c21 = a02 * a10 - a00 * a12;
    const double c22 = a00 * a11 - a01 * a10;
    det_out = a00 * c00 + a01 * c01 + a02 * c02;
    const double inv_det = 1.0 / det_out;
    inv_row_major[0] = c00 * inv_det; inv_row_major[1] = c10 * inv_det; inv_row_major[2] = c20 * inv_det;
    inv_row_major[3] = c01 * inv_det; inv_row_major[4] = c11 * inv_det; inv_row_major[5] = c21 * inv_det;
    inv_row_major[6] = c02 * inv_det; inv_row_major[7] = c12 * inv_det; inv_row_major[8] = c22 * inv_det;
}

inline int floor_mod(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}

inline int floor_div(int i, int n) {
    int r = i % n;
    int q = i / n;
    if (r < 0) --q;
    return q;
}

struct ActiveOrbEntry {
    int32_t orb;
    int32_t s0;
    int32_t s1;
    int32_t s2;
    double val;
    double gx;
    double gy;
    double gz;
    double hxx;
    double hxy;
    double hxz;
    double hyy;
    double hyz;
    double hzz;
};

struct ActiveGridCache {
    std::vector<int32_t> grid_offsets;
    std::vector<ActiveOrbEntry> entries;
    double dvol = 0.0;
    int64_t n_grid = 0;
};

// Build active orbital values (and optionally gradients/Hessians wrt d = r_p - R_I - A T) per grid point.
ActiveGridCache build_active_grid_cache(
    const double* positions,   // (n_atoms, 3)
    const double* cell,        // (3, 3) column vectors A_{alpha, j}
    const double* pao_r_max,   // (n_shells,)
    const double* pao_rcut,    // (n_shells,)
    const int32_t* pao_l,      // (n_shells,)
    const double* pao_values,  // (n_shells, n_pao_pts)
    const int32_t* orb_atom,   // (n_orb,)
    const int32_t* orb_g_shell,// (n_orb,)
    const int32_t* orb_m,      // (n_orb,)
    const double* orb_rcut,    // (n_orb,)
    int32_t n_atoms,
    int32_t n_orb,
    int32_t n_shells,
    int32_t n_pao_pts,
    int32_t n1, int32_t n2, int32_t n3,
    bool need_grad,
    bool need_hess = false) {

    ActiveGridCache cache;
    const int64_t n_grid = static_cast<int64_t>(n1) * n2 * n3;
    cache.n_grid = n_grid;

    double inv_cell[9], det = 0.0;
    mat3_inv(cell, inv_cell, det);
    cache.dvol = std::abs(det) / static_cast<double>(n_grid);

    // Build radial cubic splines for R_s(r) / r^l to avoid 0/0 singularities at r = 0
    std::vector<jsiesta::math::UniformCubicSpline> rad_splines(n_shells);
    int l_max_all = 0;
    for (int s = 0; s < n_shells; ++s) {
        const int l = pao_l[s];
        l_max_all = std::max(l_max_all, l);
        const double r_max = pao_r_max[s];
        const double dr = r_max / static_cast<double>(n_pao_pts - 1);
        const double* src = pao_values + static_cast<size_t>(s) * n_pao_pts;
        std::vector<double> f_over_rl(n_pao_pts, 0.0);
        if (l == 0) {
            f_over_rl.assign(src, src + n_pao_pts);
        } else {
            for (int ir = 1; ir < n_pao_pts; ++ir) {
                const double r = static_cast<double>(ir) * dr;
                f_over_rl[ir] = src[ir] / std::pow(r, l);
            }
            // Quadratic extrapolation at r = 0 (since R_l(r)/r^l is even in r): f(0) = (4 f(dr) - f(2dr))/3
            if (n_pao_pts >= 3) {
                f_over_rl[0] = (4.0 * f_over_rl[1] - f_over_rl[2]) / 3.0;
            }
        }
        rad_splines[s].build(f_over_rl.data(), n_pao_pts, r_max, pao_rcut[s]);
    }

    // Group orbitals by atom
    std::vector<std::vector<int32_t>> atom_orbs(n_atoms);
    std::vector<double> atom_max_rc(n_atoms, 0.0);
    for (int32_t mu = 0; mu < n_orb; ++mu) {
        const int ia = orb_atom[mu];
        atom_orbs[ia].push_back(mu);
        atom_max_rc[ia] = std::max(atom_max_rc[ia], orb_rcut[mu]);
    }

    double b_norms[3];
    for (int ax = 0; ax < 3; ++ax) {
        b_norms[ax] = std::sqrt(
            inv_cell[3 * ax + 0] * inv_cell[3 * ax + 0] +
            inv_cell[3 * ax + 1] * inv_cell[3 * ax + 1] +
            inv_cell[3 * ax + 2] * inv_cell[3 * ax + 2]);
    }

    // Two sweeps over the same loops: the first counts the entries of every grid point, the second writes them in
    // place (same order as one sweep with a scatter, without a second copy of all entries in memory). Each thread
    // takes a contiguous range of atoms with its own per-point counters; the counters of thread t become write
    // positions after those of threads 0 .. t - 1, so the entries of a point keep the serial (atom-ascending) order.
    const int n_sh_all = (l_max_all + 1) * (l_max_all + 1);
    const int n_thr = std::max(1, std::min(resolve_num_threads(n_grid), static_cast<int>(n_atoms)));
    std::vector<std::vector<int32_t>> cnt(static_cast<size_t>(n_thr), std::vector<int32_t>(n_grid, 0));

    auto sweep = [&](const bool fill, const int t) {
    std::vector<int32_t>& pos_t = cnt[static_cast<size_t>(t)];
    std::vector<double> c_lm(n_sh_all);
    std::vector<double> dc_lm((need_grad || need_hess) ? 3 * n_sh_all : 1);
    std::vector<double> hc_lm(need_hess ? 9 * n_sh_all : 1);
    const int32_t a_lo = static_cast<int32_t>(static_cast<int64_t>(n_atoms) * t / n_thr);
    const int32_t a_hi = static_cast<int32_t>(static_cast<int64_t>(n_atoms) * (t + 1) / n_thr);
    for (int32_t ia = a_lo; ia < a_hi; ++ia) {
        const double rc_a = atom_max_rc[ia];
        if (rc_a <= 0.0 || atom_orbs[ia].empty()) continue;

        const double px = positions[3 * ia + 0];
        const double py = positions[3 * ia + 1];
        const double pz = positions[3 * ia + 2];

        const double fc0 = inv_cell[0] * px + inv_cell[1] * py + inv_cell[2] * pz;
        const double fc1 = inv_cell[3] * px + inv_cell[4] * py + inv_cell[5] * pz;
        const double fc2 = inv_cell[6] * px + inv_cell[7] * py + inv_cell[8] * pz;

        // Loop over grid points in the unwrapped bounding box around fractional position fc:
        // Any grid point r_p + A*T within rc_a of R_I corresponds to an unwrapped grid index (i0, i1, i2)
        // in [ceil(N_a * (fc_a - df_a)), floor(N_a * (fc_a + df_a))], where T = floor_div(i_a, N_a)!
        const int i0_min = static_cast<int>(std::ceil(n1 * (fc0 - rc_a * b_norms[0])));
        const int i0_max = static_cast<int>(std::floor(n1 * (fc0 + rc_a * b_norms[0])));
        const int i1_min = static_cast<int>(std::ceil(n2 * (fc1 - rc_a * b_norms[1])));
        const int i1_max = static_cast<int>(std::floor(n2 * (fc1 + rc_a * b_norms[1])));
        const int i2_min = static_cast<int>(std::ceil(n3 * (fc2 - rc_a * b_norms[2])));
        const int i2_max = static_cast<int>(std::floor(n3 * (fc2 + rc_a * b_norms[2])));

        for (int i0 = i0_min; i0 <= i0_max; ++i0) {
            const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
            const int g0 = floor_mod(i0, n1);
            const int t0 = -floor_div(i0, n1);  // R_orb_img = R_I + A * T where T = -floor(i/N) => d = r_p - R_orb_img
            for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
                const int g1 = floor_mod(i1, n2);
                const int t1 = -floor_div(i1, n2);
                for (int i2 = i2_min; i2 <= i2_max; ++i2) {
                    const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
                    const double rx = cell[0] * f0 + cell[1] * f1 + cell[2] * f2;
                    const double ry = cell[3] * f0 + cell[4] * f1 + cell[5] * f2;
                    const double rz = cell[6] * f0 + cell[7] * f1 + cell[8] * f2;

                    const double dx = rx - px;
                    const double dy = ry - py;
                    const double dz = rz - pz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_a * rc_a) continue;

                    const double r = std::sqrt(r2);
                    const int g2 = floor_mod(i2, n3);
                    const int t2 = -floor_div(i2, n3);
                    const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;
                    if (!fill) {
                        for (int32_t mu : atom_orbs[ia]) {
                            if (r < orb_rcut[mu]) pos_t[p_idx] += 1;
                        }
                        continue;
                    }

                    jsiesta::math::real_solid_harmonics_and_grad(
                        dx, dy, dz, l_max_all, c_lm.data(),
                        (need_grad || need_hess) ? dc_lm.data() : nullptr,
                        need_hess ? hc_lm.data() : nullptr);

                    for (int32_t mu : atom_orbs[ia]) {
                        if (r >= orb_rcut[mu]) continue;
                        const int sh = orb_g_shell[mu];
                        const int l = pao_l[sh];
                        const int m = orb_m[mu];
                        const int lm_idx = l * l + (m + l);
                        double dfdr = 0.0, d2fdr2 = 0.0;
                        const double g_val = rad_splines[sh].eval(
                            r,
                            (need_grad || need_hess) ? &dfdr : nullptr,
                            need_hess ? &d2fdr2 : nullptr);
                        const double y_poly = c_lm[lm_idx];
                        const double phi = g_val * y_poly;
                        double gx = 0.0, gy = 0.0, gz = 0.0;
                        double hxx = 0.0, hxy = 0.0, hxz = 0.0, hyy = 0.0, hyz = 0.0, hzz = 0.0;
                        if (need_grad || need_hess) {
                            // d/dd [g(r) * C_lm(d)] = (g'(r)/r) * d * C_lm(d) + g(r) * grad C_lm(d)
                            // Note at r->0, g'(r)/r -> finite g''(0) because g(r) = R_l(r)/r^l is even!
                            double g_over_r = 0.0;
                            if (r > 1e-8) {
                                g_over_r = dfdr / r;
                            } else {
                                double df_eps = 0.0;
                                rad_splines[sh].eval(1e-4, &df_eps);
                                g_over_r = df_eps / 1e-4;
                            }
                            const double dcx = dc_lm[3 * lm_idx + 0];
                            const double dcy = dc_lm[3 * lm_idx + 1];
                            const double dcz = dc_lm[3 * lm_idx + 2];
                            gx = g_over_r * dx * y_poly + g_val * dcx;
                            gy = g_over_r * dy * y_poly + g_val * dcy;
                            gz = g_over_r * dz * y_poly + g_val * dcz;
                            if (need_hess) {
                                double s_r = 0.0;
                                if (r > 1e-5) {
                                    s_r = (d2fdr2 - g_over_r) / (r * r);
                                }
                                const double* hlm = hc_lm.data() + 9 * lm_idx;
                                hxx = s_r * dx * dx * y_poly + g_over_r * (y_poly + 2.0 * dx * dcx) + g_val * hlm[0];
                                hxy = s_r * dx * dy * y_poly + g_over_r * (dx * dcy + dy * dcx) + g_val * hlm[1];
                                hxz = s_r * dx * dz * y_poly + g_over_r * (dx * dcz + dz * dcx) + g_val * hlm[2];
                                hyy = s_r * dy * dy * y_poly + g_over_r * (y_poly + 2.0 * dy * dcy) + g_val * hlm[4];
                                hyz = s_r * dy * dz * y_poly + g_over_r * (dy * dcz + dz * dcy) + g_val * hlm[5];
                                hzz = s_r * dz * dz * y_poly + g_over_r * (y_poly + 2.0 * dz * dcz) + g_val * hlm[8];
                            }
                        }
                        cache.entries[static_cast<size_t>(pos_t[p_idx]++)] =
                            ActiveOrbEntry{mu, t0, t1, t2, phi, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz};
                    }
                }
            }
        }
    }
    };

    run_threads(n_thr, [&](int t) { sweep(false, t); });
    cache.grid_offsets.resize(n_grid + 1, 0);
    for (int64_t p = 0; p < n_grid; ++p) {
        int32_t running = cache.grid_offsets[p];
        for (int t = 0; t < n_thr; ++t) {
            const int32_t c = cnt[static_cast<size_t>(t)][p];
            cnt[static_cast<size_t>(t)][p] = running;
            running += c;
        }
        cache.grid_offsets[p + 1] = running;
    }
    cache.entries.resize(static_cast<size_t>(cache.grid_offsets[n_grid]));
    run_threads(n_thr, [&](int t) { sweep(true, t); });
    return cache;
}

struct EdgeHashMap {
    std::vector<uint64_t> keys;
    std::vector<int32_t> vals;
    uint32_t mask = 0;

    static inline uint64_t pack_key(int32_t mu, int32_t nu, int32_t dt0, int32_t dt1, int32_t dt2) {
        uint64_t h = static_cast<uint64_t>(static_cast<uint16_t>(mu));
        h = (h << 14) ^ static_cast<uint64_t>(static_cast<uint16_t>(nu));
        h = (h << 12) ^ static_cast<uint64_t>(dt0 + 512);
        h = (h << 12) ^ static_cast<uint64_t>(dt1 + 512);
        h = (h << 12) ^ static_cast<uint64_t>(dt2 + 512);
        return h + 1ULL;  // never 0 (0 is empty sentinel)
    }

    static inline uint64_t mix64(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    void build(
        int32_t n_orb,
        const int32_t* row_ptr,
        const int32_t* dst_unit_orb,
        const int32_t* edge_cell_idx,
        const int32_t* cell_shifts) {
        const int32_t n_edges = row_ptr[n_orb];
        uint32_t cap = 16;
        while (cap < static_cast<uint32_t>(n_edges) * 2U + 16U) cap <<= 1;
        mask = cap - 1;
        keys.assign(cap, 0ULL);
        vals.assign(cap, -1);
        for (int32_t mu = 0; mu < n_orb; ++mu) {
            for (int32_t e = row_ptr[mu]; e < row_ptr[mu + 1]; ++e) {
                const int32_t nu = dst_unit_orb[e];
                const int32_t c = edge_cell_idx[e];
                const uint64_t k = pack_key(
                    mu, nu, cell_shifts[3 * c + 0], cell_shifts[3 * c + 1], cell_shifts[3 * c + 2]);
                uint32_t slot = static_cast<uint32_t>(mix64(k)) & mask;
                while (keys[slot] != 0ULL) slot = (slot + 1U) & mask;
                keys[slot] = k;
                vals[slot] = e;
            }
        }
    }

    inline int32_t find(int32_t mu, int32_t nu, int32_t dt0, int32_t dt1, int32_t dt2) const {
        const uint64_t k = pack_key(mu, nu, dt0, dt1, dt2);
        uint32_t slot = static_cast<uint32_t>(mix64(k)) & mask;
        while (keys[slot] != 0ULL) {
            if (keys[slot] == k) return vals[slot];
            slot = (slot + 1U) & mask;
        }
        return -1;
    }
};


struct GridAtomImage {
    int32_t atom;
    int32_t s0;
    int32_t s1;
    int32_t s2;
    int32_t val_offset;
};

struct AtomPairBlock {
    int32_t atom_a;
    int32_t atom_b;
    int32_t c_idx;
    int32_t blk_scalar_off;  // offset into packed (noa * nob) buffer
    int32_t edge_list_off;   // offset into compact active_edges list
    int32_t n_valid_edges;
};

struct BlockEdgeMapEntry {
    int32_t blk_rel;  // index inside atom_a * nob + atom_b block
    int32_t e_idx;    // sparse edge index
};

struct GridKernelPlan {
    // Key arrays for exact cache validation
    int64_t n1 = 0, n2 = 0, n3 = 0;
    int32_t n_atoms = 0, n_orb = 0, n_shells = 0, n_pao_pts = 0, n_edges = 0, n_shifts = 0;
    bool need_grad = false;
    bool need_hess = false;  // orbital Hessians stored (implies need_grad); used by the meta-GGA backward
    uint64_t hash = 0;       // cache key of this variant (inputs and the two flags)
    uint64_t base_hash = 0;  // hash of the inputs alone
    std::vector<double> key_positions, key_cell, key_pao_rmax, key_pao_rcut, key_pao_vals, key_orb_rcut;
    std::vector<int32_t> key_pao_l, key_orb_atom, key_orb_gsh, key_orb_m;
    std::vector<int32_t> key_row_ptr, key_dst_orb, key_eci, key_shifts;

    ActiveGridCache active;
    EdgeHashMap edge_map;

    bool use_atom_blocks = false;
    std::vector<int32_t> atom_n_orbs;
    std::vector<int32_t> orb_local;
    int32_t s_min[3] = {0, 0, 0}, s_max[3] = {0, 0, 0};
    int32_t s_dim[3] = {1, 1, 1};
    std::vector<int32_t> shift_box;       // (s_dim[0]*s_dim[1]*s_dim[2]) -> c_idx or -1
    std::vector<int32_t> pair_block_id;   // ((atom_a*n_atoms + atom_b)*n_shifts + c_idx) -> block_id or -1
    std::vector<AtomPairBlock> blocks;
    std::vector<BlockEdgeMapEntry> active_block_edges;
    size_t total_blk_scalars = 0;

    std::vector<int32_t> grid_img_offsets;
    std::vector<GridAtomImage> img_groups;
    std::vector<double> img_vals;
    std::vector<double> img_gx;
    std::vector<double> img_gy;
    std::vector<double> img_gz;
    std::vector<double> img_hxx, img_hxy, img_hxz, img_hyy, img_hyz, img_hzz;  // need_hess only

    inline int32_t lookup_shift(int32_t ds0, int32_t ds1, int32_t ds2) const {
        const int32_t i0 = ds0 - s_min[0];
        if (static_cast<uint32_t>(i0) >= static_cast<uint32_t>(s_dim[0])) return -1;
        const int32_t i1 = ds1 - s_min[1];
        if (static_cast<uint32_t>(i1) >= static_cast<uint32_t>(s_dim[1])) return -1;
        const int32_t i2 = ds2 - s_min[2];
        if (static_cast<uint32_t>(i2) >= static_cast<uint32_t>(s_dim[2])) return -1;
        return shift_box[(static_cast<size_t>(i0) * s_dim[1] + i1) * s_dim[2] + i2];
    }

    inline int32_t lookup_block(int32_t atom_a, int32_t atom_b, int32_t c_idx) const {
        return pair_block_id[(static_cast<size_t>(atom_a) * n_atoms + atom_b) * n_shifts + c_idx];
    }

    template <typename T>
    static uint64_t hash_bytes(uint64_t seed, const T* ptr, size_t count) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(ptr);
        const size_t nbytes = count * sizeof(T);
        uint64_t h = seed ^ 0x9e3779b97f4a7c15ULL;
        const size_t n8 = nbytes / 8;
        for (size_t i = 0; i < n8; ++i) {
            uint64_t w;
            std::memcpy(&w, bytes + i * 8, 8);
            h = EdgeHashMap::mix64(h ^ w);
        }
        uint64_t tail = 0;
        for (size_t i = n8 * 8; i < nbytes; ++i) {
            tail = (tail << 8) | bytes[i];
        }
        return EdgeHashMap::mix64(h ^ tail ^ nbytes);
    }

    static uint64_t compute_input_hash(
        const double* positions, const double* cell,
        const double* pao_r_max, const double* pao_rcut, const int32_t* pao_l, const double* pao_values,
        const int32_t* orb_atom, const int32_t* orb_g_shell, const int32_t* orb_m, const double* orb_rcut,
        const int32_t* row_ptr, const int32_t* dst_unit_orb, const int32_t* edge_cell_idx, const int32_t* cell_shifts,
        int32_t n_atoms, int32_t n_orb, int32_t n_shells, int32_t n_pao_pts,
        int32_t n_edges, int32_t n_shifts,
        int64_t n1, int64_t n2, int64_t n3, bool need_grad, bool need_hess) {
        return with_flags(
            compute_base_hash(positions, cell, pao_r_max, pao_rcut, pao_l, pao_values, orb_atom, orb_g_shell, orb_m,
                              orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts, n_atoms, n_orb, n_shells,
                              n_pao_pts, n_edges, n_shifts, n1, n2, n3),
            need_grad, need_hess);
    }

    static uint64_t compute_base_hash(
        const double* positions, const double* cell,
        const double* pao_r_max, const double* pao_rcut, const int32_t* pao_l, const double* pao_values,
        const int32_t* orb_atom, const int32_t* orb_g_shell, const int32_t* orb_m, const double* orb_rcut,
        const int32_t* row_ptr, const int32_t* dst_unit_orb, const int32_t* edge_cell_idx, const int32_t* cell_shifts,
        int32_t n_atoms, int32_t n_orb, int32_t n_shells, int32_t n_pao_pts,
        int32_t n_edges, int32_t n_shifts,
        int64_t n1, int64_t n2, int64_t n3) {
        uint64_t h = static_cast<uint64_t>(n1) * 1315423911ULL ^ static_cast<uint64_t>(n2) * 2654435761ULL ^
                     static_cast<uint64_t>(n3);
        h = hash_bytes(h, positions, static_cast<size_t>(n_atoms) * 3);
        h = hash_bytes(h, cell, 9);
        h = hash_bytes(h, pao_r_max, n_shells);
        h = hash_bytes(h, pao_rcut, n_shells);
        h = hash_bytes(h, pao_l, n_shells);
        h = hash_bytes(h, pao_values, static_cast<size_t>(n_shells) * n_pao_pts);
        h = hash_bytes(h, orb_atom, n_orb);
        h = hash_bytes(h, orb_g_shell, n_orb);
        h = hash_bytes(h, orb_m, n_orb);
        h = hash_bytes(h, orb_rcut, n_orb);
        h = hash_bytes(h, row_ptr, n_orb + 1);
        h = hash_bytes(h, dst_unit_orb, n_edges);
        h = hash_bytes(h, edge_cell_idx, n_edges);
        h = hash_bytes(h, cell_shifts, static_cast<size_t>(n_shifts) * 3);
        return h;
    }

    // the stored-arrays flags enter last, so one hash of the inputs gives the key of every plan variant
    static uint64_t with_flags(uint64_t base, bool need_grad, bool need_hess) {
        return EdgeHashMap::mix64(base ^ (need_grad ? 0x55555555ULL : 0ULL) ^
                                  (need_hess ? 0xAAAAAAAA00000000ULL : 0ULL));
    }

    template <typename T>
    static bool vec_eq(const std::vector<T>& v, const T* ptr) {
        return v.empty() || std::memcmp(v.data(), ptr, v.size() * sizeof(T)) == 0;
    }

    bool matches(
        uint64_t h,
        const double* positions, const double* cell,
        const double* pao_r_max, const double* pao_rcut, const int32_t* pao_l, const double* pao_values,
        const int32_t* orb_atom, const int32_t* orb_g_shell, const int32_t* orb_m, const double* orb_rcut,
        const int32_t* row_ptr, const int32_t* dst_unit_orb, const int32_t* edge_cell_idx, const int32_t* cell_shifts,
        int32_t na, int32_t no, int32_t ns, int32_t npts, int32_t ne, int32_t nsh,
        int64_t g1, int64_t g2, int64_t g3, bool grad, bool hess) const {
        // h is the input hash; a plan that stores more arrays than requested also matches
        if (base_hash != h || n1 != g1 || n2 != g2 || n3 != g3 ||
            n_atoms != na || n_orb != no || n_shells != ns || n_pao_pts != npts ||
            n_edges != ne || n_shifts != nsh || (grad && !need_grad) || (hess && !need_hess)) {
            return false;
        }
        return vec_eq(key_positions, positions) && vec_eq(key_cell, cell) &&
               vec_eq(key_pao_rmax, pao_r_max) && vec_eq(key_pao_rcut, pao_rcut) &&
               vec_eq(key_pao_l, pao_l) && vec_eq(key_pao_vals, pao_values) &&
               vec_eq(key_orb_atom, orb_atom) && vec_eq(key_orb_gsh, orb_g_shell) &&
               vec_eq(key_orb_m, orb_m) && vec_eq(key_orb_rcut, orb_rcut) &&
               vec_eq(key_row_ptr, row_ptr) && vec_eq(key_dst_orb, dst_unit_orb) &&
               vec_eq(key_eci, edge_cell_idx) && vec_eq(key_shifts, cell_shifts);
    }
};

std::shared_ptr<const GridKernelPlan> get_or_build_grid_plan(
    const double* positions, const double* cell,
    const double* pao_r_max, const double* pao_rcut, const int32_t* pao_l, const double* pao_values,
    const int32_t* orb_atom, const int32_t* orb_g_shell, const int32_t* orb_m, const double* orb_rcut,
    const int32_t* row_ptr, const int32_t* dst_unit_orb, const int32_t* edge_cell_idx, const int32_t* cell_shifts,
    int32_t n_atoms, int32_t n_orb, int32_t n_shells, int32_t n_pao_pts,
    int32_t n_edges, int32_t n_shifts,
    int64_t n1, int64_t n2, int64_t n3, bool need_grad, bool need_hess = false) {

    if (need_hess) need_grad = true;
    static std::mutex cache_mu;
    static std::vector<std::shared_ptr<const GridKernelPlan>> lru;

    const uint64_t bh = GridKernelPlan::compute_base_hash(
        positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut,
        row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3);
    const uint64_t h = GridKernelPlan::with_flags(bh, need_grad, need_hess);

    {
        std::lock_guard<std::mutex> lock(cache_mu);
        for (size_t idx = 0; idx < lru.size(); ++idx) {
            if (lru[idx]->matches(
                    bh, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
                    orb_atom, orb_g_shell, orb_m, orb_rcut,
                    row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
                    n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3, need_grad, need_hess)) {
                auto found = lru[idx];
                if (idx != 0) {
                    lru.erase(lru.begin() + static_cast<std::ptrdiff_t>(idx));
                    lru.insert(lru.begin(), found);
                }
                return found;
            }
        }
        // cached plans of the same inputs that store fewer arrays are superseded by the one built below; drop them
        // first, so they do not add to the peak memory of the build
        lru.erase(std::remove_if(lru.begin(), lru.end(),
                                 [&](const std::shared_ptr<const GridKernelPlan>& q) {
                                     return q->base_hash == bh && (need_grad || !q->need_grad) &&
                                            (need_hess || !q->need_hess);
                                 }),
                  lru.end());
    }

    auto plan = std::make_shared<GridKernelPlan>();
    plan->n1 = n1; plan->n2 = n2; plan->n3 = n3;
    plan->n_atoms = n_atoms; plan->n_orb = n_orb; plan->n_shells = n_shells;
    plan->n_pao_pts = n_pao_pts; plan->n_edges = n_edges; plan->n_shifts = n_shifts;
    plan->need_grad = need_grad; plan->need_hess = need_hess; plan->hash = h; plan->base_hash = bh;

    plan->key_positions.assign(positions, positions + static_cast<size_t>(n_atoms) * 3);
    plan->key_cell.assign(cell, cell + 9);
    plan->key_pao_rmax.assign(pao_r_max, pao_r_max + n_shells);
    plan->key_pao_rcut.assign(pao_rcut, pao_rcut + n_shells);
    plan->key_pao_l.assign(pao_l, pao_l + n_shells);
    plan->key_pao_vals.assign(pao_values, pao_values + static_cast<size_t>(n_shells) * n_pao_pts);
    plan->key_orb_atom.assign(orb_atom, orb_atom + n_orb);
    plan->key_orb_gsh.assign(orb_g_shell, orb_g_shell + n_orb);
    plan->key_orb_m.assign(orb_m, orb_m + n_orb);
    plan->key_orb_rcut.assign(orb_rcut, orb_rcut + n_orb);
    plan->key_row_ptr.assign(row_ptr, row_ptr + n_orb + 1);
    plan->key_dst_orb.assign(dst_unit_orb, dst_unit_orb + n_edges);
    plan->key_eci.assign(edge_cell_idx, edge_cell_idx + n_edges);
    plan->key_shifts.assign(cell_shifts, cell_shifts + static_cast<size_t>(n_shifts) * 3);

    plan->active = build_active_grid_cache(
        positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut,
        n_atoms, n_orb, n_shells, n_pao_pts, n1, n2, n3, need_grad, need_hess);
    plan->edge_map.build(n_orb, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts);

    // Build dense atom-image & atom-pair block representation for forward rhoofd/tauofd/vmat when compact
    if (n_shifts > 0 && n_atoms > 0 &&
        static_cast<int64_t>(n_atoms) * n_atoms * n_shifts <= 64000000LL) {
        int32_t smin[3] = {cell_shifts[0], cell_shifts[1], cell_shifts[2]};
        int32_t smax[3] = {cell_shifts[0], cell_shifts[1], cell_shifts[2]};
        for (int32_t c = 1; c < n_shifts; ++c) {
            for (int ax = 0; ax < 3; ++ax) {
                const int32_t sv = cell_shifts[3 * c + ax];
                smin[ax] = std::min(smin[ax], sv);
                smax[ax] = std::max(smax[ax], sv);
            }
        }
        const int64_t b0 = static_cast<int64_t>(smax[0]) - smin[0] + 1;
        const int64_t b1 = static_cast<int64_t>(smax[1]) - smin[1] + 1;
        const int64_t b2 = static_cast<int64_t>(smax[2]) - smin[2] + 1;
        if (b0 > 0 && b1 > 0 && b2 > 0 &&
            b0 <= 262144LL && b1 <= 262144LL && b2 <= 262144LL &&
            b0 * b1 <= 262144LL && b0 * b1 * b2 <= 262144LL) {
            plan->use_atom_blocks = true;
            for (int ax = 0; ax < 3; ++ax) {
                plan->s_min[ax] = smin[ax];
                plan->s_max[ax] = smax[ax];
                plan->s_dim[ax] = smax[ax] - smin[ax] + 1;
            }
            plan->shift_box.assign(static_cast<size_t>(b0 * b1 * b2), -1);
            for (int32_t c = 0; c < n_shifts; ++c) {
                const int32_t i0 = cell_shifts[3 * c + 0] - smin[0];
                const int32_t i1 = cell_shifts[3 * c + 1] - smin[1];
                const int32_t i2 = cell_shifts[3 * c + 2] - smin[2];
                plan->shift_box[(static_cast<size_t>(i0) * plan->s_dim[1] + i1) * plan->s_dim[2] + i2] = c;
            }

            plan->atom_n_orbs.assign(n_atoms, 0);
            plan->orb_local.assign(n_orb, 0);
            for (int32_t mu = 0; mu < n_orb; ++mu) {
                const int32_t ia = orb_atom[mu];
                plan->orb_local[mu] = plan->atom_n_orbs[ia]++;
            }

            const size_t n_pair_slots = static_cast<size_t>(n_atoms) * n_atoms * n_shifts;
            plan->pair_block_id.assign(n_pair_slots, -1);

            // Partition CSR edges into (atom_a, atom_b, c_idx) blocks
            std::vector<std::vector<BlockEdgeMapEntry>> tmp_block_edges;
            for (int32_t mu = 0; mu < n_orb; ++mu) {
                const int32_t ia = orb_atom[mu];
                const int32_t la = plan->orb_local[mu];
                for (int32_t e = row_ptr[mu]; e < row_ptr[mu + 1]; ++e) {
                    const int32_t nu = dst_unit_orb[e];
                    const int32_t ib = orb_atom[nu];
                    const int32_t lb = plan->orb_local[nu];
                    const int32_t c = edge_cell_idx[e];
                    const size_t slot = (static_cast<size_t>(ia) * n_atoms + ib) * n_shifts + c;
                    int32_t bid = plan->pair_block_id[slot];
                    if (bid < 0) {
                        bid = static_cast<int32_t>(plan->blocks.size());
                        plan->pair_block_id[slot] = bid;
                        AtomPairBlock blk{};
                        blk.atom_a = ia;
                        blk.atom_b = ib;
                        blk.c_idx = c;
                        blk.blk_scalar_off = static_cast<int32_t>(plan->total_blk_scalars);
                        plan->total_blk_scalars +=
                            static_cast<size_t>(plan->atom_n_orbs[ia]) * plan->atom_n_orbs[ib];
                        plan->blocks.push_back(blk);
                        tmp_block_edges.emplace_back();
                    }
                    const int32_t nob = plan->atom_n_orbs[ib];
                    tmp_block_edges[bid].push_back({la * nob + lb, e});
                }
            }
            for (size_t bid = 0; bid < plan->blocks.size(); ++bid) {
                plan->blocks[bid].edge_list_off = static_cast<int32_t>(plan->active_block_edges.size());
                plan->blocks[bid].n_valid_edges = static_cast<int32_t>(tmp_block_edges[bid].size());
                plan->active_block_edges.insert(
                    plan->active_block_edges.end(),
                    tmp_block_edges[bid].begin(), tmp_block_edges[bid].end());
            }

            // Group active orbital entries per grid point by (atom, s0, s1, s2). Two passes over the points, threaded
            // over point ranges: the first counts the images and values of every point, so the per-image arrays are
            // allocated once at their exact size; the second fills them at the prefix-sum offsets. Same output as a
            // serial pass.
            const int64_t n_pts = plan->active.n_grid;
            struct TempRef {
                int32_t atom;
                int32_t s0, s1, s2;
                int32_t entry_idx;
            };
            auto sorted_refs = [&](int64_t p, std::vector<TempRef>& refs) {
                const int32_t beg = plan->active.grid_offsets[p];
                const int32_t end = plan->active.grid_offsets[p + 1];
                refs.clear();
                for (int32_t idx = beg; idx < end; ++idx) {
                    const auto& e = plan->active.entries[idx];
                    refs.push_back({orb_atom[e.orb], e.s0, e.s1, e.s2, idx});
                }
                std::sort(refs.begin(), refs.end(), [](const TempRef& x, const TempRef& y) {
                    if (x.atom != y.atom) return x.atom < y.atom;
                    if (x.s0 != y.s0) return x.s0 < y.s0;
                    if (x.s1 != y.s1) return x.s1 < y.s1;
                    return x.s2 < y.s2;
                });
            };
            auto same_image = [](const TempRef& x, const TempRef& y) {
                return x.atom == y.atom && x.s0 == y.s0 && x.s1 == y.s1 && x.s2 == y.s2;
            };
            const int n_thr_pts = resolve_num_threads(n_pts);
            auto point_range = [&](int t, int64_t& p0, int64_t& p1) {
                p0 = n_pts * t / n_thr_pts;
                p1 = n_pts * (t + 1) / n_thr_pts;
            };
            std::vector<int64_t> grp_off(static_cast<size_t>(n_pts) + 1, 0), val_off(static_cast<size_t>(n_pts) + 1, 0);
            run_threads(n_thr_pts, [&](int t) {
                std::vector<TempRef> refs;
                int64_t p0 = 0, p1 = 0;
                point_range(t, p0, p1);
                for (int64_t p = p0; p < p1; ++p) {
                    sorted_refs(p, refs);
                    int64_t ng = 0, nv = 0;
                    for (size_t k = 0; k < refs.size(); ++k) {
                        if (k == 0 || !same_image(refs[k], refs[k - 1])) {
                            ++ng;
                            nv += plan->atom_n_orbs[refs[k].atom];
                        }
                    }
                    grp_off[static_cast<size_t>(p) + 1] = ng;
                    val_off[static_cast<size_t>(p) + 1] = nv;
                }
            });
            for (int64_t p = 0; p < n_pts; ++p) {
                grp_off[static_cast<size_t>(p) + 1] += grp_off[static_cast<size_t>(p)];
                val_off[static_cast<size_t>(p) + 1] += val_off[static_cast<size_t>(p)];
            }
            const int64_t n_groups = grp_off[static_cast<size_t>(n_pts)];
            const int64_t n_vals_tot = val_off[static_cast<size_t>(n_pts)];
            constexpr int64_t kInt32Max = 2147483647;
            if (n_groups > kInt32Max || n_vals_tot > kInt32Max) {
                // image and value offsets are int32: keep the generic representation (entries are not freed)
                plan->use_atom_blocks = false;
            } else {
                plan->grid_img_offsets.assign(static_cast<size_t>(n_pts) + 1, 0);
                for (int64_t p = 0; p <= n_pts; ++p) {
                    plan->grid_img_offsets[static_cast<size_t>(p)] = static_cast<int32_t>(grp_off[static_cast<size_t>(p)]);
                }
                const size_t nv = static_cast<size_t>(n_vals_tot);
                plan->img_groups.resize(static_cast<size_t>(n_groups));
                plan->img_vals.assign(nv, 0.0);
                if (need_grad) {
                    plan->img_gx.assign(nv, 0.0);
                    plan->img_gy.assign(nv, 0.0);
                    plan->img_gz.assign(nv, 0.0);
                }
                if (need_hess) {
                    for (auto* hv : {&plan->img_hxx, &plan->img_hxy, &plan->img_hxz,
                                     &plan->img_hyy, &plan->img_hyz, &plan->img_hzz}) {
                        hv->assign(nv, 0.0);
                    }
                }
                run_threads(n_thr_pts, [&](int t) {
                    std::vector<TempRef> refs;
                    int64_t p0 = 0, p1 = 0;
                    point_range(t, p0, p1);
                    for (int64_t p = p0; p < p1; ++p) {
                        sorted_refs(p, refs);
                        int64_t g = grp_off[static_cast<size_t>(p)];
                        int64_t v = val_off[static_cast<size_t>(p)];
                        size_t k = 0;
                        while (k < refs.size()) {
                            const TempRef head = refs[k];
                            const int32_t voff = static_cast<int32_t>(v);
                            while (k < refs.size() && same_image(refs[k], head)) {
                                const auto& e = plan->active.entries[refs[k].entry_idx];
                                const size_t idx_off = static_cast<size_t>(voff + plan->orb_local[e.orb]);
                                plan->img_vals[idx_off] = e.val;
                                if (need_grad) {
                                    plan->img_gx[idx_off] = e.gx;
                                    plan->img_gy[idx_off] = e.gy;
                                    plan->img_gz[idx_off] = e.gz;
                                }
                                if (need_hess) {
                                    plan->img_hxx[idx_off] = e.hxx;
                                    plan->img_hxy[idx_off] = e.hxy;
                                    plan->img_hxz[idx_off] = e.hxz;
                                    plan->img_hyy[idx_off] = e.hyy;
                                    plan->img_hyz[idx_off] = e.hyz;
                                    plan->img_hzz[idx_off] = e.hzz;
                                }
                                ++k;
                            }
                            plan->img_groups[static_cast<size_t>(g)] = {head.atom, head.s0, head.s1, head.s2, voff};
                            ++g;
                            v += plan->atom_n_orbs[head.atom];
                        }
                    }
                });
            }
            // The per-entry arrays are only read by the generic path (no atom blocks); free them.
            if (plan->use_atom_blocks) std::vector<ActiveOrbEntry>().swap(plan->active.entries);
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache_mu);
        lru.insert(lru.begin(), plan);
        if (lru.size() > 4) lru.resize(4);
        if (std::getenv("JSIESTA_GRID_PLAN_VERBOSE") != nullptr) {
            auto bytes = [](const GridKernelPlan& q) {
                return q.active.entries.capacity() * sizeof(ActiveOrbEntry) +
                       q.img_groups.capacity() * sizeof(GridAtomImage) +
                       (q.img_vals.capacity() + q.img_gx.capacity() + q.img_gy.capacity() + q.img_gz.capacity() +
                        q.img_hxx.capacity() + q.img_hxy.capacity() + q.img_hxz.capacity() + q.img_hyy.capacity() +
                        q.img_hyz.capacity() + q.img_hzz.capacity()) * sizeof(double);
            };
            std::size_t total = 0;
            for (const auto& q : lru) total += bytes(*q);
            const std::size_t n_entries =
                plan->active.grid_offsets.empty() ? 0 : static_cast<std::size_t>(plan->active.grid_offsets.back());
            std::fprintf(stderr,
                         "jsiesta: host grid plan built: %.2f GB kept, %.2f GB of per-entry arrays during the build "
                         "(grad=%d hess=%d); cache %.2f GB in %zu plans\n",
                         bytes(*plan) / 1e9, n_entries * sizeof(ActiveOrbEntry) / 1e9, need_grad ? 1 : 0,
                         need_hess ? 1 : 0, total / 1e9, lru.size());
        }
    }
    return plan;
}

ffi::Error ScatterOrbGridImpl(
    bool is_tau,
    ffi::Buffer<ffi::DataType::F64> dm_edges,
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut,
    ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values,
    ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell,
    ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb,
    ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_grid) {

    const int32_t n_spin_mat = (nspin == 1) ? 1 : (nspin == 2 ? 2 : 4);
    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_orb = static_cast<int32_t>(orb_atom.dimensions()[0]);
    const int32_t n_shells = static_cast<int32_t>(pao_l.dimensions()[0]);
    const int32_t n_pao_pts = static_cast<int32_t>(pao_values.dimensions()[1]);
    const int32_t n_edges = static_cast<int32_t>(dm_edges.dimensions()[1]);
    const int32_t n_shifts = static_cast<int32_t>(cell_shifts.dimensions()[0]);

    const auto plan = get_or_build_grid_plan(
        positions.typed_data(), cell.typed_data(),
        pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
        orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
        row_ptr.typed_data(), dst_unit_orb.typed_data(), edge_cell_idx.typed_data(), cell_shifts.typed_data(),
        n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3, is_tau);

    const double* dm_ptr = dm_edges.typed_data();
    double* out_ptr = out_grid->typed_data();
    const int64_t n_grid = plan->active.n_grid;
    std::fill(out_ptr, out_ptr + static_cast<size_t>(n_spin_mat) * n_grid, 0.0);

    const int n_threads = resolve_num_threads(n_grid);
    const double scale = is_tau ? 0.5 : 1.0;

    if (plan->use_atom_blocks) {
        const size_t blk_stride = plan->total_blk_scalars;
        std::vector<double> dm_blk(blk_stride * static_cast<size_t>(n_spin_mat), 0.0);
        for (const auto& blk : plan->blocks) {
            const int32_t end_e = blk.edge_list_off + blk.n_valid_edges;
            for (int32_t ei = blk.edge_list_off; ei < end_e; ++ei) {
                const auto& me = plan->active_block_edges[ei];
                for (int32_t s = 0; s < n_spin_mat; ++s) {
                    dm_blk[static_cast<size_t>(s) * blk_stride + blk.blk_scalar_off + me.blk_rel] =
                        dm_ptr[static_cast<size_t>(s) * n_edges + me.e_idx];
                }
            }
        }
        const int n_comp = is_tau ? 3 : 1;
        const double* comp_vecs[3] = {
            is_tau ? plan->img_gx.data() : plan->img_vals.data(),
            is_tau ? plan->img_gy.data() : nullptr,
            is_tau ? plan->img_gz.data() : nullptr,
        };

        auto worker = [&](int64_t p_lo, int64_t p_hi) {
            for (int64_t p = p_lo; p < p_hi; ++p) {
                const int32_t beg = plan->grid_img_offsets[p];
                const int32_t end = plan->grid_img_offsets[p + 1];
                if (n_spin_mat == 1) {
                    double acc0 = 0.0;
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            const double* d0 = dm_blk.data() + plan->blocks[bid].blk_scalar_off;
                            for (int comp = 0; comp < n_comp; ++comp) {
                                const double* va = comp_vecs[comp] + ga.val_offset;
                                const double* vb = comp_vecs[comp] + gb.val_offset;
                                for (int a = 0; a < noa; ++a) {
                                    const double* row0 = d0 + a * nob;
                                    double rs0 = 0.0;
                                    for (int b = 0; b < nob; ++b) rs0 += row0[b] * vb[b];
                                    acc0 += va[a] * rs0;
                                }
                            }
                        }
                    }
                    out_ptr[p] = scale * acc0;
                } else if (n_spin_mat == 2) {
                    double acc0 = 0.0, acc1 = 0.0;
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            const double* d0 = dm_blk.data() + plan->blocks[bid].blk_scalar_off;
                            const double* d1 = d0 + blk_stride;
                            for (int comp = 0; comp < n_comp; ++comp) {
                                const double* va = comp_vecs[comp] + ga.val_offset;
                                const double* vb = comp_vecs[comp] + gb.val_offset;
                                for (int a = 0; a < noa; ++a) {
                                    const double* r0 = d0 + a * nob;
                                    const double* r1 = d1 + a * nob;
                                    double rs0 = 0.0, rs1 = 0.0;
                                    for (int b = 0; b < nob; ++b) {
                                        const double pb = vb[b];
                                        rs0 += r0[b] * pb;
                                        rs1 += r1[b] * pb;
                                    }
                                    const double pa = va[a];
                                    acc0 += pa * rs0;
                                    acc1 += pa * rs1;
                                }
                            }
                        }
                    }
                    out_ptr[p] = scale * acc0;
                    out_ptr[n_grid + p] = scale * acc1;
                } else {
                    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            const double* d0 = dm_blk.data() + plan->blocks[bid].blk_scalar_off;
                            const double* d1 = d0 + blk_stride;
                            const double* d2 = d1 + blk_stride;
                            const double* d3 = d2 + blk_stride;
                            for (int comp = 0; comp < n_comp; ++comp) {
                                const double* va = comp_vecs[comp] + ga.val_offset;
                                const double* vb = comp_vecs[comp] + gb.val_offset;
                                for (int a = 0; a < noa; ++a) {
                                    const double* r0 = d0 + a * nob;
                                    const double* r1 = d1 + a * nob;
                                    const double* r2 = d2 + a * nob;
                                    const double* r3 = d3 + a * nob;
                                    double rs0 = 0.0, rs1 = 0.0, rs2 = 0.0, rs3 = 0.0;
                                    for (int b = 0; b < nob; ++b) {
                                        const double pb = vb[b];
                                        rs0 += r0[b] * pb;
                                        rs1 += r1[b] * pb;
                                        rs2 += r2[b] * pb;
                                        rs3 += r3[b] * pb;
                                    }
                                    const double pa = va[a];
                                    acc0 += pa * rs0;
                                    acc1 += pa * rs1;
                                    acc2 += pa * rs2;
                                    acc3 += pa * rs3;
                                }
                            }
                        }
                    }
                    out_ptr[p] = scale * acc0;
                    out_ptr[n_grid + p] = scale * acc1;
                    out_ptr[2 * n_grid + p] = scale * acc2;
                    out_ptr[3 * n_grid + p] = scale * acc3;
                }
            }
        };

        if (n_threads <= 1) {
            worker(0, n_grid);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(n_threads);
            const int64_t chunk = (n_grid + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; ++t) {
                const int64_t p_lo = t * chunk;
                const int64_t p_hi = std::min(n_grid, p_lo + chunk);
                if (p_lo < p_hi) pool.emplace_back(worker, p_lo, p_hi);
            }
            for (auto& th : pool) th.join();
        }
        return ffi::Error::Success();
    }

    // Fallback generic path using cached ActiveGridCache & EdgeHashMap
    const auto& cache = plan->active;
    const auto& edge_map = plan->edge_map;
    for (int64_t p = 0; p < cache.n_grid; ++p) {
        const int32_t beg = cache.grid_offsets[p];
        const int32_t end = cache.grid_offsets[p + 1];
        for (int32_t ia = beg; ia < end; ++ia) {
            const auto& ea = cache.entries[ia];
            for (int32_t ib = beg; ib < end; ++ib) {
                const auto& eb = cache.entries[ib];
                const int32_t e_idx = edge_map.find(
                    ea.orb, eb.orb, eb.s0 - ea.s0, eb.s1 - ea.s1, eb.s2 - ea.s2);
                if (e_idx < 0) continue;
                const double prod = is_tau
                    ? 0.5 * (ea.gx * eb.gx + ea.gy * eb.gy + ea.gz * eb.gz)
                    : (ea.val * eb.val);
                for (int32_t s = 0; s < n_spin_mat; ++s) {
                    out_ptr[static_cast<size_t>(s) * cache.n_grid + p] +=
                        dm_ptr[static_cast<size_t>(s) * n_edges + e_idx] * prod;
                }
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error RhoofdImpl(
    ffi::Buffer<ffi::DataType::F64> dm_edges, ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell, ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut, ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values, ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell, ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut, ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb, ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_rho) {
    return ScatterOrbGridImpl(
        false, dm_edges, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n1, n2, n3, nspin, out_rho);
}

ffi::Error TauofdImpl(
    ffi::Buffer<ffi::DataType::F64> dm_edges, ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell, ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut, ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values, ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell, ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut, ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb, ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_tau) {
    return ScatterOrbGridImpl(
        true, dm_edges, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n1, n2, n3, nspin, out_tau);
}

ffi::Error GatherVmatGridImpl(
    bool is_tau,
    ffi::Buffer<ffi::DataType::F64> v_grid,
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut,
    ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values,
    ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell,
    ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb,
    ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sparse) {

    const int32_t n_spin_mat = (nspin == 1) ? 1 : (nspin == 2 ? 2 : 4);
    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_orb = static_cast<int32_t>(orb_atom.dimensions()[0]);
    const int32_t n_shells = static_cast<int32_t>(pao_l.dimensions()[0]);
    const int32_t n_pao_pts = static_cast<int32_t>(pao_values.dimensions()[1]);
    const int32_t n_edges = static_cast<int32_t>(out_v_sparse->dimensions()[1]);
    const int32_t n_shifts = static_cast<int32_t>(cell_shifts.dimensions()[0]);

    const auto plan = get_or_build_grid_plan(
        positions.typed_data(), cell.typed_data(),
        pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
        orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
        row_ptr.typed_data(), dst_unit_orb.typed_data(), edge_cell_idx.typed_data(), cell_shifts.typed_data(),
        n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3, is_tau);

    const double* vg_ptr = v_grid.typed_data();
    double* vs_ptr = out_v_sparse->typed_data();
    const int64_t n_grid = plan->active.n_grid;
    const double dvol_eff = plan->active.dvol * (is_tau ? 0.5 : 1.0);
    std::fill(vs_ptr, vs_ptr + static_cast<size_t>(n_spin_mat) * n_edges, 0.0);

    const int n_threads = resolve_num_threads(n_grid);

    if (plan->use_atom_blocks) {
        const size_t blk_stride = plan->total_blk_scalars;
        const size_t one_thread_bytes = blk_stride * static_cast<size_t>(n_spin_mat) * sizeof(double);
        const size_t max_tls_bytes = 256ULL * 1024ULL * 1024ULL;
        const int max_safe_threads = one_thread_bytes == 0
            ? 1
            : static_cast<int>(std::max<size_t>(1, max_tls_bytes / one_thread_bytes));
        const int actual_threads = std::max(1, std::min(n_threads, max_safe_threads));
        std::vector<std::vector<double>> tls_vs(
            actual_threads, std::vector<double>(blk_stride * static_cast<size_t>(n_spin_mat), 0.0));
        const int n_comp = is_tau ? 3 : 1;
        const double* comp_vecs[3] = {
            is_tau ? plan->img_gx.data() : plan->img_vals.data(),
            is_tau ? plan->img_gy.data() : nullptr,
            is_tau ? plan->img_gz.data() : nullptr,
        };

        auto worker = [&](int t_id, int64_t p_lo, int64_t p_hi) {
            double* my_vs = tls_vs[t_id].data();
            for (int64_t p = p_lo; p < p_hi; ++p) {
                const int32_t beg = plan->grid_img_offsets[p];
                const int32_t end = plan->grid_img_offsets[p + 1];
                if (n_spin_mat == 1) {
                    const double w0 = dvol_eff * vg_ptr[p];
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            double* v0 = my_vs + plan->blocks[bid].blk_scalar_off;
                            for (int comp = 0; comp < n_comp; ++comp) {
                                const double* va = comp_vecs[comp] + ga.val_offset;
                                const double* vb = comp_vecs[comp] + gb.val_offset;
                                for (int a = 0; a < noa; ++a) {
                                    const double w0a = w0 * va[a];
                                    double* row0 = v0 + a * nob;
                                    for (int b = 0; b < nob; ++b) row0[b] += w0a * vb[b];
                                }
                            }
                        }
                    }
                } else if (n_spin_mat == 2) {
                    const double w0 = dvol_eff * vg_ptr[p];
                    const double w1 = dvol_eff * vg_ptr[n_grid + p];
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            double* v0 = my_vs + plan->blocks[bid].blk_scalar_off;
                            double* v1 = v0 + blk_stride;
                            for (int comp = 0; comp < n_comp; ++comp) {
                                const double* va = comp_vecs[comp] + ga.val_offset;
                                const double* vb = comp_vecs[comp] + gb.val_offset;
                                for (int a = 0; a < noa; ++a) {
                                    const double w0a = w0 * va[a];
                                    const double w1a = w1 * va[a];
                                    double* r0 = v0 + a * nob;
                                    double* r1 = v1 + a * nob;
                                    for (int b = 0; b < nob; ++b) {
                                        const double pb = vb[b];
                                        r0[b] += w0a * pb;
                                        r1[b] += w1a * pb;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    const double w0 = dvol_eff * vg_ptr[p];
                    const double w1 = dvol_eff * vg_ptr[n_grid + p];
                    const double w2 = dvol_eff * vg_ptr[2 * n_grid + p];
                    const double w3 = dvol_eff * vg_ptr[3 * n_grid + p];
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            double* v0 = my_vs + plan->blocks[bid].blk_scalar_off;
                            double* v1 = v0 + blk_stride;
                            double* v2 = v1 + blk_stride;
                            double* v3 = v2 + blk_stride;
                            for (int comp = 0; comp < n_comp; ++comp) {
                                const double* va = comp_vecs[comp] + ga.val_offset;
                                const double* vb = comp_vecs[comp] + gb.val_offset;
                                for (int a = 0; a < noa; ++a) {
                                    const double pa = va[a];
                                    const double w0a = w0 * pa, w1a = w1 * pa, w2a = w2 * pa, w3a = w3 * pa;
                                    double* r0 = v0 + a * nob;
                                    double* r1 = v1 + a * nob;
                                    double* r2 = v2 + a * nob;
                                    double* r3 = v3 + a * nob;
                                    for (int b = 0; b < nob; ++b) {
                                        const double pb = vb[b];
                                        r0[b] += w0a * pb;
                                        r1[b] += w1a * pb;
                                        r2[b] += w2a * pb;
                                        r3[b] += w3a * pb;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        if (actual_threads <= 1) {
            worker(0, 0, n_grid);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(actual_threads);
            const int64_t chunk = (n_grid + actual_threads - 1) / actual_threads;
            for (int t = 0; t < actual_threads; ++t) {
                const int64_t p_lo = t * chunk;
                const int64_t p_hi = std::min(n_grid, p_lo + chunk);
                if (p_lo < p_hi) pool.emplace_back(worker, t, p_lo, p_hi);
            }
            for (auto& th : pool) th.join();
        }

        for (int t = 0; t < actual_threads; ++t) {
            const double* my_vs = tls_vs[t].data();
            for (const auto& blk : plan->blocks) {
                const int32_t end_e = blk.edge_list_off + blk.n_valid_edges;
                for (int32_t ei = blk.edge_list_off; ei < end_e; ++ei) {
                    const auto& me = plan->active_block_edges[ei];
                    for (int32_t s = 0; s < n_spin_mat; ++s) {
                        vs_ptr[static_cast<size_t>(s) * n_edges + me.e_idx] +=
                            my_vs[static_cast<size_t>(s) * blk_stride + blk.blk_scalar_off + me.blk_rel];
                    }
                }
            }
        }
        return ffi::Error::Success();
    }

    const auto& cache = plan->active;
    const auto& edge_map = plan->edge_map;
    for (int64_t p = 0; p < cache.n_grid; ++p) {
        const int32_t beg = cache.grid_offsets[p];
        const int32_t end = cache.grid_offsets[p + 1];
        for (int32_t ia = beg; ia < end; ++ia) {
            const auto& ea = cache.entries[ia];
            for (int32_t ib = beg; ib < end; ++ib) {
                const auto& eb = cache.entries[ib];
                const int32_t e_idx = edge_map.find(
                    ea.orb, eb.orb, eb.s0 - ea.s0, eb.s1 - ea.s1, eb.s2 - ea.s2);
                if (e_idx < 0) continue;
                const double w_prod = is_tau
                    ? 0.5 * cache.dvol * (ea.gx * eb.gx + ea.gy * eb.gy + ea.gz * eb.gz)
                    : (cache.dvol * ea.val * eb.val);
                for (int32_t s = 0; s < n_spin_mat; ++s) {
                    vs_ptr[static_cast<size_t>(s) * n_edges + e_idx] +=
                        vg_ptr[static_cast<size_t>(s) * cache.n_grid + p] * w_prod;
                }
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error VmatImpl(
    ffi::Buffer<ffi::DataType::F64> v_grid, ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell, ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut, ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values, ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell, ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut, ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb, ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sparse) {
    return GatherVmatGridImpl(
        false, v_grid, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n1, n2, n3, nspin, out_v_sparse);
}

ffi::Error VmatTauImpl(
    ffi::Buffer<ffi::DataType::F64> v_tau_grid, ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell, ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut, ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values, ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell, ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut, ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb, ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sparse) {
    return GatherVmatGridImpl(
        true, v_tau_grid, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n1, n2, n3, nspin, out_v_sparse);
}

ffi::Error DfscfGradCoreImpl(
    bool is_tau,
    ffi::Buffer<ffi::DataType::F64> dm_edges,
    ffi::Buffer<ffi::DataType::F64> v_cotangent,
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::F64> pao_r_max,
    ffi::Buffer<ffi::DataType::F64> pao_rcut,
    ffi::Buffer<ffi::DataType::S32> pao_l,
    ffi::Buffer<ffi::DataType::F64> pao_values,
    ffi::Buffer<ffi::DataType::S32> orb_atom,
    ffi::Buffer<ffi::DataType::S32> orb_g_shell,
    ffi::Buffer<ffi::DataType::S32> orb_m,
    ffi::Buffer<ffi::DataType::F64> orb_rcut,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> dst_unit_orb,
    ffi::Buffer<ffi::DataType::S32> edge_cell_idx,
    ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_dm,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_pos,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_cell) {

    const int32_t n_spin_mat = (nspin == 1) ? 1 : (nspin == 2 ? 2 : 4);
    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_orb = static_cast<int32_t>(orb_atom.dimensions()[0]);
    const int32_t n_shells = static_cast<int32_t>(pao_l.dimensions()[0]);
    const int32_t n_pao_pts = static_cast<int32_t>(pao_values.dimensions()[1]);
    const int32_t n_edges = static_cast<int32_t>(dm_edges.dimensions()[1]);

    // Fast path: cached atom-image plan with orbital values and gradients (and, for the meta-GGA backward, Hessians),
    // dense atom-pair DM blocks and per-thread accumulators. Same terms as the generic loop below; pairs without an
    // edge have a zero DM entry and their d_dm entries are not mapped back, exactly as the edge lookup skips them.
    {
        const int32_t n_shifts = static_cast<int32_t>(cell_shifts.dimensions()[0]);
        const auto plan = get_or_build_grid_plan(
            positions.typed_data(), cell.typed_data(),
            pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
            orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
            row_ptr.typed_data(), dst_unit_orb.typed_data(), edge_cell_idx.typed_data(), cell_shifts.typed_data(),
            n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3, true, is_tau);
        if (plan->use_atom_blocks) {
            const int64_t n_grid = plan->active.n_grid;
            const size_t blk_stride = plan->total_blk_scalars;
            const double* dm_ptr = dm_edges.typed_data();
            const double* vc_ptr = v_cotangent.typed_data();
            std::vector<double> dm_blk(blk_stride * static_cast<size_t>(n_spin_mat), 0.0);
            for (const auto& blk : plan->blocks) {
                const int32_t end_e = blk.edge_list_off + blk.n_valid_edges;
                for (int32_t ei = blk.edge_list_off; ei < end_e; ++ei) {
                    const auto& me = plan->active_block_edges[ei];
                    for (int32_t s = 0; s < n_spin_mat; ++s) {
                        dm_blk[static_cast<size_t>(s) * blk_stride + blk.blk_scalar_off + me.blk_rel] =
                            dm_ptr[static_cast<size_t>(s) * n_edges + me.e_idx];
                    }
                }
            }
            int32_t max_no = 1;
            for (const int32_t no : plan->atom_n_orbs) max_no = std::max(max_no, no);

            const int n_threads = resolve_num_threads(n_grid);
            const size_t one_thread_bytes = blk_stride * static_cast<size_t>(n_spin_mat) * sizeof(double);
            const size_t max_tls_bytes = 256ULL * 1024ULL * 1024ULL;
            const int max_safe_threads = one_thread_bytes == 0
                ? 1
                : static_cast<int>(std::max<size_t>(1, max_tls_bytes / one_thread_bytes));
            const int actual_threads = std::max(1, std::min(n_threads, max_safe_threads));
            std::vector<std::vector<double>> tls_ddm(
                actual_threads, std::vector<double>(blk_stride * static_cast<size_t>(n_spin_mat), 0.0));
            std::vector<std::vector<double>> tls_dpos(actual_threads, std::vector<double>(3 * static_cast<size_t>(n_atoms), 0.0));
            std::vector<std::vector<double>> tls_dcell(actual_threads, std::vector<double>(9, 0.0));

            auto worker = [&](int t_id, int64_t p_lo, int64_t p_hi) {
                double* my_ddm = tls_ddm[t_id].data();
                double* my_dpos = tls_dpos[t_id].data();
                double* my_dcell = tls_dcell[t_id].data();
                std::vector<double> q_b(static_cast<size_t>(max_no), 0.0);
                std::vector<double> qx_b, qy_b, qz_b;  // meta-GGA: sum_a w_ab grad phi_a
                if (is_tau) {
                    qx_b.assign(static_cast<size_t>(max_no), 0.0);
                    qy_b.assign(static_cast<size_t>(max_no), 0.0);
                    qz_b.assign(static_cast<size_t>(max_no), 0.0);
                }
                double vc[4] = {0.0, 0.0, 0.0, 0.0};
                for (int64_t p = p_lo; p < p_hi; ++p) {
                    const int32_t beg = plan->grid_img_offsets[p];
                    const int32_t end = plan->grid_img_offsets[p + 1];
                    if (beg == end) continue;
                    const int32_t g2 = static_cast<int32_t>(p % n3);
                    const int32_t g1 = static_cast<int32_t>((p / n3) % n2);
                    const int32_t g0 = static_cast<int32_t>(p / (static_cast<int64_t>(n2) * n3));
                    const double f0 = static_cast<double>(g0) / static_cast<double>(n1);
                    const double f1 = static_cast<double>(g1) / static_cast<double>(n2);
                    const double f2 = static_cast<double>(g2) / static_cast<double>(n3);
                    for (int32_t s = 0; s < n_spin_mat; ++s) vc[s] = vc_ptr[static_cast<size_t>(s) * n_grid + p];
                    for (int32_t ia = beg; ia < end; ++ia) {
                        const auto& ga = plan->img_groups[ia];
                        const int noa = plan->atom_n_orbs[ga.atom];
                        const double* va = plan->img_vals.data() + ga.val_offset;
                        const double* gxa = plan->img_gx.data() + ga.val_offset;
                        const double* gya = plan->img_gy.data() + ga.val_offset;
                        const double* gza = plan->img_gz.data() + ga.val_offset;
                        for (int32_t ib = beg; ib < end; ++ib) {
                            const auto& gb = plan->img_groups[ib];
                            const int32_t c_idx = plan->lookup_shift(gb.s0 - ga.s0, gb.s1 - ga.s1, gb.s2 - ga.s2);
                            if (c_idx < 0) continue;
                            const int32_t bid = plan->lookup_block(ga.atom, gb.atom, c_idx);
                            if (bid < 0) continue;
                            const int nob = plan->atom_n_orbs[gb.atom];
                            const double* vb = plan->img_vals.data() + gb.val_offset;
                            const double* gxb = plan->img_gx.data() + gb.val_offset;
                            const double* gyb = plan->img_gy.data() + gb.val_offset;
                            const double* gzb = plan->img_gz.data() + gb.val_offset;
                            const size_t off = plan->blocks[bid].blk_scalar_off;
                            double fa_x = 0.0, fa_y = 0.0, fa_z = 0.0;
                            double fb_x = 0.0, fb_y = 0.0, fb_z = 0.0;
                            if (!is_tau) {
                                // rho = phi_a D phi_b: f_a = grad phi_a sum_b w_ab phi_b, f_b = grad phi_b sum_a w_ab phi_a
                                std::fill(q_b.begin(), q_b.begin() + nob, 0.0);
                                for (int a = 0; a < noa; ++a) {
                                    const double pa = va[a];
                                    double r_a = 0.0;
                                    for (int b = 0; b < nob; ++b) {
                                        const size_t rel = off + static_cast<size_t>(a) * nob + b;
                                        const double prod = pa * vb[b];
                                        double w_ab = 0.0;
                                        for (int32_t s = 0; s < n_spin_mat; ++s) {
                                            my_ddm[static_cast<size_t>(s) * blk_stride + rel] += vc[s] * prod;
                                            w_ab += vc[s] * dm_blk[static_cast<size_t>(s) * blk_stride + rel];
                                        }
                                        r_a += w_ab * vb[b];
                                        q_b[b] += w_ab * pa;
                                    }
                                    fa_x += gxa[a] * r_a;
                                    fa_y += gya[a] * r_a;
                                    fa_z += gza[a] * r_a;
                                }
                                for (int b = 0; b < nob; ++b) {
                                    fb_x += gxb[b] * q_b[b];
                                    fb_y += gyb[b] * q_b[b];
                                    fb_z += gzb[b] * q_b[b];
                                }
                            } else {
                                // tau = 1/2 grad phi_a . D grad phi_b: f_a = 1/2 H_a sum_b w_ab grad phi_b,
                                // f_b = 1/2 H_b sum_a w_ab grad phi_a (H: orbital Hessian wrt r)
                                std::fill(qx_b.begin(), qx_b.begin() + nob, 0.0);
                                std::fill(qy_b.begin(), qy_b.begin() + nob, 0.0);
                                std::fill(qz_b.begin(), qz_b.begin() + nob, 0.0);
                                const size_t vo_a = static_cast<size_t>(ga.val_offset);
                                for (int a = 0; a < noa; ++a) {
                                    double rx = 0.0, ry = 0.0, rz = 0.0;
                                    for (int b = 0; b < nob; ++b) {
                                        const size_t rel = off + static_cast<size_t>(a) * nob + b;
                                        const double prod = 0.5 * (gxa[a] * gxb[b] + gya[a] * gyb[b] + gza[a] * gzb[b]);
                                        double w_ab = 0.0;
                                        for (int32_t s = 0; s < n_spin_mat; ++s) {
                                            my_ddm[static_cast<size_t>(s) * blk_stride + rel] += vc[s] * prod;
                                            w_ab += vc[s] * dm_blk[static_cast<size_t>(s) * blk_stride + rel];
                                        }
                                        rx += w_ab * gxb[b];
                                        ry += w_ab * gyb[b];
                                        rz += w_ab * gzb[b];
                                        qx_b[b] += w_ab * gxa[a];
                                        qy_b[b] += w_ab * gya[a];
                                        qz_b[b] += w_ab * gza[a];
                                    }
                                    const size_t ka = vo_a + static_cast<size_t>(a);
                                    fa_x += 0.5 * (plan->img_hxx[ka] * rx + plan->img_hxy[ka] * ry + plan->img_hxz[ka] * rz);
                                    fa_y += 0.5 * (plan->img_hxy[ka] * rx + plan->img_hyy[ka] * ry + plan->img_hyz[ka] * rz);
                                    fa_z += 0.5 * (plan->img_hxz[ka] * rx + plan->img_hyz[ka] * ry + plan->img_hzz[ka] * rz);
                                }
                                const size_t vo_b = static_cast<size_t>(gb.val_offset);
                                for (int b = 0; b < nob; ++b) {
                                    const size_t kb = vo_b + static_cast<size_t>(b);
                                    fb_x += 0.5 * (plan->img_hxx[kb] * qx_b[b] + plan->img_hxy[kb] * qy_b[b] + plan->img_hxz[kb] * qz_b[b]);
                                    fb_y += 0.5 * (plan->img_hxy[kb] * qx_b[b] + plan->img_hyy[kb] * qy_b[b] + plan->img_hyz[kb] * qz_b[b]);
                                    fb_z += 0.5 * (plan->img_hxz[kb] * qx_b[b] + plan->img_hyz[kb] * qy_b[b] + plan->img_hzz[kb] * qz_b[b]);
                                }
                            }
                            my_dpos[3 * ga.atom + 0] -= fa_x;
                            my_dpos[3 * ga.atom + 1] -= fa_y;
                            my_dpos[3 * ga.atom + 2] -= fa_z;
                            my_dpos[3 * gb.atom + 0] -= fb_x;
                            my_dpos[3 * gb.atom + 1] -= fb_y;
                            my_dpos[3 * gb.atom + 2] -= fb_z;
                            const double df_a[3] = {f0 - ga.s0, f1 - ga.s1, f2 - ga.s2};
                            const double df_b[3] = {f0 - gb.s0, f1 - gb.s1, f2 - gb.s2};
                            for (int j = 0; j < 3; ++j) {
                                my_dcell[0 * 3 + j] += fa_x * df_a[j] + fb_x * df_b[j];
                                my_dcell[1 * 3 + j] += fa_y * df_a[j] + fb_y * df_b[j];
                                my_dcell[2 * 3 + j] += fa_z * df_a[j] + fb_z * df_b[j];
                            }
                        }
                    }
                }
            };

            if (actual_threads <= 1) {
                worker(0, 0, n_grid);
            } else {
                std::vector<std::thread> pool;
                pool.reserve(actual_threads);
                const int64_t chunk = (n_grid + actual_threads - 1) / actual_threads;
                for (int t = 0; t < actual_threads; ++t) {
                    const int64_t p_lo = t * chunk;
                    const int64_t p_hi = std::min(n_grid, p_lo + chunk);
                    if (p_lo < p_hi) pool.emplace_back(worker, t, p_lo, p_hi);
                }
                for (auto& th : pool) th.join();
            }

            double* d_dm_ptr = out_d_dm->typed_data();
            double* d_pos_ptr = out_d_pos->typed_data();
            double* d_cell_ptr = out_d_cell->typed_data();
            std::fill(d_dm_ptr, d_dm_ptr + static_cast<size_t>(n_spin_mat) * n_edges, 0.0);
            std::fill(d_pos_ptr, d_pos_ptr + static_cast<size_t>(n_atoms) * 3, 0.0);
            std::fill(d_cell_ptr, d_cell_ptr + 9, 0.0);
            for (int t = 0; t < actual_threads; ++t) {
                const double* my_ddm = tls_ddm[t].data();
                for (const auto& blk : plan->blocks) {
                    const int32_t end_e = blk.edge_list_off + blk.n_valid_edges;
                    for (int32_t ei = blk.edge_list_off; ei < end_e; ++ei) {
                        const auto& me = plan->active_block_edges[ei];
                        for (int32_t s = 0; s < n_spin_mat; ++s) {
                            d_dm_ptr[static_cast<size_t>(s) * n_edges + me.e_idx] +=
                                my_ddm[static_cast<size_t>(s) * blk_stride + blk.blk_scalar_off + me.blk_rel];
                        }
                    }
                }
                for (size_t k = 0; k < 3 * static_cast<size_t>(n_atoms); ++k) d_pos_ptr[k] += tls_dpos[t][k];
                for (int k = 0; k < 9; ++k) d_cell_ptr[k] += tls_dcell[t][k];
            }
            return ffi::Error::Success();
        }
    }

    const auto cache = build_active_grid_cache(
        positions.typed_data(), cell.typed_data(),
        pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
        orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
        n_atoms, n_orb, n_shells, n_pao_pts, n1, n2, n3, true, is_tau);

    const double* dm_ptr = dm_edges.typed_data();
    const double* vc_ptr = v_cotangent.typed_data();
    const int32_t* oa_ptr = orb_atom.typed_data();
    EdgeHashMap edge_map;
    edge_map.build(
        n_orb, row_ptr.typed_data(), dst_unit_orb.typed_data(),
        edge_cell_idx.typed_data(), cell_shifts.typed_data());

    double* d_dm_ptr = out_d_dm->typed_data();
    double* d_pos_ptr = out_d_pos->typed_data();
    double* d_cell_ptr = out_d_cell->typed_data();
    std::fill(d_dm_ptr, d_dm_ptr + static_cast<size_t>(n_spin_mat) * n_edges, 0.0);
    std::fill(d_pos_ptr, d_pos_ptr + static_cast<size_t>(n_atoms) * 3, 0.0);
    std::fill(d_cell_ptr, d_cell_ptr + 9, 0.0);

    for (int64_t p = 0; p < cache.n_grid; ++p) {
        const int32_t g2 = static_cast<int32_t>(p % n3);
        const int32_t g1 = static_cast<int32_t>((p / n3) % n2);
        const int32_t g0 = static_cast<int32_t>(p / (static_cast<int64_t>(n2) * n3));
        const double f0 = static_cast<double>(g0) / static_cast<double>(n1);
        const double f1 = static_cast<double>(g1) / static_cast<double>(n2);
        const double f2 = static_cast<double>(g2) / static_cast<double>(n3);

        const int32_t beg = cache.grid_offsets[p];
        const int32_t end = cache.grid_offsets[p + 1];
        for (int32_t ia = beg; ia < end; ++ia) {
            const auto& ea = cache.entries[ia];
            const int32_t atom_a = oa_ptr[ea.orb];
            for (int32_t ib = beg; ib < end; ++ib) {
                const auto& eb = cache.entries[ib];
                const int32_t e_idx = edge_map.find(
                    ea.orb, eb.orb, eb.s0 - ea.s0, eb.s1 - ea.s1, eb.s2 - ea.s2);
                if (e_idx < 0) continue;
                const int32_t atom_b = oa_ptr[eb.orb];

                double w_pair = 0.0;
                const double prod = is_tau
                    ? 0.5 * (ea.gx * eb.gx + ea.gy * eb.gy + ea.gz * eb.gz)
                    : (ea.val * eb.val);
                for (int32_t s = 0; s < n_spin_mat; ++s) {
                    const double vc = vc_ptr[static_cast<size_t>(s) * cache.n_grid + p];
                    d_dm_ptr[static_cast<size_t>(s) * n_edges + e_idx] += vc * prod;
                    w_pair += vc * dm_ptr[static_cast<size_t>(s) * n_edges + e_idx];
                }
                double fa_x = 0.0, fa_y = 0.0, fa_z = 0.0;
                double fb_x = 0.0, fb_y = 0.0, fb_z = 0.0;
                if (is_tau) {
                    const double w_half = 0.5 * w_pair;
                    fa_x = w_half * (ea.hxx * eb.gx + ea.hxy * eb.gy + ea.hxz * eb.gz);
                    fa_y = w_half * (ea.hxy * eb.gx + ea.hyy * eb.gy + ea.hyz * eb.gz);
                    fa_z = w_half * (ea.hxz * eb.gx + ea.hyz * eb.gy + ea.hzz * eb.gz);
                    fb_x = w_half * (eb.hxx * ea.gx + eb.hxy * ea.gy + eb.hxz * ea.gz);
                    fb_y = w_half * (eb.hxy * ea.gx + eb.hyy * ea.gy + eb.hyz * ea.gz);
                    fb_z = w_half * (eb.hxz * ea.gx + eb.hyz * ea.gy + eb.hzz * ea.gz);
                } else {
                    fa_x = w_pair * ea.gx * eb.val;
                    fa_y = w_pair * ea.gy * eb.val;
                    fa_z = w_pair * ea.gz * eb.val;
                    fb_x = w_pair * ea.val * eb.gx;
                    fb_y = w_pair * ea.val * eb.gy;
                    fb_z = w_pair * ea.val * eb.gz;
                }
                d_pos_ptr[3 * atom_a + 0] -= fa_x;
                d_pos_ptr[3 * atom_a + 1] -= fa_y;
                d_pos_ptr[3 * atom_a + 2] -= fa_z;
                d_pos_ptr[3 * atom_b + 0] -= fb_x;
                d_pos_ptr[3 * atom_b + 1] -= fb_y;
                d_pos_ptr[3 * atom_b + 2] -= fb_z;

                const double df_a[3] = {
                    f0 - static_cast<double>(ea.s0),
                    f1 - static_cast<double>(ea.s1),
                    f2 - static_cast<double>(ea.s2),
                };
                const double df_b[3] = {
                    f0 - static_cast<double>(eb.s0),
                    f1 - static_cast<double>(eb.s1),
                    f2 - static_cast<double>(eb.s2),
                };
                for (int j = 0; j < 3; ++j) {
                    d_cell_ptr[0 * 3 + j] += fa_x * df_a[j] + fb_x * df_b[j];
                    d_cell_ptr[1 * 3 + j] += fa_y * df_a[j] + fb_y * df_b[j];
                    d_cell_ptr[2 * 3 + j] += fa_z * df_a[j] + fb_z * df_b[j];
                }
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error DfscfGradImpl(
    ffi::Buffer<ffi::DataType::F64> dm_edges, ffi::Buffer<ffi::DataType::F64> v_cotangent,
    ffi::Buffer<ffi::DataType::F64> positions, ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::F64> pao_r_max, ffi::Buffer<ffi::DataType::F64> pao_rcut,
    ffi::Buffer<ffi::DataType::S32> pao_l, ffi::Buffer<ffi::DataType::F64> pao_values,
    ffi::Buffer<ffi::DataType::S32> orb_atom, ffi::Buffer<ffi::DataType::S32> orb_g_shell,
    ffi::Buffer<ffi::DataType::S32> orb_m, ffi::Buffer<ffi::DataType::F64> orb_rcut,
    ffi::Buffer<ffi::DataType::S32> row_ptr, ffi::Buffer<ffi::DataType::S32> dst_unit_orb,
    ffi::Buffer<ffi::DataType::S32> edge_cell_idx, ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_dm,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_pos,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_cell) {
    return DfscfGradCoreImpl(
        false, dm_edges, v_cotangent, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n1, n2, n3, nspin, out_d_dm, out_d_pos, out_d_cell);
}

ffi::Error TauDfscfGradImpl(
    ffi::Buffer<ffi::DataType::F64> dm_edges, ffi::Buffer<ffi::DataType::F64> v_cotangent,
    ffi::Buffer<ffi::DataType::F64> positions, ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::F64> pao_r_max, ffi::Buffer<ffi::DataType::F64> pao_rcut,
    ffi::Buffer<ffi::DataType::S32> pao_l, ffi::Buffer<ffi::DataType::F64> pao_values,
    ffi::Buffer<ffi::DataType::S32> orb_atom, ffi::Buffer<ffi::DataType::S32> orb_g_shell,
    ffi::Buffer<ffi::DataType::S32> orb_m, ffi::Buffer<ffi::DataType::F64> orb_rcut,
    ffi::Buffer<ffi::DataType::S32> row_ptr, ffi::Buffer<ffi::DataType::S32> dst_unit_orb,
    ffi::Buffer<ffi::DataType::S32> edge_cell_idx, ffi::Buffer<ffi::DataType::S32> cell_shifts,
    int64_t n1, int64_t n2, int64_t n3, int64_t nspin,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_dm,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_pos,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_cell) {
    return DfscfGradCoreImpl(
        true, dm_edges, v_cotangent, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values,
        orb_atom, orb_g_shell, orb_m, orb_rcut, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n1, n2, n3, nspin, out_d_dm, out_d_pos, out_d_cell);
}

ffi::Error NaGridImpl(
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::S32> atom_species,
    ffi::Buffer<ffi::DataType::F64> sp_rmax,
    ffi::Buffer<ffi::DataType::F64> sp_rcut_na,
    ffi::Buffer<ffi::DataType::F64> sp_rho_na,
    ffi::Buffer<ffi::DataType::F64> sp_v_na,
    ffi::Buffer<ffi::DataType::F64> sp_rho_core,
    int64_t n1, int64_t n2, int64_t n3,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_rho_na,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_na,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_rho_core) {

    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_sp = static_cast<int32_t>(sp_rmax.dimensions()[0]);
    const int32_t n_pts = static_cast<int32_t>(sp_rho_na.dimensions()[1]);
    const int64_t n_grid = static_cast<int64_t>(n1) * n2 * n3;

    const double* pos_ptr = positions.typed_data();
    const double* cell_ptr = cell.typed_data();
    const int32_t* sp_idx_ptr = atom_species.typed_data();
    const double* rmax_ptr = sp_rmax.typed_data();
    const double* rc_ptr = sp_rcut_na.typed_data();
    const double* rna_ptr = sp_rho_na.typed_data();
    const double* vna_ptr = sp_v_na.typed_data();
    const double* rcore_ptr = sp_rho_core.typed_data();

    double* out_r = out_rho_na->typed_data();
    double* out_v = out_v_na->typed_data();
    double* out_c = out_rho_core->typed_data();
    std::fill(out_r, out_r + n_grid, 0.0);
    std::fill(out_v, out_v + n_grid, 0.0);
    std::fill(out_c, out_c + n_grid, 0.0);

    std::vector<jsiesta::math::UniformCubicSpline> s_rna(n_sp), s_vna(n_sp), s_rcore(n_sp);
    for (int s = 0; s < n_sp; ++s) {
        s_rna[s].build(rna_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rc_ptr[s]);
        s_vna[s].build(vna_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rc_ptr[s]);
        s_rcore[s].build(rcore_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rmax_ptr[s]);
    }

    double inv_cell[9], det = 0.0;
    mat3_inv(cell_ptr, inv_cell, det);
    double b_norms[3];
    for (int ax = 0; ax < 3; ++ax) {
        b_norms[ax] = std::sqrt(
            inv_cell[3 * ax + 0] * inv_cell[3 * ax + 0] +
            inv_cell[3 * ax + 1] * inv_cell[3 * ax + 1] +
            inv_cell[3 * ax + 2] * inv_cell[3 * ax + 2]);
    }

    for (int32_t ia = 0; ia < n_atoms; ++ia) {
        const int sp = sp_idx_ptr[ia];
        const double rc_a = rmax_ptr[sp];
        const double px = pos_ptr[3 * ia + 0];
        const double py = pos_ptr[3 * ia + 1];
        const double pz = pos_ptr[3 * ia + 2];
        const double fc0 = inv_cell[0] * px + inv_cell[1] * py + inv_cell[2] * pz;
        const double fc1 = inv_cell[3] * px + inv_cell[4] * py + inv_cell[5] * pz;
        const double fc2 = inv_cell[6] * px + inv_cell[7] * py + inv_cell[8] * pz;

        const int i0_min = static_cast<int>(std::ceil(n1 * (fc0 - rc_a * b_norms[0])));
        const int i0_max = static_cast<int>(std::floor(n1 * (fc0 + rc_a * b_norms[0])));
        const int i1_min = static_cast<int>(std::ceil(n2 * (fc1 - rc_a * b_norms[1])));
        const int i1_max = static_cast<int>(std::floor(n2 * (fc1 + rc_a * b_norms[1])));
        const int i2_min = static_cast<int>(std::ceil(n3 * (fc2 - rc_a * b_norms[2])));
        const int i2_max = static_cast<int>(std::floor(n3 * (fc2 + rc_a * b_norms[2])));

        for (int i0 = i0_min; i0 <= i0_max; ++i0) {
            const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
            const int g0 = floor_mod(i0, n1);
            for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
                const int g1 = floor_mod(i1, n2);
                for (int i2 = i2_min; i2 <= i2_max; ++i2) {
                    const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
                    const double rx = cell_ptr[0] * f0 + cell_ptr[1] * f1 + cell_ptr[2] * f2;
                    const double ry = cell_ptr[3] * f0 + cell_ptr[4] * f1 + cell_ptr[5] * f2;
                    const double rz = cell_ptr[6] * f0 + cell_ptr[7] * f1 + cell_ptr[8] * f2;
                    const double dx = rx - px, dy = ry - py, dz = rz - pz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_a * rc_a) continue;
                    const double r = std::sqrt(r2);
                    const int g2 = floor_mod(i2, n3);
                    const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;
                    if (r < rc_ptr[sp]) {
                        out_r[p_idx] += s_rna[sp].eval(r);
                        out_v[p_idx] += s_vna[sp].eval(r);
                    }
                    out_c[p_idx] += s_rcore[sp].eval(r);
                }
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error NaGridGradImpl(
    ffi::Buffer<ffi::DataType::F64> w_rho_na,
    ffi::Buffer<ffi::DataType::F64> w_v_na,
    ffi::Buffer<ffi::DataType::F64> w_rho_core,
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::S32> atom_species,
    ffi::Buffer<ffi::DataType::F64> sp_rmax,
    ffi::Buffer<ffi::DataType::F64> sp_rcut_na,
    ffi::Buffer<ffi::DataType::F64> sp_rho_na,
    ffi::Buffer<ffi::DataType::F64> sp_v_na,
    ffi::Buffer<ffi::DataType::F64> sp_rho_core,
    int64_t n1, int64_t n2, int64_t n3,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_pos,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_cell) {

    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_sp = static_cast<int32_t>(sp_rmax.dimensions()[0]);
    const int32_t n_pts = static_cast<int32_t>(sp_rho_na.dimensions()[1]);

    const double* wr_ptr = w_rho_na.typed_data();
    const double* wv_ptr = w_v_na.typed_data();
    const double* wc_ptr = w_rho_core.typed_data();
    const double* pos_ptr = positions.typed_data();
    const double* cell_ptr = cell.typed_data();
    const int32_t* sp_idx_ptr = atom_species.typed_data();
    const double* rmax_ptr = sp_rmax.typed_data();
    const double* rc_ptr = sp_rcut_na.typed_data();
    const double* rna_ptr = sp_rho_na.typed_data();
    const double* vna_ptr = sp_v_na.typed_data();
    const double* rcore_ptr = sp_rho_core.typed_data();

    double* d_pos_ptr = out_d_pos->typed_data();
    double* d_cell_ptr = out_d_cell->typed_data();
    std::fill(d_pos_ptr, d_pos_ptr + static_cast<size_t>(n_atoms) * 3, 0.0);
    std::fill(d_cell_ptr, d_cell_ptr + 9, 0.0);

    std::vector<jsiesta::math::UniformCubicSpline> s_rna(n_sp), s_vna(n_sp), s_rcore(n_sp);
    for (int s = 0; s < n_sp; ++s) {
        s_rna[s].build(rna_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rc_ptr[s]);
        s_vna[s].build(vna_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rc_ptr[s]);
        s_rcore[s].build(rcore_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rmax_ptr[s]);
    }

    double inv_cell[9], det = 0.0;
    mat3_inv(cell_ptr, inv_cell, det);
    double b_norms[3];
    for (int ax = 0; ax < 3; ++ax) {
        b_norms[ax] = std::sqrt(
            inv_cell[3 * ax + 0] * inv_cell[3 * ax + 0] +
            inv_cell[3 * ax + 1] * inv_cell[3 * ax + 1] +
            inv_cell[3 * ax + 2] * inv_cell[3 * ax + 2]);
    }

    for (int32_t ia = 0; ia < n_atoms; ++ia) {
        const int sp = sp_idx_ptr[ia];
        const double rc_a = rmax_ptr[sp];
        const double px = pos_ptr[3 * ia + 0];
        const double py = pos_ptr[3 * ia + 1];
        const double pz = pos_ptr[3 * ia + 2];
        const double fc0 = inv_cell[0] * px + inv_cell[1] * py + inv_cell[2] * pz;
        const double fc1 = inv_cell[3] * px + inv_cell[4] * py + inv_cell[5] * pz;
        const double fc2 = inv_cell[6] * px + inv_cell[7] * py + inv_cell[8] * pz;

        const int i0_min = static_cast<int>(std::ceil(n1 * (fc0 - rc_a * b_norms[0])));
        const int i0_max = static_cast<int>(std::floor(n1 * (fc0 + rc_a * b_norms[0])));
        const int i1_min = static_cast<int>(std::ceil(n2 * (fc1 - rc_a * b_norms[1])));
        const int i1_max = static_cast<int>(std::floor(n2 * (fc1 + rc_a * b_norms[1])));
        const int i2_min = static_cast<int>(std::ceil(n3 * (fc2 - rc_a * b_norms[2])));
        const int i2_max = static_cast<int>(std::floor(n3 * (fc2 + rc_a * b_norms[2])));

        for (int i0 = i0_min; i0 <= i0_max; ++i0) {
            const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
            const int g0 = floor_mod(i0, n1);
            for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
                const int g1 = floor_mod(i1, n2);
                for (int i2 = i2_min; i2 <= i2_max; ++i2) {
                    const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
                    const double rx = cell_ptr[0] * f0 + cell_ptr[1] * f1 + cell_ptr[2] * f2;
                    const double ry = cell_ptr[3] * f0 + cell_ptr[4] * f1 + cell_ptr[5] * f2;
                    const double rz = cell_ptr[6] * f0 + cell_ptr[7] * f1 + cell_ptr[8] * f2;
                    const double dx = rx - px, dy = ry - py, dz = rz - pz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_a * rc_a || r2 < 1e-24) continue;
                    const double r = std::sqrt(r2);
                    const int g2 = floor_mod(i2, n3);
                    const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;

                    double drna = 0.0, dvna = 0.0, drc = 0.0;
                    if (r < rc_ptr[sp]) {
                        s_rna[sp].eval(r, &drna);
                        s_vna[sp].eval(r, &dvna);
                    }
                    s_rcore[sp].eval(r, &drc);

                    const double coeff = (wr_ptr[p_idx] * drna + wv_ptr[p_idx] * dvna + wc_ptr[p_idx] * drc) / r;
                    const double gx = coeff * dx, gy = coeff * dy, gz = coeff * dz;
                    d_pos_ptr[3 * ia + 0] -= gx;
                    d_pos_ptr[3 * ia + 1] -= gy;
                    d_pos_ptr[3 * ia + 2] -= gz;

                    // f0, f1, f2 are already unwrapped fractional coordinates (i / N)
                    const double df[3] = {f0, f1, f2};
                    for (int j = 0; j < 3; ++j) {
                        d_cell_ptr[0 * 3 + j] += gx * df[j];
                        d_cell_ptr[1 * 3 + j] += gy * df[j];
                        d_cell_ptr[2 * 3 + j] += gz * df[j];
                    }
                }
            }
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RhoofdHandler, RhoofdImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // dm_edges
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // positions
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // cell
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_r_max
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_rcut
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pao_l
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_values
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_atom
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_g_shell
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_m
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // orb_rcut
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // row_ptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // dst_unit_orb
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // edge_cell_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // cell_shifts
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VmatHandler, VmatImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // v_grid
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // positions
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // cell
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_r_max
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_rcut
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pao_l
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_values
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_atom
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_g_shell
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_m
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // orb_rcut
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // row_ptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // dst_unit_orb
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // edge_cell_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // cell_shifts
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DfscfGradHandler, DfscfGradImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // dm_edges
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // v_cotangent
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // positions
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // cell
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_r_max
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_rcut
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pao_l
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // pao_values
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_atom
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_g_shell
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // orb_m
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // orb_rcut
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // row_ptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // dst_unit_orb
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // edge_cell_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // cell_shifts
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // d_dm
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // d_pos
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // d_cell

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NaGridHandler, NaGridImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // positions
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // cell
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // atom_species
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rmax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rcut_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rho_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_v_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rho_core
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // rho_na
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // v_na
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // rho_core

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NaGridGradHandler, NaGridGradImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // w_rho_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // w_v_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // w_rho_core
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // positions
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // cell
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // atom_species
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rmax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rcut_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rho_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_v_na
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // sp_rho_core
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // d_pos
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // d_cell

ffi::Error TauCoreGridImpl(
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::S32> atom_species,
    ffi::Buffer<ffi::DataType::F64> sp_rmax,
    ffi::Buffer<ffi::DataType::F64> sp_tau_core,
    int64_t n1, int64_t n2, int64_t n3,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_tau_core) {

    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_sp = static_cast<int32_t>(sp_rmax.dimensions()[0]);
    const int32_t n_pts = static_cast<int32_t>(sp_tau_core.dimensions()[1]);
    const int64_t n_grid = static_cast<int64_t>(n1) * n2 * n3;

    const double* pos_ptr = positions.typed_data();
    const double* cell_ptr = cell.typed_data();
    const int32_t* sp_idx_ptr = atom_species.typed_data();
    const double* rmax_ptr = sp_rmax.typed_data();
    const double* tcore_ptr = sp_tau_core.typed_data();
    double* out_t = out_tau_core->typed_data();
    std::fill(out_t, out_t + n_grid, 0.0);

    std::vector<jsiesta::math::UniformCubicSpline> s_tcore(n_sp);
    for (int s = 0; s < n_sp; ++s) {
        s_tcore[s].build(tcore_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rmax_ptr[s]);
    }

    double inv_cell[9], det = 0.0;
    mat3_inv(cell_ptr, inv_cell, det);
    double b_norms[3];
    for (int ax = 0; ax < 3; ++ax) {
        b_norms[ax] = std::sqrt(
            inv_cell[3 * ax + 0] * inv_cell[3 * ax + 0] +
            inv_cell[3 * ax + 1] * inv_cell[3 * ax + 1] +
            inv_cell[3 * ax + 2] * inv_cell[3 * ax + 2]);
    }

    for (int32_t ia = 0; ia < n_atoms; ++ia) {
        const int sp = sp_idx_ptr[ia];
        const double rc_a = rmax_ptr[sp];
        const double px = pos_ptr[3 * ia + 0], py = pos_ptr[3 * ia + 1], pz = pos_ptr[3 * ia + 2];
        const double fc0 = inv_cell[0] * px + inv_cell[1] * py + inv_cell[2] * pz;
        const double fc1 = inv_cell[3] * px + inv_cell[4] * py + inv_cell[5] * pz;
        const double fc2 = inv_cell[6] * px + inv_cell[7] * py + inv_cell[8] * pz;
        const int i0_min = static_cast<int>(std::ceil(n1 * (fc0 - rc_a * b_norms[0])));
        const int i0_max = static_cast<int>(std::floor(n1 * (fc0 + rc_a * b_norms[0])));
        const int i1_min = static_cast<int>(std::ceil(n2 * (fc1 - rc_a * b_norms[1])));
        const int i1_max = static_cast<int>(std::floor(n2 * (fc1 + rc_a * b_norms[1])));
        const int i2_min = static_cast<int>(std::ceil(n3 * (fc2 - rc_a * b_norms[2])));
        const int i2_max = static_cast<int>(std::floor(n3 * (fc2 + rc_a * b_norms[2])));
        for (int i0 = i0_min; i0 <= i0_max; ++i0) {
            const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
            const int g0 = floor_mod(i0, n1);
            for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
                const int g1 = floor_mod(i1, n2);
                for (int i2 = i2_min; i2 <= i2_max; ++i2) {
                    const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
                    const double rx = cell_ptr[0] * f0 + cell_ptr[1] * f1 + cell_ptr[2] * f2;
                    const double ry = cell_ptr[3] * f0 + cell_ptr[4] * f1 + cell_ptr[5] * f2;
                    const double rz = cell_ptr[6] * f0 + cell_ptr[7] * f1 + cell_ptr[8] * f2;
                    const double dx = rx - px, dy = ry - py, dz = rz - pz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_a * rc_a) continue;
                    const int g2 = floor_mod(i2, n3);
                    const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;
                    out_t[p_idx] += s_tcore[sp].eval(std::sqrt(r2));
                }
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error TauCoreGridGradImpl(
    ffi::Buffer<ffi::DataType::F64> w_tau_core,
    ffi::Buffer<ffi::DataType::F64> positions,
    ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::S32> atom_species,
    ffi::Buffer<ffi::DataType::F64> sp_rmax,
    ffi::Buffer<ffi::DataType::F64> sp_tau_core,
    int64_t n1, int64_t n2, int64_t n3,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_pos,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_d_cell) {

    const int32_t n_atoms = static_cast<int32_t>(positions.dimensions()[0]);
    const int32_t n_sp = static_cast<int32_t>(sp_rmax.dimensions()[0]);
    const int32_t n_pts = static_cast<int32_t>(sp_tau_core.dimensions()[1]);

    const double* wt_ptr = w_tau_core.typed_data();
    const double* pos_ptr = positions.typed_data();
    const double* cell_ptr = cell.typed_data();
    const int32_t* sp_idx_ptr = atom_species.typed_data();
    const double* rmax_ptr = sp_rmax.typed_data();
    const double* tcore_ptr = sp_tau_core.typed_data();

    double* d_pos_ptr = out_d_pos->typed_data();
    double* d_cell_ptr = out_d_cell->typed_data();
    std::fill(d_pos_ptr, d_pos_ptr + static_cast<size_t>(n_atoms) * 3, 0.0);
    std::fill(d_cell_ptr, d_cell_ptr + 9, 0.0);

    std::vector<jsiesta::math::UniformCubicSpline> s_tcore(n_sp);
    for (int s = 0; s < n_sp; ++s) {
        s_tcore[s].build(tcore_ptr + static_cast<size_t>(s) * n_pts, n_pts, rmax_ptr[s], rmax_ptr[s]);
    }

    double inv_cell[9], det = 0.0;
    mat3_inv(cell_ptr, inv_cell, det);
    double b_norms[3];
    for (int ax = 0; ax < 3; ++ax) {
        b_norms[ax] = std::sqrt(
            inv_cell[3 * ax + 0] * inv_cell[3 * ax + 0] +
            inv_cell[3 * ax + 1] * inv_cell[3 * ax + 1] +
            inv_cell[3 * ax + 2] * inv_cell[3 * ax + 2]);
    }

    for (int32_t ia = 0; ia < n_atoms; ++ia) {
        const int sp = sp_idx_ptr[ia];
        const double rc_a = rmax_ptr[sp];
        const double px = pos_ptr[3 * ia + 0], py = pos_ptr[3 * ia + 1], pz = pos_ptr[3 * ia + 2];
        const double fc0 = inv_cell[0] * px + inv_cell[1] * py + inv_cell[2] * pz;
        const double fc1 = inv_cell[3] * px + inv_cell[4] * py + inv_cell[5] * pz;
        const double fc2 = inv_cell[6] * px + inv_cell[7] * py + inv_cell[8] * pz;
        const int i0_min = static_cast<int>(std::ceil(n1 * (fc0 - rc_a * b_norms[0])));
        const int i0_max = static_cast<int>(std::floor(n1 * (fc0 + rc_a * b_norms[0])));
        const int i1_min = static_cast<int>(std::ceil(n2 * (fc1 - rc_a * b_norms[1])));
        const int i1_max = static_cast<int>(std::floor(n2 * (fc1 + rc_a * b_norms[1])));
        const int i2_min = static_cast<int>(std::ceil(n3 * (fc2 - rc_a * b_norms[2])));
        const int i2_max = static_cast<int>(std::floor(n3 * (fc2 + rc_a * b_norms[2])));
        for (int i0 = i0_min; i0 <= i0_max; ++i0) {
            const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
            const int g0 = floor_mod(i0, n1);
            for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
                const int g1 = floor_mod(i1, n2);
                for (int i2 = i2_min; i2 <= i2_max; ++i2) {
                    const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
                    const double rx = cell_ptr[0] * f0 + cell_ptr[1] * f1 + cell_ptr[2] * f2;
                    const double ry = cell_ptr[3] * f0 + cell_ptr[4] * f1 + cell_ptr[5] * f2;
                    const double rz = cell_ptr[6] * f0 + cell_ptr[7] * f1 + cell_ptr[8] * f2;
                    const double dx = rx - px, dy = ry - py, dz = rz - pz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_a * rc_a || r2 < 1e-24) continue;
                    const double r = std::sqrt(r2);
                    const int g2 = floor_mod(i2, n3);
                    const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;
                    double dtc = 0.0;
                    s_tcore[sp].eval(r, &dtc);
                    const double coeff = (wt_ptr[p_idx] * dtc) / r;
                    const double gx = coeff * dx, gy = coeff * dy, gz = coeff * dz;
                    d_pos_ptr[3 * ia + 0] -= gx;
                    d_pos_ptr[3 * ia + 1] -= gy;
                    d_pos_ptr[3 * ia + 2] -= gz;
                    const double df[3] = {f0, f1, f2};
                    for (int j = 0; j < 3; ++j) {
                        d_cell_ptr[0 * 3 + j] += gx * df[j];
                        d_cell_ptr[1 * 3 + j] += gy * df[j];
                        d_cell_ptr[2 * 3 + j] += gz * df[j];
                    }
                }
            }
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauofdHandler, TauofdImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VmatTauHandler, VmatTauImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauDfscfGradHandler, TauDfscfGradImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>() .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauCoreGridHandler, TauCoreGridImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauCoreGridGradHandler, TauCoreGridGradImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>() .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>() .Ret<ffi::Buffer<ffi::DataType::F64>>());

}  // namespace

namespace jsiesta::grid {

static_assert(sizeof(GridAtomImage) == 5 * sizeof(std::int32_t), "GridAtomImage must be five packed int32");

std::uint64_t grid_plan_input_hash(
    const double* positions, const double* cell,
    const double* pao_r_max, const double* pao_rcut, const std::int32_t* pao_l, const double* pao_values,
    const std::int32_t* orb_atom, const std::int32_t* orb_g_shell, const std::int32_t* orb_m, const double* orb_rcut,
    const std::int32_t* row_ptr, const std::int32_t* dst_unit_orb, const std::int32_t* edge_cell_idx,
    const std::int32_t* cell_shifts,
    std::int32_t n_atoms, std::int32_t n_orb, std::int32_t n_shells, std::int32_t n_pao_pts,
    std::int32_t n_edges, std::int32_t n_shifts,
    std::int64_t n1, std::int64_t n2, std::int64_t n3) {
    return GridKernelPlan::compute_base_hash(
        positions, cell, pao_r_max, pao_rcut, pao_l, pao_values, orb_atom, orb_g_shell, orb_m, orb_rcut,
        row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3);
}

std::uint64_t grid_plan_cache_key(std::uint64_t input_hash, bool need_grad, bool need_hess) {
    if (need_hess) need_grad = true;  // as in get_or_build_grid_plan
    return GridKernelPlan::with_flags(input_hash, need_grad, need_hess);
}

FlatGridPlan get_flat_grid_plan(
    const double* positions, const double* cell,
    const double* pao_r_max, const double* pao_rcut, const std::int32_t* pao_l, const double* pao_values,
    const std::int32_t* orb_atom, const std::int32_t* orb_g_shell, const std::int32_t* orb_m, const double* orb_rcut,
    const std::int32_t* row_ptr, const std::int32_t* dst_unit_orb, const std::int32_t* edge_cell_idx,
    const std::int32_t* cell_shifts,
    std::int32_t n_atoms, std::int32_t n_orb, std::int32_t n_shells, std::int32_t n_pao_pts,
    std::int32_t n_edges, std::int32_t n_shifts,
    std::int64_t n1, std::int64_t n2, std::int64_t n3, bool need_grad, bool need_hess) {
    const auto plan = get_or_build_grid_plan(
        positions, cell, pao_r_max, pao_rcut, pao_l, pao_values, orb_atom, orb_g_shell, orb_m, orb_rcut,
        row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts,
        n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3, need_grad, need_hess);
    FlatGridPlan out;
    out.owner = plan;
    // key of the requested variant (the cached plan may store more arrays than requested)
    out.hash = GridKernelPlan::with_flags(plan->base_hash, need_grad || need_hess, need_hess);
    out.use_atom_blocks = plan->use_atom_blocks;
    out.has_grad = need_grad || need_hess;
    out.has_hess = need_hess;
    out.n_grid = plan->active.n_grid;
    out.n_atoms = plan->n_atoms;
    out.n_shifts = plan->n_shifts;
    out.dvol = plan->active.dvol;
    if (!plan->use_atom_blocks) return out;
    for (int ax = 0; ax < 3; ++ax) {
        out.s_min[ax] = plan->s_min[ax];
        out.s_dim[ax] = plan->s_dim[ax];
    }
    out.total_blk_scalars = plan->total_blk_scalars;
    out.grid_img_offsets = plan->grid_img_offsets.data();
    out.img = reinterpret_cast<const std::int32_t*>(plan->img_groups.data());
    out.n_img = plan->img_groups.size();
    out.img_vals = plan->img_vals.data();
    out.n_vals = plan->img_vals.size();
    if (need_grad || need_hess) {
        out.img_gx = plan->img_gx.data();
        out.img_gy = plan->img_gy.data();
        out.img_gz = plan->img_gz.data();
    }
    if (need_hess) {
        out.img_hxx = plan->img_hxx.data();
        out.img_hxy = plan->img_hxy.data();
        out.img_hxz = plan->img_hxz.data();
        out.img_hyy = plan->img_hyy.data();
        out.img_hyz = plan->img_hyz.data();
        out.img_hzz = plan->img_hzz.data();
    }
    out.atom_n_orbs = plan->atom_n_orbs.data();
    out.shift_box = plan->shift_box.data();
    out.shift_box_len = plan->shift_box.size();
    out.pair_block_id = plan->pair_block_id.data();
    out.pair_block_len = plan->pair_block_id.size();
    out.block_offset.reserve(plan->blocks.size());
    for (const auto& blk : plan->blocks) out.block_offset.push_back(blk.blk_scalar_off);
    out.block_edge_abs.reserve(plan->active_block_edges.size());
    out.block_edge_idx.reserve(plan->active_block_edges.size());
    for (const auto& blk : plan->blocks) {
        const std::int32_t end_e = blk.edge_list_off + blk.n_valid_edges;
        for (std::int32_t ei = blk.edge_list_off; ei < end_e; ++ei) {
            const auto& me = plan->active_block_edges[ei];
            out.block_edge_abs.push_back(blk.blk_scalar_off + me.blk_rel);
            out.block_edge_idx.push_back(me.e_idx);
        }
    }
    return out;
}

}  // namespace jsiesta::grid

extern "C" {

XLA_FFI_Error* jsiesta_rhoofd_f64(XLA_FFI_CallFrame* call) { return RhoofdHandler(call); }
XLA_FFI_Error* jsiesta_vmat_f64(XLA_FFI_CallFrame* call) { return VmatHandler(call); }
XLA_FFI_Error* jsiesta_dfscf_grad_f64(XLA_FFI_CallFrame* call) { return DfscfGradHandler(call); }
XLA_FFI_Error* jsiesta_na_grid_f64(XLA_FFI_CallFrame* call) { return NaGridHandler(call); }
XLA_FFI_Error* jsiesta_na_grid_grad_f64(XLA_FFI_CallFrame* call) { return NaGridGradHandler(call); }
XLA_FFI_Error* jsiesta_tauofd_f64(XLA_FFI_CallFrame* call) { return TauofdHandler(call); }
XLA_FFI_Error* jsiesta_vmat_tau_f64(XLA_FFI_CallFrame* call) { return VmatTauHandler(call); }
XLA_FFI_Error* jsiesta_tau_dfscf_grad_f64(XLA_FFI_CallFrame* call) { return TauDfscfGradHandler(call); }
XLA_FFI_Error* jsiesta_tau_core_f64(XLA_FFI_CallFrame* call) { return TauCoreGridHandler(call); }
XLA_FFI_Error* jsiesta_tau_core_grad_f64(XLA_FFI_CallFrame* call) { return TauCoreGridGradHandler(call); }

}  // extern "C"
