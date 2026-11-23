/**
 * @file rhoofd_vmat.cu
 * @brief CUDA XLA FFI device kernels for GPU-resident \c rhoofd, \c vmat, \c dfscf, \c na_grid, and \c xc_eval.
 *
 * @par Theory
 * Executes bounded-sphere orbital-grid density scatter (\c rhoofd), adjoint potential
 * gather (\c vmat), orbital/neutral-atom coordinate and cell gradients (\c dfscf_grad,
 * \c na_grid_grad), neutral-atom grid assembly (\c na_grid), and semilocal Dual-AD
 * exchange-correlation (\c xc_eval) directly on CUDA device memory using the same
 * natural cubic radial splines and solid harmonics as the CPU FFI backend.
 *
 * @par Algorithm
 * Launches stream-ordered setup kernels (Thomas tridiagonal splines + cell inverse)
 * followed by parallel edge-box / atom-box / grid-stride CUDA kernels on
 * \c xla::ffi::PlatformStream<cudaStream_t>, with \c cudaMallocAsync/\c cudaFreeAsync
 * workspace buffers so zero host-device synchronization occurs during execution.
 *
 * @par Invariants
 * - Zero host synchronization inside SCF or force/stress steps.
 * - Matches CPU \c grid_ffi.cpp and \c xc_ffi.cpp to IEEE-754 double precision.
 *
 * @par SIESTA Mapping
 * GPU counterpart to \c rhoofd.F90, \c vmat.F90, \c dfscf.F, \c dhscf.F, and \c cellxc.F.
 */

#include "kernels.cuh"
#include "jsiesta/grid_xc_ffi.hpp"
#include "jsiesta/grid_plan.hpp"
#include "jsiesta/xc_semilocal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"
#endif

namespace jsiesta::cuda {

GridTileConfig select_optimal_tile_config(int n1, int n2, int n3, int n_orb_avg) {
    if (n_orb_avg <= 64 && n3 >= 2 * std::max(n1, n2)) {
        return GridTileConfig{4, 4, 8};
    }
    if (n_orb_avg <= 64 && n1 >= 2 * std::max(n2, n3)) {
        return GridTileConfig{16, 4, 4};
    }
    return GridTileConfig{8, 8, 4};
}

#ifdef __CUDACC__

namespace ffi = xla::ffi;

namespace {

constexpr double kDevPi = 3.1415926535897932384626433832795;
constexpr int kMaxL = 6;
constexpr int kStride = kMaxL + 1;
constexpr int kShCount = kStride * kStride;

#define CUDA_FFI_CHECK(expr)                                                          \
    do {                                                                              \
        cudaError_t _err = (expr);                                                    \
        if (_err != cudaSuccess) {                                                    \
            return ffi::Error::Internal(std::string("CUDA error: ") +                 \
                                        cudaGetErrorString(_err));                    \
        }                                                                             \
    } while (0)

__device__ __forceinline__ int dev_floor_mod(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}

__device__ __forceinline__ int dev_binary_search_src_orb(
    const int32_t* __restrict__ row_ptr, int32_t n_orb, int32_t edge_idx) {
    int lo = 0;
    int hi = n_orb - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (row_ptr[mid] <= edge_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

__device__ __forceinline__ double dev_eval_spline(
    const double* __restrict__ y,
    const double* __restrict__ y2,
    int n_pts,
    double r_max,
    double r_cut,
    double r,
    double* dfdr,
    double* d2fdr2 = nullptr) {
    if (r >= r_cut || n_pts < 2 || r_max <= 0.0) {
        if (dfdr) *dfdr = 0.0;
        if (d2fdr2) *d2fdr2 = 0.0;
        return 0.0;
    }
    const double dr = r_max / static_cast<double>(n_pts - 1);
    const double pos = fmax(0.0, r) / dr;
    int klo = static_cast<int>(pos);
    if (klo >= n_pts - 1) {
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

template <bool NeedGrad, bool NeedHess = false>
__device__ void dev_real_solid_harmonics(
    double x, double y, double z, int l_max, double* c_lm, double* dc_lm, double* hc_lm = nullptr) {
    double A[kShCount];
    double B[kShCount];
    for (int i = 0; i < kShCount; ++i) {
        A[i] = 0.0;
        B[i] = 0.0;
    }
    double dAx[kShCount], dAy[kShCount], dAz[kShCount];
    double dBx[kShCount], dBy[kShCount], dBz[kShCount];
    if constexpr (NeedGrad || NeedHess) {
        for (int i = 0; i < kShCount; ++i) {
            dAx[i] = dAy[i] = dAz[i] = 0.0;
            dBx[i] = dBy[i] = dBz[i] = 0.0;
        }
    }
    double hxxA[kShCount], hxyA[kShCount], hxzA[kShCount], hyyA[kShCount], hyzA[kShCount], hzzA[kShCount];
    double hxxB[kShCount], hxyB[kShCount], hxzB[kShCount], hyyB[kShCount], hyzB[kShCount], hzzB[kShCount];
    if constexpr (NeedHess) {
        for (int i = 0; i < kShCount; ++i) {
            hxxA[i] = hxyA[i] = hxzA[i] = hyyA[i] = hyzA[i] = hzzA[i] = 0.0;
            hxxB[i] = hxyB[i] = hxzB[i] = hyyB[i] = hyzB[i] = hzzB[i] = 0.0;
        }
    }

    auto idx = [](int l, int m) -> int { return l * kStride + m; };
    const double r2 = x * x + y * y + z * z;
    A[idx(0, 0)] = 1.0;

    for (int l = 0; l < l_max; ++l) {
        const double f = static_cast<double>(2 * l + 1);
        const int cur = idx(l, l);
        const int nxt = idx(l + 1, l + 1);
        A[nxt] = f * (x * A[cur] - y * B[cur]);
        B[nxt] = f * (x * B[cur] + y * A[cur]);
        if constexpr (NeedGrad || NeedHess) {
            dAx[nxt] = f * (A[cur] + x * dAx[cur] - y * dBx[cur]);
            dAy[nxt] = f * (x * dAy[cur] - B[cur] - y * dBy[cur]);
            dAz[nxt] = f * (x * dAz[cur] - y * dBz[cur]);
            dBx[nxt] = f * (B[cur] + x * dBx[cur] + y * dAx[cur]);
            dBy[nxt] = f * (x * dBy[cur] + A[cur] + y * dAy[cur]);
            dBz[nxt] = f * (x * dBz[cur] + y * dAz[cur]);
        }
        if constexpr (NeedHess) {
            hxxA[nxt] = f * (2.0 * dAx[cur] + x * hxxA[cur] - y * hxxB[cur]);
            hxyA[nxt] = f * (dAy[cur] + x * hxyA[cur] - dBx[cur] - y * hxyB[cur]);
            hxzA[nxt] = f * (dAz[cur] + x * hxzA[cur] - y * hxzB[cur]);
            hyyA[nxt] = f * (x * hyyA[cur] - 2.0 * dBy[cur] - y * hyyB[cur]);
            hyzA[nxt] = f * (x * hyzA[cur] - dBz[cur] - y * hyzB[cur]);
            hzzA[nxt] = f * (x * hzzA[cur] - y * hzzB[cur]);

            hxxB[nxt] = f * (2.0 * dBx[cur] + x * hxxB[cur] + y * hxxA[cur]);
            hxyB[nxt] = f * (dBy[cur] + x * hxyB[cur] + dAx[cur] + y * hxyA[cur]);
            hxzB[nxt] = f * (dBz[cur] + x * hxzB[cur] + y * hxzA[cur]);
            hyyB[nxt] = f * (x * hyyB[cur] + 2.0 * dAy[cur] + y * hyyA[cur]);
            hyzB[nxt] = f * (x * hyzB[cur] + dAz[cur] + y * hyzA[cur]);
            hzzB[nxt] = f * (x * hzzB[cur] + y * hzzA[cur]);
        }
    }

    for (int m = 0; m < l_max; ++m) {
        const double f = static_cast<double>(2 * m + 1);
        const int cur = idx(m, m);
        const int nxt = idx(m + 1, m);
        A[nxt] = f * z * A[cur];
        B[nxt] = f * z * B[cur];
        if constexpr (NeedGrad || NeedHess) {
            dAx[nxt] = f * z * dAx[cur];
            dAy[nxt] = f * z * dAy[cur];
            dAz[nxt] = f * (A[cur] + z * dAz[cur]);
            dBx[nxt] = f * z * dBx[cur];
            dBy[nxt] = f * z * dBy[cur];
            dBz[nxt] = f * (B[cur] + z * dBz[cur]);
        }
        if constexpr (NeedHess) {
            hxxA[nxt] = f * z * hxxA[cur];
            hxyA[nxt] = f * z * hxyA[cur];
            hxzA[nxt] = f * (dAx[cur] + z * hxzA[cur]);
            hyyA[nxt] = f * z * hyyA[cur];
            hyzA[nxt] = f * (dAy[cur] + z * hyzA[cur]);
            hzzA[nxt] = f * (2.0 * dAz[cur] + z * hzzA[cur]);

            hxxB[nxt] = f * z * hxxB[cur];
            hxyB[nxt] = f * z * hxyB[cur];
            hxzB[nxt] = f * (dBx[cur] + z * hxzB[cur]);
            hyyB[nxt] = f * z * hyyB[cur];
            hyzB[nxt] = f * (dBy[cur] + z * hyzB[cur]);
            hzzB[nxt] = f * (2.0 * dBz[cur] + z * hzzB[cur]);
        }
        for (int l = m + 1; l < l_max; ++l) {
            const double c1 = static_cast<double>(2 * l + 1) / static_cast<double>(l + 1 - m);
            const double c2 = static_cast<double>(l + m) / static_cast<double>(l + 1 - m);
            const int i_cur = idx(l, m);
            const int i_prev = idx(l - 1, m);
            const int i_nxt = idx(l + 1, m);
            A[i_nxt] = c1 * z * A[i_cur] - c2 * r2 * A[i_prev];
            B[i_nxt] = c1 * z * B[i_cur] - c2 * r2 * B[i_prev];
            if constexpr (NeedGrad || NeedHess) {
                dAx[i_nxt] = c1 * z * dAx[i_cur] - c2 * (2.0 * x * A[i_prev] + r2 * dAx[i_prev]);
                dAy[i_nxt] = c1 * z * dAy[i_cur] - c2 * (2.0 * y * A[i_prev] + r2 * dAy[i_prev]);
                dAz[i_nxt] = c1 * (A[i_cur] + z * dAz[i_cur]) - c2 * (2.0 * z * A[i_prev] + r2 * dAz[i_prev]);
                dBx[i_nxt] = c1 * z * dBx[i_cur] - c2 * (2.0 * x * B[i_prev] + r2 * dBx[i_prev]);
                dBy[i_nxt] = c1 * z * dBy[i_cur] - c2 * (2.0 * y * B[i_prev] + r2 * dBy[i_prev]);
                dBz[i_nxt] = c1 * (B[i_cur] + z * dBz[i_cur]) - c2 * (2.0 * z * B[i_prev] + r2 * dBz[i_prev]);
            }
            if constexpr (NeedHess) {
                hxxA[i_nxt] = c1 * z * hxxA[i_cur] - c2 * (2.0 * A[i_prev] + 4.0 * x * dAx[i_prev] + r2 * hxxA[i_prev]);
                hxyA[i_nxt] = c1 * z * hxyA[i_cur] - c2 * (2.0 * x * dAy[i_prev] + 2.0 * y * dAx[i_prev] + r2 * hxyA[i_prev]);
                hxzA[i_nxt] = c1 * (dAx[i_cur] + z * hxzA[i_cur]) - c2 * (2.0 * x * dAz[i_prev] + 2.0 * z * dAx[i_prev] + r2 * hxzA[i_prev]);
                hyyA[i_nxt] = c1 * z * hyyA[i_cur] - c2 * (2.0 * A[i_prev] + 4.0 * y * dAy[i_prev] + r2 * hyyA[i_prev]);
                hyzA[i_nxt] = c1 * (dAy[i_cur] + z * hyzA[i_cur]) - c2 * (2.0 * y * dAz[i_prev] + 2.0 * z * dAy[i_prev] + r2 * hyzA[i_prev]);
                hzzA[i_nxt] = c1 * (2.0 * dAz[i_cur] + z * hzzA[i_cur]) - c2 * (2.0 * A[i_prev] + 4.0 * z * dAz[i_prev] + r2 * hzzA[i_prev]);

                hxxB[i_nxt] = c1 * z * hxxB[i_cur] - c2 * (2.0 * B[i_prev] + 4.0 * x * dBx[i_prev] + r2 * hxxB[i_prev]);
                hxyB[i_nxt] = c1 * z * hxyB[i_cur] - c2 * (2.0 * x * dBy[i_prev] + 2.0 * y * dBx[i_prev] + r2 * hxyB[i_prev]);
                hxzB[i_nxt] = c1 * (dBx[i_cur] + z * hxzB[i_cur]) - c2 * (2.0 * x * dBz[i_prev] + 2.0 * z * dBx[i_prev] + r2 * hxzB[i_prev]);
                hyyB[i_nxt] = c1 * z * hyyB[i_cur] - c2 * (2.0 * B[i_prev] + 4.0 * y * dBy[i_prev] + r2 * hyyB[i_prev]);
                hyzB[i_nxt] = c1 * (dBy[i_cur] + z * hyzB[i_cur]) - c2 * (2.0 * y * dBz[i_prev] + 2.0 * z * dBy[i_prev] + r2 * hyzB[i_prev]);
                hzzB[i_nxt] = c1 * (2.0 * dBz[i_cur] + z * hzzB[i_cur]) - c2 * (2.0 * B[i_prev] + 4.0 * z * dBz[i_prev] + r2 * hzzB[i_prev]);
            }
        }
    }

    double fact[2 * kMaxL + 2];
    fact[0] = 1.0;
    for (int k = 1; k < 2 * l_max + 2; ++k) {
        fact[k] = fact[k - 1] * static_cast<double>(k);
    }

    for (int l = 0; l <= l_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int am = m < 0 ? -m : m;
            const double norm = sqrt(
                (static_cast<double>(2 * l + 1) / (4.0 * kDevPi)) *
                (fact[l - am] / fact[l + am]) *
                (m == 0 ? 1.0 : 2.0));
            const int out_idx = l * l + (m + l);
            const int tbl_idx = idx(l, am);
            if (m < 0) {
                c_lm[out_idx] = norm * B[tbl_idx];
                if constexpr (NeedGrad || NeedHess) {
                    dc_lm[3 * out_idx + 0] = norm * dBx[tbl_idx];
                    dc_lm[3 * out_idx + 1] = norm * dBy[tbl_idx];
                    dc_lm[3 * out_idx + 2] = norm * dBz[tbl_idx];
                }
                if constexpr (NeedHess) {
                    double* h = hc_lm + 9 * out_idx;
                    h[0] = norm * hxxB[tbl_idx]; h[1] = norm * hxyB[tbl_idx]; h[2] = norm * hxzB[tbl_idx];
                    h[3] = h[1];                  h[4] = norm * hyyB[tbl_idx]; h[5] = norm * hyzB[tbl_idx];
                    h[6] = h[2];                  h[7] = h[5];                  h[8] = norm * hzzB[tbl_idx];
                }
            } else {
                c_lm[out_idx] = norm * A[tbl_idx];
                if constexpr (NeedGrad || NeedHess) {
                    dc_lm[3 * out_idx + 0] = norm * dAx[tbl_idx];
                    dc_lm[3 * out_idx + 1] = norm * dAy[tbl_idx];
                    dc_lm[3 * out_idx + 2] = norm * dAz[tbl_idx];
                }
                if constexpr (NeedHess) {
                    double* h = hc_lm + 9 * out_idx;
                    h[0] = norm * hxxA[tbl_idx]; h[1] = norm * hxyA[tbl_idx]; h[2] = norm * hxzA[tbl_idx];
                    h[3] = h[1];                  h[4] = norm * hyyA[tbl_idx]; h[5] = norm * hyzA[tbl_idx];
                    h[6] = h[2];                  h[7] = h[5];                  h[8] = norm * hzzA[tbl_idx];
                }
            }
        }
    }
}

template <bool NeedHess>
__device__ __forceinline__ void dev_eval_orb_grad_hess(
    double dx, double dy, double dz, double r,
    const double* __restrict__ f, const double* __restrict__ y2,
    int n_pao_pts, double rmax, double rcut,
    int lm, const double* c_lm, const double* dc_lm, const double* hc_lm,
    double* grad_out, double* hess_sym6_out) {
    double df = 0.0, d2f = 0.0;
    const double g = dev_eval_spline(f, y2, n_pao_pts, rmax, rcut, r, &df, NeedHess ? &d2f : nullptr);
    double g_over_r = 0.0;
    if (r > 1e-8) {
        g_over_r = df / r;
    } else {
        double df_eps = 0.0;
        dev_eval_spline(f, y2, n_pao_pts, rmax, rcut, 1e-4, &df_eps);
        g_over_r = df_eps / 1e-4;
    }
    const double c = c_lm[lm];
    const double dcx = dc_lm[3 * lm + 0];
    const double dcy = dc_lm[3 * lm + 1];
    const double dcz = dc_lm[3 * lm + 2];
    grad_out[0] = g_over_r * dx * c + g * dcx;
    grad_out[1] = g_over_r * dy * c + g * dcy;
    grad_out[2] = g_over_r * dz * c + g * dcz;
    if constexpr (NeedHess) {
        const double s_r = (r > 1e-5) ? ((d2f - g_over_r) / (r * r)) : 0.0;
        const double* hlm = hc_lm + 9 * lm;
        hess_sym6_out[0] = s_r * dx * dx * c + g_over_r * (c + 2.0 * dx * dcx) + g * hlm[0];
        hess_sym6_out[1] = s_r * dx * dy * c + g_over_r * (dx * dcy + dy * dcx) + g * hlm[1];
        hess_sym6_out[2] = s_r * dx * dz * c + g_over_r * (dx * dcz + dz * dcx) + g * hlm[2];
        hess_sym6_out[3] = s_r * dy * dy * c + g_over_r * (c + 2.0 * dy * dcy) + g * hlm[4];
        hess_sym6_out[4] = s_r * dy * dz * c + g_over_r * (dy * dcz + dz * dcy) + g * hlm[5];
        hess_sym6_out[5] = s_r * dz * dz * c + g_over_r * (c + 2.0 * dz * dcz) + g * hlm[8];
    }
}

// geom_dev stores: [0..8] = inv_cell (row major), [9..11] = b_norms, [12] = dvol, [13] = l_max (as double)
__global__ void prepare_geom_and_pao_splines_kernel(
    const double* __restrict__ cell,
    const double* __restrict__ pao_r_max,
    const int32_t* __restrict__ pao_l,
    const double* __restrict__ pao_values,
    int32_t n_shells,
    int32_t n_pao_pts,
    int64_t n_grid,
    double* __restrict__ geom_dev,
    double* __restrict__ f_dev,
    double* __restrict__ y2_dev,
    double* __restrict__ u_dev) {
    const int s = blockIdx.x;
    if (s == 0 && threadIdx.x == 0) {
        const double a00 = cell[0], a01 = cell[1], a02 = cell[2];
        const double a10 = cell[3], a11 = cell[4], a12 = cell[5];
        const double a20 = cell[6], a21 = cell[7], a22 = cell[8];
        const double c00 = a11 * a22 - a12 * a21;
        const double c01 = a12 * a20 - a10 * a22;
        const double c02 = a10 * a21 - a11 * a20;
        const double c10 = a02 * a21 - a01 * a22;
        const double c11 = a00 * a22 - a02 * a20;
        const double c12 = a01 * a20 - a00 * a21;
        const double c20 = a01 * a12 - a02 * a11;
        const double c21 = a02 * a10 - a00 * a12;
        const double c22 = a00 * a11 - a01 * a10;
        const double det = a00 * c00 + a01 * c01 + a02 * c02;
        const double inv_det = 1.0 / det;
        geom_dev[0] = c00 * inv_det; geom_dev[1] = c10 * inv_det; geom_dev[2] = c20 * inv_det;
        geom_dev[3] = c01 * inv_det; geom_dev[4] = c11 * inv_det; geom_dev[5] = c21 * inv_det;
        geom_dev[6] = c02 * inv_det; geom_dev[7] = c12 * inv_det; geom_dev[8] = c22 * inv_det;
        for (int ax = 0; ax < 3; ++ax) {
            const double bx = geom_dev[3 * ax + 0];
            const double by = geom_dev[3 * ax + 1];
            const double bz = geom_dev[3 * ax + 2];
            geom_dev[9 + ax] = sqrt(bx * bx + by * by + bz * bz);
        }
        geom_dev[12] = fabs(det) / static_cast<double>(n_grid);
        int max_l = 0;
        for (int i = 0; i < n_shells; ++i) {
            if (pao_l[i] > max_l) max_l = pao_l[i];
        }
        geom_dev[13] = static_cast<double>(max_l);
    }

    if (s < n_shells && threadIdx.x == 0) {
        const int l = pao_l[s];
        const double r_max = pao_r_max[s];
        const double dr = (n_pao_pts > 1 && r_max > 0.0) ? (r_max / static_cast<double>(n_pao_pts - 1)) : 1.0;
        const double* src = pao_values + static_cast<size_t>(s) * n_pao_pts;
        double* f = f_dev + static_cast<size_t>(s) * n_pao_pts;
        double* y2 = y2_dev + static_cast<size_t>(s) * n_pao_pts;
        double* u = u_dev + static_cast<size_t>(s) * n_pao_pts;

        for (int ir = 0; ir < n_pao_pts; ++ir) {
            y2[ir] = 0.0;
            u[ir] = 0.0;
            if (l == 0) {
                f[ir] = src[ir];
            } else if (ir > 0) {
                const double r = static_cast<double>(ir) * dr;
                f[ir] = src[ir] / pow(r, l);
            } else {
                f[0] = 0.0;
            }
        }
        if (l > 0 && n_pao_pts >= 3) {
            f[0] = (4.0 * f[1] - f[2]) / 3.0;
        }
        if (n_pao_pts >= 3 && r_max > 0.0) {
            for (int i = 1; i < n_pao_pts - 1; ++i) {
                const double p = 0.5 * y2[i - 1] + 2.0;
                y2[i] = -0.5 / p;
                const double rhs = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (dr * dr);
                u[i] = (3.0 * rhs - 0.5 * u[i - 1]) / p;
            }
            for (int k = n_pao_pts - 2; k >= 0; --k) {
                y2[k] = y2[k] * y2[k + 1] + u[k];
            }
        }
    }
}

__global__ void prepare_na_geom_and_splines_kernel(
    const double* __restrict__ cell,
    const double* __restrict__ sp_rmax,
    const double* __restrict__ sp_rho_na,
    const double* __restrict__ sp_v_na,
    const double* __restrict__ sp_rho_core,
    int32_t n_sp,
    int32_t n_pts,
    double* __restrict__ geom_dev,
    double* __restrict__ y2_dev,
    double* __restrict__ u_dev) {
    const int tid = blockIdx.x;
    if (tid == 0 && threadIdx.x == 0) {
        const double a00 = cell[0], a01 = cell[1], a02 = cell[2];
        const double a10 = cell[3], a11 = cell[4], a12 = cell[5];
        const double a20 = cell[6], a21 = cell[7], a22 = cell[8];
        const double c00 = a11 * a22 - a12 * a21;
        const double c01 = a12 * a20 - a10 * a22;
        const double c02 = a10 * a21 - a11 * a20;
        const double c10 = a02 * a21 - a01 * a22;
        const double c11 = a00 * a22 - a02 * a20;
        const double c12 = a01 * a20 - a00 * a21;
        const double c20 = a01 * a12 - a02 * a11;
        const double c21 = a02 * a10 - a00 * a12;
        const double c22 = a00 * a11 - a01 * a10;
        const double det = a00 * c00 + a01 * c01 + a02 * c02;
        const double inv_det = 1.0 / det;
        geom_dev[0] = c00 * inv_det; geom_dev[1] = c10 * inv_det; geom_dev[2] = c20 * inv_det;
        geom_dev[3] = c01 * inv_det; geom_dev[4] = c11 * inv_det; geom_dev[5] = c21 * inv_det;
        geom_dev[6] = c02 * inv_det; geom_dev[7] = c12 * inv_det; geom_dev[8] = c22 * inv_det;
        for (int ax = 0; ax < 3; ++ax) {
            const double bx = geom_dev[3 * ax + 0];
            const double by = geom_dev[3 * ax + 1];
            const double bz = geom_dev[3 * ax + 2];
            geom_dev[9 + ax] = sqrt(bx * bx + by * by + bz * bz);
        }
    }
    if (tid < 3 * n_sp && threadIdx.x == 0) {
        const int sp = tid / 3;
        const int kind = tid % 3;
        const double* src = (kind == 0 ? sp_rho_na : (kind == 1 ? sp_v_na : sp_rho_core))
                            + static_cast<size_t>(sp) * n_pts;
        double* y2 = y2_dev + static_cast<size_t>(tid) * n_pts;
        double* u = u_dev + static_cast<size_t>(tid) * n_pts;
        for (int i = 0; i < n_pts; ++i) {
            y2[i] = 0.0;
            u[i] = 0.0;
        }
        const double r_max = sp_rmax[sp];
        if (n_pts >= 3 && r_max > 0.0) {
            const double dr = r_max / static_cast<double>(n_pts - 1);
            for (int i = 1; i < n_pts - 1; ++i) {
                const double p = 0.5 * y2[i - 1] + 2.0;
                y2[i] = -0.5 / p;
                const double rhs = (src[i + 1] - 2.0 * src[i] + src[i - 1]) / (dr * dr);
                u[i] = (3.0 * rhs - 0.5 * u[i - 1]) / p;
            }
            for (int k = n_pts - 2; k >= 0; --k) {
                y2[k] = y2[k] * y2[k + 1] + u[k];
            }
        }
    }
}

template <int Mode>  // 0: rhoofd, 1: vmat, 2: dfscf_grad, 3: tauofd, 4: vmat_tau, 5: tau_dfscf_grad
__global__ void edge_orbital_grid_kernel(
    const double* __restrict__ primary_buf,     // Mode 0/2: dm_edges, Mode 1: v_grid
    const double* __restrict__ v_cot_buf,       // Mode 2 only: v_cotangent
    const double* __restrict__ positions,
    const double* __restrict__ cell,
    const double* __restrict__ geom_dev,
    const double* __restrict__ pao_r_max,
    const double* __restrict__ pao_rcut,
    const int32_t* __restrict__ pao_l,
    const double* __restrict__ f_dev,
    const double* __restrict__ y2_dev,
    const int32_t* __restrict__ orb_atom,
    const int32_t* __restrict__ orb_g_shell,
    const int32_t* __restrict__ orb_m,
    const double* __restrict__ orb_rcut,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ dst_unit_orb,
    const int32_t* __restrict__ edge_cell_idx,
    const int32_t* __restrict__ cell_shifts,
    int32_t n_orb,
    int32_t n_pao_pts,
    int32_t n_edges,
    int32_t n1,
    int32_t n2,
    int32_t n3,
    int32_t n_spin_mat,
    double* __restrict__ out_main,      // Mode 0: out_rho, Mode 1: out_v, Mode 2: out_d_dm
    double* __restrict__ out_d_pos,     // Mode 2 only
    double* __restrict__ out_d_cell) {  // Mode 2 only

    const int32_t e = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_edges) return;

    const int32_t mu = dev_binary_search_src_orb(row_ptr, n_orb, e);
    const int32_t nu = dst_unit_orb[e];
    const int32_t c_idx = edge_cell_idx[e];
    const int32_t t0 = cell_shifts[3 * c_idx + 0];
    const int32_t t1 = cell_shifts[3 * c_idx + 1];
    const int32_t t2 = cell_shifts[3 * c_idx + 2];

    const int32_t atom_a = orb_atom[mu];
    const int32_t atom_b = orb_atom[nu];
    const double rc_a = orb_rcut[mu];
    const double rc_b = orb_rcut[nu];

    const double ax = positions[3 * atom_a + 0];
    const double ay = positions[3 * atom_a + 1];
    const double az = positions[3 * atom_a + 2];

    const double bx = positions[3 * atom_b + 0]
        + cell[0] * t0 + cell[1] * t1 + cell[2] * t2;
    const double by = positions[3 * atom_b + 1]
        + cell[3] * t0 + cell[4] * t1 + cell[5] * t2;
    const double bz = positions[3 * atom_b + 2]
        + cell[6] * t0 + cell[7] * t1 + cell[8] * t2;

    const double abx = bx - ax, aby = by - ay, abz = bz - az;
    const double rsum = rc_a + rc_b;
    constexpr bool kIsTau = (Mode >= 3);
    constexpr bool kIsScatter = (Mode == 0 || Mode == 3);
    constexpr bool kIsVmat = (Mode == 1 || Mode == 4);
    constexpr bool kIsBwd = (Mode == 2 || Mode == 5);
    if (abx * abx + aby * aby + abz * abz >= rsum * rsum) {
        if constexpr (kIsVmat || kIsBwd) {
            for (int32_t s = 0; s < n_spin_mat; ++s) {
                out_main[static_cast<size_t>(s) * n_edges + e] = 0.0;
            }
        }
        return;
    }

    const double inv0 = geom_dev[0], inv1 = geom_dev[1], inv2 = geom_dev[2];
    const double inv3 = geom_dev[3], inv4 = geom_dev[4], inv5 = geom_dev[5];
    const double inv6 = geom_dev[6], inv7 = geom_dev[7], inv8 = geom_dev[8];
    const double bn0 = geom_dev[9], bn1 = geom_dev[10], bn2 = geom_dev[11];
    const double dvol = geom_dev[12];

    const double fca0 = inv0 * ax + inv1 * ay + inv2 * az;
    const double fca1 = inv3 * ax + inv4 * ay + inv5 * az;
    const double fca2 = inv6 * ax + inv7 * ay + inv8 * az;
    const double fcb0 = inv0 * bx + inv1 * by + inv2 * bz;
    const double fcb1 = inv3 * bx + inv4 * by + inv5 * bz;
    const double fcb2 = inv6 * bx + inv7 * by + inv8 * bz;

    const int i0_lo = max(static_cast<int>(ceil(n1 * (fca0 - rc_a * bn0))),
                          static_cast<int>(ceil(n1 * (fcb0 - rc_b * bn0))));
    const int i0_hi = min(static_cast<int>(floor(n1 * (fca0 + rc_a * bn0))),
                          static_cast<int>(floor(n1 * (fcb0 + rc_b * bn0))));
    const int i1_lo = max(static_cast<int>(ceil(n2 * (fca1 - rc_a * bn1))),
                          static_cast<int>(ceil(n2 * (fcb1 - rc_b * bn1))));
    const int i1_hi = min(static_cast<int>(floor(n2 * (fca1 + rc_a * bn1))),
                          static_cast<int>(floor(n2 * (fcb1 + rc_b * bn1))));
    const int i2_lo = max(static_cast<int>(ceil(n3 * (fca2 - rc_a * bn2))),
                          static_cast<int>(ceil(n3 * (fcb2 - rc_b * bn2))));
    const int i2_hi = min(static_cast<int>(floor(n3 * (fca2 + rc_a * bn2))),
                          static_cast<int>(floor(n3 * (fcb2 + rc_b * bn2))));

    if (i0_lo > i0_hi || i1_lo > i1_hi || i2_lo > i2_hi) {
        if constexpr (kIsVmat || kIsBwd) {
            for (int32_t s = 0; s < n_spin_mat; ++s) {
                out_main[static_cast<size_t>(s) * n_edges + e] = 0.0;
            }
        }
        return;
    }

    const int sh_a = orb_g_shell[mu];
    const int sh_b = orb_g_shell[nu];
    const int l_a = pao_l[sh_a];
    const int l_b = pao_l[sh_b];
    const int lm_a = l_a * l_a + (orb_m[mu] + l_a);
    const int lm_b = l_b * l_b + (orb_m[nu] + l_b);
    const double* fa = f_dev + static_cast<size_t>(sh_a) * n_pao_pts;
    const double* y2a = y2_dev + static_cast<size_t>(sh_a) * n_pao_pts;
    const double* fb = f_dev + static_cast<size_t>(sh_b) * n_pao_pts;
    const double* y2b = y2_dev + static_cast<size_t>(sh_b) * n_pao_pts;
    const double rmax_a = pao_r_max[sh_a], rcut_a = pao_rcut[sh_a];
    const double rmax_b = pao_r_max[sh_b], rcut_b = pao_rcut[sh_b];
    const int l_req_a = l_a > kMaxL ? kMaxL : l_a;
    const int l_req_b = l_b > kMaxL ? kMaxL : l_b;

    const int64_t n_grid = static_cast<int64_t>(n1) * n2 * n3;
    double edge_acc[4] = {0.0, 0.0, 0.0, 0.0};
    double d_pa[3] = {0.0, 0.0, 0.0};
    double d_pb[3] = {0.0, 0.0, 0.0};
    double d_cell_loc[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    double dm_local[4] = {0.0, 0.0, 0.0, 0.0};
    if constexpr (kIsScatter || kIsBwd) {
        for (int32_t s = 0; s < n_spin_mat; ++s) {
            dm_local[s] = primary_buf[static_cast<size_t>(s) * n_edges + e];
        }
    }

    double c_a[kShCount], c_b[kShCount];
    double dc_a[3 * kShCount], dc_b[3 * kShCount];

    for (int i0 = i0_lo; i0 <= i0_hi; ++i0) {
        const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
        const int g0 = dev_floor_mod(i0, n1);
        for (int i1 = i1_lo; i1 <= i1_hi; ++i1) {
            const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
            const int g1 = dev_floor_mod(i1, n2);
            for (int i2 = i2_lo; i2 <= i2_hi; ++i2) {
                const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
                const double rx = cell[0] * f0 + cell[1] * f1 + cell[2] * f2;
                const double ry = cell[3] * f0 + cell[4] * f1 + cell[5] * f2;
                const double rz = cell[6] * f0 + cell[7] * f1 + cell[8] * f2;

                const double dax = rx - ax, day = ry - ay, daz = rz - az;
                const double ra2 = dax * dax + day * day + daz * daz;
                if (ra2 >= rc_a * rc_a) continue;

                const double dbx = rx - bx, dby = ry - by, dbz = rz - bz;
                const double rb2 = dbx * dbx + dby * dby + dbz * dbz;
                if (rb2 >= rc_b * rc_b) continue;

                const double ra = sqrt(ra2);
                const double rb = sqrt(rb2);
                const int g2 = dev_floor_mod(i2, n3);
                const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;

                if constexpr (Mode == 2) {
                    dev_real_solid_harmonics<true, false>(dax, day, daz, l_req_a, c_a, dc_a);
                    dev_real_solid_harmonics<true, false>(dbx, dby, dbz, l_req_b, c_b, dc_b);
                    const double ga = dev_eval_spline(fa, y2a, n_pao_pts, rmax_a, rcut_a, ra, nullptr);
                    const double gb = dev_eval_spline(fb, y2b, n_pao_pts, rmax_b, rcut_b, rb, nullptr);
                    const double phi_a = ga * c_a[lm_a];
                    const double phi_b = gb * c_b[lm_b];

                    double w_pair = 0.0;
                    const double prod = phi_a * phi_b;
                    for (int32_t s = 0; s < n_spin_mat; ++s) {
                        const double vc = v_cot_buf[static_cast<size_t>(s) * n_grid + p_idx];
                        edge_acc[s] += vc * prod;
                        w_pair += vc * dm_local[s];
                    }

                    double grad_a[3], grad_b[3];
                    dev_eval_orb_grad_hess<false>(dax, day, daz, ra, fa, y2a, n_pao_pts, rmax_a, rcut_a, lm_a, c_a, dc_a, nullptr, grad_a, nullptr);
                    dev_eval_orb_grad_hess<false>(dbx, dby, dbz, rb, fb, y2b, n_pao_pts, rmax_b, rcut_b, lm_b, c_b, dc_b, nullptr, grad_b, nullptr);

                    const double fax = w_pair * grad_a[0] * phi_b;
                    const double fay = w_pair * grad_a[1] * phi_b;
                    const double faz = w_pair * grad_a[2] * phi_b;
                    const double fbx = w_pair * phi_a * grad_b[0];
                    const double fby = w_pair * phi_a * grad_b[1];
                    const double fbz = w_pair * phi_a * grad_b[2];

                    d_pa[0] -= fax; d_pa[1] -= fay; d_pa[2] -= faz;
                    d_pb[0] -= fbx; d_pb[1] -= fby; d_pb[2] -= fbz;

                    const double dfa[3] = {f0, f1, f2};
                    const double dfb[3] = {
                        f0 - static_cast<double>(t0),
                        f1 - static_cast<double>(t1),
                        f2 - static_cast<double>(t2),
                    };
                    for (int j = 0; j < 3; ++j) {
                        d_cell_loc[0 * 3 + j] += fax * dfa[j] + fbx * dfb[j];
                        d_cell_loc[1 * 3 + j] += fay * dfa[j] + fby * dfb[j];
                        d_cell_loc[2 * 3 + j] += faz * dfa[j] + fbz * dfb[j];
                    }
                } else if constexpr (Mode == 5) {
                    double hc_a[9 * kShCount], hc_b[9 * kShCount];
                    dev_real_solid_harmonics<true, true>(dax, day, daz, l_req_a, c_a, dc_a, hc_a);
                    dev_real_solid_harmonics<true, true>(dbx, dby, dbz, l_req_b, c_b, dc_b, hc_b);
                    double grad_a[3], grad_b[3];
                    double hess_a[6], hess_b[6];
                    dev_eval_orb_grad_hess<true>(dax, day, daz, ra, fa, y2a, n_pao_pts, rmax_a, rcut_a, lm_a, c_a, dc_a, hc_a, grad_a, hess_a);
                    dev_eval_orb_grad_hess<true>(dbx, dby, dbz, rb, fb, y2b, n_pao_pts, rmax_b, rcut_b, lm_b, c_b, dc_b, hc_b, grad_b, hess_b);

                    const double prod = 0.5 * (grad_a[0] * grad_b[0] + grad_a[1] * grad_b[1] + grad_a[2] * grad_b[2]);
                    double w_pair = 0.0;
                    for (int32_t s = 0; s < n_spin_mat; ++s) {
                        const double vc = v_cot_buf[static_cast<size_t>(s) * n_grid + p_idx];
                        edge_acc[s] += vc * prod;
                        w_pair += 0.5 * vc * dm_local[s];
                    }

                    const double fax = w_pair * (hess_a[0] * grad_b[0] + hess_a[1] * grad_b[1] + hess_a[2] * grad_b[2]);
                    const double fay = w_pair * (hess_a[1] * grad_b[0] + hess_a[3] * grad_b[1] + hess_a[4] * grad_b[2]);
                    const double faz = w_pair * (hess_a[2] * grad_b[0] + hess_a[4] * grad_b[1] + hess_a[5] * grad_b[2]);
                    const double fbx = w_pair * (grad_a[0] * hess_b[0] + grad_a[1] * hess_b[1] + grad_a[2] * hess_b[2]);
                    const double fby = w_pair * (grad_a[0] * hess_b[1] + grad_a[1] * hess_b[3] + grad_a[2] * hess_b[4]);
                    const double fbz = w_pair * (grad_a[0] * hess_b[2] + grad_a[1] * hess_b[4] + grad_a[2] * hess_b[5]);

                    d_pa[0] -= fax; d_pa[1] -= fay; d_pa[2] -= faz;
                    d_pb[0] -= fbx; d_pb[1] -= fby; d_pb[2] -= fbz;

                    const double dfa[3] = {f0, f1, f2};
                    const double dfb[3] = {
                        f0 - static_cast<double>(t0),
                        f1 - static_cast<double>(t1),
                        f2 - static_cast<double>(t2),
                    };
                    for (int j = 0; j < 3; ++j) {
                        d_cell_loc[0 * 3 + j] += fax * dfa[j] + fbx * dfb[j];
                        d_cell_loc[1 * 3 + j] += fay * dfa[j] + fby * dfb[j];
                        d_cell_loc[2 * 3 + j] += faz * dfa[j] + fbz * dfb[j];
                    }
                } else {
                    double prod = 0.0;
                    if constexpr (kIsTau) {
                        dev_real_solid_harmonics<true, false>(dax, day, daz, l_req_a, c_a, dc_a);
                        dev_real_solid_harmonics<true, false>(dbx, dby, dbz, l_req_b, c_b, dc_b);
                        double grad_a[3], grad_b[3];
                        dev_eval_orb_grad_hess<false>(dax, day, daz, ra, fa, y2a, n_pao_pts, rmax_a, rcut_a, lm_a, c_a, dc_a, nullptr, grad_a, nullptr);
                        dev_eval_orb_grad_hess<false>(dbx, dby, dbz, rb, fb, y2b, n_pao_pts, rmax_b, rcut_b, lm_b, c_b, dc_b, nullptr, grad_b, nullptr);
                        prod = 0.5 * (grad_a[0] * grad_b[0] + grad_a[1] * grad_b[1] + grad_a[2] * grad_b[2]);
                    } else {
                        dev_real_solid_harmonics<false, false>(dax, day, daz, l_req_a, c_a, nullptr);
                        dev_real_solid_harmonics<false, false>(dbx, dby, dbz, l_req_b, c_b, nullptr);
                        const double ga = dev_eval_spline(fa, y2a, n_pao_pts, rmax_a, rcut_a, ra, nullptr);
                        const double gb = dev_eval_spline(fb, y2b, n_pao_pts, rmax_b, rcut_b, rb, nullptr);
                        prod = (ga * c_a[lm_a]) * (gb * c_b[lm_b]);
                    }
                    if constexpr (kIsScatter) {
                        for (int32_t s = 0; s < n_spin_mat; ++s) {
                            atomicAdd(&out_main[static_cast<size_t>(s) * n_grid + p_idx], dm_local[s] * prod);
                        }
                    } else {
                        for (int32_t s = 0; s < n_spin_mat; ++s) {
                            edge_acc[s] += dvol * primary_buf[static_cast<size_t>(s) * n_grid + p_idx] * prod;
                        }
                    }
                }
            }
        }
    }

    if constexpr (kIsVmat || kIsBwd) {
        for (int32_t s = 0; s < n_spin_mat; ++s) {
            out_main[static_cast<size_t>(s) * n_edges + e] = edge_acc[s];
        }
    }
    if constexpr (kIsBwd) {
        atomicAdd(&out_d_pos[3 * atom_a + 0], d_pa[0]);
        atomicAdd(&out_d_pos[3 * atom_a + 1], d_pa[1]);
        atomicAdd(&out_d_pos[3 * atom_a + 2], d_pa[2]);
        atomicAdd(&out_d_pos[3 * atom_b + 0], d_pb[0]);
        atomicAdd(&out_d_pos[3 * atom_b + 1], d_pb[1]);
        atomicAdd(&out_d_pos[3 * atom_b + 2], d_pb[2]);
        for (int k = 0; k < 9; ++k) {
            atomicAdd(&out_d_cell[k], d_cell_loc[k]);
        }
    }
}

__global__ void na_grid_atom_kernel(
    const double* __restrict__ positions,
    const double* __restrict__ cell,
    const double* __restrict__ geom_dev,
    const int32_t* __restrict__ atom_species,
    const double* __restrict__ sp_rmax,
    const double* __restrict__ sp_rcut_na,
    const double* __restrict__ sp_rho_na,
    const double* __restrict__ sp_v_na,
    const double* __restrict__ sp_rho_core,
    const double* __restrict__ y2_dev,
    int32_t n_atoms,
    int32_t n_pts,
    int32_t n1,
    int32_t n2,
    int32_t n3,
    double* __restrict__ out_rho_na,
    double* __restrict__ out_v_na,
    double* __restrict__ out_rho_core) {
    const int32_t ia = blockIdx.x;
    if (ia >= n_atoms) return;

    const int32_t sp = atom_species[ia];
    const double r_max = sp_rmax[sp];
    const double r_cut = sp_rcut_na[sp];
    const double px = positions[3 * ia + 0];
    const double py = positions[3 * ia + 1];
    const double pz = positions[3 * ia + 2];

    const double fc0 = geom_dev[0] * px + geom_dev[1] * py + geom_dev[2] * pz;
    const double fc1 = geom_dev[3] * px + geom_dev[4] * py + geom_dev[5] * pz;
    const double fc2 = geom_dev[6] * px + geom_dev[7] * py + geom_dev[8] * pz;

    const int i0_min = static_cast<int>(ceil(n1 * (fc0 - r_max * geom_dev[9])));
    const int i0_max = static_cast<int>(floor(n1 * (fc0 + r_max * geom_dev[9])));
    const int i1_min = static_cast<int>(ceil(n2 * (fc1 - r_max * geom_dev[10])));
    const int i1_max = static_cast<int>(floor(n2 * (fc1 + r_max * geom_dev[10])));
    const int i2_min = static_cast<int>(ceil(n3 * (fc2 - r_max * geom_dev[11])));
    const int i2_max = static_cast<int>(floor(n3 * (fc2 + r_max * geom_dev[11])));

    const int d0 = max(0, i0_max - i0_min + 1);
    const int d1 = max(0, i1_max - i1_min + 1);
    const int d2 = max(0, i2_max - i2_min + 1);
    const int64_t total = static_cast<int64_t>(d0) * d1 * d2;

    const double* yr = sp_rho_na + static_cast<size_t>(sp) * n_pts;
    const double* yv = sp_v_na + static_cast<size_t>(sp) * n_pts;
    const double* yc = sp_rho_core + static_cast<size_t>(sp) * n_pts;
    const double* y2r = y2_dev + static_cast<size_t>(sp * 3 + 0) * n_pts;
    const double* y2v = y2_dev + static_cast<size_t>(sp * 3 + 1) * n_pts;
    const double* y2c = y2_dev + static_cast<size_t>(sp * 3 + 2) * n_pts;

    for (int64_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        const int i2 = i2_min + static_cast<int>(tid % d2);
        const int64_t t_rem = tid / d2;
        const int i1 = i1_min + static_cast<int>(t_rem % d1);
        const int i0 = i0_min + static_cast<int>(t_rem / d1);
        const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
        const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
        const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
        const double rx = cell[0] * f0 + cell[1] * f1 + cell[2] * f2;
        const double ry = cell[3] * f0 + cell[4] * f1 + cell[5] * f2;
        const double rz = cell[6] * f0 + cell[7] * f1 + cell[8] * f2;
        const double dx = rx - px, dy = ry - py, dz = rz - pz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= r_max * r_max) continue;
        const double r = sqrt(r2);
        const int g0 = dev_floor_mod(i0, n1);
        const int g1 = dev_floor_mod(i1, n2);
        const int g2 = dev_floor_mod(i2, n3);
        const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;
        if (r < r_cut) {
            atomicAdd(&out_rho_na[p_idx], dev_eval_spline(yr, y2r, n_pts, r_max, r_cut, r, nullptr));
            atomicAdd(&out_v_na[p_idx], dev_eval_spline(yv, y2v, n_pts, r_max, r_cut, r, nullptr));
        }
        atomicAdd(&out_rho_core[p_idx], dev_eval_spline(yc, y2c, n_pts, r_max, r_max, r, nullptr));
    }
}

__global__ void na_grid_grad_atom_kernel(
    const double* __restrict__ w_rho_na,
    const double* __restrict__ w_v_na,
    const double* __restrict__ w_rho_core,
    const double* __restrict__ positions,
    const double* __restrict__ cell,
    const double* __restrict__ geom_dev,
    const int32_t* __restrict__ atom_species,
    const double* __restrict__ sp_rmax,
    const double* __restrict__ sp_rcut_na,
    const double* __restrict__ sp_rho_na,
    const double* __restrict__ sp_v_na,
    const double* __restrict__ sp_rho_core,
    const double* __restrict__ y2_dev,
    int32_t n_atoms,
    int32_t n_pts,
    int32_t n1,
    int32_t n2,
    int32_t n3,
    double* __restrict__ out_d_pos,
    double* __restrict__ out_d_cell) {
    const int32_t ia = blockIdx.x;
    if (ia >= n_atoms) return;

    __shared__ double smem[12 * 256];

    const int32_t sp = atom_species[ia];
    const double r_max = sp_rmax[sp];
    const double r_cut = sp_rcut_na[sp];
    const double px = positions[3 * ia + 0];
    const double py = positions[3 * ia + 1];
    const double pz = positions[3 * ia + 2];

    const double fc0 = geom_dev[0] * px + geom_dev[1] * py + geom_dev[2] * pz;
    const double fc1 = geom_dev[3] * px + geom_dev[4] * py + geom_dev[5] * pz;
    const double fc2 = geom_dev[6] * px + geom_dev[7] * py + geom_dev[8] * pz;

    const int i0_min = static_cast<int>(ceil(n1 * (fc0 - r_max * geom_dev[9])));
    const int i0_max = static_cast<int>(floor(n1 * (fc0 + r_max * geom_dev[9])));
    const int i1_min = static_cast<int>(ceil(n2 * (fc1 - r_max * geom_dev[10])));
    const int i1_max = static_cast<int>(floor(n2 * (fc1 + r_max * geom_dev[10])));
    const int i2_min = static_cast<int>(ceil(n3 * (fc2 - r_max * geom_dev[11])));
    const int i2_max = static_cast<int>(floor(n3 * (fc2 + r_max * geom_dev[11])));

    const int d0 = max(0, i0_max - i0_min + 1);
    const int d1 = max(0, i1_max - i1_min + 1);
    const int d2 = max(0, i2_max - i2_min + 1);
    const int64_t total = static_cast<int64_t>(d0) * d1 * d2;

    const double* yr = sp_rho_na + static_cast<size_t>(sp) * n_pts;
    const double* yv = sp_v_na + static_cast<size_t>(sp) * n_pts;
    const double* yc = sp_rho_core + static_cast<size_t>(sp) * n_pts;
    const double* y2r = y2_dev + static_cast<size_t>(sp * 3 + 0) * n_pts;
    const double* y2v = y2_dev + static_cast<size_t>(sp * 3 + 1) * n_pts;
    const double* y2c = y2_dev + static_cast<size_t>(sp * 3 + 2) * n_pts;

    double loc[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        const int i2 = i2_min + static_cast<int>(tid % d2);
        const int64_t t_rem = tid / d2;
        const int i1 = i1_min + static_cast<int>(t_rem % d1);
        const int i0 = i0_min + static_cast<int>(t_rem / d1);
        const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
        const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
        const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
        const double rx = cell[0] * f0 + cell[1] * f1 + cell[2] * f2;
        const double ry = cell[3] * f0 + cell[4] * f1 + cell[5] * f2;
        const double rz = cell[6] * f0 + cell[7] * f1 + cell[8] * f2;
        const double dx = rx - px, dy = ry - py, dz = rz - pz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= r_max * r_max || r2 < 1e-24) continue;
        const double r = sqrt(r2);
        const int g0 = dev_floor_mod(i0, n1);
        const int g1 = dev_floor_mod(i1, n2);
        const int g2 = dev_floor_mod(i2, n3);
        const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;

        double drna = 0.0, dvna = 0.0, drc = 0.0;
        if (r < r_cut) {
            dev_eval_spline(yr, y2r, n_pts, r_max, r_cut, r, &drna);
            dev_eval_spline(yv, y2v, n_pts, r_max, r_cut, r, &dvna);
        }
        dev_eval_spline(yc, y2c, n_pts, r_max, r_max, r, &drc);

        const double coeff = (w_rho_na[p_idx] * drna + w_v_na[p_idx] * dvna + w_rho_core[p_idx] * drc) / r;
        const double gx = coeff * dx, gy = coeff * dy, gz = coeff * dz;
        loc[0] -= gx; loc[1] -= gy; loc[2] -= gz;
        const double df[3] = {f0, f1, f2};
        for (int j = 0; j < 3; ++j) {
            loc[3 + 0 * 3 + j] += gx * df[j];
            loc[3 + 1 * 3 + j] += gy * df[j];
            loc[3 + 2 * 3 + j] += gz * df[j];
        }
    }

    for (int k = 0; k < 12; ++k) {
        smem[k * blockDim.x + threadIdx.x] = loc[k];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int k = 0; k < 12; ++k) {
                smem[k * blockDim.x + threadIdx.x] += smem[k * blockDim.x + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_d_pos[3 * ia + 0] = smem[0 * blockDim.x];
        out_d_pos[3 * ia + 1] = smem[1 * blockDim.x];
        out_d_pos[3 * ia + 2] = smem[2 * blockDim.x];
        for (int k = 0; k < 9; ++k) {
            atomicAdd(&out_d_cell[k], smem[(3 + k) * blockDim.x]);
        }
    }
}

__global__ void prepare_geom_and_single_spline_kernel(
    const double* __restrict__ cell,
    const double* __restrict__ sp_rmax,
    const double* __restrict__ sp_vals,
    int32_t n_sp,
    int32_t n_pts,
    double* __restrict__ geom_dev,
    double* __restrict__ y2_dev,
    double* __restrict__ u_dev) {
    const int tid = blockIdx.x;
    if (tid == 0 && threadIdx.x == 0) {
        const double a00 = cell[0], a01 = cell[1], a02 = cell[2];
        const double a10 = cell[3], a11 = cell[4], a12 = cell[5];
        const double a20 = cell[6], a21 = cell[7], a22 = cell[8];
        const double c00 = a11 * a22 - a12 * a21;
        const double c01 = a12 * a20 - a10 * a22;
        const double c02 = a10 * a21 - a11 * a20;
        const double c10 = a02 * a21 - a01 * a22;
        const double c11 = a00 * a22 - a02 * a20;
        const double c12 = a01 * a20 - a00 * a21;
        const double c20 = a01 * a12 - a02 * a11;
        const double c21 = a02 * a10 - a00 * a12;
        const double c22 = a00 * a11 - a01 * a10;
        const double det = a00 * c00 + a01 * c01 + a02 * c02;
        const double inv_det = 1.0 / det;
        geom_dev[0] = c00 * inv_det; geom_dev[1] = c10 * inv_det; geom_dev[2] = c20 * inv_det;
        geom_dev[3] = c01 * inv_det; geom_dev[4] = c11 * inv_det; geom_dev[5] = c21 * inv_det;
        geom_dev[6] = c02 * inv_det; geom_dev[7] = c12 * inv_det; geom_dev[8] = c22 * inv_det;
        for (int ax = 0; ax < 3; ++ax) {
            const double bx = geom_dev[3 * ax + 0];
            const double by = geom_dev[3 * ax + 1];
            const double bz = geom_dev[3 * ax + 2];
            geom_dev[9 + ax] = sqrt(bx * bx + by * by + bz * bz);
        }
    }
    if (tid < n_sp && threadIdx.x == 0) {
        const double* src = sp_vals + static_cast<size_t>(tid) * n_pts;
        double* y2 = y2_dev + static_cast<size_t>(tid) * n_pts;
        double* u = u_dev + static_cast<size_t>(tid) * n_pts;
        for (int i = 0; i < n_pts; ++i) {
            y2[i] = 0.0;
            u[i] = 0.0;
        }
        const double r_max = sp_rmax[tid];
        if (n_pts >= 3 && r_max > 0.0) {
            const double dr = r_max / static_cast<double>(n_pts - 1);
            for (int i = 1; i < n_pts - 1; ++i) {
                const double p = 0.5 * y2[i - 1] + 2.0;
                y2[i] = -0.5 / p;
                const double rhs = (src[i + 1] - 2.0 * src[i] + src[i - 1]) / (dr * dr);
                u[i] = (3.0 * rhs - 0.5 * u[i - 1]) / p;
            }
            for (int k = n_pts - 2; k >= 0; --k) {
                y2[k] = y2[k] * y2[k + 1] + u[k];
            }
        }
    }
}

template <bool IsGrad>
__global__ void tau_core_atom_kernel(
    const double* __restrict__ w_tau_core,
    const double* __restrict__ positions,
    const double* __restrict__ cell,
    const double* __restrict__ geom_dev,
    const int32_t* __restrict__ atom_species,
    const double* __restrict__ sp_rmax,
    const double* __restrict__ sp_tau_core,
    const double* __restrict__ y2_dev,
    int32_t n_atoms,
    int32_t n_pts,
    int32_t n1,
    int32_t n2,
    int32_t n3,
    double* __restrict__ out_main,
    double* __restrict__ out_d_cell) {
    const int32_t ia = blockIdx.x;
    if (ia >= n_atoms) return;

    __shared__ double smem[12 * 256];

    const int32_t sp = atom_species[ia];
    const double r_max = sp_rmax[sp];
    const double px = positions[3 * ia + 0];
    const double py = positions[3 * ia + 1];
    const double pz = positions[3 * ia + 2];

    const double fc0 = geom_dev[0] * px + geom_dev[1] * py + geom_dev[2] * pz;
    const double fc1 = geom_dev[3] * px + geom_dev[4] * py + geom_dev[5] * pz;
    const double fc2 = geom_dev[6] * px + geom_dev[7] * py + geom_dev[8] * pz;

    const int i0_min = static_cast<int>(ceil(n1 * (fc0 - r_max * geom_dev[9])));
    const int i0_max = static_cast<int>(floor(n1 * (fc0 + r_max * geom_dev[9])));
    const int i1_min = static_cast<int>(ceil(n2 * (fc1 - r_max * geom_dev[10])));
    const int i1_max = static_cast<int>(floor(n2 * (fc1 + r_max * geom_dev[10])));
    const int i2_min = static_cast<int>(ceil(n3 * (fc2 - r_max * geom_dev[11])));
    const int i2_max = static_cast<int>(floor(n3 * (fc2 + r_max * geom_dev[11])));

    const int d0 = max(0, i0_max - i0_min + 1);
    const int d1 = max(0, i1_max - i1_min + 1);
    const int d2 = max(0, i2_max - i2_min + 1);
    const int64_t total = static_cast<int64_t>(d0) * d1 * d2;

    const double* ytc = sp_tau_core + static_cast<size_t>(sp) * n_pts;
    const double* y2tc = y2_dev + static_cast<size_t>(sp) * n_pts;

    double loc[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        const int i2 = i2_min + static_cast<int>(tid % d2);
        const int64_t t_rem = tid / d2;
        const int i1 = i1_min + static_cast<int>(t_rem % d1);
        const int i0 = i0_min + static_cast<int>(t_rem / d1);
        const double f0 = static_cast<double>(i0) / static_cast<double>(n1);
        const double f1 = static_cast<double>(i1) / static_cast<double>(n2);
        const double f2 = static_cast<double>(i2) / static_cast<double>(n3);
        const double rx = cell[0] * f0 + cell[1] * f1 + cell[2] * f2;
        const double ry = cell[3] * f0 + cell[4] * f1 + cell[5] * f2;
        const double rz = cell[6] * f0 + cell[7] * f1 + cell[8] * f2;
        const double dx = rx - px, dy = ry - py, dz = rz - pz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= r_max * r_max) continue;
        const int g0 = dev_floor_mod(i0, n1);
        const int g1 = dev_floor_mod(i1, n2);
        const int g2 = dev_floor_mod(i2, n3);
        const int64_t p_idx = (static_cast<int64_t>(g0) * n2 + g1) * n3 + g2;
        if constexpr (!IsGrad) {
            const double r = sqrt(r2);
            atomicAdd(&out_main[p_idx], dev_eval_spline(ytc, y2tc, n_pts, r_max, r_max, r, nullptr));
        } else {
            if (r2 < 1e-24) continue;
            const double r = sqrt(r2);
            double dtc = 0.0;
            dev_eval_spline(ytc, y2tc, n_pts, r_max, r_max, r, &dtc);
            const double coeff = (w_tau_core[p_idx] * dtc) / r;
            const double gx = coeff * dx, gy = coeff * dy, gz = coeff * dz;
            loc[0] -= gx; loc[1] -= gy; loc[2] -= gz;
            const double df[3] = {f0, f1, f2};
            for (int j = 0; j < 3; ++j) {
                loc[3 + 0 * 3 + j] += gx * df[j];
                loc[3 + 1 * 3 + j] += gy * df[j];
                loc[3 + 2 * 3 + j] += gz * df[j];
            }
        }
    }

    if constexpr (IsGrad) {
        for (int k = 0; k < 12; ++k) {
            smem[k * blockDim.x + threadIdx.x] = loc[k];
        }
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                for (int k = 0; k < 12; ++k) {
                    smem[k * blockDim.x + threadIdx.x] += smem[k * blockDim.x + threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_main[3 * ia + 0] = smem[0 * blockDim.x];
            out_main[3 * ia + 1] = smem[1 * blockDim.x];
            out_main[3 * ia + 2] = smem[2 * blockDim.x];
            for (int k = 0; k < 9; ++k) {
                atomicAdd(&out_d_cell[k], smem[(3 + k) * blockDim.x]);
            }
        }
    }
}

// ---------- GPU LDA/GGA XC kernel (built-in functionals of jsiesta/xc_semilocal.hpp, matched to Libxc) ----------

__global__ void xc_eval_cuda_kernel(
    const double* __restrict__ r_ptr,
    const double* __restrict__ s_ptr,
    int64_t n_grid,
    int32_t xc_id,
    int32_t n_spin_pol,
    double* __restrict__ e_ptr,
    double* __restrict__ vr_ptr,
    double* __restrict__ vs_ptr) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t p = tid; p < n_grid; p += stride) {
        double vr[2], vs[3];
        jsiesta::xc::eval_semilocal_xc_point(xc_id, n_spin_pol, r_ptr[p], r_ptr[n_grid + p], s_ptr[p], s_ptr[n_grid + p],
                                             s_ptr[2 * n_grid + p], &e_ptr[p], vr, vs);
        vr_ptr[p] = vr[0];
        vr_ptr[n_grid + p] = vr[1];
        vs_ptr[p] = vs[0];
        vs_ptr[n_grid + p] = vs[1];
        vs_ptr[2 * n_grid + p] = vs[2];
    }
}


// ---------------------------------------------------------------------------------------------------------------
// Cached-plan path for rhoofd / vmat (D-036), tauofd / vmat_tau (D-037) and their backward modes dfscf_grad /
// tau_dfscf_grad (D-038). The host plan of grid_ffi.cpp (orbital values, gradients and, for the meta-GGA backward,
// Hessians per atom image and grid point; dense atom-pair DM blocks) is built once per geometry, uploaded once, and
// contracted with one warp per grid point. The edge-parallel kernel above stays as the fallback (oversized plans, device
// memory short, or JSIESTA_CUDA_GRID_LEGACY=1).
// ---------------------------------------------------------------------------------------------------------------

struct DeviceGridPlan {
    std::uint64_t hash = 0;
    bool has_grad = false;
    bool has_hess = false;
    std::int64_t n_grid = 0;
    std::int32_t n_atoms = 0, n_shifts = 0;
    std::int32_t s_min[3] = {0, 0, 0}, s_dim[3] = {1, 1, 1};
    std::size_t blk_stride = 0, n_block_edges = 0;
    double dvol = 0.0;
    std::int32_t* img_off = nullptr;
    std::int32_t* img = nullptr;
    double* vals = nullptr;
    double* gx = nullptr;  // orbital gradients (has_grad only)
    double* gy = nullptr;
    double* gz = nullptr;
    double* hxx = nullptr;  // orbital Hessians (has_hess only)
    double* hxy = nullptr;
    double* hxz = nullptr;
    double* hyy = nullptr;
    double* hyz = nullptr;
    double* hzz = nullptr;
    std::int32_t* atom_no = nullptr;
    std::int32_t* shift_box = nullptr;
    std::int32_t* pair_bid = nullptr;
    std::int32_t* blk_off = nullptr;
    std::int32_t* be_abs = nullptr;
    std::int32_t* be_idx = nullptr;
    // streamed mode (D-043): only the index arrays above are on the device; images and value arrays stay in the host
    // plan (kept alive by host_owner) and are copied chunk by chunk of grid points in each call
    bool streamed = false;
    std::shared_ptr<const void> host_owner;
    const std::int32_t* h_img = nullptr;
    const double* h_arr[10] = {nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr};  // vals, g{x,y,z}, h{xx,xy,xz,yy,yz,zz}
    struct Chunk {
        std::int64_t p0, p1, img0, img1, val0, val1;
    };
    std::vector<Chunk> chunks;
    std::size_t max_chunk_imgs = 0, max_chunk_vals = 0;
    // D-044: host ranges this plan page-locked (cudaHostRegister), so its copies are asynchronous DMA; double buffering
    // with a copy stream overlaps the copy of one chunk with the kernels of the previous one
    std::vector<void*> pinned;
    bool all_pinned = false;
    bool overlap = false;
    std::vector<void*> allocs;
    ~DeviceGridPlan() {
        for (void* ptr : pinned) cudaHostUnregister(ptr);  // before host_owner (a member) releases the host plan
        for (void* ptr : allocs) cudaFree(ptr);
        cudaGetLastError();
    }
};

template <typename T>
bool upload_array(DeviceGridPlan& dp, const T* host, std::size_t n, T** out) {
    *out = nullptr;
    const std::size_t bytes = std::max<std::size_t>(1, n) * sizeof(T);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    dp.allocs.push_back(ptr);
    if (n > 0 && cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    *out = static_cast<T*>(ptr);
    return true;
}

// JSIESTA_CUDA_PLAN_STREAM: unset = automatic (resident if it fits, else streamed), 1 = always streamed,
// 0 = never streamed (resident or the edge kernel).
int plan_stream_mode() {
    const char* env = std::getenv("JSIESTA_CUDA_PLAN_STREAM");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
}

std::size_t env_size(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return 0;
    return static_cast<std::size_t>(std::strtoull(env, nullptr, 10));
}

// JSIESTA_CUDA_PLAN_PIN=0 / JSIESTA_CUDA_PLAN_OVERLAP=0 switch off page-locking and copy/kernel overlap (A/B runs).
bool env_flag_on(const char* name) {
    const char* env = std::getenv(name);
    return env == nullptr || env[0] == '\0' || env[0] != '0';
}

// One non-blocking stream per process for the chunk copies of streamed plans (nullptr if it cannot be created).
cudaStream_t plan_copy_stream() {
    static std::once_flag once;
    static cudaStream_t s = nullptr;
    std::call_once(once, [] {
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            s = nullptr;
        }
    });
    return s;
}

bool plan_path_disabled() {
    const char* env = std::getenv("JSIESTA_CUDA_GRID_LEGACY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

// Host copies of the inputs the plan depends on (small compared with the grid).
struct PlanInputsHost {
    std::vector<double> positions, cell, pao_r_max, pao_rcut, pao_values, orb_rcut;
    std::vector<std::int32_t> pao_l, orb_atom, orb_g_shell, orb_m, row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts;
};

template <typename T, ffi::DataType DT>
cudaError_t copy_buffer_to_host(cudaStream_t stream, const ffi::Buffer<DT>& buf, std::vector<T>& out) {
    out.resize(buf.element_count());
    if (out.empty()) return cudaSuccess;
    return cudaMemcpyAsync(out.data(), buf.typed_data(), out.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

// Returns nullptr (and the caller falls back to the edge kernel) when the plan has no atom-block form or does not fit.
std::shared_ptr<const DeviceGridPlan> get_device_grid_plan(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F64> positions, ffi::Buffer<ffi::DataType::F64> cell,
    ffi::Buffer<ffi::DataType::F64> pao_r_max, ffi::Buffer<ffi::DataType::F64> pao_rcut,
    ffi::Buffer<ffi::DataType::S32> pao_l, ffi::Buffer<ffi::DataType::F64> pao_values,
    ffi::Buffer<ffi::DataType::S32> orb_atom, ffi::Buffer<ffi::DataType::S32> orb_g_shell,
    ffi::Buffer<ffi::DataType::S32> orb_m, ffi::Buffer<ffi::DataType::F64> orb_rcut,
    ffi::Buffer<ffi::DataType::S32> row_ptr, ffi::Buffer<ffi::DataType::S32> dst_unit_orb,
    ffi::Buffer<ffi::DataType::S32> edge_cell_idx, ffi::Buffer<ffi::DataType::S32> cell_shifts,
    std::int32_t n_edges, std::int64_t n1, std::int64_t n2, std::int64_t n3, bool need_grad, bool need_hess = false) {
    if (plan_path_disabled()) return nullptr;
    if (need_hess) need_grad = true;
    PlanInputsHost h;
    cudaError_t err = cudaSuccess;
    err = err == cudaSuccess ? copy_buffer_to_host<double>(stream, positions, h.positions) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<double>(stream, cell, h.cell) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<double>(stream, pao_r_max, h.pao_r_max) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<double>(stream, pao_rcut, h.pao_rcut) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, pao_l, h.pao_l) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<double>(stream, pao_values, h.pao_values) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, orb_atom, h.orb_atom) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, orb_g_shell, h.orb_g_shell) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, orb_m, h.orb_m) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<double>(stream, orb_rcut, h.orb_rcut) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, row_ptr, h.row_ptr) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, dst_unit_orb, h.dst_unit_orb) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, edge_cell_idx, h.edge_cell_idx) : err;
    err = err == cudaSuccess ? copy_buffer_to_host<std::int32_t>(stream, cell_shifts, h.cell_shifts) : err;
    if (err != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    const std::int32_t n_atoms = static_cast<std::int32_t>(positions.dimensions()[0]);
    const std::int32_t n_orb = static_cast<std::int32_t>(orb_atom.dimensions()[0]);
    const std::int32_t n_shells = static_cast<std::int32_t>(pao_l.dimensions()[0]);
    const std::int32_t n_pao_pts = static_cast<std::int32_t>(pao_values.dimensions()[1]);
    const std::int32_t n_shifts = static_cast<std::int32_t>(cell_shifts.dimensions()[0]);
    static std::mutex mu;
    static std::vector<std::shared_ptr<const DeviceGridPlan>> lru;
    const int stream_mode = plan_stream_mode();
    {
        // a cached plan that stores at least what this call needs (the cache key includes the gradient and Hessian
        // flags) is used without building the host plan again
        const bool combos[3][2] = {{need_grad, need_hess}, {true, need_hess}, {true, true}};
        const std::uint64_t input_hash = jsiesta::grid::grid_plan_input_hash(
            h.positions.data(), h.cell.data(), h.pao_r_max.data(), h.pao_rcut.data(), h.pao_l.data(),
            h.pao_values.data(), h.orb_atom.data(), h.orb_g_shell.data(), h.orb_m.data(), h.orb_rcut.data(),
            h.row_ptr.data(), h.dst_unit_orb.data(), h.edge_cell_idx.data(), h.cell_shifts.data(),
            n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3);
        std::lock_guard<std::mutex> lock(mu);
        for (const auto& c : combos) {
            const std::uint64_t key = jsiesta::grid::grid_plan_cache_key(input_hash, c[0], c[1]);
            for (std::size_t i = 0; i < lru.size(); ++i) {
                if (lru[i]->hash == key && (stream_mode < 0 || lru[i]->streamed == (stream_mode == 1))) {
                    auto found = lru[i];
                    lru.erase(lru.begin() + static_cast<std::ptrdiff_t>(i));
                    lru.insert(lru.begin(), found);
                    return found;
                }
            }
        }
    }
    const jsiesta::grid::FlatGridPlan flat = jsiesta::grid::get_flat_grid_plan(
        h.positions.data(), h.cell.data(), h.pao_r_max.data(), h.pao_rcut.data(), h.pao_l.data(), h.pao_values.data(),
        h.orb_atom.data(), h.orb_g_shell.data(), h.orb_m.data(), h.orb_rcut.data(), h.row_ptr.data(),
        h.dst_unit_orb.data(), h.edge_cell_idx.data(), h.cell_shifts.data(),
        n_atoms, n_orb, n_shells, n_pao_pts, n_edges, n_shifts, n1, n2, n3, need_grad, need_hess);
    if (!flat.use_atom_blocks) return nullptr;

    std::lock_guard<std::mutex> lock(mu);
    for (std::size_t i = 0; i < lru.size(); ++i) {
        if (lru[i]->hash == flat.hash && lru[i]->n_grid == flat.n_grid && lru[i]->has_grad == need_grad &&
            lru[i]->has_hess == need_hess && (stream_mode < 0 || lru[i]->streamed == (stream_mode == 1))) {
            auto found = lru[i];
            lru.erase(lru.begin() + static_cast<std::ptrdiff_t>(i));
            lru.insert(lru.begin(), found);
            return found;
        }
    }
    // free the oldest upload before allocating a new one; two entries hold the value plan and the gradient plan of a
    // meta-GGA SCF step at the same time
    if (lru.size() >= 2) lru.pop_back();

    const std::size_t n_val_arrays = 1 + (need_grad ? 3 : 0) + (need_hess ? 6 : 0);
    const std::size_t index_bytes =
        (static_cast<std::size_t>(flat.n_grid) + 1 + flat.shift_box_len + flat.pair_block_len +
         flat.block_offset.size() + 2 * flat.block_edge_idx.size() + static_cast<std::size_t>(flat.n_atoms)) *
        sizeof(std::int32_t);
    const std::size_t bulk_bytes = n_val_arrays * flat.n_vals * sizeof(double) + 5 * flat.n_img * sizeof(std::int32_t);
    const std::size_t plan_bytes = index_bytes + bulk_bytes;
    // resident: everything on the device; otherwise only the index arrays (streamed mode)
    auto upload_plan = [&](bool resident) -> std::shared_ptr<DeviceGridPlan> {
        auto dp = std::make_shared<DeviceGridPlan>();
        dp->hash = flat.hash;
        dp->has_grad = need_grad;
        dp->has_hess = need_hess;
        dp->n_grid = flat.n_grid;
        dp->n_atoms = flat.n_atoms;
        dp->n_shifts = flat.n_shifts;
        for (int ax = 0; ax < 3; ++ax) {
            dp->s_min[ax] = flat.s_min[ax];
            dp->s_dim[ax] = flat.s_dim[ax];
        }
        dp->blk_stride = flat.total_blk_scalars;
        dp->n_block_edges = flat.block_edge_idx.size();
        dp->dvol = flat.dvol;
        const bool ok_index =
            upload_array(*dp, flat.grid_img_offsets, static_cast<std::size_t>(flat.n_grid) + 1, &dp->img_off) &&
            upload_array(*dp, flat.atom_n_orbs, static_cast<std::size_t>(flat.n_atoms), &dp->atom_no) &&
            upload_array(*dp, flat.shift_box, flat.shift_box_len, &dp->shift_box) &&
            upload_array(*dp, flat.pair_block_id, flat.pair_block_len, &dp->pair_bid) &&
            upload_array(*dp, flat.block_offset.data(), flat.block_offset.size(), &dp->blk_off) &&
            upload_array(*dp, flat.block_edge_abs.data(), flat.block_edge_abs.size(), &dp->be_abs) &&
            upload_array(*dp, flat.block_edge_idx.data(), flat.block_edge_idx.size(), &dp->be_idx);
        if (!ok_index) return nullptr;
        if (!resident) return dp;
        const bool ok =
            upload_array(*dp, flat.img, 5 * flat.n_img, &dp->img) &&
            upload_array(*dp, flat.img_vals, flat.n_vals, &dp->vals) &&
            (!need_grad || (upload_array(*dp, flat.img_gx, flat.n_vals, &dp->gx) &&
                            upload_array(*dp, flat.img_gy, flat.n_vals, &dp->gy) &&
                            upload_array(*dp, flat.img_gz, flat.n_vals, &dp->gz))) &&
            (!need_hess || (upload_array(*dp, flat.img_hxx, flat.n_vals, &dp->hxx) &&
                            upload_array(*dp, flat.img_hxy, flat.n_vals, &dp->hxy) &&
                            upload_array(*dp, flat.img_hxz, flat.n_vals, &dp->hxz) &&
                            upload_array(*dp, flat.img_hyy, flat.n_vals, &dp->hyy) &&
                            upload_array(*dp, flat.img_hyz, flat.n_vals, &dp->hyz) &&
                            upload_array(*dp, flat.img_hzz, flat.n_vals, &dp->hzz)));
        return ok ? dp : nullptr;
    };
    // splits the grid into chunks of consecutive points whose images and values fit the chunk budget; images and
    // values are stored in grid-point order, so a chunk is one contiguous slice of every array
    auto make_streamed = [&]() -> std::shared_ptr<DeviceGridPlan> {
        auto dp = upload_plan(false);
        if (!dp) return nullptr;
        dp->streamed = true;
        dp->host_owner = flat.owner;
        dp->h_img = flat.img;
        const double* h[10] = {flat.img_vals, flat.img_gx, flat.img_gy, flat.img_gz, flat.img_hxx,
                               flat.img_hxy,  flat.img_hxz, flat.img_hyy, flat.img_hyz, flat.img_hzz};
        for (int k = 0; k < 10; ++k) dp->h_arr[k] = h[k];
        if (!need_grad) for (int k = 1; k < 10; ++k) dp->h_arr[k] = nullptr;
        if (!need_hess) for (int k = 4; k < 10; ++k) dp->h_arr[k] = nullptr;
        if (env_flag_on("JSIESTA_CUDA_PLAN_PIN")) {
            // page-lock the host arrays in place; a range another plan already registered is fine (still pinned)
            auto pin = [&](const void* ptr, std::size_t bytes) -> bool {
                if (ptr == nullptr || bytes == 0) return true;
                const cudaError_t e = cudaHostRegister(const_cast<void*>(ptr), bytes, cudaHostRegisterDefault);
                if (e == cudaSuccess) {
                    dp->pinned.push_back(const_cast<void*>(ptr));
                    return true;
                }
                cudaGetLastError();
                return e == cudaErrorHostMemoryAlreadyRegistered;
            };
            bool ok = pin(flat.img, 5 * flat.n_img * sizeof(std::int32_t));
            for (int k = 0; k < 10; ++k) ok = pin(dp->h_arr[k], flat.n_vals * sizeof(double)) && ok;
            dp->all_pinned = ok;
        }
        dp->overlap = env_flag_on("JSIESTA_CUDA_PLAN_OVERLAP") && plan_copy_stream() != nullptr;
        std::size_t budget = env_size("JSIESTA_CUDA_PLAN_CHUNK_MB") << 20;
        if (budget == 0) {
            std::size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            cudaGetLastError();
            budget = free_b / 2;
        }
        if (dp->overlap) budget /= 2;  // two buffer sets
        budget = std::max<std::size_t>(budget, std::size_t(1) << 20);
        const double imgs_per_val = flat.n_vals > 0 ? static_cast<double>(flat.n_img) / flat.n_vals : 0.0;
        const double bytes_per_val = 8.0 * n_val_arrays + 20.0 * imgs_per_val;
        std::size_t vcap = static_cast<std::size_t>(static_cast<double>(budget) / bytes_per_val);
        if (const std::size_t forced = env_size("JSIESTA_CUDA_PLAN_CHUNK_VALS")) vcap = forced;
        vcap = std::max<std::size_t>(vcap, 1);
        const std::int32_t* off = flat.grid_img_offsets;
        auto val_at = [&](std::int64_t pt) -> std::int64_t {  // value offset of the first image at or after point pt
            const std::int64_t im = off[pt];
            return im < static_cast<std::int64_t>(flat.n_img) ? flat.img[5 * im + 4]
                                                               : static_cast<std::int64_t>(flat.n_vals);
        };
        std::int64_t p0 = 0;
        while (p0 < flat.n_grid) {
            const std::int64_t v0 = val_at(p0);
            std::int64_t p1 = p0 + 1;
            while (p1 < flat.n_grid && static_cast<std::size_t>(val_at(p1 + 1) - v0) <= vcap) ++p1;
            DeviceGridPlan::Chunk c{p0, p1, off[p0], off[p1], v0, val_at(p1)};
            dp->max_chunk_imgs = std::max<std::size_t>(dp->max_chunk_imgs, static_cast<std::size_t>(c.img1 - c.img0));
            dp->max_chunk_vals = std::max<std::size_t>(dp->max_chunk_vals, static_cast<std::size_t>(c.val1 - c.val0));
            dp->chunks.push_back(c);
            p0 = p1;
        }
        return dp;
    };
    std::shared_ptr<DeviceGridPlan> dp;
    if (stream_mode != 1) {
        dp = upload_plan(true);
        if (!dp && !lru.empty()) {
            // device memory short: drop the cached plans (a value plan of the same geometry is a subset of this one)
            // and retry once
            lru.clear();
            dp = upload_plan(true);
        }
    }
    if (!dp && stream_mode != 0) {
        lru.clear();
        dp = make_streamed();
        if (dp) {
            static bool told = false;
            if (!told || std::getenv("JSIESTA_CUDA_PLAN_VERBOSE") != nullptr) {
                told = true;
                const std::size_t chunk_bytes =
                    dp->max_chunk_vals * n_val_arrays * sizeof(double) + 5 * dp->max_chunk_imgs * sizeof(std::int32_t);
                std::fprintf(stderr,
                             "jsiesta: grid plan (%.2f GB, %zu orbital values%s) streamed from the host in %zu chunks "
                             "of up to %.2f GB per call (host memory %s, %s)\n",
                             plan_bytes / 1e9, flat.n_vals,
                             need_hess ? " with gradients and Hessians" : (need_grad ? " with gradients" : ""),
                             dp->chunks.size(), chunk_bytes / 1e9, dp->all_pinned ? "pinned" : "pageable",
                             dp->overlap ? "copies overlapped with kernels" : "no overlap");
            }
        }
    }
    if (!dp) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            cudaGetLastError();
            std::fprintf(stderr,
                         "jsiesta: grid plan (%.2f GB, %zu orbital values%s) does not fit in device memory (%.2f of "
                         "%.2f GB free); using the edge kernel, which is much slower. Lower "
                         "XLA_PYTHON_CLIENT_MEM_FRACTION to leave room.\n",
                         plan_bytes / 1e9, flat.n_vals, need_hess ? " with gradients and Hessians"
                                                                  : (need_grad ? " with gradients" : ""),
                         free_b / 1e9, total_b / 1e9);
        }
        return nullptr;
    }
    if (!dp->streamed && std::getenv("JSIESTA_CUDA_PLAN_VERBOSE") != nullptr) {
        std::fprintf(stderr, "jsiesta: grid plan uploaded: %.2f GB, %zu orbital values, grad=%d hess=%d\n",
                     plan_bytes / 1e9, flat.n_vals, need_grad ? 1 : 0, need_hess ? 1 : 0);
    }
    lru.insert(lru.begin(), dp);
    return dp;
}

__global__ void edges_to_blocks_kernel(
    const double* __restrict__ edges, std::int64_t n_edges, const std::int32_t* __restrict__ be_abs,
    const std::int32_t* __restrict__ be_idx, std::int64_t n_be, std::int32_t n_spin_mat, std::int64_t stride,
    double* __restrict__ blk) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n_be) return;
    for (std::int32_t s = 0; s < n_spin_mat; ++s) {
        blk[s * stride + be_abs[k]] = edges[s * n_edges + be_idx[k]];
    }
}

__global__ void blocks_to_edges_kernel(
    const double* __restrict__ blk, std::int64_t stride, const std::int32_t* __restrict__ be_abs,
    const std::int32_t* __restrict__ be_idx, std::int64_t n_be, std::int32_t n_spin_mat, std::int64_t n_edges,
    double* __restrict__ edges) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n_be) return;
    for (std::int32_t s = 0; s < n_spin_mat; ++s) {
        edges[s * n_edges + be_idx[k]] = blk[s * stride + be_abs[k]];
    }
}

__device__ __forceinline__ std::int32_t dev_plan_block(
    const std::int32_t* __restrict__ ga, const std::int32_t* __restrict__ gb, const std::int32_t* __restrict__ shift_box,
    int sm0, int sm1, int sm2, int sd0, int sd1, int sd2, const std::int32_t* __restrict__ pair_bid,
    std::int32_t n_atoms, std::int32_t n_shifts) {
    const int i0 = gb[1] - ga[1] - sm0;
    const int i1 = gb[2] - ga[2] - sm1;
    const int i2 = gb[3] - ga[3] - sm2;
    if (static_cast<unsigned>(i0) >= static_cast<unsigned>(sd0) || static_cast<unsigned>(i1) >= static_cast<unsigned>(sd1) ||
        static_cast<unsigned>(i2) >= static_cast<unsigned>(sd2)) {
        return -1;
    }
    const std::int32_t c = shift_box[(static_cast<std::int64_t>(i0) * sd1 + i1) * sd2 + i2];
    if (c < 0) return -1;
    return pair_bid[(static_cast<std::int64_t>(ga[0]) * n_atoms + gb[0]) * n_shifts + c];
}

// Product of the orbital factors of entries a (image ga) and b (image gb) at one grid point: phi_a phi_b (NC = 1) or
// grad phi_a . grad phi_b (NC = 3, kinetic-energy density).
template <int NC>
__device__ __forceinline__ double dev_plan_prod(
    const double* __restrict__ c0, const double* __restrict__ c1, const double* __restrict__ c2,
    std::int64_t oa, std::int64_t ob) {
    if constexpr (NC == 1) {
        return c0[oa] * c0[ob];
    } else {
        return c0[oa] * c0[ob] + c1[oa] * c1[ob] + c2[oa] * c2[ob];
    }
}

// rho_s(p) = sum_{images a, b at p} phi_a^T D_s,ab phi_b (NC = 1), or
// tau_s(p) = scale * sum_{a, b} sum_c d_c phi_a^T D_s,ab d_c phi_b (NC = 3, scale 1/2);
// one warp per grid point, lanes over block entries.
template <int NS, int NC>
__global__ void plan_rhoofd_kernel(
    const std::int32_t* __restrict__ img_off, const std::int32_t* __restrict__ img, const double* __restrict__ c0,
    const double* __restrict__ c1, const double* __restrict__ c2,
    const std::int32_t* __restrict__ atom_no, const std::int32_t* __restrict__ shift_box,
    int sm0, int sm1, int sm2, int sd0, int sd1, int sd2, const std::int32_t* __restrict__ pair_bid,
    std::int32_t n_atoms, std::int32_t n_shifts, const std::int32_t* __restrict__ blk_off,
    const double* __restrict__ dm_blk, double scale, std::int64_t stride, std::int64_t n_grid, std::int64_t p0,
    std::int64_t p1, std::int64_t img_base, std::int64_t val_base, double* __restrict__ out) {
    const std::int64_t p = p0 + ((static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (p >= p1) return;  // uniform per warp
    double acc[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) acc[s] = 0.0;
    const std::int32_t beg = img_off[p], end = img_off[p + 1];
    for (std::int32_t ia = beg; ia < end; ++ia) {
        const std::int32_t* ga = img + 5 * (static_cast<std::int64_t>(ia) - img_base);
        const int noa = atom_no[ga[0]];
        for (std::int32_t ib = beg; ib < end; ++ib) {
            const std::int32_t* gb = img + 5 * (static_cast<std::int64_t>(ib) - img_base);
            const std::int32_t bid = dev_plan_block(ga, gb, shift_box, sm0, sm1, sm2, sd0, sd1, sd2, pair_bid, n_atoms, n_shifts);
            if (bid < 0) continue;
            const int nob = atom_no[gb[0]];
            const double* d0 = dm_blk + blk_off[bid];
            for (int idx = lane; idx < noa * nob; idx += 32) {
                const int a = idx / nob;
                const double prod = dev_plan_prod<NC>(c0, c1, c2, ga[4] - val_base + a, gb[4] - val_base + (idx - a * nob));
#pragma unroll
                for (int s = 0; s < NS; ++s) acc[s] += d0[s * stride + idx] * prod;
            }
        }
    }
#pragma unroll
    for (int s = 0; s < NS; ++s) {
        double v = acc[s];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) out[s * n_grid + p] = scale * v;
    }
}

// V_s,ab += dvol_eff V_s(p) phi_a phi_b^T (NC = 1), or dvol_eff V_tau,s(p) sum_c d_c phi_a d_c phi_b^T (NC = 3,
// dvol_eff = dvol / 2); one warp per grid point, atomics on consecutive block entries.
template <int NS, int NC>
__global__ void plan_vmat_kernel(
    const std::int32_t* __restrict__ img_off, const std::int32_t* __restrict__ img, const double* __restrict__ c0,
    const double* __restrict__ c1, const double* __restrict__ c2,
    const std::int32_t* __restrict__ atom_no, const std::int32_t* __restrict__ shift_box,
    int sm0, int sm1, int sm2, int sd0, int sd1, int sd2, const std::int32_t* __restrict__ pair_bid,
    std::int32_t n_atoms, std::int32_t n_shifts, const std::int32_t* __restrict__ blk_off,
    const double* __restrict__ v_grid, double dvol_eff, std::int64_t stride, std::int64_t n_grid, std::int64_t p0,
    std::int64_t p1, std::int64_t img_base, std::int64_t val_base, double* __restrict__ v_blk) {
    const std::int64_t p = p0 + ((static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (p >= p1) return;
    double w[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) w[s] = dvol_eff * v_grid[s * n_grid + p];
    const std::int32_t beg = img_off[p], end = img_off[p + 1];
    for (std::int32_t ia = beg; ia < end; ++ia) {
        const std::int32_t* ga = img + 5 * (static_cast<std::int64_t>(ia) - img_base);
        const int noa = atom_no[ga[0]];
        for (std::int32_t ib = beg; ib < end; ++ib) {
            const std::int32_t* gb = img + 5 * (static_cast<std::int64_t>(ib) - img_base);
            const std::int32_t bid = dev_plan_block(ga, gb, shift_box, sm0, sm1, sm2, sd0, sd1, sd2, pair_bid, n_atoms, n_shifts);
            if (bid < 0) continue;
            const int nob = atom_no[gb[0]];
            double* v0 = v_blk + blk_off[bid];
            for (int idx = lane; idx < noa * nob; idx += 32) {
                const int a = idx / nob;
                const double prod = dev_plan_prod<NC>(c0, c1, c2, ga[4] - val_base + a, gb[4] - val_base + (idx - a * nob));
#pragma unroll
                for (int s = 0; s < NS; ++s) atomicAdd(&v0[s * stride + idx], w[s] * prod);
            }
        }
    }
}

// Device pointers and grid-point range of one plan chunk: the whole plan when it is resident, or one streamed chunk.
// Image i of the chunk is img[5 (i - img_base)], value k of the chunk is vals[k - val_base].
struct PlanView {
    const std::int32_t* img = nullptr;
    const double* vals = nullptr;
    const double* gx = nullptr;
    const double* gy = nullptr;
    const double* gz = nullptr;
    const double* hxx = nullptr;
    const double* hxy = nullptr;
    const double* hxz = nullptr;
    const double* hyy = nullptr;
    const double* hyz = nullptr;
    const double* hzz = nullptr;
    std::int64_t p0 = 0, p1 = 0, img_base = 0, val_base = 0;
};

inline unsigned plan_launch_blocks(const PlanView& v, int blk) {
    const std::int64_t n_threads = (v.p1 - v.p0) * 32;
    return static_cast<unsigned>(std::max<std::int64_t>(1, (n_threads + blk - 1) / blk));
}

template <int NS, int NC>
void launch_plan_rhoofd(cudaStream_t stream, const DeviceGridPlan& dp, const PlanView& v, const double* dm_blk,
                        double* out) {
    const int blk = 256;
    const double* c0 = NC == 1 ? v.vals : v.gx;
    plan_rhoofd_kernel<NS, NC><<<plan_launch_blocks(v, blk), blk, 0, stream>>>(
        dp.img_off, v.img, c0, v.gy, v.gz, dp.atom_no, dp.shift_box, dp.s_min[0], dp.s_min[1], dp.s_min[2],
        dp.s_dim[0], dp.s_dim[1], dp.s_dim[2], dp.pair_bid, dp.n_atoms, dp.n_shifts, dp.blk_off, dm_blk,
        NC == 1 ? 1.0 : 0.5, static_cast<std::int64_t>(dp.blk_stride), dp.n_grid, v.p0, v.p1, v.img_base, v.val_base,
        out);
}

template <int NS, int NC>
void launch_plan_vmat(cudaStream_t stream, const DeviceGridPlan& dp, const PlanView& v, const double* v_grid,
                      double* v_blk) {
    const int blk = 256;
    const double* c0 = NC == 1 ? v.vals : v.gx;
    plan_vmat_kernel<NS, NC><<<plan_launch_blocks(v, blk), blk, 0, stream>>>(
        dp.img_off, v.img, c0, v.gy, v.gz, dp.atom_no, dp.shift_box, dp.s_min[0], dp.s_min[1], dp.s_min[2],
        dp.s_dim[0], dp.s_dim[1], dp.s_dim[2], dp.pair_bid, dp.n_atoms, dp.n_shifts, dp.blk_off, v_grid,
        dp.dvol * (NC == 1 ? 1.0 : 0.5), static_cast<std::int64_t>(dp.blk_stride), dp.n_grid, v.p0, v.p1, v.img_base,
        v.val_base, v_blk);
}

template <int NC>
void launch_plan_rhoofd_spin(cudaStream_t stream, const DeviceGridPlan& dp, const PlanView& v, std::int32_t n_spin_mat,
                             const double* dm_blk, double* out) {
    if (n_spin_mat == 1) launch_plan_rhoofd<1, NC>(stream, dp, v, dm_blk, out);
    else if (n_spin_mat == 2) launch_plan_rhoofd<2, NC>(stream, dp, v, dm_blk, out);
    else launch_plan_rhoofd<4, NC>(stream, dp, v, dm_blk, out);
}

template <int NC>
void launch_plan_vmat_spin(cudaStream_t stream, const DeviceGridPlan& dp, const PlanView& v, std::int32_t n_spin_mat,
                           const double* v_grid, double* v_blk) {
    if (n_spin_mat == 1) launch_plan_vmat<1, NC>(stream, dp, v, v_grid, v_blk);
    else if (n_spin_mat == 2) launch_plan_vmat<2, NC>(stream, dp, v, v_grid, v_blk);
    else launch_plan_vmat<4, NC>(stream, dp, v, v_grid, v_blk);
}

// Which value arrays a launch reads.
constexpr unsigned kPlanVals = 1u, kPlanGrad = 2u, kPlanHess = 4u;

// Calls launch(view) once for a resident plan, or once per chunk for a streamed plan after copying the chunk's images
// and the requested value arrays from the host plan (stream-ordered, so one set of buffers serves all chunks).
// Returns false, before any launch, when the chunk buffers cannot be allocated.
template <class Launch>
bool run_on_plan(cudaStream_t stream, const DeviceGridPlan& dp, unsigned arrays, Launch&& launch) {
    if (!dp.streamed) {
        PlanView v;
        v.img = dp.img;
        v.vals = dp.vals;
        v.gx = dp.gx;
        v.gy = dp.gy;
        v.gz = dp.gz;
        v.hxx = dp.hxx;
        v.hxy = dp.hxy;
        v.hxz = dp.hxz;
        v.hyy = dp.hyy;
        v.hyz = dp.hyz;
        v.hzz = dp.hzz;
        v.p0 = 0;
        v.p1 = dp.n_grid;
        launch(v);
        return true;
    }
    // slots: 0 vals, 1-3 gradient, 4-9 Hessian (DeviceGridPlan::h_arr order)
    int slots[10];
    int n_slots = 0;
    if (arrays & kPlanVals) slots[n_slots++] = 0;
    if (arrays & kPlanGrad) for (int k = 1; k <= 3; ++k) slots[n_slots++] = k;
    if (arrays & kPlanHess) for (int k = 4; k <= 9; ++k) slots[n_slots++] = k;
    for (int k = 0; k < n_slots; ++k) {
        if (dp.h_arr[slots[k]] == nullptr) return false;
    }
    const std::size_t img_elems = std::max<std::size_t>(1, 5 * dp.max_chunk_imgs);
    const std::size_t val_elems = std::max<std::size_t>(1, dp.max_chunk_vals);
    auto round256 = [](std::size_t n) { return (n + 255) / 256 * 256; };  // keep every sub-buffer aligned
    const std::size_t img_bytes = round256(img_elems * sizeof(std::int32_t));
    const std::size_t set_bytes = round256(img_bytes + static_cast<std::size_t>(n_slots) * val_elems * sizeof(double));
    // with overlap: two buffer sets, copies on the copy stream, kernels on the call stream, ordered by events
    cudaStream_t cs = dp.overlap ? plan_copy_stream() : nullptr;
    const int n_sets = (cs != nullptr && dp.chunks.size() > 1) ? 2 : 1;
    if (n_sets == 1) cs = stream;
    void* buf = nullptr;
    if (cudaMallocAsync(&buf, n_sets * set_bytes, stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaEvent_t ready = nullptr, copied[2] = {nullptr, nullptr}, consumed[2] = {nullptr, nullptr};
    if (n_sets == 2) {
        bool ev_ok = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) == cudaSuccess;
        for (int b = 0; b < 2; ++b) {
            ev_ok = ev_ok && cudaEventCreateWithFlags(&copied[b], cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&consumed[b], cudaEventDisableTiming) == cudaSuccess;
        }
        if (!ev_ok) {
            cudaGetLastError();
            for (cudaEvent_t e : {ready, copied[0], copied[1], consumed[0], consumed[1]}) {
                if (e != nullptr) cudaEventDestroy(e);
            }
            cudaFreeAsync(buf, stream);
            return false;
        }
        // the copy stream must not write the buffers before their allocation on the call stream
        cudaEventRecord(ready, stream);
        cudaStreamWaitEvent(cs, ready, 0);
    }
    for (std::size_t i = 0; i < dp.chunks.size(); ++i) {
        const auto& c = dp.chunks[i];
        const int b = n_sets == 2 ? static_cast<int>(i & 1) : 0;
        char* set = static_cast<char*>(buf) + static_cast<std::size_t>(b) * set_bytes;
        auto* img_buf = reinterpret_cast<std::int32_t*>(set);
        auto* val_buf = reinterpret_cast<double*>(set + img_bytes);
        // buffer set b is refilled only after the kernels of chunk i - 2 have read it
        if (n_sets == 2 && i >= 2) cudaStreamWaitEvent(cs, consumed[b], 0);
        const std::size_t n_img = static_cast<std::size_t>(c.img1 - c.img0);
        const std::size_t n_val = static_cast<std::size_t>(c.val1 - c.val0);
        if (n_img > 0) {
            cudaMemcpyAsync(img_buf, dp.h_img + 5 * c.img0, 5 * n_img * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                            cs);
        }
        const double* dev[10] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        for (int k = 0; k < n_slots; ++k) {
            double* dst = val_buf + static_cast<std::size_t>(k) * val_elems;
            if (n_val > 0) {
                cudaMemcpyAsync(dst, dp.h_arr[slots[k]] + c.val0, n_val * sizeof(double), cudaMemcpyHostToDevice, cs);
            }
            dev[slots[k]] = dst;
        }
        if (n_sets == 2) {
            cudaEventRecord(copied[b], cs);
            cudaStreamWaitEvent(stream, copied[b], 0);
        }
        PlanView v;
        v.img = img_buf;
        v.vals = dev[0];
        v.gx = dev[1];
        v.gy = dev[2];
        v.gz = dev[3];
        v.hxx = dev[4];
        v.hxy = dev[5];
        v.hxz = dev[6];
        v.hyy = dev[7];
        v.hyz = dev[8];
        v.hzz = dev[9];
        v.p0 = c.p0;
        v.p1 = c.p1;
        v.img_base = c.img0;
        v.val_base = c.val0;
        launch(v);
        if (n_sets == 2) cudaEventRecord(consumed[b], stream);
    }
    // every copy precedes, through the events, a kernel on the call stream, so freeing on the call stream is safe
    cudaFreeAsync(buf, stream);
    if (n_sets == 2) {
        for (cudaEvent_t e : {ready, copied[0], copied[1], consumed[0], consumed[1]}) cudaEventDestroy(e);
    }
    return true;
}

// Returns true when the plan path handled the call.
// is_tau selects the kinetic-energy density (gradient plan, NC = 3).
bool plan_rhoofd(cudaStream_t stream, const DeviceGridPlan& dp, const double* dm_edges, std::int64_t n_edges,
                 std::int32_t n_spin_mat, bool is_tau, double* out, ffi::Error* err) {
    if (is_tau && !dp.has_grad) return false;
    const std::size_t stride = dp.blk_stride;
    double* dm_blk = nullptr;
    const std::size_t bytes = std::max<std::size_t>(1, stride * static_cast<std::size_t>(n_spin_mat)) * sizeof(double);
    if (cudaMallocAsync(reinterpret_cast<void**>(&dm_blk), bytes, stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaMemsetAsync(dm_blk, 0, bytes, stream);
    const int blk = 256;
    const std::int64_t n_be = static_cast<std::int64_t>(dp.n_block_edges);
    if (n_be > 0) {
        edges_to_blocks_kernel<<<static_cast<unsigned>((n_be + blk - 1) / blk), blk, 0, stream>>>(
            dm_edges, n_edges, dp.be_abs, dp.be_idx, n_be, n_spin_mat, static_cast<std::int64_t>(stride), dm_blk);
    }
    const bool ran = run_on_plan(stream, dp, is_tau ? kPlanGrad : kPlanVals, [&](const PlanView& v) {
        if (is_tau) launch_plan_rhoofd_spin<3>(stream, dp, v, n_spin_mat, dm_blk, out);
        else launch_plan_rhoofd_spin<1>(stream, dp, v, n_spin_mat, dm_blk, out);
    });
    cudaFreeAsync(dm_blk, stream);
    if (!ran) return false;
    const cudaError_t e = cudaGetLastError();
    *err = e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(e));
    return true;
}

bool plan_vmat(cudaStream_t stream, const DeviceGridPlan& dp, const double* v_grid, std::int64_t n_edges,
               std::int32_t n_spin_mat, bool is_tau, double* out_edges, ffi::Error* err) {
    if (is_tau && !dp.has_grad) return false;
    const std::size_t stride = dp.blk_stride;
    double* v_blk = nullptr;
    const std::size_t bytes = std::max<std::size_t>(1, stride * static_cast<std::size_t>(n_spin_mat)) * sizeof(double);
    if (cudaMallocAsync(reinterpret_cast<void**>(&v_blk), bytes, stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaMemsetAsync(v_blk, 0, bytes, stream);
    const bool ran = run_on_plan(stream, dp, is_tau ? kPlanGrad : kPlanVals, [&](const PlanView& v) {
        if (is_tau) launch_plan_vmat_spin<3>(stream, dp, v, n_spin_mat, v_grid, v_blk);
        else launch_plan_vmat_spin<1>(stream, dp, v, n_spin_mat, v_grid, v_blk);
    });
    if (!ran) {
        cudaFreeAsync(v_blk, stream);
        return false;
    }
    cudaMemsetAsync(out_edges, 0, static_cast<std::size_t>(n_spin_mat) * static_cast<std::size_t>(n_edges) * sizeof(double), stream);
    const int blk = 256;
    const std::int64_t n_be = static_cast<std::int64_t>(dp.n_block_edges);
    if (n_be > 0) {
        blocks_to_edges_kernel<<<static_cast<unsigned>((n_be + blk - 1) / blk), blk, 0, stream>>>(
            v_blk, static_cast<std::int64_t>(stride), dp.be_abs, dp.be_idx, n_be, n_spin_mat, n_edges, out_edges);
    }
    cudaFreeAsync(v_blk, stream);
    const cudaError_t e = cudaGetLastError();
    *err = e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(e));
    return true;
}

// Backward of rhoofd (TAU = false) and tauofd (TAU = true) on the plan, for forces and stress at fixed DM:
//   d_dm_s,ab      = sum_p vc_s(p) prod_ab(p),  prod = phi_a phi_b  or  1/2 grad phi_a . grad phi_b,
//   d_pos[atom X] -= sum_p f_X(p),  d_cell[i][j] += sum_p f_X,i(p) (frac_j(p) - shift_X,j),
// with w_ab(p) = sum_s vc_s(p) D_s,ab and, for every atom image X at p, the a-side term of the blocks (X, Y) plus the
// b-side term of the blocks (Y, X):
//   rho: f_X = sum_{a in X} grad phi_a sum_b w_ab phi_b + sum_{b in X} grad phi_b sum_a w_ab phi_a,
//   tau: the same with grad phi -> 1/2 H phi (orbital Hessian) and phi -> grad phi.
// These are the terms of DfscfGradCoreImpl (grid_ffi.cpp) regrouped per image, so no per-pair reduction is needed.
// One warp per grid point; lanes over block entries; d_dm and d_pos by atomics, d_cell reduced per thread block.
template <int NS, bool TAU>
__global__ void plan_dfscf_kernel(
    const std::int32_t* __restrict__ img_off, const std::int32_t* __restrict__ img, const double* __restrict__ vals,
    const double* __restrict__ gx, const double* __restrict__ gy, const double* __restrict__ gz,
    const double* __restrict__ hxx, const double* __restrict__ hxy, const double* __restrict__ hxz,
    const double* __restrict__ hyy, const double* __restrict__ hyz, const double* __restrict__ hzz,
    const std::int32_t* __restrict__ atom_no, const std::int32_t* __restrict__ shift_box,
    int sm0, int sm1, int sm2, int sd0, int sd1, int sd2, const std::int32_t* __restrict__ pair_bid,
    std::int32_t n_atoms, std::int32_t n_shifts, const std::int32_t* __restrict__ blk_off,
    const double* __restrict__ dm_blk, const double* __restrict__ v_cot, std::int64_t stride, std::int64_t n_grid,
    std::int64_t p0, std::int64_t p1, std::int64_t img_base, std::int64_t val_base,
    std::int32_t n1, std::int32_t n2, std::int32_t n3,
    double* __restrict__ ddm_blk, double* __restrict__ d_pos, double* __restrict__ d_cell) {
    __shared__ double s_cell[9];
    if (threadIdx.x < 9) s_cell[threadIdx.x] = 0.0;
    __syncthreads();
    const std::int64_t p = p0 + ((static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    double cell_acc[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) cell_acc[k] = 0.0;
    if (p < p1) {  // uniform per warp: the shuffles below see all 32 lanes
        double vc[NS];
#pragma unroll
        for (int s = 0; s < NS; ++s) vc[s] = v_cot[s * n_grid + p];
        const std::int64_t n23 = static_cast<std::int64_t>(n2) * n3;
        const double fr[3] = {
            static_cast<double>(p / n23) / static_cast<double>(n1),
            static_cast<double>((p / n3) % n2) / static_cast<double>(n2),
            static_cast<double>(p % n3) / static_cast<double>(n3),
        };
        const std::int32_t beg = img_off[p], end = img_off[p + 1];
        for (std::int32_t ix = beg; ix < end; ++ix) {
            const std::int32_t* gX = img + 5 * (static_cast<std::int64_t>(ix) - img_base);
            const int noX = atom_no[gX[0]];
            const std::int64_t oX = gX[4] - val_base;
            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (std::int32_t iy = beg; iy < end; ++iy) {
                const std::int32_t* gY = img + 5 * (static_cast<std::int64_t>(iy) - img_base);
                const int noY = atom_no[gY[0]];
                const std::int64_t oY = gY[4] - val_base;
                // a-side: block (X, Y), entries (a in X, b in Y); also the d_dm contribution of every ordered pair
                const std::int32_t bid = dev_plan_block(gX, gY, shift_box, sm0, sm1, sm2, sd0, sd1, sd2, pair_bid, n_atoms, n_shifts);
                if (bid >= 0) {
                    const double* d0 = dm_blk + blk_off[bid];
                    double* dd0 = ddm_blk + blk_off[bid];
                    for (int idx = lane; idx < noX * noY; idx += 32) {
                        const int a = idx / noY;
                        const std::int64_t ka = oX + a, kb = oY + (idx - a * noY);
                        double w = 0.0;
#pragma unroll
                        for (int s = 0; s < NS; ++s) w += vc[s] * d0[s * stride + idx];
                        double prod;
                        if constexpr (!TAU) {
                            prod = vals[ka] * vals[kb];
                            const double t = w * vals[kb];
                            fx += gx[ka] * t;
                            fy += gy[ka] * t;
                            fz += gz[ka] * t;
                        } else {
                            prod = 0.5 * (gx[ka] * gx[kb] + gy[ka] * gy[kb] + gz[ka] * gz[kb]);
                            const double hw = 0.5 * w;
                            fx += hw * (hxx[ka] * gx[kb] + hxy[ka] * gy[kb] + hxz[ka] * gz[kb]);
                            fy += hw * (hxy[ka] * gx[kb] + hyy[ka] * gy[kb] + hyz[ka] * gz[kb]);
                            fz += hw * (hxz[ka] * gx[kb] + hyz[ka] * gy[kb] + hzz[ka] * gz[kb]);
                        }
#pragma unroll
                        for (int s = 0; s < NS; ++s) atomicAdd(&dd0[s * stride + idx], vc[s] * prod);
                    }
                }
                // b-side: block (Y, X), entries (a in Y, b in X)
                const std::int32_t bid2 = dev_plan_block(gY, gX, shift_box, sm0, sm1, sm2, sd0, sd1, sd2, pair_bid, n_atoms, n_shifts);
                if (bid2 >= 0) {
                    const double* d1 = dm_blk + blk_off[bid2];
                    for (int idx = lane; idx < noY * noX; idx += 32) {
                        const int a = idx / noX;
                        const std::int64_t ka = oY + a, kb = oX + (idx - a * noX);
                        double w = 0.0;
#pragma unroll
                        for (int s = 0; s < NS; ++s) w += vc[s] * d1[s * stride + idx];
                        if constexpr (!TAU) {
                            const double t = w * vals[ka];
                            fx += gx[kb] * t;
                            fy += gy[kb] * t;
                            fz += gz[kb] * t;
                        } else {
                            const double hw = 0.5 * w;
                            fx += hw * (hxx[kb] * gx[ka] + hxy[kb] * gy[ka] + hxz[kb] * gz[ka]);
                            fy += hw * (hxy[kb] * gx[ka] + hyy[kb] * gy[ka] + hyz[kb] * gz[ka]);
                            fz += hw * (hxz[kb] * gx[ka] + hyz[kb] * gy[ka] + hzz[kb] * gz[ka]);
                        }
                    }
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                fx += __shfl_down_sync(0xffffffffu, fx, off);
                fy += __shfl_down_sync(0xffffffffu, fy, off);
                fz += __shfl_down_sync(0xffffffffu, fz, off);
            }
            if (lane == 0) {
                const std::int64_t at = gX[0];
                atomicAdd(&d_pos[3 * at + 0], -fx);
                atomicAdd(&d_pos[3 * at + 1], -fy);
                atomicAdd(&d_pos[3 * at + 2], -fz);
                const double df[3] = {fr[0] - gX[1], fr[1] - gX[2], fr[2] - gX[3]};
                const double f[3] = {fx, fy, fz};
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int j = 0; j < 3; ++j) cell_acc[3 * i + j] += f[i] * df[j];
                }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(&s_cell[k], cell_acc[k]);
    }
    __syncthreads();
    if (threadIdx.x < 9) atomicAdd(&d_cell[threadIdx.x], s_cell[threadIdx.x]);
}

template <int NS, bool TAU>
void launch_plan_dfscf(cudaStream_t stream, const DeviceGridPlan& dp, const PlanView& v, const double* dm_blk,
                       const double* v_cot, std::int32_t n1, std::int32_t n2, std::int32_t n3, double* ddm_blk,
                       double* d_pos, double* d_cell) {
    const int blk = 256;
    plan_dfscf_kernel<NS, TAU><<<plan_launch_blocks(v, blk), blk, 0, stream>>>(
        dp.img_off, v.img, v.vals, v.gx, v.gy, v.gz, v.hxx, v.hxy, v.hxz, v.hyy, v.hyz, v.hzz,
        dp.atom_no, dp.shift_box, dp.s_min[0], dp.s_min[1], dp.s_min[2], dp.s_dim[0], dp.s_dim[1], dp.s_dim[2],
        dp.pair_bid, dp.n_atoms, dp.n_shifts, dp.blk_off, dm_blk, v_cot, static_cast<std::int64_t>(dp.blk_stride),
        dp.n_grid, v.p0, v.p1, v.img_base, v.val_base, n1, n2, n3, ddm_blk, d_pos, d_cell);
}

template <bool TAU>
void launch_plan_dfscf_spin(cudaStream_t stream, const DeviceGridPlan& dp, const PlanView& v, std::int32_t n_spin_mat,
                            const double* dm_blk, const double* v_cot, std::int32_t n1, std::int32_t n2,
                            std::int32_t n3, double* ddm_blk, double* d_pos, double* d_cell) {
    if (n_spin_mat == 1) launch_plan_dfscf<1, TAU>(stream, dp, v, dm_blk, v_cot, n1, n2, n3, ddm_blk, d_pos, d_cell);
    else if (n_spin_mat == 2) launch_plan_dfscf<2, TAU>(stream, dp, v, dm_blk, v_cot, n1, n2, n3, ddm_blk, d_pos, d_cell);
    else launch_plan_dfscf<4, TAU>(stream, dp, v, dm_blk, v_cot, n1, n2, n3, ddm_blk, d_pos, d_cell);
}

// Returns true when the plan path handled the call. d_pos and d_cell must already be zero.
bool plan_dfscf(cudaStream_t stream, const DeviceGridPlan& dp, const double* dm_edges, const double* v_cot,
                std::int64_t n_edges, std::int32_t n_spin_mat, bool is_tau, std::int32_t n1, std::int32_t n2,
                std::int32_t n3, double* out_d_dm, double* d_pos, double* d_cell, ffi::Error* err) {
    if (!dp.has_grad || (is_tau && !dp.has_hess)) return false;
    const std::size_t stride = dp.blk_stride;
    const std::size_t n_blk = std::max<std::size_t>(1, stride * static_cast<std::size_t>(n_spin_mat));
    double* work = nullptr;  // DM blocks, then d_dm blocks
    if (cudaMallocAsync(reinterpret_cast<void**>(&work), 2 * n_blk * sizeof(double), stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    double* dm_blk = work;
    double* ddm_blk = work + n_blk;
    cudaMemsetAsync(work, 0, 2 * n_blk * sizeof(double), stream);
    const int blk = 256;
    const std::int64_t n_be = static_cast<std::int64_t>(dp.n_block_edges);
    if (n_be > 0) {
        edges_to_blocks_kernel<<<static_cast<unsigned>((n_be + blk - 1) / blk), blk, 0, stream>>>(
            dm_edges, n_edges, dp.be_abs, dp.be_idx, n_be, n_spin_mat, static_cast<std::int64_t>(stride), dm_blk);
    }
    const bool ran = run_on_plan(stream, dp, is_tau ? (kPlanGrad | kPlanHess) : (kPlanVals | kPlanGrad),
                                 [&](const PlanView& v) {
        if (is_tau) launch_plan_dfscf_spin<true>(stream, dp, v, n_spin_mat, dm_blk, v_cot, n1, n2, n3, ddm_blk, d_pos, d_cell);
        else launch_plan_dfscf_spin<false>(stream, dp, v, n_spin_mat, dm_blk, v_cot, n1, n2, n3, ddm_blk, d_pos, d_cell);
    });
    if (!ran) {
        cudaFreeAsync(work, stream);
        return false;
    }
    cudaMemsetAsync(out_d_dm, 0, static_cast<std::size_t>(n_spin_mat) * static_cast<std::size_t>(n_edges) * sizeof(double), stream);
    if (n_be > 0) {
        blocks_to_edges_kernel<<<static_cast<unsigned>((n_be + blk - 1) / blk), blk, 0, stream>>>(
            ddm_blk, static_cast<std::int64_t>(stride), dp.be_abs, dp.be_idx, n_be, n_spin_mat, n_edges, out_d_dm);
    }
    cudaFreeAsync(work, stream);
    const cudaError_t e = cudaGetLastError();
    *err = e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(e));
    return true;
}

template <int Mode>
ffi::Error OrbGridForwardCudaImpl(
    cudaStream_t stream,
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
    const int32_t n_orb = static_cast<int32_t>(orb_atom.dimensions()[0]);
    const int32_t n_shells = static_cast<int32_t>(pao_l.dimensions()[0]);
    const int32_t n_pao_pts = static_cast<int32_t>(pao_values.dimensions()[1]);
    const int32_t n_edges = static_cast<int32_t>(dm_edges.dimensions()[1]);
    const int64_t n_grid = n1 * n2 * n3;

    CUDA_FFI_CHECK(cudaMemsetAsync(
        out_grid->typed_data(), 0,
        static_cast<size_t>(n_spin_mat) * static_cast<size_t>(n_grid) * sizeof(double), stream));
    if (n_edges == 0) return ffi::Error::Success();

    if constexpr (Mode == 0 || Mode == 3) {  // rhoofd, tauofd
        constexpr bool kTau = Mode == 3;
        const auto dp = get_device_grid_plan(
            stream, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values, orb_atom, orb_g_shell, orb_m, orb_rcut,
            row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts, n_edges, n1, n2, n3, kTau);
        ffi::Error plan_err = ffi::Error::Success();
        if (dp && plan_rhoofd(stream, *dp, dm_edges.typed_data(), n_edges, n_spin_mat, kTau, out_grid->typed_data(),
                              &plan_err)) {
            return plan_err;
        }
    }

    const size_t shell_elems = static_cast<size_t>(n_shells) * static_cast<size_t>(n_pao_pts);
    const size_t ws_bytes = (14ULL + 3ULL * shell_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));

    double* geom_dev = ws_dev;
    double* f_dev = ws_dev + 14;
    double* y2_dev = f_dev + shell_elems;
    double* u_dev = y2_dev + shell_elems;

    prepare_geom_and_pao_splines_kernel<<<std::max(1, n_shells), 1, 0, stream>>>(
        cell.typed_data(), pao_r_max.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
        n_shells, n_pao_pts, n_grid, geom_dev, f_dev, y2_dev, u_dev);

    const int blk = 128;
    const int grid_e = (n_edges + blk - 1) / blk;
    edge_orbital_grid_kernel<Mode><<<grid_e, blk, 0, stream>>>(
        dm_edges.typed_data(), nullptr, positions.typed_data(), cell.typed_data(), geom_dev,
        pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), f_dev, y2_dev,
        orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
        row_ptr.typed_data(), dst_unit_orb.typed_data(), edge_cell_idx.typed_data(), cell_shifts.typed_data(),
        n_orb, n_pao_pts, n_edges, static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        n_spin_mat, out_grid->typed_data(), nullptr, nullptr);

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

template <int Mode>
ffi::Error VmatGridCudaImpl(
    cudaStream_t stream,
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
    const int32_t n_orb = static_cast<int32_t>(orb_atom.dimensions()[0]);
    const int32_t n_shells = static_cast<int32_t>(pao_l.dimensions()[0]);
    const int32_t n_pao_pts = static_cast<int32_t>(pao_values.dimensions()[1]);
    const int32_t n_edges = static_cast<int32_t>(out_v_sparse->dimensions()[1]);
    const int64_t n_grid = n1 * n2 * n3;
    if (n_edges == 0) return ffi::Error::Success();
    if constexpr (Mode == 1 || Mode == 4) {  // vmat, vmat_tau
        constexpr bool kTau = Mode == 4;
        const auto dp = get_device_grid_plan(
            stream, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values, orb_atom, orb_g_shell, orb_m, orb_rcut,
            row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts, n_edges, n1, n2, n3, kTau);
        ffi::Error plan_err = ffi::Error::Success();
        if (dp && plan_vmat(stream, *dp, v_grid.typed_data(), n_edges, n_spin_mat, kTau, out_v_sparse->typed_data(),
                            &plan_err)) {
            return plan_err;
        }
    }

    const size_t shell_elems = static_cast<size_t>(n_shells) * static_cast<size_t>(n_pao_pts);
    const size_t ws_bytes = (14ULL + 3ULL * shell_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));

    double* geom_dev = ws_dev;
    double* f_dev = ws_dev + 14;
    double* y2_dev = f_dev + shell_elems;
    double* u_dev = y2_dev + shell_elems;

    prepare_geom_and_pao_splines_kernel<<<std::max(1, n_shells), 1, 0, stream>>>(
        cell.typed_data(), pao_r_max.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
        n_shells, n_pao_pts, n_grid, geom_dev, f_dev, y2_dev, u_dev);

    const int blk = 128;
    const int grid_e = (n_edges + blk - 1) / blk;
    edge_orbital_grid_kernel<Mode><<<grid_e, blk, 0, stream>>>(
        v_grid.typed_data(), nullptr, positions.typed_data(), cell.typed_data(), geom_dev,
        pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), f_dev, y2_dev,
        orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
        row_ptr.typed_data(), dst_unit_orb.typed_data(), edge_cell_idx.typed_data(), cell_shifts.typed_data(),
        n_orb, n_pao_pts, n_edges, static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        n_spin_mat, out_v_sparse->typed_data(), nullptr, nullptr);

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

template <int Mode>
ffi::Error DfscfGradGridCudaImpl(
    cudaStream_t stream,
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
    const int64_t n_grid = n1 * n2 * n3;

    CUDA_FFI_CHECK(cudaMemsetAsync(
        out_d_pos->typed_data(), 0, static_cast<size_t>(n_atoms) * 3ULL * sizeof(double), stream));
    CUDA_FFI_CHECK(cudaMemsetAsync(out_d_cell->typed_data(), 0, 9ULL * sizeof(double), stream));
    if (n_edges == 0) return ffi::Error::Success();

    {  // plan path (D-038): Mode 2 needs values and gradients, Mode 5 gradients and Hessians
        constexpr bool kTau = Mode == 5;
        const auto dp = get_device_grid_plan(
            stream, positions, cell, pao_r_max, pao_rcut, pao_l, pao_values, orb_atom, orb_g_shell, orb_m, orb_rcut,
            row_ptr, dst_unit_orb, edge_cell_idx, cell_shifts, n_edges, n1, n2, n3, true, kTau);
        ffi::Error plan_err = ffi::Error::Success();
        if (dp && plan_dfscf(stream, *dp, dm_edges.typed_data(), v_cotangent.typed_data(), n_edges, n_spin_mat, kTau,
                             static_cast<std::int32_t>(n1), static_cast<std::int32_t>(n2), static_cast<std::int32_t>(n3),
                             out_d_dm->typed_data(), out_d_pos->typed_data(), out_d_cell->typed_data(), &plan_err)) {
            return plan_err;
        }
    }

    const size_t shell_elems = static_cast<size_t>(n_shells) * static_cast<size_t>(n_pao_pts);
    const size_t ws_bytes = (14ULL + 3ULL * shell_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));

    double* geom_dev = ws_dev;
    double* f_dev = ws_dev + 14;
    double* y2_dev = f_dev + shell_elems;
    double* u_dev = y2_dev + shell_elems;

    prepare_geom_and_pao_splines_kernel<<<std::max(1, n_shells), 1, 0, stream>>>(
        cell.typed_data(), pao_r_max.typed_data(), pao_l.typed_data(), pao_values.typed_data(),
        n_shells, n_pao_pts, n_grid, geom_dev, f_dev, y2_dev, u_dev);

    const int blk = 128;
    const int grid_e = (n_edges + blk - 1) / blk;
    edge_orbital_grid_kernel<Mode><<<grid_e, blk, 0, stream>>>(
        dm_edges.typed_data(), v_cotangent.typed_data(), positions.typed_data(), cell.typed_data(), geom_dev,
        pao_r_max.typed_data(), pao_rcut.typed_data(), pao_l.typed_data(), f_dev, y2_dev,
        orb_atom.typed_data(), orb_g_shell.typed_data(), orb_m.typed_data(), orb_rcut.typed_data(),
        row_ptr.typed_data(), dst_unit_orb.typed_data(), edge_cell_idx.typed_data(), cell_shifts.typed_data(),
        n_orb, n_pao_pts, n_edges, static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        n_spin_mat, out_d_dm->typed_data(), out_d_pos->typed_data(), out_d_cell->typed_data());

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

ffi::Error NaGridCudaImpl(
    cudaStream_t stream,
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
    const size_t n_grid_bytes = static_cast<size_t>(n1 * n2 * n3) * sizeof(double);

    CUDA_FFI_CHECK(cudaMemsetAsync(out_rho_na->typed_data(), 0, n_grid_bytes, stream));
    CUDA_FFI_CHECK(cudaMemsetAsync(out_v_na->typed_data(), 0, n_grid_bytes, stream));
    CUDA_FFI_CHECK(cudaMemsetAsync(out_rho_core->typed_data(), 0, n_grid_bytes, stream));
    if (n_atoms == 0) return ffi::Error::Success();

    const size_t sp_elems = static_cast<size_t>(3 * n_sp) * static_cast<size_t>(n_pts);
    const size_t ws_bytes = (12ULL + 2ULL * sp_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));

    double* geom_dev = ws_dev;
    double* y2_dev = ws_dev + 12;
    double* u_dev = y2_dev + sp_elems;

    prepare_na_geom_and_splines_kernel<<<std::max(1, 3 * n_sp), 1, 0, stream>>>(
        cell.typed_data(), sp_rmax.typed_data(), sp_rho_na.typed_data(), sp_v_na.typed_data(),
        sp_rho_core.typed_data(), n_sp, n_pts, geom_dev, y2_dev, u_dev);

    na_grid_atom_kernel<<<n_atoms, 256, 0, stream>>>(
        positions.typed_data(), cell.typed_data(), geom_dev, atom_species.typed_data(),
        sp_rmax.typed_data(), sp_rcut_na.typed_data(), sp_rho_na.typed_data(),
        sp_v_na.typed_data(), sp_rho_core.typed_data(), y2_dev, n_atoms, n_pts,
        static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        out_rho_na->typed_data(), out_v_na->typed_data(), out_rho_core->typed_data());

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

ffi::Error NaGridGradCudaImpl(
    cudaStream_t stream,
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

    CUDA_FFI_CHECK(cudaMemsetAsync(
        out_d_pos->typed_data(), 0, static_cast<size_t>(n_atoms) * 3ULL * sizeof(double), stream));
    CUDA_FFI_CHECK(cudaMemsetAsync(out_d_cell->typed_data(), 0, 9ULL * sizeof(double), stream));
    if (n_atoms == 0) return ffi::Error::Success();

    const size_t sp_elems = static_cast<size_t>(3 * n_sp) * static_cast<size_t>(n_pts);
    const size_t ws_bytes = (12ULL + 2ULL * sp_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));

    double* geom_dev = ws_dev;
    double* y2_dev = ws_dev + 12;
    double* u_dev = y2_dev + sp_elems;

    prepare_na_geom_and_splines_kernel<<<std::max(1, 3 * n_sp), 1, 0, stream>>>(
        cell.typed_data(), sp_rmax.typed_data(), sp_rho_na.typed_data(), sp_v_na.typed_data(),
        sp_rho_core.typed_data(), n_sp, n_pts, geom_dev, y2_dev, u_dev);

    na_grid_grad_atom_kernel<<<n_atoms, 256, 0, stream>>>(
        w_rho_na.typed_data(), w_v_na.typed_data(), w_rho_core.typed_data(),
        positions.typed_data(), cell.typed_data(), geom_dev, atom_species.typed_data(),
        sp_rmax.typed_data(), sp_rcut_na.typed_data(), sp_rho_na.typed_data(),
        sp_v_na.typed_data(), sp_rho_core.typed_data(), y2_dev, n_atoms, n_pts,
        static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        out_d_pos->typed_data(), out_d_cell->typed_data());

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

ffi::Error XcEvalCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F64> rho_spin,
    ffi::Buffer<ffi::DataType::F64> sigma_spin,
    int64_t xc_id,
    int64_t n_spin_pol,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_exc_density,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_rho,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sigma) {

    const int64_t n_grid = static_cast<int64_t>(out_exc_density->element_count());
    if (n_grid <= 0) return ffi::Error::Success();
    // LDA (PZ81, PW92), PBE and PBEsol run on the device with the built-in kernel, whose parameters match the Libxc ids
    // the CPU path calls (D-039). Libxc on the host (copy out, evaluate, copy back) is kept only for other ids and for
    // A/B runs with JSIESTA_CUDA_XC_LIBXC=1.
    const bool builtin_id = (xc_id == 1 || xc_id == 2 || xc_id == 10 || xc_id == 11);
#ifdef JSIESTA_HAS_LIBXC
    const char* force_libxc = std::getenv("JSIESTA_CUDA_XC_LIBXC");
    if (!builtin_id || (force_libxc != nullptr && force_libxc[0] == '1')) {
        const size_t np = static_cast<size_t>(n_grid);
        std::vector<double> h_r(2 * np), h_s(3 * np);
        std::vector<double> h_e(np), h_vr(2 * np), h_vs(3 * np);

        CUDA_FFI_CHECK(cudaMemcpyAsync(h_r.data(), rho_spin.typed_data(), 2 * np * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_FFI_CHECK(cudaMemcpyAsync(h_s.data(), sigma_spin.typed_data(), 3 * np * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_FFI_CHECK(cudaStreamSynchronize(stream));

        const int rc = jsiesta::evaluate_xc_libxc_host(
            n_grid, h_r.data(), h_s.data(), xc_id, n_spin_pol,
            h_e.data(), h_vr.data(), h_vs.data());
        if (rc != 0) return ffi::Error::InvalidArgument("Libxc XC evaluation failed in XcEvalCudaImpl.");
        CUDA_FFI_CHECK(cudaMemcpyAsync(out_exc_density->typed_data(), h_e.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_FFI_CHECK(cudaMemcpyAsync(out_v_rho->typed_data(), h_vr.data(), 2 * np * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_FFI_CHECK(cudaMemcpyAsync(out_v_sigma->typed_data(), h_vs.data(), 3 * np * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_FFI_CHECK(cudaStreamSynchronize(stream));
        return ffi::Error::Success();
    }
#endif
    if (!builtin_id) return ffi::Error::InvalidArgument("Unknown LDA/GGA functional id in XcEvalCudaImpl.");
    const int blk = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((n_grid + blk - 1) / blk, 4096));
    xc_eval_cuda_kernel<<<blocks, blk, 0, stream>>>(
        rho_spin.typed_data(), sigma_spin.typed_data(), n_grid, static_cast<int32_t>(xc_id),
        static_cast<int32_t>(n_spin_pol), out_exc_density->typed_data(), out_v_rho->typed_data(), out_v_sigma->typed_data());
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

ffi::Error TauCoreCudaImpl(
    cudaStream_t stream,
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
    const size_t n_grid_bytes = static_cast<size_t>(n1 * n2 * n3) * sizeof(double);

    CUDA_FFI_CHECK(cudaMemsetAsync(out_tau_core->typed_data(), 0, n_grid_bytes, stream));
    if (n_atoms == 0) return ffi::Error::Success();

    const size_t sp_elems = static_cast<size_t>(n_sp) * static_cast<size_t>(n_pts);
    const size_t ws_bytes = (12ULL + 2ULL * sp_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));
    double* geom_dev = ws_dev;
    double* y2_dev = ws_dev + 12;
    double* u_dev = y2_dev + sp_elems;

    prepare_geom_and_single_spline_kernel<<<std::max(1, n_sp), 1, 0, stream>>>(
        cell.typed_data(), sp_rmax.typed_data(), sp_tau_core.typed_data(),
        n_sp, n_pts, geom_dev, y2_dev, u_dev);

    tau_core_atom_kernel<false><<<n_atoms, 256, 0, stream>>>(
        nullptr, positions.typed_data(), cell.typed_data(), geom_dev, atom_species.typed_data(),
        sp_rmax.typed_data(), sp_tau_core.typed_data(), y2_dev, n_atoms, n_pts,
        static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        out_tau_core->typed_data(), nullptr);

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

ffi::Error TauCoreGradCudaImpl(
    cudaStream_t stream,
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

    CUDA_FFI_CHECK(cudaMemsetAsync(
        out_d_pos->typed_data(), 0, static_cast<size_t>(n_atoms) * 3ULL * sizeof(double), stream));
    CUDA_FFI_CHECK(cudaMemsetAsync(out_d_cell->typed_data(), 0, 9ULL * sizeof(double), stream));
    if (n_atoms == 0) return ffi::Error::Success();

    const size_t sp_elems = static_cast<size_t>(n_sp) * static_cast<size_t>(n_pts);
    const size_t ws_bytes = (12ULL + 2ULL * sp_elems) * sizeof(double);
    double* ws_dev = nullptr;
    CUDA_FFI_CHECK(cudaMallocAsync(&ws_dev, ws_bytes, stream));
    double* geom_dev = ws_dev;
    double* y2_dev = ws_dev + 12;
    double* u_dev = y2_dev + sp_elems;

    prepare_geom_and_single_spline_kernel<<<std::max(1, n_sp), 1, 0, stream>>>(
        cell.typed_data(), sp_rmax.typed_data(), sp_tau_core.typed_data(),
        n_sp, n_pts, geom_dev, y2_dev, u_dev);

    tau_core_atom_kernel<true><<<n_atoms, 256, 0, stream>>>(
        w_tau_core.typed_data(), positions.typed_data(), cell.typed_data(), geom_dev,
        atom_species.typed_data(), sp_rmax.typed_data(), sp_tau_core.typed_data(), y2_dev,
        n_atoms, n_pts, static_cast<int32_t>(n1), static_cast<int32_t>(n2), static_cast<int32_t>(n3),
        out_d_pos->typed_data(), out_d_cell->typed_data());

    CUDA_FFI_CHECK(cudaFreeAsync(ws_dev, stream));
    CUDA_FFI_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

ffi::Error XcEvalMggaCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F64> rho_spin,
    ffi::Buffer<ffi::DataType::F64> sigma_spin,
    ffi::Buffer<ffi::DataType::F64> tau_spin,
    int64_t x_id,
    int64_t c_id,
    int64_t n_spin_pol,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_exc_density,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_rho,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_sigma,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_v_tau) {

    const int64_t n_grid = static_cast<int64_t>(out_exc_density->element_count());
    if (n_grid <= 0) return ffi::Error::Success();
    const size_t np = static_cast<size_t>(n_grid);

    std::vector<double> h_r(2 * np), h_s(3 * np), h_t(2 * np);
    std::vector<double> h_e(np), h_vr(2 * np), h_vs(3 * np), h_vt(2 * np);

    CUDA_FFI_CHECK(cudaMemcpyAsync(h_r.data(), rho_spin.typed_data(), 2 * np * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_FFI_CHECK(cudaMemcpyAsync(h_s.data(), sigma_spin.typed_data(), 3 * np * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_FFI_CHECK(cudaMemcpyAsync(h_t.data(), tau_spin.typed_data(), 2 * np * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_FFI_CHECK(cudaStreamSynchronize(stream));

    const int rc = jsiesta::evaluate_mgga_libxc_host(
        n_grid, h_r.data(), h_s.data(), h_t.data(), x_id, c_id, n_spin_pol,
        h_e.data(), h_vr.data(), h_vs.data(), h_vt.data());
    if (rc == 1) {
        return ffi::Error::InvalidArgument(
            "meta-GGA XC evaluation requires jsiesta compiled with external libxc (JSIESTA_HAS_LIBXC=1).");
    }
    if (rc == 2) {
        return ffi::Error::InvalidArgument("Unknown libxc functional ID in XcEvalMggaCudaImpl.");
    }

    CUDA_FFI_CHECK(cudaMemcpyAsync(out_exc_density->typed_data(), h_e.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_FFI_CHECK(cudaMemcpyAsync(out_v_rho->typed_data(), h_vr.data(), 2 * np * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_FFI_CHECK(cudaMemcpyAsync(out_v_sigma->typed_data(), h_vs.data(), 3 * np * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_FFI_CHECK(cudaMemcpyAsync(out_v_tau->typed_data(), h_vt.data(), 2 * np * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_FFI_CHECK(cudaStreamSynchronize(stream));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RhoofdCudaHandler, OrbGridForwardCudaImpl<0>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VmatCudaHandler, VmatGridCudaImpl<1>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DfscfGradCudaHandler, DfscfGradGridCudaImpl<2>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NaGridCudaHandler, NaGridCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NaGridGradCudaHandler, NaGridGradCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XcEvalCudaHandler, XcEvalCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("xc_id").Attr<int64_t>("n_spin_pol")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauofdCudaHandler, OrbGridForwardCudaImpl<3>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VmatTauCudaHandler, VmatGridCudaImpl<4>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauDfscfGradCudaHandler, DfscfGradGridCudaImpl<5>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3").Attr<int64_t>("nspin")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauCoreCudaHandler, TauCoreCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TauCoreGradCudaHandler, TauCoreGradCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("n1").Attr<int64_t>("n2").Attr<int64_t>("n3")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XcEvalMggaCudaHandler, XcEvalMggaCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<int64_t>("x_id").Attr<int64_t>("c_id").Attr<int64_t>("n_spin_pol")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

}  // namespace

#endif  // __CUDACC__

}  // namespace jsiesta::cuda

#ifdef __CUDACC__
extern "C" {
XLA_FFI_Error* jsiesta_rhoofd_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::RhoofdCudaHandler(call);
}
XLA_FFI_Error* jsiesta_vmat_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::VmatCudaHandler(call);
}
XLA_FFI_Error* jsiesta_dfscf_grad_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::DfscfGradCudaHandler(call);
}
XLA_FFI_Error* jsiesta_na_grid_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::NaGridCudaHandler(call);
}
XLA_FFI_Error* jsiesta_na_grid_grad_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::NaGridGradCudaHandler(call);
}
XLA_FFI_Error* jsiesta_xc_eval_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::XcEvalCudaHandler(call);
}
XLA_FFI_Error* jsiesta_tauofd_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::TauofdCudaHandler(call);
}
XLA_FFI_Error* jsiesta_vmat_tau_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::VmatTauCudaHandler(call);
}
XLA_FFI_Error* jsiesta_tau_dfscf_grad_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::TauDfscfGradCudaHandler(call);
}
XLA_FFI_Error* jsiesta_tau_core_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::TauCoreCudaHandler(call);
}
XLA_FFI_Error* jsiesta_tau_core_grad_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::TauCoreGradCudaHandler(call);
}
XLA_FFI_Error* jsiesta_xc_eval_mgga_f64_cuda(XLA_FFI_CallFrame* call) {
    return jsiesta::cuda::XcEvalMggaCudaHandler(call);
}
}
#endif
