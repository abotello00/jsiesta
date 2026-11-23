/**
 * @file kb_three_center.cpp
 * @brief Three-centre Kleinman-Bylander terms on the sparse pattern: \f$V_{KB}\f$ assembly, energy, forces and stress.
 *
 * @par Theory
 * With projector overlaps \f$P_{\mu,km}(\mathbf{t}) = \langle \phi_\mu(\mathbf{r} - \mathbf{R}_\mu) | \chi_{km}(\mathbf{r} -
 * \mathbf{R}_k - A\mathbf{t}) \rangle\f$ and KB energies \f$E_k\f$, the non-local matrix is
 * \f$V_{\mu\nu}(\mathbf{T}) = \sum_{k,\mathbf{t}} E_k \sum_m P_{\mu,km}(\mathbf{t}) P_{\nu,km}(\mathbf{t} - \mathbf{T})\f$
 * and the energy \f$E_{KB} = \sum_{\mu\nu\mathbf{T}} D_{\mu\nu}(\mathbf{T}) V_{\mu\nu}(\mathbf{T})\f$. Its derivative with
 * respect to the position of \f$\mu\f$'s atom is \f$2 E_k \sum_m \nabla P_{\mu,km} W_{\mu,km}\f$ with
 * \f$W_{\mu,km}(\mathbf{t}) = \sum_{\nu\mathbf{t}'} D_{\mu\nu}(\mathbf{t} - \mathbf{t}') P_{\nu,km}(\mathbf{t}')\f$; the
 * projector atom takes the opposite force, and the stress is the force times the bond vector
 * \f$\mathbf{R}_k + A\mathbf{t} - \mathbf{R}_\mu\f$.
 *
 * @par Algorithm
 * Links \f$(\mu, k, \mathbf{t}, m)\f$ arrive sorted by \f$(k, \mu, \mathbf{t}, m)\f$; consecutive links with the same
 * \f$(k, \mu, \mathbf{t})\f$ form a group of \f$2l+1\f$ values. Rows \f$\mu\f$ are split into contiguous ranges, one per
 * thread; for every group of a row, every group of the same shell is a partner, and the pattern edge
 * \f$(\mu, \nu, \mathbf{t} - \mathbf{t}')\f$ is found by binary search in the row's sorted \f$(\nu, \mathbf{T})\f$ keys. A
 * thread writes only the \f$V_{KB}\f$ entries of its own rows and keeps private force, stress and energy sums, which are
 * added in thread order, so the result does not depend on scheduling.
 *
 * @par Invariants
 * - The same pairs as the NumPy reference (\c _iter_kb_shell_pairs): groups whose size differs from \f$2l+1\f$ are
 *   skipped, pairs without a pattern edge contribute nothing.
 * - Spin-orbit shells (nspin 8) are not handled here; the caller keeps the NumPy path for them.
 *
 * @par SIESTA Mapping
 * \c nlefsm.f (matrix elements, energy and forces of the KB projectors) as called from \c setup_H0 and
 * \c final_H_f_stress.
 */

#include "jsiesta/threading.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr int kShiftBits = 11;
constexpr int64_t kShiftHalf = 1 << (kShiftBits - 1);  // |T| < 1024

inline int64_t pack_key(int64_t nu, int64_t t0, int64_t t1, int64_t t2) {
    return (nu << (3 * kShiftBits)) | ((t0 + kShiftHalf) << (2 * kShiftBits)) | ((t1 + kShiftHalf) << kShiftBits) |
           (t2 + kShiftHalf);
}

int resolve_threads(int64_t work) { return jsiesta::threading::thread_count(work, 2048); }

using jsiesta::threading::parallel_ranges;

}  // namespace

extern "C" {

/**
 * @brief Three-centre KB terms on the sparse pattern: optional \f$V_{KB}\f$ assembly and optional energy, forces and
 *        stress at a given density matrix.
 *
 * @par Theory
 * See the file description: \f$V_{\mu\nu}(\mathbf{T}) = \sum E_k P P\f$, \f$E_{KB} = \sum D V\f$,
 * \f$\mathbf{F} = -\partial E_{KB} / \partial \mathbf{R}\f$ through the projector overlaps.
 *
 * @par Algorithm
 * Groups the sorted links, sorts each pattern row's \f$(\nu, \mathbf{T})\f$ keys, and loops rows in parallel as in the
 * file description.
 *
 * @par Invariants
 * Deterministic for a fixed thread count; returns 1 on an inconsistent input (shift or index out of range) and writes
 * nothing then.
 *
 * @par SIESTA Mapping
 * \c nlefsm.f.
 *
 * @param n_links Number of links.
 * @param link_mu Orbital of each link (sorted with the others by shell, orbital, shift, m).
 * @param link_k KB shell instance of each link.
 * @param link_t Cell shift of the projector, (n_links, 3).
 * @param p Projector overlaps, (n_links,).
 * @param dp Gradients of the overlaps with respect to the orbital's atom position, (n_links, 3); may be null when no
 *        forces are requested.
 * @param n_shells Number of KB shell instances.
 * @param shell_atom Atom of each shell instance.
 * @param shell_ek Effective KB energy of each shell instance (Ha).
 * @param shell_dim Number of m components (2l+1) of each shell instance.
 * @param n_orb Number of orbitals in the unit cell.
 * @param orb_atom Atom of each orbital.
 * @param n_atoms Number of atoms.
 * @param pos Atomic positions, (n_atoms, 3), Bohr.
 * @param cell Lattice vectors as columns, row-major (3, 3), Bohr.
 * @param n_edges Number of pattern edges.
 * @param row_ptr CSR row pointer of the pattern, (n_orb + 1).
 * @param edge_dst Unit-cell orbital of each edge's column.
 * @param edge_t Cell shift of each edge, (n_edges, 3).
 * @param dm Total density matrix on the edges, (n_edges,); null for assembly only.
 * @param v_out Output \f$V_{KB}\f$ on the edges, (n_edges,), accumulated into; null to skip assembly.
 * @param e_out Output KB energy (Ha); ignored when \c dm is null.
 * @param f_out Output forces, (n_atoms, 3), Ha/Bohr, overwritten; ignored when \c dm or \c dp is null.
 * @param s_out Output \f$\sum \mathbf{g} \otimes \mathbf{d}\f$, row-major (3, 3), not divided by the volume.
 * @return 0 on success, 1 on invalid input.
 */
int32_t jsiesta_kb_three_center(int64_t n_links, const int32_t* link_mu, const int32_t* link_k,
                                const int32_t* link_t, const double* p, const double* dp, int32_t n_shells,
                                const int32_t* shell_atom, const double* shell_ek, const int32_t* shell_dim,
                                int32_t n_orb, const int32_t* orb_atom, int32_t n_atoms, const double* pos,
                                const double* cell, int64_t n_edges, const int64_t* row_ptr, const int32_t* edge_dst,
                                const int32_t* edge_t, const double* dm, double* v_out, double* e_out, double* f_out,
                                double* s_out) {
    const bool want_forces = (dm != nullptr && dp != nullptr && f_out != nullptr && s_out != nullptr);
    if (n_links < 0 || n_shells < 0 || n_orb < 0 || n_atoms < 0 || n_edges < 0) return 1;
    for (int64_t e = 0; e < 3 * n_edges; ++e) {
        if (edge_t[e] <= -kShiftHalf / 2 || edge_t[e] >= kShiftHalf / 2) return 1;
    }
    for (int64_t i = 0; i < n_links; ++i) {
        if (link_mu[i] < 0 || link_mu[i] >= n_orb || link_k[i] < 0 || link_k[i] >= n_shells) return 1;
        for (int c = 0; c < 3; ++c) {
            if (link_t[3 * i + c] <= -kShiftHalf / 2 || link_t[3 * i + c] >= kShiftHalf / 2) return 1;
        }
        if (i > 0 && link_k[i] < link_k[i - 1]) return 1;  // must be sorted by shell
    }

    // Groups of consecutive links with the same (k, mu, t).
    std::vector<int64_t> g_start;
    g_start.reserve(static_cast<size_t>(n_links / 3 + 1));
    for (int64_t i = 0; i < n_links; ++i) {
        const bool same = i > 0 && link_k[i] == link_k[i - 1] && link_mu[i] == link_mu[i - 1] &&
                          link_t[3 * i] == link_t[3 * i - 3] && link_t[3 * i + 1] == link_t[3 * i - 2] &&
                          link_t[3 * i + 2] == link_t[3 * i - 1];
        if (!same) g_start.push_back(i);
    }
    const int64_t n_groups_all = static_cast<int64_t>(g_start.size());
    g_start.push_back(n_links);
    // Keep groups with exactly 2l+1 components (as the NumPy reference).
    std::vector<int64_t> grp;  // indices into g_start
    grp.reserve(static_cast<size_t>(n_groups_all));
    for (int64_t g = 0; g < n_groups_all; ++g) {
        const int32_t k = link_k[g_start[g]];
        if (g_start[g + 1] - g_start[g] == shell_dim[k]) grp.push_back(g);
    }
    const int64_t n_groups = static_cast<int64_t>(grp.size());
    // Shell ranges (groups are sorted by shell) and the groups of each row.
    std::vector<int64_t> shell_lo(static_cast<size_t>(n_shells) + 1, n_groups);
    for (int64_t gi = n_groups - 1; gi >= 0; --gi) shell_lo[link_k[g_start[grp[gi]]]] = gi;
    for (int32_t k = n_shells - 1; k >= 0; --k) shell_lo[k] = std::min(shell_lo[k], shell_lo[k + 1]);
    std::vector<int64_t> row_cnt(static_cast<size_t>(n_orb) + 1, 0);
    for (int64_t gi = 0; gi < n_groups; ++gi) ++row_cnt[link_mu[g_start[grp[gi]]] + 1];
    for (int32_t mu = 0; mu < n_orb; ++mu) row_cnt[mu + 1] += row_cnt[mu];
    std::vector<int64_t> row_groups(static_cast<size_t>(n_groups));
    {
        std::vector<int64_t> fill(row_cnt.begin(), row_cnt.end() - 1);
        for (int64_t gi = 0; gi < n_groups; ++gi) row_groups[fill[link_mu[g_start[grp[gi]]]]++] = gi;
    }

    // Sorted (nu, T) keys per pattern row.
    std::vector<int64_t> keys(static_cast<size_t>(n_edges));
    std::vector<int32_t> keyed_edge(static_cast<size_t>(n_edges));
    const int n_threads = resolve_threads(n_edges + n_links);
    parallel_ranges(n_orb, n_threads, [&](int, int64_t lo, int64_t hi) {
        std::vector<std::pair<int64_t, int32_t>> tmp;
        for (int64_t mu = lo; mu < hi; ++mu) {
            const int64_t a = row_ptr[mu], b = row_ptr[mu + 1];
            tmp.clear();
            for (int64_t e = a; e < b; ++e) {
                tmp.emplace_back(pack_key(edge_dst[e], edge_t[3 * e], edge_t[3 * e + 1], edge_t[3 * e + 2]),
                                 static_cast<int32_t>(e));
            }
            std::sort(tmp.begin(), tmp.end());
            for (int64_t e = a; e < b; ++e) {
                keys[e] = tmp[e - a].first;
                keyed_edge[e] = tmp[e - a].second;
            }
        }
    });

    std::vector<double> f_t(want_forces ? static_cast<size_t>(n_threads) * 3 * n_atoms : 0, 0.0);
    std::vector<double> s_t(static_cast<size_t>(n_threads) * 9, 0.0);
    std::vector<double> e_t(static_cast<size_t>(n_threads), 0.0);
    parallel_ranges(n_orb, n_threads, [&](int t, int64_t lo, int64_t hi) {
        double* ft = want_forces ? f_t.data() + static_cast<size_t>(t) * 3 * n_atoms : nullptr;
        double* st = s_t.data() + 9 * t;
        double e_acc = 0.0;
        double w[16];
        for (int64_t mu = lo; mu < hi; ++mu) {
            const int64_t ra = row_ptr[mu], rb = row_ptr[mu + 1];
            const int64_t* kb = keys.data() + ra;
            const int64_t* ke = keys.data() + rb;
            for (int64_t q = row_cnt[mu]; q < row_cnt[mu + 1]; ++q) {
                const int64_t gr = row_groups[q];
                const int64_t ir = g_start[grp[gr]];
                const int32_t k = link_k[ir];
                const int dim = shell_dim[k];
                if (dim > 16) continue;
                const double ek = shell_ek[k];
                const int32_t tr0 = link_t[3 * ir], tr1 = link_t[3 * ir + 1], tr2 = link_t[3 * ir + 2];
                for (int m = 0; m < dim; ++m) w[m] = 0.0;
                for (int64_t gc = shell_lo[k]; gc < shell_lo[k + 1]; ++gc) {
                    const int64_t ic = g_start[grp[gc]];
                    const int64_t key = pack_key(link_mu[ic], tr0 - link_t[3 * ic], tr1 - link_t[3 * ic + 1],
                                                 tr2 - link_t[3 * ic + 2]);
                    const int64_t* it = std::lower_bound(kb, ke, key);
                    if (it == ke || *it != key) continue;
                    const int64_t e = keyed_edge[ra + (it - kb)];
                    if (v_out != nullptr) {
                        double pp = 0.0;
                        for (int m = 0; m < dim; ++m) pp += p[ir + m] * p[ic + m];
                        v_out[e] += ek * pp;
                    }
                    if (dm != nullptr) {
                        const double d = dm[e];
                        for (int m = 0; m < dim; ++m) w[m] += d * p[ic + m];
                    }
                }
                if (dm == nullptr) continue;
                for (int m = 0; m < dim; ++m) e_acc += ek * p[ir + m] * w[m];
                if (!want_forces) continue;
                double g[3] = {0.0, 0.0, 0.0};
                for (int m = 0; m < dim; ++m) {
                    for (int c = 0; c < 3; ++c) g[c] += 2.0 * ek * dp[3 * (ir + m) + c] * w[m];
                }
                const int32_t ia = orb_atom[mu], ka = shell_atom[k];
                double dvec[3];
                for (int c = 0; c < 3; ++c) {
                    dvec[c] = pos[3 * ka + c] + cell[3 * c] * tr0 + cell[3 * c + 1] * tr1 + cell[3 * c + 2] * tr2 -
                              pos[3 * ia + c];
                    ft[3 * ia + c] += g[c];
                    ft[3 * ka + c] -= g[c];
                }
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) st[3 * i + j] += g[i] * dvec[j];
                }
            }
        }
        e_t[t] = e_acc;
    });

    if (dm != nullptr && e_out != nullptr) {
        double e_sum = 0.0;
        for (int t = 0; t < n_threads; ++t) e_sum += e_t[t];
        *e_out = e_sum;
    }
    if (want_forces) {
        for (int64_t i = 0; i < 3 * static_cast<int64_t>(n_atoms); ++i) f_out[i] = 0.0;
        for (int i = 0; i < 9; ++i) s_out[i] = 0.0;
        for (int t = 0; t < n_threads; ++t) {
            for (int64_t i = 0; i < 3 * static_cast<int64_t>(n_atoms); ++i) f_out[i] += f_t[static_cast<size_t>(t) * 3 * n_atoms + i];
            for (int i = 0; i < 9; ++i) s_out[i] += s_t[9 * t + i];
        }
    }
    return 0;
}

}  // extern "C"
