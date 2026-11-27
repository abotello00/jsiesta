/**
 * @file hfx.cpp
 * @brief Screened, threaded construction of the exact-exchange matrix on the sparse pattern and of its derivatives.
 *
 * @par Theory
 * See \c hfx.hpp: lattice sums of four-centre integrals contracted with the density matrix.
 *
 * @par Algorithm
 * A handle holds the shell pairs and the pattern blocks of one geometry; the build and the gradient share the
 * quartet loop and its screening.
 *
 * @par Invariants
 * A handle is read-only after its creation, so builds on different threads are safe.
 *
 * @par SIESTA Mapping
 * None in SIESTA; \c setup_hfx in HONPAS.
 */

#include "jsiesta/hfx.hpp"

#include "jsiesta/eri.hpp"
#include "jsiesta/threading.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

using jsiesta::eri::Kernel;
using jsiesta::eri::ShellPair;

/// Shell pair (I in the home cell, J in cell \c cell) with its Hermite expansion and Schwarz bound.
struct Pair {
    int shell_j = 0;
    int cell[3] = {0, 0, 0};
    double q = 0.0;
    // Gaussian envelope of the pair density, |phi_i phi_j|(r) <= w exp(-gamma |r - centre|^2):
    double charge = 0.0;                   // w (pi / gamma)^{3/2}
    double gamma = 0.0;
    double centre[3] = {0.0, 0.0, 0.0};
    ShellPair sp;
};

/// Shell-pair block of the sparse pattern: first shell in the home cell, second shell in pattern cell \c cell.
struct Block {
    int shell_a = 0;
    int shell_b = 0;
    int cell = 0;
    int n_a = 1;
    int n_b = 1;
    std::int64_t offset = 0;         // into the block-ordered value arrays
    std::vector<std::int64_t> edge;  // (n_a, n_b) edge ids, -1 where the pattern has no edge
};

inline std::int64_t pack_cell(int a, int b, int c) {
    return (static_cast<std::int64_t>(a + 4096) << 26) | (static_cast<std::int64_t>(b + 4096) << 13) |
           static_cast<std::int64_t>(c + 4096);
}

/// The eight orderings (x z | y w) of the centres (a b | c d) of one integral. Each gives a contribution
/// K[x, y] += P[z, w] (x z | y w) with the same integral values; entries are the positions of x, z, y, w.
constexpr int kImage[8][4] = {{0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
                              {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

/// Images of a quartet under the permutations of \c kImage that the loops over (bra pair, block, ket pair)
/// would visit as separate items. The item at hand is image 0.
struct Images {
    int k_block[8];
    int p_block[8];
    bool use[8];
    bool canonical = true;  // image 0 is the first of the distinct valid images in the order of their keys
};

struct Hfx {
    int n_shell = 0;
    int n_cells = 0;
    std::int64_t n_edges = 0;
    Kernel kernel;
    double lattice[9] = {0};  // columns are lattice vectors
    std::vector<int> shell_l;
    std::vector<std::vector<Pair>> pairs;            // per home-cell shell
    std::vector<Block> blocks;
    std::vector<std::vector<int>> blocks_from;       // per first shell: block ids
    std::unordered_map<std::int64_t, int> block_of;  // (shell_a, shell_b, cell) -> block id
    std::unordered_map<std::int64_t, int> cell_of;   // packed integer triple -> pattern cell
    std::vector<int> cell_shift;                     // (n_cells, 3)
    std::int64_t n_block_values = 0;
    std::int64_t n_pairs = 0;
    bool has_envelopes = false;  // screened kernel only: the pairs carry Gaussian envelopes
    bool use_distance = false;   // use them for a bound from the distance between the pair densities
    // Shell data kept for the derivative expansions, which are built on the fly.
    int n_atoms = 0;
    double prim_tol = 0.0;
    std::vector<double> positions;
    std::vector<int> shell_atom;
    std::vector<int> shell_prim_ptr;
    std::vector<double> exps;
    std::vector<double> coefs;

    std::int64_t block_key(int a, int b, int c) const {
        return (static_cast<std::int64_t>(a) * n_shell + b) * n_cells + c;
    }
    /// Block (shell \c s in the home cell, shell \c t in the cell displaced by \c d), or -1.
    int find_block(int s, int t, const int* d) const {
        const auto cit = cell_of.find(pack_cell(d[0], d[1], d[2]));
        if (cit == cell_of.end()) return -1;
        const auto bit = block_of.find(block_key(s, t, cit->second));
        return bit == block_of.end() ? -1 : bit->second;
    }
    void lattice_vector(const int* n, double* t) const {
        for (int d = 0; d < 3; ++d) t[d] = lattice[3 * d] * n[0] + lattice[3 * d + 1] * n[1] + lattice[3 * d + 2] * n[2];
    }
};

constexpr double kPiHfx = 3.14159265358979323846;
constexpr int kRadialTable = 257;

/// |f(r)| of a shell from its table (\c kRadialTable points up to the cut-off radius), zero beyond.
inline double radial_abs(const double* tab, double rcut, double r) {
    if (r >= rcut) return 0.0;
    const double x = r / rcut * (kRadialTable - 1);
    const int n = static_cast<int>(x);
    const double w = x - n;
    return (1.0 - w) * tab[n] + w * tab[n + 1];
}

/// Gaussian envelope of the product of two shells at distance \c dist: the radial functions out to the radii of
/// the tables, the angular parts replaced by their largest value. The product is sampled in cylindrical
/// coordinates about the line of centres (first shell at the origin); the envelope is centred at the centroid of
/// the sampled product, with the exponent among three fractions of \c p_min (the sum of the smallest exponents of
/// the two shells, beyond which no envelope exists) that gives the smallest charge. The height is raised by 30 %
/// for the sampling. Returns the position of the centre along the line, the exponent and the charge.
inline void pair_envelope(const double* tab_a, double rc_a, int la, const double* tab_b, double rc_b, int lb,
                          double dist, double p_min, double* z_centre, double* gamma, double* charge) {
    const double ang = std::sqrt((2.0 * la + 1.0) * (2.0 * lb + 1.0)) / (4.0 * kPiHfx);
    constexpr int nz = 32, nr = 24;
    const double z_lo = std::max(-rc_a, dist - rc_b), z_hi = std::min(rc_a, dist + rc_b);
    const double rho_max = std::min(rc_a, rc_b);
    *z_centre = 0.5 * dist;
    *gamma = 0.5 * p_min;
    *charge = 0.0;
    if (z_hi <= z_lo) return;
    const double dz = (z_hi - z_lo) / nz, drho = rho_max / nr;
    double val[nz][nr];
    double sum = 0.0, sum_z = 0.0;
    for (int iz = 0; iz < nz; ++iz) {
        const double z = z_lo + (iz + 0.5) * dz;
        for (int ir = 0; ir < nr; ++ir) {
            const double rho = (ir + 0.5) * drho;
            const double ra = std::sqrt(rho * rho + z * z), rb = std::sqrt(rho * rho + (z - dist) * (z - dist));
            const double v = ang * radial_abs(tab_a, rc_a, ra) * radial_abs(tab_b, rc_b, rb);
            val[iz][ir] = v;
            sum += rho * v;
            sum_z += rho * v * z;
        }
    }
    if (sum <= 0.0) return;
    const double zc = sum_z / sum;
    *z_centre = zc;
    double best = std::numeric_limits<double>::infinity();
    for (const double frac : {0.25, 0.5, 0.75}) {
        const double g = frac * p_min;
        double w = 0.0;
        for (int iz = 0; iz < nz; ++iz) {
            const double z = z_lo + (iz + 0.5) * dz - zc;
            for (int ir = 0; ir < nr; ++ir) {
                const double rho = (ir + 0.5) * drho;
                w = std::max(w, val[iz][ir] * std::exp(g * (rho * rho + z * z)));
            }
        }
        const double q = 1.3 * w * std::pow(kPiHfx / g, 1.5);
        if (q < best) {
            best = q;
            *gamma = g;
        }
    }
    *charge = best;
}

/// Bound of a short-range integral from the distance between the two pair densities: the interaction of their
/// Gaussian envelopes, \f$q_1 q_2[\mathrm{erf}(\sqrt{\alpha}R) - \mathrm{erf}(\sqrt{\alpha_s}R)]/R\f$ with
/// \f$\alpha = \gamma_1\gamma_2/(\gamma_1+\gamma_2)\f$ and \f$\alpha_s = \alpha\omega^2/(\alpha+\omega^2)\f$.
/// Infinite when the kernel has a full-range part or the bound is switched off.
inline double far_bound(const Hfx& h, const Pair& bra, const Pair& ket, const double* t_g) {
    if (!h.use_distance) return std::numeric_limits<double>::infinity();
    const double dx = ket.centre[0] + t_g[0] - bra.centre[0];
    const double dy = ket.centre[1] + t_g[1] - bra.centre[1];
    const double dz = ket.centre[2] + t_g[2] - bra.centre[2];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double alpha = bra.gamma * ket.gamma / (bra.gamma + ket.gamma);
    const double w2 = h.kernel.omega * h.kernel.omega;
    const double sa = std::sqrt(alpha), ss = std::sqrt(alpha * w2 / (alpha + w2));
    const double g = r > 1e-6 ? (std::erf(sa * r) - std::erf(ss * r)) / r : 1.1283791670955126 * (sa - ss);
    return std::abs(h.kernel.c_sr) * bra.charge * ket.charge * g;
}

/// Finds which of the first \c n_img images of the quartet (shells \c sh, cells \c cc of the four centres) exist
/// as items of the loops, removes duplicates, and decides whether image 0 is the one that computes the integral.
inline void find_images(const Hfx& h, const int* sh, const int (*cc)[3], int b_k0, int b_p0, int n_img, Images& im) {
    int key[8][13];
    im.canonical = true;
    for (int u = 0; u < n_img; ++u) {
        const int x = kImage[u][0], z = kImage[u][1], y = kImage[u][2], w = kImage[u][3];
        int* q = key[u];
        q[0] = sh[x];
        q[1] = sh[z];
        q[5] = sh[y];
        q[9] = sh[w];
        for (int d = 0; d < 3; ++d) {
            q[2 + d] = cc[z][d] - cc[x][d];
            q[6 + d] = cc[y][d] - cc[x][d];
            q[10 + d] = cc[w][d] - cc[x][d];
        }
        if (u == 0) {
            im.k_block[0] = b_k0;
            im.p_block[0] = b_p0;
            im.use[0] = true;
            continue;
        }
        im.use[u] = false;
        int cmp = 0;
        for (int t = 0; t < 13 && cmp == 0; ++t) cmp = (q[t] > key[0][t]) - (q[t] < key[0][t]);
        if (cmp == 0) continue;  // the same item as image 0
        bool dup = false;
        for (int v = 1; v < u && !dup; ++v) {
            if (im.use[v]) dup = std::equal(q, q + 13, key[v]);
        }
        if (dup) continue;
        const int d_p[3] = {q[10] - q[2], q[11] - q[3], q[12] - q[4]};
        im.k_block[u] = h.find_block(sh[x], sh[y], q + 6);
        if (im.k_block[u] < 0) continue;
        im.p_block[u] = h.find_block(sh[z], sh[w], d_p);
        if (im.p_block[u] < 0) continue;
        if (cmp < 0) {
            im.canonical = false;
            return;
        }
        im.use[u] = true;
    }
}

}  // namespace

extern "C" {

void* jsiesta_hfx_create(int32_t n_atoms, const double* positions, const double* cell, int32_t n_shell,
                         const int32_t* shell_atom, const int32_t* shell_l, const double* shell_rcut,
                         const int32_t* shell_orb, const int32_t* shell_prim_ptr, const double* exps,
                         const double* coefs, int32_t n_orb, int64_t n_edges, const int32_t* edge_src,
                         const int32_t* edge_dst, const int32_t* edge_cell, int32_t n_cells,
                         const int32_t* cell_shifts, double c_full, double c_sr, double omega, double prim_tol,
                         double pair_tol) {
    if (n_atoms <= 0 || n_shell <= 0 || n_orb <= 0 || n_cells <= 0) return nullptr;
    for (int s = 0; s < n_shell; ++s) {
        if (shell_l[s] < 0 || shell_l[s] > jsiesta::eri::kMaxL) return nullptr;
    }
    auto* h = new Hfx();
    h->n_shell = n_shell;
    h->n_cells = n_cells;
    h->n_edges = n_edges;
    h->kernel.c_full = c_full;
    h->kernel.c_sr = c_sr;
    h->kernel.omega = omega;
    // Higher Hermite integrals exceed one by a few orders of magnitude; the margin keeps the neglected
    // primitive quartets well below the primitive-pair tolerance.
    h->kernel.prim_cut = 1e-4 * prim_tol;
    std::copy(cell, cell + 9, h->lattice);
    h->shell_l.assign(shell_l, shell_l + n_shell);
    h->n_atoms = n_atoms;
    h->prim_tol = prim_tol;
    h->positions.assign(positions, positions + 3 * static_cast<std::size_t>(n_atoms));
    h->shell_atom.assign(shell_atom, shell_atom + n_shell);
    h->shell_prim_ptr.assign(shell_prim_ptr, shell_prim_ptr + n_shell + 1);
    h->exps.assign(exps, exps + shell_prim_ptr[n_shell]);
    h->coefs.assign(coefs, coefs + shell_prim_ptr[n_shell]);
    h->cell_shift.assign(cell_shifts, cell_shifts + 3 * static_cast<std::size_t>(n_cells));
    for (int c = 0; c < n_cells; ++c) {
        h->cell_of[pack_cell(cell_shifts[3 * c], cell_shifts[3 * c + 1], cell_shifts[3 * c + 2])] = c;
    }

    // Blocks of the pattern.
    std::vector<int> orb_shell(n_orb, -1);
    for (int s = 0; s < n_shell; ++s) {
        for (int m = 0; m < 2 * shell_l[s] + 1; ++m) orb_shell[shell_orb[s] + m] = s;
    }
    h->blocks_from.resize(n_shell);
    for (std::int64_t e = 0; e < n_edges; ++e) {
        const int a = orb_shell[edge_src[e]], b = orb_shell[edge_dst[e]];
        if (a < 0 || b < 0) continue;
        const std::int64_t key = h->block_key(a, b, edge_cell[e]);
        auto it = h->block_of.find(key);
        int id;
        if (it == h->block_of.end()) {
            id = static_cast<int>(h->blocks.size());
            Block blk;
            blk.shell_a = a;
            blk.shell_b = b;
            blk.cell = edge_cell[e];
            blk.n_a = 2 * shell_l[a] + 1;
            blk.n_b = 2 * shell_l[b] + 1;
            blk.offset = h->n_block_values;
            blk.edge.assign(static_cast<std::size_t>(blk.n_a) * blk.n_b, -1);
            h->n_block_values += static_cast<std::int64_t>(blk.n_a) * blk.n_b;
            h->blocks.push_back(std::move(blk));
            h->block_of[key] = id;
            h->blocks_from[a].push_back(id);
        } else {
            id = it->second;
        }
        Block& blk = h->blocks[id];
        blk.edge[(edge_src[e] - shell_orb[a]) * blk.n_b + (edge_dst[e] - shell_orb[b])] = e;
    }

    // Translations to search: all cells within the largest sum of radii of any atom pair.
    double rc_max = 0.0;
    for (int s = 0; s < n_shell; ++s) rc_max = std::max(rc_max, shell_rcut[s]);
    const double* a1 = h->lattice;
    double col[3][3];
    for (int j = 0; j < 3; ++j) {
        for (int d = 0; d < 3; ++d) col[j][d] = a1[3 * d + j];
    }
    auto cross = [](const double* u, const double* v, double* w) {
        w[0] = u[1] * v[2] - u[2] * v[1];
        w[1] = u[2] * v[0] - u[0] * v[2];
        w[2] = u[0] * v[1] - u[1] * v[0];
    };
    double c01[3], c12[3], c20[3];
    cross(col[0], col[1], c01);
    cross(col[1], col[2], c12);
    cross(col[2], col[0], c20);
    const double vol = std::abs(col[0][0] * c12[0] + col[0][1] * c12[1] + col[0][2] * c12[2]);
    auto nrm = [](const double* u) { return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); };
    const double height[3] = {vol / nrm(c12), vol / nrm(c20), vol / nrm(c01)};
    // Atoms need not lie in the home cell: one extra cell covers positions folded by up to a lattice vector.
    int reach[3];
    for (int d = 0; d < 3; ++d) reach[d] = static_cast<int>(std::ceil(2.0 * rc_max / height[d])) + 1;

    // |f(r)| of every shell on a radial table, for the bound of the integral of |phi_i phi_j|.
    h->use_distance = (c_full == 0.0) && (c_sr != 0.0) && (omega > 0.0);
    // |f(r)| of every shell on a radial table for the envelopes of the pair densities. The fits are not confined,
    // so the table runs to the radius beyond which the norm of the fit is below 1e-12, not to the cut-off radius.
    h->has_envelopes = (c_full == 0.0) && (c_sr != 0.0) && (omega > 0.0);
    h->use_distance = h->has_envelopes;
    std::vector<double> radial(static_cast<std::size_t>(n_shell) * kRadialTable, 0.0);
    std::vector<double> r_ext(n_shell, 0.0), exp_min(n_shell, 0.0);
    if (h->has_envelopes) {
        for (int s = 0; s < n_shell; ++s) {
            exp_min[s] = *std::min_element(exps + shell_prim_ptr[s], exps + shell_prim_ptr[s + 1]);
            auto f_abs = [&](double r) {
                double f = 0.0;
                for (int q = shell_prim_ptr[s]; q < shell_prim_ptr[s + 1]; ++q) f += coefs[q] * std::exp(-exps[q] * r * r);
                return std::abs(f * std::pow(r, shell_l[s]));
            };
            const int n_r = 2000;
            const double dr = 4.0 * shell_rcut[s] / n_r;
            double acc = 0.0;
            int n = n_r;
            for (; n > 0; --n) {
                const double r = (n - 0.5) * dr;
                const double f = f_abs(r);
                acc += f * f * r * r * dr;
                if (acc > 1e-12) break;
            }
            r_ext[s] = std::max(shell_rcut[s], n * dr);
            for (int k = 0; k < kRadialTable; ++k) {
                radial[static_cast<std::size_t>(s) * kRadialTable + k] = f_abs(r_ext[s] * k / (kRadialTable - 1));
            }
        }
    }

    h->pairs.resize(n_shell);
    const int n_thr = jsiesta::threading::thread_count(n_shell, 2);
    std::atomic<int> next{0};
    jsiesta::threading::run_threads(n_thr, [&](int) {
        std::vector<double> work, diag;
        const double zero[3] = {0.0, 0.0, 0.0};
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= n_shell) break;
            const double* pi = positions + 3 * shell_atom[i];
            const int npi = shell_prim_ptr[i + 1] - shell_prim_ptr[i];
            for (int n0 = -reach[0]; n0 <= reach[0]; ++n0) {
                for (int n1 = -reach[1]; n1 <= reach[1]; ++n1) {
                    for (int n2 = -reach[2]; n2 <= reach[2]; ++n2) {
                        const int nn[3] = {n0, n1, n2};
                        double t[3];
                        h->lattice_vector(nn, t);
                        for (int j = 0; j < n_shell; ++j) {
                            const double* pj0 = positions + 3 * shell_atom[j];
                            const double pj[3] = {pj0[0] + t[0], pj0[1] + t[1], pj0[2] + t[2]};
                            const double dx = pi[0] - pj[0], dy = pi[1] - pj[1], dz = pi[2] - pj[2];
                            const double rc = shell_rcut[i] + shell_rcut[j];
                            if (dx * dx + dy * dy + dz * dz >= rc * rc) continue;
                            Pair pr;
                            pr.shell_j = j;
                            pr.cell[0] = n0;
                            pr.cell[1] = n1;
                            pr.cell[2] = n2;
                            const int npj = shell_prim_ptr[j + 1] - shell_prim_ptr[j];
                            jsiesta::eri::make_shell_pair(
                                shell_l[i], npi, exps + shell_prim_ptr[i], coefs + shell_prim_ptr[i], pi,
                                shell_l[j], npj, exps + shell_prim_ptr[j], coefs + shell_prim_ptr[j], pj,
                                prim_tol, pr.sp);
                            if (pr.sp.n_prim == 0) continue;
                            const int n_ab = pr.sp.n_ab;
                            diag.resize(static_cast<std::size_t>(n_ab) * n_ab);
                            jsiesta::eri::eri_quartet(pr.sp, pr.sp, zero, h->kernel, diag.data(), work);
                            double qmax = 0.0;
                            for (int ab = 0; ab < n_ab; ++ab) qmax = std::max(qmax, std::abs(diag[ab * n_ab + ab]));
                            pr.q = std::sqrt(qmax);
                            if (pr.q < pair_tol) continue;
                            if (h->has_envelopes) {
                                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                                double zc = 0.0;
                                pair_envelope(radial.data() + static_cast<std::size_t>(i) * kRadialTable, r_ext[i],
                                              shell_l[i], radial.data() + static_cast<std::size_t>(j) * kRadialTable,
                                              r_ext[j], shell_l[j], dist, exp_min[i] + exp_min[j], &zc, &pr.gamma,
                                              &pr.charge);
                                const double f = dist > 1e-12 ? zc / dist : 0.0;
                                for (int c = 0; c < 3; ++c) pr.centre[c] = pi[c] + f * (pj[c] - pi[c]);
                            }
                            h->pairs[i].push_back(std::move(pr));
                        }
                    }
                }
            }
        }
    });
    for (const auto& v : h->pairs) h->n_pairs += static_cast<std::int64_t>(v.size());
    return h;
}

int32_t jsiesta_hfx_build(void* handle, int32_t n_spin, const double* dm, double screen_tol, double* k_out,
                          int64_t* stats) {
    if (handle == nullptr || n_spin < 1) return 2;
    const Hfx& h = *static_cast<const Hfx*>(handle);
    const std::int64_t nbv = h.n_block_values;
    const int n_blocks = static_cast<int>(h.blocks.size());

    // Density matrix in block order and its largest element per block.
    std::vector<double> p_blk(static_cast<std::size_t>(n_spin) * nbv, 0.0);
    std::vector<double> p_max(n_blocks, 0.0);
    for (int b = 0; b < n_blocks; ++b) {
        const Block& blk = h.blocks[b];
        const int nv = blk.n_a * blk.n_b;
        for (int s = 0; s < n_spin; ++s) {
            double* dst = p_blk.data() + static_cast<std::size_t>(s) * nbv + blk.offset;
            for (int q = 0; q < nv; ++q) {
                const std::int64_t e = blk.edge[q];
                const double v = (e >= 0) ? dm[static_cast<std::size_t>(s) * h.n_edges + e] : 0.0;
                dst[q] = v;
                p_max[b] = std::max(p_max[b], std::abs(v));
            }
        }
    }

    const double p_max_all = p_max.empty() ? 0.0 : *std::max_element(p_max.begin(), p_max.end());

    // Work items: (first shell, bra pair).
    std::vector<std::pair<int, int>> items;
    items.reserve(static_cast<std::size_t>(h.n_pairs));
    for (int i = 0; i < h.n_shell; ++i) {
        for (int p = 0; p < static_cast<int>(h.pairs[i].size()); ++p) items.emplace_back(i, p);
    }
    const std::int64_t n_items = static_cast<std::int64_t>(items.size());
    const int n_thr = jsiesta::threading::thread_count(n_items, 4);
    std::vector<std::vector<double>> k_thr(n_thr);
    std::vector<std::int64_t> n_done(n_thr, 0), n_skip(n_thr, 0);
    std::atomic<std::int64_t> next{0};

    jsiesta::threading::run_threads(n_thr, [&](int tid) {
        std::vector<double>& k_blk = k_thr[tid];
        k_blk.assign(static_cast<std::size_t>(n_spin) * nbv, 0.0);
        std::vector<double> work, buf;
        Images im;
        for (;;) {
            const std::int64_t it = next.fetch_add(1);
            if (it >= n_items) break;
            const int i = items[it].first;
            const Pair& bra = h.pairs[i][items[it].second];
            const int j = bra.shell_j;
            const int n_i = 2 * h.shell_l[i] + 1, n_j = 2 * h.shell_l[j] + 1;
            for (const int b_ik : h.blocks_from[i]) {
                const Block& blk = h.blocks[b_ik];
                const int k = blk.shell_b;
                const int* g = h.cell_shift.data() + 3 * blk.cell;
                double t_g[3];
                h.lattice_vector(g, t_g);
                const int n_k = blk.n_b;
                for (const Pair& ket : h.pairs[k]) {
                    // sigma sits in cell H = G + M; the density-matrix block is (J, L, H - N).
                    const auto cit = h.cell_of.find(pack_cell(g[0] + ket.cell[0] - bra.cell[0],
                                                              g[1] + ket.cell[1] - bra.cell[1],
                                                              g[2] + ket.cell[2] - bra.cell[2]));
                    if (cit == h.cell_of.end()) continue;
                    const int l = ket.shell_j;
                    const auto bit = h.block_of.find(h.block_key(j, l, cit->second));
                    if (bit == h.block_of.end()) continue;
                    const int b_jl = bit->second;
                    const double qq = std::min(bra.q * ket.q, far_bound(h, bra, ket, t_g));
                    if (qq * p_max_all < screen_tol) {
                        ++n_skip[tid];
                        continue;
                    }
                    // The integral is shared by up to eight items of these loops; the first in key order computes it.
                    const int sh[4] = {i, j, k, l};
                    const int cc[4][3] = {{0, 0, 0},
                                          {bra.cell[0], bra.cell[1], bra.cell[2]},
                                          {g[0], g[1], g[2]},
                                          {g[0] + ket.cell[0], g[1] + ket.cell[1], g[2] + ket.cell[2]}};
                    find_images(h, sh, cc, b_ik, b_jl, 8, im);
                    if (!im.canonical) continue;
                    double pm = 0.0;
                    for (int u = 0; u < 8; ++u) {
                        if (im.use[u]) pm = std::max(pm, p_max[im.p_block[u]]);
                    }
                    if (qq * pm < screen_tol) {
                        ++n_skip[tid];
                        continue;
                    }
                    ++n_done[tid];
                    const int n_l = 2 * h.shell_l[l] + 1;
                    const int nn[4] = {n_i, n_j, n_k, n_l};
                    buf.resize(static_cast<std::size_t>(n_i) * n_j * n_k * n_l);
                    jsiesta::eri::eri_quartet(bra.sp, ket.sp, t_g, h.kernel, buf.data(), work);
                    for (int u = 0; u < 8; ++u) {
                        if (!im.use[u]) continue;
                        const int x = kImage[u][0], z = kImage[u][1], y = kImage[u][2], w = kImage[u][3];
                        const int n_y = nn[y], n_w = nn[w];
                        const std::int64_t off_k = h.blocks[im.k_block[u]].offset;
                        const std::int64_t off_p = h.blocks[im.p_block[u]].offset;
                        for (int s = 0; s < n_spin; ++s) {
                            const double* p = p_blk.data() + static_cast<std::size_t>(s) * nbv + off_p;
                            double* kk = k_blk.data() + static_cast<std::size_t>(s) * nbv + off_k;
                            const double* val = buf.data();
                            int idx[4];
                            for (idx[0] = 0; idx[0] < n_i; ++idx[0]) {
                                for (idx[1] = 0; idx[1] < n_j; ++idx[1]) {
                                    for (idx[2] = 0; idx[2] < n_k; ++idx[2]) {
                                        for (idx[3] = 0; idx[3] < n_l; ++idx[3]) {
                                            kk[idx[x] * n_y + idx[y]] += p[idx[z] * n_w + idx[w]] * (*val++);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    std::fill(k_out, k_out + static_cast<std::size_t>(n_spin) * h.n_edges, 0.0);
    std::int64_t done = 0, skipped = 0;
    for (int t = 0; t < n_thr; ++t) {
        done += n_done[t];
        skipped += n_skip[t];
        if (k_thr[t].empty()) continue;
        for (int b = 0; b < n_blocks; ++b) {
            const Block& blk = h.blocks[b];
            const int nv = blk.n_a * blk.n_b;
            for (int s = 0; s < n_spin; ++s) {
                const double* src = k_thr[t].data() + static_cast<std::size_t>(s) * nbv + blk.offset;
                for (int q = 0; q < nv; ++q) {
                    if (blk.edge[q] >= 0) k_out[static_cast<std::size_t>(s) * h.n_edges + blk.edge[q]] += src[q];
                }
            }
        }
    }
    if (stats != nullptr) {
        stats[0] = done;
        stats[1] = skipped;
        stats[2] = h.n_pairs;
        stats[3] = n_thr;
    }
    return 0;
}

int32_t jsiesta_hfx_gradient(void* handle, int32_t n_spin, const double* dm, double screen_tol, double* d_first,
                             double* virial_first, double* virial_pair, int64_t* stats) {
    if (handle == nullptr || n_spin < 1) return 2;
    const Hfx& h = *static_cast<const Hfx*>(handle);
    const std::int64_t nbv = h.n_block_values;
    const int n_blocks = static_cast<int>(h.blocks.size());

    std::vector<double> p_blk(static_cast<std::size_t>(n_spin) * nbv, 0.0);
    std::vector<double> p_max(n_blocks, 0.0);
    for (int b = 0; b < n_blocks; ++b) {
        const Block& blk = h.blocks[b];
        const int nv = blk.n_a * blk.n_b;
        for (int s = 0; s < n_spin; ++s) {
            double* dst = p_blk.data() + static_cast<std::size_t>(s) * nbv + blk.offset;
            for (int q = 0; q < nv; ++q) {
                const std::int64_t e = blk.edge[q];
                const double v = (e >= 0) ? dm[static_cast<std::size_t>(s) * h.n_edges + e] : 0.0;
                dst[q] = v;
                p_max[b] = std::max(p_max[b], std::abs(v));
            }
        }
    }

    const double p_max_all = p_max.empty() ? 0.0 : *std::max_element(p_max.begin(), p_max.end());

    std::vector<std::pair<int, int>> items;
    items.reserve(static_cast<std::size_t>(h.n_pairs));
    for (int i = 0; i < h.n_shell; ++i) {
        for (int p = 0; p < static_cast<int>(h.pairs[i].size()); ++p) items.emplace_back(i, p);
    }
    const std::int64_t n_items = static_cast<std::int64_t>(items.size());
    const int n_thr = jsiesta::threading::thread_count(n_items, 4);
    const std::size_t n_acc = 3 * static_cast<std::size_t>(h.n_atoms) + 18;  // d_first, virial_first, virial_pair
    std::vector<std::vector<double>> acc_thr(n_thr, std::vector<double>(n_acc, 0.0));
    std::vector<std::int64_t> n_done(n_thr, 0), n_skip(n_thr, 0);
    std::atomic<std::int64_t> next{0};

    jsiesta::threading::run_threads(n_thr, [&](int tid) {
        double* acc = acc_thr[tid].data();
        double* vir_first = acc + 3 * static_cast<std::size_t>(h.n_atoms);
        double* vir_pair = vir_first + 9;
        std::vector<double> work, buf, w_pp;
        ShellPair deriv[6];
        Images im;
        for (;;) {
            const std::int64_t it = next.fetch_add(1);
            if (it >= n_items) break;
            const int i = items[it].first;
            const Pair& bra = h.pairs[i][items[it].second];
            const int j = bra.shell_j;
            const int n_i = 2 * h.shell_l[i] + 1, n_j = 2 * h.shell_l[j] + 1;
            const double* r_a = h.positions.data() + 3 * h.shell_atom[i];
            double t_n[3];
            h.lattice_vector(bra.cell, t_n);
            const double* r_b0 = h.positions.data() + 3 * h.shell_atom[j];
            const double r_b[3] = {r_b0[0] + t_n[0], r_b0[1] + t_n[1], r_b0[2] + t_n[2]};
            bool have_deriv = false;
            for (const int b_ik : h.blocks_from[i]) {
                const Block& blk = h.blocks[b_ik];
                const int k = blk.shell_b;
                const int* g = h.cell_shift.data() + 3 * blk.cell;
                double t_g[3];
                h.lattice_vector(g, t_g);
                const int n_k = blk.n_b;
                const double* r_c0 = h.positions.data() + 3 * h.shell_atom[k];
                for (const Pair& ket : h.pairs[k]) {
                    const auto cit = h.cell_of.find(pack_cell(g[0] + ket.cell[0] - bra.cell[0],
                                                              g[1] + ket.cell[1] - bra.cell[1],
                                                              g[2] + ket.cell[2] - bra.cell[2]));
                    if (cit == h.cell_of.end()) continue;
                    const int l = ket.shell_j;
                    const auto bit = h.block_of.find(h.block_key(j, l, cit->second));
                    if (bit == h.block_of.end()) continue;
                    const int b_jl = bit->second;
                    const double qq = std::min(bra.q * ket.q, far_bound(h, bra, ket, t_g));
                    if (qq * p_max_all * p_max_all < screen_tol) {
                        ++n_skip[tid];
                        continue;
                    }
                    // The derivatives with respect to the first centre and to the translation of the bra pair give
                    // those with respect to the second centre, so the items that differ by the order within the
                    // bra pair or within the ket pair share one set of derivative integrals.
                    const int sh[4] = {i, j, k, l};
                    const int cc[4][3] = {{0, 0, 0},
                                          {bra.cell[0], bra.cell[1], bra.cell[2]},
                                          {g[0], g[1], g[2]},
                                          {g[0] + ket.cell[0], g[1] + ket.cell[1], g[2] + ket.cell[2]}};
                    find_images(h, sh, cc, b_ik, b_jl, 4, im);
                    if (!im.canonical) continue;
                    double pm = 0.0;
                    for (int u = 0; u < 4; ++u) {
                        if (im.use[u]) pm = std::max(pm, p_max[im.k_block[u]] * p_max[im.p_block[u]]);
                    }
                    if (qq * pm < screen_tol) {
                        ++n_skip[tid];
                        continue;
                    }
                    ++n_done[tid];
                    if (!have_deriv) {
                        // Derivative expansions of the bra pair: first centre (x, y, z), pair translation (x, y, z).
                        for (int m = 0; m < 6; ++m) {
                            jsiesta::eri::make_shell_pair_mode(
                                h.shell_l[i], h.shell_prim_ptr[i + 1] - h.shell_prim_ptr[i],
                                h.exps.data() + h.shell_prim_ptr[i], h.coefs.data() + h.shell_prim_ptr[i], r_a,
                                h.shell_l[j], h.shell_prim_ptr[j + 1] - h.shell_prim_ptr[j],
                                h.exps.data() + h.shell_prim_ptr[j], h.coefs.data() + h.shell_prim_ptr[j], r_b,
                                h.prim_tol, static_cast<jsiesta::eri::PairMode>(m + 1), deriv[m]);
                        }
                        have_deriv = true;
                    }
                    const int n_l = 2 * h.shell_l[l] + 1;
                    const int nn[4] = {n_i, n_j, n_k, n_l};
                    const std::size_t n_q = static_cast<std::size_t>(n_i) * n_j * n_k * n_l;
                    buf.resize(6 * n_q);
                    jsiesta::eri::eri_quartet_multi(deriv, 6, ket.sp, t_g, h.kernel, buf.data(), work);
                    const double* r_d0 = h.positions.data() + 3 * h.shell_atom[l];
                    double t_h[3];
                    h.lattice_vector(cc[3], t_h);
                    const double r_c[3] = {r_c0[0] + t_g[0], r_c0[1] + t_g[1], r_c0[2] + t_g[2]};
                    const double r_d[3] = {r_d0[0] + t_h[0], r_d0[1] + t_h[1], r_d0[2] + t_h[2]};
                    const double* pos[4] = {r_a, r_b, r_c, r_d};
                    for (int u = 0; u < 4; ++u) {
                        if (!im.use[u]) continue;
                        const int x = kImage[u][0], z = kImage[u][1], y = kImage[u][2], w = kImage[u][3];
                        const int n_y = nn[y], n_w = nn[w];
                        // w[a, b, c, d] = sum_s P^s[x, y] P^s[z, w] in the index order of the integral
                        w_pp.assign(n_q, 0.0);
                        for (int s = 0; s < n_spin; ++s) {
                            const double* p_xy =
                                p_blk.data() + static_cast<std::size_t>(s) * nbv + h.blocks[im.k_block[u]].offset;
                            const double* p_zw =
                                p_blk.data() + static_cast<std::size_t>(s) * nbv + h.blocks[im.p_block[u]].offset;
                            double* out = w_pp.data();
                            int idx[4];
                            for (idx[0] = 0; idx[0] < n_i; ++idx[0]) {
                                for (idx[1] = 0; idx[1] < n_j; ++idx[1]) {
                                    for (idx[2] = 0; idx[2] < n_k; ++idx[2]) {
                                        for (idx[3] = 0; idx[3] < n_l; ++idx[3]) {
                                            *out++ += p_xy[idx[x] * n_y + idx[y]] * p_zw[idx[z] * n_w + idx[w]];
                                        }
                                    }
                                }
                            }
                        }
                        double grad[6];
                        for (int m = 0; m < 6; ++m) {
                            const double* val = buf.data() + m * n_q;
                            double sum = 0.0;
                            for (std::size_t q = 0; q < n_q; ++q) sum += w_pp[q] * val[q];
                            grad[m] = sum;
                        }
                        // Derivative with respect to the first centre of this item: a, or b = pair - a.
                        const double g_x[3] = {x == 0 ? grad[0] : grad[3] - grad[0], x == 0 ? grad[1] : grad[4] - grad[1],
                                               x == 0 ? grad[2] : grad[5] - grad[2]};
                        double* f = acc + 3 * static_cast<std::size_t>(h.shell_atom[sh[x]]);
                        for (int a = 0; a < 3; ++a) {
                            f[a] += g_x[a];
                            for (int b = 0; b < 3; ++b) {
                                vir_first[3 * a + b] += g_x[a] * (pos[x][b] - pos[z][b]);
                                vir_pair[3 * a + b] += grad[3 + a] * (pos[y][b] - pos[x][b]);
                            }
                        }
                    }
                }
            }
        }
    });

    std::fill(d_first, d_first + 3 * static_cast<std::size_t>(h.n_atoms), 0.0);
    std::fill(virial_first, virial_first + 9, 0.0);
    std::fill(virial_pair, virial_pair + 9, 0.0);
    std::int64_t done = 0, skipped = 0;
    for (int t = 0; t < n_thr; ++t) {
        done += n_done[t];
        skipped += n_skip[t];
        const double* acc = acc_thr[t].data();
        for (std::size_t q = 0; q < 3 * static_cast<std::size_t>(h.n_atoms); ++q) d_first[q] += acc[q];
        for (int q = 0; q < 9; ++q) {
            virial_first[q] += acc[3 * static_cast<std::size_t>(h.n_atoms) + q];
            virial_pair[q] += acc[3 * static_cast<std::size_t>(h.n_atoms) + 9 + q];
        }
    }
    if (stats != nullptr) {
        stats[0] = done;
        stats[1] = skipped;
        stats[2] = h.n_pairs;
        stats[3] = n_thr;
    }
    return 0;
}

void jsiesta_hfx_set_distance_screening(void* handle, int32_t on) {
    if (handle == nullptr) return;
    auto* h = static_cast<Hfx*>(handle);
    h->use_distance = h->has_envelopes && on != 0;
}

void jsiesta_hfx_destroy(void* handle) {
    delete static_cast<Hfx*>(handle);
}

}  // extern "C"
