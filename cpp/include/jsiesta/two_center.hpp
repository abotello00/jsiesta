/**
 * @file two_center.hpp
 * @brief C++ Fourier-Bessel two-center integral engine (\c S, \c T, \c Vkb, \c E_NA_pair).
 *
 * @par Theory
 * Evaluates two-center overlap \f$S_{\mu\nu}\f$, kinetic \f$T_{\mu\nu}\f$, and
 * orbital-to-KB-projector overlaps \f$P_{\mu k}\f$ via spherical Bessel convolution:
 * \f[
 *   I_{l_1 m_1, l_2 m_2}(\mathbf{R}) = 8 \sum_{L=|l_1-l_2|}^{l_1+l_2} (-1)^{(l_2-l_1+L)/2}
 *   \left(\sum_{M=-L}^{L} G^{LM}_{l_1 m_1, l_2 m_2} C_{LM}(\mathbf{R})\right)
 *   \frac{F_{l_1 l_2 L}(R)}{R^L},
 * \f]
 * plus exact short-range neutral-atom pair energies \f$E_{\text{NA-pair}}(R)\f$
 * and analytical pair displacement derivatives \f$\nabla_{\mathbf{R}} I\f$.
 *
 * @par Algorithm
 * Precomputes real Gaunt coefficients on a Gauss-Legendre x trapezoidal spherical
 * quadrature grid, transforms radial orbitals/projectors to \f$k\f$-space via
 * Simpson/trapezoidal quadrature, tabulates \f$F_{l_1 l_2 L}(R)/R^L\f$ on a dense
 * uniform radial grid with natural cubic splines, and evaluates sparse edge pairs.
 *
 * @par Invariants
 * - Exact Hermiticity across \f$(\mu,\nu,\mathbf{T}) \leftrightarrow (\nu,\mu,-\mathbf{T})\f$.
 * - Analytical derivatives match central finite differences to quadrature precision.
 *
 * @par SIESTA Mapping
 * Replaces \c matel.F90, \c kinefsm.F, \c naefs.F, and \c nlefsm.f.
 */

#pragma once

#include <cstdint>

extern "C" {

/**
 * @brief Evaluate sparse two-center \f$S_{\mu\nu}\f$, \f$T_{\mu\nu}\f$, and derivatives in C++.
 *
 * @par Theory
 * Computes overlap \f$S_{\mu\nu}\f$, kinetic energy \f$T_{\mu\nu}\f$, and their
 * displacement gradients wrt \f$\mathbf{R}_{\nu} - \mathbf{R}_{\mu}\f$ via Fourier-Bessel \c matel.
 *
 * @par Algorithm
 * Builds radial splines for each unique orbital shell pair \f$(s_1, s_2)\f$ and
 * evaluates all requested directed edges in a single C++ pass.
 *
 * @param n_shells Total unique radial PAO shells across all species.
 * @param n_pao_pts Number of uniform grid points per radial PAO shell.
 * @param pao_r_max Maximum radius of each PAO radial table (length \c n_shells).
 * @param pao_rcut Cutoff radius of each PAO shell (length \c n_shells).
 * @param pao_l Angular momentum \f$l\f$ of each PAO shell (length \c n_shells).
 * @param pao_values Flattened PAO radial tables, shape \c (n_shells, n_pao_pts).
 * @param n_edges Number of sparse orbital-pair edges.
 * @param edge_shell1 Global shell index of source orbital (length \c n_edges).
 * @param edge_m1 Magnetic quantum number \f$m_1\f$ of source orbital (length \c n_edges).
 * @param edge_shell2 Global shell index of target orbital (length \c n_edges).
 * @param edge_m2 Magnetic quantum number \f$m_2\f$ of target orbital (length \c n_edges).
 * @param edge_vectors Pair displacement vectors \f$\mathbf{R}_2 - \mathbf{R}_1\f$ (shape \c (n_edges, 3)).
 * @param out_s Output overlap matrix values (length \c n_edges).
 * @param out_t Output kinetic matrix values (length \c n_edges).
 * @param out_ds Output overlap derivatives \f$\partial S / \partial \mathbf{R}\f$ (shape \c (n_edges, 3)).
 * @param out_dt Output kinetic derivatives \f$\partial T / \partial \mathbf{R}\f$ (shape \c (n_edges, 3)).
 * @return 0 on success, non-zero on invalid arguments.
 *
 * @par Invariants
 * Output buffers must be pre-allocated to exact lengths.
 *
 * @par SIESTA Mapping
 * Replaces \c matel.F90 and \c kinefsm.F.
 */
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
    double* out_dt);

/**
 * @brief Evaluate orbital-to-KB-projector overlaps \f$\langle \phi_\mu | \chi_k \rangle\f$ and derivatives.
 *
 * @par Theory
 * Computes two-center overlaps between PAOs and nonlocal Kleinman-Bylander projectors
 * so \f$V^{\text{KB}}\f$ (both scalar and spin-orbit) and its forces/stress assemble cleanly.
 *
 * @par Algorithm
 * Reuses the identical Fourier-Bessel + Gaunt + cubic spline convolution kernel as
 * \c jsiesta_eval_two_center_st.
 *
 * @param n_pao_shells Number of unique PAO shells.
 * @param n_pao_pts Number of uniform grid points per PAO shell.
 * @param pao_r_max Max grid radius per PAO shell (length \c n_pao_shells).
 * @param pao_rcut Cutoff radius per PAO shell (length \c n_pao_shells).
 * @param pao_l Angular momentum per PAO shell (length \c n_pao_shells).
 * @param pao_values PAO tables, shape \c (n_pao_shells, n_pao_pts).
 * @param n_kb_shells Number of unique KB projector shells.
 * @param n_kb_pts Number of uniform grid points per KB projector shell.
 * @param kb_r_max Max grid radius per KB shell (length \c n_kb_shells).
 * @param kb_rcut Cutoff radius per KB shell (length \c n_kb_shells).
 * @param kb_l Angular momentum per KB shell (length \c n_kb_shells).
 * @param kb_values KB radial tables, shape \c (n_kb_shells, n_kb_pts).
 * @param n_links Number of interacting \c (orb, kb_proj) pairs.
 * @param link_pao_shell PAO shell index per link (length \c n_links).
 * @param link_m1 PAO \f$m_1\f$ per link (length \c n_links).
 * @param link_kb_shell KB shell index per link (length \c n_links).
 * @param link_m2 KB \f$m_2\f$ per link (length \c n_links).
 * @param link_vectors Displacement \f$\mathbf{R}_K - \mathbf{R}_\mu\f$ (shape \c (n_links, 3)).
 * @param out_p Output overlap \f$\langle \phi_\mu | \chi_k \rangle\f$ (length \c n_links).
 * @param out_dp Output gradient \f$\partial P / \partial (\mathbf{R}_K - \mathbf{R}_\mu)\f$ (shape \c (n_links, 3)).
 * @return 0 on success.
 *
 * @par Invariants
 * Output buffers must be pre-allocated.
 *
 * @par SIESTA Mapping
 * Replaces projector-overlap tables in \c nlefsm.f and \c matel.F90.
 */
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
    double* out_dp);

/**
 * @brief Evaluate short-range neutral-atom electrostatic pair correction \f$E_{\text{NA-pair}}\f$.
 *
 * @par Theory
 * Evaluates \f$U_{\text{NA}}(R_{IJ}) = \frac{Z_I Z_J}{R_{IJ}} - \int \rho^{\text{NA}}_I V_H[\rho^{\text{NA}}_J]\,d\mathbf{r}\f$
 * and its radial derivative for overlapping neutral-atom spheres.
 *
 * @par Algorithm
 * Evaluates the singularity-free Fourier-Bessel integral
 * \f$32\pi \int_0^\infty [\frac{Z_I Z_J}{16\pi^2} - \tilde{\rho}_I(k)\tilde{\rho}_J(k)] j_0(kR)\,dk\f$
 * via precomputed cubic splines.
 *
 * @param n_species Number of species.
 * @param n_pts Grid points per species \c rho_na table.
 * @param r_max Max radial grid extent per species (length \c n_species).
 * @param rcut_na Neutral-atom cutoff radius per species (length \c n_species).
 * @param z_val Valence charges per species (length \c n_species).
 * @param rho_na Flattened neutral-atom densities, shape \c (n_species, n_pts).
 * @param n_pairs Number of overlapping atom pairs \f$I \ne J\f$.
 * @param pair_sp1 Species index of atom 1 (length \c n_pairs).
 * @param pair_sp2 Species index of atom 2 (length \c n_pairs).
 * @param pair_dist Pair distances \f$R_{IJ}\f$ (length \c n_pairs).
 * @param out_e Output pair energy \f$U_{\text{NA}}(R_{IJ})\f$ (length \c n_pairs).
 * @param out_de_dr Output radial derivative \f$dU_{\text{NA}}/dR\f$ (length \c n_pairs).
 * @return 0 on success.
 *
 * @par Invariants
 * Vanishes smoothly to zero when \f$R_{IJ} \ge r_{c,I}^{\text{NA}} + r_{c,J}^{\text{NA}}\f$.
 *
 * @par SIESTA Mapping
 * Replaces short-range neutral-atom pair terms in \c naefs.F.
 */
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
    double* out_de_dr);

/**
 * @brief Tabulate \f$W(R) = \int V_{NA,v}(\mathbf r)\, n_{\text{at},n}(\mathbf r - \mathbf R)\, d^3r\f$ for one species pair.
 *
 * @par Theory
 * Both functions are spherical, so \f$W(R) = \frac{1}{2\pi^2}\int_0^\infty k^2\,\hat V(k)\,\hat n(k)\, j_0(kR)\, dk\f$ with
 * \f$\hat f(k) = 4\pi\int_0^\infty r^2 f(r)\, j_0(kr)\, dr\f$. It is the exact atomic part of the neutral-atom grid
 * energy (D-017).
 *
 * @par Algorithm
 * Trapezoid rules on the given radial grids for \f$\hat V\f$ and \f$\hat n\f$ on a uniform k grid
 * \f$[0, k_{\max}]\f$ of \c n_k points, threaded over k; then a trapezoid rule over k for each of \c n_big_r uniform
 * points \f$R \in [0, r_{\text{cut}}]\f$, threaded over R. The last value is set to zero.
 *
 * @param n_rv Number of radial points of \c v.
 * @param r_v Radial grid of \c v (Bohr).
 * @param v Neutral-atom potential \f$V_{NA}(r)\f$ (Ha).
 * @param n_rn Number of radial points of \c n.
 * @param r_n Radial grid of \c n (Bohr).
 * @param n Atomic valence density \f$n_{\text{at}}(r)\f$ (electrons/Bohr^3).
 * @param n_k Number of k points.
 * @param k_max Largest k (1/Bohr).
 * @param n_big_r Number of R points.
 * @param r_cut Largest R (Bohr), \f$r_{c,v} + r_{c,n}\f$.
 * @param out_w Output \f$W(R_i)\f$ (Ha), length \c n_big_r.
 * @return 0 on success, 1 on invalid sizes.
 *
 * @par Invariants
 * Equals the NumPy reference (\c numpy.trapezoid on the same grids) to rounding; \f$W(r_{\text{cut}}) = 0\f$.
 *
 * @par SIESTA Mapping
 * SIESTA's \c Ena two-centre term (\c naefs); SIESTA tabulates it with its own radial Fourier transforms.
 */
int32_t jsiesta_na_density_table(
    int32_t n_rv,
    const double* r_v,
    const double* v,
    int32_t n_rn,
    const double* r_n,
    const double* n,
    int32_t n_k,
    double k_max,
    int32_t n_big_r,
    double r_cut,
    double* out_w);

}  // extern "C"
