/**
 * @file hfx.hpp
 * @brief Exact-exchange matrix on the sparse orbital pattern from Gaussian fits of the numerical orbitals.
 *
 * @par Theory
 * With the density matrix \f$P\f$ and the exchange matrix \f$K\f$ stored on the edges
 * \f$(\mu, \lambda, \mathbf{G})\f$ of the sparse pattern (orbital \f$\mu\f$ in the home cell, \f$\lambda\f$ in cell
 * \f$\mathbf{G}\f$),
 * \f[ K^{\mathbf{G}}_{\mu\lambda} = \sum_{\nu\sigma} \sum_{\mathbf{N}\mathbf{H}}
 *     P^{\mathbf{H}-\mathbf{N}}_{\nu\sigma}\, (\mu\mathbf{0}\, \nu\mathbf{N} | \lambda\mathbf{G}\, \sigma\mathbf{H})_w , \f]
 * with the kernel \f$w(r) = c_\mathrm{full}/r + c_\mathrm{sr}\,\mathrm{erfc}(\omega r)/r\f$ and the integrals of
 * \c eri.hpp over the Gaussian fits of the orbitals (NAO2GTO). The Hamiltonian contribution is \f$-K^s\f$ for the
 * density matrix of one spin channel; the caller applies sign and spin factors. Density-matrix elements and
 * exchange-matrix elements outside the pattern are neglected (the convention of HONPAS).
 *
 * @par Algorithm
 * Once per geometry: the list of shell pairs \f$(I\mathbf{0}, J\mathbf{N})\f$ whose numerical orbitals overlap, their
 * Hermite expansions and Schwarz bounds \f$Q_{IJ} = \max\sqrt{(ij|ij)}\f$; and the shell-pair blocks of the pattern.
 * Per density matrix: for every bra pair, every pattern block \f$(I, K\mathbf{G})\f$ and every ket pair
 * \f$(K\mathbf{0}, L\mathbf{M})\f$ translated by \f$\mathbf{G}\f$, the quartet is skipped when
 * \f$Q_{IJ} Q_{KL} \max|P_{JL}|\f$ is below the tolerance, else its integrals are contracted with the block of
 * \f$P\f$. Threads take bra pairs from a shared counter and accumulate into private copies of \f$K\f$.
 *
 * @par Invariants
 * \f$K\f$ is linear in \f$P\f$ and symmetric, \f$K^{\mathbf{G}}_{\mu\lambda} = K^{-\mathbf{G}}_{\lambda\mu}\f$, when
 * \f$P\f$ is and the pattern is closed under transposition (up to the screening tolerance).
 *
 * @par SIESTA Mapping
 * None in SIESTA. HONPAS: the HFX matrix construction of Shang et al., Comput. Phys. Commun. 254, 107204 (2020),
 * Algorithm 4, without the MPI distribution.
 */
#pragma once

#include <cstdint>

extern "C" {

/**
 * @brief Prepare the exchange build of one geometry: shell pairs, Schwarz bounds, pattern blocks.
 *
 * @par Theory
 * See the file description.
 *
 * @par Algorithm
 * Shell pairs are searched over all lattice translations within the sum of the orbital radii.
 *
 * @par Invariants
 * Returns a handle to release with \c jsiesta_hfx_destroy, or null on invalid input.
 *
 * @par SIESTA Mapping
 * HONPAS shell-pair lists (list-IJ, list-KL).
 *
 * @param n_atoms Number of atoms in the unit cell.
 * @param positions Atomic positions, shape (n_atoms, 3), Bohr.
 * @param cell Lattice vectors as the columns of a row-major 3x3 matrix, Bohr.
 * @param n_shell Number of shells in the unit cell.
 * @param shell_atom Atom of each shell.
 * @param shell_l Angular momentum of each shell.
 * @param shell_rcut Radius of the numerical orbital of each shell, Bohr.
 * @param shell_orb First orbital of each shell (its 2l+1 orbitals are contiguous, m from -l to l).
 * @param shell_prim_ptr Offsets of the primitives of each shell, length n_shell + 1.
 * @param exps Exponents of all primitives, Bohr^-2.
 * @param coefs Coefficients of all primitives (of the solid harmonic times the bare Gaussian).
 * @param n_orb Number of orbitals in the unit cell.
 * @param n_edges Number of edges of the sparse pattern.
 * @param edge_src,edge_dst,edge_cell Edge list: home-cell orbital, orbital, index of the cell of the second.
 * @param n_cells Number of cells of the pattern.
 * @param cell_shifts Integer lattice coordinates of the cells, shape (n_cells, 3).
 * @param c_full,c_sr,omega Weights of the full-range and short-range kernels and the range parameter (Bohr^-1).
 * @param prim_tol Primitive pairs with a prefactor below this are dropped.
 * @param pair_tol Shell pairs with a Schwarz bound below this are dropped.
 */
void* jsiesta_hfx_create(int32_t n_atoms, const double* positions, const double* cell, int32_t n_shell,
                         const int32_t* shell_atom, const int32_t* shell_l, const double* shell_rcut,
                         const int32_t* shell_orb, const int32_t* shell_prim_ptr, const double* exps,
                         const double* coefs, int32_t n_orb, int64_t n_edges, const int32_t* edge_src,
                         const int32_t* edge_dst, const int32_t* edge_cell, int32_t n_cells,
                         const int32_t* cell_shifts, double c_full, double c_sr, double omega, double prim_tol,
                         double pair_tol);

/**
 * @brief Exchange matrices \f$K^s[P^s]\f$ of the density matrices of \p n_spin channels.
 *
 * @par Theory
 * See the file description; no sign or spin factor is applied.
 *
 * @par Algorithm
 * Screened quartet loop, threaded.
 *
 * @par Invariants
 * Returns 0 on success. \p k_out is overwritten.
 *
 * @par SIESTA Mapping
 * HONPAS \c setup_hfx.
 *
 * @param handle Handle from \c jsiesta_hfx_create.
 * @param n_spin Number of density matrices.
 * @param dm Density matrices on the edges, shape (n_spin, n_edges).
 * @param screen_tol Quartets whose Schwarz bounds times the largest density-matrix element are below this are skipped.
 * @param k_out Output, shape (n_spin, n_edges).
 * @param stats Output, length 4: quartets computed, quartets screened out, shell pairs, threads.
 */
int32_t jsiesta_hfx_build(void* handle, int32_t n_spin, const double* dm, double screen_tol, double* k_out,
                          int64_t* stats);

/**
 * @brief Derivative sums of the exchange energy with respect to the centres, at fixed density matrices.
 *
 * @par Theory
 * With \f$I = (\mu\mathbf{0}\,\nu\mathbf{N}|\lambda\mathbf{G}\,\sigma\mathbf{H})_w\f$ and
 * \f$W = \sum_s P^s_{\mu\lambda} P^s_{\nu\sigma}\f$, the routine returns
 * \f$D_A = \sum W\,\partial_{\mathbf{R}_\mu} I\f$ per atom of \f$\mu\f$,
 * \f$V_1 = \sum W\,\partial_{\mathbf{R}_\mu} I \otimes (\mathbf{R}_\mu - \mathbf{R}_\nu)\f$ and
 * \f$V_2 = \sum W\,\partial_{\mathbf{P}} I \otimes (\mathbf{R}_\lambda - \mathbf{R}_\mu)\f$, where
 * \f$\partial_{\mathbf{P}}\f$ is the derivative with respect to a common translation of \f$\mu\f$ and \f$\nu\f$.
 * The sum over quartets is symmetric under the exchange of the two orbitals of each pair and of the two pairs, so
 * the four centres contribute equally: for \f$E = -\kappa \sum W I\f$ the derivative with respect to an atom is
 * \f$-4\kappa D_A\f$ and the derivative with respect to a homogeneous strain is \f$-\kappa (2 V_1 - V_2)\f$
 * (translation invariance gives \f$\partial_\lambda + \partial_\sigma = -\partial_{\mathbf{P}}\f$).
 *
 * @par Algorithm
 * The quartet loop of \c jsiesta_hfx_build with six derivative expansions of the bra pair (\c make_shell_pair_mode),
 * built once per bra pair; each integral block is contracted with \f$W\f$ immediately.
 *
 * @par Invariants
 * \f$\sum_A D_A = 0\f$ (no net force) to the screening tolerance. Returns 0 on success.
 *
 * @par SIESTA Mapping
 * None in SIESTA; the HFX force routine of HONPAS.
 *
 * @param handle Handle from \c jsiesta_hfx_create.
 * @param n_spin Number of density matrices.
 * @param dm Density matrices on the edges, shape (n_spin, n_edges).
 * @param screen_tol Quartets whose Schwarz bounds times both density-matrix blocks are below this are skipped.
 * @param d_first Output, shape (n_atoms, 3).
 * @param virial_first Output, shape (3, 3): derivative index first.
 * @param virial_pair Output, shape (3, 3): derivative index first.
 * @param stats Output, length 4: quartets computed, quartets screened out, shell pairs, threads.
 */
int32_t jsiesta_hfx_gradient(void* handle, int32_t n_spin, const double* dm, double screen_tol, double* d_first,
                             double* virial_first, double* virial_pair, int64_t* stats);

/**
 * @brief Switch the distance bound of the screened kernel on or off.
 *
 * @par Theory
 * For a kernel without full-range part, a quartet is also bounded by the short-range interaction of Gaussian
 * envelopes of its two pair densities, which decays with their distance; the Schwarz bound does not.
 *
 * @par Algorithm
 * Sets a flag read by the build and the gradient. The envelopes are prepared by \c jsiesta_hfx_create.
 *
 * @par Invariants
 * No effect for a kernel with a full-range part. On by default for a screened kernel.
 *
 * @par SIESTA Mapping
 * None in SIESTA; the far-field screening of HONPAS has the same purpose.
 *
 * @param handle Handle from \c jsiesta_hfx_create.
 * @param on Non-zero to use the bound.
 */
void jsiesta_hfx_set_distance_screening(void* handle, int32_t on);

/**
 * @brief Release a handle of \c jsiesta_hfx_create.
 *
 * @par Theory
 * The handle owns the shell-pair expansions of one geometry.
 *
 * @par Algorithm
 * Deletes the object.
 *
 * @par Invariants
 * A null handle is ignored.
 *
 * @par SIESTA Mapping
 * None.
 *
 * @param handle Handle to release.
 */
void jsiesta_hfx_destroy(void* handle);

}  // extern "C"
