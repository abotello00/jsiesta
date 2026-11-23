/**
 * @file types.hpp
 * @brief Core C++ data types, atomic unit constants, and FFI ABI descriptors for jsiesta.
 *
 * @par Theory
 * Defines atomic-unit constants (Bohr \f$a_0\f$, Hartree \f$E_h\f$) and POD data views
 * shared between Python/JAX PyTrees and compiled C++/CUDA XLA FFI kernels.
 *
 * @par Algorithm
 * Uses flat, contiguous, read-only span descriptors with zero heap-allocated global
 * state so all kernels remain thread-safe and XLA-graph-compatible.
 *
 * @par Invariants
 * - All coordinate and lattice arrays are in Bohr.
 * - Spin modes are strictly \c nspin in {1, 2, 4, 8}.
 *
 * @par SIESTA Mapping
 * Replaces global module arrays from \c precision.F, \c m_spin.F90, and \c atm_types.F.
 */

#pragma once

#include <cstdint>

namespace jsiesta {

/** @brief Conversion factor from Bohr to Angstrom (CODATA). */
constexpr double kBohrToAngstrom = 0.529177210903;

/** @brief Conversion factor from Hartree to electron-volt (CODATA). */
constexpr double kHartreeToEv = 27.211386245988;

/** @brief Conversion factor from Hartree to Rydberg. */
constexpr double kHartreeToRy = 2.0;

/**
 * @brief Real-space orbital-grid spin modes matching SIESTA \c nspin.
 *
 * @par Theory
 * Enumerates scalar, collinear, non-collinear axial-vector, and SOC 2-spinor modes.
 *
 * @par Algorithm
 * Passed as an integer attribute to \c rhoofd, \c vmat, and \c xc_eval FFI kernels.
 *
 * @par Invariants
 * Strictly one of 1, 2, 4, or 8.
 *
 * @par SIESTA Mapping
 * Replaces \c nspin values in \c m_spin.F90.
 */
enum class SpinMode : int32_t {
    Unpolarized = 1,   /**< Scalar density \f$\rho\f$ */
    Collinear = 2,     /**< Spin-up and spin-down densities \f$(\rho_\uparrow, \rho_\downarrow)\f$ */
    NonCollinear = 4,  /**< 2x2 Hermitian density matrix \f$(\rho_{\uparrow\uparrow}, \rho_{\downarrow\downarrow}, \Re\rho_{\uparrow\downarrow}, \Im\rho_{\uparrow\downarrow})\f$ */
    SpinOrbit = 8      /**< Full SOC 2-spinor density matrix */
};

/**
 * @brief Descriptor for uniform radial cubic spline tables in C++ kernels.
 *
 * @par Theory
 * Represents piecewise cubic Hermite/natural spline interpolants for PAOs, KB projectors, and \f$V^{\text{NA}}(r)\f$.
 *
 * @par Algorithm
 * Stores tabulated grid points, values, and second derivatives for \f$O(1)\f$ radial lookup.
 *
 * @par Invariants
 * \c n_pts >= 2 and \c rcut >= 0.
 *
 * @par SIESTA Mapping
 * Replaces radial spline tables in \c radial.f and \c matel.F90.
 */
struct RadialSplineView {
    const double* r;       /**< Radial grid points (Bohr), length \c n_pts. */
    const double* values;  /**< Function values \f$f(r)\f$, length \c n_pts. */
    const double* y2;      /**< Second-derivative cubic spline coefficients, length \c n_pts. */
    int32_t n_pts;         /**< Number of radial grid points. */
    double rcut;           /**< Cutoff radius beyond which \f$f(r) = 0\f$ (Bohr). */
};

/**
 * @brief CSR neighbor-list view for sparse orbital/atom interactions.
 *
 * @par Theory
 * Encodes the symmetry-closed interacting orbital-pair graph \f$(\mu, \nu, \mathbf{R})\f$.
 *
 * @par Algorithm
 * Compressed sparse row offsets paired with target unit-cell orbitals and pair displacement vectors.
 *
 * @par Invariants
 * \c row_ptr has length \c n_orb + 1 with \c row_ptr[n_orb] == n_edges.
 *
 * @par SIESTA Mapping
 * Replaces \c numh, \c listhptr, \c listh, and \c xijo in \c sparse_matrices.F.
 */
struct SparseCSRView {
    const int32_t* row_ptr;        /**< CSR row offsets, length \c n_orb + 1. */
    const int32_t* dst_unit_orb;   /**< Target unit-cell orbital indices, length \c n_edges. */
    const double* edge_vectors;    /**< Pair displacement vectors \f$\mathbf{R}_{\nu} - \mathbf{R}_{\mu}\f$ (Bohr), shape \c (n_edges, 3). */
    int32_t n_orb;                 /**< Number of unit-cell orbitals. */
    int32_t n_edges;               /**< Total directed sparse edges. */
};

}  // namespace jsiesta
