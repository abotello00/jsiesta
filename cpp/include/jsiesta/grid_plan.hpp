/**
 * @file grid_plan.hpp
 * @brief Flat, read-only view of the cached host grid plan (active orbitals per grid point, atom-pair DM blocks).
 *
 * @par Theory
 * At fixed geometry the orbital values \f$\phi_\mu(\mathbf{r}_p - \mathbf{R}_I - \mathbf{A}\mathbf{T})\f$ on the mesh do
 * not change, so \f$\rho(\mathbf{r}_p) = \sum_{IJ} \boldsymbol{\phi}_I(p)^\mathsf{T} D_{IJ}(\mathbf{T}) \boldsymbol{\phi}_J(p)\f$
 * and \f$V_{IJ} = d\Omega \sum_p V(\mathbf{r}_p)\, \boldsymbol{\phi}_I(p) \boldsymbol{\phi}_J(p)^\mathsf{T}\f$ only need
 * the stored values and dense atom-pair blocks \f$D_{IJ}(\mathbf{T})\f$.
 *
 * @par Algorithm
 * The CPU kernels in \c grid_ffi.cpp build the plan once per geometry (LRU of four, keyed by all input bytes). This
 * header exposes it as flat arrays so that the CUDA kernels can upload it and contract on the device.
 *
 * @par Invariants
 * - The arrays stay valid while \c owner is held.
 * - \c use_atom_blocks is false when the dense pair table would be too large; then the other fields are empty.
 *
 * @par SIESTA Mapping
 * The stored-\f$\phi\f$ mode of SIESTA (\c DirectPhi \c .false., \c Src/meshsubs.F, \c rhoofd.F90, \c vmat.F90).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace jsiesta::grid {

/**
 * @brief Flat view of one cached host grid plan.
 *
 * @par Theory
 * Active atom images per grid point with their orbital values (and gradients and Hessians when requested), and the dense
 * atom-pair block layout of the sparse density matrix.
 *
 * @par Algorithm
 * Filled by \c get_flat_grid_plan from the plan cache of \c grid_ffi.cpp; block-to-edge maps are flattened with
 * absolute block offsets.
 *
 * @par Invariants
 * - \c grid_img_offsets has \c n_grid + 1 entries; image \c i is \c img[5 i .. 5 i + 4] = (atom, s0, s1, s2, value
 *   offset).
 * - \c block_edge_abs[k] indexes a block scalar and \c block_edge_idx[k] its sparse edge; every edge appears once.
 *
 * @par SIESTA Mapping
 * Corresponds to SIESTA's stored orbital values on the mesh and its per-big-point orbital lists.
 */
struct FlatGridPlan {
    std::shared_ptr<const void> owner;          ///< Keeps the host plan alive.
    std::uint64_t hash = 0;                     ///< Hash of all inputs the plan depends on.
    bool use_atom_blocks = false;               ///< Whether the atom-block representation exists.
    bool has_grad = false;                      ///< Whether gradient arrays are filled.
    bool has_hess = false;                      ///< Whether Hessian arrays are filled (implies \c has_grad).
    std::int64_t n_grid = 0;                    ///< Number of mesh points.
    std::int32_t n_atoms = 0;                   ///< Number of atoms.
    std::int32_t n_shifts = 0;                  ///< Number of cell shifts of the pattern.
    std::int32_t s_min[3] = {0, 0, 0};          ///< Smallest cell shift per axis.
    std::int32_t s_dim[3] = {1, 1, 1};          ///< Extent of the shift box per axis.
    double dvol = 0.0;                          ///< Volume per mesh point.
    std::size_t total_blk_scalars = 0;          ///< Scalars in all dense atom-pair blocks (per spin component).
    const std::int32_t* grid_img_offsets = nullptr;  ///< Image ranges per mesh point (\c n_grid + 1).
    const std::int32_t* img = nullptr;               ///< Images, five \c int32 each.
    std::size_t n_img = 0;                           ///< Number of images.
    const double* img_vals = nullptr;                ///< Orbital values of all images.
    const double* img_gx = nullptr;                  ///< Orbital x gradients (\c has_grad only).
    const double* img_gy = nullptr;                  ///< Orbital y gradients (\c has_grad only).
    const double* img_gz = nullptr;                  ///< Orbital z gradients (\c has_grad only).
    const double* img_hxx = nullptr;                 ///< Orbital Hessian xx (\c has_hess only).
    const double* img_hxy = nullptr;                 ///< Orbital Hessian xy (\c has_hess only).
    const double* img_hxz = nullptr;                 ///< Orbital Hessian xz (\c has_hess only).
    const double* img_hyy = nullptr;                 ///< Orbital Hessian yy (\c has_hess only).
    const double* img_hyz = nullptr;                 ///< Orbital Hessian yz (\c has_hess only).
    const double* img_hzz = nullptr;                 ///< Orbital Hessian zz (\c has_hess only).
    std::size_t n_vals = 0;                          ///< Length of the value arrays.
    const std::int32_t* atom_n_orbs = nullptr;       ///< Orbitals per atom (\c n_atoms).
    const std::int32_t* shift_box = nullptr;         ///< Shift-box cell -> shift index or -1.
    std::size_t shift_box_len = 0;                   ///< Length of \c shift_box.
    const std::int32_t* pair_block_id = nullptr;     ///< (atom_a, atom_b, shift) -> block id or -1.
    std::size_t pair_block_len = 0;                  ///< Length of \c pair_block_id.
    std::vector<std::int32_t> block_offset;          ///< Scalar offset of each block.
    std::vector<std::int32_t> block_edge_abs;        ///< Absolute block scalar of each mapped edge.
    std::vector<std::int32_t> block_edge_idx;        ///< Sparse edge index of each mapped scalar.
};

/**
 * @brief Hash of the grid plan inputs, without building a plan.
 *
 * Hashes the same inputs as the plan cache without building a plan, so a caller can look for a cached plan that
 * stores more (gradients or Hessians) than it needs.
 *
 * @par Theory
 * The plan is a pure function of geometry, basis tables, sparsity pattern and mesh; equal inputs give equal plans.
 *
 * @par Algorithm
 * 64-bit mixing hash over the mesh dimensions and the bytes of every input array, in a fixed order.
 *
 * @param positions Atomic positions \c (n_atoms, 3), Bohr.
 * @param cell Lattice vectors \c (3, 3), Bohr.
 * @param pao_r_max Radial table extent per shell.
 * @param pao_rcut Cut-off radius per shell.
 * @param pao_l Angular momentum per shell.
 * @param pao_values Radial tables \c (n_shells, n_pao_pts).
 * @param orb_atom Atom of each orbital.
 * @param orb_g_shell Global shell of each orbital.
 * @param orb_m Magnetic quantum number of each orbital.
 * @param orb_rcut Cut-off radius of each orbital.
 * @param row_ptr CSR row pointer of the pattern.
 * @param dst_unit_orb Column orbital of each edge.
 * @param edge_cell_idx Cell-shift index of each edge.
 * @param cell_shifts Integer cell shifts \c (n_shifts, 3).
 * @param n_atoms Number of atoms.
 * @param n_orb Number of orbitals.
 * @param n_shells Number of radial shells.
 * @param n_pao_pts Points per radial table.
 * @param n_edges Number of pattern edges.
 * @param n_shifts Number of cell shifts.
 * @param n1 Mesh points along the first axis.
 * @param n2 Mesh points along the second axis.
 * @param n3 Mesh points along the third axis.
 * @return Hash of the inputs; grid_plan_cache_key() turns it into the cache key of each plan variant.
 *
 * @par Invariants
 * Pure function of its inputs.
 *
 * @par SIESTA Mapping
 * None; jsiesta cache bookkeeping.
 */
std::uint64_t grid_plan_input_hash(
    const double* positions, const double* cell,
    const double* pao_r_max, const double* pao_rcut, const std::int32_t* pao_l, const double* pao_values,
    const std::int32_t* orb_atom, const std::int32_t* orb_g_shell, const std::int32_t* orb_m, const double* orb_rcut,
    const std::int32_t* row_ptr, const std::int32_t* dst_unit_orb, const std::int32_t* edge_cell_idx,
    const std::int32_t* cell_shifts,
    std::int32_t n_atoms, std::int32_t n_orb, std::int32_t n_shells, std::int32_t n_pao_pts,
    std::int32_t n_edges, std::int32_t n_shifts,
    std::int64_t n1, std::int64_t n2, std::int64_t n3);

/**
 * @brief Cache key of a plan variant, equal to the \c FlatGridPlan::hash get_flat_grid_plan() stores.
 *
 * @par Theory
 * Plans of the same inputs differ only in which per-image arrays they store (values, gradients, Hessians).
 *
 * @par Algorithm
 * Mixes the gradient and Hessian flags into the input hash.
 *
 * @param input_hash Result of grid_plan_input_hash() for the plan inputs.
 * @param need_grad Whether the plan stores orbital gradients.
 * @param need_hess Whether the plan stores orbital Hessians; implies \p need_grad.
 * @return The cache key.
 *
 * @par Invariants
 * Pure function; cheap, so one input hash gives the key of every variant.
 *
 * @par SIESTA Mapping
 * None; jsiesta cache bookkeeping.
 */
std::uint64_t grid_plan_cache_key(std::uint64_t input_hash, bool need_grad, bool need_hess);

/**
 * @brief Return the cached host grid plan for the given inputs as flat arrays (built on first use).
 *
 * @par Theory
 * See \c FlatGridPlan; the plan is a pure function of geometry, basis tables, sparsity pattern and mesh.
 *
 * @par Algorithm
 * Calls the CPU plan cache of \c grid_ffi.cpp (exact byte comparison of all inputs) and flattens the block-to-edge
 * maps.
 *
 * @param positions Atomic positions \c (n_atoms, 3).
 * @param cell Cell matrix, column vectors.
 * @param pao_r_max Radial table extents per shell.
 * @param pao_rcut Cut-off radii per shell.
 * @param pao_l Angular momenta per shell.
 * @param pao_values Radial tables \c (n_shells, n_pao_pts).
 * @param orb_atom Atom of each orbital.
 * @param orb_g_shell Global shell of each orbital.
 * @param orb_m Magnetic quantum number of each orbital.
 * @param orb_rcut Cut-off radius of each orbital.
 * @param row_ptr CSR row pointers of the pattern.
 * @param dst_unit_orb CSR destination orbitals.
 * @param edge_cell_idx Cell-shift index of each edge.
 * @param cell_shifts Integer cell shifts \c (n_shifts, 3).
 * @param n_atoms Number of atoms.
 * @param n_orb Number of orbitals.
 * @param n_shells Number of radial shells.
 * @param n_pao_pts Points per radial table.
 * @param n_edges Number of pattern edges.
 * @param n_shifts Number of cell shifts.
 * @param n1 Mesh points along the first axis.
 * @param n2 Mesh points along the second axis.
 * @param n3 Mesh points along the third axis.
 * @param need_grad Whether orbital gradients are stored.
 * @param need_hess Whether orbital Hessians are stored as well (meta-GGA backward); implies \p need_grad.
 * @return Flat view; \c use_atom_blocks is false when only the generic representation exists.
 *
 * @par Invariants
 * Thread-safe; equal inputs return views of the same cached plan.
 *
 * @par SIESTA Mapping
 * Plays the role of SIESTA's \c PhiOnMesh set-up in \c meshsubs.F.
 */
FlatGridPlan get_flat_grid_plan(
    const double* positions, const double* cell,
    const double* pao_r_max, const double* pao_rcut, const std::int32_t* pao_l, const double* pao_values,
    const std::int32_t* orb_atom, const std::int32_t* orb_g_shell, const std::int32_t* orb_m, const double* orb_rcut,
    const std::int32_t* row_ptr, const std::int32_t* dst_unit_orb, const std::int32_t* edge_cell_idx,
    const std::int32_t* cell_shifts,
    std::int32_t n_atoms, std::int32_t n_orb, std::int32_t n_shells, std::int32_t n_pao_pts,
    std::int32_t n_edges, std::int32_t n_shifts,
    std::int64_t n1, std::int64_t n2, std::int64_t n3, bool need_grad, bool need_hess = false);

}  // namespace jsiesta::grid
