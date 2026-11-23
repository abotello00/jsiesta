/**
 * @file grid_xc_ffi.hpp
 * @brief XLA/JAX FFI declarations for real-space grid (\c rhoofd, \c vmat, \c dfscf, \c na_grid) and XC kernels.
 *
 * @par Theory
 * Declares XLA CPU/GPU FFI handlers for bounded-sphere orbital-grid density scatter
 * (\c rhoofd), adjoint potential gather (\c vmat), orbital/neutral-atom grid gradients
 * (\c dfscf), and pointwise exchange-correlation evaluation (\c xc_eval).
 *
 * @par Algorithm
 * All grid handlers operate on flat contiguous buffers passed directly from JAX via
 * \c jax.extend.ffi.ffi_call without host-device round-trips inside SCF.
 *
 * @par Invariants
 * - Exact discrete adjoint identity: \f$\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]\f$.
 * - Thread-safe, pure functions of input buffers.
 *
 * @par SIESTA Mapping
 * Replaces \c rhoofd.F90, \c vmat.F90, \c dfscf.F, \c dhscf.F, and \c libgridxc/cellxc.F.
 */

#pragma once

#include "xla/ffi/api/c_api.h"

extern "C" {

/**
 * @brief XLA FFI handler scattering sparse density matrix onto the 3D real-space mesh.
 *
 * @par Theory
 * Evaluates \f$\rho_s(\mathbf{r}_p) = \sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T})\phi_\mu(\mathbf{r}_p)\phi_\nu(\mathbf{r}_p - \mathbf{A}\mathbf{T})\f$.
 *
 * @par Algorithm
 * Loops over unit-cell atoms and overlapping periodic images inside each orbital's
 * spherical cutoff \f$r_c(\mu)\f$, evaluating radial cubic splines and solid harmonics.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Supports \c nspin in {1, 2, 4, 8}.
 *
 * @par SIESTA Mapping
 * Replaces \c rhoofd.F90.
 */
XLA_FFI_Error* jsiesta_rhoofd_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI handler gathering effective grid potential into sparse Hamiltonian matrix elements.
 *
 * @par Theory
 * Evaluates \f$V^{\text{grid}}_{s,\mu\nu}(\mathbf{T}) = d\Omega \sum_p V_s(\mathbf{r}_p)\phi_\mu(\mathbf{r}_p)\phi_\nu(\mathbf{r}_p - \mathbf{A}\mathbf{T})\f$.
 *
 * @par Algorithm
 * Exact discrete adjoint of \c jsiesta_rhoofd_f64 over active orbitals at each grid point.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Satisfies \f$\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]\f$ to machine precision.
 *
 * @par SIESTA Mapping
 * Replaces \c vmat.F90.
 */
XLA_FFI_Error* jsiesta_vmat_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI backward VJP handler for \c dfscf orbital-grid forces and cell gradients.
 *
 * @par Theory
 * Evaluates \f$\partial \langle V | \rho(D) \rangle / \partial \mathbf{R}_I\f$ and
 * \f$\partial \langle V | \rho(D) \rangle / \partial \mathbf{A}_{\alpha\beta}\f$ at fixed \f$D, V\f$.
 *
 * @par Algorithm
 * Differentiates radial cubic splines and solid harmonics analytically at each overlapping
 * grid point without numerical finite differences.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches central finite differences of \f$\langle V | \rho(D) \rangle\f$ wrt atomic positions and cell.
 *
 * @par SIESTA Mapping
 * Replaces orbital-grid derivatives in \c dfscf.F.
 */
XLA_FFI_Error* jsiesta_dfscf_grad_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI handler scattering spherical neutral-atom density, potential, and NLCC core charge onto the 3D grid.
 *
 * @par Theory
 * Evaluates \f$\rho^{\text{NA}}(\mathbf{r})\f$, \f$V^{\text{NA}}(\mathbf{r})\f$, and
 * \f$\rho^{\text{core}}(\mathbf{r})\f$ summed over periodic atomic images within \f$r_c^{\text{NA}}\f$.
 *
 * @par Algorithm
 * Bounded-sphere fractional box loop per atom with natural cubic radial spline lookup.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Integrated \f$\rho^{\text{NA}}\f$ equals total valence charge as mesh cutoff increases.
 *
 * @par SIESTA Mapping
 * Replaces neutral-atom / pseudo-core grid assembly in \c dhscf.F.
 */
XLA_FFI_Error* jsiesta_na_grid_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI backward VJP handler for neutral-atom and NLCC grid position/cell derivatives.
 *
 * @par Theory
 * Computes exact gradients of \f$\int [w_{\rho}\rho^{\text{NA}} + w_V V^{\text{NA}} + w_c \rho^{\text{core}}]\,d\mathbf{r}\f$
 * wrt atomic coordinates and lattice matrix.
 *
 * @par Algorithm
 * Accumulates analytical radial spline derivatives over each atom's grid sphere.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Matches finite differences of \c jsiesta_na_grid_f64 wrt positions and cell.
 *
 * @par SIESTA Mapping
 * Replaces \c dfscf.F neutral-atom and NLCC grid force/stress terms.
 */
XLA_FFI_Error* jsiesta_na_grid_grad_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI pointwise exchange-correlation evaluator (\c libxc or built-in C++ LDA/PBE).
 *
 * @par Theory
 * Computes energy density \f$\varepsilon_{\text{xc}}(\mathbf{r})\f$ and derivatives
 * \f$\partial (n\varepsilon_{\text{xc}})/\partial \rho_\sigma\f$,
 * \f$\partial (n\varepsilon_{\text{xc}})/\partial |\nabla\rho|^2\f$ pointwise on the mesh.
 *
 * @par Algorithm
 * Calls \c libxc when \c JSIESTA_HAS_LIBXC=1, or exact analytical C++ Slater/PZ/VWN/PBE/PBEsol kernels.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 *
 * @par Invariants
 * Functional derivatives match finite perturbations of density and gradient invariants.
 *
 * @par SIESTA Mapping
 * Replaces \c libgridxc and \c cellxc.F.
 */
XLA_FFI_Error* jsiesta_xc_eval_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI meta-GGA pointwise exchange-correlation evaluator (\c libxc).
 *
 * @par Theory
 * Evaluates \f$e_{\text{xc}}(\rho_\sigma, \sigma_{\alpha\beta}, \tau_\sigma)\f$ and
 * exact partial derivatives wrt density, gradient invariants, and kinetic energy density \f$\tau\f$.
 *
 * @par Algorithm
 * Loops over grid points calling \c jsiesta::evaluate_mgga_libxc_host with
 * \c xc_mgga_exc_vxc from Libxc (thresholded at \f$n \le 10^{-9}\f$).
 *
 * @par Invariants
 * Vacuum grid points with \f$n_{\text{tot}} \le 10^{-9}\f$ return identically zero
 * energy density and potential derivatives without NaNs.
 *
 * @par SIESTA Mapping
 * Replaces \c libgridxc meta-GGA pointwise calls inside \c mggaxc.F90.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 */
XLA_FFI_Error* jsiesta_xc_eval_mgga_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI handler scattering orbital kinetic energy density \f$\tau_s(\mathbf{r})\f$ onto the 3D mesh.
 *
 * @par Theory
 * Evaluates \f$\tau_s(\mathbf{r}_p) = \frac{1}{2}\sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T}) \nabla\phi_\mu(\mathbf{r}_p) \cdot \nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})\f$.
 *
 * @par Algorithm
 * Iterates over sparse orbital edges and intersects compact bounding boxes with the 3D mesh,
 * evaluating orbital spatial gradients via cubic splines and real solid harmonics.
 *
 * @par Invariants
 * Symmetric density matrices produce real, non-negative collinear \f$\tau_\sigma(\mathbf{r})\f$.
 *
 * @par SIESTA Mapping
 * Extends \c rhoofd.F90 to assemble meta-GGA kinetic energy density \f$\tau\f$.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 */
XLA_FFI_Error* jsiesta_tauofd_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI handler gathering meta-GGA potential \f$v_{\tau,s}(\mathbf{r})\f$ into sparse Hamiltonian edges.
 *
 * @par Theory
 * Exact discrete adjoint of \c jsiesta_tauofd_f64:
 * \f$V^\tau_{s,\mu\nu}(\mathbf{T}) = \frac{1}{2} d\Omega \sum_p v_{\tau,s}(\mathbf{r}_p) \nabla\phi_\mu(\mathbf{r}_p) \cdot \nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})\f$.
 *
 * @par Algorithm
 * Uses the identical bounding-box mesh loop and orbital gradient evaluation as \c jsiesta_tauofd_f64,
 * accumulating dot-product weighted edge contributions scaled by voxel volume \f$d\Omega\f$.
 *
 * @par Invariants
 * Satisfies exact inner-product adjointness with \c jsiesta_tauofd_f64 to machine precision.
 *
 * @par SIESTA Mapping
 * Extends \c vmat.F90 for meta-GGA \f$\tau\f$-potential matrix elements.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 */
XLA_FFI_Error* jsiesta_vmat_tau_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI backward VJP handler for meta-GGA \f$\tau\f$ orbital-Hessian forces and stress.
 *
 * @par Theory
 * Contracts incoming cotangent \f$\bar{\tau}_s(\mathbf{r}_p)\f$ with spatial orbital Hessians
 * \f$\nabla\nabla\phi\f$ and voxel-volume variation \f$\delta(d\Omega)/d\Omega\f$.
 *
 * @par Algorithm
 * Evaluates second spatial derivatives of splines and real solid harmonics on active edge-overlap
 * grid points and accumulates atomic forces and metric virial stress tensors.
 *
 * @par Invariants
 * Matches central finite differences of \f$\langle \bar{\tau} | \tau(\mathbf{R}, \mathbf{A}) \rangle\f$.
 *
 * @par SIESTA Mapping
 * Extends \c dfscf.F90 with orbital-Hessian Pulay forces for meta-GGA functionals.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 */
XLA_FFI_Error* jsiesta_tau_dfscf_grad_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI handler scattering spherical NLCC \f$\tau^{\text{core}}(\mathbf{r})\f$ onto the 3D mesh.
 *
 * @par Theory
 * Sums radially tabulated pseudo-core kinetic energy densities around every periodic atom image.
 *
 * @par Algorithm
 * Evaluates cubic radial splines on mesh points inside each atom's core cutoff radius.
 *
 * @par Invariants
 * Produces smooth non-negative core kinetic energy density on periodic grids.
 *
 * @par SIESTA Mapping
 * Provides the meta-GGA NLCC analogue of \c rhoat.F / \c nlefsm core terms.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 */
XLA_FFI_Error* jsiesta_tau_core_f64(XLA_FFI_CallFrame* call);

/**
 * @brief XLA FFI backward VJP handler for spherical NLCC \f$\tau^{\text{core}}\f$ forces and stress.
 *
 * @par Theory
 * Computes gradients of \f$\sum_p \bar{\tau}^{\text{core}}_p \tau^{\text{core}}(\mathbf{r}_p)\f$
 * with respect to atomic coordinates \f$\mathbf{R}_I\f$ and cell matrix \f$\mathbf{A}\f$.
 *
 * @par Algorithm
 * Uses radial spline derivatives \f$d\tau^{\text{core}}/dr\f$ to assemble force and virial sums.
 *
 * @par Invariants
 * Total atomic force sums to zero by translational invariance.
 *
 * @par SIESTA Mapping
 * Computes Pulay forces and stress from the NLCC pseudo-core kinetic density.
 *
 * @param call XLA FFI call frame pointer.
 * @return \c XLA_FFI_Error* or \c nullptr on success.
 */
XLA_FFI_Error* jsiesta_tau_core_grad_f64(XLA_FFI_CallFrame* call);

}  // extern "C"

namespace jsiesta {

/**
 * @brief Host Libxc meta-GGA pointwise evaluator shared by CPU and CUDA FFI handlers.
 *
 * @par Theory
 * Evaluates semilocal meta-GGA exchange and correlation energy density and partial
 * derivatives via Libxc \c xc_mgga_exc_vxc for collinear spin channels.
 *
 * @par Algorithm
 * Sets Libxc density threshold to \f$10^{-10}\f$, packs transposed AoS buffers, calls Libxc
 * for non-zero exchange/correlation IDs, and masks vacuum points with \f$n_{\text{tot}} \le 10^{-9}\f$.
 *
 * @par Invariants
 * Vacuum grid points emit exact zeros; thread-safe across concurrent XLA calls.
 *
 * @par SIESTA Mapping
 * Shared host Libxc bridge replacing \c libgridxc meta-GGA calls for both CPU and GPU backends.
 *
 * @param n_grid Number of real-space mesh points.
 * @param r_ptr Spin density buffer ``(2, n_grid)``.
 * @param s_ptr Gradient invariant buffer ``(3, n_grid)``.
 * @param t_ptr Kinetic energy density buffer ``(2, n_grid)``.
 * @param x_id Libxc exchange functional ID (or 0).
 * @param c_id Libxc correlation functional ID (or 0).
 * @param n_spin_pol Effective Libxc spin polarization (1 or 2).
 * @param e_ptr Output energy density ``(n_grid,)``.
 * @param vr_ptr Output density potential ``(2, n_grid)``.
 * @param vs_ptr Output gradient invariant derivative ``(3, n_grid)``.
 * @param vt_ptr Output kinetic energy density derivative ``(2, n_grid)``.
 * @return 0 on success, 1 if Libxc support is unavailable, 2 if an ID is invalid.
 */
int evaluate_mgga_libxc_host(
    int64_t n_grid,
    const double* r_ptr,
    const double* s_ptr,
    const double* t_ptr,
    int64_t x_id,
    int64_t c_id,
    int64_t n_spin_pol,
    double* e_ptr,
    double* vr_ptr,
    double* vs_ptr,
    double* vt_ptr);

/**
 * @brief Host Libxc LDA/GGA pointwise evaluator shared by CPU and CUDA FFI handlers.
 *
 * @par Theory
 * Evaluates semilocal LDA (Slater + PZ81 / PW92) and GGA (PBE / PBEsol) exchange and
 * correlation energy density \f$e_{\text{xc}}\f$ and derivatives \f$\partial e_{\text{xc}}/\partial n_\sigma\f$,
 * \f$\partial e_{\text{xc}}/\partial \gamma_{\alpha\beta}\f$ via Libxc.
 *
 * @par Algorithm
 * Maps \c xc_id and \c n_spin_pol to Libxc functional IDs, packs unpolarized or collinear
 * densities and gradient invariants, evaluates Libxc \c xc_lda_exc_vxc / \c xc_gga_exc_vxc,
 * and zeros vacuum points with \f$n_{\text{tot}} \le 10^{-14}\f$.
 *
 * @par Invariants
 * Produces bit-identical Libxc outputs on CPU and CUDA hosts; vacuum points emit exact zeros.
 *
 * @par SIESTA Mapping
 * Shared host Libxc bridge replacing \c libgridxc LDA/GGA calls for both CPU and GPU backends.
 *
 * @param n_grid Number of real-space mesh points.
 * @param r_ptr Spin density buffer ``(2, n_grid)``.
 * @param s_ptr Gradient invariant buffer ``(3, n_grid)``.
 * @param xc_id Functional selector (1=PZ81, 2=PW92, 3=PBE, 4=PBEsol).
 * @param n_spin_pol Effective Libxc spin polarization (1 or 2).
 * @param e_ptr Output energy density ``(n_grid,)``.
 * @param vr_ptr Output density potential ``(2, n_grid)``.
 * @param vs_ptr Output gradient invariant derivative ``(3, n_grid)``.
 * @return 0 on success, 1 if Libxc support is unavailable.
 */
int evaluate_xc_libxc_host(
    int64_t n_grid,
    const double* r_ptr,
    const double* s_ptr,
    int64_t xc_id,
    int64_t n_spin_pol,
    double* e_ptr,
    double* vr_ptr,
    double* vs_ptr);

}  // namespace jsiesta
