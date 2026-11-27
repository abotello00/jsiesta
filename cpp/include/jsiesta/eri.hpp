/**
 * @file eri.hpp
 * @brief Four-centre electron repulsion integrals over contracted Gaussian shells of real solid harmonics.
 *
 * @par Theory
 * A shell is \f$\chi_{lm}(\mathbf{r}) = C_{lm}(\mathbf{r}-\mathbf{A}) \sum_i c_i e^{-a_i |\mathbf{r}-\mathbf{A}|^2}\f$
 * with \f$C_{lm} = r^l Y_{lm}\f$ the real solid harmonics of \c math_utils.hpp (jsiesta's \f$m\f$ order and sign
 * convention, no further normalisation). The integrals are
 * \f$(ab|cd) = \iint \chi_a(\mathbf{r})\chi_b(\mathbf{r})\, w(|\mathbf{r}-\mathbf{r}'|)\,
 * \chi_c(\mathbf{r}')\chi_d(\mathbf{r}')\, d\mathbf{r}\, d\mathbf{r}'\f$ with the kernel
 * \f$w(r) = c_\mathrm{full}/r + c_\mathrm{sr}\,\mathrm{erfc}(\omega r)/r\f$.
 *
 * @par Algorithm
 * McMurchie-Davidson. A shell pair is expanded in Hermite Gaussians at the product centre \f$\mathbf{P}\f$ with
 * coefficients \f$E^{ab}_{tuv}\f$; the integral of two Hermite Gaussians is
 * \f$2\pi^{5/2} / (pq\sqrt{p+q})\, (-1)^{\tau+\nu+\phi} R_{t+\tau,u+\nu,v+\phi}(\alpha, \mathbf{P}-\mathbf{Q})\f$,
 * \f$\alpha = pq/(p+q)\f$, where \f$R\f$ follows by recursion from \f$R^n_{000} = (-2\alpha)^n F_n(\alpha R^2)\f$ and
 * \f$F_n\f$ is the Boys function. The error-function kernel \f$\mathrm{erf}(\omega r)/r\f$ gives the same recursion
 * from \f$\sqrt{\alpha'/\alpha}\,(-2\alpha')^n F_n(\alpha' R^2)\f$, \f$\alpha' = \alpha\omega^2/(\alpha+\omega^2)\f$,
 * so the two kernels are combined in the starting values.
 *
 * @par Invariants
 * \f$(ab|cd) = (ba|cd) = (ab|dc) = (cd|ab)\f$; \f$(ab|ab) \ge 0\f$ for \f$w \ge 0\f$; invariance under a common
 * translation of the four centres.
 *
 * @par SIESTA Mapping
 * None in SIESTA. HONPAS computes the same integrals with libint after fitting the numerical orbitals with
 * Gaussians (NAO2GTO).
 */
#pragma once

#include <cstdint>
#include <vector>

namespace jsiesta::eri {

/// Highest angular momentum of a shell.
constexpr int kMaxL = 4;

/**
 * @brief Coulomb kernel \f$w(r) = c_\mathrm{full}/r + c_\mathrm{sr}\,\mathrm{erfc}(\omega r)/r\f$.
 *
 * @par Theory
 * The full-range and the short-range part of a range-separated hybrid in the convention of Libxc.
 *
 * @par Algorithm
 * Plain record read by \c eri_quartet.
 *
 * @par Invariants
 * A zero \c c_sr or \c omega means the plain Coulomb kernel.
 *
 * @par SIESTA Mapping
 * None in SIESTA; HONPAS has the HSE06 values built in.
 */
struct Kernel {
    double c_full = 1.0; /**< Weight of the full-range Coulomb kernel. */
    double c_sr = 0.0;   /**< Weight of the short-range kernel. */
    double omega = 0.0;  /**< Range-separation parameter in Bohr^-1 (used when \c c_sr is not zero). */
    double prim_cut = 0.0; /**< Primitive quartets with (largest Hermite coefficients) x prefactor below this are skipped. */
};

/**
 * @brief Number of Hermite functions \f$(t,u,v)\f$ with \f$t+u+v \le L\f$.
 *
 * @par Theory
 * A product of two shells of total angular momentum \f$L\f$ needs Hermite Gaussians up to order \f$L\f$.
 *
 * @par Algorithm
 * Closed form \f$(L+1)(L+2)(L+3)/6\f$.
 *
 * @par Invariants
 * Positive and increasing in \f$L\f$.
 *
 * @par SIESTA Mapping
 * None.
 */
inline int n_hermite(int L) { return (L + 1) * (L + 2) * (L + 3) / 6; }

/**
 * @brief Hermite expansion of the product of two contracted shells.
 *
 * @par Theory
 * \f$\chi_a\chi_b = \sum_{tuv} E^{ab}_{tuv}\Lambda_{tuv}\f$ per primitive pair, with Hermite Gaussians at the product centre.
 *
 * @par Algorithm
 * Filled by \c make_shell_pair or \c make_shell_pair_mode; primitive pairs whose prefactor \f$|c_i c_j| e^{-\mu R_{AB}^2}\f$ is below the tolerance are dropped.
 *
 * @par Invariants
 * The arrays have one entry (or block) per primitive pair kept.
 *
 * @par SIESTA Mapping
 * None in SIESTA; a shell pair of libint in HONPAS.
 */
struct ShellPair {
    int la = 0;      /**< Angular momentum of the first shell. */
    int lb = 0;      /**< Angular momentum of the second shell. */
    int l_herm = 0;  /**< Highest Hermite order: la + lb, plus one for a derivative expansion. */
    int n_ab = 1;    /**< \f$(2l_a+1)(2l_b+1)\f$. */
    int n_herm = 1;  /**< \c n_hermite(l_herm). */
    int n_prim = 0;  /**< Number of primitive pairs kept. */
    std::vector<double> p;      /**< Exponent sum of each primitive pair. */
    std::vector<double> centre; /**< Product centre of each primitive pair, 3 per pair. */
    std::vector<double> e;      /**< Hermite coefficients, shape (n_prim, n_herm, n_ab), contraction included. */
    std::vector<double> e_max;  /**< Largest absolute Hermite coefficient of each primitive pair. */
};

/**
 * @brief Boys function \f$F_n(T) = \int_0^1 t^{2n} e^{-T t^2} dt\f$ for \f$n = 0..n_\mathrm{max}\f$.
 *
 * @par Theory
 * The starting values of the Hermite integrals.
 *
 * @par Algorithm
 * Below \f$T = 30\f$: Taylor expansion (eight terms) about the nearest point of a table with spacing 0.05, built
 * once from the series and downward recursion; error below 1e-14. Above: closed form of \f$F_0\f$ and upward
 * recursion.
 *
 * @par Invariants
 * \f$F_n(T) > F_{n+1}(T) > 0\f$; \f$F_n(0) = 1/(2n+1)\f$.
 *
 * @par SIESTA Mapping
 * None.
 *
 * @param n_max Highest order (at most 4 * kMaxL + 1).
 * @param t Argument, non-negative.
 * @param f Output, length \p n_max + 1.
 */
void boys(int n_max, double t, double* f);

/**
 * @brief Build the Hermite expansion of a shell pair.
 *
 * @par Theory
 * See \c ShellPair.
 *
 * @par Algorithm
 * \c make_shell_pair_mode with \c PairMode::kValue.
 *
 * @par Invariants
 * Symmetric under the exchange of the two shells up to the order of the indices.
 *
 * @par SIESTA Mapping
 * None in SIESTA; HONPAS shell-pair set-up.
 *
 * @param la,lb Angular momenta (at most \c kMaxL).
 * @param npa,npb Numbers of primitives.
 * @param exp_a,exp_b Exponents in Bohr^-2.
 * @param coef_a,coef_b Contraction coefficients of the solid harmonic times the bare Gaussian.
 * @param ca,cb Centres in Bohr.
 * @param prim_tol Primitive pairs whose coefficient product times Gaussian prefactor is below this are dropped.
 * @param out Result.
 */
void make_shell_pair(int la, int npa, const double* exp_a, const double* coef_a, const double* ca,
                     int lb, int npb, const double* exp_b, const double* coef_b, const double* cb,
                     double prim_tol, ShellPair& out);

/**
 * @brief What a shell-pair expansion represents: the product of two shells or one of its derivatives.
 *
 * @par Theory
 * Forces and stress of the exchange energy need the derivative of a shell pair with respect to its first centre
 * and with respect to a common translation of both centres; the derivative with respect to the second centre is
 * their difference.
 *
 * @par Algorithm
 * Selector of \c make_shell_pair_mode.
 *
 * @par Invariants
 * The values 1 to 3 and 4 to 6 are in the order x, y, z.
 *
 * @par SIESTA Mapping
 * None.
 */
enum class PairMode : int {
    kValue = 0,   /**< The product \f$\chi_a\chi_b\f$. */
    kFirstX = 1,  /**< Its derivative with respect to the x coordinate of the first centre. */
    kFirstY = 2,  /**< Same, y. */
    kFirstZ = 3,  /**< Same, z. */
    kPairX = 4,   /**< Its derivative with respect to a common x translation of both centres. */
    kPairY = 5,   /**< Same, y. */
    kPairZ = 6,   /**< Same, z. */
};

/**
 * @brief Hermite expansion of a shell pair or of one of its derivatives with respect to the centres.
 *
 * @par Theory
 * \f$\partial_{A_x} [x_A^i e^{-a x_A^2}] = 2a\,x_A^{i+1} e^{-a x_A^2} - i\,x_A^{i-1} e^{-a x_A^2}\f$ gives the
 * derivative with respect to the first centre from the one-dimensional Hermite coefficients of the neighbouring
 * powers. A common translation of both centres moves the product centre: \f$\partial_{P_x}\Lambda_{tuv} =
 * \Lambda_{t+1,u,v}\f$, an index shift. The derivative with respect to the second centre is their difference.
 *
 * @par Algorithm
 * As \c make_shell_pair, with the factor of the differentiated direction replaced; the Hermite order is one higher.
 * The result is used by \c eri_quartet like any shell pair.
 *
 * @par Invariants
 * Mode \c kValue reproduces \c make_shell_pair. The primitive pairs kept are the same in every mode.
 *
 * @par SIESTA Mapping
 * None in SIESTA; derivative integrals of HONPAS (libint).
 *
 * @param la,lb Angular momenta (at most \c kMaxL).
 * @param npa,npb Numbers of primitives.
 * @param exp_a,exp_b Exponents in Bohr^-2.
 * @param coef_a,coef_b Contraction coefficients of the solid harmonic times the bare Gaussian.
 * @param ca,cb Centres in Bohr.
 * @param prim_tol Primitive pairs whose coefficient product times Gaussian prefactor is below this are dropped.
 * @param mode What to expand.
 * @param out Result.
 */
void make_shell_pair_mode(int la, int npa, const double* exp_a, const double* coef_a, const double* ca,
                          int lb, int npb, const double* exp_b, const double* coef_b, const double* cb,
                          double prim_tol, PairMode mode, ShellPair& out);

/**
 * @brief Contracted integrals of several expansions of one bra pair with one ket pair.
 *
 * @par Theory
 * The Hermite integrals and their contraction with the ket depend on the bra only through its primitive
 * exponents and centres. The value and derivative expansions of one shell pair share these, so the integrals
 * of all of them cost one pass at the highest Hermite order plus one contraction per expansion.
 *
 * @par Algorithm
 * As \c eri_quartet, with the last contraction repeated for each bra expansion.
 *
 * @par Invariants
 * The expansions must come from \c make_shell_pair_mode for the same two shells, centres and tolerance, with the
 * same Hermite order (the six derivative modes, or the value alone). An expansion with no primitive is skipped
 * and its output is zero. Thread-safe with one \c work vector per thread.
 *
 * @par SIESTA Mapping
 * None in SIESTA; libderiv in HONPAS.
 *
 * @param bras Expansions of the bra pair; the first must have primitives.
 * @param n_bra Number of expansions.
 * @param ket Second shell pair.
 * @param ket_shift Translation added to the centres of the ket pair (a lattice vector), Bohr.
 * @param kernel Coulomb kernel.
 * @param out Output, \c n_bra blocks of shape (n_a, n_b, n_c, n_d); overwritten.
 * @param work Scratch storage, resized as needed (one per thread).
 */
void eri_quartet_multi(const ShellPair* bras, int n_bra, const ShellPair& ket, const double* ket_shift,
                       const Kernel& kernel, double* out, std::vector<double>& work);

/**
 * @brief Contracted integrals \f$(ab|cd)\f$ of two shell pairs.
 *
 * @par Theory
 * See the file description. Either pair may be a derivative expansion (\c make_shell_pair_mode).
 *
 * @par Algorithm
 * For each primitive pair of the bra, the Hermite integrals with all primitive pairs of the ket are contracted with the ket coefficients, then once with the bra coefficients. An AVX2 copy is selected at load time where available.
 *
 * @par Invariants
 * Thread-safe with one \c work vector per thread.
 *
 * @par SIESTA Mapping
 * None in SIESTA; libint in HONPAS.
 *
 * @param bra First shell pair.
 * @param ket Second shell pair.
 * @param ket_shift Translation added to the centres of the ket pair (a lattice vector), Bohr.
 * @param kernel Coulomb kernel.
 * @param out Output, shape (n_a, n_b, n_c, n_d) in row-major order, m from -l to l in each index; overwritten.
 * @param work Scratch storage, resized as needed (one per thread).
 */
void eri_quartet(const ShellPair& bra, const ShellPair& ket, const double* ket_shift, const Kernel& kernel,
                 double* out, std::vector<double>& work);

}  // namespace jsiesta::eri

extern "C" {

/**
 * @brief Integrals \f$(ab|cd)\f$ of four contracted shells (test and reference entry point).
 *
 * @par Theory
 * See the file description.
 *
 * @par Algorithm
 * Two calls of \c make_shell_pair and one of \c eri_quartet, without primitive screening.
 *
 * @par Invariants
 * Returns 0 on success, 2 for an angular momentum above \c kMaxL or a non-positive number of primitives.
 *
 * @par SIESTA Mapping
 * None.
 *
 * @param l Angular momenta of the four shells.
 * @param n_prim Numbers of primitives of the four shells.
 * @param exps Exponents of the four shells, concatenated (Bohr^-2).
 * @param coefs Contraction coefficients, concatenated like \p exps.
 * @param centres Centres, shape (4, 3), Bohr.
 * @param c_full,c_sr,omega Weights of the full-range and short-range kernels and the range parameter (Bohr^-1).
 * @param out Output, one index per shell with 2l+1 values each, row-major.
 */
int32_t jsiesta_eri_quartet(const int32_t* l, const int32_t* n_prim, const double* exps, const double* coefs,
                            const double* centres, double c_full, double c_sr, double omega, double* out);

/**
 * @brief Integrals of four contracted shells with the bra pair replaced by one of its derivatives.
 *
 * @par Theory
 * Mode 0 is \f$(ab|cd)\f$; modes 1 to 3 are its derivatives with respect to the x, y, z coordinates of the
 * centre of the first shell; modes 4 to 6 with respect to a common translation of the first two centres.
 *
 * @par Algorithm
 * \c make_shell_pair_mode for the bra pair, then \c eri_quartet.
 *
 * @par Invariants
 * Returns 0 on success, 2 for invalid input. Mode 0 equals \c jsiesta_eri_quartet.
 *
 * @par SIESTA Mapping
 * None.
 *
 * @param l Angular momenta of the four shells.
 * @param n_prim Numbers of primitives of the four shells.
 * @param exps Exponents of the four shells, concatenated (Bohr^-2).
 * @param coefs Contraction coefficients, concatenated like exps.
 * @param centres Centres, shape (4, 3), Bohr.
 * @param c_full,c_sr,omega Weights of the full-range and short-range kernels and the range parameter (Bohr^-1).
 * @param bra_mode Value of \c jsiesta::eri::PairMode.
 * @param out Output, one index per shell with 2l+1 values each, row-major.
 */
int32_t jsiesta_eri_quartet_mode(const int32_t* l, const int32_t* n_prim, const double* exps, const double* coefs,
                                 const double* centres, double c_full, double c_sr, double omega, int32_t bra_mode,
                                 double* out);

}  // extern "C"
