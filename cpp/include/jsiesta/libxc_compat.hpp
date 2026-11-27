/**
 * @file libxc_compat.hpp
 * @brief One classification of Libxc functionals for every supported Libxc release (5.1 to 7.x).
 *
 * @par Theory
 * A hybrid functional is a semilocal functional (LDA, GGA or meta-GGA ingredients) plus a fraction of exact
 * exchange. The grid kernels need the semilocal family; the exact-exchange stage needs to know that it is a hybrid.
 *
 * @par Algorithm
 * Libxc 5, 6 and the 7.x releases mark hybrids with the families \c XC_FAMILY_HYB_LDA, \c XC_FAMILY_HYB_GGA and
 * \c XC_FAMILY_HYB_MGGA. The development line that preceded 7.0 dropped these families for \c xc_hyb_type(); that
 * interface was not released. The code tests for the macros of the header it is compiled against, never for the
 * version number.
 *
 * @par Invariants
 * \c libxc_semilocal_family returns \c XC_FAMILY_LDA, \c XC_FAMILY_GGA or \c XC_FAMILY_MGGA for every functional
 * with one of these ingredient sets, hybrid or not, and the input family otherwise.
 *
 * @par SIESTA Mapping
 * \c libgridxc classifies functionals in \c gridxc_setXC; it has no hybrids.
 */
#pragma once

#ifdef JSIESTA_HAS_LIBXC
#include <xc.h>

namespace jsiesta {

/// Semilocal family of a Libxc family value (hybrid families folded onto LDA / GGA / MGGA).
inline int libxc_semilocal_family(int family) {
#ifdef XC_FAMILY_HYB_LDA
    if (family == XC_FAMILY_HYB_LDA) return XC_FAMILY_LDA;
#endif
#ifdef XC_FAMILY_HYB_GGA
    if (family == XC_FAMILY_HYB_GGA) return XC_FAMILY_GGA;
#endif
#ifdef XC_FAMILY_HYB_MGGA
    if (family == XC_FAMILY_HYB_MGGA) return XC_FAMILY_MGGA;
#endif
    return family;
}

/// True when the initialised functional \p func mixes exact exchange.
inline bool libxc_is_hybrid(const xc_func_type* func) {
#if defined(XC_FAMILY_HYB_GGA)
    const int family = xc_func_info_get_family(func->info);
    return libxc_semilocal_family(family) != family;
#else
    return xc_hyb_type(func) != XC_HYB_NONE;
#endif
}

}  // namespace jsiesta
#endif  // JSIESTA_HAS_LIBXC
