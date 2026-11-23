r"""Physics-informed SCF conditioning descriptors and parameter recommender (``scf_params``).

Theory:
    Estimates the physical conditioning of the Kohn-Sham fixed-point problem from
    structure, basis, and spin mode before SCF starts:

    - Long-wavelength Coulomb condition number :math:`\kappa_\varepsilon \sim 1 + (k_{\text{TF}} L_{\text{eff}} / 2\pi)^2`
      driving :math:`1/q^2` charge sloshing in extended cells and metals
    - Open :math:`d/f`-shell stiffness (:math:`f_d + 1.5 f_f`) driving multipole/spin coupling
    - Vacuum / slab dimensionality factor capturing surface dipole sloshing
    - Electronegativity variance across species
    - Spin/SOC/BdG mode coupling

Algorithm:
    Computes dimensionless descriptors and recommends cold-start mixer hyperparameters
    (:math:`w_{\text{base}}`, :math:`N_{\text{hist}}`, :math:`\lambda_{\text{Tikhonov}}`,
    ``mix_target="Hamiltonian"``), explicitly tagging heuristic priors vs calibrated
    surrogate outputs.

Invariants:
    - Recommended mixing weight satisfies :math:`w_{\text{base}} \in [0.020, 0.250]`.
    - Recommended Krylov depth satisfies :math:`N_{\text{hist}} \in [4, 12]`.

SIESTA Mapping:
    Directly incorporates ``scf_params/features.py`` and ``scf_params/recommender.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from jax_siesta.core.types import BasisData, SCFConfig, Structure

# Pauling electronegativity estimates for common elements (fallback 1.9)
_PAULING_EN: dict[int, float] = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 14: 1.90, 18: 3.00, 27: 1.88,
}


@dataclass(frozen=True)
class ConditioningDescriptors:
    """Physical conditioning descriptors of the SCF fixed-point operator.

    Theory:
        Summarizes Coulomb sloshing scale, localized :math:`d/f` stiffness, slab
        vacuum fraction, electronegativity spread, and spin complexity.

    Algorithm:
        Computed once at initialization from ``Structure``, ``BasisData``, and ``SCFConfig``.

    Invariants:
        All descriptors are finite and non-negative.

    SIESTA Mapping:
        Mirrors ``scf_params/features.py`` feature vector.
    """

    kappa_eps: float
    l_eff_bohr: float
    df_stiffness: float
    vacuum_fraction: float
    en_variance: float
    is_slab_or_molecule: bool
    nspin: int


@dataclass(frozen=True)
class RecommendedSCFParams:
    r"""Physics-informed cold-start SCF mixing recommendation.

    Theory:
        Provides safe, well-conditioned defaults scaled inversely with
        :math:`\ln \kappa_\varepsilon` and :math:`d/f` stiffness.

    Algorithm:
        Maps ``ConditioningDescriptors`` into bounded mixer hyperparameters.

    Invariants:
        - ``0.020 <= w_base <= 0.250``.
        - ``4 <= n_hist <= 12``.

    SIESTA Mapping:
        Mirrors ``scf_params/recommender.py`` output contract.
    """

    w_base: float
    n_hist: int
    tikhonov_reg: float
    mix_target: str
    mix_spin: str
    is_heuristic_prior: bool = True
    descriptors: ConditioningDescriptors | None = None


#: Half-extent (Bohr) attributed to each atom when measuring empty space; roughly a valence orbital radius.
_VACUUM_ATOM_RADIUS_BOHR = 3.0


def estimate_vacuum_fraction(structure: Structure) -> float:
    r"""Fraction of the cell that is vacuum, from the largest empty gap along each lattice direction.

    Theory:
        Along lattice direction :math:`a` the distance between lattice planes is
        :math:`d_a = 1/|\mathbf b_a|` (:math:`\mathbf b` reciprocal vectors without :math:`2\pi`). The largest
        periodic gap :math:`g_a` between sorted fractional coordinates, minus one atomic diameter, is empty:
        :math:`v_a = \max(0, g_a d_a - 2 r_{\text{at}})/d_a`. The vacuum fraction is
        :math:`1 - \prod_a (1 - v_a)`. Bulk crystals give 0; slabs and molecules give their vacuum share.

    Algorithm:
        Wrap fractional coordinates into [0, 1), sort per axis, take the largest gap including the periodic
        wrap-around, convert to a length with :math:`d_a`. Non-periodic axes count as fully padded by the box.

    Parameters:
        structure: ``Structure`` PyTree.

    Returns:
        Vacuum fraction in [0, 0.95].

    Invariants:
        Invariant under rigid translations and atom permutations. Diamond Si -> 0.0
        (the former coordinate-spread estimate returned 0.875 and classified every bulk case as a slab; review M2).

    SIESTA Mapping:
        None (replaces the heuristic in ``scf_params/features.py``).
    """
    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    frac = (np.linalg.solve(cell, pos.T).T) % 1.0
    plane_spacing = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=1)
    keep = 1.0
    for a in range(3):
        xs = np.sort(frac[:, a])
        gaps = np.diff(np.concatenate([xs, [xs[0] + 1.0]])) if len(xs) > 0 else np.array([1.0])
        empty = max(0.0, float(np.max(gaps)) * plane_spacing[a] - 2.0 * _VACUUM_ATOM_RADIUS_BOHR)
        keep *= 1.0 - min(empty / plane_spacing[a], 1.0)
    return float(np.clip(1.0 - keep, 0.0, 0.95))

def estimate_conditioning_descriptors(
    structure: Structure,
    basis: BasisData,
    config: SCFConfig,
) -> ConditioningDescriptors:
    r"""Compute physical SCF conditioning descriptors from structure and basis.

    Theory:
        Estimates Thomas-Fermi wavevector :math:`k_{\text{TF}} = (4 k_F / \pi)^{1/2}`
        from average valence density :math:`\bar{n} = N_e / \Omega_{\text{active}}`,
        long-wavelength condition number :math:`\kappa_\varepsilon = 1 + (k_{\text{TF}} L_{\text{eff}} / 2\pi)^2`,
        and orbital angular-momentum stiffness :math:`f_d + 1.5 f_f`.

    Algorithm:
        Inspects lattice column lengths, active valence density, orbital ``l`` fractions,
        and Pauling electronegativity variance.

    Parameters:
        structure: ``Structure`` PyTree.
        basis: ``BasisData`` PyTree.
        config: ``SCFConfig``.

    Returns:
        ``ConditioningDescriptors`` instance.

    Invariants:
        ``kappa_eps >= 1.0`` and ``0.0 <= df_stiffness <= 1.5``.

    SIESTA Mapping:
        Replaces heuristic descriptor extraction in ``scf_params/features.py``.
    """
    cell = np.asarray(structure.cell, dtype=np.float64)
    col_norms = np.linalg.norm(cell, axis=0)
    l_eff = float(np.max(col_norms))
    vol = max(float(abs(np.linalg.det(cell))), 1.0)

    vac_frac = estimate_vacuum_fraction(structure)

    n_e = max(basis.total_valence_electrons - config.net_charge, 0.5)
    active_vol = vol * max(1.0 - vac_frac, 0.1)
    n_avg = n_e / active_vol
    k_f = (3.0 * (math.pi**2) * n_avg) ** (1.0 / 3.0)
    k_tf = math.sqrt(4.0 * k_f / math.pi)
    kappa_eps = 1.0 + (k_tf * l_eff / (2.0 * math.pi)) ** 2

    n_orb = max(basis.n_orb, 1)
    f_d = sum(1 for l_q in basis.orb_l if l_q == 2) / n_orb
    f_f = sum(1 for l_q in basis.orb_l if l_q >= 3) / n_orb
    df_stiff = f_d + 1.5 * f_f

    ens = np.array([_PAULING_EN.get(z, 1.90) for z in structure.atomic_numbers], dtype=np.float64)
    en_var = float(np.var(ens)) if len(ens) > 1 else 0.0

    is_slab_or_mol = (not all(structure.pbc)) or (vac_frac > 0.35)
    return ConditioningDescriptors(
        kappa_eps=kappa_eps,
        l_eff_bohr=l_eff,
        df_stiffness=df_stiff,
        vacuum_fraction=vac_frac,
        en_variance=en_var,
        is_slab_or_molecule=is_slab_or_mol,
        nspin=config.nspin,
    )


def recommend_scf_defaults(
    structure: Structure,
    basis: BasisData,
    config: SCFConfig,
) -> RecommendedSCFParams:
    r"""Recommend safe cold-start SCF mixing parameters honoring explicit user overrides.

    Theory:
        Defaults to Hamiltonian mixing (``mix_target="Hamiltonian"``) because
        grid potentials are pre-smoothed, scales :math:`w_{\text{base}}` inversely
        with :math:`\ln \kappa_\varepsilon` and :math:`d/f` stiffness, sets Krylov
        depth :math:`N_{\text{hist}} \in [4, 12]`, and sets Tikhonov regularization
        :math:`\lambda = 1.0\times 10^{-3}` (bulk) or :math:`1.5\times 10^{-3}` (slabs/SOC).

    Algorithm:
        Computes ``ConditioningDescriptors`` and merges them with any non-None
        user overrides in ``SCFConfig``.

    Parameters:
        structure: ``Structure`` PyTree.
        basis: ``BasisData`` PyTree.
        config: ``SCFConfig`` (user overrides take precedence).

    Returns:
        ``RecommendedSCFParams`` instance.

    Invariants:
        - ``0.020 <= w_base <= 0.250`` when not explicitly overridden.
        - ``4 <= n_hist <= 12`` when not explicitly overridden.

    SIESTA Mapping:
        Implements ``scf_params/recommender.py`` policy.
    """
    desc = estimate_conditioning_descriptors(structure, basis, config)

    # Base weight inversely scaled with ln(kappa_eps) and d/f stiffness
    denom = 1.0 + 0.55 * math.log1p(desc.kappa_eps) + 2.2 * desc.df_stiffness + 0.5 * desc.en_variance
    if desc.nspin in (4, 8):
        denom += 0.4
    w_auto = float(np.clip(0.28 / denom, 0.020, 0.250))

    n_hist_auto = 6
    if desc.kappa_eps > 4.0 or desc.df_stiffness > 0.15 or desc.is_slab_or_molecule:
        n_hist_auto = 8
    if desc.nspin in (4, 8) or (desc.df_stiffness > 0.25 and desc.kappa_eps > 6.0):
        n_hist_auto = 10
    n_hist_auto = int(np.clip(n_hist_auto, 4, 12))

    reg_auto = 1.5e-3 if (desc.is_slab_or_molecule or desc.nspin == 8) else 1.0e-3

    return RecommendedSCFParams(
        w_base=float(config.mix_weight if config.mix_weight is not None else w_auto),
        n_hist=int(config.mix_history if config.mix_history is not None else n_hist_auto),
        tikhonov_reg=float(config.tikhonov_reg if config.tikhonov_reg is not None else reg_auto),
        mix_target=config.mix_target,
        mix_spin=config.mix_spin,
        is_heuristic_prior=(config.mix_weight is None and config.mix_history is None),
        descriptors=desc,
    )
