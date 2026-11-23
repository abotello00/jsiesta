r"""Basis Enthalpy (Junquera et al.) optimization with bijective constraints.

Theory:
    Finds optimal Numerical Atomic Orbital (NAO) basis parameters by minimizing the
    SIESTA **Basis Enthalpy** functional (Junquera et al., *Phys. Rev. B* **64**, 235111,
    2001):

    .. math::
        H_{\text{basis}}(\mathbf{p}) = E_{\text{KS}}(\mathbf{p})
        + P_{\text{basis}} \sum_{I=1}^{N_{\text{atoms}}}
          \sum_{(n,l,\zeta)\in I} \frac{4\pi}{3} r_c(I,n,l,\zeta)^3,

    over the explicit per-shell parameter set requested for transferable NAO optimization:

    - Explicit outer & inner cutoff radii :math:`r_c^{(z)}` per :math:`\zeta` per shell
      (using direct parabolic splitting rather than implicit ``split_norm`` algorithms)
    - Soft confinement potential depth :math:`V_0` per shell (Hartree)
    - Soft confinement inner radius :math:`r_i` per shell (Bohr)
    - Net atomic ionic charge :math:`q` per species (:math:`\Delta Q` adjusting atomic
      reference occupations during orbital generation)

Algorithm:
    Uses the unconstrained-to-bounded arccos bijection with dynamic ``depends_on`` upper
    bounds and bottom-up cascading lower bounds (ported from ``SiestaOpt``) so every
    optimizer vector decodes to a physically valid, strictly ordered set of radii
    (:math:`r_c^{(1)} > r_c^{(2)} > \dots > r_i \ge 0`). Each objective evaluation
    generates radial PAOs directly in memory via :func:`generate_species_from_psml`,
    executes ``run_scf`` + post-SCF ``DM_out`` evaluation inside ``jax_siesta``, and
    supports ``"nm"``, ``"powell"``, ``"de"``, and two-stage ``"de+nm"`` solvers.

Invariants:
    - Orbital shell volume :math:`V_{\text{basis}}` counts each :math:`(n, l, \zeta)` radial
      shell once per atom (not multiplied by :math:`2l+1`), matching ``basis_enthalpy.f90``.
    - Every decoded parameter point satisfies all ordering and bound invariants by construction
      (zero invalid basis sets).
    - Float64 ``np.clip(|cos(u)|, 0, 1)`` prevents cosine rounding overshoot at multiples of :math:`\pi`.

SIESTA Mapping:
    Combines SIESTA's ``Src/basis_enthalpy.f90`` enthalpy objective and ``SiestaOpt`` /
    ``simplex-0.9.6`` bijective parameter space inside pure JAX-SIESTA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import differential_evolution, minimize

from jax_siesta.core.types import BasisData, HARTREE_TO_EV, SCFConfig, Structure
from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.pao_generator import (
    ShellBasisSpec,
    SpeciesBasisSpec,
    _parse_psml_for_atom,
    generate_species_from_psml,
)
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.symmetry.engine import NativeSymmetrySetupCache

# 1 GPa in eV / Bohr^3, from SIESTA's own units (Src/libunits/units_codata2018.f90, internal energy unit Ry):
# GPa = 6.79786184348648780e-5 Ry/Bohr^3 and eV = 7.34986443513115789e-2 Ry. Until 2026-11-22 this constant held the Ry
# value (6.798e-5) as if it were eV, so p V was 13.6x too small (D-040).
EV_PER_BOHR3_PER_GPA: float = 6.79786184348648780e-5 / 7.34986443513115789e-2
_L_NAMES = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}


def basis_volume_bohr3(structure: Structure, basis: BasisData) -> float:
    r"""Compute total orbital sphere volume :math:`V_{\text{basis}}` in :math:`\text{Bohr}^3`.

    Theory:
        Evaluates :math:`\sum_{I=1}^{N_{\text{atoms}}} \sum_{s \in \text{shells}(I)} \frac{4\pi}{3} r_c(I, s)^3`.

    Algorithm:
        Iterates unit-cell atoms, looks up each atom's species ``pao_rcut`` shell radii,
        and sums their spherical volumes.

    Parameters:
        structure: Atomic ``Structure``.
        basis: Constructed ``BasisData`` holding ``species[sp].pao_rcut``.

    Returns:
        Total orbital shell volume in :math:`\text{Bohr}^3`.

    Invariants:
        Counts each radial ``(n, l, zeta)`` shell once per atom regardless of :math:`2l+1`.

    SIESTA Mapping:
        Matches ``orb_volumes`` in SIESTA ``Src/basis_enthalpy.f90``.
    """
    vol = 0.0
    for sp_idx in structure.species_indices:
        sp = basis.species[sp_idx]
        for rc in sp.pao_rcut:
            vol += (4.0 / 3.0) * math.pi * (float(rc) ** 3)
    return float(vol)


def basis_enthalpy_ev(
    free_energy_ha: float,
    structure: Structure,
    basis: BasisData,
    pressure_gpa: float = 0.2,
) -> float:
    r"""Evaluate SIESTA Basis Enthalpy :math:`H_{\text{basis}} = E + P V_{\text{basis}}` in eV.

    Theory:
        Adds the geometric confinement penalty :math:`P_{\text{basis}} V_{\text{basis}}`
        to the post-SCF Kohn-Sham free energy :math:`E_{\text{free}}`.

    Algorithm:
        Converts ``free_energy_ha`` to eV and adds ``pressure_gpa * EV_PER_BOHR3_PER_GPA * V``.

    Parameters:
        free_energy_ha: Electronic free energy in Hartree.
        structure: Atomic ``Structure``.
        basis: Basis PyTree.
        pressure_gpa: Confining basis pressure in GPa (default ``0.2 GPa``).

    Returns:
        Basis enthalpy in eV.

    Invariants:
        Monotonically increases with cutoff radii for ``pressure_gpa > 0``.

    SIESTA Mapping:
        Matches ``write_basis_enthalpy`` in SIESTA ``Src/basis_enthalpy.f90``.
    """
    e_ev = float(free_energy_ha) * HARTREE_TO_EV
    vol = basis_volume_bohr3(structure, basis)
    return float(e_ev + float(pressure_gpa) * EV_PER_BOHR3_PER_GPA * vol)


def to_optimizer_u(x: float, lo: float, hi: float) -> float:
    """Map bounded physical value ``x in [lo, hi]`` into unconstrained space via ``arccos``.

    Theory:
        Inverts :math:`x = x_{\\text{lo}} + (x_{\\text{hi}} - x_{\\text{lo}})|\\cos u|`
        so initial physical guesses map smoothly onto unconstrained real vectors.

    Algorithm:
        Clips normalized fraction to ``[0, 1]`` and evaluates ``arccos(frac)``.

    Parameters:
        x: Physical parameter value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Unconstrained real coordinate ``u in [0, pi/2]``.

    Invariants:
        Raises ``ValueError`` if ``hi < lo - 1e-10``.

    SIESTA Mapping:
        Ports ``map_siesta_2_amoeba`` from ``simplex-0.9.6/Src/mapping.f`` and ``SiestaOpt``.
    """
    if hi < lo - 1e-10:
        raise ValueError(f"Infeasible parameter interval [{lo}, {hi}]")
    hi = max(hi, lo)
    if np.isclose(hi, lo):
        return 0.0
    frac = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    return float(np.arccos(frac))


def to_physical_x(u: float, lo: float, hi: float) -> float:
    """Map unconstrained scalar ``u`` back into physical interval ``[lo, hi]``.

    Theory:
        Evaluates the periodic smooth surjection :math:`x = x_{\\text{lo}} + (x_{\\text{hi}}-x_{\\text{lo}})|\\cos u|`.

    Algorithm:
        Clips ``abs(cos(u))`` to ``[0, 1]`` in float64 to prevent rounding overshoot beyond bounds.

    Parameters:
        u: Unconstrained coordinate.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Physical parameter strictly inside ``[lo, hi]``.

    Invariants:
        ``lo <= to_physical_x(u, lo, hi) <= hi`` for all finite ``u``.

    SIESTA Mapping:
        Ports ``map_amoeba_2_siesta`` with float64 safety clip from ``SiestaOpt``.
    """
    if hi < lo - 1e-10:
        raise ValueError(f"Infeasible parameter interval [{lo}, {hi}]")
    hi = max(hi, lo)
    factor = float(np.clip(abs(math.cos(float(u))), 0.0, 1.0))
    return float(lo + (hi - lo) * factor)


@dataclass(frozen=True)
class BasisParamSpec:
    """Specification of a single bounded optimization coordinate in a basis parameter vector.

    Theory:
        Represents one physical basis parameter (:math:`r_c`, :math:`V_0`, :math:`r_i`, or :math:`q`)
        whose upper bound is either static or dynamically dependent on an outer radius.

    Algorithm:
        Ordered so any spec with ``depends_on`` appears strictly after its parent key in
        :class:`MappedBasisSpace`.

    Invariants:
        - Either ``hi`` is a float or ``depends_on`` names an earlier parameter.

    SIESTA Mapping:
        Matches ``ParamSpec`` in ``SiestaOpt/src/siesta_opt/mapped_params.py``.
    """

    name: str
    lo: float
    hi: float | None
    initial: float
    depends_on: str | None = None
    gap: float = 0.0


def _static_upper_bound(spec: BasisParamSpec) -> float:
    """Return the fixed upper bound of an independent parameter.

    Parameters:
        spec: Parameter specification without ``depends_on``.

    Returns:
        ``spec.hi`` as a float.

    Raises:
        ValueError: If an independent parameter has no upper bound.

    Invariants:
        Only parameters with ``depends_on`` may leave ``hi`` unset; their bound is derived from the
        parent value minus ``gap``.
    """
    if spec.hi is None:
        raise ValueError(f"Parameter {spec.name!r} has no 'depends_on' and therefore needs an explicit 'hi'.")
    return float(spec.hi)


@dataclass(frozen=True)
class ShellOptConfig:
    """User configuration for one shell's basis optimization parameters.

    Theory:
        Defines orbital quantum numbers ``(n, l)``, explicit zeta count ``n_zeta``,
        initial outer radius guess ``rc_guess`` (or explicit ``rc_initials``), and initial
        soft-confinement values ``(v0_ha, ri_bohr)``.

    Algorithm:
        Consumed by :meth:`MappedBasisSpace.from_species_configs` to generate cascading
        ``rc_z1 > rc_z2 > ... > ri`` bounds.

    Invariants:
        - ``n_zeta >= 1``.
        - ``is_polarization=True`` sets ``polarization_mode`` (``"perturbative"`` or ``"eigenstate"``).

    SIESTA Mapping:
        Generalizes shell entries in ``PAO.Basis`` for automated basis optimization.
    """

    n: int
    l: int  # noqa: E741
    n_zeta: int = 2
    rc_guess: float = 6.0
    rc_initials: tuple[float, ...] | None = None
    v0_initial_ha: float = 20.0
    ri_initial_bohr: float = 3.0
    is_polarization: bool = False
    polarization_mode: Literal["perturbative", "eigenstate"] = "eigenstate"
    parent_l: int | None = None
    optimize_soft_confinement: bool = True


class MappedBasisSpace:
    r"""Bijective parameter mapping over explicit per-zeta ``rc``, ``V0``, ``ri``, and ``q``.

    Theory:
        Maps unconstrained real vectors :math:`\mathbf{u} \in \mathbb{R}^d` bijectively onto
        valid species basis parameters guaranteeing strict ordering:

        .. math::
            r_c^{(1)} > r_c^{(2)} + \Delta r_c, \quad \dots, \quad r_c^{(N_\zeta)} > r_i + \Delta r_i.

    Algorithm:
        Constructs bottom-up cascading lower bounds ``chain_lo`` per shell so multi-zeta
        intervals are never degenerate, decodes active coordinates sequentially, and converts
        results directly into :class:`SpeciesBasisSpec` objects.

    Invariants:
        - Every decoded parameter set is feasible by construction (0% invalid rate).
        - Frozen parameters are excluded from unconstrained vector dimension ``ndim``.

    SIESTA Mapping:
        Ports ``MappedParameterSpace`` from ``SiestaOpt/mapped_params.py`` tailored to
        in-memory ``jsiesta`` ``SpeciesBasisSpec`` objects.
    """

    def __init__(
        self,
        specs: list[BasisParamSpec],
        species_shells: dict[str, tuple[ShellOptConfig, ...]],
        xc: str = "PBE",
        frozen: dict[str, float] | None = None,
        log_grid_step: float = 0.004,
    ) -> None:
        self.specs = list(specs)
        self.species_shells = species_shells
        self.xc = xc
        self.log_grid_step = float(log_grid_step)
        self._intrinsic_frozen: dict[str, float] = {
            sp.name: float(sp.lo)
            for sp in self.specs
            if sp.hi is not None and np.isclose(sp.hi, sp.lo)
        }
        self._frozen: dict[str, float] = dict(self._intrinsic_frozen)
        if frozen:
            self._frozen.update({k: float(v) for k, v in frozen.items()})
        self._active_specs = [sp for sp in self.specs if sp.name not in self._frozen]

    @property
    def ndim(self) -> int:
        """Number of active unconstrained optimization dimensions."""
        return len(self._active_specs)

    @property
    def active_names(self) -> list[str]:
        """Ordered list of active parameter names."""
        return [sp.name for sp in self._active_specs]

    @classmethod
    def from_species_configs(
        cls,
        species_shells: dict[str, tuple[ShellOptConfig, ...]],
        *,
        xc: str = "PBE",
        optimize_rc: bool = True,
        optimize_v0: bool = True,
        optimize_ri: bool = True,
        optimize_q: bool = True,
        rc_lo_mult: float = 0.6,
        rc_hi_mult: float = 1.6,
        rc_abs_max: float = 9.5,
        rc_gap: float = 0.35,
        ri_lo: float = 0.5,
        ri_gap: float = 0.3,
        rc_inner_lo: float = 1.5,
        v0_bounds_ha: tuple[float, float] = (0.5, 60.0),
        q_bounds: tuple[float, float] = (-0.75, 1.5),
        q_initial: dict[str, float] | None = None,
    ) -> MappedBasisSpace:
        """Build a bijective parameter space for one or more species automatically.

        Theory:
            Precomputes bottom-up feasible lower bounds so multi-:math:`\\zeta` radii and
            soft-confinement inner radii always form non-empty valid intervals.

        Algorithm:
            Adds species net charge ``{sp}_q`` followed by each shell's ``rc_z1``, ``v0_ha``,
            ``rc_z2..rc_zN``, and ``ri`` in topological dependency order, freezing any groups
            disabled by flags.

        Parameters:
            species_shells: Mapping ``symbol -> tuple[ShellOptConfig, ...]``.
            xc: XC functional identifier used when generating PAOs.
            optimize_rc: Whether cutoff radii ``rc_z*`` are active.
            optimize_v0: Whether soft confinement depth ``v0_ha`` is active.
            optimize_ri: Whether soft confinement inner radius ``ri`` is active.
            optimize_q: Whether species ionic charge ``q`` is active.
            rc_lo_mult: Lower-bound multiplier on initial ``rc_guess`` for ``rc_z1``.
            rc_hi_mult: Upper-bound multiplier on initial ``rc_guess`` for ``rc_z1``.
            rc_abs_max: Global maximum orbital cutoff radius (Bohr).
            rc_gap: Minimum separation ``rc_z{k} - rc_z{k+1}`` (Bohr).
            ri_lo: Minimum soft confinement inner radius ``ri`` (Bohr).
            ri_gap: Minimum gap ``rc_innermost - ri`` (Bohr).
            rc_inner_lo: Absolute minimum cutoff for innermost zeta (Bohr).
            v0_bounds_ha: ``(lo, hi)`` bounds on soft confinement potential ``V0`` (Ha).
            q_bounds: ``(lo, hi)`` bounds on ionic net charge ``q``.
            q_initial: Optional initial ionic charge map per species.

        Returns:
            Configured :class:`MappedBasisSpace`.

        Invariants:
            Always yields non-degenerate intervals whenever ``rc_abs_max`` fits all gaps.

        SIESTA Mapping:
            Ports ``MappedParameterSpace.from_shell_structure`` from ``SiestaOpt``.
        """
        specs: list[BasisParamSpec] = []
        frozen: dict[str, float] = {}
        q_init_map = q_initial or {}

        for sp_lbl in sorted(species_shells.keys()):
            q_val = float(q_init_map.get(sp_lbl, 0.0))
            q_name = f"{sp_lbl}_q"
            specs.append(
                BasisParamSpec(
                    name=q_name,
                    lo=float(q_bounds[0]),
                    hi=float(q_bounds[1]),
                    initial=float(np.clip(q_val, q_bounds[0], q_bounds[1])),
                )
            )
            if not optimize_q:
                frozen[q_name] = float(np.clip(q_val, q_bounds[0], q_bounds[1]))

            for sh in species_shells[sp_lbl]:
                ll = _L_NAMES.get(sh.l, str(sh.l))
                prefix = f"{sp_lbl}_{sh.n}{ll}"
                use_sc = sh.optimize_soft_confinement

                rc_inner_lo_eff = max(rc_inner_lo, ri_lo + ri_gap + 0.1) if use_sc else rc_inner_lo
                chain_lo: list[float] = [0.0] * (sh.n_zeta + 1)
                chain_lo[sh.n_zeta] = rc_inner_lo_eff
                for z in range(sh.n_zeta - 1, 1, -1):
                    chain_lo[z] = chain_lo[z + 1] + rc_gap
                min_rc1 = (chain_lo[2] + rc_gap + 0.05) if sh.n_zeta >= 2 else (rc_inner_lo_eff + 0.05)

                rc1_guess = sh.rc_initials[0] if (sh.rc_initials and len(sh.rc_initials) > 0) else sh.rc_guess
                rc1_lo = max(min_rc1, rc_lo_mult * rc1_guess)
                rc1_hi = min(rc_abs_max, max(rc_hi_mult * rc1_guess, rc1_lo + 0.5))
                if rc1_hi < rc1_lo:
                    raise ValueError(f"{prefix}: infeasible rc_z1 bounds [{rc1_lo}, {rc1_hi}]")
                rc1_init = float(np.clip(rc1_guess, rc1_lo, rc1_hi))
                rc1_name = f"{prefix}_rc_z1"
                specs.append(BasisParamSpec(name=rc1_name, lo=rc1_lo, hi=rc1_hi, initial=rc1_init))
                if not optimize_rc:
                    frozen[rc1_name] = rc1_init

                v0_name = f"{prefix}_v0_ha"
                if use_sc:
                    v0_init = float(np.clip(sh.v0_initial_ha, v0_bounds_ha[0], v0_bounds_ha[1]))
                    specs.append(
                        BasisParamSpec(
                            name=v0_name,
                            lo=float(v0_bounds_ha[0]),
                            hi=float(v0_bounds_ha[1]),
                            initial=v0_init,
                        )
                    )
                    if not optimize_v0:
                        frozen[v0_name] = v0_init

                prev_name = rc1_name
                prev_init = rc1_init
                for z in range(2, sh.n_zeta + 1):
                    z_name = f"{prefix}_rc_z{z}"
                    lo_z = chain_lo[z]
                    raw_z = (
                        sh.rc_initials[z - 1]
                        if (sh.rc_initials and len(sh.rc_initials) >= z)
                        else 0.75 * prev_init
                    )
                    z_init = float(min(max(lo_z, raw_z), prev_init - rc_gap))
                    specs.append(
                        BasisParamSpec(
                            name=z_name,
                            lo=lo_z,
                            hi=None,
                            initial=z_init,
                            depends_on=prev_name,
                            gap=rc_gap,
                        )
                    )
                    if not optimize_rc:
                        frozen[z_name] = z_init
                    prev_name = z_name
                    prev_init = z_init

                if use_sc:
                    ri_name = f"{prefix}_ri"
                    ri_init = float(min(max(ri_lo, sh.ri_initial_bohr), prev_init - ri_gap))
                    specs.append(
                        BasisParamSpec(
                            name=ri_name,
                            lo=ri_lo,
                            hi=None,
                            initial=ri_init,
                            depends_on=prev_name,
                            gap=ri_gap,
                        )
                    )
                    if not optimize_ri:
                        frozen[ri_name] = ri_init

        return cls(specs=specs, species_shells=species_shells, xc=xc, frozen=frozen)

    def initial_u(self) -> np.ndarray:
        """Encode initial physical parameter guesses into unconstrained vector ``u0``.

        Theory:
            Applies the inverse arccos mapping to all active initial parameter values.

        Algorithm:
            Builds ``{name: spec.initial}`` and delegates to :meth:`encode`.

        Returns:
            Initial unconstrained vector ``u0`` of length ``ndim``.

        Invariants:
            ``decode(initial_u())`` reproduces active initial values to machine precision.

        SIESTA Mapping:
            Initializes unconstrained optimizer vectors from baseline guesses.
        """
        return self.encode({sp.name: sp.initial for sp in self.specs})

    def encode(self, params: dict[str, float]) -> np.ndarray:
        """Encode a physical parameter dictionary into active unconstrained coordinates ``u``.

        Theory:
            Inverts the arccos surjection sequentially following topological dependency order
            so dynamic upper bounds resolve consistently.

        Algorithm:
            Clips each parameter into ``[lo, resolved_hi]`` and applies ``to_optimizer_u``.

        Parameters:
            params: Physical parameter mapping containing all active species/shell keys.

        Returns:
            Unconstrained parameter vector ``u`` of shape ``(ndim,)``.

        Invariants:
            Frozen parameters are omitted from the returned vector ``u``.

        SIESTA Mapping:
            Ports ``MappedParameterSpace.encode`` from ``SiestaOpt``.
        """
        resolved: dict[str, float] = {}
        u_vals: list[float] = []
        for sp in self.specs:
            hi = (resolved[sp.depends_on] - sp.gap) if sp.depends_on is not None else _static_upper_bound(sp)
            val = float(self._frozen.get(sp.name, params[sp.name]))
            val = float(np.clip(val, sp.lo, max(hi, sp.lo)))
            resolved[sp.name] = val
            if sp.name not in self._frozen:
                u_vals.append(to_optimizer_u(val, sp.lo, hi))
        return np.asarray(u_vals, dtype=np.float64)

    def decode(self, u: np.ndarray) -> dict[str, float]:
        """Decode unconstrained vector ``u`` into a physically valid parameter dictionary.

        Theory:
            Evaluates dynamic cascading intervals sequentially via ``to_physical_x``.

        Algorithm:
            Resolves outer cutoffs first so inner cutoffs and ``ri`` receive valid
            dynamic upper bounds ``resolved[depends_on] - gap``.

        Parameters:
            u: Unconstrained coordinate vector of shape ``(ndim,)``.

        Returns:
            Full physical parameter dictionary including frozen parameters.

        Invariants:
            Every returned parameter satisfies strict ordering and bound constraints.

        SIESTA Mapping:
            Ports ``MappedParameterSpace.decode`` from ``SiestaOpt``.
        """
        resolved: dict[str, float] = {}
        u_idx = 0
        for sp in self.specs:
            hi = (resolved[sp.depends_on] - sp.gap) if sp.depends_on is not None else _static_upper_bound(sp)
            if sp.name in self._frozen:
                val = float(np.clip(self._frozen[sp.name], sp.lo, max(hi, sp.lo)))
            else:
                val = to_physical_x(float(u[u_idx]), sp.lo, hi)
                u_idx += 1
            resolved[sp.name] = val
        return resolved

    def decode_to_species_specs(self, u: np.ndarray) -> tuple[dict[str, float], dict[str, SpeciesBasisSpec]]:
        """Decode ``u`` into ``(params_dict, {symbol: SpeciesBasisSpec})`` for PAO generation.

        Theory:
            Translates flat mapped ``(rc_z*, v0_ha, ri, q)`` parameters into structured
            :class:`ShellBasisSpec` and :class:`SpeciesBasisSpec` objects.

        Algorithm:
            Calls :meth:`decode` and groups shell radii ``rc_z1..rc_zN``, confinement
            parameters ``(v0_ha, ri)``, and species charge ``q`` per species.

        Parameters:
            u: Unconstrained coordinate vector of shape ``(ndim,)``.

        Returns:
            Tuple ``(params_dict, species_specs_dict)``.

        Invariants:
            All ``rc_list`` sequences are strictly decreasing and larger than ``ri_bohr``.

        SIESTA Mapping:
            Replaces ``generate_pao_basis_block`` for direct in-memory PAO construction.
        """
        params = self.decode(u)
        out: dict[str, SpeciesBasisSpec] = {}
        for sp_lbl, sh_tuple in self.species_shells.items():
            shells_out: list[ShellBasisSpec] = []
            for sh in sh_tuple:
                ll = _L_NAMES.get(sh.l, str(sh.l))
                prefix = f"{sp_lbl}_{sh.n}{ll}"
                rcs = tuple(float(params[f"{prefix}_rc_z{z}"]) for z in range(1, sh.n_zeta + 1))
                v0 = float(params.get(f"{prefix}_v0_ha", 0.0))
                ri = float(params.get(f"{prefix}_ri", 0.0))
                shells_out.append(
                    ShellBasisSpec(
                        n=sh.n,
                        l=sh.l,
                        n_zeta=sh.n_zeta,
                        rc_list=rcs,
                        v0_ha=v0,
                        ri_bohr=ri,
                        is_polarization=sh.is_polarization,
                        polarization_mode=sh.polarization_mode,
                        parent_l=sh.parent_l,
                    )
                )
            out[sp_lbl] = SpeciesBasisSpec(
                symbol=sp_lbl,
                shells=tuple(shells_out),
                ionic_charge=float(params[f"{sp_lbl}_q"]),
                xc=self.xc,
                log_grid_step=self.log_grid_step,
            )
        return params, out


def _effective_rc_params(sp_lbl: str, diag: dict) -> dict[str, float]:
    """Map generator diagnostics to ``{"<sp>_<n><l>_rc_z<k>": rc_used}`` (radii after grid snapping)."""
    letters = "spdfg"
    out: dict[str, float] = {}
    for sh in diag.get("shells", []):
        name = f"{sp_lbl}_{sh['n']}{letters[int(sh['l'])]}_rc_z{sh['zeta']}"
        out[name] = float(sh["rc_bohr"])
    return out


@dataclass(frozen=True)
class BasisTrainingStructure:
    """One geometry target included in a Basis Enthalpy objective.

    Theory:
        Allows optimizing NAO bases simultaneously over one or multiple crystal/molecular
        environments (e.g., compressed + equilibrium + strained lattices) weighted by ``weight``.

    Algorithm:
        Bundles a ``Structure``, k-grid, k-shifts, and relative weight.

    Invariants:
        - ``weight > 0``.

    SIESTA Mapping:
        Supports multi-geometry basis optimization generalizing single-structure simplex runs.
    """

    structure: Structure
    k_grid: tuple[int, int, int] = (2, 2, 2)
    k_shifts: tuple[float, float, float] = (0.0, 0.0, 0.0)
    weight: float = 1.0


@dataclass
class BasisOptimizationResult:
    """Result container from :func:`optimize_basis_enthalpy`.

    Theory:
        Summarizes optimal unconstrained coordinates, decoded physical parameters, generated
        :class:`SpeciesBasisSpec` / ``SpeciesData`` / ``BasisData``, final Basis Enthalpy,
        electronic free energy, orbital volume, and full evaluation trace.

    Algorithm:
        Returned by :func:`optimize_basis_enthalpy` with reproducible PAO metadata.

    Invariants:
        - ``best_enthalpy_ev`` equals ``best_energy_ev + pressure_gpa * EV_PER_BOHR3_PER_GPA * best_volume_bohr3`` (for unit-weight single structure).

    SIESTA Mapping:
        Python equivalent of ``OptimizationResult`` in ``SiestaOpt`` plus live PyTrees.
    """

    best_u: np.ndarray
    best_params: dict[str, float]
    #: Parameters as actually used by the PAO generator (cut-off radii snapped to its radial grid). The
    #: requested ``best_params`` can differ by up to one grid step (review H4).
    best_effective_params: dict[str, float]
    best_species_specs: dict[str, SpeciesBasisSpec]
    best_species_data: dict[str, Any]
    best_basis: BasisData
    best_enthalpy_ev: float
    best_energy_ev: float
    best_volume_bohr3: float
    pressure_gpa: float
    n_evaluations: int
    history: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    message: str = "ok"


class BasisEnthalpyObjective:
    r"""Callable Basis Enthalpy evaluator over explicit ``(rc, V0, ri, q)`` basis vectors.

    Theory:
        Maps candidate unconstrained vector :math:`\mathbf{u}` -> radial PAO generation
        -> static ``jax_siesta`` SCF -> weighted sum of :math:`H_{\text{basis}}(\mathbf{p})`.

    Algorithm:
        Pre-parses geometry/PSML metadata once, evaluates each candidate strictly in memory
        (zero external subprocesses or temporary `.ion` files), tracks the best evaluation,
        and records convergence history. With symmetry, the set-up of each target is kept in a
        :class:`~jax_siesta.symmetry.engine.NativeSymmetrySetupCache` and reused while the orbital
        layout and the neighbour set do not change.

    Invariants:
        - Strictly valid orbital parameters on every evaluation via :class:`MappedBasisSpace`.

    SIESTA Mapping:
        In-memory differentiable-DFT replacement for ``SiestaOpt/objective.py``.
    """

    def __init__(
        self,
        space: MappedBasisSpace,
        psml_paths: dict[str, str | Path],
        targets: list[BasisTrainingStructure],
        scf_config: SCFConfig,
        pressure_gpa: float = 0.2,
    ) -> None:
        self.space = space
        self.psml_paths = {k: str(v) for k, v in psml_paths.items()}
        self.targets = list(targets)
        self.scf_config = scf_config
        self.pressure_gpa = float(pressure_gpa)
        self._loader = PSMLBasisLoader()
        self._tc_eval = TwoCenterEvaluator()
        self._obs_eval = StaticObservableEvaluator()

        self.n_evaluations: int = 0
        self.best_enthalpy_ev: float = float("inf")
        self.best_energy_ev: float = float("inf")
        self.best_volume_bohr3: float = 0.0
        self.best_u: np.ndarray | None = None
        self.best_params: dict[str, float] | None = None
        self.best_effective_params: dict[str, float] | None = None
        self.best_specs: dict[str, SpeciesBasisSpec] | None = None
        self.history: list[dict[str, Any]] = []
        # symmetry set-up per target, reused while the orbital layout and the neighbour set do not change
        self._sym_caches: dict[int, NativeSymmetrySetupCache] = {}

    def _symmetry_cache(self, i_tgt: int) -> NativeSymmetrySetupCache:
        """The symmetry set-up cache of target ``i_tgt`` (created on first use)."""
        cache = self._sym_caches.get(i_tgt)
        if cache is None:
            tgt = self.targets[i_tgt]
            cache = NativeSymmetrySetupCache(tgt.structure, tgt.k_grid, self.scf_config, k_shifts=tgt.k_shifts)
            self._sym_caches[i_tgt] = cache
        return cache

    def evaluate_details(
        self, u: np.ndarray
    ) -> tuple[float, float, float, dict[str, float], dict[str, SpeciesBasisSpec]]:
        """Evaluate ``(enthalpy_ev, energy_ev, volume_bohr3, params, species_specs)`` at ``u``.

        Theory:
            Solves static Kohn-Sham DFT for each training geometry using PAOs generated
            in memory from ``decode_to_species_specs(u)`` and returns the pressure-weighted
            Basis Enthalpy average.

        Algorithm:
            Decodes ``u``, builds ``SpeciesData`` per species, constructs ``BasisData``
            and symmetry/neighbor patterns per target, runs ``run_scf`` and post-SCF
            energy evaluation, and updates best-so-far history records.

        Parameters:
            u: Unconstrained coordinate vector of shape ``(ndim,)``.

        Returns:
            Tuple ``(enthalpy_ev, energy_ev, volume_bohr3, params, species_specs)``.

        Invariants:
            Updates ``n_evaluations`` and ``history`` monotonically on every evaluation.

        SIESTA Mapping:
            Executes in-memory SCF + ``basis_enthalpy`` evaluation replacing subprocess runs.
        """
        params, sp_specs = self.space.decode_to_species_specs(u)
        species_map = {}
        effective = dict(params)
        for sp_lbl, sp_spec in sp_specs.items():
            sp_data, diag = generate_species_from_psml(
                self.psml_paths[sp_lbl], basis_spec=sp_spec, xc=self.scf_config.xc_functional
            )
            species_map[sp_lbl] = sp_data
            effective.update(_effective_rc_params(sp_lbl, diag))

        w_sum = 0.0
        h_acc = 0.0
        e_acc = 0.0
        v_acc = 0.0
        for i_tgt, tgt in enumerate(self.targets):
            basis = self._loader.build_basis(
                tgt.structure,
                species_map,
                include_soc_projectors=(self.scf_config.nspin == 8),
            )
            if self.scf_config.use_symmetry:
                sym_proj, pat, kmesh, mesh_shape = self._symmetry_cache(i_tgt).build(basis)
                # the threaded full evaluation on the symmetry-closed pattern is faster than the irreducible-orbit path
                # (Python orbit bookkeeping) and equal to it to rounding (1e-14)
                tc = self._tc_eval.evaluate(tgt.structure, basis, pat, self.scf_config)
            else:
                sym_proj = None
                pat = build_sparse_pattern(tgt.structure, basis)
                kmesh = build_monkhorst_pack_mesh(tgt.k_grid, tgt.k_shifts)
                mesh_shape = choose_mesh_dimensions(
                    tgt.structure.cell,
                    self.scf_config.mesh_cutoff_ry,
                    criterion=self.scf_config.mesh_criterion,
                    mesh_subdivisions=self.scf_config.mesh_subdivisions,
                )
                tc = self._tc_eval.evaluate(tgt.structure, basis, pat, self.scf_config)

            st, mesh_shape = run_scf(
                tgt.structure,
                basis,
                pat,
                tc,
                kmesh,
                self.scf_config,
                mesh_shape=mesh_shape,
                sym_proj=sym_proj,
            )
            obs = self._obs_eval.evaluate(
                tgt.structure,
                basis,
                pat,
                tc,
                kmesh,
                st,
                self.scf_config,
                mesh_shape=mesh_shape,
                sym_proj=sym_proj,
                compute_forces_stress=False,
            )
            e_ha = float(obs.final_post_scf_free_e_ha)
            e_ev = e_ha * HARTREE_TO_EV
            vol = basis_volume_bohr3(tgt.structure, basis)
            h_ev = e_ev + self.pressure_gpa * EV_PER_BOHR3_PER_GPA * vol
            w = float(tgt.weight)
            w_sum += w
            h_acc += w * h_ev
            e_acc += w * e_ev
            v_acc += w * vol

        h_avg = h_acc / w_sum
        e_avg = e_acc / w_sum
        v_avg = v_acc / w_sum
        self.n_evaluations += 1
        self.history.append(
            {
                "eval": self.n_evaluations,
                "enthalpy_ev": h_avg,
                "energy_ev": e_avg,
                "volume_bohr3": v_avg,
                "params": dict(params),
                "effective_params": effective,
            }
        )
        if h_avg < self.best_enthalpy_ev:
            self.best_enthalpy_ev = h_avg
            self.best_energy_ev = e_avg
            self.best_volume_bohr3 = v_avg
            self.best_u = np.array(u, dtype=np.float64, copy=True)
            self.best_params = dict(params)
            self.best_effective_params = effective
            self.best_specs = sp_specs
        return h_avg, e_avg, v_avg, params, sp_specs

    def __call__(self, u: np.ndarray) -> float:
        """Return scalar Basis Enthalpy in eV for optimizer vector ``u``."""
        try:
            h_ev, _, _, _, _ = self.evaluate_details(np.asarray(u, dtype=np.float64))
            return float(h_ev) if np.isfinite(h_ev) else 1e10
        except RuntimeError:
            return 1e10


def default_opt_space_from_psml(
    psml_paths: dict[str, str | Path],
    basis_size: Literal["SZ", "DZ", "DZP"] = "DZP",
    xc: str = "PBE",
    *,
    optimize_rc: bool = True,
    optimize_v0: bool = True,
    optimize_ri: bool = True,
    optimize_q: bool = True,
) -> MappedBasisSpace:
    """Create a ready-to-optimize :class:`MappedBasisSpace` seeded from PSML defaults.

    Theory:
        Generates initial ``SZ``/``DZ``/``DZP`` shell cutoffs in memory from each ``.psml``
        file and constructs explicit ``(rc_z*, v0_ha, ri, q)`` parameter bounds around them.

    Algorithm:
        Calls :func:`generate_species_from_psml` for initial guesses and feeds them into
        :meth:`MappedBasisSpace.from_species_configs`.

    Parameters:
        psml_paths: Mapping ``symbol -> psml_path``.
        basis_size: Initial shell template (``"SZ"``, ``"DZ"``, ``"DZP"``).
        xc: XC functional identifier.
        optimize_rc: Enable ``rc`` optimization per zeta per shell.
        optimize_v0: Enable soft confinement potential ``V0`` optimization per shell.
        optimize_ri: Enable soft confinement inner radius ``ri`` optimization per shell.
        optimize_q: Enable species ionic charge ``q`` optimization.

    Returns:
        Configured :class:`MappedBasisSpace`.

    Invariants:
        Initial decoded specs cover every valence (and polarization for DZP) shell explicitly.

    SIESTA Mapping:
        Replaces ``dry_run.py`` in ``SiestaOpt`` without invoking SIESTA externally.
    """
    species_shells: dict[str, tuple[ShellOptConfig, ...]] = {}
    for sym, path in psml_paths.items():
        sp_init, diag = generate_species_from_psml(path, basis_spec=basis_size, xc=xc)
        _ = _parse_psml_for_atom(path)
        grouped: dict[tuple[int, int], list[float]] = {}
        pol_flags: dict[tuple[int, int], bool] = {}
        for sh in diag["shells"]:
            key = (int(sh["n"]), int(sh["l"]))
            grouped.setdefault(key, []).append(float(sh["rc_bohr"]))
            pol_flags[key] = bool(sh.get("polarization", False))
        # Also infer from sp_init if needed
        configs: list[ShellOptConfig] = []
        for (n_q, l_q), rcs in grouped.items():
            rcs_sorted = tuple(sorted(rcs, reverse=True))
            configs.append(
                ShellOptConfig(
                    n=n_q,
                    l=l_q,
                    n_zeta=len(rcs_sorted),
                    rc_guess=rcs_sorted[0],
                    rc_initials=rcs_sorted,
                    v0_initial_ha=15.0,
                    ri_initial_bohr=min(0.6 * rcs_sorted[-1], rcs_sorted[-1] - 0.5),
                    is_polarization=pol_flags[(n_q, l_q)],
                    polarization_mode="perturbative" if pol_flags[(n_q, l_q)] else "eigenstate",
                    parent_l=(l_q - 1) if pol_flags[(n_q, l_q)] else None,
                    optimize_soft_confinement=True,
                )
            )
        _ = sp_init
        species_shells[sym] = tuple(configs)

    return MappedBasisSpace.from_species_configs(
        species_shells,
        xc=xc,
        optimize_rc=optimize_rc,
        optimize_v0=optimize_v0,
        optimize_ri=optimize_ri,
        optimize_q=optimize_q,
    )


def optimize_basis_enthalpy(
    objective: BasisEnthalpyObjective,
    *,
    method: Literal["nm", "powell", "de", "de+nm"] = "nm",
    max_evaluations: int = 40,
    seed: int = 42,
) -> BasisOptimizationResult:
    r"""Minimize Basis Enthalpy :math:`H_{\text{basis}}` over explicit ``(rc, V0, ri, q)``.

    Theory:
        Minimizes the scalar ``BasisEnthalpyObjective`` in the unconstrained arccos-mapped
        coordinate space using derivative-free local (Nelder-Mead, Powell) or global + local
        (``de``, ``de+nm``) optimization.

    Algorithm:
        Starts from ``objective.space.initial_u()``, enforces the ``max_evaluations`` budget,
        reconstructs the optimal ``SpeciesData`` and ``BasisData`` PyTrees, and returns
        :class:`BasisOptimizationResult`.

    Parameters:
        objective: Configured :class:`BasisEnthalpyObjective`.
        method: Optimization algorithm (``"nm"``, ``"powell"``, ``"de"``, or ``"de+nm"``).
        max_evaluations: Maximum number of SCF evaluations.
        seed: Random seed for stochastic global search (``de`` / ``de+nm``).

    Returns:
        :class:`BasisOptimizationResult` containing best basis parameters and PyTrees.

    Invariants:
        - Returned ``best_enthalpy_ev`` is guaranteed less than or equal to initial guess.
        - Every tried basis is physically valid by bijection.

    SIESTA Mapping:
        Replaces ``SiestaOpt/optimizers.py`` driving ``jsiesta`` SCF directly in memory.
    """
    u0 = objective.space.initial_u()
    ndim = len(u0)
    algo = method.lower()

    if ndim == 0:
        objective(u0)
    elif algo == "nm":
        minimize(
            objective,
            u0,
            method="Nelder-Mead",
            options={"maxfev": max(1, max_evaluations), "xatol": 1e-4, "fatol": 1e-5},
        )
    elif algo == "powell":
        minimize(
            objective,
            u0,
            method="Powell",
            options={"maxfev": max(1, max_evaluations), "xtol": 1e-4, "ftol": 1e-5},
        )
    elif algo in ("de", "de+nm"):
        de_budget = max_evaluations if algo == "de" else max(10, int(0.65 * max_evaluations))
        popsize = max(2, min(6, de_budget // max(1, 2 * ndim)))
        maxiter = max(1, de_budget // max(1, popsize * ndim))
        bounds = [(-math.pi, math.pi)] * ndim
        differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            polish=False,
            seed=seed,
            x0=u0,
        )
        if algo == "de+nm":
            rem = max(5, max_evaluations - objective.n_evaluations)
            u_start = objective.best_u if objective.best_u is not None else u0
            minimize(
                objective,
                u_start,
                method="Nelder-Mead",
                options={"maxfev": rem, "xatol": 1e-4, "fatol": 1e-5},
            )
    else:
        raise ValueError(f"Unsupported optimizer method {method!r}; expected 'nm', 'powell', 'de', or 'de+nm'.")

    best_u = objective.best_u if objective.best_u is not None else u0
    best_params, best_specs = objective.space.decode_to_species_specs(best_u)
    best_species = {}
    for sym, sp_spec in best_specs.items():
        sp_data, _ = generate_species_from_psml(
            objective.psml_paths[sym], basis_spec=sp_spec, xc=objective.scf_config.xc_functional
        )
        best_species[sym] = sp_data
    best_basis = objective._loader.build_basis(
        objective.targets[0].structure,
        best_species,
        include_soc_projectors=(objective.scf_config.nspin == 8),
    )
    return BasisOptimizationResult(
        best_u=best_u,
        best_params=best_params,
        best_effective_params=dict(objective.best_effective_params or best_params),
        best_species_specs=best_specs,
        best_species_data=best_species,
        best_basis=best_basis,
        best_enthalpy_ev=float(objective.best_enthalpy_ev),
        best_energy_ev=float(objective.best_energy_ev),
        best_volume_bohr3=float(objective.best_volume_bohr3),
        pressure_gpa=objective.pressure_gpa,
        n_evaluations=objective.n_evaluations,
        history=objective.history,
        success=np.isfinite(objective.best_enthalpy_ev),
        message="ok" if np.isfinite(objective.best_enthalpy_ev) else "failed",
    )


def format_pao_basis_block(species_specs: dict[str, SpeciesBasisSpec]) -> str:
    """Format optimized :class:`SpeciesBasisSpec` dictionaries as a SIESTA ``%block PAO.Basis``.

    Theory:
        Emits exact explicit per-zeta ``rc`` values, soft confinement ``E (2*V0_Ha) ri``,
        and net ionic charge ``q`` in standard SIESTA FDF units (Bohr, Rydberg, e).

    Algorithm:
        Iterates species and shells, formatting FDF lines matching ``SiestaOpt/basis_block.py``.

    Parameters:
        species_specs: Mapping ``symbol -> SpeciesBasisSpec``.

    Returns:
        Formatted ``%block PAO.Basis ... %endblock PAO.Basis`` string.

    Invariants:
        Soft confinement depth is exported in Rydberg (``2 * v0_ha``) matching SIESTA FDF.

    SIESTA Mapping:
        Mirrors ``generate_pao_basis_block`` in ``SiestaOpt/src/siesta_opt/basis_block.py``.
    """
    lines = ["%block PAO.Basis"]
    for sym, sp in species_specs.items():
        lines.append(f"  {sym}  {len(sp.shells)}  {sp.ionic_charge:.5f}")
        for sh in sp.shells:
            rcs = sh.rc_list or (5.0,) * sh.n_zeta
            hdr = [f"n={sh.n}", f"{sh.l}", f"{sh.n_zeta}"]
            if sh.v0_ha != 0.0 or sh.ri_bohr > 0.0:
                hdr.append(f"E {2.0 * sh.v0_ha:.3f} {sh.ri_bohr:.4f}")
            lines.append("  " + "  ".join(hdr))
            lines.append("    " + " ".join(f"{r:.4f}" for r in rcs))
            lines.append("    " + " ".join("1.0000" for _ in rcs))
    lines.append("%endblock PAO.Basis")
    return "\n".join(lines)


__all__ = [
    "EV_PER_BOHR3_PER_GPA",
    "BasisEnthalpyObjective",
    "BasisOptimizationResult",
    "BasisParamSpec",
    "BasisTrainingStructure",
    "MappedBasisSpace",
    "ShellOptConfig",
    "basis_enthalpy_ev",
    "basis_volume_bohr3",
    "default_opt_space_from_psml",
    "format_pao_basis_block",
    "optimize_basis_enthalpy",
    "to_optimizer_u",
    "to_physical_x",
]
