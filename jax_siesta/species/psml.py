r"""PSML-only pseudopotential parser and confined NAO radial basis loader.

Theory:
    Parses norm-conserving ``.psml`` (v1.0/v1.1) XML pseudopotentials exclusively
    (rejecting legacy ``.psf`` files) to extract:

    - Semilocal / local pseudopotential :math:`V_{\text{loc}}(r)` (Hartree)
    - Separable Kleinman-Bylander (KB) projectors :math:`\chi_{lj}(r)` and energies
      :math:`E^{\text{KB}}_{lj}` (supporting both scalar-relativistic and :math:`j`-dependent
      spin-orbit coupling channels)
    - Valence density :math:`\rho^{\text{NA}}(r)` and pseudo-core charge :math:`\rho^{\text{core}}(r)`
    - Neutral-atom screening potential :math:`V^{\text{NA}}(r) = V_{\text{loc}}(r) + V_H[\rho^{\text{NA}}](r)`

    Confined Numerical Atomic Orbitals (NAOs) :math:`\phi_{nl\zeta m}(\mathbf{r}) = R_{nl\zeta}(r) Y_{lm}(\hat{\mathbf{r}})`
    are loaded from ``.ion.json`` tables (or generated as smooth Sankey-confined
    split-valence orbitals) with unique sequential ``pao_iz`` shell indices.

Algorithm:
    Uses ``lxml.etree`` / ``xml.etree.ElementTree`` to parse PSML XML into
    uniform radial tables on ``[0, r_max]``, solves the 1D radial Poisson integral
    for :math:`V_H[\rho^{\text{NA}}](r)` via cumulative trapezoidal quadrature, and
    constructs ``SpeciesData``, ``BasisData``, and ``SparsePattern`` PyTrees.

Invariants:
    - Legacy ``.psf`` files raise ``ValueError`` explicitly.
    - Every orbital shell gets a strictly unique sequential ``pao_iz`` counter.
    - :math:`4\pi \int_0^\infty r^2 \rho^{\text{NA}}(r)\,dr = Z_{\text{val}}` (normalized).
    - :math:`V^{\text{NA}}(r) \to 0` smoothly beyond :math:`r_c^{\text{NA}}`.

SIESTA Mapping:
    Replaces ``atom.F``, ``basis_specs.f``, ``psml_wrappers.F90``, ``atmfuncs.F``,
    and supercell neighbor list initialization in ``listsc_init``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from scipy.interpolate import CubicSpline

from jax_siesta.core.types import BasisData, SparsePattern, SpeciesData, Structure

_L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_HA_PER_UNIT = {
    "hartree": 1.0,
    "ha": 1.0,
    "rydberg": 0.5,
    "ry": 0.5,
    "ev": 1.0 / 27.211386245988,
}
_BOHR_PER_UNIT = {
    "bohr": 1.0,
    "a0": 1.0,
    "angstrom": 1.0 / 0.529177210903,
    "ang": 1.0 / 0.529177210903,
}


_MIN_RADIAL_POINTS = 401


def _uniform_grid_from_sources(sources: list[np.ndarray], r_max: float, min_points: int = _MIN_RADIAL_POINTS) -> np.ndarray:
    """Uniform grid on ``[0, r_max]`` at least as fine as the finest source grid spacing.

    SIESTA ``.ion.xml`` tables use a separate fine grid per function (e.g. 0.0046 Bohr for a Co 3s semicore
    PAO); a fixed coarse grid with linear interpolation loses accuracy in S, T and V_KB (review H2).
    """
    steps = [float(np.min(np.diff(r))) for r in sources if len(r) > 1 and float(np.min(np.diff(r))) > 0.0]
    d_min = min(steps) if steps else r_max / (min_points - 1)
    n_pts = max(min_points, int(np.ceil(r_max / d_min)) + 1)
    return np.linspace(0.0, r_max, n_pts, dtype=np.float64)


def _spline_resample(r_src: np.ndarray, f_src: np.ndarray, r_dst: np.ndarray, power: int = 0) -> np.ndarray:
    """Cubic-spline resampling of ``f_src(r) * r**power``, zero beyond the last source point."""
    cs = CubicSpline(np.asarray(r_src, dtype=np.float64), np.asarray(f_src, dtype=np.float64))
    inside = r_dst <= float(r_src[-1]) + 1e-12
    vals = cs(np.clip(r_dst, float(r_src[0]), float(r_src[-1]))) * (r_dst**power if power else 1.0)
    return np.where(inside, vals, 0.0)


def _parse_float_array(text: str | None) -> np.ndarray:
    if not text or not text.strip():
        return np.array([], dtype=np.float64)
    return np.fromstring(text.strip(), sep=" ", dtype=np.float64)


def _strip_xml_namespaces(root: ET.Element) -> None:
    """Strip XML namespace prefixes in-place for uniform PSML v1.0/v1.1 querying."""
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _parse_ion_xml_two_col(radfunc_el: ET.Element | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse SIESTA ``.ion.xml`` ``<radfunc><data>`` two-column ``(r_bohr, value)`` table."""
    if radfunc_el is None:
        return None
    data_el = radfunc_el.find("data")
    if data_el is None or not data_el.text:
        return None
    arr = np.fromstring(data_el.text.strip(), sep=" ", dtype=np.float64)
    if len(arr) < 4 or len(arr) % 2 != 0:
        return None
    pts = arr.reshape(-1, 2)
    return pts[:, 0], pts[:, 1]


def _select_nonlocal_block(
    root: ET.Element,
    prefer_lj: bool = False,
) -> ET.Element | None:
    """Select the canonical ``<nonlocal-projectors>`` block from PSML v1.0/v1.1."""
    blocks = root.findall("nonlocal-projectors")
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]
    target_set = "lj" if prefer_lj else "scalar_relativistic"
    for blk in blocks:
        if blk.get("set", "").lower() == target_set:
            return blk
    return blocks[0]


def _radial_hartree_potential(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r"""Compute :math:`V_H[\rho](r) = \frac{4\pi}{r}\int_0^r r'^2\rho\,dr' + 4\pi\int_r^\infty r'\rho\,dr'`."""
    n = len(r)
    dr = np.diff(r)
    integrand_inner = 4.0 * np.pi * (r**2) * rho
    integrand_outer = 4.0 * np.pi * r * rho

    q_inner = np.zeros(n, dtype=np.float64)
    q_inner[1:] = np.cumsum(0.5 * (integrand_inner[:-1] + integrand_inner[1:]) * dr)

    tail_outer = np.zeros(n, dtype=np.float64)
    rev_int = 0.5 * (integrand_outer[:-1] + integrand_outer[1:]) * dr
    tail_outer[:-1] = np.cumsum(rev_int[::-1])[::-1]

    v_h = np.zeros(n, dtype=np.float64)
    v_h[1:] = q_inner[1:] / r[1:] + tail_outer[1:]
    v_h[0] = tail_outer[0]
    return v_h


def _generate_confined_pao_orbitals(
    r_grid: np.ndarray,
    shells: list[tuple[int, int]],
    z_val: float,
    rcut: float = 6.0,
    zeta_count: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[int], list[float]]:
    """Generate smooth Sankey-confined split-valence PAO radial orbitals on ``[0, rcut]``."""
    pao_r = np.linspace(0.0, rcut, 201, dtype=np.float64)
    vals: list[np.ndarray] = []
    pao_l: list[int] = []
    pao_n: list[int] = []
    pao_zeta: list[int] = []
    pao_rcut: list[float] = []

    for zeta in range(1, zeta_count + 1):
        for n_q, l_q in shells:
            alpha = (1.05 if zeta == 1 else 2.15) * (1.0 + 0.22 * z_val / max(n_q, 1))
            cut = np.exp(-0.15 / np.maximum(rcut - pao_r, 1e-4))
            cut[-1] = 0.0
            rad = (pao_r**l_q) * np.exp(-alpha * pao_r) * cut
            norm = float(np.sqrt(np.trapezoid((rad**2) * (pao_r**2), pao_r)))
            if norm > 0.0:
                rad = rad / norm
            vals.append(rad)
            pao_l.append(l_q)
            pao_n.append(n_q)
            pao_zeta.append(zeta)
            pao_rcut.append(rcut)
    return pao_r, np.stack(vals, axis=0), pao_l, pao_n, pao_zeta, pao_rcut


def _resample_last_axis(r: np.ndarray, values: np.ndarray, n_new: int) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-spline resample ``values[..., :]`` given on ``r`` onto ``n_new`` uniform points of ``[r[0], r[-1]]``."""
    r_new = np.linspace(float(r[0]), float(r[-1]), n_new)
    return r_new, np.asarray(CubicSpline(r, values, axis=-1)(r_new), dtype=np.float64)


def unify_species_tables(species: list[SpeciesData]) -> list[SpeciesData]:
    r"""Give all species radial tables of equal length so the C++ kernels can stack them.

    Theory:
        The FFI kernels store the PAO, Kleinman-Bylander and neutral-atom tables of all species in one rectangular
        array on a uniform grid ``[0, r_max]``. Species whose tables come from sources of different density (for
        example H and O ``.ion.xml`` files) must therefore be put on a common point count without losing accuracy:
        the finest table sets the count and shorter ones are resampled with a natural cubic spline (not linearly,
        review H2 / D-017).

    Algorithm:
        Takes the largest ``len(r_grid)``, ``len(pao_r)`` and ``len(kb_r)`` over the species and resamples the
        arrays defined on each grid; species already at that length are returned unchanged.

    Parameters:
        species: Species of one calculation.

    Returns:
        New list in the same order (the input objects are not modified).

    Invariants:
        - Each ``r_max`` is unchanged; a single species, or species with equal lengths, are returned as is.
        - Afterwards all ``r_grid``, all ``pao_r`` and all ``kb_r`` have one common length.

    SIESTA Mapping:
        None; SIESTA keeps per-species tables and interpolates each one separately.
    """
    if len(species) < 2:
        return list(species)
    n_r = max(int(sp.r_grid.shape[0]) for sp in species)
    n_pao = max(int(sp.pao_r.shape[0]) for sp in species)
    n_kb = max(int(sp.kb_r.shape[0]) for sp in species)
    out: list[SpeciesData] = []
    for sp in species:
        changes: dict[str, Any] = {}
        if int(sp.r_grid.shape[0]) != n_r:
            r0 = np.asarray(sp.r_grid, dtype=np.float64)
            r_new = None
            for name in ("v_local", "v_na", "rho_na", "rho_core", "tau_core"):
                arr = getattr(sp, name)
                if arr is None:
                    continue
                r_new, new = _resample_last_axis(r0, np.asarray(arr, dtype=np.float64), n_r)
                changes[name] = jnp.asarray(new, dtype=jnp.float64)
            assert r_new is not None
            changes["r_grid"] = jnp.asarray(r_new, dtype=jnp.float64)
        if int(sp.pao_r.shape[0]) != n_pao:
            r_new, new = _resample_last_axis(
                np.asarray(sp.pao_r, dtype=np.float64), np.asarray(sp.pao_values, dtype=np.float64), n_pao
            )
            changes["pao_r"], changes["pao_values"] = jnp.asarray(r_new), jnp.asarray(new)
        if int(sp.kb_r.shape[0]) != n_kb:
            r_new, new = _resample_last_axis(
                np.asarray(sp.kb_r, dtype=np.float64), np.asarray(sp.kb_projectors, dtype=np.float64), n_kb
            )
            changes["kb_r"], changes["kb_projectors"] = jnp.asarray(r_new), jnp.asarray(new)
        out.append(replace(sp, **changes) if changes else sp)
    return out


class PSMLBasisLoader:
    """PSML-only pseudopotential & NAO basis loader implementing ``BasisAndPseudos``.

    Theory:
        Enforces PSML-only norm-conserving pseudopotential ingestion, computes
        short-range neutral-atom screening :math:`V^{\text{NA}}(r)`, and organizes
        orbital and KB projector index tables across unit-cell atoms.

    Algorithm:
        Parses ``.psml`` XML and optional ``.ion.json`` radial basis tables,
        resampling radial functions onto dense uniform grids for fast C++ cubic
        spline interpolation.

    Invariants:
        - Rejects ``.psf`` files with ``ValueError``.
        - Guarantees unique sequential ``pao_iz`` shell indices per ``(l, m)``.

    SIESTA Mapping:
        Implements ``BasisAndPseudos`` replacing ``atom.F`` and ``basis_specs.f``.
    """

    def load_species(
        self,
        symbol: str,
        psml_path: str,
        ion_path: str | None = None,
        **kwargs: Any,
    ) -> SpeciesData:
        r"""Load one chemical species from a ``.psml`` XML file and optional ``.ion.json``.

        Theory:
            Extracts :math:`V_{\text{loc}}(r)`, :math:`\rho^{\text{NA}}(r)`, :math:`\rho^{\text{core}}(r)`,
            KB projectors :math:`(\chi_{lj}, E^{\text{KB}}_{lj})`, and confined PAOs
            :math:`R_{nl\zeta}(r)`, converting all units explicitly to Bohr and Hartree.

        Algorithm:
            Reads PSML XML grid and radial datasets, normalizes :math:`\rho^{\text{NA}}(r)`
            to :math:`Z_{\text{val}}`, solves 1D radial Poisson for :math:`V^{\text{NA}}(r)`,
            and aligns PAO radial tables onto uniform grids.

        Parameters:
            symbol: Element/species label (e.g. ``"Si"``, ``"Co"``).
            psml_path: Path to ``.psml`` XML pseudopotential file.
            ion_path: Optional path to ``.ion.json`` PAO radial basis table.
            **kwargs: Optional ``rcut_bohr`` or ``zeta_count`` overrides when generating PAOs.

        Returns:
            Immutable ``SpeciesData`` PyTree.

        Invariants:
            Raises ``ValueError`` if ``psml_path`` ends with ``.psf`` or lacks ``<psml>``.

        SIESTA Mapping:
            Replaces ``read_psml`` and ``read_ion`` in ``m_psml_parse`` and ``atom.F``.
        """
        p = Path(psml_path)
        if p.suffix.lower() == ".psf":
            raise ValueError(
                f"Legacy .psf format is explicitly unsupported ({p}); provide a .psml pseudopotential."
            )
        if not p.is_file():
            raise FileNotFoundError(f"PSML file not found: {p}")

        raw_bytes = p.read_bytes()
        psml_sha = hashlib.sha256(raw_bytes).hexdigest()
        root = ET.fromstring(raw_bytes)
        _strip_xml_namespaces(root)
        if root.tag != "psml":
            raise ValueError(f"Expected <psml> root element in {p}, got <{root.tag}>.")

        e_unit = root.get("energy_unit", "hartree").lower()
        l_unit = root.get("length_unit", "bohr").lower()
        e_scale = _HA_PER_UNIT.get(e_unit, 1.0)
        l_scale = _BOHR_PER_UNIT.get(l_unit, 1.0)

        pa_spec = root.find("pseudo-atom-spec")
        if pa_spec is None:
            raise ValueError(f"Missing <pseudo-atom-spec> in {p}.")
        atomic_number = int(pa_spec.get("atomic-number", "0"))
        z_val = float(pa_spec.get("z-pseudo", "0.0"))
        mass = float(pa_spec.get("atomic-mass", str(max(1.0, 2.0 * atomic_number))))
        relativity = pa_spec.get("relativity", "scalar").lower()

        shells: list[tuple[int, int]] = []
        occ_map: dict[tuple[int, int], float] = {}
        vc = pa_spec.find("valence-configuration")
        if vc is not None:
            if z_val <= 0.0:
                z_val = float(vc.get("total-valence-charge", "0.0"))
            for sh in vc.findall("shell"):
                n_q = int(sh.get("n", "1"))
                l_raw = sh.get("l", "s").lower()
                l_q = _L_MAP[l_raw] if l_raw in _L_MAP else int(l_raw)
                shells.append((n_q, l_q))
                occ_str = sh.get("occupation")
                if occ_str is not None:
                    occ_map[(n_q, l_q)] = occ_map.get((n_q, l_q), 0.0) + float(occ_str)
        if not shells:
            shells = [(1, 0)]

        grid_el = root.find("grid/grid-data")
        if grid_el is None:
            raise ValueError(f"Missing <grid><grid-data> in {p}.")
        r_raw = _parse_float_array(grid_el.text) * l_scale
        if len(r_raw) < 10:
            raise ValueError(f"PSML radial grid too short ({len(r_raw)} pts) in {p}.")

        # Resolve optional .ion.json or SIESTA-generated .ion.xml basis table early
        # so missing <local-potential> in semilocal PSMLs can fall back cleanly.
        if ion_path is None:
            for cand in (p.with_suffix(".ion.json"), p.with_suffix(".ion.xml")):
                if cand.is_file():
                    ion_path = str(cand)
                    break

        ion_xml_root: ET.Element | None = None
        if ion_path is not None and str(ion_path).endswith(".ion.xml") and Path(ion_path).is_file():
            ion_xml_root = ET.fromstring(Path(ion_path).read_bytes())
            _strip_xml_namespaces(ion_xml_root)
            z_xml = ion_xml_root.findtext("valence")
            if z_xml and z_val <= 0.0:
                z_val = float(z_xml)
            m_xml = ion_xml_root.findtext("mass")
            if m_xml:
                mass = float(m_xml)

        # Parse local potential (with Coulombic -z_val/r tail beyond tabulated span)
        # With a SIESTA .ion.xml, use the local potential SIESTA actually used to build V_NA and the KB
        # projectors (``reduced_vlocal``); it can differ from the PSML one (e.g. Co: SIESTA regenerates it
        # from ``chlocal``, changing the ion self-energy by 24 meV/atom; review H2, D-017).
        vloc_parent = root.find("local-potential")
        vloc_el = vloc_parent.find("radfunc/data") if vloc_parent is not None else None
        ion_has_vlocal = ion_xml_root is not None and ion_xml_root.find("reduced_vlocal/radfunc") is not None
        if vloc_el is not None and vloc_parent is not None and not ion_has_vlocal:
            v_loc_raw = _parse_float_array(vloc_el.text) * e_scale
            loc_grid_el = vloc_parent.find("grid/grid-data")
            r_loc = (
                _parse_float_array(loc_grid_el.text) * l_scale
                if loc_grid_el is not None
                else r_raw[: len(v_loc_raw)]
            )
        elif ion_xml_root is not None:
            rvl_pair = _parse_ion_xml_two_col(ion_xml_root.find("reduced_vlocal/radfunc"))
            if rvl_pair is None:
                raise ValueError(f"Missing both PSML <local-potential> and .ion.xml <reduced_vlocal> in {p}.")
            r_loc, rvl_ry = rvl_pair
            v_loc_raw = np.zeros_like(r_loc)
            nz_mask = r_loc > 1e-10
            v_loc_raw[nz_mask] = (0.5 * rvl_ry[nz_mask] - z_val) / r_loc[nz_mask]
            if len(v_loc_raw) > 2:
                v_loc_raw[0] = v_loc_raw[1] - r_loc[1] * (v_loc_raw[2] - v_loc_raw[1]) / max(
                    r_loc[2] - r_loc[1], 1e-12
                )
        else:
            raise ValueError(f"Missing <local-potential> in {p}.")

        rho_parent = root.find("valence-charge")
        rho_el = rho_parent.find("radfunc/data") if rho_parent is not None else None
        if rho_el is not None:
            rho_na_raw = _parse_float_array(rho_el.text) / (l_scale**3)
            # Official PSML 1.1 stores 4*pi*rho(r) in valence-charge; rescaled below anyway.
            r_rho = r_raw[: len(rho_na_raw)]
        else:
            r_rho = r_raw
            sigma = 1.2
            rho_na_raw = (z_val / (np.pi**1.5 * sigma**3)) * np.exp(-((r_raw / sigma) ** 2))

        # Resample onto a clean uniform radial grid for fast C++ cubic splines
        r_max = min(float(r_raw[-1]), 12.0)
        n_unif = 301
        r_unif = np.linspace(0.0, r_max, n_unif, dtype=np.float64)
        v_loc = np.where(
            r_unif <= float(r_loc[-1]),
            np.interp(r_unif, r_loc, v_loc_raw),
            -z_val / np.maximum(r_unif, 1e-12),
        )
        rho_na = np.maximum(np.interp(r_unif, r_rho, rho_na_raw, right=0.0), 0.0)
        # Ensure exact integral 4*pi*r^2 rho_na dr = z_val
        q_na = float(np.trapezoid(4.0 * np.pi * (r_unif**2) * rho_na, r_unif))
        if q_na > 1e-12:
            rho_na = rho_na * (z_val / q_na)

        v_h_na = _radial_hartree_potential(r_unif, rho_na)
        v_na = v_loc + v_h_na
        # Determine neutral-atom screening cutoff where |v_na| and rho_na drop below 1e-7
        active_idx = np.where((np.abs(v_na) > 1e-7) | (rho_na > 1e-7))[0]
        rcut_na = float(r_unif[min(int(active_idx[-1]) + 5, n_unif - 1)]) if len(active_idx) else 4.0
        taper = np.where(r_unif < rcut_na, 1.0, 0.0)
        v_na = v_na * taper
        rho_na = rho_na * taper
        q_na_cut = float(np.trapezoid(4.0 * np.pi * (r_unif**2) * rho_na, r_unif))
        if q_na_cut > 1e-12:
            rho_na = rho_na * (z_val / q_na_cut)

        # Parse NLCC pseudo-core charge: official PSML v1.1 uses <pseudocore-charge> (4*pi*rho_core)
        # while synthetic fixtures may use <pseudo-core-charge> (already rho_core).
        pc_parent = root.find("pseudocore-charge")
        pc_div_4pi = True
        if pc_parent is None:
            pc_parent = root.find("pseudo-core-charge")
            pc_div_4pi = False
        pc_el = pc_parent.find("radfunc/data") if pc_parent is not None else None
        if pc_el is not None:
            rho_core_raw = _parse_float_array(pc_el.text) / (l_scale**3)
            if pc_div_4pi:
                rho_core_raw = rho_core_raw / (4.0 * np.pi)
            r_core = r_raw[: len(rho_core_raw)]
            rho_core = np.maximum(np.interp(r_unif, r_core, rho_core_raw, right=0.0), 0.0)
            has_nlcc = bool(np.max(rho_core) > 1e-12)
        else:
            rho_core = np.zeros_like(r_unif)
            has_nlcc = False

        # Parse NLCC pseudo-core kinetic energy density for meta-GGA functionals:
        # official PSML v1.1 stores 4*pi*tau_core (matching <pseudocore-charge>).
        tk_parent = root.find("pseudocore-kinetic-energy-density")
        tk_div_4pi = True
        if tk_parent is None:
            tk_parent = root.find("pseudo-core-kinetic-energy-density")
            tk_div_4pi = False
        tk_el = tk_parent.find("radfunc/data") if tk_parent is not None else None
        if tk_el is not None:
            tau_core_raw = _parse_float_array(tk_el.text) / (l_scale**5)
            if tk_div_4pi:
                tau_core_raw = tau_core_raw / (4.0 * np.pi)
            r_tcore = r_raw[: len(tau_core_raw)]
            tau_core = np.maximum(np.interp(r_unif, r_tcore, tau_core_raw, right=0.0), 0.0)
            has_tau_core = bool(np.max(tau_core) > 1e-12)
        else:
            tau_core = np.zeros_like(r_unif)
            has_tau_core = False

        # Parse Kleinman-Bylander projectors from .ion.xml (if present) or PSML
        kb_projs: list[np.ndarray] = []
        kb_l: list[int] = []
        kb_j2: list[int] = []
        kb_energies: list[float] = []
        kb_rcut: list[float] = []
        has_soc = False

        prefer_lj = bool(kwargs.get("prefer_lj_projectors", False))
        if ion_xml_root is not None and not prefer_lj and ion_xml_root.find("kbs/projector") is not None:
            kb_raw_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            max_kb_rc = 4.0
            for proj in ion_xml_root.findall("kbs/projector"):
                parsed = _parse_ion_xml_two_col(proj.find("radfunc"))
                if parsed is None:
                    continue
                r_k, y_k = parsed
                l_val = int(proj.attrib["l"])
                ekb = 0.5 * float(proj.attrib["ref_energy"])  # Ry -> Ha
                rc_kb = float(r_k[-1])
                max_kb_rc = max(max_kb_rc, rc_kb)
                kb_raw_pairs.append((r_k, y_k))
                kb_l.append(l_val)
                kb_j2.append(0)
                kb_energies.append(ekb)
                kb_rcut.append(rc_kb)
            kb_r_unif = _uniform_grid_from_sources([r for r, _ in kb_raw_pairs], max_kb_rc, min_points=201)
            for (r_k, y_k), l_k in zip(kb_raw_pairs, kb_l, strict=True):
                kb_projs.append(_spline_resample(r_k, y_k, kb_r_unif, power=l_k))
        else:
            nl_el = _select_nonlocal_block(root, prefer_lj=prefer_lj)
            raw_kb_records: list[tuple[np.ndarray, np.ndarray, float]] = []
            max_kb_rc = 4.0
            if nl_el is not None:
                for proj in nl_el.findall("proj"):
                    l_str = proj.get("l", "s").lower()
                    l_val = _L_MAP[l_str] if l_str in _L_MAP else int(l_str)
                    ekb = float(proj.get("ekb", "0.0")) * e_scale
                    j_attr = proj.get("j")
                    if j_attr is not None:
                        j2_val = int(round(2.0 * float(j_attr)))
                        has_soc = True
                    else:
                        j2_val = 0
                    pdata_el = proj.find("radfunc/data")
                    if pdata_el is None:
                        continue
                    chi_raw = _parse_float_array(pdata_el.text)
                    r_proj = r_raw[: len(chi_raw)]
                    rcut_attr = proj.get("rcut")
                    if rcut_attr is not None:
                        rc_kb = float(rcut_attr) * l_scale
                    else:
                        nz = np.where(np.abs(chi_raw) > 1e-10)[0]
                        rc_idx = min(int(nz[-1]) + 2, len(r_proj) - 1) if len(nz) else 0
                        rc_kb = max(float(r_proj[rc_idx]), 0.5)
                    max_kb_rc = max(max_kb_rc, rc_kb)
                    raw_kb_records.append((r_proj, chi_raw, rc_kb))
                    kb_l.append(l_val)
                    kb_j2.append(j2_val)
                    kb_energies.append(ekb)
                    kb_rcut.append(rc_kb)
            kb_r_unif = np.linspace(0.0, max_kb_rc, 201 if raw_kb_records else 161, dtype=np.float64)
            for r_proj, chi_raw, rc_kb in raw_kb_records:
                chi_u = np.where(kb_r_unif <= rc_kb, np.interp(kb_r_unif, r_proj, chi_raw, right=0.0), 0.0)
                kb_projs.append(chi_u)

        if not kb_projs:
            kb_r_unif = np.linspace(0.0, 4.0, 161, dtype=np.float64)
            kb_projs = [np.zeros_like(kb_r_unif)]
            kb_l = [0]
            kb_j2 = [0]
            kb_energies = [0.0]
            kb_rcut = [1.0]
        if relativity in ("dirac", "spin-orbit", "spin_orbit"):
            has_soc = has_soc or any(j2 > 0 for j2 in kb_j2)

        # Load .ion.xml, .ion.json basis, or auto-generate confined split-valence PAOs
        if ion_xml_root is not None:
            orbs = ion_xml_root.findall("paos/orbital")
            pao_raw: list[tuple[np.ndarray, np.ndarray, float]] = []
            pao_l_list: list[int] = []
            pao_n_list: list[int] = []
            pao_zeta_list: list[int] = []
            pao_rcut_list: list[float] = []
            max_rc = 0.0
            for orb in orbs:
                parsed = _parse_ion_xml_two_col(orb.find("radfunc"))
                if parsed is None:
                    continue
                r_o, y_o = parsed
                l_o = int(orb.attrib["l"])
                n_o = int(orb.attrib["n"])
                z_o = int(orb.attrib["z"])
                rc_o = float(r_o[-1])
                max_rc = max(max_rc, rc_o)
                # SIESTA .ion.xml stores R_{nl}(r) / r^l; jsiesta stores R_{nl}(r)
                pao_raw.append((r_o, y_o, rc_o))
                pao_l_list.append(l_o)
                pao_n_list.append(n_o)
                pao_zeta_list.append(z_o)
                pao_rcut_list.append(rc_o)
            pao_r = _uniform_grid_from_sources([r for r, _, _ in pao_raw], max_rc)
            pao_vals_list = [
                _spline_resample(r_o, y_o, pao_r, power=l_o)
                for (r_o, y_o, _rc), l_o in zip(pao_raw, pao_l_list, strict=True)
            ]
            pao_values = np.stack(pao_vals_list, axis=0)

            # Use SIESTA exact V_NA, core, and atomic PAO rho_atm on aligned uniform grid
            vna_pair = _parse_ion_xml_two_col(ion_xml_root.find("vna/radfunc"))
            core_pair = _parse_ion_xml_two_col(ion_xml_root.find("core/radfunc"))
            if vna_pair is not None:
                r_vna_x, vna_ry_x = vna_pair
                rcut_na = float(r_vna_x[-1])
                r_unif_new = _uniform_grid_from_sources([r_vna_x], max(rcut_na, 8.0))
                v_na = 0.5 * _spline_resample(r_vna_x, vna_ry_x, r_unif_new)
                v_loc = np.where(
                    r_unif_new <= float(r_loc[-1]),
                    _spline_resample(r_loc, v_loc_raw, r_unif_new),
                    -z_val / np.maximum(r_unif_new, 1e-12),
                )
                # Reconstruct exact SIESTA neutral-atom density from first-zeta PAOs
                rho_atm = np.zeros_like(r_unif_new)
                for l_o, n_o, z_o, r_vals_u in zip(
                    pao_l_list, pao_n_list, pao_zeta_list, pao_vals_list, strict=True
                ):
                    if z_o == 1:
                        occ = occ_map.get((n_o, l_o), 0.0)
                        if occ > 0.0:
                            r_on_u = _spline_resample(pao_r, r_vals_u, r_unif_new)
                            rho_atm += (occ / (4.0 * np.pi)) * (r_on_u**2)
                q_atm = float(np.trapezoid(4.0 * np.pi * (r_unif_new**2) * rho_atm, r_unif_new))
                if q_atm > 1e-12:
                    rho_na = rho_atm * (z_val / q_atm)
                else:
                    rho_na = np.maximum(np.interp(r_unif_new, r_rho, rho_na_raw, right=0.0), 0.0)
                    q_fallback = float(np.trapezoid(4.0 * np.pi * (r_unif_new**2) * rho_na, r_unif_new))
                    if q_fallback > 1e-12:
                        rho_na = rho_na * (z_val / q_fallback)
                if core_pair is not None:
                    r_c_x, rho_c_x = core_pair
                    rho_core = np.maximum(_spline_resample(r_c_x, rho_c_x, r_unif_new), 0.0)
                    has_nlcc = bool(np.max(rho_core) > 1e-12)
                else:
                    rho_core = np.interp(r_unif_new, r_unif, rho_core, right=0.0)
                tau_core = np.maximum(np.interp(r_unif_new, r_unif, tau_core, right=0.0), 0.0)
                r_unif = r_unif_new
        elif ion_path is not None and Path(ion_path).is_file():
            ion_data = json.loads(Path(ion_path).read_text(encoding="utf-8"))
            orbs = ion_data["orbitals"]
            max_rc = max(float(o["rcut_bohr"]) for o in orbs)
            pao_r = np.linspace(0.0, max_rc, 201, dtype=np.float64)
            pao_vals_list = []
            pao_l_list = []
            pao_n_list = []
            pao_zeta_list = []
            pao_rcut_list = []
            for o in orbs:
                r_o = np.asarray(o["r_bohr"], dtype=np.float64)
                v_o = np.asarray(o["values"], dtype=np.float64)
                rc_o = float(o["rcut_bohr"])
                interp_v = np.where(pao_r <= rc_o, np.interp(pao_r, r_o, v_o, right=0.0), 0.0)
                pao_vals_list.append(interp_v)
                pao_l_list.append(int(o["l"]))
                pao_n_list.append(int(o["n"]))
                pao_zeta_list.append(int(o["zeta"]))
                pao_rcut_list.append(rc_o)
            pao_values = np.stack(pao_vals_list, axis=0)
        else:
            pao_r, pao_values, pao_l_list, pao_n_list, pao_zeta_list, pao_rcut_list = (
                _generate_confined_pao_orbitals(
                    r_unif,
                    shells,
                    z_val,
                    rcut=float(kwargs.get("rcut_bohr", 6.0)),
                    zeta_count=int(kwargs.get("zeta_count", 2)),
                )
            )

        # Assign unique sequential radial pao_iz counter per angular momentum l
        iz_counter: dict[int, int] = {}
        pao_iz_list: list[int] = []
        for l_val in pao_l_list:
            cnt = iz_counter.get(l_val, 0)
            pao_iz_list.append(cnt)
            iz_counter[l_val] = cnt + 1

        return SpeciesData(
            symbol=symbol,
            atomic_number=atomic_number,
            valence_charge=z_val,
            mass_amu=mass,
            r_grid=jnp.asarray(r_unif, dtype=jnp.float64),
            v_local=jnp.asarray(v_loc, dtype=jnp.float64),
            v_na=jnp.asarray(v_na, dtype=jnp.float64),
            rho_na=jnp.asarray(rho_na, dtype=jnp.float64),
            rho_core=jnp.asarray(rho_core, dtype=jnp.float64),
            pao_r=jnp.asarray(pao_r, dtype=jnp.float64),
            pao_values=jnp.asarray(pao_values, dtype=jnp.float64),
            pao_l=tuple(pao_l_list),
            pao_n=tuple(pao_n_list),
            pao_zeta=tuple(pao_zeta_list),
            pao_iz=tuple(pao_iz_list),
            pao_rcut=tuple(pao_rcut_list),
            kb_r=jnp.asarray(kb_r_unif, dtype=jnp.float64),
            kb_projectors=jnp.asarray(np.stack(kb_projs, axis=0), dtype=jnp.float64),
            kb_l=tuple(kb_l),
            kb_j2=tuple(kb_j2),
            kb_energies=jnp.asarray(kb_energies, dtype=jnp.float64),
            kb_rcut=tuple(kb_rcut),
            has_nlcc=has_nlcc,
            has_spin_orbit=has_soc,
            rcut_na=rcut_na,
            psml_sha256=psml_sha,
            tau_core=jnp.asarray(tau_core, dtype=jnp.float64),
            has_tau_core=has_tau_core,
        )

    def build_basis(
        self,
        structure: Structure,
        species_map: dict[str, SpeciesData],
        include_soc_projectors: bool = False,
    ) -> BasisData:
        """Assemble flat unit-cell NAO and KB projector index tables across all atoms.

        Theory:
            Expands each radial shell :math:`(n, l, \\zeta)` into :math:`2l + 1` real
            spherical harmonic orbitals ordered canonically :math:`m \\in [-l, l]`,
            and similarly expands KB projectors.

        Algorithm:
            Iterates unit-cell atoms, accumulating flat tuples and JAX arrays for
            O(1) orbital-to-atom and orbital-to-shell lookup.

        Parameters:
            structure: Periodic or molecular ``Structure``.
            species_map: Mapping from species label to ``SpeciesData``.
            include_soc_projectors: If ``False``, collapses ``j``-split KB channels
                to scalar-relativistic channels when present, or keeps scalar channels.

        Returns:
            Immutable ``BasisData`` PyTree.

        Invariants:
            - ``len(atom_orb_offsets) == structure.n_atoms + 1``.
            - ``total_valence_electrons == sum(valence_charge)``.

        SIESTA Mapping:
            Replaces ``atomlist.F`` (``iphorb``, ``iaorb``, ``lasto``, ``iphkb``).
        """
        ordered_species: list[SpeciesData] = unify_species_tables(
            [species_map[lbl] for lbl in structure.species_labels]
        )
        orb_atom: list[int] = []
        orb_species: list[int] = []
        orb_shell: list[int] = []
        orb_l: list[int] = []
        orb_m: list[int] = []
        orb_iz: list[int] = []
        orb_rcut: list[float] = []
        atom_offsets: list[int] = [0]

        kb_atom: list[int] = []
        kb_species: list[int] = []
        kb_l: list[int] = []
        kb_m: list[int] = []
        kb_j2: list[int] = []
        kb_energies: list[float] = []
        kb_rcut: list[float] = []

        total_z_val = 0.0
        for ia, sp_idx in enumerate(structure.species_indices):
            sp = ordered_species[sp_idx]
            total_z_val += sp.valence_charge
            for sh_idx, (l_q, iz_q, rc_q) in enumerate(
                zip(sp.pao_l, sp.pao_iz, sp.pao_rcut, strict=True)
            ):
                for m_q in range(-l_q, l_q + 1):
                    orb_atom.append(ia)
                    orb_species.append(sp_idx)
                    orb_shell.append(sh_idx)
                    orb_l.append(l_q)
                    orb_m.append(m_q)
                    orb_iz.append(iz_q)
                    orb_rcut.append(rc_q)
            atom_offsets.append(len(orb_atom))

            sp_kb_e = np.asarray(sp.kb_energies, dtype=np.float64)
            for k_idx, (l_k, j2_k, rc_k) in enumerate(
                zip(sp.kb_l, sp.kb_j2, sp.kb_rcut, strict=True)
            ):
                if not include_soc_projectors and j2_k > 0 and l_k > 0:
                    # Weight by (2j+1)/(2*(2l+1)) when using scalar average or keep all if SOC
                    weight = (j2_k + 1.0) / (2.0 * (2 * l_k + 1.0))
                else:
                    weight = 1.0
                ek_val = float(sp_kb_e[k_idx]) * weight
                for m_k in range(-l_k, l_k + 1):
                    kb_atom.append(ia)
                    kb_species.append(sp_idx)
                    kb_l.append(l_k)
                    kb_m.append(m_k)
                    kb_j2.append(j2_k)
                    kb_energies.append(ek_val)
                    kb_rcut.append(rc_k)

        return BasisData(
            species=tuple(ordered_species),
            n_orb=len(orb_atom),
            n_kb=len(kb_atom),
            total_valence_electrons=total_z_val,
            orb_atom=tuple(orb_atom),
            orb_species=tuple(orb_species),
            orb_shell=tuple(orb_shell),
            orb_l=tuple(orb_l),
            orb_m=tuple(orb_m),
            orb_iz=tuple(orb_iz),
            orb_rcut=jnp.asarray(orb_rcut, dtype=jnp.float64),
            atom_orb_offsets=tuple(atom_offsets),
            kb_atom=tuple(kb_atom),
            kb_species=tuple(kb_species),
            kb_l=tuple(kb_l),
            kb_m=tuple(kb_m),
            kb_j2=tuple(kb_j2),
            kb_energies=jnp.asarray(kb_energies, dtype=jnp.float64),
            kb_rcut=jnp.asarray(kb_rcut, dtype=jnp.float64),
        )


def _expand_ranges(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For ranges ``[starts[g], starts[g] + counts[g])`` return ``(group_of_item, item)`` in range order."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(np.sum(counts))
    group = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    item = np.repeat(np.asarray(starts, dtype=np.int64), counts) + (np.arange(total, dtype=np.int64) - first)
    return group, item


def _triplet_keys(idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unique ``int64`` codes of ``(idx, T)`` rows (integer shifts ``T``), for deduplication."""
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    half = int(np.max(np.abs(t))) + 1
    stride = 2 * half + 1
    return (((idx * stride + (t[:, 0] + half)) * stride + (t[:, 1] + half)) * stride) + (t[:, 2] + half)


def build_sparse_pattern(
    structure: Structure,
    basis: BasisData,
    extra_edges: list[tuple[int, int, tuple[int, int, int]]] | None = None,
    symmetry_closed: bool = False,
) -> SparsePattern:
    r"""Build the interacting static CSR/COO orbital-pair ``SparsePattern``.

    Theory:
        Collects all directed orbital pairs :math:`(\mu, \nu, \mathbf{T})` whose center
        distance :math:`|\mathbf{R}_{I(\nu)} + \mathbf{A}\mathbf{T} - \mathbf{R}_{I(\mu)}|`
        lies within :math:`r_c(\mu) + r_c(\nu)` OR that share an indirect Kleinman-Bylander
        projector bridge :math:`(\mu \leftrightarrow \chi_k \leftrightarrow \nu)`.
        Includes explicit transpose edges :math:`(\nu, \mu, -\mathbf{T})` for exact Hermiticity.

    Algorithm:
        Determines supercell bounding box ``nsc`` from max orbital + KB cutoffs and
        reciprocal lattice norms. Per atom, one vectorised distance table over all atoms and
        images gives (a) the atom pairs within the largest orbital cut-offs, refined per orbital
        pair, and (b) the atom-to-KB-projector links, joined through their common projector atom
        into the indirect pairs. Symmetry-closure ``extra_edges`` and the transposes are merged,
        duplicates are removed through ``int64`` edge codes, and the CSR ``row_ptr`` and
        ``transpose_edge_idx`` follow from one sort (cost linear in the number of atoms times the
        number of images, plus the number of edges).

    Parameters:
        structure: Periodic/molecular ``Structure``.
        basis: Unit-cell ``BasisData``.
        extra_edges: Optional extra ``(src_orb, dst_unit_orb, (sx, sy, sz))`` edges
            for symmetry closure.
        symmetry_closed: Whether the returned pattern is certified symmetry-closed.

    Returns:
        Immutable ``SparsePattern`` PyTree.

    Invariants:
        - Every edge ``(u, v, T)`` has its exact Hermitian partner ``(v, u, -T)`` in the table.
        - ``row_ptr[-1] == n_edges``.
        - Edges are sorted by ``(src, cell index, dst)``; the box shifts come first, in
          lexicographic order with ``(0, 0, 0)`` at index 0, and any further shift (from
          ``extra_edges``) is appended in sorted order.

    SIESTA Mapping:
        Replaces ``nsc`` determination and ``hsparse`` in ``hsparse.F`` and ``listsc_init``.
    """
    cell = np.asarray(structure.cell, dtype=np.float64)
    pos = np.asarray(structure.positions, dtype=np.float64)
    rc_orb = np.asarray(basis.orb_rcut, dtype=np.float64)
    rc_kb = np.asarray(basis.kb_rcut, dtype=np.float64)
    max_rc_orb = float(np.max(rc_orb)) if len(rc_orb) else 0.0
    max_rc_kb = float(np.max(rc_kb)) if len(rc_kb) else 0.0
    max_interaction_r = 2.0 * max_rc_orb + 2.0 * max_rc_kb

    inv_cell = np.linalg.inv(cell)
    # Bounding supercell half-widths along each lattice vector
    b_norms = np.linalg.norm(inv_cell, axis=1)
    n_half = [
        int(np.ceil(max_interaction_r * b_norms[ax])) if structure.pbc[ax] else 0
        for ax in range(3)
    ]
    nsc = tuple(2 * h + 1 for h in n_half)

    # Enumerate supercell shift triplets with (0,0,0) first
    shifts_list: list[tuple[int, int, int]] = [(0, 0, 0)]
    for s0 in range(-n_half[0], n_half[0] + 1):
        for s1 in range(-n_half[1], n_half[1] + 1):
            for s2 in range(-n_half[2], n_half[2] + 1):
                if (s0, s1, s2) != (0, 0, 0):
                    shifts_list.append((s0, s1, s2))
    box_shifts = np.asarray(shifts_list, dtype=np.int64)
    t_box = np.stack([cell @ np.asarray(t, dtype=np.float64) for t in shifts_list], axis=0)

    n_orb = basis.n_orb
    n_atoms = structure.n_atoms
    orb_atom = np.asarray(basis.orb_atom, dtype=np.int64)
    offs = np.asarray(basis.atom_orb_offsets, dtype=np.int64)
    n_orb_atom = offs[1:] - offs[:-1]

    atom_max_kb = np.zeros(n_atoms, dtype=np.float64)
    if rc_kb.size:
        np.maximum.at(atom_max_kb, np.asarray(basis.kb_atom, dtype=np.int64), rc_kb)
    atom_max_orb = np.zeros(n_atoms, dtype=np.float64)
    if rc_orb.size:
        np.maximum.at(atom_max_orb, orb_atom, rc_orb)

    # Atom-level distances |R_j + A T - R_i| for all (i, j, T): shape (n_atoms, n_atoms, n_shifts) per row i.
    shifted = pos[:, None, :] + t_box[None, :, :]

    def _dist_row(ia: int) -> np.ndarray:
        d = shifted - pos[ia][None, None, :]
        return np.sqrt(np.einsum("jsa,jsa->js", d, d))

    # Atom -> KB-atom links (i, K, T_k): |R_K + A T_k - R_i| <= max rc(orb on i) + max rc(KB on K)
    link_i: list[np.ndarray] = []
    link_k: list[np.ndarray] = []
    link_s: list[np.ndarray] = []
    has_kb = atom_max_kb > 0.0
    dist_rows: list[np.ndarray] = []
    for ia in range(n_atoms):
        d_row = _dist_row(ia)
        dist_rows.append(d_row)
        ka, si = np.nonzero((d_row <= (atom_max_orb[ia] + atom_max_kb)[:, None] + 1e-9) & has_kb[:, None])
        link_i.append(np.full(ka.size, ia, dtype=np.int64))
        link_k.append(ka.astype(np.int64))
        link_s.append(si.astype(np.int64))
    l_i = np.concatenate(link_i) if link_i else np.zeros(0, dtype=np.int64)
    l_k = np.concatenate(link_k) if link_k else np.zeros(0, dtype=np.int64)
    l_t = box_shifts[np.concatenate(link_s)] if link_s else np.zeros((0, 3), dtype=np.int64)
    k_order = np.argsort(l_k, kind="stable")
    k_counts = np.bincount(l_k, minlength=n_atoms).astype(np.int64)
    k_offs = np.concatenate(([0], np.cumsum(k_counts)[:-1])).astype(np.int64)
    i_counts = np.bincount(l_i, minlength=n_atoms).astype(np.int64)
    i_offs = np.concatenate(([0], np.cumsum(i_counts)[:-1])).astype(np.int64)

    edge_mu: list[np.ndarray] = []
    edge_nu: list[np.ndarray] = []
    edge_t: list[np.ndarray] = []
    for ia in range(n_atoms):
        mu_lo, n_mu = int(offs[ia]), int(n_orb_atom[ia])
        if n_mu == 0:
            continue
        mu_loc = np.arange(mu_lo, mu_lo + n_mu, dtype=np.int64)
        # Direct orbital-orbital overlaps: atom-level test, then per orbital pair
        d_row = dist_rows[ia]
        ja, si = np.nonzero(d_row <= (atom_max_orb[ia] + atom_max_orb)[:, None] + 1e-9)
        if ja.size:
            grp, nu = _expand_ranges(offs[ja], n_orb_atom[ja])
            dist = d_row[ja[grp], si[grp]]
            mu_b = np.repeat(mu_loc[None, :], nu.size, axis=0)
            keep = dist[:, None] <= (rc_orb[mu_b] + rc_orb[nu][:, None]) + 1e-9
            pr, pc = np.nonzero(keep)
            edge_mu.append(mu_loc[pc])
            edge_nu.append(nu[pr])
            edge_t.append(box_shifts[si[grp[pr]]])
        # KB-bridged pairs mu - chi_K - nu: T_ij = T_ik - T_jk for links (i, K, T_ik) and (j, K, T_jk)
        rows_i = np.arange(i_offs[ia], i_offs[ia] + i_counts[ia], dtype=np.int64)
        if rows_i.size:
            ks = l_k[rows_i]
            g2, rows_j_pos = _expand_ranges(k_offs[ks], k_counts[ks])
            rows_j = k_order[rows_j_pos]
            ja_b = l_i[rows_j]
            t_ij = l_t[rows_i[g2]] - l_t[rows_j]
            atom_key = _triplet_keys(ja_b, t_ij)
            _, uniq = np.unique(atom_key, return_index=True)
            ja_b, t_ij = ja_b[uniq], t_ij[uniq]
            grp, nu = _expand_ranges(offs[ja_b], n_orb_atom[ja_b])
            edge_mu.append(np.repeat(mu_loc[None, :], nu.size, axis=0).ravel())
            edge_nu.append(np.repeat(nu, n_mu))
            edge_t.append(np.repeat(t_ij[grp], n_mu, axis=0))

    e_mu = np.concatenate(edge_mu) if edge_mu else np.zeros(0, dtype=np.int64)
    e_nu = np.concatenate(edge_nu) if edge_nu else np.zeros(0, dtype=np.int64)
    e_t = np.concatenate(edge_t, axis=0) if edge_t else np.zeros((0, 3), dtype=np.int64)
    if extra_edges:
        x = np.asarray([(m, n, t[0], t[1], t[2]) for m, n, t in extra_edges], dtype=np.int64).reshape(-1, 5)
        e_mu = np.concatenate([e_mu, x[:, 0]])
        e_nu = np.concatenate([e_nu, x[:, 1]])
        e_t = np.concatenate([e_t, x[:, 2:5]], axis=0)

    # Guarantee Hermitian transpose closure (u, v, T) <-> (v, u, -T), then deduplicate
    a_mu = np.concatenate([e_mu, e_nu])
    a_nu = np.concatenate([e_nu, e_mu])

    # Shift table: the box shifts first (in their order), then any further shifts in sorted order
    t_half = int(max(np.max(np.abs(e_t)) if e_t.size else 0, np.max(np.abs(box_shifts)))) + 1
    t_stride = 2 * t_half + 1

    def _t_code(t: np.ndarray) -> np.ndarray:
        return ((t[:, 0] + t_half) * t_stride + (t[:, 1] + t_half)) * t_stride + (t[:, 2] + t_half)

    # Codes of all edges: those of the transposes follow from code(-T) = 2 t_half (s^2 + s + 1) - code(T).
    n_codes = t_stride**3
    e_code = _t_code(e_t) if e_t.size else np.zeros(0, dtype=np.int64)
    a_code = np.concatenate([e_code, 2 * t_half * (t_stride * t_stride + t_stride + 1) - e_code])
    used_codes = np.flatnonzero(np.bincount(a_code, minlength=n_codes)) if a_code.size else np.zeros(0, dtype=np.int64)
    box_codes = _t_code(box_shifts)
    extra_codes = np.setdiff1d(used_codes, box_codes)
    if extra_codes.size:
        ex = np.stack(
            [extra_codes // (t_stride * t_stride) - t_half, (extra_codes // t_stride) % t_stride - t_half,
             extra_codes % t_stride - t_half],
            axis=1,
        )
        ex_order = np.lexsort((ex[:, 2], ex[:, 1], ex[:, 0]))
        shifts_list.extend((int(row[0]), int(row[1]), int(row[2])) for row in ex[ex_order])
    shifts_np = np.asarray(shifts_list, dtype=np.int64)
    all_codes = _t_code(shifts_np)
    n_sh = len(shifts_list)
    # Shift codes are small integers: a lookup table maps them to shift indices.
    code_to_idx = np.full(n_codes, -1, dtype=np.int64)
    code_to_idx[all_codes] = np.arange(n_sh, dtype=np.int64)
    c_all = code_to_idx[a_code]

    # Sort by (src, shift index, dst) through one monotone key; dedupe
    keys = np.unique((a_mu * n_sh + c_all) * max(n_orb, 1) + a_nu)
    src_orb = (keys // (n_sh * max(n_orb, 1))).astype(np.int32)
    edge_cell_idx = ((keys // max(n_orb, 1)) % n_sh).astype(np.int32)
    dst_unit_orb = (keys % max(n_orb, 1)).astype(np.int32)

    t_vecs = np.stack([cell @ np.asarray(t, dtype=np.float64) for t in shifts_list], axis=0)
    edge_vectors = (pos[orb_atom[dst_unit_orb]] + t_vecs[edge_cell_idx]) - pos[orb_atom[src_orb]]
    neg_idx = code_to_idx[_t_code(-shifts_np)]
    t_keys = (dst_unit_orb.astype(np.int64) * n_sh + neg_idx[edge_cell_idx]) * max(n_orb, 1) + src_orb
    transpose_idx = np.searchsorted(keys, t_keys).astype(np.int32)
    row_counts = np.bincount(src_orb, minlength=n_orb).astype(np.int32)
    shifts_arr = np.asarray(shifts_list, dtype=np.int32)

    row_ptr = np.zeros(n_orb + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(row_counts)

    return SparsePattern(
        n_orb=n_orb,
        n_super_cells=len(shifts_list),
        nsc=(int(nsc[0]), int(nsc[1]), int(nsc[2])),
        cell_shifts=jnp.asarray(shifts_arr, dtype=jnp.int32),
        src_orb=jnp.asarray(src_orb, dtype=jnp.int32),
        dst_unit_orb=jnp.asarray(dst_unit_orb, dtype=jnp.int32),
        edge_cell_idx=jnp.asarray(edge_cell_idx, dtype=jnp.int32),
        edge_vectors=jnp.asarray(edge_vectors, dtype=jnp.float64),
        row_ptr=jnp.asarray(row_ptr, dtype=jnp.int32),
        transpose_edge_idx=jnp.asarray(transpose_idx, dtype=jnp.int32),
        symmetry_closed=symmetry_closed,
    )
