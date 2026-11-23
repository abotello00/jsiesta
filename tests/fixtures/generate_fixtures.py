"""Generate frozen norm-conserving PSML v1.1 XML pseudopotentials and .ion basis fixtures.

Theory:
    Constructs smooth, norm-conserving Troullier-style local potentials,
    Kleinman-Bylander projectors (including scalar-relativistic and j-dependent
    spin-orbit channels for Co), valence electron density, NLCC pseudo-core
    density, and confined numerical atomic orbitals (NAOs) vanishing smoothly at r_c.

Algorithm:
    Uses a logarithmic/shifted radial grid r_i = r0 * (exp(a * i) - 1) and
    writes standard PSML v1.1 XML files (no legacy .psf files).
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SPECIES_SPECS = [
    {
        "symbol": "H",
        "z": 1,
        "z_val": 1.0,
        "mass": 1.008,
        "rc_loc": 1.1,
        "shells": [(1, 0, 1, 1.0)],  # (n, l, occ_max, occ)
        "kb": [(0, 1.1, -0.35, 0)],
        "nlcc": False,
        "soc": False,
        "pao_rcut": 5.0,
    },
    {
        "symbol": "C",
        "z": 6,
        "z_val": 4.0,
        "mass": 12.011,
        "rc_loc": 1.4,
        "shells": [(2, 0, 2, 2.0), (2, 1, 6, 2.0)],
        "kb": [(0, 1.4, 2.2, 0), (1, 1.4, -0.85, 0)],
        "nlcc": False,
        "soc": False,
        "pao_rcut": 5.5,
    },
    {
        "symbol": "O",
        "z": 8,
        "z_val": 6.0,
        "mass": 15.999,
        "rc_loc": 1.25,
        "shells": [(2, 0, 2, 2.0), (2, 1, 6, 4.0)],
        "kb": [(0, 1.25, 3.1, 0), (1, 1.25, -1.2, 0)],
        "nlcc": False,
        "soc": False,
        "pao_rcut": 5.0,
    },
    {
        "symbol": "Si",
        "z": 14,
        "z_val": 4.0,
        "mass": 28.085,
        "rc_loc": 1.8,
        "shells": [(3, 0, 2, 2.0), (3, 1, 6, 2.0)],
        "kb": [(0, 1.8, 1.4, 0), (1, 1.8, 0.35, 0), (2, 1.8, -0.45, 0)],
        "nlcc": True,
        "soc": False,
        "pao_rcut": 6.5,
    },
    {
        "symbol": "Co",
        "z": 27,
        "z_val": 9.0,
        "mass": 58.933,
        "rc_loc": 1.9,
        "shells": [(4, 0, 2, 2.0), (3, 2, 10, 7.0)],
        # Include j-dependent spin-orbit channels (j2 = 2*j = 3, 5 for l=2; 1, 3 for l=1)
        "kb": [
            (0, 1.9, 1.8, 0),
            (1, 1.9, 0.75, 1),
            (1, 1.9, 0.72, 3),
            (2, 1.9, -1.35, 3),
            (2, 1.9, -1.28, 5),
        ],
        "nlcc": True,
        "soc": True,
        "pao_rcut": 6.5,
    },
]


def _make_grid(n_pts: int = 350, r_max: float = 12.0) -> np.ndarray:
    a = 0.015
    i = np.arange(n_pts, dtype=np.float64)
    r = (np.exp(a * i) - 1.0)
    r = r * (r_max / r[-1])
    return r


def _erf_vec(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf
    return erf(x)


def write_psml_fixture(spec: dict, out_path: Path) -> None:
    r = _make_grid()
    z_val = spec["z_val"]
    rc_loc = spec["rc_loc"]

    # Smooth norm-conserving local pseudopotential V_loc(r) -> -Z_val / r as r -> inf
    r_safe = np.maximum(r, 1e-8)
    v_loc = -z_val * _erf_vec(r_safe / rc_loc) / r_safe
    v_loc[0] = -z_val * (2.0 / (np.sqrt(np.pi) * rc_loc))

    # Smooth Gaussian valence charge integrating to z_val: int 4*pi*r^2 rho(r) dr = z_val
    sigma_v = rc_loc * 0.9
    rho_val = (z_val / (np.pi ** 1.5 * sigma_v ** 3)) * np.exp(-((r / sigma_v) ** 2))

    root = ET.Element("psml", {"version": "1.1", "energy_unit": "hartree", "length_unit": "bohr"})
    prov = ET.SubElement(root, "provenance", {"record-number": "1", "creator": "jsiesta-phase0-fixture"})
    ET.SubElement(prov, "annotation", {"action": "frozen-reference-fixture"})

    pa = ET.SubElement(
        root,
        "pseudo-atom-spec",
        {
            "atomic-label": spec["symbol"],
            "atomic-number": str(spec["z"]),
            "z-pseudo": f"{z_val:.6f}",
            "atomic-mass": f"{spec['mass']:.6f}",
            "relativity": "dirac" if spec["soc"] else "scalar",
            "core-corrections": "yes" if spec["nlcc"] else "no",
        },
    )
    vc = ET.SubElement(pa, "valence-configuration", {"total-valence-charge": f"{z_val:.6f}"})
    for n, l_ang, _, occ in spec["shells"]:
        ET.SubElement(
            vc,
            "shell",
            {"n": str(n), "l": ["s", "p", "d", "f"][l_ang], "occupation": f"{occ:.6f}"},
        )

    grid = ET.SubElement(root, "grid", {"npts": str(len(r))})
    ga = ET.SubElement(grid, "grid-data")
    ga.text = " ".join(f"{val:.12e}" for val in r)

    loc_el = ET.SubElement(root, "local-potential", {"type": "psml"})
    loc_rad = ET.SubElement(loc_el, "radfunc")
    loc_data = ET.SubElement(loc_rad, "data")
    loc_data.text = " ".join(f"{val:.12e}" for val in v_loc)

    # Valence charge (4*pi*r^2 * rho_val or rho_val; store pointwise radial density rho(r))
    vc_el = ET.SubElement(root, "valence-charge", {"total-charge": f"{z_val:.6f}"})
    vc_rad = ET.SubElement(vc_el, "radfunc")
    vc_data = ET.SubElement(vc_rad, "data")
    vc_data.text = " ".join(f"{val:.12e}" for val in rho_val)

    if spec["nlcc"]:
        sigma_c = rc_loc * 0.45
        z_core_model = 0.35 * z_val
        rho_core = (z_core_model / (np.pi ** 1.5 * sigma_c ** 3)) * np.exp(-((r / sigma_c) ** 2))
        pc_el = ET.SubElement(root, "pseudo-core-charge", {"matching-radius": f"{rc_loc:.4f}"})
        pc_rad = ET.SubElement(pc_el, "radfunc")
        pc_data = ET.SubElement(pc_rad, "data")
        pc_data.text = " ".join(f"{val:.12e}" for val in rho_core)

    nl_el = ET.SubElement(root, "nonlocal-projectors", {"set": "spin_orbit" if spec["soc"] else "scalar_relativistic"})
    for idx, (l_ang, rc_kb, ekb, j2) in enumerate(spec["kb"]):
        # Smooth localized KB projector chi_l(r) = A * r^l * exp(-(r/sigma)^2) * cutoff(r/rc_kb)
        sigma_kb = rc_kb * 0.55
        cut = np.where(r < rc_kb, (1.0 - (r / rc_kb) ** 2) ** 2, 0.0)
        chi = (r ** l_ang) * np.exp(-((r / sigma_kb) ** 2)) * cut
        norm = np.sqrt(np.trapezoid((chi ** 2) * (r ** 2), r))
        if norm > 0:
            chi = chi / norm
        attrs = {
            "seq": str(idx + 1),
            "l": ["s", "p", "d", "f"][l_ang],
            "ekb": f"{ekb:.12e}",
            "rcut": f"{rc_kb:.6f}",
        }
        if j2 > 0:
            attrs["j"] = f"{0.5 * j2:.1f}"
        proj = ET.SubElement(nl_el, "proj", attrs)
        prad = ET.SubElement(proj, "radfunc")
        pdata = ET.SubElement(prad, "data")
        pdata.text = " ".join(f"{val:.12e}" for val in chi)

    ET.indent(root, space="  ")
    out_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def write_ion_fixture(spec: dict, out_path: Path) -> None:
    """Write a companion confined NAO .ion.json reference basis table."""
    import json

    r = np.linspace(0.0, spec["pao_rcut"], 201)
    rc = spec["pao_rcut"]
    orbitals = []
    for zeta in (1, 2):
        for n, l_ang, _, _ in spec["shells"]:
            # Confined Slater-like radial function R_nl(r) with smooth Sankey cutoff
            alpha = (1.1 if zeta == 1 else 2.2) * (1.0 + 0.25 * spec["z_val"] / max(n, 1))
            cut = np.exp(-0.15 / np.maximum(rc - r, 1e-4))
            cut[-1] = 0.0
            rad = (r ** l_ang) * np.exp(-alpha * r) * cut
            norm = np.sqrt(np.trapezoid((rad ** 2) * (r ** 2), r))
            if norm > 0:
                rad = rad / norm
            orbitals.append({
                "n": n,
                "l": l_ang,
                "zeta": zeta,
                "rcut_bohr": rc,
                "r_bohr": [float(x) for x in r],
                "values": [float(x) for x in rad],
            })
    payload = {
        "symbol": spec["symbol"],
        "atomic_number": spec["z"],
        "valence_charge": spec["z_val"],
        "orbitals": orbitals,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parent / "psml"
    base.mkdir(parents=True, exist_ok=True)
    for spec in SPECIES_SPECS:
        write_psml_fixture(spec, base / f"{spec['symbol']}.psml")
        write_ion_fixture(spec, base / f"{spec['symbol']}.ion.json")


if __name__ == "__main__":
    main()
