"""jsiesta's neutral-atom pair energy equals the exact value of SIESTA's own formula (open issue O-8, D-025).

SIESTA's printed ``Ena`` deviates smoothly from the exact evaluation of ``naefs + dnaefs`` because its tables have a
fixed numerical resolution; jsiesta's two-centre integral is converged and matches the exact value. The displacement
derivative of this term is what drives the force and stress differences seen against SIESTA for H2O, triclinic Si and
Co (``tests/test_siesta_parity.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import jax
import numpy as np
import pytest

from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.testing.external_refs import ref_path
from jax_siesta.testing.siesta_adapter import run_siesta_reference
from jax_siesta.testing.siesta_na_reference import siesta_pair_energy_ha

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parent
HA_EV = 27.211386245988


def _siesta_ena_ev(ref_dir: Path) -> float:
    return float(re.findall(r"siesta:\s+Ena\s*=\s*([-+0-9.eE]+)", (ref_dir / "si.out").read_text())[-1])


@pytest.mark.slow
@pytest.mark.external_refs("siesta_bin_or_cache", "pseudodojo_sr_pbe_dir")
@pytest.mark.parametrize("case_name", ["h2o_gamma", "si_triclinic"])
def test_jsiesta_pair_energy_matches_exact_siesta_formula(case_name: str, tmp_path: Path, record_residuals) -> None:
    src = json.loads((ROOT / "manifests" / "cases" / case_name / "case.json").read_text())
    base = {k: v for k, v in src.items() if k != "dispersion"}
    base.update({"d_tol": 1.0e-6, "h_tol_ha": 1.0e-6, "e_tol_ha": 1.0e-8, "max_scf_iterations": 300, "xc": "PBE"})
    h = 0.02  # Bohr, atom 0 along z
    out: dict[int, dict[str, float]] = {}
    for sgn in (1, -1):
        work = tmp_path / f"{case_name}_{sgn:+d}"
        work.mkdir()
        for label in set(base["species"]):
            shutil.copy2(ref_path("pseudodojo_sr_pbe_dir") / f"{label}.psml", work / f"{label}.psml")
        pos = np.array(base["positions_bohr"], dtype=float)
        pos[0, 2] += sgn * h
        case = {**base, "positions_bohr": pos.tolist()}
        (work / "case.json").write_text(json.dumps(case), encoding="utf-8")
        ref = run_siesta_reference(work / "case.json", work / "ref", mode="siesta", psml_dir=work, timeout=3600)
        c = load_case_json(work / "case.json")
        structure, config = c["structure"], c["config"]
        loader = PSMLBasisLoader()
        species = {
            lbl: loader.load_species(
                lbl, str(work / f"{lbl}.psml"), ion_path=str(work / "ref" / f"{lbl}.ion.xml"), prefer_lj_projectors=False
            )
            for lbl in structure.species_labels
        }
        basis = loader.build_basis(structure, species)
        two_center = TwoCenterEvaluator().evaluate(structure, basis, build_sparse_pattern(structure, basis), config)
        naefs, dnaefs = siesta_pair_energy_ha(structure, work / "ref")
        out[sgn] = {
            "jsiesta": float(two_center.e_na_pair) * HA_EV,
            "exact": (naefs + dnaefs) * HA_EV,
            "siesta": _siesta_ena_ev(work / "ref"),
        }
        del ref

    # Absolute: the exact formula reproduces SIESTA's printed Ena to 0.04 % (self term included).
    for key in ("siesta",):
        assert abs(out[1]["exact"] - out[1][key]) < 0.05, (out[1]["exact"], out[1][key])

    d = {k: out[1][k] - out[-1][k] for k in ("jsiesta", "exact", "siesta")}
    err_jsiesta = abs(d["jsiesta"] - d["exact"])
    err_siesta = abs(d["siesta"] - d["exact"])
    record_residuals(
        {f"{case_name}.jsiesta_vs_exact_ev": err_jsiesta, f"{case_name}.siesta_vs_exact_ev": err_siesta},
        {f"{case_name}.jsiesta_vs_exact_ev": 3.0e-3 * abs(d["exact"])},
    )
    assert err_jsiesta < 3.0e-3 * abs(d["exact"]), d  # measured 0.05-0.07 %
    assert err_siesta > 2.0 * err_jsiesta, d  # SIESTA's tables are the less accurate side (measured 4x-45x)


@pytest.mark.slow
@pytest.mark.external_refs("siesta_bin_or_cache", "co_pilot_dir")
def test_co_strain_pair_energy_against_exact_siesta_formula(tmp_path: Path, record_residuals) -> None:
    """Co hcp (nspin 2, NLCC, semicore) under a +-1e-3 strain along z: both codes deviate ~0.1 % from the exact formula.

    Unlike H2O and PseudoDojo Si, jsiesta's pair energy is not the exact value of SIESTA's formula here (-0.075 %): the
    two are the same quantity only if the ``vna`` table is the Poisson potential of the density jsiesta uses (the
    PAO-based ``rho_NA``), which holds to 0.05-0.07 % for PseudoDojo tables and 0.075 % for this older Co pseudopotential.
    SIESTA's printed ``Ena`` is +0.097 % off the exact value in the other direction; together they make the
    -0.0495 eV per unit strain of D-021.
    """
    ang = 1.0 / 0.529177210903
    a_co, c_co = 2.507100, 4.069500
    out: dict[float, dict[str, float]] = {}
    for eps in (1.0e-3, -1.0e-3):
        work = tmp_path / f"co_{eps:+.4f}"
        work.mkdir()
        shutil.copy2(ref_path("co_pilot_dir") / "Co.psml", work / "Co.psml")
        cell = np.array(
            [[a_co, -0.5 * a_co, 0.0], [0.0, a_co * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c_co * (1.0 + eps)]]
        ) * ang
        frac = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]])
        case = {
            "name": "co_strain", "cell_columns_bohr": cell.tolist(), "positions_bohr": (cell @ frac.T).T.tolist(),
            "species": ["Co", "Co"], "k_grid": [4, 4, 2], "mesh_cutoff_ry": 120.0, "xc": "PBE", "nspin": 2,
            "initial_moments": [[0, 0, 2.0], [0, 0, 2.0]],
        }
        (work / "case.json").write_text(json.dumps(case), encoding="utf-8")
        run_siesta_reference(work / "case.json", work / "ref", mode="siesta", psml_dir=work, timeout=3600)
        c = load_case_json(work / "case.json")
        structure, config = c["structure"], c["config"]
        loader = PSMLBasisLoader()
        species = {
            "Co": loader.load_species(
                "Co", str(work / "Co.psml"), ion_path=str(work / "ref" / "Co.ion.xml"), prefer_lj_projectors=False
            )
        }
        basis = loader.build_basis(structure, species)
        two_center = TwoCenterEvaluator().evaluate(structure, basis, build_sparse_pattern(structure, basis), config)
        naefs, dnaefs = siesta_pair_energy_ha(structure, work / "ref")
        out[eps] = {
            "jsiesta": float(two_center.e_na_pair) * HA_EV,
            "exact": (naefs + dnaefs) * HA_EV,
            "siesta": _siesta_ena_ev(work / "ref"),
        }
    # The exact formula reproduces SIESTA's printed Ena to better than 1 meV in absolute value for this pseudopotential.
    assert abs(out[1.0e-3]["exact"] - out[1.0e-3]["siesta"]) < 5.0e-3, out[1.0e-3]
    d = {k: (out[1.0e-3][k] - out[-1.0e-3][k]) / 2.0e-3 for k in ("jsiesta", "exact", "siesta")}  # eV per unit strain
    rel_js = (d["jsiesta"] - d["exact"]) / d["exact"]
    rel_si = (d["siesta"] - d["exact"]) / d["exact"]
    record_residuals({"co.jsiesta_vs_exact_rel": abs(rel_js), "co.siesta_vs_exact_rel": abs(rel_si)}, {})
    assert abs(rel_js) < 2.0e-3 and abs(rel_si) < 2.0e-3, (d, rel_js, rel_si)  # measured -7.5e-4 and +9.7e-4
    assert rel_js < 0.0 < rel_si  # opposite directions: the -0.0495 gap splits into 43 % jsiesta definition, 57 % SIESTA
