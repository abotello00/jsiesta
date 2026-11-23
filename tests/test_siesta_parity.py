r"""Cross-code parity tests against real reference SIESTA and ``siesta_spglib_full_impl`` binaries."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import numpy as np
import pytest

from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.testing.siesta_adapter import (
    observables_to_component_validation_dict,
    run_jsiesta_matched_case,
    run_siesta_directory,
    run_siesta_reference,
)
import validation

from jax_siesta.testing.external_refs import ref_path

_SI_PSML = (ref_path("siesta_pseudos_dir") or Path("/nonexistent")) / "Si.tm2.nrl.psml"
_CO_DIR = ref_path("co_pilot_dir") or Path("/nonexistent")


@pytest.mark.external_refs("siesta_bin_or_cache", "siesta_sym_bin_or_cache", "siesta_pseudos_dir")
def test_siesta_bulk_and_symmetry_parity(tmp_path: Path, record_residuals) -> None:
    """Verify diamond Si bulk parity across standard SIESTA, ``siesta_spglib_full_impl``, and ``jsiesta``."""
    work = tmp_path / "si_bulk_parity"
    work.mkdir(parents=True)
    shutil.copy2(_SI_PSML, work / "Si.psml")

    cell_cols = 5.13150248 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    pos_bohr = (cell_cols @ np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]).T).T
    case_path = work / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "name": "si_bulk_real_psml",
                "cell_columns_bohr": cell_cols.tolist(),
                "positions_bohr": pos_bohr.tolist(),
                "species": ["Si", "Si"],
                "k_grid": [4, 4, 4],
                "mesh_cutoff_ry": 150.0,
                "xc": "PBE",
                "nspin": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ref_std = run_siesta_reference(case_path, work / "ref_std", mode="siesta", psml_dir=work)
    ref_sym = run_siesta_reference(case_path, work / "ref_sym", mode="siesta-sym", psml_dir=work)
    obs_j, meta_j = run_jsiesta_matched_case(
        case_path,
        work,
        ion_dir=work / "ref_std",
        use_symmetry=False,
        reference_mesh_dims=ref_std["_meta"]["mesh_dims"],
    )
    obs_sym, meta_sym = run_jsiesta_matched_case(
        case_path,
        work,
        ion_dir=work / "ref_sym",
        use_symmetry=True,
        reference_mesh_dims=ref_sym["_meta"]["mesh_dims"],
    )

    assert ref_sym["_meta"]["n_kpoints"] == 8
    assert meta_sym["n_kpoints"] == 8
    assert meta_sym["active_symmetry_ops"] == 48

    d_j = observables_to_component_validation_dict(obs_j, 1)
    d_sym = observables_to_component_validation_dict(obs_sym, 1)
    resid: dict[str, float] = {}
    tols: dict[str, float] = {}
    for key in ("e_kinetic_ev", "e_kb_ev", "e_hartree_ev", "e_xc_ev", "fermi_level_ev"):
        resid[f"std.{key}"] = abs(d_j[key] - ref_std[key])
        resid[f"sym.{key}"] = abs(d_sym[key] - ref_sym[key])
        tols[f"std.{key}"] = tols[f"sym.{key}"] = 2.0e-3
    resid["std.forces_max_ev_ang"] = float(np.max(np.abs(np.asarray(d_j["forces"]) - np.asarray(ref_std["forces"]))))
    resid["std.stress_max_ev_ang3"] = float(np.max(np.abs(np.asarray(d_j["stress"]) - np.asarray(ref_std["stress"]))))
    resid["sym.stress_max_ev_ang3"] = float(
        np.max(np.abs(np.asarray(d_sym["stress"]) - np.asarray(ref_sym["stress"])))
    )
    tols.update({"std.forces_max_ev_ang": 2.0e-4, "std.stress_max_ev_ang3": 2.5e-4, "sym.stress_max_ev_ang3": 2.5e-4})
    record_residuals(resid, tols)
    for key, val in resid.items():
        assert val < tols[key], (key, val, tols[key])


@pytest.mark.external_refs("siesta_bin_or_cache", "siesta_sym_bin_or_cache", "siesta_pseudos_dir")
def test_siesta_strained_displaced_via_validation_harness(tmp_path: Path) -> None:
    """Verify non-zero forces and anisotropic stress parity through ``validation.py``."""
    case_dir = tmp_path / "inputs" / "si_disp"
    case_dir.mkdir(parents=True)
    shutil.copy2(_SI_PSML, case_dir / "Si.psml")

    cell_st = np.array(
        [
            [0.0, 5.13150248, 5.28544755],
            [5.13150248, 0.0, 5.28544755],
            [5.13150248, 5.13150248, 0.0],
        ]
    )
    pos_st = (cell_st @ np.array([[0.0, 0.0, 0.0], [0.258, 0.252, 0.247]]).T).T
    case_path = case_dir / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "name": "si_strained_displaced",
                "cell_columns_bohr": cell_st.tolist(),
                "positions_bohr": pos_st.tolist(),
                "species": ["Si", "Si"],
                "k_grid": [3, 3, 3],
                "mesh_cutoff_ry": 150.0,
                "xc": "PBE",
                "nspin": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Pre-generate Si.ion.xml from SIESTA so both baseline and candidate share exact PAO/KB basis tables
    ion_dir = tmp_path / "ion_stage"
    run_siesta_reference(case_path, ion_dir, mode="siesta-sym", psml_dir=case_dir)
    shutil.copy2(ion_dir / "Si.ion.xml", case_dir / "Si.ion.xml")

    repo_root = Path(__file__).resolve().parent.parent
    adapter_script = str(repo_root / "jax_siesta" / "testing" / "siesta_adapter.py")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "timeout_seconds": 300,
        "baseline": {
            "command": [
                "{python}",
                adapter_script,
                "{workdir}/case.json",
                "--mode",
                "siesta-sym",
                "--output",
                "{workdir}/observables.json",
            ],
            "identity": {"build": "siesta_spglib_full_impl", "revision": "5.4.2-32-g673822fa1"},
        },
        "candidate": {
            "command": [
                "{python}",
                adapter_script,
                "{workdir}/case.json",
                "--mode",
                "jsiesta-sym",
                "--output",
                "{workdir}/observables.json",
            ],
            "identity": {"build": "jax_siesta", "revision": "phase3"},
        },
        "cases": [
            {
                "name": "si_strained_displaced_sym",
                "input_dir": str(case_dir),
                "tolerances": {
                    "e_kinetic_ev": {"atol": 5.0e-3, "rtol": 1.0e-4, "unit": "eV"},
                    "e_kb_ev": {"atol": 5.0e-3, "rtol": 1.0e-4, "unit": "eV"},
                    "e_hartree_ev": {"atol": 2.0e-3, "rtol": 5.0e-3, "unit": "eV"},
                    "e_xc_ev": {"atol": 5.0e-3, "rtol": 1.0e-4, "unit": "eV"},
                    "fermi_level_ev": {"atol": 5.0e-3, "rtol": 1.0e-3, "unit": "eV"},
                    "e_na_ev": {"atol": 1.0e-2, "rtol": 0.0, "unit": "eV"},
                    "e_total_ev": {"atol": 1.0e-2, "rtol": 0.0, "unit": "eV"},
                    "final_post_scf_free_e_ev": {"atol": 1.0e-2, "rtol": 0.0, "unit": "eV"},
                    **{
                        f"forces[{ia}][{ax}]": {"atol": 2.0e-2, "rtol": 1.0e-2, "unit": "eV/Ang"}
                        for ia in range(2)
                        for ax in range(3)
                    },
                    **{
                        f"stress[{a}][{b}]": {"atol": 5.0e-4, "rtol": 5.0e-2, "unit": "eV/Ang^3"}
                        for a in range(3)
                        for b in range(3)
                    },
                },
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _ = repo_root
    report_path, passed = validation.run_manifest(manifest_path, tmp_path / "val_out")
    assert passed, report_path.read_text(encoding="utf-8")


@pytest.mark.external_refs("siesta_bin_or_cache", "siesta_sym_bin_or_cache", "co_pilot_dir")
def test_siesta_co_hcp_collinear_symmetry_parity(tmp_path: Path, record_residuals) -> None:
    """Verify spin-polarized NLCC HCP Co parity against ``siesta`` and ``siesta_spglib_full_impl``."""
    work = tmp_path / "co_hcp"
    work.mkdir(parents=True)
    shutil.copy2(_CO_DIR / "Co.psml", work / "Co.psml")

    ang_to_bohr = 1.0 / 0.529177210903
    a_co = 2.507100
    cell_cols = (
        np.array(
            [
                [a_co, -0.5 * a_co, 0.000000],
                [0.000000, a_co * np.sqrt(3.0) / 2.0, 0.000000],
                [0.000000, 0.000000, 4.069500],
            ],
            dtype=np.float64,
        )
        * ang_to_bohr
    )
    frac = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]], dtype=np.float64)
    pos_bohr = (cell_cols @ frac.T).T

    case_path = work / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "name": "co_hcp_collinear",
                "cell_columns_bohr": cell_cols.tolist(),
                "positions_bohr": pos_bohr.tolist(),
                "species": ["Co", "Co"],
                "k_grid": [4, 4, 2],
                "mesh_cutoff_ry": 120.0,
                "xc": "PBE",
                "nspin": 2,
                "initial_moments": [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Matched meshes (review M5) and tolerances from measured residuals (review H2, D-017). Before the
    # radial-table fix the kinetic energy was off by 0.44-0.85 eV and the test allowed 1.2 eV.
    tol = {
        "e_kinetic_ev": 2.0e-2,
        "e_kb_ev": 5.0e-3,
        "e_hartree_ev": 2.0e-3,
        "e_xc_ev": 5.0e-3,
        "e_na_ev": 3.0e-2,
        "e_total_ev": 2.5e-2,
        "final_post_scf_free_e_ev": 2.5e-2,
        "fermi_level_ev": 2.0e-3,
        "total_spin_moment": 1.0e-3,
    }
    for mode, use_sym in (("siesta", False), ("siesta-sym", True)):
        ref = run_siesta_reference(case_path, work / f"ref_{mode}", mode=mode, psml_dir=work, timeout=1200)
        obs, meta = run_jsiesta_matched_case(
            case_path,
            work,
            ion_dir=work / f"ref_{mode}",
            use_symmetry=use_sym,
            reference_mesh_dims=ref["_meta"]["mesh_dims"],
        )
        d = observables_to_component_validation_dict(obs, 2)
        if use_sym:
            assert meta["n_kpoints"] == ref["_meta"]["n_kpoints"]
        record_residuals(
            {f"{mode}.{key}": abs(d[key] - ref[key]) for key in tol},
            {f"{mode}.{key}": t for key, t in tol.items()},
        )
        for key, t in tol.items():
            assert abs(d[key] - ref[key]) < t, (mode, key, d[key], ref[key])
        if use_sym:
            # Perfect hcp: forces vanish by symmetry. jsiesta projects them exactly; siesta_spglib_full_impl
            # leaves ~1e-3 eV/Ang of unsymmetrised mesh noise, so it is only a loose reference here.
            assert np.max(np.abs(np.asarray(d["forces"]))) < 1.0e-8
            assert np.max(np.abs(np.asarray(ref["forces"]))) < 3.0e-3
        else:
            assert np.max(np.abs(np.asarray(d["forces"]) - np.asarray(ref["forces"]))) < 1.0e-4
        # Stress: 3e-3 eV/Ang^3 against measured 2.3e-3 / 2.5e-3. The 2.8 % sigma_zz difference is a real difference
        # in the strain dependence of the energy (neutral-atom pair term, kinetic), not an error of the analytic
        # derivative of either code: logs/DECISIONS.md D-021. In symmetric mode jsiesta projects the stress onto
        # the hexagonal group, while siesta_spglib_full_impl does not (the shifted 4x4x2 grid gives it
        # sigma_xy = 0.0095), so the reference is symmetrised the same way: <xx, yy> in plane, xy -> 0.
        s_ref = np.asarray(ref["stress"], dtype=float)
        if use_sym:
            inplane = 0.5 * (s_ref[0, 0] + s_ref[1, 1])
            s_ref = np.diag([inplane, inplane, s_ref[2, 2]])
        stress_resid = float(np.max(np.abs(np.asarray(d["stress"]) - s_ref)))
        szz_rel = float((np.asarray(d["stress"])[2, 2] - s_ref[2, 2]) / s_ref[2, 2])
        record_residuals(
            {f"{mode}.stress_max_ev_ang3": stress_resid, f"{mode}.stress_zz_relative": abs(szz_rel)},
            {f"{mode}.stress_max_ev_ang3": 3.0e-3},
        )
        assert stress_resid < 3.0e-3, (mode, d["stress"], s_ref.tolist())


@pytest.mark.external_refs("siesta_bin_or_cache", "siesta_pseudos_dir")
def test_siesta_total_and_free_energy_parity(tmp_path: Path, record_residuals) -> None:
    """Absolute and relative total/free energies agree with SIESTA (review finding H1).

    Three Si geometries on SIESTA's own mesh: equilibrium, an atom displacement and an anisotropic strain.
    The absolute energies include the one-centre neutral-atom constants (``neutral_atom_self_energy``);
    the remaining few-meV/atom offset is SIESTA's reciprocal-space evaluation of its one-centre ``Ena``
    (isolated Si atom: SIESTA 112.6987 eV vs exact radial 112.6969 eV) and must be geometry independent.
    """
    base = 5.13150248 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    geoms = {
        "eq": (base, [[0, 0, 0], [0.25, 0.25, 0.25]]),
        "disp": (base, [[0, 0, 0], [0.26, 0.245, 0.25]]),
        "strain": (base @ np.diag([1.02, 1.0, 0.99]), [[0, 0, 0], [0.25, 0.25, 0.25]]),
    }
    diffs: dict[str, dict[str, float]] = {}
    for name, (cell, frac) in geoms.items():
        work = tmp_path / name
        work.mkdir()
        shutil.copy2(_SI_PSML, work / "Si.psml")
        pos = (cell @ np.asarray(frac, dtype=float).T).T
        case_path = work / "case.json"
        case_path.write_text(
            json.dumps(
                {
                    "name": name,
                    "cell_columns_bohr": cell.tolist(),
                    "positions_bohr": pos.tolist(),
                    "species": ["Si", "Si"],
                    "k_grid": [3, 3, 3],
                    "mesh_cutoff_ry": 100.0,
                    "xc": "PBE",
                    "nspin": 1,
                }
            ),
            encoding="utf-8",
        )
        ref = run_siesta_reference(case_path, work / "ref", mode="siesta", psml_dir=work)
        obs, _ = run_jsiesta_matched_case(
            case_path, work, ion_dir=work / "ref", reference_mesh_dims=ref["_meta"]["mesh_dims"]
        )
        d = observables_to_component_validation_dict(obs, 1)
        diffs[name] = {k: d[k] - ref[k] for k in ("e_total_ev", "final_post_scf_free_e_ev", "e_na_ev")}

    record_residuals(
        {f"{name}.{key}": abs(val) for name, dd in diffs.items() for key, val in dd.items()},
        {f"{name}.{key}": 1.0e-2 for name, dd in diffs.items() for key in dd},
    )
    for name, dd in diffs.items():
        # Absolute: within 5 meV/atom of SIESTA (observed ~3.2 meV/atom, see docstring).
        for key, val in dd.items():
            assert abs(val) < 1.0e-2, (name, key, val)
        # Relative to equilibrium: the offset must be geometry independent.
        for key in ("e_total_ev", "final_post_scf_free_e_ev"):
            assert abs(dd[key] - diffs["eq"][key]) < 3.0e-4, (name, key, dd[key], diffs["eq"][key])


def _siesta_init_mesh(work: Path, cell_cols: np.ndarray, cutoff_ry: float, nsm: int) -> tuple[int, int, int]:
    """Run SIESTA's setup (one SCF step, one Si atom) and return its ``InitMesh`` dimensions."""
    work.mkdir(parents=True)
    shutil.copy2(_SI_PSML, work / "Si.psml")
    rows = "\n".join(" ".join(f"{x:.10f}" for x in cell_cols[:, i]) for i in range(3))
    (work / "input.fdf").write_text(
        "SystemLabel mesh\nNumberOfAtoms 1\nNumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n 1 14 Si\n%endblock ChemicalSpeciesLabel\n"
        "PAO.BasisSize SZ\nLatticeConstant 1.0 Bohr\n"
        f"%block LatticeVectors\n{rows}\n%endblock LatticeVectors\n"
        "AtomicCoordinatesFormat Bohr\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n 0.0 0.0 0.0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n"
        f"MeshCutoff {cutoff_ry} Ry\nMeshSubDivisions {nsm}\nMaxSCFIterations 1\nSCF.MustConverge false\n",
        encoding="utf-8",
    )
    out = run_siesta_directory(work, "input.fdf", stdout_name="out.txt", timeout=300)
    m = re.search(r"InitMesh:\s+MESH\s*=\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", out.read_text())
    assert m is not None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


@pytest.mark.external_refs("siesta_bin_or_cache", "siesta_pseudos_dir")
def test_siesta_mesh_selection_matches_siesta(tmp_path: Path, record_residuals) -> None:
    """``choose_mesh_dimensions(criterion="siesta")`` picks the same mesh as SIESTA (open issue O-6).

    Covers every cell of the Phase 0 matrix, the Si and Co parity cells and three ``MeshSubDivisions`` values on a
    skewed cell, so interplanar spacing, ``nsm`` divisibility and the ``chkgmx`` loop are all exercised.
    """
    manifests = Path(__file__).resolve().parent / "manifests" / "cases"
    cells: dict[str, tuple[np.ndarray, float, int]] = {}
    for case_dir in sorted(manifests.iterdir()):
        data = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        cells[case_dir.name] = (np.asarray(data["cell_columns_bohr"], dtype=float), float(data["mesh_cutoff_ry"]), 2)
    ang = 1.0 / 0.529177210903
    a, c = 2.5071 * ang, 4.0695 * ang
    cells["co_parity"] = (
        np.array([[a, -0.5 * a, 0.0], [0.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c]]), 120.0, 2,
    )
    si = 5.13150248 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    cells["si_150"], cells["si_100"] = (si, 150.0, 2), (si, 100.0, 2)
    skew = np.array([[5.2, 0.35, 0.15], [0.2, 5.05, 0.4], [0.1, 0.25, 5.3]]).T
    for nsm in (1, 2, 4):
        cells[f"triclinic_nsm{nsm}"] = (skew, 100.0, nsm)
    cells["skewed_hex_slab"] = (np.array([[9.3, 0.0, 0.0], [-7.0, 8.0, 0.0], [1.5, 2.0, 30.0]]).T, 90.0, 2)

    mismatches = {}
    for name, (cell, cutoff, nsm) in cells.items():
        ref = _siesta_init_mesh(tmp_path / name, cell, cutoff, nsm)
        mine = choose_mesh_dimensions(cell, cutoff, criterion="siesta", mesh_subdivisions=nsm)
        if mine != ref:
            mismatches[name] = (mine, ref)
    record_residuals({"n_cells": len(cells), "n_mismatches": len(mismatches)}, {"n_mismatches": 0})
    assert not mismatches, mismatches


_MATRIX_CASES = ("h2o_gamma", "graphene_slab", "si_triclinic", "c_bilayer", "co_afm")
# Tolerances: components (eV), forces (eV/Ang), stress (eV/Ang^3), spin moment (mu_B), set per case from the measured
# residuals (about 1.3x; D-024). Kinetic, KB, Hartree and XC agree to <= 4 meV in every case. What dominates is the
# neutral-atom pair term: SIESTA's printed ``Ena`` deviates smoothly from the exact value of its own formula
# (fixed-resolution tables, D-025), by 2 % of its displacement derivative for H2O and 0.3 % for triclinic Si (whose pair
# forces are large), while jsiesta equals the exact value to 0.07 %. This is why forces/stress of H2O and the triclinic Si
# cell are only reproduced to ~0.06 / ~0.1 eV/Ang: the difference is SIESTA's.
_MATRIX_TOL: dict[str, dict[str, float]] = {
    "h2o_gamma": {"component": 5.0e-3, "forces": 8.0e-2, "stress": 2.0e-4},
    "graphene_slab": {"component": 5.0e-3, "forces": 2.0e-4, "stress": 2.0e-4},
    "si_triclinic": {"component": 7.0e-2, "forces": 1.2e-1, "stress": 7.0e-3},
    "c_bilayer": {"component": 8.0e-3, "forces": 5.0e-4, "stress": 2.0e-4},
    "co_afm": {"component": 6.0e-3, "forces": 1.0e-6, "stress": 1.0e-3},
}
_MATRIX_DEFAULT_TOL = {"component": 2.0e-2, "forces": 2.0e-3, "stress": 5.0e-4, "moment": 1.0e-3}
# The Fermi level is only defined for (semi-)metals; in a gap it is an arbitrary value between the band edges.
# (Graphene on its 4x4 Gamma-centred grid misses the K point, so it has a gap there as well.)
_MATRIX_FERMI_CASES = ("c_bilayer", "co_afm")


def _run_matrix_case(case_name: str, pseudo_key: str, tmp_path: Path, record_residuals) -> None:
    """Phase 0 matrix cases (molecule, slab, triclinic, bilayer, AFM) against SIESTA with real PseudoDojo pseudos.

    The cases of ``tests/manifests/cases`` run with the PseudoDojo nc-sr-05 PBE scalar-relativistic PSML set (the
    manifest's own PSML files are synthetic fixtures). The bilayer runs without DFT-D3: the adapter refuses matched
    dispersion runs, and D3 is validated in ``tests/test_dispersion.py``. SIESTA's mesh is reproduced by the
    ``"siesta"`` mesh criterion (asserted).
    """
    src = json.loads((Path(__file__).resolve().parent / "manifests" / "cases" / case_name / "case.json").read_text())
    case = {k: v for k, v in src.items() if k != "dispersion"}
    case.update({"d_tol": 1.0e-6, "h_tol_ha": 1.0e-6, "e_tol_ha": 1.0e-8, "max_scf_iterations": 300, "xc": "PBE"})
    work = tmp_path / case_name
    work.mkdir()
    for label in set(case["species"]):
        shutil.copy2(ref_path(pseudo_key) / f"{label}.psml", work / f"{label}.psml")
    case_path = work / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    nspin = int(case.get("nspin", 1))

    ref = run_siesta_reference(case_path, work / "ref", mode="siesta", psml_dir=work, timeout=3600)
    obs, meta = run_jsiesta_matched_case(
        case_path, work, ion_dir=work / "ref", use_symmetry=False, reference_mesh_dims=ref["_meta"]["mesh_dims"]
    )
    d = observables_to_component_validation_dict(obs, nspin)
    tol = {**_MATRIX_DEFAULT_TOL, **_MATRIX_TOL.get(case_name, {})}
    resid = {
        f"{k}": abs(d[k] - ref[k])
        for k in ("e_kinetic_ev", "e_kb_ev", "e_hartree_ev", "e_xc_ev", "e_na_ev", "e_total_ev", "final_post_scf_free_e_ev")
    }
    resid["fermi_level_ev"] = abs(d["fermi_level_ev"] - ref["fermi_level_ev"])
    resid["forces_max"] = float(np.max(np.abs(np.asarray(d["forces"]) - np.asarray(ref["forces"]))))
    resid["stress_max"] = float(np.max(np.abs(np.asarray(d["stress"]) - np.asarray(ref["stress"]))))
    tols = {k: tol["component"] for k in resid}
    tols.update({"forces_max": tol["forces"], "stress_max": tol["stress"], "fermi_level_ev": 5.0e-3})
    if case_name not in _MATRIX_FERMI_CASES:
        tols["fermi_level_ev"] = float("inf")  # recorded, not asserted
    if nspin >= 2:
        resid["total_spin_moment"] = abs(d["total_spin_moment"] - ref["total_spin_moment"])
        tols["total_spin_moment"] = tol["moment"]
    record_residuals(resid, tols)
    print(f"RESIDUALS {case_name}: " + json.dumps({k: float(f"{v:.3e}") for k, v in resid.items()}))
    bad = {k: (v, tols[k]) for k, v in resid.items() if v > tols[k]}
    assert not bad, bad



@pytest.mark.slow
@pytest.mark.external_refs("siesta_bin_or_cache", "pseudodojo_sr_pbe_dir")
@pytest.mark.parametrize("case_name", _MATRIX_CASES)
def test_phase0_matrix_parity_against_siesta(case_name: str, tmp_path: Path, record_residuals) -> None:
    """Phase 0 matrix cases against SIESTA with the PseudoDojo scalar-relativistic PSML set (see ``_run_matrix_case``)."""
    _run_matrix_case(case_name, "pseudodojo_sr_pbe_dir", tmp_path, record_residuals)


@pytest.mark.slow
@pytest.mark.external_refs("siesta_bin_or_cache", "pseudodojo_fr_pbe_dir")
@pytest.mark.parametrize("case_name", ["h2o_gamma", "si_triclinic"])
def test_fully_relativistic_psml_below_nspin8_matches_siesta(case_name: str, tmp_path: Path, record_residuals) -> None:
    """A fully relativistic PSML (``relativity="dirac"``) used without spin-orbit (``nspin < 8``) reads the same projectors.

    The PseudoDojo nc-fr PSML carries ``scalar_relativistic``, ``spin_orbit`` and ``lj`` projector sets. Below nspin 8
    jsiesta selects the ``scalar_relativistic`` set; this checks that SIESTA does the same by comparing every energy
    component, forces and stress on the same cases as the scalar-relativistic matrix (D-028).
    """
    _run_matrix_case(case_name, "pseudodojo_fr_pbe_dir", tmp_path, record_residuals)
