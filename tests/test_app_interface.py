"""User interface (``jax_siesta.app``): FDF front end, ``Calculation``/``Results``, the ``jsiesta`` command.

What is checked against something independent: band energies against a dense NumPy/SciPy solution of the Bloch
problem and against the electron count; forces from the high-level API against central finite differences of
its energy; the files against their own readers. The comparison with a SIESTA run of the same input file is
``test_same_fdf_input_in_siesta_and_jsiesta`` (needs ``siesta_bin``).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest
import scipy.linalg

import jax_siesta as js
from jax_siesta.app import build_species, parse_pao_basis_block, write_outputs
from jax_siesta.app.cli import main as jsiesta_main
from jax_siesta.core.types import ANGSTROM_TO_BOHR, HARTREE_TO_EV, KB_HARTREE_PER_KELVIN
from jax_siesta.io import parse_fdf, read_xv, tokenize_fdf
from jax_siesta.species import format_pao_basis_block

REPO_ROOT = Path(__file__).resolve().parent.parent
PSEUDOS = REPO_ROOT / "examples" / "pseudos"

_SI_FDF = """
SystemLabel si_test
%block ChemicalSpeciesLabel
  1 14 Si
%endblock ChemicalSpeciesLabel
LatticeConstant 5.47 Ang
%block LatticeVectors
  0.0 0.5 0.5
  0.5 0.0 0.5
  0.5 0.5 0.0
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
  0.00 0.00 0.00 1
  0.25 0.25 0.26 1
%endblock AtomicCoordinatesAndAtomicSpecies
PAO.BasisSize SZ
XC.Functional GGA
XC.Authors PBE
MeshCutoff 60 Ry
%block kgrid.MonkhorstPack
  2 0 0 0.0
  0 2 0 0.0
  0 0 2 0.0
%endblock kgrid.MonkhorstPack
"""


def test_fdf_front_end_keywords_units_includes_and_unused_labels(tmp_path: Path) -> None:
    """Includes, units, spin blocks, SIESTA defaults and the report of labels that are not used."""
    (tmp_path / "geom.fdf").write_text(
        "LatticeConstant 4.0 Ang\n%block LatticeParameters\n 1.0 1.0 1.5 90 90 120\n%endblock LatticeParameters\n"
        "AtomicCoordinatesFormat ScaledByLatticeVectors\n%block AtomicCoordinatesAndAtomicSpecies < coords.dat\n"
    )
    (tmp_path / "coords.dat").write_text("0.0 0.0 0.0 1\n0.5 0.5 0.5 2  # second atom\n")
    fdf = tmp_path / "in.fdf"
    fdf.write_text(
        "SystemLabel  demo ; trailing comment\n%include geom.fdf\n"
        "%block Chemical_Species_Label\n 1 26 Fe\n 2 8 O\n%endblock Chemical_Species_Label\n"
        "Spin non-collinear\n%block DM.InitSpin\n 1 2.0 90. 0.\n 2 -\n%endblock DM.InitSpin\n"
        "mesh-cutoff 0.2 mHa\nMeshCutoff 100 Ry\nElectronicTemperature 25 meV\nSCF.H.Tolerance 1 meV\n"
        "SCF.DM.Tolerance 1e-4\nMaxSCFIterations 37\nNetCharge 1.0\nPAO.EnergyShift 50 meV\nPAO.BasisSize tzp\n"
        "PAO.SplitNormH 0.5\n"
        "%block kgrid_Monkhorst_Pack\n 3 0 0 0.5\n 0 3 0 0.5\n 0 0 1 0.0\n%endblock kgrid_Monkhorst_Pack\n"
        "MD.TypeOfRun CG\nMD.Steps 20\nWriteForces T\nLDAU.Proj whatever\nSCF.MustConverge F\nJSiesta.UseSymmetry T\n"
    )
    scalars, blocks, names = tokenize_fdf(fdf)
    assert names["chemicalspecieslabel"] == "Chemical_Species_Label" and len(blocks["atomiccoordinatesandatomicspecies"]) == 2
    assert scalars["meshcutoff"] == ["0.2", "mHa"]  # first definition wins, as in SIESTA

    p = parse_fdf(fdf)
    st, cfg = p["structure"], p["config"]
    cell = np.asarray(st.cell)  # columns
    a = 4.0 * ANGSTROM_TO_BOHR
    np.testing.assert_allclose(np.linalg.norm(cell, axis=0), [a, a, 1.5 * a], rtol=1e-12)
    np.testing.assert_allclose(cell[:, 0] @ cell[:, 1], a * a * np.cos(np.radians(120.0)), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(st.positions)[1], cell @ np.array([0.5, 0.5, 0.5]), rtol=1e-12)
    assert st.species_labels == ("Fe", "O") and st.atomic_numbers == (26, 8)
    np.testing.assert_allclose(np.asarray(st.initial_moments), [[2.0, 0.0, 0.0], [0.0, 0.0, -1.0]], atol=1e-12)
    assert cfg.nspin == 4 and cfg.max_scf_iterations == 37 and cfg.net_charge == 1.0 and cfg.use_symmetry
    assert cfg.mesh_cutoff_ry == pytest.approx(0.2e-3 * 2.0)
    assert cfg.electronic_temp_ha == pytest.approx(0.025 / HARTREE_TO_EV)
    assert cfg.h_tol_ha == pytest.approx(1e-3 / HARTREE_TO_EV) and cfg.d_tol == 1e-4
    assert p["k_grid"] == (3, 3, 1) and p["k_shift"] == (0.5, 0.5, 0.0)
    assert p["basis"]["size"] == "TZP" and p["basis"]["energy_shift_ha"] == pytest.approx(0.05 / HARTREE_TO_EV)
    assert p["basis"]["split_norm"] == 0.15 and p["basis"]["split_norm_h"] == 0.5
    assert p["io"]["system_label"] == "demo" and p["io"]["must_converge"] is False
    assert p["unused"] == ["LDAU.Proj"]  # WriteForces is known and harmless; LDAU is not implemented
    assert len(p["warnings"]) == 1 and "single points only" in p["warnings"][0]

    # SIESTA's defaults where the keyword is absent: LDA (CA), 300 Ry, 300 K, DZP, 0.02 Ry, Gamma point.
    bare = tmp_path / "bare.fdf"
    bare.write_text(
        "%block ChemicalSpeciesLabel\n 1 1 H\n%endblock ChemicalSpeciesLabel\nSpinPolarized T\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n 0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    q = parse_fdf(bare)
    assert q["config"].xc_functional == "LDA_PZ" and q["config"].mesh_cutoff_ry == 300.0 and q["config"].nspin == 2
    assert q["config"].electronic_temp_ha == pytest.approx(300.0 * KB_HARTREE_PER_KELVIN)
    assert q["basis"]["size"] == "DZP" and q["basis"]["energy_shift_ha"] == pytest.approx(0.01)
    assert q["k_grid"] == (1, 1, 1) and float(np.asarray(q["structure"].initial_moments)[0, 2]) == 1.0

    for bad in ("Spin sideways\n", "OccupationFunction MP\n", "%block kgrid.MonkhorstPack\n 2 1 0\n 0 2 0\n 0 0 2\n%endblock\n"):
        wrong = tmp_path / "wrong.fdf"
        wrong.write_text(bare.read_text().replace("SpinPolarized T\n", "") + bad)
        with pytest.raises(ValueError):
            parse_fdf(wrong)


def test_pao_basis_block_reader_inverts_the_writer_and_sizes_expand() -> None:
    """``parse_pao_basis_block`` reads back what ``format_pao_basis_block`` writes; size names give the shells."""
    text = """
      Si 2 0.10
        n=3 0 2 P 1 E 40.0 4.5
          6.0 4.2
          1.0 1.0
        n=3 1 2 S 0.20
          0.0 0.0
      O 1
        n=2 0 1
          3.5
    """
    rows = [line.split() for line in text.strip().splitlines()]
    specs = parse_pao_basis_block(rows, xc="PBE")
    si = specs["Si"]
    assert si.ionic_charge == 0.10 and [(s.n, s.l, s.n_zeta, s.is_polarization) for s in si.shells] == [
        (3, 0, 2, False), (3, 1, 1, True), (3, 1, 2, False)
    ]
    assert si.shells[0].rc_list == (6.0, 4.2) and si.shells[0].v0_ha == 20.0 and si.shells[0].ri_bohr == 4.5
    assert si.shells[2].rc_list is None and si.shells[2].split_norm == 0.20
    assert specs["O"].shells[0].rc_list == (3.5,)

    explicit = {"Si": parse_pao_basis_block([r.split() for r in ("Si 2", "n=3 0 2 E 30.0 3.0", "5.5 4.0", "n=3 1 2 P 1", "6.5 4.5")])["Si"]}
    block = format_pao_basis_block(explicit)
    body = [line.split() for line in block.splitlines()[1:-1]]
    again = parse_pao_basis_block(body)["Si"]
    assert [(s.n, s.l, s.n_zeta, s.rc_list, s.v0_ha, s.ri_bohr, s.is_polarization) for s in again.shells] == [
        (s.n, s.l, s.n_zeta, s.rc_list, s.v0_ha, s.ri_bohr, s.is_polarization) for s in explicit["Si"].shells
    ]
    for bad in (["Si 1", "0 2", "5.0 4.0"], ["Si 1", "n=3 0 2 Q 3.0", "5.0 4.0"], ["Si 1", "n=3 0 2", "5.0 0.0"]):
        with pytest.raises(ValueError):
            parse_pao_basis_block([r.split() for r in bad])

    counts = {}
    for size in ("SZ", "SZP", "DZ", "DZP", "TZ2P"):
        sp, info = build_species("Si", PSEUDOS / "Si.psml", size)
        counts[size] = [(l_q, sp.pao_l.count(l_q)) for l_q in sorted(set(sp.pao_l))]
        assert info["source"] == "generated" and sp.symbol == "Si"
    assert counts == {
        "SZ": [(0, 1), (1, 1)], "SZP": [(0, 1), (1, 1), (2, 1)], "DZ": [(0, 2), (1, 2)],
        "DZP": [(0, 2), (1, 2), (2, 1)], "TZ2P": [(0, 3), (1, 3), (2, 2)],
    }
    # PAO.SplitNormH acts on hydrogen only: a larger split norm gives a more compact second zeta.
    water = js.make_structure(np.eye(3) * 8.0, ["O", "H"], [[4.0, 4.0, 4.0], [4.96, 4.0, 4.0]])
    plain = js.Calculation.from_structure(water, PSEUDOS, basis="DZ")
    split_h = js.Calculation.from_structure(water, PSEUDOS, basis="DZ", split_norm_h=0.5)
    assert split_h.species["O"].pao_rcut == plain.species["O"].pao_rcut
    assert split_h.species["H"].pao_rcut[0] == plain.species["H"].pao_rcut[0]
    assert split_h.species["H"].pao_rcut[1] < plain.species["H"].pao_rcut[1] - 0.5
    with pytest.raises(ValueError):
        build_species("Si", PSEUDOS / "Si.psml", "XXL")
    with pytest.raises(FileNotFoundError, match="PSML files only"):
        (PSEUDOS / "Legacy.psf").touch()
        try:
            js.app.find_pseudopotential("Legacy", [PSEUDOS])
        finally:
            (PSEUDOS / "Legacy.psf").unlink()


@pytest.fixture(scope="module")
def si_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[js.Calculation, js.Results, Path]:
    """One small Si calculation (SZ, 60 Ry, 2x2x2) shared by the tests of results and files."""
    work = tmp_path_factory.mktemp("si")
    (work / "si.fdf").write_text(_SI_FDF)
    calc = js.Calculation.from_fdf(work / "si.fdf", pseudo_dir=PSEUDOS)
    return calc, calc.run(), work


def test_band_energies_match_dense_solution_and_electron_count(si_run, record_residuals) -> None:
    """Eigenvalues equal an independent dense ``scipy.linalg.eigh`` of the Bloch sums; occupations sum to N."""
    _, res, _ = si_run
    assert res.converged
    pat = res.pattern
    src, dst = np.asarray(pat.src_orb), np.asarray(pat.dst_unit_orb)
    shifts = np.asarray(pat.cell_shifts)[np.asarray(pat.edge_cell_idx)]
    h = np.asarray(res.scf_state.h_sparse)[0]
    s = np.asarray(res.overlap)
    k = np.array([0.31, -0.12, 0.07])
    phase = np.exp(2j * np.pi * (shifts @ k))
    h_k = np.zeros((pat.n_orb, pat.n_orb), dtype=complex)
    s_k = np.zeros_like(h_k)
    np.add.at(h_k, (src, dst), h * phase)
    np.add.at(s_k, (src, dst), s * phase)
    assert np.abs(h_k - h_k.conj().T).max() < 1e-10
    ref = scipy.linalg.eigh(h_k, s_k, eigvals_only=True) * HARTREE_TO_EV
    got = res.eigenvalues_ev(k[None, :])[0, 0]
    err_dense = float(np.abs(got - ref).max())

    e = res.eigenvalues_ev()  # SCF k-points
    kt = res.config.electronic_temp_ha * HARTREE_TO_EV
    occ = 2.0 / (1.0 + np.exp(np.clip((e[0] - res.fermi_level_ev) / kt, -500, 500)))
    n_el = float((np.asarray(res.kmesh.weights)[:, None] * occ).sum())
    record_residuals({"eig_vs_dense_ev": err_dense, "electron_count": abs(n_el - 8.0)}, {"eig_vs_dense_ev": 1e-8, "electron_count": 1e-8})
    assert err_dense < 1e-8 and abs(n_el - 8.0) < 1e-8

    bs = res.band_structure([("L", (0.5, 0.5, 0.5)), ("\\Gamma", (0.0, 0.0, 0.0)), ("X", (0.5, 0.0, 0.5))], [4, 6])
    assert bs.energies_ev.shape == (1, 11, 8) and len(bs.ticks) == 3 and bs.n_occupied == 4
    assert np.all(np.diff(bs.distances) > 0) and np.all(np.diff(bs.energies_ev, axis=2) >= -1e-9)
    assert 0.0 < bs.band_gap_ev < 3.0


def test_results_units_and_output_files_round_trip(si_run) -> None:
    """Unit conversions of ``Results`` and the ``.FA``, ``.EIG``, ``.bands``, ``.XV``, ``.json`` files."""
    calc, res, work = si_run
    obs = res.observables
    assert res.energy_ev == pytest.approx(float(obs.total_energy_ha) * HARTREE_TO_EV)
    np.testing.assert_allclose(res.forces_ev_ang, np.asarray(obs.forces_ha_per_bohr) * HARTREE_TO_EV * ANGSTROM_TO_BOHR)
    np.testing.assert_allclose(res.stress_ev_ang3, np.asarray(obs.stress_ha_per_bohr3) * HARTREE_TO_EV * ANGSTROM_TO_BOHR**3)
    assert res.pressure_gpa == pytest.approx(-np.trace(res.stress_ev_ang3) / 3.0 * 160.21766208)
    assert len(res.scf_history) == res.n_iterations and res.forces_ev_ang.shape == (2, 3)
    terms = res.energies_ev
    assert terms["free"] == pytest.approx(terms["total"] + terms["entropy_term"], abs=1e-9)

    lines = {"rows": [r.split() for r in ("1 1.0 1.0 1.0 L", "5 0.0 0.0 0.0 \\Gamma", "5 2.0 0.0 0.0 X")],
             "scale": "pi/a", "lattice_constant_bohr": 5.47 * ANGSTROM_TO_BOHR}
    written = write_outputs(res, work / "out", band_lines=lines)
    assert set(written) == {"FA", "EIG", "XV", "DM", "bands", "json"}

    fa = (work / "out" / "si_test.FA").read_text().split()
    np.testing.assert_allclose(np.array(fa[1:], dtype=float).reshape(2, 4)[:, 1:], res.forces_ev_ang, atol=5e-7)
    eig = (work / "out" / "si_test.EIG").read_text().split()
    assert float(eig[0]) == pytest.approx(res.fermi_level_ev, abs=1e-8) and [int(x) for x in eig[1:4]] == [8, 1, 8]
    np.testing.assert_allclose(np.array(eig[5:13], dtype=float), res.eigenvalues_ev()[0, 0], atol=1e-8)
    bands = (work / "out" / "si_test.bands").read_text().splitlines()
    assert [int(x) for x in bands[3].split()] == [8, 1, 11] and bands[-3:] == [
        "    0.000000   'L'", f"{float(bands[-2].split()[0]):12.6f}   '\\Gamma'", f"{float(bands[1].split()[1]):12.6f}   'X'"
    ]
    # pi/a Cartesian points equal the fractional path L - Gamma - X of the fcc lattice
    ref = res.band_structure([("L", (0.5, 0.5, 0.5)), ("G", (0.0, 0.0, 0.0)), ("X", (0.0, 0.5, 0.5))], 5)
    np.testing.assert_allclose(float(bands[1].split()[1]), ref.distances[-1], atol=1e-10)
    np.testing.assert_allclose(np.array(bands[4].split()[1:], dtype=float), ref.energies_ev[0, 0], atol=1e-4)
    xv = read_xv(work / "out" / "si_test.XV")
    np.testing.assert_allclose(np.asarray(xv.positions), np.asarray(res.structure.positions), atol=1e-8)
    summary = json.loads((work / "out" / "si_test.json").read_text())
    assert summary["converged"] and summary["energies"]["total"] == pytest.approx(res.energy_ev)
    assert calc.with_config(mesh_cutoff_ry=80.0).config.mesh_cutoff_ry == 80.0 and calc.config.mesh_cutoff_ry == 60.0


def test_forces_of_generated_basis_run_are_derivatives_of_its_energy(record_residuals) -> None:
    """High-level API on H2 with a basis generated from the PSML file: force = -dE/dx by central differences."""
    def h2(d_ang: float) -> js.Structure:
        return js.make_structure(np.eye(3) * 5.0, ["H", "H"], [[2.0, 2.5, 2.5], [2.0 + d_ang, 2.5, 2.5]])

    calc = js.Calculation.from_structure(
        h2(0.80), PSEUDOS, basis="SZ", xc_functional="PBE", mesh_cutoff_ry=200.0, d_tol=1e-9, h_tol_ha=1e-9, e_tol_ha=1e-11
    )
    res = calc.run()
    step = 0.002
    e_p = calc.with_structure(h2(0.80 + step)).run().free_energy_ev
    e_m = calc.with_structure(h2(0.80 - step)).run().free_energy_ev
    fd = -(e_p - e_m) / (2.0 * step)
    err = abs(fd - res.forces_ev_ang[1, 0])
    record_residuals({"force_fd_ev_per_ang": err}, {"force_fd_ev_per_ang": 2e-4})
    assert res.converged and abs(res.forces_ev_ang[1, 0]) > 0.1  # away from equilibrium: a real test of the value
    # Measured 1.1e-5 eV/Ang at 200 Ry. At 80 Ry the difference stays near 1e-3 eV/Ang for any step, so the
    # mesh, not the step, sets the tolerance.
    assert err < 2e-4


def test_jsiesta_command_runs_restarts_and_reports_errors(tmp_path: Path, capsys, monkeypatch) -> None:
    """``jsiesta in.fdf``: log, files, restart from ``<label>.DM``, exit codes for input errors and non-convergence."""
    monkeypatch.chdir(tmp_path)
    fdf = tmp_path / "h2.fdf"
    base = (
        "SystemLabel h2\n%block ChemicalSpeciesLabel\n 1 1 H\n%endblock ChemicalSpeciesLabel\n"
        "LatticeConstant 5.0 Ang\n%block LatticeVectors\n 1 0 0\n 0 1 0\n 0 0 1\n%endblock LatticeVectors\n"
        "AtomicCoordinatesFormat Ang\n%block AtomicCoordinatesAndAtomicSpecies\n 2.0 2.5 2.5 1\n 2.75 2.5 2.5 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\nPAO.BasisSize SZ\nXC.Functional GGA\nXC.Authors PBE\n"
        "MeshCutoff 80 Ry\nSolutionMethod diagon\nDM.UseNewKeyword 3\n"
    )
    fdf.write_text(base)
    assert jsiesta_main([str(fdf), "--pseudo-dir", str(PSEUDOS), "--check"]) == 0
    assert "no calculation was run" in capsys.readouterr().out and not (tmp_path / "h2.XV").exists()

    assert jsiesta_main([str(fdf), "--pseudo-dir", str(PSEUDOS)]) == 0
    out = capsys.readouterr().out
    assert "input labels not used by jsiesta" in out and "DM.UseNewKeyword" in out and "SolutionMethod" not in out
    assert "SCF cycle converged" in out and "Job completed" in out
    n_first = len(re.findall(r"^   scf:", out, flags=re.M))
    e_first = float(re.search(r"Total =\s+([-0-9.]+)", out).group(1))
    for ext in ("XV", "DM", "FA", "EIG", "json"):
        assert (tmp_path / f"h2.{ext}").is_file()
    assert json.loads((tmp_path / "h2.json").read_text())["energies"]["total"] == pytest.approx(e_first, abs=1e-5)

    fdf.write_text(base + "DM.UseSaveDM T\n")
    assert jsiesta_main([str(fdf), "--pseudo-dir", str(PSEUDOS)]) == 0
    out = capsys.readouterr().out
    assert "density matrix read from h2.DM" in out
    assert len(re.findall(r"^   scf:", out, flags=re.M)) < n_first
    assert float(re.search(r"Total =\s+([-0-9.]+)", out).group(1)) == pytest.approx(e_first, abs=1e-4)

    fdf.write_text(base + "MaxSCFIterations 2\n")
    assert jsiesta_main([str(fdf), "--pseudo-dir", str(PSEUDOS)]) == 1
    assert "did NOT converge" in capsys.readouterr().out
    fdf.write_text(base + "MaxSCFIterations 2\nSCF.MustConverge F\n")
    assert jsiesta_main([str(fdf), "--pseudo-dir", str(PSEUDOS)]) == 0
    capsys.readouterr()

    assert jsiesta_main([str(fdf)]) == 2  # no H.psml next to the input
    assert "No pseudopotential H.psml" in capsys.readouterr().err
    assert jsiesta_main([str(tmp_path / "missing.fdf")]) == 2
    capsys.readouterr()


@pytest.mark.external_refs("siesta_bin")
def test_same_fdf_input_in_siesta_and_jsiesta(tmp_path: Path, record_residuals) -> None:
    """The same ``.fdf`` and PSML files in SIESTA and in ``jsiesta``: each code generates its own DZP basis.

    The two basis generators follow the same recipe but are separate implementations, so this measures how
    close a native run is to SIESTA (D-040 found 28 meV per cell for Si DZ with identical ``PAO.Basis``).
    """
    from jax_siesta.testing.siesta_adapter import run_siesta_directory

    (tmp_path / "si.fdf").write_text(
        _SI_FDF.replace("PAO.BasisSize SZ", "PAO.BasisSize DZP").replace("MeshCutoff 60 Ry", "MeshCutoff 200 Ry")
        + "PAO.EnergyShift 0.02 Ry\nElectronicTemperature 300 K\n"
    )
    (tmp_path / "Si.psml").write_bytes((PSEUDOS / "Si.psml").read_bytes())
    out = run_siesta_directory(tmp_path, "si.fdf", timeout=600).read_text()
    num = r"([-+0-9.eE]+)"
    e_siesta = float(re.findall(r"siesta:\s+Total =\s+" + num, out)[-1])
    f_siesta = np.array(
        re.search(
            r"siesta: Atomic forces \(eV/Ang\):\s*\nsiesta:\s+1\s+" + r"\s+".join([num] * 3)
            + r"\s*\nsiesta:\s+2\s+" + r"\s+".join([num] * 3), out,
        ).groups(), dtype=float,
    ).reshape(2, 3)
    res = js.Calculation.from_fdf(tmp_path / "si.fdf").run()
    resid = {"total_energy_ev": abs(res.energy_ev - e_siesta), "force_ev_per_ang": float(np.abs(res.forces_ev_ang - f_siesta).max())}
    tol = {"total_energy_ev": 0.1, "force_ev_per_ang": 0.03}
    record_residuals(resid, tol)
    assert all(resid[k] < tol[k] for k in tol), resid
