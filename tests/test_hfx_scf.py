"""Self-consistent calculations with hybrid functionals (Stages 6 to 9 of ``docs/design/HYBRID_FUNCTIONALS_PLAN.md``).

These are identity, self-consistency and finite-difference tests. The external reference for a hybrid SCF (HONPAS, same orbitals and
Gaussians) is the gated test at the end (D-059).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_siesta.app.calculation import Calculation, make_structure
from jax_siesta.core.ffi import ffi_backend_info

PSEUDOS = Path(__file__).resolve().parents[1] / "examples" / "pseudos"
pytestmark = pytest.mark.skipif(not ffi_backend_info().get("has_libxc", False), reason="hybrids need Libxc")
TIGHT = {"hfx_n_gauss": 3, "hfx_screen_tol": 1e-10, "d_tol": 1e-7, "h_tol_ha": 1e-7, "e_tol_ha": 1e-9}


def _chain(n_cells: int, k_x: int, xc: str, **extra):
    a, vacuum = 2.2, 6.0  # Angstrom: H2 molecules along x, no images along y and z
    cell = np.diag([n_cells * a, vacuum, vacuum])
    positions = [[0.2 + i * a + d, 3.0, 3.0] for i in range(n_cells) for d in (0.0, 0.74)]
    structure = make_structure(cell, ["H"] * (2 * n_cells), positions)
    return Calculation.from_structure(
        structure, pseudo_dir=PSEUDOS, basis="SZ", k_grid=(k_x, 1, 1), mesh_shape=(12 * n_cells, 30, 30), xc_functional=xc, **TIGHT, **extra
    ).run()


def test_hse06_k_points_equal_supercell() -> None:
    """HSE06 for a hydrogen chain: one cell with four k-points equals two cells with two k-points.

    Independent identity for the whole periodic path (density matrix from k-points, lattice sums of the exchange
    matrix, Bloch assembly of the result): total energy, exact-exchange energy and Fermi level per cell agree, and
    the Kohn-Sham and Harris energies agree at convergence.
    """
    unit, double = _chain(1, 4, "HSE06"), _chain(2, 2, "HSE06")
    assert unit.converged and double.converged
    e1, e2 = unit.energies_ev, double.energies_ev
    assert e1["exact_exchange"] < -1.0
    assert e2["total"] / 2 == pytest.approx(e1["total"], abs=2e-6)
    assert e2["exact_exchange"] / 2 == pytest.approx(e1["exact_exchange"], abs=2e-6)
    assert double.fermi_level_ev == pytest.approx(unit.fermi_level_ev, abs=1e-5)
    assert e1["harris"] == pytest.approx(e1["total"], abs=1e-6)
    # the hybrid is not the semilocal functional in disguise
    pbe = _chain(1, 4, "PBE")
    assert abs(e1["total"] - pbe.energies_ev["total"]) > 0.01
    assert pbe.energies_ev["exact_exchange"] == 0.0


def test_extended_pattern_for_the_exchange() -> None:
    """``HFX.PatternScale``: orbital pairs beyond overlap carry the density matrix and the exchange matrix.

    Identities. A semilocal functional must not notice the extra pairs (their overlap and Hamiltonian elements
    vanish): energy, forces and stress of PBE are unchanged, which checks that every other term of the code copes
    with pattern edges without overlap. With HSE06 on the extended pattern, one cell with four k-points equals two
    cells with two, and the Kohn-Sham and Harris energies agree. That the extension converges the truncation of
    the exchange is a measurement (D-061), not asserted here.
    """
    plain, wide = _chain(1, 4, "PBE"), _chain(1, 4, "PBE", hfx_pattern_scale=1.5)
    assert wide.pattern.n_edges > plain.pattern.n_edges
    assert wide.energies_ev["total"] == pytest.approx(plain.energies_ev["total"], abs=1e-8)
    assert np.max(np.abs(wide.forces_ev_ang - plain.forces_ev_ang)) < 1e-7
    assert np.max(np.abs(wide.stress_ev_ang3 - plain.stress_ev_ang3)) < 1e-8

    unit, double = _chain(1, 4, "HSE06", hfx_pattern_scale=1.5), _chain(2, 2, "HSE06", hfx_pattern_scale=1.5)
    assert unit.converged and double.converged
    e1, e2 = unit.energies_ev, double.energies_ev
    assert e2["total"] / 2 == pytest.approx(e1["total"], abs=2e-6)
    assert e2["exact_exchange"] / 2 == pytest.approx(e1["exact_exchange"], abs=2e-6)
    assert e1["harris"] == pytest.approx(e1["total"], abs=1e-6)
    assert np.max(np.abs(double.forces_ev_ang[:2] - unit.forces_ev_ang)) < 1e-5


@pytest.mark.parametrize("xc", ["HSE06", "PBE0"])
def test_hybrid_scf_for_a_molecule(xc: str) -> None:
    """Water with a screened and with a global hybrid: convergence, energy bookkeeping and what is not available.

    Self-consistency: the Kohn-Sham energy recomputed from the output density matrix, the in-flight SCF energy and
    the Harris energy agree, which requires ``dE_x/dP = H_x``. Forces and stress are available and finite.
    """
    structure = make_structure(np.eye(3) * 8.0, ["O", "H", "H"], [[4.0, 4.0, 4.0], [4.757, 4.586, 4.0], [3.243, 4.586, 4.0]])
    res = Calculation.from_structure(
        structure, pseudo_dir=PSEUDOS, basis="SZ", mesh_cutoff_ry=80.0, xc_functional=xc, hfx_n_gauss=3, hfx_screen_tol=1e-9
    ).run()
    assert res.converged
    e = res.energies_ev
    assert e["exact_exchange"] < -5.0
    assert e["harris"] == pytest.approx(e["total"], abs=1e-4)
    assert float(res.scf_state.last_scf_free_e_ha) * 27.211386 == pytest.approx(e["free"], abs=1e-3)
    assert res.observables.forces_available
    assert np.all(np.isfinite(res.forces_ev_ang)) and np.all(np.isfinite(res.stress_ev_ang3))
    # The net force is the egg-box error of the coarse 80 Ry mesh (measured: 0.18 eV/Ang for PBE, 0.18 and 0.21 for HSE06 and PBE0); the exchange
    # term itself has no net force (test_exchange_forces_and_strain_derivative_against_finite_differences).
    assert np.max(np.abs(res.forces_ev_ang.sum(axis=0))) < 0.5
    assert np.max(np.abs(res.forces_ev_ang)) > 1.0  # the SZ geometry is far from equilibrium


def test_hse06_forces_and_stress_against_finite_differences_of_the_total_energy() -> None:
    """End to end: the force on an atom of water and the stress of a hydrogen chain equal energy differences.

    Central differences of the free energy of complete HSE06 calculations. The tolerances are those the same
    comparison reaches with PBE at these meshes (2e-3 eV/Ang and 1.3e-5 eV/Ang**3, measured): the mesh, not the
    exchange term, limits the agreement.
    """
    tight = {"hfx_n_gauss": 3, "hfx_screen_tol": 1e-10, "d_tol": 1e-7, "h_tol_ha": 1e-7, "e_tol_ha": 1e-10}

    def water(dx: float):  # type: ignore[no-untyped-def]
        pos = np.array([[4.0, 4.0, 4.0], [4.757 + dx, 4.586, 4.0], [3.243, 4.586, 4.0]])
        structure = make_structure(np.eye(3) * 8.0, ["O", "H", "H"], pos)
        return Calculation.from_structure(
            structure, pseudo_dir=PSEUDOS, basis="SZ", mesh_cutoff_ry=150.0, xc_functional="HSE06", **tight
        ).run()

    step = 0.01
    force = water(0.0).forces_ev_ang[1, 0]
    fd = -(water(step).energies_ev["free"] - water(-step).energies_ev["free"]) / (2 * step)
    assert force == pytest.approx(fd, abs=4e-3)

    def chain(eps: float):  # type: ignore[no-untyped-def]
        positions = [[(0.2 + d) * (1 + eps), 3.0, 3.0] for d in (0.0, 0.74)]
        structure = make_structure(np.diag([2.2 * (1 + eps), 6.0, 6.0]), ["H", "H"], positions)
        return Calculation.from_structure(
            structure, pseudo_dir=PSEUDOS, basis="SZ", k_grid=(4, 1, 1), mesh_shape=(16, 36, 36), xc_functional="HSE06", **tight
        ).run()

    eps_step = 2e-3
    ref = chain(0.0)
    volume = float(np.linalg.det(ref.cell_ang))
    fd_stress = (chain(eps_step).energies_ev["free"] - chain(-eps_step).energies_ev["free"]) / (2 * eps_step) / volume
    assert ref.stress_ev_ang3[0, 0] == pytest.approx(fd_stress, abs=4e-5)


def test_full_range_hybrid_in_a_periodic_system() -> None:
    """PBE0 for a hydrogen chain: refused when the k-point mesh is too coarse, consistent with a supercell otherwise.

    The orbital pairs of the chain span four cells, so the density matrix of a mesh with fewer than five
    divisions repeats within the range of the lattice sum; the full-range sum over repeated values grows with the
    range and is refused. With six k-points the unit cell equals two cells with three k-points (identity for the
    lattice sums of the full-range kernel). The error of truncating the density matrix to the sparse pattern is
    not tested here; it is measured in D-058.
    """
    with pytest.raises(NotImplementedError, match="k-point mesh"):
        _chain(1, 4, "PBE0")
    unit, double = _chain(1, 6, "PBE0"), _chain(2, 3, "PBE0")
    assert unit.converged and double.converged
    e1, e2 = unit.energies_ev, double.energies_ev
    assert e1["exact_exchange"] < -1.0
    assert e2["total"] / 2 == pytest.approx(e1["total"], abs=2e-6)
    assert e2["exact_exchange"] / 2 == pytest.approx(e1["exact_exchange"], abs=2e-6)
    assert e1["harris"] == pytest.approx(e1["total"], abs=1e-6)


def _rotation(axis: list[float], angle: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def test_noncollinear_hybrid_is_covariant_under_a_rotation_of_the_spin_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two SCF iterations of HSE06 with ``nspin = 4`` give the same energy as ``nspin = 8`` in rotated spin axes.

    Independent identity for the spinor path of the driver. In the first frame the moments lie in the xz plane
    and every spin block is real; in the rotated frame the exchange matrix of the diagonal blocks has an
    imaginary part, which the driver has to keep in the Hamiltonian, in the energy traces and in the mixing.
    Measured: 3e-11 eV between the frames; 8e-5 eV when the imaginary part is dropped. Two iterations only,
    because the mixers are not covariant from the third iteration on, with or without a hybrid (measured with
    PBE: 2.6e-3 eV at three iterations; the converged energies agree again).
    """
    from jax_siesta.hfx import Nao2GtoExchange

    largest_imag: list[float] = []
    original = Nao2GtoExchange.matrix_and_energy

    def spy(self, dm, nspin):  # type: ignore[no-untyped-def]
        h_x, e_x = original(self, dm, nspin)
        largest_imag.append(float(np.max(np.abs(np.imag(h_x)))))
        return h_x, e_x

    monkeypatch.setattr(Nao2GtoExchange, "matrix_and_energy", spy)
    moments = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    def run(nspin: int, m: np.ndarray):  # type: ignore[no-untyped-def]
        structure = make_structure(np.eye(3) * 6.0, ["H", "H"], [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]], initial_moments=m)
        return Calculation.from_structure(
            structure, pseudo_dir=PSEUDOS, basis="SZ", mesh_shape=(30, 30, 30), xc_functional="HSE06", nspin=nspin,
            max_scf_iterations=2, **TIGHT,
        ).run()  # fmt: skip

    first = run(4, moments)
    assert max(largest_imag) < 1e-12
    rotated = run(8, moments @ _rotation([0.3, -0.5, 0.81], 0.9).T)
    assert max(largest_imag) > 1e-4  # Hartree
    assert rotated.energies_ev["total"] == pytest.approx(first.energies_ev["total"], abs=1e-8)
    assert rotated.energies_ev["exact_exchange"] == pytest.approx(first.energies_ev["exact_exchange"], abs=1e-8)
    assert first.energies_ev["exact_exchange"] < -1.0


def test_noncollinear_hybrid_stress_against_finite_differences_of_the_total_energy() -> None:
    """A frustrated H3 triangle with a non-collinear ground state: stress equals the strain derivative of the energy.

    End-to-end finite-difference test of ``nspin = 4`` with exact exchange (complex Hamiltonian, complex density
    matrix in the exchange derivative). The strain derivative has no egg-box error because the mesh scales with
    the cell. The functional is the LDA-based global hybrid LDA0 (Libxc id 177), because the non-collinear GGA
    route is not variational on its own (PBE without exact exchange deviates by 3e-4 eV/Ang**3 in this test).
    Measured: 3.8e-6 eV/Ang**3 (0.2 %); Slater exchange through the same Libxc route, without exact exchange,
    deviates by 2.5e-6, and the exchange term alone at a fixed density matrix by 1e-9.
    """
    tight = {**TIGHT, "e_tol_ha": 1e-10}
    angle = 2.0 * np.pi / 3.0
    moments = np.array([[0.0, 0.0, 1.0], [np.sin(angle), 0.0, np.cos(angle)], [-np.sin(angle), 0.0, np.cos(angle)]])
    moments = moments @ _rotation([0.3, -0.5, 0.81], 0.9).T
    pos = np.array([[3.0, 3.2, 4.0], [5.0, 3.2, 4.0], [4.0, 4.93, 4.0]])

    def triangle(eps: float):  # type: ignore[no-untyped-def]
        scale = np.array([1.0, 1.0 + eps, 1.0])
        structure = make_structure(np.diag(8.0 * scale), ["H"] * 3, pos * scale, initial_moments=moments)
        res = Calculation.from_structure(
            structure, pseudo_dir=PSEUDOS, basis="SZ", mesh_shape=(40, 40, 40), xc_functional="LIBXC-177", nspin=4, **tight
        ).run()
        assert res.converged
        return res

    ref = triangle(0.0)
    m = ref.atomic_spin_moments
    cosines = [float(m[i] @ m[j] / np.linalg.norm(m[i]) / np.linalg.norm(m[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    assert all(-0.9 < c < -0.1 for c in cosines)  # neither parallel nor antiparallel
    assert ref.energies_ev["exact_exchange"] < -1.0
    assert ref.energies_ev["harris"] == pytest.approx(ref.energies_ev["total"], abs=1e-6)
    step = 1e-3
    fd = (triangle(step).energies_ev["free"] - triangle(-step).energies_ev["free"]) / (2 * step) / 512.0
    assert abs(ref.stress_ev_ang3[1, 1]) > 1e-3
    assert ref.stress_ev_ang3[1, 1] == pytest.approx(fd, abs=1e-5)


def test_hybrid_with_spin_orbit_coupling_for_an_atom() -> None:
    """HSE06 with ``nspin = 8`` for a Co atom: the energy does not depend on the direction of the moment.

    Independent identity for exact exchange together with spin-orbit coupling (an isolated atom has no preferred
    direction; the two directions give different real and imaginary parts of every spin block). Also checked:
    convergence, the Harris free energy, and that the moment keeps its size. The electronic temperature is high
    (0.02 Ha) on purpose: at 0.005 Ha the degenerate open shell converges in energy (1e-7 Ha) but the residual of
    the Hamiltonian stays at 2e-5 Ha for 150 iterations (measured; PBE converges there in 29 iterations).
    """
    fixtures = Path(__file__).resolve().parent / "fixtures" / "psml"
    settings = {"hfx_n_gauss": 3, "hfx_screen_tol": 1e-9, "d_tol": 1e-6, "h_tol_ha": 1e-6, "e_tol_ha": 1e-8}
    results = []
    for moment in ([0.6, 0.0, 0.8], [0.0, 0.0, 1.0]):
        structure = make_structure(np.eye(3) * 7.0, ["Co"], [[3.5, 3.5, 3.5]], initial_moments=[moment])
        res = Calculation.from_structure(
            structure, pseudo_dir=fixtures, basis="SZ", mesh_cutoff_ry=80.0, xc_functional="HSE06", nspin=8,
            electronic_temp_ha=0.02, max_scf_iterations=80, **settings,
        ).run()  # fmt: skip
        assert res.converged
        assert res.energies_ev["harris"] == pytest.approx(res.energies_ev["free"], abs=1e-5)
        direction = res.spin_moment / np.linalg.norm(res.spin_moment)
        assert np.allclose(direction, moment, atol=1e-3)
        results.append(res)
    tilted, along_z = results
    assert along_z.energies_ev["exact_exchange"] < -1.0
    assert tilted.energies_ev["total"] == pytest.approx(along_z.energies_ev["total"], abs=1e-5)
    assert np.linalg.norm(tilted.spin_moment) == pytest.approx(np.linalg.norm(along_z.spin_moment), abs=1e-4)
    assert np.max(np.abs(np.imag(np.asarray(tilted.scf_state.h_sparse)[:2]))) > 1e-4  # complex diagonal blocks


_HONPAS_SI = """SystemName si
SystemLabel si
NumberOfAtoms 2
NumberOfSpecies 1
%block ChemicalSpeciesLabel
  1  14  Si
%endblock ChemicalSpeciesLabel
LatticeConstant 5.43 Ang
%block LatticeVectors
  0.00  0.50  0.50
  0.50  0.00  0.50
  0.50  0.50  0.00
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
  0.00  0.00  0.00   1
  0.27  0.25  0.25   1
%endblock AtomicCoordinatesAndAtomicSpecies
%block kgrid_Monkhorst_Pack
   3  0  0  0.0
   0  3  0  0.0
   0  0  3  0.0
%endblock kgrid_Monkhorst_Pack
XC.functional GGA
XC.authors {xc}
MeshCutoff 100.0 Ry
MaxSCFIterations 100
PAO.BasisSize SZ
DM.Tolerance 1e-6
WriteForces T
"""


def _honpas_results(workdir: Path) -> dict:
    """Total energy, exact-exchange energy, forces (eV, eV/Ang) and band extrema over the k-points of a HONPAS run."""
    import re

    text = (workdir / "siesta.out").read_text()
    total = float(re.findall(r"siesta:\s+Total =\s+(\S+)", text)[-1])
    hfx = [float(x) for x in re.findall(r"HFX energy:\s+(\S+)\s+eV", text)]
    block = text.split("siesta: Atomic forces (eV/Ang):")[-1].splitlines()[1:3]
    forces = np.array([[float(x) for x in line.split()[2:5]] for line in block])
    tokens = (workdir / "si.EIG").read_text().split()
    n_band, n_k = int(tokens[1]), int(tokens[3])
    eig = np.array(tokens[4:], dtype=float).reshape(n_k, n_band + 1)[:, 1:]
    return {"total": total, "exact_exchange": hfx[-1] if hfx else 0.0, "forces": forces, "band_min": eig.min(axis=0), "band_max": eig.max(axis=0)}


@pytest.mark.external_refs("honpas_bin")
def test_hse06_silicon_against_honpas(tmp_path: Path) -> None:
    """HSE06 for bulk Si with a displaced atom against HONPAS, with the same orbitals and the same Gaussians.

    External reference for a hybrid SCF. The two codes are made to evaluate the same functional (D-059): jsiesta
    reads the orbitals HONPAS writes (``.ion.xml``), HONPAS reads the Gaussian fit of jsiesta (``%block NAO2GTO``)
    and keeps the numerical orbitals outside the exchange (``HFX.UseFittedNAOs false``). PBE runs of both codes
    give the difference that exists without a hybrid (5.6 meV in the total energy, the neutral-atom term of
    D-025), so what is compared is the change from PBE to HSE06: total energy, forces and band energies, and the
    exact-exchange energy itself. Measured: 0.24 meV, 2e-4 eV/Ang, 4 meV, 0.28 meV. The stress is not compared:
    the one HONPAS prints for a hybrid lacks the exchange term (D-059).
    """
    import shutil

    from jax_siesta.hfx import fit_species
    from jax_siesta.species.psml import PSMLBasisLoader
    from jax_siesta.testing.siesta_adapter import run_siesta_directory

    n_gauss = 3
    runs = {}
    for name in ("honpas_pbe", "honpas_hse", "jsiesta"):
        (tmp_path / name).mkdir()
        shutil.copy(PSEUDOS / "Si.psml", tmp_path / name / "Si.psml")
    (tmp_path / "honpas_pbe" / "in.fdf").write_text(_HONPAS_SI.format(xc="PBE"))
    run_siesta_directory(tmp_path / "honpas_pbe", "in.fdf", executable_key="honpas_bin", timeout=900)
    runs["honpas_pbe"] = _honpas_results(tmp_path / "honpas_pbe")
    shutil.copy(tmp_path / "honpas_pbe" / "Si.ion.xml", tmp_path / "jsiesta" / "Si.ion.xml")

    species = PSMLBasisLoader().load_species("Si", str(tmp_path / "jsiesta" / "Si.psml"), ion_path=str(tmp_path / "jsiesta" / "Si.ion.xml"))
    lines, zeta = ["%block NAO2GTO", f"Si {len(species.pao_l)}"], {}
    for shell, fit in enumerate(fit_species(species, n_gauss)):
        key = (int(species.pao_n[shell]), fit.l)
        zeta[key] = zeta.get(key, 0) + 1
        lines.append(f"{key[0]} {fit.l} {zeta[key]} {len(fit.exponents)}")
        lines += [f"{a:.16e} {c:.16e}" for a, c in zip(fit.exponents, fit.coefficients)]
    lines += ["%endblock NAO2GTO", "HFX.UseFittedNAOs false", "HFX.FarField false"]
    lines += [f"HFX.{key}Tolerance 1e-8" for key in ("Schwarz", "PairList", "StoreERIs", "FarField")]
    (tmp_path / "honpas_hse" / "in.fdf").write_text(_HONPAS_SI.format(xc="HSE06") + "\n".join(lines) + "\n")
    run_siesta_directory(tmp_path / "honpas_hse", "in.fdf", executable_key="honpas_bin", timeout=3000)
    runs["honpas_hse"] = _honpas_results(tmp_path / "honpas_hse")

    for xc in ("PBE", "HSE06"):
        fdf = tmp_path / "jsiesta" / f"{xc}.fdf"
        fdf.write_text(_HONPAS_SI.format(xc=xc) + f"User.Basis T\nHFX.NumberOfGaussians {n_gauss}\nHFX.ScreeningTolerance 1e-8\n")
        res = Calculation.from_fdf(fdf).run()
        assert res.converged
        eig = np.asarray(res.eigenvalues_ev()).reshape(-1, res.basis.n_orb)
        runs[f"jsiesta_{xc}"] = {
            "total": res.energies_ev["total"], "exact_exchange": res.energies_ev["exact_exchange"],
            "forces": res.forces_ev_ang, "band_min": eig.min(axis=0), "band_max": eig.max(axis=0),
        }  # fmt: skip
    j_pbe, j_hse, h_pbe, h_hse = runs["jsiesta_PBE"], runs["jsiesta_HSE06"], runs["honpas_pbe"], runs["honpas_hse"]

    assert h_hse["exact_exchange"] < -5.0
    assert j_hse["exact_exchange"] == pytest.approx(h_hse["exact_exchange"], abs=1e-3)
    assert j_hse["total"] - j_pbe["total"] == pytest.approx(h_hse["total"] - h_pbe["total"], abs=1e-3)
    assert abs(h_hse["total"] - h_pbe["total"]) > 1.0
    d_force = (j_hse["forces"] - j_pbe["forces"]) - (h_hse["forces"] - h_pbe["forces"])
    assert np.max(np.abs(d_force)) < 1e-3
    assert np.max(np.abs(h_hse["forces"] - h_pbe["forces"])) > 0.02
    for key in ("band_min", "band_max"):
        assert np.max(np.abs(j_pbe[key] - h_pbe[key])) < 1e-3
        assert np.max(np.abs(j_hse[key] - h_hse[key])) < 1e-2
    gap = {name: run["band_min"][4] - run["band_max"][3] for name, run in runs.items()}
    assert gap["jsiesta_HSE06"] == pytest.approx(gap["honpas_hse"], abs=1e-2)
    assert gap["honpas_hse"] - gap["honpas_pbe"] > 0.3  # the hybrid opens the gap
