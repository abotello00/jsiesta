r"""DFT-D3 validation: reference library, finite differences, and cross-code parity with SIESTA."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core.types import Structure
from jax_siesta.testing.external_refs import ref_path
from jax_siesta.testing.siesta_adapter import run_siesta_directory
from jax_siesta.xc.dispersion import DFTD3Dispersion, d3_method_for_functional

dftd3 = pytest.importorskip("dftd3.interface")

HA_EV = 27.211386245988
BOHR_ANG = 0.529177210903
ANG_BOHR = 1.0 / BOHR_ANG
# PBE-D3(BJ): Grimme, Ehrlich, Goerigk, JCC 32, 1456 (2011)
PBE_D3BJ = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}


def _si_displaced(a_ang: float = 5.43, frac2: tuple[float, float, float] = (0.25, 0.25, 0.27)) -> Structure:
    a = a_ang * ANG_BOHR
    cell = a * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]).T  # columns = lattice vectors
    frac = np.array([[0.0, 0.0, 0.0], list(frac2)])
    return Structure(
        cell=jnp.asarray(cell),
        positions=jnp.asarray(frac @ cell.T),
        species_indices=jnp.array([0, 0]),
        atomic_numbers=(14, 14),
        species_labels=("Si",),
        initial_moments=jnp.zeros((2, 3)),
    )


def _with(structure: Structure, cell: np.ndarray, pos: np.ndarray) -> Structure:
    return Structure(
        cell=jnp.asarray(cell),
        positions=jnp.asarray(pos),
        species_indices=structure.species_indices,
        atomic_numbers=structure.atomic_numbers,
        species_labels=structure.species_labels,
        initial_moments=structure.initial_moments,
        pbc=structure.pbc,
    )


def test_d3_method_mapping() -> None:
    assert d3_method_for_functional("PBE") == "pbe"
    assert d3_method_for_functional("r2SCAN01") == "r2scan"
    with pytest.raises(ValueError):
        d3_method_for_functional("LDA")


def test_d3_equals_reference_library() -> None:
    """The wrapper must return exactly what simple-dftd3 returns (no re-implementation)."""
    s = _si_displaced()
    e, f, sig = DFTD3Dispersion("PBE", three_body=False).evaluate(s)
    model = dftd3.DispersionModel(
        np.array([14, 14]), np.asarray(s.positions), np.asarray(s.cell).T, periodic=np.array([True] * 3)
    )
    ref = model.get_dispersion(dftd3.RationalDampingParam(method="pbe"), grad=True)
    assert abs(float(e) - ref["energy"]) < 1e-14
    np.testing.assert_allclose(np.asarray(f), -ref["gradient"], atol=1e-14)
    # Explicit parameters reproduce the published set.
    e2, _, _ = DFTD3Dispersion("PBE", three_body=False, params=PBE_D3BJ).evaluate(s)
    assert abs(float(e2) - float(e)) < 1e-12
    # Published Si value sanity: about -0.63 eV for this cell (two-body only).
    assert -0.70 < float(e) * HA_EV < -0.55


def test_d3_forces_and_stress_match_finite_differences() -> None:
    """Checks jsiesta's sign/unit conventions: F = -dE/dR, sigma = (1/Omega) dE/deps."""
    s = _si_displaced()
    d3 = DFTD3Dispersion("PBE")
    _, f, sig = d3.evaluate(s)
    assert float(jnp.max(jnp.abs(jnp.sum(f, axis=0)))) < 1e-12
    np.testing.assert_allclose(np.asarray(sig), np.asarray(sig).T, atol=1e-14)

    cell = np.asarray(s.cell)
    pos = np.asarray(s.positions)
    h = 1e-5
    for atom, ax in [(1, 2), (0, 0)]:
        pp, pm = pos.copy(), pos.copy()
        pp[atom, ax] += h
        pm[atom, ax] -= h
        fd = -(float(d3.energy(_with(s, cell, pp))) - float(d3.energy(_with(s, cell, pm)))) / (2 * h)
        assert abs(fd - float(f[atom, ax])) < 1e-9

    vol = abs(np.linalg.det(cell))
    for a, b in [(0, 0), (0, 1), (2, 2)]:
        eps = np.zeros((3, 3))
        eps[a, b] = h
        dp, dm = np.eye(3) + eps, np.eye(3) - eps
        ep = float(d3.energy(_with(s, dp @ cell, pos @ dp.T)))
        em = float(d3.energy(_with(s, dm @ cell, pos @ dm.T)))
        assert abs((ep - em) / (2 * h * vol) - float(sig[a, b])) < 1e-10


_SI_FDF = """SystemName si_d3
SystemLabel si_d3
NumberOfSpecies 1
NumberOfAtoms 2
%block ChemicalSpeciesLabel
 1 14 Si
%endblock ChemicalSpeciesLabel
PAO.BasisSize SZ
LatticeConstant 5.43 Ang
%block LatticeVectors
{lattice}
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
 0.00 0.00 0.00 1
 0.25 0.25 0.27 1
%endblock AtomicCoordinatesAndAtomicSpecies
XC.functional GGA
XC.authors PBE
MeshCutoff 60 Ry
%block kgrid_Monkhorst_Pack
 2 0 0 0.0
 0 2 0 0.0
 0 0 2 0.0
%endblock kgrid_Monkhorst_Pack
MaxSCFIterations 1
SCF.MustConverge F
"""

# SIESTA 5.4.2 overwrites the functional's D3 parameters with generic defaults
# (s8=1, a1=0.4, a2=5.0) when DFTD3.UseXCDefaults is T (grimme_dispersion_m.F90, lines 236-251),
# so the published PBE parameters are passed explicitly. See logs/DECISIONS.md D-014.
_D3_BLOCK = """DFTD3 T
DFTD3.UseXCDefaults F
DFTD3.s6 1.0
DFTD3.s8 0.7875
DFTD3.a1 0.4289
DFTD3.a2 4.4407
DFTD3.s9 1.0
DFTD3.CoordinationCutoff 40.0 Bohr
"""


def _run_siesta(workdir: Path, extra: str, eps_xx: float = 0.0) -> dict[str, np.ndarray | float]:
    """Run SIESTA (one SCF step; D3 depends on geometry only) on the cell strained by ``eps_xx``."""
    workdir.mkdir(parents=True)
    shutil.copy2(Path(ref_path("siesta_pseudos_dir") or "") / "Si.tm2.nrl.psml", workdir / "Si.psml")
    vecs = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) @ np.diag([1.0 + eps_xx, 1.0, 1.0])
    lattice = "\n".join(" ".join(f"{x:.12f}" for x in v) for v in vecs)
    (workdir / "si.fdf").write_text(_SI_FDF.format(lattice=lattice) + extra)
    out = run_siesta_directory(workdir, "si.fdf", timeout=600).read_text()
    num = r"([-+0-9.eE]+)"
    forces = re.search(
        r"siesta: Atomic forces \(eV/Ang\):\s*\nsiesta:\s+1\s+" + r"\s+".join([num] * 3)
        + r"\s*\nsiesta:\s+2\s+" + r"\s+".join([num] * 3),
        out,
    )
    stress = re.search(
        r"siesta: Stress tensor \(static\) \(eV/Ang\*\*3\):\s*\n"
        + r"\s*\n".join([r"siesta:\s+" + r"\s+".join([num] * 3)] * 3),
        out,
    )
    assert forces is not None and stress is not None, out[-2000:]
    edisp = re.search(r"siesta: D3 dispersion =\s+" + num, out)
    return {
        "edisp": float(edisp.group(1)) if edisp else 0.0,
        "forces": np.array([float(x) for x in forces.groups()]).reshape(2, 3),
        "stress": np.array([float(x) for x in stress.groups()]).reshape(3, 3),
    }


@pytest.mark.external_refs("siesta_bin_or_cache", "siesta_pseudos_dir")
def test_d3_matches_siesta(tmp_path: Path, record_residuals) -> None:
    """D3 energy, forces and stress agree with SIESTA's D3 (same s-dftd3 library family).

    * Energy: compared directly.
    * Forces: D3-on minus D3-off SIESTA forces. SIESTA 5.4.2 bundles s-dftd3 1.1.0, whose analytic
      gradient is ~0.3% off its own energy (finite differences agree with s-dftd3 >= 1.7 and with
      jsiesta), hence the 3e-5 eV/Ang tolerance.
    * Stress: SIESTA adds the D3 virial with the wrong sign (final_H_f_stress -> dftd3_energy_forces),
      so the reference is the finite-difference strain derivative of SIESTA's own D3 energies.
    See logs/DECISIONS.md D-014.
    """
    on = _run_siesta(tmp_path / "on", _D3_BLOCK)
    off = _run_siesta(tmp_path / "off", "")
    e, f, sig = DFTD3Dispersion("PBE", three_body=True, params=PBE_D3BJ).evaluate(_si_displaced())

    f_siesta = np.asarray(on["forces"]) - np.asarray(off["forces"])
    record_residuals(
        {
            "energy_ev": abs(float(e) * HA_EV - float(on["edisp"])),
            "forces_ev_ang": float(np.max(np.abs(np.asarray(f) * HA_EV * ANG_BOHR - f_siesta))),
        },
        {"energy_ev": 2e-6, "forces_ev_ang": 3e-5},
    )
    assert abs(float(e) * HA_EV - float(on["edisp"])) < 2e-6
    np.testing.assert_allclose(np.asarray(f) * HA_EV * ANG_BOHR, f_siesta, atol=3e-5)

    h = 2e-3
    e_p = float(_run_siesta(tmp_path / "p", _D3_BLOCK, eps_xx=h)["edisp"])
    e_m = float(_run_siesta(tmp_path / "m", _D3_BLOCK, eps_xx=-h)["edisp"])
    vol_ang3 = abs(float(np.linalg.det(np.asarray(_si_displaced().cell)))) * BOHR_ANG**3
    sxx_siesta_fd = (e_p - e_m) / (2 * h * vol_ang3)
    sxx_js = float(sig[0, 0]) * HA_EV / BOHR_ANG**3
    record_residuals({"sigma_xx_ev_ang3": abs(sxx_js - sxx_siesta_fd)}, {"sigma_xx_ev_ang3": 2e-6})
    assert abs(sxx_js - sxx_siesta_fd) < 2e-6
    assert sxx_js > 0.0  # dispersion is attractive: stretching raises the energy


# ----------------------------------------------------------------------------------------- rVV10

_QE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "qe_rvv10" / "c_bilayer_rho.npz"
_QE_ENL_RY = 0.070718182102919  # pw.x 7.5, see tests/fixtures/qe_rvv10/README.md


def test_rvv10_kernel_zero_k_is_analytic() -> None:
    """4*pi * int R^2 phi(q, q, R) dR = -3 pi^2 / (16 q^{3/2}) (no truncation issue for large q)."""
    from jax_siesta.xc.dispersion import rvv10_kernel_table

    q = np.array([0.5, 0.3, 0.2])
    table, _ = rvv10_kernel_table(q)
    for i, qi in enumerate(q):
        exact = -3.0 * np.pi**2 / (16.0 * qi**1.5)
        assert abs(table[0, i, i] / exact - 1.0) < 1e-4


def test_rvv10_matches_quantum_espresso(record_residuals) -> None:
    """Non-local energy on QE's own converged density equals QE's printed value."""
    from jax_siesta.xc.dispersion import RVV10Dispersion

    data = np.load(_QE_FIXTURE)
    e = float(RVV10Dispersion().energy_from_density(jnp.asarray(data["rho"]), jnp.asarray(data["cell"])))
    record_residuals({"enl_ry": abs(2.0 * e - _QE_ENL_RY)}, {"enl_ry": 1e-9})
    assert abs(2.0 * e - _QE_ENL_RY) < 1e-9


def test_rvv10_stress_matches_finite_difference() -> None:
    """sigma = (1/Omega) dE/deps under a homogeneous, charge-conserving strain."""
    from jax_siesta.xc.dispersion import RVV10Dispersion

    s = _si_displaced()
    cell = jnp.asarray(s.cell)
    n = 12
    x = jnp.arange(n) / n
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing="ij")
    rho = 0.03 + 0.02 * jnp.cos(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * (yy + zz)) + 0.005 * jnp.sin(2 * jnp.pi * zz)
    rvv = RVV10Dispersion()
    _, _, sig = rvv.potential_and_energy(rho, cell)
    vol = float(jnp.abs(jnp.linalg.det(cell)))
    h = 1e-5
    for a, b in [(0, 0), (0, 1), (2, 1)]:
        eps = np.zeros((3, 3))
        eps[a, b] = h
        dp, dm = jnp.eye(3) + eps, jnp.eye(3) - eps
        ep = float(rvv.energy_from_density(rho / jnp.linalg.det(dp), dp @ cell))
        em = float(rvv.energy_from_density(rho / jnp.linalg.det(dm), dm @ cell))
        assert abs((ep - em) / (2 * h * vol) - float(sig[a, b])) < 1e-9
