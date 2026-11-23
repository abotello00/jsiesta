r"""Reference SIESTA binary adapter and parser for cross-code parity validation.

Theory:
    Bridges ``jax_siesta`` cases and FDF inputs directly to compiled reference
    SIESTA executables (standard non-symmetrized ``siesta`` and space/magnetic-group
    symmetrized ``siesta_spglib_full_impl``) using identical PSML pseudopotentials,
    mesh cutoffs, electronic smearing, and k-grids. Compares invariant Kohn-Sham
    energy components (:math:`T`, :math:`E^{\text{KB}} + E^{\text{SO}}`,
    :math:`E_H[\delta\rho]`, :math:`E_{\text{xc}}`), Fermi level, forces, stress
    tensor, spin moment, and IBZ symmetry reduction.

Algorithm:
    Translates ``case.json`` geometry and ``SCFConfig`` into a Cartesian Bohr FDF
    with ``WriteIonPlotFiles true`` (and ``kgrid.UseSymmetry true`` / ``DM.UseSymmetry true``
    when symmetry is enabled), executes the reference binary in an isolated directory,
    and parses ``si.out`` and ``<label>.FA`` into ``validation.py`` schema-v1 JSON.

Invariants:
    - FDF writes ``LatticeConstant 1.0 Bohr`` and ``NotScaledCartesianBohr`` coordinates verbatim.
    - Forces parsed in ``eV/Ang``; stress parsed in ``eV/Ang^3``; spin moments in :math:`\mu_B`.

SIESTA Mapping:
    Invokes upstream SIESTA 5.4+ (``siesta_forces``, ``compute_energies``,
    ``m_symmetry``, ``m_dm_symmetrize``) as ground-truth reference baselines.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

# Ensure editable/repo imports work when invoked by path inside isolated validation.py workdirs.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jax_siesta.core.types import (  # noqa: E402
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    Observables,
    SCFConfig,
    Structure,
)
from jax_siesta.io.fdf import load_case_json  # noqa: E402
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern  # noqa: E402
from jax_siesta.integrals.two_center import TwoCenterEvaluator  # noqa: E402
from jax_siesta.observables.forces import StaticObservableEvaluator  # noqa: E402
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf  # noqa: E402
from jax_siesta.grid.mesh import choose_mesh_dimensions  # noqa: E402

from jax_siesta.testing.external_refs import ref_cache_dir, ref_path  # noqa: E402

# Resolved from tests/external_refs.toml / JSIESTA_<KEY> (see jax_siesta.testing.external_refs).
DEFAULT_SIESTA_BIN = ref_path("siesta_bin") or Path("siesta")
DEFAULT_SIESTA_SYM_BIN = ref_path("siesta_sym_bin") or Path("siesta")
DEFAULT_SIESTA_MGGA_BIN = ref_path("siesta_mgga_bin") or Path("siesta")

_SPIN_FDF = {
    1: "non-polarized",
    2: "collinear",
    4: "non-collinear",
    8: "spin-orbit",
}


def resolve_siesta_executable(mode: str = "siesta", executable: Path | None = None) -> Path:
    r"""Resolve the target reference SIESTA binary path from override, env, or default.

    Theory:
        Maps execution mode (``"siesta"`` standard vs ``"siesta-sym"`` spglib branch)
        to the compiled SIESTA executable used as parity reference.

    Algorithm:
        Checks explicit ``executable``, then ``JSIESTA_SIESTA_SYM_BIN`` or
        ``JSIESTA_SIESTA_BIN``, then compiled workspace defaults.

    Parameters:
        mode: Reference mode (``"siesta"`` or ``"siesta-sym"``).
        executable: Optional explicit binary path override.

    Returns:
        Resolved executable ``Path``.

    Invariants:
        Raises ``FileNotFoundError`` if the resolved binary does not exist.

    SIESTA Mapping:
        Selects standard ``Src/siesta`` or ``siesta_spglib_full_impl/build/Src/siesta``.
    """
    if executable is not None:
        cand = Path(executable)
    elif mode == "siesta-sym":
        env_val = os.environ.get("JSIESTA_SIESTA_SYM_BIN")
        cand = Path(env_val) if env_val else DEFAULT_SIESTA_SYM_BIN
    elif mode == "siesta-mgga":
        env_val = os.environ.get("JSIESTA_SIESTA_MGGA_BIN")
        cand = Path(env_val) if env_val else DEFAULT_SIESTA_MGGA_BIN
    else:
        env_val = os.environ.get("JSIESTA_SIESTA_BIN")
        cand = Path(env_val) if env_val else DEFAULT_SIESTA_BIN

    if not cand.is_file():
        raise FileNotFoundError(f"SIESTA reference executable not found: {cand}")
    return cand


def write_siesta_fdf(
    workdir: Path,
    case: dict[str, Any],
    *,
    system_label: str = "jsiesta_ref",
    use_symmetry: bool = False,
    xc_functional_override: str | None = None,
) -> Path:
    r"""Generate an exact Bohr-coordinate SIESTA ``.fdf`` input file in ``workdir``.

    Theory:
        Transposes column lattice vectors :math:`\mathbf{A}` back into SIESTA row
        ``LatticeVectors`` and emits Cartesian Bohr positions, k-grid, XC authors,
        electronic smearing, and optional spglib symmetry toggles.

    Algorithm:
        Formats clean FDF text blocks using CODATA atomic units and SIESTA spin strings.

    Parameters:
        workdir: Destination directory (must already contain copied ``.psml`` files).
        case: Loaded ``case.json`` dictionary from ``load_case_json``.
        system_label: SIESTA ``SystemLabel`` prefix for output files.
        use_symmetry: Enables ``kgrid.UseSymmetry true`` and ``DM.UseSymmetry true``.
        xc_functional_override: Optional XC identifier override (e.g. ``"LDA"`` or ``"PBE"``).

    Returns:
        Path to written ``input.fdf``.

    Invariants:
        Preserves exact double-precision geometry and k-grid dimensions.

    SIESTA Mapping:
        Generates ``input.fdf`` read by ``fdf_read`` in reference SIESTA.
    """
    structure: Structure = case["structure"]
    config: SCFConfig = case["config"]
    k_grid: tuple[int, int, int] = case["k_grid"]
    k_shift: tuple[float, float, float] = tuple(case.get("k_shift", (0.0, 0.0, 0.0)))

    xc_name = (xc_functional_override or config.xc_functional).upper()
    xc_mix_block: list[str] | None = None
    if "R2SCAN01" in xc_name:
        xc_mix_block = ["%block XC.mix", "2", "MGGA LIBXC-0645 1.0 0.0", "MGGA LIBXC-0642 0.0 1.0", "%endblock XC.mix"]
        xc_func, xc_auth = "MGGA", "LIBXC-0645"
    elif "R2SCAN" in xc_name:
        xc_mix_block = ["%block XC.mix", "2", "MGGA LIBXC-0497 1.0 0.0", "MGGA LIBXC-0498 0.0 1.0", "%endblock XC.mix"]
        xc_func, xc_auth = "MGGA", "LIBXC-0497"
    elif "SCAN" in xc_name:
        xc_mix_block = ["%block XC.mix", "2", "MGGA LIBXC-0263 1.0 0.0", "MGGA LIBXC-0267 0.0 1.0", "%endblock XC.mix"]
        xc_func, xc_auth = "MGGA", "LIBXC-0263"
    elif "TPSS" in xc_name:
        xc_mix_block = ["%block XC.mix", "2", "MGGA LIBXC-0202 1.0 0.0", "MGGA LIBXC-0231 0.0 1.0", "%endblock XC.mix"]
        xc_func, xc_auth = "MGGA", "LIBXC-0202"
    elif "PBESOL" in xc_name:
        xc_func, xc_auth = "GGA", "PBEsol"
    elif "PBE" in xc_name:
        xc_func, xc_auth = "GGA", "PBE"
    else:
        xc_func, xc_auth = "LDA", "CA"

    unique_species: list[tuple[int, str]] = []
    seen_labels: set[str] = set()
    for sp_idx, z_num in zip(structure.species_indices, structure.atomic_numbers, strict=True):
        lbl = structure.species_labels[sp_idx]
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            unique_species.append((sp_idx + 1, lbl if isinstance(lbl, str) else str(lbl)))
    species_z_map = {
        structure.species_labels[sp_idx]: int(z_num)
        for sp_idx, z_num in zip(structure.species_indices, structure.atomic_numbers, strict=True)
    }

    cell_rows = np.asarray(structure.cell, dtype=np.float64).T
    pos_bohr = np.asarray(structure.positions, dtype=np.float64)
    moments = np.asarray(structure.initial_moments, dtype=np.float64)

    lines: list[str] = [
        f"SystemName  {system_label}",
        f"SystemLabel {system_label}",
        f"NumberOfAtoms {structure.n_atoms}",
        f"NumberOfSpecies {len(structure.species_labels)}",
        "%block ChemicalSpeciesLabel",
    ]
    for sp_i, lbl in enumerate(structure.species_labels, start=1):
        lines.append(f"  {sp_i} {species_z_map[lbl]} {lbl}")
    lines.extend(
        [
            "%endblock ChemicalSpeciesLabel",
            "PAO.BasisSize DZP",
            "LatticeConstant 1.0 Bohr",
            "%block LatticeVectors",
        ]
    )
    for row in cell_rows:
        lines.append(f"  {row[0]:.14f}  {row[1]:.14f}  {row[2]:.14f}")
    lines.extend(
        [
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat NotScaledCartesianBohr",
            "%block AtomicCoordinatesAndAtomicSpecies",
        ]
    )
    for ia in range(structure.n_atoms):
        p = pos_bohr[ia]
        lines.append(f"  {p[0]:.14f}  {p[1]:.14f}  {p[2]:.14f}  {structure.species_indices[ia] + 1}")
    lines.extend(
        [
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "%block kgrid.MonkhorstPack",
            f"  {k_grid[0]} 0 0 {k_shift[0]:.6f}",
            f"  0 {k_grid[1]} 0 {k_shift[1]:.6f}",
            f"  0 0 {k_grid[2]} {k_shift[2]:.6f}",
            "%endblock kgrid.MonkhorstPack",
            *(xc_mix_block if xc_mix_block is not None else [f"XC.functional {xc_func}", f"XC.authors    {xc_auth}"]),
            f"Spin {_SPIN_FDF[config.nspin]}",
            f"MeshCutoff {config.mesh_cutoff_ry:.6f} Ry",
            f"MeshSubDivisions {config.mesh_subdivisions}",
            f"ElectronicTemperature {config.electronic_temp_ha:.12e} Ha",
            f"MaxSCFIterations {config.max_scf_iterations}",
            f"SCF.DM.Tolerance {config.d_tol:.12e}",
            f"SCF.H.Tolerance  {config.h_tol_ha:.12e} Ha",
            "SCF.MustConverge true",
            "WriteIonPlotFiles true",
            "WriteCoorInitial false",
        ]
    )

    if config.nspin in (2, 4, 8):
        lines.append("%block DM.InitSpin")
        for ia in range(structure.n_atoms):
            mx, my, mz = moments[ia]
            if config.nspin == 2:
                lines.append(f"  {ia + 1} {mz:+.6f}")
            else:
                norm_m = float(np.linalg.norm([mx, my, mz]))
                if norm_m > 1e-12:
                    theta = float(np.degrees(np.arccos(np.clip(mz / norm_m, -1.0, 1.0))))
                    phi = float(np.degrees(np.arctan2(my, mx)))
                    lines.append(f"  {ia + 1} {norm_m:.6f} {theta:.6f} {phi:.6f}")
                else:
                    lines.append(f"  {ia + 1} +0.0")
        lines.append("%endblock DM.InitSpin")

    if use_symmetry:
        lines.extend(
            [
                "UseSymmetry true",
                "kgrid.UseSymmetry true",
                "DM.UseSymmetry true",
                f"Symmetry.Tolerance {config.symprec:.12e} Bohr",
            ]
        )
    if config.dispersion.upper() not in ("NONE", ""):
        # SIESTA has no rVV10 ("XC.authors VV" is VV10 with rPW86 exchange, a different functional), and its
        # DFT-D3 defaults are unreliable (logs/DECISIONS.md D-014). Dispersion parity is tested in isolation
        # in tests/test_dispersion.py instead of through full matched runs.
        raise NotImplementedError(
            f"Matched SIESTA runs do not support dispersion={config.dispersion!r}; see tests/test_dispersion.py."
        )

    fdf_path = workdir / "input.fdf"
    fdf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fdf_path


def parse_siesta_outputs(workdir: Path, system_label: str, nspin: int) -> dict[str, Any]:
    r"""Parse SIESTA standard output, ``.FA`` forces, stress tensor, and IBZ k-point count.

    Theory:
        Extracts invariant Kohn-Sham energy components, Fermi energy, forces, stress,
        spin polarization, real-space mesh dimensions, and k-grid reduction counts.

    Algorithm:
        Reads ``si.out`` and ``<system_label>.FA`` using regexes anchored on SIESTA's
        canonical output markers.

    Parameters:
        workdir: Directory where SIESTA executed.
        system_label: SIESTA ``SystemLabel`` prefix.
        nspin: Spin mode (``1, 2, 4, 8``).

    Returns:
        Dictionary of parsed observables in eV / Angstrom / Bohr-free validation units
        alongside diagnostic provenance fields.

    Invariants:
        Raises ``RuntimeError`` if normal completion marker or ``Final energy`` is missing.

    SIESTA Mapping:
        Parses ``siesta_forces``, ``write_energies``, ``outcell``, and ``m_kpoints`` output.
    """
    out_path = workdir / "si.out"
    text = out_path.read_text(encoding="utf-8")
    if "Job completed" not in text and "siesta: Final energy (eV):" not in text:
        raise RuntimeError(f"SIESTA output incomplete in {out_path}")

    def _last_float(pattern: str, default: float | None = None) -> float:
        matches = re.findall(pattern, text)
        if not matches:
            if default is not None:
                return default
            raise ValueError(f"Missing SIESTA pattern {pattern!r} in {out_path}")
        return float(matches[-1])

    e_kin = _last_float(r"siesta:\s+Kinetic\s*=\s*([-+0-9.eE]+)")
    e_nl = _last_float(r"siesta:\s+Enl(?:\(\+so\))?\s*=\s*([-+0-9.eE]+)")
    e_so = _last_float(r"siesta:\s+Eso(?:\(nil\))?\s*=\s*([-+0-9.eE]+)", 0.0)
    e_hartree = _last_float(r"siesta:\s+DUscf\s*=\s*([-+0-9.eE]+)")
    e_xc = _last_float(r"siesta:\s+Exch\.-corr\.\s*=\s*([-+0-9.eE]+)")
    e_tot = _last_float(r"siesta:\s+Total\s*=\s*([-+0-9.eE]+)")
    # SIESTA prints the Mermin free energy as "siesta: FreeEng =" (there is no "Free =" line).
    e_free = _last_float(r"siesta:\s+FreeEng\s*=\s*([-+0-9.eE]+)")
    e_na_neutral = (
        _last_float(r"siesta:\s+Ena\s*=\s*([-+0-9.eE]+)")
        + _last_float(r"siesta:\s+DEna\s*=\s*([-+0-9.eE]+)")
        - _last_float(r"siesta:\s+Eions\s*=\s*([-+0-9.eE]+)")
    )
    e_fermi = _last_float(r"siesta:\s+Fermi\s*=\s*([-+0-9.eE]+)")
    n_kpts = int(_last_float(r"siesta: k-grid: Number of k-points\s*=\s*(\d+)"))

    mesh_matches = re.findall(r"InitMesh:\s+MESH\s*=\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", text)
    mesh_dims = tuple(int(x) for x in mesh_matches[-1]) if mesh_matches else None

    scf_matches = re.findall(r"scf:\s+(\d+)\s+", text)
    n_scf = int(scf_matches[-1]) if scf_matches else 0

    stress_match = re.search(
        r"siesta:\s+Stress tensor \(static\) \(eV/Ang\*\*3\):\s*\n"
        r"siesta:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\n"
        r"siesta:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\n"
        r"siesta:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        text,
    )
    if stress_match is None:
        raise ValueError(f"Missing static Stress tensor in {out_path}")
    s_vals = [float(stress_match.group(i)) for i in range(1, 10)]
    stress = [s_vals[0:3], s_vals[3:6], s_vals[6:9]]

    fa_lines = [ln.strip() for ln in (workdir / f"{system_label}.FA").read_text().splitlines() if ln.strip()]
    n_at = int(fa_lines[0])
    forces = [[float(x) for x in fa_lines[1 + ia].split()[1:4]] for ia in range(n_at)]

    res: dict[str, Any] = {
        "e_kinetic_ev": e_kin,
        "e_kb_ev": e_nl + e_so,
        "e_hartree_ev": e_hartree,
        "e_xc_ev": e_xc,
        "fermi_level_ev": e_fermi,
        "e_na_ev": e_na_neutral,
        "e_total_ev": e_tot,
        "final_post_scf_free_e_ev": e_free,
        "forces": forces,
        "stress": stress,
        "_meta": {
            "e_total_siesta_ev": e_tot,
            "n_kpoints": n_kpts,
            "mesh_dims": mesh_dims,
            "n_scf_iterations": n_scf,
        },
    }
    if nspin in (2, 4, 8):
        spin_matches = re.findall(r"spin moment:[^\n]*\}\s+([-+0-9.eE]+)", text)
        res["total_spin_moment"] = abs(float(spin_matches[-1])) if spin_matches else 0.0
    return res


def observables_to_component_validation_dict(
    obs: Observables,
    nspin: int,
) -> dict[str, Any]:
    r"""Convert ``jsiesta`` ``Observables`` into component keys matching ``parse_siesta_outputs``.

    Theory:
        Extracts the reference-comparable Kohn-Sham terms (kinetic, nonlocal KB/SO,
        residual Hartree :math:`E_H[\delta\rho]`, exchange-correlation, neutral-atom energy,
        Fermi energy, forces, stress, spin magnitude) and the absolute total and free energies
        (``e_total_ev``, ``final_post_scf_free_e_ev``), which include the one-centre
        neutral-atom constants and so compare directly with SIESTA's ``Total``/``Free``.

    Algorithm:
        Applies exact CODATA Hartree/Bohr conversions to match SIESTA ``eV``, ``eV/Ang``,
        and ``eV/Ang^3`` units.

    Parameters:
        obs: Converged ``Observables`` PyTree.
        nspin: Spin mode (``1, 2, 4, 8``).

    Returns:
        Strictly finite JSON dictionary accepted by ``validation.py``.

    Invariants:
        Exact key parity with ``parse_siesta_outputs`` (excluding ``_meta``).

    SIESTA Mapping:
        Matches ``siesta: Kinetic``, ``Enl+Eso``, ``DUscf``, ``Exch.-corr.``, ``Fermi``,
        ``.FA``, and ``Stress tensor (static)``.
    """
    f_conv = HARTREE_TO_EV / BOHR_TO_ANGSTROM
    s_conv = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**3)
    out: dict[str, Any] = {
        "e_kinetic_ev": float(obs.e_kinetic_ha) * HARTREE_TO_EV,
        "e_kb_ev": float(obs.e_kb_ha) * HARTREE_TO_EV,
        "e_hartree_ev": float(obs.e_hartree_ha) * HARTREE_TO_EV,
        "e_xc_ev": float(obs.e_xc_ha) * HARTREE_TO_EV,
        "fermi_level_ev": float(obs.fermi_level_ha) * HARTREE_TO_EV,
        "e_na_ev": float(obs.e_na_ha) * HARTREE_TO_EV,
        "e_total_ev": float(obs.total_energy_ha) * HARTREE_TO_EV,
        "final_post_scf_free_e_ev": float(obs.final_post_scf_free_e_ha) * HARTREE_TO_EV,
        "forces": (np.asarray(obs.forces_ha_per_bohr, dtype=np.float64) * f_conv).tolist(),
        "stress": (np.asarray(obs.stress_ha_per_bohr3, dtype=np.float64) * s_conv).tolist(),
    }
    if nspin in (2, 4, 8):
        out["total_spin_moment"] = float(np.linalg.norm(np.asarray(obs.total_spin_moment)))
    return out


def run_siesta_reference(
    case_json: Path,
    workdir: Path,
    *,
    mode: str = "siesta",
    executable: Path | None = None,
    psml_dir: Path | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    r"""Execute a real reference SIESTA binary on ``case_json`` inside ``workdir``.

    Theory:
        Stages pseudopotentials and ``input.fdf``, invokes ``siesta`` or
        ``siesta_spglib_full_impl``, and harvests energies, forces, stress, and
        SIESTA-generated ``<Species>.ion.xml`` radial tables.

    Algorithm:
        Copies ``.psml`` files, writes ``input.fdf``, runs subprocess with stdin/stdout
        files, and calls ``parse_siesta_outputs``.

    Parameters:
        case_json: Path to ``case.json`` benchmark descriptor.
        workdir: Isolated scratch directory for SIESTA execution.
        mode: ``"siesta"`` (non-symmetrized) or ``"siesta-sym"`` (spglib symmetry).
        executable: Optional explicit SIESTA executable override.
        psml_dir: Directory containing ``<Species>.psml`` files.
        timeout: Maximum subprocess runtime in seconds.

    Returns:
        Dictionary of parsed observables plus ``_meta`` provenance.

    Invariants:
        Generates ``<Species>.ion.xml`` inside ``workdir`` via ``WriteIonPlotFiles true``.

    SIESTA Mapping:
        Drives standalone ``siesta`` execution for cross-code regression testing.
    """
    case = load_case_json(case_json)
    structure: Structure = case["structure"]
    config: SCFConfig = case["config"]
    use_sym = mode == "siesta-sym" or config.use_symmetry

    if psml_dir is None:
        psml_dir = Path(case_json).resolve().parent
    workdir.mkdir(parents=True, exist_ok=True)
    for lbl in structure.species_labels:
        src = Path(psml_dir) / f"{lbl}.psml"
        if not src.is_file():
            raise FileNotFoundError(f"Missing PSML pseudopotential: {src}")
        shutil.copy2(src, workdir / f"{lbl}.psml")

    label = "jsiesta_ref"
    fdf_path = write_siesta_fdf(workdir, case, system_label=label, use_symmetry=use_sym)

    # Reference cache: a run is identified by the mode, the FDF text and the pseudopotential bytes. A hit replays the
    # stored SIESTA output (and ion tables) without a SIESTA binary; a miss runs SIESTA and stores the result.
    cache_root = ref_cache_dir()
    entry: Path | None = None
    if cache_root is not None:
        digest_in = hashlib.sha256()
        digest_in.update(mode.encode())
        digest_in.update(fdf_path.read_bytes())
        for lbl in structure.species_labels:
            digest_in.update((workdir / f"{lbl}.psml").read_bytes())
        entry = cache_root / digest_in.hexdigest()[:32]
        if _cache_replay(entry, workdir):
            parsed = parse_siesta_outputs(workdir, label, config.nspin)
            parsed["_meta"]["wall_seconds"] = 0.0
            parsed["_meta"]["cache_hit"] = True
            return parsed

    exe = resolve_siesta_executable(mode=mode, executable=executable)
    t0 = time.perf_counter()
    with fdf_path.open("rb") as fin, (workdir / "si.out").open("wb") as fout:
        proc = subprocess.run(
            [str(exe)],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.PIPE,
            cwd=workdir,
            timeout=timeout,
            check=False,
        )
    wall_s = float(time.perf_counter() - t0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SIESTA binary exited with {proc.returncode}: {proc.stderr.decode(errors='replace')}"
        )
    parsed = parse_siesta_outputs(workdir, label, config.nspin)
    parsed["_meta"]["wall_seconds"] = wall_s
    parsed["_meta"]["cache_hit"] = False
    if entry is not None:
        _cache_store(entry, workdir, f"mode={mode}\nexecutable={exe}\n")
    return parsed


def _cache_replay(entry: Path, workdir: Path) -> bool:
    """Copy a completed cache entry into ``workdir``; return whether the entry existed."""
    if not (entry / "COMPLETE").is_file():
        return False
    for item in entry.iterdir():
        if item.is_file() and item.name != "COMPLETE":
            shutil.copy2(item, workdir / item.name)
    return True


def _cache_store(entry: Path, workdir: Path, note: str) -> None:
    """Store the small files of ``workdir`` as a cache entry (atomically, via a temporary directory)."""
    tmp = entry.with_name(entry.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for item in workdir.iterdir():
        # Keep inputs, text output, forces and ion tables; skip bulky binaries (density matrices, wavefunctions).
        if item.is_file() and item.stat().st_size < 20_000_000 and item.suffix not in (".DM", ".WFSX", ".TSHS", ".bin"):
            shutil.copy2(item, tmp / item.name)
    (tmp / "COMPLETE").write_text(note, encoding="utf-8")
    shutil.rmtree(entry, ignore_errors=True)
    tmp.rename(entry)


def run_siesta_directory(
    workdir: Path,
    fdf_name: str,
    *,
    executable_key: str = "siesta_bin",
    stdout_name: str = "siesta.out",
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> Path:
    r"""Run a SIESTA binary on a prepared directory, replaying a cached run when ``JSIESTA_SIESTA_REF_CACHE`` has one.

    Theory:
        Several cross-code tests drive SIESTA with hand-written FDF files (mesh set-up, D3, the ``pr_mgga`` suite).
        A real run is a pure function of the FDF text, the pseudopotentials and the executable's identity, so its text
        output can be cached exactly like :func:`run_siesta_reference` and replayed on machines without a SIESTA binary.

    Algorithm:
        Hashes the executable key, the FDF bytes and every ``*.psml`` file in ``workdir``. On a hit the stored files are
        copied into ``workdir``. On a miss the binary resolved by ``executable_key`` runs with the FDF on stdin and
        stdout/stderr in ``stdout_name``; the result is stored when a cache directory is configured.

    Parameters:
        workdir: Directory holding the FDF and the PSML files; the run happens here.
        fdf_name: Name of the FDF file inside ``workdir`` (fed on stdin).
        executable_key: External-reference key of the executable (``siesta_bin``, ``siesta_sym_bin``, ``siesta_mgga_bin``).
        stdout_name: File receiving SIESTA's standard output and error.
        timeout: Maximum runtime in seconds.
        env: Optional environment for the subprocess.

    Returns:
        Path of the text output file in ``workdir``.

    Raises:
        RuntimeError: If SIESTA fails.
        FileNotFoundError: If neither a cache entry nor the executable exists.

    Invariants:
        A replayed run produces byte-identical output files to the run that filled the cache.

    SIESTA Mapping:
        Standalone ``siesta`` execution, as in the ``Tests/`` harness.
    """
    out_path = workdir / stdout_name
    cache_root = ref_cache_dir()
    entry: Path | None = None
    if cache_root is not None:
        digest = hashlib.sha256()
        digest.update(b"directory-run\0" + executable_key.encode() + b"\0")
        digest.update((workdir / fdf_name).read_bytes())
        for psml in sorted(workdir.glob("*.psml")):
            digest.update(psml.name.encode() + b"\0" + psml.read_bytes())
        entry = cache_root / digest.hexdigest()[:32]
        if _cache_replay(entry, workdir):
            return out_path
    exe = ref_path(executable_key)
    if exe is None or not exe.exists():
        raise FileNotFoundError(f"SIESTA executable {executable_key!r} is not available and no cached run matches")
    with (workdir / fdf_name).open("rb") as fin, out_path.open("wb") as fout:
        proc = subprocess.run(
            [str(exe)], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=workdir, env=env, timeout=timeout, check=False
        )
    if proc.returncode != 0:
        raise RuntimeError(f"SIESTA exited with {proc.returncode}; see {out_path}")
    if entry is not None:
        _cache_store(entry, workdir, f"directory-run\nexecutable={exe}\n")
    return out_path


def run_jsiesta_matched_case(
    case_json: Path,
    psml_dir: Path,
    *,
    ion_dir: Path | None = None,
    use_symmetry: bool = False,
    mesh_shape_override: tuple[int, int, int] | None = None,
    symmetrize_grid: bool = True,
    reference_mesh_dims: tuple[int, int, int] | None = None,
) -> tuple[Observables, dict[str, Any]]:
    r"""Execute ``jax_siesta`` using matched SIESTA ``.psml`` and optional ``.ion.xml`` tables.

    Theory:
        Loads exact PSML and (when supplied) SIESTA-generated ``.ion.xml`` PAO/KB/VNA
        radial datasets so ``jax_siesta`` and SIESTA evaluate identical Kohn-Sham
        discrete equations without basis-generation discrepancies.

    Algorithm:
        Constructs ``BasisData``, builds ``SparsePattern`` and ``KPointMesh`` (via
        spglib native symmetry when ``use_symmetry=True``), executes ``run_scf``, and
        evaluates post-SCF ``StaticObservableEvaluator``.

    Parameters:
        case_json: Path to ``case.json``.
        psml_dir: Directory containing ``<Species>.psml`` (and optionally ``.ion.xml``).
        ion_dir: Optional separate directory containing SIESTA ``<Species>.ion.xml`` files.
        use_symmetry: Enables native Phase 2 ``spglib`` symmetry reduction.
        symmetrize_grid: Symmetrise grid densities/potentials (``False`` mimics ``siesta_spglib_full_impl``).
        mesh_shape_override: Optional explicit 3D mesh matching SIESTA ``InitMesh`` (escape hatch).
        reference_mesh_dims: SIESTA's ``InitMesh`` dimensions. The run uses SIESTA's mesh *criterion*
            (``mesh_criterion="siesta"``) and raises ``AssertionError`` unless it selects exactly these dims.

    Returns:
        Tuple ``(Observables, provenance_meta_dict)``.

    Invariants:
        Raises ``RuntimeError`` if SCF fails to converge.

    SIESTA Mapping:
        Matched twin of ``run_siesta_reference`` inside ``jax_siesta``.
    """
    case = load_case_json(case_json)
    structure: Structure = case["structure"]
    config: SCFConfig = replace(
        case["config"], use_symmetry=use_symmetry, symmetrize_grid=symmetrize_grid, mesh_criterion="siesta"
    )
    k_grid: tuple[int, int, int] = case["k_grid"]
    k_shift: tuple[float, float, float] = tuple(case.get("k_shift", (0.0, 0.0, 0.0)))

    loader = PSMLBasisLoader()
    species_map = {}
    for lbl in structure.species_labels:
        psml_path = Path(psml_dir) / f"{lbl}.psml"
        ion_path: str | None = None
        if ion_dir is not None:
            cand = Path(ion_dir) / f"{lbl}.ion.xml"
            if cand.is_file():
                ion_path = str(cand)
        elif (Path(psml_dir) / f"{lbl}.ion.xml").is_file():
            ion_path = str(Path(psml_dir) / f"{lbl}.ion.xml")
        species_map[lbl] = loader.load_species(
            lbl,
            str(psml_path),
            ion_path=ion_path,
            prefer_lj_projectors=(config.nspin == 8),
        )

    t_start = time.perf_counter()
    basis = loader.build_basis(structure, species_map, include_soc_projectors=(config.nspin == 8))
    sym_proj = None
    if config.use_symmetry:
        from jax_siesta.symmetry.engine import build_native_symmetry

        sym_proj, pattern, kmesh, mesh_shape = build_native_symmetry(
            structure, basis, k_grid, config, mesh_shape_override=mesh_shape_override
        )
        two_center = sym_proj.evaluate_irreducible_two_center(structure, basis, pattern, config)
    else:
        pattern = build_sparse_pattern(structure, basis)
        kmesh = build_monkhorst_pack_mesh(k_grid, shifts=k_shift)
        mesh_shape = mesh_shape_override or choose_mesh_dimensions(
            structure.cell, config.mesh_cutoff_ry, criterion=config.mesh_criterion,
            mesh_subdivisions=config.mesh_subdivisions,
        )
        two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    t_setup = time.perf_counter()
    if reference_mesh_dims is not None and tuple(int(x) for x in mesh_shape) != tuple(
        int(x) for x in reference_mesh_dims
    ):
        raise AssertionError(
            f"mesh criterion selected {tuple(int(x) for x in mesh_shape)} but SIESTA used "
            f"{tuple(int(x) for x in reference_mesh_dims)}"
        )

    scf_state, mesh_shape = run_scf(
        structure, basis, pattern, two_center, kmesh, config, mesh_shape=mesh_shape, sym_proj=sym_proj
    )
    t_scf = time.perf_counter()
    obs = StaticObservableEvaluator().evaluate(
        structure, basis, pattern, two_center, kmesh, scf_state, config, mesh_shape=mesh_shape, sym_proj=sym_proj
    )
    t_obs = time.perf_counter()
    if sym_proj is not None:
        obs = replace(obs, active_symmetry_ops=sym_proj.n_ops, magnetic_subgroup=sym_proj.subgroup_label)
    if not obs.converged:
        raise RuntimeError(f"jsiesta SCF failed to converge: outcome={obs.outcome}")
    meta = {
        "n_kpoints": int(kmesh.n_k),
        "mesh_dims": tuple(int(x) for x in mesh_shape),
        "n_scf_iterations": int(obs.n_iterations),
        "active_symmetry_ops": int(obs.active_symmetry_ops),
        "magnetic_subgroup": obs.magnetic_subgroup,
        "xc_backend": obs.xc_backend,
        "setup_seconds": float(t_setup - t_start),
        "scf_seconds": float(t_scf - t_setup),
        "observables_seconds": float(t_obs - t_scf),
        "wall_seconds": float(t_obs - t_start),
    }
    return obs, meta


def main() -> None:
    r"""CLI adapter for running SIESTA vs ``jax_siesta`` under ``validation.py``.

    Theory:
        Dispatches either a reference SIESTA executable or ``jax_siesta`` using matched
        ``case.json`` inputs and emits component-wise ``observables.json``.

    Algorithm:
        Parses CLI flags, executes selected backend, strips non-numeric metadata from
        ``observables.json`` to satisfy ``validation.flatten()``.

    Invariants:
        Exits non-zero on non-convergence or missing reference binaries.

    SIESTA Mapping:
        Connects ``validation.py`` manifests directly to real ``siesta`` executables.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case_json", type=Path)
    p.add_argument("--mode", choices=("siesta", "siesta-sym", "jsiesta", "jsiesta-sym"), required=True)
    p.add_argument("--executable", type=Path, default=None)
    p.add_argument("--psml-dir", type=Path, default=None)
    p.add_argument("--ion-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("observables.json"))
    args = p.parse_args()

    case = load_case_json(args.case_json)
    nspin = case["config"].nspin
    psml_dir = args.psml_dir or args.case_json.resolve().parent

    if args.mode in ("siesta", "siesta-sym"):
        workdir = args.output.resolve().parent / f"siesta_run_{args.mode}"
        raw = run_siesta_reference(
            args.case_json,
            workdir,
            mode=args.mode,
            executable=args.executable,
            psml_dir=psml_dir,
        )
        payload = {k: v for k, v in raw.items() if k != "_meta"}
    else:
        obs, _meta = run_jsiesta_matched_case(
            args.case_json,
            psml_dir,
            ion_dir=args.ion_dir,
            use_symmetry=(args.mode == "jsiesta-sym"),
        )
        payload = observables_to_component_validation_dict(obs, nspin=nspin)

    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
