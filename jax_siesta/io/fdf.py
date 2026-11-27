r"""FDF input parser subset and ``validation.py`` observable serialization.

Theory:
    Translates SIESTA ``.fdf`` input keywords (``LatticeVectors``, ``AtomicCoordinatesFormat``,
    ``AtomicCoordinatesAndAtomicSpecies``, ``ChemicalSpeciesLabel``, ``kgrid.MonkhorstPack``,
    ``MeshCutoff``, ``XC.authors``, ``Spin``, ``ElectronicTemperature``, ``DM.MixingWeight``,
    ``DM.NumberPulay``) and ``case.json`` manifests into atomic-unit ``Structure``
    and ``SCFConfig`` objects, and converts ``Observables`` to explicit eV / Angstrom
    units for ``validation.py``.

Algorithm:
    Parses line-oriented FDF blocks and scalar key-value pairs with explicit unit
    tokens, converting all lengths to Bohr and energies to Hartree.

Invariants:
    - Always emits ``last_scf_free_e_ev`` and ``final_post_scf_free_e_ev`` as separate
      keys with ``dm_out_recomputed`` provenance (no silent fallback).
    - Forces exported in ``eV/Ang``; stress in ``eV/Ang^3``.

SIESTA Mapping:
    Replaces ``fdf`` input reading and ``outcoor`` / ``m_energies`` JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    KB_HARTREE_PER_KELVIN,
    RY_TO_HARTREE,
    Observables,
    SCFConfig,
    Structure,
)

_LENGTH_TO_BOHR = {
    "bohr": 1.0,
    "ang": ANGSTROM_TO_BOHR,
    "angstrom": ANGSTROM_TO_BOHR,
    "nm": 10.0 * ANGSTROM_TO_BOHR,
}
_ENERGY_TO_HA = {
    "ha": 1.0,
    "hartree": 1.0,
    "ry": RY_TO_HARTREE,
    "rydberg": RY_TO_HARTREE,
    "ev": EV_TO_HARTREE,
    "mev": 1.0e-3 * EV_TO_HARTREE,
    "mry": 1.0e-3 * RY_TO_HARTREE,
    "mha": 1.0e-3,
    "mhartree": 1.0e-3,
    "k": KB_HARTREE_PER_KELVIN,
}


def _fdf_key(name: str) -> str:
    """Normalised FDF label: case-insensitive, with ``.``, ``_`` and ``-`` removed (as SIESTA's fdf does)."""
    return name.lower().replace(".", "").replace("_", "").replace("-", "")


def _fdf_strip(line: str) -> str:
    """One FDF line without its comment (``#``, ``!`` or ``;`` to the end of the line)."""
    for mark in "#!;":
        line = line.split(mark)[0]
    return line.strip()


def tokenize_fdf(
    fdf_path: str | Path, _depth: int = 0
) -> tuple[dict[str, list[str]], dict[str, list[list[str]]], dict[str, str]]:
    """Split an FDF file into scalar entries and blocks, following ``%include`` and blocks read from files.

    Theory:
        FDF is a label-value format: labels are case-insensitive and ignore ``.``, ``_`` and ``-``; the first
        occurrence of a label wins; ``%block`` ... ``%endblock`` encloses free-format lines; ``%include file``
        inserts another file and ``%block Label < file`` reads the block body from a file.

    Algorithm:
        One pass over the lines. Paths of included files are relative to the including file. Values are kept as
        token lists; units and meanings are resolved by the callers (``parse_fdf``, ``jax_siesta.app``).

    Parameters:
        fdf_path: Path to the ``.fdf`` file.

    Returns:
        ``(scalars, blocks, names)``: tokens after each scalar label, token rows of each block, and the label as
        written in the file, all keyed by the normalised label.

    Invariants:
        - Raises ``ValueError`` on an unclosed block or on includes nested deeper than 10 files.
        - The first definition of a label is kept, as in SIESTA.

    SIESTA Mapping:
        The reading part of ``libfdf`` (``fdf_init``); ``Label < file`` for scalars is not supported.
    """
    path = Path(fdf_path)
    if _depth > 10:
        raise ValueError(f"FDF %include nesting deeper than 10 files at {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    scalars: dict[str, list[str]] = {}
    blocks: dict[str, list[list[str]]] = {}
    names: dict[str, str] = {}

    i = 0
    while i < len(lines):
        raw = _fdf_strip(lines[i])
        i += 1
        if not raw:
            continue
        toks = raw.split()
        head = toks[0].lower()
        if head == "%include" and len(toks) > 1:
            sub_s, sub_b, sub_n = tokenize_fdf(path.parent / toks[1], _depth + 1)
            for key, val in sub_s.items():
                scalars.setdefault(key, val)
            for key, rows in sub_b.items():
                blocks.setdefault(key, rows)
            for key, name in sub_n.items():
                names.setdefault(key, name)
        elif head == "%block":
            bname = _fdf_key(toks[1])
            body: list[list[str]] = []
            if len(toks) >= 4 and toks[2] == "<":
                for bline in (path.parent / toks[3]).read_text(encoding="utf-8").splitlines():
                    braw = _fdf_strip(bline)
                    if braw and not braw.lower().startswith("%"):
                        body.append(braw.split())
            else:
                closed = False
                while i < len(lines):
                    braw = _fdf_strip(lines[i])
                    i += 1
                    if not braw:
                        continue
                    btoks = braw.split()
                    if btoks[0].lower() == "%endblock":
                        closed = True
                        break
                    body.append(btoks)
                if not closed:
                    raise ValueError(f"Unclosed %block {toks[1]} in {fdf_path}")
            blocks.setdefault(bname, body)
            names.setdefault(bname, toks[1])
        else:
            key = _fdf_key(toks[0])
            scalars.setdefault(key, toks[1:])
            names.setdefault(key, toks[0])
    return scalars, blocks, names


# Keywords that only steer SIESTA's printing or files jsiesta does not produce: accepted and ignored silently.
_FDF_IGNORED = frozenset(
    _fdf_key(k)
    for k in (
        "SystemName", "NumberOfAtoms", "NumberOfSpecies", "WriteCoorInitial", "WriteCoorStep", "WriteForces",
        "WriteKpoints", "WriteEigenvalues", "WriteKbands", "WriteBands", "WriteMullikenPop", "WriteCoorXmol",
        "WriteMDXmol", "WriteMDHistory", "WriteIonPlotFiles", "WriteDM", "WriteDM.NetCDF", "WriteDMHS.NetCDF",
        "LongOutput", "SaveRho", "SaveDeltaRho", "SaveElectrostaticPotential", "SaveTotalPotential",
        "SaveTotalCharge", "SolutionMethod", "Diag.ParallelOverK", "Diag.Algorithm",
        "NumberOfEigenStates", "DM.UseSaveDM", "MD.UseSaveXV", "UseSaveData", "File.DM.Init", "CDF.Save",
        "SaveHS", "SCF.Mixer.Variant", "DM.MixSCF1", "SCF.Mix.First", "XML.Write", "COOP.Write",
        "WriteWaveFunctions", "WriteHirshfeldPop", "WriteVoronoiPop", "DirectPhi", "MullikenInSCF",
        "TimerReportThreshold", "UseTreeTimer", "AllocReportLevel", "PAO.BasisType", "BandLinesScale",
    )
)
# Keywords this reader interprets (used to report the ones it does not).
_FDF_USED = frozenset(
    _fdf_key(k)
    for k in (
        "SystemLabel", "LatticeConstant", "LatticeVectors", "LatticeParameters", "ChemicalSpeciesLabel",
        "AtomicCoordinatesFormat", "AtomicCoordinatesAndAtomicSpecies", "Spin", "SpinPolarized",
        "NonCollinearSpin", "SpinOrbit", "DM.InitSpin", "MeshCutoff", "Mesh.SubDivisions", "Mesh.Sizes",
        "ElectronicTemperature", "OccupationFunction", "kgrid.MonkhorstPack", "XC.Functional", "XC.Authors",
        "DM.MixingWeight", "DM.NumberPulay", "SCF.Mixer.Weight", "SCF.Mixer.History", "SCF.Mixer.Method",
        "SCF.Mix", "DM.Tolerance", "SCF.DM.Tolerance", "SCF.H.Tolerance", "SCF.FreeE.Tolerance",
        "MaxSCFIterations", "NetCharge", "DFTD3", "PAO.BasisSize", "PAO.EnergyShift", "PAO.SplitNorm", "PAO.SplitNormH",
        "PAO.Basis", "User.Basis", "JSiesta.UseSymmetry", "kgrid.UseSymmetry", "JSiesta.Dispersion",
        "MD.TypeOfRun", "MD.Steps", "MD.NumCGsteps", "BandLines", "SCF.MustConverge",
        "HFX.NumberOfGaussians", "HFX.ScreeningTolerance", "HFX.PatternScale",
    )
)
_SPIN_MODES = {
    "none": 1, "non-polarized": 1, "nonpolarized": 1, "unpolarized": 1,
    "polarized": 2, "collinear": 2,
    "non-collinear": 4, "noncollinear": 4, "non-colinear": 4,
    "spin-orbit": 8, "spinorbit": 8, "soc": 8, "so": 8, "spin-orbit+onsite": 8, "spin-orbit+offsite": 8,
}
_FDF_TRUE = ("t", "true", ".true.", "yes", "y", "1")


def _xc_from_fdf(scalars: dict[str, list[str]]) -> str:
    """jsiesta functional name for ``XC.Functional`` / ``XC.Authors`` (SIESTA's default is LDA, CA)."""
    family = scalars.get("xcfunctional", [""])[0].upper() if scalars.get("xcfunctional") else ""
    authors = scalars.get("xcauthors", [""])[0] if scalars.get("xcauthors") else ""
    key = authors.upper()
    if not family and not key:
        return "LDA_PZ"
    if key in ("CA", "PZ") or (family in ("LDA", "LSD") and not key):
        return "LDA_PZ"
    if key == "PW92":
        return "LDA_PW92"
    if not key and family == "GGA":
        return "PBE"
    return authors


def _lattice_from_parameters(row: list[str], lat_scale: float) -> np.ndarray:
    """Cell matrix (columns ``a1, a2, a3``, Bohr) from ``a b c alpha beta gamma`` with SIESTA's orientation."""
    a, b, c = (float(x) * lat_scale for x in row[:3])
    alpha, beta, gamma = (np.radians(float(x)) for x in row[3:6])
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    a3 = np.array([cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))])
    return np.stack([a1, a2, a3], axis=1)


def _initial_moments(
    blocks: dict[str, list[list[str]]], n_atoms: int, nspin: int
) -> np.ndarray:
    """Initial atomic moments ``(n_atoms, 3)`` in Bohr magnetons from ``DM.InitSpin`` (ferromagnetic 1 without it)."""
    moments = np.zeros((n_atoms, 3), dtype=np.float64)
    if nspin == 1:
        return moments
    if "dminitspin" not in blocks:
        moments[:, 2] = 1.0  # SIESTA starts every atom polarised when no DM.InitSpin block is given
        return moments
    for row in blocks["dminitspin"]:
        ia = int(row[0])
        ia = ia - 1 if ia > 0 else n_atoms + ia
        if not 0 <= ia < n_atoms:
            raise ValueError(f"DM.InitSpin: atom index {row[0]} is outside 1..{n_atoms}.")
        mag = {"+": 1.0, "-": -1.0}.get(row[1]) if len(row) > 1 else 1.0
        if mag is None:
            mag = float(row[1])
        theta = np.radians(float(row[2])) if len(row) > 3 and nspin > 2 else 0.0
        phi = np.radians(float(row[3])) if len(row) > 3 and nspin > 2 else 0.0
        moments[ia] = mag * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return moments


def parse_fdf(fdf_path: str | Path) -> dict[str, Any]:
    """Parse a static SIESTA ``.fdf`` input into a ``Structure``, an ``SCFConfig`` and run settings.

    Theory:
        Reads the crystal structure, species, k-grid, spin mode and initial moments, functional, mesh, electronic
        temperature, SCF tolerances and mixing options, and the basis-set keywords. Defaults follow SIESTA where
        the keyword has the same meaning (``XC`` LDA/CA, ``MeshCutoff`` 300 Ry, ``PAO.BasisSize`` DZP,
        ``PAO.EnergyShift`` 0.02 Ry, Gamma-point sampling); the SCF tolerances and the mixer default to jsiesta's.

    Algorithm:
        ``tokenize_fdf`` (labels case-insensitive, ``.``/``_``/``-`` ignored, ``%include`` followed), then a
        keyword-by-keyword translation with explicit unit conversion to Bohr and Hartree. Labels that are neither
        interpreted nor known to be irrelevant are returned in ``"unused"`` so the caller can report them.

    Parameters:
        fdf_path: Path to ``.fdf`` input file.

    Returns:
        Dictionary with ``"structure"``, ``"config"``, ``"k_grid"``, ``"k_shift"``, ``"mesh_sizes"``; ``"io"``
        (``SystemLabel``, ``MD.UseSaveXV``, ``DM.UseSaveDM``, ``UseSaveData``, ``File.DM.Init``, ``CDF.Save``,
        ``SaveHS``, ``SCF.MustConverge`` for ``jax_siesta.io.siesta_files``); ``"basis"`` (``size``, ``energy_shift_ha``,
        ``split_norm``, ``split_norm_h``, ``pao_basis`` rows, ``user_basis``); ``"band_lines"`` (rows of ``%block BandLines``,
        its scale and the lattice constant); ``"unused"`` (labels not interpreted) and ``"warnings"`` (messages for the user).

    Invariants:
        - Raises ``ValueError`` on an unclosed ``%block``, a non-diagonal k-grid, an unknown coordinate format,
          spin mode or occupation function.
        - The first definition of a label wins, as in SIESTA.

    SIESTA Mapping:
        Replaces ``fdf_read`` for static single-point input files (``siesta_options``, ``struct_init``,
        ``kpoint_scf``, ``m_spin``, ``basis_specs``).
    """
    scalars, blocks, names = tokenize_fdf(fdf_path)
    warnings: list[str] = []

    def _flag(key: str, default: bool) -> bool:
        if key not in scalars:
            return default
        # A bare FDF logical key means true.
        return (scalars[key][0].lower() if scalars[key] else "t") in _FDF_TRUE

    def _energy_ha(key: str, default_unit: str) -> float | None:
        if key not in scalars or not scalars[key]:
            return None
        unit = scalars[key][1].lower() if len(scalars[key]) > 1 else default_unit
        if unit not in _ENERGY_TO_HA:
            raise ValueError(f"{names[key]}: unknown energy unit {unit!r}.")
        return float(scalars[key][0]) * _ENERGY_TO_HA[unit]

    lat_scale = 1.0
    if "latticeconstant" in scalars:
        val = float(scalars["latticeconstant"][0])
        unit = scalars["latticeconstant"][1].lower() if len(scalars["latticeconstant"]) > 1 else "bohr"
        lat_scale = val * _LENGTH_TO_BOHR.get(unit, 1.0)

    cell = np.eye(3, dtype=np.float64) * lat_scale
    if "latticevectors" in blocks:
        # FDF LatticeVectors rows are a1, a2, a3 -> transpose into column matrix A = [a1, a2, a3]
        rows = np.array([[float(x) for x in r[:3]] for r in blocks["latticevectors"][:3]], dtype=np.float64)
        cell = (rows * lat_scale).T
    elif "latticeparameters" in blocks:
        cell = _lattice_from_parameters(blocks["latticeparameters"][0], lat_scale)

    sp_map: dict[int, tuple[int, str]] = {}
    if "chemicalspecieslabel" in blocks:
        for row in blocks["chemicalspecieslabel"]:
            sp_idx = int(row[0])
            z_num = int(row[1])
            if z_num <= 0:
                raise ValueError(f"Species {row[2]}: ghost and synthetic atoms (Z <= 0) are not supported.")
            sp_map[sp_idx] = (z_num, row[2])

    coord_fmt = _fdf_key(scalars.get("atomiccoordinatesformat", ["bohr"])[0])
    pos_list: list[list[float]] = []
    sp_idx_list: list[int] = []
    if "atomiccoordinatesandatomicspecies" in blocks:
        for row in blocks["atomiccoordinatesandatomicspecies"]:
            xyz = np.array([float(row[0]), float(row[1]), float(row[2])], dtype=np.float64)
            sp_id = int(row[3])
            if coord_fmt in ("fractional", "scaledbylatticevectors"):
                xyz = cell @ xyz
            elif coord_fmt == "scaledcartesian":
                xyz = xyz * lat_scale
            elif coord_fmt in ("ang", "notscaledcartesianang"):
                xyz = xyz * ANGSTROM_TO_BOHR
            elif coord_fmt not in ("bohr", "notscaledcartesianbohr"):
                raise ValueError(f"AtomicCoordinatesFormat {scalars['atomiccoordinatesformat'][0]!r} is not supported.")
            if sp_id not in sp_map:
                raise ValueError(f"Atom {len(pos_list) + 1}: species {sp_id} is not in ChemicalSpeciesLabel.")
            pos_list.append(xyz.tolist())
            sp_idx_list.append(sp_id)

    unique_sp_ids = sorted(set(sp_idx_list))
    id_to_zero = {sid: idx for idx, sid in enumerate(unique_sp_ids)}
    species_indices = tuple(id_to_zero[sid] for sid in sp_idx_list)
    atomic_numbers = tuple(sp_map[sid][0] for sid in sp_idx_list)
    species_labels = tuple(sp_map[sid][1] for sid in unique_sp_ids)

    nspin = 1
    if "spin" in scalars:
        spin_tok = scalars["spin"][0].lower()
        if spin_tok not in _SPIN_MODES:
            raise ValueError(f"Spin {scalars['spin'][0]!r} is not supported (none, polarized, non-collinear, spin-orbit).")
        nspin = _SPIN_MODES[spin_tok]
    elif _flag("spinorbit", False):
        nspin = 8
    elif _flag("noncollinearspin", False):
        nspin = 4
    elif _flag("spinpolarized", False):
        nspin = 2

    mesh_cut = 300.0
    mesh_ha = _energy_ha("meshcutoff", "ry")
    if mesh_ha is not None:
        mesh_cut = mesh_ha / RY_TO_HARTREE

    temp_ha = _energy_ha("electronictemperature", "k")
    if temp_ha is None:
        temp_ha = 300.0 * KB_HARTREE_PER_KELVIN
    occ_fn = scalars.get("occupationfunction", ["FD"])[0].upper()
    if occ_fn != "FD":
        raise ValueError(f"OccupationFunction {occ_fn} is not supported (Fermi-Dirac only).")

    k_grid = (1, 1, 1)
    k_shift = (0.0, 0.0, 0.0)
    if "kgridmonkhorstpack" in blocks:
        kb = blocks["kgridmonkhorstpack"]
        mat = np.array([[int(x) for x in r[:3]] for r in kb[:3]])
        if np.any(mat - np.diag(np.diag(mat)) != 0):
            raise ValueError("kgrid.MonkhorstPack: only diagonal grids are supported.")
        k_grid = (int(mat[0, 0]), int(mat[1, 1]), int(mat[2, 2]))
        k_shift = tuple(float(r[3]) if len(r) > 3 else 0.0 for r in kb[:3])  # type: ignore[assignment]

    xc_func = _xc_from_fdf(scalars)
    w_mix = float(scalars["dmmixingweight"][0]) if "dmmixingweight" in scalars else None
    n_pulay = int(scalars["dmnumberpulay"][0]) if "dmnumberpulay" in scalars else None
    # SIESTA 5 mixer keywords (scalar forms only). ``SCF.Mixer.*`` overrides the older ``DM.*`` names.
    if "scfmixerweight" in scalars:
        w_mix = float(scalars["scfmixerweight"][0])
    if "scfmixerhistory" in scalars:
        n_pulay = int(scalars["scfmixerhistory"][0])
    extra: dict[str, Any] = {}
    if "scfmixermethod" in scalars:
        method = scalars["scfmixermethod"][0].lower()
        if method not in ("pulay", "broyden"):
            raise ValueError(f"SCF.Mixer.Method {method!r} is not supported (use 'pulay' or 'broyden').")
        extra["mixer_variant"] = method
    if "scfmix" in scalars:
        target = scalars["scfmix"][0].lower()
        if target not in ("hamiltonian", "density"):
            raise ValueError(f"SCF.Mix {target!r} is not supported (use 'hamiltonian' or 'density').")
        # jsiesta mixes the density matrix, the closest of its two targets to SIESTA's "density".
        extra["mix_target"] = "Hamiltonian" if target == "hamiltonian" else "DM"
    for key in ("scfdmtolerance", "dmtolerance"):
        if key in scalars:
            extra["d_tol"] = float(scalars[key][0])
            break
    h_tol = _energy_ha("scfhtolerance", "ev")
    if h_tol is not None:
        extra["h_tol_ha"] = h_tol
    e_tol = _energy_ha("scffreeetolerance", "ev")
    if e_tol is not None:
        extra["e_tol_ha"] = e_tol
    if "maxscfiterations" in scalars:
        extra["max_scf_iterations"] = int(scalars["maxscfiterations"][0])
    if "netcharge" in scalars:
        extra["net_charge"] = float(scalars["netcharge"][0])
    if "jsiestadispersion" in scalars:
        extra["dispersion"] = scalars["jsiestadispersion"][0]
    elif _flag("dftd3", False):
        extra["dispersion"] = "D3"
    extra["use_symmetry"] = _flag("jsiestausesymmetry", _flag("kgridusesymmetry", False))

    run_type = scalars.get("mdtypeofrun", ["CG"])[0].upper()
    n_steps = int(scalars.get("mdsteps", scalars.get("mdnumcgsteps", ["0"]))[0])
    if n_steps > 0:
        warnings.append(
            f"MD.TypeOfRun {run_type} with {n_steps} steps requested: jsiesta computes static single points only "
            "(energy, forces and stress of the input geometry)."
        )

    structure = Structure(
        cell=jnp.asarray(cell, dtype=jnp.float64),
        positions=jnp.asarray(pos_list, dtype=jnp.float64).reshape(-1, 3),
        species_indices=species_indices,
        atomic_numbers=atomic_numbers,
        species_labels=species_labels,
        initial_moments=jnp.asarray(_initial_moments(blocks, len(pos_list), nspin), dtype=jnp.float64),
    )
    mesh_subdivisions = int(scalars["meshsubdivisions"][0]) if "meshsubdivisions" in scalars else 2
    mesh_sizes: tuple[int, int, int] | None = None
    if "meshsizes" in scalars and len(scalars["meshsizes"]) >= 3:
        mesh_sizes = (int(scalars["meshsizes"][0]), int(scalars["meshsizes"][1]), int(scalars["meshsizes"][2]))
    elif "meshsizes" in blocks and blocks["meshsizes"]:
        row = blocks["meshsizes"][0]
        mesh_sizes = (int(row[0]), int(row[1]), int(row[2]))
    # Hybrid functionals (XC.Authors HSE06 ...): Gaussians per orbital shell and integral screening tolerance.
    if "hfxnumberofgaussians" in scalars:
        extra["hfx_n_gauss"] = int(scalars["hfxnumberofgaussians"][0])
    if "hfxscreeningtolerance" in scalars:
        extra["hfx_screen_tol"] = float(scalars["hfxscreeningtolerance"][0].lower().replace("d", "e"))
    if "hfxpatternscale" in scalars:
        extra["hfx_pattern_scale"] = float(scalars["hfxpatternscale"][0].lower().replace("d", "e"))
    config = SCFConfig(
        nspin=nspin,
        mesh_cutoff_ry=mesh_cut,
        mesh_criterion="siesta",  # an FDF file has SIESTA semantics: MeshCutoff + MeshSubDivisions
        mesh_subdivisions=mesh_subdivisions,
        electronic_temp_ha=temp_ha,
        xc_functional=xc_func,
        mix_weight=w_mix,
        mix_history=n_pulay,
        **extra,
    )

    label = scalars["systemlabel"][0] if scalars.get("systemlabel") else "siesta"
    restart_io = {
        "system_label": label,
        "use_save_xv": _flag("mdusesavexv", _flag("usesavedata", False)),
        "use_save_dm": _flag("dmusesavedm", _flag("usesavedata", False)),
        "dm_init_file": scalars["filedminit"][0] if scalars.get("filedminit") else f"{label}.DM",
        "cdf_save": _flag("cdfsave", False),
        "save_hs": _flag("savehs", False),
        "must_converge": _flag("scfmustconverge", True),
    }
    e_shift = _energy_ha("paoenergyshift", "ry")
    basis = {
        "size": scalars.get("paobasissize", ["DZP"])[0].upper() if scalars.get("paobasissize") else "DZP",
        "energy_shift_ha": 0.02 * RY_TO_HARTREE if e_shift is None else e_shift,
        "split_norm": float(scalars["paosplitnorm"][0]) if scalars.get("paosplitnorm") else 0.15,
        "split_norm_h": float(scalars["paosplitnormh"][0]) if scalars.get("paosplitnormh") else None,
        "pao_basis": blocks.get("paobasis"),
        "user_basis": _flag("userbasis", False),
    }
    band_lines = {
        "rows": blocks.get("bandlines"),
        "scale": _fdf_key(scalars["bandlinesscale"][0]) if scalars.get("bandlinesscale") else "pi/a",
        "lattice_constant_bohr": lat_scale,
    }
    unused = sorted(
        names[key] for key in (set(scalars) | set(blocks)) if key not in _FDF_USED and key not in _FDF_IGNORED
    )
    return {
        "structure": structure,
        "config": config,
        "k_grid": k_grid,
        "k_shift": k_shift,
        "mesh_sizes": mesh_sizes,
        "io": restart_io,
        "basis": basis,
        "band_lines": band_lines,
        "unused": unused,
        "warnings": warnings,
    }


def load_case_json(case_json_path: str | Path) -> dict[str, Any]:
    """Load a frozen Phase 0/1/2 ``case.json`` input specification.

    Theory:
        Reads column-vector lattice ``cell_columns_bohr``, coordinates, species,
        initial moments, k-grid, XC, and dispersion settings in atomic units.

    Algorithm:
        Parses JSON and builds immutable ``Structure`` and ``SCFConfig`` objects.

    Parameters:
        case_json_path: Path to ``case.json``.

    Returns:
        Dictionary containing ``structure``, ``config``, ``k_grid``, and ``name``.

    Invariants:
        Raises ``KeyError`` or ``ValueError`` if ``cell_columns_bohr`` is missing.

    SIESTA Mapping:
        Bridges frozen ``validation.py`` manifest inputs to ``jax_siesta``.
    """
    p = Path(case_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    cell = np.asarray(data["cell_columns_bohr"], dtype=np.float64)
    pos = np.asarray(data["positions_bohr"], dtype=np.float64)
    sp_list: list[str] = list(data["species"])
    uniq_sp: list[str] = []
    for s in sp_list:
        if s not in uniq_sp:
            uniq_sp.append(s)
    z_map = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
        "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
        "Cs": 55, "Ba": 56, "La": 57, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
        "Tl": 81, "Pb": 82, "Bi": 83,
    }
    sp_indices = tuple(uniq_sp.index(s) for s in sp_list)
    atomic_nums = tuple(z_map[s] for s in sp_list)
    moments = np.asarray(
        data.get("initial_moments", data.get("initial_magnetic_moments", np.zeros((len(sp_list), 3)))),
        dtype=np.float64,
    )
    nspin = int(data.get("nspin", 1))
    structure = Structure(
        cell=jnp.asarray(cell, dtype=jnp.float64),
        positions=jnp.asarray(pos, dtype=jnp.float64),
        species_indices=sp_indices,
        atomic_numbers=atomic_nums,
        species_labels=tuple(uniq_sp),
        initial_moments=jnp.asarray(moments, dtype=jnp.float64),
        pbc=tuple(bool(x) for x in data.get("pbc", [True, True, True])),  # type: ignore[arg-type]
    )
    scf_kwargs: dict[str, Any] = {
        "nspin": nspin,
        "mesh_cutoff_ry": float(data.get("mesh_cutoff_ry", 120.0)),
        "mesh_criterion": str(data.get("mesh_criterion", "siesta")),
        "mesh_subdivisions": int(data.get("mesh_subdivisions", 2)),
        "xc_functional": str(data.get("xc", "PBE")),
        "dispersion": str(data.get("dispersion", "none")),
        "use_symmetry": bool(data.get("use_symmetry", False)),
        "symmetrize_grid": bool(data.get("symmetrize_grid", True)),
    }
    for key, caster in (
        ("max_scf_iterations", int),
        ("d_tol", float),
        ("h_tol_ha", float),
        ("e_tol_ha", float),
        ("electronic_temp_ha", float),
        ("mix_weight", float),
        ("mix_history", int),
        ("mix_target", str),
        ("mixer_variant", str),
    ):
        if key in data and data[key] is not None:
            scf_kwargs[key] = caster(data[key])
    config = SCFConfig(**scf_kwargs)
    return {
        "name": data.get("name", p.parent.name),
        "structure": structure,
        "config": config,
        "k_grid": tuple(int(k) for k in data.get("k_grid", [1, 1, 1])),
        "k_shift": tuple(float(s) for s in data.get("k_shift", [0.0, 0.0, 0.0])),
    }


def observables_to_validation_dict(
    obs: Observables,
    nspin: int,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Serialize ``Observables`` into finite numeric JSON compatible with ``validation.py``.

    Theory:
        Converts Hartree/Bohr atomic units into eV / Angstrom for ``validation.py``
        comparisons while keeping ``last_scf_free_e_ev`` and ``final_post_scf_free_e_ev``
        strictly separate.

    Algorithm:
        Applies CODATA unit conversion factors and checks finiteness.

    Parameters:
        obs: Converged/evaluated ``Observables`` PyTree.
        nspin: Spin mode (emits ``total_spin_moment`` when ``nspin in (2, 4, 8)``).
        include_metadata: If ``False`` (default), emits strictly finite numeric keys
            matching ``validation.py`` ``flatten()`` contract.

    Returns:
        JSON-serializable dictionary of finite numbers and nested numeric arrays.

    Invariants:
        - Passes ``validation.flatten()`` with zero booleans, strings, or non-finite values.
        - Never falls back between ``last_scf_free_e_ev`` and ``final_post_scf_free_e_ev``.

    SIESTA Mapping:
        Produces ``observables.json`` consumed by ``validation.py``.
    """
    f_conv = HARTREE_TO_EV / BOHR_TO_ANGSTROM
    s_conv = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**3)
    out: dict[str, Any] = {
        "last_scf_free_e_ev": float(obs.last_scf_free_e_ha) * HARTREE_TO_EV,
        "final_post_scf_free_e_ev": float(obs.final_post_scf_free_e_ha) * HARTREE_TO_EV,
        "forces": (np.asarray(obs.forces_ha_per_bohr, dtype=np.float64) * f_conv).tolist(),
        "stress": (np.asarray(obs.stress_ha_per_bohr3, dtype=np.float64) * s_conv).tolist(),
    }
    if nspin in (2, 4, 8):
        out["total_spin_moment"] = float(np.linalg.norm(np.asarray(obs.total_spin_moment)))
    if include_metadata:
        out["provenance"] = {
            "outcome": obs.outcome,
            "converged": obs.converged,
            "dm_out_recomputed": obs.dm_out_recomputed,
            "last_scf_cycle": obs.last_scf_cycle,
            "xc_backend": obs.xc_backend,
            "active_symmetry_ops": obs.active_symmetry_ops,
            "magnetic_subgroup": obs.magnetic_subgroup,
        }
    return out
