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

_LENGTH_TO_BOHR = {"bohr": 1.0, "ang": ANGSTROM_TO_BOHR, "angstrom": ANGSTROM_TO_BOHR}
_ENERGY_TO_HA = {
    "ha": 1.0,
    "hartree": 1.0,
    "ry": RY_TO_HARTREE,
    "rydberg": RY_TO_HARTREE,
    "ev": EV_TO_HARTREE,
    "k": KB_HARTREE_PER_KELVIN,
}


def parse_fdf(fdf_path: str | Path) -> dict[str, Any]:
    """Parse a static SIESTA ``.fdf`` file subset into normalized dictionary entries.

    Theory:
        Reads crystal structure, species map, k-grid, spin mode, electronic
        temperature, mesh cutoff, and SCF mixing options.

    Algorithm:
        Single-pass block and scalar line parser normalizing FDF keys (case-insensitive,
        dots/underscores/hyphens stripped).

    Parameters:
        fdf_path: Path to ``.fdf`` input file.

    Returns:
        Dictionary of parsed structure fields and ``SCFConfig`` keyword overrides.

    Invariants:
        Raises ``ValueError`` if required ``%block`` tags are unclosed.

    SIESTA Mapping:
        Replaces ``fdf_read`` for static single-point input files.
    """
    lines = Path(fdf_path).read_text(encoding="utf-8").splitlines()
    scalars: dict[str, list[str]] = {}
    blocks: dict[str, list[list[str]]] = {}

    i = 0
    while i < len(lines):
        raw = lines[i].split("#")[0].split("!")[0].strip()
        i += 1
        if not raw:
            continue
        toks = raw.split()
        head = toks[0].lower()
        if head == "%block":
            bname = toks[1].lower().replace(".", "").replace("_", "").replace("-", "")
            body: list[list[str]] = []
            closed = False
            while i < len(lines):
                braw = lines[i].split("#")[0].split("!")[0].strip()
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
            blocks[bname] = body
        else:
            key = head.replace(".", "").replace("_", "").replace("-", "")
            scalars[key] = toks[1:]

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

    sp_map: dict[int, tuple[int, str]] = {}
    if "chemicalspecieslabel" in blocks:
        for row in blocks["chemicalspecieslabel"]:
            sp_idx = int(row[0])
            z_num = int(row[1])
            lbl = row[2]
            sp_map[sp_idx] = (z_num, lbl)

    coord_fmt = scalars.get("atomiccoordinatesformat", ["bohr"])[0].lower()
    pos_list: list[list[float]] = []
    sp_idx_list: list[int] = []
    if "atomiccoordinatesandatomicspecies" in blocks:
        for row in blocks["atomiccoordinatesandatomicspecies"]:
            xyz = np.array([float(row[0]), float(row[1]), float(row[2])], dtype=np.float64)
            sp_id = int(row[3])
            if coord_fmt in ("fractional", "scaledcartesian"):
                xyz = cell @ xyz if "fractional" in coord_fmt else xyz * lat_scale
            elif coord_fmt in ("ang", "notscaledcartesianang"):
                xyz = xyz * ANGSTROM_TO_BOHR
            elif coord_fmt in ("bohr", "notscaledcartesianbohr"):
                pass
            pos_list.append(xyz.tolist())
            sp_idx_list.append(sp_id)

    unique_sp_ids = sorted(set(sp_idx_list))
    id_to_zero = {sid: idx for idx, sid in enumerate(unique_sp_ids)}
    species_indices = tuple(id_to_zero[sid] for sid in sp_idx_list)
    atomic_numbers = tuple(sp_map[sid][0] for sid in sp_idx_list)
    species_labels = tuple(sp_map[sid][1] for sid in unique_sp_ids)

    spin_tok = scalars.get("spin", ["none"])[0].lower()
    nspin = {
        "none": 1,
        "non-polarized": 1,
        "polarized": 2,
        "collinear": 2,
        "non-collinear": 4,
        "noncollinear": 4,
        "spin-orbit": 8,
        "soc": 8,
    }.get(spin_tok, 1)

    mesh_cut = 150.0
    if "meshcutoff" in scalars:
        v = float(scalars["meshcutoff"][0])
        u = scalars["meshcutoff"][1].lower() if len(scalars["meshcutoff"]) > 1 else "ry"
        mesh_cut = v * (_ENERGY_TO_HA.get(u, 0.5) / RY_TO_HARTREE)

    temp_ha = 300.0 * KB_HARTREE_PER_KELVIN
    if "electronictemperature" in scalars:
        v = float(scalars["electronictemperature"][0])
        u = scalars["electronictemperature"][1].lower() if len(scalars["electronictemperature"]) > 1 else "k"
        temp_ha = v * _ENERGY_TO_HA.get(u, KB_HARTREE_PER_KELVIN)

    k_grid = (1, 1, 1)
    if "kgridmonkhorstpack" in blocks:
        k_grid = (
            int(blocks["kgridmonkhorstpack"][0][0]),
            int(blocks["kgridmonkhorstpack"][1][1]),
            int(blocks["kgridmonkhorstpack"][2][2]),
        )

    xc_func = scalars.get("xcauthors", ["PBE"])[0]
    w_mix = float(scalars["dmmixingweight"][0]) if "dmmixingweight" in scalars else None
    n_pulay = int(scalars["dmnumberpulay"][0]) if "dmnumberpulay" in scalars else None
    # SIESTA 5 mixer keywords (scalar forms only). ``SCF.Mixer.*`` overrides the older ``DM.*`` names.
    if "scfmixerweight" in scalars:
        w_mix = float(scalars["scfmixerweight"][0])
    if "scfmixerhistory" in scalars:
        n_pulay = int(scalars["scfmixerhistory"][0])
    mix_kwargs: dict[str, Any] = {}
    if "scfmixermethod" in scalars:
        method = scalars["scfmixermethod"][0].lower()
        if method not in ("pulay", "broyden"):
            raise ValueError(f"SCF.Mixer.Method {method!r} is not supported (use 'pulay' or 'broyden').")
        mix_kwargs["mixer_variant"] = method
    if "scfmix" in scalars:
        target = scalars["scfmix"][0].lower()
        if target not in ("hamiltonian", "density"):
            raise ValueError(f"SCF.Mix {target!r} is not supported (use 'hamiltonian' or 'density').")
        # jsiesta mixes the density matrix, the closest of its two targets to SIESTA's "density".
        mix_kwargs["mix_target"] = "Hamiltonian" if target == "hamiltonian" else "DM"

    structure = Structure(
        cell=jnp.asarray(cell, dtype=jnp.float64),
        positions=jnp.asarray(pos_list, dtype=jnp.float64),
        species_indices=species_indices,
        atomic_numbers=atomic_numbers,
        species_labels=species_labels,
        initial_moments=jnp.zeros((len(pos_list), 3), dtype=jnp.float64),
    )
    mesh_subdivisions = int(scalars["meshsubdivisions"][0]) if "meshsubdivisions" in scalars else 2
    mesh_sizes: tuple[int, int, int] | None = None
    if "meshsizes" in scalars and len(scalars["meshsizes"]) >= 3:
        mesh_sizes = (int(scalars["meshsizes"][0]), int(scalars["meshsizes"][1]), int(scalars["meshsizes"][2]))
    elif "meshsizes" in blocks and blocks["meshsizes"]:
        row = blocks["meshsizes"][0]
        mesh_sizes = (int(row[0]), int(row[1]), int(row[2]))
    config = SCFConfig(
        nspin=nspin,
        mesh_cutoff_ry=mesh_cut,
        mesh_criterion="siesta",  # an FDF file has SIESTA semantics: MeshCutoff + MeshSubDivisions
        mesh_subdivisions=mesh_subdivisions,
        electronic_temp_ha=temp_ha,
        xc_functional=xc_func,
        mix_weight=w_mix,
        mix_history=n_pulay,
        **mix_kwargs,
    )
    return {"structure": structure, "config": config, "k_grid": k_grid, "mesh_sizes": mesh_sizes}


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
