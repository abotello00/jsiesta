r"""FDF input parser subset, SIESTA restart files (``.XV``, ``.DM``, ``.nc``) and ``validation.py`` adapters.

Theory:
    Translates standard SIESTA ``.fdf`` input specifications and ``case.json`` files
    into immutable ``Structure`` / ``SCFConfig`` objects and writes schema-v1
    ``observables.json`` outputs with explicit unit declarations.

Algorithm:
    Pure Python parsers with explicit unit conversion to internal atomic units
    (Bohr, Hartree) and structured JSON export compatible with ``validation.py``.
    ``siesta_files`` reads and writes SIESTA's ``.XV`` geometry, ``.DM`` density-matrix and ``.nc`` sparse-matrix
    files, converting sparsity layout, spin components and units (Ry) to and from the jsiesta ones.

Invariants:
    - Every exported observable carries explicit atomic or eV/Angstrom units.
    - Both ``last_scf_free_e`` and ``final_post_scf_free_e`` are emitted separately.

SIESTA Mapping:
    Replaces ``fdf`` input reading and ``io`` output formatting for static runs; ``ioxv``, ``m_iodm`` and
    ``m_ncdf_siesta`` for the restart files.
"""

from jax_siesta.io.fdf import load_case_json, observables_to_validation_dict, parse_fdf
from jax_siesta.io.siesta_files import (
    MappingInfo,
    SiestaNCData,
    SiestaSparsity,
    XVData,
    convert_spin,
    dm_from_siesta,
    dm_to_siesta,
    h_from_siesta,
    h_to_siesta,
    map_siesta_to_edges,
    normalize_dm,
    pattern_to_siesta,
    read_dm,
    read_dm_file,
    read_nc_restart,
    read_siesta_nc,
    read_xv,
    siesta_supercell_offsets,
    structure_from_nc,
    structure_from_xv,
    write_dm,
    write_dm_file,
    write_restart_files,
    write_siesta_nc,
    write_xv,
)

__all__ = [
    "MappingInfo",
    "SiestaNCData",
    "SiestaSparsity",
    "XVData",
    "convert_spin",
    "dm_from_siesta",
    "dm_to_siesta",
    "h_from_siesta",
    "h_to_siesta",
    "load_case_json",
    "map_siesta_to_edges",
    "normalize_dm",
    "observables_to_validation_dict",
    "parse_fdf",
    "pattern_to_siesta",
    "read_dm",
    "read_dm_file",
    "read_nc_restart",
    "read_siesta_nc",
    "read_xv",
    "siesta_supercell_offsets",
    "structure_from_nc",
    "structure_from_xv",
    "write_dm",
    "write_dm_file",
    "write_restart_files",
    "write_siesta_nc",
    "write_xv",
]
