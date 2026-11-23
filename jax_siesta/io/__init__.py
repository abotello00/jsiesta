r"""FDF input parser subset, ``.ion`` table I/O, and ``validation.py`` adapters.

Theory:
    Translates standard SIESTA ``.fdf`` input specifications and ``case.json`` files
    into immutable ``Structure`` / ``SCFConfig`` objects and writes schema-v1
    ``observables.json`` outputs with explicit unit declarations.

Algorithm:
    Pure Python parsers with explicit unit conversion to internal atomic units
    (Bohr, Hartree) and structured JSON export compatible with ``validation.py``.

Invariants:
    - Every exported observable carries explicit atomic or eV/Angstrom units.
    - Both ``last_scf_free_e`` and ``final_post_scf_free_e`` are emitted separately.

SIESTA Mapping:
    Replaces ``fdf`` input reading and ``io`` output formatting for static runs.
"""

from jax_siesta.io.fdf import load_case_json, observables_to_validation_dict, parse_fdf

__all__ = ["load_case_json", "observables_to_validation_dict", "parse_fdf"]
