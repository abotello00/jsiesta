r"""Native crystal and Shubnikov magnetic space-group symmetry package.

Theory:
    Provides space/magnetic group detection (``extract_space_group``), IBZ k-point
    reduction (``build_ibz_kpoint_mesh``), symmetry-commensurate grid mapping
    (``choose_commensurate_mesh_dimensions``, ``precompute_grid_permutations``),
    symmetry-closed sparse patterns and Wigner-D/SU(2) ``segment_sum`` projection
    (``compile_projection_table``), and the unified ``NativeSymmetryEngine``.

Algorithm:
    Organizes the 4-step symmetry architecture across ``spacegroup``, ``projection``,
    and ``engine`` modules without mutable global state.

Invariants:
    - Exposes ``build_native_symmetry`` consumed by ``jax_siesta.testing.run_case``.

SIESTA Mapping:
    Replaces ``m_symmetry.F90``, ``m_dm_symmetrize.F90``, and ``kpoint_scf.F90``.
"""

from jax_siesta.symmetry.engine import NativeSymmetryEngine, build_native_symmetry
from jax_siesta.symmetry.projection import (
    ProjectionTable,
    apply_projection_table,
    build_symmetry_closed_pattern,
    compile_projection_table,
    compute_wigner_d_matrices,
    su2_spinor_matrix,
)
from jax_siesta.symmetry.spacegroup import (
    SpaceGroupData,
    build_ibz_kpoint_mesh,
    choose_commensurate_mesh_dimensions,
    extract_space_group,
    precompute_grid_permutations,
)

__all__ = [
    "NativeSymmetryEngine",
    "ProjectionTable",
    "SpaceGroupData",
    "apply_projection_table",
    "build_ibz_kpoint_mesh",
    "build_native_symmetry",
    "build_symmetry_closed_pattern",
    "choose_commensurate_mesh_dimensions",
    "compile_projection_table",
    "compute_wigner_d_matrices",
    "extract_space_group",
    "precompute_grid_permutations",
    "su2_spinor_matrix",
]
