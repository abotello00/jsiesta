#!/usr/bin/env python3
r"""Single-point jsiesta driver and ``validation.py`` case/contract runner.

Theory:
    Executes the full static Kohn-Sham pipeline (PSML loading, neighbor pattern,
    two-center integrals, SCF cycle, post-SCF ``DM_out`` pass, and fixed-(D,E)
    variational forces/stress) for one ``case.json`` input and writes
    ``observables.json`` in the isolated working directory.

Algorithm:
    Supports ``--mode baseline|candidate|symmetrized|unsymmetrized|mixer-<variant>``
    plus independent physical-identity checks (acoustic sum rule, electron conservation).

Invariants:
    - Always writes finite numeric JSON matching ``validation.py`` schema v1.
    - Exits non-zero if convergence or physical bounds fail.

SIESTA Mapping:
    Provides the candidate adapter interface for ``validation.py`` and ``docs/design/EVIDENCE.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jax_siesta.core.types import Observables, SCFConfig
from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json, observables_to_validation_dict
from jax_siesta.io.siesta_files import read_dm, read_nc_restart, structure_from_xv, write_restart_files
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern

if getattr(jax, "config", None) is not None:
    jax.config.update("jax_enable_x64", True)


def run_single_point(
    case_path: str | Path,
    psml_dir: str | Path | None = None,
    config_overrides: dict | None = None,
    read_xv_path: str | Path | None = None,
    read_dm_path: str | Path | None = None,
    read_nc_path: str | Path | None = None,
    write_restart_dir: str | Path | None = None,
    system_label: str | None = None,
    restart_formats: tuple[str, ...] = ("XV", "DM", "nc"),
) -> Observables:
    r"""Execute an end-to-end static single-point KS-DFT calculation for ``case.json``.

    Theory:
        Runs PSML species loading, sparse pattern construction, C++ two-center
        integrals, adaptive SCF, and mandatory post-SCF ``DM_out`` observables.

    Algorithm:
        Wires together ``PSMLBasisLoader``, ``TwoCenterEvaluator``, ``run_scf``,
        and ``StaticObservableEvaluator`` (activating ``NativeSymmetryEngine`` when
        ``config.use_symmetry`` is ``True``). SIESTA restart files are optional at both ends: an ``.XV``
        file replaces the geometry of the case, a ``.DM`` file and/or a ``.nc`` file (``DM`` and ``H``) warm-start
        the SCF, and ``<label>.XV``, ``<label>.DM`` and ``<label>.nc`` are written from the final state.

    Parameters:
        case_path: Path to ``case.json``.
        psml_dir: Directory containing ``<Element>.psml`` (defaults to ``tests/fixtures/psml``).
        config_overrides: Optional dictionary of ``SCFConfig`` field overrides.
        read_xv_path: SIESTA ``.XV`` file whose cell and positions replace those of the case.
        read_dm_path: SIESTA ``.DM`` file used as the initial density matrix.
        read_nc_path: SIESTA ``.nc`` file; its ``H`` (and its ``DM`` unless ``read_dm_path`` is given) start the SCF.
        write_restart_dir: Directory for the restart files of the final state (not written when ``None``).
        system_label: ``SystemLabel`` of the files written (defaults to the case name).
        restart_formats: Subset of ``("XV", "DM", "nc")`` to write.

    Returns:
        Provenanced ``Observables`` PyTree.

    Invariants:
        - Raises ``RuntimeError`` if SCF fails with ``numerical_breakdown`` or ``invalid_input``.

    SIESTA Mapping:
        Single-point static driver replacing ``siesta.F`` + ``siesta_forces.F``.
    """
    case = load_case_json(case_path)
    structure = case["structure"]
    config: SCFConfig = case["config"]
    if config_overrides:
        config = replace(config, **config_overrides)
    if read_xv_path is not None:
        structure = structure_from_xv(read_xv_path, template=structure)

    if psml_dir is None:
        psml_dir = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "psml"
    psml_dir = Path(psml_dir)

    loader = PSMLBasisLoader()
    species_map = {
        lbl: loader.load_species(lbl, str(psml_dir / f"{lbl}.psml"))
        for lbl in structure.species_labels
    }
    basis = loader.build_basis(
        structure, species_map, include_soc_projectors=(config.nspin == 8)
    )

    k_shift = tuple(case.get("k_shift", (0.0, 0.0, 0.0)))
    sym_proj = None
    if config.use_symmetry:
        from jax_siesta.symmetry.engine import build_native_symmetry

        sym_proj, pattern, kmesh, mesh_shape = build_native_symmetry(
            structure, basis, case["k_grid"], config, k_shifts=k_shift
        )
        two_center = sym_proj.evaluate_irreducible_two_center(structure, basis, pattern, config)
    else:
        pattern = build_sparse_pattern(structure, basis)
        kmesh = build_monkhorst_pack_mesh(case["k_grid"], k_shift)
        mesh_shape = choose_mesh_dimensions(
            structure.cell, config.mesh_cutoff_ry, criterion=config.mesh_criterion, mesh_subdivisions=config.mesh_subdivisions
        )
        two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)

    initial_dm = None
    initial_h = None
    if read_nc_path is not None:
        initial_dm, initial_h, _ = read_nc_restart(read_nc_path, pattern, config.nspin)
    if read_dm_path is not None:
        initial_dm = read_dm(read_dm_path, pattern, config.nspin)

    scf_state, mesh_shape = run_scf(
        structure,
        basis,
        pattern,
        two_center,
        kmesh,
        config,
        mesh_shape=mesh_shape,
        sym_proj=sym_proj,
        initial_dm=initial_dm,
        initial_h=initial_h,
    )
    obs = StaticObservableEvaluator().evaluate(
        structure,
        basis,
        pattern,
        two_center,
        kmesh,
        scf_state,
        config,
        mesh_shape=mesh_shape,
        sym_proj=sym_proj,
    )
    if sym_proj is not None:
        obs = replace(
            obs,
            active_symmetry_ops=sym_proj.n_ops,
            magnetic_subgroup=sym_proj.subgroup_label,
        )
    if write_restart_dir is not None:
        write_restart_files(
            write_restart_dir,
            system_label or str(case["name"]),
            structure,
            basis,
            pattern,
            two_center.overlap,
            scf_state,
            config,
            forces_ha_per_bohr=obs.forces_ha_per_bohr,
            stress_ha_per_bohr3=obs.stress_ha_per_bohr3,
            k_grid=case["k_grid"],
            formats=restart_formats,
        )
    return obs


def main() -> None:
    r"""CLI entrypoint for ``validation.py`` adapters.

    Theory:
        Runs one ``case.json`` single-point calculation and serializes observables into
        SIESTA-compatible units (eV, eV/Ang, eV/Ang^3) for automated parity checks.

    Algorithm:
        Parses CLI mode flags (``unsymmetrized``, ``symmetrized``, ``mixer-*``), invokes
        ``run_single_point``, and writes ``observables.json``.

    Invariants:
        Exits non-zero if SCF fails to converge; never emits NaN or Inf fields.

    SIESTA Mapping:
        Provides the executable adapter expected by ``validation.py``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case_json", type=Path)
    p.add_argument("--mode", default="candidate")
    p.add_argument("--psml-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("observables.json"))
    p.add_argument("--read-xv", type=Path, default=None, help="SIESTA .XV file: start from its cell and positions")
    p.add_argument("--read-dm", type=Path, default=None, help="SIESTA .DM file: warm-start density matrix")
    p.add_argument("--read-nc", type=Path, default=None, help="SIESTA .nc file: warm-start Hamiltonian (and DM)")
    p.add_argument("--write-restart", type=Path, default=None, help="directory for <label>.XV, .DM and .nc")
    p.add_argument("--label", default=None, help="SystemLabel of the restart files (default: case name)")
    args = p.parse_args()

    overrides: dict = {}
    if args.mode == "unsymmetrized":
        overrides["use_symmetry"] = False
    elif args.mode == "symmetrized":
        overrides["use_symmetry"] = True
    elif args.mode.startswith("mixer-"):
        overrides["mixer_variant"] = args.mode.removeprefix("mixer-")

    case = load_case_json(args.case_json)
    nspin = overrides.get("nspin", case["config"].nspin)
    obs = run_single_point(
        args.case_json,
        psml_dir=args.psml_dir,
        config_overrides=overrides,
        read_xv_path=args.read_xv,
        read_dm_path=args.read_dm,
        read_nc_path=args.read_nc,
        write_restart_dir=args.write_restart,
        system_label=args.label,
    )
    if not obs.converged:
        raise SystemExit(f"SCF did not converge: outcome={obs.outcome}")

    payload = observables_to_validation_dict(obs, nspin=nspin, include_metadata=False)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
