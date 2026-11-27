r"""Output files of a run in SIESTA's formats: forces, eigenvalues, bands, restart files and a JSON summary.

Theory:
    SIESTA leaves its results in small text files named after ``SystemLabel``, which post-processing tools read.
    They are ``.FA`` (forces), ``.EIG`` (eigenvalues at the SCF k-points), ``.bands`` (eigenvalues along
    ``BandLines``), and the restart files ``.XV`` and ``.DM``. Writing the same files lets those tools and SIESTA
    itself continue from a jsiesta run.

Algorithm:
    Fortran record layouts reproduced with Python formatting (``iofa``, ``ioeig``, ``bands``); the restart files
    are written by ``jax_siesta.io.write_restart_files``.

Invariants:
    - Units are those of SIESTA (eV, eV/Ang, Bohr^-1 for the band path).
    - The layouts follow SIESTA 5 source; ``.XV`` and ``.DM`` are checked byte for byte against SIESTA-written
      files by the test suite, the text files written here have not been compared with a SIESTA run.

SIESTA Mapping:
    ``iofa.f``, ``ioeig.f``, ``bands.F`` (output part), ``ioxv.F``, ``m_iodm.F90``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jax_siesta.app.calculation import BandStructure, Results
from jax_siesta.io.siesta_files import write_restart_files


def _wrapped(values: np.ndarray, first: str, pad: str, fmt: str, per_line: int = 10) -> str:
    """Fortran-style record: ``first`` + up to ``per_line`` values, continuation lines indented by ``pad``."""
    chunks = [values[i : i + per_line] for i in range(0, len(values), per_line)] or [values]
    return "\n".join(
        (first if i == 0 else pad) + "".join(fmt.format(v) for v in chunk) for i, chunk in enumerate(chunks)
    )


def write_forces(path: str | Path, results: Results) -> None:
    """Write ``<SystemLabel>.FA``: number of atoms, then index and force (eV/Ang) of each atom.

    Theory:
        Forces at the converged density matrix, the quantity relaxation and phonon tools consume.

    Algorithm:
        SIESTA's formats ``(i6)`` and ``(i6,3f12.6)``.

    Parameters:
        path: File name.
        results: Outcome of a run.

    Invariants:
        One line per atom after the header.

    SIESTA Mapping:
        ``iofa.f``.
    """
    forces = results.forces_ev_ang
    lines = [f"{len(forces):6d}"] + [f"{i + 1:6d}{f[0]:12.6f}{f[1]:12.6f}{f[2]:12.6f}" for i, f in enumerate(forces)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eigenvalues(path: str | Path, results: Results) -> None:
    """Write ``<SystemLabel>.EIG``: Fermi level and the eigenvalues (eV) at every SCF k-point.

    Theory:
        The spectrum on the SCF mesh, from which densities of states are built.

    Algorithm:
        ``Results.eigenvalues_ev`` in the record layout of ``ioeig``: Fermi level, then ``n_bands nspin n_k``,
        then per k-point its index and the eigenvalues (spin up followed by spin down for collinear spin).

    Parameters:
        path: File name.
        results: Outcome of a run.

    Invariants:
        The k-points are those of ``results.kmesh`` (the irreducible ones when symmetry is on), in that order.

    SIESTA Mapping:
        ``ioeig.f``.
    """
    eig = results.eigenvalues_ev()  # (n_ch, n_k, n_bands)
    n_ch, n_k, n_bands = eig.shape
    nspin = results.config.nspin
    lines = [f"{results.fermi_level_ev:17.9e}", f" {n_bands:10d} {min(nspin, 4)} {n_k:10d}"]
    for ik in range(n_k):
        vals = eig[:, ik, :].reshape(-1)
        lines.append(_wrapped(vals, f"{ik + 1:10d}", " " * 10, " {:17.9e}"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def band_lines_path(
    rows: list[list[str]], scale: str, lattice_constant_bohr: float, reciprocal_cell: Any
) -> tuple[list[tuple[str, tuple[float, float, float]]], list[int]]:
    """Convert the rows of ``%block BandLines`` into a path for ``Results.band_structure``.

    Theory:
        Each row is ``n k1 k2 k3 [label]``: the end point of a line with ``n`` k-points from the previous point.
        ``BandLinesScale ReciprocalLatticeVectors`` gives the points in units of the reciprocal lattice vectors;
        the default ``pi/a`` gives Cartesian coordinates in units of pi over the lattice constant.

    Algorithm:
        For ``pi/a`` the points are converted to fractional coordinates with the inverse reciprocal cell.

    Parameters:
        rows: Token rows of the block.
        scale: Normalised ``BandLinesScale`` (``"pi/a"`` or ``"reciprocallatticevectors"``).
        lattice_constant_bohr: ``LatticeConstant`` in Bohr.
        reciprocal_cell: ``Structure.reciprocal_cell`` (columns are the reciprocal vectors, 1/Bohr).

    Returns:
        ``(path, points_per_segment)`` as taken by ``Results.band_structure``.

    Invariants:
        Raises ``ValueError`` for another scale.

    SIESTA Mapping:
        The input part of ``bands.F`` (``initbands``).
    """
    if scale not in ("pi/a", "reciprocallatticevectors"):
        raise ValueError(f"BandLinesScale {scale!r} is not supported (pi/a or ReciprocalLatticeVectors).")
    b = np.asarray(reciprocal_cell, dtype=np.float64)
    path: list[tuple[str, tuple[float, float, float]]] = []
    counts: list[int] = []
    for row in rows:
        k = np.array([float(row[1]), float(row[2]), float(row[3])])
        if scale == "pi/a":
            k = np.linalg.solve(b, k * np.pi / lattice_constant_bohr)
        label = row[4] if len(row) > 4 else " "
        if path:
            counts.append(int(row[0]))
        path.append((label, (float(k[0]), float(k[1]), float(k[2]))))
    return path, counts


def write_bands(path: str | Path, bands: BandStructure) -> None:
    """Write a band structure as a SIESTA ``<SystemLabel>.bands`` file.

    Theory:
        Band energies in eV against the path length in Bohr^-1, with the positions and labels of the line ends.

    Algorithm:
        Record layout of ``bands.F``: Fermi level; path range; energy range; ``n_bands n_spin n_k``; per k-point
        the path length and the eigenvalues, ``(f10.6,10f12.4,/,(10x,10f12.4))``; number of lines; one
        ``(f12.6,3x,a)`` record per line end with the label in quotes.

    Parameters:
        path: File name.
        bands: Result of ``Results.band_structure``.

    Invariants:
        Non-collinear and spin-orbit runs are written as one spin channel with twice the number of bands.

    SIESTA Mapping:
        The output part of ``bands.F``; the file read by ``gnubands`` and ``sisl``.
    """
    e = bands.energies_ev
    n_ch, n_k, n_bands = e.shape
    lines = [
        f" {bands.fermi_level_ev:22.14f}",
        f" {0.0:22.14f} {bands.distances[-1]:22.14f}",
        f" {e.min():22.14f} {e.max():22.14f}",
        f" {n_bands:11d} {n_ch:11d} {n_k:11d}",
    ]
    for ik in range(n_k):
        lines.append(_wrapped(e[:, ik, :].reshape(-1), f"{bands.distances[ik]:10.6f}", " " * 10, "{:12.4f}"))
    lines.append(f" {len(bands.ticks):11d}")
    lines += [f"{x:12.6f}   '{lbl}'" for x, lbl in bands.ticks]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    results: Results,
    directory: str | Path = ".",
    restart_formats: tuple[str, ...] = ("XV", "DM"),
    band_lines: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the standard set of output files of a run, named ``<label>.<ext>``.

    Theory:
        The files a SIESTA user expects after a single point: results for post-processing and the state for
        restarts.

    Algorithm:
        ``.FA``, ``.EIG`` and ``.json`` always; ``.XV`` / ``.DM`` / ``.nc`` according to ``restart_formats``;
        ``.bands`` when ``band_lines`` (from ``parse_fdf``) has rows.

    Parameters:
        results: Outcome of a run.
        directory: Output directory (created if missing).
        restart_formats: Subset of ``("XV", "DM", "nc")``; ``"nc"`` needs the ``netCDF4`` package.
        band_lines: The ``"band_lines"`` entry of ``parse_fdf``.

    Returns:
        Mapping from extension to the path written.

    Invariants:
        Existing files with the same names are overwritten, as SIESTA does.

    SIESTA Mapping:
        ``siesta_analysis`` and ``write_subs`` (``siesta_write_forces``), ``ioeig``, ``bands``, ``iodm``, ``ioxv``.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    label = results.label
    written: dict[str, Path] = {}
    if results.observables.forces_available:
        write_forces(out / f"{label}.FA", results)
        written["FA"] = out / f"{label}.FA"
    write_eigenvalues(out / f"{label}.EIG", results)
    written["EIG"] = out / f"{label}.EIG"
    if restart_formats:
        obs = results.observables
        written.update(
            write_restart_files(
                out, label, results.structure, results.basis, results.pattern, results.overlap, results.scf_state,
                results.config, forces_ha_per_bohr=obs.forces_ha_per_bohr,
                stress_ha_per_bohr3=obs.stress_ha_per_bohr3, k_grid=results.k_grid, formats=restart_formats,
            )
        )
    if band_lines and band_lines.get("rows"):
        path, counts = band_lines_path(
            band_lines["rows"], band_lines["scale"], band_lines["lattice_constant_bohr"],
            results.structure.reciprocal_cell,
        )
        write_bands(out / f"{label}.bands", results.band_structure(path, counts))
        written["bands"] = out / f"{label}.bands"
    (out / f"{label}.json").write_text(json.dumps(results.to_dict(), indent=2) + "\n", encoding="utf-8")
    written["json"] = out / f"{label}.json"
    return written


__all__ = ["band_lines_path", "write_bands", "write_eigenvalues", "write_forces", "write_outputs"]
