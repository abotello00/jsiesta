r"""Text report of a run, laid out like a SIESTA output.

Theory:
    A calculation should state what it did: the species and their basis, the size of the problem, the progress
    of the self-consistent cycle, and the final energies, forces and stress with their units.

Algorithm:
    Pure formatting functions returning lists of lines; ``Calculation.run`` sends them to its ``log`` argument
    and the ``jsiesta`` command prints them. Quantities are converted to SIESTA's output units (eV, Ang, Ry).

Invariants:
    - No function here changes or computes physics; they only format.
    - Lines carry the prefix ``jsiesta:`` where SIESTA writes ``siesta:``. The layout is similar to SIESTA's, not
      identical: tools that parse SIESTA output should read ``<SystemLabel>.json`` or the SIESTA-format files.

SIESTA Mapping:
    The standard output of ``siesta`` (``siesta_init`` header, ``scf:`` lines, ``siesta_analysis`` summary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from jax_siesta.app.basis import describe_species
from jax_siesta.core.types import BOHR_TO_ANGSTROM, HARTREE_TO_EV, KB_HARTREE_PER_KELVIN
from jax_siesta.grid.mesh import siesta_mesh_cutoff_ry

if TYPE_CHECKING:
    from jax_siesta.app.calculation import Calculation, Results

SCF_HEADER = "        iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    dHmax(eV)    Ef(eV)"
_SPIN_NAME = {1: "none", 2: "polarized (collinear)", 4: "non-collinear", 8: "spin-orbit"}
EV_PER_ANG3_TO_KBAR = 1602.1766208


def header_lines(calc: Calculation) -> list[str]:
    """Opening lines of a run: program, input file, ignored keywords, species and basis.

    Theory:
        Records the inputs that define the calculation, and warns about input the program did not use, so that
        a keyword that jsiesta does not implement is never silently dropped.

    Algorithm:
        Formats ``calc.info`` (from ``Calculation.from_fdf``) and ``describe_species`` of every species.

    Parameters:
        calc: The calculation about to run.

    Returns:
        Lines without trailing newlines.

    Invariants:
        Every label of the input file that was not interpreted appears in a ``WARNING`` line.

    SIESTA Mapping:
        The header, ``redata`` and ``<basis_specs>`` sections of a SIESTA output.
    """
    from jax_siesta import __version__

    info = calc.info
    lines = [
        f"jsiesta {__version__}: static Kohn-Sham calculation (JAX + C++ implementation of the SIESTA method)",
        "",
    ]
    if info.get("input_file"):
        lines.append(f"jsiesta: input file   {info['input_file']}")
    lines.append(f"jsiesta: system label {calc.label}")
    for msg in info.get("warnings", []):
        lines.append(f"jsiesta: WARNING: {msg}")
    if info.get("unused"):
        lines.append(
            "jsiesta: WARNING: input labels not used by jsiesta (not implemented or misspelt): "
            + ", ".join(info["unused"])
        )
    lines += ["", "jsiesta: species and basis sets"]
    for lbl, sp in calc.species.items():
        binfo = info.get("basis", {}).get(lbl, {})
        if binfo.get("pseudopotential"):
            lines.append(f"  {lbl}: pseudopotential {binfo['pseudopotential']}")
        if binfo.get("source") not in (None, "generated"):
            lines.append(f"  {lbl}: basis read from {binfo['source']}")
        lines += describe_species(sp)
    return lines


def system_lines(
    calc: Calculation, basis: Any, pattern: Any, kmesh: Any, mesh_shape: tuple[int, int, int], sym_proj: Any
) -> list[str]:
    """Description of the system after set-up: cell, atoms, problem size, sampling and options.

    Theory:
        The cost and accuracy of the run follow from the number of orbitals, of k-points and of mesh points, and
        from the options that control the physics (functional, spin, temperature).

    Algorithm:
        Formats the structure in Angstrom and the sizes of the set-up objects; the mesh cutoff printed is the
        one the chosen mesh actually gives (SIESTA's definition).

    Parameters:
        calc: The calculation.
        basis: ``BasisData`` of the structure.
        pattern: ``SparsePattern``.
        kmesh: ``KPointMesh`` in use (irreducible when symmetry is on).
        mesh_shape: Real-space mesh.
        sym_proj: Symmetry engine or ``None``.

    Returns:
        Lines without trailing newlines.

    Invariants:
        Lengths are printed in Angstrom, the mesh cutoff in Ry.

    SIESTA Mapping:
        ``outcell``, ``outcoor``, ``siesta: k-grid``, ``InitMesh`` lines of a SIESTA output.
    """
    st, cfg = calc.structure, calc.config
    cell = np.asarray(st.cell, dtype=np.float64).T * BOHR_TO_ANGSTROM
    pos = np.asarray(st.positions, dtype=np.float64) * BOHR_TO_ANGSTROM
    lines = ["", "jsiesta: unit cell vectors (Ang):"]
    lines += [f"    {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}" for v in cell]
    lines.append(f"jsiesta: cell volume = {abs(np.linalg.det(cell)):.4f} Ang**3")
    lines.append("jsiesta: atomic coordinates (Ang) and species:")
    for ia in range(st.n_atoms):
        lines.append(
            f"    {pos[ia, 0]:12.6f} {pos[ia, 1]:12.6f} {pos[ia, 2]:12.6f}  {st.species_labels[st.species_indices[ia]]:<6s}{ia + 1:6d}"
        )
    n_el = float(basis.total_valence_electrons) - cfg.net_charge
    lines += [
        "",
        f"jsiesta: atoms, orbitals, projectors: {st.n_atoms:6d} {basis.n_orb:6d} {basis.n_kb:6d}",
        f"jsiesta: electrons                  : {n_el:.4f}",
        f"jsiesta: non-zero matrix elements   : {pattern.n_edges}",
        f"jsiesta: functional                 : {cfg.xc_functional}"
        + (f" + {cfg.dispersion}" if cfg.dispersion.lower() != "none" else ""),
        f"jsiesta: spin                       : {_SPIN_NAME[cfg.nspin]}",
        f"jsiesta: electronic temperature     : {cfg.electronic_temp_ha / KB_HARTREE_PER_KELVIN:.1f} K",
        f"jsiesta: k-grid                     : {calc.k_grid[0]} x {calc.k_grid[1]} x {calc.k_grid[2]}, "
        f"{kmesh.n_k} k-points" + (" (irreducible)" if getattr(kmesh, "is_ibz", False) else ""),
        f"jsiesta: mesh                       : {mesh_shape[0]} x {mesh_shape[1]} x {mesh_shape[2]}, "
        f"cutoff {siesta_mesh_cutoff_ry(st.cell, mesh_shape):.2f} Ry (requested {cfg.mesh_cutoff_ry:.2f} Ry)",
    ]
    if sym_proj is not None:
        lines.append(f"jsiesta: symmetry                   : {sym_proj.n_ops} operations ({sym_proj.subgroup_label})")
    lines.append(
        f"jsiesta: SCF                        : mix {cfg.mix_target}, {cfg.mixer_variant}, tolerances dDmax "
        f"{cfg.d_tol:.1e}, dHmax {cfg.h_tol_ha * HARTREE_TO_EV:.1e} eV, max {cfg.max_scf_iterations} iterations"
    )
    return lines


def scf_line(rec: dict[str, float]) -> str:
    """One line of the SCF table from a ``run_scf`` callback record.

    Theory:
        The Harris and Kohn-Sham energies approach each other and the changes of the density matrix and of the
        Hamiltonian go to zero as the cycle converges.

    Algorithm:
        Converts Hartree to eV and formats in the column order of ``SCF_HEADER``.

    Parameters:
        rec: Record with ``iteration``, ``harris_energy_ha``, ``free_energy_ha``, ``d_max``, ``h_max_ha``,
            ``fermi_level_ha``.

    Returns:
        The formatted line.

    Invariants:
        During the cycle jsiesta tracks the free energy; the E_KS column shows the same value until the final
        evaluation separates the two.

    SIESTA Mapping:
        The ``scf:`` lines of a SIESTA output.
    """
    ev = HARTREE_TO_EV
    return (
        f"   scf: {int(rec['iteration']):4d} {rec['harris_energy_ha'] * ev:15.6f} {rec['free_energy_ha'] * ev:15.6f} "
        f"{rec['free_energy_ha'] * ev:15.6f} {rec['d_max']:9.6f} {rec['h_max_ha'] * ev:11.6f} "
        f"{rec['fermi_level_ha'] * ev:10.6f}"
    )


def final_lines(results: Results) -> list[str]:
    """Final section: convergence, energy terms, forces, stress, pressure, spin moments and timings.

    Theory:
        The energy is evaluated once more at the output density matrix of the last iteration; forces and stress
        are its derivatives at fixed density and energy-density matrices.

    Algorithm:
        Formats ``Results`` in eV, eV/Ang and eV/Ang**3; the pressure also in kBar, as SIESTA prints it.

    Parameters:
        results: Outcome of ``Calculation.run``.

    Returns:
        Lines without trailing newlines.

    Invariants:
        A run that did not converge is flagged in the first lines of the section.

    SIESTA Mapping:
        ``siesta: Final energy (eV)``, ``siesta: Atomic forces``, ``siesta: Stress tensor``, ``siesta: Pressure``.
    """
    e = results.energies_ev
    forces = results.forces_ev_ang
    stress = results.stress_ev_ang3
    lines = [""]
    if results.converged:
        lines.append(f"SCF cycle converged after {results.n_iterations} iterations")
    else:
        lines.append(
            f"jsiesta: WARNING: SCF did NOT converge in {results.n_iterations} iterations "
            f"({results.observables.outcome}); the results below are not self-consistent"
        )
    lines += [
        "",
        "jsiesta: Final energy (eV):",
        f"jsiesta:       Kinetic = {e['kinetic']:18.6f}",
        f"jsiesta:  Neutral atom = {e['neutral_atom']:18.6f}",
        f"jsiesta:     Non-local = {e['nonlocal']:18.6f}",
        f"jsiesta:       Hartree = {e['hartree']:18.6f}",
        f"jsiesta:    Exch.-corr = {e['xc']:18.6f}",
    ]
    if results.config.dispersion.lower() != "none":
        lines.append(f"jsiesta:    Dispersion = {e['dispersion']:18.6f}")
    lines += [
        f"jsiesta:   -T*S (el.)  = {e['entropy_term']:18.6f}",
        f"jsiesta:        Harris = {e['harris']:18.6f}",
        f"jsiesta:         Total = {e['total']:18.6f}",
        f"jsiesta:       FreeEng = {e['free']:18.6f}",
        f"jsiesta:         Fermi = {results.fermi_level_ev:18.6f}",
        "",
        "jsiesta: Atomic forces (eV/Ang):",
    ]
    lines += [f"jsiesta: {ia + 1:6d} {f[0]:12.6f} {f[1]:12.6f} {f[2]:12.6f}" for ia, f in enumerate(forces)]
    lines.append("jsiesta: " + "-" * 46)
    tot = forces.sum(axis=0)
    lines.append(f"jsiesta:    Tot {tot[0]:12.6f} {tot[1]:12.6f} {tot[2]:12.6f}")
    lines.append(f"jsiesta:    Max {np.abs(forces).max():12.6f}")
    lines += ["", "jsiesta: Stress tensor (static) (eV/Ang**3):"]
    lines += [f"jsiesta: {row[0]:14.6f} {row[1]:14.6f} {row[2]:14.6f}" for row in stress]
    p_ev = -np.trace(stress) / 3.0
    lines += [
        "",
        f"jsiesta: Pressure (static): {p_ev * EV_PER_ANG3_TO_KBAR:14.6f} kBar  = {results.pressure_gpa:.6f} GPa",
    ]
    if results.config.nspin > 1:
        m = results.spin_moment
        lines += [
            "",
            f"jsiesta: Total spin polarization (Bohr magnetons): |S| = {np.linalg.norm(m):.6f}, "
            f"S = ({m[0]:.6f}, {m[1]:.6f}, {m[2]:.6f})",
            "jsiesta: Atomic spin moments (Mulliken, Bohr magnetons):",
        ]
        lines += [
            f"jsiesta: {ia + 1:6d} {a[0]:12.6f} {a[1]:12.6f} {a[2]:12.6f}"
            for ia, a in enumerate(results.atomic_spin_moments)
        ]
    lines += ["", "jsiesta: Elapsed time (s):"]
    lines += [f"jsiesta:   {name:<20s} {sec:10.2f}" for name, sec in results.timings.items()]
    return lines


__all__ = ["SCF_HEADER", "final_lines", "header_lines", "scf_line", "system_lines"]
