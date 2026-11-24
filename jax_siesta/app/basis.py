r"""Pseudopotential lookup and basis-set construction for SIESTA-style runs.

Theory:
    A SIESTA run names its species in ``ChemicalSpeciesLabel`` and expects one pseudopotential file per label.
    The basis of confined numerical atomic orbitals is either generated from the pseudopotential
    (``PAO.BasisSize``, ``PAO.EnergyShift``, ``PAO.SplitNorm`` or an explicit ``%block PAO.Basis``) or read from
    the ``.ion.xml`` file of an earlier SIESTA run (``User.Basis``).

Algorithm:
    ``find_pseudopotential`` searches ``<Label>.psml`` in the given directories and in ``SIESTA_PS_PATH``;
    ``parse_pao_basis_block`` turns the rows of ``%block PAO.Basis`` into ``SpeciesBasisSpec`` objects;
    ``build_species`` calls the native generator (``generate_species_from_psml``) or the ``.ion.xml`` reader
    (``PSMLBasisLoader``).

Invariants:
    - Only PSML pseudopotentials are accepted; a ``.psf`` or ``.vps`` file gives an error that says so.
    - A generated basis and its neutral-atom potential always come from the same atomic calculation.

SIESTA Mapping:
    ``basis_specs.f`` (reading of the basis keywords), ``atom.F`` (generation) and ``pseudopotential.f`` (file
    lookup).
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

from jax_siesta.core.types import RY_TO_HARTREE, SpeciesData
from jax_siesta.species.pao_generator import ShellBasisSpec, SpeciesBasisSpec, generate_species_from_psml
from jax_siesta.species.psml import PSMLBasisLoader

_L_LETTER = "spdfg"


def atomic_xc_name(xc_functional: str) -> str:
    """Functional used for the free-atom calculation that generates the basis.

    Theory:
        SIESTA generates the basis with the functional of the run. The radial atomic solver of jsiesta has the
        Perdew-Zunger and Perdew-Wang LDAs and PBE; other functionals (PBEsol, meta-GGAs, Libxc ids) use the PBE
        atom, which changes the basis functions slightly but not the functional of the crystal calculation.

    Algorithm:
        Name lookup.

    Parameters:
        xc_functional: ``SCFConfig.xc_functional``.

    Returns:
        ``"LDA_PZ"``, ``"LDA_PW92"`` or ``"PBE"``.

    Invariants:
        Never raises; unknown names map to ``"PBE"``.

    SIESTA Mapping:
        The functional passed to ``atom.F`` by ``basis_specs``.
    """
    key = xc_functional.strip().upper().replace("-", "_")
    if key in ("LDA", "LDA_PZ", "PZ", "CA"):
        return "LDA_PZ"
    if key in ("LDA_PW92", "PW92"):
        return "LDA_PW92"
    return "PBE"


def find_pseudopotential(label: str, search_dirs: list[Path] | tuple[Path, ...]) -> Path:
    """Locate ``<label>.psml`` for a species.

    Theory:
        SIESTA looks for the pseudopotential of each species label in the working directory and then in the
        directories of the ``SIESTA_PS_PATH`` environment variable.

    Algorithm:
        First match of ``<label>.psml`` over ``search_dirs`` followed by ``SIESTA_PS_PATH`` (colon-separated).

    Parameters:
        label: Species label of ``ChemicalSpeciesLabel``.
        search_dirs: Directories searched first, in order.

    Returns:
        Path of the PSML file.

    Invariants:
        Raises ``FileNotFoundError`` naming every directory searched; when a ``.psf``/``.vps`` file with that
        label exists the message says that only PSML is supported.

    SIESTA Mapping:
        ``pseudo_read`` in ``pseudopotential.f``.
    """
    dirs = [Path(d) for d in search_dirs]
    dirs += [Path(d) for d in os.environ.get("SIESTA_PS_PATH", "").split(os.pathsep) if d]
    for d in dirs:
        cand = d / f"{label}.psml"
        if cand.is_file():
            return cand
    legacy = [str(d / f"{label}{ext}") for d in dirs for ext in (".psf", ".vps") if (d / f"{label}{ext}").is_file()]
    hint = (
        f" Found {', '.join(legacy)}: jsiesta reads PSML files only (PseudoDojo distributes them; SIESTA's psml "
        "tools convert other formats)."
        if legacy
        else ""
    )
    raise FileNotFoundError(
        f"No pseudopotential {label}.psml in {', '.join(str(d) for d in dirs) or 'any directory'}." + hint
    )


def parse_pao_basis_block(
    rows: list[list[str]],
    xc: str = "PBE",
    energy_shift_ha: float = 0.02 * RY_TO_HARTREE,
    split_norm: float = 0.15,
) -> dict[str, SpeciesBasisSpec]:
    """Translate the rows of ``%block PAO.Basis`` into basis specifications.

    Theory:
        Each species entry is ``Label n_shells [type] [ionic_charge]`` followed, per shell, by a header
        ``n=<n> <l> <n_zeta> [P <n_pol>] [E <V0 Ry> <ri Bohr>] [S <split_norm>]``, a line with one cut-off radius
        per zeta (Bohr; all zeros mean: from ``PAO.EnergyShift`` and the split norm) and an optional line of
        scale factors. ``P`` adds perturbative polarisation orbitals of angular momentum ``l + 1``; ``E`` is the
        soft-confinement potential of Junquera et al.

    Algorithm:
        Sequential read of the token rows. A negative ``ri`` is SIESTA's fraction of the first cut-off radius.

    Parameters:
        rows: Token rows of the block (from ``tokenize_fdf`` / ``parse_fdf``).
        xc: Functional of the atomic calculation.
        energy_shift_ha: ``PAO.EnergyShift`` in Hartree for shells with automatic radii.
        split_norm: ``PAO.SplitNorm`` for shells with automatic inner radii.

    Returns:
        Mapping from species label to ``SpeciesBasisSpec``.

    Invariants:
        Raises ``ValueError`` for what the generator cannot do: shells without ``n=``, charge confinement
        (``Q``), filtering (``F``), scale factors other than 1, and radii that are only partly given.

    SIESTA Mapping:
        ``basis_specs.f`` (``repaobasis``).
    """
    specs: dict[str, SpeciesBasisSpec] = {}
    i = 0
    while i < len(rows):
        head = rows[i]
        i += 1
        label, n_shells = head[0], int(head[1])
        charge = 0.0
        for tok in head[2:]:
            try:
                charge = float(tok)
            except ValueError:
                continue
        shells: list[ShellBasisSpec] = []
        for _ in range(n_shells):
            if i >= len(rows):
                raise ValueError(f"PAO.Basis: species {label} declares {n_shells} shells but the block ends early.")
            sh = rows[i]
            i += 1
            if not sh[0].lower().startswith("n="):
                raise ValueError(f"PAO.Basis, species {label}: write every shell as 'n=<n> <l> <n_zeta>' (got {' '.join(sh)!r}).")
            n_q, l_q, n_zeta = int(sh[0][2:]), int(sh[1]), int(sh[2])
            n_pol, v0_ry, ri, shell_split = 0, 0.0, 0.0, None
            j = 3
            while j < len(sh):
                opt = sh[j].upper()
                if opt == "P":
                    has_count = j + 1 < len(sh) and sh[j + 1].isdigit()
                    n_pol = int(sh[j + 1]) if has_count else 1
                    j += 2 if has_count else 1
                elif opt == "E":
                    v0_ry, ri = float(sh[j + 1]), float(sh[j + 2])
                    j += 3
                elif opt == "S":
                    shell_split = float(sh[j + 1])
                    j += 2
                else:
                    raise ValueError(f"PAO.Basis, species {label}: option {sh[j]!r} is not supported (use P, E, S).")
            rcs = tuple(float(x) for x in rows[i][:n_zeta])
            i += 1
            if len(rcs) != n_zeta:
                raise ValueError(f"PAO.Basis, species {label}, n={n_q} l={l_q}: expected {n_zeta} cut-off radii.")
            if i < len(rows) and not rows[i][0].lower().startswith("n=") and _all_floats(rows[i]):
                if any(abs(float(x) - 1.0) > 1e-12 for x in rows[i]):
                    raise ValueError(f"PAO.Basis, species {label}: scale factors other than 1 are not supported.")
                i += 1
            if all(rc == 0.0 for rc in rcs):
                rc_list = None
            elif any(rc <= 0.0 for rc in rcs):
                raise ValueError(
                    f"PAO.Basis, species {label}, n={n_q} l={l_q}: give all cut-off radii or none (all zeros)."
                )
            else:
                rc_list = rcs
            if ri < 0.0:
                if rc_list is None:
                    raise ValueError(f"PAO.Basis, species {label}: a relative ri needs explicit cut-off radii.")
                ri = -ri * rc_list[0]
            shells.append(
                ShellBasisSpec(
                    n=n_q, l=l_q, n_zeta=n_zeta, rc_list=rc_list, v0_ha=v0_ry * RY_TO_HARTREE, ri_bohr=ri,
                    split_norm=shell_split,
                )
            )
            if n_pol > 0:
                shells.append(
                    ShellBasisSpec(
                        n=n_q, l=l_q + 1, n_zeta=n_pol, is_polarization=True, polarization_mode="perturbative",
                        parent_l=l_q,
                    )
                )
        specs[label] = SpeciesBasisSpec(
            symbol=label, shells=tuple(shells), ionic_charge=charge, xc=xc, energy_shift_ha=energy_shift_ha,
            split_norm=split_norm,
        )
    return specs


def _all_floats(tokens: list[str]) -> bool:
    """True when every token parses as a number (a scale-factor line of ``PAO.Basis``)."""
    try:
        [float(t) for t in tokens]
    except ValueError:
        return False
    return True


def build_species(
    label: str,
    psml_path: str | Path,
    basis: str | SpeciesBasisSpec = "DZP",
    xc: str = "PBE",
    energy_shift_ha: float = 0.02 * RY_TO_HARTREE,
    split_norm: float = 0.15,
    ion_path: str | Path | None = None,
) -> tuple[SpeciesData, dict[str, Any]]:
    """Build the pseudopotential and basis tables of one species.

    Theory:
        The species data hold the Kleinman-Bylander projectors, the neutral-atom potential and the confined
        orbitals. With ``ion_path`` the orbitals are those of a SIESTA run (same basis in both codes, the route
        used for the parity tests); otherwise they are generated from the pseudopotential by jsiesta's own atomic
        solver, which follows SIESTA's recipe (energy shift, split norm, perturbative polarisation) but is a
        separate implementation: radii and energies are close to SIESTA's, not identical.

    Algorithm:
        ``PSMLBasisLoader.load_species`` with an ``.ion.xml`` file, else ``generate_species_from_psml``.

    Parameters:
        label: Species label (also the key of the species in the calculation).
        psml_path: PSML pseudopotential file.
        basis: Basis size (``"SZ"``, ``"DZP"``, ``"TZ2P"``...) or an explicit ``SpeciesBasisSpec``.
        xc: Functional of the atomic calculation (see ``atomic_xc_name``).
        energy_shift_ha: ``PAO.EnergyShift`` in Hartree (SIESTA's default 0.02 Ry).
        split_norm: ``PAO.SplitNorm``.
        ion_path: SIESTA ``<Label>.ion.xml`` file with the basis to use instead of generating one.

    Returns:
        ``(species, info)``: ``info`` has ``"source"`` (``"generated"`` or the ion file) and, for a generated
        basis, the generator's ``"shells"`` (radii and confined eigenvalues).

    Invariants:
        ``species.symbol == label``.

    SIESTA Mapping:
        ``atom.F`` (generation) or ``read_ion`` (``User.Basis``).
    """
    if ion_path is not None:
        species = PSMLBasisLoader().load_species(label, str(psml_path), ion_path=str(ion_path))
        return replace(species, symbol=label), {"source": str(ion_path)}
    species, diag = generate_species_from_psml(
        psml_path, basis, xc=atomic_xc_name(xc), energy_shift_ha=energy_shift_ha, split_norm=split_norm
    )
    return replace(species, symbol=label), {"source": "generated", "shells": diag["shells"]}


def describe_species(species: SpeciesData) -> list[str]:
    """Text lines describing the basis and projectors of a species (label, charge, shells with cut-off radii).

    Theory:
        The quality of a numerical-orbital calculation is set by the number of radial functions per angular
        momentum and by their cut-off radii, so a run should show them.

    Algorithm:
        Formats ``pao_n``, ``pao_l``, ``pao_zeta`` and ``pao_rcut`` and counts the projectors per ``l``.

    Parameters:
        species: Species tables.

    Returns:
        Lines without trailing newlines.

    Invariants:
        One line per radial orbital shell.

    SIESTA Mapping:
        The ``<basis_specs>`` and ``atom:`` sections of a SIESTA output.
    """
    n_orb = sum(2 * l_q + 1 for l_q in species.pao_l)
    lines = [
        f"  {species.symbol}: Z = {species.atomic_number}, valence charge = {species.valence_charge:.2f}, "
        f"{len(species.pao_l)} radial functions, {n_orb} orbitals per atom"
        + (", core correction" if species.has_nlcc else "")
    ]
    for n_q, l_q, zeta, rc in zip(species.pao_n, species.pao_l, species.pao_zeta, species.pao_rcut, strict=True):
        lines.append(f"      n={n_q} {_L_LETTER[l_q]}  zeta {zeta}   rc = {rc:8.4f} Bohr")
    kb = ", ".join(f"{species.kb_l.count(l_q)} {_L_LETTER[l_q]}" for l_q in sorted(set(species.kb_l)))
    lines.append(f"      Kleinman-Bylander projectors: {kb}")
    return lines


__all__ = ["atomic_xc_name", "build_species", "describe_species", "find_pseudopotential", "parse_pao_basis_block"]
