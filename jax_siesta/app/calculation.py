r"""High-level single-point calculation: one object in, energies, forces and stress out.

Theory:
    A static Kohn-Sham calculation is a fixed sequence: species tables (pseudopotential and basis), the sparsity
    pattern of the orbital pairs, the two-centre integrals, the self-consistent cycle, and one final evaluation
    of the energy, the forces and the stress at the output density matrix. ``Calculation`` holds the inputs of
    that sequence and ``Results`` its outputs, in the units SIESTA prints (eV, Angstrom).

Algorithm:
    ``Calculation.run`` chains ``PSMLBasisLoader.build_basis``, ``build_sparse_pattern`` (or
    ``build_native_symmetry``), ``TwoCenterEvaluator``, ``run_scf`` and ``StaticObservableEvaluator``; these are
    the same calls a user can make one by one for finer control. ``Calculation.from_fdf`` fills the inputs from a
    SIESTA input file.

Invariants:
    - Inputs are never modified: ``with_structure`` / ``with_config`` return new calculations.
    - ``Results`` always carries the post-SCF energy evaluated at the output density matrix.

SIESTA Mapping:
    The top-level flow of ``siesta.F`` for a single geometry (``siesta_init``, ``siesta_forces``,
    ``siesta_analysis``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_siesta.app.basis import build_species, find_pseudopotential, parse_pao_basis_block
from jax_siesta.core.types import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    BasisData,
    KPointMesh,
    Observables,
    SCFConfig,
    SCFState,
    SparsePattern,
    SpeciesData,
    Structure,
)
from jax_siesta.grid.mesh import choose_mesh_dimensions
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io.fdf import parse_fdf
from jax_siesta.io.siesta_files import read_dm, structure_from_xv
from jax_siesta.observables.bands import band_energies
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species.pao_generator import SpeciesBasisSpec
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern

EV_PER_ANG3_TO_GPA: float = 160.21766208
_Z_OF = {
    s: z
    for z, s in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y "
        "Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
        "Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split(),
        start=1,
    )
}


def make_structure(
    cell: Any,
    symbols: list[str] | tuple[str, ...],
    positions: Any,
    fractional: bool = False,
    units: str = "Ang",
    initial_moments: Any | None = None,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> Structure:
    """Build a ``Structure`` from lattice vectors, element symbols and positions.

    Theory:
        jsiesta stores the lattice vectors as the columns of the cell matrix, in Bohr; structure files and
        SIESTA inputs give them as rows, usually in Angstrom. This helper takes the common form and converts.

    Algorithm:
        Unit conversion, optional fractional-to-Cartesian conversion, and species indices in order of first
        appearance of each symbol.

    Parameters:
        cell: Lattice vectors as rows ``(3, 3)``.
        symbols: Element symbol of each atom; a suffix after ``_`` distinguishes species of one element
            (``"Fe_up"``), and each distinct symbol needs its own pseudopotential file.
        positions: Atomic positions ``(n_atoms, 3)``.
        fractional: Positions are fractions of the lattice vectors.
        units: ``"Ang"`` or ``"Bohr"`` for ``cell`` and Cartesian ``positions``.
        initial_moments: Starting magnetic moments: ``(n_atoms,)`` along z or ``(n_atoms, 3)`` vectors, in Bohr
            magnetons. Only used by spin-polarised runs.
        pbc: Periodicity along each lattice vector.

    Returns:
        Immutable ``Structure`` in atomic units.

    Invariants:
        Raises ``ValueError`` for an unknown unit or element, or a cell with zero volume.

    SIESTA Mapping:
        ``LatticeVectors``, ``ChemicalSpeciesLabel`` and ``AtomicCoordinatesAndAtomicSpecies``.
    """
    if units.lower() not in ("ang", "angstrom", "bohr"):
        raise ValueError(f"units must be 'Ang' or 'Bohr', not {units!r}.")
    scale = 1.0 if units.lower() == "bohr" else ANGSTROM_TO_BOHR
    cell_cols = np.asarray(cell, dtype=np.float64).reshape(3, 3).T * scale
    if abs(np.linalg.det(cell_cols)) < 1e-12:
        raise ValueError("The lattice vectors are linearly dependent (zero cell volume).")
    pos_in = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    pos = (cell_cols @ pos_in.T).T if fractional else pos_in * scale
    labels: list[str] = []
    for sym in symbols:
        if sym not in labels:
            labels.append(sym)
    try:
        numbers = tuple(_Z_OF[sym.split("_")[0]] for sym in symbols)
    except KeyError as exc:
        raise ValueError(f"Unknown element symbol {exc.args[0]!r}.") from None
    moments = np.zeros((len(symbols), 3), dtype=np.float64)
    if initial_moments is not None:
        m = np.asarray(initial_moments, dtype=np.float64)
        if m.ndim == 1:
            moments[:, 2] = m
        else:
            moments[:] = m
    return Structure(
        cell=jnp.asarray(cell_cols),
        positions=jnp.asarray(pos),
        species_indices=tuple(labels.index(sym) for sym in symbols),
        atomic_numbers=numbers,
        species_labels=tuple(labels),
        initial_moments=jnp.asarray(moments),
        pbc=pbc,
    )


def _species_atomic_numbers(structure: Structure) -> dict[str, int]:
    """Atomic number of each species label of a structure."""
    return {
        structure.species_labels[sp]: int(z)
        for sp, z in zip(structure.species_indices, structure.atomic_numbers, strict=True)
    }


@dataclass(frozen=True)
class BandStructure:
    """Band energies along a path in the Brillouin zone.

    Theory:
        Eigenvalues of the converged Hamiltonian at k-points along straight segments between named points,
        plotted against the path length.

    Algorithm:
        Produced by ``Results.band_structure``; plain NumPy arrays.

    Invariants:
        - ``energies_ev`` has shape ``(n_channels, n_k, n_bands)``; ``distances`` has length ``n_k``.
        - Energies are absolute (not shifted to the Fermi level).

    SIESTA Mapping:
        The contents of ``<SystemLabel>.bands``.
    """

    kpoints_frac: np.ndarray
    kpoints_cart: np.ndarray
    distances: np.ndarray
    energies_ev: np.ndarray
    fermi_level_ev: float
    ticks: tuple[tuple[float, str], ...]
    n_electrons: float
    nspin: int

    @property
    def n_occupied(self) -> int:
        """Number of fully occupied bands per channel at zero temperature (insulator filling)."""
        per_band = 2.0 if self.nspin == 1 else 1.0
        return int(round(self.n_electrons / per_band / (2 if self.nspin == 2 else 1)))

    @property
    def band_gap_ev(self) -> float:
        """Gap between the highest occupied and lowest empty band along the path (eV); 0 or less for a metal.

        For ``nspin`` 2 the occupation is taken from the Fermi level instead of an electron count per channel.
        """
        e = self.energies_ev
        if self.nspin == 2:
            occ = e[e <= self.fermi_level_ev]
            emp = e[e > self.fermi_level_ev]
            return float(emp.min() - occ.max()) if occ.size and emp.size else 0.0
        n_occ = self.n_occupied
        if n_occ <= 0 or n_occ >= e.shape[2]:
            return 0.0
        return float(e[0, :, n_occ].min() - e[0, :, n_occ - 1].max())

    def plot(self, ax: Any | None = None, shift_to_fermi: bool = True, **line_kwargs: Any) -> Any:
        """Plot the bands with matplotlib.

        Theory:
            Bands are drawn against the path length, with vertical lines at the named points.

        Algorithm:
            One line per band and channel (second spin channel dashed).

        Parameters:
            ax: Matplotlib axes (a new figure when ``None``).
            shift_to_fermi: Subtract the Fermi level, so that zero is the Fermi level.
            **line_kwargs: Passed to ``ax.plot``.

        Returns:
            The axes.

        Invariants:
            Needs ``matplotlib``; raises ``ImportError`` with an install hint otherwise.

        SIESTA Mapping:
            What ``gnubands`` plus a plotting program do with a ``.bands`` file.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("BandStructure.plot needs matplotlib (pip install matplotlib).") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 4.0))
        e0 = self.fermi_level_ev if shift_to_fermi else 0.0
        style: dict[str, Any] = {"color": "C0", "lw": 1.2} | line_kwargs
        for ich in range(self.energies_ev.shape[0]):
            ax.plot(self.distances, self.energies_ev[ich] - e0, ls="-" if ich == 0 else "--", **style)
        for x, _ in self.ticks:
            ax.axvline(x, color="0.6", lw=0.6)
        ax.set_xticks([x for x, _ in self.ticks])
        ax.set_xticklabels([lbl.replace("\\Gamma", "Γ").replace("Gamma", "Γ") for _, lbl in self.ticks])
        ax.set_xlim(self.distances[0], self.distances[-1])
        ax.set_ylabel("E − E$_F$ (eV)" if shift_to_fermi else "E (eV)")
        if shift_to_fermi:
            ax.axhline(0.0, color="0.3", lw=0.6, ls=":")
        return ax


@dataclass(frozen=True)
class Results:
    """Outcome of a single-point calculation, with SIESTA's output units (eV, Angstrom).

    Theory:
        Holds the converged state (Hamiltonian, density matrix), the observables of the final evaluation at the
        output density matrix, and what is needed for post-processing (basis, sparsity pattern, overlap,
        k-points). Energies are in eV, forces in eV/Ang, the stress in eV/Ang^3; the underlying objects keep
        Hartree atomic units.

    Algorithm:
        A frozen container; the unit conversions are properties.

    Invariants:
        - ``energy_ev`` and ``free_energy_ev`` come from the post-SCF evaluation, never from a mixed state.
        - ``scf_history`` has one entry per SCF iteration.

    SIESTA Mapping:
        The final section of a SIESTA output (``siesta: Final energy``, ``Atomic forces``, ``Stress tensor``)
        plus the state SIESTA keeps in ``.DM`` / ``.HSX`` files.
    """

    label: str
    structure: Structure
    basis: BasisData
    pattern: SparsePattern
    kmesh: KPointMesh
    config: SCFConfig
    mesh_shape: tuple[int, int, int]
    overlap: Any
    scf_state: SCFState
    observables: Observables
    k_grid: tuple[int, int, int] = (1, 1, 1)
    scf_history: tuple[dict[str, float], ...] = ()
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Whether the SCF cycle reached its tolerances."""
        return bool(self.observables.converged)

    @property
    def n_iterations(self) -> int:
        """Number of SCF iterations performed."""
        return len(self.scf_history) or int(self.observables.n_iterations)

    @property
    def energy_ev(self) -> float:
        """Kohn-Sham total energy (SIESTA's ``Total``), in eV."""
        return float(self.observables.total_energy_ha) * HARTREE_TO_EV

    @property
    def free_energy_ev(self) -> float:
        """Electronic free energy ``E - T S`` (SIESTA's ``FreeEng``), in eV: the quantity forces derive from."""
        return float(self.observables.final_post_scf_free_e_ha) * HARTREE_TO_EV

    @property
    def fermi_level_ev(self) -> float:
        """Fermi level in eV."""
        return float(self.observables.fermi_level_ha) * HARTREE_TO_EV

    @property
    def forces_ev_ang(self) -> np.ndarray:
        """Atomic forces ``(n_atoms, 3)`` in eV/Ang."""
        return np.asarray(self.observables.forces_ha_per_bohr, dtype=np.float64) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)

    @property
    def stress_ev_ang3(self) -> np.ndarray:
        """Stress tensor ``(3, 3)`` in eV/Ang^3, with SIESTA's sign (negative under compression)."""
        return np.asarray(self.observables.stress_ha_per_bohr3, dtype=np.float64) * (
            HARTREE_TO_EV / BOHR_TO_ANGSTROM**3
        )

    @property
    def pressure_gpa(self) -> float:
        """Hydrostatic pressure ``-trace(stress) / 3`` in GPa (positive under compression)."""
        return float(-np.trace(self.stress_ev_ang3) / 3.0 * EV_PER_ANG3_TO_GPA)

    @property
    def spin_moment(self) -> np.ndarray:
        """Total spin moment vector ``(mx, my, mz)`` in Bohr magnetons (zero for ``nspin`` 1)."""
        return np.asarray(self.observables.total_spin_moment, dtype=np.float64)

    @property
    def atomic_spin_moments(self) -> np.ndarray:
        """Mulliken spin moment of each atom ``(n_atoms, 3)`` in Bohr magnetons."""
        return np.asarray(self.observables.atomic_spin_moments, dtype=np.float64)

    @property
    def energies_ev(self) -> dict[str, float]:
        """Energy terms in eV: total, free, Harris, kinetic, neutral-atom, non-local, Hartree, xc, dispersion, -TS.

        ``neutral_atom`` corresponds to SIESTA's ``Ena + DEna``, ``nonlocal`` to ``Enl``, ``hartree`` to
        ``DUscf`` (the Hartree energy of the density minus the neutral-atom density).
        """
        o = self.observables
        return {
            "total": float(o.total_energy_ha) * HARTREE_TO_EV,
            "free": float(o.final_post_scf_free_e_ha) * HARTREE_TO_EV,
            "harris": float(o.harris_energy_ha) * HARTREE_TO_EV,
            "kinetic": float(o.e_kinetic_ha) * HARTREE_TO_EV,
            "neutral_atom": float(o.e_na_ha) * HARTREE_TO_EV,
            "nonlocal": float(o.e_kb_ha) * HARTREE_TO_EV,
            "hartree": float(o.e_hartree_ha) * HARTREE_TO_EV,
            "xc": float(o.e_xc_ha) * HARTREE_TO_EV,
            "dispersion": float(o.e_disp_ha) * HARTREE_TO_EV,
            "entropy_term": float(o.e_entropy_ha) * HARTREE_TO_EV,
        }

    @property
    def positions_ang(self) -> np.ndarray:
        """Atomic positions ``(n_atoms, 3)`` in Angstrom."""
        return np.asarray(self.structure.positions, dtype=np.float64) * BOHR_TO_ANGSTROM

    @property
    def cell_ang(self) -> np.ndarray:
        """Lattice vectors as rows ``(3, 3)`` in Angstrom."""
        return np.asarray(self.structure.cell, dtype=np.float64).T * BOHR_TO_ANGSTROM

    def eigenvalues_ev(self, kpoints_frac: Any | None = None) -> np.ndarray:
        """Kohn-Sham eigenvalues in eV at the SCF k-points or at given k-points.

        Theory:
            Eigenvalues of the converged Hamiltonian; no further self-consistency is involved.

        Algorithm:
            ``jax_siesta.observables.band_energies`` on ``scf_state.h_sparse``.

        Parameters:
            kpoints_frac: k-points ``(n_k, 3)`` in fractional reciprocal coordinates (default: the k-points of
                the SCF mesh, ``Results.kmesh``).

        Returns:
            Array ``(n_channels, n_k, n_bands)``; two channels for collinear spin, one otherwise.

        Invariants:
            Ascending in the band index.

        SIESTA Mapping:
            ``<SystemLabel>.EIG``.
        """
        kpts = self.kmesh.kpoints_frac if kpoints_frac is None else kpoints_frac
        return band_energies(self.scf_state.h_sparse, self.overlap, self.pattern, kpts, self.config.nspin) * HARTREE_TO_EV

    def band_structure(
        self,
        path: list[tuple[str, tuple[float, float, float]]],
        points_per_segment: int | list[int] = 30,
    ) -> BandStructure:
        """Band energies along straight lines between named k-points.

        Theory:
            The band structure is a non-self-consistent property of the converged Hamiltonian.

        Algorithm:
            Linear interpolation between consecutive points (fractional reciprocal coordinates), the path
            length measured in Cartesian reciprocal space (1/Bohr), and ``eigenvalues_ev`` on all points.

        Parameters:
            path: ``[(label, (k1, k2, k3)), ...]`` in units of the reciprocal lattice vectors.
            points_per_segment: Number of k-points of each segment (one integer, or one per segment).

        Returns:
            ``BandStructure`` with energies in eV.

        Invariants:
            The first point of the path is included once; every later named point ends its segment.

        SIESTA Mapping:
            ``%block BandLines`` with ``BandLinesScale ReciprocalLatticeVectors``.
        """
        n_seg = len(path) - 1
        if n_seg < 1:
            raise ValueError("A band path needs at least two points.")
        counts = [points_per_segment] * n_seg if isinstance(points_per_segment, int) else list(points_per_segment)
        kpts = [np.asarray(path[0][1], dtype=np.float64)]
        tick_idx = [0]
        for (_, k0), (_, k1), n in zip(path[:-1], path[1:], counts, strict=True):
            a, b = np.asarray(k0, dtype=np.float64), np.asarray(k1, dtype=np.float64)
            kpts.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
            tick_idx.append(len(kpts) - 1)
        k_frac = np.asarray(kpts)
        k_cart = k_frac @ np.asarray(self.structure.reciprocal_cell, dtype=np.float64).T
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(k_cart, axis=0), axis=1))])
        return BandStructure(
            kpoints_frac=k_frac,
            kpoints_cart=k_cart,
            distances=dist,
            energies_ev=self.eigenvalues_ev(k_frac),
            fermi_level_ev=self.fermi_level_ev,
            ticks=tuple((float(dist[i]), lbl) for i, (lbl, _) in zip(tick_idx, path, strict=True)),
            n_electrons=float(self.basis.total_valence_electrons - self.config.net_charge),
            nspin=self.config.nspin,
        )

    def to_dict(self) -> dict[str, Any]:
        """Results as plain Python numbers and lists (for JSON), in eV and Angstrom.

        Theory:
            A machine-readable record of the run, complementary to the text output.

        Algorithm:
            Collects the unit-converted properties.

        Returns:
            Dictionary with the label, convergence data, energies, Fermi level, forces, stress, pressure, spin
            moments, geometry, mesh and timings.

        Invariants:
            Only JSON types (no arrays).

        SIESTA Mapping:
            The information of SIESTA's final output section, as written to ``<SystemLabel>.json`` by ``jsiesta``.
        """
        return {
            "label": self.label,
            "converged": self.converged,
            "outcome": str(self.observables.outcome),
            "n_scf_iterations": self.n_iterations,
            "units": {"energy": "eV", "length": "Ang", "force": "eV/Ang", "stress": "eV/Ang^3", "pressure": "GPa"},
            "energies": self.energies_ev,
            "fermi_level": self.fermi_level_ev,
            "forces": self.forces_ev_ang.tolist(),
            "stress": self.stress_ev_ang3.tolist(),
            "pressure": self.pressure_gpa,
            "total_spin_moment": self.spin_moment.tolist(),
            "atomic_spin_moments": self.atomic_spin_moments.tolist(),
            "cell": self.cell_ang.tolist(),
            "positions": self.positions_ang.tolist(),
            "species": [self.structure.species_labels[i] for i in self.structure.species_indices],
            "n_orbitals": int(self.basis.n_orb),
            "n_kpoints": int(self.kmesh.n_k),
            "mesh": list(self.mesh_shape),
            "scf_history": [dict(h) for h in self.scf_history],
            "timings_s": dict(self.timings),
        }


@dataclass(frozen=True)
class Calculation:
    """Inputs of a single-point calculation: geometry, species, settings and k-point sampling.

    Theory:
        Everything a static Kohn-Sham calculation depends on, in one immutable object: the ``Structure``, one
        ``SpeciesData`` (pseudopotential and basis) per species label, the ``SCFConfig`` (functional, spin, mesh,
        tolerances, mixing) and the Monkhorst-Pack grid.

    Algorithm:
        Built directly, with ``Calculation.from_fdf`` (SIESTA input file) or ``Calculation.from_structure``
        (structure plus pseudopotential files); ``run`` executes it.

    Invariants:
        - ``species`` has an entry for every label of ``structure.species_labels``.
        - Frozen: variations are made with ``with_structure`` and ``with_config``.

    SIESTA Mapping:
        The contents of an ``.fdf`` file together with the pseudopotential files it refers to.
    """

    structure: Structure
    species: dict[str, SpeciesData]
    config: SCFConfig = field(default_factory=SCFConfig)
    k_grid: tuple[int, int, int] = (1, 1, 1)
    k_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    label: str = "jsiesta"
    mesh_shape: tuple[int, int, int] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        pseudo_dir: str | Path = ".",
        basis: str | dict[str, str | SpeciesBasisSpec] = "DZP",
        k_grid: tuple[int, int, int] = (1, 1, 1),
        k_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: str = "jsiesta",
        energy_shift_ry: float = 0.02,
        split_norm: float = 0.15,
        split_norm_h: float | None = None,
        mesh_shape: tuple[int, int, int] | None = None,
        config: SCFConfig | None = None,
        **settings: Any,
    ) -> Calculation:
        """Set up a calculation from a structure and a directory of PSML pseudopotentials.

        Theory:
            The Python counterpart of writing an input file: the basis is generated from each species'
            pseudopotential with the given size, energy shift and split norm.

        Algorithm:
            ``find_pseudopotential`` and ``build_species`` per species label, then ``SCFConfig(**settings)``.

        Parameters:
            structure: Geometry (see ``make_structure``).
            pseudo_dir: Directory with ``<Label>.psml`` files.
            basis: Basis size for all species, or a mapping from label to size or ``SpeciesBasisSpec``.
            k_grid: Monkhorst-Pack divisions.
            k_shift: Monkhorst-Pack displacement in grid units (0.5 for a shifted grid).
            label: Name used for output files.
            energy_shift_ry: ``PAO.EnergyShift`` in Ry.
            split_norm: ``PAO.SplitNorm``.
            split_norm_h: ``PAO.SplitNormH``, the split norm of hydrogen species (default: ``split_norm``).
                The default 0.15 gives hydrogen a second zeta that is too extended; 0.5 is SIESTA's
                recommendation and brings the H2 bond length from 0.777 to 0.748 Ang with a DZP basis (PBE).
            mesh_shape: Explicit real-space mesh ``(n1, n2, n3)`` instead of the one ``mesh_cutoff_ry`` gives
                (``Mesh.Sizes``); useful to keep the mesh fixed while the cell changes.
            config: A complete ``SCFConfig``; ``settings`` then override its fields.
            **settings: ``SCFConfig`` fields, e.g. ``xc_functional="PBE"``, ``mesh_cutoff_ry=200``,
                ``nspin=2``, ``use_symmetry=True``.

        Returns:
            ``Calculation``.

        Invariants:
            Raises ``FileNotFoundError`` when a pseudopotential is missing.

        SIESTA Mapping:
            ``ChemicalSpeciesLabel`` + ``PAO.BasisSize`` + the general options of an ``.fdf`` file.
        """
        cfg = replace(config, **settings) if config is not None else SCFConfig(**settings)
        species: dict[str, SpeciesData] = {}
        basis_info: dict[str, Any] = {}
        z_of_label = _species_atomic_numbers(structure)
        for lbl in structure.species_labels:
            size = basis.get(lbl, "DZP") if isinstance(basis, dict) else basis
            psml = find_pseudopotential(lbl, [Path(pseudo_dir)])
            split = split_norm_h if (split_norm_h is not None and z_of_label[lbl] == 1) else split_norm
            species[lbl], basis_info[lbl] = build_species(
                lbl, psml, size, xc=cfg.xc_functional, energy_shift_ha=0.5 * energy_shift_ry, split_norm=split
            )
            basis_info[lbl]["pseudopotential"] = str(psml)
        return cls(
            structure, species, cfg, tuple(k_grid), tuple(k_shift), label, mesh_shape, {"basis": basis_info}  # type: ignore[arg-type]
        )

    @classmethod
    def from_fdf(cls, fdf_path: str | Path, pseudo_dir: str | Path | None = None) -> Calculation:
        """Set up a calculation from a SIESTA ``.fdf`` input file.

        Theory:
            The input file gives the geometry and the options; the pseudopotential of each species is the file
            ``<Label>.psml``; the basis follows ``PAO.BasisSize`` / ``PAO.EnergyShift`` / ``PAO.SplitNorm`` /
            ``%block PAO.Basis``, or is read from ``<Label>.ion.xml`` with ``User.Basis T``.

        Algorithm:
            ``parse_fdf``; pseudopotentials are searched in ``pseudo_dir``, the directory of the input file and
            ``SIESTA_PS_PATH``; ``MD.UseSaveXV T`` replaces the geometry by that of ``<SystemLabel>.XV``.
            Restart of the density matrix (``DM.UseSaveDM``) is handled by ``run``.

        Parameters:
            fdf_path: Input file.
            pseudo_dir: Directory searched first for pseudopotentials and ``.ion.xml`` files.

        Returns:
            ``Calculation``; ``info`` keeps the parsed ``io`` flags, the band lines, the labels of the input file
            that were not used (``"unused"``) and warnings.

        Invariants:
            Raises ``ValueError`` for an input without atoms.

        SIESTA Mapping:
            ``siesta_init`` (reading of the input and of the species files).
        """
        fdf_path = Path(fdf_path)
        parsed = parse_fdf(fdf_path)
        structure: Structure = parsed["structure"]
        config: SCFConfig = parsed["config"]
        if structure.n_atoms == 0:
            raise ValueError(f"{fdf_path}: no atoms (AtomicCoordinatesAndAtomicSpecies block missing or empty).")
        work_dir = fdf_path.resolve().parent
        dirs = ([Path(pseudo_dir)] if pseudo_dir is not None else []) + [work_dir]
        label = parsed["io"]["system_label"]
        warnings = list(parsed["warnings"])
        if parsed["io"]["use_save_xv"]:
            xv = work_dir / f"{label}.XV"
            if xv.is_file():
                structure = structure_from_xv(xv, template=structure)
            else:
                warnings.append(f"MD.UseSaveXV: {xv.name} not found, using the geometry of the input file.")
        bset = parsed["basis"]
        explicit = (
            parse_pao_basis_block(bset["pao_basis"], config.xc_functional, bset["energy_shift_ha"], bset["split_norm"])
            if bset["pao_basis"]
            else {}
        )
        species: dict[str, SpeciesData] = {}
        basis_info: dict[str, Any] = {}
        z_of_label = _species_atomic_numbers(structure)
        for lbl in structure.species_labels:
            psml = find_pseudopotential(lbl, dirs)
            split = bset["split_norm"]
            if bset["split_norm_h"] is not None and z_of_label[lbl] == 1:
                split = bset["split_norm_h"]
            ion = None
            if bset["user_basis"]:
                ion = next((d / f"{lbl}.ion.xml" for d in dirs if (d / f"{lbl}.ion.xml").is_file()), None)
                if ion is None:
                    raise FileNotFoundError(f"User.Basis: {lbl}.ion.xml not found in {', '.join(map(str, dirs))}.")
            species[lbl], basis_info[lbl] = build_species(
                lbl, psml, explicit.get(lbl, bset["size"]), xc=config.xc_functional,
                energy_shift_ha=bset["energy_shift_ha"], split_norm=split, ion_path=ion,
            )
            basis_info[lbl]["pseudopotential"] = str(psml)
        info = {
            "basis": basis_info,
            "io": parsed["io"],
            "band_lines": parsed["band_lines"],
            "unused": parsed["unused"],
            "warnings": warnings,
            "work_dir": str(work_dir),
            "input_file": str(fdf_path),
        }
        return cls(
            structure, species, config, parsed["k_grid"], parsed["k_shift"], label, parsed["mesh_sizes"], info
        )

    def with_structure(self, structure: Structure) -> Calculation:
        """The same calculation for another geometry (same species, settings and k-grid).

        Theory:
            Energy-volume curves, finite-difference checks and scans repeat a calculation on modified
            structures with everything else fixed.

        Algorithm:
            ``dataclasses.replace``; compiled functions are reused when the array shapes do not change.

        Parameters:
            structure: New geometry with the same species labels.

        Returns:
            New ``Calculation``.

        Invariants:
            ``self`` is unchanged.

        SIESTA Mapping:
            Editing the geometry blocks of an input file.
        """
        return replace(self, structure=structure)

    def with_config(self, **settings: Any) -> Calculation:
        """The same calculation with some ``SCFConfig`` fields changed.

        Theory:
            Convergence studies vary one numerical parameter (mesh cutoff, temperature, tolerance) at a time.

        Algorithm:
            ``dataclasses.replace`` on the configuration; ``k_grid``, ``k_shift``, ``label`` and ``mesh_shape``
            may be given too. The basis is not regenerated: build a new calculation to change it.

        Parameters:
            **settings: ``SCFConfig`` fields and/or ``k_grid``, ``k_shift``, ``label``, ``mesh_shape``.

        Returns:
            New ``Calculation``.

        Invariants:
            ``self`` is unchanged.

        SIESTA Mapping:
            Editing options of an input file.
        """
        own = {k: settings.pop(k) for k in ("k_grid", "k_shift", "label", "mesh_shape") if k in settings}
        return replace(self, config=replace(self.config, **settings), **own)

    def run(
        self,
        log: Callable[[str], None] | None = None,
        initial_dm: Any | None = None,
        initial_h: Any | None = None,
    ) -> Results:
        """Run the self-consistent cycle and evaluate energy, forces and stress.

        Theory:
            Solves the Kohn-Sham equations self-consistently for the fixed geometry, then evaluates the energy
            at the output density matrix and its derivatives with respect to the atomic positions (forces) and
            to strain (stress).

        Algorithm:
            1. ``build_basis``; 2. sparsity pattern, k-points and mesh (through ``build_native_symmetry`` when
            ``config.use_symmetry``); 3. two-centre integrals; 4. ``run_scf``; 5. ``StaticObservableEvaluator``.
            With ``DM.UseSaveDM T`` in the originating input file and no ``initial_dm``, ``<SystemLabel>.DM`` is
            read if present.

        Parameters:
            log: Function receiving progress lines (e.g. ``print``); silent when ``None``.
            initial_dm: Starting density matrix on the pattern of this calculation (see ``jax_siesta.io.read_dm``).
            initial_h: Starting Hamiltonian in Hartree (see ``jax_siesta.io.read_nc_restart``).

        Returns:
            ``Results``. A run that does not converge returns normally with ``Results.converged == False``.

        Invariants:
            Raises ``RuntimeError`` only when the SCF breaks down numerically.

        SIESTA Mapping:
            ``siesta_forces`` for one geometry.
        """
        from jax_siesta.app import report

        say = log if log is not None else (lambda _line: None)
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        structure, config = self.structure, self.config
        missing = [lbl for lbl in structure.species_labels if lbl not in self.species]
        if missing:
            raise ValueError(f"No species data for {missing}; species available: {sorted(self.species)}.")
        basis = PSMLBasisLoader().build_basis(structure, self.species, include_soc_projectors=(config.nspin == 8))

        sym_proj = None
        mesh_shape = self.mesh_shape
        if config.use_symmetry:
            from jax_siesta.symmetry.engine import build_native_symmetry

            # An explicit mesh must be given to the engine, which tabulates how each operation permutes the
            # mesh points (and rejects a mesh that is not commensurate with the group).
            sym_proj, pattern, kmesh, mesh_shape = build_native_symmetry(
                structure, basis, self.k_grid, config, k_shifts=self.k_shift, mesh_shape_override=mesh_shape
            )
            two_center = sym_proj.evaluate_irreducible_two_center(structure, basis, pattern, config)
        else:
            pattern = build_sparse_pattern(structure, basis)
            kmesh = build_monkhorst_pack_mesh(self.k_grid, self.k_shift)
            mesh_shape = mesh_shape or choose_mesh_dimensions(
                structure.cell, config.mesh_cutoff_ry, criterion=config.mesh_criterion,
                mesh_subdivisions=config.mesh_subdivisions,
            )
            two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
        timings["setup"] = time.perf_counter() - t_start
        for line in report.system_lines(self, basis, pattern, kmesh, mesh_shape, sym_proj):
            say(line)

        io = self.info.get("io", {})
        if initial_dm is None and initial_h is None and io.get("use_save_dm"):
            dm_file = Path(self.info.get("work_dir", ".")) / io["dm_init_file"]
            if dm_file.is_file():
                initial_dm = read_dm(dm_file, pattern, config.nspin)
                say(f"jsiesta: density matrix read from {dm_file.name}")
            else:
                say(f"jsiesta: {dm_file.name} not found, starting from atomic densities")

        history: list[dict[str, float]] = []

        def _on_iteration(rec: dict[str, float]) -> None:
            history.append(rec)
            say(report.scf_line(rec))

        say("")
        say(report.SCF_HEADER)
        t_scf = time.perf_counter()
        state, mesh_shape = run_scf(
            structure, basis, pattern, two_center, kmesh, config, mesh_shape=mesh_shape, sym_proj=sym_proj,
            initial_dm=initial_dm, initial_h=initial_h, callback=_on_iteration,
        )
        timings["scf"] = time.perf_counter() - t_scf
        if state.outcome in ("numerical_breakdown", "invalid_input"):
            raise RuntimeError(f"SCF failed: {state.outcome} after {len(history)} iterations.")
        t_obs = time.perf_counter()
        obs = StaticObservableEvaluator().evaluate(
            structure, basis, pattern, two_center, kmesh, state, config, mesh_shape=mesh_shape, sym_proj=sym_proj
        )
        if sym_proj is not None:
            obs = replace(obs, active_symmetry_ops=sym_proj.n_ops, magnetic_subgroup=sym_proj.subgroup_label)
        timings["forces_and_stress"] = time.perf_counter() - t_obs
        timings["total"] = time.perf_counter() - t_start
        results = Results(
            label=self.label, structure=structure, basis=basis, pattern=pattern, kmesh=kmesh, config=config,
            mesh_shape=tuple(int(n) for n in mesh_shape), overlap=two_center.overlap, scf_state=state,  # type: ignore[arg-type]
            observables=obs, k_grid=tuple(self.k_grid), scf_history=tuple(history), timings=timings,  # type: ignore[arg-type]
        )
        for line in report.final_lines(results):
            say(line)
        return results


__all__ = ["EV_PER_ANG3_TO_GPA", "BandStructure", "Calculation", "Results", "make_structure"]
