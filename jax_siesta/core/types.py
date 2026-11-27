"""Immutable JAX PyTree data structures for static KS-DFT state.

Theory:
    Represents the complete state of a static SIESTA Kohn-Sham calculation in
    explicit atomic units (lengths in Bohr :math:`a_0`, energies in Hartree
    :math:`E_h`, electron charge :math:`|e|=1`). Lattice vectors are stored as
    column vectors in a :math:`3 \\times 3` matrix :math:`\\mathbf{A} = [\\mathbf{a}_1, \\mathbf{a}_2, \\mathbf{a}_3]`
    so Cartesian positions satisfy :math:`\\mathbf{r} = \\mathbf{A} \\mathbf{f}` and
    reciprocal lattice vectors satisfy :math:`\\mathbf{A}^T \\mathbf{B} = 2\\pi \\mathbf{I}`.

Algorithm:
    Each container is an immutable ``@dataclass(frozen=True)`` registered with
    ``jax.tree_util.register_pytree_node_class``. Array-valued quantities are
    returned as PyTree children so ``jax.jit``, ``jax.vmap``, and ``jax.grad``
    can trace them directly, while metadata, shapes, and discrete topology
    remain static auxiliary data.

Invariants:
    - ``cell`` has positive or non-zero determinant (Bohr^3).
    - ``nspin`` is restricted to SIESTA's four spin modes: ``1`` (unpolarized),
      ``2`` (collinear polarized), ``4`` (non-collinear real density / 2-spinor),
      or ``8`` (full spin-orbit coupling complex 2-spinor).
    - Every ``Observables`` instance explicitly records both ``last_scf_free_e``
      and ``final_post_scf_free_e`` with ``dm_out_recomputed`` provenance.

SIESTA Mapping:
    Replaces module-level globals in ``atomlist.F``, ``sparse_matrices.F``,
    ``m_spin.F90``, ``m_energies.F90``, and ``siesta_options.F90``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV: float = 27.211386245988
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV
HARTREE_TO_RY: float = 2.0
RY_TO_HARTREE: float = 0.5
KB_HARTREE_PER_KELVIN: float = 3.1668115634556e-6


def _register_dataclass_pytree(cls: type[Any]) -> type[Any]:
    """Register a frozen dataclass as a JAX PyTree with explicit child/aux split."""

    def tree_flatten(self: Any) -> tuple[list[Any], tuple[tuple[str, ...], dict[str, Any]]]:
        child_names = getattr(self, "_pytree_children", ())
        children = [getattr(self, name) for name in child_names]
        aux = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in child_names
        }
        return children, (child_names, aux)

    @classmethod  # type: ignore[misc]
    def tree_unflatten(
        klass: type[Any],
        aux_data: tuple[tuple[str, ...], dict[str, Any]],
        children: list[Any],
    ) -> Any:
        child_names, aux = aux_data
        kwargs = dict(aux)
        for name, val in zip(child_names, children, strict=True):
            kwargs[name] = val
        return klass(**kwargs)

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    return jax.tree_util.register_pytree_node_class(cls)


@_register_dataclass_pytree
@dataclass(frozen=True)
class Structure:
    r"""Periodic crystal or molecular atomic geometry in atomic units.

    Theory:
        Encodes lattice vectors :math:`\mathbf{A} \in \mathbb{R}^{3\times 3}` (Bohr,
        columns are :math:`\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3`), Cartesian coordinates
        :math:`\mathbf{R} \in \mathbb{R}^{N_{\text{at}}\times 3}` (Bohr), species indices,
        and initial atomic magnetic moments :math:`\mathbf{m}_I \in \mathbb{R}^3`.

    Algorithm:
        Immutable JAX PyTree holding ``cell``, ``positions``, and ``initial_moments``
        as differentiable array children so geometric/strain derivatives flow cleanly.

    Invariants:
        - ``cell`` shape is ``(3, 3)``; volume ``abs(det(cell)) > 0``.
        - ``positions`` shape is ``(n_atoms, 3)``.
        - ``species_indices`` tuple length equals ``n_atoms``.

    SIESTA Mapping:
        Replaces ``ucell``, ``xa``, ``isa``, ``iza``, and ``na_u`` in ``siesta_geom.F90``.
    """

    cell: jax.Array
    positions: jax.Array
    species_indices: tuple[int, ...]
    atomic_numbers: tuple[int, ...]
    species_labels: tuple[str, ...]
    initial_moments: jax.Array
    pbc: tuple[bool, bool, bool] = (True, True, True)

    _pytree_children = ("cell", "positions", "initial_moments")

    @property
    def n_atoms(self) -> int:
        """Number of unit-cell atoms."""
        return len(self.species_indices)

    @property
    def volume(self) -> jax.Array:
        """Unit cell volume :math:`\\Omega = |\\det(\\mathbf{A})|` in :math:`a_0^3`."""
        return jnp.abs(jnp.linalg.det(self.cell))

    @property
    def reciprocal_cell(self) -> jax.Array:
        """Reciprocal lattice column matrix :math:`\\mathbf{B} = 2\\pi (\\mathbf{A}^{-1})^T`."""
        return 2.0 * jnp.pi * jnp.linalg.inv(self.cell).T


@_register_dataclass_pytree
@dataclass(frozen=True)
class SpeciesData:
    r"""PSML pseudopotential and confined NAO radial basis tables for one species.

    Theory:
        Stores norm-conserving PSML semilocal potentials, separable Kleinman-Bylander
        (KB) projectors :math:`|\chi_{lm}^j\rangle` with energies :math:`E^{\text{KB}}_{lj}`,
        neutral-atom screening potential :math:`V^{\text{NA}}(r)` and density
        :math:`\rho^{\text{NA}}(r)`, pseudo-core density :math:`\rho^{\text{core}}(r)` for
        non-linear core corrections (NLCC), and confined numerical atomic orbitals
        :math:`\phi_{nl\zeta}(r) = R_{nl\zeta}(r) Y_{lm}(\hat{\mathbf{r}})`.

    Algorithm:
        Radial functions are stored on uniform or logarithmic grids with precomputed
        natural cubic spline coefficients for :math:`O(1)` evaluation in C++/JAX.
        Sequential ``pao_iz`` keys distinguish semicore and valence shells with
        identical ``(l, m)`` quantum numbers.

    Invariants:
        - Every radial orbital vanishes smoothly at its cutoff radius :math:`r_c`.
        - KB energies satisfy finite real bounds; ``len(kb_l) == len(kb_energies)``.
        - ``pao_iz`` is strictly unique per ``(l, m)`` shell on a species.

    SIESTA Mapping:
        Replaces ``species_info`` in ``atm_types.F`` and ``atmfuncs.F`` (PSML-only;
        legacy ``.psf`` is excluded by design).
    """

    symbol: str
    atomic_number: int
    valence_charge: float
    mass_amu: float
    r_grid: jax.Array
    v_local: jax.Array
    v_na: jax.Array
    rho_na: jax.Array
    rho_core: jax.Array
    pao_r: jax.Array
    pao_values: jax.Array
    pao_l: tuple[int, ...]
    pao_n: tuple[int, ...]
    pao_zeta: tuple[int, ...]
    pao_iz: tuple[int, ...]
    pao_rcut: tuple[float, ...]
    kb_r: jax.Array
    kb_projectors: jax.Array
    kb_l: tuple[int, ...]
    kb_j2: tuple[int, ...]
    kb_energies: jax.Array
    kb_rcut: tuple[float, ...]
    has_nlcc: bool = False
    has_spin_orbit: bool = False
    rcut_na: float = 0.0
    psml_sha256: str = ""
    tau_core: jax.Array | None = None
    has_tau_core: bool = False

    _pytree_children = (
        "r_grid",
        "v_local",
        "v_na",
        "rho_na",
        "rho_core",
        "pao_r",
        "pao_values",
        "kb_r",
        "kb_projectors",
        "kb_energies",
        "tau_core",
    )


@_register_dataclass_pytree
@dataclass(frozen=True)
class BasisData:
    """Unit-cell and supercell orbital index maps across all atoms.

    Theory:
        Maps every unit-cell atomic orbital :math:`\\mu \\in [0, N_{\\text{orb}})` to its
        owning atom :math:`I`, species :math:`s`, angular momentum :math:`(l, m)`,
        sequential radial shell index :math:`i_z`, and cutoff radius :math:`r_c(\\mu)`.

    Algorithm:
        Precomputed flat integer/float arrays enable vectorized sparse assembly and
        O(1) orbital-to-atom lookup inside C++ FFI kernels and symmetry projectors.

    Invariants:
        - ``n_orb == len(orb_atom) == len(orb_l) == len(orb_m) == len(orb_iz)``.
        - ``total_valence_electrons >= 0``.

    SIESTA Mapping:
        Replaces ``iphorb``, ``iaorb``, ``lasto``, ``no_u``, and ``no_s`` in ``atomlist.F``.
    """

    species: tuple[SpeciesData, ...]
    n_orb: int
    n_kb: int
    total_valence_electrons: float
    orb_atom: tuple[int, ...]
    orb_species: tuple[int, ...]
    orb_shell: tuple[int, ...]
    orb_l: tuple[int, ...]
    orb_m: tuple[int, ...]
    orb_iz: tuple[int, ...]
    orb_rcut: jax.Array
    atom_orb_offsets: tuple[int, ...]
    kb_atom: tuple[int, ...]
    kb_species: tuple[int, ...]
    kb_l: tuple[int, ...]
    kb_m: tuple[int, ...]
    kb_j2: tuple[int, ...]
    kb_energies: jax.Array
    kb_rcut: jax.Array

    _pytree_children = ("species", "orb_rcut", "kb_energies", "kb_rcut")


@_register_dataclass_pytree
@dataclass(frozen=True)
class SparsePattern:
    r"""Symmetry-closed static CSR/COO orbital-pair sparsity pattern.

    Theory:
        Stores all interacting unit-cell-to-supercell orbital pairs
        :math:`(\mu, \nu, \mathbf{R}_\nu)` whose center-to-center distance satisfies
        :math:`|\mathbf{R}_{I(\nu)} + \mathbf{T}_\nu - \mathbf{R}_{I(\mu)}| \le r_c(\mu) + r_c(\nu)`
        (or share a KB projector overlap). In symmetry-enabled runs, the edge set
        is closed under every space/magnetic group operation :math:`g \in G`.

    Algorithm:
        Stored in CSR (``row_ptr``, ``dst_unit_orb``, ``edge_cell_idx``, ``cell_shifts``)
        and edge-list COO (``src_orb``, ``dst_unit_orb``, ``edge_vectors``) form so
        Bloch summation :math:`M_{\mu\nu}(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} M_{\mu,\nu\mathbf{R}}`
        and symmetry ``segment_sum`` execute on GPU without dynamic allocation.

    Invariants:
        - ``n_edges == len(src_orb) == len(dst_unit_orb) == len(edge_cell_idx)``.
        - When ``symmetry_closed=True``, every edge's image under :math:`g \in G`
          exists in the edge table.

    SIESTA Mapping:
        Replaces ``numh``, ``listhptr``, ``listh``, ``xijo``, and ``indxuo`` in
        ``sparse_matrices.F`` and ``listsc``.
    """

    n_orb: int
    n_super_cells: int
    nsc: tuple[int, int, int]
    cell_shifts: jax.Array
    src_orb: jax.Array
    dst_unit_orb: jax.Array
    edge_cell_idx: jax.Array
    edge_vectors: jax.Array
    row_ptr: jax.Array
    transpose_edge_idx: jax.Array
    symmetry_closed: bool = False

    _pytree_children = (
        "cell_shifts",
        "src_orb",
        "dst_unit_orb",
        "edge_cell_idx",
        "edge_vectors",
        "row_ptr",
        "transpose_edge_idx",
    )

    @property
    def n_edges(self) -> int:
        """Total number of directed orbital-pair edges."""
        return int(self.src_orb.shape[0])


@_register_dataclass_pytree
@dataclass(frozen=True)
class KPointMesh:
    """Brillouin-zone sampling mesh (full BZ or irreducible BZ via spglib).

    Theory:
        Represents wavevectors :math:`\\mathbf{k}_q` in fractional reciprocal coordinates
        and normalized weights :math:`w_q` satisfying :math:`\\sum_q w_q = 1`.

    Algorithm:
        Produced either as a uniform Monkhorst-Pack grid or reduced to the
        irreducible Brillouin zone (IBZ) using ``spglib.get_stabilized_reciprocal_mesh``
        with proper ``is_time_reversal`` handling for magnetic/SOC sectors.

    Invariants:
        - ``kpoints_frac`` has shape ``(n_k, 3)``.
        - ``weights`` has shape ``(n_k,)``, all strictly positive, summing to ``1.0``.

    SIESTA Mapping:
        Replaces ``kpoint_scf_m`` / ``Src/kpoint_scf.F90``.
    """

    kpoints_frac: jax.Array
    weights: jax.Array
    grid_shape: tuple[int, int, int]
    shifts: tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_ibz: bool = False
    used_time_reversal: bool = True

    _pytree_children = ("kpoints_frac", "weights")

    @property
    def n_k(self) -> int:
        """Number of k-points in the mesh."""
        return int(self.kpoints_frac.shape[0])


@_register_dataclass_pytree
@dataclass(frozen=True)
class TwoCenterMatrices:
    """Precomputed static sparse two-center integrals and geometric derivatives.

    Theory:
        Holds the overlap :math:`S_{\\mu\\nu}`, kinetic :math:`T_{\\mu\\nu}`,
        neutral-atom screening :math:`V^{\\text{NA}}_{\\mu\\nu}`, and Kleinman-Bylander
        nonlocal :math:`V^{\\text{KB}}_{\\mu\\nu}` (including scalar and spin-orbit
        channels for ``nspin=8``) matrices on the ``SparsePattern``, together with
        their exact derivatives wrt pair displacement :math:`\\mathbf{R}_{\\mu\\nu}`
        for Pulay and Hellmann-Feynman forces/stresses.

    Algorithm:
        Evaluated once per geometry via C++ Fourier-Bessel ``matel`` kernels
        (and on orbit representatives only when symmetry is active).

    Invariants:
        - All sparse edge arrays have leading dimension ``n_edges``.
        - :math:`H_0 = T + V^{\\text{NA}} + V^{\\text{KB}}` is Hermitian across supercell pairs.

    SIESTA Mapping:
        Replaces ``S``, ``H0`` (``T + Vna + Vkb``) from ``overlap``, ``kinefsm``,
        ``naefs``, and ``nlefsm``.
    """

    overlap: jax.Array
    kinetic: jax.Array
    v_na: jax.Array
    v_kb: jax.Array
    v_kb_so: jax.Array | None
    d_overlap_dr: jax.Array
    d_kinetic_dr: jax.Array
    d_v_na_dr: jax.Array
    d_v_kb_dr: jax.Array
    e_na_pair: jax.Array
    f_na_atoms: jax.Array
    stress_na: jax.Array
    kb_link_p: jax.Array | None = None
    kb_link_dp: jax.Array | None = None
    kb_link_orb: tuple[int, ...] = ()
    kb_link_kb: tuple[int, ...] = ()
    kb_link_cell_shift: jax.Array | None = None

    _pytree_children = (
        "overlap",
        "kinetic",
        "v_na",
        "v_kb",
        "v_kb_so",
        "d_overlap_dr",
        "d_kinetic_dr",
        "d_v_na_dr",
        "d_v_kb_dr",
        "e_na_pair",
        "f_na_atoms",
        "stress_na",
        "kb_link_p",
        "kb_link_dp",
        "kb_link_cell_shift",
    )


@dataclass(frozen=True)
class SCFConfig:
    """Immutable configuration controlling SCF mixing, smearing, XC, and symmetry.

    Theory:
        Collects physical parameters (``nspin``, electronic temperature :math:`T_e`
        in Hartree, XC functional IDs, dispersion model, mesh cutoff) and
        ``scf_params`` conditioning/adaptive controller hyperparameters.

    Algorithm:
        Static Python dataclass (no array leaves) passed as compile-time/runtime
        configuration to the SCF driver and adaptive trust-region controller.

    Invariants:
        - ``nspin in (1, 2, 4, 8)``.
        - ``electronic_temp_ha >= 0.0``.
        - ``mix_target in ("Hamiltonian", "DM")``.
        - ``mesh_criterion in ("jsiesta", "siesta")``; ``mesh_subdivisions >= 1`` (SIESTA ``MeshSubDivisions``).
        - ``hfx_n_gauss`` (Gaussians per orbital shell) and ``hfx_screen_tol`` (screening of the four-centre
          integrals, Hartree) are used only when ``xc_functional`` is a hybrid. ``hfx_pattern_scale >= 1`` extends
          the sparse pattern to orbital pairs within that multiple of the sum of their radii.
        - ``mixer_variant in ("tikhonov", "pulay", "broyden", "adiis")``; ``"adiis"`` requires
          ``mix_target == "Hamiltonian"``. (Until D-022 ``"adiis"``/``"broyden"`` were Pulay with a different diagonal
          shift and raised ``NotImplementedError``.)

    SIESTA Mapping:
        Consolidates ``siesta_options``, ``m_mixing_scf``, and ``scf_adaptive.lua``.
    """

    nspin: int = 1
    max_scf_iterations: int = 100
    d_tol: float = 1e-5
    h_tol_ha: float = 1e-5
    e_tol_ha: float = 1e-6
    electronic_temp_ha: float = 300.0 * KB_HARTREE_PER_KELVIN
    mesh_cutoff_ry: float = 150.0
    mesh_criterion: str = "siesta"
    mesh_subdivisions: int = 2
    xc_functional: str = "PBE"
    dispersion: str = "none"
    mix_target: str = "Hamiltonian"
    mixer_variant: str = "tikhonov"
    mix_spin: str = "decoupled"
    mix_weight: float | None = None
    mix_history: int | None = None
    tikhonov_reg: float | None = None
    svd_tol: float = 1e-12
    enable_adaptive_controller: bool = True
    enable_smearing_annealing: bool = False
    use_symmetry: bool = False
    symmetrize_grid: bool = True
    symprec: float = 1e-5
    mag_symprec: float = 1e-3
    net_charge: float = 0.0
    hfx_n_gauss: int = 5
    hfx_screen_tol: float = 1e-7
    hfx_pattern_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nspin not in (1, 2, 4, 8):
            raise ValueError(f"Unsupported nspin={self.nspin}; expected one of (1, 2, 4, 8).")
        if self.mesh_criterion not in ("jsiesta", "siesta"):
            raise ValueError(f"Unsupported mesh_criterion={self.mesh_criterion!r}; expected 'jsiesta' or 'siesta'.")
        if self.mesh_subdivisions < 1:
            raise ValueError("mesh_subdivisions must be >= 1.")
        if self.mix_target not in ("Hamiltonian", "DM"):
            raise ValueError(f"Unsupported mix_target={self.mix_target!r}.")
        if self.mixer_variant not in ("tikhonov", "pulay", "broyden", "adiis"):
            raise ValueError(f"Unsupported mixer_variant={self.mixer_variant!r}.")
        if self.mixer_variant == "adiis" and self.mix_target != "Hamiltonian":
            raise ValueError("mixer_variant='adiis' needs mix_target='Hamiltonian' (it uses H[D_out] and E[D_out]).")
        if self.mix_spin not in ("coupled", "decoupled"):
            raise ValueError(f"Unsupported mix_spin={self.mix_spin!r}.")


@_register_dataclass_pytree
@dataclass(frozen=True)
class SCFState:
    """Complete mutable-free state snapshot of one SCF iteration.

    Theory:
        Tracks sparse Hamiltonian :math:`H`, density matrix :math:`D`, energy-density
        matrix :math:`E`, Fermi level :math:`\\mu_F`, adaptive trust-region state
        (:math:`w_{\\text{cur}}`, ``streak_bad``, active :math:`T_e`), and Krylov history.

    Algorithm:
        Passed functionally from step ``iscf`` to ``iscf + 1`` inside ``jax.lax``
        or the adaptive SCF driver, enabling exact checkpointing and zero hidden state.

    Invariants:
        - ``dm`` and ``edm`` have shape ``(n_spin_mat, n_edges)`` (real or complex).
        - ``history_x`` and ``history_f`` share identical leading history capacity.
        - ``harris_energy_ha`` is the Harris-Foulkes free energy of the last iteration (``None`` before the first);
          ``harris_provenance`` is ``"exact_dm_mixing"`` (``H_in = H[D_in]``) or ``"siesta_h_mixing"``
          (``D_in`` = previous ``D_out``, as SIESTA does when mixing the Hamiltonian; not variational).
        - ``history_d`` (output density matrices) and ``history_e`` (their energies) exist only for the ADIIS mixer
          and share that capacity; they are ``None`` otherwise.

    SIESTA Mapping:
        Replaces ``Dscf``, ``Escf``, ``H``, ``Efermi``, and ``m_mixing`` internal buffers.
    """

    iteration: int
    h_sparse: jax.Array
    dm_in: jax.Array
    dm_out: jax.Array
    edm_out: jax.Array
    fermi_level_ha: jax.Array
    entropy_ha: jax.Array
    last_scf_free_e_ha: jax.Array
    w_cur: jax.Array
    w_base: jax.Array
    active_temp_ha: jax.Array
    streak_bad: int
    prev_d_max: jax.Array
    prev_h_max: jax.Array
    history_x: jax.Array
    history_f: jax.Array
    history_count: int
    converged: bool = False
    outcome: str = "running"
    history_d: jax.Array | None = None
    history_e: jax.Array | None = None
    harris_energy_ha: jax.Array | None = None
    harris_provenance: str = "not_computed"

    _pytree_children = (
        "h_sparse",
        "dm_in",
        "dm_out",
        "edm_out",
        "fermi_level_ha",
        "entropy_ha",
        "last_scf_free_e_ha",
        "w_cur",
        "w_base",
        "active_temp_ha",
        "prev_d_max",
        "prev_h_max",
        "history_x",
        "history_f",
        "history_d",
        "history_e",
        "harris_energy_ha",
    )


@_register_dataclass_pytree
@dataclass(frozen=True)
class Observables:
    """Provenanced static DFT observables satisfying the ``DM_out`` energy contract.

    Theory:
        Separates the in-flight SCF loop energy (``last_scf_free_e_ha``) from the
        mandatory fresh non-mixed post-SCF evaluation on ``DM_out``
        (``final_post_scf_free_e_ha``), alongside decomposed Kohn-Sham terms,
        Harris functional energy (of the last SCF iteration, with ``harris_provenance``), fixed-(D,E) variational atomic forces, stress
        tensor, and magnetic moments.

    Algorithm:
        Can be serialized directly to ``validation.py`` schema-v1 ``observables.json``
        in explicit Hartree / eV / Bohr units with no silent fallback between
        ``last_scf_free_e`` and ``final_post_scf_free_e``.

    Invariants:
        - ``dm_out_recomputed`` must be ``True`` for any accepted ``converged`` result.
        - ``outcome`` belongs to the five-state taxonomy:
          ``converged``, ``iteration_cap_exhausted``, ``numerical_breakdown``,
          ``invalid_input``, ``normal_state_collapse``.
        - ``e_exact_exchange_ha`` is the exact-exchange energy of a hybrid functional (zero otherwise); it is part
          of ``total_energy_ha`` and not of ``e_xc_ha``, which holds the semilocal remainder.
        - ``forces_available`` is false when forces and stress were not computed
          (``compute_forces_stress=False``); the arrays are then zero.

    SIESTA Mapping:
        Consolidates ``m_energies``, ``post_scf_work``, ``compute_energies``,
        ``fa``, and ``stress`` from SIESTA and ``scf_params/ENERGY_CONTRACT.md``.
    """

    outcome: str
    converged: bool
    n_iterations: int
    dm_out_recomputed: bool
    last_scf_free_e_ha: jax.Array
    last_scf_cycle: int
    final_post_scf_free_e_ha: jax.Array
    total_energy_ha: jax.Array
    harris_energy_ha: jax.Array
    e_kinetic_ha: jax.Array
    e_na_ha: jax.Array
    e_kb_ha: jax.Array
    e_hartree_ha: jax.Array
    e_xc_ha: jax.Array
    e_disp_ha: jax.Array
    e_entropy_ha: jax.Array
    fermi_level_ha: jax.Array
    forces_ha_per_bohr: jax.Array
    stress_ha_per_bohr3: jax.Array
    total_spin_moment: jax.Array
    atomic_spin_moments: jax.Array
    charge_residual: jax.Array
    d_max: jax.Array
    h_max_ha: jax.Array
    active_symmetry_ops: int = 1
    magnetic_subgroup: str = "P1"
    xc_backend: str = "builtin_cpp"
    harris_provenance: str = "not_computed"
    e_exact_exchange_ha: float = 0.0
    forces_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    _pytree_children = (
        "last_scf_free_e_ha",
        "final_post_scf_free_e_ha",
        "total_energy_ha",
        "harris_energy_ha",
        "e_kinetic_ha",
        "e_na_ha",
        "e_kb_ha",
        "e_hartree_ha",
        "e_xc_ha",
        "e_disp_ha",
        "e_entropy_ha",
        "fermi_level_ha",
        "forces_ha_per_bohr",
        "stress_ha_per_bohr3",
        "total_spin_moment",
        "atomic_spin_moments",
        "charge_residual",
        "d_max",
        "h_max_ha",
    )
