"""Strict Python Protocol extension seams for modular KS-DFT components.

Theory:
    Decouples every major physical and numerical stage of the SIESTA method behind
    structural subtyping interfaces (``typing.Protocol``), enabling independent
    implementations (e.g., DFT+U, hybrid functionals, BdG pairing, linear-scaling
    solvers, or CUDA vs CPU FFI kernels) without altering the core SCF pipeline.

Algorithm:
    Runtime-checkable ``Protocol`` definitions specify exact typed input/output
    contracts using ``jax_siesta.core.types`` PyTrees.

Invariants:
    - All protocol methods are pure functions with no hidden global mutation.
    - Physical quantities obey atomic units (Bohr, Hartree).

SIESTA Mapping:
    Replaces implicit Fortran module coupling across ``basis_specs``, ``matel``,
    ``m_dm_symmetrize``, ``dhscf``, ``diagon``, ``mixer``, and ``siesta_forces``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax

from jax_siesta.core.types import (
    BasisData,
    KPointMesh,
    Observables,
    SCFConfig,
    SCFState,
    SparsePattern,
    SpeciesData,
    Structure,
    TwoCenterMatrices,
)


@runtime_checkable
class BasisAndPseudos(Protocol):
    """Protocol for PSML pseudopotential parsing and confined NAO basis loading.

    Theory:
        Loads norm-conserving ``.psml`` XML data and constructs confined numerical
        atomic orbitals :math:`\\phi_{nl\\zeta}(r)` and neutral-atom screening fields.

    Algorithm:
        Parses PSML XML exclusively (legacy ``.psf`` unsupported) and matches
        radial grids and cutoff radii into immutable ``SpeciesData`` and ``BasisData``.

    Invariants:
        - Must reject ``.psf`` inputs explicitly.
        - Must assign unique sequential ``pao_iz`` shell indices.

    SIESTA Mapping:
        Replaces ``atom.F``, ``basis_specs.f``, and ``psml_wrappers``.
    """

    def load_species(self, symbol: str, psml_path: str, **kwargs: Any) -> SpeciesData:
        """Load one species from a ``.psml`` file and basis specification."""
        ...

    def build_basis(self, structure: Structure, species_map: dict[str, SpeciesData]) -> BasisData:
        """Assemble unit-cell orbital and KB projector index tables."""
        ...


@runtime_checkable
class TwoCenterIntegrals(Protocol):
    """Protocol for sparse two-center integral evaluation (:math:`S, T, V^{\\text{NA}}, V^{\\text{KB}}`).

    Theory:
        Evaluates overlap, kinetic, neutral-atom, and Kleinman-Bylander nonlocal
        integrals and their Cartesian pair derivatives :math:`\\partial/\\partial\\mathbf{R}`.

    Algorithm:
        Computes integrals over the ``SparsePattern`` (on orbit representatives
        only when symmetry is active, rotating partners via Wigner-D matrices).

    Invariants:
        - Output matrices preserve Hermiticity and symmetry commutation.

    SIESTA Mapping:
        Replaces ``matel``, ``kinefsm``, ``naefs``, and ``nlefsm``.
    """

    def evaluate(
        self,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        config: SCFConfig,
    ) -> TwoCenterMatrices:
        """Evaluate sparse two-center matrices and pair derivatives."""
        ...


@runtime_checkable
class SymmetryProjector(Protocol):
    """Protocol for crystal/magnetic space-group symmetry operations.

    Theory:
        Implements space/magnetic group maps, IBZ k-point reduction, commensurate
        grid permutations :math:`\\pi_S`, and sparse matrix projection
        :math:`\\mathcal{P}_{\\text{sym}}` for density and energy-density matrices.

    Algorithm:
        Precompiles static COO/CSR projection tables at startup using 16-point
        Fibonacci-sphere Wigner-D matrices and SU(2) spinor maps.

    Invariants:
        - Idempotency: :math:`\\mathcal{P}_{\\text{sym}}^2 = \\mathcal{P}_{\\text{sym}}`.
        - Electron-count preservation: :math:`\\mathrm{Tr}[\\mathcal{P}_{\\text{sym}}(D) S] = \\mathrm{Tr}[DS]`.

    SIESTA Mapping:
        Replaces and upgrades ``m_symmetry.F90``, ``m_dm_symmetrize.F90``, and ``kpoint_scf.F90``.
    """

    def project_matrix(self, matrix_edges: jax.Array, nspin: int) -> jax.Array:
        """Apply precompiled sparse symmetry projection to ``D`` or ``E`` edges."""
        ...

    def symmetrize_grid(self, grid_field: jax.Array) -> jax.Array:
        """Apply exact group averaging over commensurate 3D grid permutations."""
        ...

    def project_forces(self, forces: jax.Array) -> jax.Array:
        """Project atomic forces onto the invariant subspace of the group."""
        ...

    def project_stress(self, stress: jax.Array) -> jax.Array:
        """Project stress tensor onto the invariant subspace of the group."""
        ...


@runtime_checkable
class RealSpaceGridOps(Protocol):
    """Protocol for sparse orbital-grid scatter/gather and Poisson electrostatics.

    Theory:
        Evaluates :math:`\\rho(\\mathbf{r}) = \\sum_{\\mu\\nu} D_{\\mu\\nu}\\phi_\\mu(\\mathbf{r})\\phi_\\nu(\\mathbf{r})`,
        solves :math:`\\nabla^2 V_H(\\mathbf{r}) = -4\\pi (\\rho(\\mathbf{r}) - \\rho^{\\text{NA}}(\\mathbf{r}))`
        via 3D FFT, and gathers :math:`V_{\\text{eff}}(\\mathbf{r})` back to sparse
        Hamiltonian elements :math:`H^{\\text{grid}}_{\\mu\\nu}`.

    Algorithm:
        Uses C++/CUDA XLA FFI kernels for sparse bounded-sphere orbital-grid loops
        and pure JAX ``jnp.fft`` for Poisson and grid gradients.

    Invariants:
        - Exact adjoint identity: :math:`\\langle V | \\rho(D) \\rangle = \\mathrm{Tr}[D V_{\\text{grid}}]`.

    SIESTA Mapping:
        Replaces ``rhoofd.F90``, ``vmat.F90``, ``poison.F``, and grid ``dhscf.F``.
    """

    def density_on_grid(
        self,
        dm_edges: jax.Array,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        mesh_shape: tuple[int, int, int],
        nspin: int,
    ) -> jax.Array:
        """Scatter sparse density matrix onto the 3D real-space grid."""
        ...

    def potential_to_matrix(
        self,
        v_eff_grid: jax.Array,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        nspin: int,
    ) -> jax.Array:
        """Gather effective grid potential into sparse Hamiltonian matrix elements."""
        ...

    def hartree_potential(
        self,
        delta_rho_grid: jax.Array,
        cell: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Solve periodic 3D Poisson equation returning ``(v_hartree, e_hartree, stress_hartree)``."""
        ...


@runtime_checkable
class ExchangeCorrelation(Protocol):
    """Protocol for semilocal exchange-correlation evaluation (LDA/GGA/meta-GGA).

    Theory:
        Computes exchange-correlation energy density :math:`\\epsilon_{\\text{xc}}(\\mathbf{r})`
        and functional derivatives :math:`V_{\\text{xc}}(\\mathbf{r}) = \\delta E_{\\text{xc}}/\\delta \\rho(\\mathbf{r})`
        plus stress contributions.

    Algorithm:
        Computes :math:`\\nabla\\rho` via JAX 3D FFTs on GPU/CPU and calls ``libxc``
        (or built-in C++ analytical XC kernel) pointwise via JAX XLA FFI.

    Invariants:
        - Satisfies finite-difference functional derivative check :math:`\\delta E_{\\text{xc}}/\\delta\\rho`.

    SIESTA Mapping:
        Replaces ``cellxc`` and ``libgridxc`` semilocal XC drivers.
    """

    def evaluate(
        self,
        rho_grid: jax.Array,
        cell: jax.Array,
        nspin: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate ``(e_xc, v_xc_grid, stress_xc)`` from real-space spin density."""
        ...


@runtime_checkable
class DispersionModel(Protocol):
    """Protocol for van der Waals dispersion corrections (DFT-D3 / rVV10).

    Theory:
        Adds long-range non-local correlation energy, forces, and stress either
        atom-pairwise (Grimme DFT-D3 with Becke-Johnson damping) or density-based
        (Sabatini rVV10 3D-FFT kernel).

    Algorithm:
        Implemented natively in differentiable JAX for automatic forces/stress.

    Invariants:
        - Pairwise/grid energies vanish cleanly in isolated single-atom limits.

    SIESTA Mapping:
        Replaces ``libgridxc`` nonlocal vdW-DF / external DFT-D3 calls.
    """

    def evaluate(
        self,
        structure: Structure,
        rho_grid: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(e_disp, forces_disp, stress_disp)`` in atomic units."""
        ...


@runtime_checkable
class EigenSolver(Protocol):
    """Protocol for generalized Kohn-Sham eigensolution and density matrix assembly.

    Theory:
        Solves :math:`H(\\mathbf{k}) C(\\mathbf{k}) = S(\\mathbf{k}) C(\\mathbf{k}) \\varepsilon(\\mathbf{k})`
        across Brillouin-zone k-points, determines Fermi level :math:`\\mu_F` for
        electronic temperature :math:`T_e`, and builds :math:`D` and :math:`E`.

    Algorithm:
        Batched Cholesky/Löwdin orthogonalization and ``jnp.linalg.eigh`` via ``jax.vmap``.

    Invariants:
        - :math:`C(\\mathbf{k})^\\dagger S(\\mathbf{k}) C(\\mathbf{k}) = I`.
        - :math:`\\mathrm{Tr}[D S] = N_e`.

    SIESTA Mapping:
        Replaces ``diagon.F`` and ``compute_dm.F``.
    """

    def solve(
        self,
        h_sparse: jax.Array,
        s_sparse: jax.Array,
        pattern: SparsePattern,
        kmesh: KPointMesh,
        n_electrons: float,
        electronic_temp_ha: float,
        nspin: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return ``(dm_out, edm_out, fermi_ha, e_band_ha, entropy_ha)``."""
        ...


@runtime_checkable
class SCFMixer(Protocol):
    """Protocol for SCF fixed-point mixing (Tikhonov-Pulay, Pulay, Broyden, ADIIS-to-Pulay hybrid).

    Theory:
        Accelerates fixed-point convergence of :math:`x_{k+1} = \\mathcal{M}(\\{x_j, F_j\\})`
        with residual :math:`F_k = x_{\\text{out}, k} - x_{\\text{in}, k}`, supporting
        decoupled charge/spin preconditioning and deterministic scalar SVD handling.

    Algorithm:
        Maintains functional Krylov subspace history inside ``SCFState`` with
        multiplicative Tikhonov regularization on the Pulay Gram matrix.

    Invariants:
        - Deterministic handling of :math:`1 \\times 1` Gram matrices (no undefined rank/info).
        - Phase-locked secant coefficients across charge and spin channels.

    SIESTA Mapping:
        Replaces ``mixer.F``, ``m_mixing.F90``, and ``scf_params`` mixer prototypes.
    """

    def step(
        self,
        x_in: jax.Array,
        x_out: jax.Array,
        state: SCFState,
        config: SCFConfig,
        aux: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, SCFState]:
        """Compute mixed next iterate and updated ``SCFState`` (``aux = (d_out, energy)`` for ADIIS)."""
        ...


@runtime_checkable
class ObservableEvaluator(Protocol):
    """Protocol for mandatory post-SCF ``DM_out`` evaluation and variational gradients.

    Theory:
        Evaluates total Kohn-Sham free energy, Harris energy, and fixed-(D,E)
        variational forces and stress tensor directly from converged ``DM_out``.

    Algorithm:
        Executes one clean non-mixed grid/potential pass from ``DM_out`` and
        evaluates Hellmann-Feynman + Pulay (:math:`-\\mathrm{Tr}[E\\,\\partial S/\\partial\\mathbf{R}]`)
        + discrete grid ``dfscf`` derivatives without differentiating through SCF loops.

    Invariants:
        - Must set ``dm_out_recomputed=True``; no silent fallback to ``last_scf_free_e``.

    SIESTA Mapping:
        Replaces ``post_scf_work.F``, ``compute_energies.F90``, ``dfscf.F``, and ``siesta_forces.F``.
    """

    def evaluate(
        self,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        two_center: TwoCenterMatrices,
        kmesh: KPointMesh,
        scf_state: SCFState,
        config: SCFConfig,
    ) -> Observables:
        """Run clean post-SCF ``DM_out`` pass and compute variational forces and stress."""
        ...
