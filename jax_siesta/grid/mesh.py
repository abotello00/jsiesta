r"""Real-space mesh operations, Poisson solver, and C++ XLA FFI wrappers for ``rhoofd``/``vmat``.

Theory:
    Provides:
    - Symmetry-aware uniform 3D periodic mesh sizing from ``MeshCutoff`` (:math:`G_{\max}^2 = E_{\text{cut,Ry}}`)
    - Bounded-sphere orbital-grid density scatter ``rhoofd`` (:math:`\rho_s(\mathbf{r})`)
      and exact discrete adjoint potential gather ``vmat`` (:math:`V^{\text{grid}}_{s,\mu\nu}`)
      via compiled C++ XLA FFI kernels with ``jax.custom_vjp`` backward passes (``dfscf``)
    - Neutral-atom screening density :math:`\rho^{\text{NA}}(\mathbf{r})`, potential
      :math:`V^{\text{NA}}(\mathbf{r})`, and NLCC pseudo-core charge :math:`\rho^{\text{core}}(\mathbf{r})`
    - Periodic 3D FFT Poisson solver :math:`\nabla^2 V_H = -4\pi \delta\rho` and
      analytical Hartree reciprocal-space stress tensor

Algorithm:
    Uses ``jax.extend.ffi.ffi_call`` for ``jsiesta_rhoofd_f64``, ``jsiesta_vmat_f64``,
    ``jsiesta_dfscf_grad_f64``, ``jsiesta_na_grid_f64``, and ``jsiesta_na_grid_grad_f64``,
    paired with ``jnp.fft.rfftn``/``irfftn`` for Poisson electrostatics.

Invariants:
    - Exact discrete adjoint identity: :math:`\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]`.
    - Differentiable wrt atomic coordinates and cell matrix via ``custom_vjp``.

SIESTA Mapping:
    Implements ``RealSpaceGridOps`` replacing ``rhoofd.F90``, ``vmat.F90``,
    ``dfscf.F``, ``poison.F``, and ``mesh.F``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax import ffi as _jax_ffi  # type: ignore[attr-defined]
except ImportError:
    from jax.extend import ffi as _jax_ffi  # type: ignore[no-redef]

from jax_siesta.core.ffi import register_grid_xc_target
from jax_siesta.core.types import BasisData, SparsePattern, Structure


def _n_grid_spin(nspin: int) -> int:
    return 1 if nspin == 1 else (2 if nspin == 2 else 4)


def _smooth_grid_dim(n_min: int, divisor: int = 1) -> int:
    """Round ``n_min`` up to the smallest multiple of ``divisor`` with prime factors in ``{2, 3, 5}``."""
    n = max(divisor, ((n_min + divisor - 1) // divisor) * divisor)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += divisor


MESH_CRITERIA = ("jsiesta", "siesta")


def _shortest_lattice_vector_norm(basis: np.ndarray) -> float:
    """Length of the shortest non-zero vector of the lattice spanned by the columns of ``basis``.

    The basis is first reduced (Gauss-Lagrange sweeps over all column pairs, the role of SIESTA's ``minvec``);
    the minimum is then taken over all integer combinations with coefficients in ``[-2, 2]``, which contains
    the shortest vector of a reduced basis (SIESTA only scans ``[-1, 1]`` of its reduced basis).
    """
    b = np.array(basis, dtype=np.float64)
    changed = True
    while changed:
        changed = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                mu = round(float(b[:, i] @ b[:, j]) / float(b[:, j] @ b[:, j]))
                if mu:
                    cand = b[:, i] - mu * b[:, j]
                    if float(cand @ cand) < float(b[:, i] @ b[:, i]) - 1e-12:
                        b[:, i] = cand
                        changed = True
    best = math.inf
    for c in itertools.product(range(-2, 3), repeat=3):
        if c != (0, 0, 0):
            best = min(best, float(np.linalg.norm(b @ np.asarray(c, dtype=np.float64))))
    return best


def siesta_mesh_cutoff_ry(cell: jax.Array | np.ndarray, mesh_shape: tuple[int, int, int]) -> float:
    r"""Plane-wave cutoff (Ry) that an ``(N1, N2, N3)`` mesh supports without aliasing, as SIESTA's ``chkgmx``.

    Theory:
        The reciprocal lattice of the mesh unit cell :math:`\mathbf a_i / N_i` is spanned by
        :math:`N_i \mathbf b_i`. Densities on the mesh are alias free for
        :math:`|\mathbf G| \le \tfrac12 \min_{\mathbf m \ne 0} |\sum_i m_i N_i \mathbf b_i|`, so
        :math:`G^2_{\text{mesh}}` (in Bohr\ :sup:`-2` = Ry) is the square of that bound.

    Algorithm:
        Builds :math:`\mathbf b_i = 2\pi\,(\mathbf A^{-1})^T` columns, scales them by ``mesh_shape`` and
        finds the shortest non-zero lattice vector after basis reduction. Like SIESTA, it subtracts 1e-8.

    Parameters:
        cell: Lattice matrix :math:`\mathbf A` of shape ``(3, 3)`` (columns in Bohr).
        mesh_shape: Mesh divisions ``(N1, N2, N3)``.

    Returns:
        Largest alias-free mesh cutoff in Rydberg.

    Invariants:
        Equals ``(\pi N / |a|)^2`` for an orthogonal cell with ``N_i`` points along vectors of length ``|a_i|``.

    SIESTA Mapping:
        ``chkgmx`` / ``meshKcut`` in ``Src/chkgmx.f``.
    """
    a_np = np.asarray(cell, dtype=np.float64)
    recip = 2.0 * math.pi * np.linalg.inv(a_np).T
    scaled = recip * np.asarray(mesh_shape, dtype=np.float64)[None, :]
    half_min = 0.5 * _shortest_lattice_vector_norm(scaled)
    return half_min * half_min - 1.0e-8


def _siesta_mesh_dimensions(
    a_np: np.ndarray,
    mesh_cutoff_ry: float,
    nsm: int,
    divisors: tuple[int, int, int],
    equiv_axes: tuple[tuple[int, ...], ...],
) -> tuple[int, int, int]:
    """Port of ``get_mesh_sizes`` in SIESTA ``Src/meshsubs.F`` (see :func:`choose_mesh_dimensions`)."""
    recip = 2.0 * math.pi * np.linalg.inv(a_np).T
    g_max = math.sqrt(max(mesh_cutoff_ry, 1.0e-12))
    ntm = [int(2.0 * g_max / float(np.linalg.norm(recip[:, ax])) + 1.0) for ax in range(3)]
    divs = [math.lcm(nsm, int(d)) for d in divisors]
    for grp in equiv_axes:
        if not grp:
            continue
        req_min = max(ntm[ax] for ax in grp)
        req_div = math.lcm(*(divs[ax] for ax in grp))
        for ax in grp:
            ntm[ax] = req_min
            divs[ax] = req_div
    for _ in range(1000):
        dims = (
            _smooth_grid_dim(ntm[0], divs[0]),
            _smooth_grid_dim(ntm[1], divs[1]),
            _smooth_grid_dim(ntm[2], divs[2]),
        )
        if siesta_mesh_cutoff_ry(a_np, dims) >= mesh_cutoff_ry:
            return dims
        ntm = [d + 1 for d in dims]
    raise RuntimeError("SIESTA-style mesh search did not reach the requested cutoff.")


def choose_mesh_dimensions(
    cell: jax.Array,
    mesh_cutoff_ry: float,
    divisors: tuple[int, int, int] = (1, 1, 1),
    equiv_axes: tuple[tuple[int, ...], ...] = (),
    criterion: str = "siesta",
    mesh_subdivisions: int = 2,
) -> tuple[int, int, int]:
    r"""Determine FFT-friendly, symmetry-commensurate 3D mesh dimensions ``(N1, N2, N3)``.

    Theory:
        Enforces Nyquist resolution for plane-wave cutoff :math:`E_{\text{cut,Ry}} = G_{\max}^2`:
        :math:`N_\alpha \ge \frac{1}{\pi} |\mathbf{a}_\alpha| \sqrt{E_{\text{cut,Ry}}}`.
        When space-group symmetry is enabled, ``N_alpha`` is constrained to be a
        multiple of all fractional-translation denominators along axis :math:`\alpha`
        and equal across symmetry-equivalent lattice vectors.

    Algorithm:
        Computes minimum Nyquist count along each column of ``cell``, enforces
        per-axis divisors and equivalence groups, and rounds up to 5-smooth integers.

    Parameters:
        cell: Lattice matrix :math:`\mathbf{A}` of shape ``(3, 3)`` (columns in Bohr).
        mesh_cutoff_ry: Equivalent kinetic energy mesh cutoff in Rydberg.
        divisors: Required integer divisors ``(d1, d2, d3)`` for commensurate symmetry.
        equiv_axes: Groups of axis indices that must share equal mesh counts.
        criterion: ``"siesta"`` (default; since D-027) follows
            ``get_mesh_sizes``: :math:`N_i = \mathrm{int}(2 G_{\max}/|\mathbf b_i| + 1)` from the reciprocal vectors
            (:math:`|\mathbf b_i| = 2\pi/d_i` with the interplanar spacing :math:`d_i`), rounded to a 5-smooth
            multiple of ``mesh_subdivisions`` (SIESTA ``MeshSubDivisions``), and all :math:`N_i` are increased
            together until :func:`siesta_mesh_cutoff_ry` reaches ``mesh_cutoff_ry``. ``"jsiesta"`` is the legacy criterion
            (Nyquist along each lattice vector, 2/3/5-smooth), kept for reproducing results from before D-027.
        mesh_subdivisions: ``MeshSubDivisions``; only used by the ``"siesta"`` criterion.

    Returns:
        Tuple ``(N1, N2, N3)`` of positive integers.

    Invariants:
        ``N_alpha % divisors[alpha] == 0`` for all :math:`\alpha \in \{0, 1, 2\}`.

    SIESTA Mapping:
        Replaces and upgrades ``chkgmx`` / ``m_mesh_init`` in ``dhscf.F`` and ``mesh.F``.
    """
    a_np = np.asarray(cell, dtype=np.float64)
    if criterion not in MESH_CRITERIA:
        raise ValueError(f"Unsupported mesh criterion {criterion!r}; expected one of {MESH_CRITERIA}.")
    if mesh_subdivisions < 1:
        raise ValueError("mesh_subdivisions must be >= 1.")
    if criterion == "siesta":
        return _siesta_mesh_dimensions(a_np, mesh_cutoff_ry, int(mesh_subdivisions), divisors, equiv_axes)
    g_max = math.sqrt(max(mesh_cutoff_ry, 1.0))
    raw_mins = [
        max(8, int(math.ceil(np.linalg.norm(a_np[:, ax]) * g_max / math.pi)))
        for ax in range(3)
    ]
    divs = list(divisors)
    for grp in equiv_axes:
        if not grp:
            continue
        req_min = max(raw_mins[ax] for ax in grp)
        req_div = math.lcm(*(divs[ax] for ax in grp))
        for ax in grp:
            raw_mins[ax] = req_min
            divs[ax] = req_div
    return (
        _smooth_grid_dim(raw_mins[0], divs[0]),
        _smooth_grid_dim(raw_mins[1], divs[1]),
        _smooth_grid_dim(raw_mins[2], divs[2]),
    )


def _pack_basis_grid_buffers(basis: BasisData, pattern: SparsePattern) -> dict[str, jax.Array]:
    """Pack immutable PAO and CSR arrays into contiguous C++-FFI-ready JAX arrays."""
    pao_tables: list[np.ndarray] = []
    pao_r_max: list[float] = []
    pao_rcut: list[float] = []
    pao_l: list[int] = []
    sp_shell_offsets: list[int] = [0]
    n_pao_pts = int(basis.species[0].pao_r.shape[0])
    for sp in basis.species:
        r_max = float(sp.pao_r[-1])
        vals = np.asarray(sp.pao_values, dtype=np.float64)
        if vals.shape[1] != n_pao_pts:
            r_unif = np.linspace(0.0, r_max, n_pao_pts)
            vals = np.stack([np.interp(r_unif, np.asarray(sp.pao_r), row) for row in vals], axis=0)
        for sh_i in range(vals.shape[0]):
            pao_tables.append(vals[sh_i])
            pao_r_max.append(r_max)
            pao_rcut.append(float(sp.pao_rcut[sh_i]))
            pao_l.append(int(sp.pao_l[sh_i]))
        sp_shell_offsets.append(len(pao_tables))

    orb_g_shell = np.array(
        [sp_shell_offsets[basis.orb_species[mu]] + basis.orb_shell[mu] for mu in range(basis.n_orb)],
        dtype=np.int32,
    )
    return {
        "pao_r_max": jnp.asarray(pao_r_max, dtype=jnp.float64),
        "pao_rcut": jnp.asarray(pao_rcut, dtype=jnp.float64),
        "pao_l": jnp.asarray(pao_l, dtype=jnp.int32),
        "pao_values": jnp.asarray(np.stack(pao_tables, axis=0), dtype=jnp.float64),
        "orb_atom": jnp.asarray(basis.orb_atom, dtype=jnp.int32),
        "orb_g_shell": jnp.asarray(orb_g_shell, dtype=jnp.int32),
        "orb_m": jnp.asarray(basis.orb_m, dtype=jnp.int32),
        "orb_rcut": jnp.asarray(basis.orb_rcut, dtype=jnp.float64),
        "row_ptr": jnp.asarray(pattern.row_ptr, dtype=jnp.int32),
        "dst_unit_orb": jnp.asarray(pattern.dst_unit_orb, dtype=jnp.int32),
        "edge_cell_idx": jnp.asarray(pattern.edge_cell_idx, dtype=jnp.int32),
        "cell_shifts": jnp.asarray(pattern.cell_shifts, dtype=jnp.int32),
    }


def _pack_na_buffers(structure: Structure, basis: BasisData) -> dict[str, jax.Array]:
    n_sp = len(basis.species)
    n_pts = int(basis.species[0].r_grid.shape[0])
    sp_rmax = np.array([float(sp.r_grid[-1]) for sp in basis.species], dtype=np.float64)
    sp_rcut_na = np.array([float(sp.rcut_na) for sp in basis.species], dtype=np.float64)
    sp_rho_na = np.stack([np.asarray(sp.rho_na, dtype=np.float64) for sp in basis.species], axis=0)
    sp_v_na = np.stack([np.asarray(sp.v_na, dtype=np.float64) for sp in basis.species], axis=0)
    sp_rho_core = np.stack([np.asarray(sp.rho_core, dtype=np.float64) for sp in basis.species], axis=0)
    sp_tau_core = np.stack(
        [
            np.asarray(sp.tau_core, dtype=np.float64)
            if sp.tau_core is not None
            else np.zeros_like(sp.rho_core, dtype=np.float64)
            for sp in basis.species
        ],
        axis=0,
    )
    if sp_rho_na.shape != (n_sp, n_pts):
        raise ValueError("All species r_grid arrays must share length for FFI na_grid.")
    return {
        "atom_species": jnp.asarray(structure.species_indices, dtype=jnp.int32),
        "sp_rmax": jnp.asarray(sp_rmax, dtype=jnp.float64),
        "sp_rcut_na": jnp.asarray(sp_rcut_na, dtype=jnp.float64),
        "sp_rho_na": jnp.asarray(sp_rho_na, dtype=jnp.float64),
        "sp_v_na": jnp.asarray(sp_v_na, dtype=jnp.float64),
        "sp_rho_core": jnp.asarray(sp_rho_core, dtype=jnp.float64),
        "sp_tau_core": jnp.asarray(sp_tau_core, dtype=np.float64),
    }


def _orb_grid_scatter_differentiable(
    dm_edges: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    mesh_shape: tuple[int, int, int],
    nspin: int,
    *,
    fwd_target: str,
    bwd_target: str,
) -> jax.Array:
    """Shared differentiable FFI orbital-grid scatter (``rhoofd`` or ``tauofd``)."""
    n1, n2, n3 = mesh_shape
    n_spin_mat = _n_grid_spin(nspin)
    register_grid_xc_target(fwd_target)
    register_grid_xc_target(bwd_target)

    fwd_call = _jax_ffi.ffi_call(
        fwd_target,
        jax.ShapeDtypeStruct((n_spin_mat, n1, n2, n3), jnp.float64),
    )
    bwd_call = _jax_ffi.ffi_call(
        bwd_target,
        (
            jax.ShapeDtypeStruct(dm_edges.shape, jnp.float64),
            jax.ShapeDtypeStruct(positions.shape, jnp.float64),
            jax.ShapeDtypeStruct(cell.shape, jnp.float64),
        ),
    )

    static_args = (
        packed["pao_r_max"],
        packed["pao_rcut"],
        packed["pao_l"],
        packed["pao_values"],
        packed["orb_atom"],
        packed["orb_g_shell"],
        packed["orb_m"],
        packed["orb_rcut"],
        packed["row_ptr"],
        packed["dst_unit_orb"],
        packed["edge_cell_idx"],
        packed["cell_shifts"],
    )

    @jax.custom_vjp
    def _scatter(dm_e: jax.Array, pos: jax.Array, c: jax.Array) -> jax.Array:
        return fwd_call(dm_e, pos, c, *static_args, n1=n1, n2=n2, n3=n3, nspin=nspin)

    def _scatter_fwd(
        dm_e: jax.Array, pos: jax.Array, c: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        return _scatter(dm_e, pos, c), (dm_e, pos, c)

    def _scatter_bwd(
        res: tuple[jax.Array, jax.Array, jax.Array], v_cot: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dm_e, pos, c = res
        return bwd_call(dm_e, v_cot, pos, c, *static_args, n1=n1, n2=n2, n3=n3, nspin=nspin)

    _scatter.defvjp(_scatter_fwd, _scatter_bwd)
    return _scatter(dm_edges, positions, cell)


def rhoofd_differentiable(
    dm_edges: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    mesh_shape: tuple[int, int, int],
    nspin: int,
) -> jax.Array:
    r"""Differentiable C++ XLA FFI ``rhoofd`` scattering ``dm_edges`` to the 3D mesh.

    Theory:
        Evaluates :math:`\rho_s(\mathbf{r}_p) = \sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T}) \phi_\mu(\mathbf{r}_p) \phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})`.

    Algorithm:
        Dispatches to ``_orb_grid_scatter_differentiable`` with forward target
        ``jsiesta_rhoofd_f64`` and custom VJP backward target ``jsiesta_dfscf_grad_f64``.

    Parameters:
        dm_edges: Sparse density-matrix edge values ``(n_spin_dim, n_edges)``.
        positions: Atomic Cartesian coordinates ``(n_atoms, 3)``.
        cell: Lattice vectors as rows ``(3, 3)``.
        packed: Pre-packed species/edge buffer dictionary from ``_pack_basis_grid_buffers``.
        mesh_shape: Uniform real-space grid dimensions ``(N1, N2, N3)``.
        nspin: Spin channel mode (1, 2, 4, or 8).

    Returns:
        Spin density tensor ``(n_rho_comp, N1, N2, N3)`` on the 3D mesh.

    Invariants:
        Total electron count equals :math:`d\Omega \sum_p \rho_{\text{tot}}(\mathbf{r}_p)`.

    SIESTA Mapping:
        Replaces SIESTA ``rhoofd.F90`` and Pulay adjoint ``dfscf.F90``.
    """
    return _orb_grid_scatter_differentiable(
        dm_edges,
        positions,
        cell,
        packed,
        mesh_shape,
        nspin,
        fwd_target="jsiesta_rhoofd_f64",
        bwd_target="jsiesta_dfscf_grad_f64",
    )


def tauofd_differentiable(
    dm_edges: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    mesh_shape: tuple[int, int, int],
    nspin: int,
) -> jax.Array:
    r"""Differentiable C++ XLA FFI ``tauofd`` scattering :math:`\tau_s(\mathbf{r})` to the 3D mesh.

    Theory:
        Evaluates :math:`\tau_s(\mathbf{r}_p) = \frac{1}{2}\sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T}) \nabla\phi_\mu(\mathbf{r}_p) \cdot \nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})`.

    Algorithm:
        Dispatches to ``_orb_grid_scatter_differentiable`` with forward target
        ``jsiesta_tauofd_f64`` and orbital-Hessian VJP target ``jsiesta_tau_dfscf_grad_f64``.

    Parameters:
        dm_edges: Sparse density-matrix edge values ``(n_spin_dim, n_edges)``.
        positions: Atomic Cartesian coordinates ``(n_atoms, 3)``.
        cell: Lattice vectors as rows ``(3, 3)``.
        packed: Pre-packed species/edge buffer dictionary from ``_pack_basis_grid_buffers``.
        mesh_shape: Uniform real-space grid dimensions ``(N1, N2, N3)``.
        nspin: Spin channel mode (1, 2, 4, or 8).

    Returns:
        Orbital kinetic energy density tensor ``(n_tau_comp, N1, N2, N3)`` on the 3D mesh.

    Invariants:
        Matches exact discrete adjoint of ``vmat_tau_differentiable`` modulo voxel volume :math:`d\Omega`.

    SIESTA Mapping:
        Extends SIESTA ``rhoofd.F90`` to meta-GGA kinetic energy density ``tau``.
    """
    return _orb_grid_scatter_differentiable(
        dm_edges,
        positions,
        cell,
        packed,
        mesh_shape,
        nspin,
        fwd_target="jsiesta_tauofd_f64",
        bwd_target="jsiesta_tau_dfscf_grad_f64",
    )


def tau_core_grid_differentiable(
    positions: jax.Array,
    cell: jax.Array,
    na_packed: dict[str, jax.Array],
    mesh_shape: tuple[int, int, int],
) -> jax.Array:
    r"""Differentiable C++ XLA FFI scatter for NLCC :math:`\tau^{\text{core}}(\mathbf{r})`.

    Theory:
        Sums spherical pseudo-core kinetic energy density splines around periodic atoms.

    Algorithm:
        Wraps ``jsiesta_tau_core_f64`` with a custom VJP backed by ``jsiesta_tau_core_grad_f64``.

    Parameters:
        positions: Atomic Cartesian coordinates ``(n_atoms, 3)``.
        cell: Lattice vectors as rows ``(3, 3)``.
        na_packed: Pre-packed neutral-atom/NLCC radial spline dictionary.
        mesh_shape: Uniform real-space grid dimensions ``(N1, N2, N3)``.

    Returns:
        Total core kinetic energy density array of shape ``(N1, N2, N3)``.

    Invariants:
        Zero net force on rigid translation; smooth non-negative values inside core radii.

    SIESTA Mapping:
        Provides meta-GGA NLCC pseudo-core kinetic density analogous to ``rhoat`` in SIESTA.
    """
    n1, n2, n3 = mesh_shape
    register_grid_xc_target("jsiesta_tau_core_f64")
    register_grid_xc_target("jsiesta_tau_core_grad_f64")
    fwd_call = _jax_ffi.ffi_call(
        "jsiesta_tau_core_f64",
        jax.ShapeDtypeStruct((n1, n2, n3), jnp.float64),
    )
    bwd_call = _jax_ffi.ffi_call(
        "jsiesta_tau_core_grad_f64",
        (
            jax.ShapeDtypeStruct(positions.shape, jnp.float64),
            jax.ShapeDtypeStruct(cell.shape, jnp.float64),
        ),
    )
    static_args = (
        na_packed["atom_species"],
        na_packed["sp_rmax"],
        na_packed["sp_tau_core"],
    )

    @jax.custom_vjp
    def _tc(pos: jax.Array, c: jax.Array) -> jax.Array:
        return fwd_call(pos, c, *static_args, n1=n1, n2=n2, n3=n3)

    def _tc_fwd(pos: jax.Array, c: jax.Array):
        return _tc(pos, c), (pos, c)

    def _tc_bwd(res: tuple[jax.Array, jax.Array], w_tc: jax.Array):
        pos, c = res
        return bwd_call(w_tc, pos, c, *static_args, n1=n1, n2=n2, n3=n3)

    _tc.defvjp(_tc_fwd, _tc_bwd)
    return _tc(positions, cell)


def neutral_atom_grid_differentiable(
    positions: jax.Array,
    cell: jax.Array,
    na_packed: dict[str, jax.Array],
    mesh_shape: tuple[int, int, int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Differentiable C++ XLA FFI scatter for :math:`(\rho^{\text{NA}}, V^{\text{NA}}, \rho^{\text{core}})`.

    Theory:
        Evaluates atomic neutral-atom screening densities, short-range local
        potentials, and NLCC pseudo-core densities on the 3D mesh with exact
        analytical derivatives wrt atomic positions and lattice matrix.

    Algorithm:
        Calls ``jsiesta_na_grid_f64`` and ``jsiesta_na_grid_grad_f64`` via ``@jax.custom_vjp``.

    Parameters:
        positions: Atomic positions of shape ``(n_atoms, 3)`` (Bohr).
        cell: Lattice matrix of shape ``(3, 3)`` (Bohr).
        na_packed: Prepacked radial species tables from ``_pack_na_buffers``.
        mesh_shape: Grid dimensions ``(N1, N2, N3)``.

    Returns:
        Tuple ``(rho_na_grid, v_na_grid, rho_core_grid)``, each of shape ``(N1, N2, N3)``.

    Invariants:
        Consistent forward evaluation and VJP gradients for ``positions`` and ``cell``.

    SIESTA Mapping:
        Replaces neutral-atom and pseudo-core grid assembly in ``dhscf.F`` and ``dfscf.F``.
    """
    n1, n2, n3 = mesh_shape
    register_grid_xc_target("jsiesta_na_grid_f64")
    register_grid_xc_target("jsiesta_na_grid_grad_f64")

    fwd_call = _jax_ffi.ffi_call(
        "jsiesta_na_grid_f64",
        (
            jax.ShapeDtypeStruct((n1, n2, n3), jnp.float64),
            jax.ShapeDtypeStruct((n1, n2, n3), jnp.float64),
            jax.ShapeDtypeStruct((n1, n2, n3), jnp.float64),
        ),
    )
    bwd_call = _jax_ffi.ffi_call(
        "jsiesta_na_grid_grad_f64",
        (
            jax.ShapeDtypeStruct(positions.shape, jnp.float64),
            jax.ShapeDtypeStruct(cell.shape, jnp.float64),
        ),
    )
    static_args = (
        na_packed["atom_species"],
        na_packed["sp_rmax"],
        na_packed["sp_rcut_na"],
        na_packed["sp_rho_na"],
        na_packed["sp_v_na"],
        na_packed["sp_rho_core"],
    )

    @jax.custom_vjp
    def _na(pos: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return fwd_call(pos, c, *static_args, n1=n1, n2=n2, n3=n3)

    def _na_fwd(pos: jax.Array, c: jax.Array):
        outs = _na(pos, c)
        return outs, (pos, c)

    def _na_bwd(res: tuple[jax.Array, jax.Array], cotangents: tuple[jax.Array, jax.Array, jax.Array]):
        pos, c = res
        wr, wv, wc = cotangents
        return bwd_call(wr, wv, wc, pos, c, *static_args, n1=n1, n2=n2, n3=n3)

    _na.defvjp(_na_fwd, _na_bwd)
    return _na(positions, cell)


def reciprocal_grid_vectors(mesh_shape: tuple[int, int, int], cell: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Compute 3D ``rfftn`` reciprocal vectors :math:`\mathbf{G}` and :math:`|\mathbf{G}|^2`.

    Theory:
        For lattice columns :math:`\mathbf{A}` and :math:`\mathbf{B} = 2\pi (\mathbf{A}^{-1})^T`,
        the discrete plane-wave wavevector at FFT integer index :math:`\mathbf{m}` is
        :math:`\mathbf{G}_{\mathbf{m}} = \mathbf{B}\mathbf{m}`.

    Algorithm:
        Constructs ``fftfreq`` / ``rfftfreq`` integer frequencies scaled by :math:`N_\alpha`
        and transforms by :math:`\mathbf{B}`.

    Parameters:
        mesh_shape: Grid dimensions ``(N1, N2, N3)``.
        cell: Lattice matrix ``(3, 3)`` (columns in Bohr).

    Returns:
        Tuple ``(g_vecs, g2)`` with shapes ``(N1, N2, N3//2 + 1, 3)`` and ``(N1, N2, N3//2 + 1)``.

    Invariants:
        :math:`|\mathbf{G}|^2 = 0` at index ``(0, 0, 0)`` and strictly positive elsewhere.

    SIESTA Mapping:
        Replaces reciprocal-space loops in ``poison.F`` and ``fft3d``.
    """
    n1, n2, n3 = mesh_shape
    b_mat = 2.0 * jnp.pi * jnp.linalg.inv(cell).T  # columns are b1, b2, b3
    m1 = jnp.fft.fftfreq(n1, d=1.0 / n1)
    m2 = jnp.fft.fftfreq(n2, d=1.0 / n2)
    m3 = jnp.fft.rfftfreq(n3, d=1.0 / n3)
    mm1, mm2, mm3 = jnp.meshgrid(m1, m2, m3, indexing="ij")
    m_grid = jnp.stack([mm1, mm2, mm3], axis=-1)
    g_vecs = jnp.einsum("ai,xyzi->xyza", b_mat, m_grid)
    g2 = jnp.sum(g_vecs**2, axis=-1)
    return g_vecs, g2


def _vmat_target_call(
    target_name: str,
    v_eff_grid: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    n_edges: int,
    nspin: int,
) -> jax.Array:
    n1, n2, n3 = v_eff_grid.shape[1:]
    n_spin_mat = _n_grid_spin(nspin)
    register_grid_xc_target(target_name)
    call = _jax_ffi.ffi_call(
        target_name,
        jax.ShapeDtypeStruct((n_spin_mat, n_edges), jnp.float64),
    )
    return call(
        v_eff_grid,
        positions,
        cell,
        packed["pao_r_max"],
        packed["pao_rcut"],
        packed["pao_l"],
        packed["pao_values"],
        packed["orb_atom"],
        packed["orb_g_shell"],
        packed["orb_m"],
        packed["orb_rcut"],
        packed["row_ptr"],
        packed["dst_unit_orb"],
        packed["edge_cell_idx"],
        packed["cell_shifts"],
        n1=int(n1),
        n2=int(n2),
        n3=int(n3),
        nspin=int(nspin),
    )


def vmat_differentiable(
    v_eff_grid: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    n_edges: int,
    nspin: int,
) -> jax.Array:
    r"""Gather effective grid potential into sparse matrix elements using pre-packed buffers.

    Theory:
        Evaluates :math:`V_{s,\mu\nu}(\mathbf{T}) = d\Omega \sum_p V^{\text{eff}}_s(\mathbf{r}_p) \phi_\mu(\mathbf{r}_p) \phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})`.

    Algorithm:
        Dispatches to ``_vmat_target_call("jsiesta_vmat_f64", ...)`` backed by exact discrete
        adjointness with ``rhoofd_differentiable`` and custom VJP wrt grid potential and geometry.

    Parameters:
        v_eff_grid: Real-space effective potential ``(n_rho_comp, N1, N2, N3)``.
        positions: Atomic Cartesian coordinates ``(n_atoms, 3)``.
        cell: Lattice vectors as rows ``(3, 3)``.
        packed: Pre-packed species/edge buffer dictionary from ``_pack_basis_grid_buffers``.
        n_edges: Number of sparse orbital edges.
        nspin: Spin channel mode (1, 2, 4, or 8).

    Returns:
        Sparse Hamiltonian edge potential ``(n_spin_dim, n_edges)``.

    Invariants:
        Exact linear adjoint of ``rhoofd_differentiable`` weighted by voxel volume :math:`d\Omega`.

    SIESTA Mapping:
        Replaces SIESTA ``vmat.F90``.
    """
    return _vmat_target_call("jsiesta_vmat_f64", v_eff_grid, positions, cell, packed, n_edges, nspin)


def vmat_tau_differentiable(
    v_tau_grid: jax.Array,
    positions: jax.Array,
    cell: jax.Array,
    packed: dict[str, jax.Array],
    n_edges: int,
    nspin: int,
) -> jax.Array:
    r"""Gather meta-GGA :math:`v_{\tau,s}(\mathbf{r})` potential into sparse Hamiltonian edges.

    Theory:
        Evaluates :math:`V^\tau_{s,\mu\nu}(\mathbf{T}) = \frac{1}{2} d\Omega \sum_p v_{\tau,s}(\mathbf{r}_p) \nabla\phi_\mu(\mathbf{r}_p) \cdot \nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})`.

    Algorithm:
        Dispatches to ``_vmat_target_call("jsiesta_vmat_tau_f64", ...)`` reusing orbital
        gradients on compact spherical bounding boxes.

    Parameters:
        v_tau_grid: Real-space meta-GGA potential ``(n_tau_comp, N1, N2, N3)``.
        positions: Atomic Cartesian coordinates ``(n_atoms, 3)``.
        cell: Lattice vectors as rows ``(3, 3)``.
        packed: Pre-packed species/edge buffer dictionary from ``_pack_basis_grid_buffers``.
        n_edges: Number of sparse orbital edges.
        nspin: Spin channel mode (1, 2, 4, or 8).

    Returns:
        Sparse meta-GGA Hamiltonian edge contribution ``(n_spin_dim, n_edges)``.

    Invariants:
        Exact discrete adjoint of ``tauofd_differentiable`` scaled by :math:`d\Omega`.

    SIESTA Mapping:
        Extends SIESTA ``vmat.F90`` to gather meta-GGA :math:`v_\tau` into :math:`H_{\mu\nu}`.
    """
    return _vmat_target_call("jsiesta_vmat_tau_f64", v_tau_grid, positions, cell, packed, n_edges, nspin)


class GridOps:
    r"""Real-space mesh operator suite implementing ``RealSpaceGridOps``.

    Theory:
        Couples compiled C++ XLA FFI bounded-sphere orbital-grid operations with
        pure JAX 3D FFT Poisson electrostatics.

    Algorithm:
        Provides ``density_on_grid``, ``potential_to_matrix``, ``neutral_atom_grid``,
        and ``hartree_potential`` as stateless methods over PyTrees.

    Invariants:
        - Exact discrete adjointness :math:`\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]`.

    SIESTA Mapping:
        Implements ``RealSpaceGridOps`` replacing ``rhoofd.F90``, ``vmat.F90``,
        ``poison.F``, and ``dhscf.F``.
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
        r"""Scatter sparse density matrix onto the 3D real-space mesh via C++ FFI ``rhoofd``.

        Theory:
            Computes :math:`\rho_s(\mathbf{r}_p)` across all spin channels.

        Algorithm:
            Packs basis/CSR buffers and invokes ``rhoofd_differentiable``.

        Parameters:
            dm_edges: Sparse density matrix ``(n_spin_mat, n_edges)``.
            structure: ``Structure`` PyTree.
            basis: ``BasisData`` PyTree.
            pattern: ``SparsePattern`` PyTree.
            mesh_shape: Uniform grid counts ``(N1, N2, N3)``.
            nspin: Spin mode ``1, 2, 4, 8``.

        Returns:
            Grid density array of shape ``(n_grid_spin, N1, N2, N3)``.

        Invariants:
            Total integral :math:`d\Omega \sum_p (\rho_0 + \dots)` equals :math:`\mathrm{Tr}[DS]`.

        SIESTA Mapping:
            Replaces ``rhoofd.F90``.
        """
        packed = _pack_basis_grid_buffers(basis, pattern)
        return rhoofd_differentiable(
            dm_edges, structure.positions, structure.cell, packed, mesh_shape, nspin
        )

    def potential_to_matrix(
        self,
        v_eff_grid: jax.Array,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        nspin: int,
    ) -> jax.Array:
        r"""Gather effective grid potential into sparse matrix elements via C++ FFI ``vmat``.

        Theory:
            Computes :math:`V^{\text{grid}}_{s,\mu\nu}(\mathbf{T}) = d\Omega \sum_p V_s(\mathbf{r}_p)\phi_\mu(\mathbf{r}_p)\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})`.

        Algorithm:
            Dispatches ``jsiesta_vmat_f64`` via JAX XLA FFI.

        Parameters:
            v_eff_grid: Effective potential on mesh ``(n_grid_spin, N1, N2, N3)``.
            structure: ``Structure`` PyTree.
            basis: ``BasisData`` PyTree.
            pattern: ``SparsePattern`` PyTree.
            nspin: Spin mode ``1, 2, 4, 8``.

        Returns:
            Sparse matrix elements of shape ``(n_spin_mat, n_edges)``.

        Invariants:
            Exact discrete adjoint of ``density_on_grid``.

        SIESTA Mapping:
            Replaces ``vmat.F90``.
        """
        packed = _pack_basis_grid_buffers(basis, pattern)
        return vmat_differentiable(
            v_eff_grid, structure.positions, structure.cell, packed, pattern.n_edges, nspin
        )

    def neutral_atom_grid(
        self,
        structure: Structure,
        basis: BasisData,
        mesh_shape: tuple[int, int, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Evaluate :math:`(\rho^{\text{NA}}, V^{\text{NA}}, \rho^{\text{core}})` on the 3D grid.

        Theory:
            Accumulates atomic neutral-atom and NLCC radial tables onto the 3D mesh.

        Algorithm:
            Calls ``neutral_atom_grid_differentiable``.

        Parameters:
            structure: ``Structure`` PyTree.
            basis: ``BasisData`` PyTree.
            mesh_shape: Grid counts ``(N1, N2, N3)``.

        Returns:
            Tuple ``(rho_na, v_na, rho_core)``, each of shape ``(N1, N2, N3)``.

        Invariants:
            Differentiable wrt atomic positions and lattice matrix.

        SIESTA Mapping:
            Replaces neutral-atom and pseudo-core grid assembly in ``dhscf.F``.
        """
        na_packed = _pack_na_buffers(structure, basis)
        return neutral_atom_grid_differentiable(
            structure.positions, structure.cell, na_packed, mesh_shape
        )

    def kinetic_density_on_grid(
        self,
        dm_edges: jax.Array,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        mesh_shape: tuple[int, int, int],
        nspin: int,
    ) -> jax.Array:
        r"""Scatter sparse density matrix onto 3D orbital kinetic energy density ``tau_grid``.

        Theory:
            Assembles :math:`\tau_s(\mathbf{r}) = \frac{1}{2}\sum_{\mu\nu,\mathbf{T}} D_{s,\mu\nu}(\mathbf{T}) \nabla\phi_\mu(\mathbf{r}) \cdot \nabla\phi_\nu(\mathbf{r}-\mathbf{A}\mathbf{T})`.

        Algorithm:
            Packs basis/edge buffers and invokes ``tauofd_differentiable``.

        Parameters:
            dm_edges: Sparse density-matrix edge array.
            structure: Atomic structure and unit cell.
            basis: Atomic orbital radial spline basis data.
            pattern: Sparse neighbor graph pattern.
            mesh_shape: 3D uniform mesh dimensions ``(N1, N2, N3)``.
            nspin: Spin channel mode (1, 2, 4, or 8).

        Returns:
            Real-space orbital kinetic energy density ``(n_tau_comp, N1, N2, N3)``.

        Invariants:
            Symmetric collinear ``dm_edges`` yield real non-negative :math:`\tau_\sigma(\mathbf{r})`.

        SIESTA Mapping:
            Implements the kinetic-energy-density analogue of ``rhoofd.F90``.
        """
        packed = _pack_basis_grid_buffers(basis, pattern)
        return tauofd_differentiable(
            dm_edges, structure.positions, structure.cell, packed, mesh_shape, nspin
        )

    def tau_potential_to_matrix(
        self,
        v_tau_grid: jax.Array,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        nspin: int,
    ) -> jax.Array:
        r"""Gather meta-GGA ``v_tau`` potential into sparse Hamiltonian edges.

        Theory:
            Integrates :math:`\frac{1}{2} d\Omega \sum_p v_{\tau,s}(\mathbf{r}_p) \nabla\phi_\mu(\mathbf{r}_p)\cdot\nabla\phi_\nu(\mathbf{r}_p-\mathbf{A}\mathbf{T})`.

        Algorithm:
            Packs basis/edge buffers and invokes ``vmat_tau_differentiable``.

        Parameters:
            v_tau_grid: Real-space meta-GGA potential ``v_tau`` array.
            structure: Atomic structure and unit cell.
            basis: Atomic orbital radial spline basis data.
            pattern: Sparse neighbor graph pattern.
            nspin: Spin channel mode (1, 2, 4, or 8).

        Returns:
            Sparse meta-GGA Hamiltonian edge array ``(n_spin_dim, n_edges)``.

        Invariants:
            Exact discrete adjoint of ``kinetic_density_on_grid`` modulo voxel volume :math:`d\Omega`.

        SIESTA Mapping:
            Extends ``vmat.F90`` to assemble meta-GGA :math:`v_\tau` matrix elements.
        """
        packed = _pack_basis_grid_buffers(basis, pattern)
        return vmat_tau_differentiable(
            v_tau_grid, structure.positions, structure.cell, packed, pattern.n_edges, nspin
        )

    def tau_core_grid(
        self,
        structure: Structure,
        basis: BasisData,
        mesh_shape: tuple[int, int, int],
    ) -> jax.Array:
        r"""Evaluate spherical NLCC pseudo-core kinetic energy density on the 3D mesh.

        Theory:
            Sums atomic spherical NLCC pseudo-core kinetic energy densities over periodic images.

        Algorithm:
            Packs neutral-atom/NLCC splines and invokes ``tau_core_grid_differentiable``.

        Parameters:
            structure: Atomic structure and unit cell.
            basis: Basis data containing per-species ``tau_core`` splines.
            mesh_shape: 3D uniform mesh dimensions ``(N1, N2, N3)``.

        Returns:
            Total core kinetic energy density array ``(N1, N2, N3)``.

        Invariants:
            Matches analytic spline radial sums within core cutoffs and is zero outside.

        SIESTA Mapping:
            Provides the NLCC pseudo-core kinetic density analogue of ``rhoat.F``.
        """
        na_packed = _pack_na_buffers(structure, basis)
        return tau_core_grid_differentiable(structure.positions, structure.cell, na_packed, mesh_shape)

    def hartree_potential(
        self,
        delta_rho_grid: jax.Array,
        cell: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Solve periodic 3D Poisson equation via ``jnp.fft.rfftn`` for :math:`\delta\rho`.

        Theory:
            Solves :math:`\nabla^2 V_H(\mathbf{r}) = -4\pi \delta\rho(\mathbf{r})` with
            :math:`\mathbf{G}=\mathbf{0}` neutralizing background, returning:

            .. math::
                E_H = \frac{1}{2} d\Omega \sum_p \delta\rho(\mathbf{r}_p) V_H(\mathbf{r}_p), \qquad
                \sigma^H_{\alpha\beta} = \frac{1}{\Omega}\frac{\partial E_H}{\partial \epsilon_{\alpha\beta}}.

        Algorithm:
            Uses real-to-complex 3D FFT ``rfftn``/``irfftn`` and computes exact
            reciprocal-space virial stress tensor.

        Parameters:
            delta_rho_grid: Neutral-atom-subtracted density :math:`\rho - \rho^{\text{NA}}`,
                shape ``(N1, N2, N3)``.
            cell: Lattice column matrix ``(3, 3)`` (Bohr).

        Returns:
            Tuple ``(v_hartree, e_hartree, stress_hartree)``.

        Invariants:
            :math:`\nabla^2 V_H + 4\pi (\delta\rho - \overline{\delta\rho}) = 0` to machine precision.

        SIESTA Mapping:
            Replaces ``poison.F``.
        """
        mesh_shape = delta_rho_grid.shape
        n1, n2, n3 = int(mesh_shape[0]), int(mesh_shape[1]), int(mesh_shape[2])
        vol = jnp.abs(jnp.linalg.det(cell))
        dvol = vol / float(n1 * n2 * n3)
        g_vecs, g2 = reciprocal_grid_vectors((n1, n2, n3), cell)

        dr_k = jnp.fft.rfftn(delta_rho_grid)
        inv_g2 = jnp.where(g2 > 1e-14, 1.0 / jnp.maximum(g2, 1e-14), 0.0)
        vh_k = 4.0 * jnp.pi * dr_k * inv_g2
        v_hartree = jnp.fft.irfftn(vh_k, s=(n1, n2, n3))
        e_hartree = 0.5 * dvol * jnp.sum(delta_rho_grid * v_hartree)

        # Analytical reciprocal stress at fixed fractional charges Q_p = dvol * delta_rho_p:
        # dE_H/d(eps_{ab}) = (2*pi/Omega) * sum_{G!=0} |Q_G|^2 / G^2 * (-delta_{ab} + 2 G_a G_b / G^2)
        # Note Parseval weights for rfftn: interior kz planes (1..n3//2 - 1) have multiplicity 2
        mult = jnp.ones((n1, n2, n3 // 2 + 1), dtype=jnp.float64)
        if n3 > 2:
            end_k = n3 // 2 if (n3 % 2 == 0) else (n3 // 2 + 1)
            mult = mult.at[:, :, 1:end_k].set(2.0)
        q_k = dr_k * dvol
        e_k_density = (2.0 * jnp.pi / vol) * mult * (jnp.abs(q_k) ** 2) * inv_g2
        g_norm_outer = jnp.einsum("xyza,xyzb->xyzab", g_vecs, g_vecs) * (2.0 * inv_g2[..., None, None])
        stress_hartree = (
            jnp.sum(e_k_density[..., None, None] * (g_norm_outer - jnp.eye(3)[None, None, None, :, :]), axis=(0, 1, 2))
            / vol
        )
        return v_hartree, e_hartree, stress_hartree

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience alias disabled; call explicit protocol methods."""
        raise NotImplementedError("Use explicit RealSpaceGridOps methods.")
