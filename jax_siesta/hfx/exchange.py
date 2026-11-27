r"""Exact-exchange stage: matrix and energy on the sparse pattern (NAO2GTO).

Theory:
    For the density matrix :math:`P^s` of spin channel :math:`s` on the edges of the sparse pattern,
    :math:`K^s_{\mu\lambda} = \sum_{\nu\sigma} P^s_{\nu\sigma} (\mu\nu|\lambda\sigma)_w` with the kernel
    :math:`w(r) = \alpha/r + \beta\,\mathrm{erfc}(\omega r)/r` of the hybrid functional. The contribution to the
    Hamiltonian is :math:`H^s_x = -K^s` and to the energy :math:`E_x = -\frac{1}{2}\sum_s \mathrm{Tr}[P^s K^s]`.
    For ``nspin = 1`` the stored density matrix is the sum over spin: :math:`H_x = -\frac{1}{2} K[P]`,
    :math:`E_x = -\frac{1}{4}\mathrm{Tr}[P K[P]]`. In both cases :math:`E_x = \frac{1}{2}\mathrm{Tr}[P H_x]` and
    :math:`\partial E_x / \partial P = H_x`.

    For ``nspin`` 4 and 8 the density matrix has four spin blocks :math:`P^{ss'}` (complex), and
    :math:`H^{ss'}_x = -K[P^{ss'}]`, :math:`E_x = -\frac{1}{2}\sum_{ss'} \mathrm{Re}\,\mathrm{Tr}[P^{ss'\dagger} K[P^{ss'}]]`.
    The integrals are real, so :math:`K` acts on the real and imaginary parts separately. The exchange matrix of
    the diagonal blocks has an imaginary antisymmetric part whenever the density matrix has one, also without
    spin-orbit coupling.

Algorithm:
    The orbitals are fitted with Gaussians (``nao2gto``); the C++ kernel ``jsiesta_hfx_build`` forms the screened
    lattice sums. Density-matrix and exchange-matrix elements outside the pattern are neglected.

Invariants:
    ``H_x`` is linear in the density matrix and Hermitian under edge transposition (to the screening tolerance).

SIESTA Mapping:
    None in SIESTA. HONPAS: ``setup_hfx`` (Shang et al., Comput. Phys. Commun. 254, 107204 (2020)).
"""

from __future__ import annotations

import ctypes

import numpy as np

from jax_siesta.core.ffi import load_ffi_library
from jax_siesta.core.types import BasisData, SparsePattern, Structure
from jax_siesta.hfx.nao2gto import GaussianShell, fit_species
from jax_siesta.xc.functional import ExactExchangeSpec

__all__ = ["Nao2GtoExchange"]

_DPTR = ctypes.POINTER(ctypes.c_double)
_IPTR = ctypes.POINTER(ctypes.c_int32)
_LPTR = ctypes.POINTER(ctypes.c_int64)


def _bind(lib: ctypes.CDLL) -> None:
    lib.jsiesta_hfx_create.restype = ctypes.c_void_p
    lib.jsiesta_hfx_create.argtypes = [
        ctypes.c_int32, _DPTR, _DPTR, ctypes.c_int32, _IPTR, _IPTR, _DPTR, _IPTR, _IPTR, _DPTR, _DPTR,
        ctypes.c_int32, ctypes.c_int64, _IPTR, _IPTR, _IPTR, ctypes.c_int32, _IPTR,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]  # fmt: skip
    lib.jsiesta_hfx_build.restype = ctypes.c_int32
    lib.jsiesta_hfx_build.argtypes = [ctypes.c_void_p, ctypes.c_int32, _DPTR, ctypes.c_double, _DPTR, _LPTR]
    lib.jsiesta_hfx_gradient.restype = ctypes.c_int32
    lib.jsiesta_hfx_gradient.argtypes = [ctypes.c_void_p, ctypes.c_int32, _DPTR, ctypes.c_double, _DPTR, _DPTR, _DPTR, _LPTR]
    lib.jsiesta_hfx_set_distance_screening.restype = None
    lib.jsiesta_hfx_set_distance_screening.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.jsiesta_hfx_destroy.restype = None
    lib.jsiesta_hfx_destroy.argtypes = [ctypes.c_void_p]


def _spinor_real_parts(dm: np.ndarray) -> np.ndarray:
    """Six real matrices ``[Re uu, Re dd, Re ud, Im ud, Im uu, Im dd]`` of a spinor density matrix ``(4, n_edges)``."""
    return np.ascontiguousarray(
        np.stack([dm[0].real, dm[1].real, dm[2].real, dm[3].real, dm[0].imag, dm[1].imag]).astype(np.float64)
    )


def _pauli_real_parts(dm: np.ndarray, transpose: np.ndarray) -> np.ndarray:
    """Eight real matrices of a spinor density matrix, each symmetric or antisymmetric under edge transposition.

    With ``P = (P_0 + sum_a P_a sigma_a) / 2`` every Pauli component is Hermitian in the orbital indices, so its
    real part is symmetric and its imaginary part antisymmetric. The exchange energy is ``-1/4`` times the sum of
    the quadratic forms of the eight parts, which is the form the gradient kernel needs (its sum over the four
    centres of an integral assumes one of the two symmetries).
    """
    r_ud, i_ud = dm[2].real, dm[3].real
    r_t, i_t = r_ud[transpose], i_ud[transpose]
    parts = [
        dm[0].real + dm[1].real, dm[0].imag + dm[1].imag,  # P_0
        dm[0].real - dm[1].real, dm[0].imag - dm[1].imag,  # P_z
        r_ud + r_t, i_ud - i_t,  # P_x = ud + du
        -(i_ud + i_t), r_ud - r_t,  # P_y = i (ud - du)
    ]  # fmt: skip
    return np.ascontiguousarray(np.stack(parts).astype(np.float64))


class Nao2GtoExchange:
    r"""Exact exchange of one geometry from Gaussian fits of the numerical orbitals.

    Theory:
        See the module description.

    Algorithm:
        The constructor fits the species (cached), lists the shells of the unit cell and calls
        ``jsiesta_hfx_create`` (shell pairs, Schwarz bounds, pattern blocks). ``matrix_and_energy`` calls
        ``jsiesta_hfx_build`` and applies the spin factors.

    Parameters:
        structure: Geometry (Bohr).
        basis: Orbital basis.
        pattern: Sparse pattern on which the density matrix and the result live.
        spec: Exact-exchange parameters of the functional.
        n_gauss: Gaussians per shell.
        screen_tol: Quartets with ``Q_IJ Q_KL max|P|`` below this are skipped.
        pair_tol: Shell pairs with a Schwarz bound below this are dropped.
        prim_tol: Primitive pairs with a prefactor below this are dropped.
        shells: Optional explicit Gaussian shells per species (replaces the fit; used by tests).
        incremental: Build the exchange of the change of the density matrix between calls (for a self-consistent cycle).
        max_incremental: Number of incremental steps between two full builds.
        incremental_tol_factor: Factor on ``screen_tol`` in an incremental step.
        distance_screening: For a kernel without full-range part, also bound a quartet by the short-range
            interaction of Gaussian envelopes of its two pair densities.

    Invariants:
        The object is tied to the geometry and pattern it was built for.

    SIESTA Mapping:
        None in SIESTA; HONPAS ``nao2gto`` set-up and ``setup_hfx``.
    """

    def __init__(
        self,
        structure: Structure,
        basis: BasisData,
        pattern: SparsePattern,
        spec: ExactExchangeSpec,
        n_gauss: int = 5,
        screen_tol: float = 1e-7,
        pair_tol: float = 1e-10,
        prim_tol: float = 1e-12,
        shells: tuple[tuple[GaussianShell, ...], ...] | None = None,
        incremental: bool = False,
        max_incremental: int = 8,
        incremental_tol_factor: float = 0.1,
        distance_screening: bool = True,
    ) -> None:
        if not spec.is_hybrid:
            raise ValueError("The functional has no exact exchange.")
        self.spec = spec
        self.screen_tol = float(screen_tol)
        self.incremental = bool(incremental)
        self.max_incremental = int(max_incremental)
        self.incremental_tol_factor = float(incremental_tol_factor)
        self._reference: tuple[np.ndarray, np.ndarray] | None = None
        self._n_incremental = 0
        self.species_shells = shells if shells is not None else tuple(fit_species(sp, n_gauss) for sp in basis.species)
        self.n_edges = int(pattern.src_orb.shape[0])
        self.n_atoms = int(structure.n_atoms)
        self._transpose = np.asarray(pattern.transpose_edge_idx, dtype=np.int64)
        self.last_stats: dict[str, int] = {}

        shell_atom: list[int] = []
        shell_l: list[int] = []
        shell_rcut: list[float] = []
        shell_orb: list[int] = []
        prim_ptr: list[int] = [0]
        exps: list[float] = []
        coefs: list[float] = []
        orb_shell = tuple(int(s) for s in basis.orb_shell)
        for ia, sp_idx in enumerate(structure.species_indices):
            first = int(basis.atom_orb_offsets[ia])
            for sh_idx, gs in enumerate(self.species_shells[sp_idx]):
                orb = first + sum(2 * int(g.l) + 1 for g in self.species_shells[sp_idx][:sh_idx])
                if orb_shell[orb] != sh_idx or int(basis.orb_l[orb]) != gs.l:
                    raise ValueError("Shell order of the basis does not match the species tables.")
                shell_atom.append(ia)
                shell_l.append(gs.l)
                shell_rcut.append(gs.rcut)
                shell_orb.append(orb)
                exps.extend(float(a) for a in gs.exponents)
                coefs.extend(float(c) for c in gs.coefficients)
                prim_ptr.append(len(exps))

        self._lib = load_ffi_library()
        if not hasattr(self._lib, "jsiesta_hfx_set_distance_screening"):
            raise RuntimeError("libjsiesta_ffi is too old for hybrid functionals; rebuild it (make ffi).")
        _bind(self._lib)
        pos = np.ascontiguousarray(np.asarray(structure.positions, dtype=np.float64))
        cell = np.ascontiguousarray(np.asarray(structure.cell, dtype=np.float64))
        a_shell_atom = np.asarray(shell_atom, dtype=np.int32)
        a_shell_l = np.asarray(shell_l, dtype=np.int32)
        a_shell_rcut = np.asarray(shell_rcut, dtype=np.float64)
        a_shell_orb = np.asarray(shell_orb, dtype=np.int32)
        a_prim_ptr = np.asarray(prim_ptr, dtype=np.int32)
        a_exps = np.asarray(exps, dtype=np.float64)
        a_coefs = np.asarray(coefs, dtype=np.float64)
        src = np.ascontiguousarray(np.asarray(pattern.src_orb, dtype=np.int32))
        dst = np.ascontiguousarray(np.asarray(pattern.dst_unit_orb, dtype=np.int32))
        ecell = np.ascontiguousarray(np.asarray(pattern.edge_cell_idx, dtype=np.int32))
        shifts = np.ascontiguousarray(np.rint(np.asarray(pattern.cell_shifts)).astype(np.int32))
        self.n_shells = len(shell_l)
        self._handle = self._lib.jsiesta_hfx_create(
            pos.shape[0], pos.ctypes.data_as(_DPTR), cell.ctypes.data_as(_DPTR), self.n_shells,
            a_shell_atom.ctypes.data_as(_IPTR), a_shell_l.ctypes.data_as(_IPTR), a_shell_rcut.ctypes.data_as(_DPTR),
            a_shell_orb.ctypes.data_as(_IPTR), a_prim_ptr.ctypes.data_as(_IPTR), a_exps.ctypes.data_as(_DPTR),
            a_coefs.ctypes.data_as(_DPTR), int(basis.n_orb), self.n_edges, src.ctypes.data_as(_IPTR),
            dst.ctypes.data_as(_IPTR), ecell.ctypes.data_as(_IPTR), shifts.shape[0], shifts.ctypes.data_as(_IPTR),
            float(spec.alpha), float(spec.beta), float(spec.omega), float(prim_tol), float(pair_tol),
        )  # fmt: skip
        if not self._handle:
            raise RuntimeError("jsiesta_hfx_create failed (angular momentum above the limit of the integral engine?).")
        if not distance_screening:
            self._lib.jsiesta_hfx_set_distance_screening(self._handle, 0)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.jsiesta_hfx_destroy(handle)
            self._handle = None

    def exchange_matrices(self, dm: np.ndarray) -> np.ndarray:
        r"""Bare exchange matrices :math:`K[P]` of density matrices, without sign or spin factor.

        Theory:
            :math:`K_{\mu\lambda} = \sum_{\nu\sigma} P_{\nu\sigma} (\mu\nu|\lambda\sigma)_w`, one per row of ``dm``.

        Algorithm:
            One call of ``jsiesta_hfx_build``; the screening statistics go to ``last_stats``. With
            ``incremental``, the kernel is applied to the difference to the density matrices of the previous call
            and added to the previous result: the screening then acts on the change, which is small near
            self-consistency. A step uses ``incremental_tol_factor`` times the tolerance, and after
            ``max_incremental`` steps the full matrix is built again, so that the neglected terms do not
            accumulate.

        Parameters:
            dm: Real density matrices on the edges, shape ``(n, n_edges)``.

        Returns:
            Array of the same shape.

        Invariants:
            Linear in ``dm`` up to the screening tolerance. An incremental result differs from a full build by
            at most the neglected terms of the steps since the last full build.

        SIESTA Mapping:
            None in SIESTA; HONPAS ``setup_hfx``.
        """
        p = np.ascontiguousarray(np.asarray(dm, dtype=np.float64))
        if p.ndim != 2 or p.shape[1] != self.n_edges:
            raise ValueError(f"Density matrix of shape {p.shape}; expected (n, {self.n_edges}).")
        ref = self._reference
        step = self.incremental and ref is not None and ref[0].shape == p.shape
        step = step and self._n_incremental < self.max_incremental
        arg = np.ascontiguousarray(p - ref[0]) if step and ref is not None else p
        # The terms neglected in a step are not those neglected in a full build, so they act as noise on the
        # self-consistent cycle; a tighter tolerance keeps them below it at little cost (the change is small).
        tol = self.screen_tol * (self.incremental_tol_factor if step else 1.0)
        out = np.zeros_like(p)
        stats = np.zeros(4, dtype=np.int64)
        rc = self._lib.jsiesta_hfx_build(
            self._handle, p.shape[0], arg.ctypes.data_as(_DPTR), tol, out.ctypes.data_as(_DPTR),
            stats.ctypes.data_as(_LPTR),
        )
        if rc != 0:
            raise RuntimeError(f"jsiesta_hfx_build failed with code {rc}.")
        if step and ref is not None:
            out += ref[1]
        self._n_incremental = self._n_incremental + 1 if step else 0
        if self.incremental:
            self._reference = (p.copy(), out.copy())
        self.last_stats = {
            "quartets_computed": int(stats[0]), "quartets_screened": int(stats[1]),
            "shell_pairs": int(stats[2]), "threads": int(stats[3]), "incremental": int(bool(step)),
        }  # fmt: skip
        return out

    def matrix_and_energy(self, dm: np.ndarray, nspin: int) -> tuple[np.ndarray, float]:
        r"""Exchange contribution to the Hamiltonian and its energy for a density matrix.

        Theory:
            ``nspin = 1`` gives :math:`H_x = -\frac{1}{2}K[P]`, ``nspin = 2`` gives :math:`H^s_x = -K[P^s]`, and
            ``nspin`` 4 and 8 give :math:`H^{ss'}_x = -K[P^{ss'}]` for the four spin blocks. In all cases
            :math:`E_x = \frac{1}{2}\sum \mathrm{Re}\,\mathrm{Tr}[P^\dagger H_x]` over the spin blocks.

        Algorithm:
            ``exchange_matrices`` and the spin factor; the trace is the sum over edges. For ``nspin`` 4 and 8 the
            kernel acts on the six real matrices ``Re uu, Re dd, Re ud, Im ud, Im uu, Im dd``; the block ``du``
            is the Hermitian conjugate of ``ud`` and counts through a factor two.

        Parameters:
            dm: Density matrix, shape ``(n_spin_mat, n_edges)``; complex for ``nspin`` 4 and 8, channels
                ``[uu, dd, Re ud, Im ud]``.
            nspin: 1, 2, 4 or 8.

        Returns:
            ``(h_x, e_x)`` in Hartree, ``h_x`` with the shape of ``dm`` (complex for ``nspin`` 4 and 8).

        Invariants:
            ``e_x <= 0`` for a positive kernel and a physical density matrix. A collinear density matrix given in
            the spinor layout returns the ``nspin = 2`` result.

        SIESTA Mapping:
            None in SIESTA; HONPAS adds the same term to ``H`` and to the total energy.
        """
        if nspin in (4, 8):
            d = np.asarray(dm, dtype=np.complex128).reshape(4, self.n_edges)
            p6 = _spinor_real_parts(d)
            k6 = self.exchange_matrices(p6)
            h_x = np.stack([-(k6[0] + 1j * k6[4]), -(k6[1] + 1j * k6[5]), -k6[2] + 0j, -k6[3] + 0j])
            weight = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])[:, None]
            return h_x, -0.5 * float(np.sum(weight * p6 * k6))
        if nspin not in (1, 2):
            raise ValueError(f"Unknown spin mode {nspin}.")
        p = np.asarray(np.real(dm), dtype=np.float64).reshape(nspin, self.n_edges)
        h_x = -(0.5 if nspin == 1 else 1.0) * self.exchange_matrices(p)
        return h_x, 0.5 * float(np.sum(p * h_x))

    def forces_and_strain_derivative(self, dm: np.ndarray, nspin: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Forces and strain derivative of the exchange energy at a fixed density matrix.

        Theory:
            :math:`E_x = -\kappa \sum_s \sum P^s_{\mu\lambda} P^s_{\nu\sigma} (\mu\nu|\lambda\sigma)_w` with
            :math:`\kappa = 1/4` for ``nspin = 1`` (the density matrix is the sum over spin) and :math:`1/2` for
            ``nspin = 2``. The sum is symmetric under the exchange of the two orbitals of each pair and of the two
            pairs, so the four centres of an integral contribute equally: the force on an atom is
            :math:`4\kappa \sum W \partial_{\mathbf{R}_\mu}(\mu\nu|\lambda\sigma)` over the orbitals
            :math:`\mu` of the atom, and :math:`\partial E_x/\partial\epsilon = -\kappa (2 V_1 - V_2)` with the
            two virial sums of ``jsiesta_hfx_gradient``. For ``nspin`` 4 and 8 the energy is written with the
            Pauli components of the density matrix, :math:`E_x = -\frac{1}{4}\sum_c \sum M^c_{\mu\lambda}
            M^c_{\nu\sigma} (\mu\nu|\lambda\sigma)_w` over their real (symmetric) and imaginary (antisymmetric)
            parts :math:`M^c`, for which the same sum over centres holds. The Gaussian fits move rigidly with the atoms. The part
            of the force that comes from the change of the orbitals' overlap is in the energy-density-matrix
            term of the caller, as for every other contribution to the Hamiltonian.

        Algorithm:
            One call of ``jsiesta_hfx_gradient`` (derivative integrals of the bra pair, contracted on the fly).

        Parameters:
            dm: Density matrix, shape ``(n_spin_mat, n_edges)`` (complex for ``nspin`` 4 and 8).
            nspin: 1, 2, 4 or 8.

        Returns:
            ``(forces, de_deps)``: forces in Hartree/Bohr, shape ``(n_atoms, 3)``, and the derivative of the
            energy with respect to a homogeneous strain in Hartree, shape ``(3, 3)`` (divide by the volume for the
            stress).

        Invariants:
            The forces sum to zero to the screening tolerance. ``de_deps`` is symmetric for a density matrix
            that transforms with the orbitals under rotation (any self-consistent one), not for an arbitrary one.

        SIESTA Mapping:
            None in SIESTA; the HFX forces of HONPAS.
        """
        if nspin in (4, 8):
            p = _pauli_real_parts(np.asarray(dm, dtype=np.complex128).reshape(4, self.n_edges), self._transpose)
        elif nspin in (1, 2):
            p = np.ascontiguousarray(np.asarray(np.real(dm), dtype=np.float64).reshape(nspin, self.n_edges))
        else:
            raise ValueError(f"Unknown spin mode {nspin}.")
        d_first = np.zeros((self.n_atoms, 3))
        v_first, v_pair = np.zeros((3, 3)), np.zeros((3, 3))
        stats = np.zeros(4, dtype=np.int64)
        rc = self._lib.jsiesta_hfx_gradient(
            self._handle, p.shape[0], p.ctypes.data_as(_DPTR), self.screen_tol, d_first.ctypes.data_as(_DPTR),
            v_first.ctypes.data_as(_DPTR), v_pair.ctypes.data_as(_DPTR), stats.ctypes.data_as(_LPTR),
        )
        if rc != 0:
            raise RuntimeError(f"jsiesta_hfx_gradient failed with code {rc}.")
        self.last_stats = {
            "quartets_computed": int(stats[0]), "quartets_screened": int(stats[1]),
            "shell_pairs": int(stats[2]), "threads": int(stats[3]),
        }  # fmt: skip
        kappa = 0.5 if nspin == 2 else 0.25
        return 4.0 * kappa * d_first, -kappa * (2.0 * v_first - v_pair)
