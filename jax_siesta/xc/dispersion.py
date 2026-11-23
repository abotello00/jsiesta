r"""Van der Waals dispersion models: DFT-D3 via simple-dftd3 and 3D-FFT rVV10.

Theory:
    Implements two dispersion models behind the ``DispersionModel`` protocol:

    1. **DFT-D3 (Grimme et al., JCP 132, 154104 (2010); BJ damping, JCC 32, 1456 (2011)):**
       :math:`E = -\sum_{n=6,8} s_n \sum_{I<J} C_n^{IJ}(CN_I, CN_J) / (R_{IJ}^n + f(R_0^{IJ})^n)`
       plus the Axilrod–Teller–Muto three-body term, with :math:`C_6` interpolated from Grimme's
       reference tables and :math:`C_8 = 3 C_6 \sqrt{Q_I Q_J}`. Evaluated by the reference
       implementation ``simple-dftd3`` (S. Ehlert, Python package ``dftd3``), not re-implemented.
    2. **rVV10 (Sabatini et al., PRB 87, 041108(R) 2013):** nonlocal correlation in 3D Fourier
       space (see :class:`RVV10Dispersion`).

Algorithm:
    DFT-D3 is a function of positions and cell only: the energy, analytic gradient and strain
    derivative come from ``simple-dftd3`` on the host. rVV10 is a pure JAX functional of the
    density supporting ``jax.grad``.

Invariants:
    - Single isolated atoms yield zero pairwise DFT-D3 energy.
    - DFT-D3 forces obey the acoustic sum rule; the stress is symmetric.

SIESTA Mapping:
    DFT-D3 matches SIESTA's ``DFTD3 T`` (SIESTA links the same ``s-dftd3`` library);
    rVV10 replaces the ``libgridxc`` nonlocal vdW path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import Structure
from jax_siesta.grid.mesh import reciprocal_grid_vectors
from jax_siesta.xc.functional import _fft_gradient_3d

#: jsiesta XC labels -> simple-dftd3 method names (parameters from Grimme's published tables).
_D3_METHOD_ALIASES: dict[str, str] = {
    "PBE": "pbe",
    "GGA": "pbe",
    "PBESOL": "pbesol",
    "REVPBE": "revpbe",
    "RPBE": "rpbe",
    "BLYP": "blyp",
    "R2SCAN": "r2scan",
    "R2SCAN01": "r2scan",
    "SCAN": "scan",
    "TPSS": "tpss",
}


def d3_method_for_functional(xc_functional: str) -> str:
    """Map a jsiesta XC label to the ``simple-dftd3`` damping-parameter method name.

    Theory:
        D3 damping parameters are fitted per density functional; using the wrong set changes
        the dispersion energy by tens of percent.

    Algorithm:
        Upper-case lookup in ``_D3_METHOD_ALIASES``; otherwise the lower-cased label is passed
        through so ``simple-dftd3`` can resolve or reject it.

    Parameters:
        xc_functional: jsiesta functional label (e.g. ``"PBE"``, ``"R2SCAN"``).

    Returns:
        ``simple-dftd3`` method name.

    Raises:
        ValueError: For LDA functionals, which have no D3 parametrisation.

    Invariants:
        Deterministic and case-insensitive.

    SIESTA Mapping:
        SIESTA derives the same choice from ``XC.authors`` in ``m_dftd3``.
    """
    key = xc_functional.strip().upper()
    if key in ("LDA", "PW92", "CA", "PZ"):
        raise ValueError(f"DFT-D3 has no damping parameters for the LDA functional {xc_functional!r}.")
    return _D3_METHOD_ALIASES.get(key, key.lower())


class DFTD3Dispersion:
    """Grimme DFT-D3 with Becke–Johnson damping, evaluated by ``simple-dftd3``.

    Theory:
        Coordination-number–dependent :math:`C_6`/:math:`C_8` pair terms with rational (BJ)
        damping and the ATM three-body term, as published (see module docstring).

    Algorithm:
        Builds ``dftd3.interface.DispersionModel`` with Bohr coordinates and lattice rows,
        loads ``RationalDampingParam(method=...)`` and requests the analytic gradient and
        virial. Forces are :math:`-\\partial E/\\partial \\mathbf R`; stress is
        :math:`\\Omega^{-1}\\partial E/\\partial\\epsilon`, the convention used by the other
        jsiesta stress terms.

    Invariants:
        - Acoustic sum rule :math:`\\sum_I \\mathbf{F}_I^{\\text{D3}} = \\mathbf{0}`.
        - Results equal ``simple-dftd3`` to machine precision (no re-implementation).

    SIESTA Mapping:
        Same library and parameters as SIESTA's ``DFTD3 T`` (``Src/m_dftd3.F90``).
    """

    def __init__(
        self,
        functional: str = "PBE",
        three_body: bool = True,
        params: dict[str, float] | None = None,
        cutoff_2b_bohr: float = 60.0,
        cutoff_3b_bohr: float = 40.0,
        cutoff_cn_bohr: float = 40.0,
    ) -> None:
        """Configure the D3(BJ) model.

        Parameters:
            functional: XC label used to load the published damping parameters.
            three_body: Include the ATM three-body term (``s9 = 1``).
            params: Explicit ``s6, s8, a1, a2`` (and optionally ``s9``, ``alp``) overriding the
                published set; used to reproduce custom SIESTA ``DFTD3.*`` inputs.
            cutoff_2b_bohr: Two-body real-space cutoff (library and SIESTA default 60 Bohr).
            cutoff_3b_bohr: Three-body real-space cutoff (default 40 Bohr).
            cutoff_cn_bohr: Coordination-number cutoff (library default 40 Bohr; SIESTA's
                ``DFTD3.CoordinationCutoff`` default is 10 Bohr).
        """
        self.functional = functional
        self.method = d3_method_for_functional(functional)
        self.three_body = three_body
        self.params = dict(params) if params is not None else None
        self.cutoffs = (float(cutoff_2b_bohr), float(cutoff_3b_bohr), float(cutoff_cn_bohr))

    def _run(self, structure: Structure) -> dict[str, np.ndarray]:
        try:
            from dftd3.interface import DispersionModel, RationalDampingParam
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
            raise ImportError(
                "DFT-D3 needs the optional 'dftd3' package (simple-dftd3): pip install dftd3"
            ) from exc
        numbers = np.asarray(structure.atomic_numbers, dtype=np.int32)
        positions = np.asarray(structure.positions, dtype=np.float64)
        lattice_rows = np.asarray(structure.cell, dtype=np.float64).T
        periodic = np.asarray(structure.pbc, dtype=bool)
        model = DispersionModel(numbers, positions, lattice_rows, periodic=periodic)
        model.set_realspace_cutoff(*self.cutoffs)
        if self.params is not None:
            explicit = {"s9": 1.0 if self.three_body else 0.0, **self.params}
            param = RationalDampingParam(**explicit)
        else:
            param = RationalDampingParam(method=self.method, atm=self.three_body)
        return model.get_dispersion(param, grad=True)

    def energy(self, structure: Structure) -> jax.Array:
        r"""Compute the DFT-D3(BJ) dispersion energy in Hartree.

        Theory:
            See :class:`DFTD3Dispersion`.

        Algorithm:
            Calls ``simple-dftd3`` and returns its energy.

        Parameters:
            structure: ``Structure`` PyTree with concrete (non-traced) arrays.

        Returns:
            Scalar energy in Hartree.

        Invariants:
            Invariant under rigid translations and lattice-vector relabelling.

        SIESTA Mapping:
            ``dftd3`` energy in SIESTA's ``m_dftd3``.
        """
        return jnp.asarray(self._run(structure)["energy"], dtype=jnp.float64)

    def evaluate(
        self,
        structure: Structure,
        rho_grid: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Return ``(e_disp, forces_disp, stress_disp)`` for DFT-D3(BJ) in atomic units.

        Theory:
            Analytic position and strain derivatives of the D3 energy.

        Algorithm:
            One ``simple-dftd3`` call with ``grad=True``.

        Parameters:
            structure: ``Structure`` PyTree with concrete arrays.
            rho_grid: Unused for pairwise DFT-D3.

        Returns:
            Tuple ``(e_disp_ha, forces_ha_per_bohr, stress_ha_per_bohr3)``.

        Invariants:
            :math:`\sum_I \mathbf{F}_I = \mathbf{0}` and symmetric stress tensor.

        SIESTA Mapping:
            Implements ``DispersionModel.evaluate``.
        """
        del rho_grid
        res = self._run(structure)
        vol = abs(float(np.linalg.det(np.asarray(structure.cell, dtype=np.float64))))
        e_disp = jnp.asarray(res["energy"], dtype=jnp.float64)
        forces = -jnp.asarray(res["gradient"], dtype=jnp.float64)
        stress = jnp.asarray(res["virial"], dtype=jnp.float64) / vol
        return e_disp, forces, stress


#: rVV10 q-mesh (Sabatini et al. 2013 interpolation points; same numerical parameters as Quantum ESPRESSO's
#: ``xc_rVV10``, so the two implementations can be compared term by term).
RVV10_Q_MIN = 1.0e-4
RVV10_Q_CUT = 0.5
RVV10_Q_MESH = np.array(
    [
        RVV10_Q_MIN, 3.0e-4, 5.893850845618885e-4, 1.008103720396345e-3, 1.613958359589310e-3,
        2.490584839564653e-3, 3.758997979748929e-3, 5.594297198907115e-3, 8.249838297569416e-3,
        1.209220822453922e-2, 1.765183095571029e-2, 2.569619042667097e-2, 3.733577865542191e-2,
        5.417739477463518e-2, 7.854595729872216e-2, 0.113805449932145, 0.164823306218807,
        0.238642339497217, 0.345452975434964, RVV10_Q_CUT,
    ],
    dtype=np.float64,
)
_RVV10_M_CUT = 12
_RVV10_RHO_EPS = 1.0e-12


def rvv10_phi(q1: np.ndarray, q2: np.ndarray, r: np.ndarray) -> np.ndarray:
    r"""Reduced rVV10 kernel :math:`\phi(q, q', R)` (Hartree units).

    Theory:
        Sabatini et al., PRB 87, 041108(R) (2013), Eq. (5): with :math:`g = \kappa (q R^2 + 1)` and
        the approximation :math:`\kappa(qR^2+1) + \kappa'(q'R^2+1) \approx \sqrt{\kappa\kappa'}(qR^2+q'R^2+2)`,

        .. math:: \Phi = \frac{1}{(\kappa\kappa')^{3/2}}\,\phi, \qquad
                  \phi = -\frac{3}{2}\,\frac{1}{(qR^2+1)(q'R^2+1)(qR^2+q'R^2+2)}.

    Algorithm:
        Direct evaluation with NumPy broadcasting.

    Parameters:
        q1: :math:`q` values.
        q2: :math:`q'` values.
        r: Distances in Bohr.

    Returns:
        :math:`\phi` (dimensionless; energy follows from :math:`\theta = n/\kappa^{3/2}`).

    Invariants:
        Symmetric in ``q1, q2``; :math:`\phi(R=0) = -3/4`; :math:`\phi \sim -3/(2 q q'(q+q') R^6)` for large R.

    SIESTA Mapping:
        No SIESTA equivalent (SIESTA's ``libgridxc`` implements VV10, not rVV10).
    """
    d1 = q1 * r**2
    d2 = q2 * r**2
    return -1.5 / ((d1 + 1.0) * (d2 + 1.0) * (d1 + d2 + 2.0))


def _natural_spline_cardinal_y2(x: np.ndarray) -> np.ndarray:
    """Second derivatives ``y2[i, a]`` at node ``i`` of the natural cubic spline through unit vector ``e_a``."""
    n = x.size
    h = np.diff(x)
    mat = np.zeros((n, n))
    rhs = np.zeros((n, n))
    mat[0, 0] = mat[-1, -1] = 1.0
    for i in range(1, n - 1):
        mat[i, i - 1] = h[i - 1] / 6.0
        mat[i, i] = (h[i - 1] + h[i]) / 3.0
        mat[i, i + 1] = h[i] / 6.0
        rhs[i, i + 1] += 1.0 / h[i]
        rhs[i, i] -= 1.0 / h[i] + 1.0 / h[i - 1]
        rhs[i, i - 1] += 1.0 / h[i - 1]
    return np.linalg.solve(mat, rhs)


def rvv10_kernel_table(
    q_mesh: np.ndarray = RVV10_Q_MESH, r_max: float = 100.0, n_r: int = 1024
) -> tuple[np.ndarray, float]:
    r"""Tabulate the 3D Fourier transform :math:`\hat\phi_{\alpha\beta}(k)` of the rVV10 kernel.

    Theory:
        :math:`\hat\phi(k) = 4\pi\int_0^{\infty} R^2 \phi(R)\,\frac{\sin kR}{kR}\,dR`, truncated at
        ``r_max``.

    Algorithm:
        Trapezoidal rule on :math:`R_i = i\,\Delta R`, :math:`\Delta R = r_{\max}/n_r`, evaluated at
        :math:`k_j = j\,\Delta k`, :math:`\Delta k = 2\pi/r_{\max}`, :math:`j = 0..n_r`.

    Parameters:
        q_mesh: Interpolation nodes.
        r_max: Radial cut-off in Bohr.
        n_r: Number of radial intervals.

    Returns:
        ``(table, dk)`` with ``table`` of shape ``(n_r + 1, Nq, Nq)``.

    Invariants:
        Symmetric in the last two indices. For large ``q`` (no truncation) the ``k = 0`` value equals
        the analytic :math:`-3\pi^2/(16 q^{3/2})` when :math:`q = q'`.

    SIESTA Mapping:
        No SIESTA equivalent; numerical parameters match Quantum ESPRESSO ``xc_rVV10``.
    """
    dr = r_max / n_r
    dk = 2.0 * np.pi / r_max
    r = dr * np.arange(n_r + 1)
    k = dk * np.arange(n_r + 1)
    w = np.full(n_r + 1, dr)
    w[0] = w[-1] = 0.5 * dr
    kr = np.outer(k, r)
    sinc = np.where(kr > 0.0, np.sin(kr) / np.where(kr > 0.0, kr, 1.0), 1.0)
    weights = 4.0 * np.pi * sinc * (w * r**2)[None, :]  # (n_k, n_r)
    nq = q_mesh.size
    phi_r = rvv10_phi(q_mesh[:, None, None], q_mesh[None, :, None], r[None, None, :])  # (Nq, Nq, n_r)
    table = np.einsum("kr,abr->kab", weights, phi_r)
    assert table.shape == (n_r + 1, nq, nq)
    return table, dk


def _uniform_natural_spline_y2(table: np.ndarray, dx: float) -> np.ndarray:
    """Natural cubic-spline second derivatives along axis 0 of ``table`` on a uniform grid."""
    n = table.shape[0]
    mat = np.zeros((n, n))
    rhs = np.zeros_like(table)
    mat[0, 0] = mat[-1, -1] = 1.0
    for i in range(1, n - 1):
        mat[i, i - 1] = dx / 6.0
        mat[i, i] = 2.0 * dx / 3.0
        mat[i, i + 1] = dx / 6.0
        rhs[i] = (table[i + 1] - 2.0 * table[i] + table[i - 1]) / dx
    flat = np.linalg.solve(mat, rhs.reshape(n, -1))
    return flat.reshape(table.shape)


class RVV10Dispersion:
    r"""rVV10 nonlocal correlation (Sabatini, Gorni, de Gironcoli, PRB 87, 041108(R) 2013).

    Theory:
        .. math::
            E_c^{\mathrm{nl}} = \int n(\mathbf r)\Big[\beta + \tfrac12\int n(\mathbf r')
            \Phi(\mathbf r,\mathbf r')\,d\mathbf r'\Big] d\mathbf r,\quad
            \beta = \tfrac{1}{32}\big(3/b^2\big)^{3/4},

        with :math:`\Phi = \phi(q,q',R)/(\kappa\kappa')^{3/2}` (see :func:`rvv10_phi`),
        :math:`q = \omega_0/\kappa`, :math:`\omega_0 = \sqrt{\omega_g^2 + \omega_p^2/3}`,
        :math:`\omega_p^2 = 4\pi n`, :math:`\omega_g^2 = C|\nabla n/n|^4`,
        :math:`\kappa = b\,\tfrac{3\pi}{2}\,(n/9\pi)^{1/6}` (Hartree atomic units; b = 6.3, C = 0.0093).
        :math:`q` is saturated smoothly at :math:`q_{\mathrm{cut}}` and bounded below by :math:`q_{\min}`.

    Algorithm:
        Román-Pérez–Soler factorisation: :math:`\theta_\alpha = (n/\kappa^{3/2})\,p_\alpha(q_0)` with
        natural cubic-spline cardinal functions :math:`p_\alpha` on the q-mesh, so that
        :math:`E_{\mathrm{nl}} = \tfrac{1}{2\Omega}\sum_{\mathbf G}\sum_{\alpha\beta}
        \tilde\theta_\alpha^*\,\hat\phi_{\alpha\beta}(|\mathbf G|)\,\tilde\theta_\beta`
        (:math:`\tilde\theta = \int_\Omega \theta e^{-i\mathbf G\cdot\mathbf r}`). The kernel table is
        interpolated in :math:`|\mathbf G|` with natural cubic splines. Density gradients use FFTs.
        Potential and stress are exact derivatives of this discrete energy (``jax.grad``).

    Invariants:
        - ``V = dE/dn`` exactly (autodiff of the energy).
        - Energy is invariant under rigid translations of the density.

    SIESTA Mapping:
        No SIESTA rVV10 (``libgridxc`` provides VV10). Cross-checked against Quantum ESPRESSO.
    """

    def __init__(
        self,
        b: float = 6.3,
        c_param: float = 0.0093,
        q_mesh: np.ndarray = RVV10_Q_MESH,
        r_max: float = 100.0,
        n_r: int = 1024,
    ) -> None:
        self.b = float(b)
        self.c_param = float(c_param)
        self.beta = (1.0 / 32.0) * (3.0 / self.b**2) ** 0.75
        self.q_mesh_np = np.asarray(q_mesh, dtype=np.float64)
        self.q_cut = float(self.q_mesh_np[-1])
        self.q_min = float(self.q_mesh_np[0])
        self.q_mesh = jnp.asarray(self.q_mesh_np)
        self._q_y2 = jnp.asarray(_natural_spline_cardinal_y2(self.q_mesh_np))
        table, dk = rvv10_kernel_table(self.q_mesh_np, r_max=r_max, n_r=n_r)
        self.dk = float(dk)
        self.k_max = float(dk * n_r)
        # Natural cubic spline of the kernel along k (uniform grid): store values and second derivatives.
        self._k_vals = jnp.asarray(table)
        self._k_y2 = jnp.asarray(_uniform_natural_spline_y2(table, dk))

    # ------------------------------------------------------------------ helpers
    def q0_and_theta(self, rho: jax.Array, cell: jax.Array) -> tuple[jax.Array, jax.Array]:
        r"""Return the saturated ``q0`` and ``theta = n / kappa^{3/2}`` on the grid.

        Theory:
            :math:`q = \omega_0/\kappa`, saturated as :math:`q_0 = q_c(1 - e^{-\sum_{m=1}^{12}(q/q_c)^m/m})` and
            bounded below by :math:`q_{\min}`; :math:`\theta = n/\kappa^{3/2}` (Hartree units).

        Algorithm:
            FFT density gradient, pointwise formulas, masking of points with :math:`n \le 10^{-12}`.

        Parameters:
            rho: Total density ``(N1, N2, N3)``.
            cell: Lattice column matrix ``(3, 3)`` in Bohr.

        Returns:
            ``(q0, theta)`` grids; ``theta = 0`` and ``q0 = q_cut`` where the density is negligible.

        Invariants:
            ``q_min <= q0 <= q_cut`` everywhere.

        SIESTA Mapping:
            None (QE ``get_q0_on_grid`` / ``get_thetas_on_grid`` equivalent).
        """
        n1, n2, n3 = (int(x) for x in rho.shape)
        g_vecs, _ = reciprocal_grid_vectors((n1, n2, n3), cell)
        grad = _fft_gradient_3d(rho, g_vecs)
        grad2 = jnp.sum(grad**2, axis=0)
        mask = rho > _RVV10_RHO_EPS
        n_safe = jnp.where(mask, rho, 1.0)
        wp2 = 4.0 * jnp.pi * n_safe
        wg2 = self.c_param * (grad2 / n_safe**2) ** 2
        omega0 = jnp.sqrt(wg2 + wp2 / 3.0)
        kappa = self.b * 1.5 * jnp.pi * (n_safe / (9.0 * jnp.pi)) ** (1.0 / 6.0)
        q = omega0 / kappa
        ratio = q / self.q_cut
        expo = sum(ratio**m / m for m in range(1, _RVV10_M_CUT + 1))
        q0 = self.q_cut * (1.0 - jnp.exp(-expo))
        q0 = jnp.where(mask, jnp.maximum(q0, self.q_min), self.q_cut)
        theta = jnp.where(mask, n_safe / kappa**1.5, 0.0)
        return q0, theta

    def _q_weights(self, q0: jax.Array) -> jax.Array:
        """Natural cubic-spline cardinal weights ``p_a(q0)`` with shape ``(Nq, *q0.shape)``."""
        x = self.q_mesh
        j = jnp.clip(jnp.searchsorted(x, q0, side="right") - 1, 0, x.shape[0] - 2)
        h = x[j + 1] - x[j]
        a = (x[j + 1] - q0) / h
        bb = 1.0 - a
        nq = x.shape[0]
        eye = jnp.eye(nq, dtype=jnp.float64)
        w = (
            a[..., None] * eye[j]
            + bb[..., None] * eye[j + 1]
            + ((a**3 - a)[..., None] * self._q_y2[j] + (bb**3 - bb)[..., None] * self._q_y2[j + 1])
            * (h**2)[..., None]
            / 6.0
        )
        return jnp.moveaxis(w, -1, 0)

    def _apply_kernel(self, gmod: jax.Array, theta_k: jax.Array) -> jax.Array:
        """Return ``u_a(G) = sum_b phi_ab(|G|) theta_b(G)`` (shape of ``theta_k``); kernel is 0 beyond ``k_max``.

        The kernel is cubic-spline interpolated in ``|G|`` row by row (``lax.map`` over ``a``) so that
        the ``(Nq, Nq, N_G)`` table is never materialised.
        """
        n_int = self._k_vals.shape[0] - 1
        u = gmod / self.dk
        i = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_int - 1)
        a = (i + 1).astype(jnp.float64) - u
        bb = 1.0 - a
        h2 = self.dk**2 / 6.0
        inside = gmod <= self.k_max
        ca = jnp.where(inside, a, 0.0)
        cb = jnp.where(inside, bb, 0.0)
        c2a = jnp.where(inside, (a**3 - a) * h2, 0.0)
        c2b = jnp.where(inside, (bb**3 - bb) * h2, 0.0)

        def _row(alpha: jax.Array) -> jax.Array:
            kv = self._k_vals[:, alpha, :]  # (n_k, Nq)
            ky = self._k_y2[:, alpha, :]
            phi = (
                ca[..., None] * kv[i]
                + cb[..., None] * kv[i + 1]
                + c2a[..., None] * ky[i]
                + c2b[..., None] * ky[i + 1]
            )  # (N1, N2, N3h, Nq)
            return jnp.einsum("ijkb,bijk->ijk", phi, theta_k)

        return jax.lax.map(_row, jnp.arange(self._k_vals.shape[1]))

    # ------------------------------------------------------------------ public API
    def energy_from_density(self, rho_tot: jax.Array, cell: jax.Array) -> jax.Array:
        r"""Compute the rVV10 energy (Hartree) of the total density ``rho_tot``.

        Theory:
            :math:`E = E_{\mathrm{nl}} + \beta\int n` (see class docstring).

        Algorithm:
            :math:`q_0`, :math:`\theta`, spline weights, real FFT of the :math:`N_q` fields,
            contraction with the interpolated kernel over the half reciprocal grid.

        Parameters:
            rho_tot: Total density ``(N1, N2, N3)`` in electrons/Bohr³ (valence + core, as for XC).
            cell: Lattice column matrix ``(3, 3)`` in Bohr.

        Returns:
            Scalar energy in Hartree.

        Invariants:
            Differentiable with respect to ``rho_tot`` and ``cell``.

        SIESTA Mapping:
            None (QE ``xc_rVV10`` / ``vdW_energy`` equivalent).
        """
        n1, n2, n3 = (int(x) for x in rho_tot.shape)
        vol = jnp.abs(jnp.linalg.det(cell))
        dvol = vol / float(n1 * n2 * n3)
        q0, theta = self.q0_and_theta(rho_tot, cell)
        theta_a = theta[None, ...] * self._q_weights(q0)
        theta_k = jnp.fft.rfftn(theta_a, axes=(1, 2, 3)) * dvol
        _, g2 = reciprocal_grid_vectors((n1, n2, n3), cell)
        gmod = jnp.sqrt(jnp.where(g2 > 0.0, g2, 1.0)) * (g2 > 0.0)
        mult = jnp.ones((n1, n2, n3 // 2 + 1), dtype=jnp.float64)
        if n3 > 1:
            end_k = n3 // 2 if (n3 % 2 == 0) else (n3 // 2 + 1)
            mult = mult.at[:, :, 1:end_k].set(2.0)
        u_k = self._apply_kernel(gmod, theta_k)
        e_nl = (0.5 / vol) * jnp.sum(mult[None, ...] * jnp.real(jnp.conj(theta_k) * u_k))
        return e_nl + self.beta * dvol * jnp.sum(rho_tot)

    def potential_and_energy(
        self, rho_tot: jax.Array, cell: jax.Array, compute_stress: bool = True
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Return ``(e_rvv10, v_rvv10_grid, stress_rvv10)`` via exact ``jax.grad``.

        Theory:
            Computes :math:`V_{\text{rVV10}}(\mathbf{r}) = \frac{1}{d\Omega}\frac{\partial E}{\partial \rho(\mathbf{r})}`
            and mass-conserving strain stress :math:`\sigma_{\alpha\beta}`.

        Algorithm:
            Calls ``jax.value_and_grad`` wrt ``rho_tot`` and affine strain tensor.

        Parameters:
            rho_tot: Total density ``(N1, N2, N3)`` (valence + core).
            cell: Lattice column matrix ``(3, 3)`` (Bohr).
            compute_stress: Skip the strain derivative (zeros returned) inside the SCF loop.

        Returns:
            Tuple ``(e_rvv10_ha, v_rvv10_grid, stress_ha_per_bohr3)``.

        Invariants:
            Exact functional derivative consistency.

        SIESTA Mapping:
            Replaces ``libgridxc`` rVV10 potential and stress calculation.
        """
        n1, n2, n3 = int(rho_tot.shape[0]), int(rho_tot.shape[1]), int(rho_tot.shape[2])
        vol = jnp.abs(jnp.linalg.det(cell))
        dvol = vol / float(n1 * n2 * n3)
        e_rvv, de_drho = jax.value_and_grad(lambda r: self.energy_from_density(r, cell))(rho_tot)
        v_rvv = de_drho / dvol

        def _strain_e(eps: jax.Array) -> jax.Array:
            deform = jnp.eye(3) + eps
            return self.energy_from_density(rho_tot / jnp.linalg.det(deform), deform @ cell)

        if not compute_stress:
            return e_rvv, v_rvv, jnp.zeros((3, 3), dtype=jnp.float64)
        stress = jax.grad(_strain_e)(jnp.zeros((3, 3), dtype=jnp.float64)) / vol
        return e_rvv, v_rvv, stress

    def evaluate(
        self,
        structure: Structure,
        rho_grid: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Evaluate ``DispersionModel.evaluate`` for rVV10 given ``rho_grid``.

        Theory:
            Returns grid rVV10 energy and stress; grid-orbital forces enter through
            ``dfscf`` via :math:`V_{\text{rVV10}}` in :math:`V_{\text{eff}}`.

        Algorithm:
            Sums spin channels of ``rho_grid`` and delegates to ``potential_and_energy``.

        Parameters:
            structure: ``Structure`` PyTree.
            rho_grid: Real-space density ``(n_grid_spin, N1, N2, N3)``.

        Returns:
            Tuple ``(e_rvv10, zeros_forces, stress_rvv10)``.

        Invariants:
            Requires non-None ``rho_grid``.

        SIESTA Mapping:
            Implements ``DispersionModel.evaluate`` for grid rVV10.
        """
        if rho_grid is None:
            raise ValueError("RVV10Dispersion requires rho_grid.")
        rho_tot = rho_grid[0] if rho_grid.shape[0] == 1 else (rho_grid[0] + rho_grid[1])
        e_rvv, _, stress = self.potential_and_energy(rho_tot, structure.cell)
        return e_rvv, jnp.zeros((structure.n_atoms, 3), dtype=jnp.float64), stress
