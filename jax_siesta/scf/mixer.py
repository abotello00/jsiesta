r"""Tikhonov-Pulay, Pulay (DIIS), Broyden and ADIIS SCF mixers with decoupled spin & deterministic SVD.

Theory:
    Solves the SCF fixed-point problem :math:`x = \mathcal{F}(x)` via Krylov/DIIS
    extrapolation over history pairs :math:`\{x_j, F_j = \mathcal{F}(x_j) - x_j\}`:

    - **Scale-invariant multiplicative Tikhonov-Pulay:** Solves
      :math:`(G + \lambda \operatorname{diag}(G) + \epsilon_{\text{abs}} I) c = \mathbf{1}`
      and normalizes :math:`c / \sum_j c_j`, regularizing near-collinear Krylov
      directions without shrinking the stationary preconditioner :math:`G = w I`.
    - **Decoupled charge/spin preconditioning (``mix_spin="decoupled"``):** Applies
      separate preconditioner weights :math:`w_{\text{charge}} = w_{\text{cur}}` and
      :math:`w_{\text{spin}} = s_{\text{spin}} w_{\text{cur}}` (:math:`s_{\text{spin}} = 1.6`
      collinear, :math:`1.4` NC/SOC) while keeping Pulay weights :math:`c_j` phase-locked
      across charge and spin channels.
    - **Deterministic scalar (1x1) SVD contract:** For :math:`1\times 1` Gram matrices,
      returns ``info=0, rank=0, inv=0`` if :math:`|A_{11}| \le \epsilon_{\text{svd}}`,
      and ``info=0, rank=1, inv=1/A_{11}`` otherwise.
    - **Broyden (Johnson's modified Broyden, equivalent to Anderson/type-II mixing):** with differences
      :math:`\Delta F_i = F_{i+1} - F_i`, :math:`\Delta x_i = x_{i+1} - x_i` and the newest residual :math:`F_n`,
      :math:`\gamma = (w_0^2 I + \Delta F \Delta F^T)^{-1} \Delta F\, F_n` and
      :math:`x_{n+1} = x_n + W F_n - \sum_i \gamma_i (\Delta x_i + W \Delta F_i)`. It updates an approximate inverse
      Jacobian from secant pairs instead of solving a constrained least-squares problem for :math:`\sum c_j = 1`.
    - **ADIIS (Hu & Yang 2010), Hamiltonian mixing only:** minimises the augmented-Roothaan-Hall energy model
      :math:`f(c) = E_n + 2\sum_i c_i \langle D_i - D_n | F_n \rangle + \sum_{ij} c_i c_j \langle D_i - D_n | F_j - F_n \rangle`
      over :math:`c_i \ge 0, \sum c_i = 1`, where :math:`D_i` is the output density matrix, :math:`F_i` the Hamiltonian
      built from it and :math:`E_i` the energy at :math:`D_i`. The ADIIS coefficients are blended with the
      Tikhonov-Pulay ones, :math:`c = (1-s)\,c_{\text{ADIIS}} + s\,c_{\text{Pulay}}`, with :math:`s` rising from 0 to 1
      as the largest residual falls from ``ADIIS_ERR_HI`` to ``ADIIS_ERR_LO`` (the ADIIS-to-Pulay hybrid of plan §4.2.1).

Algorithm:
    Maintains fixed-capacity circular/FIFO Krylov history inside immutable ``SCFState``
    buffers and supports the ``tikhonov``, ``pulay``, ``broyden`` and ``adiis`` variants,
    plus an optional ``anomalous_channels`` slice hook for future BdG pairing fields.

Invariants:
    - Deterministic 1x1 SVD pseudoinverse (zero uninitialized rank/info).
    - Iteration-1 post-seed guard prevents premature convergence before an update step.

SIESTA Mapping:
    Implements ``SCFMixer`` replacing ``mixer.F``, ``m_mixing.F90``, and
    ``scf_params`` mixer fixes (``0001-mixing-initialize-scalar-svd.patch``).
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import math

import jax
import jax.numpy as jnp

from jax_siesta.core.types import SCFConfig, SCFState

#: Johnson's weight ``w0``: the Broyden Gram diagonal is multiplied by ``1 + w0^2``.
BROYDEN_W0 = 0.01
#: ADIIS-to-Pulay blend: pure ADIIS above ``ADIIS_ERR_HI`` (Ha, max-norm residual), pure Pulay below ``ADIIS_ERR_LO``.
ADIIS_ERR_HI = 0.1
ADIIS_ERR_LO = 1.0e-4
_ADIIS_PG_ITERATIONS = 300


@partial(
    jax.jit,
    static_argnames=(
        "nspin",
        "mix_spin",
        "mixer_variant",
        "tikhonov_reg",
        "svd_tol",
        "n_anomalous_channels",
    ),
)
def _mixer_step_jit(
    x_in: jax.Array,
    x_out: jax.Array,
    history_x: jax.Array,
    history_f: jax.Array,
    history_count: jax.Array,
    w_cur: jax.Array,
    nspin: int,
    mix_spin: str,
    mixer_variant: str,
    tikhonov_reg: float | None,
    svd_tol: float,
    n_anomalous_channels: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compiled pure-JAX Tikhonov-Pulay / Pulay step with fixed-shape padding."""
    res = x_out - x_in
    max_hist = history_x.shape[0]
    cur_count = jnp.asarray(history_count, dtype=jnp.int32)

    def _append_branch(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        hx = history_x.at[cur_count].set(x_in)
        hf = history_f.at[cur_count].set(res)
        return hx, hf, cur_count + 1

    def _roll_branch(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        hx = jnp.roll(history_x, shift=-1, axis=0).at[-1].set(x_in)
        hf = jnp.roll(history_f, shift=-1, axis=0).at[-1].set(res)
        return hx, hf, jnp.asarray(max_hist, dtype=jnp.int32)

    hist_x, hist_f, m = jax.lax.cond(cur_count < max_hist, _append_branch, _roll_branch, None)

    hx_cs = jax.vmap(lambda a: _to_charge_spin_basis(a, nspin))(hist_x)
    hf_cs = jax.vmap(lambda a: _to_charge_spin_basis(a, nspin))(hist_f)
    w_vec = _channel_preconditioner_weights(
        x_in.shape[0],
        nspin,
        w_cur,
        mix_spin,
        n_anomalous_channels,
    )

    def _linear_step(_: None) -> jax.Array:
        return hx_cs[0] + w_vec[:, None] * hf_cs[0]

    def _krylov_step(_: None) -> jax.Array:
        if mixer_variant == "broyden":
            return _broyden_update(hx_cs, hf_cs, m, w_vec)
        valid_1d = jnp.arange(max_hist, dtype=jnp.int32) < m
        valid_f = valid_1d.astype(jnp.float64)
        valid_2d = valid_1d[:, None] & valid_1d[None, :]

        f_flat = hf_cs.reshape((max_hist, -1))
        f_norms = jnp.linalg.norm(f_flat, axis=1) * valid_f
        f_scale = jnp.maximum(jnp.max(f_norms), 1e-15)
        f_scaled = (f_flat / f_scale) * valid_f[:, None]
        gram = f_scaled @ f_scaled.T

        if mixer_variant == "tikhonov":
            reg = tikhonov_reg if tikhonov_reg is not None else 1.0e-3
            diag_g = jnp.maximum(jnp.diag(gram), 1e-16) * valid_f
            gram_reg = gram + reg * jnp.diag(diag_g) + 1e-14 * jnp.diag(valid_f)
        else:  # "pulay": plain DIIS ("broyden" returned above; "adiis" uses _adiis_step_jit)
            gram_reg = gram + 1e-14 * jnp.diag(valid_f)

        gram_reg = jnp.where(valid_2d, gram_reg, 0.0)
        g_inv, rank, _singular_values = _deterministic_svd_pinv_arrays(gram_reg, svd_tol=svd_tol)

        fallback = jnp.zeros((max_hist,), dtype=jnp.float64).at[m - 1].set(1.0)
        raw_c = (g_inv @ valid_f) * valid_f
        c_sum = jnp.sum(raw_c)
        c_ok = (rank > 0) & (jnp.abs(c_sum) > 1e-14)
        coeffs = jnp.where(
            c_ok,
            raw_c / jnp.where(c_ok, c_sum, 1.0),
            fallback,
        )
        x_bar = jnp.einsum("m,mck->ck", coeffs, hx_cs)
        f_bar = jnp.einsum("m,mck->ck", coeffs, hf_cs)
        return x_bar + w_vec[:, None] * f_bar

    x_next_cs = jax.lax.cond(m <= 1, _linear_step, _krylov_step, None)
    x_next = _from_charge_spin_basis(x_next_cs, nspin)
    return x_next, hist_x, hist_f, m


def _broyden_update(hx_cs: jax.Array, hf_cs: jax.Array, m: jax.Array, w_vec: jax.Array) -> jax.Array:
    """Johnson's modified Broyden step on the valid history ``hx_cs[:m]``, ``hf_cs[:m]`` (fixed shapes)."""
    max_hist = hx_cs.shape[0]
    last = m - 1
    x_last = hx_cs[last]
    f_last = hf_cs[last]
    valid = (jnp.arange(max_hist - 1, dtype=jnp.int32) + 1 < m).astype(jnp.float64)
    d_f = (hf_cs[1:] - hf_cs[:-1]) * valid[:, None, None]
    d_x = (hx_cs[1:] - hx_cs[:-1]) * valid[:, None, None]
    flat = d_f.reshape((max_hist - 1, -1))
    scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.linalg.norm(hf_cs.reshape((max_hist, -1)), axis=1)), jnp.linalg.norm(f_last)),
        1e-15,
    )
    flat_s = flat / scale
    gram0 = flat_s @ flat_s.T
    # Multiplicative (scale-invariant) w0^2 regulariser: an absolute one would swamp the shrinking residual differences.
    gram = gram0 + (BROYDEN_W0**2) * jnp.diag(jnp.maximum(jnp.diag(gram0), 1e-30)) + jnp.diag(1.0 - valid)
    gamma = jnp.linalg.solve(gram, flat_s @ (f_last.reshape(-1) / scale)) * valid
    correction = jnp.einsum("i,ick->ck", gamma, d_x + w_vec[None, :, None] * d_f)
    return x_last + w_vec[:, None] * f_last - correction


def _project_simplex(v: jax.Array, valid: jax.Array) -> jax.Array:
    """Euclidean projection of ``v`` onto ``{c >= 0, sum c = 1, c_i = 0 where not valid}``."""
    n = v.shape[0]
    neg = -1.0e30
    u = jnp.sort(jnp.where(valid, v, neg))[::-1]
    live = u > 0.5 * neg
    css = jnp.cumsum(jnp.where(live, u, 0.0))
    k = jnp.arange(1, n + 1, dtype=jnp.float64)
    rho = jnp.maximum(jnp.sum((u - (css - 1.0) / k > 0.0) & live), 1)
    theta = (css[rho - 1] - 1.0) / rho.astype(jnp.float64)
    return jnp.maximum(v - theta, 0.0) * valid.astype(jnp.float64)


def adiis_coefficients(
    hist_d: jax.Array,
    hist_f: jax.Array,
    hist_e: jax.Array,
    m: jax.Array,
    trace_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""ADIIS coefficients minimising the augmented-Roothaan-Hall energy model over the simplex.

    Theory:
        :math:`f(c) = E_n + 2 b\cdot c + c^T A c` with :math:`b_i = \langle D_i - D_n | F_n \rangle`,
        :math:`A_{ij} = \langle D_i - D_n | F_j - F_n \rangle` (symmetrised) and the channel-weighted trace inner
        product :math:`\langle X | Y \rangle = \sum_{c,e} w_c X_{c,e} Y_{c,e}` (Hu & Yang, J. Chem. Phys. 132, 054109).

    Algorithm:
        With :math:`F = \partial E/\partial D` (``D`` the total density matrix, ``Tr[D H]`` the band energy) the exact energy of
        :math:`D_{\text{mix}} = \sum_i c_i D_i` is :math:`E_n + (f - E_n)/2` for a quadratic ``E``; Hu & Yang's factor 2 comes
        from their spin-restricted convention and does not change the minimiser. Projected gradient descent on the simplex from the newest entry :math:`c = e_n` (fixed iteration count, step
        :math:`1/\|2A\|_F`), keeping the better of the result and the start so :math:`f(c) \le E_n` always.

    Parameters:
        hist_d: Output density matrices ``(max_hist, n_ch, n_edges)``.
        hist_f: Hamiltonians built from them ``(max_hist, n_ch, n_edges)``.
        hist_e: Energies at those densities ``(max_hist,)``.
        m: Number of valid entries (the newest is ``m - 1``).
        trace_weights: Per-channel weights ``(n_ch,)`` of the trace inner product.

    Returns:
        ``(c, f_c)``: coefficients ``(max_hist,)`` (zero beyond ``m``) and the model energy at ``c``.

    Invariants:
        ``c >= 0``, ``sum(c) = 1`` and ``f_c <= hist_e[m - 1]``.

    SIESTA Mapping:
        None; SIESTA has no ADIIS mixer.
    """
    max_hist = hist_d.shape[0]
    valid = jnp.arange(max_hist, dtype=jnp.int32) < m
    last = m - 1
    d_n, f_n, e_n = hist_d[last], hist_f[last], hist_e[last]
    dd = hist_d - d_n[None]
    df = hist_f - f_n[None]
    b = jnp.einsum("hck,ck,c->h", dd, f_n, trace_weights)
    a_mat = jnp.einsum("hck,jck,c->hj", dd, df, trace_weights)
    a_sym = 0.5 * (a_mat + a_mat.T)
    vf = valid.astype(jnp.float64)
    a_sym = a_sym * vf[:, None] * vf[None, :]
    b = b * vf
    step = 1.0 / (2.0 * jnp.linalg.norm(a_sym) + 1.0e-30)

    def _model(c: jax.Array) -> jax.Array:
        return e_n + 2.0 * (b @ c) + c @ (a_sym @ c)

    c0 = jnp.zeros((max_hist,), dtype=jnp.float64).at[last].set(1.0)

    def _body(_: int, c: jax.Array) -> jax.Array:
        return _project_simplex(c - step * (2.0 * b + 2.0 * (a_sym @ c)), valid)

    c_pg = jax.lax.fori_loop(0, _ADIIS_PG_ITERATIONS, _body, c0)
    take = _model(c_pg) < _model(c0)
    c_best = jnp.where(take, c_pg, c0)
    return c_best, _model(c_best)


@partial(
    jax.jit,
    static_argnames=("nspin", "mix_spin", "tikhonov_reg", "svd_tol", "n_anomalous_channels", "trace_weights"),
)
def _adiis_step_jit(
    x_in: jax.Array,
    x_out: jax.Array,
    d_out: jax.Array,
    energy: jax.Array,
    history_x: jax.Array,
    history_f: jax.Array,
    history_d: jax.Array,
    history_e: jax.Array,
    history_count: jax.Array,
    w_cur: jax.Array,
    nspin: int,
    mix_spin: str,
    tikhonov_reg: float | None,
    svd_tol: float,
    n_anomalous_channels: int,
    trace_weights: tuple[float, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """ADIIS-to-Tikhonov-Pulay hybrid step (Hamiltonian mixing): ``x`` is H, ``d_out`` the density matrix of H_in."""
    res = x_out - x_in
    max_hist = history_x.shape[0]
    cur = jnp.asarray(history_count, dtype=jnp.int32)

    def _push(hist: jax.Array, new: jax.Array) -> jax.Array:
        appended = hist.at[cur].set(new)
        rolled = jnp.roll(hist, shift=-1, axis=0).at[-1].set(new)
        return jnp.where(cur < max_hist, appended, rolled)

    hist_x = _push(history_x, x_in)
    hist_f = _push(history_f, res)
    hist_d = _push(history_d, d_out)
    hist_e = _push(history_e, jnp.asarray(energy, dtype=jnp.float64))
    m = jnp.minimum(cur + 1, max_hist)

    hx_cs = jax.vmap(lambda a: _to_charge_spin_basis(a, nspin))(hist_x)
    hf_cs = jax.vmap(lambda a: _to_charge_spin_basis(a, nspin))(hist_f)
    w_vec = _channel_preconditioner_weights(x_in.shape[0], nspin, w_cur, mix_spin, n_anomalous_channels)

    # ADIIS: F_i = x_in,i + res_i is the Hamiltonian built from D_i.
    c_adiis, _ = adiis_coefficients(hist_d, hist_x + hist_f, hist_e, m, jnp.asarray(trace_weights, dtype=jnp.float64))

    # Tikhonov-Pulay coefficients on the same history.
    valid_f = (jnp.arange(max_hist, dtype=jnp.int32) < m).astype(jnp.float64)
    f_flat = hf_cs.reshape((max_hist, -1))
    f_norms = jnp.linalg.norm(f_flat, axis=1) * valid_f
    f_scaled = (f_flat / jnp.maximum(jnp.max(f_norms), 1e-15)) * valid_f[:, None]
    gram = f_scaled @ f_scaled.T
    reg = tikhonov_reg if tikhonov_reg is not None else 1.0e-3
    gram_reg = gram + reg * jnp.diag(jnp.maximum(jnp.diag(gram), 1e-16) * valid_f) + 1e-14 * jnp.diag(valid_f)
    gram_reg = jnp.where((valid_f[:, None] * valid_f[None, :]) > 0, gram_reg, 0.0)
    g_inv, rank, _ = _deterministic_svd_pinv_arrays(gram_reg, svd_tol=svd_tol)
    raw_c = (g_inv @ valid_f) * valid_f
    c_sum = jnp.sum(raw_c)
    ok = (rank > 0) & (jnp.abs(c_sum) > 1e-14)
    fallback = jnp.zeros((max_hist,), dtype=jnp.float64).at[m - 1].set(1.0)
    c_pulay = jnp.where(ok, raw_c / jnp.where(ok, c_sum, 1.0), fallback)

    err = jnp.maximum(jnp.max(jnp.abs(res)), 1e-300)
    s = jnp.clip(
        (math.log10(ADIIS_ERR_HI) - jnp.log10(err)) / (math.log10(ADIIS_ERR_HI) - math.log10(ADIIS_ERR_LO)),
        0.0,
        1.0,
    )
    coeffs = (1.0 - s) * c_adiis + s * c_pulay
    x_bar = jnp.einsum("m,mck->ck", coeffs, hx_cs)
    f_bar = jnp.einsum("m,mck->ck", coeffs, hf_cs)
    mixed = x_bar + w_vec[:, None] * f_bar
    linear = hx_cs[0] + w_vec[:, None] * hf_cs[0]
    x_next_cs = jnp.where(m <= 1, linear, mixed)
    return _from_charge_spin_basis(x_next_cs, nspin), hist_x, hist_f, hist_d, hist_e, m


def _trace_weights(nspin: int, n_channels: int) -> tuple[float, ...]:
    """Channel weights of ``Tr[D H]`` for the stored real channels (off-diagonal spin blocks count twice)."""
    if nspin in (4, 8) and n_channels >= 4:
        return (1.0, 1.0, 2.0, 2.0) + (0.0,) * (n_channels - 4)
    return (1.0,) * n_channels


def _deterministic_svd_pinv_arrays(
    gram: jax.Array,
    svd_tol: float = 1e-12,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pure-JAX JIT-safe SVD pseudoinverse returning array scalars ``(pinv, rank, info)``."""
    m = int(gram.shape[0])
    if m == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)
    if m == 1:
        a11 = gram[0, 0]
        is_finite = jnp.isfinite(a11)
        is_nz = is_finite & (jnp.abs(a11) > svd_tol)
        inv = jnp.where(is_nz, 1.0 / jnp.where(is_nz, a11, 1.0), 0.0)
        rank = jnp.where(is_nz, 1, 0).astype(jnp.int32)
        info = jnp.where(is_finite, 0, 1).astype(jnp.int32)
        return jnp.array([[inv]], dtype=jnp.float64), rank, info

    u, s, vt = jnp.linalg.svd(gram, full_matrices=False)
    thresh = svd_tol * jnp.maximum(jnp.max(s), 1.0)
    keep = s > thresh
    s_inv = jnp.where(keep, 1.0 / jnp.maximum(s, thresh), 0.0)
    pinv = (vt.T * s_inv[None, :]) @ u.T
    rank = jnp.sum(keep).astype(jnp.int32)
    return pinv, rank, jnp.asarray(0, dtype=jnp.int32)


def deterministic_svd_pinv(
    gram: jax.Array,
    svd_tol: float = 1e-12,
) -> tuple[jax.Array, int, int]:
    r"""Exact 1x1-safe Moore-Penrose pseudoinverse of Pulay Gram matrix.

    Theory:
        Solves symmetric least-squares systems :math:`A x = b` via thin SVD
        with deterministic rank truncation below :math:`\epsilon_{\text{svd}}`.

    Algorithm:
        Handles :math:`m = 1` explicitly without calling LAPACK ``dgesdd`` (preventing
        uninitialized ``rank``/``info`` defects) and uses ``jnp.linalg.svd`` for :math:`m \ge 2`.

    Parameters:
        gram: Square matrix of shape ``(m, m)``.
        svd_tol: Singular value cutoff threshold.

    Returns:
        Tuple ``(pinv_matrix, rank, info)`` where ``info == 0`` on finite inputs.

    Invariants:
        For ``m == 1`` and ``|A[0, 0]| <= svd_tol``, returns ``(zeros((1,1)), 0, 0)``.

    SIESTA Mapping:
        Fixes the scalar SVD defect in ``0001-mixing-initialize-scalar-svd.patch``.
    """
    pinv, rank, info = _deterministic_svd_pinv_arrays(gram, svd_tol=svd_tol)
    return pinv, int(rank), int(info)


def _channel_preconditioner_weights(
    n_channels: int,
    nspin: int,
    w_cur: jax.Array,
    mix_spin: str,
    n_anomalous_channels: int = 0,
) -> jax.Array:
    """Compute per-channel preconditioner weights for decoupled charge/spin/(BdG) mixing."""
    if mix_spin == "coupled" or nspin == 1:
        return jnp.full((n_channels,), w_cur, dtype=jnp.float64)

    spin_ratio = 1.6 if nspin == 2 else 1.4
    w_spin = jnp.minimum(0.65, spin_ratio * w_cur)
    weights = [w_cur]  # channel 0: charge (n = x0 + x1)
    n_spin_only = n_channels - 1 - n_anomalous_channels
    for _ in range(max(0, n_spin_only)):
        weights.append(w_spin)
    for _ in range(n_anomalous_channels):
        weights.append(w_cur)
    return jnp.stack(weights)


def _to_charge_spin_basis(arr: jax.Array, nspin: int) -> jax.Array:
    """Transform spin matrix channels ``(up, dn, ...)`` into ``(charge, m_z, m_x, m_y, ...)``."""
    if nspin == 1 or arr.shape[0] < 2:
        return arr
    charge = arr[0] + arr[1]
    mz = arr[0] - arr[1]
    if arr.shape[0] == 2:
        return jnp.stack([charge, mz], axis=0)
    return jnp.concatenate([charge[None, :], mz[None, :], arr[2:]], axis=0)


def _from_charge_spin_basis(cs: jax.Array, nspin: int) -> jax.Array:
    """Inverse of ``_to_charge_spin_basis``."""
    if nspin == 1 or cs.shape[0] < 2:
        return cs
    up = 0.5 * (cs[0] + cs[1])
    dn = 0.5 * (cs[0] - cs[1])
    if cs.shape[0] == 2:
        return jnp.stack([up, dn], axis=0)
    return jnp.concatenate([up[None, :], dn[None, :], cs[2:]], axis=0)


class TikhonovPulayMixer:
    r"""Adaptive Tikhonov-Pulay / Pulay SCF mixer implementing ``SCFMixer``.

    Theory:
        Accelerates fixed-point convergence using DIIS / Pulay extrapolation with
        scale-invariant multiplicative Tikhonov regularization, decoupled charge/spin
        preconditioning, and deterministic scalar SVD fallback.

    Algorithm:
        Transforms inputs to charge/spin basis, appends ``(x_in, res = x_out - x_in)``
        into the Krylov ring buffer, solves the regularized Gram equation, and applies
        phase-locked extrapolation + preconditioned residual update.

    Invariants:
        - Preserves exact linear combination sum :math:`\sum_j c_j = 1`.
        - Never mutates inputs; returns a new ``SCFState``.

    SIESTA Mapping:
        Implements ``SCFMixer`` replacing ``mixer.F`` and ``m_mixing.F90``.
    """

    def __init__(self, n_anomalous_channels: int = 0) -> None:
        self.n_anomalous_channels = n_anomalous_channels

    def step(
        self,
        x_in: jax.Array,
        x_out: jax.Array,
        state: SCFState,
        config: SCFConfig,
        aux: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, SCFState]:
        r"""Compute mixed next iterate and updated ``SCFState`` with Krylov history.

        Theory:
            Forms residual :math:`F_k = x_{\text{out}} - x_{\text{in}}`, updates
            the :math:`m`-vector Krylov subspace, computes extrapolation weights
            :math:`c` via (Tikhonov-regularised) Pulay, and steps
            :math:`x_{\text{next}} = \sum_j c_j (x_j + W F_j)`.

        Algorithm:
            Executes linear mixing when ``history_count == 0``, otherwise builds
            the :math:`m\times m` residual Gram matrix and solves via
            ``deterministic_svd_pinv``.

        Parameters:
            x_in: Current input iterate ``(n_spin_mat, n_edges)``.
            x_out: Current output iterate ``(n_spin_mat, n_edges)``.
            state: Current ``SCFState``.
            config: ``SCFConfig`` selecting mixer variant and spin preconditioning.
            aux: ``(d_out, energy)`` for ``mixer_variant="adiis"``: the output density matrix in the layout of
                ``x_in`` and the energy at that density. Ignored by the other variants.

        Returns:
            Tuple ``(x_next, updated_state)``.

        Invariants:
            - ``x_next.shape == x_in.shape``.
            - ``updated_state.history_count <= state.history_x.shape[0]``.

        SIESTA Mapping:
            Replaces ``mixingSCF`` and ``pulayx`` in ``m_mixing_scf.F90``.
        """
        if config.mixer_variant == "adiis":
            if aux is None or state.history_d is None or state.history_e is None:
                raise ValueError("mixer_variant='adiis' needs aux=(d_out, energy) and ADIIS history buffers in SCFState.")
            x_next, hist_x, hist_f, hist_d, hist_e, m = _adiis_step_jit(
                x_in,
                x_out,
                aux[0],
                aux[1],
                state.history_x,
                state.history_f,
                state.history_d,
                state.history_e,
                jnp.asarray(state.history_count, dtype=jnp.int32),
                state.w_cur,
                nspin=config.nspin,
                mix_spin=config.mix_spin,
                tikhonov_reg=config.tikhonov_reg,
                svd_tol=config.svd_tol,
                n_anomalous_channels=self.n_anomalous_channels,
                trace_weights=_trace_weights(config.nspin, int(x_in.shape[0])),
            )
            return x_next, replace(
                state, history_x=hist_x, history_f=hist_f, history_d=hist_d, history_e=hist_e, history_count=int(m)
            )
        x_next, hist_x, hist_f, m = _mixer_step_jit(
            x_in,
            x_out,
            state.history_x,
            state.history_f,
            jnp.asarray(state.history_count, dtype=jnp.int32),
            state.w_cur,
            nspin=config.nspin,
            mix_spin=config.mix_spin,
            mixer_variant=config.mixer_variant,
            tikhonov_reg=config.tikhonov_reg,
            svd_tol=config.svd_tol,
            n_anomalous_channels=self.n_anomalous_channels,
        )
        new_state = replace(state, history_x=hist_x, history_f=hist_f, history_count=int(m))
        return x_next, new_state
