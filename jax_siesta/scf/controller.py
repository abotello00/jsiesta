r"""Native closed-loop adaptive SCF controller and outcome classifier (``scf_params``).

Theory:
    Implements the three cooperative feedback loops from ``scf_adaptive.lua``
    directly inside the pure-JAX SCF state machine without external Lua/Flook coupling:

    1. **Stationary metric preservation:** Holds :math:`w_{\text{cur}} = w_{\text{base}}`
       fixed during initial Krylov construction (``iscf <= 4``) and during normal
       monotone contraction so all secant pairs share a consistent metric.
    2. **Trust-region spike backoff & selective history quarantine:** Detects severe
       residual spikes (``ratio_max > 3.0`` with ``dH > 0.25 eV``, or ``ratio_D > 2.2``
       and ``ratio_H > 2.2`` with ``dH > 0.15 eV``), backing off
       :math:`w_{\text{cur}} \leftarrow \max(w_{\min}, 0.65\,w_{\text{cur}})`, and
       flushing oldest Krylov history vectors only after 3 consecutive divergent steps
       (``streak_bad >= 3``). Relaxes :math:`w_{\text{cur}}` smoothly toward
       :math:`w_{\text{base}}` once contraction resumes.
    3. **Optional Fermi-surface smearing annealing:** Temporarily boosts
       :math:`T_e` during early cold-start transients (``dDmax > 0.03`` and ``iscf <= 6``),
       strictly restoring the exact user target ``config.electronic_temp_ha`` before
       declaring convergence.

Algorithm:
    Pure function ``update_adaptive_state`` taking residual metrics and returning
    updated ``SCFState`` adaptive fields plus convergence/breakdown status.

Invariants:
    - Convergence can NEVER be declared while ``active_temp_ha != config.electronic_temp_ha``.
    - Iteration-1 post-seed guard prevents premature convergence on ``iteration <= 1``.

SIESTA Mapping:
    Replaces ``scf_adaptive.lua`` and convergence checks in ``siesta_forces.F``.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from jax_siesta.core.types import HARTREE_TO_EV, SCFConfig, SCFState

#: Outcome taxonomy shared with ``scf_params``. Produced by the SCF loop: ``converged``,
#: ``iteration_cap_exhausted``, ``numerical_breakdown``. ``invalid_input`` is signalled by ``ValueError`` before the
#: loop starts, and ``normal_state_collapse`` is reserved for a future BdG solver; neither is emitted today (review L4).
VALID_OUTCOMES = (
    "converged",
    "iteration_cap_exhausted",
    "numerical_breakdown",
    "invalid_input",
    "normal_state_collapse",
)


def update_adaptive_state(
    state: SCFState,
    config: SCFConfig,
    d_max: float,
    h_max_ha: float,
    e_diff_ha: float,
) -> tuple[SCFState, bool, str]:
    r"""Evaluate convergence, trust-region backoff, history quarantine, and smearing annealing.

    Theory:
        Applies adaptive trust-region control and verifies convergence criteria
        strictly at the user's exact target electronic temperature.

    Algorithm:
        Checks numerical finiteness first (classifying NaN/Inf as ``numerical_breakdown``),
        evaluates spike ratios against ``state.prev_d_max`` / ``state.prev_h_max``,
        adjusts ``w_cur``, ``history_count``, and ``active_temp_ha``, and checks
        convergence only when ``iteration >= 2`` and ``active_temp_ha == target_temp``.

    Parameters:
        state: Current ``SCFState``.
        config: Immutable ``SCFConfig``.
        d_max: Max norm of :math:`D_{\text{out}} - D_{\text{in}}`.
        h_max_ha: Max norm of :math:`H_{\text{out}} - H_{\text{in}}` (Hartree).
        e_diff_ha: Absolute free-energy step change (Hartree).

    Returns:
        Tuple ``(updated_state, converged, outcome_label)``.

    Invariants:
        - ``outcome_label`` is ``"running"``, ``"converged"``, or one of ``VALID_OUTCOMES``.
        - ``converged == True`` implies ``iteration >= 2`` and exact target temperature.

    SIESTA Mapping:
        Replaces ``scf_adaptive.lua`` and ``scfconvergence_test.F``.
    """
    if not (jnp.isfinite(d_max) and jnp.isfinite(h_max_ha) and jnp.isfinite(e_diff_ha)):
        return replace(state, converged=False, outcome="numerical_breakdown"), False, "numerical_breakdown"

    iscf = state.iteration + 1
    target_temp = config.electronic_temp_ha
    w_base = float(state.w_base)
    w_cur = float(state.w_cur)
    w_min = max(0.01, 0.2 * w_base)
    streak_bad = state.streak_bad
    hist_count = state.history_count
    hist_x = state.history_x
    hist_f = state.history_f
    hist_d = state.history_d
    hist_e = state.history_e

    if config.enable_adaptive_controller and iscf > 4:
        prev_d = max(float(state.prev_d_max), 1e-12)
        prev_h = max(float(state.prev_h_max), 1e-12)
        ratio_d = d_max / prev_d
        ratio_h = h_max_ha / prev_h
        ratio_max = max(ratio_d, ratio_h)
        dh_ev = h_max_ha * HARTREE_TO_EV

        severe_spike = (ratio_max > 3.0 and dh_ev > 0.25) or (
            ratio_d > 2.2 and ratio_h > 2.2 and dh_ev > 0.15
        )
        if severe_spike:
            w_cur = max(w_min, 0.65 * w_cur)
            streak_bad += 1
            if streak_bad >= 3 and hist_count > 2:
                # Quarantine oldest Krylov vectors, keep 2 most recent
                hist_x = jnp.roll(hist_x, shift=-(hist_count - 2), axis=0)
                hist_f = jnp.roll(hist_f, shift=-(hist_count - 2), axis=0)
                if hist_d is not None and hist_e is not None:
                    hist_d = jnp.roll(hist_d, shift=-(hist_count - 2), axis=0)
                    hist_e = jnp.roll(hist_e, shift=-(hist_count - 2), axis=0)
                hist_count = 2
                streak_bad = 0
        else:
            streak_bad = 0
            # Smoothly relax w_cur back toward stationary w_base
            if ratio_max <= 1.05 and w_cur < w_base:
                w_cur = min(w_base, 1.15 * w_cur)

    # Optional Fermi-surface smearing annealing (monotonically non-increasing; restored strictly before convergence)
    if config.enable_smearing_annealing and d_max > 0.03 and iscf <= 6:
        anneal_temp = max(target_temp, 0.01)  # ~3150 K annealing floor
        active_temp = anneal_temp if iscf == 1 else min(float(state.active_temp_ha), anneal_temp)
    else:
        active_temp = target_temp

    # The DM just evaluated was built at the temperature used in *this* iteration (state.active_temp_ha), not
    # at the one chosen for the next iteration; only that one certifies the occupations (review L1).
    used_temp = float(state.active_temp_ha)
    at_target_temp = abs(used_temp - target_temp) <= 1e-14 and abs(active_temp - target_temp) <= 1e-14
    # Mandatory iteration-1 post-seed guard: require iscf >= 2
    converged = bool(
        iscf >= 2
        and at_target_temp
        and d_max <= config.d_tol
        and h_max_ha <= config.h_tol_ha
        and e_diff_ha <= config.e_tol_ha
    )
    outcome = "converged" if converged else "running"

    updated = replace(
        state,
        iteration=iscf,
        w_cur=jnp.asarray(w_cur, dtype=jnp.float64),
        active_temp_ha=jnp.asarray(active_temp, dtype=jnp.float64),
        streak_bad=streak_bad,
        prev_d_max=jnp.asarray(d_max, dtype=jnp.float64),
        prev_h_max=jnp.asarray(h_max_ha, dtype=jnp.float64),
        history_x=hist_x,
        history_f=hist_f,
        history_d=hist_d,
        history_e=hist_e,
        history_count=hist_count,
        converged=converged,
        outcome=outcome,
    )
    return updated, converged, outcome
