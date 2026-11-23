"""Broyden and ADIIS mixers (review M1, open issue O-1): algorithmic properties on model problems.

The integration test on real SCF runs (Co hcp endpoint agreement, differing trajectories) lives in
``test_phase1_scf_and_forces.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_siesta.core.types import SCFConfig, SCFState
from jax_siesta.scf.mixer import (
    ADIIS_ERR_HI,
    ADIIS_ERR_LO,
    TikhonovPulayMixer,
    _project_simplex,
    adiis_coefficients,
)

jax.config.update("jax_enable_x64", True)


def _state(x0: jax.Array, n_hist: int, adiis: bool = False) -> SCFState:
    n_ch, n_edges = x0.shape
    z = jnp.zeros((n_hist, n_ch, n_edges))
    return SCFState(
        iteration=0, h_sparse=x0, dm_in=x0, dm_out=x0, edm_out=x0,
        fermi_level_ha=jnp.array(0.0), entropy_ha=jnp.array(0.0), last_scf_free_e_ha=jnp.array(0.0),
        w_cur=jnp.array(0.5), w_base=jnp.array(0.5), active_temp_ha=jnp.array(1e-3),
        streak_bad=0, prev_d_max=jnp.array(1.0), prev_h_max=jnp.array(1.0),
        history_x=z, history_f=z, history_count=0,
        history_d=z if adiis else None, history_e=jnp.zeros((n_hist,)) if adiis else None,
    )


def _affine_problem(n: int, seed: int = 3):
    """x -> M x + c with eigenvalues of M in [-0.6, 0.97]: slow for linear mixing, easy for Krylov methods."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    m = q @ np.diag(np.linspace(-0.6, 0.97, n)) @ q.T
    c = rng.standard_normal(n)
    return jnp.asarray(m), jnp.asarray(c), jnp.asarray(np.linalg.solve(np.eye(n) - m, c))


def _iterations(variant: str, n: int = 10, w: float = 0.5, tol: float = 1e-9, max_iter: int = 400) -> int:
    m, c, x_star = _affine_problem(n)
    cfg = SCFConfig(nspin=1, mixer_variant=variant, mix_target="Hamiltonian")
    mixer = TikhonovPulayMixer()
    state = _state(jnp.zeros((1, n)), n_hist=12)
    state = state.__class__(**{**state.__dict__, "w_cur": jnp.array(w)})
    x = jnp.zeros((1, n))
    for it in range(1, max_iter + 1):
        x_out = (m @ x[0] + c)[None, :]
        if float(jnp.max(jnp.abs(x_out - x))) < tol:
            assert float(jnp.max(jnp.abs(x[0] - x_star))) < 1e-6
            return it
        x, state = mixer.step(x, x_out, state, cfg)
    return max_iter


def test_broyden_converges_far_faster_than_linear_mixing_and_differs_from_pulay() -> None:
    it_broyden = _iterations("broyden")
    it_pulay = _iterations("pulay")
    # Plain linear mixing with w = 0.5 needs ~1300 steps for a 0.97 eigenvalue (1 - 0.5*0.03 per step to 1e-9).
    assert it_broyden <= 25, it_broyden
    assert it_pulay <= 40, it_pulay
    assert it_broyden < it_pulay  # measured 17 vs 29 on this problem
    # A different algorithm must not reproduce DIIS: the iteration counts or at least the iterates differ.
    m, c, _ = _affine_problem(10)
    cfg_b, cfg_p = SCFConfig(nspin=1, mixer_variant="broyden"), SCFConfig(nspin=1, mixer_variant="pulay")
    xb, xp = jnp.zeros((1, 10)), jnp.zeros((1, 10))
    sb, sp = _state(xb, 12), _state(xp, 12)
    mixer = TikhonovPulayMixer()
    for _ in range(4):
        xb, sb = mixer.step(xb, (m @ xb[0] + c)[None, :], sb, cfg_b)
        xp, sp = mixer.step(xp, (m @ xp[0] + c)[None, :], sp, cfg_p)
    assert float(jnp.max(jnp.abs(xb - xp))) > 1e-6


def test_broyden_satisfies_the_secant_condition_on_an_affine_map() -> None:
    """Finite termination on an affine map: Broyden with enough history converges in about n + 4 steps."""
    n = 6
    m, c, _ = _affine_problem(n, seed=5)
    cfg = SCFConfig(nspin=1, mixer_variant="broyden")
    mixer = TikhonovPulayMixer()
    x = jnp.zeros((1, n))
    state = _state(x, 12)
    for _ in range(n + 5):
        x, state = mixer.step(x, (m @ x[0] + c)[None, :], state, cfg)
    # n secant pairs span the n-dimensional affine problem (measured: 1e-12 after n + 4 steps).
    assert float(jnp.max(jnp.abs((m @ x[0] + c) - x[0]))) < 1e-9


def test_simplex_projection_and_adiis_coefficients_are_feasible_and_descend() -> None:
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.standard_normal(7))
    valid = jnp.asarray([True] * 5 + [False] * 2)
    p = _project_simplex(v, valid)
    assert float(jnp.min(p)) >= 0.0 and abs(float(jnp.sum(p)) - 1.0) < 1e-12 and float(jnp.sum(p[5:])) == 0.0

    n_hist, n_ch, n_edges, m = 6, 2, 9, 5
    d = jnp.asarray(rng.standard_normal((n_hist, n_ch, n_edges)))
    f = jnp.asarray(rng.standard_normal((n_hist, n_ch, n_edges)))
    e = jnp.asarray(rng.standard_normal(n_hist))
    w = jnp.array([1.0, 1.0])
    c, fc = adiis_coefficients(d, f, e, jnp.asarray(m), w)
    assert float(jnp.min(c)) >= 0.0 and abs(float(jnp.sum(c)) - 1.0) < 1e-12 and float(jnp.sum(c[m:])) == 0.0
    assert float(fc) <= float(e[m - 1]) + 1e-12  # never worse than keeping the newest entry
    # A single valid entry gives c = (1) and f = E_n.
    c1, f1 = adiis_coefficients(d, f, e, jnp.asarray(1), w)
    assert float(c1[0]) == 1.0 and abs(float(f1) - float(e[0])) < 1e-12


def test_adiis_reduces_the_energy_model_below_the_best_single_entry() -> None:
    """Quadratic energy E(D) = h.D + D.U.D/2 with F = h + U D: ADIIS finds a convex combination with lower E."""
    rng = np.random.default_rng(8)
    n, m = 8, 5
    a = rng.standard_normal((n, n))
    u = a @ a.T / n + np.eye(n)
    h = rng.standard_normal(n)
    d = rng.standard_normal((m, n))
    energies = np.array([h @ di + 0.5 * di @ u @ di for di in d])
    focks = np.array([h + u @ di for di in d])
    pad = lambda arr: jnp.asarray(np.concatenate([arr, np.zeros((1,) + arr.shape[1:])]))[:, None, :]  # noqa: E731
    c, fc = adiis_coefficients(pad(d), pad(focks), jnp.asarray(np.concatenate([energies, [0.0]])), jnp.asarray(m), jnp.ones(1))
    d_mix = np.asarray(c[:m]) @ d
    e_mix = h @ d_mix + 0.5 * d_mix @ u @ d_mix
    # With F = dE/dD the exact energy of the interpolated density is E_n + (f - E_n)/2 (Hu & Yang's f carries the
    # factor 2 of their spin-restricted convention; the minimiser is the same). For a quadratic E this is exact.
    assert abs(e_mix - (energies[m - 1] + 0.5 * (float(fc) - energies[m - 1]))) < 1e-9
    assert e_mix <= energies[m - 1] + 1e-12  # never worse than the newest entry
    assert e_mix < energies[m - 1] - 1e-3  # and strictly lower here


def test_adiis_step_blend_and_requirements() -> None:
    cfg = SCFConfig(nspin=1, mixer_variant="adiis", mix_target="Hamiltonian")
    mixer = TikhonovPulayMixer()
    x = jnp.zeros((1, 6))
    with pytest.raises(ValueError):
        mixer.step(x, x + 0.1, _state(x, 4), cfg)  # no aux, no ADIIS buffers
    with pytest.raises(ValueError):
        SCFConfig(mixer_variant="adiis", mix_target="DM")
    state = _state(x, 4, adiis=True)
    rng = np.random.default_rng(2)
    for _ in range(3):
        x_out = x + jnp.asarray(0.05 * rng.standard_normal((1, 6)))
        x_new, state = mixer.step(x, x_out, state, cfg, aux=(jnp.asarray(rng.standard_normal((1, 6))), jnp.array(-1.0)))
        assert x_new.shape == x.shape and bool(jnp.all(jnp.isfinite(x_new)))
        x = x_new
    assert state.history_count == 3 and state.history_d is not None and state.history_e is not None
    assert ADIIS_ERR_LO < ADIIS_ERR_HI


def test_fdf_mixer_keywords(tmp_path) -> None:
    """``SCF.Mixer.Method``, ``SCF.Mix``, ``SCF.Mixer.Weight`` and ``SCF.Mixer.History`` reach ``SCFConfig``."""
    from jax_siesta.io import parse_fdf

    base = (
        "NumberOfAtoms 1\nNumberOfSpecies 1\n%block ChemicalSpeciesLabel\n 1 14 Si\n%endblock ChemicalSpeciesLabel\n"
        "LatticeConstant 10.0 Bohr\n%block LatticeVectors\n 1 0 0\n 0 1 0\n 0 0 1\n%endblock LatticeVectors\n"
        "AtomicCoordinatesFormat Bohr\n%block AtomicCoordinatesAndAtomicSpecies\n 0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    f = tmp_path / "a.fdf"
    f.write_text(base + "SCF.Mixer.Method broyden\nSCF.Mix hamiltonian\nSCF.Mixer.Weight 0.2\nSCF.Mixer.History 7\n")
    cfg = parse_fdf(f)["config"]
    assert (cfg.mixer_variant, cfg.mix_target, cfg.mix_weight, cfg.mix_history) == ("broyden", "Hamiltonian", 0.2, 7)
    g = tmp_path / "b.fdf"
    g.write_text(base + "SCF.Mixer.Method linear\n")
    with pytest.raises(ValueError):
        parse_fdf(g)
