"""Generalized eigensolver: hoisted overlap transform, real Bloch path and one ``eigh`` per k-point (D-032)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_siesta.scf.eigensolver as eig_mod
from jax_siesta.integrals import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json
from jax_siesta.scf.driver import build_h0_sparse, build_monkhorst_pack_mesh
from jax_siesta.scf.eigensolver import GeneralizedEigenSolver, batched_generalized_eigh, batched_overlap_transform
from jax_siesta.species import PSMLBasisLoader, build_sparse_pattern

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parent


def _random_pencil(n_k: int, m: int, seed: int, min_s_eig: float | None = None) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_k, m, m)) + 1j * rng.standard_normal((n_k, m, m))
    h = 0.5 * (a + np.conj(np.swapaxes(a, 1, 2)))
    b = rng.standard_normal((n_k, m, m)) + 1j * rng.standard_normal((n_k, m, m))
    s = np.eye(m)[None] + 0.1 * (b @ np.conj(np.swapaxes(b, 1, 2))) / m
    if min_s_eig is not None:
        w, v = np.linalg.eigh(s)
        w[:, 0] = min_s_eig
        s = (v * w[:, None, :]) @ np.conj(np.swapaxes(v, 1, 2))
    return jnp.asarray(h), jnp.asarray(s)


def test_precomputed_transform_is_identical_and_solves_the_pencil() -> None:
    h, s = _random_pencil(3, 24, seed=1)
    e0, c0 = batched_generalized_eigh(h, s)
    x = batched_overlap_transform(s)
    e1, c1 = batched_generalized_eigh(h, s, x_k=x)
    assert np.array_equal(np.asarray(e0), np.asarray(e1)) and np.array_equal(np.asarray(c0), np.asarray(c1))
    c = np.asarray(c1)
    ch = np.conj(np.swapaxes(c, 1, 2))
    np.testing.assert_allclose(ch @ np.asarray(s) @ c, np.broadcast_to(np.eye(24), (3, 24, 24)), atol=1e-12)
    np.testing.assert_allclose(np.asarray(h) @ c, np.asarray(s) @ c * np.asarray(e1)[:, None, :], atol=1e-10)
    ref = np.stack([np.sort(np.real(np.linalg.eigvals(np.linalg.solve(s[k], h[k])))) for k in range(3)])
    np.testing.assert_allclose(np.asarray(e1), ref, atol=1e-10)


def test_lowdin_branch_for_near_singular_overlap() -> None:
    _, s = _random_pencil(2, 12, seed=2, min_s_eig=1e-9)
    x = np.asarray(batched_overlap_transform(s))
    w, v = np.linalg.eigh(np.asarray(s))
    inv_sqrt = np.where(w > 1e-10, 1.0 / np.sqrt(np.maximum(w, 1e-10)), 0.0)
    # S^-1/2 is Hermitian (the Cholesky transform is triangular); its 1/sqrt(1e-9) column is conditioned to ~1e-7.
    np.testing.assert_allclose(x, np.conj(np.swapaxes(x, 1, 2)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x, (v * inv_sqrt[:, None, :]) @ np.conj(np.swapaxes(v, 1, 2)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("k_grid", [(1, 1, 1), (2, 2, 2)])
def test_real_bloch_path_matches_complex_path(monkeypatch: pytest.MonkeyPatch, k_grid: tuple[int, int, int]) -> None:
    case = load_case_json(ROOT / "manifests" / "cases" / "si_bulk" / "case.json")
    structure, config = case["structure"], case["config"]
    loader = PSMLBasisLoader()
    basis = loader.build_basis(structure, {"Si": loader.load_species("Si", str(ROOT / "fixtures" / "psml" / "Si.psml"))})
    pattern = build_sparse_pattern(structure, basis)
    tc = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
    kmesh = build_monkhorst_pack_mesh(k_grid)
    h0 = build_h0_sparse(tc, config.nspin)
    n_el = basis.total_valence_electrons
    solver = GeneralizedEigenSolver()
    assert eig_mod._real_bloch_problem(kmesh.kpoints_frac, config.nspin)

    def _run(s_x: jax.Array | None) -> list[np.ndarray]:
        out = solver.solve(h0, tc.overlap, pattern, kmesh, n_el, config.electronic_temp_ha, config.nspin, s_transform=s_x)
        return [np.asarray(o) for o in out]

    real_cached = _run(solver.overlap_transform(tc.overlap, pattern, kmesh, config.nspin))
    real_plain = _run(None)
    for a, b in zip(real_cached, real_plain, strict=True):
        assert np.array_equal(a, b)
    monkeypatch.setattr(eig_mod, "_real_bloch_problem", lambda *_: False)
    cplx = _run(None)
    for a, b in zip(real_cached, cplx, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-11)
    # Tr[D S] = N_e
    assert abs(float(np.sum(real_cached[0][0] * np.asarray(tc.overlap))) - n_el) < 1e-9


def test_one_eigh_per_solve_with_precomputed_transform() -> None:
    h, s = _random_pencil(2, 8, seed=3)
    x = batched_overlap_transform(s)
    jaxpr = jax.make_jaxpr(lambda hh, xx: batched_generalized_eigh(hh, xx, x_k=xx))(h, x)

    def _count(jx: Any) -> int:
        # duck-typed walk over sub-jaxprs (ClosedJaxpr has `.jaxpr`, Jaxpr has `.eqns`); the
        # `jax.core.ClosedJaxpr` alias is gone in newer jax releases
        n = 0
        for eqn in jx.eqns:
            n += eqn.primitive.name == "eigh"
            for p in eqn.params.values():
                for q in p if isinstance(p, (tuple, list)) else (p,):
                    inner = getattr(q, "jaxpr", q)
                    if hasattr(inner, "eqns"):
                        n += _count(inner)
        return n

    assert _count(jaxpr.jaxpr) == 1
    full = jax.make_jaxpr(lambda hh, ss: batched_generalized_eigh(hh, ss))(h, s)
    assert _count(full.jaxpr) == 2  # eigh(S) for the branch choice, eigh(X^H H X)


def test_gather_assembly_equals_scatter_assembly() -> None:
    """The host-planned gather assembly equals the scatter-add (used when the pattern is traced)."""
    rng = np.random.default_rng(4)
    n, n_e, n_k = 17, 900, 3
    src = jnp.asarray(np.sort(rng.integers(0, n, n_e)), dtype=jnp.int32)
    dst = jnp.asarray(rng.integers(0, n, n_e), dtype=jnp.int32)
    shifts = jnp.asarray(rng.integers(-2, 3, (n_e, 3)), dtype=jnp.float64)
    kpts = jnp.asarray(rng.uniform(-0.5, 0.5, (n_k, 3)))
    vals = jnp.asarray(rng.standard_normal(n_e) + 1j * rng.standard_normal(n_e))
    planned = eig_mod._bloch_assemble(vals, src, dst, shifts, kpts, n)
    scattered = jax.jit(lambda s_, d_: eig_mod._bloch_assemble(vals, s_, d_, shifts, kpts, n))(src, dst)
    assert eig_mod._assembly_plan(src, dst, n) is not None
    np.testing.assert_allclose(np.asarray(planned), np.asarray(scattered), atol=1e-13, rtol=0)
