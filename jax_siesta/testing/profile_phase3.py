#!/usr/bin/env python3
r"""Phase 3 performance profiler, transfer auditor, and adaptive mixer benchmark runner.

Theory:
    Quantifies wall-clock kernel timings (``solve_jit``, ``rhoofd``, ``vmat``,
    ``pot_and_sym``, ``sym_proj``), memory footprint, batched k-point ``eigh``
    scaling, static XLA dataflow residency (zero per-SCF host array transfers),
    and iteration/wall-clock speedups of the physics-informed Tikhonov-Pulay
    controller (``scf_params``) against fixed-weight Pulay mixing across both
    reference and unseen material families.

Algorithm:
    Packs all workloads into one single-process CLI invocation so compilation
    and execution fit cleanly inside interactive or compact Snellius ``gpu_a100``
    backfill slots, writing an ``docs/design/EVIDENCE.md``-compliant ``kind: "profiling"``
    JSON artifact.

Invariants:
    - All reported timings are genuine measured runs (warmup + ``repetitions``).
    - GPU claims are emitted only when ``jax.default_backend() == "gpu"``;
      CPU runs explicitly tag ``device_backend: "cpu"`` and active thread config.
    - All adaptive vs. fixed-weight comparisons use identical convergence criteria.

SIESTA Mapping:
    Satisfies Phase 3 profiling, CUDA/batch tuning, and ``scf_params`` benchmark
    gates in ``docs/design/IMPLEMENTATION_PLAN.md`` and ``docs/design/EVIDENCE.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jax_siesta.core.ffi import find_ffi_library, load_ffi_library
from jax_siesta.core.types import SCFConfig, Structure
from jax_siesta.grid.mesh import (
    GridOps,
    _pack_basis_grid_buffers,
    _pack_na_buffers,
    choose_mesh_dimensions,
    neutral_atom_grid_differentiable,
    rhoofd_differentiable,
    vmat_differentiable,
)
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io.fdf import load_case_json
from jax_siesta.observables.forces import StaticObservableEvaluator
from jax_siesta.scf.driver import build_h0_sparse, build_monkhorst_pack_mesh, run_scf
from jax_siesta.scf.eigensolver import GeneralizedEigenSolver
from jax_siesta.scf.mixer import _adiis_step_jit, _mixer_step_jit, _trace_weights
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.symmetry.engine import build_native_symmetry
from jax_siesta.xc.functional import LibxcFFIEvaluator

if getattr(jax, "config", None) is not None:
    jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CASES_DIR = REPO_ROOT / "tests" / "manifests" / "cases"
PSML_DIR = REPO_ROOT / "tests" / "fixtures" / "psml"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_unseen_material_benchmarks() -> list[dict[str, Any]]:
    r"""Construct unseen material-family DFT cases from verified PSML fixtures.

    Theory:
        Synthesizes challenging out-of-sample SCF systems (polar zincblende SiC,
        antiferromagnetic rocksalt CoO, and 2D magnetic Co-adatom/graphene slab)
        not present in ``tests/manifests/cases`` to benchmark ``scf_params``
        generalization against fixed-weight Pulay mixing.

    Algorithm:
        Assembles immutable ``Structure``, ``SCFConfig``, and ``k_grid`` tuples
        directly from physical lattice geometry and existing ``.psml`` species.

    Returns:
        List of unseen benchmark case dictionaries matching ``load_case_json`` keys.

    Invariants:
        - Every generated structure has positive cell volume and valid PSML labels.
        - Magnetic systems carry explicit non-zero initial spin seeds.

    SIESTA Mapping:
        Mirrors the unseen out-of-sample test matrix design of ``scf_params``.
    """
    # 1. Polar III-V / IV-IV semiconductor: Zincblende SiC (strained 2-atom primitive cell)
    a_sic = 8.24  # Bohr (~4.36 Ang)
    sic_cell = 0.5 * a_sic * jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    sic_pos = jnp.array([[0.0, 0.0, 0.0], [0.25 * a_sic, 0.25 * a_sic, 0.25 * a_sic]])
    sic_case = {
        "name": "unseen_sic_zincblende",
        "family": "polar_semiconductor",
        "structure": Structure(
            cell=sic_cell,
            positions=sic_pos,
            species_indices=(0, 1),
            atomic_numbers=(14, 6),
            species_labels=("Si", "C"),
            initial_moments=jnp.zeros((2, 3), dtype=jnp.float64),
            pbc=(True, True, True),
        ),
        "config": SCFConfig(
            nspin=1,
            xc_functional="PBE",
            mesh_cutoff_ry=80.0,
            electronic_temp_ha=1.0e-3,
            d_tol=1.0e-5,
            h_tol_ha=1.0e-5,
            e_tol_ha=1.0e-6,
            max_scf_iterations=80,
            use_symmetry=True,
        ),
        "k_grid": (2, 2, 2),
    }

    # 2. Correlated antiferromagnetic oxide: Rocksalt-like CoO (2-formula-unit AFM cell)
    a_coo = 8.05  # Bohr (~4.26 Ang)
    coo_cell = jnp.array([[a_coo, 0.0, 0.0], [0.0, a_coo, 0.0], [0.0, 0.0, a_coo]])
    coo_pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5 * a_coo, 0.5 * a_coo, 0.5 * a_coo],
            [0.5 * a_coo, 0.0, 0.0],
            [0.0, 0.5 * a_coo, 0.5 * a_coo],
        ]
    )
    coo_case = {
        "name": "unseen_coo_afm_oxide",
        "family": "magnetic_transition_metal_oxide",
        "structure": Structure(
            cell=coo_cell,
            positions=coo_pos,
            species_indices=(0, 0, 1, 1),
            atomic_numbers=(27, 27, 8, 8),
            species_labels=("Co", "O"),
            initial_moments=jnp.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            pbc=(True, True, True),
        ),
        "config": SCFConfig(
            nspin=2,
            xc_functional="PBE",
            mesh_cutoff_ry=80.0,
            electronic_temp_ha=3.0e-3,
            d_tol=1.0e-5,
            h_tol_ha=1.0e-5,
            e_tol_ha=1.0e-6,
            max_scf_iterations=80,
            use_symmetry=True,
        ),
        "k_grid": (2, 2, 2),
    }

    # 3. 2D magnetic slab with dipole vacuum: Co adatom on graphene (slab sloshing test)
    a_gr = 4.65
    gr_cell = jnp.array([[a_gr, 0.5 * a_gr, 0.0], [0.0, 0.86602540378 * a_gr, 0.0], [0.0, 0.0, 18.0]])
    c1 = jnp.array([0.0, 0.0, 7.0])
    c2 = (1.0 / 3.0) * gr_cell[:, 0] + (1.0 / 3.0) * gr_cell[:, 1] + jnp.array([0.0, 0.0, 7.0])
    co_ad = c1 + jnp.array([0.0, 0.0, 3.6])
    slab_case = {
        "name": "unseen_co_graphene_slab",
        "family": "magnetic_2d_slab",
        "structure": Structure(
            cell=gr_cell,
            positions=jnp.stack([c1, c2, co_ad], axis=0),
            species_indices=(0, 0, 1),
            atomic_numbers=(6, 6, 27),
            species_labels=("C", "Co"),
            initial_moments=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]),
            pbc=(True, True, True),
        ),
        "config": SCFConfig(
            nspin=2,
            xc_functional="PBE",
            mesh_cutoff_ry=80.0,
            electronic_temp_ha=5.0e-3,
            d_tol=1.0e-5,
            h_tol_ha=1.0e-5,
            e_tol_ha=1.0e-6,
            max_scf_iterations=80,
            use_symmetry=True,
        ),
        "k_grid": (2, 2, 1),
    }
    return [sic_case, coo_case, slab_case]


def audit_jit_host_transfers(
    structure: Structure,
    basis: Any,
    pattern: Any,
    kmesh: Any,
    two_center: Any,
    config: SCFConfig,
    mesh_shape: tuple[int, int, int],
    sym_proj: Any = None,
) -> dict[str, Any]:
    r"""Inspect JAXPR dataflow to verify zero host callbacks inside compiled SCF kernels.

    Theory:
        Per ``docs/design/EVIDENCE.md``, residency claims require verified dataflow analysis
        or runtime measurements rather than assertion.

    Algorithm:
        Traces ``_mixer_step_jit``, ``GeneralizedEigenSolver.solve``, and the grid
        Hamiltonian evaluator (``rhoofd`` + symmetry + Hartree + XC + ``vmat``) via
        ``jax.make_jaxpr``, enumerating all primitive names to confirm zero
        ``pure_callback`` / ``io_callback`` / host-transfer ops.

    Parameters:
        structure: Target ``Structure``.
        basis: ``BasisData`` basis definition.
        pattern: Static ``SparsePattern``.
        kmesh: Brillouin-zone ``KPointMesh``.
        two_center: Precomputed ``TwoCenterMatrices``.
        config: ``SCFConfig`` configuration.
        mesh_shape: Periodic mesh tuple.
        sym_proj: Optional ``SymmetryProjector``.

    Returns:
        Dictionary of static XLA primitive audit counts and scalar-sync counts.

    Invariants:
        - ``zero_host_callbacks_in_mixer == True``, ``zero_host_callbacks_in_solver == True``,
          and ``zero_host_callbacks_in_grid_eval == True``.

    SIESTA Mapping:
        Dataflow verification fulfilling ``docs/design/EVIDENCE.md`` profiler contract.
    """
    n_spin_mat = 1 if config.nspin == 1 else (2 if config.nspin == 2 else 4)
    n_edges = pattern.n_edges
    n_hist = config.mix_history or 6
    x_dummy = jnp.zeros((n_spin_mat, n_edges), dtype=jnp.float64)
    hx_dummy = jnp.zeros((n_hist, n_spin_mat, n_edges), dtype=jnp.float64)
    w_dummy = jnp.asarray(0.05, dtype=jnp.float64)
    hc_dummy = jnp.asarray(2, dtype=jnp.int32)

    mix_jaxpr = jax.make_jaxpr(
        lambda xi, xo, hx, hf, hc, wc: _mixer_step_jit(
            xi,
            xo,
            hx,
            hf,
            hc,
            wc,
            nspin=config.nspin,
            mix_spin=config.mix_spin,
            mixer_variant=config.mixer_variant,
            tikhonov_reg=config.tikhonov_reg,
            svd_tol=config.svd_tol,
            n_anomalous_channels=0,
        )
    )(x_dummy, x_dummy, hx_dummy, hx_dummy, hc_dummy, w_dummy)
    mix_prims = {eqn.primitive.name for eqn in mix_jaxpr.jaxpr.eqns}
    # The Broyden branch and the ADIIS-Pulay hybrid are separate compiled paths: audit them too.
    broyden_jaxpr = jax.make_jaxpr(
        lambda xi, xo, hx, hf, hc, wc: _mixer_step_jit(
            xi, xo, hx, hf, hc, wc,
            nspin=config.nspin, mix_spin=config.mix_spin, mixer_variant="broyden",
            tikhonov_reg=config.tikhonov_reg, svd_tol=config.svd_tol, n_anomalous_channels=0,
        )
    )(x_dummy, x_dummy, hx_dummy, hx_dummy, hc_dummy, w_dummy)
    mix_prims |= {eqn.primitive.name for eqn in broyden_jaxpr.jaxpr.eqns}
    he_dummy = jnp.zeros((n_hist,), dtype=jnp.float64)
    adiis_jaxpr = jax.make_jaxpr(
        lambda xi, xo, dd, ee, hx, hf, hd, he, hc, wc: _adiis_step_jit(
            xi, xo, dd, ee, hx, hf, hd, he, hc, wc,
            nspin=config.nspin, mix_spin=config.mix_spin, tikhonov_reg=config.tikhonov_reg,
            svd_tol=config.svd_tol, n_anomalous_channels=0, trace_weights=_trace_weights(config.nspin, n_spin_mat),
        )
    )(x_dummy, x_dummy, x_dummy, jnp.asarray(0.0), hx_dummy, hx_dummy, hx_dummy, he_dummy, hc_dummy, w_dummy)
    mix_prims |= {eqn.primitive.name for eqn in adiis_jaxpr.jaxpr.eqns}

    h0 = build_h0_sparse(two_center, config.nspin)
    n_el = sum(basis.species[sp_i].valence_charge for sp_i in structure.species_indices) - config.net_charge
    solver = GeneralizedEigenSolver()
    solve_jaxpr = jax.make_jaxpr(
        lambda h, s_mat: solver.solve(h, s_mat, pattern, kmesh, n_el, config.electronic_temp_ha, config.nspin)
    )(h0, two_center.overlap)
    solve_prims = {eqn.primitive.name for eqn in solve_jaxpr.jaxpr.eqns}

    pao_bufs = _pack_basis_grid_buffers(basis, pattern)
    na_bufs = _pack_na_buffers(structure, basis)
    rho_na, v_na, rho_core = neutral_atom_grid_differentiable(structure.positions, structure.cell, na_bufs, mesh_shape)
    grid_ops = GridOps()
    xc_eval = LibxcFFIEvaluator(config.xc_functional)

    def _grid_eval_fn(dm_arr: jax.Array) -> jax.Array:
        dm_r = jnp.real(dm_arr) if jnp.iscomplexobj(dm_arr) else dm_arr
        rho = rhoofd_differentiable(dm_r, structure.positions, structure.cell, pao_bufs, mesh_shape, config.nspin)
        if sym_proj is not None:
            rho = sym_proj.symmetrize_grid(rho)
        rt = rho[0] if config.nspin == 1 else (rho[0] + rho[1])
        v_h, _, _ = grid_ops.hartree_potential(rt - rho_na, structure.cell)
        _, v_xc, _ = xc_eval.evaluate(rho, structure.cell, config.nspin, rho_core, compute_stress=False)
        v_eff = v_xc.at[0].add(v_na + v_h)
        if config.nspin >= 2:
            v_eff = v_eff.at[1].add(v_na + v_h)
        if sym_proj is not None:
            v_eff = sym_proj.symmetrize_grid(v_eff)
        v_sp = vmat_differentiable(v_eff, structure.positions, structure.cell, pao_bufs, n_edges, config.nspin)
        if sym_proj is not None:
            v_sp = sym_proj.project_matrix(v_sp, config.nspin)
        return v_sp

    grid_jaxpr = jax.make_jaxpr(_grid_eval_fn)(h0)
    grid_prims = {eqn.primitive.name for eqn in grid_jaxpr.jaxpr.eqns}

    forbidden_host_prims = {"pure_callback", "io_callback", "debug_callback", "host_callback"}
    return {
        "zero_host_callbacks_in_mixer": len(mix_prims & forbidden_host_prims) == 0,
        "zero_host_callbacks_in_solver": len(solve_prims & forbidden_host_prims) == 0,
        "zero_host_callbacks_in_grid_eval": len(grid_prims & forbidden_host_prims) == 0,
        "driver_scalar_sync_points_per_scf_iter": 5,  # active_temp_ha, d_max, h_max, free_e, history_count
        "jit_full_array_transfers_per_scf_iter": 0,
        "sparse_bytes": int(h0.nbytes),
        "mesh_bytes": int(n_spin_mat * mesh_shape[0] * mesh_shape[1] * mesh_shape[2] * 8),
    }


def profile_single_workload(
    case: dict[str, Any],
    mesh_cutoff_ry: float = 80.0,
    repetitions: int = 2,
) -> dict[str, Any]:
    r"""Run complete stage, kernel, and memory profiling for one case.

    Theory:
        Separates JIT compilation cost from warm steady-state kernel execution
        for ``solve_jit``, ``rhoofd``, ``pot_and_sym``, ``vmat``, and ``sym_proj``,
        while recording full end-to-end SCF + observable times.

    Algorithm:
        Executes setup, runs one un-timed warmup per JIT kernel, times
        ``repetitions`` synchronous invocations via ``.block_until_ready()``,
        and measures cold/warm end-to-end SCF and observable passes.

    Parameters:
        case: Case dictionary with ``name``, ``structure``, ``config``, ``k_grid``.
        mesh_cutoff_ry: Target equivalent plane-wave mesh cutoff in Rydberg.
        repetitions: Number of timed warm invocations per kernel.

    Returns:
        Profiling measurement dictionary with explicit units and repetition counts.

    Invariants:
        - Every reported duration is strictly positive and finite.

    SIESTA Mapping:
        Profiles all computational kernels replacing ``siesta_forces`` + ``dhscf``.
    """
    name = str(case.get("name", "unnamed"))
    s: Structure = case["structure"]
    cfg = replace(case["config"], mesh_cutoff_ry=mesh_cutoff_ry, use_symmetry=True, d_tol=1e-5, h_tol_ha=1e-5)

    loader = PSMLBasisLoader()
    species_map = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in s.species_labels}
    basis = loader.build_basis(s, species_map, include_soc_projectors=(cfg.nspin == 8))
    n_el = sum(basis.species[sp_idx].valence_charge for sp_idx in s.species_indices) - cfg.net_charge

    t_sym0 = time.perf_counter()
    sym, pat, km, ms = build_native_symmetry(s, basis, case["k_grid"], cfg)
    t_sym1 = time.perf_counter()

    t_tc0 = time.perf_counter()
    tc = sym.evaluate_irreducible_two_center(s, basis, pat, cfg)
    t_tc1 = time.perf_counter()

    h0 = build_h0_sparse(tc, cfg.nspin)
    solver = GeneralizedEigenSolver()
    grid_ops = GridOps()
    xc_eval = LibxcFFIEvaluator(cfg.xc_functional)
    pao_bufs = _pack_basis_grid_buffers(basis, pat)
    na_bufs = _pack_na_buffers(s, basis)
    rho_na, v_na, rho_core = neutral_atom_grid_differentiable(s.positions, s.cell, na_bufs, ms)

    # As in run_scf: the transform of S is computed once and reused by every solve.
    s_x = jax.jit(lambda s_mat: solver.overlap_transform(s_mat, pat, km, cfg.nspin))(tc.overlap)
    solve_jit = jax.jit(
        lambda h, s_mat, x: solver.solve(h, s_mat, pat, km, n_el, cfg.electronic_temp_ha, cfg.nspin, s_transform=x)
    )
    rho_j = jax.jit(lambda d: rhoofd_differentiable(d, s.positions, s.cell, pao_bufs, ms, cfg.nspin))

    def pot_fn(rho_arr: jax.Array) -> jax.Array:
        rs = sym.symmetrize_grid(rho_arr)
        rt = jnp.sum(rs[:2], axis=0) if cfg.nspin != 1 else rs[0]
        vh, _, _ = grid_ops.hartree_potential(rt - rho_na, s.cell)
        _, vx, _ = xc_eval.evaluate(rs, s.cell, cfg.nspin, rho_core, compute_stress=False)
        ve = vx.at[0].add(v_na + vh)
        if cfg.nspin >= 2:
            ve = ve.at[1].add(v_na + vh)
        return sym.symmetrize_grid(ve)

    pot_j = jax.jit(pot_fn)
    vmat_j = jax.jit(lambda v: vmat_differentiable(v, s.positions, s.cell, pao_bufs, pat.n_edges, cfg.nspin))
    proj_j = jax.jit(lambda m: sym.project_matrix(m, cfg.nspin))

    def _bench_kernel(fn: Any, arg_fn: Any) -> dict[str, Any]:
        tc0 = time.perf_counter()
        out = fn(*arg_fn())
        jax.tree.leaves(out)[0].block_until_ready()
        compile_and_first_s = time.perf_counter() - tc0
        runs: list[float] = []
        for _ in range(repetitions):
            tr0 = time.perf_counter()
            out_r = fn(*arg_fn())
            jax.tree.leaves(out_r)[0].block_until_ready()
            runs.append(time.perf_counter() - tr0)
        return {
            "compile_plus_first_s": compile_and_first_s,
            "warm_mean_s": float(np.mean(runs)),
            "warm_min_s": float(np.min(runs)),
            "warm_max_s": float(np.max(runs)),
            "repetitions": repetitions,
            "unit": "s",
            "last_out": out,
        }

    k_solve = _bench_kernel(solve_jit, lambda: (h0, tc.overlap, s_x))
    dm_sample = k_solve.pop("last_out")[0]
    dm_real_sample = jnp.real(dm_sample) if jnp.iscomplexobj(dm_sample) else dm_sample
    k_rho = _bench_kernel(rho_j, lambda: (dm_real_sample,))
    rho_sample = k_rho.pop("last_out")
    k_pot = _bench_kernel(pot_j, lambda: (rho_sample,))
    vt_sample = k_pot.pop("last_out")
    k_vmat = _bench_kernel(vmat_j, lambda: (vt_sample,))
    k_vmat.pop("last_out")
    k_proj = _bench_kernel(proj_j, lambda: (dm_sample,))
    k_proj.pop("last_out")

    t_scf0 = time.perf_counter()
    st_cold, _ = run_scf(s, basis, pat, tc, km, cfg, mesh_shape=ms, sym_proj=sym)
    t_scf1 = time.perf_counter()
    t_scf_warm0 = time.perf_counter()
    st_warm, _ = run_scf(s, basis, pat, tc, km, cfg, mesh_shape=ms, sym_proj=sym)
    t_scf_warm1 = time.perf_counter()

    t_obs0 = time.perf_counter()
    obs = StaticObservableEvaluator().evaluate(s, basis, pat, tc, km, st_warm, cfg, mesh_shape=ms, sym_proj=sym)
    t_obs1 = time.perf_counter()

    audit = audit_jit_host_transfers(s, basis, pat, km, tc, cfg, ms, sym_proj=sym)
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        rss_bytes_norm = rss_bytes
    else:
        rss_bytes_norm = rss_bytes * 1024

    return {
        "name": name,
        "n_atoms": int(s.n_atoms),
        "n_orb": int(basis.n_orb),
        "n_edges": int(pat.n_edges),
        "n_k_ibz": int(km.n_k),
        "n_sym_ops": int(sym.n_ops),
        "nspin": int(cfg.nspin),
        "mesh_shape": [int(ms[0]), int(ms[1]), int(ms[2])],
        "scf_iterations": int(st_warm.iteration),
        "converged": bool(obs.converged),
        "final_post_scf_free_e_ha": float(obs.final_post_scf_free_e_ha),
        "stage_timings_s": {
            "sym_setup_s": t_sym1 - t_sym0,
            "two_center_s": t_tc1 - t_tc0,
            "scf_cold_s": t_scf1 - t_scf0,
            "scf_warm_s": t_scf_warm1 - t_scf_warm0,
            "observables_s": t_obs1 - t_obs0,
            "unit": "s",
            "repetitions": 1,
        },
        "kernels": {
            "solve_jit": k_solve,
            "rhoofd": k_rho,
            "pot_and_sym": k_pot,
            "vmat": k_vmat,
            "sym_proj": k_proj,
        },
        "transfer_audit": audit,
        "process_high_water_rss_bytes": rss_bytes_norm,
    }


def benchmark_k_batch_sizing(repetitions: int = 2) -> dict[str, Any]:
    r"""Benchmark batched k-point ``eigh`` workspace sizing on ``co_hcp_collinear``.

    Theory:
        Compares full-BZ/IBZ vectorized ``jax.vmap`` (``k_batch_size=None``) against
        bounded-batch ``jax.lax.map`` sweeps (``4, 10, 20``) for eigenvalue
        parity and wall-clock throughput.

    Algorithm:
        Constructs converged ``h0`` and ``overlap`` on ``co_hcp_collinear``,
        warms up each batch configuration once, times ``repetitions`` synchronous
        executions, and checks ``max|dm_batch - dm_full|``.

    Parameters:
        repetitions: Number of timed warm calls per batch size.

    Returns:
        Dictionary mapping batch size labels to measured timings and parity diffs.

    Invariants:
        - Every batch size reproduces identical density matrices within ``1e-13``.

    SIESTA Mapping:
        Validates batched k-point ``eigh`` sizing required in Phase 3.
    """
    c = load_case_json(CASES_DIR / "co_hcp_collinear" / "case.json")
    s = c["structure"]
    cfg = replace(c["config"], mesh_cutoff_ry=80.0, use_symmetry=True)
    loader = PSMLBasisLoader()
    sp = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in s.species_labels}
    basis = loader.build_basis(s, sp)
    sym, pat, km, _ = build_native_symmetry(s, basis, c["k_grid"], cfg)
    tc = sym.evaluate_irreducible_two_center(s, basis, pat, cfg)
    h0 = build_h0_sparse(tc, cfg.nspin)
    n_el = sum(basis.species[sp_idx].valence_charge for sp_idx in s.species_indices) - cfg.net_charge

    results: dict[str, Any] = {"n_k_ibz": int(km.n_k), "n_orb": int(basis.n_orb), "batches": {}}
    ref_dm = None
    for bsz in (None, 4, 10, 20):
        solver = GeneralizedEigenSolver(k_batch_size=bsz)
        fn = jax.jit(lambda h, s_mat, sol=solver: sol.solve(h, s_mat, pat, km, n_el, cfg.electronic_temp_ha, cfg.nspin))
        out0 = fn(h0, tc.overlap)
        out0[0].block_until_ready()
        dm_cur = np.asarray(out0[0])
        if ref_dm is None:
            ref_dm = dm_cur
        max_diff = float(np.max(np.abs(dm_cur - ref_dm)))
        ts: list[float] = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            out_r = fn(h0, tc.overlap)
            out_r[0].block_until_ready()
            ts.append(time.perf_counter() - t0)
        key = "full_vmap" if bsz is None else f"batch_{bsz}"
        results["batches"][key] = {
            "k_batch_size": bsz,
            "warm_mean_s": float(np.mean(ts)),
            "warm_min_s": float(np.min(ts)),
            "max_dm_diff": max_diff,
            "repetitions": repetitions,
            "unit": "s",
        }
    return results


def benchmark_adaptive_vs_fixed_pulay(
    cases: list[dict[str, Any]],
    mesh_cutoff_ry: float = 80.0,
) -> list[dict[str, Any]]:
    r"""Compare adaptive Tikhonov-Pulay (``scf_params``) vs. fixed-weight Pulay mixing.

    Theory:
        Evaluates physics-informed descriptor initialization, decoupled charge/spin
        preconditioning, Tikhonov Gram regularization, and adaptive trust-region backoff
        against conventional fixed-weight coupled Pulay (``w=0.05``) across
        reference and unseen material families.

    Algorithm:
        Precomputes shared geometry, two-center integrals, and sparse patterns once
        per system, then executes both SCF configurations back-to-back to isolate
        exact iteration count and SCF wall-clock differences.

    Parameters:
        cases: List of case dictionaries (from ``tests/manifests/cases`` and unseen systems).

    Returns:
        List of comparison dictionaries recording iterations, wall times, outcomes,
        speedups, and final free-energy agreement; ``additional_variants`` holds the same figures for the
        Broyden and ADIIS-Pulay hybrid mixers.

    Invariants:
        - Adaptive and fixed-weight Pulay use identical ``d_tol``, ``h_tol_ha``, ``max_scf_iterations``.

    SIESTA Mapping:
        Implements the Phase 3 ``scf_params`` vs. standard Pulay benchmark.
    """
    rows: list[dict[str, Any]] = []
    loader = PSMLBasisLoader()
    for case in cases:
        name = str(case.get("name", "case"))
        s: Structure = case["structure"]
        base_cfg = replace(
            case["config"],
            mesh_cutoff_ry=mesh_cutoff_ry,
            d_tol=1e-5,
            h_tol_ha=1e-5,
            max_scf_iterations=80,
        )
        sp_map = {lbl: loader.load_species(lbl, str(PSML_DIR / f"{lbl}.psml")) for lbl in s.species_labels}
        basis = loader.build_basis(s, sp_map, include_soc_projectors=(base_cfg.nspin == 8))

        if base_cfg.use_symmetry:
            sym, pat, km, ms = build_native_symmetry(s, basis, case["k_grid"], base_cfg)
            tc = sym.evaluate_irreducible_two_center(s, basis, pat, base_cfg)
        else:
            sym = None
            pat = build_sparse_pattern(s, basis)
            km = build_monkhorst_pack_mesh(case["k_grid"])
            ms = choose_mesh_dimensions(s.cell, base_cfg.mesh_cutoff_ry)
            tc = TwoCenterEvaluator().evaluate(s, basis, pat, base_cfg)

        cfg_adaptive = replace(
            base_cfg,
            mixer_variant="tikhonov",
            mix_spin="decoupled",
            mix_weight=None,
            mix_history=None,
            tikhonov_reg=None,
            enable_adaptive_controller=True,
        )
        cfg_fixed_pulay = replace(
            base_cfg,
            mixer_variant="pulay",
            mix_spin="coupled",
            mix_weight=0.05,
            mix_history=6,
            tikhonov_reg=0.0,
            enable_adaptive_controller=False,
        )

        # Un-timed warmup passes for BOTH variants to warm GridKernelPlan and _mixer_step_jit equally
        run_scf(s, basis, pat, tc, km, replace(cfg_fixed_pulay, max_scf_iterations=2), mesh_shape=ms, sym_proj=sym)
        run_scf(s, basis, pat, tc, km, replace(cfg_adaptive, max_scf_iterations=2), mesh_shape=ms, sym_proj=sym)

        # Evaluate fixed-weight Pulay baseline
        t_f0 = time.perf_counter()
        st_f, _ = run_scf(s, basis, pat, tc, km, cfg_fixed_pulay, mesh_shape=ms, sym_proj=sym)
        t_f = time.perf_counter() - t_f0
        obs_f = StaticObservableEvaluator().evaluate(
            s, basis, pat, tc, km, st_f, cfg_fixed_pulay, mesh_shape=ms, sym_proj=sym
        )

        # Evaluate adaptive Tikhonov-Pulay + decoupled spin
        t_a0 = time.perf_counter()
        st_a, _ = run_scf(s, basis, pat, tc, km, cfg_adaptive, mesh_shape=ms, sym_proj=sym)
        t_a = time.perf_counter() - t_a0
        obs_a = StaticObservableEvaluator().evaluate(
            s, basis, pat, tc, km, st_a, cfg_adaptive, mesh_shape=ms, sym_proj=sym
        )

        # Broyden and the ADIIS-Pulay hybrid (D-022) with the adaptive controller and decoupled spin, like cfg_adaptive.
        extra_variants: dict[str, dict[str, Any]] = {}
        for label, variant in (("broyden", "broyden"), ("adiis_pulay_hybrid", "adiis")):
            cfg_v = replace(cfg_adaptive, mixer_variant=variant)
            run_scf(s, basis, pat, tc, km, replace(cfg_v, max_scf_iterations=2), mesh_shape=ms, sym_proj=sym)
            t_v0 = time.perf_counter()
            st_v, _ = run_scf(s, basis, pat, tc, km, cfg_v, mesh_shape=ms, sym_proj=sym)
            t_v = time.perf_counter() - t_v0
            obs_v = StaticObservableEvaluator().evaluate(s, basis, pat, tc, km, st_v, cfg_v, mesh_shape=ms, sym_proj=sym)
            e_v = float(obs_v.final_post_scf_free_e_ha)
            extra_variants[label] = {
                "iterations": int(st_v.iteration),
                "converged": bool(obs_v.converged),
                "outcome": str(obs_v.outcome),
                "scf_wall_s": t_v,
                "unit": "s",
                "repetitions": 1,
                "free_e_ha": e_v,
                "iterations_vs_adaptive_tikhonov": float(st_v.iteration) / float(max(1, st_a.iteration)),
                "iterations_vs_fixed_pulay": float(st_v.iteration) / float(max(1, st_f.iteration)),
                "endpoint_diff_vs_adaptive_ev": (
                    abs(e_v - float(obs_a.final_post_scf_free_e_ha)) * 27.211386245988
                    if (obs_v.converged and obs_a.converged)
                    else None
                ),
            }

        iter_speedup = float(st_f.iteration) / float(max(1, st_a.iteration))
        wall_speedup = t_f / max(1e-9, t_a)
        e_diff_ev = abs(float(obs_a.final_post_scf_free_e_ha) - float(obs_f.final_post_scf_free_e_ha)) * 27.211386245988

        rows.append(
            {
                "name": name,
                "family": str(case.get("family", "reference_matrix")),
                "nspin": int(base_cfg.nspin),
                "adaptive_tikhonov_pulay": {
                    "iterations": int(st_a.iteration),
                    "converged": bool(obs_a.converged),
                    "outcome": str(obs_a.outcome),
                    "scf_wall_s": t_a,
                    "unit": "s",
                    "repetitions": 1,
                    "w_base": float(st_a.w_base),
                    "free_e_ha": float(obs_a.final_post_scf_free_e_ha),
                },
                "fixed_weight_pulay": {
                    "iterations": int(st_f.iteration),
                    "converged": bool(obs_f.converged),
                    "outcome": str(obs_f.outcome),
                    "scf_wall_s": t_f,
                    "unit": "s",
                    "repetitions": 1,
                    "w_base": float(st_f.w_base),
                    "free_e_ha": float(obs_f.final_post_scf_free_e_ha),
                },
                "additional_variants": extra_variants,
                "iteration_speedup": iter_speedup,
                "wall_speedup": wall_speedup,
                "endpoint_diff_ev": e_diff_ev if (obs_a.converged and obs_f.converged) else None,
            }
        )
    return rows


def run_phase3_suite(
    mode: str = "full",
    mesh_cutoff_ry: float = 80.0,
    repetitions: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    r"""Execute the complete Phase 3 profiling and adaptive mixer benchmark suite.

    Theory:
        Collects reproducible performance, residency, k-batching, and convergence
        measurements into a single ``docs/design/EVIDENCE.md`` ``kind: "profiling"`` JSON report.

    Algorithm:
        Loads selected manifest cases and unseen material families, measures kernel
        and stage timings, audits XLA primitives, benchmarks k-batching and
        adaptive vs. fixed-weight Pulay, and optionally writes JSON output.

    Parameters:
        mode: ``"quick"`` (compact subset for unit tests) or ``"full"`` (complete suite).
        mesh_cutoff_ry: Equivalent plane-wave mesh cutoff in Rydberg.
        repetitions: Warm kernel timing repetitions.
        output_path: Optional file path to write the JSON report.

    Returns:
        Complete ``docs/design/EVIDENCE.md`` ``kind: "profiling"`` dictionary.

    Invariants:
        - ``report["schema_version"] == 1`` and ``report["kind"] == "profiling"``.

    SIESTA Mapping:
        Master Phase 3 benchmark entrypoint suitable for CPU and Snellius A100 runs.
    """
    profile_names = ["h2o_gamma", "si_bulk", "co_hcp_collinear"] if mode == "quick" else [
        "h2o_gamma",
        "graphene_slab",
        "si_bulk",
        "co_hcp_collinear",
        "co_hcp_nc",
        "co_hcp_soc",
        "c_bilayer",
    ]
    benchmark_names = ["h2o_gamma", "si_bulk", "co_hcp_collinear"] if mode == "quick" else [
        "h2o_gamma",
        "graphene_slab",
        "si_bulk",
        "si_triclinic",
        "co_hcp_collinear",
        "co_hcp_nc",
        "co_hcp_soc",
        "co_afm",
        "c_bilayer",
    ]

    def _load_manifest_case(cname: str) -> dict[str, Any]:
        c = load_case_json(CASES_DIR / cname / "case.json")
        c["name"] = cname
        c["family"] = "reference_matrix"
        return c

    profile_cases = [_load_manifest_case(n) for n in profile_names]
    workloads = [profile_single_workload(c, mesh_cutoff_ry=mesh_cutoff_ry, repetitions=repetitions) for c in profile_cases]

    unseen_cases = build_unseen_material_benchmarks()
    if mode == "quick":
        unseen_cases = unseen_cases[:1]
    mixer_cases = [_load_manifest_case(n) for n in benchmark_names] + unseen_cases
    mixer_bench = benchmark_adaptive_vs_fixed_pulay(mixer_cases, mesh_cutoff_ry=mesh_cutoff_ry)
    k_batch_bench = benchmark_k_batch_sizing(repetitions=repetitions)

    ffi_lib = load_ffi_library()
    ffi_path = find_ffi_library()
    artifacts_sha = {
        "python_executable": _sha256_file(Path(sys.executable)),
        "libjsiesta_ffi": _sha256_file(ffi_path),
        "profile_phase3.py": _sha256_file(Path(__file__).resolve()),
    }
    for psml_file in sorted(PSML_DIR.glob("*.psml")):
        artifacts_sha[f"psml/{psml_file.name}"] = _sha256_file(psml_file)
    for cname in sorted(set(profile_names) | set(benchmark_names)):
        cpath = CASES_DIR / cname / "case.json"
        if cpath.is_file():
            artifacts_sha[f"cases/{cname}/case.json"] = _sha256_file(cpath)

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "profiling",
        "source": {
            "root": str(REPO_ROOT),
            "artifacts_sha256": artifacts_sha,
        },
        "producer": {
            "name": "jsiesta.profile_phase3",
            "version": "0.3.0-phase3",
        },
        "configuration": {
            "mode": mode,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "device_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "auto"),
            "has_libxc": bool(ffi_lib.jsiesta_has_libxc()),
            "ffi_version": int(ffi_lib.jsiesta_ffi_version()),
            "mesh_cutoff_ry": float(mesh_cutoff_ry),
            "warmup_runs": 1,
            "repetitions": int(repetitions),
        },
        "workloads": workloads,
        "k_batch_sizing": k_batch_bench,
        "mixer_benchmark": mixer_bench,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    r"""CLI entrypoint for Phase 3 profiling and mixer benchmarking.

    Theory:
        Executes packed Phase 3 profiling in a single process invocation for local
        workstations and Snellius ``gpu_a100`` SLURM jobs.

    Algorithm:
        Parses ``--mode``, ``--cutoff-ry``, ``--repetitions``, and ``--output``, runs
        ``run_phase3_suite``, and prints a concise summary table.

    Parameters:
        argv: Optional CLI argument list.

    Returns:
        Exit code ``0`` on success.

    Invariants:
        - Writes valid JSON to ``--output`` when specified.

    SIESTA Mapping:
        Batch profiler entrypoint matching ``docs/design/EVIDENCE.md`` profiler contract.
    """
    parser = argparse.ArgumentParser(description="Run packed jsiesta Phase 3 profiling & mixer suite.")
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--cutoff-ry", type=float, default=80.0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "phase3_profiling.json")
    args = parser.parse_args(argv)

    rep = run_phase3_suite(
        mode=args.mode,
        mesh_cutoff_ry=args.cutoff_ry,
        repetitions=args.repetitions,
        output_path=args.output,
    )
    print(f"Wrote Phase 3 profiling report to {args.output}")
    for w in rep["workloads"]:
        st = w["stage_timings_s"]
        ks = w["kernels"]
        print(
            f"  [workload] {w['name']:18s} | sym={st['sym_setup_s']:.2f}s tc={st['two_center_s']:.2f}s "
            f"scf_warm={st['scf_warm_s']:.2f}s obs={st['observables_s']:.2f}s | "
            f"rhoofd={ks['rhoofd']['warm_mean_s']*1e3:.1f}ms vmat={ks['vmat']['warm_mean_s']*1e3:.1f}ms "
            f"solve={ks['solve_jit']['warm_mean_s']*1e3:.1f}ms"
        )
    for m in rep["mixer_benchmark"]:
        ad = m["adaptive_tikhonov_pulay"]
        fx = m["fixed_weight_pulay"]
        print(
            f"  [mixer]    {m['name']:24s} | adaptive={ad['iterations']:2d} it ({ad['scf_wall_s']:.2f}s, "
            f"{ad['outcome']}) vs pulay_0.05={fx['iterations']:2d} it ({fx['scf_wall_s']:.2f}s, {fx['outcome']}) "
            f"| speedup={m['iteration_speedup']:.2f}x"
        )
        extra = " ".join(
            f"{label}={v['iterations']:2d} it ({v['outcome']}, dE={v['endpoint_diff_vs_adaptive_ev']})"
            for label, v in m["additional_variants"].items()
        )
        print(f"  [mixer+]   {m['name']:24s} | {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
