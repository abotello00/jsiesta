"""Phase 3 tests: profiling contract, zero host-transfer audit, k-batch eigh parity, and adaptive benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_siesta.core.types import BasisData, SCFConfig, SCFState, SparsePattern, Structure, TwoCenterMatrices
from jax_siesta.grid.mesh import _pack_basis_grid_buffers, rhoofd_differentiable, vmat_differentiable
from jax_siesta.integrals.two_center import TwoCenterEvaluator, _iter_kb_shell_pairs
from jax_siesta.observables.forces import _kb_forces_and_stress
from jax_siesta.scf.eigensolver import batched_generalized_eigh
from jax_siesta.scf.mixer import TikhonovPulayMixer, _to_charge_spin_basis
from jax_siesta.species.psml import PSMLBasisLoader, build_sparse_pattern
from jax_siesta.testing.profile_phase3 import build_unseen_material_benchmarks, run_phase3_suite

jax.config.update("jax_enable_x64", True)


def test_batched_generalized_eigh_parity() -> None:
    rng = np.random.default_rng(7)
    n_k, m = 10, 8
    h_raw = rng.standard_normal((n_k, m, m)) + 1j * rng.standard_normal((n_k, m, m))
    h_k = jnp.asarray(0.5 * (h_raw + np.swapaxes(h_raw.conj(), -1, -2)))
    a_raw = rng.standard_normal((n_k, m, m)) + 0.1j * rng.standard_normal((n_k, m, m))
    s_k = jnp.asarray(np.matmul(a_raw, np.swapaxes(a_raw.conj(), -1, -2)) + 0.5 * np.eye(m)[None, :, :])

    ev_full, _ = batched_generalized_eigh(h_k, s_k, k_batch_size=None)
    ev_b4, vec_b4 = batched_generalized_eigh(h_k, s_k, k_batch_size=4)
    assert float(jnp.max(jnp.abs(ev_full - ev_b4))) < 1e-13
    ident = jnp.einsum("kmi,kmj,kjl->kil", jnp.conj(vec_b4), s_k, vec_b4)
    err_ortho = float(jnp.max(jnp.abs(ident - jnp.eye(m)[None, :, :])))
    assert err_ortho < 1e-12


def test_phase3_profiler_evidence_contract_and_adaptive_benchmark(tmp_path: Path) -> None:
    unseen = build_unseen_material_benchmarks()
    assert len(unseen) == 3
    for case in unseen:
        assert float(case["structure"].volume) > 0.0

    out_json = tmp_path / "phase3_quick.json"
    rep = run_phase3_suite(mode="quick", mesh_cutoff_ry=70.0, repetitions=1, output_path=out_json)
    assert rep["kind"] == "profiling"
    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))

    # Verify docs/design/EVIDENCE.md profiling envelope fields
    assert loaded["schema_version"] == 1
    assert loaded["kind"] == "profiling"
    assert "libjsiesta_ffi" in loaded["source"]["artifacts_sha256"]
    assert loaded["configuration"]["repetitions"] == 1
    assert loaded["configuration"]["device_backend"] in ("cpu", "gpu", "tpu")

    assert len(loaded["workloads"]) == 3
    assert "cases/h2o_gamma/case.json" in loaded["source"]["artifacts_sha256"]
    for w in loaded["workloads"]:
        assert w["converged"] is True
        assert w["transfer_audit"]["zero_host_callbacks_in_mixer"] is True
        assert w["transfer_audit"]["zero_host_callbacks_in_solver"] is True
        assert w["transfer_audit"]["zero_host_callbacks_in_grid_eval"] is True
        assert w["transfer_audit"]["jit_full_array_transfers_per_scf_iter"] == 0
        for kname, kstats in w["kernels"].items():
            assert kstats["warm_mean_s"] > 0.0, f"{w['name']}.{kname} timing must be positive"
            assert kstats["unit"] == "s"

    # K-batch sizing parity
    batches = loaded["k_batch_sizing"]["batches"]
    assert batches["batch_4"]["max_dm_diff"] < 1e-12

    # Adaptive vs fixed Pulay benchmark records convergence & speedups
    assert len(loaded["mixer_benchmark"]) == 4
    for row in loaded["mixer_benchmark"]:
        assert row["adaptive_tikhonov_pulay"]["converged"] is True
        assert row["adaptive_tikhonov_pulay"]["unit"] == "s"
        assert row["fixed_weight_pulay"]["unit"] == "s"
        assert np.isfinite(row["iteration_speedup"])
        assert np.isfinite(row["wall_speedup"])
        # Broyden and the ADIIS-Pulay hybrid are benchmarked on the same systems and must reach the same endpoint.
        assert set(row["additional_variants"]) == {"broyden", "adiis_pulay_hybrid"}
        for label, v in row["additional_variants"].items():
            assert v["converged"] is True, (row["name"], label, v["outcome"])
            assert v["endpoint_diff_vs_adaptive_ev"] < 2e-4, (row["name"], label)


def test_nspin8_kb_forces_and_stress_against_finite_differences() -> None:
    psml_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "psml"
    loader = PSMLBasisLoader()
    sp_co = loader.load_species("Co", str(psml_dir / "Co.psml"))
    cell = np.diag([14.0, 14.0, 14.0])
    pos = np.array([[5.0, 5.5, 6.0], [8.8, 6.2, 6.7]], dtype=np.float64)
    st = Structure(
        cell=jnp.asarray(cell),
        positions=jnp.asarray(pos),
        species_indices=(0, 0),
        atomic_numbers=(27, 27),
        species_labels=("Co",),
        initial_moments=jnp.array([[1.0, 0.5, 1.2], [-0.4, 0.9, -1.0]]),
        pbc=(False, False, False),
    )
    cfg = SCFConfig(nspin=8)
    basis: BasisData = loader.build_basis(st, {"Co": sp_co}, include_soc_projectors=True)
    pat: SparsePattern = build_sparse_pattern(st, basis)
    tc: TwoCenterMatrices = TwoCenterEvaluator().evaluate(st, basis, pat, cfg)

    # Construct non-symmetric complex Hermitian DM [D_uu, D_dd, Re(D_ud), Im(D_ud)]
    rng = np.random.default_rng(19)
    n_orb = basis.n_orb
    a0 = rng.standard_normal((n_orb, n_orb)) + 1j * rng.standard_normal((n_orb, n_orb))
    a1 = rng.standard_normal((n_orb, n_orb)) + 1j * rng.standard_normal((n_orb, n_orb))
    h_uu = 0.5 * (a0 + a0.conj().T)
    h_dd = 0.5 * (a1 + a1.conj().T)
    a2 = rng.standard_normal((n_orb, n_orb)) + 1j * rng.standard_normal((n_orb, n_orb))
    src = np.asarray(pat.src_orb, dtype=np.int32)
    dst = np.asarray(pat.dst_unit_orb, dtype=np.int32)
    dm_out = jnp.asarray(
        np.stack([h_uu[src, dst], h_dd[src, dst], np.real(a2[src, dst]), np.imag(a2[src, dst])], axis=0),
        dtype=jnp.complex128,
    )

    _, f_ana, stress_ana = _kb_forces_and_stress(dm_out, st, basis, pat, tc, nspin=8)
    assert float(np.max(np.abs(np.sum(f_ana, axis=0)))) < 1e-12

    eps = 1e-5
    for atom_idx in range(2):
        for ax in range(3):
            p_p = pos.copy()
            p_m = pos.copy()
            p_p[atom_idx, ax] += eps
            p_m[atom_idx, ax] -= eps
            st_p = Structure(
                cell=jnp.asarray(cell), positions=jnp.asarray(p_p),
                species_indices=(0, 0), atomic_numbers=(27, 27), species_labels=("Co",),
                initial_moments=st.initial_moments, pbc=(False, False, False),
            )
            st_m = Structure(
                cell=jnp.asarray(cell), positions=jnp.asarray(p_m),
                species_indices=(0, 0), atomic_numbers=(27, 27), species_labels=("Co",),
                initial_moments=st.initial_moments, pbc=(False, False, False),
            )
            tc_p = TwoCenterEvaluator().evaluate(st_p, basis, pat, cfg)
            tc_m = TwoCenterEvaluator().evaluate(st_m, basis, pat, cfg)
            ep, _, _ = _kb_forces_and_stress(dm_out, st_p, basis, pat, tc_p, nspin=8)
            em, _, _ = _kb_forces_and_stress(dm_out, st_m, basis, pat, tc_m, nspin=8)
            f_fd = -(float(ep) - float(em)) / (2.0 * eps)
            assert abs(f_fd - float(f_ana[atom_idx, ax])) < 1e-9

    # Check stress component (0, 2) against homogeneous strain finite difference
    vol = float(abs(np.linalg.det(cell)))
    eps_s = 1e-5
    d_p = np.eye(3)
    d_m = np.eye(3)
    d_p[0, 2] += eps_s
    d_m[0, 2] -= eps_s
    st_sp = Structure(
        cell=jnp.asarray(d_p @ cell), positions=jnp.asarray(pos @ d_p.T),
        species_indices=(0, 0), atomic_numbers=(27, 27), species_labels=("Co",),
        initial_moments=st.initial_moments, pbc=(False, False, False),
    )
    st_sm = Structure(
        cell=jnp.asarray(d_m @ cell), positions=jnp.asarray(pos @ d_m.T),
        species_indices=(0, 0), atomic_numbers=(27, 27), species_labels=("Co",),
        initial_moments=st.initial_moments, pbc=(False, False, False),
    )
    tc_sp = TwoCenterEvaluator().evaluate(st_sp, basis, pat, cfg)
    tc_sm = TwoCenterEvaluator().evaluate(st_sm, basis, pat, cfg)
    esp, _, _ = _kb_forces_and_stress(dm_out, st_sp, basis, pat, tc_sp, nspin=8)
    esm, _, _ = _kb_forces_and_stress(dm_out, st_sm, basis, pat, tc_sm, nspin=8)
    s_fd = (float(esp) - float(esm)) / (2.0 * eps_s * vol)
    # _kb_forces_and_stress applies d(R_k - R_ia)/d(eps_02) directly cleanly
    assert abs(s_fd - float(stress_ana[0, 2])) < 1e-9


def test_iter_kb_shell_pairs_noncontiguous_and_mixer_padding_parity(monkeypatch) -> None:
    # 1. Verify _iter_kb_shell_pairs preserves exact groups on interleaved non-contiguous k_inst
    psml_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "psml"
    loader = PSMLBasisLoader()
    sp_si = loader.load_species("Si", str(psml_dir / "Si.psml"))
    st = Structure(
        cell=jnp.diag(jnp.array([10.26, 10.26, 10.26])),
        positions=jnp.array([[0.0, 0.0, 0.0], [2.565, 2.565, 2.565]]),
        species_indices=(0, 0),
        atomic_numbers=(14, 14),
        species_labels=("Si",),
        initial_moments=jnp.zeros((2, 3)),
    )
    basis = loader.build_basis(st, {"Si": sp_si})
    pat = build_sparse_pattern(st, basis)
    tc = TwoCenterEvaluator().evaluate(st, basis, pat, SCFConfig(nspin=1))
    assert tc.kb_link_p is not None and tc.kb_link_cell_shift is not None
    p_np = np.asarray(tc.kb_link_p)
    l_mu = np.asarray(tc.kb_link_orb, dtype=np.int64)
    l_ki = np.asarray(tc.kb_link_kb, dtype=np.int64)
    l_tk = np.asarray(tc.kb_link_cell_shift, dtype=np.int64)
    shell_dims = [2 * int(k_l) + 1 for _ in range(2) for k_l in sp_si.kb_l]

    ref_vals = {
        k_i: (r, c, e, pm)
        for k_i, _, _, r, c, e, pm, _ in _iter_kb_shell_pairs(
            pat, basis.n_orb, len(shell_dims), shell_dims, l_mu, l_ki, l_tk, p_np
        )
    }
    # Scramble across k_inst/tk blocks while preserving m2 order inside each (k_inst, tk, mu)
    perm_blocks = np.lexsort((np.arange(len(l_mu)), l_tk[:, 2], l_ki))
    shuf_vals = {
        k_i: (r, c, e, pm)
        for k_i, _, _, r, c, e, pm, _ in _iter_kb_shell_pairs(
            pat, basis.n_orb, len(shell_dims), shell_dims,
            l_mu[perm_blocks], l_ki[perm_blocks], l_tk[perm_blocks], p_np[perm_blocks],
        )
    }
    assert set(ref_vals) == set(shuf_vals)
    for k_i, (r0, c0, e0, pm0) in ref_vals.items():
        r1, c1, e1, pm1 = shuf_vals[k_i]
        assert np.allclose(pm0[r0] * pm0[c0], pm1[r1] * pm1[c1], atol=1e-13)
        assert np.array_equal(e0, e1)

    # 2. Verify padded JIT mixer vs tight unpadded (m, m) Gram solve parity across all variants
    rng = np.random.default_rng(11)
    n_edges, max_hist, m_prev = 16, 6, 3
    hx_garbage = jnp.asarray(rng.standard_normal((max_hist, 2, n_edges)))
    hf_garbage = jnp.asarray(rng.standard_normal((max_hist, 2, n_edges)))
    # Replace first m_prev slots cleanly
    hx_clean = hx_garbage.at[:m_prev].set(jnp.asarray(rng.standard_normal((m_prev, 2, n_edges))))
    hf_clean = hf_garbage.at[:m_prev].set(jnp.asarray(0.05 * rng.standard_normal((m_prev, 2, n_edges))))
    x_in = jnp.asarray(rng.standard_normal((2, n_edges)))
    x_out = x_in + jnp.asarray(0.05 * rng.standard_normal((2, n_edges)))
    state_pad = SCFState(
        iteration=m_prev, h_sparse=x_in, dm_in=x_in, dm_out=x_out, edm_out=x_in,
        fermi_level_ha=jnp.array(0.0), entropy_ha=jnp.array(0.0), last_scf_free_e_ha=jnp.array(0.0),
        w_cur=jnp.array(0.1), w_base=jnp.array(0.1), active_temp_ha=jnp.array(1e-3),
        streak_bad=0, prev_d_max=jnp.array(0.1), prev_h_max=jnp.array(0.1),
        history_x=hx_clean, history_f=hf_clean, history_count=m_prev,
    )
    state_tight = SCFState(
        iteration=m_prev, h_sparse=x_in, dm_in=x_in, dm_out=x_out, edm_out=x_in,
        fermi_level_ha=jnp.array(0.0), entropy_ha=jnp.array(0.0), last_scf_free_e_ha=jnp.array(0.0),
        w_cur=jnp.array(0.1), w_base=jnp.array(0.1), active_temp_ha=jnp.array(1e-3),
        streak_bad=0, prev_d_max=jnp.array(0.1), prev_h_max=jnp.array(0.1),
        history_x=jnp.concatenate([hx_clean[:m_prev], x_in[None, ...]], axis=0),
        history_f=jnp.concatenate([hf_clean[:m_prev], (x_out - x_in)[None, ...]], axis=0),
        history_count=m_prev,
    )
    mixer = TikhonovPulayMixer()
    for variant in ("tikhonov", "pulay", "broyden"):
        cfg_v = SCFConfig(nspin=2, mixer_variant=variant, mix_spin="decoupled")
        xn_pad, _ = mixer.step(x_in, x_out, state_pad, cfg_v)
        # With exact (m_prev+1)-size buffer, no unused padding slots exist
        xn_tight, _ = mixer.step(x_in, x_out, state_tight, cfg_v)
        assert float(jnp.max(jnp.abs(xn_pad - xn_tight))) < 1e-13
        # Keep _to_charge_spin_basis exercised
        assert _to_charge_spin_basis(xn_pad, 2).shape == xn_pad.shape

    # 3. Verify grid FFI OMP_NUM_THREADS=1 vs multi-thread parity
    packed = _pack_basis_grid_buffers(basis, pat)
    dm_test = jnp.ones((1, pat.n_edges), dtype=jnp.float64) * 0.05
    v_test = jnp.ones((1, 16, 16, 16), dtype=jnp.float64) * 0.25
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rho_1 = rhoofd_differentiable(dm_test, st.positions, st.cell, packed, (16, 16, 16), 1)
    v_1 = vmat_differentiable(v_test, st.positions, st.cell, packed, pat.n_edges, 1)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    rho_4 = rhoofd_differentiable(dm_test, st.positions, st.cell, packed, (16, 16, 16), 1)
    v_4 = vmat_differentiable(v_test, st.positions, st.cell, packed, pat.n_edges, 1)
    assert float(jnp.max(jnp.abs(rho_1 - rho_4))) < 1e-13
    assert float(jnp.max(jnp.abs(v_1 - v_4))) < 1e-13
