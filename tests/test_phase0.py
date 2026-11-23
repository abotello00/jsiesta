"""Phase 0 verification gate tests: PyTrees, FFI ping, Knowledge Graph, Docs, and Frozen Matrix."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from jax_siesta.core import (
    BasisAndPseudos,
    BasisData,
    DispersionModel,
    EigenSolver,
    ExchangeCorrelation,
    KPointMesh,
    ObservableEvaluator,
    Observables,
    RealSpaceGridOps,
    SCFConfig,
    SCFMixer,
    SCFState,
    SparsePattern,
    SpeciesData,
    Structure,
    SymmetryProjector,
    TwoCenterIntegrals,
    TwoCenterMatrices,
)
from jax_siesta.core.ffi import ffi_backend_info, ffi_ping
from jax_siesta.testing.external_refs import ref_path
from kg.build_kg import build_kg
from kg.server_mcp import ProgressiveKG

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_core_pytrees_and_protocols() -> None:
    """Verify two-call jax.jit stability and protocol contracts for all Phase 0 PyTrees."""
    struct = Structure(
        cell=jnp.eye(3) * 10.0,
        positions=jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        species_indices=(0, 0),
        atomic_numbers=(14, 14),
        species_labels=("Si",),
        initial_moments=jnp.zeros((2, 3)),
    )
    jit_vol = jax.jit(lambda s: s.volume + jnp.sum(s.positions))
    assert jnp.allclose(jit_vol(struct), jit_vol(struct))

    sp = SpeciesData(
        symbol="H",
        atomic_number=1,
        valence_charge=1.0,
        mass_amu=1.008,
        r_grid=jnp.linspace(0.0, 5.0, 4),
        v_local=jnp.zeros(4),
        v_na=jnp.zeros(4),
        rho_na=jnp.zeros(4),
        rho_core=jnp.zeros(4),
        pao_r=jnp.linspace(0.0, 5.0, 4),
        pao_values=jnp.ones((1, 4)),
        pao_l=(0,),
        pao_n=(1,),
        pao_zeta=(1,),
        pao_iz=(0,),
        pao_rcut=(5.0,),
        kb_r=jnp.linspace(0.0, 5.0, 4),
        kb_projectors=jnp.zeros((1, 4)),
        kb_l=(0,),
        kb_j2=(0,),
        kb_energies=jnp.array([-0.2]),
        kb_rcut=(1.1,),
    )
    basis = BasisData(
        species=(sp,),
        n_orb=1,
        n_kb=1,
        total_valence_electrons=1.0,
        orb_atom=(0,),
        orb_species=(0,),
        orb_shell=(0,),
        orb_l=(0,),
        orb_m=(0,),
        orb_iz=(0,),
        orb_rcut=jnp.array([5.0]),
        atom_orb_offsets=(0, 1),
        kb_atom=(0,),
        kb_species=(0,),
        kb_l=(0,),
        kb_m=(0,),
        kb_j2=(0,),
        kb_energies=jnp.array([-0.2]),
        kb_rcut=jnp.array([1.1]),
    )
    pattern = SparsePattern(
        n_orb=1,
        n_super_cells=1,
        nsc=(1, 1, 1),
        cell_shifts=jnp.zeros((1, 3), dtype=jnp.int32),
        src_orb=jnp.array([0], dtype=jnp.int32),
        dst_unit_orb=jnp.array([0], dtype=jnp.int32),
        edge_cell_idx=jnp.array([0], dtype=jnp.int32),
        edge_vectors=jnp.zeros((1, 3)),
        row_ptr=jnp.array([0, 1], dtype=jnp.int32),
        transpose_edge_idx=jnp.array([0], dtype=jnp.int32),
    )
    jit_pat = jax.jit(lambda p: p.edge_vectors.sum() + p.src_orb.sum() + p.row_ptr.sum())
    assert jnp.allclose(jit_pat(pattern), jit_pat(pattern))

    kmesh = KPointMesh(
        kpoints_frac=jnp.array([[0.0, 0.0, 0.0]]),
        weights=jnp.array([1.0]),
        grid_shape=(1, 1, 1),
    )
    two_c = TwoCenterMatrices(
        overlap=jnp.array([1.0]),
        kinetic=jnp.array([0.5]),
        v_na=jnp.array([0.0]),
        v_kb=jnp.array([0.0]),
        v_kb_so=None,
        d_overlap_dr=jnp.zeros((1, 3)),
        d_kinetic_dr=jnp.zeros((1, 3)),
        d_v_na_dr=jnp.zeros((1, 3)),
        d_v_kb_dr=jnp.zeros((1, 3)),
        e_na_pair=jnp.array(0.0),
        f_na_atoms=jnp.zeros((2, 3)),
        stress_na=jnp.zeros((3, 3)),
    )
    scf_state = SCFState(
        iteration=0,
        h_sparse=jnp.zeros((1, 1)),
        dm_in=jnp.zeros((1, 1)),
        dm_out=jnp.zeros((1, 1)),
        edm_out=jnp.zeros((1, 1)),
        fermi_level_ha=jnp.array(0.0),
        entropy_ha=jnp.array(0.0),
        last_scf_free_e_ha=jnp.array(0.0),
        w_cur=jnp.array(0.1),
        w_base=jnp.array(0.1),
        active_temp_ha=jnp.array(1e-3),
        streak_bad=0,
        prev_d_max=jnp.array(1.0),
        prev_h_max=jnp.array(1.0),
        history_x=jnp.zeros((4, 1, 1)),
        history_f=jnp.zeros((4, 1, 1)),
        history_count=0,
    )
    obs = Observables(
        outcome="converged",
        converged=True,
        n_iterations=1,
        dm_out_recomputed=True,
        last_scf_free_e_ha=jnp.array(-1.0),
        last_scf_cycle=1,
        final_post_scf_free_e_ha=jnp.array(-1.0),
        total_energy_ha=jnp.array(-1.0),
        harris_energy_ha=jnp.array(-1.0),
        e_kinetic_ha=jnp.array(0.5),
        e_na_ha=jnp.array(0.0),
        e_kb_ha=jnp.array(0.0),
        e_hartree_ha=jnp.array(0.0),
        e_xc_ha=jnp.array(-0.5),
        e_disp_ha=jnp.array(0.0),
        e_entropy_ha=jnp.array(0.0),
        fermi_level_ha=jnp.array(-0.2),
        forces_ha_per_bohr=jnp.zeros((2, 3)),
        stress_ha_per_bohr3=jnp.zeros((3, 3)),
        total_spin_moment=jnp.array(0.0),
        atomic_spin_moments=jnp.zeros((2, 3)),
        charge_residual=jnp.array(0.0),
        d_max=jnp.array(0.0),
        h_max_ha=jnp.array(0.0),
    )
    for tree_obj in (struct, sp, basis, pattern, kmesh, two_c, scf_state, obs):
        leaves, treedef = jax.tree_util.tree_flatten(tree_obj)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jax.tree_util.tree_structure(rebuilt) == treedef

    cfg = SCFConfig(nspin=2)
    assert cfg.nspin == 2

    for proto in (
        BasisAndPseudos,
        TwoCenterIntegrals,
        SymmetryProjector,
        RealSpaceGridOps,
        ExchangeCorrelation,
        DispersionModel,
        EigenSolver,
        SCFMixer,
        ObservableEvaluator,
    ):
        assert getattr(proto, "_is_protocol", False)


def test_cpp_xla_ffi_ping_roundtrip() -> None:
    info = ffi_backend_info()
    assert info["ffi_version"] == 300
    x = jnp.array([0.5, -1.5, 3.0, 7.25], dtype=jnp.float64)
    y = jax.jit(ffi_ping)(x)
    assert jnp.allclose(y, 2.0 * x + 1.0)


def test_psml_fixtures_and_frozen_manifest() -> None:
    psml_dir = REPO_ROOT / "tests" / "fixtures" / "psml"
    for sym in ("H", "C", "O", "Si", "Co"):
        p = psml_dir / f"{sym}.psml"
        assert p.is_file()
        root = ET.parse(p).getroot()
        assert root.tag == "psml"
        assert root.get("version") == "1.1"
        assert root.find("pseudo-atom-spec").get("atomic-label") == sym
        assert root.find("grid/grid-data") is not None
        assert root.find("local-potential/radfunc/data") is not None
        assert root.find("nonlocal-projectors") is not None

    manifest_path = REPO_ROOT / "tests" / "manifests" / "phase0_matrix.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "frozen"
    assert len(manifest["cases"]) == 9
    for case in manifest["cases"]:
        case_dir = manifest_path.parent / case["input_dir"]
        cdata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        assert "cell_columns_bohr" in cdata
        assert "final_post_scf_free_e_ev" in case["tolerances"]
        assert "forces[0][0]" in case["tolerances"]
        assert "stress[0][0]" in case["tolerances"]


def test_knowledge_graph_build_and_resumption(tmp_path: Path) -> None:
    # Build in an isolated tmp_path so read-only sandboxes remain clean
    kg_out = tmp_path / "kg"
    db_path = build_kg(REPO_ROOT, out_dir=kg_out)
    assert db_path.is_file()
    kg = ProgressiveKG(kg_out / "current" / "docs_kg.db")

    ctx = kg.resume_session_context()
    assert ctx["undocumented_public_entities"] == [], f"Undocumented: {ctx['undocumented_public_entities']}"
    assert len(ctx["decisions"]) >= 6
    assert len(ctx["recent_work_logs"]) >= 1

    staleness = kg.check_staleness()
    assert staleness["up_to_date"], staleness

    lineage = kg.trace_physics_lineage("BasisAndPseudos")
    assert len(lineage["code_entities"]) >= 1
    assert len(lineage["theory_nodes"]) >= 1

    # Verify atomic preservation when build fails
    prev_target = (kg_out / "current").resolve()
    bad_root = tmp_path / "bad_repo"
    bad_root.mkdir()
    (bad_root / "jax_siesta").mkdir()
    (bad_root / "jax_siesta" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    try:
        build_kg(bad_root, out_dir=kg_out)
        raise AssertionError("Expected SyntaxError on invalid Python file")
    except SyntaxError:
        pass
    assert (kg_out / "current").resolve() == prev_target


_SYNTHETIC_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" tests="6" failures="1" skipped="1">
<properties><property name="git_commit" value="0123456789abcdef"/><property name="platform" value="synthetic"/></properties>
<testcase classname="tests.test_dispersion" name="test_d3_equals_reference_library" time="0.1"/>
<testcase classname="tests.test_dispersion" name="test_d3_matches_siesta" time="0.2">
<properties><property name="resid:energy_ev" value="1.5e-07"/><property name="tol:energy_ev" value="2e-06"/></properties></testcase>
<testcase classname="tests.test_mgga_soc" name="test_mgga_scf_forces_and_stress_vs_finite_differences" time="1.0">
<failure message="AssertionError: boom">trace</failure></testcase>
<testcase classname="tests.test_phase3_cuda" name="test_cuda_xc_parity[1-LDA]" time="0.0"><skipped type="pytest.skip" message="no GPU"/></testcase>
<testcase classname="tests.test_tools.IndexTests" name="test_missing_everything" time="0.0"/>
<testcase classname="tests.test_phase1_grid_xc" name="test_unrelated_names_do_not_matter" time="0.0"/>
</testsuite></testsuites>
"""


def test_knowledge_graph_ingests_junit_results(tmp_path: Path) -> None:
    """``verified_by`` edges carry pass/fail/skip/never-run status and residuals from JUnit XML (review M7)."""
    import sqlite3

    junit = tmp_path / "synthetic.junit.xml"
    junit.write_text(_SYNTHETIC_JUNIT, encoding="utf-8")
    db_path = build_kg(REPO_ROOT, out_dir=tmp_path / "kg", junit=[("synthetic", junit)])
    conn = sqlite3.connect(db_path)
    try:
        status = dict(conn.execute("SELECT id, status FROM validation_nodes WHERE kind='pytest_suite'"))
        assert status["val:pytest:test_mgga_soc"] == "fail"
        assert status["val:pytest:test_phase3_cuda"] == "skip"
        assert status["val:pytest:test_dispersion"] == "pass"
        assert status["val:pytest:test_basis_enthalpy"] == "never-run"  # absent from the JUnit file
        assert status["val:pytest:test_tools"] == "pass"  # unittest class suffix is stripped

        details = json.loads(
            conn.execute("SELECT details_json FROM validation_nodes WHERE id='val:pytest:test_dispersion'").fetchone()[0]
        )
        assert details["external_reference"] is True
        assert "test_d3_matches_siesta" in details["external_tests"]
        # Only the external-reference test counts for "verified"; it passed, the self-consistency ones are separate.
        assert details["external_status"] == "pass"
        assert details["residuals"]["test_d3_matches_siesta"]["energy_ev"] == {"resid": 1.5e-07, "tol": 2e-06}
        assert "test_rvv10_kernel_zero_k_is_analytic" in details["missing_tests"]

        edge_statuses = {
            r[0]: r[1]
            for r in conn.execute("SELECT dst, status FROM edges WHERE kind='verified_by' AND dst LIKE 'val:pytest:%'")
        }
        assert edge_statuses["val:pytest:test_mgga_soc"] == "fail"
        assert conn.execute("SELECT count(*) FROM test_results").fetchone()[0] == 6
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        assert json.loads(meta["junit_files"])[0]["git_commit"] == "0123456789abcdef"
    finally:
        conn.close()

    kg = ProgressiveKG(db_path)
    ctx = kg.resume_session_context()
    states = {m: v["state"] for m, v in ctx["verification"].items()}
    # A module whose only suites failed or never ran must not be reported as verified.
    assert "jax_siesta.xc.dispersion" in ctx["verified_modules"]  # external reference test passed
    assert states["jax_siesta.species.basis_opt"] == "never_run"  # only test_basis_enthalpy imports it
    assert "jax_siesta.species.basis_opt" not in ctx["verified_modules"]
    assert states["jax_siesta.xc.functional"] in ("self_consistent_only", "failing")  # no external pass
    assert "jax_siesta.xc.functional" not in ctx["verified_modules"]
    assert not any(states[m] == "failing" and m in ctx["verified_modules"] for m in states)
    lineage = kg.trace_physics_lineage("DFTD3Dispersion")
    dispersion = [v for v in lineage["validation_nodes"] if v["id"] == "val:pytest:test_dispersion"]
    assert dispersion and dispersion[0]["status"] == "pass" and dispersion[0]["residuals"]
    assert kg.check_staleness()["results_stale"]["stale"]  # synthetic commit != HEAD


def test_knowledge_graph_without_junit_reports_never_run(tmp_path: Path) -> None:
    """Without results nothing is called verified: every suite is ``never-run`` and results are stale."""
    db_path = build_kg(REPO_ROOT, out_dir=tmp_path / "kg", junit=[])
    kg = ProgressiveKG(db_path)
    ctx = kg.resume_session_context()
    assert ctx["verified_modules"] == []
    assert kg.check_staleness()["results_stale"] == {"stale": True, "reasons": ["no JUnit results ingested"]}


def test_doxygen_and_sphinx_strict_build(tmp_path: Path) -> None:
    doxy = shutil.which("doxygen")
    assert doxy is not None, "doxygen must be installed"
    assert (REPO_ROOT / "docs" / "_doxygen" / "xml" / "index.xml").is_file()

    venv_sphinx = Path(sys.executable).parent / "sphinx-build"
    sphinx = (
        str(venv_sphinx)
        if venv_sphinx.is_file()
        else (shutil.which("sphinx-build") or "/usr/local/Cellar/sphinx-doc/9.1.0_5/bin/sphinx-build")
    )
    assert Path(sphinx).exists(), "sphinx-build must be available"
    tmp_html = tmp_path / "html"
    tmp_doctrees = tmp_path / "doctrees"
    subprocess.run(
        [sphinx, "-W", "-d", str(tmp_doctrees), "-b", "html", "docs", str(tmp_html)],
        cwd=REPO_ROOT,
        check=True,
    )
    assert (tmp_html / "index.html").is_file()


@pytest.mark.external_refs("siesta_doc_index")
def test_knowledge_graph_links_legacy_siesta_routines(tmp_path: Path) -> None:
    # The SIESTA documentation index describes an external source tree, so it is an external reference.
    build_kg(REPO_ROOT, out_dir=tmp_path / "kg", legacy_db=ref_path("siesta_doc_index"))
    kg = ProgressiveKG(tmp_path / "kg" / "current" / "docs_kg.db")
    lineage = kg.trace_physics_lineage("BasisAndPseudos")
    legacy_names = {n["name"] for n in lineage["legacy_siesta_nodes"]}
    assert {"atom", "basis_specs"} <= legacy_names
    assert "diagon" not in legacy_names  # no sibling protocol leakage


def test_kg_mcp_stdio_session(tmp_path: Path) -> None:
    # Build a fresh graph in tmp_path: the test must not depend on whether someone remembered to
    # republish kg/current after editing handoff notes, and must not write into the source tree.
    db = build_kg(REPO_ROOT, out_dir=tmp_path / "kg")
    assert db.is_file()

    async def _run() -> None:
        server = REPO_ROOT / "kg" / "server_mcp.py"
        params = StdioServerParameters(command=sys.executable, args=[str(server), str(db)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {"resume_session_context", "trace_physics_lineage", "check_staleness", "search_kg"} <= names

                res = await session.call_tool("resume_session_context", {})
                assert not res.isError
                ctx = json.loads(res.content[0].text)
                assert ctx["undocumented_public_entities"] == []

                staleness = await session.call_tool("check_staleness", {})
                assert not staleness.isError
                assert json.loads(staleness.content[0].text)["up_to_date"]

                lineage = await session.call_tool("trace_physics_lineage", {"term": "Structure"})
                assert not lineage.isError
                assert json.loads(lineage.content[0].text)["code_entities"]

    asyncio.run(_run())
