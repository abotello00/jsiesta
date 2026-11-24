#!/usr/bin/env python3
"""Deterministic progressive Knowledge Graph builder for jsiesta.

Theory:
    Constructs a unified, revision-bound graph linking:
    - ``TheoryNode`` (``docs/theory/*.md``)
    - ``CodeEntity`` (Python ``ast`` over ``jax_siesta/`` + Doxygen XML over ``cpp/``)
    - ``LegacySIESTANode`` (resolved against a SIESTA documentation index, see :func:`legacy_index_path`)
    - ``ValidationNode`` (manifests in ``tests/manifests/``, reports in ``reports/``,
      and pytest verification modules in ``tests/``)
    - ``DecisionAndTaskNode`` (``MEMORY.md``, ``logs/DECISIONS.md``, ``logs/WORK_LOG.md``)

Algorithm:
    1. Parse Python modules via ``ast`` (no fragile regexes for code entities).
    2. Parse Doxygen XML compounds/members from ``docs/_doxygen/xml/``.
    3. Resolve ``SIESTA Mapping`` references against the SIESTA documentation index.
    4. Emit SQLite/FTS5 database + JSON export + Mermaid diagrams into staging dir.
    5. Verify integrity and atomically swap ``kg/current`` symlink.

Invariants:
    - Previous valid ``kg/current`` is preserved if any build step fails.
    - Every ``CodeEntity`` carries exact file path, SHA256 digest, and line span.

SIESTA Mapping:
    Extends ``index_docs.py`` and ``rebuild_index.py`` for the clean-room
    ``jax_siesta`` + ``cpp`` codebase.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge import digest, git_identity
from kg.junit_ingest import JunitRun, TestResult, aggregate_status, parse_junit, suite_summary


SCHEMA_VERSION = "2"
REQUIRED_SECTIONS = ("Theory", "Algorithm", "Invariants", "SIESTA Mapping")

THEORY_MAP: dict[str, tuple[str, ...]] = {
    "jax_siesta": ("theory.units_and_geometry",),
    "jax_siesta.core": ("theory.units_and_geometry", "theory.real_spherical_harmonics_and_spinors"),
    "jax_siesta.species": ("theory.psml_and_kb", "theory.neutral_atom_screening"),
    "jax_siesta.integrals": ("theory.two_center_matel",),
    "jax_siesta.symmetry": ("theory.native_symmetry",),
    "jax_siesta.grid": ("theory.grid_poisson_xc_dispersion",),
    "jax_siesta.xc": ("theory.grid_poisson_xc_dispersion",),
    "jax_siesta.scf": ("theory.adaptive_scf_and_dm_out",),
    "jax_siesta.observables": ("theory.variational_forces_and_stress",),
    "jax_siesta.io": ("theory.units_and_geometry",),
    "jax_siesta.testing": ("theory.variational_forces_and_stress",),
    "jax_siesta.app": ("theory.units_and_geometry", "theory.adaptive_scf_and_dm_out"),
    "cpp": ("theory.grid_poisson_xc_dispersion", "theory.two_center_matel"),
}


def _parse_docstring_sections(doc: str | None) -> dict[str, str]:
    """Extract structured multi-layer docstring sections."""
    if not doc:
        return {}
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    known = {"Theory", "Algorithm", "Parameters", "Returns", "Invariants", "SIESTA Mapping"}
    for line in doc.splitlines():
        stripped = line.strip()
        matched = None
        for sec in known:
            if stripped == f"{sec}:" or stripped.startswith(f"{sec}: "):
                matched = sec
                rest = stripped[len(sec) + 1 :].strip()
                break
        if matched is not None:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = matched
            buf = [rest] if rest else []
        elif current is not None:
            buf.append(stripped)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _extract_siesta_tokens(mapping_text: str) -> list[str]:
    """Extract referenced legacy SIESTA module/routine names from ``SIESTA Mapping``."""
    tokens = re.findall(r"`{1,2}([A-Za-z0-9_./]+)`{1,2}", mapping_text)
    cleaned: list[str] = []
    for tok in tokens:
        base = Path(tok).stem
        if base and base not in cleaned:
            cleaned.append(base)
    return cleaned


def extract_python_entities(repo_root: Path) -> tuple[list[dict], dict[str, str]]:
    """Deterministically extract ``CodeEntity`` nodes from ``jax_siesta/`` via ``ast``.

    Theory:
        Inspects the Python Abstract Syntax Tree so symbol spans, signatures,
        and multi-layer docstring contracts are exact without regex heuristics.

    Algorithm:
        Walks all ``.py`` files under ``jax_siesta/``, computing per-file SHA256
        and emitting module, class, protocol, and public function entities.

    Parameters:
        repo_root: Workspace root directory.

    Returns:
        Tuple of ``(entities_list, file_hashes_dict)``.

    Invariants:
        Every entity includes valid ``file``, ``sha256``, ``line_start``, ``line_end``.

    SIESTA Mapping:
        Replaces Doxygen Fortran parsing for the Python/JAX layer.
    """
    pkg_dir = repo_root / "jax_siesta"
    entities: list[dict] = []
    file_hashes: dict[str, str] = {}
    if not pkg_dir.is_dir():
        return entities, file_hashes

    for py_path in sorted(pkg_dir.rglob("*.py")):
        rel = py_path.relative_to(repo_root).as_posix()
        sha = digest(py_path)
        file_hashes[rel] = sha
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        lines = source.splitlines()
        mod_parts = list(py_path.relative_to(repo_root).with_suffix("").parts)
        if mod_parts[-1] == "__init__":
            mod_parts = mod_parts[:-1]
        mod_name = ".".join(mod_parts)
        mod_doc = ast.get_docstring(tree) or ""
        mod_sec = _parse_docstring_sections(mod_doc)
        entities.append({
            "id": f"py:{mod_name}",
            "kind": "module",
            "name": mod_parts[-1],
            "qualname": mod_name,
            "parent_id": f"py:{'.'.join(mod_parts[:-1])}" if len(mod_parts) > 1 else "",
            "language": "python",
            "file": rel,
            "sha256": sha,
            "line_start": 1,
            "line_end": max(1, len(lines)),
            "signature": f"module {mod_name}",
            "docstring": mod_doc,
            "sections_json": json.dumps(mod_sec, sort_keys=True),
            "is_public": True,
            "has_required_docs": all(k in mod_sec for k in REQUIRED_SECTIONS),
        })

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node) or ""
                sec = _parse_docstring_sections(doc)
                bases = [ast.unparse(b) for b in node.bases]
                kind = "protocol" if "Protocol" in bases else "class"
                cls_id = f"py:{mod_name}.{node.name}"
                entities.append({
                    "id": cls_id,
                    "kind": kind,
                    "name": node.name,
                    "qualname": f"{mod_name}.{node.name}",
                    "parent_id": f"py:{mod_name}",
                    "language": "python",
                    "file": rel,
                    "sha256": sha,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "signature": f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
                    "docstring": doc,
                    "sections_json": json.dumps(sec, sort_keys=True),
                    "is_public": True,
                    "has_required_docs": all(k in sec for k in REQUIRED_SECTIONS),
                })
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                        fdoc = ast.get_docstring(item) or ""
                        fsec = _parse_docstring_sections(fdoc)
                        args_str = ast.unparse(item.args)
                        ret_str = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                        is_prop = any(
                            isinstance(d, ast.Name) and d.id == "property"
                            for d in item.decorator_list
                        )
                        is_proto_stub = kind == "protocol"
                        has_docs = bool(fdoc) if (is_proto_stub or is_prop) else all(k in fsec for k in REQUIRED_SECTIONS)
                        entities.append({
                            "id": f"{cls_id}.{item.name}",
                            "kind": "property" if is_prop else "method",
                            "name": item.name,
                            "qualname": f"{mod_name}.{node.name}.{item.name}",
                            "parent_id": cls_id,
                            "language": "python",
                            "file": rel,
                            "sha256": sha,
                            "line_start": item.lineno,
                            "line_end": item.end_lineno or item.lineno,
                            "signature": f"def {item.name}({args_str}){ret_str}",
                            "docstring": fdoc,
                            "sections_json": json.dumps(fsec, sort_keys=True),
                            "is_public": True,
                            "has_required_docs": has_docs,
                        })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                # typing.overload stubs share the implementation's name; only
                # the implementation is an entity.
                if any(
                    (isinstance(d, ast.Name) and d.id == "overload")
                    or (isinstance(d, ast.Attribute) and d.attr == "overload")
                    for d in node.decorator_list
                ):
                    continue
                fdoc = ast.get_docstring(node) or ""
                fsec = _parse_docstring_sections(fdoc)
                args_str = ast.unparse(node.args)
                ret_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                entities.append({
                    "id": f"py:{mod_name}.{node.name}",
                    "kind": "function",
                    "name": node.name,
                    "qualname": f"{mod_name}.{node.name}",
                    "parent_id": f"py:{mod_name}",
                    "language": "python",
                    "file": rel,
                    "sha256": sha,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "signature": f"def {node.name}({args_str}){ret_str}",
                    "docstring": fdoc,
                    "sections_json": json.dumps(fsec, sort_keys=True),
                    "is_public": True,
                    "has_required_docs": all(k in fsec for k in REQUIRED_SECTIONS),
                })
    return entities, file_hashes


def _doxygen_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


def extract_cpp_entities(repo_root: Path) -> tuple[list[dict], dict[str, str]]:
    """Extract C++/CUDA ``CodeEntity`` nodes from Doxygen XML in ``docs/_doxygen/xml``.

    Theory:
        Captures compiled C++/CUDA structs, functions, and XLA FFI handlers with
        exact header/source line locations and Doxygen ``@par`` sections.

    Algorithm:
        Reads Doxygen XML compounds (``file``, ``struct``, ``class``, ``namespace``)
        and computes SHA256 digests over all ``cpp/`` source and header files.

    Parameters:
        repo_root: Workspace root directory.

    Returns:
        Tuple of ``(entities_list, file_hashes_dict)``.

    Invariants:
        File paths are normalized relative to ``repo_root``.

    SIESTA Mapping:
        Adapts ``index_docs.py`` XML ingestion for ``cpp/include``, ``cpp/src``, and ``cpp/cuda``.
    """
    entities: list[dict] = []
    file_hashes: dict[str, str] = {}
    cpp_dir = repo_root / "cpp"
    if cpp_dir.is_dir():
        for src in sorted(cpp_dir.rglob("*")):
            if src.is_file() and src.suffix in (".hpp", ".h", ".cpp", ".cuh", ".cu") and "build" not in src.parts:
                file_hashes[src.relative_to(repo_root).as_posix()] = digest(src)

    xml_dir = repo_root / "docs" / "_doxygen" / "xml"
    if not xml_dir.is_dir():
        return entities, file_hashes

    seen_ids: set[str] = set()
    for xml_file in sorted(xml_dir.glob("*.xml")):
        if xml_file.name in ("index.xml", "Doxyfile.xml"):
            continue
        root = ET.parse(xml_file).getroot()
        for comp in root.iter("compounddef"):
            kind = comp.get("kind", "")
            if kind in ("dir", "page", "group"):
                continue
            cname = comp.findtext("compoundname", "")
            if cname == "std" or cname.startswith("@"):
                continue
            loc = comp.find("location")
            raw_file = loc.get("file", "") if loc is not None else ""
            rel_file = ""
            if raw_file:
                p = Path(raw_file)
                resolved = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
                if resolved.is_relative_to(repo_root):
                    rel_file = resolved.relative_to(repo_root).as_posix()
            if rel_file and rel_file not in file_hashes and (repo_root / rel_file).is_file():
                file_hashes[rel_file] = digest(repo_root / rel_file)

            brief = _doxygen_text(comp.find("briefdescription"))
            detail = _doxygen_text(comp.find("detaileddescription"))
            doc_combined = f"{brief}\n{detail}".strip()
            comp_id = f"cpp:{comp.get('id', cname)}"
            if comp_id not in seen_ids and rel_file:
                seen_ids.add(comp_id)
                entities.append({
                    "id": comp_id,
                    "kind": f"cpp_{kind}",
                    "name": cname.split("::")[-1],
                    "qualname": cname,
                    "parent_id": "",
                    "language": "cuda" if rel_file.endswith((".cu", ".cuh")) else "cpp",
                    "file": rel_file,
                    "sha256": file_hashes.get(rel_file, ""),
                    "line_start": int(loc.get("line", 1) or 1) if loc is not None else 1,
                    "line_end": int((loc.get("bodyend", 0) or loc.get("line", 1) or 1) if loc is not None else 1),
                    "signature": f"{kind} {cname}",
                    "docstring": doc_combined,
                    "sections_json": json.dumps({"brief": brief, "detail": detail}, sort_keys=True),
                    "is_public": True,
                    "has_required_docs": True if kind == "namespace" else all(
                        s in doc_combined for s in ("Theory", "Algorithm", "Invariants", "SIESTA Mapping")
                    ),
                })

            for mem in comp.iter("memberdef"):
                mkind = mem.get("kind", "")
                if mkind in ("define", "typedef"):
                    continue
                mname = mem.findtext("name", "")
                if not mname or mname.startswith("_") or mname == "PingImpl" or mname == "PingHandler":
                    continue
                mloc = mem.find("location")
                mfile_raw = mloc.get("bodyfile") or mloc.get("file", "") if mloc is not None else ""
                mfile = rel_file
                if mfile_raw:
                    mp = Path(mfile_raw)
                    mres = mp.resolve() if mp.is_absolute() else (repo_root / mp).resolve()
                    if mres.is_relative_to(repo_root):
                        mfile = mres.relative_to(repo_root).as_posix()
                mbrief = _doxygen_text(mem.find("briefdescription"))
                mdetail = _doxygen_text(mem.find("detaileddescription"))
                mdoc = f"{mbrief}\n{mdetail}".strip()
                qual = mem.findtext("qualifiedname", "") or f"{cname}::{mname}"
                sig = (mem.findtext("definition", "") or qual) + (mem.findtext("argsstring", "") or "")
                mid = f"cpp:{mem.get('id', qual)}"
                dedup_key = f"cpp_member:{mkind}:{qual}"
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                # Prefer entries with richer documentation when declared in both .hpp and .cpp
                existing_idx = next(
                    (i for i, e in enumerate(entities) if e.get("_dedup_key") == dedup_key),
                    None,
                )
                entry = {
                    "_dedup_key": dedup_key,
                    "id": mid,
                    "kind": f"cpp_{mkind}",
                    "name": mname,
                    "qualname": qual,
                    "parent_id": comp_id,
                    "language": "cuda" if mfile.endswith((".cu", ".cuh")) else "cpp",
                    "file": mfile,
                    "sha256": file_hashes.get(mfile, ""),
                    "line_start": int(mloc.get("line", 1) or 1) if mloc is not None else 1,
                    "line_end": int((mloc.get("bodyend", 0) or mloc.get("line", 1) or 1) if mloc is not None else 1),
                    "signature": sig.strip(),
                    "docstring": mdoc,
                    "sections_json": json.dumps({"brief": mbrief, "detail": mdetail}, sort_keys=True),
                    "is_public": mem.get("prot", "public") == "public",
                    "has_required_docs": bool(mbrief) and (
                        mkind == "variable" or all(s in mdoc for s in ("Theory", "Algorithm", "Invariants", "SIESTA Mapping"))
                    ),
                }
                if existing_idx is not None:
                    if len(str(entry["docstring"])) > len(str(entities[existing_idx]["docstring"])):
                        entities[existing_idx] = entry
                else:
                    entities.append(entry)
    for ent in entities:
        ent.pop("_dedup_key", None)
    return entities, file_hashes


def extract_theory_nodes(repo_root: Path) -> tuple[list[dict], dict[str, str]]:
    """Extract ``TheoryNode`` definitions from ``docs/theory/*.md``.

    Theory:
        Indexes named mathematical derivations, sign/gauge conventions, and unit
        contracts as first-class graph nodes.

    Algorithm:
        Scans Markdown files in ``docs/theory/`` for ``## TheoryNode: <id>`` headers.

    Parameters:
        repo_root: Workspace root directory.

    Returns:
        Tuple of ``(theory_nodes, file_hashes)``.

    Invariants:
        Node IDs are globally unique across ``docs/theory/``.

    SIESTA Mapping:
        Provides traceable mathematical specifications for implemented modules.
    """
    theory_dir = repo_root / "docs" / "theory"
    nodes: list[dict] = []
    hashes: dict[str, str] = {}
    if not theory_dir.is_dir():
        return nodes, hashes

    for md in sorted(theory_dir.glob("*.md")):
        rel = md.relative_to(repo_root).as_posix()
        sha = digest(md)
        hashes[rel] = sha
        lines = md.read_text(encoding="utf-8").splitlines()
        doc_title = lines[0].lstrip("# ").strip() if lines else md.stem
        current_id: str | None = None
        start_line = 1
        buf: list[str] = []
        for idx, line in enumerate(lines, start=1):
            m = re.match(r"^##\s+TheoryNode:\s*(\S+)\s*$", line)
            if m:
                if current_id is not None:
                    nodes.append({
                        "id": current_id,
                        "title": doc_title,
                        "file": rel,
                        "sha256": sha,
                        "line_start": start_line,
                        "line_end": idx - 1,
                        "summary": "\n".join(buf).strip(),
                    })
                current_id = m.group(1)
                start_line = idx
                buf = []
            elif current_id is not None:
                buf.append(line)
        if current_id is not None:
            nodes.append({
                "id": current_id,
                "title": doc_title,
                "file": rel,
                "sha256": sha,
                "line_start": start_line,
                "line_end": len(lines),
                "summary": "\n".join(buf).strip(),
            })
    return nodes, hashes


def extract_decisions_and_tasks(repo_root: Path) -> tuple[list[dict], dict[str, str]]:
    """Extract ``DecisionAndTaskNode`` records from ``MEMORY.md`` and ``logs/``.

    Theory:
        Preserves architectural rationale and resumable milestone status inside
        the same queryable graph as code and validation evidence.

    Algorithm:
        Parses structured headings from ``logs/DECISIONS.md``, ``logs/WORK_LOG.md``,
        and ``MEMORY.md``.

    Parameters:
        repo_root: Workspace root directory.

    Returns:
        Tuple of ``(nodes, file_hashes)``.

    Invariants:
        Record IDs uniquely identify each decision, log entry, or memory snapshot.

    SIESTA Mapping:
        Extends session-continuity tracking modeled on ``scf_params/Plan.md`` and ``MEMORY.md``.
    """
    nodes: list[dict] = []
    hashes: dict[str, str] = {}

    memory_path = repo_root / "MEMORY.md"
    if memory_path.is_file():
        rel = "MEMORY.md"
        sha = digest(memory_path)
        hashes[rel] = sha
        text = memory_path.read_text(encoding="utf-8")
        nodes.append({
            "id": "task:memory_state",
            "kind": "memory",
            "title": "Live Project Memory",
            "status": "active",
            "file": rel,
            "sha256": sha,
            "content": text,
        })

    for fname, kind, prefix in [
        ("logs/DECISIONS.md", "decision", "decision"),
        ("logs/WORK_LOG.md", "work_log", "worklog"),
    ]:
        fpath = repo_root / fname
        if not fpath.is_file():
            continue
        sha = digest(fpath)
        hashes[fname] = sha
        lines = fpath.read_text(encoding="utf-8").splitlines()
        cur_title: str | None = None
        buf: list[str] = []
        idx_num = 0
        for line in lines:
            if line.startswith("## "):
                if cur_title is not None:
                    idx_num += 1
                    slug = re.sub(r"[^a-z0-9]+", "_", cur_title.lower()).strip("_")[:48]
                    nodes.append({
                        "id": f"{prefix}:{idx_num:03d}_{slug}",
                        "kind": kind,
                        "title": cur_title,
                        "status": "recorded",
                        "file": fname,
                        "sha256": sha,
                        "content": "\n".join(buf).strip(),
                    })
                cur_title = line[3:].strip()
                buf = []
            elif cur_title is not None:
                buf.append(line)
        if cur_title is not None:
            idx_num += 1
            slug = re.sub(r"[^a-z0-9]+", "_", cur_title.lower()).strip("_")[:48]
            nodes.append({
                "id": f"{prefix}:{idx_num:03d}_{slug}",
                "kind": kind,
                "title": cur_title,
                "status": "recorded",
                "file": fname,
                "sha256": sha,
                "content": "\n".join(buf).strip(),
            })
    return nodes, hashes


def _declared_tests(tree: ast.AST) -> tuple[list[str], list[str]]:
    """Return ``(all_test_names, external_reference_test_names)`` declared in a test module AST."""
    names: list[str] = []
    external: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "external_refs":
                    external.append(node.name)
                    break
    return sorted(set(names)), sorted(set(external))


def extract_validation_nodes(
    repo_root: Path,
    junit_runs: list[JunitRun] | None = None,
) -> tuple[list[dict], list[tuple[str, str, str]], dict[str, str], list[TestResult]]:
    """Extract ``ValidationNode`` records and ``verified_by`` edges from tests and manifests.

    Theory:
        Connects physical modules and functions directly to frozen ``validation.py``
        manifests, pytest test modules, and saved verification reports.

    Algorithm:
        Scans ``tests/test_*.py`` AST for imports of ``jax_siesta`` modules and for
        ``external_refs`` markers, attaches JUnit statuses, and inspects ``tests/manifests/*.json`` and ``reports/*/report.json``.

    Parameters:
        repo_root: Workspace root directory.
        junit_runs: Parsed pytest JUnit files. A test suite takes its status from them; a suite that
            appears in none of them is ``never-run``. Without any JUnit file every suite is ``never-run``.

    Returns:
        Tuple of ``(validation_nodes, edges, file_hashes, test_results)``.

    Invariants:
        Manifests and test files are hashed with SHA256 for staleness checks.

    SIESTA Mapping:
        Integrates ``validation.py`` numerical contracts with the code graph.
    """
    nodes: list[dict] = []
    edges: list[tuple[str, str, str]] = []
    hashes: dict[str, str] = {}
    all_results = [r for run in (junit_runs or []) for r in run.results]
    summaries = suite_summary(all_results)

    tests_dir = repo_root / "tests"
    if tests_dir.is_dir():
        for test_file in sorted(tests_dir.glob("test_*.py")):
            rel = test_file.relative_to(repo_root).as_posix()
            sha = digest(test_file)
            hashes[rel] = sha
            src = test_file.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=rel)
            targets: list[str] = []
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("jax_siesta"):
                    targets.append(f"py:{node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("jax_siesta"):
                            targets.append(f"py:{alias.name}")
            vid = f"val:pytest:{test_file.stem}"
            declared, external = _declared_tests(tree)
            n_tests = len(declared)
            suite_results = [r for r in all_results if r.suite == test_file.stem]
            summary = summaries.get(test_file.stem)
            status = summary["status"] if summary else "never-run"
            ran = {r.name.split("[")[0] for r in suite_results}
            ext_status = aggregate_status(
                [r.status for r in suite_results if r.name.split("[")[0] in external]
            ) if external else "none"
            residuals = {
                f"{r.name}": r.residuals() for r in suite_results if r.residuals()
            }
            nodes.append({
                "id": vid,
                "kind": "pytest_suite",
                "name": test_file.name,
                "status": status,
                "file": rel,
                "sha256": sha,
                "summary": (
                    f"{n_tests} test functions covering {', '.join(sorted(set(targets))) or 'infrastructure'}"
                    f"; status {status}"
                    + (f"; external reference {ext_status}" if external else "; self-consistency only")
                ),
                "details_json": json.dumps(
                    {
                        "n_tests": n_tests,
                        "targets": sorted(set(targets)),
                        "external_reference": bool(external),
                        "external_tests": external,
                        "external_status": ext_status,
                        "counts": summary["counts"] if summary else {},
                        "by_platform": summary["by_platform"] if summary else {},
                        "missing_tests": sorted(set(declared) - ran) if suite_results else declared,
                        "residuals": residuals,
                    },
                    sort_keys=True,
                ),
            })
            for tgt in sorted(set(targets)):
                edges.append((tgt, vid, "verified_by"))

    manifests_dir = tests_dir / "manifests"
    if manifests_dir.is_dir():
        for mfile in sorted(manifests_dir.glob("*.json")):
            rel = mfile.relative_to(repo_root).as_posix()
            sha = digest(mfile)
            hashes[rel] = sha
            data = json.loads(mfile.read_text(encoding="utf-8"))
            vid = f"val:manifest:{mfile.stem}"
            cases = [c.get("name", "") for c in data.get("cases", [])]
            nodes.append({
                "id": vid,
                "kind": "validation_manifest",
                "name": mfile.name,
                "status": data.get("status", "frozen"),
                "file": rel,
                "sha256": sha,
                "summary": f"Frozen validation manifest with {len(cases)} cases: {', '.join(cases[:6])}",
                "details_json": json.dumps(
                    {"cases": cases, "schema_version": data.get("schema_version"), "external_reference": False},
                    sort_keys=True,
                ),
            })
            for mod_target in data.get("covered_modules", []):
                edges.append((f"py:{mod_target}", vid, "verified_by"))

    reports_dir = repo_root / "reports"
    if reports_dir.is_dir():
        for rfile in sorted(reports_dir.glob("*/report.json")):
            rel = rfile.relative_to(repo_root).as_posix()
            sha = digest(rfile)
            hashes[rel] = sha
            data = json.loads(rfile.read_text(encoding="utf-8"))
            vid = f"val:report:{rfile.parent.name}"
            passed = bool(data.get("passed", False))
            nodes.append({
                "id": vid,
                "kind": "validation_report",
                "name": rfile.parent.name,
                "status": "passed" if passed else "failed",
                "file": rel,
                "sha256": sha,
                "summary": f"validation.py report ({len(data.get('cases', []))} cases, passed={passed})",
                "details_json": json.dumps(
                    {"passed": passed, "created_at": data.get("created_at"), "external_reference": False},
                    sort_keys=True,
                ),
            })
        for pfile in sorted(reports_dir.glob("*.json")):
            rel = pfile.relative_to(repo_root).as_posix()
            sha = digest(pfile)
            hashes[rel] = sha
            data = json.loads(pfile.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            if data.get("kind") == "profiling":
                vid = f"val:profiling:{pfile.stem}"
                workloads = data.get("workloads", [])
                all_conv = all(w.get("converged", False) for w in workloads)
                nodes.append({
                    "id": vid,
                    "kind": "profiling_report",
                    "name": pfile.name,
                    "status": "passed" if all_conv else "partial",
                    "file": rel,
                    "sha256": sha,
                    "summary": (
                        f"docs/design/EVIDENCE.md profiling report ({len(workloads)} workloads, "
                        f"{len(data.get('mixer_benchmark', []))} mixer comparisons, "
                        f"backend={data.get('configuration', {}).get('device_backend', 'unknown')})"
                    ),
                    "details_json": json.dumps(
                        {
                            "kind": "profiling",
                            "configuration": data.get("configuration", {}),
                            "external_reference": False,
                        },
                        sort_keys=True,
                    ),
                })
                for mod in (
                    "py:jax_siesta.testing.profile_phase3",
                    "py:jax_siesta.scf.mixer",
                    "py:jax_siesta.scf.eigensolver",
                    "py:jax_siesta.integrals.two_center",
                    "py:jax_siesta.observables.forces",
                    "py:jax_siesta.grid.mesh",
                ):
                    edges.append((mod, vid, "verified_by"))
    return nodes, edges, hashes, all_results


#: Environment variable naming the SIESTA documentation index (``docs.db`` built by ``rebuild_index.py``). The same
#: variable configures the ``siesta_doc_index`` external reference of the tests.
LEGACY_INDEX_ENV = "JSIESTA_SIESTA_DOC_INDEX"


def legacy_index_path(repo_root: Path, legacy_db: Path | None = None) -> Path:
    """Return the SIESTA documentation index that legacy provenance is resolved against.

    Theory:
        The index describes an external SIESTA source tree, so its location is configuration, not part of the
        repository.

    Algorithm:
        The explicit argument, else ``$JSIESTA_SIESTA_DOC_INDEX``, else ``indexes/default/current/docs.db`` under
        ``repo_root`` (where ``rebuild_index.py`` publishes it by default).

    Parameters:
        repo_root: Workspace root directory.
        legacy_db: Explicit index path, or ``None``.

    Returns:
        The index path; it may not exist.

    Invariants:
        Pure function of its arguments and the environment.

    SIESTA Mapping:
        None (configuration of the provenance index).
    """
    if legacy_db is not None:
        return Path(legacy_db).expanduser()
    env = os.environ.get(LEGACY_INDEX_ENV)
    if env:
        return Path(env).expanduser()
    return repo_root / "indexes" / "default" / "current" / "docs.db"


def resolve_legacy_siesta_nodes(
    repo_root: Path,
    code_entities: list[dict],
    legacy_db: Path | None = None,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Link ``CodeEntity`` nodes to legacy SIESTA entities in a SIESTA documentation index.

    Theory:
        Maintains explicit provenance edges (``implements_equivalent_of`` /
        ``replaces_routine``) between new JAX/C++ entities and historical Fortran SIESTA routines.

    Algorithm:
        Reads ``SIESTA Mapping`` tokens from docstrings and queries the read-only
        ``docs.db`` index for matching Fortran modules/subroutines.

    Parameters:
        repo_root: Workspace root directory.
        code_entities: Extracted Python and C++ entity dictionaries.
        legacy_db: SIESTA documentation index; ``None`` resolves it with :func:`legacy_index_path`.

    Returns:
        Tuple of ``(legacy_nodes, edges)``.

    Invariants:
        Read-only access to the index; returns empty lists if it is absent.

    SIESTA Mapping:
        Queries the ``docs.db`` index of a SIESTA source tree built by ``rebuild_index.py`` / ``index_docs.py``.
    """
    index_db = legacy_index_path(repo_root, legacy_db)
    if not index_db.is_file():
        return [], []

    conn = sqlite3.connect(index_db.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    legacy_nodes: dict[str, dict] = {}
    edges: set[tuple[str, str, str]] = set()
    try:
        for ent in code_entities:
            sec = json.loads(ent["sections_json"])
            mapping_text = sec.get("SIESTA Mapping", "") or sec.get("detail", "")
            if not mapping_text:
                continue
            for tok in _extract_siesta_tokens(mapping_text):
                rows = conn.execute(
                    "SELECT id, kind, name, qualname, file FROM entities "
                    "WHERE lower(name)=lower(?) OR lower(qualname)=lower(?) LIMIT 3",
                    (tok, tok),
                ).fetchall()
                for r in rows:
                    lid = f"siesta:{r['id']}"
                    legacy_nodes[lid] = {
                        "id": lid,
                        "legacy_id": r["id"],
                        "kind": r["kind"],
                        "name": r["name"],
                        "qualname": r["qualname"],
                        "file": r["file"],
                    }
                    rel_kind = "replaces_routine" if r["kind"] == "function" else "implements_equivalent_of"
                    edges.add((ent["id"], lid, rel_kind))
    finally:
        conn.close()
    return sorted(legacy_nodes.values(), key=lambda x: x["id"]), sorted(edges)


def emit_mermaid_diagrams(
    arch_dir: Path,
    code_entities: list[dict],
    theory_nodes: list[dict],
    validation_nodes: list[dict],
    legacy_nodes: list[dict],
    edges: list[tuple[str, str, str]],
) -> None:
    """Emit Mermaid architecture, validation-coverage, and lineage diagrams into ``arch_dir``.

    Theory:
        Provides human-readable visual representations of module dependencies,
        validation coverage, and legacy-to-modern provenance.

    Algorithm:
        Generates deterministic Mermaid ``flowchart LR`` files from graph nodes and edges.

    Parameters:
        arch_dir: Output directory. ``docs/architecture`` for a published build; a directory inside
            ``out_dir`` for an isolated build, so tests never write into the source tree.
        code_entities: Extracted code entities.
        theory_nodes: Extracted theory nodes.
        validation_nodes: Extracted validation nodes.
        legacy_nodes: Resolved legacy SIESTA nodes.
        edges: Directed graph edges ``(src, dst, kind)``.

    Invariants:
        Node labels with special characters are quoted per Mermaid requirements.

    SIESTA Mapping:
        Replaces static hand-drawn architecture figures with graph-derived diagrams.
    """
    arch_dir.mkdir(parents=True, exist_ok=True)

    modules = [e for e in code_entities if e["kind"] == "module"]
    dep_lines = ["flowchart LR", '    core["jax_siesta.core"]']
    for mod in modules:
        q = mod["qualname"]
        if q in ("jax_siesta", "jax_siesta.core"):
            continue
        short = q.replace("jax_siesta.", "")
        dep_lines.append(f'    {short}["{q}"]')
        dep_lines.append(f"    core --> {short}")
    dep_lines.append('    cpp_ffi["cpp/libjsiesta_ffi"]')
    dep_lines.append("    core --> cpp_ffi")
    dep_lines.append("    grid --> cpp_ffi")
    dep_lines.append("    xc --> cpp_ffi")
    dep_lines.append("    integrals --> cpp_ffi")
    (arch_dir / "dependency_graph.mmd").write_text("\n".join(dep_lines) + "\n", encoding="utf-8")

    val_lines = ["flowchart LR"]
    val_map = {v["id"]: v for v in validation_nodes}
    for src, dst, kind in edges:
        if kind == "verified_by" and dst in val_map:
            s_id = re.sub(r"[^A-Za-z0-9_]", "_", src)
            d_id = re.sub(r"[^A-Za-z0-9_]", "_", dst)
            v = val_map[dst]
            val_lines.append(f'    {s_id}["{src.removeprefix("py:")}"] --> {d_id}["{v["name"]} ({v["status"]})"]')
    if len(val_lines) == 1:
        val_lines.append('    placeholder["No validation edges yet"]')
    (arch_dir / "validation_coverage.mmd").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    lin_lines = ["flowchart LR"]
    t_map = {t["id"]: t for t in theory_nodes}
    l_map = {leg["id"]: leg for leg in legacy_nodes}
    shown = 0
    for src, dst, kind in edges:
        if kind == "implements_theory" and dst in t_map:
            s_id = re.sub(r"[^A-Za-z0-9_]", "_", src)
            d_id = re.sub(r"[^A-Za-z0-9_]", "_", dst)
            lin_lines.append(f'    {d_id}["{dst}"] --> {s_id}["{src.removeprefix("py:")}"]')
            shown += 1
        elif kind in ("replaces_routine", "implements_equivalent_of") and dst in l_map and shown < 30:
            s_id = re.sub(r"[^A-Za-z0-9_]", "_", src)
            d_id = re.sub(r"[^A-Za-z0-9_]", "_", dst)
            lin_lines.append(f'    {s_id}["{src.removeprefix("py:")}"] -. "{kind}" .-> {d_id}["SIESTA:{l_map[dst]["name"]}"]')
            shown += 1
    if len(lin_lines) == 1:
        lin_lines.append('    placeholder["No lineage edges yet"]')
    (arch_dir / "lineage_graph.mmd").write_text("\n".join(lin_lines) + "\n", encoding="utf-8")


def build_kg(
    repo_root: Path,
    out_dir: Path | None = None,
    junit: list[tuple[str, Path]] | None = None,
    legacy_db: Path | None = None,
) -> Path:
    """Build and atomically publish the progressive Knowledge Graph snapshot.

    Theory:
        Consolidates code AST, Doxygen XML, mathematical theory, validation
        artifacts, legacy SIESTA provenance, and session decisions into a
        single revision-bound SQLite/FTS5 + JSON database.

    Algorithm:
        Stages into a temporary directory under ``kg/generations/``, validates
        SQLite integrity and foreign keys, and atomically swaps ``kg/current``.

    Parameters:
        repo_root: Root directory of the ``jsiesta`` repository.
        out_dir: Optional output root (defaults to ``repo_root / "kg"``). When given, the Mermaid
            diagrams go to ``out_dir / "architecture"`` instead of ``docs/architecture`` so an
            isolated build (e.g. in a test) leaves the source tree untouched.
        junit: ``(platform, path)`` pairs of pytest JUnit XML files whose results become the status of
            ``verified_by`` edges. Defaults to ``reports/pytest.junit.xml`` when that file exists.
        legacy_db: SIESTA documentation index for the legacy provenance nodes; ``None`` resolves it with
            :func:`legacy_index_path`. Without an index the graph has no ``LegacySIESTANode`` entries.

    Returns:
        Path to the newly published ``docs_kg.db``.

    Invariants:
        Preserves the prior ``kg/current`` target if any failure occurs.

    SIESTA Mapping:
        Mirrors the atomic generation publication pattern of ``index_docs.py``.
    """
    root = Path(repo_root).resolve()
    out = (out_dir or (root / "kg")).resolve()
    generations = out / "generations"
    generations.mkdir(parents=True, exist_ok=True)

    py_entities, py_hashes = extract_python_entities(root)
    cpp_entities, cpp_hashes = extract_cpp_entities(root)
    code_entities = py_entities + cpp_entities
    theory_nodes, theory_hashes = extract_theory_nodes(root)
    decision_nodes, dec_hashes = extract_decisions_and_tasks(root)
    if junit is None:
        default_junit = root / "reports" / "pytest.junit.xml"
        junit = [("local", default_junit)] if default_junit.is_file() else []
    junit_runs = [parse_junit(Path(path), platform) for platform, path in junit]
    val_nodes, val_edges, val_hashes, test_results = extract_validation_nodes(root, junit_runs)
    legacy_nodes, legacy_edges = resolve_legacy_siesta_nodes(root, code_entities, legacy_db)

    all_hashes = {**py_hashes, **cpp_hashes, **theory_hashes, **dec_hashes, **val_hashes}

    edges: set[tuple[str, str, str]] = set()
    code_ids = {e["id"] for e in code_entities}
    theory_ids = {t["id"] for t in theory_nodes}
    val_ids = {v["id"] for v in val_nodes}
    ent_file = {e["id"]: e["file"] for e in code_entities}

    for ent in code_entities:
        if ent["parent_id"] and ent["parent_id"] in code_ids:
            edges.add((ent["parent_id"], ent["id"], "contains"))
        for prefix, tids in THEORY_MAP.items():
            if ent["qualname"] == prefix or ent["qualname"].startswith(prefix + ".") or (
                prefix == "cpp" and ent["id"].startswith("cpp:")
            ):
                for tid in tids:
                    if tid in theory_ids:
                        edges.add((ent["id"], tid, "implements_theory"))

    for src, dst, kind in val_edges:
        if src in code_ids and dst in val_ids:
            edges.add((src, dst, kind))
            if kind == "verified_by":
                # Spread a module's verification to the classes/functions defined in the *same file* only.
                # A test importing a package (``from jax_siesta.species import X``) says nothing about the
                # package's other submodules, so those do not inherit the edge.
                for cid in code_ids:
                    if cid.startswith(src + ".") and ent_file.get(cid) == ent_file.get(src):
                        edges.add((cid, dst, kind))
                # Mark parent subpackage node verified when its submodule is verified
                parts = src.split(".")
                if len(parts) >= 3:
                    parent_pkg = ".".join(parts[:-1])
                    if parent_pkg in code_ids:
                        edges.add((parent_pkg, dst, kind))
    for src, dst, kind in legacy_edges:
        if src in code_ids:
            edges.add((src, dst, kind))

    sorted_edges = sorted(edges)
    val_by_id = {v["id"]: v for v in val_nodes}
    edge_rows: list[tuple[str, str, str, str, str]] = []
    for src, dst, kind in sorted_edges:
        if kind == "verified_by" and dst in val_by_id:
            node = val_by_id[dst]
            external = bool(json.loads(node["details_json"]).get("external_reference", False))
            edge_rows.append((src, dst, kind, node["status"], json.dumps({"external_reference": external})))
        else:
            edge_rows.append((src, dst, kind, "", ""))
    arch_dir = root / "docs" / "architecture" if out_dir is None else out / "architecture"
    emit_mermaid_diagrams(arch_dir, code_entities, theory_nodes, val_nodes, legacy_nodes, sorted_edges)

    git_id = git_identity(root)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(root),
        "builder_sha256": digest(__file__),
        "git_commit": git_id["commit"] or "not-a-git-repository",
        "git_dirty": git_id["dirty_status"] or "",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_code_entities": str(len(code_entities)),
        "n_theory_nodes": str(len(theory_nodes)),
        "n_legacy_nodes": str(len(legacy_nodes)),
        "n_validation_nodes": str(len(val_nodes)),
        "n_decision_nodes": str(len(decision_nodes)),
        "n_edges": str(len(sorted_edges)),
        "junit_files": json.dumps(
            [
                {"path": r.path, "platform": r.platform, "git_commit": r.git_commit, "mtime": r.mtime,
                 "n_results": len(r.results)}
                for r in junit_runs
            ],
            sort_keys=True,
        ),
        "snapshot_sha256": hashlib.sha256(
            "".join(f"{k}:{v}\n" for k, v in sorted(all_hashes.items())).encode()
        ).hexdigest(),
    }

    with tempfile.TemporaryDirectory(prefix=".building-", dir=generations) as temporary:
        tmp_path = Path(temporary)
        db_path = tmp_path / "docs_kg.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE code_entities(
                    id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualname TEXT,
                    parent_id TEXT, language TEXT, file TEXT, sha256 TEXT,
                    line_start INTEGER, line_end INTEGER, signature TEXT,
                    docstring TEXT, sections_json TEXT, is_public INTEGER,
                    has_required_docs INTEGER
                );
                CREATE TABLE theory_nodes(
                    id TEXT PRIMARY KEY, title TEXT, file TEXT, sha256 TEXT,
                    line_start INTEGER, line_end INTEGER, summary TEXT
                );
                CREATE TABLE legacy_siesta_nodes(
                    id TEXT PRIMARY KEY, legacy_id TEXT, kind TEXT, name TEXT,
                    qualname TEXT, file TEXT
                );
                CREATE TABLE validation_nodes(
                    id TEXT PRIMARY KEY, kind TEXT, name TEXT, status TEXT,
                    file TEXT, sha256 TEXT, summary TEXT, details_json TEXT
                );
                CREATE TABLE decision_task_nodes(
                    id TEXT PRIMARY KEY, kind TEXT, title TEXT, status TEXT,
                    file TEXT, sha256 TEXT, content TEXT
                );
                CREATE TABLE edges(
                    src TEXT, dst TEXT, kind TEXT,
                    status TEXT NOT NULL DEFAULT '', details_json TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(src, dst, kind)
                );
                CREATE TABLE test_results(
                    validation_id TEXT, test_name TEXT, status TEXT, time_s REAL, message TEXT,
                    platform TEXT, properties_json TEXT
                );
                CREATE TABLE tracked_files(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
                CREATE VIRTUAL TABLE fts USING fts5(
                    node_id UNINDEXED, node_type UNINDEXED, title, body
                );
                """
            )
            conn.executemany(
                "INSERT INTO code_entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        e["id"], e["kind"], e["name"], e["qualname"], e["parent_id"],
                        e["language"], e["file"], e["sha256"], e["line_start"],
                        e["line_end"], e["signature"], e["docstring"], e["sections_json"],
                        int(e["is_public"]), int(e["has_required_docs"]),
                    )
                    for e in code_entities
                ],
            )
            conn.executemany(
                "INSERT INTO theory_nodes VALUES (?,?,?,?,?,?,?)",
                [
                    (t["id"], t["title"], t["file"], t["sha256"], t["line_start"], t["line_end"], t["summary"])
                    for t in theory_nodes
                ],
            )
            conn.executemany(
                "INSERT INTO legacy_siesta_nodes VALUES (?,?,?,?,?,?)",
                [(leg["id"], leg["legacy_id"], leg["kind"], leg["name"], leg["qualname"], leg["file"]) for leg in legacy_nodes],
            )
            conn.executemany(
                "INSERT INTO validation_nodes VALUES (?,?,?,?,?,?,?,?)",
                [
                    (v["id"], v["kind"], v["name"], v["status"], v["file"], v["sha256"], v["summary"], v["details_json"])
                    for v in val_nodes
                ],
            )
            conn.executemany(
                "INSERT INTO decision_task_nodes VALUES (?,?,?,?,?,?,?)",
                [
                    (d["id"], d["kind"], d["title"], d["status"], d["file"], d["sha256"], d["content"])
                    for d in decision_nodes
                ],
            )
            conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", edge_rows)
            conn.executemany(
                "INSERT INTO test_results VALUES (?,?,?,?,?,?,?)",
                [
                    (f"val:pytest:{r.suite}", r.name, r.status, r.time_s, r.message, r.platform,
                     json.dumps(r.properties, sort_keys=True))
                    for r in test_results
                ],
            )
            conn.executemany("INSERT INTO tracked_files VALUES (?,?)", sorted(all_hashes.items()))
            conn.executemany("INSERT INTO meta VALUES (?,?)", sorted(metadata.items()))

            fts_rows = [
                (e["id"], "code", e["qualname"], f"{e['signature']}\n{e['docstring']}")
                for e in code_entities
            ] + [
                (t["id"], "theory", t["title"], t["summary"])
                for t in theory_nodes
            ] + [
                (leg["id"], "legacy_siesta", leg["qualname"], leg["file"])
                for leg in legacy_nodes
            ] + [
                (v["id"], "validation", v["name"], v["summary"])
                for v in val_nodes
            ] + [
                (d["id"], "decision_task", d["title"], d["content"])
                for d in decision_nodes
            ]
            conn.executemany("INSERT INTO fts VALUES (?,?,?,?)", fts_rows)
            conn.commit()
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("KG SQLite integrity check failed.")
        finally:
            conn.close()

        graph_export = {
            "metadata": metadata,
            "code_entities": code_entities,
            "theory_nodes": theory_nodes,
            "legacy_siesta_nodes": legacy_nodes,
            "validation_nodes": val_nodes,
            "decision_task_nodes": decision_nodes,
            "edges": [
                {"src": s, "dst": d, "kind": k, "status": st}
                for s, d, k, st, _details in edge_rows
            ],
            "test_results": [
                {"validation_id": f"val:pytest:{r.suite}", "test": r.name, "status": r.status,
                 "platform": r.platform, "message": r.message}
                for r in test_results
            ],
        }
        (tmp_path / "graph.json").write_text(json.dumps(graph_export, indent=2) + "\n", encoding="utf-8")

        gen_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + tmp_path.name.removeprefix(".building-")
        generation = generations / gen_name
        os.replace(tmp_path, generation)
        link = out / (".current-" + gen_name)
        try:
            link.symlink_to(Path("generations") / gen_name, target_is_directory=True)
            os.replace(link, out / "current")
        finally:
            link.unlink(missing_ok=True)
    return generation / "docs_kg.db"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--junit",
        action="append",
        default=None,
        metavar="[PLATFORM=]PATH",
        help="pytest JUnit XML to ingest (repeatable; default reports/pytest.junit.xml if present)",
    )
    parser.add_argument(
        "--legacy-db",
        type=Path,
        default=None,
        help=f"SIESTA documentation index (default ${LEGACY_INDEX_ENV}, else indexes/default/current/docs.db)",
    )
    args = parser.parse_args()
    junit_arg: list[tuple[str, Path]] | None = None
    if args.junit is not None:
        junit_arg = []
        for item in args.junit:
            platform_name, _, path_text = item.rpartition("=")
            junit_arg.append((platform_name or "local", Path(path_text)))
    print(build_kg(args.repo_root, args.out_dir, junit_arg, args.legacy_db))
