#!/usr/bin/env python3
"""Read-only MCP server & Python API for the progressive jsiesta Knowledge Graph.

Theory:
    Provides session resumption, cross-layer physics lineage traversal, and
    SHA256 staleness verification over ``kg/current/docs_kg.db``.

Algorithm:
    Opens ``docs_kg.db`` in read-only SQLite URI mode per request so newly
    published generations under ``kg/current`` are visible immediately.

Invariants:
    - Read-only; never mutates source files or database state.
    - Reports exact SHA256 staleness against live workspace files.

SIESTA Mapping:
    Extends ``siesta_mcp.py`` and ``knowledge.py`` with ``resume_session_context``,
    ``trace_physics_lineage``, and ``check_staleness``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP

from knowledge import digest, git_identity
from kg.build_kg import SCHEMA_VERSION

BASE = Path(__file__).resolve().parent.parent
KG_DB_PATH = BASE / "kg" / "current" / "docs_kg.db"

mcp = FastMCP("jsiesta-kg")


class ProgressiveKG:
    """Read-only interface to ``kg/current/docs_kg.db``."""

    def __init__(self, db_path: Path | str = KG_DB_PATH) -> None:
        self.db_path = Path(db_path).resolve()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not row or row[0] != SCHEMA_VERSION:
            conn.close()
            raise ValueError("Unsupported or missing progressive KG schema_version.")
        try:
            yield conn
        finally:
            conn.close()

    def metadata(self) -> dict[str, str]:
        with self.connect() as c:
            return dict(c.execute("SELECT key, value FROM meta"))

    def resume_session_context(self) -> dict[str, Any]:
        """Return phase status, module verification state, blockers, decisions, and next step.

        ``verified_modules`` are modules with at least one passing test that compares against an external
        reference (SIESTA, Quantum ESPRESSO, a library). ``verification`` gives every module's state:
        ``verified``, ``failing``, ``self_consistent_only`` (passing tests without an external reference),
        ``skipped`` (its tests were skipped) or ``never_run`` (no result, or no test imports it).
        """
        with self.connect() as c:
            meta = dict(c.execute("SELECT key, value FROM meta"))
            mem = c.execute("SELECT content FROM decision_task_nodes WHERE id='task:memory_state'").fetchone()
            decisions = [
                {"id": r["id"], "title": r["title"]}
                for r in c.execute("SELECT id, title FROM decision_task_nodes WHERE kind='decision' ORDER BY id")
            ]
            recent_logs = [
                {"id": r["id"], "title": r["title"], "content": r["content"]}
                for r in c.execute("SELECT id, title, content FROM decision_task_nodes WHERE kind='work_log' ORDER BY id DESC LIMIT 3")
            ]
            modules = [dict(r) for r in c.execute("SELECT id, qualname, has_required_docs FROM code_entities WHERE kind='module' ORDER BY qualname")]
            verification = _module_verification(c, modules)
            verified_mods = [q for q, v in verification.items() if v["state"] == "verified"]
            unverified_mods = [q for q, v in verification.items() if v["state"] != "verified"]
            undocumented = [
                r["qualname"]
                for r in c.execute("SELECT qualname FROM code_entities WHERE is_public=1 AND has_required_docs=0 ORDER BY qualname")
            ]
        return {
            "metadata": meta,
            "memory_summary": mem["content"] if mem else "",
            "verified_modules": verified_mods,
            "unverified_modules": unverified_mods,
            "verification": verification,
            "undocumented_public_entities": undocumented,
            "decisions": decisions,
            "recent_work_logs": recent_logs,
        }

    def trace_physics_lineage(self, term: str) -> dict[str, Any]:
        """Walk ``TheoryNode <-> CodeEntity <-> LegacySIESTANode <-> ValidationNode`` for a term."""
        with self.connect() as c:
            q_like = f"%{term.lower()}%"
            code_rows = [
                dict(r)
                for r in c.execute(
                    "SELECT id, kind, qualname, file, line_start, line_end, signature FROM code_entities "
                    "WHERE lower(id) LIKE ? OR lower(qualname) LIKE ? OR lower(docstring) LIKE ? ORDER BY qualname LIMIT 30",
                    (q_like, q_like, q_like),
                )
            ]
            theory_rows = [
                dict(r)
                for r in c.execute(
                    "SELECT id, title, file, line_start, line_end FROM theory_nodes "
                    "WHERE lower(id) LIKE ? OR lower(title) LIKE ? OR lower(summary) LIKE ? LIMIT 20",
                    (q_like, q_like, q_like),
                )
            ]
            matched_ids = {r["id"] for r in code_rows} | {r["id"] for r in theory_rows}
            # Walk up ancestor parent_id chain only for theory & validation inheritance
            parent_map = {
                r["id"]: r["parent_id"]
                for r in c.execute("SELECT id, parent_id FROM code_entities")
            }
            ancestor_ids: set[str] = set()
            for mid in matched_ids:
                cur = parent_map.get(mid, "")
                while cur:
                    ancestor_ids.add(cur)
                    cur = parent_map.get(cur, "")

            all_edges = [dict(r) for r in c.execute("SELECT src, dst, kind, status FROM edges")]
            related_edges = [
                e for e in all_edges
                if e["src"] in matched_ids
                or e["dst"] in matched_ids
                or (e["src"] in ancestor_ids and e["kind"] in ("implements_theory", "verified_by"))
            ]
            theory_val_ids = {
                e["dst"] for e in related_edges if e["kind"] in ("implements_theory", "verified_by")
            } | matched_ids
            legacy_ids = {
                e["dst"]
                for e in all_edges
                if e["src"] in matched_ids and e["kind"] in ("replaces_routine", "implements_equivalent_of")
            }
            theory_rows = [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "file": r["file"],
                    "line_start": r["line_start"],
                    "line_end": r["line_end"],
                }
                for r in c.execute("SELECT id, title, file, line_start, line_end, summary FROM theory_nodes")
                if r["id"] in theory_val_ids or term.lower() in r["id"].lower() or term.lower() in r["summary"].lower()
            ]
            legacy = [
                dict(r)
                for r in c.execute("SELECT * FROM legacy_siesta_nodes")
                if r["id"] in legacy_ids or term.lower() == r["name"].lower()
            ]
            validations = []
            for r in c.execute("SELECT id, kind, name, status, file, summary, details_json FROM validation_nodes"):
                if r["id"] not in theory_val_ids:
                    continue
                row = {k: r[k] for k in ("id", "kind", "name", "status", "file", "summary")}
                details = json.loads(r["details_json"] or "{}")
                for key in ("external_reference", "external_status", "counts", "by_platform", "residuals", "missing_tests"):
                    if key in details:
                        row[key] = details[key]
                validations.append(row)
        return {
            "term": term,
            "code_entities": code_rows,
            "theory_nodes": theory_rows,
            "legacy_siesta_nodes": legacy,
            "validation_nodes": validations,
            "edges": related_edges,
        }

    def check_staleness(self) -> dict[str, Any]:
        """Compare live workspace SHA256 hashes against recorded KG tracked_files.

        Source, test and theory files decide ``up_to_date``. Handoff documents (``MEMORY.md`` and
        ``logs/*.md``) are reported separately under ``stale_handoff_docs``: editing session notes
        must not make the code graph look stale, but agents still see that a rebuild is due.
        """
        meta = self.metadata()
        root = Path(meta["repo_root"])
        stale, missing = [], []
        with self.connect() as c:
            tracked = {r["path"]: r["sha256"] for r in c.execute("SELECT path, sha256 FROM tracked_files")}
        for rel, expected in sorted(tracked.items()):
            p = root / rel
            if not p.is_file():
                missing.append(rel)
            elif digest(p) != expected:
                stale.append(rel)
        live_files: set[str] = set()
        for base, exts in [
            (root / "jax_siesta", (".py",)),
            (root / "cpp", (".hpp", ".h", ".cpp", ".cuh", ".cu")),
            (root / "docs" / "theory", (".md",)),
            (root / "tests", (".py", ".json")),
        ]:
            if not base.is_dir():
                continue
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if "build" in p.parts or "__pycache__" in p.parts or "fixtures" in p.parts or "cases" in p.parts:
                    continue
                if base.name == "tests" and p.suffix == ".py" and not p.name.startswith("test_"):
                    continue
                if p.suffix in exts:
                    live_files.add(p.relative_to(root).as_posix())
        added = sorted(live_files - set(tracked.keys()))
        handoff = [p for p in stale + missing if _is_handoff_doc(p)]
        stale = [p for p in stale if not _is_handoff_doc(p)]
        missing = [p for p in missing if not _is_handoff_doc(p)]
        return {
            "up_to_date": not stale and not missing and not added,
            "results_stale": self._results_staleness(meta, root, tracked),
            "stale_handoff_docs": handoff,
            "stale_files": stale,
            "missing_files": missing,
            "untracked_files": added,
            "snapshot_sha256": meta.get("snapshot_sha256"),
        }

    @staticmethod
    def _results_staleness(meta: dict[str, str], root: Path, tracked: dict[str, str]) -> dict[str, Any]:
        """Say whether the ingested test results still describe the working tree.

        Results are stale when no JUnit file was ingested, when a file was recorded at another commit than
        HEAD, or when a tracked source/test file is newer than the JUnit file. This never affects
        ``up_to_date``: running pytest must not make the code graph look stale.
        """
        files = json.loads(meta.get("junit_files", "[]"))
        if not files:
            return {"stale": True, "reasons": ["no JUnit results ingested"]}
        reasons: list[str] = []
        head = git_identity(root)["commit"]
        newest_code = max(
            (
                (root / rel).stat().st_mtime
                for rel in tracked
                if not _is_handoff_doc(rel) and not rel.startswith("reports/") and (root / rel).is_file()
            ),
            default=0.0,
        )
        for f in files:
            if f["git_commit"] and f["git_commit"] != head:
                # without a HEAD (no git, or no commit yet) recorded results cannot be confirmed either
                now = head[:8] if head else "unknown"
                reasons.append(f"{f['platform']}: results from commit {f['git_commit'][:8]}, HEAD is {now}")
            if newest_code > f["mtime"]:
                reasons.append(f"{f['platform']}: a tracked file changed after the results were written")
        return {"stale": bool(reasons), "reasons": reasons, "files": files}

    def search(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT node_id, node_type, title, snippet(fts, 3, '[', ']', '...', 16) AS snippet "
                "FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
                (query, max(1, min(limit, 50))),
            ).fetchall()
            return [dict(r) for r in rows]


def _module_verification(c: sqlite3.Connection, modules: list[dict]) -> dict[str, dict[str, Any]]:
    """Classify each module from the status of its ``verified_by`` edges (see ``resume_session_context``)."""
    suites = {
        r["id"]: {"status": r["status"], "details": json.loads(r["details_json"] or "{}")}
        for r in c.execute("SELECT id, status, details_json FROM validation_nodes")
    }
    by_src: dict[str, list[str]] = {}
    for r in c.execute("SELECT src, dst FROM edges WHERE kind='verified_by'"):
        by_src.setdefault(r["src"], []).append(r["dst"])
    out: dict[str, dict[str, Any]] = {}
    for m in modules:
        dsts = sorted(by_src.get(m["id"], []))
        infos = [suites[d] for d in dsts if d in suites]
        external_pass = [d for d in dsts if d in suites and suites[d]["details"].get("external_status") == "pass"]
        if any(i["status"] == "fail" for i in infos):
            state = "failing"
        elif external_pass:
            state = "verified"
        elif any(i["status"] in ("pass", "partial", "passed", "frozen") for i in infos):
            state = "self_consistent_only"
        elif any(i["status"] == "skip" for i in infos):
            state = "skipped"
        else:
            state = "never_run"
        out[m["qualname"]] = {
            "state": state,
            "suites": {d: suites[d]["status"] for d in dsts if d in suites},
            "external_reference_suites": external_pass,
        }
    return out


def _is_handoff_doc(rel_path: str) -> bool:
    """Return True for session handoff documents (``MEMORY.md``, ``logs/*.md``)."""
    return rel_path == "MEMORY.md" or (rel_path.startswith("logs/") and rel_path.endswith(".md"))


def _kg() -> ProgressiveKG:
    return ProgressiveKG(KG_DB_PATH)


@mcp.tool()
def resume_session_context() -> dict[str, Any]:
    """Return current phase, verified vs unverified modules, blockers, decisions, and next task."""
    return _kg().resume_session_context()


@mcp.tool()
def trace_physics_lineage(term: str) -> dict[str, Any]:
    """Walk TheoryNode <-> CodeEntity <-> LegacySIESTANode <-> ValidationNode for a physics term."""
    return _kg().trace_physics_lineage(term)


@mcp.tool()
def check_staleness() -> dict[str, Any]:
    """Verify live source SHA256 hashes against the published Knowledge Graph snapshot."""
    return _kg().check_staleness()


@mcp.tool()
def search_kg(query: str, limit: int = 12) -> dict[str, Any]:
    """FTS5 search across CodeEntity, TheoryNode, LegacySIESTANode, ValidationNode, and Decisions."""
    kg = _kg()
    return {"results": kg.search(query, limit), "metadata": kg.metadata()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database", nargs="?", type=Path, default=KG_DB_PATH)
    args = p.parse_args()
    KG_DB_PATH = args.database
    mcp.run()
