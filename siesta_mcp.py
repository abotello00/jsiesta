#!/usr/bin/env python3
"""Read-only MCP interface to a revision-bound SIESTA index and evidence reports."""
import argparse
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from knowledge import KnowledgeBase, compare_indexes

mcp = FastMCP('siesta-dev')
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / 'indexes' / 'default' / 'current' / 'docs.db'
REPORT_ROOT = BASE / 'reports'


def kb():
    # Resolve once per request: a running server sees newly published generations.
    return KnowledgeBase(DB_PATH)


@mcp.tool()
def search_docs(query: str, limit: int = 8) -> dict:
    """FTS5 search of documentation, qualified names, and signatures."""
    index = kb()
    return {'results': index.search(query, limit), 'provenance': index.metadata()}


@mcp.tool()
def get_entity(name: str) -> dict:
    """Find symbol cards by ID, qualified name, bare name, or substring. Lists ambiguity."""
    return kb().entity(name)


@mcp.tool()
def get_source(file: str, start: int, end: int) -> dict:
    """Read at most 400 lines of indexed source; refuse changed files and ambiguous paths."""
    return kb().source(file, start, end)


@mcp.tool()
def get_callers(name: str) -> dict:
    """Incoming Doxygen references, not proven calls. Use a unique qualified name or ID."""
    return kb().relations(name, reverse=True)


@mcp.tool()
def get_callees(name: str) -> dict:
    """Outgoing Doxygen references and unresolved relationships; not a complete call graph."""
    return kb().relations(name)


@mcp.tool()
def graph_path(from_name: str, to_name: str, max_depth: int = 6) -> dict:
    """Bounded BFS through references. No path does not establish independence."""
    return kb().graph_path(from_name, to_name, max_depth)


@mcp.tool()
def module_overview(module: str) -> dict:
    """List module entities including undocumented routines and variables."""
    return kb().module(module)


@mcp.tool()
def stats() -> dict:
    """Index provenance, current Git identity, file freshness, coverage and counts."""
    return kb().health()


@mcp.tool()
def extraction_report(component: str) -> dict:
    """Boundary references for a module or symbol; lists unknown state/ownership analyses."""
    return kb().extraction_report(component)


@mcp.tool()
def compare_index(baseline_db: str) -> dict:
    """Compare this index's structural metadata against another generated index."""
    return compare_indexes(KnowledgeBase(baseline_db), kb())


@mcp.tool()
def list_evidence() -> dict:
    """List saved numerical comparison reports. Reports are not proofs of physical equivalence."""
    root = REPORT_ROOT.resolve()
    return {'reports': [p.relative_to(root).as_posix() for p in sorted(root.rglob('report.json'))]} if root.is_dir() else {'reports': []}


@mcp.tool()
def read_evidence(report: str) -> dict:
    """Read one JSON evidence report below the configured reports root (maximum 2 MB)."""
    root = REPORT_ROOT.resolve()
    path = (root / report).resolve()
    if not path.is_relative_to(root) or path.name != 'report.json':
        raise ValueError('Provide a report.json path below the reports directory.')
    if path.stat().st_size > 2_000_000:
        raise ValueError('Report too large; inspect the local artifact.')
    return json.loads(path.read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('database', nargs='?', type=Path, default=DB_PATH)
    p.add_argument('--reports', type=Path, default=REPORT_ROOT)
    args = p.parse_args()
    DB_PATH, REPORT_ROOT = args.database, args.reports
    mcp.run()
