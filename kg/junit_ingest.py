"""Ingest pytest JUnit XML into per-suite verification status for the knowledge graph (review item M7).

A test suite is one ``tests/test_<stem>.py`` file; its graph id is ``val:pytest:<stem>``. The JUnit
``classname`` is ``tests.test_<stem>`` (plus a class suffix for ``unittest`` classes), so the suite is the
second dotted component. Residuals and tolerances recorded by tests with ``record_property("resid:<key>", v)``
and ``record_property("tol:<key>", t)`` (see ``tests/conftest.py``) are carried along per test.

Statuses: per test ``pass | fail | error | skip``; per suite ``pass`` (all pass), ``fail`` (any fail/error),
``partial`` (passes and skips), ``skip`` (all skipped) and ``never-run`` (no entry in any ingested file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

# Worst-first ordering used when several runs (platforms) report the same test.
_SEVERITY = {"error": 0, "fail": 1, "skip": 2, "pass": 3}


@dataclass
class TestResult:
    """One executed testcase from one JUnit file."""

    __test__ = False  # not a pytest test class

    suite: str
    name: str
    status: str
    time_s: float
    message: str
    platform: str
    properties: dict[str, str] = field(default_factory=dict)

    def residuals(self) -> dict[str, dict[str, float]]:
        """Return ``{key: {"resid": r, "tol": t}}`` from ``resid:``/``tol:`` properties."""
        out: dict[str, dict[str, float]] = {}
        for prop, value in self.properties.items():
            for tag, label in (("resid:", "resid"), ("tol:", "tol")):
                if prop.startswith(tag):
                    try:
                        out.setdefault(prop[len(tag):], {})[label] = float(value)
                    except ValueError:
                        pass
        return out


@dataclass
class JunitRun:
    """Parsed content of one JUnit file."""

    path: str
    platform: str
    git_commit: str
    mtime: float
    results: list[TestResult]


def suite_of(classname: str) -> str | None:
    """Return the test-module stem (``test_foo``) of a JUnit ``classname``, or ``None``."""
    for part in classname.split("."):
        if part.startswith("test_"):
            return part
    return None


def parse_junit(path: Path, platform: str = "local") -> JunitRun:
    """Parse a pytest JUnit XML file (xunit1 or xunit2 layout)."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    suite_props: dict[str, str] = {}
    for ts in suites:
        for prop in ts.findall("properties/property"):
            suite_props[prop.get("name", "")] = prop.get("value", "")
    results: list[TestResult] = []
    for tc in root.iter("testcase"):
        stem = suite_of(tc.get("classname", ""))
        if stem is None:
            continue
        status, message = "pass", ""
        for tag, label in (("error", "error"), ("failure", "fail"), ("skipped", "skip")):
            el = tc.find(tag)
            if el is not None:
                status = label
                message = (el.get("message") or el.text or "").strip()[:500]
                break
        props = {p.get("name", ""): p.get("value", "") for p in tc.findall("properties/property")}
        results.append(
            TestResult(
                suite=stem,
                name=tc.get("name", ""),
                status=status,
                time_s=float(tc.get("time", "0") or 0.0),
                message=message,
                platform=platform,
                properties=props,
            )
        )
    return JunitRun(
        path=str(path),
        platform=suite_props.get("platform", platform),
        git_commit=suite_props.get("git_commit", ""),
        mtime=path.stat().st_mtime,
        results=results,
    )


def aggregate_status(statuses: list[str]) -> str:
    """Collapse test statuses into a suite status (see module docstring)."""
    if not statuses:
        return "never-run"
    if any(s in ("fail", "error") for s in statuses):
        return "fail"
    if all(s == "skip" for s in statuses):
        return "skip"
    if any(s == "skip" for s in statuses):
        return "partial"
    return "pass"


def suite_summary(results: list[TestResult]) -> dict[str, dict]:
    """Group results by suite: overall status, counts, and status per platform."""
    by_suite: dict[str, list[TestResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)
    out: dict[str, dict] = {}
    for stem, rs in by_suite.items():
        platforms = sorted({r.platform for r in rs})
        counts = {s: sum(1 for r in rs if r.status == s) for s in _SEVERITY}
        out[stem] = {
            "status": aggregate_status([r.status for r in rs]),
            "counts": counts,
            "by_platform": {p: aggregate_status([r.status for r in rs if r.platform == p]) for p in platforms},
        }
    return out
