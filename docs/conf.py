"""Sphinx configuration for jsiesta multi-layer documentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DOCS_DIR))

if importlib.util.find_spec("jax") is None:
    # Provide clean lightweight stubs for Sphinx autodoc when run under Python 3.14
    class _JaxArray:
        pass

    _jax = types.ModuleType("jax")
    _jax.Array = _JaxArray  # type: ignore[attr-defined]
    _jax_tree_util = types.ModuleType("jax.tree_util")
    _jax_tree_util.register_pytree_node_class = lambda cls: cls  # type: ignore[attr-defined]
    _jax_numpy = types.ModuleType("jax.numpy")
    _jax_lax = types.ModuleType("jax.lax")
    _jax_extend = types.ModuleType("jax.extend")
    _jax_extend_ffi = types.ModuleType("jax.extend.ffi")
    _jax.tree_util = _jax_tree_util  # type: ignore[attr-defined]
    _jax.numpy = _jax_numpy  # type: ignore[attr-defined]
    _jax.lax = _jax_lax  # type: ignore[attr-defined]
    _jax.extend = _jax_extend  # type: ignore[attr-defined]
    _jax_extend.ffi = _jax_extend_ffi  # type: ignore[attr-defined]
    _jax.jit = lambda fn=None, **_kw: fn if fn is not None else (lambda f: f)  # type: ignore[attr-defined]
    _jax.vmap = lambda fn=None, **_kw: fn if fn is not None else (lambda f: f)  # type: ignore[attr-defined]
    sys.modules.update({
        "jax": _jax,
        "jax.tree_util": _jax_tree_util,
        "jax.numpy": _jax_numpy,
        "jax.lax": _jax_lax,
        "jax.extend": _jax_extend,
        "jax.extend.ffi": _jax_extend_ffi,
    })

project = "jsiesta"
author = "jsiesta contributors"
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

HAS_MYST = importlib.util.find_spec("myst_parser") is not None
HAS_BREATHE = importlib.util.find_spec("breathe") is not None
if HAS_MYST:
    extensions.append("myst_parser")
if HAS_BREATHE:
    extensions.append("breathe")
    breathe_projects = {"jsiesta_cpp": str(DOCS_DIR / "_doxygen" / "xml")}
    breathe_default_project = "jsiesta_cpp"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Use plain custom section labels without params_style colon-parsing issues
napoleon_custom_sections = [
    ("Theory", "returns_style"),
    ("Algorithm", "returns_style"),
    ("Invariants", "returns_style"),
    ("SIESTA Mapping", "returns_style"),
]

autodoc_mock_imports = [
    m for m in ("numpy", "scipy", "spglib", "lxml") if importlib.util.find_spec(m) is None
]
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = False

exclude_patterns = ["_build", "_doxygen", "theory/*.md", "architecture/*.mmd", "design/*"]
if importlib.util.find_spec("sphinx_immaterial") is not None:
    extensions.append("sphinx_immaterial")
    html_theme = "sphinx_immaterial"
    html_theme_options = {
        "font": False,
        "features": [
            "navigation.expand",
            "navigation.sections",
            "navigation.top",
            "toc.follow",
        ],
        "palette": [
            {
                "media": "(prefers-color-scheme: light)",
                "scheme": "default",
                "primary": "indigo",
                "accent": "deep-purple",
                "toggle": {"icon": "material/weather-night", "name": "Dark mode"},
            },
            {
                "media": "(prefers-color-scheme: dark)",
                "scheme": "slate",
                "primary": "indigo",
                "accent": "deep-purple",
                "toggle": {"icon": "material/weather-sunny", "name": "Light mode"},
            },
        ],
    }
else:
    html_theme = "alabaster"
nitpicky = False


def _md_to_rst(text: str) -> str:
    """Convert theory/log Markdown with $...$ / $$...$$ math into clean RST."""
    import re

    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip() or "text"
            out.append(f".. code-block:: {lang}")
            out.append("")
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                out.append("   " + lines[i])
                i += 1
            out.append("")
            i += 1
            continue
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4:
            expr = line.strip()[2:-2].strip()
            out.extend([".. math::", "", f"   {expr}", ""])
            i += 1
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            out.extend([title, "=" * len(title), ""])
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            out.extend([title, "-" * len(title), ""])
            i += 1
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            out.extend([title, "^" * len(title), ""])
            i += 1
            continue
        converted = re.sub(r"\$([^$\n]+?)\$", r":math:`\1`", line)
        converted = re.sub(r"\*\*([^*]+?)\*\*", r"**\1**", converted)
        out.append(converted)
        i += 1
    return "\n".join(out) + "\n"


def _sync_generated_rst() -> None:
    gen_dir = DOCS_DIR / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # 1. Theory pages
    theory_entries: list[str] = []
    for md in sorted((DOCS_DIR / "theory").glob("*.md")):
        rst_name = f"theory_{md.stem}.rst"
        (gen_dir / rst_name).write_text(_md_to_rst(md.read_text(encoding="utf-8")), encoding="utf-8")
        theory_entries.append(f"   {rst_name}")
    (gen_dir / "theory_index.rst").write_text(
        "Physics & Mathematical Theory\n=============================\n\n.. toctree::\n   :maxdepth: 2\n\n"
        + "\n".join(theory_entries)
        + "\n",
        encoding="utf-8",
    )

    # 2. Architecture Mermaid diagrams + logs
    arch_lines = ["Architecture & Decision Logs", "============================", ""]
    for mmd in sorted((DOCS_DIR / "architecture").glob("*.mmd")):
        title = mmd.stem.replace("_", " ").title()
        arch_lines.extend([title, "-" * len(title), "", ".. code-block:: text", ""])
        for ml in mmd.read_text(encoding="utf-8").splitlines():
            arch_lines.append("   " + ml)
        arch_lines.append("")
    for log_path in (REPO_ROOT / "MEMORY.md", REPO_ROOT / "logs" / "DECISIONS.md", REPO_ROOT / "logs" / "WORK_LOG.md"):
        if log_path.is_file():
            arch_lines.append(_md_to_rst(log_path.read_text(encoding="utf-8")))
    (gen_dir / "architecture_index.rst").write_text("\n".join(arch_lines), encoding="utf-8")

    # 3. Python API modules discovered dynamically from jax_siesta/
    py_mods: list[tuple[str, bool]] = []
    for py_file in sorted((REPO_ROOT / "jax_siesta").rglob("*.py")):
        parts = list(py_file.relative_to(REPO_ROOT).with_suffix("").parts)
        is_init = parts[-1] == "__init__"
        if is_init:
            parts = parts[:-1]
        mod_name = ".".join(parts)
        if not any(m == mod_name for m, _ in py_mods):
            py_mods.append((mod_name, is_init))
    py_api = ["Python / JAX API Reference", "==========================", ""]
    for mod, is_init in py_mods:
        if is_init:
            py_api.extend([mod, "-" * len(mod), "", f".. automodule:: {mod}", ""])
        else:
            py_api.extend([mod, "-" * len(mod), "", f".. automodule:: {mod}", "   :members:", "   :undoc-members:", ""])
    (gen_dir / "python_api.rst").write_text("\n".join(py_api), encoding="utf-8")

    # 4. C++ / CUDA FFI API extracted from Doxygen XML
    def _doxy_elem_to_rst(elem: ET.Element | None) -> str:
        if elem is None:
            return ""
        parts: list[str] = []

        def _walk(node: ET.Element, is_root: bool = False) -> None:
            if node.tag == "formula":
                raw = (node.text or "").strip()
                if raw.startswith("$") and raw.endswith("$") and len(raw) > 2:
                    parts.append(f" :math:`{raw[1:-1]}` ")
                elif raw.startswith("\\[") and raw.endswith("\\]"):
                    parts.append(f" :math:`{raw[2:-2].strip()}` ")
                elif raw:
                    parts.append(raw)
            elif node.tag == "computeroutput":
                txt = " ".join("".join(node.itertext()).split())
                if txt:
                    parts.append(f" ``{txt}`` ")
            elif node.tag == "simplesect":
                title_el = node.find("title")
                title_txt = " ".join("".join(title_el.itertext()).split()) if title_el is not None else ""
                if title_txt:
                    parts.append(f" **{title_txt}:** ")
                for child in node:
                    if child is not title_el:
                        _walk(child, False)
            elif node.tag == "parameterlist":
                parts.append(" **Parameters:** ")
                for item in node.findall("parameteritem"):
                    pnames = ", ".join(
                        " ".join("".join(pn.itertext()).split())
                        for pn in item.findall(".//parametername")
                    )
                    pd_el = item.find("parameterdescription")
                    pdesc = " ".join("".join(pd_el.itertext()).split()) if pd_el is not None else ""
                    if pnames:
                        parts.append(f" ``{pnames}`` ({pdesc}); ")
            else:
                if node.text:
                    parts.append(node.text)
                for child in node:
                    _walk(child, False)
            if not is_root and node.tail:
                parts.append(node.tail)

        _walk(elem, True)
        return " ".join("".join(parts).split())

    cpp_api = ["C++ / CUDA FFI Reference", "========================", ""]
    xml_dir = DOCS_DIR / "_doxygen" / "xml"
    if xml_dir.is_dir():
        for xf in sorted(xml_dir.glob("*.xml")):
            if xf.name in ("index.xml", "Doxyfile.xml"):
                continue
            root = ET.parse(xf).getroot()
            for comp in root.iter("compounddef"):
                kind = comp.get("kind", "")
                cname = comp.findtext("compoundname", "")
                if kind in ("dir", "page", "group") or cname == "std" or cname.startswith("@"):
                    continue
                header = f"{kind}: {cname}"
                cpp_api.extend([header, "-" * len(header), ""])
                brief = _doxy_elem_to_rst(comp.find("briefdescription"))
                detail = _doxy_elem_to_rst(comp.find("detaileddescription"))
                if brief:
                    cpp_api.extend([brief, ""])
                if detail:
                    cpp_api.extend([detail, ""])
                for mem in comp.iter("memberdef"):
                    mname = mem.findtext("name", "")
                    if not mname or mname.startswith("_"):
                        continue
                    sig = ((mem.findtext("definition", "") or mname) + (mem.findtext("argsstring", "") or "")).strip()
                    mbrief = _doxy_elem_to_rst(mem.find("briefdescription"))
                    mdetail = _doxy_elem_to_rst(mem.find("detaileddescription"))
                    cpp_api.extend([f"- ``{sig}``: {mbrief}"])
                    if mdetail:
                        cpp_api.extend([f"  {mdetail}"])
                cpp_api.append("")
    (gen_dir / "cpp_api.rst").write_text("\n".join(cpp_api), encoding="utf-8")


_sync_generated_rst()
