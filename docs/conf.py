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
try:
    from jax_siesta import __version__ as release
except Exception:  # the package needs its compiled dependencies only at run time; fall back if the import fails
    release = "0.1.0"
version = release
html_title = f"jsiesta {release}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

HAS_MYST_NB = importlib.util.find_spec("myst_nb") is not None
HAS_MYST = importlib.util.find_spec("myst_parser") is not None
HAS_BREATHE = importlib.util.find_spec("breathe") is not None
if HAS_MYST_NB:
    # The tutorials are notebooks stored with their outputs: they need the compiled library to run, so the
    # documentation build renders them as they are (`make tutorials` executes them).
    extensions.append("myst_nb")
    nb_execution_mode = "off"
    myst_enable_extensions = ["dollarmath", "amsmath"]
elif HAS_MYST:
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
# The summary tables of the API sections list objects by full name on the page that documents their module.
suppress_warnings = ["autosummary.import_cycle"]

exclude_patterns = [
    "_build",
    "_doxygen",
    "theory/*.md",
    "architecture/*.mmd",
    "design/*",
    "_generated/api/*",
    "tutorials/*.py",
    "tutorials/work",
    "tutorials/.ipynb_checkpoints",
    "**/.ipynb_checkpoints",
]
if not HAS_MYST_NB:
    exclude_patterns.append("tutorials/*.ipynb")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
if importlib.util.find_spec("sphinx_immaterial") is not None:
    extensions.append("sphinx_immaterial")
    html_theme = "sphinx_immaterial"
    html_theme_options = {
        "font": False,
        "site_url": "https://abotello00.github.io/jsiesta/",
        "repo_url": "https://github.com/abotello00/jsiesta",
        "repo_name": "abotello00/jsiesta",
        "icon": {"repo": "fontawesome/brands/github"},
        "globaltoc_collapse": True,
        "features": [
            "navigation.tabs",
            "navigation.tabs.sticky",
            "navigation.sections",
            "navigation.top",
            "navigation.tracking",
            "toc.follow",
            "content.code.copy",
            "search.highlight",
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


# Sections of the Python API reference (docs/api/<section>.rst) and the packages each one documents.
API_SECTIONS: dict[str, list[str]] = {
    "interface": ["jax_siesta.app"],
    "core": ["jax_siesta", "jax_siesta.core"],
    "species": ["jax_siesta.species"],
    "integrals": ["jax_siesta.integrals"],
    "grid": ["jax_siesta.grid"],
    "xc": ["jax_siesta.xc", "jax_siesta.hfx"],
    "symmetry": ["jax_siesta.symmetry"],
    "scf": ["jax_siesta.scf"],
    "observables": ["jax_siesta.observables"],
    "io": ["jax_siesta.io"],
    "testing": ["jax_siesta.testing"],
}
# Sections of the C++ / CUDA reference: (title, substrings of the Doxygen compound names that belong to it).
CPP_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("Two-centre integrals", ("two_center", "kb_three_center")),
    ("Orbital-grid kernels", ("grid_ffi", "grid_plan", "jsiesta::grid")),
    ("Exchange-correlation kernels", ("xc_ffi", "grid_xc_ffi", "xc_semilocal", "jsiesta::xc", "libxc_compat")),
    ("Exact exchange", ("eri.", "hfx.", "jsiesta::eri")),
    ("CUDA kernels", ("kernels.cuh", "rhoofd_vmat", "jsiesta::cuda")),
]
CPP_FALLBACK = "FFI registry, types and utilities"
CPP_INTROS = {
    "Two-centre integrals": "Overlap, kinetic, neutral-atom and Kleinman-Bylander matrix elements and their "
    "derivatives, evaluated from radial tables for every orbital pair of the sparsity pattern.",
    "Orbital-grid kernels": "The density on the mesh from the density matrix (``rhoofd``), the matrix elements "
    "of a mesh potential (``vmat``), and the plan that stores which orbitals touch which mesh points.",
    "Exchange-correlation kernels": "Pointwise LDA, GGA and meta-GGA evaluation on the mesh, built in or through "
    "Libxc.",
    "Exact exchange": "Four-centre electron repulsion integrals over Gaussian fits of the orbitals and the "
    "screened construction of the exchange matrix of hybrid functionals.",
    "CUDA kernels": "GPU versions of the orbital-grid kernels and of the semilocal exchange-correlation.",
    CPP_FALLBACK: "Registration of the kernels with the XLA foreign-function interface, shared views of the "
    "arrays passed from Python, splines and threading helpers.",
}
_ARCH_INTRO = (
    "Diagrams regenerated by ``make kg`` from the knowledge graph (Mermaid source; paste into a Mermaid viewer "
    "to render): module dependencies, the lineage from theory notes to code, and which tests validate what."
)


def _public_names(py_file: Path) -> list[str]:
    """Names listed in ``__all__`` of a module, or its public top-level functions and classes."""
    import ast

    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                names = [str(n) for n in ast.literal_eval(node.value)]
            except ValueError:
                break
            defined = {
                n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            return [n for n in names if n in defined]
    return [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_")
    ]


def _sync_generated_rst() -> None:
    gen_dir = DOCS_DIR / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("architecture_index", "cpp_api", "python_api", "theory_index"):  # pages of the old layout
        (gen_dir / f"{stale}.rst").unlink(missing_ok=True)

    # 1. Theory pages (the hand-written docs/theory/index.rst lists them)
    for md in sorted((DOCS_DIR / "theory").glob("*.md")):
        rst_name = f"theory_{md.stem}.rst"
        (gen_dir / rst_name).write_text(_md_to_rst(md.read_text(encoding="utf-8")), encoding="utf-8")

    # 2. Development pages: generated diagrams, project status and logs, one page each
    arch_lines = ["Architecture diagrams", "=====================", "", _ARCH_INTRO, ""]
    for mmd in sorted((DOCS_DIR / "architecture").glob("*.mmd")):
        title = mmd.stem.replace("_", " ").title()
        arch_lines.extend([title, "-" * len(title), "", ".. code-block:: text", ""])
        for ml in mmd.read_text(encoding="utf-8").splitlines():
            arch_lines.append("   " + ml)
        arch_lines.append("")
    (gen_dir / "dev_architecture.rst").write_text("\n".join(arch_lines), encoding="utf-8")
    for name, log_path in (
        ("dev_status", REPO_ROOT / "MEMORY.md"),
        ("dev_decisions", REPO_ROOT / "logs" / "DECISIONS.md"),
        ("dev_work_log", REPO_ROOT / "logs" / "WORK_LOG.md"),
    ):
        body = _md_to_rst(log_path.read_text(encoding="utf-8")) if log_path.is_file() else f"{name}\n{'=' * len(name)}\n"
        (gen_dir / f"{name}.rst").write_text(body, encoding="utf-8")

    # 3. Python API: one fragment per section of docs/api/, included by the hand-written page of that section.
    #    Every package of jax_siesta must belong to a section, so a new package cannot be left undocumented.
    api_dir = gen_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    packages = sorted(
        ".".join(p.parent.relative_to(REPO_ROOT).parts) for p in (REPO_ROOT / "jax_siesta").rglob("__init__.py")
    )
    assigned = {pkg for pkgs in API_SECTIONS.values() for pkg in pkgs}
    missing = [pkg for pkg in packages if pkg not in assigned]
    if missing:
        raise RuntimeError(f"docs/conf.py: packages without an API section (add them to API_SECTIONS): {missing}")
    for section, pkgs in API_SECTIONS.items():
        lines: list[str] = []
        for pkg in pkgs:
            pkg_dir = REPO_ROOT.joinpath(*pkg.split("."))
            lines.extend([f"Package ``{pkg}``", "-" * (len(pkg) + 12), "", f".. automodule:: {pkg}", "   :no-index:", ""])
            for py_file in sorted(pkg_dir.glob("*.py")):
                if py_file.name.startswith("__"):
                    continue
                mod = f"{pkg}.{py_file.stem}"
                names = _public_names(py_file)
                lines.extend([f"``{mod}``", "^" * (len(mod) + 4), ""])
                if names:
                    lines.extend([".. autosummary::", "   :nosignatures:", ""])
                    lines.extend(f"   {mod}.{name}" for name in names)
                    lines.append("")
                lines.extend([f".. automodule:: {mod}", "   :members:", "   :undoc-members:", ""])
        (api_dir / f"{section}.inc").write_text("\n".join(lines), encoding="utf-8")

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

    cpp_groups: dict[str, list[str]] = {title: [] for title, _ in CPP_SECTIONS}
    cpp_groups[CPP_FALLBACK] = []
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
                cpp_api = cpp_groups[
                    next((title for title, keys in CPP_SECTIONS if any(k in cname for k in keys)), CPP_FALLBACK)
                ]
                cpp_api.extend([header, "^" * len(header), ""])
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
    cpp_out: list[str] = []
    for title, body in cpp_groups.items():
        if body:
            cpp_out.extend([title, "-" * len(title), "", CPP_INTROS.get(title, ""), "", *body])
    (api_dir / "cpp.inc").write_text("\n".join(cpp_out), encoding="utf-8")


_sync_generated_rst()
