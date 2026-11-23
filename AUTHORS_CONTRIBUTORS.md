# Authors and Contributors

Commits in this repository carry a neutral identity (`jsiesta <noreply@jsiesta.invalid>`) and one-line
messages. This file records who did the work.

## Project lead

- **Project owner** — scientific direction, `docs/design/IMPLEMENTATION_PLAN.md`, reference SIESTA builds
  (standard, `siesta_spglib_full_impl`, `pr_mgga`), the `scf_params` pilots, and review of all changes.
  *(Name to be filled in by the owner.)*

## Implementation

| Period | Contributor | Scope |
|---|---|---|
| 2026-11-18 → 2026-11-19 | Gemini agent (Antigravity CLI, conversation `a72012ba-3450-43b2-99aa-de5fe7d1d4ea`) | Original implementation of Phases 0–3 imported as the baseline commit: JAX/C++/CUDA engine, symmetry engine, meta-GGA/NC/SOC, Tier B PAO generator, basis-enthalpy optimiser, knowledge graph, docs |
| 2026-11-19 | Claude Code (Claude Opus 5.5) | Local performance/accuracy benchmark and the independent review that defines the v0.1 work plan |
| 2026-11-20 → 2026-11-23 | Claude Code (Claude Opus 5.5) | v0.1 remediation (quality gates, external-reference configuration, validation fixes, physics corrections) and the performance work D-031 to D-044 (GPU kernels, forces, memory), listed in `logs/WORK_LOG.md` and `logs/DECISIONS.md` |

## Third-party code and data

- `knowledge.py`, `index_docs.py`, `rebuild_index.py`, `siesta_mcp.py`, `validation.py` — vendored from the
  owner's `siesta-mcp` project.
- Reference physics follows SIESTA (GPL-3.0) conventions; no SIESTA source code is included (clean-room).
- Libxc, spglib, JAX — external dependencies under their own licences.
