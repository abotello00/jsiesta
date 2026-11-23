"""Progressive dual-audience Knowledge Graph for jsiesta.

Theory:
    Maintains a revision-bound SQLite/FTS5 + JSON knowledge graph connecting
    mathematical derivations (``TheoryNode``), Python/JAX AST and C++/CUDA Doxygen
    implementations (``CodeEntity``), legacy SIESTA provenance
    (``LegacySIESTANode``), numerical verification evidence (``ValidationNode``),
    and architectural decisions/tasks (``DecisionAndTaskNode``).

Algorithm:
    Builds each generation in a staging directory and publishes atomically via
    ``kg/current`` symlink swaps while emitting Mermaid architecture diagrams.

Invariants:
    - Failed builds never overwrite ``kg/current``.
    - Every recorded ``CodeEntity`` pins its exact source file SHA256 and line range.

SIESTA Mapping:
    Extends the revision-bound indexing architecture of ``knowledge.py`` and
    ``siesta_mcp.py`` into a full progressive agent+human development graph.
"""
