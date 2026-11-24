"""``python -m jax_siesta input.fdf``, the same as the ``jsiesta`` command.

Theory:
    None of its own; a calculation is defined by its input file (see ``jax_siesta.app.cli``).

Algorithm:
    Calls ``jax_siesta.app.cli.main`` and exits with its status.

Invariants:
    Behaves exactly as the ``jsiesta`` console script.

SIESTA Mapping:
    The ``siesta`` executable.
"""

from jax_siesta.app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
