r"""The ``jsiesta`` command: run a SIESTA input file.

Theory:
    SIESTA is driven by one input file and writes a text log plus files named after ``SystemLabel``. The command
    reproduces that way of working for the static single-point scope of jsiesta, so existing inputs can be tried
    without writing Python.

Algorithm:
    ``Calculation.from_fdf`` -> ``Calculation.run`` (log to standard output) -> ``write_outputs``. As with
    SIESTA the input may be given on standard input (``jsiesta < in.fdf > out``); it may also be a file argument.

Invariants:
    - Exit status 0 for a converged run, 1 when the SCF did not converge and ``SCF.MustConverge`` is true (the
      default, as in SIESTA), 2 for input errors.
    - Output files are written in the directory of the run, never next to the package.

SIESTA Mapping:
    The ``siesta`` executable for ``MD.TypeOfRun`` single point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

from jax_siesta.app import report
from jax_siesta.app.calculation import Calculation
from jax_siesta.app.outputs import write_outputs


def _say(line: str) -> None:
    """Print a log line immediately (the SCF table should appear as it is computed)."""
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``jsiesta`` command (also ``python -m jax_siesta``).

    Theory:
        One static calculation per invocation: energies, forces and stress of the geometry in the input file.

    Algorithm:
        Parses the arguments, reads the input file (or standard input), runs, prints the report and writes
        ``<SystemLabel>.XV``, ``.DM``, ``.FA``, ``.EIG``, ``.json`` (and ``.bands`` with ``%block BandLines``,
        ``.nc`` with ``CDF.Save T``) in the output directory.

    Parameters:
        argv: Command-line arguments (default ``sys.argv[1:]``).

    Returns:
        Exit status: 0 converged, 1 not converged with ``SCF.MustConverge``, 2 input error.

    Invariants:
        Input errors are reported in one line on standard error, without a traceback.

    SIESTA Mapping:
        ``siesta < input.fdf > output``.
    """
    parser = argparse.ArgumentParser(
        prog="jsiesta",
        description="Static Kohn-Sham calculation (energy, forces, stress) from a SIESTA .fdf input file and "
        "PSML pseudopotentials.",
        epilog="Pseudopotentials <Label>.psml are searched in --pseudo-dir, next to the input file and in "
        "SIESTA_PS_PATH. Documentation: https://abotello00.github.io/jsiesta/",
    )
    parser.add_argument("input", nargs="?", help="input .fdf file (read from standard input when omitted)")
    parser.add_argument("--pseudo-dir", type=Path, default=None, help="directory with <Label>.psml files")
    parser.add_argument("--out-dir", type=Path, default=None, help="directory for output files (default: current)")
    parser.add_argument("--check", action="store_true", help="read the input, build the basis, report and stop")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from jax_siesta import __version__

        print(f"jsiesta {__version__}")
        return 0

    stdin_copy: Path | None = None
    try:
        if args.input is None:
            if sys.stdin.isatty():
                parser.print_usage(sys.stderr)
                print("jsiesta: error: no input file (give one as argument or on standard input)", file=sys.stderr)
                return 2
            # An input on standard input is stored in the current directory so that %include and
            # pseudopotential lookups resolve as they do for SIESTA.
            with tempfile.NamedTemporaryFile("w", suffix=".fdf", prefix=".jsiesta_stdin_", dir=".", delete=False) as fh:
                fh.write(sys.stdin.read())
                stdin_copy = Path(fh.name)
            fdf = stdin_copy
        else:
            fdf = Path(args.input)
            if not fdf.is_file():
                print(f"jsiesta: error: input file {fdf} not found", file=sys.stderr)
                return 2
        try:
            calc = Calculation.from_fdf(fdf, pseudo_dir=args.pseudo_dir)
        except (ValueError, FileNotFoundError, KeyError, IndexError) as exc:
            print(f"jsiesta: error in the input: {exc}", file=sys.stderr)
            return 2
    finally:
        if stdin_copy is not None:
            stdin_copy.unlink(missing_ok=True)
    if stdin_copy is not None:
        calc.info["input_file"] = "<standard input>"

    for line in report.header_lines(calc):
        _say(line)
    if args.check:
        _say("")
        _say("jsiesta: --check: input and species are valid; no calculation was run")
        return 0

    try:
        results = calc.run(log=_say)
    except FileNotFoundError as exc:  # the compiled library is missing
        print(f"jsiesta: error: {exc}", file=sys.stderr)
        return 2
    io = calc.info["io"]
    formats = ("XV", "DM", "nc") if io.get("cdf_save") else ("XV", "DM")
    out_dir = args.out_dir if args.out_dir is not None else Path(".")
    try:
        written = write_outputs(results, out_dir, restart_formats=formats, band_lines=calc.info["band_lines"])
    except ImportError as exc:
        print(f"jsiesta: WARNING: {exc}; writing without the .nc file", file=sys.stderr)
        written = write_outputs(results, out_dir, restart_formats=("XV", "DM"), band_lines=calc.info["band_lines"])
    _say("")
    _say("jsiesta: files written: " + ", ".join(str(p) for p in written.values()))
    if not results.converged and io.get("must_converge", True):
        print("jsiesta: SCF did not converge (SCF.MustConverge is true): exit status 1", file=sys.stderr)
        return 1
    _say("")
    _say("Job completed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
