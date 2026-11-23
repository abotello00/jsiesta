#!/usr/bin/env python3
"""Generate fresh Doxygen XML and atomically publish an index for one worktree."""
import argparse
from pathlib import Path
import subprocess
import tempfile
from index_docs import main, source_snapshot


def rebuild(source, output, references=True, timeout=600, excludes=(), inputs=()):
    source = Path(source).resolve()
    config_dir = source / 'Docs' / 'developer'
    config = (config_dir / 'doxygen-siesta').read_text()
    before = source_snapshot(source)
    excluded = []
    for item in excludes:
        path = (source / item).resolve()
        if not path.is_relative_to(source) or not path.exists():
            raise ValueError('Exclusions must exist below the source root.')
        excluded.append(path)
    version = subprocess.check_output(['doxygen', '--version'], text=True).strip()
    with tempfile.TemporaryDirectory(prefix='siesta-doxygen-') as tmp:
        overrides = (f'\nOUTPUT_DIRECTORY = "{tmp}"\nGENERATE_XML = YES\n'
                     'GENERATE_HTML = NO\nGENERATE_LATEX = NO\nEXTRACT_ALL = YES\n'
                     'HAVE_DOT = NO\nCALL_GRAPH = NO\nCALLER_GRAPH = NO\n'
                     f'REFERENCES_RELATION = {"YES" if references else "NO"}\n'
                     'REFERENCED_BY_RELATION = NO\n')
        if inputs:
            selected = [(source / item).resolve() for item in inputs]
            if any(not p.is_relative_to(source) or not p.exists() for p in selected):
                raise ValueError('Inputs must exist below the source root.')
            overrides += 'INPUT = ' + ' '.join(chr(34)+str(p)+chr(34) for p in selected) + '\n'
        if excluded:
            overrides += 'EXCLUDE += ' + ' '.join(chr(34)+str(p)+chr(34) for p in excluded) + '\n'
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        # Keep diagnostics even when generation fails. Never copy old edges.
        log_path = output / 'doxygen-last.log'
        with log_path.open('w') as log:
            result = subprocess.run(['doxygen', '-'], input=config+overrides, text=True,
                                    cwd=config_dir, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Doxygen failed ({result.returncode}); see {log_path}. Active index unchanged. '
                               'Retry --no-references if extraction crashes; graph coverage will be explicit.')
        if before != source_snapshot(source):
            raise RuntimeError('Source changed during Doxygen generation; active index unchanged.')
        db = main(Path(tmp) / 'xml', output, source,
                  alignment='fresh_generation_source_hashes_checked',
                  generator=f'doxygen {version}; references={references}; exclusions={[str(p.relative_to(source)) for p in excluded]}; inputs={list(inputs) or ["source configuration"]}',
                  expected_snapshot=before, effective_config=config+overrides)
        return db


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('output')
    p.add_argument('--no-references', action='store_true')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--input', action='append', default=[], help='Source-relative input subset; recorded as partial coverage')
    p.add_argument('--exclude', action='append', default=[], help='Explicit source-relative exclusion, recorded in provenance')
    args = p.parse_args()
    print(rebuild(args.source, args.output, not args.no_references, args.timeout, args.exclude, args.input))
