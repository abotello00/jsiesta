#!/usr/bin/env python3
"""Run explicit baseline/candidate commands and compare finite numeric observables.

Commands run only through this local CLI, never through an MCP read operation.
Each command must write observables.json inside its isolated working directory.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from knowledge import digest, git_identity


def flatten(value, prefix=''):
    result = {}
    if not prefix and not isinstance(value, dict):
        raise ValueError('Observables must be a named object.')
    if isinstance(value, dict):
        if not value:
            raise ValueError(f'Empty observable object: {prefix}')
        for key, child in value.items():
            if not key or any(c in key for c in '.[]'):
                raise ValueError('Observable keys must be nonempty and may not contain . [ ].')
            result.update(flatten(child, f'{prefix}.{key}' if prefix else key))
    elif isinstance(value, list):
        if not value:
            raise ValueError(f'Empty observable array: {prefix}')
        for i, child in enumerate(value):
            result.update(flatten(child, f'{prefix}[{i}]'))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        if not prefix:
            raise ValueError('Observables must be a named object.')
        result[prefix] = value
    else:
        raise ValueError(f'Observable {prefix} must be finite numeric data.')
    return result


def compare_observables(baseline, candidate, tolerances):
    a, b = flatten(baseline), flatten(candidate)
    if a.keys() != b.keys() or a.keys() != tolerances.keys():
        raise ValueError('Baseline, candidate, and explicit tolerance keys must match exactly. '
                         f'baseline={sorted(a)}, candidate={sorted(b)}, tolerances={sorted(tolerances)}')
    comparisons = []
    for key in sorted(a):
        policy = tolerances[key]
        atol, rtol = policy['atol'], policy['rtol']
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (atol, rtol)):
            raise ValueError('Tolerances must be finite nonnegative numbers.')
        if not isinstance(policy.get('unit'), str) or not policy['unit']:
            raise ValueError('Every tolerance requires an explicit unit (use dimensionless when appropriate).')
        difference = abs(b[key]-a[key])
        threshold = atol + rtol*abs(a[key])
        if not math.isfinite(difference) or not math.isfinite(threshold):
            raise ValueError('Numeric overflow during comparison.')
        comparisons.append({'observable': key, 'baseline': a[key], 'candidate': b[key],
                            'absolute_difference': difference, 'allowed_difference': threshold,
                            'policy': policy, 'passed': difference <= threshold})
    return {'passed': all(x['passed'] for x in comparisons), 'comparisons': comparisons}


def tree_hashes(directory):
    root = Path(directory).resolve()
    hashes = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Input/source evidence trees may not contain symlinks: {path}')
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = digest(path)
    return hashes


def run_variant(spec, manifest_dir, input_dir, destination, timeout, environment):
    work = destination.resolve()
    shutil.copytree(input_dir, work)
    # Inputs may never supply a result which a failed adapter could leave untouched.
    (work / 'observables.json').unlink(missing_ok=True)
    args = spec['command']
    if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
        raise ValueError('command must be a nonempty argument list; shell strings are unsupported.')
    executable = Path(sys.executable if args[0] == '{python}' else args[0])
    if not executable.is_absolute():
        executable = (manifest_dir / executable).resolve()
    args = [str(executable), *[x.replace('{workdir}', str(work)).replace('{manifestdir}', str(manifest_dir)) for x in args[1:]]]
    artifacts = [executable, *[(manifest_dir / p).resolve() for p in spec.get('artifacts', [])]]
    artifact_hashes = {str(p): digest(p) for p in artifacts}
    identity = spec.get('identity')
    if not isinstance(identity, dict) or not identity.get('build') or not identity.get('revision'):
        raise ValueError('Each variant requires identity.build and identity.revision.')
    source = spec.get('source_root')
    source_identity = git_identity((manifest_dir / source).resolve()) if source else None
    env = os.environ.copy()
    env.update(environment)
    started = time.monotonic()
    timed_out = False
    with (work / 'stdout.log').open('wb') as stdout, (work / 'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(args, cwd=work, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    changed_artifacts = [str(p) for p in artifacts if not p.is_file() or digest(p) != artifact_hashes[str(p)]]
    result = {'command': args, 'identity': identity, 'source_git': source_identity,
              'artifacts_sha256': artifact_hashes, 'changed_artifacts': changed_artifacts,
              'returncode': code, 'timeout': timed_out, 'seconds': time.monotonic()-started,
              'workdir': str(work), 'observables': None}
    if code == 0 and not timed_out and not changed_artifacts:
        result['observables'] = json.loads((work / 'observables.json').read_text())
        flatten(result['observables'])
    return result


def run_manifest(manifest_path, output_root):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('cases'), list) or not manifest['cases']:
        raise ValueError('Require schema_version=1 and nonempty cases.')
    timeout = manifest.get('timeout_seconds', 300)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout_seconds must be positive and finite.')
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='comparison-', dir=output_root))
    # Keep the exact contract; environment capture is restricted to named scientific
    # runtime controls to avoid recording credentials from the inherited environment.
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    environment = manifest.get('environment', {})
    allowed = {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES',
               'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_SCHEDULE', 'MKL_CBWR'}
    if not isinstance(environment, dict) or not all(k in allowed and isinstance(v, str) for k,v in environment.items()):
        raise ValueError('environment supports named scientific runtime controls only.')
    report = {'schema_version': 1, 'kind': 'numerical_comparison',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': digest(manifest_path), 'manifest': manifest,
              'environment': {k: environment.get(k, os.environ.get(k)) for k in sorted(allowed)},
              'environment_coverage': 'Inherited environment is not fully captured; this is not a hermetic runner.',
              'cases': [], 'passed': False,
              'limitations': 'Only declared observables and cases were compared. This does not establish physical equivalence. Build/revision labels are user supplied; executable/artifact hashes are measured. Timing is observational, not a controlled performance benchmark.'}
    try:
        for i, case in enumerate(manifest['cases']):
            entry = {'name': case['name'], 'passed': False}
            report['cases'].append(entry)
            inputs = (manifest_path.parent / case['input_dir']).resolve()
            if not inputs.is_dir():
                raise ValueError('input_dir must be a directory.')
            if output.is_relative_to(inputs):
                raise ValueError('Report output must be outside the input tree.')
            entry['input_sha256'] = tree_hashes(inputs)
            case_root = output / f'case-{i:03d}'
            case_root.mkdir()
            for variant in ('baseline', 'candidate'):
                entry[variant] = run_variant(manifest[variant], manifest_path.parent, inputs,
                                             case_root / variant, timeout, environment)
            if tree_hashes(inputs) != entry['input_sha256']:
                raise ValueError('Input tree changed during execution.')
            if any(entry[v]['returncode'] != 0 or entry[v]['timeout'] or entry[v]['changed_artifacts'] for v in ('baseline', 'candidate')):
                entry['error'] = 'Execution failed, timed out, or an executable/artifact changed.'
            else:
                entry.update(compare_observables(entry['baseline']['observables'], entry['candidate']['observables'], case['tolerances']))
        report['passed'] = all(c['passed'] for c in report['cases'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return output / 'report.json', report['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'reports')
    args = p.parse_args()
    report, passed = run_manifest(args.manifest, args.output)
    print(json.dumps({'report': str(report), 'passed': passed}))
    raise SystemExit(0 if passed else 1)
