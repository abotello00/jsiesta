import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

from index_docs import main
from knowledge import KnowledgeBase, compare_indexes
from validation import compare_observables, run_manifest


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'tree'
        (self.source / 'Src').mkdir(parents=True)
        (self.source / 'Src/a.f90').write_text('module a\ncontains\nsubroutine step(x)\nend subroutine\nend module\n')
        (self.source / 'Src/b.f90').write_text('module b\ncontains\nsubroutine step(x)\nend subroutine\nend module\n')
        self.xml = self.base / 'xml'
        self.xml.mkdir()
        self.out = self.base / 'index'
        for module in ('a', 'b'):
            refs = '<references refid="b_step">misleading_name</references><references>step</references><references refid="missing">absent</references>' if module == 'a' else ''
            (self.xml / f'{module}.xml').write_text(f'''<doxygen><compounddef id="{module}" kind="namespace">
                <compoundname>{module}</compoundname><location file="Src/{module}.f90" line="1"/>
                <memberdef id="{module}_step" kind="function"><name>step</name>
                <definition>subroutine {module}::step</definition><argsstring>(x)</argsstring>
                <briefdescription><para>Advance state</para></briefdescription>
                <location file="Src/{module}.f90" line="3" bodystart="3" bodyend="4"/>{refs}
                </memberdef></compounddef></doxygen>''')
        main(self.xml, self.out, self.source, alignment='fresh_generation_source_hashes_checked')
        self.kb = KnowledgeBase(self.out / 'current/docs.db')

    def test_signature_resolution_and_explicit_ambiguity(self):
        row = self.kb.entity('A::STEP')['entities'][0]
        self.assertEqual(row['signature'], 'subroutine a::step(x)')
        self.assertEqual(len(self.kb.entity('step')['entities']), 2)
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            self.kb.relations('step')
        result = self.kb.relations('a_step')
        self.assertEqual(result['references'][0]['id'], 'b_step')
        self.assertEqual({r['reason'] for r in result['unresolved_outgoing']}, {'ambiguous', 'unresolved'})
        self.assertEqual(result['references'][0]['relation'], 'references')
        self.assertTrue(self.kb.search('"a::step"'))

    def test_source_freshness_and_confinement(self):
        self.assertTrue(self.kb.health()['source_matches'])
        self.assertIn('subroutine step', self.kb.source('a.f90', 3, 4)['text'])
        (self.source / 'Src/a.f90').write_text('changed\n')
        self.assertFalse(self.kb.health()['source_matches'])
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.kb.source('a.f90', 1, 1)
        with self.assertRaises(ValueError):
            self.kb.source('/etc/passwd', 1, 2)
        with self.assertRaises(ValueError):
            self.kb.source('b.f90', 3, 1)

    def test_duplicate_source_basenames_and_missing_files(self):
        (self.source / 'Src/other').mkdir()
        (self.source / 'Src/other/a.f90').write_text('! second file\n')
        out = self.base / 'duplicate-files'
        main(self.xml, out, self.source, alignment='fresh_generation_source_hashes_checked')
        index = KnowledgeBase(out / 'current/docs.db')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            index.source('a.f90', 1, 1)
        self.assertTrue(index.source('Src/a.f90', 1, 1)['text'])
        (self.source / 'Src/b.f90').unlink()
        self.assertIn('Src/b.f90', index.health()['missing_files'])

    def test_depth_and_isolated_nodes(self):
        self.assertIsNone(self.kb.graph_path('a_step', 'b_step', 0)['path'])
        self.assertEqual(len(self.kb.graph_path('a_step', 'b_step', 1)['path']), 2)
        self.assertEqual(len(self.kb.graph_path('a', 'a', 0)['path']), 1)
        self.assertIsNone(self.kb.graph_path('b', 'a', 5)['path'])

    def test_publication_failure_preserves_active_index(self):
        previous = (self.out / 'current').resolve()
        (self.xml / 'bad.xml').write_text('<bad>')
        with self.assertRaises(Exception):
            main(self.xml, self.out, self.source)
        self.assertEqual((self.out / 'current').resolve(), previous)
        self.assertTrue(self.kb.entity('a_step')['entities'])

    def test_exports_and_component_report(self):
        path = self.out / 'current'
        nodes = json.loads((path / 'graph_nodes.json').read_text())
        edges = json.loads((path / 'graph_edges.json').read_text())
        ids = {n['id'] for n in nodes['nodes']}
        self.assertTrue(all(e['source'] in ids and e['target'] in ids for e in edges['edges']))
        self.assertEqual(nodes['metadata']['xml_source_alignment'], 'fresh_generation_source_hashes_checked')
        report = self.kb.extraction_report('a')
        self.assertEqual(len(report['boundary_references']), 1)
        self.assertIn('global reads/writes', report['unknown'])
        self.assertEqual(compare_indexes(self.kb, self.kb)['added'], [])

    def test_external_xml_cannot_claim_current_source_ranges(self):
        out = self.base / 'imported'
        main(self.xml, out, self.source)
        imported = KnowledgeBase(out / 'current/docs.db')
        self.assertFalse(imported.health()['ready_for_source_navigation'])
        with self.assertRaisesRegex(ValueError, 'unverified'):
            imported.source('a.f90', 1, 2)

    def test_new_source_and_structural_changes(self):
        (self.source / 'Src/new.f90').write_text('subroutine new()\nend subroutine\n')
        self.assertEqual(self.kb.health()['added_files'], ['Src/new.f90'])
        self.assertFalse(self.kb.health()['source_matches'])
        other = self.base / 'other'
        main(self.xml, other, self.source)
        diff = compare_indexes(self.kb, KnowledgeBase(other / 'current/docs.db'))
        self.assertEqual(diff['source_files_added'], ['Src/new.f90'])

    def test_legacy_index_rejected_without_mutation(self):
        path = self.base / 'old.db'
        with sqlite3.connect(path) as c:
            c.execute('create table meta(key,value)')
        c.close()
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            KnowledgeBase(path).metadata()
        self.assertEqual(path.read_bytes(), before)


class ValidationTests(unittest.TestCase):
    def test_contracts(self):
        tolerance = {'x': {'atol': 0.01, 'rtol': 0, 'unit': 'eV'}}
        self.assertTrue(compare_observables({'x': 1}, {'x': 1.001}, tolerance)['passed'])
        self.assertFalse(compare_observables({'x': 1}, {'x': 2}, tolerance)['passed'])
        for a, b in [({'x': float('nan')}, {'x': 1}), ({'x': True}, {'x': 1}), ({'x': []}, {'x': []}), ({'x': 1}, {'y': 1})]:
            with self.assertRaises(ValueError):
                compare_observables(a, b, tolerance)
        with self.assertRaises(ValueError):
            compare_observables({'x': 1}, {'x': 1}, {'x': {'atol': -1, 'rtol': 0, 'unit': 'eV'}})

    def test_runner_pass_fail_timeout_and_bad_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'inputs').mkdir()
            # A stale input result must not be accepted if an adapter writes nothing.
            (root / 'inputs/observables.json').write_text('{"x":1}')
            identity = {'build': 'test', 'revision': 'test'}
            script = root / 'adapter.py'
            script.write_text('from pathlib import Path\nPath("observables.json").write_text(\'{"x":1}\')\n')
            variant = {'command': [sys.executable, str(script)], 'artifacts': ['adapter.py'], 'identity': identity}
            manifest = {'schema_version': 1, 'timeout_seconds': 5,
                        'baseline': variant, 'candidate': dict(variant),
                        'cases': [{'name': 'test', 'input_dir': 'inputs', 'tolerances': {'x': {'atol': 0, 'rtol': 0, 'unit': 'eV'}}}]}
            path = root / 'manifest.json'
            def run():
                path.write_text(json.dumps(manifest))
                report, passed = run_manifest(path, root / 'reports')
                return json.loads(report.read_text()), passed
            report, passed = run()
            self.assertTrue(passed)
            self.assertIn(str(script.resolve()), report['cases'][0]['baseline']['artifacts_sha256'])
            manifest['candidate']['command'] = [sys.executable, '-c', 'from pathlib import Path; Path("observables.json").write_text(\'{"x":2}\')']
            self.assertFalse(run()[1])
            manifest['candidate']['command'] = [sys.executable, '-c', 'raise SystemExit(3)']
            self.assertFalse(run()[1])
            manifest['candidate']['command'] = [sys.executable, '-c', 'pass']
            self.assertIn('error', run()[0])
            manifest['candidate']['command'] = [sys.executable, '-c', 'import time; time.sleep(3)']
            manifest['timeout_seconds'] = 0.2
            report, passed = run()
            self.assertFalse(passed)
            self.assertTrue(report['cases'][0]['candidate']['timeout'])


if __name__ == '__main__':
    unittest.main()
