"""Revision-bound, read-only access to a generated SIESTA knowledge index."""
from pathlib import Path
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import subprocess

SCHEMA_VERSION = '2'
LIMITATIONS = ('Doxygen references are not a complete call graph. Missing edges do not '
               'prove independence. Global state, aliasing, MPI/I/O effects and GPU '
               'suitability have not been analyzed.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_identity(root):
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': git('rev-parse', 'HEAD'),
            'dirty_status': git('status', '--porcelain', '--untracked-files=normal')}


class KnowledgeBase:
    def __init__(self, path):
        self.path = Path(path).resolve()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not version or version[0] != SCHEMA_VERSION:
            conn.close()
            raise ValueError('Legacy/unsupported index: rebuild with index_docs.py; original data is preserved.')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def metadata(self):
        with self.connect() as c:
            return dict(c.execute('SELECT key,value FROM meta'))

    def resolve(self, c, name):
        rows = c.execute('SELECT * FROM entities WHERE id=?', (name,)).fetchall()
        if not rows:
            rows = c.execute('SELECT * FROM entities WHERE qualname=? COLLATE NOCASE OR name=? COLLATE NOCASE ORDER BY qualname,id', (name, name)).fetchall()
        return rows

    def unique(self, c, name):
        rows = self.resolve(c, name)
        if len(rows) != 1:
            raise ValueError('Symbol must resolve uniquely; use an ID or qualified name. Candidates: ' +
                             json.dumps([{'id': r['id'], 'qualname': r['qualname']} for r in rows]))
        return rows[0]

    def entity(self, name):
        with self.connect() as c:
            rows = self.resolve(c, name)
            if not rows:
                rows = c.execute('SELECT * FROM entities WHERE instr(lower(qualname),lower(?))>0 ORDER BY qualname LIMIT 25', (name,)).fetchall()
            return {'entities': [dict(r) for r in rows], 'provenance': self.metadata(), 'limitations': LIMITATIONS}

    def search(self, query, limit=8):
        with self.connect() as c:
            try:
                rows = c.execute('SELECT e.* FROM fts JOIN entities e ON e.id=fts.entity_id WHERE fts MATCH ? ORDER BY rank LIMIT ?', (query, max(1, min(limit, 25)))).fetchall()
            except sqlite3.OperationalError as exc:
                raise ValueError(f'Invalid FTS query: {exc}') from exc
            return [dict(r) for r in rows]

    def health(self):
        meta = self.metadata()
        root = Path(meta['source_root'])
        stale, missing = [], []
        with self.connect() as c:
            for row in c.execute('SELECT path,sha256 FROM source_files ORDER BY path'):
                path = root / row['path']
                if not path.is_file():
                    missing.append(row['path'])
                elif digest(path) != row['sha256']:
                    stale.append(row['path'])
            counts = {t: c.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
                      for t in ('entities', 'edges', 'unresolved_references', 'source_files')}
        known = set()
        with self.connect() as c:
            known = {r[0] for r in c.execute('SELECT path FROM source_files')}
        added = [p.relative_to(root).as_posix() for p in sorted((root / 'Src').rglob('*'))
                 if p.is_file() and not p.is_symlink() and p.suffix.lower() in
                 ('.f', '.f90', '.f95', '.f03', '.f08', '.for', '.inc', '.h', '.c', '.cpp')
                 and p.relative_to(root).as_posix() not in known]
        current = git_identity(root)
        return {'metadata': meta, 'current_git': current,
                'revision_matches': current['commit'] == meta.get('source_commit') if current['commit'] else None,
                'changed_files': stale, 'missing_files': missing, 'counts': counts,
                'source_matches': not stale and not missing and not added,
                'ready_for_source_navigation': (not stale and not missing and not added and
                    meta['xml_source_alignment'] == 'fresh_generation_source_hashes_checked'),
                'added_files': added,
                'limitations': LIMITATIONS}

    def source(self, file, start, end):
        if start < 1 or end < start:
            raise ValueError('Require 1 <= start <= end.')
        meta = self.metadata()
        if meta['xml_source_alignment'] != 'fresh_generation_source_hashes_checked':
            raise ValueError('XML/source alignment is unverified; regenerate fresh XML before using indexed source ranges.')
        root = Path(meta['source_root']).resolve()
        supplied = Path(file)
        if supplied.is_absolute():
            try:
                file = supplied.resolve().relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError('Source path is outside the indexed worktree.') from exc
        with self.connect() as c:
            rows = c.execute('SELECT * FROM source_files WHERE path=?', (file,)).fetchall()
            if not rows and '/' not in file:
                rows = [r for r in c.execute('SELECT * FROM source_files') if Path(r['path']).name == file]
            if len(rows) != 1:
                raise ValueError('Source path missing or ambiguous; provide an indexed relative path.')
            row = rows[0]
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source path escapes the indexed worktree.')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('Source changed since indexing; rebuild before using indexed line ranges.')
        lines = content.decode(errors='replace').splitlines()
        if start > len(lines):
            raise ValueError('Start is beyond the end of the file.')
        end = min(end, start + 399, len(lines))
        return {'file': row['path'], 'sha256': row['sha256'], 'start': start, 'end': end,
                'xml_source_alignment': meta['xml_source_alignment'],
                'text': '\n'.join(f'{i}: {lines[i-1]}' for i in range(start, end+1))}

    def relations(self, name, reverse=False):
        with self.connect() as c:
            entity = self.unique(c, name)
            side, other = ('dst', 'src') if reverse else ('src', 'dst')
            rows = c.execute(f'SELECT e.*,x.kind AS relation,x.provenance FROM edges x JOIN entities e ON e.id=x.{other} WHERE x.{side}=? ORDER BY e.qualname', (entity['id'],)).fetchall()
            unresolved = c.execute('SELECT * FROM unresolved_references WHERE src=?', (entity['id'],)).fetchall()
            return {'entity': dict(entity), 'direction': 'incoming' if reverse else 'outgoing',
                    'references': [dict(r) for r in rows],
                    'unresolved_outgoing': [dict(r) for r in unresolved], 'provenance': self.metadata(), 'limitations': LIMITATIONS}

    def graph_path(self, source, target, max_depth=6):
        if not 0 <= max_depth <= 50:
            raise ValueError('max_depth must be between 0 and 50.')
        with self.connect() as c:
            s, t = self.unique(c, source), self.unique(c, target)
            queue, seen = deque([(s['id'], [s['id']])]), {s['id']}
            while queue:
                node, path = queue.popleft()
                if node == t['id']:
                    return {'path': [dict(c.execute('SELECT * FROM entities WHERE id=?', (i,)).fetchone()) for i in path], 'provenance': self.metadata(), 'limitations': LIMITATIONS}
                if len(path)-1 >= max_depth:
                    continue
                for row in c.execute('SELECT dst FROM edges WHERE src=? ORDER BY dst', (node,)):
                    if row[0] not in seen:
                        seen.add(row[0])
                        queue.append((row[0], path+[row[0]]))
        return {'path': None, 'max_depth': max_depth, 'provenance': self.metadata(), 'limitations': LIMITATIONS}

    def module(self, name):
        with self.connect() as c:
            rows = c.execute('SELECT * FROM entities WHERE parent IN (SELECT id FROM compounds WHERE name=? COLLATE NOCASE) ORDER BY qualname,id', (name,)).fetchall()
            return {'entities': [dict(r) for r in rows], 'provenance': self.metadata(), 'limitations': LIMITATIONS}

    def extraction_report(self, name):
        with self.connect() as c:
            rows = c.execute('SELECT id FROM entities WHERE parent IN (SELECT id FROM compounds WHERE name=? COLLATE NOCASE)', (name,)).fetchall()
            ids = {r[0] for r in rows}
            if not ids:
                ids = {self.unique(c, name)['id']}
            boundaries = []
            for r in c.execute('SELECT x.*,s.qualname AS source_name,d.qualname AS target_name FROM edges x JOIN entities s ON s.id=x.src JOIN entities d ON d.id=x.dst'):
                if (r['src'] in ids) != (r['dst'] in ids):
                    boundaries.append(dict(r))
            unresolved = [dict(r) for r in c.execute('SELECT * FROM unresolved_references') if r['src'] in ids]
        return {'component': name, 'entity_ids': sorted(ids), 'boundary_references': boundaries,
                'unresolved_outgoing': unresolved, 'unknown': ['global reads/writes', 'pointer aliasing',
                'allocation/ownership', 'MPI and I/O side effects', 'initialization order', 'GPU data lifetime'],
                'provenance': self.metadata(), 'limitations': LIMITATIONS}


def compare_indexes(baseline, candidate):
    def symbols(kb):
        with kb.connect() as c:
            return {r['id']: dict(r) for r in c.execute('SELECT * FROM entities')}
    a, b = symbols(baseline), symbols(candidate)
    def key(r):
        return (r['qualname'].lower(), r['kind'], r['file'])
    # IDs can change across revisions. Semantic keys may also be ambiguous;
    # refuse to collapse overloads into one entry.
    def grouped(rows):
        result = {}
        for r in rows.values():
            result.setdefault(key(r), []).append(r)
        return result
    ga, gb = grouped(a), grouped(b)
    common = ga.keys() & gb.keys()
    ambiguous = [list(k) for k in sorted(ga.keys() | gb.keys()) if len(ga.get(k, [])) > 1 or len(gb.get(k, [])) > 1]
    changed = []
    for k in sorted(common):
        if len(ga[k]) == len(gb[k]) == 1:
            left, right = ga[k][0], gb[k][0]
            fields = [f for f in ('signature', 'brief', 'detail', 'bodystart', 'bodyend') if left[f] != right[f]]
            if fields:
                changed.append({'symbol': list(k), 'changed_fields': fields})
    def file_hashes(kb):
        with kb.connect() as c:
            return dict(c.execute('SELECT path,sha256 FROM source_files'))
    fa, fb = file_hashes(baseline), file_hashes(candidate)
    def relationships(kb, groups):
        keys = {row['id']: key(row) for rows in groups.values() if len(rows) == 1 for row in rows}
        with kb.connect() as c:
            return {(keys[r['src']], keys[r['dst']], r['kind']) for r in c.execute('SELECT * FROM edges')
                    if r['src'] in keys and r['dst'] in keys}
    ea, eb = relationships(baseline, ga), relationships(candidate, gb)
    return {'baseline': baseline.metadata(), 'candidate': candidate.metadata(),
            'source_files_added': sorted(fb.keys()-fa.keys()),
            'source_files_removed': sorted(fa.keys()-fb.keys()),
            'source_files_changed': sorted(f for f in fa.keys() & fb.keys() if fa[f] != fb[f]),
            'references_added': sorted(eb-ea), 'references_removed': sorted(ea-eb),
            'added': [list(k) for k in sorted(gb.keys()-ga.keys())],
            'removed': [list(k) for k in sorted(ga.keys()-gb.keys())],
            'changed': changed, 'ambiguous_keys': ambiguous,
            'limitations': 'Structural metadata comparison only; unchanged metadata does not imply unchanged implementation. ' + LIMITATIONS}
