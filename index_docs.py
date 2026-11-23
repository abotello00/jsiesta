#!/usr/bin/env python3
"""Build a revision-bound index from Doxygen XML.

python index_docs.py XML_DIR OUTPUT_DIR --source-root WORKTREE
External XML is explicitly marked as unverified against the source snapshot.
Use rebuild_index.py for freshly generated XML and source alignment checking.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import sys
import xml.etree.ElementTree as ET

from knowledge import SCHEMA_VERSION, digest, git_identity


def text_of(element):
    return ' '.join(''.join(element.itertext()).split()) if element is not None else ''


def source_snapshot(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted((root / 'Src').rglob('*'))
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in
            ('.f', '.f90', '.f95', '.f03', '.f08', '.for', '.inc', '.h', '.c', '.cpp')}


def build(xml_dir, destination, source_root, alignment='unverified_external_xml', generator='external', exclude_xml=(), expected_snapshot=None, effective_config=None):
    root, xml = Path(source_root).resolve(), Path(xml_dir).resolve()
    if not (root / 'Src').is_dir():
        raise ValueError('source-root must contain Src/.')
    paths = sorted(p for p in xml.glob('*.xml') if p.name != 'index.xml' and p.name not in exclude_xml)
    unknown = set(exclude_xml) - {p.name for p in xml.glob('*.xml')}
    if unknown:
        raise ValueError(f'Unknown XML exclusions: {sorted(unknown)}')
    if not paths:
        raise ValueError('No Doxygen XML files found.')
    before = source_snapshot(root)
    if expected_snapshot is not None and before != expected_snapshot:
        raise ValueError('Source changed after XML generation; retry.')
    trees = [ET.parse(p) for p in paths]  # malformed XML fails the entire build
    compounds, entities, references = {}, {}, []
    priorities = {}

    def relative(file):
        if not file:
            return ''
        p = Path(file)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            # Doxygen paths are often relative to Docs/developer.
            candidates = [root / p, root / 'Docs' / 'developer' / p]
            resolved = next((q.resolve() for q in candidates if q.is_file()), candidates[0].resolve())
        if not resolved.is_relative_to(root):
            raise ValueError(f'XML source path outside selected worktree: {file}')
        return resolved.relative_to(root).as_posix()

    for tree in trees:
        for comp in tree.getroot().iter('compounddef'):
            cid, kind = comp.get('id'), comp.get('kind')
            if kind in ('dir', 'page', 'group'):
                continue
            name = comp.findtext('compoundname', '')
            compounds[cid] = (cid, kind, name, text_of(comp.find('briefdescription')), text_of(comp.find('detaileddescription')))
            location = comp.find('location')
            if kind != 'file':
                entities[cid] = (cid, kind, name.split('::')[-1], name, '', compounds[cid][3], compounds[cid][4],
                                 relative(location.get('file', '')) if location is not None else '',
                                 int(location.get('line', 0)) if location is not None else 0, 0, 0, name)
            for mem in comp.iter('memberdef'):
                mid, mkind = mem.get('id'), mem.get('kind')
                if mkind in ('define', 'directory'):
                    continue
                mname = mem.findtext('name', '')
                qualified = mem.findtext('qualifiedname', '') or (name+'::'+mname if kind != 'file' else mname)
                loc = mem.find('location')
                file = relative(loc.get('bodyfile', loc.get('file', ''))) if loc is not None else ''
                numbers = [int(loc.get(k, 0) or 0) if loc is not None else 0 for k in ('line', 'bodystart', 'bodyend')]
                definition = mem.findtext('definition', '') or qualified
                signature = definition + mem.findtext('argsstring', '')
                priority = 1 if kind != 'file' else 0
                if mid not in priorities or priority > priorities[mid]:
                    entities[mid] = (mid, mkind, mname, qualified, cid, text_of(mem.find('briefdescription')),
                                     text_of(mem.find('detaileddescription')), file, *numbers, signature)
                    priorities[mid] = priority
                for ref in mem.findall('references'):
                    references.append((mid, ref.get('refid', ''), text_of(ref)))
    if not entities:
        raise ValueError('XML contains no indexable entities.')
    names = defaultdict(set)
    for eid, row in entities.items():
        names[row[2].lower()].add(eid)
        names[row[3].lower()].add(eid)
    edges, unresolved = set(), set()
    for src, refid, label in references:
        if src not in entities:
            continue
        if refid:
            candidates = {refid} if refid in entities else set()
            provenance = 'doxygen_refid'
        else:
            candidates = names.get(label.lower(), set())
            provenance = 'doxygen_unique_name'
        if len(candidates) == 1:
            dst = next(iter(candidates))
            edges.add((src, dst, entities[src][3], entities[dst][3], 'references', provenance))
        else:
            unresolved.add((src, refid, label, 'ambiguous' if candidates else 'unresolved', json.dumps(sorted(candidates))))
    # Hash every referenced source too, including files outside Src/.
    hashes = dict(before)
    for row in entities.values():
        if row[7]:
            path = root / row[7]
            if not path.is_file():
                raise ValueError(f'Indexed source is missing: {path}')
            hashes[row[7]] = digest(path)
    if before != source_snapshot(root):
        raise ValueError('Source changed during indexing; retry with an idle worktree.')
    identity = git_identity(root)
    metadata = {'schema_version': SCHEMA_VERSION, 'source_root': str(root),
                'indexer_sha256': digest(__file__), 'python_version': sys.version,
                'source_commit': identity['commit'] or 'not-a-git-repository',
                'source_dirty_status': identity['dirty_status'] or '',
                'indexed_at': datetime.now(timezone.utc).isoformat(), 'xml_dir': str(xml),
                'xml_sha256': hashlib.sha256(''.join(p.name+':'+digest(p)+'\n' for p in paths).encode()).hexdigest(),
                'xml_source_alignment': alignment, 'generator': generator,
                'excluded_xml': json.dumps(sorted(exclude_xml)),
                'edge_coverage': 'partial_doxygen_references', 'n_unresolved_refs': str(len(unresolved))}
    if effective_config is not None:
        metadata['doxygen_config_sha256'] = hashlib.sha256(effective_config.encode()).hexdigest()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if effective_config is not None:
        (destination / 'doxygen-config.txt').write_text(effective_config)
    db = sqlite3.connect(destination / 'docs.db')
    try:
        db.executescript('''
        CREATE TABLE entities(id TEXT PRIMARY KEY,kind TEXT,name TEXT,qualname TEXT,parent TEXT,
            brief TEXT,detail TEXT,file TEXT,line INTEGER,bodystart INTEGER,bodyend INTEGER,signature TEXT);
        CREATE INDEX entity_name ON entities(name COLLATE NOCASE);
        CREATE INDEX entity_qualified ON entities(qualname COLLATE NOCASE);
        CREATE INDEX entity_parent ON entities(parent);
        CREATE TABLE compounds(id TEXT PRIMARY KEY,kind TEXT,name TEXT,brief TEXT,detail TEXT);
        CREATE TABLE source_files(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL);
        CREATE TABLE edges(src TEXT REFERENCES entities(id),dst TEXT REFERENCES entities(id),
            src_name TEXT,dst_name TEXT,kind TEXT,provenance TEXT,PRIMARY KEY(src,dst,kind));
        CREATE INDEX edges_dst ON edges(dst);
        CREATE TABLE unresolved_references(src TEXT,refid TEXT,label TEXT,reason TEXT,candidates TEXT);
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE VIRTUAL TABLE fts USING fts5(entity_id UNINDEXED,name,brief,detail,qualname,signature);
        ''')
        db.executemany('INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', sorted(entities.values()))
        db.executemany('INSERT INTO compounds VALUES (?,?,?,?,?)', sorted(compounds.values()))
        db.executemany('INSERT INTO edges VALUES (?,?,?,?,?,?)', sorted(edges))
        db.executemany('INSERT INTO unresolved_references VALUES (?,?,?,?,?)', sorted(unresolved))
        db.executemany('INSERT INTO source_files VALUES (?,?)', sorted(hashes.items()))
        db.executemany('INSERT INTO meta VALUES (?,?)', metadata.items())
        db.execute('INSERT INTO fts SELECT id,name,brief,detail,qualname,signature FROM entities')
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Index integrity check failed.')
        db.row_factory = sqlite3.Row
        nodes = [dict(r) for r in db.execute('SELECT * FROM entities ORDER BY id')]
        exported_edges = [dict(r) for r in db.execute('SELECT src AS source,dst AS target,kind,provenance FROM edges ORDER BY src,dst')]
        for name, content in [('graph_nodes.json', {'nodes': nodes}), ('graph_edges.json', {'edges': exported_edges})]:
            (destination / name).write_text(json.dumps({'metadata': metadata, **content}, indent=2)+'\n')
    finally:
        db.close()
    return metadata


def main(xml_dir, out_dir, source_root, alignment='unverified_external_xml', generator='external', exclude_xml=(), expected_snapshot=None, effective_config=None):
    """Publish an entire generation atomically; callers use OUT/current/docs.db."""
    out = Path(out_dir).resolve()
    generations = out / 'generations'
    generations.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.building-', dir=generations) as temporary:
        build(xml_dir, temporary, source_root, alignment, generator, exclude_xml, expected_snapshot, effective_config)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+Path(temporary).name.removeprefix('.building-')
        generation = generations / name
        os.replace(temporary, generation)
        link = out / ('.current-'+name)
        try:
            link.symlink_to(Path('generations') / name, target_is_directory=True)
            os.replace(link, out / 'current')
        finally:
            link.unlink(missing_ok=True)
    return generation / 'docs.db'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--exclude-xml', action='append', default=[], help='Explicit XML filename exclusion, recorded in metadata')
    args = parser.parse_args()
    print(main(args.xml_dir, args.out_dir, args.source_root, exclude_xml=args.exclude_xml))
