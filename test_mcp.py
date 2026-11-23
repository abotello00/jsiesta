#!/usr/bin/env python3
"""Assert a real stdio MCP session against a generated index (optional DB argument)."""
import asyncio
import json
from pathlib import Path
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def exercise(database):
    server = Path(__file__).resolve().with_name('siesta_mcp.py')
    params = StdioServerParameters(command=sys.executable, args=[str(server), str(database)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {'stats', 'get_entity', 'get_source', 'extraction_report', 'read_evidence'} <= names
            async def call(tool_name, **args):
                result = await session.call_tool(tool_name, args)
                assert not result.isError, result
                return json.loads(result.content[0].text)
            health = await call('stats')
            assert health['metadata']['schema_version'] == '2'
            assert health['counts']['entities'] > 0
            assert health['source_matches'], health
            # Search an indexed symbol, rather than relying on a particular SIESTA version.
            from knowledge import KnowledgeBase
            index = KnowledgeBase(database)
            with index.connect() as conn:
                symbol = dict(conn.execute("SELECT * FROM entities WHERE kind='function' AND bodystart>0 ORDER BY id LIMIT 1").fetchone())
            entity = await call('get_entity', name=symbol['id'])
            assert entity['entities'][0]['signature']
            if health['ready_for_source_navigation']:
                source = await call('get_source', file=symbol['file'], start=symbol['bodystart'], end=max(symbol['bodystart'], symbol['bodyend']))
                assert source['text']
            else:
                unverified = await session.call_tool('get_source', {'file': symbol['file'], 'start': 1, 'end': 2})
                assert unverified.isError
            isolated = await call('graph_path', from_name=symbol['id'], to_name=symbol['id'], max_depth=0)
            assert len(isolated['path']) == 1
            report = await call('extraction_report', component=symbol['id'])
            assert report['unknown']
            search = await call('search_docs', query='"'+symbol['name']+'"')
            assert search['results']
            for direction in ('get_callers', 'get_callees'):
                references = await call(direction, name=symbol['id'])
                assert 'limitations' in references
            overview = await call('module_overview', module='nonexistent_test_module')
            assert overview['entities'] == []
            comparison = await call('compare_index', baseline_db=str(Path(database).resolve()))
            assert comparison['added'] == comparison['removed'] == comparison['source_files_changed'] == []
            evidence = await call('list_evidence')
            if evidence['reports']:
                saved = await call('read_evidence', report=evidence['reports'][0])
                assert saved['schema_version'] == 1
            denied = await session.call_tool('read_evidence', {'report': '../docs.db'})
            assert denied.isError
            bad = await session.call_tool('get_source', {'file': symbol['file'], 'start': 0, 'end': 1})
            assert bad.isError
            print(json.dumps({'mcp_tools': sorted(names), 'counts': health['counts'], 'transport_checks': 'passed'}, indent=2))


if __name__ == '__main__':
    default = Path(__file__).resolve().parent / 'indexes/default/current/docs.db'
    asyncio.run(exercise(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
