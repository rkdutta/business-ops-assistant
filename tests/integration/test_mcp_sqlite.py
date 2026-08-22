"""Exercises the real mcp-sqlite server via MultiServerMCPClient, pointed
at an isolated temp DB rather than data/business_ops.db — MCP tools are
async-only (see agents/business_ops_agent.py), so this drives them through
asyncio.run() rather than pulling in pytest-asyncio for one test file."""

import asyncio

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


@pytest.fixture
def mcp_tools(test_db_path):
    async def _get_tools():
        client = MultiServerMCPClient(
            {
                "business_ops_sqlite": {
                    "command": "uvx",
                    "args": ["--with", "mcp<2", "mcp-sqlite", str(test_db_path)],
                    "transport": "stdio",
                }
            }
        )
        return await client.get_tools()

    return asyncio.run(_get_tools())


def _find_tool(tools, name):
    return next(t for t in tools if t.name == name)


def _text(result) -> str:
    # format="content_and_artifact" tools return a list of content blocks
    # (e.g. [{"type": "text", "text": "..."}]), not a plain string.
    return result[0]["text"] if isinstance(result, list) else result


def test_catalog_reflects_the_real_schema(mcp_tools):
    catalog_tool = _find_tool(mcp_tools, "sqlite_get_catalog")
    result = _text(asyncio.run(catalog_tool.ainvoke({})))
    assert "invoices" in result
    assert "customers" in result


def test_execute_runs_a_real_query(mcp_tools):
    execute_tool = _find_tool(mcp_tools, "sqlite_execute")
    result = _text(
        asyncio.run(execute_tool.ainvoke({"sql": "SELECT name FROM customers WHERE id = 2"}))
    )
    assert "Blue Fern Cafe" in result
