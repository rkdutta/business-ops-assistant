import asyncio
import json
from pathlib import Path
from typing import Annotated, TypedDict

from deepagents import create_deep_agent
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from models.llm import llm as LLM

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"
MCP_CONFIG_PATH = Path(__file__).parent.parent / "mcp.json"


class ChatbotState(TypedDict):
    # The outer graph's state: just the running chat history for this thread_id.
    messages: Annotated[list[BaseMessage], add_messages]

# get an instance of the model
model = LLM(local=True).get_llm()

# A single event loop lives for the process's lifetime. ChatOllama (and the MCP
# client) lazily cache an async HTTP client tied to whichever loop is active on
# first use — calling asyncio.run() per chat turn would create/destroy a loop
# each time, leaving that cached client pointing at a dead loop on the next turn.
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

# MCP servers are declared in mcp.json (repo root) rather than hardcoded here,
# so the server list can be edited/extended without touching this file.
_repo_root = MCP_CONFIG_PATH.parent
_mcp_servers = json.loads(MCP_CONFIG_PATH.read_text())["servers"]
_mcp_client = MultiServerMCPClient(
    {
        # cwd pins the subprocess's working directory to the repo root, so the
        # relative DB path in mcp.json resolves the same regardless of where
        # this app is launched from.
        name: {**cfg, "transport": "stdio", "cwd": _repo_root}
        for name, cfg in _mcp_servers.items()
    }
)
# Fetches the live tool list (schema, args) from the running MCP server once,
# at import time, rather than re-listing tools on every chat turn.
mcp_tools = _loop.run_until_complete(_mcp_client.get_tools())

# Deep agent gets the MCP server's tools directly — sqlite_get_catalog (schema)
# and sqlite_execute (read-only SQL) — instead of a hand-written query tool.
agent = create_deep_agent(
    model=model,
    tools=mcp_tools,
    system_prompt=(
        "You are a business operations assistant with access to a SQLite database "
        "via MCP tools. Call sqlite_get_catalog() first to see the available tables "
        "and columns, then use sqlite_execute() with a read-only SQL query to answer "
        "the user's question. Only state facts that come from tool results — never "
        "invent customer, invoice, or supplier details."
    ),
)


def chat_node(state: ChatbotState) -> ChatbotState:
    # The inner deep agent has its own multi-node graph (model <-> tools loop);
    # this just hands it the running history and takes its final answer back.
    messages = state.get("messages")
    # MCP tools only expose an async interface, so the agent must run via
    # ainvoke; run_until_complete bridges that into this graph's sync node.
    response = _loop.run_until_complete(agent.ainvoke({"messages": messages}))
    return {"messages": [response["messages"][-1]]}


checkpointer = InMemorySaver()

# Outer graph: START -> chat_node -> END. Deliberately minimal — its only job
# is per-thread chat history + checkpointing around the deep agent in chat_node.
graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
