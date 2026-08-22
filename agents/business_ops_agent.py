import asyncio
import contextvars
import json
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.rag_tool import search_correspondence
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

# Structured DB tools (MCP) + semantic search over correspondence (RAG).
all_tools = [*mcp_tools, search_correspondence]

# Three domain subagents. None declare their own `tools` — per SubAgent's
# contract, that means each inherits the main agent's tools (all_tools), so
# the same structured-data + correspondence-search tools back all of them;
# the domain split lives entirely in each system_prompt's table scope.
_TOOL_USAGE = (
    "For structured data (amounts, statuses, dates), call sqlite_get_catalog() "
    "first to see the schema, then sqlite_execute() with a read-only SQL query. "
    "For questions about agreed terms, past correspondence, or notes (e.g. "
    "payment terms, delivery arrangements, discounts agreed with someone), use "
    "search_correspondence instead. Only state facts that come from tool "
    "results — never invent data."
)

billing_agent: SubAgent = {
    "name": "billing_agent",
    "description": (
        "Handles invoices, payments, overdue tracking, and payment reminders. "
        "Delegate to this agent for anything about what a customer owes, has "
        "paid, or is late on."
    ),
    "system_prompt": (
        "You are the billing agent for a small business operations assistant. "
        "You handle invoices and payments using the invoices table (joined to "
        "customers on customer_id when a customer name is needed). "
        f"{_TOOL_USAGE}"
    ),
}

customer_agent: SubAgent = {
    "name": "customer_agent",
    "description": (
        "Handles customer lookups, history, and communication drafting. "
        "Delegate to this agent for anything about who a customer is, their "
        "contact details, or drafting a message to them."
    ),
    "system_prompt": (
        "You are the customer agent for a small business operations assistant. "
        "You handle customer information and communication drafting using the "
        f"customers table. {_TOOL_USAGE}"
    ),
}

supplier_agent: SubAgent = {
    "name": "supplier_agent",
    "description": (
        "Handles supplier info, purchase orders, and order/delivery tracking. "
        "Delegate to this agent for anything about a supplier or what's been "
        "ordered from them."
    ),
    "system_prompt": (
        "You are the supplier agent for a small business operations assistant. "
        "You handle supplier information and purchase orders using the suppliers "
        "and purchase_orders tables (joined on supplier_id). "
        f"{_TOOL_USAGE}"
    ),
}

# The main agent is the router: it classifies each request and delegates via
# the task() tool deepagents wires up automatically for `subagents`. It can
# call more than one subagent in a turn for multi-domain asks (e.g. "email
# all customers with overdue invoices" needs billing_agent then
# customer_agent), and answers directly itself for anything that isn't
# billing/customer/supplier-specific.
agent = create_deep_agent(
    model=model,
    tools=all_tools,
    subagents=[billing_agent, customer_agent, supplier_agent],
    system_prompt=(
        "You are the router for a business operations assistant. For each "
        "request, decide whether it belongs to billing_agent (invoices, "
        "payments, overdue tracking), customer_agent (customer lookups, "
        "history, communication drafting), or supplier_agent (supplier info, "
        "purchase orders). Delegate using the task tool. If a request spans "
        "multiple domains (e.g. 'email all customers with overdue invoices'), "
        "call the relevant subagents in sequence and combine their results. "
        "For anything general that doesn't fit those domains, you may use "
        "your own tools directly. Only state facts that come from tool "
        "results — never invent customer, invoice, or supplier details."
    ),
)


def chat_node(state: ChatbotState) -> ChatbotState:
    # The inner deep agent has its own multi-node graph (model <-> tools loop);
    # this just hands it the running history and takes its final answer back.
    messages = state.get("messages")
    # MCP tools only expose an async interface, so the agent must run via
    # ainvoke; run_until_complete bridges that into this graph's sync node.
    # LangGraph schedules chat_node inside a captured contextvars.Context, so
    # the inner agent would otherwise inherit the outer graph's checkpointer
    # through that context regardless of what config= is passed explicitly
    # here — the inner agent has no state of its own to persist across turns,
    # and SqliteSaver doesn't support the async calls it would receive. A
    # fresh empty Context breaks that inheritance.
    response = contextvars.Context().run(
        _loop.run_until_complete, agent.ainvoke({"messages": messages})
    )
    return {"messages": [response["messages"][-1]]}


# Persisted (not in-memory) so a separate process — the chat-naming controller
# in chat_name_controller_agent.py — can read the same thread history.
_checkpoint_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_checkpoint_conn)
checkpointer.setup()

# Outer graph: START -> chat_node -> END. Deliberately minimal — its only job
# is per-thread chat history + checkpointing around the deep agent in chat_node.
graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
