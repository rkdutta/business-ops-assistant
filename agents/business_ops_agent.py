import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Annotated, TypedDict

import aiosqlite
from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agents.memory_tool import recall_facts, remember_fact
from agents.rag_tool import search_correspondence
from agents.write_tools import create_invoice, mark_invoice_paid, send_reminder_email
from models.llm import llm as LLM

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"
MCP_CONFIG_PATH = Path(__file__).parent.parent / "mcp.json"


class ChatbotState(TypedDict):
    # The outer graph's state: just the running chat history for this thread_id.
    messages: Annotated[list[BaseMessage], add_messages]

# get an instance of the model
model = LLM(local=True).get_llm()

# A single event loop lives for the process's lifetime, running continuously
# on a dedicated background thread. Two reasons it must actually run
# continuously rather than being pumped on demand per call (the original
# asyncio.new_event_loop() + run_until_complete()-per-call approach):
# 1. ChatOllama (and the MCP client) lazily cache an async HTTP client tied to
#    whichever loop is active on first use — a loop that gets created and
#    destroyed per call would leave that cached client pointing at a dead
#    loop on the next call.
# 2. AsyncSqliteSaver's sync methods (called internally by create_deep_agent
#    during graph construction) bridge back to async via
#    asyncio.run_coroutine_threadsafe(coro, loop).result() — which blocks
#    forever unless something is concurrently pumping that loop on another
#    thread. run_until_complete() only pumps it for the duration of the one
#    coroutine passed to it, so this reliably deadlocked at construction time
#    until the loop was moved onto its own always-running background thread.
_loop = asyncio.new_event_loop()


def _run_loop_forever() -> None:
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_run_loop_forever, daemon=True).start()


def _run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()

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
mcp_tools = _run_async(_mcp_client.get_tools())

# Structured DB tools (MCP) + semantic search over correspondence (RAG) +
# long-term memory (business preferences/patterns that persist across threads,
# unlike chat history which is per-thread) + money/communication write tools
# (gated by interrupt_on below — human approval required before they run).
all_tools = [
    *mcp_tools,
    search_correspondence,
    remember_fact,
    recall_facts,
    create_invoice,
    mark_invoice_paid,
    send_reminder_email,
]

# Money-related and outbound-communication actions pause for approval before
# executing. Declared on the top-level agent so it's inherited by every
# subagent automatically (verified: an interrupt raised inside a subagent's
# tool call — invoked via task() — correctly propagates up and pauses this
# whole graph, not just the subagent). Requires a checkpointer (below) since
# pausing/resuming relies on persisted graph state.
WRITE_TOOL_NAMES = {"create_invoice", "mark_invoice_paid", "send_reminder_email"}
interrupt_on = {name: {"allowed_decisions": ["approve", "reject"]} for name in WRITE_TOOL_NAMES}

# Persisted (not in-memory) so a separate process — the chat-naming controller
# in chat_name_controller_agent.py — can read the same thread history, and so
# a paused (interrupted) agent can be resumed correctly.
#
# Two separate checkpointer instances, two separate DB files (same thread_id
# reused across both, but checkpoint_ns isn't a general namespacing knob for
# a top-level graph — get_state()/aget_state() treat any non-empty value as
# a named-subgraph lookup and error since `agent` isn't literally a subgraph
# of anything, so sharing one file with a custom ns doesn't work; separate
# files sidesteps that entirely):
# - The outer graph (below) has no async tools, is invoked synchronously from
#   Streamlit, and uses the sync SqliteSaver, backed by business_ops.db.
# - The inner deep agent has async-only MCP tools, so it's invoked via
#   ainvoke — which SqliteSaver can't support (aget_tuple etc. are sync-only)
#   — so it needs AsyncSqliteSaver instead, backed by an aiosqlite connection
#   to its own file.
_checkpoint_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_checkpoint_conn)
checkpointer.setup()
INNER_CHECKPOINT_DB_PATH = Path(__file__).parent.parent / "data" / "deep_agent_checkpoints.db"


async def _setup_inner_checkpointer() -> AsyncSqliteSaver:
    conn = await aiosqlite.connect(INNER_CHECKPOINT_DB_PATH)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


inner_checkpointer = _run_async(_setup_inner_checkpointer())

# Three domain subagents. None declare their own `tools` — per SubAgent's
# contract, that means each inherits the main agent's tools (all_tools), so
# the same structured-data + correspondence-search tools back all of them;
# the domain split lives entirely in each system_prompt's table scope.
_TOOL_USAGE = (
    "For structured data (amounts, statuses, dates), call sqlite_get_catalog() "
    "first to see the schema, then sqlite_execute() with a read-only SQL query. "
    "For questions about agreed terms, past correspondence, or notes (e.g. "
    "payment terms, delivery arrangements, discounts agreed with someone), use "
    "search_correspondence instead. Before drafting a communication or acting "
    "on a specific customer/supplier, call recall_facts (scope='global', and "
    "scope='customer'/'supplier' with entity_name for that specific one) to "
    "check standing preferences and known patterns — e.g. always CC accounts@ "
    "on invoice emails, or a customer being a frequent late payer. When the "
    "user states a new lasting preference or you notice a recurring pattern "
    "(not a one-off instruction), call remember_fact to save it for future "
    "conversations. Only state facts that come from tool results — never "
    "invent data."
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
    interrupt_on=interrupt_on,
    checkpointer=inner_checkpointer,
    system_prompt=(
        "You are the router for a business operations assistant. For each "
        "request, decide whether it belongs to billing_agent (invoices, "
        "payments, overdue tracking), customer_agent (customer lookups, "
        "history, communication drafting), or supplier_agent (supplier info, "
        "purchase orders). Delegate using the task tool. If a request spans "
        "multiple domains (e.g. 'email all customers with overdue invoices'), "
        "call the relevant subagents in sequence and combine their results. "
        "For anything general that doesn't fit those domains, you may use "
        "your own tools directly. create_invoice, mark_invoice_paid, and "
        "send_reminder_email will pause for the user's approval automatically "
        "— call them directly when appropriate, don't ask for confirmation "
        "yourself first. Only state facts that come from tool results — "
        "never invent customer, invoice, or supplier details."
    ),
)


def _inner_config(thread_id) -> dict:
    # Same thread_id as the outer graph, but a separate checkpointer/file
    # (inner_checkpointer / INNER_CHECKPOINT_DB_PATH above) — no collision
    # to avoid, so the default empty checkpoint_ns is used, same as any
    # normal top-level graph invocation.
    return {"configurable": {"thread_id": str(thread_id)}}


def _run_inner(coro):
    # Runs on the dedicated background loop thread (see _run_async above),
    # which is itself already enough to stop the inner agent from inheriting
    # the outer graph's checkpointer via ambient contextvars propagation —
    # that propagation only happens within a single thread's call stack, and
    # this coroutine executes on a different thread entirely.
    return _run_async(coro)


def chat_node(state: ChatbotState, config: RunnableConfig) -> ChatbotState:
    # The inner deep agent has its own multi-node graph (model <-> tools loop)
    # and, now, its own checkpointer — so only the newest message needs to be
    # sent in; the inner agent's persisted history (keyed by the same
    # thread_id) already has everything before it.
    thread_id = config["configurable"]["thread_id"]
    new_message = state["messages"][-1]

    response = _run_inner(
        agent.ainvoke({"messages": [new_message]}, config=_inner_config(thread_id))
    )

    if "__interrupt__" in response:
        action_requests = response["__interrupt__"][0].value["action_requests"]
        summary = "; ".join(
            f"{a['name']}({', '.join(f'{k}={v!r}' for k, v in a['args'].items())})"
            for a in action_requests
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"⏸️ Waiting for your approval before: {summary}. "
                        "Approve or reject in the sidebar to continue."
                    )
                )
            ]
        }

    return {"messages": [response["messages"][-1]]}


def get_pending_approval(thread_id) -> dict | None:
    """Returns the pending action_requests for this thread if the inner agent
    is currently paused on an interrupt, else None."""
    state = _run_inner(agent.aget_state(_inner_config(thread_id)))
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def resume_pending_action(thread_id, decision: str, message: str | None = None) -> str:
    """Resumes a paused thread with an approve/reject decision for every
    pending action request, and appends the result to the outer graph's
    checkpointed history too, so the chat transcript stays coherent."""
    pending = get_pending_approval(thread_id)
    count = len(pending["action_requests"]) if pending else 1
    single_decision = {"type": decision, **({"message": message} if message else {})}

    response = _run_inner(
        agent.ainvoke(
            Command(resume={"decisions": [dict(single_decision) for _ in range(count)]}),
            config=_inner_config(thread_id),
        )
    )
    final_message = response["messages"][-1]
    chatbot.update_state({"configurable": {"thread_id": str(thread_id)}}, {"messages": [final_message]})
    return final_message.content


# Outer graph: START -> chat_node -> END. Deliberately minimal — its only job
# is per-thread chat history + checkpointing around the deep agent in chat_node.
graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
