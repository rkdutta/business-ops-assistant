import asyncio
import json
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Annotated, TypedDict

import aiosqlite
from deepagents import FilesystemPermission, SubAgent, create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agents.context_tool import get_customer_context, get_supplier_context
from agents.logging_utils import scrub_args, tool_call_logger
from agents.memory_tool import recall_facts, remember_fact
from agents.rag_tool import search_correspondence
from agents.sandbox_tool import run_analysis_script
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
    get_customer_context,
    get_supplier_context,
    run_analysis_script,
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
    "search_correspondence instead. For computation over data — totals by "
    "month, aggregates, trends, anything a CSV-style analysis script would "
    "do — use run_analysis_script instead of sqlite_execute; it runs "
    "against an isolated snapshot rather than the live database. When you "
    "need a rounded picture of one "
    "specific customer or supplier — history plus correspondence together, "
    "not just a single fact — call get_customer_context or "
    "get_supplier_context instead of separate sqlite_execute and "
    "search_correspondence calls; it returns both already merged, with long "
    "histories capped and summarized rather than dumped in full. Before drafting a communication or acting "
    "on a specific customer/supplier, call recall_facts (scope='global', and "
    "scope='customer'/'supplier' with entity_name for that specific one) to "
    "check standing preferences and known patterns — e.g. always CC accounts@ "
    "on invoice emails, or a customer being a frequent late payer. When the "
    "user states a new lasting preference or you notice a recurring pattern "
    "(not a one-off instruction), call remember_fact to save it for future "
    "conversations. Only state facts — especially financial figures like "
    "amounts, balances, and totals — that come from tool results. Never "
    "invent or estimate a number the tools didn't return."
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

# Filesystem backend for the skills library, scoped to skills/ only (not the
# repo root) — even though FilesystemBackend blocks path traversal outside
# root_dir regardless, pointing root_dir at the whole repo would still hand
# the agent's filesystem tools (read_file/write_file/ls/...) access to
# business_ops.db and source code for no reason. Scoping root_dir itself to
# skills/ means there's nothing outside skills/ to reach in the first place.
_skills_backend = FilesystemBackend(root_dir=_repo_root / "skills")
_skills_readonly = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]

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
    # customer-summary skill: prompt + get_customer_context (merged
    # get_customer_details + RAG over correspondence) for "summarize this
    # customer"-style requests. See skills/customer-summary/SKILL.md.
    "skills": [("/", "Business Ops")],
    "permissions": _skills_readonly,
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
    backend=_skills_backend,
    permissions=_skills_readonly,
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
        "yourself first. Only state facts — especially financial figures "
        "like amounts, balances, and totals — that come from tool results. "
        "Never invent or estimate a number, or any customer, invoice, or "
        "supplier detail, that the tools didn't return."
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


# Friendly progress labels per tool, for event streaming. Falls back to a
# generic "Calling X..." for anything not listed here (e.g. new tools added
# later don't need this table updated to still show *something*).
_PROGRESS_LABELS = {
    "task": lambda args: f"Delegating to {args.get('subagent_type', 'a specialist agent')}...",
    "sqlite_get_catalog": lambda args: "Checking the database schema...",
    "sqlite_execute": lambda args: "Querying the database...",
    "search_correspondence": lambda args: f"Searching past correspondence for \"{args.get('query', '')}\"...",
    "get_customer_context": lambda args: f"Pulling full context for {args.get('customer', 'customer')}...",
    "get_supplier_context": lambda args: f"Pulling full context for {args.get('supplier', 'supplier')}...",
    "run_analysis_script": lambda args: f"Running analysis on {', '.join(args.get('tables', []))} in an isolated sandbox...",
    "read_file": lambda args: f"Reading {args.get('file_path', 'a file')}...",
    "recall_facts": lambda args: "Checking remembered preferences...",
    "remember_fact": lambda args: "Saving a new preference...",
    "create_invoice": lambda args: "Preparing to create an invoice...",
    "mark_invoice_paid": lambda args: f"Preparing to mark invoice #{args.get('invoice_id')} as paid...",
    "send_reminder_email": lambda args: "Drafting a reminder email...",
}


def _progress_label(tool_name: str, args: dict) -> str:
    fn = _PROGRESS_LABELS.get(tool_name)
    if fn is None:
        return f"Calling {tool_name}..."
    try:
        return fn(args)
    except Exception:
        return f"Calling {tool_name}..."


async def _stream_inner_run(thread_id, new_message, on_progress):
    """Drains the inner agent's tool-call events (for progress narration) and
    returns its final StateSnapshot. astream_events' own on_chain_end output
    doesn't reliably surface a pending interrupt the way ainvoke's return
    value does, so the definitive final state comes from a follow-up
    aget_state call once the run has actually finished."""
    inner_config = _inner_config(thread_id)
    async for event in agent.astream_events({"messages": [new_message]}, config=inner_config, version="v2"):
        if event["event"] == "on_tool_start":
            args = event["data"].get("input", {})
            # Scrubbed before it ever reaches the logger — the UI-facing
            # progress label below is built from the unscrubbed args, since
            # that's shown only to this session's own user, not persisted.
            tool_call_logger.info("%s(%s)", event["name"], scrub_args(args))
            on_progress(_progress_label(event["name"], args))
    return await agent.aget_state(inner_config)


def chat_node(state: ChatbotState, config: RunnableConfig) -> ChatbotState:
    # The inner deep agent has its own multi-node graph (model <-> tools loop)
    # and, now, its own checkpointer — so only the newest message needs to be
    # sent in; the inner agent's persisted history (keyed by the same
    # thread_id) already has everything before it.
    thread_id = config["configurable"]["thread_id"]
    new_message = state["messages"][-1]

    # get_stream_writer() must be called on THIS (the outer graph's) thread —
    # it reads LangGraph's streaming context, which doesn't cross the thread
    # boundary into the background loop thread the inner run executes on. A
    # plain thread-safe queue.Queue bridges progress messages back here so
    # they can be hand-delivered to the writer from the correct thread.
    writer = get_stream_writer()
    progress_q: queue.Queue = queue.Queue()
    SENTINEL = object()

    async def runner():
        try:
            snapshot = await _stream_inner_run(
                thread_id, new_message, lambda msg: progress_q.put(msg)
            )
            progress_q.put((SENTINEL, snapshot, None))
        except Exception as e:  # noqa: BLE001 — re-raised on the calling thread below
            progress_q.put((SENTINEL, None, e))

    future = asyncio.run_coroutine_threadsafe(runner(), _loop)

    while True:
        item = progress_q.get()
        if isinstance(item, tuple) and item[0] is SENTINEL:
            _, snapshot, error = item
            break
        writer(item)

    future.result()  # propagate any unexpected exception from runner()
    if error is not None:
        raise error

    for task in snapshot.tasks:
        if task.interrupts:
            action_requests = task.interrupts[0].value["action_requests"]
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

    return {"messages": [snapshot.values["messages"][-1]]}


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
