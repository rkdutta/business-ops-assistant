"""Unit tests for the human-in-the-loop gate around money/communication
tools, using GenericFakeChatModel instead of the real local Ollama model —
scripted responses make the tool-calling trajectory deterministic and free
to run. This builds a small standalone deep agent in each test rather than
importing agents.business_ops_agent, since that module's import triggers
the real MCP subprocess, a live Ollama connection, and a background event
loop thread — exactly what a unit test should avoid.
"""

from deepagents import create_deep_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.write_tools import create_invoice, mark_invoice_paid

CONFIG = {"configurable": {"thread_id": "test-thread"}}


class FakeToolCallingModel(GenericFakeChatModel):
    """GenericFakeChatModel.bind_tools() raises NotImplementedError by
    default (it has no notion of tools at all) — create_deep_agent calls
    bind_tools() internally regardless, so it has to be a no-op here. The
    fake's scripted `messages` iterator is what actually determines
    tool-calling behavior, independent of which tools were "bound"."""

    def bind_tools(self, tools, **kwargs):
        return self


def _agent(responses):
    return create_deep_agent(
        model=FakeToolCallingModel(messages=iter(responses)),
        tools=[create_invoice, mark_invoice_paid],
        interrupt_on={
            "create_invoice": {"allowed_decisions": ["approve", "reject"]},
            "mark_invoice_paid": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=InMemorySaver(),
    )


def test_create_invoice_pauses_for_approval_before_running(patched_db):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "create_invoice", "args": {"customer_id": 1, "amount": 250, "due_date": "2026-04-01"}, "id": "call_1"}],
    )
    agent = _agent([tool_call])

    agent.invoke({"messages": [HumanMessage(content="Invoice Acme for $250")]}, config=CONFIG)

    state = agent.get_state(CONFIG)
    assert len(state.tasks) == 1
    interrupts = state.tasks[0].interrupts
    assert len(interrupts) == 1
    action_requests = interrupts[0].value["action_requests"]
    assert action_requests[0]["name"] == "create_invoice"
    assert action_requests[0]["args"]["amount"] == 250


def test_approving_the_interrupt_runs_the_tool(patched_db, test_db_path):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "create_invoice", "args": {"customer_id": 1, "amount": 250, "due_date": "2026-04-01"}, "id": "call_1"}],
    )
    final = AIMessage(content="Done — invoice created.")
    agent = _agent([tool_call, final])

    agent.invoke({"messages": [HumanMessage(content="Invoice Acme for $250")]}, config=CONFIG)
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=CONFIG)

    assert result["messages"][-1].content == "Done — invoice created."

    import sqlite3
    conn = sqlite3.connect(test_db_path)
    row = conn.execute("SELECT amount FROM invoices WHERE customer_id = 1 AND amount = 250").fetchone()
    conn.close()
    assert row is not None


def test_rejecting_the_interrupt_skips_the_tool(patched_db, test_db_path):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "create_invoice", "args": {"customer_id": 1, "amount": 999, "due_date": "2026-04-01"}, "id": "call_1"}],
    )
    final = AIMessage(content="Okay, I won't create that invoice.")
    agent = _agent([tool_call, final])

    agent.invoke({"messages": [HumanMessage(content="Invoice Acme for $999")]}, config=CONFIG)
    agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config=CONFIG)

    import sqlite3
    conn = sqlite3.connect(test_db_path)
    row = conn.execute("SELECT amount FROM invoices WHERE amount = 999").fetchone()
    conn.close()
    assert row is None


def test_mark_invoice_paid_also_pauses_for_approval(patched_db):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "mark_invoice_paid", "args": {"invoice_id": 2}, "id": "call_1"}],
    )
    agent = _agent([tool_call])

    agent.invoke({"messages": [HumanMessage(content="Mark invoice 2 paid")]}, config=CONFIG)

    state = agent.get_state(CONFIG)
    action_requests = state.tasks[0].interrupts[0].value["action_requests"]
    assert action_requests[0]["name"] == "mark_invoice_paid"
    assert action_requests[0]["args"]["invoice_id"] == 2
