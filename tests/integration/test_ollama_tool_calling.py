"""Same HITL interrupt/approve/reject behavior as
tests/test_hitl_fake_model.py, but driven by the real local Ollama model
instead of a scripted fake — this is what actually proves the model
reliably produces a valid tool call for these prompts, not just that our
graph wiring handles a tool call correctly once one exists."""

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.write_tools import create_invoice, mark_invoice_paid
from models.llm import llm as LLM

CONFIG = {"configurable": {"thread_id": "integration-test-thread"}}


def _agent():
    return create_deep_agent(
        model=LLM(local=True).get_llm(),
        tools=[create_invoice, mark_invoice_paid],
        interrupt_on={
            "create_invoice": {"allowed_decisions": ["approve", "reject"]},
            "mark_invoice_paid": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=InMemorySaver(),
    )


def test_real_model_pauses_before_marking_invoice_paid(patched_db, require_ollama):
    agent = _agent()
    agent.invoke(
        {"messages": [HumanMessage(content="Mark invoice 2 as paid.")]},
        config=CONFIG,
    )

    state = agent.get_state(CONFIG)
    assert len(state.tasks) == 1, "model did not call a gated tool"
    action_requests = state.tasks[0].interrupts[0].value["action_requests"]
    assert action_requests[0]["name"] == "mark_invoice_paid"
    assert action_requests[0]["args"]["invoice_id"] == 2


def test_real_model_flow_completes_after_approval(patched_db, test_db_path, require_ollama):
    agent = _agent()
    agent.invoke(
        {"messages": [HumanMessage(content="Mark invoice 2 as paid.")]},
        config=CONFIG,
    )
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=CONFIG)

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content

    import sqlite3
    conn = sqlite3.connect(test_db_path)
    status = conn.execute("SELECT status FROM invoices WHERE id = 2").fetchone()[0]
    conn.close()
    assert status == "paid"
