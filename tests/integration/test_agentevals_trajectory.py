"""AgentEvals trajectory-match check against the real local model: does
asking for a customer summary actually result in a get_customer_context
call? "superset" mode means the real trajectory must contain at least the
reference tool calls (it's free to do additional exploration around them);
tool_args_match_mode="ignore" because the real model's exact phrasing of
the customer argument isn't the thing under test — the tool choice is."""

from agentevals.trajectory.match import create_trajectory_match_evaluator
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.context_tool import get_customer_context
from models.llm import llm as LLM

CONFIG = {"configurable": {"thread_id": "agentevals-test-thread"}}


def _agent():
    # Without a domain system_prompt, deepagents' auto-wired filesystem
    # tools (grep/glob/ls) distract a small local model into searching the
    # filesystem for "Blue Fern Cafe" instead of calling the one real tool
    # it has — this prompt is what the production customer_agent supplies
    # (see agents/business_ops_agent.py) and is load-bearing for this test
    # to be representative of real behavior, not just of what's possible.
    return create_deep_agent(
        model=LLM(local=True).get_llm(),
        tools=[get_customer_context],
        system_prompt=(
            "You are a customer information agent. For any question about a "
            "specific customer, call get_customer_context — never search "
            "the filesystem, there is no relevant data there."
        ),
    )


def test_customer_summary_request_calls_get_customer_context(patched_db, fake_store, require_ollama):
    agent = _agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content="Give me a summary of Blue Fern Cafe.")]},
        config=CONFIG,
    )

    reference_trajectory = [
        HumanMessage(content="Give me a summary of Blue Fern Cafe."),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_customer_context", "args": {"customer": "Blue Fern Cafe"}, "id": "call_1"}],
        ),
        ToolMessage(content="...", tool_call_id="call_1"),
        AIMessage(content="Here's a summary of Blue Fern Cafe."),
    ]

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    evaluation = evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)

    assert evaluation["score"], evaluation.get("comment")
