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
    messages: Annotated[list[BaseMessage], add_messages]


model = LLM(local=True).get_llm()

# A single event loop lives for the process's lifetime. ChatOllama (and the MCP
# client) lazily cache an async HTTP client tied to whichever loop is active on
# first use — calling asyncio.run() per chat turn would create/destroy a loop
# each time, leaving that cached client pointing at a dead loop on the next turn.
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

_repo_root = MCP_CONFIG_PATH.parent
_mcp_servers = json.loads(MCP_CONFIG_PATH.read_text())["servers"]
_mcp_client = MultiServerMCPClient(
    {
        name: {**cfg, "transport": "stdio", "cwd": _repo_root}
        for name, cfg in _mcp_servers.items()
    }
)
mcp_tools = _loop.run_until_complete(_mcp_client.get_tools())

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
    messages = state.get("messages")
    response = _loop.run_until_complete(agent.ainvoke({"messages": messages}))
    return {"messages": [response["messages"][-1]]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
