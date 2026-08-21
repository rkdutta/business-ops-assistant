from datetime import datetime
import os
from ddgs import DDGS
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from models.llm import llm as LLM
import uuid
import sqlite3

unique_id = uuid.uuid4()

os.environ.setdefault("AWS_REGION", "eu-west-1")

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model =  LLM(local=True).get_llm()

@tool
def internet_search(query: str, max_results: int = 5) -> str:
    """Search the internet for a given query and return the top results."""
    results = DDGS().text(query, max_results=max_results)
    return "\n\n".join(
        f"{r['title']}\n{r['href']}\n{r['body']}" for r in results
    )

@tool
def get_current_date() -> str:
    """Get today's date and day of the week."""
    now = datetime.now()
    return now.strftime("%A, %Y-%m-%d")

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return internet_search.invoke({"query": f"current weather in {city}"})


# Deep agent wired up with a single tool and a basic system prompt
agent = create_deep_agent(
    model=model,
    tools=[get_weather,get_current_date,internet_search],
    system_prompt=(
        "You are a helpful weather assistant. "
    ),
)

def chat_node(state: ChatbotState) -> ChatbotState:
    messages = state.get("messages")
    response = agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

checkpointer = InMemorySaver()
# conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# thread_id = 'thread_1'

# while True:
#     # user_message = input("Type here: ")

#     if user_message.lower() in ["exit", "quit", "bye"]:
#         break

#     config = {"configurable":{ "thread_id": thread_id }}
#     response = chatbot.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)
#     print("\n\n AI:", response.get("messages")[-1].content)