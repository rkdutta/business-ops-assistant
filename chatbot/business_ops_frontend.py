import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.business_ops_agent import chatbot

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"


def get_thread_title(thread_id) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    # chat_threads is created by chat_name_controller_agent.py, which may not
    # have run yet on a fresh DB — treat a missing table as "no title yet".
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_threads "
        "(thread_id TEXT PRIMARY KEY, title TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    row = conn.execute(
        "SELECT title FROM chat_threads WHERE thread_id = ?", (str(thread_id),)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def reset_chat():
    st.session_state.thread_id = uuid.uuid4()
    st.session_state.messages_history = []
    add_thread(st.session_state.thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def init_session_state():
    st.sidebar.title("Business Operations Assistant")
    st.sidebar.button("New Chat", on_click=reset_chat)
    st.sidebar.header("History")
    if "messages_history" not in st.session_state:
        st.session_state.messages_history = []

    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4()
    add_thread(st.session_state.thread_id)


init_session_state()

for tid in st.session_state.chat_threads:
    label = get_thread_title(tid) or f"New chat ({str(tid)[:8]})"
    if st.sidebar.button(label):
        st.session_state.thread_id = tid
        messages = load_conversation(tid)
        st.session_state.messages_history = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]


with st.chat_message("assistant"):
    st.markdown("Hello! I'm your business ops assistant. Ask me about a customer.")

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")
if user_input:
    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        ai_message = response.get("messages")[-1].content
        st.markdown(ai_message)
    st.session_state["messages_history"].append({"role": "assistant", "content": ai_message})
