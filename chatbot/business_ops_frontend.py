import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.business_ops_agent import chatbot, get_pending_approval, resume_pending_action

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

pending = get_pending_approval(st.session_state.thread_id)
if pending:
    st.sidebar.header("Pending Approval")
    for action in pending["action_requests"]:
        st.sidebar.markdown(f"**{action['name']}**")
        st.sidebar.json(action["args"])

    # One decision applies to every pending action_request in this batch —
    # resume_pending_action() applies it to all of them together.
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Approve"):
        result = resume_pending_action(st.session_state.thread_id, "approve")
        st.session_state["messages_history"].append({"role": "assistant", "content": result})
        st.rerun()
    if col2.button("Reject"):
        result = resume_pending_action(st.session_state.thread_id, "reject")
        st.session_state["messages_history"].append({"role": "assistant", "content": result})
        st.rerun()


with st.chat_message("assistant"):
    st.markdown("Hello! I'm your business ops assistant. Ask me about a customer.")

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Resolve the pending approval above before continuing..." if pending else "Type here...",
    disabled=bool(pending),
)
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
    if get_pending_approval(st.session_state.thread_id):
        st.rerun()
