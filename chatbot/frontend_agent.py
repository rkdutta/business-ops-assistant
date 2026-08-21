import streamlit as st
from langchain_core.messages import HumanMessage
from backend_agent import chatbot
import uuid


def reset_chat():
    st.session_state.thread_id = uuid.uuid4()
    st.session_state.messages_history = []
    add_thread(st.session_state.thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def load_conversation(thread_id):
    print(chatbot.get_state(config={"configurable": {"thread_id": thread_id}}))
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

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
    if st.sidebar.button(f"{tid}"):
        st.session_state.thread_id = tid
        messages = load_conversation(tid)
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state.messages_history = temp_messages



with st.chat_message("assistant"):
    st.markdown(
        "Hello! I am your assistant. How can I help you today?"
    )



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