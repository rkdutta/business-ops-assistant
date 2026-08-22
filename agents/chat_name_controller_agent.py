"""Standalone process that titles chat threads based on their content.

Runs independently of the Streamlit app. Polls business_ops.db's checkpoint
table for threads that have at least one full exchange but no title yet,
asks the LLM for a short descriptive title, and writes it to the
chat_threads table — which chatbot/business_ops_frontend.py reads to label
the sidebar instead of showing raw thread_id UUIDs.

Run with: .venv/bin/python agents/chat_name_controller_agent.py
"""

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from models.llm import llm as LLM

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"
POLL_INTERVAL_SECONDS = 10
MIN_MESSAGES_TO_TITLE = 2  # at least one user message + one reply

model = LLM(local=True).get_llm()


def ensure_titles_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_untitled_thread_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT c.thread_id
        FROM checkpoints c
        LEFT JOIN chat_threads t ON t.thread_id = c.thread_id
        WHERE t.thread_id IS NULL
        """
    ).fetchall()
    return [r[0] for r in rows]


def generate_title(saver: SqliteSaver, thread_id: str) -> str | None:
    tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if tup is None:
        return None
    messages = tup.checkpoint.get("channel_values", {}).get("messages", [])
    if len(messages) < MIN_MESSAGES_TO_TITLE:
        return None

    convo = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages[:4]
    )
    prompt = (
        "Give a short, descriptive title (3-6 words, no quotes, no trailing "
        "punctuation) summarizing what this conversation is about:\n\n"
        f"{convo}"
    )
    response = model.invoke(prompt)
    return response.content.strip().strip('"')


def run_once(conn: sqlite3.Connection, saver: SqliteSaver) -> None:
    for thread_id in get_untitled_thread_ids(conn):
        title = generate_title(saver, thread_id)
        if title is None:
            continue
        conn.execute(
            "INSERT INTO chat_threads (thread_id, title, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title, "
            "updated_at = excluded.updated_at",
            (thread_id, title, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        print(f"Titled {thread_id}: {title}")


def main() -> None:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    ensure_titles_table(conn)
    saver = SqliteSaver(conn)
    saver.setup()
    print(f"Chat name controller watching {DB_PATH} (every {POLL_INTERVAL_SECONDS}s)")
    while True:
        run_once(conn, saver)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
