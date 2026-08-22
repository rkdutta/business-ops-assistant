"""Long-term memory tools: business-wide preferences and per-customer/
supplier patterns the agent learns over time, persisted across sessions
(unlike the chat history, which is per-thread).

Backed by a memory_facts table in business_ops.db, created on first use.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL CHECK (scope IN ('global', 'customer', 'supplier')),
            entity_id INTEGER,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _resolve_entity_id(conn: sqlite3.Connection, scope: str, entity_name: str) -> int | None:
    table = "customers" if scope == "customer" else "suppliers"
    row = conn.execute(f"SELECT id FROM {table} WHERE name = ?", (entity_name,)).fetchone()
    return row[0] if row else None


@tool
def remember_fact(fact: str, scope: str = "global", entity_name: str | None = None) -> str:
    """Persist a lasting business preference or observed pattern for future
    conversations — not a one-off instruction for the current task. Examples:
    "always CC accounts@ on invoice emails" (scope="global"), or "pays late
    almost every cycle" for a specific customer (scope="customer",
    entity_name="Blue Fern Cafe"). scope must be "global", "customer", or
    "supplier"; entity_name is required unless scope is "global"."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)

    entity_id = None
    if scope != "global":
        if not entity_name:
            conn.close()
            return f"entity_name is required for scope='{scope}'."
        entity_id = _resolve_entity_id(conn, scope, entity_name)
        if entity_id is None:
            conn.close()
            return f"No {scope} named '{entity_name}' found — fact not saved."

    conn.execute(
        "INSERT INTO memory_facts (scope, entity_id, fact, created_at) VALUES (?, ?, ?, ?)",
        (scope, entity_id, fact, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return f"Remembered ({scope}{f': {entity_name}' if entity_name else ''}): {fact}"


@tool
def recall_facts(scope: str = "global", entity_name: str | None = None) -> str:
    """Retrieve remembered business preferences/patterns. Call with
    scope="global" to check standing business-wide rules (e.g. always CC
    accounts@ on invoice emails) before drafting emails or similar actions.
    Call with scope="customer"/"supplier" and entity_name to check patterns
    remembered about that specific customer/supplier (e.g. frequent late
    payer)."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)

    if scope == "global":
        rows = conn.execute(
            "SELECT fact FROM memory_facts WHERE scope = 'global' ORDER BY created_at"
        ).fetchall()
    else:
        if not entity_name:
            conn.close()
            return f"entity_name is required for scope='{scope}'."
        entity_id = _resolve_entity_id(conn, scope, entity_name)
        if entity_id is None:
            conn.close()
            return f"No {scope} named '{entity_name}' found."
        rows = conn.execute(
            "SELECT fact FROM memory_facts WHERE scope = ? AND entity_id = ? ORDER BY created_at",
            (scope, entity_id),
        ).fetchall()

    conn.close()
    if not rows:
        return "No remembered facts for this scope."
    return "\n".join(f"- {r[0]}" for r in rows)
