"""Merged-context tool (goals doc "Context engineering": deciding how much
invoice/customer history to include per query, summarizing long histories,
merging RAG results with structured DB lookups into one coherent context).

Without this, a subagent has to call sqlite_execute for history and
search_correspondence for notes/agreements separately, dump everything it
gets back into context uncapped, and stitch the two together itself. These
tools do both lookups in one call, cap/summarize the history instead of
listing every row, and return one formatted block covering all of an
entity's correspondence (a metadata filter on entity_id, not a similarity
search — there's no specific query here, we want everything on file for
that entity).
"""

import sqlite3
from pathlib import Path

from langchain_core.tools import tool

from agents.rag_tool import _store

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"
HISTORY_LIMIT = 5


def _resolve(conn: sqlite3.Connection, table: str, identifier: str) -> tuple[int, str] | None:
    identifier = str(identifier).strip()
    if identifier.isdigit():
        row = conn.execute(f"SELECT id, name FROM {table} WHERE id = ?", (int(identifier),)).fetchone()
    else:
        row = conn.execute(f"SELECT id, name FROM {table} WHERE name LIKE ?", (f"%{identifier}%",)).fetchone()
    return tuple(row) if row else None


def _correspondence_block(entity_type: str, entity_id: int) -> str:
    hits = _store.get(
        where={"$and": [{"entity_type": entity_type}, {"entity_id": entity_id}]},
        include=["documents", "metadatas"],
    )
    if not hits["ids"]:
        return "No correspondence on file."
    entries = [
        f"- {meta['title']} ({meta['date']}): {doc}"
        for doc, meta in zip(hits["documents"], hits["metadatas"])
    ]
    return "\n".join(entries)


@tool
def get_customer_context(customer: str) -> str:
    """Get a merged context block for a customer: invoice history (capped
    and summarized if long) plus all correspondence on file. customer may
    be an id or a name/partial name. Use this instead of separate
    sqlite_execute + search_correspondence calls when you need a rounded
    picture of a specific customer, not just one fact."""
    conn = sqlite3.connect(DB_PATH)
    found = _resolve(conn, "customers", customer)
    if found is None:
        conn.close()
        return f"No customer matching '{customer}'."
    customer_id, name = found

    rows = conn.execute(
        "SELECT id, amount, status, issued_date, due_date FROM invoices "
        "WHERE customer_id = ? ORDER BY issued_date DESC",
        (customer_id,),
    ).fetchall()
    conn.close()

    if not rows:
        history = "No invoices on record."
    else:
        overdue = [r for r in rows if r[2] == "overdue"]
        header = f"{len(rows)} invoice(s) total"
        if overdue:
            header += f", {len(overdue)} overdue totaling ${sum(r[1] for r in overdue):.2f}"
        if len(rows) > HISTORY_LIMIT:
            header += f" — showing {HISTORY_LIMIT} most recent"
        lines = [f"#{r[0]} ${r[1]:.2f} {r[2]} (issued {r[3]}, due {r[4]})" for r in rows[:HISTORY_LIMIT]]
        history = header + ":\n" + "\n".join(lines)

    correspondence = _correspondence_block("customer", customer_id)
    return (
        f"=== {name} (customer #{customer_id}) ===\n"
        f"Invoice history:\n{history}\n\n"
        f"Correspondence:\n{correspondence}"
    )


@tool
def get_supplier_context(supplier: str) -> str:
    """Get a merged context block for a supplier: purchase-order history
    (capped and summarized if long) plus all correspondence on file.
    supplier may be an id or a name/partial name. Use this instead of
    separate sqlite_execute + search_correspondence calls when you need a
    rounded picture of a specific supplier, not just one fact."""
    conn = sqlite3.connect(DB_PATH)
    found = _resolve(conn, "suppliers", supplier)
    if found is None:
        conn.close()
        return f"No supplier matching '{supplier}'."
    supplier_id, name = found

    rows = conn.execute(
        "SELECT id, item, quantity, status, order_date FROM purchase_orders "
        "WHERE supplier_id = ? ORDER BY order_date DESC",
        (supplier_id,),
    ).fetchall()
    conn.close()

    if not rows:
        history = "No purchase orders on record."
    else:
        pending = [r for r in rows if r[3] in ("pending", "in_transit")]
        header = f"{len(rows)} purchase order(s) total"
        if pending:
            header += f", {len(pending)} pending/in transit"
        if len(rows) > HISTORY_LIMIT:
            header += f" — showing {HISTORY_LIMIT} most recent"
        lines = [f"#{r[0]} {r[1]} x{r[2]} {r[3]} (ordered {r[4]})" for r in rows[:HISTORY_LIMIT]]
        history = header + ":\n" + "\n".join(lines)

    correspondence = _correspondence_block("supplier", supplier_id)
    return (
        f"=== {name} (supplier #{supplier_id}) ===\n"
        f"Purchase order history:\n{history}\n\n"
        f"Correspondence:\n{correspondence}"
    )
