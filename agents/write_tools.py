"""Money/communication write tools. These are the tools human-in-the-loop
gates in business_ops_agent.py — by the time one of these bodies actually
runs, the user has already approved it.

send_reminder_email is simulated (no real SMTP/email API integration) — it
just logs the "sent" email to a table, consistent with the project's
fake/sample-data-only scope.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"


def _ensure_sent_emails_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sent_emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            sent_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


@tool
def create_invoice(customer_id: int, amount: float, due_date: str) -> str:
    """Create a new invoice for a customer. amount must be positive.
    due_date should be an ISO date string (YYYY-MM-DD)."""
    if amount <= 0:
        return "Invoice amount must be positive — not created."
    conn = sqlite3.connect(DB_PATH)
    customer = conn.execute(
        "SELECT name FROM customers WHERE id = ?", (customer_id,)
    ).fetchone()
    if customer is None:
        conn.close()
        return f"No customer with id {customer_id} — invoice not created."
    issued_date = datetime.now(timezone.utc).date().isoformat()
    cur = conn.execute(
        "INSERT INTO invoices (customer_id, amount, status, issued_date, due_date) "
        "VALUES (?, ?, 'pending', ?, ?)",
        (customer_id, amount, issued_date, due_date),
    )
    invoice_id = cur.lastrowid
    conn.commit()
    conn.close()
    return (
        f"Created invoice #{invoice_id} for {customer[0]}: ${amount:.2f}, "
        f"due {due_date}."
    )


@tool
def mark_invoice_paid(invoice_id: int) -> str:
    """Mark an existing invoice as paid."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT customer_id, amount, status FROM invoices WHERE id = ?", (invoice_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return f"No invoice with id {invoice_id}."
    if row[2] == "paid":
        conn.close()
        return f"Invoice #{invoice_id} was already marked paid."
    conn.execute("UPDATE invoices SET status = 'paid' WHERE id = ?", (invoice_id,))
    conn.commit()
    conn.close()
    return f"Invoice #{invoice_id} (${row[1]:.2f}) marked as paid."


@tool
def send_reminder_email(customer_id: int, subject: str, body: str) -> str:
    """Send a payment reminder (or other) email to a customer. Simulated —
    no real email is sent; it's logged as sent for this demo."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_sent_emails_table(conn)
    customer = conn.execute(
        "SELECT name, email FROM customers WHERE id = ?", (customer_id,)
    ).fetchone()
    if customer is None:
        conn.close()
        return f"No customer with id {customer_id} — email not sent."
    conn.execute(
        "INSERT INTO sent_emails (customer_id, subject, body, sent_at) VALUES (?, ?, ?, ?)",
        (customer_id, subject, body, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return f"Email sent to {customer[0]} <{customer[1]}>: \"{subject}\""
