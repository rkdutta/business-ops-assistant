"""Shared fixtures for unit tests.

The write/context/memory/sandbox tools each hardcode a module-level DB_PATH
pointing at data/business_ops.db, since there's no dependency-injection
layer for it (and adding one just for tests would be more machinery than
this project needs). `patched_db` gives each test an isolated temp SQLite
file with the same schema instead, and monkeypatches DB_PATH on every tool
module so calling the tools under test never touches the real database.
"""

import sqlite3

import pytest

from agents import context_tool, memory_tool, sandbox_tool, write_tools


def pytest_collection_modifyitems(items):
    # A conftest.py's own `pytestmark` does NOT auto-apply to every test
    # under its directory (that only works inside an actual test module) —
    # this hook is the correct way to mark everything under tests/integration/
    # from one place instead of repeating `pytestmark` in each file there.
    for item in items:
        if "/integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    notes TEXT
);
CREATE TABLE invoices (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    amount REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'overdue', 'pending')),
    issued_date TEXT NOT NULL,
    due_date TEXT NOT NULL
);
CREATE TABLE suppliers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    notes TEXT
);
CREATE TABLE purchase_orders (
    id INTEGER PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    item TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_transit', 'delivered')),
    order_date TEXT NOT NULL
);
"""

CUSTOMERS = [
    (1, "Acme Roasters", "acme@example.com", "+1-555-0001", "Wholesale buyer, pays on time."),
    (2, "Blue Fern Cafe", "blue@example.com", "+1-555-0002", "Occasionally late on payment."),
]
INVOICES = [
    (1, 1, 100.00, "paid", "2026-01-01", "2026-01-31"),
    (2, 2, 50.00, "overdue", "2026-01-10", "2026-02-09"),
    (3, 2, 25.00, "pending", "2026-02-01", "2026-03-01"),
]
SUPPLIERS = [
    (1, "Highland Bean Co", "sales@example.com", "+1-555-0101", "Reliable lead times."),
]
PURCHASE_ORDERS = [
    (1, 1, "Coffee beans (50kg)", 10, "delivered", "2026-01-05"),
    (2, 1, "Coffee beans (50kg)", 5, "pending", "2026-02-01"),
]


@pytest.fixture
def test_db_path(tmp_path):
    db_path = tmp_path / "test_business_ops.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", CUSTOMERS)
    conn.executemany("INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?)", INVOICES)
    conn.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?)", SUPPLIERS)
    conn.executemany("INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?)", PURCHASE_ORDERS)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def patched_db(monkeypatch, test_db_path):
    for module in (write_tools, context_tool, memory_tool, sandbox_tool):
        monkeypatch.setattr(module, "DB_PATH", test_db_path)
    return test_db_path


class FakeCorrespondenceStore:
    """Stands in for the real Chroma store in context_tool — get_customer_
    context/get_supplier_context only ever call .get(where=...) (a metadata
    filter, not a similarity search), so this only needs to replicate that
    one method against an in-memory list of (metadata, document) records."""

    def __init__(self):
        self.records: list[tuple[dict, str]] = []

    def get(self, where=None, include=None):
        def matches(meta: dict) -> bool:
            if where is None:
                return True
            conditions = where.get("$and", [where])
            return all(meta.get(k) == v for cond in conditions for k, v in cond.items())

        matched = [(meta, doc) for meta, doc in self.records if matches(meta)]
        return {
            "ids": [str(i) for i in range(len(matched))],
            "documents": [doc for _, doc in matched],
            "metadatas": [meta for meta, _ in matched],
        }


@pytest.fixture
def fake_store(monkeypatch):
    """Isolates correspondence lookups from the real, persisted Chroma
    index (data/chroma_correspondence) — without this, get_customer_context
    tests would read real production RAG data instead of test fixtures."""
    store = FakeCorrespondenceStore()
    monkeypatch.setattr(context_tool, "_store", store)
    return store
