"""Seeds data/business_ops.db with fake customer data for Phase 0.

Run with: .venv/bin/python data/seed_data.py
Re-running drops and recreates the customers table, so it's safe to repeat.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "business_ops.db"

CUSTOMERS = [
    (1, "Acme Roasters", "orders@acmeroasters.com", "+1-555-0101",
     "Wholesale coffee buyer, orders monthly, pays net-30."),
    (2, "Blue Fern Cafe", "hello@bluefern.cafe", "+1-555-0102",
     "Small independent cafe, orders biweekly, occasionally late on payment."),
    (3, "Cedar & Oak Catering", "accounts@cedaroak.com", "+1-555-0103",
     "Large catering company, big orders around holidays, always pays on time."),
    (4, "Daybreak Bakery", "info@daybreakbakery.com", "+1-555-0104",
     "New customer, first order placed last month."),
    (5, "Evergreen Grocers", "purchasing@evergreengrocers.com", "+1-555-0105",
     "Regional grocery chain, multiple locations, requires PO numbers on invoices."),
    (6, "Fig & Thyme", "owner@figandthyme.com", "+1-555-0106",
     "Boutique restaurant, low volume but high margin orders."),
    (7, "Golden Hour Coffee Co", "team@goldenhourcoffee.com", "+1-555-0107",
     "Frequent late payer, has needed reminder emails twice this year."),
]


def seed():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS customers")
    cur.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            notes TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO customers (id, name, email, phone, notes) VALUES (?, ?, ?, ?, ?)",
        CUSTOMERS,
    )
    conn.commit()
    conn.close()
    print(f"Seeded {len(CUSTOMERS)} customers into {DB_PATH}")


if __name__ == "__main__":
    seed()
