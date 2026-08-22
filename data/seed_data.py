"""Seeds data/business_ops.db with fake customer/invoice/supplier data.

Run with: .venv/bin/python data/seed_data.py
Re-running drops and recreates these tables, so it's safe to repeat. Only
touches customers/invoices/suppliers/purchase_orders — leaves the LangGraph
checkpoint tables and chat_threads (also stored in this DB) untouched.
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

# customer_id references CUSTOMERS.id
INVOICES = [
    (1, 1, 1250.00, "paid", "2026-06-01", "2026-07-01"),
    (2, 2, 340.50, "overdue", "2026-06-15", "2026-07-15"),
    (3, 3, 4800.00, "paid", "2026-07-01", "2026-07-31"),
    (4, 4, 210.00, "pending", "2026-08-10", "2026-09-09"),
    (5, 5, 2100.00, "paid", "2026-07-20", "2026-08-19"),
    (6, 6, 675.25, "pending", "2026-08-15", "2026-09-14"),
    (7, 7, 512.00, "overdue", "2026-06-20", "2026-07-20"),
    (8, 7, 488.75, "overdue", "2026-07-25", "2026-08-24"),
]

SUPPLIERS = [
    (1, "Highland Bean Co", "sales@highlandbean.com", "+1-555-0201",
     "Green coffee bean supplier, net-45 terms, reliable lead times."),
    (2, "Packrite Supplies", "orders@packrite.com", "+1-555-0202",
     "Packaging and cups, occasional shipping delays."),
    (3, "Dairy Direct", "accounts@dairydirect.com", "+1-555-0203",
     "Milk and dairy, weekly delivery, requires 48h order notice."),
    (4, "Sunrise Logistics", "dispatch@sunriselogistics.com", "+1-555-0204",
     "Freight/delivery partner for large catering orders."),
    (5, "Bakewell Ingredients", "hello@bakewell.com", "+1-555-0205",
     "Flour, sugar, baking staples, price increases each quarter."),
]

# supplier_id references SUPPLIERS.id
PURCHASE_ORDERS = [
    (1, 1, "Green coffee beans (Ethiopia, 50kg)", 10, "delivered", "2026-07-05"),
    (2, 2, "12oz cups (case of 1000)", 20, "delivered", "2026-07-10"),
    (3, 3, "Whole milk (gallons)", 100, "delivered", "2026-08-14"),
    (4, 1, "Green coffee beans (Colombia, 50kg)", 15, "pending", "2026-08-18"),
    (5, 4, "Catering delivery - Cedar & Oak order", 1, "in_transit", "2026-08-19"),
    (6, 5, "All-purpose flour (25kg bags)", 30, "pending", "2026-08-20"),
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

    cur.execute("DROP TABLE IF EXISTS invoices")
    cur.execute(
        """
        CREATE TABLE invoices (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            amount REAL NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('paid', 'overdue', 'pending')),
            issued_date TEXT NOT NULL,
            due_date TEXT NOT NULL
        )
        """
    )
    cur.executemany(
        "INSERT INTO invoices (id, customer_id, amount, status, issued_date, due_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        INVOICES,
    )

    cur.execute("DROP TABLE IF EXISTS suppliers")
    cur.execute(
        """
        CREATE TABLE suppliers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            notes TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO suppliers (id, name, email, phone, notes) VALUES (?, ?, ?, ?, ?)",
        SUPPLIERS,
    )

    cur.execute("DROP TABLE IF EXISTS purchase_orders")
    cur.execute(
        """
        CREATE TABLE purchase_orders (
            id INTEGER PRIMARY KEY,
            supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
            item TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'in_transit', 'delivered')),
            order_date TEXT NOT NULL
        )
        """
    )
    cur.executemany(
        "INSERT INTO purchase_orders (id, supplier_id, item, quantity, status, order_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        PURCHASE_ORDERS,
    )

    conn.commit()
    conn.close()
    print(
        f"Seeded {len(CUSTOMERS)} customers, {len(INVOICES)} invoices, "
        f"{len(SUPPLIERS)} suppliers, {len(PURCHASE_ORDERS)} purchase orders into {DB_PATH}"
    )


if __name__ == "__main__":
    seed()
