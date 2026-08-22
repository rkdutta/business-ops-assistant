from agents.context_tool import get_customer_context, get_supplier_context


def test_get_customer_context_unknown_customer(patched_db, fake_store):
    result = get_customer_context.invoke({"customer": "Nonexistent Co"})
    assert "No customer matching" in result


def test_get_customer_context_by_name(patched_db, fake_store):
    result = get_customer_context.invoke({"customer": "Blue Fern"})
    assert "Blue Fern Cafe (customer #2)" in result
    assert "2 invoice(s) total" in result
    assert "1 overdue totaling $50.00" in result
    assert "#2 $50.00 overdue" in result
    assert "No correspondence on file." in result


def test_get_customer_context_by_id(patched_db, fake_store):
    result = get_customer_context.invoke({"customer": "1"})
    assert "Acme Roasters (customer #1)" in result
    assert "1 invoice(s) total" in result
    assert "overdue" not in result.split("Correspondence:")[0].lower()


def test_get_customer_context_no_invoices(patched_db, fake_store, test_db_path):
    import sqlite3

    conn = sqlite3.connect(test_db_path)
    conn.execute(
        "INSERT INTO customers VALUES (3, 'No Invoice Co', 'x@example.com', NULL, NULL)"
    )
    conn.commit()
    conn.close()

    result = get_customer_context.invoke({"customer": "No Invoice Co"})
    assert "No invoices on record." in result


def test_get_customer_context_merges_correspondence(patched_db, fake_store):
    fake_store.records.append(
        (
            {"entity_type": "customer", "entity_id": 2, "title": "Payment plan", "date": "2026-01-15"},
            "Agreed to split the balance into two installments.",
        )
    )
    result = get_customer_context.invoke({"customer": "Blue Fern"})
    assert "Payment plan (2026-01-15)" in result
    assert "split the balance into two installments" in result


def test_get_supplier_context_unknown_supplier(patched_db, fake_store):
    result = get_supplier_context.invoke({"supplier": "Nonexistent Supplier"})
    assert "No supplier matching" in result


def test_get_supplier_context_by_name(patched_db, fake_store):
    result = get_supplier_context.invoke({"supplier": "Highland"})
    assert "Highland Bean Co (supplier #1)" in result
    assert "2 purchase order(s) total" in result
    assert "1 pending/in transit" in result
