from agents.write_tools import create_invoice, mark_invoice_paid, send_reminder_email


def test_create_invoice_rejects_non_positive_amount(patched_db):
    result = create_invoice.invoke({"customer_id": 1, "amount": -5, "due_date": "2026-04-01"})
    assert "must be positive" in result
    assert "not created" in result


def test_create_invoice_rejects_unknown_customer(patched_db):
    result = create_invoice.invoke({"customer_id": 999, "amount": 50, "due_date": "2026-04-01"})
    assert "No customer with id 999" in result


def test_create_invoice_happy_path(patched_db):
    result = create_invoice.invoke({"customer_id": 1, "amount": 250, "due_date": "2026-04-01"})
    assert "Created invoice" in result
    assert "Acme Roasters" in result
    assert "$250.00" in result


def test_mark_invoice_paid_rejects_non_positive_id(patched_db):
    result = mark_invoice_paid.invoke({"invoice_id": -1})
    assert "must be positive" in result


def test_mark_invoice_paid_rejects_unknown_invoice(patched_db):
    result = mark_invoice_paid.invoke({"invoice_id": 999})
    assert "No invoice with id 999" in result


def test_mark_invoice_paid_rejects_already_paid(patched_db):
    result = mark_invoice_paid.invoke({"invoice_id": 1})
    assert "already marked paid" in result


def test_mark_invoice_paid_happy_path(patched_db):
    result = mark_invoice_paid.invoke({"invoice_id": 2})
    assert "marked as paid" in result
    assert "$50.00" in result


def test_send_reminder_email_rejects_empty_subject(patched_db):
    result = send_reminder_email.invoke({"customer_id": 1, "subject": "  ", "body": "hi"})
    assert "must not be empty" in result


def test_send_reminder_email_rejects_empty_body(patched_db):
    result = send_reminder_email.invoke({"customer_id": 1, "subject": "hi", "body": ""})
    assert "must not be empty" in result


def test_send_reminder_email_rejects_unknown_customer(patched_db):
    result = send_reminder_email.invoke({"customer_id": 999, "subject": "hi", "body": "hi"})
    assert "not sent" in result


def test_send_reminder_email_happy_path(patched_db):
    result = send_reminder_email.invoke(
        {"customer_id": 2, "subject": "Overdue reminder", "body": "Please pay."}
    )
    assert "Email sent to Blue Fern Cafe" in result
    assert "blue@example.com" in result
