from agents.logging_utils import scrub_args


def test_scrub_args_redacts_sensitive_keys_outright():
    args = {"amount": 150.0, "email": "blue@bluefern.cafe", "phone": "+1-555-0102"}
    scrubbed = scrub_args(args)
    assert scrubbed == {"amount": "[REDACTED]", "email": "[REDACTED]", "phone": "[REDACTED]"}


def test_scrub_args_redacts_embedded_email_in_free_text():
    args = {"notes": "contact them at hello@bluefern.cafe about the balance"}
    scrubbed = scrub_args(args)
    assert "hello@bluefern.cafe" not in scrubbed["notes"]
    assert "[REDACTED_EMAIL]" in scrubbed["notes"]


def test_scrub_args_redacts_embedded_phone_in_free_text():
    args = {"notes": "call 555-123-4567 to confirm"}
    scrubbed = scrub_args(args)
    assert "555-123-4567" not in scrubbed["notes"]
    assert "[REDACTED_PHONE]" in scrubbed["notes"]


def test_scrub_args_redacts_embedded_dollar_amount_in_free_text():
    args = {"notes": "balance of $1,341.25 is overdue"}
    scrubbed = scrub_args(args)
    assert "$1,341.25" not in scrubbed["notes"]
    assert "[REDACTED_AMOUNT]" in scrubbed["notes"]


def test_scrub_args_leaves_non_sensitive_values_untouched():
    args = {"customer": "Blue Fern Cafe", "invoice_id": 2}
    scrubbed = scrub_args(args)
    assert scrubbed == {"customer": "Blue Fern Cafe", "invoice_id": 2}
