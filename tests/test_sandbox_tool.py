"""Only the Docker-independent parts of sandbox_tool are unit tested here —
table validation and the CSV snapshot step. Actually running a script in
the container is exercised in the integration tests instead, since it
needs Docker."""

import csv

from agents.sandbox_tool import ALLOWED_TABLES, _snapshot_tables, run_analysis_script


def test_run_analysis_script_rejects_no_valid_tables(patched_db):
    result = run_analysis_script.invoke({"tables": ["not_a_real_table"], "script": "print(1)"})
    assert "No valid tables requested" in result
    assert all(t in result for t in sorted(ALLOWED_TABLES))


def test_run_analysis_script_rejects_empty_tables(patched_db):
    result = run_analysis_script.invoke({"tables": [], "script": "print(1)"})
    assert "No valid tables requested" in result


def test_snapshot_tables_writes_expected_csv(patched_db, tmp_path):
    written = _snapshot_tables(["invoices", "not_a_real_table"], tmp_path)
    assert written == ["invoices"]

    with open(tmp_path / "invoices.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["id", "customer_id", "amount", "status", "issued_date", "due_date"]
    assert len(rows) == 4  # header + 3 seeded invoices
    assert not (tmp_path / "not_a_real_table.csv").exists()


def test_snapshot_tables_skips_disallowed_table(patched_db, tmp_path):
    written = _snapshot_tables(["customers", "sqlite_master"], tmp_path)
    assert written == ["customers"]
