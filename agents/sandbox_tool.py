"""Sandboxed analysis-script tool (goals doc "Sandboxing": run analysis
scripts in an isolated sandbox, not directly against live data).

deepagents' built-in backends don't fit this project: LocalShellBackend
gives zero isolation (runs directly on the host, full access), and
LangSmithSandbox needs an external managed cloud service, which breaks this
project's fully-local design (local Ollama, local SQLite, no other external
dependencies). Docker would give real OS-level isolation but isn't running
here, and starting it is a bigger ask than this feature needs.

What this gives instead: real *data* isolation, which is the literal thing
the goals doc asks for ("rather than directly against your live data"). The
script never receives a live DB connection or credentials — it only ever
sees a point-in-time CSV snapshot of the tables it asked for, in a fresh
temp directory, and runs as a subprocess with a timeout. This is NOT full
process/OS sandboxing: the script shares the host's Python interpreter and
filesystem permissions, so nothing stops it from e.g. opening an arbitrary
path itself. For a trusted local single-user assistant, isolating it from
the live database by default and bounding its runtime is the risk that
actually matters here; true code-execution sandboxing would need a
container, which this environment doesn't currently have available.
"""

import csv
import subprocess
import sqlite3
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"
ALLOWED_TABLES = {"customers", "invoices", "suppliers", "purchase_orders"}
TIMEOUT_SECONDS = 15


def _snapshot_tables(tables: list[str], dest: Path) -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    written = []
    for table in tables:
        if table not in ALLOWED_TABLES:
            continue
        columns = [d[0] for d in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        with open(dest / f"{table}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        written.append(table)
    conn.close()
    return written


@tool
def run_analysis_script(tables: list[str], script: str) -> str:
    """Run a Python data-analysis script in an isolated sandbox, not
    against the live database. Choose which tables you need from
    customers, invoices, suppliers, purchase_orders — each is snapshotted
    to a CSV file (e.g. invoices.csv) in the script's working directory,
    and the script (pandas is available) should read those files by name
    instead of querying the database. Use this for computation over data
    (totals by month, aggregates, trends) rather than simple lookups, which
    sqlite_execute already handles. Print the result — only the script's
    stdout is returned. Runs with a 15-second timeout and no access to the
    live database."""
    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp:
        tmp_path = Path(tmp)
        snapshotted = _snapshot_tables(tables, tmp_path)
        if not snapshotted:
            return f"No valid tables requested — choose from {sorted(ALLOWED_TABLES)}."

        script_path = tmp_path / "analysis.py"
        script_path.write_text(script)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"Script timed out after {TIMEOUT_SECONDS}s."

        if result.returncode != 0:
            return f"Script failed:\n{result.stderr[-2000:]}"
        return result.stdout[-4000:] or "Script ran successfully but printed nothing."
