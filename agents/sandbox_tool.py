"""Sandboxed analysis-script tool (goals doc "Sandboxing": run analysis
scripts in an isolated sandbox, not directly against live data).

deepagents' built-in backends don't fit this project: LocalShellBackend
gives zero isolation (runs directly on the host, full access), and
LangSmithSandbox needs an external managed cloud service, which breaks this
project's fully-local design. Docker gives real process isolation instead:
the script runs in a throwaway container (image built from
sandbox/Dockerfile — python:3.11-slim + pandas, non-root user) with no
network, capped CPU/memory/process count, all Linux capabilities dropped,
and a read-only root filesystem. It only ever sees a point-in-time CSV
snapshot of the tables it asked for, mounted read-only — never a live DB
connection or credentials.
"""

import csv
import shutil
import subprocess
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent / "data" / "business_ops.db"
SANDBOX_DIR = Path(__file__).parent.parent / "sandbox"
IMAGE_NAME = "business-ops-sandbox:latest"
ALLOWED_TABLES = {"customers", "invoices", "suppliers", "purchase_orders"}
TIMEOUT_SECONDS = 15

# Docker Desktop's file sharing on this machine is scoped to the project
# tree (/Users/...), not the macOS system temp dir (/var/folders, /tmp) —
# a bind mount from tempfile.TemporaryDirectory() silently shows up empty
# inside the container. Scratch runs live under the project instead, in a
# directory that's already shared, and are deleted immediately after.
_RUNS_DIR = SANDBOX_DIR / "runs"


@contextmanager
def _scratch_dir():
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = _RUNS_DIR / uuid.uuid4().hex
    run_dir.mkdir()
    try:
        yield run_dir
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

_image_ready = False


def _ensure_image() -> str | None:
    """Builds the sandbox image on first use if it doesn't exist yet.
    Returns an error message on failure, else None."""
    global _image_ready
    if _image_ready:
        return None
    inspect = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME], capture_output=True, text=True
    )
    if inspect.returncode == 0:
        _image_ready = True
        return None
    build = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, str(SANDBOX_DIR)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if build.returncode != 0:
        return f"Failed to build sandbox image:\n{build.stderr[-2000:]}"
    _image_ready = True
    return None


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
    """Run a Python data-analysis script in an isolated Docker sandbox, not
    against the live database. Choose which tables you need from
    customers, invoices, suppliers, purchase_orders — each is snapshotted
    to a CSV file (e.g. invoices.csv) in the script's working directory,
    and the script (pandas is available) should read those files by name
    instead of querying the database. Status columns (e.g. invoices.status)
    are ground truth — 'paid'/'overdue'/'pending' — use them directly rather
    than re-deriving status from date comparisons. Use this for computation over data
    (totals by month, aggregates, trends) rather than simple lookups, which
    sqlite_execute already handles. Print the result — only the script's
    stdout is returned. The container has no network access, capped
    CPU/memory, and a 15-second timeout."""
    image_error = _ensure_image()
    if image_error:
        return image_error

    with _scratch_dir() as tmp_path:
        snapshotted = _snapshot_tables(tables, tmp_path)
        if not snapshotted:
            return f"No valid tables requested — choose from {sorted(ALLOWED_TABLES)}."

        script_path = tmp_path / "analysis.py"
        script_path.write_text(script)

        container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", "256m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{tmp_path}:/workspace:ro",
            IMAGE_NAME,
            "python", "analysis.py",
        ]

        try:
            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            # subprocess's own timeout only kills the `docker run` CLI
            # process, not the container it started — dockerd keeps it
            # running independently, so it has to be stopped explicitly.
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return f"Script timed out after {TIMEOUT_SECONDS}s."

        if result.returncode != 0:
            return f"Script failed:\n{result.stderr[-2000:]}"
        return result.stdout[-4000:] or "Script ran successfully but printed nothing."
