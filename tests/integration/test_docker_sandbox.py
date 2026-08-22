"""Runs run_analysis_script for real, against the actual Docker daemon —
this is what test_sandbox_tool.py's unit tests deliberately don't cover,
since building/running a container is exactly the kind of slow, external
dependency unit tests are supposed to avoid."""

from agents.sandbox_tool import run_analysis_script


def test_runs_pandas_analysis_for_real(patched_db, require_docker):
    script = (
        "import pandas as pd\n"
        "df = pd.read_csv('invoices.csv')\n"
        "print('overdue total:', df[df.status == 'overdue']['amount'].sum())\n"
    )
    result = run_analysis_script.invoke({"tables": ["invoices"], "script": script})
    assert "overdue total: 50.0" in result


def test_network_is_blocked(patched_db, require_docker):
    script = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=3)\n"
        "    print('REACHED')\n"
        "except Exception as e:\n"
        "    print('blocked:', type(e).__name__)\n"
    )
    result = run_analysis_script.invoke({"tables": ["invoices"], "script": script})
    assert "blocked:" in result
    assert "REACHED" not in result


def test_filesystem_is_read_only(patched_db, require_docker):
    script = (
        "try:\n"
        "    open('invoices.csv', 'a').write('x')\n"
        "    print('WROTE')\n"
        "except Exception as e:\n"
        "    print('blocked:', type(e).__name__)\n"
    )
    result = run_analysis_script.invoke({"tables": ["invoices"], "script": script})
    assert "blocked:" in result
    assert "WROTE" not in result


def test_timeout_kills_the_script(patched_db, require_docker):
    result = run_analysis_script.invoke(
        {"tables": ["invoices"], "script": "import time; time.sleep(60)"}
    )
    assert "timed out" in result
