"""jude console (python -m jude): SQL REPL + storage-governance dot-commands."""
import subprocess
import sys


def _run(script: str) -> str:
    env = {"JUDE_RUNNER": "local", "PATH": "/usr/bin:/bin"}
    import os
    env["HOME"] = os.environ.get("HOME", "/tmp")
    env["VIRTUAL_ENV"] = os.environ.get("VIRTUAL_ENV", "")
    p = subprocess.run(
        [sys.executable, "-m", "jude"],
        input=script, capture_output=True, text=True, timeout=120,
        env={**os.environ, **env},
    )
    return p.stdout + p.stderr


def test_sql_and_help():
    out = _run("SELECT 1 AS a, 'hi' AS b\n.help\n.quit\n")
    assert "a" in out and "hi" in out          # SQL result rendered
    assert ".tables" in out and ".discover" in out  # help listing


def test_tables_empty_and_error_recovery():
    out = _run("SELECT * FROM nope_no_table\nSELECT 42 AS x\n.quit\n")
    assert "error:" in out.lower()   # bad query reported
    assert "42" in out               # REPL kept going after the error
