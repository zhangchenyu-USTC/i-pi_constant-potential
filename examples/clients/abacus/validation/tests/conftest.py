"""Require independently identified executables for designated acceptance runs."""

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

IPI_ROOT = Path(os.environ.get("IPI_ROOT", str(Path(__file__).resolve().parents[5])))
os.environ.setdefault("ABACUS_TEST_IPI_ROOT", str(IPI_ROOT))
sys.path.insert(0, os.environ["ABACUS_TEST_IPI_ROOT"])
if "ABACUS_PP_ORB" not in os.environ and "ABACUS_SOURCE_ROOT" in os.environ:
    os.environ["ABACUS_PP_ORB"] = str(
        Path(os.environ["ABACUS_SOURCE_ROOT"]) / "tests/PP_ORB"
    )


def pytest_sessionstart(session):
    if os.environ.get("ABACUS_REQUIRE_REAL_TESTS") != "1":
        return
    required = (
        "ABACUS_CP_EXECUTABLE",
        "ABACUS_REFERENCE_EXECUTABLE",
        "ABACUS_PP_ORB",
        "ABACUS_TEST_IPI_ROOT",
    )
    for name in required:
        value = os.environ.get(name)
        if not value or not Path(value).exists():
            raise pytest.UsageError(f"Acceptance requires an existing {name} path")
    hashes = {}
    for name in required[:2]:
        binary = Path(os.environ[name]).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise pytest.UsageError(f"{name} must be an executable file")
        hashes[name] = {
            "path": str(binary),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
    if hashes[required[0]]["sha256"] == hashes[required[1]]["sha256"]:
        raise pytest.UsageError(
            "CP and reference binaries must be independently built, not identical copies"
        )
    output = os.environ.get("ABACUS_TEST_IDENTITY_FILE")
    if output:
        Path(output).write_text(json.dumps(hashes, indent=2) + "\n")


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("ABACUS_REQUIRE_REAL_TESTS") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter and reporter.stats.get("skipped"):
        reporter.write_sep("!", "Designated acceptance suite must not skip tests")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
