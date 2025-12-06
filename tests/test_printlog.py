"""Unit tests for the PrintLog logging utility.

This module verifies that PrintLog correctly captures and logs output to files.
"""

from src.printlog import PrintLog


def test_printlog():
    """Verify that PrintLog captures output to file when context manager exits.

    Tests that:
    1. PrintLog context manager executes without errors
    2. Output written during context (print statements) is captured to file
    3. File is accessible and contains expected content
    """
    local_log = PrintLog(extra_name="_test_unit", enable=False)
    with local_log:
        print("test")

    with open(local_log.filename, "r") as f:
        assert "test" in f.read()
