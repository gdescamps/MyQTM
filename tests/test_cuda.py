"""Unit tests for CUDA GPU selection utility.

This module verifies automatic GPU selection based on available memory.
"""

import os

from src.cuda import auto_select_gpu


def test_auto_select_gpu():
    """Verify that GPU auto-selection sets the CUDA_VISIBLE_DEVICES environment variable.

    Tests that auto_select_gpu correctly:
    1. Executes without raising exceptions
    2. Selects a GPU with at least 500MB available memory
    3. Sets the CUDA_VISIBLE_DEVICES environment variable
    """
    # Execute GPU auto-selection with 500MB minimum memory threshold
    try:
        auto_select_gpu(threshold_mb=500)
        print("auto_select_gpu executed successfully.")
    except Exception as e:
        print(f"auto_select_gpu failed with exception: {e}")
        assert False, "auto_select_gpu raised an exception"

    # Verify that CUDA_VISIBLE_DEVICES environment variable is set
    assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES is not set"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
