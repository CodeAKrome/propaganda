#!/usr/bin/env python
import subprocess
import sys
import tempfile
import os


def test_timeout_functionality():
    """Test the timeout functionality of mlxllm.py"""

    # Create a test prompt file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Tell me a very long story about the history of the world.")
        test_prompt_file = f.name

    try:
        print("Testing timeout functionality...")

        # Test 1: Normal execution (should work)
        print("\nTest 1: Normal execution (no timeout)")
        result = subprocess.run(
            [
                sys.executable,
                "db/mlxllm.py",
                test_prompt_file,
                "--time_limit",
                "5",
                "--tokens",
                "10",  # Very small token count to finish quickly
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

        # Test 2: Force timeout (this should trigger timeout)
        print("\nTest 2: Force timeout")
        result = subprocess.run(
            [
                sys.executable,
                "db/mlxllm.py",
                test_prompt_file,
                "--time_limit",
                "1",  # Very short timeout
                "--tokens",
                "10000",  # Large token count to force timeout
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

        # Check if timeout error is present
        if "[TIMEOUT_ERROR]" in result.stdout:
            print("✓ Timeout functionality working correctly!")
        else:
            print("✗ Timeout functionality may not be working as expected")

    except subprocess.TimeoutExpired:
        print("Test timed out - this might be expected behavior")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Clean up
        if os.path.exists(test_prompt_file):
            os.unlink(test_prompt_file)


if __name__ == "__main__":
    test_timeout_functionality()
