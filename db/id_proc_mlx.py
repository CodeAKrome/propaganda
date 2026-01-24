#!/usr/bin/env python3
import sys
import subprocess
import argparse


def fmt_command(model, mongo_id):
    """
    Format a shell command using the model name and MongoDB ID.
    """
    # Using raw string and proper escaping for the complex command
    command = (
        f"time cat prompt/right.txt | "
        f"mlx_lm.generate --model {model} --prompt - --max-tokens 100000 "
        f"--verbose FALSE --prompt-cache-file prompt/claudeopus_CoVe.safetensors | "
        f'grep "Now produce final output" | '
        f"perl -0777 -ne '@m = /\\{{(?:[^{{}}]|(?0))*\\}}/g; print $m[-1]'"
    )
    return command


def main():
    parser = argparse.ArgumentParser(description="Process MongoDB IDs with MLX model")
    parser.add_argument("model", help="Name of MLX model to use")
    args = parser.parse_args()

    # Read MongoDB IDs from stdin
    for line in sys.stdin:
        mongo_id = line.strip()
        if not mongo_id:
            continue

        # Get the command for this ID
        command = fmt_command(args.model, mongo_id)

        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout, adjust as needed
            )

            # Print output
            if result.stdout:
                print(f"[{mongo_id}] Output: {result.stdout.strip()}")

            # Print errors if any
            if result.returncode != 0:
                print(
                    f"[{mongo_id}] Error (exit code {result.returncode}): {result.stderr.strip()}",
                    file=sys.stderr,
                )

        except subprocess.TimeoutExpired:
            print(f"[{mongo_id}] Error: Command timeout", file=sys.stderr)
        except Exception as e:
            print(f"[{mongo_id}] Error: {str(e)}", file=sys.stderr)


if __name__ == "__main__":
    main()
