#!/usr/bin/env python3
import sys
import subprocess
import argparse
from tqdm import tqdm


def fmt_command(model, mongo_id):
    """
    Format a shell command using the model name and MongoDB ID.
    """
    # Using raw string and proper escaping for the complex command
    command = (
        f"../db/mongo_rw.py read --id={mongo_id} --field=article | "
        f"mlx_lm.generate --model {model} --prompt - --max-tokens 100000 "
        f"--verbose FALSE --prompt-cache-file prompt/claudeopus_CoVe.safetensors | "
        f'grep "final output" | '
        f"perl -0777 -ne '@m = /\\{{(?:[^{{}}]|(?0))*\\}}/g; print $m[-1]'"
    )
    return command

def main():
    parser = argparse.ArgumentParser(description="Process MongoDB IDs with MLX model")
    parser.add_argument("model", help="Name of MLX model to use")
    args = parser.parse_args()

    # Read all MongoDB IDs from stdin
    mongo_ids = [line.strip() for line in sys.stdin if line.strip()]
    
    if not mongo_ids:
        print("No MongoDB IDs provided", file=sys.stderr)
        return

    # Process each ID with progress bar
    for mongo_id in tqdm(mongo_ids, desc="Processing IDs", unit="id"):
        # Get the command for this ID
        command = fmt_command(args.model, mongo_id)

        tqdm.write(f"Processing ID: {mongo_id} {command}")

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
                tqdm.write(f"[{mongo_id}] Output: {result.stdout.strip()}")

            # Print errors if any
            if result.returncode != 0:
                tqdm.write(
                    f"[{mongo_id}] Error (exit code {result.returncode}): {result.stderr.strip()}"
                )

        except subprocess.TimeoutExpired:
            tqdm.write(f"[{mongo_id}] Error: Command timeout")
        except Exception as e:
            tqdm.write(f"[{mongo_id}] Error: {str(e)}")


if __name__ == "__main__":
    main()