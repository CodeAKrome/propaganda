#!/usr/bin/env python3

import sys
import subprocess
import os
from pathlib import Path
import fire


def count_ids(vec_file):
    """Count the number of 'ID:' entries in the vec file."""
    try:
        with open(vec_file, "r") as f:
            return sum(1 for line in f if line.startswith("ID:"))
    except FileNotFoundError:
        return 0


def run_command(cmd, timeout_sec=300):
    """Run a shell command with timeout and return exit code and success status."""
    try:
        result = subprocess.run(
            cmd, shell=True, timeout=timeout_sec, capture_output=True, text=True
        )
        return result.returncode, result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout_sec}s", file=sys.stderr)
        return 124, False  # timeout exit code
    except Exception as e:
        print(f"Command failed with exception: {e}", file=sys.stderr)
        return 1, False


def cypher(pairs, svo_prompt, vec_file, cypher_file):
    """Generate cypher relationships with failover between different LLM providers."""
    remaining_pairs = list(pairs)

    while remaining_pairs:
        src = remaining_pairs[0]
        model = remaining_pairs[1]

        print(f"Trying cypher with {src}: {model}")

        # Clear previous output
        Path(cypher_file).write_text("")

        # Build command based on source
        if src == "ollama":
            cmd = f"cat {svo_prompt} {vec_file} | ollama run --hidethinking {model} 2>/dev/null | sort | uniq > {cypher_file}"
        elif src == "gemini":
            cmd = f"cat {svo_prompt} {vec_file} | ./gemini.py {model} 2>/dev/null | sort | uniq > {cypher_file}"
        elif src == "mlx":
            cmd = f"cat {svo_prompt} {vec_file} | ./mlxllm.py - --model {model} 2>/dev/null | sort | uniq > {cypher_file}"
        else:
            print(f"Unknown cypher source: {src}", file=sys.stderr)
            remaining_pairs = remaining_pairs[2:]
            continue

        print(f"cyphering: {cmd}", file=sys.stderr)
        exit_code, _ = run_command(cmd)

        # Check if file has content
        file_exists = Path(cypher_file).exists()
        file_size = Path(cypher_file).stat().st_size if file_exists else 0

        print(
            f"Debug: exit_code={exit_code}, file exists={file_exists}, size={file_size} bytes",
            file=sys.stderr,
        )

        if exit_code == 0 and file_size > 0:
            print(f"Cypher succeeded with {src}: {model}")
            return True
        else:
            print(
                f"Cypher failed with {src}: {model} (exit_code: {exit_code})",
                file=sys.stderr,
            )

            # Remove failed pair, but keep if it's the last one
            if len(remaining_pairs) > 2:
                print(
                    f"Removing failed model {src}: {model} from retry list",
                    file=sys.stderr,
                )
                remaining_pairs = remaining_pairs[2:]
            else:
                print("Last model failed, stopping attempts", file=sys.stderr)
                break

    print("All cypher attempts failed.", file=sys.stderr)
    Path(cypher_file).write_text("")  # empty file on total failure
    return False


def report(
    pairs, reporter_prompt, cypher_file, reporter_file, vec_file, news_file, query
):
    """Generate report with failover between different LLM providers."""
    remaining_pairs = list(pairs)

    # Ensure cypher file exists
    if not Path(cypher_file).exists():
        Path(cypher_file).write_text("")

    cypher_content = Path(cypher_file).read_text()

    # Build reporter file content
    reporter_content = f"{reporter_prompt}\n<relations>\n{cypher_content}\n</relations>\n{query}\n\nUse the following data to answer:\n"
    Path(reporter_file).write_text(reporter_content)

    while remaining_pairs:
        src = remaining_pairs[0]
        model = remaining_pairs[1]

        print(f"Trying report with {src}: {model}", file=sys.stderr)

        # Clear previous output
        Path(news_file).write_text("")

        # Build command based on source
        if src == "ollama":
            cmd = f"cat {reporter_file} {vec_file} | ollama run --hidethinking {model} > {news_file} 2>/dev/null"
        elif src == "gemini":
            cmd = f"cat {reporter_file} {vec_file} | ./gemini.py {model} > {news_file} 2>/dev/null"
        elif src == "mlx":
            cmd = f"cat {reporter_file} {vec_file} | ./mlxllm.py - --model {model} > {news_file} 2>/dev/null"
        else:
            print(f"Unknown report source: {src}", file=sys.stderr)
            remaining_pairs = remaining_pairs[2:]
            continue

        exit_code, _ = run_command(cmd)

        # Check if file has content
        file_exists = Path(news_file).exists()
        file_size = Path(news_file).stat().st_size if file_exists else 0
        output_empty = "yes" if file_size == 0 else "no"

        if exit_code == 0 and file_size > 0:
            print(f"Report succeeded with {src}: {model}", file=sys.stderr)
            return True
        else:
            print(
                f"Report failed with {src}: {model} (exit_code: {exit_code}, output empty: {output_empty})",
                file=sys.stderr,
            )

            # Remove failed pair, but keep if it's the last one
            if len(remaining_pairs) > 2:
                print(
                    f"Removing failed model {src}: {model} from retry list",
                    file=sys.stderr,
                )
                remaining_pairs = remaining_pairs[2:]
            else:
                print("Last model failed, stopping attempts", file=sys.stderr)
                break

    print("All report attempts failed.", file=sys.stderr)
    Path(news_file).write_text("Nothing relevant found or generation failed.")
    return False


def main(startdate, filename, entity, query, fulltext="", svoprompt="prompt/kgsvo.txt"):
    """
    Generate news reports from vector files using LLM analysis.

    Args:
        startdate: Date offset for the report
        filename: Base filename for output files
        entity: Entity name(s) to analyze
        query: Query string for the report
        fulltext: Optional fulltext parameter
        svoprompt: Path to SVO prompt file (default: prompt/kgsvo.txt)
    """
    # File paths
    vec = f"output/{filename}.vec"
    news = f"output/{filename}.md"
    cypherfile = f"output/{filename}.cypher"
    reporterfile = f"output/{filename}.reporter"
    barenews = f"output/{filename}"
    svo_prompt = svoprompt

    # Count entries
    count = count_ids(vec)
    print(f"{filename}\t{startdate}\t{count}")

    # Exit if no entries
    if count == 0:
        return

    # Reporter prompt
    reporter_prompt = """You are an expert political analyst and news reporter called Lotta Talker.
The attached file contains the text of news articles.
Summarize the articles in an insightful fashion paying attention to detail.
Describe all the major themes.
If something is irrelevant, ignore it.
If you don't find anything relevant, just say 'Nothing relevant found.'
Describe all the major themes.

Relationships are shown in the <relations> section in (subject,object,verb,explanation) format.

Bias Analysis:
Each article, has a bias analysis in JSON format with the following structure:

a. DIRECTION - The political leaning:
- L = Left (liberal/progressive bias)
- C = Center (balanced/neutral)
- R = Right (conservative bias)

b. DEGREE - The intensity of bias:
- L = Low (minimal bias, mostly factual)
- M = Medium (noticeable bias in framing or emphasis)
- H = High (strong bias, significant editorializing)

3. REASON - A brief explanation (2-4 sentences) justifying your direction and degree ratings based on specific evidence from the article.

-Example-
{"dir": {"L": 0.1, "C": 0.4, "R": 0.5}, "deg": {"L": 0.1, "M": 0.2, "H": 0.7}, "reason": "The article uses loaded language like 'radical agenda' and 'government overreach' while exclusively quoting conservative sources. It omits counterarguments and frames policy proposals in exclusively negative terms."}

Analize the bias of the articles and summarize the bias findings in a concise paragraph at the end of your output.
Do not menntion the bias numbers directly, just summarize the bias findings in a concise paragraph.
Do not reference mongodb id article numbers.
Use the bias data to determine the overall bias of the articles and give that as a conclusion.
Be specific and list sources when mentioning which sources are biased and how.

When reporting, speak in a professional newscaster tone like Walter Kronkite.

Respond as if you are a TV reporter on camera explaining to your audience.
Use a professional newscaster tone like Walter Kronkite.
Only reply with what the reporter says, not any stage direction like musical intros or camera direction.
Do not use markup. Do not make tables. Reply with plain text only."""

    # Model configurations - define your preferred failover order
    cypher_pairs = ["ollama", "gpt-oss:20b"]

    report_pairs = [
        "gemini",
        "models/gemini-3-flash-preview",
        "gemini",
        "models/gemini-2.5-flash",
        "ollama",
        "gpt-oss:20b",
    ]

    # Run cypher generation
    cypher(cypher_pairs, svo_prompt, vec, cypherfile)

    # Run report generation
    report(report_pairs, reporter_prompt, cypherfile, reporterfile, vec, news, query)


if __name__ == "__main__":
    fire.Fire(main)
