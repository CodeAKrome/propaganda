#!/usr/bin/env python3

"""
Database utility module for managing and accessing data.

Improvements:
- Error handling with retry logic
- Timeout handling for LLM calls
- Graceful degradation on failure
- Logging instead of print statements
"""

import sys
import subprocess
import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import fire


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    exit_code: int
    success: bool
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0


LLM_TIMEOUT_SEC = 600  # 10 minutes max per request
MAX_RETRIES = 2


def count_ids(vec_file: str) -> int:
    """Count the number of 'ID:' entries in the vec file."""
    try:
        with open(vec_file, "r") as f:
            return sum(1 for line in f if line.startswith("ID:"))
    except FileNotFoundError:
        return 0


def run_command(
    cmd: str, timeout_sec: int = LLM_TIMEOUT_SEC, retry_count: int = 0
) -> Tuple[int, bool]:
    """
    Run a command with timeout and return exit code and success status.
    Uses list-based subprocess to avoid shell injection.
    """
    try:
        start_time = time.time()

        # Parse command into list (simple shlex-like parsing)
        # Handle: ollama run, python3 script, cat file, etc.
        parts = cmd.split()
        result = subprocess.run(
            parts, timeout=timeout_sec, capture_output=True, text=True
        )
        duration_ms = int((time.time() - start_time) * 1000)

        if result.returncode == 0:
            logger.debug(f"Command succeeded in {duration_ms}ms: {cmd[:50]}...")
            return result.returncode, True

        # Command failed - retry if we haven't exceeded retry count
        if retry_count < MAX_RETRIES:
            logger.warning(
                f"Command failed (exit {result.returncode}), retrying ({retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(1)  # Brief pause before retry
            return run_command(cmd, timeout_sec, retry_count + 1)

        logger.error(f"Command failed after {MAX_RETRIES} retries: {cmd[:50]}...")
        return result.returncode, False

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout_sec}s")
        return 124, False
    except Exception as e:
        logger.error(f"Command failed with exception: {e}")
        return 1, False


def cypher(
    pairs: List[str],
    svo_prompt: str,
    vec_file: str,
    cypher_file: str,
    timeout_sec: int = LLM_TIMEOUT_SEC,
    use_cache: bool = False,
) -> bool:
    """Generate cypher relationships. If use_cache=True, read from MongoDB; otherwise call LLM."""

    if use_cache:
        # Read MongoDB IDs from vec_file and concatenate cached svo from MongoDB
        try:
            ids = [
                line.split(":", 1)[1].strip()
                for line in Path(vec_file).read_text().split("\n")
                if line.startswith("ID:")
            ]
        except Exception as e:
            logger.warning(f"Could not read vec file: {e}")
            Path(cypher_file).write_text("")
            return False

        if not ids:
            Path(cypher_file).write_text("")
            return True

        import pymongo
        from bson import ObjectId

        MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
        mongo_coll = pymongo.MongoClient(MONGO_URI)["rssnews"]["articles"]

        # Fetch svo in batch
        cursor = mongo_coll.find(
            {"_id": {"$in": [ObjectId(i) for i in ids]}},
            {"_id": 1, "svo": 1},
        )
        id_to_svo = {str(d["_id"]): d.get("svo", "") for d in cursor}

        # Preserve vec_file ordering
        lines = []
        for mongo_id in ids:
            svo = id_to_svo.get(mongo_id, "")
            if svo:
                lines.append(svo)

        content = "\n".join(lines)
        Path(cypher_file).write_text(content)

        if content:
            logger.info(f"Cypher from MongoDB cache: {len(lines)} articles with svo")
            return True
        else:
            logger.warning("No cached svo found in MongoDB for these articles")
            Path(cypher_file).write_text("")
            return False

    # LLM-based generation (original path)
    remaining_pairs = list(pairs)

    while remaining_pairs:
        src = remaining_pairs[0]
        model = remaining_pairs[1]

        logger.info(f"Trying cypher with {src}: {model}")

        # Clear previous output
        Path(cypher_file).write_text("")

        try:
            # Read input files
            prompt_content = Path(svo_prompt).read_text()
            vec_content = Path(vec_file).read_text()
            full_input = prompt_content + "\n" + vec_content

            # Build command based on source
            if src == "ollama":
                result = subprocess.run(
                    ["ollama", "run", "--hidethinking", "--nowordwrap", "gpt-oss:120b"],
                    input=full_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, "ollama")

                # Filter ANSI codes
                filter_result = subprocess.run(
                    ["python3", "filter_ansi.py"],
                    input=result.stdout,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = filter_result.stdout if filter_result.returncode == 0 else result.stdout

            elif src == "gemini":
                result = subprocess.run(
                    ["python3", "gemini.py", model],
                    input=full_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, "gemini")
                output = "".join(sorted(set(result.stdout.split("\n"))))

            elif src == "mlx":
                result = subprocess.run(
                    ["python3", "mlxllm.py", "-", "--model", model],
                    input=full_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, "mlx")
                output = "".join(sorted(set(result.stdout.split("\n"))))

            else:
                logger.warning(f"Unknown cypher source: {src}")
                remaining_pairs = remaining_pairs[2:]
                continue

            if output.strip():
                Path(cypher_file).write_text(output)
                logger.info(f"Cypher succeeded with {src}: {model}")
                return True

        except subprocess.CalledProcessError as e:
            logger.warning(f"Cypher failed with {src}: {model} (exit {e.returncode})")
        except subprocess.TimeoutExpired:
            logger.error(f"Cypher timed out after {timeout_sec}s")
        except Exception as e:
            logger.error(f"Cypher failed with exception: {e}")

        # Remove failed pair, but keep if it's the last one
        if len(remaining_pairs) > 2:
            logger.info(f"Removing failed model {src}: {model} from retry list")
            remaining_pairs = remaining_pairs[2:]
        else:
            logger.error("Last model failed, stopping attempts")
            break

    logger.error("All cypher attempts failed.")
    Path(cypher_file).write_text("")  # empty file on total failure
    return False


def report(
    pairs: List[str],
    reporter_prompt: str,
    cypher_file: str,
    reporter_file: str,
    vec_file: str,
    news_file: str,
    query: str,
    timeout_sec: int = LLM_TIMEOUT_SEC,
) -> bool:
    """Generate report with failover between different LLM providers."""
    remaining_pairs = list(pairs)

    # Ensure cypher file exists
    if not Path(cypher_file).exists():
        Path(cypher_file).write_text("")

    cypher_content = Path(cypher_file).read_text()

    # Build reporter file content
    reporter_content = f"{reporter_prompt}\n<relations>\n{cypher_content}\n</relations>\n{query}\n\nUse the following data to answer:\n"
    Path(reporter_file).write_text(reporter_content)

    # Read input files for LLM
    try:
        reporter_content_full = Path(reporter_file).read_text()
        vec_content = Path(vec_file).read_text()
        full_input = reporter_content_full + "\n" + vec_content
    except Exception as e:
        logger.error(f"Could not read input files: {e}")
        Path(news_file).write_text("Nothing relevant found or generation failed.")
        return False

    while remaining_pairs:
        src = remaining_pairs[0]
        model = remaining_pairs[1]

        logger.info(f"Trying report with {src}: {model}")

        try:
            # Clear previous output
            Path(news_file).write_text("")

            if src == "ollama":
                result = subprocess.run(
                    ["ollama", "run", "--hidethinking", "--nowordwrap", "gpt-oss:120b"],
                    input=full_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, "ollama")

                # Filter ANSI codes
                filter_result = subprocess.run(
                    ["python3", "filter_ansi.py"],
                    input=result.stdout,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = filter_result.stdout if filter_result.returncode == 0 else result.stdout

            elif src == "gemini":
                result = subprocess.run(
                    ["python3", "gemini.py", model],
                    input=full_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, "gemini")
                output = result.stdout

            elif src == "mlx":
                result = subprocess.run(
                    ["python3", "mlxllm.py", "-", "--model", model],
                    input=full_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, "mlx")
                output = result.stdout

            else:
                logger.warning(f"Unknown report source: {src}")
                remaining_pairs = remaining_pairs[2:]
                continue

            if output.strip():
                Path(news_file).write_text(output)
                logger.info(f"Report succeeded with {src}: {model}")
                return True

        except subprocess.CalledProcessError as e:
            logger.warning(f"Report failed with {src}: {model} (exit {e.returncode})")
        except subprocess.TimeoutExpired:
            logger.error(f"Report timed out after {timeout_sec}s")
        except Exception as e:
            logger.error(f"Report failed with exception: {e}")

        # Remove failed pair, but keep if it's the last one
        if len(remaining_pairs) > 2:
            logger.info(f"Removing failed model {src}: {model} from retry list")
            remaining_pairs = remaining_pairs[2:]
        else:
            logger.error("Last model failed, stopping attempts")
            break

    logger.error("All report attempts failed.")
    Path(news_file).write_text("Nothing relevant found or generation failed.")
    return False


def main(
    startdate: str,
    filename: str,
    entity: str,
    query: str,
    svoprompt: str = "prompt/kgsvo.txt",
    workdir: str = "output",
    timeout: int = LLM_TIMEOUT_SEC,
    use_cache: bool = True,
):
    """
    Generate news reports from vector files using LLM analysis.

    Args:
        startdate: Date offset for the report
        filename: Base filename for output files
        entity: Entity name(s) to analyze
        query: Query string for the report
        svoprompt: Path to SVO prompt file (default: prompt/kgsvo.txt)
        workdir: Directory for input/output files (default: output)
        timeout: Timeout for LLM calls in seconds (default: 300)
        use_cache: Read cached svo from MongoDB instead of calling LLM (default: True)
    """
    # File paths
    vec = f"{workdir}/{filename}.vec"
    news = f"{workdir}/{filename}.md"
    cypherfile = f"{workdir}/{filename}.cypher"
    reporterfile = f"{workdir}/{filename}.reporter"
    barenews = f"{workdir}/{filename}"
    svo_prompt = svoprompt

    # Count entries
    count = count_ids(vec)
    logger.info(f"{filename}\t{startdate}\t{count}")

    # Exit if no entries
    if count == 0:
        logger.warning(f"No articles found for {filename}, skipping")
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

    # Model configurations - use only gpt-oss:120b via Ollama
    cypher_pairs = [
        "ollama",
        "gpt-oss:120b",
    ]

    report_pairs = [
        "ollama",
        "gpt-oss:120b",
    ]

    # Run cypher generation with timeout
    cypher(
        cypher_pairs,
        svo_prompt,
        vec,
        cypherfile,
        timeout_sec=timeout,
        use_cache=use_cache,
    )

    # Run report generation with timeout
    report(
        report_pairs,
        reporter_prompt,
        cypherfile,
        reporterfile,
        vec,
        news,
        query,
        timeout_sec=timeout,
    )


if __name__ == "__main__":
    fire.Fire(main)
