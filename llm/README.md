# LLM Processing Scripts

This directory contains shell scripts for LLM testing, SVO (Subject-Verb-Object) extraction, bias testing, and model benchmarking.

## Quick Reference

| Task | Script |
|------|--------|
| Test SVO extraction | `./svo_test.sh` |
| Large-scale SVO test | `./svo_big_test.sh` |
| Test left-bias prompts | `./lcr_left.sh` |
| Test center-bias prompts | `./lcr_cen.sh` |
| Test right-bias prompts | `./lcr_right.sh` |
| Full CoVe benchmark | `./biglcrcove.sh` |
| Start llama-server | `./start_llama-server.sh` |

## Shell Scripts

### SVO Extraction Testing

#### svo_test.sh
Tests SVO (Subject-Verb-Object) extraction using multiple Ollama models.

```bash
./svo_test.sh
# Reads: prompt/svo_test.xml, llms.txt (model list)
# Writes: out/svo_test.txt
# Runs each model in llms.txt with svo_test.xml prompt
```

**Requirements:**
- `llms.txt` - File containing model names (one per line)
- `prompt/svo_test.xml` - SVO extraction prompt
- Ollama running locally

**Example llms.txt:**
```
gemma3:27b
llama3.3:70b
qwen2.5:72b
```

#### svo_big_test.sh
Large-scale SVO testing with expanded model list.

```bash
./svo_big_test.sh
# Reads: prompt/svo_test.xml, llms_big.txt
# Writes: out/svo_bigtest.txt
# More comprehensive than svo_test.sh
```

### Bias Testing (LCR CoVe)

These scripts test Left-Center-Right bias detection using the CoVe (Chain of Verifiable Events) methodology.

#### lcr_left.sh
Tests left-bias prompt with models in llms.txt.

```bash
./lcr_left.sh
# Reads: prompt/lcr_inst.txt, prompt/left.txt, llms.txt
# Runs each model with left-bias prompt
```

#### lcr_cen.sh
Tests center-bias prompt with models in llms.txt.

```bash
./lcr_cen.sh
# Reads: prompt/lcr_inst.txt, prompt/center.txt, llms.txt
# Runs each model with center-bias prompt
```

#### lcr_right.sh
Tests right-bias prompt with models in llms.txt.

```bash
./lcr_right.sh
# Reads: prompt/lcr_inst.txt, prompt/right.txt, llms.txt
# Runs each model with right-bias prompt
```

#### lcr_cove_llms.sh
Generic LCR CoVe testing script.

```bash
./lcr_cove_llms.sh <cove_prompt> <bias_prompt>
# Example:
# ./lcr_cove_llms.sh prompt/lcr_CoVe.txt prompt/left.txt
```

#### lcrllms.sh
Runs all three LCR bias tests sequentially.

```bash
./lcrllms.sh
# Runs: lcr_left.sh, lcr_cen.sh, lcr_right.sh
# Collects results for comparison
```

#### biglcrcove.sh
Comprehensive CoVe benchmark across multiple prompt combinations.

```bash
./biglcrcove.sh
# Runs 9 combinations:
# - claudeopus_CoVe + left/center/right
# - gpt51highngrok41think + left/center/right
# - claudesonnet4520250929 + left/center/right
# Output: out/opus_l.txt, out/opus_c.txt, out/opus_r.txt, etc.
```

### Model Testing

#### test_gemcats.sh
Tests categorical sorting using Gemini.

```bash
./test_gemcats.sh
# Reads: titles_ids.txt (first 300), prompt/sort_titles.txt
# Runs: ../db/gemini.py
# Writes: tmp (categorization results)
```

#### test_cats.sh
Tests categorical processing.

```bash
./test_cats.sh
# Related to test_gemcats.sh
```

#### test_xtraction.sh
Tests extraction functionality.

```bash
./test_xtraction.sh
# Tests data extraction capabilities
```

#### testllmsvo.sh
Tests SVO with multiple LLM backends.

```bash
./testllmsvo.sh
# Tests SVO extraction across different LLM providers
```

### Server Management

#### start_llama-server.sh
Starts llama-server with specified model.

```bash
./start_llama-server.sh <model_path>

Example:
./start_llama-server.sh meta-llama/Llama-3.3-70B-Instruct

# Runs on: 127.0.0.1:8033
# Options: -c (context size), --host, --port
```

### Utility Scripts

#### rm_reporters.sh
Removes reporter output files.

```bash
./rm_reporters.sh
# Cleans up reporter.* files in output directory
```

### Docker Testing

#### docker/test.sh
Tests Docker-based LLM services.

```bash
cd docker && ./test.sh
# Tests Docker container functionality
```

## Python Scripts (Reference)

| Script | Documentation |
|--------|---------------|
| [bias_processor.py](bias_processor.py) | LLM-based bias processing |
| [geminize.py](../docs/geminize.md) | LLM processing pipeline |
| [validate_bias.py](validate_bias.py) | Bias validation |

### Available Python Services

| Script | Port | Description |
|--------|------|-------------|
| ollama_service.py | 8101 | Ollama REST API wrapper |
| gemini.py | - | Gemini API client |
| mlxllm.py | - | MLX LLM client |

## Prompt Files

The `prompt/` directory contains various prompts:

| Prompt | Purpose |
|--------|---------|
| `svo_test.xml` | SVO extraction |
| `lcr_inst.txt` | LCR instruction |
| `lcr_CoVe.txt` | Chain of Verifiable Events |
| `left.txt` / `center.txt` / `right.txt` | Bias direction prompts |
| `sort_titles.txt` | Title categorization |

## Configuration

### Required Environment Variables

```bash
# Ollama
export OLLAMA_HOST="localhost:11434"

# Gemini (optional)
export GEMINI_API_KEY="your_key"

# Groq (optional)
export GROQ_API_KEY="your_key"
```

### Model Lists

- `llms.txt` - Primary model list for testing
- `llms_big.txt` - Extended model list

## Quick Start

```bash
# 1. Create model list
echo "gemma3:27b" > llms.txt

# 2. Test SVO extraction
./svo_test.sh

# 3. Test bias detection
./lcr_left.sh
./lcr_cen.sh
./lcr_right.sh

# 4. Run full benchmark
./biglcrcove.sh
```

## Output

Test outputs are stored in:
- `out/` - Test results and logs

## See Also

- Main README: [../README.md](../README.md)
- Bias Processing: [../docs/bias_processor.md](../docs/bias_processor.md)
- LoRA Training: [../LoRA-train/README.md](../LoRA-train/README.md)

---

## mongo_rw.py

MongoDB field reader/writer tool for reading and writing individual fields in documents.

### Usage

```bash
# READ (default: output to stdout)
./mongo_rw.py read --field <field> --id <id>                    # single ID
./mongo_rw.py read --field <field> --id "id1,id2,id3"           # comma-separated
./mongo_rw.py read --field <field> --idfile <file>              # file: one ID per line
./mongo_rw.py read --field <field> --idfile -                    # stdin: one ID per line
./mongo_rw.py read --field <field> --id <id> --data <file>      # output to file
./mongo_rw.py read --field <field> --id <id> --data -            # output to stdout

# WRITE (default: read from stdin)
./mongo_rw.py write --field <field> --id <id>                    # read from stdin
./mongo_rw.py write --field <field> --id <id> --data -            # stdin explicit
./mongo_rw.py write --field <field> --id <id> --data <file>      # read from file
./mongo_rw.py write --field <field> --idfile <file> --data <data>  # batch: IDs from file, data from file
./mongo_rw.py write --field <field> --id <id> --force           # overwrite existing data
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--id` | MongoDB ID(s) - single or comma-separated (e.g., `"id1,id2"`) |
| `--idfile` | File with IDs (newline-separated, one per line) or `-` for stdin |
| `--field` | Field name to read/write (required) |
| `--data` | Data source/target: file path, `-` for stdin/stdout, or omit for default |
| `--force` | Overwrite existing field data on write |

### Examples

```bash
# Read a single field
./mongo_rw.py read --field title --id 696282dd5f8dd0157bb3d388

# Read bias JSON from multiple IDs
./mongo_rw.py read --field bias --id "id1,id2,id3"

# Read from file containing IDs (one per line)
./mongo_rw.py read --field title --idfile ids.txt

# Write data from stdin
echo '{"L": 0.5, "C": 0.3, "R": 0.2}' | ./mongo_rw.py write --field bias --id 696282dd5f8dd0157bb3d388

# Write from file with force to overwrite
./mongo_rw.py write --field bias --id 696282dd5f8dd0157bb3d388 --data bias.json --force

# Output to file instead of stdout
./mongo_rw.py read --field title --id 696282dd5f8dd0157bb3d388 --data output.txt

# Batch write with IDs from file
./mongo_rw.py write --field status --idfile ids.txt --data "processed"
```

### Tools

#### count_tokens.py
Count tokens in text files to estimate context window usage.

```bash
# Fast estimation (chars/4)
python tools/count_tokens.py article.txt

# Accurate count with tiktoken
python tools/count_tokens.py --accurate article.txt

# Show percentage of context window
python tools/count_tokens.py --context-window 128k article.txt

# Process multiple files
python tools/count_tokens.py file1.txt file2.txt file3.txt

# Read from stdin
cat article.txt | python tools/count_tokens.py --estimate
```

**Options:**
| Option | Description |
|--------|-------------|
| `--estimate` | Fast character/4 estimation (default) |
| `--accurate` | Use tiktoken for accurate count |
| `--encoder` | Tokenizer: cl100k_base, p50k_base, r50k_base (default: cl100k_base) |
| `--context-window` | Show percentage of context window (4k, 8k, 32k, 128k, 200k) |

**Common Context Windows:**
- `4k` - GPT-3.5, Claude Instant
- `8k` - GPT-4, Claude
- `32k` - GPT-4-32k, Claude 100k
- `128k` - GPT-4 Turbo, Claude 200k
- `200k` - Claude 200k

### Environment

```bash
export MONGO_URI="mongodb://root:example@localhost:27017"
```

### Notes

- For bias field writes, data is automatically parsed as JSON and stored as object
- Without `--force`, write will skip if field already has data
- Multiple IDs are processed in order, errors are reported at end
- JSON fields are output as JSON, other fields output as plain text