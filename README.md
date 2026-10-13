# Target Placement in Synthetic KV Retrieval

This project implements the experiment described in `plan.md`. It generates
reproducible synthetic subject-key-value haystacks, evaluates local Ollama
models, scores responses, and produces summary tables and figures.

Records are serialized as natural-language statements such as
`Subject_9f2c7a1b4e8d6035c1a9f0b7e2d4a6c8's date of birth is 4 March 1950`.
Subject IDs use a `Subject_` prefix followed by a unique 128-bit hexadecimal
identifier. High-similarity examples use
one consistent key type per haystack, while rotating through all 100 key types
across examples. Low-similarity examples mix the 100 key types. Values are
generated into key-specific pools and shuffled before records are built. Dates remain human-readable; every non-date value keeps a
human-readable prefix and appends a unique 128-bit hexadecimal suffix
(32 hex characters) after a dash, such as `red-9f2c...`.
Each non-date key has exactly three reusable base values; for example,
favorite colors cycle through `red`, `purple`, and `green`.
Low-similarity datasets omit the hexadecimal suffix: size 75 uses 75 distinct
categories, size 140 uses all 100 categories with at most two records per
category, and size 300 uses every category exactly three times. High-similarity
datasets retain hexadecimal suffixes.
Subject IDs are also independently shuffled in every haystack and remain
unique, so neither subjects nor values reveal an ordered index.

## Setup

The intended environment is `kv_pos` with Python 3.12:

```bash
conda activate kv_pos
pip install -r requirements.txt
```

Ollama must be running locally. The benchmark selects
`qwen3.5:9b` when it is installed and otherwise uses `qwen3:14b` as the
documented proxy.

## Workflow

Generate the 30 datasets (100 examples per size, similarity condition, and
target position):

```bash
python generate_dataset.py --output-dir data
```

Run a 300-trial pilot (10 examples per dataset):

```bash
python run_benchmark.py --data-dir data --output-dir results --pilot
```

Run the full benchmark for all available models:

```bash
python run_benchmark.py --data-dir data --output-dir results
```

Analyse one or more result files:

```bash
python analyze_results.py --results-dir results --output-dir analysis
```

Every trial stores the raw system instruction in `system_prompt`, the raw
response, scoring decision, sampling settings, model identifier, and timestamp.
The record-specific user prompt is sent to Ollama but is not stored in results.
Use `--resume` to skip
trial IDs already present in an existing result file. Dataset generation is
seeded (`42` by default), and each example records its own seed-independent
position and target metadata.

Inference disables model thinking (`think=False`) and requires exactly one
line in the form `ANSWER: <one answer>`. Responses are scored as `correct`,
`wrong_answer`, `wrong_format`, or `no_answer`; Ollama/API failures are
recorded separately as `api_error`.
The system prompt requires copying the complete literal value, including
numeric or textual suffixes such as `red-042`, rather than returning only a
semantic prefix such as `red`.

The benchmark intentionally does not silently replace an unavailable model:
the fallback substitution is reported in the run metadata and console output.
API failures are recorded as `api_error` trials rather than being discarded.
