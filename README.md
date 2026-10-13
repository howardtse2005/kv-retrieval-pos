# Target Placement in Synthetic KV Retrieval

This project implements the experiment described in `plan.md`. It generates
reproducible synthetic subject-key-value haystacks, evaluates local Ollama
models, scores responses, and produces summary tables and figures.

Records are serialized as natural-language statements such as
`Subject_0062's date of birth is 4 March 1950`. High-similarity examples use
one consistent key type per haystack, while rotating through all five key types
across examples. Low-similarity examples mix all five key types. Values are
generated into key-specific pools and shuffled before records are built, so
numeric/date values are not assigned in index order while remaining
type-correct and unique.
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

Generate the 18 datasets (100 examples per size, similarity condition, and
primary target position):

```bash
python generate_dataset.py --output-dir data
```

Run a 180-trial pilot (10 examples per dataset):

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

Every trial stores the complete prompt, raw response, scoring decision,
sampling settings, model identifier, and timestamp. Use `--resume` to skip
trial IDs already present in an existing result file. Dataset generation is
seeded (`42` by default), and each example records its own seed-independent
position and target metadata.

Inference disables model thinking (`think=False`) and requires exactly one
line in the form `ANSWER: <one answer>`. Responses are scored as `correct`,
`wrong_answer`, `wrong_format`, or `no_answer`; Ollama/API failures are
recorded separately as `api_error`.

The benchmark intentionally does not silently replace an unavailable model:
the fallback substitution is reported in the run metadata and console output.
API failures are recorded as `api_error` trials rather than being discarded.
