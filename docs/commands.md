---
title: CLI Commands
description: Command reference for Candlebench inspect, download, tokenize, embed, similarity, benchmark, and TUI commands.
permalink: /commands/
---

# CLI Commands

## Inspect SafeTensors

SafeTensors is a common format for storing model weights safely and efficiently.

Run:

```bash
cargo run -- inspect ./model.safetensors
```

Limit rows:

```bash
cargo run -- inspect ./model.safetensors --limit 10
```

JSON output:

```bash
cargo run -- inspect ./model.safetensors --json
```

What you learn from the output:

- `file size`: total file size on disk
- `tensors`: number of tensors in the file
- `tensor data`: total bytes used by tensor payloads
- `rough parameters`: rough count of stored tensor elements
- `dtype counts`: how many tensors use each dtype
- `largest tensors`: the tensors consuming the most space

## Inspect GGUF

GGUF is often used for quantized local LLMs.

Run:

```bash
cargo run -- inspect ./model.gguf
```

Show more metadata and tensors:

```bash
cargo run -- inspect ./model.gguf --limit 80
```

JSON output:

```bash
cargo run -- inspect ./model.gguf --json
```

What GGUF inspection teaches:

- model architecture metadata
- quantization type
- tokenizer metadata stored inside the file
- tensor layout
- tensor data offset
- rough memory shape of the model

## Download From Hugging Face Hub

Candlebench can download files into the local Hugging Face cache.

Download the default files for a sentence-transformer model:

```bash
cargo run -- download \
  --repo sentence-transformers/all-MiniLM-L6-v2 \
  config.json tokenizer.json model.safetensors
```

Download from a specific revision:

```bash
cargo run -- download \
  --repo sentence-transformers/all-MiniLM-L6-v2 \
  --revision main \
  config.json tokenizer.json model.safetensors
```

JSON output:

```bash
cargo run -- download \
  --repo sentence-transformers/all-MiniLM-L6-v2 \
  --json \
  config.json
```

Useful environment variables:

```bash
HF_HOME=/path/to/cache
HF_ENDPOINT=https://huggingface.co
```

## Tokenize Text

Run:

```bash
cargo run -- tokenize ./tokenizer.json "hello world"
```

JSON output:

```bash
cargo run -- tokenize ./tokenizer.json "hello world" --json
```

Example fields:

```json
{
  "id": 7592,
  "token": "hello",
  "type_id": 0,
  "attention_mask": 1,
  "offset": [0, 5]
}
```

Field meanings:

- `id`: numeric token ID passed into the model
- `token`: readable token string
- `type_id`: segment ID, often `0` for single sentence inputs
- `attention_mask`: `1` for real token, `0` for padding
- `offset`: character range in the original text

## Generate Embeddings

Run:

```bash
cargo run -- embed --text "Candle makes local inference practical."
```

Use a specific model:

```bash
cargo run -- embed \
  --repo all-MiniLM-L6-v2 \
  --text "first sentence"
```

The short repo name:

```text
all-MiniLM-L6-v2
```

is resolved to:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Batch multiple texts:

```bash
cargo run -- embed \
  --repo all-MiniLM-L6-v2 \
  --text "first sentence" \
  --text "second sentence" \
  --json
```

Use local files:

```bash
cargo run -- embed \
  --config-file ./config.json \
  --tokenizer-file ./tokenizer.json \
  --weights-file ./model.safetensors \
  --text "local model files"
```

Current embedding support is intentionally narrow: it targets BERT-compatible `model.safetensors` checkpoints such as common `sentence-transformers` models. GGUF embedding inference is not implemented.

## Compare Similarity

Run:

```bash
cargo run -- similarity \
  --repo all-MiniLM-L6-v2 \
  --text "first sentence" \
  --text "second sentence"
```

Example output:

```text
#1/#2  cosine=0.831086
```

Compare three texts:

```bash
cargo run -- similarity \
  --repo all-MiniLM-L6-v2 \
  --text "how to bake sourdough bread" \
  --text "steps for making homemade bread" \
  --text "GPU benchmark for matrix multiplication"
```

JSON output:

```bash
cargo run -- similarity \
  --text "first sentence" \
  --text "second sentence" \
  --json
```

## Use The TUI

Open the dashboard:

```bash
cargo run -- tui
```

Open a file:

```bash
cargo run -- tui ./model.safetensors
```

or:

```bash
cargo run -- tui ./model.gguf
```

Controls:

| Key | Action |
| --- | --- |
| `j` or Down | scroll down |
| `k` or Up | scroll up |
| PageDown | scroll faster down |
| PageUp | scroll faster up |
| `q` or Esc | quit |

## JSON Output

Several commands support `--json`.

Use JSON when you want to:

- pipe results into another program
- save model metadata
- build scripts around Candlebench
- compare benchmark results later
