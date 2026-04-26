---
title: Examples
description: Complete beginner workflows for Candlebench.
permalink: /examples/
---

# Examples

## Worked Beginner Example

Download the files:

```bash
cargo run -- download \
  --repo sentence-transformers/all-MiniLM-L6-v2 \
  config.json tokenizer.json model.safetensors
```

Inspect the weights:

```bash
cargo run -- inspect \
  ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/model.safetensors \
  --limit 5
```

Tokenize text:

```bash
cargo run -- tokenize \
  ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/tokenizer.json \
  "how to bake sourdough bread"
```

Embed text:

```bash
cargo run -- embed \
  --repo all-MiniLM-L6-v2 \
  --text "how to bake sourdough bread"
```

Compare meaning:

```bash
cargo run -- similarity \
  --repo all-MiniLM-L6-v2 \
  --text "how to bake sourdough bread" \
  --text "steps for making homemade bread" \
  --text "GPU benchmark for matrix multiplication"
```

Benchmark:

```bash
cargo run --release -- bench-embed \
  --repo all-MiniLM-L6-v2 \
  --text "how to bake sourdough bread" \
  --iters 20
```

## Inspect SafeTensors

Inspect a SafeTensors file:

```bash
cargo run -- inspect ./model.safetensors
```

Show JSON:

```bash
cargo run -- inspect ./model.safetensors --json
```

Limit table rows:

```bash
cargo run -- inspect ./model.safetensors --limit 50
```

## Inspect GGUF

Inspect a GGUF file:

```bash
cargo run -- inspect ./model.gguf
```

Show JSON:

```bash
cargo run -- inspect ./model.gguf --json
```

## Tokenizers

Encode text with a local tokenizer:

```bash
cargo run -- tokenize ./tokenizer.json "Candle makes local inference practical."
```

Emit token ids, token strings, masks, type ids, and offsets as JSON:

```bash
cargo run -- tokenize ./tokenizer.json "hello world" --json
```

## Embeddings

Run sentence embeddings with a BERT-style sentence-transformer model:

```bash
cargo run -- embed --text "Candle makes local inference practical."
```

Repeat `--text` for a batch:

```bash
cargo run -- embed \
  --repo all-MiniLM-L6-v2 \
  --text "first sentence" \
  --text "second sentence" \
  --json
```

Use local files by passing all three model file paths:

```bash
cargo run -- embed \
  --config-file ./config.json \
  --tokenizer-file ./tokenizer.json \
  --weights-file ./model.safetensors \
  --text "local model files"
```

## Embedding Similarity

Compute pairwise cosine similarity between two or more texts:

```bash
cargo run -- similarity \
  --text "Candle runs local model inference." \
  --text "Local embeddings can run with Candle." \
  --text "The recipe uses fresh tomatoes."
```

Emit sorted pairwise scores as JSON:

```bash
cargo run -- similarity \
  --text "first sentence" \
  --text "second sentence" \
  --json
```
