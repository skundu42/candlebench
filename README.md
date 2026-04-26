# Candlebench

A Rust CLI for inspecting, downloading, tokenizing, embedding, and benchmarking local ML model files with
[Hugging Face Candle](https://github.com/huggingface/candle).

This lab supports:

- Inspecting `.safetensors` tensor metadata without loading tensor data
- Inspecting GGUF metadata, quantized tensor shapes, dtypes, offsets, and rough byte sizes
- Downloading model files from the Hugging Face Hub cache
- Tokenizing text with a local `tokenizer.json`
- Running BERT-style sentence embedding inference with Candle
- Computing pairwise embedding similarity
- Benchmarking embedding throughput and latency
- Viewing model summaries in a `ratatui` terminal UI
- Running CPU, Metal, or auto-selected matmul benchmarks

## Setup

```bash
cargo build
```

For Apple Silicon Metal builds:

```bash
cargo build --features metal
```

For Apple Accelerate CPU BLAS builds:

```bash
cargo build --features accelerate
```

## Inspect Models

Inspect a SafeTensors file:

```bash
cargo run -- inspect ./model.safetensors
```

Inspect a GGUF file:

```bash
cargo run -- inspect ./model.gguf
```

Show JSON:

```bash
cargo run -- inspect ./model.gguf --json
```

Limit table rows:

```bash
cargo run -- inspect ./model.safetensors --limit 50
```

## Hugging Face Hub Downloads

Download files into the standard Hugging Face cache:

```bash
cargo run -- download \
  --repo sentence-transformers/all-MiniLM-L6-v2 \
  config.json tokenizer.json model.safetensors
```

Use a pinned revision and JSON output:

```bash
cargo run -- download \
  --repo sentence-transformers/all-MiniLM-L6-v2 \
  --revision main \
  --json \
  config.json tokenizer.json model.safetensors
```

The command honors the usual `hf-hub` environment and cache settings, including
`HF_HOME` and `HF_ENDPOINT`.

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

Bare repo names are resolved under `sentence-transformers/`, so the default model is
`sentence-transformers/all-MiniLM-L6-v2`. Repeat `--text` for a batch:

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

Current embedding support is intentionally narrow: it targets BERT-compatible
`model.safetensors` checkpoints such as common `sentence-transformers` models.
GGUF embedding inference is not implemented.

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

## Embedding Benchmarks

Benchmark embedding inference after loading the model once:

```bash
cargo run --release -- bench-embed \
  --text "Candlebench embedding benchmark sentence." \
  --warmup-iters 2 \
  --iters 20
```

Repeat `--text` to benchmark a larger batch:

```bash
cargo run --release -- bench-embed \
  --text "first benchmark sentence" \
  --text "second benchmark sentence" \
  --text "third benchmark sentence" \
  --iters 20
```

On Apple Silicon with Metal enabled:

```bash
cargo run --release --features metal -- bench-embed \
  --backend metal \
  --text "Candlebench embedding benchmark sentence." \
  --iters 20
```

## TUI

Open the terminal dashboard:

```bash
cargo run -- tui
```

Open a file summary directly:

```bash
cargo run -- tui ./model.gguf
```

Use `j`/`k` or arrow keys to scroll and `q` or `Esc` to quit.

## Benchmarks

Run a CPU matmul benchmark:

```bash
cargo run --release -- bench-matmul --size 1024 --iters 10 --backend cpu
```

Run an auto-selected backend benchmark. On a Metal-enabled Apple Silicon build this
selects Metal, otherwise it falls back to CPU:

```bash
cargo run --release --features metal -- bench-matmul --size 2048 --iters 20 --backend auto
```

Run Metal explicitly:

```bash
cargo run --release --features metal -- bench-matmul --size 2048 --iters 20 --backend metal
```

Run an Accelerate-backed CPU benchmark:

```bash
cargo run --release --features accelerate -- bench-matmul --size 2048 --iters 20 --backend cpu
```

## Apple Silicon Notes

There are two useful Apple Silicon paths:

- `--features accelerate` keeps tensors on CPU and uses Apple Accelerate for supported BLAS operations.
- `--features metal` enables Candle's Metal backend and allows `--backend metal` or `--backend auto`.

Benchmark both for your workload. Small matrices can favor CPU due to launch and transfer overhead,
while larger matrix sizes are more likely to show Metal gains. Use `--release`, keep `--iters`
consistent, and compare the reported average iteration time and rough GFLOP/s.
