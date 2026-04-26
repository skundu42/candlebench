---
title: Getting Started
description: Build Candlebench and run the first commands.
permalink: /getting-started/
---

# Getting Started

## Setup

Build the default CPU version:

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

Run the CLI help:

```bash
cargo run -- --help
```

During development, examples use:

```bash
cargo run -- <command>
```

That means:

```bash
cargo run -- inspect ./model.safetensors
```

is equivalent to running the compiled binary:

```bash
candlebench inspect ./model.safetensors
```

## Apple Silicon Builds

For Apple Accelerate:

```bash
cargo build --features accelerate
```

For Apple Metal:

```bash
cargo build --features metal
```

For optimized benchmark runs, use `--release`:

```bash
cargo run --release --features metal -- bench-matmul --backend metal
```

## Suggested Learning Path

If you are new, follow this order:

1. Download a small embedding model.
2. Inspect the `model.safetensors` file.
3. Tokenize a short sentence.
4. Generate one embedding.
5. Compare two similar sentences.
6. Compare unrelated sentences.
7. Benchmark embedding inference.
8. Benchmark matrix multiplication.
9. Try CPU, Accelerate, and Metal if you are on Apple Silicon.
10. Inspect a GGUF file and compare its metadata to SafeTensors.
