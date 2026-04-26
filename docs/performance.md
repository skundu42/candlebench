---
title: Performance
description: Benchmark embeddings, matrix multiplication, and backend choices.
permalink: /performance/
---

# Performance

Benchmarks teach the difference between:

- latency: how long one operation takes
- throughput: how much work is completed per second
- warmup iterations: untimed runs used to reduce first-run noise
- timed iterations: runs used for reported measurements
- batch size: how many inputs are processed at once

Candlebench includes:

- matrix multiplication benchmarks
- embedding throughput benchmarks

Matrix multiplication is the core operation behind many neural network layers, so it is a useful low-level performance check.

Embedding benchmarks measure an actual model forward pass, which is closer to real application work.

## Embedding Benchmarks

Run:

```bash
cargo run --release -- bench-embed \
  --text "Candlebench embedding benchmark sentence." \
  --warmup-iters 2 \
  --iters 20
```

Batch benchmark:

```bash
cargo run --release -- bench-embed \
  --text "first benchmark sentence" \
  --text "second benchmark sentence" \
  --text "third benchmark sentence" \
  --iters 20
```

Output fields:

- `batch size`: number of texts per forward pass
- `dimensions`: embedding vector size
- `tokens / iter`: total tokens in one batch
- `warmup iters`: untimed iterations
- `timed iters`: measured iterations
- `avg / iter`: average latency per batch
- `embeddings/sec`: throughput by number of embedded texts
- `tokens/sec`: throughput by token count

Why warmup matters:

First runs can include setup overhead such as memory allocation, cache effects, and backend initialization. Warmups reduce that noise.

## Matrix Multiplication

Matrix multiplication is a core operation behind neural network inference.

CPU benchmark:

```bash
cargo run --release -- bench-matmul \
  --size 1024 \
  --iters 10 \
  --backend cpu
```

Metal benchmark:

```bash
cargo run --release --features metal -- bench-matmul \
  --size 2048 \
  --iters 20 \
  --backend metal
```

Auto backend:

```bash
cargo run --release --features metal -- bench-matmul \
  --size 2048 \
  --iters 20 \
  --backend auto
```

For a square matrix multiply of size `N`, the rough work is:

```text
2 * N^3 floating point operations
```

Candlebench uses this estimate in `bench-matmul`:

```rust
let flops = 2.0 * (size as f64).powi(3);
let gflops = (flops / avg_seconds) / 1e9;
```

GFLOP/s means billions of floating point operations per second. It is useful for comparing compute throughput, but it is not the only thing that matters for real models.

## Backend Selection

Candlebench supports `cpu`, `metal`, and `auto` where available.

Supported backend choices:

- `cpu`
- `metal`
- `auto`

Cargo feature choices:

- default CPU build
- `accelerate` for Apple Accelerate CPU BLAS
- `metal` for Candle's Metal backend

Conceptually:

- CPU is simple and often strong for small workloads.
- Apple Accelerate can improve CPU BLAS operations.
- Metal can improve larger tensor workloads on Apple Silicon.
- Device transfer and launch overhead can dominate small examples.

This is why a tiny benchmark can make CPU look better, while a larger batch or matrix may favor Metal. Benchmark the workload shape you actually expect to use.

## Latency, Throughput, And Warmup

Embedding benchmarks report several related measurements:

| Metric | Meaning |
| --- | --- |
| `avg / iter` | average latency for one batch |
| `embeddings/sec` | how many texts are embedded per second |
| `tokens/sec` | how many real tokens are processed per second |
| `warmup iters` | untimed runs used to reduce first-run noise |

Latency and throughput are not the same thing.

If batch size increases, throughput may improve because the device gets more work per call. Latency for the whole batch may also increase. The right choice depends on whether your application is interactive, offline, or serving many requests at once.

Use consistent settings when comparing:

```bash
cargo run --release -- bench-embed \
  --text "same benchmark sentence" \
  --warmup-iters 2 \
  --iters 20
```

Then change one variable at a time:

- backend
- batch size
- model
- sequence length
- build feature
