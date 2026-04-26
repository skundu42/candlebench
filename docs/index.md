---
title: Candlebench Documentation
description: Learn local ML model inspection, embeddings, similarity, and benchmarking with Candle.
---

# Candlebench Documentation

Candlebench is a beginner-friendly Rust command line lab for learning how local machine learning models are stored, loaded, tokenized, embedded, compared, and benchmarked with [Hugging Face Candle](https://github.com/huggingface/candle).

It is not just a benchmark program. It is a learning lab for practical ML systems work:

- what model files contain
- how text becomes tokens
- how tokens become vectors
- how semantic similarity works
- why model file formats matter
- how CPU, Apple Accelerate, and Apple Metal backends differ
- how to measure inference speed without fooling yourself

<div class="section-grid">
  <a class="section-card" href="{{ '/getting-started/' | relative_url }}">
    <strong>Getting Started</strong>
    <span>Build the project, run the CLI, and follow the suggested learning path.</span>
  </a>
  <a class="section-card" href="{{ '/concepts/' | relative_url }}">
    <strong>ML Concepts</strong>
    <span>Understand tokenization, embeddings, pooling, normalization, and similarity.</span>
  </a>
  <a class="section-card" href="{{ '/model-files/' | relative_url }}">
    <strong>Model Files</strong>
    <span>Inspect SafeTensors, GGUF metadata, tensor shapes, dtypes, and quantization.</span>
  </a>
  <a class="section-card" href="{{ '/commands/' | relative_url }}">
    <strong>CLI Commands</strong>
    <span>Use inspect, download, tokenize, embed, similarity, benchmark, and TUI commands.</span>
  </a>
  <a class="section-card" href="{{ '/performance/' | relative_url }}">
    <strong>Performance</strong>
    <span>Compare CPU, Accelerate, and Metal with latency and throughput measurements.</span>
  </a>
  <a class="section-card" href="{{ '/examples/' | relative_url }}">
    <strong>Examples</strong>
    <span>Run a complete beginner workflow from download to benchmark.</span>
  </a>
</div>

## Who This Is For

Candlebench is useful if you are:

- new to local ML inference
- learning Rust ML tooling
- trying to understand `.safetensors` or `.gguf` files
- exploring embeddings and semantic search
- comparing CPU and Apple Silicon performance
- building intuition before writing a larger LLM or RAG application

You do not need to understand transformer internals before using Candlebench. The goal is to make the concepts visible from the command line.

## Command Overview

```bash
candlebench inspect       # inspect SafeTensors or GGUF files
candlebench download      # download files from Hugging Face Hub
candlebench tokenize      # inspect tokenizer output
candlebench embed         # generate sentence embeddings
candlebench similarity    # compare embeddings with cosine similarity
candlebench bench-embed   # benchmark embedding inference
candlebench bench-matmul  # benchmark matrix multiplication
candlebench tui           # open terminal UI
```
