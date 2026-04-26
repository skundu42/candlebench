# Candlebench

Candlebench is a beginner-friendly Rust command line tool for learning how local machine learning models are stored, loaded, tokenized, embedded, compared, and benchmarked with [Hugging Face Candle](https://github.com/huggingface/candle).

It is not just a benchmark program. It is a learning lab for practical ML systems work:

- what model files contain
- how text becomes tokens
- how tokens become vectors
- how semantic similarity works
- why model file formats matter
- how CPU, Apple Accelerate, and Apple Metal backends differ
- how to measure inference speed without fooling yourself

## Who This Is For

Candlebench is useful if you are:

- new to local ML inference
- learning Rust ML tooling
- trying to understand `.safetensors` or `.gguf` files
- exploring embeddings and semantic search
- comparing CPU and Apple Silicon performance
- building intuition before writing a larger LLM or RAG application

You do not need to understand transformer internals before using Candlebench. The goal is to make the concepts visible from the command line.

## What Candlebench Teaches

### 1. Model Files

Most local ML work starts with model files. Candlebench helps you inspect them before trying to run inference.

It can inspect:

- `.safetensors`
- `.gguf`

You can learn:

- how many tensors a model has
- tensor names
- tensor shapes
- data types like `F32`, `F16`, `I64`, or GGUF quantized types
- rough parameter count
- approximate tensor memory size
- GGUF metadata such as architecture, tokenizer data, context length, and quantization information

Why this matters:

- model size affects memory usage
- tensor dtype affects speed and precision
- tensor shapes reveal architecture structure
- GGUF metadata explains how a quantized model was packaged

### 2. Tokenization

Models do not read raw text directly. Text is converted into token IDs.

For example, this text:

```text
hello world
```

might become token IDs like:

```text
[101, 7592, 2088, 102]
```

Candlebench can show:

- token IDs
- token strings
- attention masks
- type IDs
- text offsets
- padding tokens

Important beginner concepts:

- A token can be a word, part of a word, punctuation, or a special marker.
- `[CLS]` and `[SEP]` are special BERT-style tokens.
- `[PAD]` fills unused positions in fixed-length batches.
- The attention mask tells the model which tokens are real and which are padding.

### 3. Embeddings

An embedding is a list of numbers that represents text.

For `sentence-transformers/all-MiniLM-L6-v2`, each sentence becomes a vector with 384 numbers.

Example:

```text
"how to bake sourdough bread"
```

becomes:

```text
[0.031, -0.082, 0.014, ... 384 values total]
```

You usually do not read the individual numbers directly. You use them for comparison:

- semantic search
- duplicate detection
- clustering
- recommendations
- retrieval augmented generation, also called RAG

### 4. Semantic Similarity

Once text is embedded, Candlebench can compare two vectors using cosine similarity.

Cosine similarity answers:

```text
Are these two texts pointing in a similar semantic direction?
```

Typical interpretation:

| Score | Meaning |
| --- | --- |
| close to `1.0` | very similar |
| around `0.5` | somewhat related |
| near `0.0` | mostly unrelated |
| below `0.0` | strongly different in embedding space |

Example:

```text
first sentence
second sentence
```

might score high because the structure and meaning are close.

Example:

```text
first sentence
how to bake sourdough bread
```

should score much lower because the topics are different.

### 5. Benchmarking

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

### 6. Backends

Candlebench can teach how different compute backends affect performance.

Supported backend choices:

- `cpu`
- `metal`
- `auto`

Cargo feature choices:

- default CPU build
- `accelerate` for Apple Accelerate CPU BLAS
- `metal` for Candle's Metal backend

Backend lessons:

- CPU can be faster for small jobs because there is less overhead.
- Metal can be faster for larger tensor workloads.
- Accelerate can improve CPU matrix operations on Apple Silicon.
- Always benchmark your actual workload, not only a synthetic example.

## ML Concepts In Depth

This section explains the machine learning ideas behind the commands. The goal is to connect each visible CLI output to the actual data structures that move through an embedding model.

The high-level pipeline is:

```text
text
  -> tokenizer
  -> token ids, type ids, attention mask
  -> tensors on a device
  -> BERT forward pass
  -> token-level hidden states
  -> masked mean pooling
  -> optional L2 normalization
  -> embedding vectors
  -> cosine similarity or benchmark metrics
```

### Model Artifacts: Config, Tokenizer, Weights

A Hugging Face model repository is usually not a single model file. For a BERT-style sentence-transformer, Candlebench needs three compatible artifacts:

| File | Role |
| --- | --- |
| `config.json` | Describes architecture settings such as hidden size, number of layers, attention heads, and vocabulary size. |
| `tokenizer.json` | Converts text into token IDs using the same vocabulary the model saw during training. |
| `model.safetensors` | Stores the learned tensor values, also called weights or parameters. |

These files must match. If the tokenizer uses a different vocabulary from the weights, token ID `7592` may no longer mean the same subword the model learned. If the config says the hidden size is 384 but the weights contain tensors shaped for another hidden size, model loading should fail.

In Candlebench, the embedding loader resolves those files from either local paths or the Hugging Face cache:

```rust
let config = read_config(&config_path)?;
let tokenizer = load_tokenizer(&tokenizer_path, max_length)?;
let weights = VarBuilder::from_mmaped_safetensors(
    &[weights_path.as_path()],
    DTYPE,
    &device,
)?;
let model = BertModel::load(weights, &config)?;
```

Conceptually:

- the config defines the model skeleton
- the weights fill that skeleton with learned numbers
- the tokenizer translates human text into the integer IDs expected by that skeleton

### Tensor Shapes And Parameters

A tensor is a rectangular block of numbers with a shape. Shape tells you how the model organizes information.

Example output:

```text
embeddings.word_embeddings.weight  F32  [30522, 384]
```

This means:

- `30522` rows: one learned vector per vocabulary entry
- `384` columns: each token is represented by 384 numeric features
- total parameters: `30522 * 384 = 11,720,448`

A parameter is one learned number. For an `F32` tensor, each parameter uses 4 bytes. A rough byte estimate is:

```text
parameters * bytes_per_value
```

For the example above:

```text
11,720,448 * 4 = 46,881,792 bytes
```

This is why shape and dtype matter before you run inference. They determine memory pressure, bandwidth needs, and often speed.

### SafeTensors Versus GGUF

SafeTensors and GGUF both store model data, but they are optimized for different workflows.

| Format | Common Use | What Candlebench Inspects |
| --- | --- | --- |
| SafeTensors | Hugging Face model weights, often full precision or half precision | tensor names, dtypes, shapes, rough parameter counts, payload bytes |
| GGUF | Quantized local LLM files used by llama.cpp-style runtimes | metadata, tokenizer data, architecture fields, quantized tensor dtypes, offsets |

SafeTensors is useful when the model code already knows the architecture. The file mostly says, "here are named tensors with shapes and dtypes."

GGUF carries more runtime metadata in the same file. A GGUF file can include architecture keys, tokenizer settings, context length, quantization information, and tensor offsets. That makes it convenient for local LLM runtimes, especially when distributing quantized models.

### Quantization

Quantization stores model weights with fewer bits than standard floating point formats.

For example:

| DType | Rough Meaning | Typical Tradeoff |
| --- | --- | --- |
| `F32` | 32-bit float | highest storage cost, strong precision |
| `F16` | 16-bit float | lower memory, usually good inference quality |
| `Q8_0` | 8-bit quantized | smaller, often still high quality |
| `Q4_0`, `Q4K` | 4-bit quantized | much smaller, more quality risk |

Quantization mainly reduces model size and memory bandwidth. It does not automatically make every workload faster. Some hardware is excellent at dense floating point math, while some quantized formats require unpacking or specialized kernels.

### Tokenization: Text Becomes Integer IDs

Neural networks do not process raw strings. The tokenizer turns text into integer IDs.

Command:

```bash
cargo run -- tokenize ./tokenizer.json "Candle makes local inference practical."
```

Typical fields:

```json
{
  "id": 101,
  "token": "[CLS]",
  "type_id": 0,
  "attention_mask": 1,
  "offset": [0, 0]
}
```

The important pieces are:

- `id`: the integer passed into the embedding table
- `token`: the human-readable token string
- `type_id`: which segment the token belongs to, usually `0` for single-sentence inputs
- `attention_mask`: whether this position is real text or padding
- `offset`: where the token came from in the original string

Candlebench uses the tokenizer library to produce those arrays:

```rust
let encoding = tokenizer.encode(text, true)?;

let ids = encoding.get_ids();
let tokens = encoding.get_tokens();
let type_ids = encoding.get_type_ids();
let attention_mask = encoding.get_attention_mask();
let offsets = encoding.get_offsets();
```

The `true` argument asks the tokenizer to add special tokens when the tokenizer configuration defines them. For BERT-like models, that usually means tokens such as `[CLS]` and `[SEP]`.

### Batching, Padding, And Truncation

Models run most efficiently on rectangular tensors. A batch of text must become a matrix shaped like:

```text
[batch_size, sequence_length]
```

Different input texts naturally have different token counts, so shorter examples are padded.

Example:

```text
"hello world"                         -> 4 real tokens
"Candle makes local inference useful" -> 8 real tokens
```

After padding, both rows may have length 8:

```text
input_ids:
[
  [101, 7592, 2088, 102, 0, 0, 0, 0],
  [101, 13541, 3084, 2334, 28937, 6179, 102, 0]
]

attention_mask:
[
  [1, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 0]
]
```

The attention mask prevents padding from being treated as meaningful text.

Candlebench configures truncation with `--max-length` and enables padding for embedding batches:

```rust
tokenizer.with_truncation(Some(TruncationParams {
    max_length,
    ..Default::default()
}))?;

tokenizer.with_padding(Some(PaddingParams::default()));
```

Practical lesson: token count is a better cost signal than character count. A short-looking string with unusual words, code, or punctuation can tokenize into many pieces.

### Forward Pass: IDs Become Hidden States

After tokenization, Candlebench converts the batch arrays into Candle tensors on the selected device:

```rust
let input_ids = Tensor::new(input_ids, &device)?;
let token_type_ids = Tensor::new(token_type_ids, &device)?;
let attention_mask = Tensor::new(attention_mask, &device)?;
```

Then it runs the BERT model:

```rust
let hidden_states = model.forward(
    &input_ids,
    &token_type_ids,
    Some(&attention_mask),
)?;
```

The result is not yet one vector per sentence. It is one vector per token:

```text
[batch_size, sequence_length, hidden_size]
```

For `all-MiniLM-L6-v2`, the hidden size is 384. If the batch has 2 texts and each row is padded to 12 tokens, the hidden states are shaped roughly like:

```text
[2, 12, 384]
```

### Pooling: Token Vectors Become Sentence Vectors

Sentence-transformer models need one vector per input text. Candlebench uses masked mean pooling:

```rust
let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
let summed = hidden_states.broadcast_mul(&mask)?.sum(1)?;
let counts = mask.sum(1)?.clamp(1e-12f32, f32::MAX)?;
let sentence_embeddings = summed.broadcast_div(&counts)?;
```

In plain language:

1. Convert the attention mask to floats.
2. Add a third dimension so it can multiply token vectors.
3. Zero out padded token vectors.
4. Sum the remaining token vectors.
5. Divide by the number of real tokens.

The mask is essential. Without it, padding tokens would pull the average away from the actual sentence meaning.

### L2 Normalization

By default, Candlebench normalizes each embedding to unit length:

```rust
let norm = pooled
    .sqr()?
    .sum_keepdim(1)?
    .sqrt()?
    .clamp(1e-12f32, f32::MAX)?;

let normalized = pooled.broadcast_div(&norm)?;
```

L2 normalization changes the vector length to `1.0` while preserving direction.

Why this matters:

- cosine similarity compares direction
- normalized embeddings make dot product and cosine similarity equivalent
- retrieval systems often normalize embeddings before indexing

You can disable this with:

```bash
cargo run -- embed \
  --text "Candle makes local inference practical." \
  --no-normalize
```

### Cosine Similarity

Cosine similarity compares vector direction:

```text
cosine(a, b) = dot(a, b) / (length(a) * length(b))
```

A minimal Rust version looks like:

```rust
fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for (&a, &b) in left.iter().zip(right) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}
```

Use it when you want semantic ranking:

```bash
cargo run -- similarity \
  --text "how to bake sourdough bread" \
  --text "steps for making homemade bread" \
  --text "GPU benchmark for matrix multiplication"
```

The two bread-related texts should rank higher because their embeddings point in a more similar direction.

### Matrix Multiplication In Neural Networks

A large part of neural network inference is matrix multiplication. Dense layers, attention projections, and feed-forward networks all use variants of:

```text
C = A x B
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

This is a synthetic benchmark. It helps you understand raw compute throughput, but it does not capture every real model bottleneck. Real inference can also be limited by memory bandwidth, kernel launch overhead, tokenizer cost, shape choices, and device transfer costs.

### Latency, Throughput, And Warmup

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

### Backend Selection

Candlebench supports `cpu`, `metal`, and `auto` where available.

Conceptually:

- CPU is simple and often strong for small workloads.
- Apple Accelerate can improve CPU BLAS operations.
- Metal can improve larger tensor workloads on Apple Silicon.
- Device transfer and launch overhead can dominate small examples.

This is why a tiny benchmark can make CPU look better, while a larger batch or matrix may favor Metal. Benchmark the workload shape you actually expect to use.

## Install And Build

Clone the repository:

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO
```

Build the default CPU version:

```bash
cargo build
```

Run the CLI help:

```bash
cargo run -- --help
```

After installation or a release build, the binary name is:

```bash
candlebench
```

During development, examples in this guide use:

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

Example interpretation:

```text
embeddings.word_embeddings.weight  F32  [30522, 384]
```

This means the model has an embedding table with:

- 30,522 vocabulary entries
- 384 dimensions per token
- `F32` storage

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

Common GGUF dtype examples:

| Type | Meaning |
| --- | --- |
| `F32` | 32-bit floating point |
| `F16` | 16-bit floating point |
| `BF16` | bfloat16 |
| `Q4_0`, `Q4K` | 4-bit quantized formats |
| `Q5K`, `Q6K`, `Q8_0` | larger quantized formats |

Beginner rule of thumb:

- lower-bit quantization usually uses less memory
- lower-bit quantization can lose some accuracy
- higher-bit or floating point formats usually preserve more quality

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

What this teaches:

- Hugging Face models are repositories
- model repos contain multiple files
- Candlebench can reuse cached downloads
- config, tokenizer, and weights are separate pieces

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

Why tokenization matters:

- token count controls compute cost
- long text may be truncated
- padding affects batching
- the tokenizer must match the model weights

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

Important output fields:

- `repo`: model used
- `device`: where inference ran
- `normalized`: whether L2 normalization was applied
- `token_count`: how many tokens the text became
- `dimensions`: embedding vector size
- `embedding`: the vector itself

Beginner note:

The raw embedding numbers are not meant to be read like a sentence. They are useful because similar meanings produce vectors that point in similar directions.

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

This means the texts are quite similar.

Try an unrelated comparison:

```bash
cargo run -- similarity \
  --repo all-MiniLM-L6-v2 \
  --text "first sentence" \
  --text "how to bake sourdough bread"
```

Example output:

```text
#1/#2  cosine=0.137627
```

This means the texts are mostly unrelated.

Compare three texts:

```bash
cargo run -- similarity \
  --repo all-MiniLM-L6-v2 \
  --text "how to bake sourdough bread" \
  --text "steps for making homemade bread" \
  --text "GPU benchmark for matrix multiplication"
```

Expected lesson:

- the two bread-related texts should rank highest
- the GPU benchmark text should rank lower against bread text

JSON output:

```bash
cargo run -- similarity \
  --text "first sentence" \
  --text "second sentence" \
  --json
```

## Benchmark Embeddings

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

## Benchmark Matrix Multiplication

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

Output fields:

- `matrix`: square matrix size
- `iters`: timed iterations
- `avg / iter`: average latency
- `rough throughput`: estimated GFLOP/s

Beginner note:

GFLOP/s means billions of floating point operations per second. It is useful for comparing compute throughput, but it is not the only thing that matters for real models.

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

Examples:

```bash
cargo run -- inspect ./model.safetensors --json
```

```bash
cargo run -- similarity \
  --text "query text" \
  --text "candidate text" \
  --json
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

## Current Limitations

Candlebench is intentionally scoped.

Current embedding inference supports BERT-compatible `model.safetensors` checkpoints such as common `sentence-transformers` models.

Not currently implemented:

- general chat or text generation
- full LLM decoding
- GGUF text generation
- ONNX inference
- vector database storage
- automatic model architecture support for every Hugging Face repo

These are good future directions, but the current tool focuses on concepts that are easier to inspect and verify.

## Glossary

### Attention Mask

A list that tells the model which tokens are real and which are padding.

### Backend

The compute device or library used to run tensor operations, such as CPU or Metal.

### Cosine Similarity

A score that compares the direction of two vectors. It is commonly used for embedding similarity.

### DType

The data type used to store tensor values, such as `F32`, `F16`, or quantized formats.

### Embedding

A vector of numbers representing the meaning of text, image, or another input.

### GGUF

A model file format commonly used for quantized local LLMs.

### Hugging Face Hub

A model repository service that stores configs, tokenizers, weights, and model cards.

### Inference

Running a trained model on new input.

### Parameter

A learned number inside a model.

### Quantization

Reducing weight precision to save memory and often improve speed.

### SafeTensors

A safe and efficient tensor file format commonly used for model weights.

### Tensor

A multidimensional array of numbers.

### Token

A piece of text converted into an integer ID for a model.

### Tokenizer

The component that converts text into tokens and token IDs.

## Publishing This Site With GitHub Pages

This documentation is stored in the repository's `docs/` directory.

To publish it:

1. Push the repository to GitHub.
2. Open the repository settings.
3. Go to Pages.
4. Set the source to deploy from a branch.
5. Choose the `main` branch.
6. Choose the `/docs` folder.
7. Save.

GitHub will publish the site at a URL similar to:

```text
https://YOUR_USER.github.io/YOUR_REPO/
```

For a user or organization site named `YOUR_USER.github.io`, GitHub will publish at:

```text
https://YOUR_USER.github.io/
```
