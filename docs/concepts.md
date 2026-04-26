---
title: ML Concepts
description: Learn the model, tokenizer, embedding, pooling, normalization, and similarity concepts behind Candlebench.
permalink: /concepts/
---

# ML Concepts

The high-level embedding pipeline is:

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

## Tokenization

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

## Embeddings

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

## Batching, Padding, And Truncation

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

## Forward Pass

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

## Pooling

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

## L2 Normalization

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

## Cosine Similarity

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
