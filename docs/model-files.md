---
title: Model Files
description: Understand SafeTensors, GGUF, tensor shapes, parameters, and quantization.
permalink: /model-files/
---

# Model Files

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

## Model Artifacts

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

## Tensor Shapes And Parameters

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

## SafeTensors Versus GGUF

SafeTensors and GGUF both store model data, but they are optimized for different workflows.

| Format | Common Use | What Candlebench Inspects |
| --- | --- | --- |
| SafeTensors | Hugging Face model weights, often full precision or half precision | tensor names, dtypes, shapes, rough parameter counts, payload bytes |
| GGUF | Quantized local LLM files used by llama.cpp-style runtimes | metadata, tokenizer data, architecture fields, quantized tensor dtypes, offsets |

SafeTensors is useful when the model code already knows the architecture. The file mostly says, "here are named tensors with shapes and dtypes."

GGUF carries more runtime metadata in the same file. A GGUF file can include architecture keys, tokenizer settings, context length, quantization information, and tensor offsets. That makes it convenient for local LLM runtimes, especially when distributing quantized models.

## Quantization

Quantization stores model weights with fewer bits than standard floating point formats.

For example:

| DType | Rough Meaning | Typical Tradeoff |
| --- | --- | --- |
| `F32` | 32-bit float | highest storage cost, strong precision |
| `F16` | 16-bit float | lower memory, usually good inference quality |
| `Q8_0` | 8-bit quantized | smaller, often still high quality |
| `Q4_0`, `Q4K` | 4-bit quantized | much smaller, more quality risk |

Quantization mainly reduces model size and memory bandwidth. It does not automatically make every workload faster. Some hardware is excellent at dense floating point math, while some quantized formats require unpacking or specialized kernels.
