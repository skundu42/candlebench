use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use comfy_table::{presets::UTF8_FULL, Cell, Table};
use humansize::{format_size, DECIMAL};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::util::{format_number, format_shape};

#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadataRow {
    pub key: String,
    pub value_type: String,
    pub value: String,
    pub length: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufTensorRow {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub parameters: usize,
    pub bytes: usize,
    pub offset: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufSummary {
    pub path: String,
    pub file_size_bytes: u64,
    pub version: String,
    pub metadata_count: usize,
    pub tensor_count: usize,
    pub tensor_data_offset: u64,
    pub total_tensor_bytes: usize,
    pub total_parameters: usize,
    pub dtype_counts: BTreeMap<String, usize>,
    pub metadata: Vec<GgufMetadataRow>,
    pub tensors: Vec<GgufTensorRow>,
}

pub fn run(path: &Path, json: bool, limit: usize) -> Result<()> {
    if !path.exists() {
        bail!("file does not exist: {}", path.display());
    }

    let summary = inspect_gguf(path)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_human_summary(&summary, limit);
    }

    Ok(())
}

pub fn inspect_gguf(path: &Path) -> Result<GgufSummary> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let file_size_bytes = file
        .metadata()
        .with_context(|| format!("failed to read file metadata for {}", path.display()))?
        .len();
    let mut reader = BufReader::new(file);
    let content = Content::read(&mut reader)
        .with_context(|| format!("failed to parse GGUF file {}", path.display()))?;

    let mut metadata = content
        .metadata
        .iter()
        .map(|(key, value)| GgufMetadataRow {
            key: key.clone(),
            value_type: gguf_value_type(value),
            value: gguf_value_display(value),
            length: gguf_value_len(value),
        })
        .collect::<Vec<_>>();
    metadata.sort_by(|a, b| a.key.cmp(&b.key));

    let mut dtype_counts = BTreeMap::new();
    let mut tensors = Vec::with_capacity(content.tensor_infos.len());
    let mut total_tensor_bytes = 0usize;
    let mut total_parameters = 0usize;

    for (name, info) in &content.tensor_infos {
        let dtype = format!("{:?}", info.ggml_dtype);
        let shape = info.shape.dims().to_vec();
        let parameters = info.shape.elem_count();
        let block_size = info.ggml_dtype.block_size();
        let bytes = parameters.div_ceil(block_size) * info.ggml_dtype.type_size();

        *dtype_counts.entry(dtype.clone()).or_insert(0) += 1;
        total_tensor_bytes += bytes;
        total_parameters += parameters;

        tensors.push(GgufTensorRow {
            name: name.clone(),
            dtype,
            shape,
            parameters,
            bytes,
            offset: info.offset,
        });
    }

    tensors.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));

    Ok(GgufSummary {
        path: path.display().to_string(),
        file_size_bytes,
        version: format!("{:?}", content.magic),
        metadata_count: metadata.len(),
        tensor_count: tensors.len(),
        tensor_data_offset: content.tensor_data_offset,
        total_tensor_bytes,
        total_parameters,
        dtype_counts,
        metadata,
        tensors,
    })
}

pub fn print_human_summary(summary: &GgufSummary, limit: usize) {
    println!("Candlebench");
    println!("===========");
    println!("file: {}", summary.path);
    println!("format: GGUF {}", summary.version);
    println!(
        "file size: {}",
        format_size(summary.file_size_bytes, DECIMAL)
    );
    println!("metadata entries: {}", summary.metadata_count);
    println!("tensors: {}", summary.tensor_count);
    println!("tensor data offset: {}", summary.tensor_data_offset);
    println!(
        "tensor data: {}",
        format_size(summary.total_tensor_bytes, DECIMAL)
    );
    println!(
        "rough parameters: {}",
        format_number(summary.total_parameters)
    );
    println!();

    if !summary.dtype_counts.is_empty() {
        println!("dtype counts:");
        for (dtype, count) in &summary.dtype_counts {
            println!("  {dtype}: {count}");
        }
        println!();
    }

    let mut metadata_table = Table::new();
    metadata_table.load_preset(UTF8_FULL);
    metadata_table.set_header(vec![
        Cell::new("#"),
        Cell::new("key"),
        Cell::new("type"),
        Cell::new("value"),
    ]);

    for (idx, row) in summary.metadata.iter().take(limit).enumerate() {
        metadata_table.add_row(vec![
            Cell::new(idx + 1),
            Cell::new(&row.key),
            Cell::new(&row.value_type),
            Cell::new(&row.value),
        ]);
    }

    println!("metadata:");
    println!("{metadata_table}");

    if summary.metadata.len() > limit {
        println!(
            "showing {limit} of {} metadata entries. Use --limit to show more.",
            summary.metadata.len()
        );
    }
    println!();

    let mut tensor_table = Table::new();
    tensor_table.load_preset(UTF8_FULL);
    tensor_table.set_header(vec![
        Cell::new("#"),
        Cell::new("name"),
        Cell::new("dtype"),
        Cell::new("shape"),
        Cell::new("params"),
        Cell::new("bytes"),
    ]);

    for (idx, row) in summary.tensors.iter().take(limit).enumerate() {
        tensor_table.add_row(vec![
            Cell::new(idx + 1),
            Cell::new(&row.name),
            Cell::new(&row.dtype),
            Cell::new(format_shape(&row.shape)),
            Cell::new(format_number(row.parameters)),
            Cell::new(format_size(row.bytes, DECIMAL)),
        ]);
    }

    println!("largest tensors:");
    println!("{tensor_table}");

    if summary.tensors.len() > limit {
        println!(
            "showing {limit} of {} tensors. Use --limit to show more.",
            summary.tensors.len()
        );
    }
}

fn gguf_value_type(value: &Value) -> String {
    match value {
        Value::Array(values) => values
            .first()
            .map(|v| format!("Array<{:?}>", v.value_type()))
            .unwrap_or_else(|| "Array<empty>".to_string()),
        _ => format!("{:?}", value.value_type()),
    }
}

fn gguf_value_len(value: &Value) -> Option<usize> {
    match value {
        Value::Array(values) => Some(values.len()),
        Value::String(value) => Some(value.len()),
        _ => None,
    }
}

fn gguf_value_display(value: &Value) -> String {
    match value {
        Value::U8(v) => v.to_string(),
        Value::I8(v) => v.to_string(),
        Value::U16(v) => v.to_string(),
        Value::I16(v) => v.to_string(),
        Value::U32(v) => v.to_string(),
        Value::I32(v) => v.to_string(),
        Value::U64(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        Value::F32(v) => v.to_string(),
        Value::F64(v) => v.to_string(),
        Value::Bool(v) => v.to_string(),
        Value::String(v) => truncate(v, 120),
        Value::Array(values) => {
            let preview = values
                .iter()
                .take(8)
                .map(gguf_value_display)
                .collect::<Vec<_>>()
                .join(", ");
            if values.len() > 8 {
                format!("[{preview}, ...] ({} items)", values.len())
            } else {
                format!("[{preview}]")
            }
        }
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let mut out = value.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}
