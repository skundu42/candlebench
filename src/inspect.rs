use anyhow::{bail, Context, Result};
use comfy_table::{presets::UTF8_FULL, Cell, Table};
use humansize::{format_size, DECIMAL};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use crate::util::{format_number, format_shape, num_elements};

#[derive(Debug, Clone, Serialize)]
pub struct TensorRow {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub parameters: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectSummary {
    pub path: String,
    pub file_size_bytes: u64,
    pub tensor_count: usize,
    pub total_tensor_bytes: usize,
    pub total_parameters: usize,
    pub dtype_counts: BTreeMap<String, usize>,
    pub tensors: Vec<TensorRow>,
}

pub fn run(path: &Path, json: bool, limit: usize) -> Result<()> {
    if !path.exists() {
        bail!("file does not exist: {}", path.display());
    }

    let summary = inspect_safetensors(path)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_human_summary(&summary, limit);
    }

    Ok(())
}

pub fn inspect_safetensors(path: &Path) -> Result<InspectSummary> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to read file metadata for {}", path.display()))?;

    // SafeTensors is designed for zero-copy inspection. We memory-map the file so
    // the OS can page data lazily instead of reading everything into a Vec.
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .with_context(|| format!("failed to memory-map {}", path.display()))?
    };

    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("failed to parse SafeTensors file {}", path.display()))?;

    let mut rows = Vec::with_capacity(tensors.len());
    let mut dtype_counts = BTreeMap::new();
    let mut total_tensor_bytes = 0usize;
    let mut total_parameters = 0usize;

    for (name, view) in tensors.iter() {
        let dtype = format!("{:?}", view.dtype());
        let shape = view.shape().to_vec();
        let parameters = num_elements(&shape);
        let bytes = view.data().len();

        *dtype_counts.entry(dtype.clone()).or_insert(0) += 1;
        total_tensor_bytes += bytes;
        total_parameters += parameters;

        rows.push(TensorRow {
            name: name.to_string(),
            dtype,
            shape,
            parameters,
            bytes,
        });
    }

    rows.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));

    Ok(InspectSummary {
        path: path.display().to_string(),
        file_size_bytes: metadata.len(),
        tensor_count: tensors.len(),
        total_tensor_bytes,
        total_parameters,
        dtype_counts,
        tensors: rows,
    })
}

pub fn print_human_summary(summary: &InspectSummary, limit: usize) {
    println!("Candlebench");
    println!("===========");
    println!("file: {}", summary.path);
    println!(
        "file size: {}",
        format_size(summary.file_size_bytes, DECIMAL)
    );
    println!("tensors: {}", summary.tensor_count);
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

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("#"),
        Cell::new("name"),
        Cell::new("dtype"),
        Cell::new("shape"),
        Cell::new("params"),
        Cell::new("bytes"),
    ]);

    for (idx, row) in summary.tensors.iter().take(limit).enumerate() {
        table.add_row(vec![
            Cell::new(idx + 1),
            Cell::new(&row.name),
            Cell::new(&row.dtype),
            Cell::new(format_shape(&row.shape)),
            Cell::new(format_number(row.parameters)),
            Cell::new(format_size(row.bytes, DECIMAL)),
        ]);
    }

    println!("largest tensors:");
    println!("{table}");

    if summary.tensors.len() > limit {
        println!(
            "showing {limit} of {} tensors. Use --limit to show more.",
            summary.tensors.len()
        );
    }
}
