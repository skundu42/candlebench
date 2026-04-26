use anyhow::{bail, Context, Result};
use comfy_table::{presets::UTF8_FULL, Cell, Table};
use serde::Serialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize)]
pub struct TokenRow {
    pub index: usize,
    pub id: u32,
    pub token: String,
    pub type_id: u32,
    pub attention_mask: u32,
    pub offset: (usize, usize),
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizeSummary {
    pub tokenizer: String,
    pub text: String,
    pub token_count: usize,
    pub vocab_size: usize,
    pub tokens: Vec<TokenRow>,
}

pub fn run(path: &Path, text: &str, json: bool) -> Result<()> {
    if !path.exists() {
        bail!("tokenizer file does not exist: {}", path.display());
    }

    let summary = tokenize(path, text)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_human_summary(&summary);
    }
    Ok(())
}

pub fn tokenize(path: &Path, text: &str) -> Result<TokenizeSummary> {
    let tokenizer = Tokenizer::from_file(path)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .context("failed to encode text")?;

    let ids = encoding.get_ids();
    let tokens = encoding.get_tokens();
    let type_ids = encoding.get_type_ids();
    let attention_mask = encoding.get_attention_mask();
    let offsets = encoding.get_offsets();

    let rows = ids
        .iter()
        .enumerate()
        .map(|(index, id)| TokenRow {
            index,
            id: *id,
            token: tokens.get(index).cloned().unwrap_or_default(),
            type_id: *type_ids.get(index).unwrap_or(&0),
            attention_mask: *attention_mask.get(index).unwrap_or(&1),
            offset: *offsets.get(index).unwrap_or(&(0, 0)),
        })
        .collect::<Vec<_>>();

    Ok(TokenizeSummary {
        tokenizer: path.display().to_string(),
        text: text.to_string(),
        token_count: rows.len(),
        vocab_size: tokenizer.get_vocab_size(true),
        tokens: rows,
    })
}

fn print_human_summary(summary: &TokenizeSummary) {
    println!("Tokenizer");
    println!("=========");
    println!("file: {}", summary.tokenizer);
    println!("vocab size: {}", summary.vocab_size);
    println!("tokens: {}", summary.token_count);
    println!();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("#"),
        Cell::new("id"),
        Cell::new("token"),
        Cell::new("type"),
        Cell::new("mask"),
        Cell::new("offset"),
    ]);

    for row in &summary.tokens {
        table.add_row(vec![
            Cell::new(row.index),
            Cell::new(row.id),
            Cell::new(&row.token),
            Cell::new(row.type_id),
            Cell::new(row.attention_mask),
            Cell::new(format!("{}..{}", row.offset.0, row.offset.1)),
        ]);
    }

    println!("{table}");
}
