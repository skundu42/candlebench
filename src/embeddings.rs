use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::backend::{device_for_backend, device_label, Backend};

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRow {
    pub text: String,
    pub token_count: usize,
    pub dimensions: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingSummary {
    pub repo: String,
    pub revision: Option<String>,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weights_path: PathBuf,
    pub device: String,
    pub normalized: bool,
    pub embeddings: Vec<EmbeddingRow>,
}

#[derive(Debug, Clone)]
pub struct EmbedOptions {
    pub repo: String,
    pub revision: Option<String>,
    pub config_file: String,
    pub tokenizer_file: String,
    pub weights_file: String,
    pub cache_dir: Option<PathBuf>,
    pub texts: Vec<String>,
    pub normalize: bool,
    pub max_length: usize,
    pub backend: Backend,
    pub json: bool,
    pub no_progress: bool,
}

pub fn run(options: EmbedOptions) -> Result<()> {
    let json = options.json;
    let summary = embed(options)?;

    if summary.embeddings.is_empty() {
        return Ok(());
    }

    if summary.embeddings.len() == 1 && summary.embeddings[0].embedding.is_empty() {
        bail!("embedding output was empty");
    }

    if summary.normalized && summary.embeddings[0].embedding.is_empty() {
        bail!("normalized embedding output was empty");
    }

    if summary.embeddings[0].dimensions == 0 {
        bail!("embedding dimension was 0");
    }

    if summary.embeddings[0].embedding.len() != summary.embeddings[0].dimensions {
        bail!("embedding length did not match reported dimension");
    }

    if summary
        .embeddings
        .iter()
        .any(|row| row.embedding.is_empty())
    {
        bail!("one or more embeddings were empty");
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_human_summary(&summary);
    }

    Ok(())
}

pub fn embed(options: EmbedOptions) -> Result<EmbeddingSummary> {
    if options.texts.is_empty() {
        bail!("at least one --text value must be provided");
    }
    if options.max_length == 0 {
        bail!("--max-length must be greater than 0");
    }

    let repo = normalize_sentence_transformers_repo(&options.repo);
    let (config_path, tokenizer_path, weights_path) = resolve_model_files(
        &repo,
        options.revision.as_deref(),
        &options.config_file,
        &options.tokenizer_file,
        &options.weights_file,
        options.cache_dir.clone(),
        options.no_progress,
    )?;

    let device = device_for_backend(options.backend)?;
    let config = read_config(&config_path)?;
    let mut tokenizer = load_tokenizer(&tokenizer_path, options.max_length)?;
    tokenizer.with_padding(Some(PaddingParams::default()));

    let encodings = tokenizer
        .encode_batch(options.texts.clone(), true)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .context("failed to tokenize texts")?;

    let input_ids = encodings
        .iter()
        .map(|encoding| encoding.get_ids().to_vec())
        .collect::<Vec<_>>();
    let token_type_ids = encodings
        .iter()
        .map(|encoding| encoding.get_type_ids().to_vec())
        .collect::<Vec<_>>();
    let attention_mask = encodings
        .iter()
        .map(|encoding| encoding.get_attention_mask().to_vec())
        .collect::<Vec<_>>();
    let token_counts = attention_mask
        .iter()
        .map(|mask| mask.iter().filter(|&&value| value != 0).count())
        .collect::<Vec<_>>();

    let embeddings = run_bert(
        &config,
        &weights_path,
        &device,
        input_ids,
        token_type_ids,
        attention_mask,
        options.normalize,
    )?;
    let embeddings = embeddings.to_vec2::<f32>()?;

    let rows = options
        .texts
        .iter()
        .zip(token_counts)
        .zip(embeddings)
        .map(|((text, token_count), embedding)| EmbeddingRow {
            text: text.clone(),
            token_count,
            dimensions: embedding.len(),
            embedding,
        })
        .collect::<Vec<_>>();

    Ok(EmbeddingSummary {
        repo,
        revision: options.revision,
        config_path,
        tokenizer_path,
        weights_path,
        device: device_label(&device).to_string(),
        normalized: options.normalize,
        embeddings: rows,
    })
}

fn print_human_summary(summary: &EmbeddingSummary) {
    println!("Embedding");
    println!("=========");
    println!("repo: {}", summary.repo);
    if let Some(revision) = &summary.revision {
        println!("revision: {revision}");
    }
    println!("device: {}", summary.device);
    println!("weights: {}", summary.weights_path.display());
    println!("normalized: {}", summary.normalized);
    println!();

    for (idx, row) in summary.embeddings.iter().enumerate() {
        println!("text #{}: {}", idx + 1, row.text);
        println!("tokens: {}", row.token_count);
        println!("dimensions: {}", row.dimensions);
        let preview = row
            .embedding
            .iter()
            .take(12)
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("preview: [{preview}, ...]");
        println!();
    }
}

fn resolve_model_files(
    repo: &str,
    revision: Option<&str>,
    config_file: &str,
    tokenizer_file: &str,
    weights_file: &str,
    cache_dir: Option<PathBuf>,
    no_progress: bool,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    if is_local_triplet(config_file, tokenizer_file, weights_file) {
        return Ok((
            Path::new(config_file).to_path_buf(),
            Path::new(tokenizer_file).to_path_buf(),
            Path::new(weights_file).to_path_buf(),
        ));
    }

    let mut builder = ApiBuilder::from_env().with_progress(!no_progress);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }
    let api = builder
        .build()
        .context("failed to initialize Hugging Face Hub client")?;
    let repo = match revision {
        Some(revision) => {
            Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string())
        }
        None => Repo::new(repo.to_string(), RepoType::Model),
    };
    let api_repo = api.repo(repo);

    let config_path = api_repo
        .get(config_file)
        .with_context(|| format!("failed to fetch {config_file}"))?;
    let tokenizer_path = api_repo
        .get(tokenizer_file)
        .with_context(|| format!("failed to fetch {tokenizer_file}"))?;
    let weights_path = api_repo
        .get(weights_file)
        .with_context(|| format!("failed to fetch {weights_file}"))?;

    Ok((config_path, tokenizer_path, weights_path))
}

fn is_local_triplet(config_file: &str, tokenizer_file: &str, weights_file: &str) -> bool {
    Path::new(config_file).exists()
        && Path::new(tokenizer_file).exists()
        && Path::new(weights_file).exists()
}

fn read_config(path: &Path) -> Result<Config> {
    let config = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    serde_json::from_str(&config).with_context(|| format!("failed to parse {}", path.display()))
}

fn load_tokenizer(path: &Path, max_length: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|err| anyhow::anyhow!("{err}"))
        .context("failed to configure tokenizer truncation")?;
    Ok(tokenizer)
}

fn run_bert(
    config: &Config,
    weights_path: &Path,
    device: &Device,
    input_ids: Vec<Vec<u32>>,
    token_type_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u32>>,
    normalize: bool,
) -> Result<Tensor> {
    let input_ids = Tensor::new(input_ids, device)?;
    let token_type_ids = Tensor::new(token_type_ids, device)?;
    let attention_mask_tensor = Tensor::new(attention_mask, device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, device)? };
    let model = BertModel::load(vb, config).context("failed to load BERT model")?;
    let embeddings = model
        .forward(&input_ids, &token_type_ids, Some(&attention_mask_tensor))
        .context("failed to run BERT forward pass")?;

    let mask = attention_mask_tensor.to_dtype(DType::F32)?.unsqueeze(2)?;
    let pooled = embeddings.broadcast_mul(&mask)?.sum(1)?;
    let denom = mask.sum(1)?.clamp(1e-12f32, f32::MAX)?;
    let pooled = pooled.broadcast_div(&denom)?;

    let pooled = if normalize {
        let norm = pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12f32, f32::MAX)?;
        pooled.broadcast_div(&norm)?
    } else {
        pooled
    };

    pooled.to_device(&Device::Cpu).map_err(Into::into)
}

fn normalize_sentence_transformers_repo(repo: &str) -> String {
    if repo.contains('/') {
        repo.to_string()
    } else {
        format!("sentence-transformers/{repo}")
    }
}
