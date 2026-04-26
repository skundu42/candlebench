use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
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

#[derive(Debug, Clone, Serialize)]
pub struct SimilarityPair {
    pub left_index: usize,
    pub right_index: usize,
    pub left_text: String,
    pub right_text: String,
    pub cosine_similarity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimilaritySummary {
    pub repo: String,
    pub revision: Option<String>,
    pub device: String,
    pub normalized: bool,
    pub dimensions: usize,
    pub pairs: Vec<SimilarityPair>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingBenchmarkSummary {
    pub repo: String,
    pub revision: Option<String>,
    pub device: String,
    pub normalized: bool,
    pub batch_size: usize,
    pub dimensions: usize,
    pub warmup_iters: usize,
    pub timed_iters: usize,
    pub tokens_per_iter: usize,
    pub total_tokens: usize,
    pub total_embeddings: usize,
    pub total_elapsed_ms: f64,
    pub avg_iter_ms: f64,
    pub embeddings_per_sec: f64,
    pub tokens_per_sec: f64,
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

pub struct EmbeddingRunner {
    repo: String,
    revision: Option<String>,
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    weights_path: PathBuf,
    device: Device,
    tokenizer: Tokenizer,
    model: BertModel,
    normalize: bool,
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
    let runner = EmbeddingRunner::load(&options)?;
    let texts = options.texts.clone();
    let batch = runner.embed_texts(texts)?;

    Ok(EmbeddingSummary {
        repo: runner.repo,
        revision: runner.revision,
        config_path: runner.config_path,
        tokenizer_path: runner.tokenizer_path,
        weights_path: runner.weights_path,
        device: device_label(&runner.device).to_string(),
        normalized: runner.normalize,
        embeddings: batch.rows,
    })
}

pub fn run_similarity(options: EmbedOptions) -> Result<()> {
    let json = options.json;
    let summary = similarity(options)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_similarity_summary(&summary);
    }

    Ok(())
}

pub fn similarity(options: EmbedOptions) -> Result<SimilaritySummary> {
    if options.texts.len() < 2 {
        bail!("similarity requires at least two --text values");
    }

    let runner = EmbeddingRunner::load(&options)?;
    let batch = runner.embed_texts(options.texts)?;
    let mut pairs = Vec::new();

    for left in 0..batch.rows.len() {
        for right in (left + 1)..batch.rows.len() {
            let left_row = &batch.rows[left];
            let right_row = &batch.rows[right];
            pairs.push(SimilarityPair {
                left_index: left,
                right_index: right,
                left_text: left_row.text.clone(),
                right_text: right_row.text.clone(),
                cosine_similarity: cosine_similarity(&left_row.embedding, &right_row.embedding)?,
            });
        }
    }

    pairs.sort_by(|a, b| {
        b.cosine_similarity
            .total_cmp(&a.cosine_similarity)
            .then_with(|| a.left_index.cmp(&b.left_index))
            .then_with(|| a.right_index.cmp(&b.right_index))
    });

    Ok(SimilaritySummary {
        repo: runner.repo,
        revision: runner.revision,
        device: device_label(&runner.device).to_string(),
        normalized: runner.normalize,
        dimensions: batch.dimensions,
        pairs,
    })
}

pub fn run_benchmark(
    mut options: EmbedOptions,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<()> {
    let json = options.json;
    if options.texts.is_empty() {
        options.texts = vec!["Candlebench embedding benchmark sentence.".to_string()];
    }
    let summary = benchmark(options, warmup_iters, timed_iters)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_benchmark_summary(&summary);
    }

    Ok(())
}

pub fn benchmark(
    options: EmbedOptions,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<EmbeddingBenchmarkSummary> {
    if warmup_iters == 0 {
        bail!("--warmup-iters must be greater than 0");
    }
    if timed_iters == 0 {
        bail!("--iters must be greater than 0");
    }

    let runner = EmbeddingRunner::load(&options)?;
    let texts = options.texts.clone();

    let mut last_batch = None;
    for _ in 0..warmup_iters {
        last_batch = Some(runner.embed_texts(texts.clone())?);
    }

    let mut elapsed = Duration::ZERO;
    for _ in 0..timed_iters {
        let start = Instant::now();
        last_batch = Some(runner.embed_texts(texts.clone())?);
        elapsed += start.elapsed();
    }

    let batch = last_batch.context("benchmark did not produce an embedding batch")?;
    let tokens_per_iter = batch.total_tokens;
    let batch_size = batch.rows.len();
    let dimensions = batch.dimensions;
    let total_tokens = tokens_per_iter * timed_iters;
    let total_embeddings = batch_size * timed_iters;
    let total_elapsed = elapsed.as_secs_f64();
    let avg_iter_ms = (total_elapsed / timed_iters as f64) * 1000.0;

    Ok(EmbeddingBenchmarkSummary {
        repo: runner.repo,
        revision: runner.revision,
        device: device_label(&runner.device).to_string(),
        normalized: runner.normalize,
        batch_size,
        dimensions,
        warmup_iters,
        timed_iters,
        tokens_per_iter,
        total_tokens,
        total_embeddings,
        total_elapsed_ms: total_elapsed * 1000.0,
        avg_iter_ms,
        embeddings_per_sec: total_embeddings as f64 / total_elapsed,
        tokens_per_sec: total_tokens as f64 / total_elapsed,
    })
}

struct EmbeddingBatch {
    rows: Vec<EmbeddingRow>,
    total_tokens: usize,
    dimensions: usize,
}

impl EmbeddingRunner {
    fn load(options: &EmbedOptions) -> Result<Self> {
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
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_path()], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &config).context("failed to load BERT model")?;

        Ok(Self {
            repo,
            revision: options.revision.clone(),
            config_path,
            tokenizer_path,
            weights_path,
            device,
            tokenizer,
            model,
            normalize: options.normalize,
        })
    }

    fn embed_texts(&self, texts: Vec<String>) -> Result<EmbeddingBatch> {
        if texts.is_empty() {
            bail!("at least one text value must be provided");
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.clone(), true)
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

        let embeddings = self.run_bert(input_ids, token_type_ids, attention_mask)?;
        let embeddings = embeddings.to_vec2::<f32>()?;

        let rows = texts
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

        let total_tokens = rows.iter().map(|row| row.token_count).sum();
        let dimensions = rows.first().map(|row| row.dimensions).unwrap_or(0);

        Ok(EmbeddingBatch {
            rows,
            total_tokens,
            dimensions,
        })
    }

    fn run_bert(
        &self,
        input_ids: Vec<Vec<u32>>,
        token_type_ids: Vec<Vec<u32>>,
        attention_mask: Vec<Vec<u32>>,
    ) -> Result<Tensor> {
        let input_ids = Tensor::new(input_ids, &self.device)?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)?;
        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)?;

        let embeddings = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask_tensor))
            .context("failed to run BERT forward pass")?;

        let mask = attention_mask_tensor.to_dtype(DType::F32)?.unsqueeze(2)?;
        let pooled = embeddings.broadcast_mul(&mask)?.sum(1)?;
        let denom = mask.sum(1)?.clamp(1e-12f32, f32::MAX)?;
        let pooled = pooled.broadcast_div(&denom)?;

        let pooled = if self.normalize {
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

fn print_similarity_summary(summary: &SimilaritySummary) {
    println!("Embedding similarity");
    println!("====================");
    println!("repo: {}", summary.repo);
    if let Some(revision) = &summary.revision {
        println!("revision: {revision}");
    }
    println!("device: {}", summary.device);
    println!("dimensions: {}", summary.dimensions);
    println!("normalized: {}", summary.normalized);
    println!();

    for pair in &summary.pairs {
        println!(
            "#{}/#{}  cosine={:.6}",
            pair.left_index + 1,
            pair.right_index + 1,
            pair.cosine_similarity
        );
        println!("  left:  {}", pair.left_text);
        println!("  right: {}", pair.right_text);
        println!();
    }
}

fn print_benchmark_summary(summary: &EmbeddingBenchmarkSummary) {
    println!("Embedding benchmark");
    println!("===================");
    println!("repo: {}", summary.repo);
    if let Some(revision) = &summary.revision {
        println!("revision: {revision}");
    }
    println!("device: {}", summary.device);
    println!("batch size: {}", summary.batch_size);
    println!("dimensions: {}", summary.dimensions);
    println!("tokens / iter: {}", summary.tokens_per_iter);
    println!("warmup iters: {}", summary.warmup_iters);
    println!("timed iters: {}", summary.timed_iters);
    println!();
    println!("total elapsed: {:.3}ms", summary.total_elapsed_ms);
    println!("avg / iter: {:.3}ms", summary.avg_iter_ms);
    println!("embeddings/sec: {:.2}", summary.embeddings_per_sec);
    println!("tokens/sec: {:.2}", summary.tokens_per_sec);
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

fn normalize_sentence_transformers_repo(repo: &str) -> String {
    if repo.contains('/') {
        repo.to_string()
    } else {
        format!("sentence-transformers/{repo}")
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "embedding dimensions differ: left={} right={}",
            left.len(),
            right.len()
        );
    }
    if left.is_empty() {
        bail!("embedding vectors are empty");
    }

    let mut dot = 0f32;
    let mut left_norm = 0f32;
    let mut right_norm = 0f32;
    for (&left_value, &right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        bail!("cannot compute cosine similarity for a zero-norm embedding");
    }

    Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
}
