mod backend;
mod bench;
mod embeddings;
mod gguf;
mod hub;
mod inspect;
mod tokenizer_lab;
mod tui;
mod util;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use backend::Backend;
use hub::HubRepoType;

#[derive(Debug, Parser)]
#[command(name = "candlebench")]
#[command(about = "Inspect and benchmark local ML models with Candle", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Inspect a .safetensors or .gguf model file without loading tensor data into Candle.
    Inspect {
        /// Path to a .safetensors or .gguf file.
        path: PathBuf,

        /// Output full summary as JSON.
        #[arg(long)]
        json: bool,

        /// Number of tensor rows to show in table output.
        #[arg(long, default_value_t = 30)]
        limit: usize,
    },

    /// Download files from a Hugging Face Hub repository into the local cache.
    Download {
        /// Hub repository id, for example sentence-transformers/all-MiniLM-L6-v2.
        #[arg(long)]
        repo: String,

        /// Hub repository type: model, dataset, or space.
        #[arg(long, default_value = "model", value_parser = parse_repo_type)]
        repo_type: HubRepoType,

        /// Optional revision, branch, tag, or commit.
        #[arg(long)]
        revision: Option<String>,

        /// Optional Hugging Face cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Disable hf-hub progress bars.
        #[arg(long)]
        no_progress: bool,

        /// Output downloaded paths as JSON.
        #[arg(long)]
        json: bool,

        /// File path(s) inside the repository.
        files: Vec<String>,
    },

    /// Tokenize text with a Hugging Face tokenizer.json file.
    Tokenize {
        /// Path to tokenizer.json.
        tokenizer: PathBuf,

        /// Text to encode.
        text: String,

        /// Output tokens as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Run BERT-style sentence embedding inference with Candle.
    Embed {
        /// Hub model repo. Bare names are resolved under sentence-transformers/.
        #[arg(long, default_value = "all-MiniLM-L6-v2")]
        repo: String,

        /// Optional revision, branch, tag, or commit.
        #[arg(long)]
        revision: Option<String>,

        /// Config filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "config.json")]
        config_file: String,

        /// Tokenizer filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer_file: String,

        /// Safetensors filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "model.safetensors")]
        weights_file: String,

        /// Optional Hugging Face cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Text to embed. Repeat for batches.
        #[arg(long = "text", required = true)]
        texts: Vec<String>,

        /// Disable L2 normalization.
        #[arg(long)]
        no_normalize: bool,

        /// Maximum tokenizer sequence length.
        #[arg(long, default_value_t = 512)]
        max_length: usize,

        /// Backend: cpu, metal, or auto.
        #[arg(long, default_value = "cpu", value_parser = parse_backend)]
        backend: Backend,

        /// Disable hf-hub progress bars.
        #[arg(long)]
        no_progress: bool,

        /// Output full embeddings as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Compute pairwise cosine similarity between embedded texts.
    Similarity {
        /// Hub model repo. Bare names are resolved under sentence-transformers/.
        #[arg(long, default_value = "all-MiniLM-L6-v2")]
        repo: String,

        /// Optional revision, branch, tag, or commit.
        #[arg(long)]
        revision: Option<String>,

        /// Config filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "config.json")]
        config_file: String,

        /// Tokenizer filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer_file: String,

        /// Safetensors filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "model.safetensors")]
        weights_file: String,

        /// Optional Hugging Face cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Text to compare. Repeat at least twice.
        #[arg(long = "text", required = true)]
        texts: Vec<String>,

        /// Disable L2 normalization before similarity calculation.
        #[arg(long)]
        no_normalize: bool,

        /// Maximum tokenizer sequence length.
        #[arg(long, default_value_t = 512)]
        max_length: usize,

        /// Backend: cpu, metal, or auto.
        #[arg(long, default_value = "cpu", value_parser = parse_backend)]
        backend: Backend,

        /// Disable hf-hub progress bars.
        #[arg(long)]
        no_progress: bool,

        /// Output pairwise scores as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Benchmark BERT-style embedding throughput with a loaded model.
    BenchEmbed {
        /// Hub model repo. Bare names are resolved under sentence-transformers/.
        #[arg(long, default_value = "all-MiniLM-L6-v2")]
        repo: String,

        /// Optional revision, branch, tag, or commit.
        #[arg(long)]
        revision: Option<String>,

        /// Config filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "config.json")]
        config_file: String,

        /// Tokenizer filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer_file: String,

        /// Safetensors filename in the Hub repo, or a local path when all three file args are local.
        #[arg(long, default_value = "model.safetensors")]
        weights_file: String,

        /// Optional Hugging Face cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Text to embed in each benchmark batch. Repeat to increase batch size.
        #[arg(long = "text")]
        texts: Vec<String>,

        /// Disable L2 normalization.
        #[arg(long)]
        no_normalize: bool,

        /// Maximum tokenizer sequence length.
        #[arg(long, default_value_t = 512)]
        max_length: usize,

        /// Backend: cpu, metal, or auto.
        #[arg(long, default_value = "cpu", value_parser = parse_backend)]
        backend: Backend,

        /// Number of untimed warmup iterations.
        #[arg(long, default_value_t = 2)]
        warmup_iters: usize,

        /// Number of timed iterations.
        #[arg(long, default_value_t = 10)]
        iters: usize,

        /// Disable hf-hub progress bars.
        #[arg(long)]
        no_progress: bool,

        /// Output benchmark metrics as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Open a small ratatui dashboard for a model file.
    Tui {
        /// Optional .safetensors or .gguf file to inspect.
        path: Option<PathBuf>,
    },

    /// Run a simple Candle matrix multiplication benchmark.
    BenchMatmul {
        /// Matrix dimension. A size of 1024 means [1024 x 1024] matmul.
        #[arg(long, default_value_t = 1024)]
        size: usize,

        /// Number of timed iterations.
        #[arg(long, default_value_t = 10)]
        iters: usize,

        /// Backend: cpu, metal, or auto.
        #[arg(long, default_value = "cpu", value_parser = parse_backend)]
        backend: Backend,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { path, json, limit } => {
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("gguf") => gguf::run(&path, json, limit)?,
                _ => inspect::run(&path, json, limit)?,
            }
        }
        Commands::Download {
            repo,
            repo_type,
            revision,
            cache_dir,
            no_progress,
            json,
            files,
        } => {
            hub::run_download(
                &repo,
                repo_type,
                revision.as_deref(),
                &files,
                cache_dir,
                no_progress,
                json,
            )?;
        }
        Commands::Tokenize {
            tokenizer,
            text,
            json,
        } => {
            tokenizer_lab::run(&tokenizer, &text, json)?;
        }
        Commands::Embed {
            repo,
            revision,
            config_file,
            tokenizer_file,
            weights_file,
            cache_dir,
            texts,
            no_normalize,
            max_length,
            backend,
            no_progress,
            json,
        } => {
            embeddings::run(embeddings::EmbedOptions {
                repo,
                revision,
                config_file,
                tokenizer_file,
                weights_file,
                cache_dir,
                texts,
                normalize: !no_normalize,
                max_length,
                backend,
                json,
                no_progress,
            })?;
        }
        Commands::Similarity {
            repo,
            revision,
            config_file,
            tokenizer_file,
            weights_file,
            cache_dir,
            texts,
            no_normalize,
            max_length,
            backend,
            no_progress,
            json,
        } => {
            embeddings::run_similarity(embeddings::EmbedOptions {
                repo,
                revision,
                config_file,
                tokenizer_file,
                weights_file,
                cache_dir,
                texts,
                normalize: !no_normalize,
                max_length,
                backend,
                json,
                no_progress,
            })?;
        }
        Commands::BenchEmbed {
            repo,
            revision,
            config_file,
            tokenizer_file,
            weights_file,
            cache_dir,
            texts,
            no_normalize,
            max_length,
            backend,
            warmup_iters,
            iters,
            no_progress,
            json,
        } => {
            embeddings::run_benchmark(
                embeddings::EmbedOptions {
                    repo,
                    revision,
                    config_file,
                    tokenizer_file,
                    weights_file,
                    cache_dir,
                    texts,
                    normalize: !no_normalize,
                    max_length,
                    backend,
                    json,
                    no_progress,
                },
                warmup_iters,
                iters,
            )?;
        }
        Commands::Tui { path } => {
            tui::run(path.as_deref())?;
        }
        Commands::BenchMatmul {
            size,
            iters,
            backend,
        } => {
            bench::run_matmul(size, iters, backend)?;
        }
    }

    Ok(())
}

fn parse_backend(value: &str) -> std::result::Result<Backend, String> {
    Backend::parse(value).map_err(|err| err.to_string())
}

fn parse_repo_type(value: &str) -> std::result::Result<HubRepoType, String> {
    HubRepoType::parse(value).map_err(|err| err.to_string())
}
