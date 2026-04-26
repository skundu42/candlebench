use anyhow::{bail, Context, Result};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
pub enum HubRepoType {
    Model,
    Dataset,
    Space,
}

impl HubRepoType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "model" => Ok(Self::Model),
            "dataset" => Ok(Self::Dataset),
            "space" => Ok(Self::Space),
            _ => bail!("--repo-type must be one of: model, dataset, space"),
        }
    }

    fn as_hf_repo_type(self) -> RepoType {
        match self {
            Self::Model => RepoType::Model,
            Self::Dataset => RepoType::Dataset,
            Self::Space => RepoType::Space,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadedFile {
    pub repo: String,
    pub repo_type: String,
    pub revision: Option<String>,
    pub file: String,
    pub path: PathBuf,
}

pub fn run_download(
    repo: &str,
    repo_type: HubRepoType,
    revision: Option<&str>,
    files: &[String],
    cache_dir: Option<PathBuf>,
    no_progress: bool,
    json: bool,
) -> Result<()> {
    if files.is_empty() {
        bail!("at least one file must be provided");
    }

    let mut builder = ApiBuilder::from_env().with_progress(!no_progress);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }
    let api = builder
        .build()
        .context("failed to initialize Hugging Face Hub client")?;

    let repo_id = repo.to_string();
    let hf_repo = match revision {
        Some(revision) => Repo::with_revision(
            repo_id.clone(),
            repo_type.as_hf_repo_type(),
            revision.to_string(),
        ),
        None => Repo::new(repo_id.clone(), repo_type.as_hf_repo_type()),
    };
    let api_repo = api.repo(hf_repo);

    let mut downloaded = Vec::with_capacity(files.len());
    for file in files {
        let path = api_repo
            .get(file)
            .with_context(|| format!("failed to download {file} from {repo}"))?;
        downloaded.push(DownloadedFile {
            repo: repo.to_string(),
            repo_type: format!("{repo_type:?}").to_lowercase(),
            revision: revision.map(ToOwned::to_owned),
            file: file.clone(),
            path,
        });
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&downloaded)?);
    } else {
        for item in downloaded {
            println!("{} -> {}", item.file, item.path.display());
        }
    }

    Ok(())
}
