use std::path::PathBuf;

use anyhow::{Context, Result};
use aws_sdk_s3::primitives::ByteStream;
use clap::Parser;
use theo_core::chunking::chunk_text;
use theo_core::index::{Chunk, Document, Index};
use theo_core::EmbeddingsClient;
use tracing::info;

/// Builds (or rebuilds) the retrieval index from a folder of `.txt` documents
/// and uploads it to S3. Run this whenever documents are added or changed.
#[derive(Parser)]
struct Args {
    /// Folder containing one `.txt` file per document.
    #[arg(long, default_value = "docs")]
    docs_dir: PathBuf,

    /// Local path to also write the built index to, for inspection.
    #[arg(long, default_value = "index.bin")]
    out: PathBuf,

    /// Words per chunk.
    #[arg(long, default_value_t = 500)]
    chunk_words: usize,

    /// Overlapping words between consecutive chunks.
    #[arg(long, default_value_t = 50)]
    overlap_words: usize,

    /// S3 bucket to upload the built index to. If omitted, only writes locally.
    #[arg(long)]
    s3_bucket: Option<String>,

    /// S3 key for the uploaded index.
    #[arg(long, default_value = "index.bin")]
    s3_key: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let sdk_config = aws_config::load_from_env().await;
    let embeddings = EmbeddingsClient::new(aws_sdk_bedrockruntime::Client::new(&sdk_config));

    let index = build_index(&args, &embeddings).await?;
    info!(documents = index.documents.len(), chunks = index.chunks.len(), "built index");

    let bytes = index.to_bytes()?;
    std::fs::write(&args.out, &bytes).with_context(|| format!("writing {}", args.out.display()))?;
    info!(path = %args.out.display(), bytes = bytes.len(), "wrote index locally");

    if let Some(bucket) = &args.s3_bucket {
        let s3 = aws_sdk_s3::Client::new(&sdk_config);
        s3.put_object()
            .bucket(bucket)
            .key(&args.s3_key)
            .body(ByteStream::from(bytes))
            .send()
            .await
            .with_context(|| format!("uploading index to s3://{bucket}/{}", args.s3_key))?;
        info!(bucket, key = %args.s3_key, "uploaded index to S3");
    }

    Ok(())
}

async fn build_index(args: &Args, embeddings: &EmbeddingsClient) -> Result<Index> {
    let mut documents = Vec::new();
    let mut chunks = Vec::new();

    let mut entries: Vec<_> = std::fs::read_dir(&args.docs_dir)
        .with_context(|| format!("reading {}", args.docs_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("txt"))
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let doc_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();
        let title = doc_id.replace(['-', '_'], " ");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;

        info!(doc_id, "chunking and embedding document");
        let mut chunk_indices = Vec::new();
        for (chunk_id, text) in chunk_text(&text, args.chunk_words, args.overlap_words).into_iter().enumerate() {
            let embedding = embeddings.embed(&text).await?;
            chunk_indices.push(chunks.len());
            chunks.push(Chunk { doc_id: doc_id.clone(), chunk_id, text, embedding });
        }

        documents.push(Document { id: doc_id, title, chunk_indices });
    }

    Ok(Index { documents, chunks })
}
