mod telegram;

use std::sync::Arc;

use anyhow::{Context, Result};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_sdk_s3::Client as S3Client;
use lambda_http::{run, service_fn, Body, Error, Request, Response};
use theo_core::index::Index;
use theo_core::prompts::{qa_user_message, QA_SYSTEM_PROMPT};
use theo_core::summarize::summarize_document;
use theo_core::{EmbeddingsClient, LlmClient};
use tokio::sync::OnceCell;
use tracing::{info, warn};

use telegram::{send_message, Update};

struct AppState {
    index: Index,
    embeddings: EmbeddingsClient,
    llm: LlmClient,
    telegram_bot_token: String,
    telegram_webhook_secret: String,
    http: reqwest::Client,
}

static STATE: OnceCell<Arc<AppState>> = OnceCell::const_new();

async fn state() -> Result<Arc<AppState>> {
    STATE
        .get_or_try_init(init_state)
        .await
        .map(Arc::clone)
        .map_err(|e: anyhow::Error| anyhow::anyhow!("{e:?}"))
}

async fn init_state() -> Result<Arc<AppState>> {
    let sdk_config = aws_config::load_from_env().await;

    let bucket = std::env::var("INDEX_S3_BUCKET").context("INDEX_S3_BUCKET not set")?;
    let key = std::env::var("INDEX_S3_KEY").unwrap_or_else(|_| "index.bin".to_string());
    let telegram_bot_token = std::env::var("TELEGRAM_BOT_TOKEN").context("TELEGRAM_BOT_TOKEN not set")?;
    let telegram_webhook_secret =
        std::env::var("TELEGRAM_WEBHOOK_SECRET").context("TELEGRAM_WEBHOOK_SECRET not set")?;

    let s3 = S3Client::new(&sdk_config);
    let object = s3
        .get_object()
        .bucket(&bucket)
        .key(&key)
        .send()
        .await
        .with_context(|| format!("downloading index from s3://{bucket}/{key}"))?;
    let bytes = object.body.collect().await.context("reading index body")?.into_bytes();
    let index = Index::from_bytes(&bytes)?;
    info!(documents = index.documents.len(), chunks = index.chunks.len(), "loaded index");

    let embeddings = EmbeddingsClient::new(BedrockClient::new(&sdk_config));
    let llm = LlmClient::new(BedrockClient::new(&sdk_config));

    Ok(Arc::new(AppState {
        index,
        embeddings,
        llm,
        telegram_bot_token,
        telegram_webhook_secret,
        http: reqwest::Client::new(),
    }))
}

async fn handler(request: Request) -> Result<Response<Body>, Error> {
    let state = state().await.map_err(|e| Error::from(e.to_string()))?;

    let provided_secret = request
        .headers()
        .get("X-Telegram-Bot-Api-Secret-Token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if provided_secret != state.telegram_webhook_secret {
        warn!("rejected webhook call with invalid secret token");
        return Ok(Response::builder().status(401).body(Body::Empty)?);
    }

    let update: Update = match serde_json::from_slice(request.body()) {
        Ok(u) => u,
        Err(e) => {
            warn!(error = %e, "failed to parse Telegram update");
            return Ok(Response::builder().status(200).body(Body::Empty)?);
        }
    };

    if let Some(message) = update.message {
        let chat_id = message.chat.id;
        let text = message.text.unwrap_or_default();

        let reply = match handle_text(&state, &text).await {
            Ok(reply) => reply,
            Err(e) => {
                warn!(error = %e, "error handling message");
                "Sorry, something went wrong processing that.".to_string()
            }
        };

        if let Err(e) = send_message(&state.http, &state.telegram_bot_token, chat_id, &reply).await {
            warn!(error = %e, "failed to send Telegram reply");
        }
    }

    Ok(Response::builder().status(200).body(Body::Empty)?)
}

async fn handle_text(state: &AppState, text: &str) -> Result<String> {
    let text = text.trim();

    if text == "/start" || text == "/help" {
        return Ok("Ask me a question about the documents, or use /docs to list them and \
/summarize <name> to get a summary of one."
            .to_string());
    }

    if text == "/docs" {
        if state.index.documents.is_empty() {
            return Ok("No documents indexed yet.".to_string());
        }
        let list = state.index.documents.iter().map(|d| format!("- {}", d.title)).collect::<Vec<_>>().join("\n");
        return Ok(list);
    }

    if let Some(query) = text.strip_prefix("/summarize") {
        let query = query.trim();
        let Some(doc) = state.index.find_document(query) else {
            return Ok(format!("Couldn't find a document matching \"{query}\". Try /docs to see available titles."));
        };
        return summarize_document(&state.llm, &state.index, doc).await;
    }

    let query_embedding = state.embeddings.embed(text).await?;
    let chunks = state.index.search(&query_embedding, 5);
    if chunks.is_empty() {
        return Ok("No documents indexed yet.".to_string());
    }

    let titles = |doc_id: &str| {
        state
            .index
            .documents
            .iter()
            .find(|d| d.id == doc_id)
            .map(|d| d.title.clone())
            .unwrap_or_else(|| doc_id.to_string())
    };
    let user_message = qa_user_message(&chunks, titles, text);
    state.llm.complete(QA_SYSTEM_PROMPT, &user_message).await
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().with_target(false).without_time().init();
    run(service_fn(handler)).await
}
