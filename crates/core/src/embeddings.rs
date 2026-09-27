use anyhow::{Context, Result};
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use serde::{Deserialize, Serialize};

/// Bedrock Titan Text Embeddings V2 model id.
pub const TITAN_EMBED_MODEL_ID: &str = "amazon.titan-embed-text-v2:0";

#[derive(Serialize)]
struct TitanEmbedRequest<'a> {
    #[serde(rename = "inputText")]
    input_text: &'a str,
}

#[derive(Deserialize)]
struct TitanEmbedResponse {
    embedding: Vec<f32>,
}

pub struct EmbeddingsClient {
    client: Client,
    model_id: String,
}

impl EmbeddingsClient {
    pub fn new(client: Client) -> Self {
        Self { client, model_id: TITAN_EMBED_MODEL_ID.to_string() }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = serde_json::to_vec(&TitanEmbedRequest { input_text: text })
            .context("serializing Titan embed request")?;

        let response = self
            .client
            .invoke_model()
            .model_id(&self.model_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .context("invoking Titan embeddings model")?;

        let parsed: TitanEmbedResponse = serde_json::from_slice(response.body.as_ref())
            .context("parsing Titan embed response")?;
        Ok(parsed.embedding)
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }
}
