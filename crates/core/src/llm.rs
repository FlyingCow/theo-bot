use anyhow::{anyhow, Context, Result};
use aws_sdk_bedrockruntime::types::{ContentBlock, ConversationRole, Message, SystemContentBlock};
use aws_sdk_bedrockruntime::Client;

/// Claude 3.5 Haiku via Bedrock. Swap for `"amazon.nova-micro-v1:0"` if preferred.
pub const DEFAULT_MODEL_ID: &str = "anthropic.claude-3-5-haiku-20241022-v1:0";

pub struct LlmClient {
    client: Client,
    model_id: String,
}

impl LlmClient {
    pub fn new(client: Client) -> Self {
        Self { client, model_id: DEFAULT_MODEL_ID.to_string() }
    }

    pub fn with_model_id(client: Client, model_id: impl Into<String>) -> Self {
        Self { client, model_id: model_id.into() }
    }

    /// Sends a single-turn (system + user) request and returns the model's text reply.
    pub async fn complete(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let message = Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(user_message.to_string()))
            .build()
            .context("building Bedrock message")?;

        let response = self
            .client
            .converse()
            .model_id(&self.model_id)
            .system(SystemContentBlock::Text(system_prompt.to_string()))
            .messages(message)
            .send()
            .await
            .context("calling Bedrock Converse API")?;

        let output = response.output.ok_or_else(|| anyhow!("Bedrock response had no output"))?;
        let message = output
            .as_message()
            .map_err(|_| anyhow!("unexpected Bedrock output variant"))?;

        let text = message
            .content
            .iter()
            .find_map(|block| block.as_text().ok())
            .ok_or_else(|| anyhow!("Bedrock response had no text content"))?;

        Ok(text.clone())
    }
}
