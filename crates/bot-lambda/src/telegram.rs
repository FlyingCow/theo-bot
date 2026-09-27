use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Update {
    pub message: Option<Message>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub chat: Chat,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Chat {
    pub id: i64,
}

#[derive(Serialize)]
struct SendMessageRequest<'a> {
    chat_id: i64,
    text: &'a str,
}

pub async fn send_message(http: &reqwest::Client, bot_token: &str, chat_id: i64, text: &str) -> Result<()> {
    let url = format!("https://api.telegram.org/bot{bot_token}/sendMessage");
    http.post(url)
        .json(&SendMessageRequest { chat_id, text })
        .send()
        .await
        .context("sending Telegram message")?
        .error_for_status()
        .context("Telegram API returned an error status")?;
    Ok(())
}
