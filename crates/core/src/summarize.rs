use anyhow::Result;

use crate::index::{Document, Index};
use crate::llm::LlmClient;
use crate::prompts::{summarize_map_prompt, summarize_reduce_prompt, SUMMARIZE_SYSTEM_PROMPT};

/// Max partial summaries combined in a single reduce call before batching further.
const REDUCE_BATCH_SIZE: usize = 10;

/// Map-reduce summarization over every chunk belonging to `doc`.
pub async fn summarize_document(llm: &LlmClient, index: &Index, doc: &Document) -> Result<String> {
    let chunks = index.chunks_for(doc);

    let mut summaries = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let prompt = summarize_map_prompt(&doc.title, &chunk.text);
        summaries.push(llm.complete(SUMMARIZE_SYSTEM_PROMPT, &prompt).await?);
    }

    reduce(llm, &doc.title, summaries).await
}

/// Recursively combines partial summaries in batches until a single summary remains.
async fn reduce(llm: &LlmClient, doc_title: &str, summaries: Vec<String>) -> Result<String> {
    if summaries.len() == 1 {
        return Ok(summaries.into_iter().next().unwrap());
    }

    let mut next_round = Vec::new();
    for batch in summaries.chunks(REDUCE_BATCH_SIZE) {
        let prompt = summarize_reduce_prompt(doc_title, batch);
        next_round.push(llm.complete(SUMMARIZE_SYSTEM_PROMPT, &prompt).await?);
    }

    Box::pin(reduce(llm, doc_title, next_round)).await
}
