use crate::index::Chunk;

pub const QA_SYSTEM_PROMPT: &str = "You are a helpful assistant that answers questions using ONLY \
the provided context excerpts from the user's documents. If the answer is not contained in the \
context, say you don't know — do not use outside knowledge. When useful, mention which document a \
fact came from. Be concise.";

pub fn qa_user_message(chunks: &[&Chunk], titles: impl Fn(&str) -> String, question: &str) -> String {
    let mut context = String::from("Context:\n");
    for chunk in chunks {
        context.push_str(&format!("[{}] {}\n\n", titles(&chunk.doc_id), chunk.text));
    }
    format!("{context}Question: {question}")
}

pub fn summarize_map_prompt(doc_title: &str, chunk_text: &str) -> String {
    format!(
        "Summarize the following excerpt from \"{doc_title}\" in 3-5 sentences, preserving key \
facts, names, and figures.\n\nExcerpt:\n{chunk_text}"
    )
}

pub fn summarize_reduce_prompt(doc_title: &str, partial_summaries: &[String]) -> String {
    let joined = partial_summaries
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are given partial summaries of sequential sections of \"{doc_title}\". Combine them \
into one coherent summary (a short paragraph followed by up to 5 bullet points of key facts). \
Avoid repetition.\n\nPartial summaries:\n{joined}"
    )
}

pub const SUMMARIZE_SYSTEM_PROMPT: &str =
    "You are a precise summarization assistant. Preserve factual accuracy; never invent details.";
