/// Splits text into overlapping word-based chunks.
///
/// Word count is a cheap proxy for token count that avoids pulling in a
/// tokenizer dependency just for chunk sizing.
pub fn chunk_text(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    assert!(chunk_words > overlap_words, "chunk_words must exceed overlap_words");

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let step = chunk_words - overlap_words;
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + chunk_words).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_short_text_into_single_chunk() {
        let chunks = chunk_text("one two three", 10, 2);
        assert_eq!(chunks, vec!["one two three"]);
    }

    #[test]
    fn overlaps_consecutive_chunks() {
        let text = (1..=25).map(|n| n.to_string()).collect::<Vec<_>>().join(" ");
        let chunks = chunk_text(&text, 10, 2);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "1 2 3 4 5 6 7 8 9 10");
        assert_eq!(chunks[1], "9 10 11 12 13 14 15 16 17 18");
        assert_eq!(chunks[2], "17 18 19 20 21 22 23 24 25");
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        assert!(chunk_text("   ", 10, 2).is_empty());
    }
}
