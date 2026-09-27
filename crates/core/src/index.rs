use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    /// Indices into `Index::chunks` that belong to this document, in order.
    pub chunk_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub doc_id: String,
    pub chunk_id: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Index {
    pub documents: Vec<Document>,
    pub chunks: Vec<Chunk>,
}

impl Index {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serde::encode_to_vec(self, bincode::config::standard())?)
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let (index, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(index)
    }

    pub fn find_document(&self, needle: &str) -> Option<&Document> {
        let needle = needle.to_lowercase();
        self.documents.iter().find(|d| {
            d.id.to_lowercase() == needle || d.title.to_lowercase().contains(&needle)
        })
    }

    /// Returns the top `k` chunks by cosine similarity to `query_embedding`,
    /// most similar first. Brute-force; fine up to tens of thousands of chunks.
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<&Chunk> {
        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .map(|c| (cosine_similarity(query_embedding, &c.embedding), c))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, c)| c).collect()
    }

    pub fn chunks_for(&self, doc: &Document) -> Vec<&Chunk> {
        doc.chunk_indices.iter().map(|&i| &self.chunks[i]).collect()
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(doc_id: &str, chunk_id: usize, embedding: Vec<f32>) -> Chunk {
        Chunk { doc_id: doc_id.into(), chunk_id, text: format!("chunk {chunk_id}"), embedding }
    }

    #[test]
    fn search_returns_most_similar_first() {
        let index = Index {
            documents: vec![],
            chunks: vec![
                chunk("d1", 0, vec![1.0, 0.0]),
                chunk("d1", 1, vec![0.0, 1.0]),
                chunk("d1", 2, vec![0.9, 0.1]),
            ],
        };
        let results = index.search(&[1.0, 0.0], 2);
        assert_eq!(results[0].chunk_id, 0);
        assert_eq!(results[1].chunk_id, 2);
    }

    #[test]
    fn find_document_matches_id_or_title_substring() {
        let index = Index {
            documents: vec![Document {
                id: "d1".into(),
                title: "Annual Report 2025".into(),
                chunk_indices: vec![],
            }],
            chunks: vec![],
        };
        assert!(index.find_document("d1").is_some());
        assert!(index.find_document("annual report").is_some());
        assert!(index.find_document("nonexistent").is_none());
    }

    #[test]
    fn roundtrips_through_bytes() {
        let index = Index {
            documents: vec![Document { id: "d1".into(), title: "T".into(), chunk_indices: vec![0] }],
            chunks: vec![chunk("d1", 0, vec![0.1, 0.2])],
        };
        let bytes = index.to_bytes().unwrap();
        let restored = Index::from_bytes(&bytes).unwrap();
        assert_eq!(restored.documents.len(), 1);
        assert_eq!(restored.chunks.len(), 1);
    }
}
