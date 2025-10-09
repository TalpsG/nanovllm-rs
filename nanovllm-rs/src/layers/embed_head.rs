use candle_core::{Error, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear};

use crate::utils::context::get_context;

pub struct VocabParallelEmbedding {
    pub embedding: Embedding,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}
impl VocabParallelEmbedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embedding: embedding(num_embeddings, embedding_dim, vb)?,
            num_embeddings,
            embedding_dim,
        })
    }
}
impl Module for VocabParallelEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.embedding.forward(xs)
    }
}

pub struct ParallelLMHead {
    pub linear: Linear,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub bias: bool,
}
impl ParallelLMHead {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear: linear(embedding_dim, num_embeddings, vb)?,
            num_embeddings,
            embedding_dim,
            bias,
        })
    }
}
impl Module for ParallelLMHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let handle = get_context();
        let context_guard = handle
            .read()
            .map_err(|e| Error::msg(format!("context lock poisoned: {e}")))?;
        let context = context_guard
            .as_ref()
            .ok_or_else(|| Error::msg("context not initialized"))?;
        if context.is_prefill {
            let mut last_indices = context.cu_seqlens_q.i(1..)?;
            let ones = Tensor::ones_like(&last_indices)?;
            last_indices = (last_indices - ones)?;
            let x = xs.i(&last_indices)?.contiguous()?;
            self.linear.forward(&x)
        } else {
            self.linear.forward(xs)
        }
    }
}
