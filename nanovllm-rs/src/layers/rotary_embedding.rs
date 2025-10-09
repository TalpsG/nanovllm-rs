use candle_core::{DType, IndexOp, Result, Tensor, bail};

use crate::engine::llm_engine::DEVICE;
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let last_dim = x.dims().len().saturating_sub(1);
    let orig_dtype = x.dtype();

    let x_f32 = x.to_dtype(DType::F32)?;
    let pieces = x_f32.chunk(2, last_dim)?;
    let x1 = &pieces[0];
    let x2 = &pieces[1];

    let y1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let y2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
    let parts = [&y1, &y2];
    let rotated = Tensor::cat(&parts, last_dim)?;
    rotated.to_dtype(orig_dtype)
}

pub struct RotaryEmbedding {
    pub sin: Tensor,
    pub cos: Tensor,
    pub head_size: usize,
}

impl RotaryEmbedding {
    pub fn new(
        head_size: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        base: f32,
    ) -> Result<Self> {
        assert_eq!(head_size, rotary_dim, "rotary_dim must equal head_size");
        assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
        let device = &DEVICE;

        let half_dim = rotary_dim / 2;
        let indices = Tensor::arange(0f32, half_dim as f32, device)?;
        let scale_two = Tensor::full(2f32, indices.shape(), device)?;
        let even_positions = (&indices * &scale_two)?;
        let inv_rotary = Tensor::full(1f32 / (rotary_dim as f32), even_positions.shape(), device)?;
        let exponent = (&even_positions * &inv_rotary)?;
        let neg_ln_base = Tensor::full(-base.ln(), exponent.shape(), device)?;
        let scaled = (&exponent * &neg_ln_base)?;
        let inv_freq = scaled.exp()?;
        let freqs = Tensor::arange(0f32, max_position_embeddings as f32, device)?
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq)?;

        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            head_size,
        })
    }

    pub fn forward(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let target_rank = query.dims().len();
        if target_rank == 0 {
            bail!("query tensor at least needs 1 dimension");
        }
        let mut sin = self.sin.i(positions)?;
        let mut cos = self.cos.i(positions)?;
        while sin.dims().len() < target_rank {
            let axis = sin.dims().len().saturating_sub(1);
            sin = sin.unsqueeze(axis)?;
            cos = cos.unsqueeze(axis)?;
        }

        let query = apply_rotary_emb(query, &cos, &sin)?;
        let key = apply_rotary_emb(key, &cos, &sin)?;
        Ok((query, key))
    }
}

#[cfg(test)]
mod tests {
    use super::RotaryEmbedding;
    use crate::engine::llm_engine::DEVICE;
    use candle_core::{IndexOp, Result, Tensor};
    use candle_nn::rotary_emb::rope_thd;

    fn assert_close(lhs: &[Vec<Vec<f32>>], rhs: &[Vec<Vec<f32>>], tol: f32) {
        for (row_l, row_r) in lhs.iter().zip(rhs.iter()) {
            for (col_l, col_r) in row_l.iter().zip(row_r.iter()) {
                for (&a, &b) in col_l.iter().zip(col_r.iter()) {
                    assert!((a - b).abs() <= tol, "mismatch {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn rotary_embedding_matches_candle_rope() -> Result<()> {
        let device = &*DEVICE;
        let seq_len = 4;
        let num_heads = 3;
        let head_dim = 8;
        let base = 10_000f32;

        let rope = RotaryEmbedding::new(head_dim, head_dim, 512, base)?;

        let positions = Tensor::arange(0i64, seq_len as i64, device)?;
        let total = (seq_len * num_heads * head_dim) as f32;
        let query = Tensor::arange(0f32, total, device)?.reshape((seq_len, num_heads, head_dim))?;
        let key =
            Tensor::arange(1f32, total + 1f32, device)?.reshape((seq_len, num_heads, head_dim))?;

        let cos_ref = rope.cos.i(&positions)?.contiguous()?;
        let sin_ref = rope.sin.i(&positions)?.contiguous()?;

        let query_rope = query.unsqueeze(0)?;
        let key_rope = key.unsqueeze(0)?;
        let expected_query =
            rope_thd(&query_rope, &cos_ref, &sin_ref)?.reshape((seq_len, num_heads, head_dim))?;
        let expected_key =
            rope_thd(&key_rope, &cos_ref, &sin_ref)?.reshape((seq_len, num_heads, head_dim))?;

        let (actual_query, actual_key) = rope.forward(&positions, &query, &key)?;

        let expected_query_vec = expected_query.to_vec3::<f32>()?;
        let actual_query_vec = actual_query.to_vec3::<f32>()?;
        assert_close(&actual_query_vec, &expected_query_vec, 1e-5);

        let expected_key_vec = expected_key.to_vec3::<f32>()?;
        let actual_key_vec = actual_key.to_vec3::<f32>()?;
        assert_close(&actual_key_vec, &expected_key_vec, 1e-5);

        Ok(())
    }
}
