use candle_core::{DType, IndexOp, Result, Tensor, bail};
use candle_flash_attn::flash_attn_varlen;

use crate::utils::context::get_context;

/// FlashAttention-backed multi-head attention with an optional KV cache.
///
/// Mirrors the behaviour of the Python implementation, relying on the
/// `flash_attn_varlen` kernel for both prefill (variable sequence lengths)
/// and decode (single-token queries) paths. Prefix-cache support via
/// block tables is not implemented yet.
pub struct Attention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub num_kv_heads: usize,
    pub k_cache: Option<Tensor>,
    pub v_cache: Option<Tensor>,
}

impl Attention {
    pub fn new(num_heads: usize, head_dim: usize, scale: f32, num_kv_heads: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            k_cache: None,
            v_cache: None,
        }
    }

    /// Attach pre-allocated KV cache tensors used during decode.
    pub fn set_cache(&mut self, k_cache: Tensor, v_cache: Tensor) {
        self.k_cache = Some(k_cache);
        self.v_cache = Some(v_cache);
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let context_handle = get_context();
        let guard = context_handle
            .read()
            .expect("context lock poisoned while reading");
        let context = guard
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("context not initialised".into()))?;

        ensure_rank3(q, "query")?;
        ensure_rank3(k, "key")?;
        ensure_rank3(v, "value")?;

        if let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) {
            store_in_cache(
                k_cache,
                k,
                &context.slot_mapping,
                self.num_kv_heads,
                self.head_dim,
            )?;
            store_in_cache(
                v_cache,
                v,
                &context.slot_mapping,
                self.num_kv_heads,
                self.head_dim,
            )?;
        }

        let target_dtype = q.dtype();

        if context.is_prefill {
            let q_flash = to_flash_dtype(q)?;
            let k_flash = to_flash_dtype(k)?;
            let v_flash = to_flash_dtype(v)?;
            if has_data(&context.block_tables) {
                bail!("prefix cache via block tables is not yet supported in Rust attention");
            }
            let output = flash_attn_varlen(
                &q_flash,
                &k_flash,
                &v_flash,
                &context.cu_seqlens_q,
                &context.cu_seqlens_k,
                context.max_seqlen_q,
                context.max_seqlen_k,
                self.scale,
                true,
            )?;
            return from_flash_dtype(output, target_dtype);
        }

        let (k_cache, v_cache) = match (&self.k_cache, &self.v_cache) {
            (Some(k_cache), Some(v_cache)) => (k_cache, v_cache),
            _ => bail!("decode path requires an initialised KV cache"),
        };
        let context_lens = context.context_lens.to_vec1::<i64>()?;
        if context_lens.is_empty() {
            bail!("decode context_lens must be provided");
        }
        let decoded_lengths: Vec<usize> = context_lens
            .iter()
            .map(|len| (*len).max(0) as usize)
            .collect();
        let batch = decoded_lengths.len();

        let device = q.device();
        let mut seqlens_q = Vec::with_capacity(batch + 1);
        for i in 0..=batch {
            seqlens_q.push(i as u32);
        }
        let seqlens_q = Tensor::from_vec(seqlens_q, (batch + 1,), device)?;

        let (k_total, v_total, seqlens_k, max_seqlen_k) = if has_data(&context.block_tables) {
            let block_tables = context.block_tables.to_vec2::<i64>()?;
            let cache_dims = k_cache.dims().to_vec();
            let block_size = match cache_dims.as_slice() {
                [_blocks, block_sz, heads, dim]
                    if *heads == self.num_kv_heads && *dim == self.head_dim =>
                {
                    *block_sz
                }
                [total, heads, dim] if *heads == self.num_kv_heads && *dim == self.head_dim => {
                    // Cache already flattened; treat entire dimension as a single block.
                    *total
                }
                shape => {
                    bail!(
                        "unexpected KV cache shape {:?}, expected (blocks, block_size, {}, {})",
                        shape,
                        self.num_kv_heads,
                        self.head_dim
                    )
                }
            };

            let mut indices = Vec::new();
            let mut cu_seqlens_k = Vec::with_capacity(batch + 1);
            cu_seqlens_k.push(0u32);
            let mut max_len = 0usize;

            for (i, blocks_row) in block_tables.iter().enumerate().take(batch) {
                let mut remaining = decoded_lengths.get(i).copied().unwrap_or(0);
                let mut collected = 0usize;
                for &block_id in blocks_row {
                    if remaining == 0 || block_id < 0 {
                        break;
                    }
                    let take = remaining.min(block_size);
                    let base = (block_id as usize).checked_mul(block_size).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "block id {block_id} overflow for sequence {i}"
                        ))
                    })?;
                    for offset in 0..take {
                        indices.push((base + offset) as i64);
                    }
                    remaining -= take;
                    collected += take;
                }
                if remaining != 0 {
                    bail!(
                        "block table for sequence {i} lacks coverage for {} tokens",
                        decoded_lengths.get(i).copied().unwrap_or(0)
                    );
                }
                cu_seqlens_k.push(indices.len() as u32);
                max_len = max_len.max(collected);
            }

            let index_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), device)?;
            let k_matrix = cache_as_matrix(k_cache, self.num_kv_heads, self.head_dim)?;
            let v_matrix = cache_as_matrix(v_cache, self.num_kv_heads, self.head_dim)?;
            let k_selected = k_matrix.i(&index_tensor)?.reshape((
                indices.len(),
                self.num_kv_heads,
                self.head_dim,
            ))?;
            let v_selected = v_matrix.i(&index_tensor)?.reshape((
                indices.len(),
                self.num_kv_heads,
                self.head_dim,
            ))?;
            let seqlens_k_tensor =
                Tensor::from_vec(cu_seqlens_k.clone(), (cu_seqlens_k.len(),), device)?;
            (k_selected, v_selected, seqlens_k_tensor, max_len.max(1))
        } else {
            let total_k: usize = decoded_lengths.iter().sum();
            let k_flat = cache_as_sequence(k_cache, self.num_kv_heads, self.head_dim)?;
            let v_flat = cache_as_sequence(v_cache, self.num_kv_heads, self.head_dim)?;

            if total_k > k_flat.dims()[0] {
                bail!("KV cache does not contain enough tokens for decode");
            }

            let k_total = if total_k == k_flat.dims()[0] {
                k_flat.clone()
            } else {
                k_flat.narrow(0, 0, total_k)?
            };
            let v_total = if total_k == v_flat.dims()[0] {
                v_flat.clone()
            } else {
                v_flat.narrow(0, 0, total_k)?
            };

            let mut cu_seqlens_k = Vec::with_capacity(batch + 1);
            cu_seqlens_k.push(0u32);
            let mut acc = 0usize;
            let mut max_len = 0usize;
            for &len in &decoded_lengths {
                acc += len;
                cu_seqlens_k.push(acc as u32);
                max_len = max_len.max(len);
            }
            let seqlens_k_tensor = Tensor::from_vec(cu_seqlens_k, (batch + 1,), device)?;
            (k_total, v_total, seqlens_k_tensor, max_len.max(1))
        };

        let q_flash = to_flash_dtype(q)?;
        let k_flash = to_flash_dtype(&k_total)?;
        let v_flash = to_flash_dtype(&v_total)?;

        let output = flash_attn_varlen(
            &q_flash,
            &k_flash,
            &v_flash,
            &seqlens_q,
            &seqlens_k,
            1,
            max_seqlen_k,
            self.scale,
            true,
        )?;
        from_flash_dtype(output, target_dtype)
    }
}

fn has_data(t: &Tensor) -> bool {
    t.elem_count() > 0
}

fn ensure_rank3(t: &Tensor, name: &str) -> Result<()> {
    if t.dims().len() != 3 {
        bail!("{name} tensor must be rank-3, got shape {:?}", t.dims());
    }
    Ok(())
}

fn to_flash_dtype(t: &Tensor) -> Result<Tensor> {
    match t.dtype() {
        DType::F16 => Ok(t.clone()),
        DType::BF16 | DType::F32 => t.to_dtype(DType::F16),
        dtype => bail!("flash attention requires F16 inputs, got {dtype:?}"),
    }
}

fn from_flash_dtype(t: Tensor, target: DType) -> Result<Tensor> {
    if target == DType::F16 {
        Ok(t)
    } else {
        t.to_dtype(target)
    }
}

fn cache_as_sequence(cache: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let dims = cache.dims();
    match dims {
        [tokens, heads, dim] if heads == &num_heads && dim == &head_dim => {
            cache.contiguous()?.reshape((*tokens, *heads, *dim))
        }
        [blocks, block_size, heads, dim] if heads == &num_heads && dim == &head_dim => {
            let total = blocks * block_size;
            cache.contiguous()?.reshape((total, *heads, *dim))
        }
        shape => bail!(
            "unsupported cache shape {:?}, expected (tokens, {}, {})",
            shape,
            num_heads,
            head_dim
        ),
    }
}

fn cache_as_matrix(cache: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let dims = cache.dims();
    match dims {
        [tokens, heads, dim] if heads == &num_heads && dim == &head_dim => {
            cache.contiguous()?.reshape((*tokens, num_heads * head_dim))
        }
        [blocks, block_size, heads, dim] if heads == &num_heads && dim == &head_dim => {
            let total = blocks * block_size;
            cache.contiguous()?.reshape((total, num_heads * head_dim))
        }
        shape => bail!(
            "unsupported cache shape {:?}, expected (tokens, {}, {})",
            shape,
            num_heads,
            head_dim
        ),
    }
}

fn store_in_cache(
    cache: &Tensor,
    values: &Tensor,
    slot_mapping: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if values.elem_count() == 0 || slot_mapping.elem_count() == 0 {
        return Ok(());
    }

    let value_dims = values.dims();
    if value_dims.len() != 3 {
        bail!(
            "expected (tokens, heads, head_dim) for cache update, got {:?}",
            value_dims
        );
    }
    if value_dims[1] != num_heads || value_dims[2] != head_dim {
        bail!(
            "value tensor shape mismatch {}, {} vs cache {} {},",
            value_dims[1],
            value_dims[2],
            num_heads,
            head_dim
        );
    }

    let slot_vec = slot_mapping.to_vec1::<i64>()?;
    if slot_vec.len() != value_dims[0] {
        bail!(
            "slot mapping length {} must match number of tokens {}",
            slot_vec.len(),
            value_dims[0]
        );
    }

    let mut selected_rows = Vec::new();
    let mut dst_slots = Vec::new();
    for (row, slot) in slot_vec.into_iter().enumerate() {
        if slot >= 0 {
            selected_rows.push(row as i64);
            dst_slots.push(slot as u32);
        }
    }

    if selected_rows.is_empty() {
        return Ok(());
    }

    let device = values.device();
    let slot_count = dst_slots.len();
    let row_tensor = Tensor::from_vec(selected_rows, (slot_count,), device)?;
    let slots_tensor = Tensor::from_vec(dst_slots, (slot_count,), device)?;

    let selected = values.i(&row_tensor)?;
    let head_size = num_heads * head_dim;
    let selected = selected.reshape((slot_count, head_size))?;

    let cache_view = cache_as_matrix(cache, num_heads, head_dim)?;
    let scatter_index = slots_tensor.unsqueeze(1)?.repeat((1, head_size))?;
    cache_view.scatter_set(&scatter_index, &selected, 0)
}
