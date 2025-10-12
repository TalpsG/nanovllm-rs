use std::path::{Path, PathBuf};

use anyhow::{Context as AnyhowContext, Result, anyhow, bail, ensure};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tracing::{debug, info, info_span, instrument, warn};

use crate::engine::sequence::Sequence;
use crate::layers::sampler::Sampler;
use crate::models::qwen3::Qwen3ForCausalLM;
use crate::utils::config::{Config, Qwen3Config};
use crate::utils::context::{Context, reset_context, set_context};

pub struct ModelRunner {
    config: Config,
    block_size: usize,
    enforce_eager: bool,
    _world_size: usize,
    _rank: usize,
    device: Device,
    sampler: Sampler,
    model: Qwen3ForCausalLM,
    kv_cache: Option<Tensor>,
    // TODO: tensor parallel specific shared-memory coordination and CUDA graph capture state.
}

impl ModelRunner {
    #[instrument(skip(config), fields(rank))]
    pub fn new(config: Config, rank: usize) -> Result<Self> {
        let world_size = config.tensor_parallel_size;
        if world_size > 1 {
            // TODO: support tensor parallel execution across ranks.
            bail!("tensor parallel execution is not yet implemented");
        }

        let device = Device::new_cuda(rank)
            .with_context(|| format!("failed to initialise CUDA device for rank {rank}"))?;

        let weight_paths = collect_weight_paths(&config.model)?;
        let dtype = resolve_dtype(&config.hf_config);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)? };
        let model = Qwen3ForCausalLM::new(&config.hf_config, vb)?;

        let sampler = Sampler::new();
        let block_size = config.kvcache_block_size;
        let enforce_eager = config.enforce_eager;

        let mut runner = Self {
            config,
            block_size,
            enforce_eager,
            _world_size: world_size,
            _rank: rank,
            device,
            sampler,
            model,
            kv_cache: None,
        };

        runner.warmup_model()?;
        runner.allocate_kv_cache()?;
        if !runner.enforce_eager {
            runner.capture_cudagraph()?;
        }

        Ok(runner)
    }

    #[instrument(skip(self))]
    fn warmup_model(&mut self) -> Result<()> {
        Ok(())
    }

    #[instrument(skip(self))]
    fn allocate_kv_cache(&mut self) -> Result<()> {
        let hf_config = &self.config.hf_config;
        let num_layers = hf_config.num_hidden_layers;
        ensure!(num_layers > 0, "model must contain at least one layer");

        let blocks_per_seq = (self.config.max_model_len + self.block_size - 1) / self.block_size;
        let theoretical_max = (self.config.max_num_seqs.max(1) * blocks_per_seq.max(1)).max(1);
        let theoretical_isize = isize::try_from(theoretical_max).unwrap_or(isize::MAX);

        let cache_dtype = resolve_dtype(hf_config);
        let dtype_bytes = match cache_dtype {
            DType::F16 | DType::BF16 => 2usize,
            DType::F32 => 4usize,
            DType::F64 => 8usize,
            other => bail!("unsupported dtype {:?} for KV cache allocation", other),
        };

        let num_kv_heads = hf_config.num_key_value_heads.max(1);
        let head_dim = hf_config.head_dim;
        let block_bytes_u128 = 2u128
            * num_layers as u128
            * self.block_size as u128
            * num_kv_heads as u128
            * head_dim as u128
            * dtype_bytes as u128;
        ensure!(block_bytes_u128 > 0, "KV cache block size must be positive");
        let block_bytes = usize::try_from(block_bytes_u128)
            .map_err(|_| anyhow!("KV cache block size exceeds usize limits"))?;

        if self.config.num_kvcache_blocks <= 0 {
            let (free_bytes, total_bytes) =
                cuda_mem_get_info().context("failed to query CUDA memory info")?;
            let util = self.config.gpu_memory_utilization.clamp(0.0, 1.0);
            ensure!(util > 0.0, "gpu_memory_utilization must be greater than 0");

            let used_bytes = total_bytes.saturating_sub(free_bytes);
            let target_bytes = ((total_bytes as f64) * util as f64) as i128;
            let mut budget_bytes = target_bytes - used_bytes as i128;
            if budget_bytes <= 0 {
                budget_bytes = free_bytes as i128;
            }
            budget_bytes = budget_bytes.min(free_bytes as i128);

            let mut blocks = if block_bytes == 0 {
                0i128
            } else {
                budget_bytes / block_bytes as i128
            };
            if blocks <= 0 {
                blocks = 1;
            }

            let capped = blocks.min(theoretical_isize as i128).max(1);
            self.config.num_kvcache_blocks = capped.try_into().unwrap_or(theoretical_isize.max(1));
            info!(
                free_bytes,
                total_bytes,
                used_bytes,
                block_bytes,
                budget_bytes,
                allocated_blocks = self.config.num_kvcache_blocks,
                "computed KV cache blocks from runtime memory"
            );
        }

        let mut num_blocks = self.config.num_kvcache_blocks as usize;
        num_blocks = num_blocks.min(theoretical_max as usize).max(1);
        ensure!(num_blocks > 0, "KV cache requires at least one block");

        let kv_cache = loop {
            let cache_shape = [
                2,
                num_layers,
                num_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
            ];
            let span = info_span!("kv_allocation_attempt", num_blocks);
            let _enter = span.enter();
            match Tensor::zeros(&cache_shape, cache_dtype, &self.device) {
                Ok(cache) => break cache,
                Err(err) => {
                    let msg = err.to_string();
                    if num_blocks <= 1 || !msg.to_lowercase().contains("out of memory") {
                        return Err(err.into());
                    }
                    num_blocks = num_blocks / 2;
                    warn!(
                        %msg,
                        num_blocks,
                        "reducing KV cache blocks after OOM"
                    );
                    continue;
                }
            }
        };
        self.config.num_kvcache_blocks = num_blocks as isize;
        info!(num_blocks, "KV cache allocation succeeded");

        let k_view = kv_cache.i(0)?;
        let v_view = kv_cache.i(1)?;
        for (layer_idx, layer) in self.model.model.layers_mut().iter_mut().enumerate() {
            let k_cache = k_view.i(layer_idx)?;
            let v_cache = v_view.i(layer_idx)?;
            layer.set_kv_cache(k_cache, v_cache);
        }

        self.kv_cache = Some(kv_cache);
        Ok(())
    }

    #[instrument(skip(self, seqs), fields(num_seqs = seqs.len()))]
    fn prepare_block_tables(&self, seqs: &[Sequence]) -> Result<Tensor> {
        if seqs.is_empty() {
            return Ok(Tensor::zeros((0, 0), DType::I64, &self.device)?);
        }
        let max_len = seqs
            .iter()
            .map(|seq| seq.block_table().len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            return Ok(Tensor::zeros((0, 0), DType::I64, &self.device)?);
        }

        let mut data = Vec::with_capacity(seqs.len() * max_len);
        for seq in seqs {
            for &block in seq.block_table() {
                let value = i64::try_from(block)
                    .with_context(|| format!("block index {block} exceeds i64 range"))?;
                data.push(value);
            }
            for _ in seq.block_table().len()..max_len {
                data.push(-1);
            }
        }

        Ok(Tensor::from_vec(data, (seqs.len(), max_len), &self.device)?)
    }

    #[instrument(skip(self, seqs), fields(num_seqs = seqs.len()))]
    fn prepare_prefill(&self, seqs: &[Sequence]) -> Result<(Tensor, Tensor)> {
        if seqs.is_empty() {
            let empty_i64 = Tensor::zeros((0,), DType::I64, &self.device)?;
            let empty_u32 = Tensor::zeros((0,), DType::U32, &self.device)?;
            let empty_tables = Tensor::zeros((0, 0), DType::I64, &self.device)?;
            set_context(Context::new(
                true,
                empty_u32.clone(),
                empty_u32.clone(),
                0,
                0,
                empty_i64.clone(),
                empty_i64.clone(),
                empty_tables,
            ));
            return Ok((empty_i64.clone(), empty_i64));
        }

        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping = Vec::new();

        for seq in seqs {
            let seqlen = seq.len();
            let cached = seq.num_cached_tokens();
            for token in &seq.token_ids()[cached..] {
                input_ids.push(*token as i64);
            }
            for pos in cached..seqlen {
                positions.push(pos as i64);
            }
            let seqlen_q = seqlen.saturating_sub(cached);
            let seqlen_k = seqlen;
            cu_seqlens_q.push(cu_seqlens_q.last().copied().unwrap_or(0) + seqlen_q as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().copied().unwrap_or(0) + seqlen_k as u32);
            max_seqlen_q = max_seqlen_q.max(seqlen_q);
            max_seqlen_k = max_seqlen_k.max(seqlen_k);

            if seq.block_table().is_empty() {
                continue;
            }
            for block_idx in seq.num_cached_blocks()..seq.num_blocks() {
                let block_id = seq.block_table()[block_idx];
                let start = block_id
                    .checked_mul(self.block_size)
                    .with_context(|| format!("block id {block_id} overflow in slot mapping"))?;
                let end = if block_idx + 1 < seq.num_blocks() {
                    start + self.block_size
                } else {
                    start + seq.last_block_num_tokens()
                };
                for slot in start..end {
                    let value = i64::try_from(slot)
                        .with_context(|| format!("slot {slot} exceeds i64 range"))?;
                    slot_mapping.push(value);
                }
            }
        }

        let block_tables = if cu_seqlens_k.last() > cu_seqlens_q.last() {
            // Prefix cache in play.
            self.prepare_block_tables(seqs)?
        } else {
            Tensor::zeros((0, 0), DType::I64, &self.device)?
        };
        let positions_len = positions.len();
        let slot_mapping_len = slot_mapping.len();
        let cu_q_len = cu_seqlens_q.len();
        let cu_k_len = cu_seqlens_k.len();
        let input_ids_tensor = Tensor::from_vec(input_ids, (positions_len,), &self.device)?;
        let positions_tensor = Tensor::from_vec(positions, (positions_len,), &self.device)?;
        let cu_seqlens_q_tensor =
            Tensor::from_vec(cu_seqlens_q.clone(), (cu_q_len,), &self.device)?;
        let cu_seqlens_k_tensor =
            Tensor::from_vec(cu_seqlens_k.clone(), (cu_k_len,), &self.device)?;
        let slot_mapping_tensor =
            Tensor::from_vec(slot_mapping, (slot_mapping_len,), &self.device)?;
        let empty_context_lens = Tensor::zeros((0,), DType::I64, &self.device)?;

        set_context(Context::new(
            true,
            cu_seqlens_q_tensor.clone(),
            cu_seqlens_k_tensor.clone(),
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping_tensor.clone(),
            empty_context_lens,
            block_tables.clone(),
        ));

        Ok((input_ids_tensor, positions_tensor))
    }

    #[instrument(skip(self, seqs), fields(num_seqs = seqs.len()))]
    fn prepare_decode(&self, seqs: &[Sequence]) -> Result<(Tensor, Tensor)> {
        if seqs.is_empty() {
            let empty_i64 = Tensor::zeros((0,), DType::I64, &self.device)?;
            let empty_u32 = Tensor::zeros((0,), DType::U32, &self.device)?;
            let empty_tables = Tensor::zeros((0, 0), DType::I64, &self.device)?;
            set_context(Context::new(
                false,
                empty_u32.clone(),
                empty_u32.clone(),
                0,
                0,
                empty_i64.clone(),
                empty_i64.clone(),
                empty_tables,
            ));
            return Ok((empty_i64.clone(), empty_i64));
        }

        let mut input_ids = Vec::with_capacity(seqs.len());
        let mut positions = Vec::with_capacity(seqs.len());
        let mut slot_mapping = Vec::with_capacity(seqs.len());
        let mut context_lens = Vec::with_capacity(seqs.len());

        for seq in seqs {
            input_ids.push(seq.last_token() as i64);
            positions.push((seq.len() - 1) as i64);
            context_lens.push(seq.len() as i64);

            let block_id = seq.block_table().last().copied().with_context(|| {
                format!("sequence {} missing block table for decode", seq.seq_id())
            })?;
            let last_tokens = seq.last_block_num_tokens();
            ensure!(last_tokens > 0, "sequence contains empty final block");
            let slot = block_id
                .checked_mul(self.block_size)
                .and_then(|base| base.checked_add(last_tokens - 1))
                .with_context(|| {
                    format!("slot computation overflow for sequence {}", seq.seq_id())
                })?;
            let value =
                i64::try_from(slot).with_context(|| format!("slot {slot} exceeds i64 range"))?;
            slot_mapping.push(value);
        }

        let block_tables = self.prepare_block_tables(seqs)?;
        let slot_mapping_len = slot_mapping.len();
        let context_lens_len = context_lens.len();
        let slot_mapping_tensor =
            Tensor::from_vec(slot_mapping, (slot_mapping_len,), &self.device)?;
        let context_lens_tensor =
            Tensor::from_vec(context_lens.clone(), (context_lens_len,), &self.device)?;
        let empty_cu = Tensor::zeros((0,), DType::U32, &self.device)?;

        set_context(Context::new(
            false,
            empty_cu.clone(),
            empty_cu.clone(),
            0,
            0,
            slot_mapping_tensor.clone(),
            context_lens_tensor.clone(),
            block_tables.clone(),
        ));

        let input_ids_tensor = Tensor::from_vec(input_ids, (seqs.len(),), &self.device)?;
        let positions_tensor = Tensor::from_vec(positions, (seqs.len(),), &self.device)?;
        Ok((input_ids_tensor, positions_tensor))
    }

    #[instrument(skip(self, seqs), fields(num_seqs = seqs.len()))]
    fn prepare_sample(&self, seqs: &[Sequence]) -> Result<Tensor> {
        if seqs.is_empty() {
            return Ok(Tensor::zeros((0,), DType::F32, &self.device)?);
        }
        let temps: Vec<f32> = seqs.iter().map(|seq| seq.temperature()).collect();
        Ok(Tensor::from_vec(temps, (seqs.len(),), &self.device)?)
    }

    #[instrument(skip(self, input_ids, positions))]
    fn run_model(
        &mut self,
        input_ids: &Tensor,
        positions: &Tensor,
        _is_prefill: bool,
        _expected_batch: usize,
    ) -> Result<Tensor> {
        debug!(
            input_shape = ?input_ids.shape(),
            pos_shape = ?positions.shape(),
            "model forward"
        );
        let hidden_states = self.model.forward(input_ids, positions)?;
        Ok(self.model.compute_logits(&hidden_states)?)
    }

    #[instrument(skip(self, seqs), fields(num_seqs = seqs.len(), is_prefill))]
    pub fn run(&mut self, seqs: &[Sequence], is_prefill: bool) -> Result<Vec<i64>> {
        let result = (|| {
            if seqs.is_empty() {
                return Ok(Vec::new());
            }

            let (input_ids, positions) = if is_prefill {
                self.prepare_prefill(seqs)?
            } else {
                self.prepare_decode(seqs)?
            };
            let logits = self.run_model(&input_ids, &positions, is_prefill, seqs.len())?;
            let temperatures = self.prepare_sample(seqs)?;
            let tokens = self.sampler.forward(&logits, &temperatures)?;
            let sampled = tokens.to_vec1::<i64>()?;
            Ok(sampled)
        })();

        reset_context();
        result
    }

    pub fn exit(&mut self) {
        // TODO: release tensor-parallel shared resources when multi-rank is supported.
        self.kv_cache = None;
    }

    fn capture_cudagraph(&mut self) -> Result<()> {
        // TODO: mirror CUDA graph capture once tensor parallel and graph pooling are wired up.
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

fn collect_weight_paths(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(model_dir)
        .with_context(|| format!("failed to list model directory {model_dir:?}"))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    ensure!(
        !paths.is_empty(),
        "no .safetensors files found under {:?}",
        model_dir
    );
    Ok(paths)
}

fn resolve_dtype(config: &Qwen3Config) -> DType {
    match config.dtype.as_deref() {
        Some("float16") | Some("fp16") => DType::F16,
        Some("float32") | Some("fp32") => DType::F32,
        Some("bfloat16") | Some("bf16") => DType::BF16,
        _ => DType::BF16,
    }
}

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
}

fn cuda_mem_get_info() -> Result<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    let status = unsafe { cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize) };
    ensure!(
        status == 0,
        "cuMemGetInfo_v2 failed with error code {status}"
    );
    Ok((free, total))
}
