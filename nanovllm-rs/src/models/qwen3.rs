use std::cell::{Ref, RefCell};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use candle_core::{Result, Tensor, bail};
use candle_nn::{Module, VarBuilder};
use tracing::instrument;

use crate::layers::{
    activation::SiluAndMul,
    attention::Attention,
    embed_head::{ParallelLMHead, VocabParallelEmbedding},
    layernorm::RMSNorm,
    linear::{LinearBase, RowParallelLinear},
    rotary_embedding::RotaryEmbedding,
};
use crate::utils::config::Qwen3Config;

pub struct Qwen3Attention {
    q_proj: LinearBase,
    k_proj: LinearBase,
    v_proj: LinearBase,
    o_proj: RowParallelLinear,
    rotary_emb: RotaryEmbedding,
    attn: Attention,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // profile
    attn_cost:RefCell<u64>,
}

impl Qwen3Attention {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = if config.num_key_value_heads == 0 {
            config.num_attention_heads
        } else {
            config.num_key_value_heads
        };
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).recip().sqrt();

        let q_proj = LinearBase::new(
            hidden_size,
            num_heads * head_dim,
            config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = LinearBase::new(
            hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = LinearBase::new(
            hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj =
            RowParallelLinear::new(num_heads * head_dim, hidden_size, false, vb.pp("o_proj"))?;
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;
        let attn = Attention::new(num_heads, head_dim, scale, num_kv_heads);
        let q_norm = RMSNorm::new(head_dim, Some(config.rms_norm_eps), vb.pp("q_norm"))?;
        let k_norm = RMSNorm::new(head_dim, Some(config.rms_norm_eps), vb.pp("k_norm"))?;

        Ok(Self {
            attn_cost: RefCell::new(0),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            attn,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn forward(&self, positions: &Tensor, hidden_states: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;
        let seq_len = q.dims()[0];
        let q = q.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = self.rotary_emb.forward(positions, &q, &k)?;
        let attn_start = Instant::now();
        let o = self.attn.forward(&q, &k, &v)?;
        *self.attn_cost.borrow_mut() = attn_start.elapsed().as_millis() as u64;
        let o = o.reshape((seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&o)
    }

    pub fn set_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) {
        self.attn.set_cache(k_cache, v_cache);
    }
    pub fn get_flash_attn_cost(&self) -> u64 {
        *self.attn.flash_attn_cost.borrow()
    }
    pub fn get_attn_cost(&self) -> u64 {
        *self.attn_cost.borrow()
    }
    pub fn get_store_in_cache_cost(&self) -> u64 {
        *self.attn.store_in_cache_cost.borrow()
    }
}

pub struct Qwen3MLP {
    gate_proj: LinearBase,
    up_proj: LinearBase,
    down_proj: RowParallelLinear,
    act_fn: SiluAndMul,
}

impl Qwen3MLP {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        hidden_act: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        if hidden_act.to_lowercase() != "silu" {
            bail!("unsupported activation '{hidden_act}' for Qwen3 MLP");
        }
        let gate_proj = LinearBase::new(hidden_size, intermediate_size, false, vb.pp("gate_proj"))?;
        let up_proj = LinearBase::new(hidden_size, intermediate_size, false, vb.pp("up_proj"))?;
        let down_proj =
            RowParallelLinear::new(intermediate_size, hidden_size, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: SiluAndMul,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(hidden_states)?;
        let up = self.up_proj.forward(hidden_states)?;
        let last_dim = gate.dims().len() - 1;
        let gate_up = Tensor::cat(&[gate, up], last_dim)?;
        let activated = self.act_fn.forward(&gate_up)?;
        self.down_proj.forward(&activated)
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen3DecoderLayer {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3Attention::new(config, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            vb.pp("mlp"),
        )?;
        let input_layernorm = RMSNorm::new(
            config.hidden_size,
            Some(config.rms_norm_eps),
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RMSNorm::new(
            config.hidden_size,
            Some(config.rms_norm_eps),
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        positions: &Tensor,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let profile = LAYER_PROFILE.load(Ordering::Relaxed);
        let layer_start = profile.then(Instant::now);
        let (hidden_states, residual) = if let Some(residual) = residual {
            self.input_layernorm
                .forward_with_residual(hidden_states, residual)?
        } else {
            let normalized = self.input_layernorm.forward(hidden_states)?;
            (normalized, hidden_states.clone())
        };

        let attn_start = profile.then(Instant::now);
        let hidden_states = self.self_attn.forward(positions, &hidden_states)?;
        let attn_elapsed = attn_start.map(|start| start.elapsed().as_millis() as u64);
        let (hidden_states, residual) = self
            .post_attention_layernorm
            .forward_with_residual(&hidden_states, &residual)?;
        let mlp_start = profile.then(Instant::now);
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let mlp_elapsed = mlp_start.map(|start| start.elapsed().as_millis() as u64);
        if let Some(total_start) = layer_start {
            let total = total_start.elapsed().as_millis() as u64;
            LAYER_STATS.with(|stats| {
                stats.borrow_mut().push(LayerTiming {
                    index: layer_idx,
                    qwen_attention: attn_elapsed.unwrap_or(0),
                    attn: self.self_attn.get_attn_cost(),
                    store_kvcache: self.self_attn.get_store_in_cache_cost(),
                    flash_attn: self.self_attn.get_flash_attn_cost(),
                    mlp: mlp_elapsed.unwrap_or(0),
                    total,
                })
            });
        }
        Ok((hidden_states, residual))
    }

    pub fn set_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) {
        self.self_attn.set_kv_cache(k_cache, v_cache);
    }
}

pub struct Qwen3Model {
    embed_tokens: VocabParallelEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
}

impl Qwen3Model {
    fn take_layer_stats() -> Vec<LayerTiming> {
        LAYER_STATS.with(|stats| stats.borrow().clone())
    }

    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = VocabParallelEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let layer_vb = vb.pp(&format!("layers.{idx}"));
            layers.push(Qwen3DecoderLayer::new(config, layer_vb)?);
        }
        let norm = RMSNorm::new(config.hidden_size, Some(config.rms_norm_eps), vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        static PROFILE_CALLS: AtomicUsize = AtomicUsize::new(0);
        let call_index = PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut layer_durations = Vec::with_capacity(self.layers.len());
        let forward_start = Instant::now();
        LAYER_STATS.with(|stats| stats.borrow_mut().clear());
        LAYER_PROFILE.store(true, Ordering::Relaxed);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_start = Instant::now();
            let (next_hidden, next_residual) =
                layer.forward(positions, &hidden_states, residual.as_ref(), layer_idx)?;
            hidden_states = next_hidden;
            residual = Some(next_residual);
            layer_durations.push((layer_idx, layer_start.elapsed().as_secs_f64()));
        }
        let detailed_stats = Self::take_layer_stats();
        if let Some(residual) = residual.as_ref() {
            let (hidden_states, _) = self.norm.forward_with_residual(&hidden_states, residual)?;
            let hidden_states = hidden_states;
            tracing::info!(
                category="perf",
                call_index,
                total = forward_start.elapsed().as_secs_f64(),
                ?layer_durations,
                ?detailed_stats,
                "decoder forward profile"
            );
            Ok(hidden_states)
        } else {
            let hidden_states = self.norm.forward(&hidden_states)?;
            tracing::info!(
                category="perf",
                call_index,
                total = forward_start.elapsed().as_secs_f64(),
                ?layer_durations,
                ?detailed_stats,
                "decoder forward profile"
            );
            Ok(hidden_states)
        }
    }

    pub fn layers_mut(&mut self) -> &mut [Qwen3DecoderLayer] {
        &mut self.layers
    }
    pub fn profile_flash_attention(&self) {
        // Log profiling information for flash attention (unit is second)
    }
}

pub struct Qwen3ForCausalLM {
    pub model: Qwen3Model,
    pub lm_head: ParallelLMHead,
    pub tie_word_embeddings: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct LayerTiming {
    index: usize,
    qwen_attention: u64,
    attn: u64,
    store_kvcache:u64,
    flash_attn: u64,
    mlp: u64,
    total: u64,
}

thread_local! {
    static LAYER_STATS: RefCell<Vec<LayerTiming>> = RefCell::new(Vec::new());
}

static LAYER_PROFILE: AtomicBool = AtomicBool::new(false);

impl Qwen3ForCausalLM {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3Model::new(config, vb.pp("model"))?;
        let lm_head = ParallelLMHead::new(
            config.vocab_size,
            config.hidden_size,
            false,
            vb.pp("lm_head"),
        )?;
        Ok(Self {
            model,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, positions)
    }

    pub fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }
}
