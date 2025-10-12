use candle_core::{Result, Tensor, bail};
use candle_nn::{Module, VarBuilder};

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
        let o = self.attn.forward(&q, &k, &v)?;
        let o = o.reshape((seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&o)
    }

    pub fn set_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) {
        self.attn.set_cache(k_cache, v_cache);
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
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = if let Some(residual) = residual {
            self.input_layernorm
                .forward_with_residual(hidden_states, residual)?
        } else {
            let normalized = self.input_layernorm.forward(hidden_states)?;
            (normalized, hidden_states.clone())
        };

        let hidden_states = self.self_attn.forward(positions, &hidden_states)?;
        let (hidden_states, residual) = self
            .post_attention_layernorm
            .forward_with_residual(&hidden_states, &residual)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
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
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;
        for layer in self.layers.iter_mut() {
            let (next_hidden, next_residual) =
                layer.forward(positions, &hidden_states, residual.as_ref())?;
            hidden_states = next_hidden;
            residual = Some(next_residual);
        }
        if let Some(residual) = residual.as_ref() {
            let (hidden_states, _) = self.norm.forward_with_residual(&hidden_states, residual)?;
            Ok(hidden_states)
        } else {
            self.norm.forward(&hidden_states)
        }
    }

    pub fn layers_mut(&mut self) -> &mut [Qwen3DecoderLayer] {
        &mut self.layers
    }
}

pub struct Qwen3ForCausalLM {
    pub model: Qwen3Model,
    pub lm_head: ParallelLMHead,
    pub tie_word_embeddings: bool,
}

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
