use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Config {
    pub model: PathBuf,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub gpu_memory_utilization: f32,
    pub tensor_parallel_size: usize,
    pub enforce_eager: bool,
    pub eos: isize,
    pub kvcache_block_size: usize,
    pub num_kvcache_blocks: isize,
    pub hf_config: Qwen3Config,
}

impl Config {
    /// Load configuration data from a HuggingFace model directory.
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_path = model_dir.as_ref();
        ensure!(
            model_path.is_dir(),
            "model directory {:?} does not exist or is not a directory",
            model_path
        );

        let hf_config_path = model_path.join("config.json");
        ensure!(
            hf_config_path.is_file(),
            "expected HuggingFace config.json at {:?}",
            hf_config_path
        );

        let config_content = fs::read_to_string(&hf_config_path).with_context(|| {
            format!(
                "failed to read HuggingFace config from {:?}",
                hf_config_path
            )
        })?;
        let raw_config: Value = serde_json::from_str(&config_content).with_context(|| {
            format!(
                "failed to parse HuggingFace config from {:?}",
                hf_config_path
            )
        })?;

        let model_type = raw_config
            .get("model_type")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("model_type missing in HuggingFace config"))?;

        ensure!(
            model_type.eq_ignore_ascii_case("qwen3"),
            "unsupported model_type '{}' in HuggingFace config (expected qwen3)",
            model_type
        );

        let mut hf_config: Qwen3Config = serde_json::from_value(raw_config)
            .context("failed to deserialize Qwen3 configuration")?;
        hf_config.apply_post_init();

        let mut config = Self {
            model: model_path.to_path_buf(),
            ..Default::default()
        };

        config.apply_hf_overrides(&hf_config)?;
        config.hf_config = hf_config;
        config.validate()?;

        Ok(config)
    }

    /// Update derived fields based on the HuggingFace configuration.
    pub fn apply_hf_overrides(&mut self, hf_config: &Qwen3Config) -> Result<()> {
        if hf_config.max_position_embeddings > 0 {
            self.max_model_len = self.max_model_len.min(hf_config.max_position_embeddings);
        }

        self.eos = extract_eos_token_id(&hf_config.eos_token_id);

        Ok(())
    }

    /// Ensure configuration invariants mirror the Python implementation.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.kvcache_block_size % 256 == 0,
            "kvcache_block_size must be a multiple of 256 (got {})",
            self.kvcache_block_size
        );
        ensure!(
            (1..=8).contains(&self.tensor_parallel_size),
            "tensor_parallel_size must be within [1, 8] (got {})",
            self.tensor_parallel_size
        );
        ensure!(
            self.max_num_batched_tokens >= self.max_model_len,
            "max_num_batched_tokens ({}) must be >= max_model_len ({})",
            self.max_num_batched_tokens,
            self.max_model_len
        );

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            max_num_batched_tokens: 16384,
            max_num_seqs: 512,
            max_model_len: 4096,
            gpu_memory_utilization: 0.9,
            tensor_parallel_size: 1,
            enforce_eager: false,
            eos: -1,
            kvcache_block_size: 256,
            num_kvcache_blocks: -1,
            hf_config: Qwen3Config::default(),
        }
    }
}

pub fn extract_eos_token_id(value: &Value) -> isize {
    match value {
        Value::Number(number) => number.as_i64().map(|v| v as isize).unwrap_or(-1),
        Value::Array(values) => values
            .iter()
            .filter_map(|item| item.as_i64())
            .map(|v| v as isize)
            .next()
            .unwrap_or(-1),
        _ => -1,
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: Value,
    #[serde(default, alias = "torch_dtype")]
    pub dtype: Option<String>,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: Value,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_layer_types")]
    pub layer_types: Vec<String>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub transformers_version: Option<String>,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default = "default_use_sliding_window")]
    pub use_sliding_window: bool,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_extra")]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl Qwen3Config {
    pub fn apply_post_init(&mut self) {
        if self.num_key_value_heads == 0 {
            self.num_key_value_heads = self.num_attention_heads;
        }

        if let Some(Value::Object(map)) = self.rope_scaling.as_mut() {
            if let Some(type_value) = map.remove("type") {
                map.entry("rope_type".to_string()).or_insert(type_value);
            }
        }

        if !self.use_sliding_window {
            self.sliding_window = None;
        }

        if self.max_window_layers > self.num_hidden_layers {
            self.max_window_layers = self.num_hidden_layers;
        }

        if self.layer_types.len() != self.num_hidden_layers {
            self.layer_types = compute_layer_types(
                self.num_hidden_layers,
                self.sliding_window,
                self.max_window_layers,
            );
        }
    }
}

impl Default for Qwen3Config {
    fn default() -> Self {
        let mut config = Self {
            architectures: None,
            attention_bias: default_attention_bias(),
            attention_dropout: default_attention_dropout(),
            bos_token_id: default_bos_token_id(),
            dtype: None,
            eos_token_id: default_eos_token_id(),
            head_dim: default_head_dim(),
            hidden_act: default_hidden_act(),
            hidden_size: default_hidden_size(),
            initializer_range: default_initializer_range(),
            intermediate_size: default_intermediate_size(),
            layer_types: default_layer_types(),
            max_position_embeddings: default_max_position_embeddings(),
            max_window_layers: default_max_window_layers(),
            model_type: default_model_type(),
            num_attention_heads: default_num_attention_heads(),
            num_hidden_layers: default_num_hidden_layers(),
            num_key_value_heads: default_num_key_value_heads(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_scaling: None,
            rope_theta: default_rope_theta(),
            sliding_window: None,
            tie_word_embeddings: default_tie_word_embeddings(),
            transformers_version: None,
            use_cache: default_use_cache(),
            use_sliding_window: default_use_sliding_window(),
            vocab_size: default_vocab_size(),
            extra: default_extra(),
        };
        config.apply_post_init();
        config
    }
}

pub fn compute_layer_types(
    num_layers: usize,
    sliding_window: Option<usize>,
    max_window_layers: usize,
) -> Vec<String> {
    let mut layers = Vec::with_capacity(num_layers);
    for idx in 0..num_layers {
        if sliding_window.is_some() && idx >= max_window_layers {
            layers.push("sliding_attention".to_string());
        } else {
            layers.push("full_attention".to_string());
        }
    }
    layers
}

pub fn default_attention_bias() -> bool {
    false
}

pub fn default_attention_dropout() -> f32 {
    0.0
}

pub fn default_bos_token_id() -> Value {
    Value::Null
}

fn default_eos_token_id() -> Value {
    Value::Null
}

fn default_head_dim() -> usize {
    128
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_hidden_size() -> usize {
    4096
}

fn default_initializer_range() -> f32 {
    0.02
}

fn default_intermediate_size() -> usize {
    22_016
}

fn default_layer_types() -> Vec<String> {
    Vec::new()
}

fn default_max_position_embeddings() -> usize {
    32_768
}

fn default_max_window_layers() -> usize {
    28
}

fn default_model_type() -> String {
    String::new()
}

fn default_num_attention_heads() -> usize {
    32
}

fn default_num_hidden_layers() -> usize {
    32
}

fn default_num_key_value_heads() -> usize {
    0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_tie_word_embeddings() -> bool {
    false
}

fn default_use_cache() -> bool {
    true
}

fn default_use_sliding_window() -> bool {
    false
}

fn default_vocab_size() -> usize {
    151_936
}

fn default_extra() -> HashMap<String, Value> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context as AnyhowContext, Result as AnyhowResult};
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn extract_eos_token_id_handles_array() {
        let eos = json!([151_643, 151_644, 151_645]);
        assert_eq!(extract_eos_token_id(&eos), 151_643);
    }

    #[test]
    fn config_rejects_non_qwen3() -> AnyhowResult<()> {
        let temp = tempdir()?;
        let cfg_path = temp.path().join("config.json");
        fs::write(&cfg_path, r#"{"model_type": "llama"}"#)?;

        let err = Config::from_model_dir(temp.path());
        assert!(err.is_err());
        let message = format!("{}", err.unwrap_err());
        assert!(
            message.contains("qwen3"),
            "unexpected error message: {}",
            message
        );

        Ok(())
    }

    #[test]
    fn config_from_qwen3_filesystem() -> AnyhowResult<()> {
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        let model_dir = Path::new(&home).join("huggingface/Qwen3-0.6B");

        if !model_dir.is_dir() {
            eprintln!(
                "Skipping config_from_qwen3_filesystem: {:?} missing",
                model_dir
            );
            return Ok(());
        }

        let config = Config::from_model_dir(&model_dir)
            .with_context(|| format!("failed to load config from {:?}", model_dir))?;
        println!("{:#?}", config);

        assert_eq!(config.model, model_dir);
        assert_eq!(config.max_num_batched_tokens, 16_384);
        assert_eq!(config.max_num_seqs, 512);
        assert_eq!(config.max_model_len, 4_096);
        assert!((config.gpu_memory_utilization - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.tensor_parallel_size, 1);
        assert!(!config.enforce_eager);

        assert_eq!(config.eos, 151_645);

        let hf = &config.hf_config;
        assert_eq!(hf.model_type.to_lowercase(), "qwen3");
        assert_eq!(hf.max_position_embeddings, 40_960);
        assert_eq!(hf.hidden_size, 1_024);
        assert_eq!(hf.num_hidden_layers, 28);
        assert_eq!(hf.num_attention_heads, 16);
        assert_eq!(hf.num_key_value_heads, 8);
        assert_eq!(hf.rope_theta, 1_000_000.0);
        assert!(hf.tie_word_embeddings);
        assert_eq!(
            hf.architectures
                .as_ref()
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("Qwen3ForCausalLM")
        );
        assert_eq!(hf.layer_types.len(), hf.num_hidden_layers);
        assert_eq!(extract_eos_token_id(&hf.eos_token_id), 151_645);
        assert_eq!(hf.dtype.as_deref(), Some("bfloat16"));

        Ok(())
    }
}
