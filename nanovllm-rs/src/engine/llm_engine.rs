use crate::utils::config::Config;
use anyhow::{Result, anyhow};
use candle_core::{CudaDevice, Device, backend::BackendDevice};
use candle_nn::VarBuilder;
use lazy_static::lazy_static;
use tokenizers::tokenizer::Tokenizer;

struct LLM_Engine {
    config: Config,
    tokenizer: Tokenizer,
    // TODO: tensor parallelism
}
lazy_static! {
    pub static ref DEVICE: Device = Device::new_cuda(0).expect("Failed to create CUDA device");
}

impl LLM_Engine {
    pub fn new(model: &str) -> Result<Self> {
        let config = Config::from_model_dir(model)?;
        let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model))
            .map_err(|err| anyhow!("Failed to load tokenizer: {}", err))?;
        let weight_map = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[format!("{}/model.safetensors", model)],
                candle_core::DType::BF16,
                &DEVICE,
            )?
        };
        Ok(Self { config, tokenizer })
    }
}
