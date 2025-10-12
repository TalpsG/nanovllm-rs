use std::path::PathBuf;

use anyhow::Result;
use nanovllm_rs::engine::llm_engine::LlmEngine;
use nanovllm_rs::utils::sampling_params::SamplingParams;
use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("nanovllm_rs=info,example=info")),
        )
        .with_target(false)
        .try_init();

    let home = std::env::var("HOME")?;
    let model_path = PathBuf::from(home).join("huggingface/Qwen3-0.6B/");

    let mut engine = LlmEngine::new(&model_path)?;
    let sampling = SamplingParams::new(0.6, 512, false)?;
    let sampling_params = vec![sampling.clone(), sampling.clone()];

    let prompts = vec![
        "introduce yourself".to_string(),
        "list all prime numbers within 100".to_string(),
    ];

    let outputs = engine.generate(&prompts, &sampling_params)?;

    info!(num_prompts = prompts.len(), "generation completed");

    for (prompt, output) in prompts.iter().zip(outputs.iter()) {
        println!("\nPrompt: {prompt:?}");
        println!("Completion: {:?}", output.text);
    }

    Ok(())
}
