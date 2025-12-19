use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use nanovllm_rs::engine::llm_engine::LlmEngine;
use nanovllm_rs::utils::sampling_params::SamplingParams;
use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let log_file = OpenOptions::new().create(true).write(true).truncate(true).open("/home/talps/repo/nanovllm-rs/rs.log").unwrap();
    let (non_blocking, _guard) = tracing_appender::non_blocking(log_file);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    println!("no env var for RUST_LOG found, defaulting to off");
                    EnvFilter::new("off")
            }),
        ).with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_writer(non_blocking)
        .with_target(false)
        .try_init();

    let home = std::env::var("HOME")?;
    let model_path = PathBuf::from(home).join("huggingface/Qwen3-0.6B/");

    let mut engine = LlmEngine::new(&model_path)?;
    let max_tokens = std::env::var("MAX_TOKENS")
        .unwrap_or_else(|_| "16".to_string())
        .parse::<usize>()
        .unwrap_or(16);
    let sampling = SamplingParams::new(0.6, max_tokens, false)?;
    let sampling_params = vec![sampling.clone(), sampling.clone()];

    let prompts = vec![
        "introduce yourself".to_string(),
        "list all prime numbers within 100".to_string(),
    ];

    let outputs = engine.generate(&prompts, &sampling_params)?;
    engine.profile_decode();
    engine.profile_prefill();

    info!(num_prompts = prompts.len(), "generation completed");

    for (prompt, output) in prompts.iter().zip(outputs.iter()) {
        println!("\nPrompt: {prompt:?}");
        println!("Completion: {:?}", output.text);
    }
    Ok(())
}
