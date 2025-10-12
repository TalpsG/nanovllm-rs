use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use candle_core::Device;
use lazy_static::lazy_static;
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument, trace};

use crate::engine::model_runner::ModelRunner;
use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::utils::config::Config;
use crate::utils::sampling_params::SamplingParams;

lazy_static! {
    pub static ref DEVICE: Device =
        Device::new_cuda(0).expect("failed to create default CUDA device");
}

/// Convenience wrapper returned during generation.
pub struct Generation {
    pub text: String,
    pub token_ids: Vec<u32>,
}

pub struct LlmEngine {
    _config: Config,
    tokenizer: Tokenizer,
    scheduler: Scheduler,
    model_runner: ModelRunner,
    // TODO: tensor-parallel worker management and CUDA graph integration.
}

impl LlmEngine {
    #[instrument(skip(model_dir))]
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let config = Config::from_model_dir(model_dir.as_ref())?;
        let tokenizer_path = config.model.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("failed to load tokenizer: {err}"))?;

        let model_runner = ModelRunner::new(config.clone(), 0)?;
        let scheduler = Scheduler::new(model_runner.config());

        info!(
            max_num_seqs = config.max_num_seqs,
            max_batched_tokens = config.max_num_batched_tokens,
            num_kv_blocks = config.num_kvcache_blocks,
            block_size = config.kvcache_block_size,
            "LlmEngine initialised"
        );

        Ok(Self {
            _config: model_runner.config().clone(),
            tokenizer,
            scheduler,
            model_runner,
        })
    }

    pub fn exit(&mut self) {
        self.model_runner.exit();
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    #[instrument(skip(self, token_ids, sampling_params))]
    pub fn add_request_from_tokens(
        &mut self,
        token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Result<()> {
        let sequence = Sequence::new(token_ids, sampling_params)?;
        let handle = Rc::new(RefCell::new(sequence));
        self.scheduler.add(handle);
        trace!("request added from tokens");
        Ok(())
    }

    #[instrument(skip(self, prompt, sampling_params))]
    pub fn add_request(&mut self, prompt: &str, sampling_params: SamplingParams) -> Result<()> {
        let prompt = Self::format_prompt(prompt);
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|err| anyhow!("failed to encode prompt: {err}"))?;
        let token_ids = encoding.get_ids().to_vec();
        self.add_request_from_tokens(token_ids, sampling_params)
    }

    #[instrument(skip(self))]
    pub fn step(&mut self) -> Result<(Vec<(u64, Vec<u32>)>, isize)> {
        let (handles, is_prefill) = self.scheduler.schedule()?;
        let batch: Vec<Sequence> = handles.iter().map(|seq| seq.borrow().clone()).collect();
        let token_ids = self.model_runner.run(&batch, is_prefill)?;
        let finished_flags = self.scheduler.postprocess(&handles, &token_ids)?;

        let mut outputs = Vec::new();
        for (seq, finished) in handles.iter().zip(finished_flags.iter()) {
            if *finished {
                let seq_ref = seq.borrow();
                outputs.push((seq_ref.seq_id(), seq_ref.completion_token_ids().to_vec()));
            }
        }

        let num_tokens = if is_prefill {
            batch.iter().map(|seq| seq.len() as isize).sum::<isize>()
        } else {
            -(batch.len() as isize)
        };

        debug!(
            batch_size = batch.len(),
            is_prefill,
            finished = finished_flags.iter().filter(|flag| **flag).count(),
            "step completed"
        );

        Ok((outputs, num_tokens))
    }

    pub fn generate(
        &mut self,
        prompts: &[String],
        sampling_params: &[SamplingParams],
    ) -> Result<Vec<Generation>> {
        if prompts.len() != sampling_params.len() {
            return Err(anyhow!(
                "mismatched prompts ({}) and sampling params ({})",
                prompts.len(),
                sampling_params.len()
            ));
        }

        for (prompt, params) in prompts.iter().zip(sampling_params.iter()) {
            self.add_request(prompt, params.clone())?;
        }

        let mut completed: Vec<(u64, Generation)> = Vec::new();
        while !self.is_finished() {
            let (results, delta_tokens) = self.step()?;
            trace!(
                delta_tokens,
                in_flight = self.scheduler.num_running(),
                waiting = self.scheduler.num_waiting()
            );
            for (seq_id, token_ids) in results {
                let text = self
                    .tokenizer
                    .decode(&token_ids, true)
                    .map_err(|err| anyhow!("failed to decode sequence {seq_id}: {err}"))?;
                completed.push((seq_id, Generation { text, token_ids }));
            }
        }
        completed.sort_by_key(|(seq_id, _)| *seq_id);
        info!(completed = completed.len(), "generation finished");
        Ok(completed
            .into_iter()
            .map(|(_, generation)| generation)
            .collect())
    }

    fn format_prompt(prompt: &str) -> String {
        if prompt.contains("<|im_start|") {
            return prompt.to_string();
        }
        let trimmed = prompt.trim();
        format!("<|im_start|>user\n{trimmed}\n<|im_end|>\n<|im_start|>assistant\n")
    }
}

impl Drop for LlmEngine {
    fn drop(&mut self) {
        self.exit();
    }
}
