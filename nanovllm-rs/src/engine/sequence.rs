use std::ops::Index;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, ensure};
use tracing::{debug, instrument};

use crate::utils::sampling_params::SamplingParams;

static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Represents the lifecycle state of a sequence within the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

/// Tracks the state of a single inference request, mirroring the Python implementation.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub seq_id: u64,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub last_token: u32,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub block_table: Vec<usize>,
    pub sampling_params: SamplingParams,
}

impl Sequence {
    pub const BLOCK_SIZE: usize = 256;

    /// Construct a new sequence from prompt tokens and sampling parameters.
    #[instrument(skip(token_ids, sampling_params))]
    pub fn new(token_ids: Vec<u32>, sampling_params: SamplingParams) -> Result<Self> {
        ensure!(
            !token_ids.is_empty(),
            "sequence requires at least one prompt token"
        );

        let seq_id = SEQUENCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let last_token = *token_ids
            .last()
            .expect("token_ids is not empty due to earlier guard");
        let num_tokens = token_ids.len();

        Ok(Self {
            seq_id,
            status: SequenceStatus::Waiting,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            sampling_params,
        })
    }

    /// Unique identifier of the sequence.
    pub fn seq_id(&self) -> u64 {
        self.seq_id
    }

    pub fn status(&self) -> SequenceStatus {
        self.status
    }

    pub fn set_status(&mut self, status: SequenceStatus) {
        self.status = status;
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn last_token(&self) -> u32 {
        self.last_token
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn num_prompt_tokens(&self) -> usize {
        self.num_prompt_tokens
    }

    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    pub fn num_cached_tokens(&self) -> usize {
        self.num_cached_tokens
    }

    pub fn set_num_cached_tokens(&mut self, num_cached_tokens: usize) {
        self.num_cached_tokens = num_cached_tokens;
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / Self::BLOCK_SIZE
    }

    pub fn num_blocks(&self) -> usize {
        (self.num_tokens + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE
    }

    pub fn last_block_num_tokens(&self) -> usize {
        if self.num_blocks() == 0 {
            0
        } else {
            self.num_tokens - (self.num_blocks() - 1) * Self::BLOCK_SIZE
        }
    }

    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.token_ids[..self.num_prompt_tokens]
    }

    pub fn completion_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }

    pub fn block_table_mut(&mut self) -> &mut Vec<usize> {
        &mut self.block_table
    }

    pub fn temperature(&self) -> f32 {
        self.sampling_params.temperature
    }

    pub fn max_tokens(&self) -> usize {
        self.sampling_params.max_tokens
    }

    pub fn ignore_eos(&self) -> bool {
        self.sampling_params.ignore_eos
    }

    pub fn sampling_params(&self) -> &SamplingParams {
        &self.sampling_params
    }

    #[instrument(skip(self))]
    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
        debug!(
            seq_id = self.seq_id,
            num_tokens = self.num_tokens,
            "appended token"
        );
    }

    pub fn block(&self, index: usize) -> &[u32] {
        assert!(index < self.num_blocks(), "block index out of bounds");
        let start = index * Self::BLOCK_SIZE;
        let end = (start + Self::BLOCK_SIZE).min(self.num_tokens);
        &self.token_ids[start..end]
    }
    pub fn len(&self) -> usize {
        self.num_tokens
    }
}

impl Index<usize> for Sequence {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.token_ids[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_initializes_from_prompt() {
        let sampling = SamplingParams::default();
        let seq = Sequence::new(vec![1, 2, 3], sampling.clone()).expect("sequence constructed");
        let seq2 = Sequence::new(vec![4, 5, 6], sampling).expect("second sequence constructed");

        assert_eq!(seq2.seq_id(), seq.seq_id() + 1);
        assert_eq!(seq.status(), SequenceStatus::Waiting);
        assert_eq!(seq.num_tokens(), 3);
        assert_eq!(seq.num_prompt_tokens(), 3);
        assert_eq!(seq.num_completion_tokens(), 0);
        assert_eq!(seq.last_token(), 3);
        assert_eq!(seq.num_cached_blocks(), 0);
        assert_eq!(seq.temperature(), 1.0);
        assert_eq!(seq.max_tokens(), 64);
        assert!(!seq.ignore_eos());
    }

    #[test]
    fn append_token_updates_counts() {
        let sampling = SamplingParams::default();
        let mut seq = Sequence::new(vec![1, 2], sampling).expect("sequence constructed");
        seq.append_token(5);
        seq.append_token(6);

        assert_eq!(seq.num_tokens(), 4);
        assert_eq!(seq.num_completion_tokens(), 2);
        assert_eq!(seq.last_token(), 6);
        assert_eq!(seq.completion_token_ids(), &[5, 6]);
    }

    #[test]
    fn block_slicing_respects_block_size() {
        let sampling = SamplingParams::default();
        let tokens: Vec<u32> = (0..300).collect();
        let mut seq = Sequence::new(tokens, sampling).expect("sequence constructed");
        seq.set_num_cached_tokens(300);

        assert_eq!(seq.num_blocks(), 2);
        assert_eq!(seq.num_cached_blocks(), 1);
        assert_eq!(seq.block(0).len(), Sequence::BLOCK_SIZE);
        assert_eq!(seq.block(1).len(), 44);
        assert_eq!(seq.last_block_num_tokens(), 44);
    }

    #[test]
    fn sequence_rejects_empty_prompt() {
        let sampling = SamplingParams::default();
        let err = Sequence::new(Vec::new(), sampling);
        assert!(err.is_err());
    }
}
