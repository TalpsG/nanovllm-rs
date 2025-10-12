use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use tracing::{debug, instrument, trace};

use crate::engine::block_manager::BlockManager;
use crate::engine::sequence::{Sequence, SequenceStatus};
use crate::utils::config::Config;

/// Shared sequence handle used throughout scheduling.
pub type SharedSequence = Rc<RefCell<Sequence>>;

/// Coordinates which sequences enter prefill vs. decode phases.
///
/// TODO: incorporate tensor-parallel specific sharding behaviour once multi-rank execution
/// is implemented.
/// TODO: hook CUDA graph specific batching heuristics when graph capture is available.
pub struct Scheduler {
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    eos: isize,
    block_manager: BlockManager,
    waiting: VecDeque<SharedSequence>,
    running: VecDeque<SharedSequence>,
}

impl Scheduler {
    #[instrument(skip(config))]
    pub fn new(config: &Config) -> Self {
        let block_manager = BlockManager::new(
            config.num_kvcache_blocks.max(0) as usize,
            config.kvcache_block_size,
        );
        Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos: config.eos,
            block_manager,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    #[instrument(skip(self, seq))]
    pub fn add(&mut self, seq: SharedSequence) {
        self.waiting.push_back(seq);
        trace!(
            waiting = self.waiting.len(),
            running = self.running.len(),
            "added sequence to waiting queue"
        );
    }

    #[instrument(skip(self))]
    pub fn schedule(&mut self) -> Result<(Vec<SharedSequence>, bool)> {
        let mut scheduled = Vec::new();
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;

        while num_seqs < self.max_num_seqs {
            let Some(seq_rc) = self.waiting.front() else {
                break;
            };
            let (seq_len, can_allocate) = {
                let seq_ref = seq_rc.borrow();
                let can_allocate = self.block_manager.can_allocate(&seq_ref);
                (seq_ref.len(), can_allocate)
            };
            if num_batched_tokens + seq_len > self.max_num_batched_tokens || !can_allocate {
                break;
            }

            let seq = self
                .waiting
                .pop_front()
                .expect("sequence peeked from waiting queue must exist");
            {
                let mut seq_mut = seq.borrow_mut();
                self.block_manager.allocate(&mut seq_mut);
                let new_tokens = seq_mut.len() - seq_mut.num_cached_tokens();
                num_batched_tokens += new_tokens;
                seq_mut.set_status(SequenceStatus::Running);
            }
            self.running.push_back(Rc::clone(&seq));
            scheduled.push(seq);
            num_seqs += 1;
        }

        if !scheduled.is_empty() {
            debug!(
                batch_size = scheduled.len(),
                num_batched_tokens, "prefill scheduling complete"
            );
            return Ok((scheduled, true));
        }

        while num_seqs < self.max_num_seqs {
            let Some(seq) = self.running.pop_front() else {
                break;
            };
            let mut preempted = false;
            trace!("considering sequence for decode batch");
            loop {
                let can_append = {
                    let seq_ref = seq.borrow();
                    self.block_manager.can_append(&seq_ref)
                };
                if can_append {
                    break;
                }
                if let Some(other) = self.running.pop_back() {
                    self.preempt(other);
                } else {
                    self.preempt(Rc::clone(&seq));
                    preempted = true;
                    break;
                }
            }
            if preempted {
                trace!("sequence preempted due to insufficient blocks");
                continue;
            }

            {
                let mut seq_mut = seq.borrow_mut();
                self.block_manager.may_append(&mut seq_mut);
            }
            scheduled.push(seq);
            num_seqs += 1;
        }

        ensure!(
            !scheduled.is_empty(),
            "scheduler produced an empty decode batch"
        );
        for seq in scheduled.iter().rev() {
            self.running.push_front(Rc::clone(seq));
        }
        debug!(batch_size = scheduled.len(), "decode scheduling complete");
        Ok((scheduled, false))
    }

    #[instrument(skip(self, seq))]
    fn preempt(&mut self, seq: SharedSequence) {
        {
            let mut seq_mut = seq.borrow_mut();
            seq_mut.set_status(SequenceStatus::Waiting);
            self.block_manager.deallocate(&mut seq_mut);
        }
        self.waiting.push_front(seq);
        trace!(
            waiting = self.waiting.len(),
            running = self.running.len(),
            "sequence preempted and returned to waiting queue"
        );
    }

    #[instrument(skip(self, seqs, token_ids), fields(batch_size = seqs.len()))]
    pub fn postprocess(&mut self, seqs: &[SharedSequence], token_ids: &[i64]) -> Result<Vec<bool>> {
        ensure!(
            seqs.len() == token_ids.len(),
            "token count {} does not match sequence count {}",
            token_ids.len(),
            seqs.len()
        );

        let mut finished_flags = Vec::with_capacity(seqs.len());
        let mut finished_ptrs = Vec::new();

        for (seq_rc, &token_id) in seqs.iter().zip(token_ids.iter()) {
            let token_u32 = u32::try_from(token_id)
                .with_context(|| format!("token id {token_id} is out of u32 range"))?;
            let finished = {
                let mut seq = seq_rc.borrow_mut();
                seq.append_token(token_u32);
                let reached_eos = !seq.ignore_eos() && token_id == self.eos as i64;
                let reached_limit = seq.num_completion_tokens() >= seq.max_tokens();
                let is_finished = reached_eos || reached_limit;
                if is_finished {
                    seq.set_status(SequenceStatus::Finished);
                    self.block_manager.deallocate(&mut seq);
                }
                trace!(
                    seq_id = seq.seq_id(),
                    finished = is_finished,
                    append_token = token_u32
                );
                is_finished
            };
            if finished {
                finished_ptrs.push(Rc::as_ptr(seq_rc));
            }
            finished_flags.push(finished);
        }

        if !finished_ptrs.is_empty() {
            self.running.retain(|seq| {
                let ptr = Rc::as_ptr(seq);
                !finished_ptrs.iter().any(|&finished| finished == ptr)
            });
        }
        debug!(finished = finished_ptrs.len(), "postprocess complete");

        Ok(finished_flags)
    }
}
