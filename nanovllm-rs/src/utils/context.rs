use std::sync::{Arc, RwLock};

use candle_core::Tensor;
use lazy_static::lazy_static;

/// Runtime context shared across embedding and decoding routines.
///
/// Mirrors the Python `nanovllm.utils.context.Context` dataclass so components
/// can query scheduling metadata (prefill/decode, cu_seqlens, etc.).
#[derive(Clone, Debug)]
pub struct Context {
    pub is_prefill: bool,
    pub cu_seqlens_q: Tensor,
    pub cu_seqlens_k: Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
}

impl Context {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        is_prefill: bool,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        slot_mapping: Tensor,
        context_lens: Tensor,
        block_tables: Tensor,
    ) -> Self {
        Self {
            is_prefill,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
        }
    }
}

lazy_static! {
    static ref CONTEXT: Arc<RwLock<Option<Context>>> = Arc::new(RwLock::new(None));
}

pub fn context_handle() -> Arc<RwLock<Option<Context>>> {
    Arc::clone(&CONTEXT)
}

pub fn get_context() -> Arc<RwLock<Option<Context>>> {
    context_handle()
}

pub fn set_context(context: Context) {
    let handle = context_handle();
    let mut guard = handle.write().expect("context lock poisoned");
    *guard = Some(context);
}

#[allow(clippy::too_many_arguments)]
pub fn set_context_with(
    is_prefill: bool,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    slot_mapping: Tensor,
    context_lens: Tensor,
    block_tables: Tensor,
) {
    set_context(Context::new(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    ));
}

pub fn update_context<F>(update: F)
where
    F: FnOnce(&mut Context),
{
    let handle = context_handle();
    let mut guard = handle.write().expect("context lock poisoned");
    let ctx = guard
        .as_mut()
        .expect("attempted to update context before initialization");
    update(ctx);
}

pub fn reset_context() {
    let handle = context_handle();
    let mut guard = handle.write().expect("context lock poisoned");
    *guard = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn context_roundtrip() {
        reset_context();

        let initial = get_context();
        {
            let guard = initial.read().expect("context lock poisoned");
            assert!(guard.is_none());
        }

        let device = Device::Cpu;
        let empty = Tensor::zeros((0,), DType::U32, &device).unwrap();
        set_context(Context::new(
            true,
            empty.clone(),
            empty.clone(),
            128,
            256,
            empty.clone(),
            empty.clone(),
            empty.clone(),
        ));
        {
            let retrieved = get_context();
            let guard = retrieved.read().expect("context lock poisoned");
            let ctx = guard.as_ref().expect("context not set");
            assert!(ctx.is_prefill);
            assert_eq!(ctx.max_seqlen_q, 128);
            assert_eq!(ctx.max_seqlen_k, 256);
        }

        update_context(|ctx| ctx.is_prefill = false);
        {
            let updated = get_context();
            let guard = updated.read().expect("context lock poisoned");
            let ctx = guard.as_ref().expect("context not set");
            assert!(!ctx.is_prefill);
        }

        reset_context();
        {
            let reset = get_context();
            let guard = reset.read().expect("context lock poisoned");
            assert!(guard.is_none());
        }
    }
}
