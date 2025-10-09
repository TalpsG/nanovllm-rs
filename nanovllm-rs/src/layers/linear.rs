// TODO:impl linear support for tp

use candle_core::Result;
use candle_nn::{Linear, Module, linear_b};

pub struct LinearBase {
    pub in_size: usize,
    pub out_size: usize,
    pub bias: bool,
    pub linear: Linear,
}
impl LinearBase {
    // use varbuilder to create linear layer
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        // TODO:TP
        Ok(LinearBase {
            linear: linear_b(in_dim, out_dim, bias, vb)?,
            in_size: in_dim,
            out_size: out_dim,
            bias,
        })
    }
}
impl Module for LinearBase {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        self.linear.forward(xs)
    }
}

pub struct ReplicateLinear {
    pub linear: LinearBase,
}

impl ReplicateLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear: LinearBase::new(in_dim, out_dim, bias, vb)?,
        })
    }
}
impl Module for ReplicateLinear {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        self.linear.forward(xs)
    }
}

pub struct ColumnParallelLinear {
    pub linear: LinearBase,
}
impl ColumnParallelLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear: LinearBase::new(in_dim, out_dim, bias, vb)?,
        })
    }
}
impl Module for ColumnParallelLinear {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        self.linear.forward(xs)
    }
}
pub struct MergedColumnParallelLinear {
    pub linear: ColumnParallelLinear,
}
impl MergedColumnParallelLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear: ColumnParallelLinear::new(in_dim, out_dim, bias, vb)?,
        })
    }
}
impl Module for MergedColumnParallelLinear {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        self.linear.forward(xs)
    }
}

pub struct QKVParallelLinear {
    pub linear: ColumnParallelLinear,
}
impl QKVParallelLinear {
    pub fn new(
        hidden_size: usize,
        head_size: usize,
        total_num_heads: usize,
        total_num_kv_heads: usize,
        bias: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let out_size = (total_num_heads + 2 * total_num_kv_heads) * head_size;
        Ok(Self {
            linear: ColumnParallelLinear::new(hidden_size, out_size, bias, vb)?,
        })
    }
}
impl Module for QKVParallelLinear {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        self.linear.forward(xs)
    }
}

pub struct RowParallelLinear {
    pub linear: LinearBase,
}
impl RowParallelLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear: LinearBase::new(in_dim, out_dim, bias, vb)?,
        })
    }
}
impl Module for RowParallelLinear {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        self.linear.forward(xs)
    }
}
