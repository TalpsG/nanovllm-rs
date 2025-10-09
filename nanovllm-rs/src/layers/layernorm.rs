use candle_core::{
    Result, Tensor,
    cuda::{cudarc::driver::sys::CUDA_EVENT_RECORD_NODE_PARAMS_st, kernels::Module},
};
use candle_nn::{Init, VarBuilder};

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(eps: Option<f32>, vb: VarBuilder) -> Self {
        let eps = eps.unwrap_or(1e-6);
        let weight = vb.get_with_hints((1,), "weight", Init::Const(1.)).unwrap();
        Self { weight, eps }
    }
    fn rms_forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let mut x1 = x.to_dtype(candle_core::DType::F32)?;
        let dims = x1.dims().len();
        let mut var = x1.powf(2.)?.mean_keepdim(dims - 1)?;
        let var_shape = var.shape();
        let eps_tensor = Tensor::full(self.eps, var_shape, x.device())?;
        var = var.broadcast_add(&eps_tensor)?;
        let rsqrt = var.sqrt()?.recip()?;
        x1 = x1.broadcast_mul(&rsqrt)?;
        x1 = x1.to_dtype(orig_dtype)?.broadcast_mul(&self.weight)?;
        Ok(x1)
    }
}

#[cfg(test)]
mod tests {
    use super::RMSNorm;
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{Module, VarBuilder, layer_norm};
    use std::collections::HashMap;

    fn build_input(batch: usize, size: usize, device: &Device) -> Result<Tensor> {
        let total = (batch * size) as f32;
        Tensor::arange(0f32, total, device)?.reshape((batch, size))
    }

    fn build_weight(size: usize, device: &Device) -> Result<Tensor> {
        Tensor::arange(1f32, (size + 1) as f32, device)
    }

    #[test]
    fn rms_norm_matches_candle() -> Result<()> {
        let device = Device::Cpu;
        let batch = 3usize;
        let size = 4usize;
        let eps = 1e-6f32;

        let xs = build_input(batch, size, &device)?;
        let weight = build_weight(size, &device)?;

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), weight.clone());
        let vb_candle = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let candle_rms = layer_norm::rms_norm(size, eps as f64, vb_candle)?;

        let vb_dummy = VarBuilder::zeros(DType::F32, &device);
        let mut custom_rms = RMSNorm::new(Some(eps), vb_dummy);
        custom_rms.weight = weight.clone();

        let expected = candle_rms.forward(&xs)?;
        let actual = custom_rms.rms_forward(&xs)?;

        let expected_data = expected.to_vec2::<f32>()?;
        let actual_data = actual.to_vec2::<f32>()?;

        for (row_expected, row_actual) in expected_data.iter().zip(actual_data.iter()) {
            for (e, a) in row_expected.iter().zip(row_actual.iter()) {
                assert!((e - a).abs() < 1e-5, "values differ: expected {e}, got {a}");
            }
        }

        Ok(())
    }
}
