use candle_core::{DType, Result, Tensor};
use candle_nn::{Init, VarBuilder};

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(size: usize, eps: Option<f32>, vb: VarBuilder) -> Result<Self> {
        let eps = eps.unwrap_or(1e-6);
        let weight = vb.get_with_hints((size,), "weight", Init::Const(1.))?;
        Ok(Self { weight, eps })
    }

    fn normalize_from_f32(&self, x_f32: &Tensor) -> Result<Tensor> {
        let last_dim = x_f32.dims().len().saturating_sub(1);
        let mut var = x_f32.powf(2.)?.mean_keepdim(last_dim)?;
        let eps_tensor = Tensor::full(self.eps, var.shape(), x_f32.device())?;
        var = var.broadcast_add(&eps_tensor)?;
        let rsqrt = var.sqrt()?.recip()?;
        x_f32.broadcast_mul(&rsqrt)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let normalized_f32 = self.normalize_from_f32(&x_f32)?;
        let normalized = normalized_f32.to_dtype(orig_dtype)?;
        normalized.broadcast_mul(&self.weight)
    }

    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let residual_f32 = residual.to_dtype(DType::F32)?;
        let summed = x_f32.broadcast_add(&residual_f32)?;
        let residual_out = summed.to_dtype(orig_dtype)?;
        let normalized_f32 = self.normalize_from_f32(&summed)?;
        let normalized = normalized_f32.to_dtype(orig_dtype)?;
        let output = normalized.broadcast_mul(&self.weight)?;
        Ok((output, residual_out))
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
        let mut custom_rms = RMSNorm::new(size, Some(eps), vb_dummy)?;
        custom_rms.weight = weight.clone();

        let expected = candle_rms.forward(&xs)?;
        let actual = custom_rms.forward(&xs)?;

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
