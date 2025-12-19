use candle_core::{DType, Device, Result, Shape, Tensor, bail};
use candle_nn::ops::softmax;

pub struct Sampler;

impl Sampler {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, logits: &Tensor, temperatures: &Tensor) -> Result<Tensor> {
        let device = logits.device();
        let probs = Self::compute_probs(logits, temperatures)?;
        let noise = Self::sample_exponential_noise(probs.shape(), device)?;
        Self::sample_from_probs(&probs, &noise)
    }

    fn compute_probs(logits: &Tensor, temperatures: &Tensor) -> Result<Tensor> {
        if logits.dims().is_empty() {
            bail!("logits tensor must have at least one dimension");
        }

        let logits = logits.to_dtype(DType::F32)?;
        let device = logits.device();
        let temps = temperatures
            .to_device(device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let scaled = logits.broadcast_div(&temps)?;
        let last_dim = scaled.dims().len() - 1;
        softmax(&scaled, last_dim)
    }

    fn sample_exponential_noise(shape: &Shape, device: &Device) -> Result<Tensor> {
        Tensor::rand(1e-6f32, 1f32, shape.clone(), device)?
            .log()?
            .neg()?
            .clamp(1e-10f32, f32::MAX)
    }

    fn sample_from_probs(probs: &Tensor, noise: &Tensor) -> Result<Tensor> {
        if probs.dims().is_empty() {
            bail!("probability tensor must have at least one dimension");
        }
        let adjusted = probs.broadcast_div(noise)?;
        let last_dim = adjusted.dims().len() - 1;
        adjusted.argmax(last_dim)?.to_dtype(DType::I64)
    }
}

#[cfg(test)]
mod tests {
    use super::Sampler;
    use crate::engine::llm_engine::DEVICE;
    use candle_core::{DType, Result, Tensor};
    use candle_nn::ops::softmax;

    #[test]
    fn compute_probs_matches_manual_softmax() -> Result<()> {
        let device = &*DEVICE;
        let logits = Tensor::from_slice(&[1f32, 3f32, 2f32, 2f32, 1f32, 0f32], (2, 3), device)?;
        let temperatures = Tensor::from_slice(&[1f32, 0.5f32], (2,), device)?;

        let probs = Sampler::compute_probs(&logits, &temperatures)?;
        let temps = temperatures.unsqueeze(1)?;
        let scaled = logits.broadcast_div(&temps)?;
        let expected = softmax(&scaled, 1)?;

        let probs_vec = probs.to_vec2::<f32>()?;
        let expected_vec = expected.to_vec2::<f32>()?;
        assert_eq!(probs_vec.len(), expected_vec.len());
        for (row_p, row_e) in probs_vec.iter().zip(expected_vec.iter()) {
            assert_eq!(row_p.len(), row_e.len());
            for (&p, &e) in row_p.iter().zip(row_e.iter()) {
                assert!((p - e).abs() < 1e-6, "mismatch {p} vs {e}");
            }
        }

        Ok(())
    }

    #[test]
    fn sample_from_probs_with_unit_noise_is_argmax() -> Result<()> {
        let device = &*DEVICE;
        let logits = Tensor::from_slice(&[1f32, 3f32, 2f32, 2f32, 1f32, 0f32], (2, 3), device)?;
        let temperatures = Tensor::from_slice(&[1f32, 0.5f32], (2,), device)?;

        let probs = Sampler::compute_probs(&logits, &temperatures)?;
        let noise = Tensor::full(1f32, probs.shape().clone(), device)?;
        let samples = Sampler::sample_from_probs(&probs, &noise)?;
        let expected = probs.argmax(1)?.to_dtype(DType::I64)?;

        let samples_vec = samples.to_vec1::<i64>()?;
        let expected_vec = expected.to_vec1::<i64>()?;
        assert_eq!(samples_vec, expected_vec);

        Ok(())
    }

    #[test]
    fn forward_produces_expected_shape() -> Result<()> {
        let device = &*DEVICE;
        let sampler = Sampler::new();
        let logits = Tensor::from_slice(&[1f32, 3f32, 2f32, 2f32, 1f32, 0f32], (2, 3), device)?;
        let temperatures = Tensor::from_slice(&[1f32, 0.5f32], (2,), device)?;

        let tokens = sampler.forward(&logits, &temperatures)?;
        assert_eq!(tokens.dims(), &[2]);
        assert_eq!(tokens.dtype(), DType::I64);

        Ok(())
    }
}
