use anyhow::{Result, ensure};

/// Configuration controlling token sampling during generation.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Temperature used to scale logits before sampling.
    pub temperature: f32,
    /// Maximum number of tokens to generate for the current request.
    pub max_tokens: usize,
    /// When set, sampling continues even if the model emits EOS.
    pub ignore_eos: bool,
}

impl SamplingParams {
    pub const MIN_TEMPERATURE: f32 = 1e-10;

    /// Create a new set of sampling parameters, ensuring invariants hold.
    pub fn new(temperature: f32, max_tokens: usize, ignore_eos: bool) -> Result<Self> {
        ensure!(
            temperature > Self::MIN_TEMPERATURE,
            "temperature ({temperature}) must be greater than {}",
            Self::MIN_TEMPERATURE
        );

        Ok(Self {
            temperature,
            max_tokens,
            ignore_eos,
        })
    }

    /// Returns a copy with an updated temperature while preserving invariants.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        ensure!(
            temperature > Self::MIN_TEMPERATURE,
            "temperature ({temperature}) must be greater than {}",
            Self::MIN_TEMPERATURE
        );
        self.temperature = temperature;
        Ok(self)
    }

    /// Returns a copy with updated max_tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Returns a copy with updated ignore_eos flag.
    pub fn with_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.ignore_eos = ignore_eos;
        self
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 64,
            ignore_eos: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_python() {
        let params = SamplingParams::default();
        assert!((params.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(params.max_tokens, 64);
        assert!(!params.ignore_eos);
    }

    #[test]
    fn new_rejects_non_positive_temperature() {
        let result = SamplingParams::new(1e-12, 32, false);
        assert!(result.is_err());
    }

    #[test]
    fn builder_helpers_work() {
        let params = SamplingParams::default()
            .with_max_tokens(128)
            .with_ignore_eos(true)
            .with_temperature(0.5)
            .expect("temperature update accepted");

        assert!((params.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.max_tokens, 128);
        assert!(params.ignore_eos);
    }
}
