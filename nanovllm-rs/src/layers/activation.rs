use candle_core::{Module, Result, Tensor};
pub struct SiluAndMul;
impl Module for SiluAndMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let pieces = xs.chunk(2, xs.dims().len() - 1)?;
        let x = &pieces[0];
        let y = &pieces[1];
        x.silu()? * y
    }
}
