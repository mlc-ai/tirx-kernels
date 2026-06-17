//! The RuntimeValues aggregate — port of `values/runtime.py`. Holds every
//! per-space value container; `state.values.*` reaches into this.

use super::cooperative::CooperativeValues;
use super::mbars::MbarValues;
use super::registers::RegisterValues;
use super::scalars::ScalarValues;
use super::smem::SmemValues;
use super::tensors::TensorValues;
use super::tmem::TmemValues;

#[derive(Clone, Debug, Default)]
pub struct RuntimeValues {
    pub scalars: ScalarValues,
    pub tensors: TensorValues,
    pub smem: SmemValues,
    pub registers: RegisterValues,
    pub tmem: TmemValues,
    pub mbars: MbarValues,
    pub cooperative: CooperativeValues,
}
