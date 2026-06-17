//! GMEM/SMEM tensor values — port of `interpreter/values/tensors.py`.
//!
//! Data is a flat row-major typed array: floats use an f32 backing tagged by
//! declared dtype (F16/BF16 are rounded on write), while integers/bool use native
//! container storage. A parallel `valid` mask fails reads of unwritten elements
//! closed.

use super::super::diagnostics::{IResult, InterpreterError};
use super::arrays::ValueArray1;
use super::indexing::numel;
use crate::ir::{DType, MemorySpace, Tensor};
use ndarray::Array1;
use std::sync::Arc;

/// Who owns a tensor instance.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TensorOwner {
    /// One global instance shared across the run (GMEM).
    Global,
    /// A CTA-local instance (SMEM).
    Cta(usize),
}

/// Identity of a tensor instance: the IR tensor (by id, via Arc) + its owner.
#[derive(Clone, Debug)]
pub struct TensorInstanceKey {
    pub tensor: Arc<Tensor>,
    pub owner: TensorOwner,
}

impl PartialEq for TensorInstanceKey {
    fn eq(&self, other: &Self) -> bool {
        // Tensor eq is by id (identity); owner by value.
        self.tensor.id == other.tensor.id && self.owner == other.owner
    }
}
impl Eq for TensorInstanceKey {}
impl std::hash::Hash for TensorInstanceKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.id.hash(state);
        self.owner.hash(state);
    }
}

/// A dense, numpy-equivalent tensor value (flat row-major native data + valid mask).
/// `all_valid` short-circuits the per-element written-mask: a fully-seeded input
/// sets it and skips allocating/checking a multi-million-element bool array.
#[derive(Clone, Debug)]
pub struct DenseTensorValue {
    pub data: ValueArray1,
    pub valid: Array1<bool>,
    pub all_valid: bool,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl DenseTensorValue {
    pub fn empty(shape: Vec<usize>, dtype: DType) -> Self {
        let n = numel(&shape);
        DenseTensorValue {
            data: ValueArray1::zeros(dtype, n),
            valid: Array1::from_elem(n, false),
            all_valid: false,
            shape,
            dtype,
        }
    }

    /// A fully-written tensor (a GMEM input) — no per-element valid mask allocated.
    pub fn from_native(data: ValueArray1, shape: Vec<usize>, dtype: DType) -> IResult<Self> {
        if data.dtype() != dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "native tensor data dtype mismatch",
            ));
        }
        if data.len() != numel(&shape) {
            return Err(InterpreterError::new(
                "tensor_value",
                "native tensor data length mismatch",
            ));
        }
        Ok(DenseTensorValue {
            data,
            valid: Array1::from_elem(0, true),
            all_valid: true,
            shape,
            dtype,
        })
    }

    fn strides(&self) -> Vec<usize> {
        row_major_strides(&self.shape)
    }

    fn check_bounds(&self, offsets: &[usize], slice_shape: &[usize]) -> IResult<()> {
        if offsets.len() != self.shape.len() || slice_shape.len() != self.shape.len() {
            return Err(InterpreterError::new(
                "tensor_value",
                "tensor slice rank mismatch",
            ));
        }
        for ((o, e), d) in offsets
            .iter()
            .zip(slice_shape.iter())
            .zip(self.shape.iter())
        {
            if o + e > *d {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "tensor slice is out of bounds",
                ));
            }
        }
        Ok(())
    }

    /// Rectangular slice read → flat row-major native values. The innermost dim is contiguous
    /// (row-major), so each row is one `copy_from_slice` (memcpy) with a bulk valid
    /// check — not a per-element gather.
    pub fn read_block(&self, offsets: &[usize], slice_shape: &[usize]) -> IResult<ValueArray1> {
        self.check_bounds(offsets, slice_shape)?;
        let k = numel(slice_shape);
        if k == 0 {
            return Ok(ValueArray1::zeros(self.dtype, 0));
        }
        let strides = self.strides();
        let base = flat_with_strides(offsets, &strides);
        let rank = slice_shape.len();
        let inner = slice_shape[rank - 1];
        let run_starts = rect_intra_offsets(&slice_shape[..rank - 1], &strides[..rank - 1]);
        let mut out = ValueArray1::zeros(self.dtype, k);
        let mut oi = 0;
        if self.all_valid {
            for &rs in &run_starts {
                let s = base + rs;
                out.copy_run_from(oi, &self.data, s, inner)?;
                oi += inner;
            }
        } else {
            let valid = self.valid.as_slice().expect("contiguous valid");
            for &rs in &run_starts {
                let s = base + rs;
                if valid[s..s + inner].contains(&false) {
                    return Err(InterpreterError::new(
                        "missing_tensor_value",
                        "tensor slice reads unwritten elements",
                    ));
                }
                out.copy_run_from(oi, &self.data, s, inner)?;
                oi += inner;
            }
        }
        Ok(out)
    }

    /// In-place rectangular slice write (row-major). Each row is one `copy_from_slice`
    /// + bulk valid fill — not a per-element scatter.
    pub fn write_slice_inplace(
        &mut self,
        offsets: &[usize],
        slice_shape: &[usize],
        values: &ValueArray1,
    ) -> IResult<()> {
        self.check_bounds(offsets, slice_shape)?;
        let k = numel(slice_shape);
        if values.len() != k {
            return Err(InterpreterError::new(
                "tensor_value",
                "slice write value count mismatch",
            ));
        }
        if values.dtype() != self.dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "slice write dtype must match the destination container",
            ));
        }
        if k == 0 {
            return Ok(());
        }
        let strides = self.strides();
        let base = flat_with_strides(offsets, &strides);
        let rank = slice_shape.len();
        let inner = slice_shape[rank - 1];
        let run_starts = rect_intra_offsets(&slice_shape[..rank - 1], &strides[..rank - 1]);
        let all_valid = self.all_valid;
        let mut vi = 0;
        for &rs in &run_starts {
            let s = base + rs;
            self.data.copy_run_from(s, values, vi, inner)?;
            vi += inner;
        }
        if !all_valid {
            let valid = self.valid.as_slice_mut().expect("contiguous valid");
            for &rs in &run_starts {
                let s = base + rs;
                valid[s..s + inner].fill(true);
            }
        }
        Ok(())
    }

    pub fn invalidate_indices(&mut self, indices: &[usize]) -> IResult<()> {
        let n = numel(&self.shape);
        if indices.iter().any(|&idx| idx >= n) {
            return Err(InterpreterError::new(
                "tensor_value",
                "tensor index is out of bounds",
            ));
        }
        if self.all_valid {
            self.valid = Array1::from_elem(n, true);
            self.all_valid = false;
        }
        let valid = self.valid.as_slice_mut().expect("contiguous valid");
        for &idx in indices {
            valid[idx] = false;
        }
        Ok(())
    }

    pub fn invalidate_slice_inplace(
        &mut self,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<()> {
        self.check_bounds(offsets, slice_shape)?;
        let k = numel(slice_shape);
        if k == 0 {
            return Ok(());
        }
        let strides = self.strides();
        let base = flat_with_strides(offsets, &strides);
        let rank = slice_shape.len();
        let inner = slice_shape[rank - 1];
        let run_starts = rect_intra_offsets(&slice_shape[..rank - 1], &strides[..rank - 1]);
        if self.all_valid {
            self.valid = Array1::from_elem(numel(&self.shape), true);
            self.all_valid = false;
        }
        let valid = self.valid.as_slice_mut().expect("contiguous valid");
        for &rs in &run_starts {
            let s = base + rs;
            valid[s..s + inner].fill(false);
        }
        Ok(())
    }

    pub fn copy_slice_from(
        &mut self,
        dst_offsets: &[usize],
        src: &DenseTensorValue,
        src_offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<()> {
        self.check_bounds(dst_offsets, slice_shape)?;
        src.check_bounds(src_offsets, slice_shape)?;
        if self.dtype != src.dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "dense copy dtype must match the destination container",
            ));
        }
        let k = numel(slice_shape);
        if k == 0 {
            return Ok(());
        }
        let rank = slice_shape.len();
        let inner = slice_shape[rank - 1];
        let dst_strides = self.strides();
        let src_strides = src.strides();
        let dst_base = flat_with_strides(dst_offsets, &dst_strides);
        let src_base = flat_with_strides(src_offsets, &src_strides);
        let dst_runs = rect_intra_offsets(&slice_shape[..rank - 1], &dst_strides[..rank - 1]);
        let src_runs = rect_intra_offsets(&slice_shape[..rank - 1], &src_strides[..rank - 1]);
        if !src.all_valid {
            let valid = src.valid.as_slice().expect("contiguous valid");
            for &rs in &src_runs {
                let s = src_base + rs;
                if valid[s..s + inner].contains(&false) {
                    return Err(InterpreterError::new(
                        "missing_tensor_value",
                        "tensor slice reads unwritten elements",
                    ));
                }
            }
        }
        for (&dst_rs, &src_rs) in dst_runs.iter().zip(src_runs.iter()) {
            self.data
                .copy_run_from(dst_base + dst_rs, &src.data, src_base + src_rs, inner)?;
        }
        if !self.all_valid {
            let valid = self.valid.as_slice_mut().expect("contiguous valid");
            for &rs in &dst_runs {
                let s = dst_base + rs;
                valid[s..s + inner].fill(true);
            }
        }
        Ok(())
    }

    pub fn f32_compute_slice(&self) -> Option<&[f32]> {
        match &self.data {
            ValueArray1::F16(a) | ValueArray1::Bf16(a) | ValueArray1::F32(a) => a.as_slice(),
            _ => None,
        }
    }
}

/// Row-major flat offsets of every element in a rectangle (relative to its base),
/// computed by stride arithmetic — no per-element coordinate vectors.
pub(crate) fn rect_intra_offsets(slice_shape: &[usize], strides: &[usize]) -> Vec<usize> {
    if slice_shape.iter().any(|&e| e == 0) {
        return Vec::new();
    }
    let mut offs = vec![0usize];
    for (d, &extent) in slice_shape.iter().enumerate() {
        let stride = strides[d];
        let mut next = Vec::with_capacity(offs.len() * extent);
        for &base in &offs {
            for step in 0..extent {
                next.push(base + step * stride);
            }
        }
        offs = next;
    }
    offs
}

pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for dim in (0..shape.len()).rev() {
        strides[dim] = acc;
        acc *= shape[dim];
    }
    strides
}

pub(crate) fn flat_with_strides(coord: &[usize], strides: &[usize]) -> usize {
    coord.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}

/// The instance key for a tensor accessed from a thread (GMEM→Global, SMEM→Cta).
pub fn tensor_instance_key(cta_id: usize, tensor: &Arc<Tensor>) -> IResult<TensorInstanceKey> {
    let owner = match tensor.space {
        MemorySpace::Gmem => TensorOwner::Global,
        MemorySpace::Smem => TensorOwner::Cta(cta_id),
        _ => {
            return Err(InterpreterError::new(
                "unsupported_tensor_space",
                "tensor instance is only modeled for GMEM/SMEM",
            ))
        }
    };
    Ok(TensorInstanceKey {
        tensor: Arc::clone(tensor),
        owner,
    })
}

/// Coerce a runtime GMEM input (an f32 ndarray) to a flat container array,
/// rounding/wrapping to the tensor's dtype.
/// All GMEM + SMEM tensor instances.
#[derive(Clone, Debug, Default)]
pub struct TensorValues {
    pub by_instance: std::collections::HashMap<TensorInstanceKey, DenseTensorValue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use ndarray::arr1;

    #[test]
    fn dense_read_write_preserves_native_dtype_and_validity() {
        let mut dense = DenseTensorValue::empty(vec![4], DType::U32);
        let missing = dense.read_block(&[0], &[1]).unwrap_err();
        assert_eq!(missing.code, "missing_tensor_value");

        let values = ValueArray1::from_i64_compute(arr1(&[-1_i64, 7]), DType::U32);
        dense.write_slice_inplace(&[1], &[2], &values).unwrap();

        let read = dense.read_block(&[1], &[2]).unwrap();
        assert_eq!(read.dtype(), DType::U32);
        assert_eq!(
            read.to_i64_compute().as_slice().unwrap(),
            &[0xFFFF_FFFF_i64, 7]
        );

        let wrong = ValueArray1::from_i64_compute(arr1(&[1]), DType::I32);
        let err = dense.write_slice_inplace(&[0], &[1], &wrong).unwrap_err();
        assert_eq!(err.code, "tensor_value");
    }
}
