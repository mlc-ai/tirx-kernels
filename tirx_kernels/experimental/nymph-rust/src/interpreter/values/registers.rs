//! Per-thread register file — port of `interpreter/values/registers.py`.
//! A reg tensor with a leading thread axis: data is `[n_cta_threads, numel(shape)]`
//! (row r = CTA-local thread warp_id*32 + lane_id). Data is stored according to
//! the typed value policy: floats in rounded f32 backing, integers/bool native.

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::threads::ThreadId;
use super::arrays::{ValueArray1, ValueArray2};
use super::dtypes::round_scalar;
use super::indexing::numel;
use crate::ir::{DType, Tensor};
use ndarray::{s, Array2};
use std::collections::HashMap;
use std::sync::Arc;

pub fn register_row(thread: &ThreadId) -> usize {
    thread.warp_id * 32 + thread.lane_id
}

/// One CTA's register-file values for one reg tensor.
#[derive(Clone, Debug)]
pub struct RegisterTensorValue {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: ValueArray2, // [n_cta_threads, numel(shape)]
    pub valid: Array2<bool>,
}

impl RegisterTensorValue {
    pub fn empty(shape: Vec<usize>, dtype: DType, n_cta_threads: usize) -> Self {
        let k = numel(&shape);
        RegisterTensorValue {
            shape,
            dtype,
            data: ValueArray2::zeros(dtype, (n_cta_threads, k)),
            valid: Array2::from_elem((n_cta_threads, k), false),
        }
    }

    /// Gather `[A,k]`: out[a,j] = data[rows[a], cols[a,j]]. Fails closed on unwritten.
    pub fn gather_rows(&self, rows: &[usize], cols: &Array2<usize>) -> IResult<ValueArray2> {
        let a = rows.len();
        let k = cols.ncols();
        for ai in 0..a {
            let r = rows[ai];
            for j in 0..k {
                let c = cols[[ai, j]];
                if !self.valid[[r, c]] {
                    return Err(InterpreterError::new(
                        "missing_tensor_value",
                        "register slice reads unwritten elements",
                    ));
                }
            }
        }
        Ok(self.data.gather_rows(rows, cols))
    }

    pub fn gather_row_range(
        &self,
        rows: &[usize],
        col_start: usize,
        len: usize,
    ) -> IResult<ValueArray2> {
        if col_start + len > self.data.ncols() {
            return Err(InterpreterError::new(
                "tensor_value",
                "register slice is out of bounds",
            ));
        }
        for &r in rows {
            if self
                .valid
                .slice(s![r, col_start..col_start + len])
                .iter()
                .any(|v| !*v)
            {
                return Err(InterpreterError::new(
                    "missing_tensor_value",
                    "register slice reads unwritten elements",
                ));
            }
        }
        Ok(self.data.gather_row_range(rows, col_start, len))
    }

    /// Scatter `[A,k]`: data[rows[a], cols[a,j]] = values[a,j]. Total (pre-validated).
    pub fn scatter_rows(
        &mut self,
        rows: &[usize],
        cols: &Array2<usize>,
        values: &ValueArray2,
    ) -> IResult<()> {
        let a = rows.len();
        let k = cols.ncols();
        self.data.scatter_rows(rows, cols, values)?;
        for ai in 0..a {
            let r = rows[ai];
            for j in 0..k {
                let c = cols[[ai, j]];
                self.valid[[r, c]] = true;
            }
        }
        Ok(())
    }

    pub fn scatter_row_range(
        &mut self,
        rows: &[usize],
        col_start: usize,
        len: usize,
        values: &ValueArray2,
    ) -> IResult<()> {
        if col_start + len > self.data.ncols() {
            return Err(InterpreterError::new(
                "tensor_value",
                "register slice is out of bounds",
            ));
        }
        self.data.scatter_row_range(rows, col_start, len, values)?;
        for &r in rows {
            self.valid
                .slice_mut(s![r, col_start..col_start + len])
                .fill(true);
        }
        Ok(())
    }

    pub fn snapshot_float_row_range(
        &self,
        rows: &[usize],
        col_start: usize,
        len: usize,
    ) -> IResult<Vec<f32>> {
        if !matches!(self.dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Err(InterpreterError::new(
                "tensor_value",
                "direct REG snapshot requires float dtype",
            ));
        }
        check_range(self, rows, col_start, len)?;
        check_valid_range(self, rows, col_start, len)?;
        let data = float_data(&self.data);
        let mut out = Vec::with_capacity(rows.len() * len);
        for &row in rows {
            for j in 0..len {
                out.push(data[[row, col_start + j]]);
            }
        }
        Ok(out)
    }

    pub fn write_float_row_range_values(
        &mut self,
        rows: &[usize],
        col_start: usize,
        len: usize,
        values: &[f32],
    ) -> IResult<()> {
        if values.len() != rows.len() * len {
            return Err(InterpreterError::new(
                "tensor_value",
                "direct REG write value count mismatch",
            ));
        }
        self.write_float_row_range_with(rows, col_start, len, |ai, j| values[ai * len + j])
    }

    pub fn write_float_row_range_with(
        &mut self,
        rows: &[usize],
        col_start: usize,
        len: usize,
        mut value_at: impl FnMut(usize, usize) -> f32,
    ) -> IResult<()> {
        if !matches!(self.dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Err(InterpreterError::new(
                "tensor_value",
                "direct REG write requires float dtype",
            ));
        }
        check_range(self, rows, col_start, len)?;
        let dtype = self.dtype;
        let data = float_data_mut(&mut self.data);
        for (ai, &row) in rows.iter().enumerate() {
            for j in 0..len {
                data[[row, col_start + j]] = round_scalar(value_at(ai, j), dtype);
            }
        }
        mark_valid_range(self, rows, col_start, len);
        Ok(())
    }

    /// Read one thread's full dense tensor (for inspection/tests).
    pub fn thread_value(&self, thread: &ThreadId) -> IResult<ValueArray1> {
        let row = register_row(thread);
        if row >= self.data.nrows() {
            return Err(InterpreterError::new(
                "tensor_value",
                "register row out of range",
            ));
        }
        let cols = Array2::from_shape_fn((1, numel(&self.shape)), |(_, j)| j);
        Ok(self.data.gather_rows(&[row], &cols).flatten_to_1d())
    }
}

fn check_range(
    tensor: &RegisterTensorValue,
    rows: &[usize],
    col_start: usize,
    len: usize,
) -> IResult<()> {
    if col_start + len > tensor.data.ncols() {
        return Err(InterpreterError::new(
            "tensor_value",
            "register slice is out of bounds",
        ));
    }
    if rows.iter().any(|&r| r >= tensor.data.nrows()) {
        return Err(InterpreterError::new(
            "tensor_value",
            "register row out of bounds",
        ));
    }
    Ok(())
}

fn check_valid_range(
    tensor: &RegisterTensorValue,
    rows: &[usize],
    col_start: usize,
    len: usize,
) -> IResult<()> {
    for &r in rows {
        if tensor
            .valid
            .slice(s![r, col_start..col_start + len])
            .iter()
            .any(|v| !*v)
        {
            return Err(InterpreterError::new(
                "missing_tensor_value",
                "register slice reads unwritten elements",
            ));
        }
    }
    Ok(())
}

fn mark_valid_range(
    tensor: &mut RegisterTensorValue,
    rows: &[usize],
    col_start: usize,
    len: usize,
) {
    for &r in rows {
        tensor
            .valid
            .slice_mut(s![r, col_start..col_start + len])
            .fill(true);
    }
}

fn float_data(values: &ValueArray2) -> &Array2<f32> {
    match values {
        ValueArray2::F16(a) | ValueArray2::Bf16(a) | ValueArray2::F32(a) => a,
        _ => unreachable!(),
    }
}

fn float_data_mut(values: &mut ValueArray2) -> &mut Array2<f32> {
    match values {
        ValueArray2::F16(a) | ValueArray2::Bf16(a) | ValueArray2::F32(a) => a,
        _ => unreachable!(),
    }
}

/// Register-file values keyed by (reg tensor, cta_id). Tensor keyed by id.
#[derive(Clone, Debug, Default)]
pub struct RegisterValues {
    pub by_instance: HashMap<RegisterKey, RegisterTensorValue>,
}

#[derive(Clone, Debug)]
pub struct RegisterKey {
    pub tensor: Arc<Tensor>,
    pub cta_id: usize,
}
impl PartialEq for RegisterKey {
    fn eq(&self, other: &Self) -> bool {
        self.tensor.id == other.tensor.id && self.cta_id == other.cta_id
    }
}
impl Eq for RegisterKey {}
impl std::hash::Hash for RegisterKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.id.hash(state);
        self.cta_id.hash(state);
    }
}

impl RegisterValues {
    pub fn get(&self, tensor: &Arc<Tensor>, cta_id: usize) -> Option<&RegisterTensorValue> {
        self.by_instance.get(&RegisterKey {
            tensor: Arc::clone(tensor),
            cta_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::threads::Coord;
    use crate::ir::{DType, MemorySpace, Tensor};
    use ndarray::{arr2, Array2};

    #[test]
    fn gather_scatter_preserves_native_dtype() {
        let mut regs = RegisterTensorValue::empty(vec![2], DType::I32, 4);
        let rows = [1usize, 3usize];
        let cols: Array2<usize> = arr2(&[[0, 1], [1, 0]]);
        let values =
            ValueArray2::from_i64_compute(arr2(&[[1_i64, 0x8000_0000_i64], [-1, 5]]), DType::I32);

        regs.scatter_rows(&rows, &cols, &values).unwrap();
        let gathered = regs.gather_rows(&rows, &cols).unwrap();
        assert_eq!(gathered.dtype(), DType::I32);
        assert_eq!(
            gathered.to_i64_compute(),
            arr2(&[[1_i64, -2147483648_i64], [-1, 5]])
        );

        let coord = Coord::from_slice(&[0]);
        let thread = ThreadId {
            cta_id: 0,
            cta_coord: coord,
            cluster_id: 0,
            ctaid_in_cluster: 0,
            cluster_coord: coord,
            cta_coord_in_cluster: coord,
            warp_id: 0,
            lane_id: 1,
        };
        let full = regs.thread_value(&thread).unwrap();
        assert_eq!(full.dtype(), DType::I32);
        assert_eq!(
            full.to_i64_compute().as_slice().unwrap(),
            &[1, -2147483648_i64]
        );
    }

    #[test]
    fn register_key_uses_tensor_id() {
        let tensor = Arc::new(Tensor {
            id: 7,
            space: MemorySpace::Reg,
            dtype: DType::F32,
            shape: vec![1],
            layout: None,
            byte_offset: None,
        });
        let mut values = RegisterValues::default();
        values.by_instance.insert(
            RegisterKey {
                tensor: Arc::clone(&tensor),
                cta_id: 0,
            },
            RegisterTensorValue::empty(vec![1], DType::F32, 1),
        );
        assert!(values.get(&tensor, 0).is_some());
    }
}
