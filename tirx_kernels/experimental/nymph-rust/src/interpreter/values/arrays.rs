//! Typed value arrays for GMEM/SMEM/REG storage.
//!
//! Float tensors use an f32 physical backing tagged with the tensor's declared
//! dtype: F16/BF16 values are rounded on write, then held exactly in f32 for hot
//! compute paths. Integer and bool tensors keep native container storage. Ops
//! widen locally through `to_f32_compute` / `to_i64_compute` and write back with
//! `coerce_to_dtype`.

use super::dtypes::{round_bf16_scalar, round_e4m3_scalar, wrap_int_to_dtype};
use super::indexing::numel;
use crate::interpreter::diagnostics::{IResult, InterpreterError};
use crate::ir::DType;
use half::f16;
use ndarray::{s, Array1, Array2, ArrayD, IxDyn};

#[derive(Clone, Debug)]
pub enum ValueArray1 {
    Bool(Array1<bool>),
    I8(Array1<i8>),
    U8(Array1<u8>),
    I16(Array1<i16>),
    U16(Array1<u16>),
    I32(Array1<i32>),
    U32(Array1<u32>),
    I64(Array1<i64>),
    U64(Array1<u64>),
    /// e4m3fn stored as the decoded (dtype-rounded) f32 value, like F16/Bf16.
    F8E4M3(Array1<f32>),
    F16(Array1<f32>),
    Bf16(Array1<f32>),
    F32(Array1<f32>),
}

#[derive(Clone, Debug)]
pub enum ValueArray2 {
    Bool(Array2<bool>),
    I8(Array2<i8>),
    U8(Array2<u8>),
    I16(Array2<i16>),
    U16(Array2<u16>),
    I32(Array2<i32>),
    U32(Array2<u32>),
    I64(Array2<i64>),
    U64(Array2<u64>),
    F8E4M3(Array2<f32>),
    F16(Array2<f32>),
    Bf16(Array2<f32>),
    F32(Array2<f32>),
}

impl ValueArray1 {
    pub fn zeros(dtype: DType, len: usize) -> Self {
        match dtype {
            DType::Bool => Self::Bool(Array1::from_elem(len, false)),
            DType::I8 => Self::I8(Array1::zeros(len)),
            DType::U8 => Self::U8(Array1::zeros(len)),
            DType::I16 => Self::I16(Array1::zeros(len)),
            DType::U16 => Self::U16(Array1::zeros(len)),
            DType::I32 => Self::I32(Array1::zeros(len)),
            DType::U32 => Self::U32(Array1::zeros(len)),
            DType::I64 => Self::I64(Array1::zeros(len)),
            DType::U64 => Self::U64(Array1::zeros(len)),
            DType::F8E4M3 => Self::F8E4M3(Array1::zeros(len)),
            DType::F16 => Self::F16(Array1::zeros(len)),
            DType::Bf16 => Self::Bf16(Array1::zeros(len)),
            DType::F32 => Self::F32(Array1::zeros(len)),
        }
    }

    pub fn from_f32_compute(values: Array1<f32>, dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool(values.mapv(|x| x != 0.0)),
            DType::I8 => Self::I8(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as i8)),
            DType::U8 => Self::U8(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u8)),
            DType::I16 => Self::I16(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as i16)),
            DType::U16 => Self::U16(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u16)),
            DType::I32 => Self::I32(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as i32)),
            DType::U32 => Self::U32(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u32)),
            DType::I64 => Self::I64(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype))),
            DType::U64 => Self::U64(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u64)),
            DType::F8E4M3 => Self::F8E4M3(values.mapv(round_e4m3_scalar)),
            DType::F16 => Self::F16(values.mapv(|x| f16::from_f32(x).to_f32())),
            DType::Bf16 => Self::Bf16(values.mapv(round_bf16_scalar)),
            DType::F32 => Self::F32(values),
        }
    }

    pub fn from_i64_compute(values: Array1<i64>, dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool(values.mapv(|x| x != 0)),
            DType::I8 => Self::I8(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as i8)),
            DType::U8 => Self::U8(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u8)),
            DType::I16 => Self::I16(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as i16)),
            DType::U16 => Self::U16(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u16)),
            DType::I32 => Self::I32(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as i32)),
            DType::U32 => Self::U32(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u32)),
            DType::I64 => Self::I64(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype))),
            DType::U64 => Self::U64(values.mapv(|x| wrap_int_to_dtype(x as i128, dtype) as u64)),
            DType::F8E4M3 => Self::F8E4M3(values.mapv(|x| round_e4m3_scalar(x as f32))),
            DType::F16 => Self::F16(values.mapv(|x| f16::from_f32(x as f32).to_f32())),
            DType::Bf16 => Self::Bf16(values.mapv(|x| round_bf16_scalar(x as f32))),
            DType::F32 => Self::F32(values.mapv(|x| x as f32)),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::U8(_) => DType::U8,
            Self::I16(_) => DType::I16,
            Self::U16(_) => DType::U16,
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::U64(_) => DType::U64,
            Self::F8E4M3(_) => DType::F8E4M3,
            Self::F16(_) => DType::F16,
            Self::Bf16(_) => DType::Bf16,
            Self::F32(_) => DType::F32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bool(a) => a.len(),
            Self::I8(a) => a.len(),
            Self::U8(a) => a.len(),
            Self::I16(a) => a.len(),
            Self::U16(a) => a.len(),
            Self::I32(a) => a.len(),
            Self::U32(a) => a.len(),
            Self::I64(a) => a.len(),
            Self::U64(a) => a.len(),
            Self::F8E4M3(a) => a.len(),
            Self::F16(a) => a.len(),
            Self::Bf16(a) => a.len(),
            Self::F32(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_f32_compute(&self) -> Array1<f32> {
        match self {
            Self::Bool(a) => a.mapv(|x| if x { 1.0 } else { 0.0 }),
            Self::I8(a) => a.mapv(|x| x as f32),
            Self::U8(a) => a.mapv(|x| x as f32),
            Self::I16(a) => a.mapv(|x| x as f32),
            Self::U16(a) => a.mapv(|x| x as f32),
            Self::I32(a) => a.mapv(|x| x as f32),
            Self::U32(a) => a.mapv(|x| x as f32),
            Self::I64(a) => a.mapv(|x| x as f32),
            Self::U64(a) => a.mapv(|x| x as f32),
            Self::F8E4M3(a) => a.clone(),
            Self::F16(a) => a.clone(),
            Self::Bf16(a) => a.clone(),
            Self::F32(a) => a.clone(),
        }
    }

    pub fn to_i64_compute(&self) -> Array1<i64> {
        match self {
            Self::Bool(a) => a.mapv(|x| if x { 1 } else { 0 }),
            Self::I8(a) => a.mapv(|x| x as i64),
            Self::U8(a) => a.mapv(|x| x as i64),
            Self::I16(a) => a.mapv(|x| x as i64),
            Self::U16(a) => a.mapv(|x| x as i64),
            Self::I32(a) => a.mapv(|x| x as i64),
            Self::U32(a) => a.mapv(|x| x as i64),
            Self::I64(a) => a.clone(),
            Self::U64(a) => a.mapv(|x| x as i64),
            Self::F8E4M3(a) => a.mapv(|x| x as i64),
            Self::F16(a) => a.mapv(|x| x as i64),
            Self::Bf16(a) => a.mapv(|x| x as i64),
            Self::F32(a) => a.mapv(|x| x as i64),
        }
    }

    pub fn coerce_to_dtype(&self, dst_dtype: DType) -> Self {
        if self.dtype() == dst_dtype {
            return self.clone();
        }
        if matches!(dst_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Self::from_f32_compute(self.to_f32_compute(), dst_dtype);
        }
        if dst_dtype == DType::Bool && matches!(self.dtype(), DType::F16 | DType::Bf16 | DType::F32)
        {
            return Self::Bool(self.to_f32_compute().mapv(|x| x != 0.0));
        }
        Self::from_i64_compute(self.to_i64_compute(), dst_dtype)
    }

    pub fn into_coerce_to_dtype(self, dst_dtype: DType) -> Self {
        if self.dtype() == dst_dtype {
            return self;
        }
        if matches!(dst_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Self::from_f32_compute(self.to_f32_compute(), dst_dtype);
        }
        if dst_dtype == DType::Bool && matches!(self.dtype(), DType::F16 | DType::Bf16 | DType::F32)
        {
            return Self::Bool(self.to_f32_compute().mapv(|x| x != 0.0));
        }
        Self::from_i64_compute(self.to_i64_compute(), dst_dtype)
    }

    pub fn reshape2(self, shape: (usize, usize)) -> IResult<ValueArray2> {
        let expected = shape.0 * shape.1;
        if self.len() != expected {
            return Err(InterpreterError::new(
                "tensor_value",
                "value reshape size mismatch",
            ));
        }
        Ok(match self {
            Self::Bool(a) => ValueArray2::Bool(a.into_shape_with_order(shape).unwrap()),
            Self::I8(a) => ValueArray2::I8(a.into_shape_with_order(shape).unwrap()),
            Self::U8(a) => ValueArray2::U8(a.into_shape_with_order(shape).unwrap()),
            Self::I16(a) => ValueArray2::I16(a.into_shape_with_order(shape).unwrap()),
            Self::U16(a) => ValueArray2::U16(a.into_shape_with_order(shape).unwrap()),
            Self::I32(a) => ValueArray2::I32(a.into_shape_with_order(shape).unwrap()),
            Self::U32(a) => ValueArray2::U32(a.into_shape_with_order(shape).unwrap()),
            Self::I64(a) => ValueArray2::I64(a.into_shape_with_order(shape).unwrap()),
            Self::U64(a) => ValueArray2::U64(a.into_shape_with_order(shape).unwrap()),
            Self::F8E4M3(a) => ValueArray2::F8E4M3(a.into_shape_with_order(shape).unwrap()),
            Self::F16(a) => ValueArray2::F16(a.into_shape_with_order(shape).unwrap()),
            Self::Bf16(a) => ValueArray2::Bf16(a.into_shape_with_order(shape).unwrap()),
            Self::F32(a) => ValueArray2::F32(a.into_shape_with_order(shape).unwrap()),
        })
    }

    pub fn gather2(&self, indices: &Array2<usize>) -> ValueArray2 {
        let shape = indices.raw_dim();
        match self {
            Self::Bool(a) => ValueArray2::Bool(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::I8(a) => ValueArray2::I8(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::U8(a) => ValueArray2::U8(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::I16(a) => ValueArray2::I16(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::U16(a) => ValueArray2::U16(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::I32(a) => ValueArray2::I32(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::U32(a) => ValueArray2::U32(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::I64(a) => ValueArray2::I64(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::U64(a) => ValueArray2::U64(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::F8E4M3(a) => {
                ValueArray2::F8E4M3(Array2::from_shape_fn(shape, |ij| a[indices[ij]]))
            }
            Self::F16(a) => ValueArray2::F16(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::Bf16(a) => ValueArray2::Bf16(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
            Self::F32(a) => ValueArray2::F32(Array2::from_shape_fn(shape, |ij| a[indices[ij]])),
        }
    }

    pub fn scatter_indices(&mut self, indices: &[usize], values: &ValueArray1) -> IResult<()> {
        if indices.len() != values.len() {
            return Err(InterpreterError::new(
                "tensor_value",
                "slice write value count mismatch",
            ));
        }
        macro_rules! scatter {
            ($dst:expr, $src:expr) => {{
                let vals = $src.as_slice().expect("contiguous values");
                for (i, &flat) in indices.iter().enumerate() {
                    $dst[flat] = vals[i];
                }
            }};
        }
        match (self, values) {
            (Self::Bool(dst), Self::Bool(src)) => scatter!(dst, src),
            (Self::I8(dst), Self::I8(src)) => scatter!(dst, src),
            (Self::U8(dst), Self::U8(src)) => scatter!(dst, src),
            (Self::I16(dst), Self::I16(src)) => scatter!(dst, src),
            (Self::U16(dst), Self::U16(src)) => scatter!(dst, src),
            (Self::I32(dst), Self::I32(src)) => scatter!(dst, src),
            (Self::U32(dst), Self::U32(src)) => scatter!(dst, src),
            (Self::I64(dst), Self::I64(src)) => scatter!(dst, src),
            (Self::U64(dst), Self::U64(src)) => scatter!(dst, src),
            (Self::F8E4M3(dst), Self::F8E4M3(src)) => scatter!(dst, src),
            (Self::F16(dst), Self::F16(src)) => scatter!(dst, src),
            (Self::Bf16(dst), Self::Bf16(src)) => scatter!(dst, src),
            (Self::F32(dst), Self::F32(src)) => scatter!(dst, src),
            _ => {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "slice write dtype must match the destination container",
                ))
            }
        }
        Ok(())
    }

    pub fn copy_run_from(
        &mut self,
        dst_start: usize,
        src: &ValueArray1,
        src_start: usize,
        len: usize,
    ) -> IResult<()> {
        macro_rules! copy_run {
            ($dst:expr, $src:expr) => {
                $dst.as_slice_mut().expect("contiguous data")[dst_start..dst_start + len]
                    .copy_from_slice(
                        &$src.as_slice().expect("contiguous values")[src_start..src_start + len],
                    );
            };
        }
        match (self, src) {
            (Self::Bool(dst), Self::Bool(src)) => {
                copy_run!(dst, src);
            }
            (Self::I8(dst), Self::I8(src)) => {
                copy_run!(dst, src);
            }
            (Self::U8(dst), Self::U8(src)) => {
                copy_run!(dst, src);
            }
            (Self::I16(dst), Self::I16(src)) => {
                copy_run!(dst, src);
            }
            (Self::U16(dst), Self::U16(src)) => {
                copy_run!(dst, src);
            }
            (Self::I32(dst), Self::I32(src)) => {
                copy_run!(dst, src);
            }
            (Self::U32(dst), Self::U32(src)) => {
                copy_run!(dst, src);
            }
            (Self::I64(dst), Self::I64(src)) => {
                copy_run!(dst, src);
            }
            (Self::U64(dst), Self::U64(src)) => {
                copy_run!(dst, src);
            }
            (Self::F8E4M3(dst), Self::F8E4M3(src)) => {
                copy_run!(dst, src);
            }
            (Self::F16(dst), Self::F16(src)) => {
                copy_run!(dst, src);
            }
            (Self::Bf16(dst), Self::Bf16(src)) => {
                copy_run!(dst, src);
            }
            (Self::F32(dst), Self::F32(src)) => {
                copy_run!(dst, src);
            }
            _ => {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "slice write dtype must match the destination container",
                ))
            }
        }
        Ok(())
    }
}

impl ValueArray2 {
    pub fn zeros(dtype: DType, shape: (usize, usize)) -> Self {
        match dtype {
            DType::Bool => Self::Bool(Array2::from_elem(shape, false)),
            DType::I8 => Self::I8(Array2::zeros(shape)),
            DType::U8 => Self::U8(Array2::zeros(shape)),
            DType::I16 => Self::I16(Array2::zeros(shape)),
            DType::U16 => Self::U16(Array2::zeros(shape)),
            DType::I32 => Self::I32(Array2::zeros(shape)),
            DType::U32 => Self::U32(Array2::zeros(shape)),
            DType::I64 => Self::I64(Array2::zeros(shape)),
            DType::U64 => Self::U64(Array2::zeros(shape)),
            DType::F8E4M3 => Self::F8E4M3(Array2::zeros(shape)),
            DType::F16 => Self::F16(Array2::zeros(shape)),
            DType::Bf16 => Self::Bf16(Array2::zeros(shape)),
            DType::F32 => Self::F32(Array2::zeros(shape)),
        }
    }

    pub fn from_f32_compute(values: Array2<f32>, dtype: DType) -> Self {
        let shape = (values.nrows(), values.ncols());
        let flat = Array1::from_iter(values.as_standard_layout().iter().copied());
        ValueArray1::from_f32_compute(flat, dtype)
            .reshape2(shape)
            .unwrap()
    }

    pub fn from_i64_compute(values: Array2<i64>, dtype: DType) -> Self {
        let shape = (values.nrows(), values.ncols());
        let flat = Array1::from_iter(values.as_standard_layout().iter().copied());
        ValueArray1::from_i64_compute(flat, dtype)
            .reshape2(shape)
            .unwrap()
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::U8(_) => DType::U8,
            Self::I16(_) => DType::I16,
            Self::U16(_) => DType::U16,
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::U64(_) => DType::U64,
            Self::F8E4M3(_) => DType::F8E4M3,
            Self::F16(_) => DType::F16,
            Self::Bf16(_) => DType::Bf16,
            Self::F32(_) => DType::F32,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Bool(a) => a.dim(),
            Self::I8(a) => a.dim(),
            Self::U8(a) => a.dim(),
            Self::I16(a) => a.dim(),
            Self::U16(a) => a.dim(),
            Self::I32(a) => a.dim(),
            Self::U32(a) => a.dim(),
            Self::I64(a) => a.dim(),
            Self::U64(a) => a.dim(),
            Self::F8E4M3(a) => a.dim(),
            Self::F16(a) => a.dim(),
            Self::Bf16(a) => a.dim(),
            Self::F32(a) => a.dim(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    pub fn to_f32_compute(&self) -> Array2<f32> {
        match self {
            Self::Bool(a) => a.mapv(|x| if x { 1.0 } else { 0.0 }),
            Self::I8(a) => a.mapv(|x| x as f32),
            Self::U8(a) => a.mapv(|x| x as f32),
            Self::I16(a) => a.mapv(|x| x as f32),
            Self::U16(a) => a.mapv(|x| x as f32),
            Self::I32(a) => a.mapv(|x| x as f32),
            Self::U32(a) => a.mapv(|x| x as f32),
            Self::I64(a) => a.mapv(|x| x as f32),
            Self::U64(a) => a.mapv(|x| x as f32),
            Self::F8E4M3(a) => a.clone(),
            Self::F16(a) => a.clone(),
            Self::Bf16(a) => a.clone(),
            Self::F32(a) => a.clone(),
        }
    }

    pub fn to_i64_compute(&self) -> Array2<i64> {
        match self {
            Self::Bool(a) => a.mapv(|x| if x { 1 } else { 0 }),
            Self::I8(a) => a.mapv(|x| x as i64),
            Self::U8(a) => a.mapv(|x| x as i64),
            Self::I16(a) => a.mapv(|x| x as i64),
            Self::U16(a) => a.mapv(|x| x as i64),
            Self::I32(a) => a.mapv(|x| x as i64),
            Self::U32(a) => a.mapv(|x| x as i64),
            Self::I64(a) => a.clone(),
            Self::U64(a) => a.mapv(|x| x as i64),
            Self::F8E4M3(a) => a.mapv(|x| x as i64),
            Self::F16(a) => a.mapv(|x| x as i64),
            Self::Bf16(a) => a.mapv(|x| x as i64),
            Self::F32(a) => a.mapv(|x| x as i64),
        }
    }

    pub fn coerce_to_dtype(&self, dst_dtype: DType) -> Self {
        if self.dtype() == dst_dtype {
            return self.clone();
        }
        if matches!(dst_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Self::from_f32_compute(self.to_f32_compute(), dst_dtype);
        }
        if dst_dtype == DType::Bool && matches!(self.dtype(), DType::F16 | DType::Bf16 | DType::F32)
        {
            return Self::Bool(self.to_f32_compute().mapv(|x| x != 0.0));
        }
        Self::from_i64_compute(self.to_i64_compute(), dst_dtype)
    }

    pub fn into_coerce_to_dtype(self, dst_dtype: DType) -> Self {
        if self.dtype() == dst_dtype {
            return self;
        }
        if matches!(dst_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Self::from_f32_compute(self.to_f32_compute(), dst_dtype);
        }
        if dst_dtype == DType::Bool && matches!(self.dtype(), DType::F16 | DType::Bf16 | DType::F32)
        {
            return Self::Bool(self.to_f32_compute().mapv(|x| x != 0.0));
        }
        Self::from_i64_compute(self.to_i64_compute(), dst_dtype)
    }

    pub fn flatten_to_1d(&self) -> ValueArray1 {
        match self {
            Self::Bool(a) => ValueArray1::Bool(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::I8(a) => ValueArray1::I8(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::U8(a) => ValueArray1::U8(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::I16(a) => ValueArray1::I16(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::U16(a) => ValueArray1::U16(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::I32(a) => ValueArray1::I32(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::U32(a) => ValueArray1::U32(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::I64(a) => ValueArray1::I64(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::U64(a) => ValueArray1::U64(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::F8E4M3(a) => ValueArray1::F8E4M3(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::F16(a) => ValueArray1::F16(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::Bf16(a) => ValueArray1::Bf16(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
            Self::F32(a) => ValueArray1::F32(Array1::from(
                a.as_standard_layout().iter().copied().collect::<Vec<_>>(),
            )),
        }
    }

    pub fn scatter_rows(
        &mut self,
        rows: &[usize],
        cols: &Array2<usize>,
        values: &ValueArray2,
    ) -> IResult<()> {
        if values.shape() != cols.dim() || rows.len() != cols.nrows() {
            return Err(InterpreterError::new(
                "tensor_value",
                "register slice write value count mismatch",
            ));
        }
        macro_rules! scatter {
            ($dst:expr, $src:expr) => {{
                for ai in 0..rows.len() {
                    let r = rows[ai];
                    for j in 0..cols.ncols() {
                        $dst[[r, cols[[ai, j]]]] = $src[[ai, j]];
                    }
                }
            }};
        }
        match (self, values) {
            (Self::Bool(dst), Self::Bool(src)) => scatter!(dst, src),
            (Self::I8(dst), Self::I8(src)) => scatter!(dst, src),
            (Self::U8(dst), Self::U8(src)) => scatter!(dst, src),
            (Self::I16(dst), Self::I16(src)) => scatter!(dst, src),
            (Self::U16(dst), Self::U16(src)) => scatter!(dst, src),
            (Self::I32(dst), Self::I32(src)) => scatter!(dst, src),
            (Self::U32(dst), Self::U32(src)) => scatter!(dst, src),
            (Self::I64(dst), Self::I64(src)) => scatter!(dst, src),
            (Self::U64(dst), Self::U64(src)) => scatter!(dst, src),
            (Self::F8E4M3(dst), Self::F8E4M3(src)) => scatter!(dst, src),
            (Self::F16(dst), Self::F16(src)) => scatter!(dst, src),
            (Self::Bf16(dst), Self::Bf16(src)) => scatter!(dst, src),
            (Self::F32(dst), Self::F32(src)) => scatter!(dst, src),
            _ => {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "register write dtype must match the destination container",
                ))
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
        if values.shape() != (rows.len(), len) {
            return Err(InterpreterError::new(
                "tensor_value",
                "register slice write value count mismatch",
            ));
        }
        macro_rules! scatter {
            ($dst:expr, $src:expr) => {{
                for (ai, &r) in rows.iter().enumerate() {
                    $dst.slice_mut(s![r, col_start..col_start + len])
                        .assign(&$src.slice(s![ai, ..]));
                }
            }};
        }
        match (self, values) {
            (Self::Bool(dst), Self::Bool(src)) => scatter!(dst, src),
            (Self::I8(dst), Self::I8(src)) => scatter!(dst, src),
            (Self::U8(dst), Self::U8(src)) => scatter!(dst, src),
            (Self::I16(dst), Self::I16(src)) => scatter!(dst, src),
            (Self::U16(dst), Self::U16(src)) => scatter!(dst, src),
            (Self::I32(dst), Self::I32(src)) => scatter!(dst, src),
            (Self::U32(dst), Self::U32(src)) => scatter!(dst, src),
            (Self::I64(dst), Self::I64(src)) => scatter!(dst, src),
            (Self::U64(dst), Self::U64(src)) => scatter!(dst, src),
            (Self::F8E4M3(dst), Self::F8E4M3(src)) => scatter!(dst, src),
            (Self::F16(dst), Self::F16(src)) => scatter!(dst, src),
            (Self::Bf16(dst), Self::Bf16(src)) => scatter!(dst, src),
            (Self::F32(dst), Self::F32(src)) => scatter!(dst, src),
            _ => {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "register write dtype must match the destination container",
                ))
            }
        }
        Ok(())
    }

    pub fn gather_rows(&self, rows: &[usize], cols: &Array2<usize>) -> ValueArray2 {
        let shape = cols.raw_dim();
        match self {
            Self::Bool(a) => Self::Bool(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::I8(a) => Self::I8(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::U8(a) => Self::U8(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::I16(a) => Self::I16(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::U16(a) => Self::U16(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::I32(a) => Self::I32(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::U32(a) => Self::U32(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::I64(a) => Self::I64(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::U64(a) => Self::U64(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::F8E4M3(a) => Self::F8E4M3(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::F16(a) => Self::F16(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::Bf16(a) => Self::Bf16(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
            Self::F32(a) => Self::F32(Array2::from_shape_fn(shape, |(ai, j)| {
                a[[rows[ai], cols[[ai, j]]]]
            })),
        }
    }

    pub fn gather_row_range(&self, rows: &[usize], col_start: usize, len: usize) -> ValueArray2 {
        let shape = (rows.len(), len);
        macro_rules! gather {
            ($src:expr) => {{
                Array2::from_shape_fn(shape, |(ai, j)| $src[[rows[ai], col_start + j]])
            }};
        }
        match self {
            Self::Bool(a) => Self::Bool(gather!(a)),
            Self::I8(a) => Self::I8(gather!(a)),
            Self::U8(a) => Self::U8(gather!(a)),
            Self::I16(a) => Self::I16(gather!(a)),
            Self::U16(a) => Self::U16(gather!(a)),
            Self::I32(a) => Self::I32(gather!(a)),
            Self::U32(a) => Self::U32(gather!(a)),
            Self::I64(a) => Self::I64(gather!(a)),
            Self::U64(a) => Self::U64(gather!(a)),
            Self::F8E4M3(a) => Self::F8E4M3(gather!(a)),
            Self::F16(a) => Self::F16(gather!(a)),
            Self::Bf16(a) => Self::Bf16(gather!(a)),
            Self::F32(a) => Self::F32(gather!(a)),
        }
    }
}

pub fn coerce_f32_arrayd(
    values: &ArrayD<f32>,
    shape: &[usize],
    dtype: DType,
) -> IResult<ValueArray1> {
    if values.shape() != shape {
        return Err(InterpreterError::new(
            "tensor_value",
            "input array shape mismatch",
        ));
    }
    let flat = values
        .to_owned()
        .into_shape_with_order(IxDyn(&[numel(shape)]))
        .map_err(|_| InterpreterError::new("tensor_value", "input array is not contiguous"))?;
    Ok(ValueArray1::from_f32_compute(
        flat.into_dimensionality().unwrap(),
        dtype,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn constructs_container_for_each_dtype() {
        let dtypes = [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::Bf16,
            DType::F32,
        ];
        for dtype in dtypes {
            let a = ValueArray1::zeros(dtype, 3);
            assert_eq!(a.dtype(), dtype);
            assert_eq!(a.len(), 3);
            let b = ValueArray2::zeros(dtype, (2, 3));
            assert_eq!(b.dtype(), dtype);
            assert_eq!(b.shape(), (2, 3));
        }
    }

    #[test]
    fn coerce_float_precisions_and_int_wrap() {
        let f = ValueArray1::from_f32_compute(
            arr1(&[1.0 + 1.0 / 2048.0, 1.0 + 1.0 / 512.0]),
            DType::F32,
        );
        let h = f.coerce_to_dtype(DType::F16);
        assert_eq!(h.to_f32_compute()[0], 1.0);
        let bf = ValueArray1::from_f32_compute(arr1(&[1.00390625]), DType::Bf16);
        assert_eq!(bf.to_f32_compute()[0], round_bf16_scalar(1.00390625));

        let i = ValueArray1::from_i64_compute(arr1(&[-1_i64, 0x1_0000_0000_i64]), DType::U32);
        assert_eq!(
            i.to_i64_compute().as_slice().unwrap(),
            &[0xFFFF_FFFF_i64, 0]
        );
        let s = ValueArray1::from_i64_compute(arr1(&[0x8000_0000_i64]), DType::I32);
        assert_eq!(s.to_i64_compute()[0], -2147483648);
        let b = ValueArray1::from_i64_compute(arr1(&[0, 7]), DType::Bool);
        assert_eq!(b.to_i64_compute().as_slice().unwrap(), &[0, 1]);
    }

    #[test]
    fn float_arrays_use_f32_backing_with_declared_dtype_tag() {
        let f16 = ValueArray1::from_f32_compute(arr1(&[1.0 + 1.0 / 2048.0]), DType::F16);
        match &f16 {
            ValueArray1::F16(values) => assert_eq!(values.as_slice().unwrap(), &[1.0]),
            _ => panic!("expected f16-tagged f32 backing"),
        }

        let bf16 = ValueArray1::from_f32_compute(arr1(&[1.00390625]), DType::Bf16);
        match &bf16 {
            ValueArray1::Bf16(values) => {
                assert_eq!(values.as_slice().unwrap(), &[round_bf16_scalar(1.00390625)])
            }
            _ => panic!("expected bf16-tagged f32 backing"),
        }
    }

    #[test]
    fn gather_scatter_preserves_native_dtype() {
        let mut data = ValueArray1::from_i64_compute(arr1(&[0_i64, 1, 2, 3]), DType::U32);
        let vals = ValueArray1::from_i64_compute(arr1(&[-1_i64, 5]), DType::U32);
        data.scatter_indices(&[1, 3], &vals).unwrap();
        assert_eq!(
            data.to_i64_compute().as_slice().unwrap(),
            &[0, 0xFFFF_FFFF_i64, 2, 5]
        );

        let gathered = data.gather2(&arr2(&[[3, 1], [0, 2]]));
        assert_eq!(gathered.dtype(), DType::U32);
        assert_eq!(
            gathered.to_i64_compute(),
            arr2(&[[5_i64, 0xFFFF_FFFF_i64], [0, 2]])
        );
    }
}
