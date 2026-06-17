//! CTA-local SMEM byte pool.
//!
//! GMEM remains dense tensor storage; SMEM is modeled as one byte-addressed pool
//! per CTA. A SMEM tensor is only a typed view: `byte_offset + flat_index *
//! dtype_size`.

use super::super::diagnostics::{IResult, InterpreterError};
use super::arrays::ValueArray1;
use super::dtypes::{decode_e4m3, encode_e4m3, round_bf16_scalar};
use super::indexing::numel;
use super::tensors::{flat_with_strides, rect_intra_offsets, row_major_strides};
use crate::ir::{DType, MemorySpace, Tensor};
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use half::{bf16, f16};
use ndarray::Array1;
use std::collections::HashMap;
use std::ops::Range;

#[inline]
pub fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::I8 | DType::U8 | DType::F8E4M3 => 1,
        DType::I16 | DType::U16 | DType::F16 | DType::Bf16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 => 8,
    }
}

pub fn tensor_byte_extent(tensor: &Tensor) -> IResult<usize> {
    numel(&tensor.shape)
        .checked_mul(dtype_size_bytes(tensor.dtype))
        .ok_or_else(|| InterpreterError::new("smem_value", "SMEM tensor byte extent overflows"))
}

#[derive(Clone, Debug, Default)]
pub struct SmemScratchpad {
    pub bytes: Vec<u8>,
    pub valid: Vec<bool>,
}

impl SmemScratchpad {
    pub fn new(size_bytes: usize) -> Self {
        Self {
            bytes: vec![0; size_bytes],
            valid: vec![false; size_bytes],
        }
    }

    fn base_offset(tensor: &Tensor) -> IResult<usize> {
        if tensor.space != MemorySpace::Smem {
            return Err(InterpreterError::new(
                "smem_value",
                "SMEM byte pool access requires an SMEM tensor",
            ));
        }
        tensor.byte_offset.ok_or_else(|| {
            InterpreterError::new("smem_value", "SMEM tensor byte_offset is missing")
        })
    }

    fn byte_range(&self, tensor: &Tensor, flat: usize) -> IResult<Range<usize>> {
        if flat >= numel(&tensor.shape) {
            return Err(InterpreterError::new(
                "tensor_value",
                "SMEM tensor index is out of bounds",
            ));
        }
        let elem_size = dtype_size_bytes(tensor.dtype);
        let start =
            Self::base_offset(tensor)?
                .checked_add(flat.checked_mul(elem_size).ok_or_else(|| {
                    InterpreterError::new("smem_value", "SMEM byte offset overflows")
                })?)
                .ok_or_else(|| InterpreterError::new("smem_value", "SMEM byte offset overflows"))?;
        let end = start
            .checked_add(elem_size)
            .ok_or_else(|| InterpreterError::new("smem_value", "SMEM byte range overflows"))?;
        if end > self.bytes.len() {
            return Err(InterpreterError::new(
                "smem_oob",
                "SMEM tensor byte range exceeds the CTA pool",
            ));
        }
        Ok(start..end)
    }

    fn check_valid(&self, range: Range<usize>) -> IResult<()> {
        if self.valid[range].iter().any(|&v| !v) {
            return Err(InterpreterError::new(
                "missing_tensor_value",
                "SMEM tensor slice reads unwritten bytes",
            ));
        }
        Ok(())
    }

    pub fn read_indices(&self, tensor: &Tensor, indices: &[usize]) -> IResult<ValueArray1> {
        match tensor.dtype {
            DType::Bool => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let range = self.byte_range(tensor, flat)?;
                    self.check_valid(range.clone())?;
                    out.push(self.bytes[range.start] != 0);
                }
                Ok(ValueArray1::Bool(Array1::from(out)))
            }
            DType::I8 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let range = self.byte_range(tensor, flat)?;
                    self.check_valid(range.clone())?;
                    out.push(self.bytes[range.start] as i8);
                }
                Ok(ValueArray1::I8(Array1::from(out)))
            }
            DType::U8 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let range = self.byte_range(tensor, flat)?;
                    self.check_valid(range.clone())?;
                    out.push(self.bytes[range.start]);
                }
                Ok(ValueArray1::U8(Array1::from(out)))
            }
            DType::I16 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<2>(tensor, flat)?;
                    out.push(i16::from_le_bytes(b));
                }
                Ok(ValueArray1::I16(Array1::from(out)))
            }
            DType::U16 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<2>(tensor, flat)?;
                    out.push(u16::from_le_bytes(b));
                }
                Ok(ValueArray1::U16(Array1::from(out)))
            }
            DType::I32 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<4>(tensor, flat)?;
                    out.push(i32::from_le_bytes(b));
                }
                Ok(ValueArray1::I32(Array1::from(out)))
            }
            DType::U32 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<4>(tensor, flat)?;
                    out.push(u32::from_le_bytes(b));
                }
                Ok(ValueArray1::U32(Array1::from(out)))
            }
            DType::I64 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<8>(tensor, flat)?;
                    out.push(i64::from_le_bytes(b));
                }
                Ok(ValueArray1::I64(Array1::from(out)))
            }
            DType::U64 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<8>(tensor, flat)?;
                    out.push(u64::from_le_bytes(b));
                }
                Ok(ValueArray1::U64(Array1::from(out)))
            }
            DType::F8E4M3 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<1>(tensor, flat)?;
                    out.push(decode_e4m3(b[0]));
                }
                Ok(ValueArray1::F8E4M3(Array1::from(out)))
            }
            DType::F16 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<2>(tensor, flat)?;
                    out.push(f16::from_bits(u16::from_le_bytes(b)).to_f32());
                }
                Ok(ValueArray1::F16(Array1::from(out)))
            }
            DType::Bf16 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<2>(tensor, flat)?;
                    out.push(f32::from_bits((u16::from_le_bytes(b) as u32) << 16));
                }
                Ok(ValueArray1::Bf16(Array1::from(out)))
            }
            DType::F32 => {
                let mut out = Vec::with_capacity(indices.len());
                for &flat in indices {
                    let b = self.read_bytes::<4>(tensor, flat)?;
                    out.push(f32::from_bits(u32::from_le_bytes(b)));
                }
                Ok(ValueArray1::F32(Array1::from(out)))
            }
        }
    }

    fn read_bytes<const N: usize>(&self, tensor: &Tensor, flat: usize) -> IResult<[u8; N]> {
        let range = self.byte_range(tensor, flat)?;
        debug_assert_eq!(range.len(), N);
        self.check_valid(range.clone())?;
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[range]);
        Ok(out)
    }

    pub fn read_u16_bits_indices(&self, tensor: &Tensor, indices: &[usize]) -> IResult<Vec<u16>> {
        if dtype_size_bytes(tensor.dtype) != 2 {
            return Err(InterpreterError::new(
                "tensor_value",
                "raw b16 read requires a 16-bit tensor dtype",
            ));
        }
        let mut out = Vec::with_capacity(indices.len());
        for &flat in indices {
            out.push(u16::from_le_bytes(self.read_bytes::<2>(tensor, flat)?));
        }
        Ok(out)
    }

    pub fn write_u16_bits_indices(
        &mut self,
        tensor: &Tensor,
        indices: &[usize],
        values: &[u16],
    ) -> IResult<()> {
        if dtype_size_bytes(tensor.dtype) != 2 {
            return Err(InterpreterError::new(
                "tensor_value",
                "raw b16 write requires a 16-bit tensor dtype",
            ));
        }
        if indices.len() != values.len() {
            return Err(InterpreterError::new(
                "tensor_value",
                "raw b16 write value count mismatch",
            ));
        }
        for (&flat, &value) in indices.iter().zip(values.iter()) {
            self.write_bytes(tensor, flat, &value.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn write_indices(
        &mut self,
        tensor: &Tensor,
        indices: &[usize],
        values: &ValueArray1,
    ) -> IResult<()> {
        if values.dtype() != tensor.dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "SMEM write dtype must match the destination tensor",
            ));
        }
        if values.len() != indices.len() {
            return Err(InterpreterError::new(
                "tensor_value",
                "SMEM slice write value count mismatch",
            ));
        }
        match values {
            ValueArray1::Bool(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &[u8::from(a[i])])?;
                }
            }
            ValueArray1::I8(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &[a[i] as u8])?;
                }
            }
            ValueArray1::U8(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &[a[i]])?;
                }
            }
            ValueArray1::I16(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_le_bytes())?;
                }
            }
            ValueArray1::U16(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_le_bytes())?;
                }
            }
            ValueArray1::I32(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_le_bytes())?;
                }
            }
            ValueArray1::U32(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_le_bytes())?;
                }
            }
            ValueArray1::I64(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_le_bytes())?;
                }
            }
            ValueArray1::U64(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_le_bytes())?;
                }
            }
            ValueArray1::F8E4M3(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &[encode_e4m3(a[i])])?;
                }
            }
            ValueArray1::F16(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &f16::from_f32(a[i]).to_bits().to_le_bytes())?;
                }
            }
            ValueArray1::Bf16(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    let bits = (round_bf16_scalar(a[i]).to_bits() >> 16) as u16;
                    self.write_bytes(tensor, flat, &bits.to_le_bytes())?;
                }
            }
            ValueArray1::F32(a) => {
                for (i, &flat) in indices.iter().enumerate() {
                    self.write_bytes(tensor, flat, &a[i].to_bits().to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    pub fn invalidate_indices(&mut self, tensor: &Tensor, indices: &[usize]) -> IResult<()> {
        for &flat in indices {
            let range = self.byte_range(tensor, flat)?;
            self.valid[range].fill(false);
        }
        Ok(())
    }

    pub fn invalidate_block(
        &mut self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<()> {
        // Symmetric with `write_block`: a rectangular slice clears its valid mask
        // one contiguous run at a time (shared `rect_row_ranges`). Without this the
        // tile falls to the per-element `block_indices`/`invalidate_indices` route
        // (one `byte_range` + `fill` per element), which dominates trace-mode TMA.
        if let Some((ranges, _)) = self.rect_row_ranges(tensor, offsets, slice_shape)? {
            for range in ranges {
                self.valid[range].fill(false);
            }
            return Ok(());
        }
        let indices = block_indices(tensor, offsets, slice_shape)?;
        self.invalidate_indices(tensor, &indices)
    }

    /// Decompose a rank-N rectangular SMEM slice into maximal contiguous byte
    /// runs, in row-major order. Trailing dims the box covers fully fold into
    /// one run (a full-width rank-2 tile or a (1, rows, k) staged box with a
    /// fully-covered tail is a SINGLE run); the remaining outer dims are walked
    /// with an odometer, one run each. `None` only when the slice rank does not
    /// match the tensor (the caller takes the per-element path). Shared by
    /// `write_block` (copy bytes + mark valid) and `invalidate_block` (clear
    /// valid): the difference is the per-range action, not the ranges.
    fn rect_row_ranges(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<Option<(Vec<std::ops::Range<usize>>, usize)>> {
        let rank = tensor.shape.len();
        if rank == 0 || offsets.len() != rank || slice_shape.len() != rank {
            return Ok(None);
        }
        check_block_bounds(tensor, offsets, slice_shape)?;
        if slice_shape.iter().any(|&d| d == 0) {
            return Ok(Some((Vec::new(), 0)));
        }
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * tensor.shape[i + 1];
        }
        // dims after `split` are fully covered: they fold into one run with dim
        // `split` itself (bounds force their offsets to 0).
        let mut split = rank - 1;
        while split > 0 && slice_shape[split] == tensor.shape[split] {
            split -= 1;
        }
        let run_elems = slice_shape[split] * strides[split];
        let corner: usize = offsets.iter().zip(&strides).map(|(&o, &st)| o * st).sum();
        let outer = &slice_shape[..split];
        let n_runs: usize = outer.iter().product();
        let mut ranges = Vec::with_capacity(n_runs);
        let mut idx = vec![0usize; split];
        'outer: loop {
            let rel: usize = idx
                .iter()
                .zip(&strides[..split])
                .map(|(&i, &st)| i * st)
                .sum();
            ranges.push(self.row_byte_range(tensor, corner + rel, run_elems)?);
            let mut d = split;
            loop {
                if d == 0 {
                    break 'outer;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < outer[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Some((ranges, run_elems)))
    }

    fn write_bytes(&mut self, tensor: &Tensor, flat: usize, src: &[u8]) -> IResult<()> {
        let range = self.byte_range(tensor, flat)?;
        debug_assert_eq!(range.len(), src.len());
        self.bytes[range.clone()].copy_from_slice(src);
        self.valid[range].fill(true);
        Ok(())
    }

    pub fn read_block(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<ValueArray1> {
        if let Some(values) = self.try_read_rect_runs(tensor, offsets, slice_shape)? {
            return Ok(values);
        }
        let indices = block_indices(tensor, offsets, slice_shape)?;
        self.read_indices(tensor, &indices)
    }

    pub fn write_block(
        &mut self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
        values: &ValueArray1,
    ) -> IResult<()> {
        if self.try_write_rect_runs(tensor, offsets, slice_shape, values)? {
            return Ok(());
        }
        let indices = block_indices(tensor, offsets, slice_shape)?;
        self.write_indices(tensor, &indices, values)
    }

    pub fn append_f32_block(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
        out: &mut Vec<f32>,
    ) -> IResult<()> {
        if !matches!(
            tensor.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F8E4M3
        ) {
            return Err(InterpreterError::new(
                "tcgen05_mma_dtype",
                "mma SMEM operand must be float",
            ));
        }
        if self.try_append_f32_rect_runs(tensor, offsets, slice_shape, out)? {
            return Ok(());
        }
        let indices = block_indices(tensor, offsets, slice_shape)?;
        out.reserve(indices.len());
        match tensor.dtype {
            DType::F16 => {
                for flat in indices {
                    let b = self.read_bytes::<2>(tensor, flat)?;
                    out.push(f16::from_bits(u16::from_le_bytes(b)).to_f32());
                }
            }
            DType::Bf16 => {
                for flat in indices {
                    let b = self.read_bytes::<2>(tensor, flat)?;
                    out.push(f32::from_bits((u16::from_le_bytes(b) as u32) << 16));
                }
            }
            DType::F32 => {
                for flat in indices {
                    let b = self.read_bytes::<4>(tensor, flat)?;
                    out.push(f32::from_bits(u32::from_le_bytes(b)));
                }
            }
            DType::F8E4M3 => {
                for flat in indices {
                    let b = self.read_bytes::<1>(tensor, flat)?;
                    out.push(decode_e4m3(b[0]));
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Bulk-materialize a rectangular slice into `out` as f32, one contiguous
    /// run at a time (rank-N: trailing fully-covered dims fold into the run).
    /// Validates the whole operand ONCE up front, then the copy loop runs
    /// check-free, decoding straight into reserved capacity.
    fn try_append_f32_rect_runs(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
        out: &mut Vec<f32>,
    ) -> IResult<bool> {
        let Some((ranges, cols)) = self.rect_row_ranges(tensor, offsets, slice_shape)? else {
            return Ok(false);
        };
        if cols == 0 {
            return Ok(true);
        }
        for range in &ranges {
            self.check_valid(range.clone())?;
        }
        out.reserve(ranges.len() * cols);
        for range in &ranges {
            let range = range.clone();
            let old_len = out.len();
            // SAFETY: every element of `dst` is written by the decode below,
            // then `set_len` commits the run.
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().add(old_len), cols) };
            match tensor.dtype {
                DType::F16 => {
                    if let Some(bits) = cast_bytes::<u16>(&self.bytes[range.clone()]) {
                        bits.reinterpret_cast::<f16>().convert_to_f32_slice(dst);
                    } else {
                        for (j, d) in dst.iter_mut().enumerate() {
                            let b0 = self.bytes[range.start + 2 * j];
                            let b1 = self.bytes[range.start + 2 * j + 1];
                            *d = f16::from_bits(u16::from_le_bytes([b0, b1])).to_f32();
                        }
                    }
                }
                DType::Bf16 => {
                    if let Some(bits) = cast_bytes::<u16>(&self.bytes[range.clone()]) {
                        bits.reinterpret_cast::<bf16>().convert_to_f32_slice(dst);
                    } else {
                        for (j, d) in dst.iter_mut().enumerate() {
                            let b0 = self.bytes[range.start + 2 * j];
                            let b1 = self.bytes[range.start + 2 * j + 1];
                            *d = f32::from_bits((u16::from_le_bytes([b0, b1]) as u32) << 16);
                        }
                    }
                }
                DType::F32 => {
                    if let Some(src) = cast_bytes::<f32>(&self.bytes[range.clone()]) {
                        dst.copy_from_slice(src);
                    } else {
                        for (j, d) in dst.iter_mut().enumerate() {
                            let st = range.start + 4 * j;
                            *d = f32::from_bits(u32::from_le_bytes([
                                self.bytes[st],
                                self.bytes[st + 1],
                                self.bytes[st + 2],
                                self.bytes[st + 3],
                            ]));
                        }
                    }
                }
                DType::F8E4M3 => {
                    for (d, &b) in dst.iter_mut().zip(self.bytes[range.clone()].iter()) {
                        *d = decode_e4m3(b);
                    }
                }
                _ => unreachable!(),
            }
            unsafe { out.set_len(old_len + cols) };
        }
        Ok(true)
    }

    fn try_write_rect_runs(
        &mut self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
        values: &ValueArray1,
    ) -> IResult<bool> {
        if values.dtype() != tensor.dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "SMEM write dtype must match the destination tensor",
            ));
        }
        if values.len() != numel(slice_shape) {
            return Err(InterpreterError::new(
                "tensor_value",
                "SMEM slice write value count mismatch",
            ));
        }
        // The same contiguous runs `invalidate_block` walks; here we additionally
        // copy each run's bytes and mark it valid — the difference from clearing is
        // the per-range action, not how the ranges are computed.
        let Some((ranges, cols)) = self.rect_row_ranges(tensor, offsets, slice_shape)? else {
            return Ok(false);
        };
        match values {
            ValueArray1::F16(a) => {
                let src_all = a.as_slice().expect("contiguous f16 backing");
                for (r, range) in ranges.iter().enumerate() {
                    let range = range.clone();
                    let src = &src_all[r * cols..(r + 1) * cols];
                    if let Some(bits) = cast_bytes_mut::<u16>(&mut self.bytes[range.clone()]) {
                        bits.reinterpret_cast_mut::<f16>()
                            .convert_from_f32_slice(src);
                    } else {
                        for (j, &v) in src.iter().enumerate() {
                            let bits = f16::from_f32(v).to_bits().to_le_bytes();
                            let s = range.start + 2 * j;
                            self.bytes[s..s + 2].copy_from_slice(&bits);
                        }
                    }
                    self.valid[range].fill(true);
                }
                Ok(true)
            }
            ValueArray1::Bf16(a) => {
                let src_all = a.as_slice().expect("contiguous bf16 backing");
                for (r, range) in ranges.iter().enumerate() {
                    let range = range.clone();
                    let src = &src_all[r * cols..(r + 1) * cols];
                    if let Some(bits) = cast_bytes_mut::<u16>(&mut self.bytes[range.clone()]) {
                        bits.reinterpret_cast_mut::<bf16>()
                            .convert_from_f32_slice(src);
                    } else {
                        for (j, &v) in src.iter().enumerate() {
                            let bits =
                                ((round_bf16_scalar(v).to_bits() >> 16) as u16).to_le_bytes();
                            let s = range.start + 2 * j;
                            self.bytes[s..s + 2].copy_from_slice(&bits);
                        }
                    }
                    self.valid[range].fill(true);
                }
                Ok(true)
            }
            ValueArray1::F32(a) => {
                let src_all = a.as_slice().expect("contiguous f32 backing");
                for (r, range) in ranges.iter().enumerate() {
                    let range = range.clone();
                    let src = &src_all[r * cols..(r + 1) * cols];
                    if let Some(dst) = cast_bytes_mut::<f32>(&mut self.bytes[range.clone()]) {
                        dst.copy_from_slice(src);
                    } else {
                        for (j, &v) in src.iter().enumerate() {
                            let bits = v.to_bits().to_le_bytes();
                            let s = range.start + 4 * j;
                            self.bytes[s..s + 4].copy_from_slice(&bits);
                        }
                    }
                    self.valid[range].fill(true);
                }
                Ok(true)
            }
            ValueArray1::F8E4M3(a) => {
                let src_all = a.as_slice().expect("contiguous f8 backing");
                for (r, range) in ranges.iter().enumerate() {
                    let range = range.clone();
                    let src = &src_all[r * cols..(r + 1) * cols];
                    let dst = &mut self.bytes[range.clone()];
                    for (d, &v) in dst.iter_mut().zip(src.iter()) {
                        *d = encode_e4m3(v);
                    }
                    self.valid[range].fill(true);
                }
                Ok(true)
            }
            ValueArray1::U32(a) => {
                let src_all = a.as_slice().expect("contiguous u32 backing");
                for (r, range) in ranges.iter().enumerate() {
                    let range = range.clone();
                    let src = &src_all[r * cols..(r + 1) * cols];
                    for (j, &v) in src.iter().enumerate() {
                        let st = range.start + 4 * j;
                        self.bytes[st..st + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    self.valid[range].fill(true);
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Bulk rectangular read: one contiguous run at a time (the same runs the
    /// write/invalidate paths walk), with a per-run validity check and a tight
    /// per-run decode loop — never a per-element flat-index/byte-range trip.
    /// `None` when the slice rank mismatches or the dtype has no bulk arm.
    fn try_read_rect_runs(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<Option<ValueArray1>> {
        if !matches!(
            tensor.dtype,
            DType::F8E4M3 | DType::U32 | DType::F16 | DType::Bf16 | DType::F32
        ) {
            return Ok(None);
        }
        let Some((ranges, _)) = self.rect_row_ranges(tensor, offsets, slice_shape)? else {
            return Ok(None);
        };
        let k = numel(slice_shape);
        for range in &ranges {
            self.check_valid(range.clone())?;
        }
        match tensor.dtype {
            DType::F8E4M3 => {
                let mut out = Vec::with_capacity(k);
                for range in &ranges {
                    out.extend(self.bytes[range.clone()].iter().map(|&b| decode_e4m3(b)));
                }
                Ok(Some(ValueArray1::F8E4M3(Array1::from(out))))
            }
            DType::U32 => {
                let mut out = Vec::with_capacity(k);
                for range in &ranges {
                    for chunk in self.bytes[range.clone()].chunks_exact(4) {
                        out.push(u32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                }
                Ok(Some(ValueArray1::U32(Array1::from(out))))
            }
            DType::F16 => {
                let mut out = Vec::with_capacity(k);
                for range in &ranges {
                    for chunk in self.bytes[range.clone()].chunks_exact(2) {
                        out.push(
                            f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32(),
                        );
                    }
                }
                Ok(Some(ValueArray1::F16(Array1::from(out))))
            }
            DType::Bf16 => {
                let mut out = Vec::with_capacity(k);
                for range in &ranges {
                    for chunk in self.bytes[range.clone()].chunks_exact(2) {
                        out.push(f32::from_bits(
                            (u16::from_le_bytes(chunk.try_into().unwrap()) as u32) << 16,
                        ));
                    }
                }
                Ok(Some(ValueArray1::Bf16(Array1::from(out))))
            }
            DType::F32 => {
                let mut out = Vec::with_capacity(k);
                for range in &ranges {
                    for chunk in self.bytes[range.clone()].chunks_exact(4) {
                        out.push(f32::from_bits(u32::from_le_bytes(
                            chunk.try_into().unwrap(),
                        )));
                    }
                }
                Ok(Some(ValueArray1::F32(Array1::from(out))))
            }
            _ => unreachable!(),
        }
    }

    fn row_byte_range(&self, tensor: &Tensor, flat: usize, cols: usize) -> IResult<Range<usize>> {
        let start = self.byte_range(tensor, flat)?.start;
        let end = start
            .checked_add(cols * dtype_size_bytes(tensor.dtype))
            .ok_or_else(|| InterpreterError::new("smem_value", "SMEM byte range overflows"))?;
        if end > self.bytes.len() {
            return Err(InterpreterError::new(
                "smem_oob",
                "SMEM tensor byte range exceeds the CTA pool",
            ));
        }
        Ok(start..end)
    }
}

fn block_indices(tensor: &Tensor, offsets: &[usize], slice_shape: &[usize]) -> IResult<Vec<usize>> {
    check_block_bounds(tensor, offsets, slice_shape)?;
    let k = numel(slice_shape);
    if k == 0 {
        return Ok(Vec::new());
    }
    let strides = row_major_strides(&tensor.shape);
    let base = flat_with_strides(offsets, &strides);
    let rank = slice_shape.len();
    let inner = slice_shape[rank - 1];
    let run_starts = rect_intra_offsets(&slice_shape[..rank - 1], &strides[..rank - 1]);
    let mut indices = Vec::with_capacity(k);
    for &rs in &run_starts {
        let start = base + rs;
        indices.extend(start..start + inner);
    }
    Ok(indices)
}

fn check_block_bounds(tensor: &Tensor, offsets: &[usize], slice_shape: &[usize]) -> IResult<()> {
    if offsets.len() != tensor.shape.len() || slice_shape.len() != tensor.shape.len() {
        return Err(InterpreterError::new(
            "tensor_value",
            "tensor slice rank mismatch",
        ));
    }
    for ((o, e), d) in offsets
        .iter()
        .zip(slice_shape.iter())
        .zip(tensor.shape.iter())
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

fn cast_bytes<T>(bytes: &[u8]) -> Option<&[T]> {
    let (prefix, middle, suffix) = unsafe { bytes.align_to::<T>() };
    if prefix.is_empty() && suffix.is_empty() {
        Some(middle)
    } else {
        None
    }
}

fn cast_bytes_mut<T>(bytes: &mut [u8]) -> Option<&mut [T]> {
    let (prefix, middle, suffix) = unsafe { bytes.align_to_mut::<T>() };
    if prefix.is_empty() && suffix.is_empty() {
        Some(middle)
    } else {
        None
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmemValues {
    pub by_cta: HashMap<usize, SmemScratchpad>,
}

impl SmemValues {
    pub fn pool_for(&self, cta_id: usize) -> IResult<&SmemScratchpad> {
        self.by_cta.get(&cta_id).ok_or_else(|| {
            InterpreterError::new("missing_tensor_value", "CTA SMEM pool has not been written")
        })
    }

    pub fn pool_for_mut(&mut self, cta_id: usize, size_bytes: usize) -> &mut SmemScratchpad {
        self.by_cta
            .entry(cta_id)
            .or_insert_with(|| SmemScratchpad::new(size_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Tensor;

    fn smem_tensor(id: u32, dtype: DType, shape: Vec<usize>, byte_offset: usize) -> Tensor {
        Tensor {
            id,
            space: MemorySpace::Smem,
            dtype,
            shape,
            layout: None,
            byte_offset: Some(byte_offset),
        }
    }

    #[test]
    fn overlapping_smem_views_reinterpret_bytes() {
        let u = smem_tensor(1, DType::U32, vec![1], 0);
        let f = smem_tensor(2, DType::F32, vec![1], 0);
        let mut pool = SmemScratchpad::new(4);
        pool.write_indices(&u, &[0], &ValueArray1::U32(Array1::from(vec![0x3f80_0000])))
            .unwrap();
        let read = pool.read_indices(&f, &[0]).unwrap().to_f32_compute();
        assert_eq!(read.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn smem_reads_fail_on_unwritten_bytes() {
        let t = smem_tensor(1, DType::U32, vec![1], 0);
        let pool = SmemScratchpad::new(4);
        let err = pool.read_indices(&t, &[0]).unwrap_err();
        assert_eq!(err.code, "missing_tensor_value");
    }
}
