//! CTA-local TMEM scratchpad — port of `interpreter/values/tmem.py`.
//!
//! A 128-lane × 512-column grid of physical 32-bit cells. Storage is lane-major:
//! `data[[lane, col]]` with shape `[TMEM_ROWS, TMEM_COLS]`, matching the Python
//! value model. Logical F32/I32/U32 TMEM values are bit-reinterpreted through
//! those cells. F16/BF16 logical values are packed two per physical cell, with
//! even logical columns in the low half and odd logical columns in the high half.

use super::super::diagnostics::{IResult, InterpreterError};
use super::arrays::ValueArray1;
use super::dtypes::round_bf16_scalar;
use super::indexing::slice_coords;
use crate::ir::{DType, Layout, MemorySpace, Tensor, TmemLayout, TmemLayoutKind};
use half::f16;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub const TMEM_ROWS: usize = 128; // lanes
pub const TMEM_COLS: usize = 512; // columns

/// (lane, col).
pub type TmemCellKey = (usize, usize);

#[derive(Clone, Copy, Debug)]
struct TmemHalfRef {
    lane: usize,
    col: usize,
    high: bool,
}

#[derive(Clone, Debug)]
pub struct TmemScratchpad {
    /// lane-major physical cell bits: `data[[lane, col]]`.
    pub data: Array2<u32>,
    pub valid: Array2<bool>,
}

impl Default for TmemScratchpad {
    fn default() -> Self {
        TmemScratchpad {
            data: Array2::zeros((TMEM_ROWS, TMEM_COLS)),
            valid: Array2::from_elem((TMEM_ROWS, TMEM_COLS), false),
        }
    }
}

impl TmemScratchpad {
    /// A temporary f32 view over the physical u32 cells for BLAS accumulator writes.
    ///
    /// TMEM stores f32 accumulator values as raw IEEE bits. The view is only used
    /// while cblas writes the result and no u32 view is accessed concurrently.
    pub fn data_as_f32_mut(&mut self) -> &mut [f32] {
        let data = self.data.as_slice_mut().expect("tmem grid is contiguous");
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<f32>(), data.len()) }
    }

    fn check_cell(&self, lane: usize, col: usize) -> IResult<()> {
        if lane >= TMEM_ROWS || col >= TMEM_COLS {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell is out of bounds",
            ));
        }
        Ok(())
    }

    fn check_supported_dtype(tensor: &Tensor) -> IResult<()> {
        if !matches!(
            tensor.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::I32 | DType::U32
        ) {
            return Err(InterpreterError::new(
                "tmem_value",
                "unsupported TMEM cell dtype",
            ));
        }
        Ok(())
    }

    /// Arbitrary (lane,col) gather. Fails closed on unwritten or unsupported dtype.
    pub fn read_cells(
        &self,
        tensor: &Tensor,
        lanes: &[usize],
        cols: &[usize],
    ) -> IResult<ValueArray1> {
        Self::check_supported_dtype(tensor)?;
        if lanes.len() != cols.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell gather index count mismatch",
            ));
        }
        let n = lanes.len();
        match tensor.dtype {
            DType::F32 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(f32::from_bits(self.data[[l, c]]));
                }
                Ok(ValueArray1::F32(Array1::from(out)))
            }
            DType::I32 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(self.data[[l, c]] as i32);
                }
                Ok(ValueArray1::I32(Array1::from(out)))
            }
            DType::U32 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(self.data[[l, c]]);
                }
                Ok(ValueArray1::U32(Array1::from(out)))
            }
            DType::F16 | DType::Bf16 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(decode_half(
                        tensor.dtype,
                        (self.data[[l, c]] & 0xffff) as u16,
                    ));
                }
                Ok(ValueArray1::from_f32_compute(
                    Array1::from(out),
                    tensor.dtype,
                ))
            }
            _ => unreachable!(),
        }
    }

    /// Inverse scatter. Total (caller pre-validates bounds/uniqueness).
    pub fn write_cells(
        &mut self,
        tensor: &Tensor,
        lanes: &[usize],
        cols: &[usize],
        values: &ValueArray1,
    ) -> IResult<()> {
        Self::check_supported_dtype(tensor)?;
        if lanes.len() != cols.len() || lanes.len() != values.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell scatter value count mismatch",
            ));
        }
        if values.dtype() != tensor.dtype {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM write dtype must match tensor dtype",
            ));
        }
        let half_values = if is_packed_half_dtype(tensor.dtype) {
            Some(values.to_f32_compute())
        } else {
            None
        };
        for i in 0..lanes.len() {
            let (l, c) = (lanes[i], cols[i]);
            self.check_cell(l, c)?;
            self.data[[l, c]] = match tensor.dtype {
                DType::F32 => match values {
                    ValueArray1::F32(a) => a[i].to_bits(),
                    _ => unreachable!(),
                },
                DType::I32 => match values {
                    ValueArray1::I32(a) => a[i] as u32,
                    _ => unreachable!(),
                },
                DType::U32 => match values {
                    ValueArray1::U32(a) => a[i],
                    _ => unreachable!(),
                },
                DType::F16 | DType::Bf16 => {
                    let old = if self.valid[[l, c]] {
                        self.data[[l, c]]
                    } else {
                        0
                    };
                    (old & 0xffff_0000)
                        | u32::from(encode_half(tensor.dtype, half_values.as_ref().unwrap()[i]))
                }
                _ => unreachable!(),
            };
            self.valid[[l, c]] = true;
        }
        Ok(())
    }

    /// Rectangular block read [l0:l1, c0:c1] → row-major logical values.
    pub fn read_cell_block(
        &self,
        tensor: &Tensor,
        l0: usize,
        l1: usize,
        c0: usize,
        c1: usize,
    ) -> IResult<ValueArray1> {
        let mut lanes = Vec::with_capacity((l1 - l0) * (c1 - c0));
        let mut cols = Vec::with_capacity((l1 - l0) * (c1 - c0));
        for l in l0..l1 {
            for c in c0..c1 {
                lanes.push(l);
                cols.push(c);
            }
        }
        self.read_cells(tensor, &lanes, &cols)
    }

    /// Rectangular block write (values lane-major). Total.
    pub fn write_cell_block(
        &mut self,
        tensor: &Tensor,
        l0: usize,
        l1: usize,
        c0: usize,
        c1: usize,
        values: &ValueArray1,
    ) -> IResult<()> {
        let mut lanes = Vec::with_capacity((l1 - l0) * (c1 - c0));
        let mut cols = Vec::with_capacity((l1 - l0) * (c1 - c0));
        for l in l0..l1 {
            for c in c0..c1 {
                lanes.push(l);
                cols.push(c);
            }
        }
        self.write_cells(tensor, &lanes, &cols, values)
    }

    pub fn clear_columns(&mut self, col_start: usize, n_cols: usize) {
        let end = (col_start + n_cols).min(TMEM_COLS);
        self.data.slice_mut(ndarray::s![.., col_start..end]).fill(0);
        self.valid
            .slice_mut(ndarray::s![.., col_start..end])
            .fill(false);
    }

    pub fn read_slice(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<ValueArray1> {
        if is_packed_half_dtype(tensor.dtype) {
            if let Some(values) = self.try_read_packed_half_rect(tensor, offsets, slice_shape)? {
                return Ok(values);
            }
            let refs = tmem_slice_half_refs(tensor, offsets, slice_shape)?;
            return self.read_half_refs(tensor, &refs);
        }
        let cells = tmem_slice_cells(tensor, offsets, slice_shape)?;
        let lanes: Vec<usize> = cells.iter().map(|c| c.0).collect();
        let cols: Vec<usize> = cells.iter().map(|c| c.1).collect();
        self.read_cells(tensor, &lanes, &cols)
    }

    pub fn write_slice(
        &mut self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
        values: &ValueArray1,
    ) -> IResult<()> {
        if is_packed_half_dtype(tensor.dtype) {
            let refs = tmem_slice_half_refs(tensor, offsets, slice_shape)?;
            if values.len() != refs.len() {
                return Err(InterpreterError::new(
                    "tmem_value",
                    "TMEM slice write value count mismatch",
                ));
            }
            return self.write_half_refs(tensor, &refs, values);
        }
        let cells = tmem_slice_cells(tensor, offsets, slice_shape)?;
        if values.len() != cells.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM slice write value count mismatch",
            ));
        }
        let lanes: Vec<usize> = cells.iter().map(|c| c.0).collect();
        let cols: Vec<usize> = cells.iter().map(|c| c.1).collect();
        self.write_cells(tensor, &lanes, &cols, values)
    }

    fn try_read_packed_half_rect(
        &self,
        tensor: &Tensor,
        offsets: &[usize],
        slice_shape: &[usize],
    ) -> IResult<Option<ValueArray1>> {
        check_tmem_slice_bounds(tensor, offsets, slice_shape)?;
        let layout = tmem_layout_for(tensor)?;
        if layout.kind != TmemLayoutKind::Lane128 || offsets[1] % 2 != 0 || slice_shape[1] % 2 != 0
        {
            return Ok(None);
        }
        let row0 = offsets[0];
        let rows = slice_shape[0];
        let col0 = layout
            .col_start
            .checked_add(offsets[1] / 2)
            .ok_or_else(|| InterpreterError::new("tmem_value", "TMEM half coordinate overflow"))?;
        let cell_cols = slice_shape[1] / 2;
        let col1 = col0
            .checked_add(cell_cols)
            .ok_or_else(|| InterpreterError::new("tmem_value", "TMEM half coordinate overflow"))?;
        if row0 + rows > TMEM_ROWS || col1 > TMEM_COLS {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell is out of bounds",
            ));
        }
        let mut out = Vec::with_capacity(rows * slice_shape[1]);
        for row in row0..row0 + rows {
            for col in col0..col1 {
                if !self.valid[[row, col]] {
                    return Err(InterpreterError::new(
                        "missing_tmem_value",
                        "TMEM cell is unwritten",
                    ));
                }
                let cell = self.data[[row, col]];
                out.push(decode_half(tensor.dtype, (cell & 0xffff) as u16));
                out.push(decode_half(tensor.dtype, (cell >> 16) as u16));
            }
        }
        Ok(Some(ValueArray1::from_f32_compute(
            Array1::from(out),
            tensor.dtype,
        )))
    }

    pub fn accumulate_f32_cell_block_from(
        &mut self,
        tensor: &Tensor,
        lane_start: usize,
        rows: usize,
        col_start: usize,
        cols: usize,
        src: &Array2<f32>,
        src_row: usize,
        src_col: usize,
        accum: bool,
    ) -> IResult<bool> {
        if tensor.dtype != DType::F32 {
            return Ok(false);
        }
        Self::check_supported_dtype(tensor)?;
        if lane_start + rows > TMEM_ROWS || col_start + cols > TMEM_COLS {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell is out of bounds",
            ));
        }
        if src_row + rows > src.nrows() || src_col + cols > src.ncols() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM source block is out of bounds",
            ));
        }
        if accum {
            for lane in lane_start..lane_start + rows {
                for col in col_start..col_start + cols {
                    if !self.valid[[lane, col]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                }
            }
        }
        {
            let grid = self.data_as_f32_mut();
            for i in 0..rows {
                let dst_base = (lane_start + i) * TMEM_COLS + col_start;
                for j in 0..cols {
                    let dst = &mut grid[dst_base + j];
                    let value = src[[src_row + i, src_col + j]];
                    if accum {
                        *dst += value;
                    } else {
                        *dst = value;
                    }
                }
            }
        }
        self.valid
            .slice_mut(ndarray::s![
                lane_start..lane_start + rows,
                col_start..col_start + cols
            ])
            .fill(true);
        Ok(true)
    }

    pub fn write_packed_half_cells(
        &mut self,
        tensor: &Tensor,
        lanes: &[usize],
        cols: &[usize],
        values: &[(f32, f32)],
    ) -> IResult<()> {
        Self::check_supported_dtype(tensor)?;
        if !is_packed_half_dtype(tensor.dtype) {
            return Err(InterpreterError::new(
                "tmem_value",
                "packed TMEM cell write requires f16 or bf16 tensor dtype",
            ));
        }
        if lanes.len() != cols.len() || lanes.len() != values.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM packed cell scatter value count mismatch",
            ));
        }
        for i in 0..lanes.len() {
            let (l, c) = (lanes[i], cols[i]);
            self.check_cell(l, c)?;
            let lo = u32::from(encode_half(tensor.dtype, values[i].0));
            let hi = u32::from(encode_half(tensor.dtype, values[i].1));
            self.data[[l, c]] = lo | (hi << 16);
            self.valid[[l, c]] = true;
        }
        Ok(())
    }

    pub fn read_packed_half_cells(
        &self,
        tensor: &Tensor,
        lanes: &[usize],
        cols: &[usize],
    ) -> IResult<Vec<(f32, f32)>> {
        Self::check_supported_dtype(tensor)?;
        if !is_packed_half_dtype(tensor.dtype) {
            return Err(InterpreterError::new(
                "tmem_value",
                "packed TMEM cell read requires f16 or bf16 tensor dtype",
            ));
        }
        if lanes.len() != cols.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM packed cell gather index count mismatch",
            ));
        }
        let mut out = Vec::with_capacity(lanes.len());
        for i in 0..lanes.len() {
            let (l, c) = (lanes[i], cols[i]);
            self.check_cell(l, c)?;
            if !self.valid[[l, c]] {
                return Err(InterpreterError::new(
                    "missing_tmem_value",
                    "TMEM cell is unwritten",
                ));
            }
            let cell = self.data[[l, c]];
            out.push((
                decode_half(tensor.dtype, (cell & 0xffff) as u16),
                decode_half(tensor.dtype, (cell >> 16) as u16),
            ));
        }
        Ok(out)
    }

    fn read_half_refs(&self, tensor: &Tensor, refs: &[TmemHalfRef]) -> IResult<ValueArray1> {
        let mut out = Vec::with_capacity(refs.len());
        for r in refs {
            self.check_cell(r.lane, r.col)?;
            if !self.valid[[r.lane, r.col]] {
                return Err(InterpreterError::new(
                    "missing_tmem_value",
                    "TMEM cell is unwritten",
                ));
            }
            let cell = self.data[[r.lane, r.col]];
            let bits = if r.high {
                (cell >> 16) as u16
            } else {
                (cell & 0xffff) as u16
            };
            out.push(decode_half(tensor.dtype, bits));
        }
        Ok(ValueArray1::from_f32_compute(
            Array1::from(out),
            tensor.dtype,
        ))
    }

    fn write_half_refs(
        &mut self,
        tensor: &Tensor,
        refs: &[TmemHalfRef],
        values: &ValueArray1,
    ) -> IResult<()> {
        if values.dtype() != tensor.dtype {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM write dtype must match tensor dtype",
            ));
        }
        let f = values.to_f32_compute();
        for (i, r) in refs.iter().enumerate() {
            self.check_cell(r.lane, r.col)?;
            let old = if self.valid[[r.lane, r.col]] {
                self.data[[r.lane, r.col]]
            } else {
                0
            };
            let bits = u32::from(encode_half(tensor.dtype, f[i]));
            self.data[[r.lane, r.col]] = if r.high {
                (old & 0x0000_ffff) | (bits << 16)
            } else {
                (old & 0xffff_0000) | bits
            };
            self.valid[[r.lane, r.col]] = true;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct TmemValues {
    pub by_cta: HashMap<usize, TmemScratchpad>,
}

impl TmemValues {
    pub fn scratchpad_for(&self, cta_id: usize) -> IResult<&TmemScratchpad> {
        self.by_cta.get(&cta_id).ok_or_else(|| {
            InterpreterError::new(
                "missing_tmem_scratchpad",
                "CTA TMEM scratchpad is not allocated",
            )
        })
    }
}

/// The TMEM layout of a tensor (default LANE_128 @ col_start 0).
pub fn tmem_layout_for(tensor: &Tensor) -> IResult<TmemLayout> {
    if tensor.space != MemorySpace::Tmem {
        return Err(InterpreterError::new(
            "tmem_value",
            "TMEM scratchpad access requires a TMEM tensor",
        ));
    }
    match &tensor.layout {
        None => Ok(TmemLayout {
            kind: TmemLayoutKind::Lane128,
            col_start: 0,
            lane_align: 0,
        }),
        Some(Layout::Tmem(tm)) => Ok(*tm),
        Some(_) => Err(InterpreterError::new(
            "tmem_layout",
            "TMEM tensor layout must be a TmemLayout",
        )),
    }
}

pub fn tmem_physical_range(tensor: &Tensor, n_cols: usize) -> IResult<(usize, usize)> {
    let layout = tmem_layout_for(tensor)?;
    Ok((layout.col_start, n_cols))
}

fn is_packed_half_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::Bf16)
}

fn encode_half(dtype: DType, value: f32) -> u16 {
    match dtype {
        DType::F16 => f16::from_f32(value).to_bits(),
        DType::Bf16 => (round_bf16_scalar(value).to_bits() >> 16) as u16,
        _ => unreachable!("encode_half on non-half dtype"),
    }
}

fn decode_half(dtype: DType, bits: u16) -> f32 {
    match dtype {
        DType::F16 => f16::from_bits(bits).to_f32(),
        DType::Bf16 => f32::from_bits(u32::from(bits) << 16),
        _ => unreachable!("decode_half on non-half dtype"),
    }
}

fn tmem_lane_for(layout: &TmemLayout, row: usize) -> IResult<usize> {
    match layout.kind {
        TmemLayoutKind::Lane128 => {
            if row >= 128 {
                return Err(InterpreterError::new(
                    "invalid_tmem_row",
                    "TMEM row out of range",
                ));
            }
            Ok(row)
        }
        TmemLayoutKind::Lane64Upper => {
            if row >= 64 {
                return Err(InterpreterError::new(
                    "invalid_tmem_row",
                    "TMEM row out of range",
                ));
            }
            Ok(row)
        }
        TmemLayoutKind::Lane64Lower => {
            if row >= 64 {
                return Err(InterpreterError::new(
                    "invalid_tmem_row",
                    "TMEM row out of range",
                ));
            }
            Ok(row + 64)
        }
        _ => Err(InterpreterError::new(
            "unsupported_tmem_layout",
            "scale-vector TMEM layouts are unsupported for value access",
        )),
    }
}

/// Map a logical (row, col) to a physical (lane, col).
pub fn tmem_cell_for(tensor: &Tensor, logical_coord: &[usize]) -> IResult<TmemCellKey> {
    if logical_coord.len() != 2 {
        return Err(InterpreterError::new(
            "tmem_value",
            "TMEM logical coord must be rank-2",
        ));
    }
    let layout = tmem_layout_for(tensor)?;
    let (row, logical_col) = (logical_coord[0], logical_coord[1]);
    let lane = tmem_lane_for(&layout, row)?;
    Ok((lane, layout.col_start + logical_col))
}

fn tmem_half_ref_for(tensor: &Tensor, logical_coord: &[usize]) -> IResult<TmemHalfRef> {
    if logical_coord.len() != 2 {
        return Err(InterpreterError::new(
            "tmem_value",
            "TMEM logical coord must be rank-2",
        ));
    }
    let layout = tmem_layout_for(tensor)?;
    let lane = tmem_lane_for(&layout, logical_coord[0])?;
    let logical_byte = logical_coord[1]
        .checked_mul(2)
        .ok_or_else(|| InterpreterError::new("tmem_value", "TMEM half coordinate overflow"))?;
    let lane_byte = layout
        .col_start
        .checked_mul(4)
        .and_then(|b| b.checked_add(logical_byte))
        .ok_or_else(|| InterpreterError::new("tmem_value", "TMEM half coordinate overflow"))?;
    Ok(TmemHalfRef {
        lane,
        col: lane_byte / 4,
        high: (lane_byte % 4) != 0,
    })
}

fn check_tmem_slice_bounds(
    tensor: &Tensor,
    offsets: &[usize],
    slice_shape: &[usize],
) -> IResult<()> {
    if tensor.shape.len() != 2 || offsets.len() != 2 || slice_shape.len() != 2 {
        return Err(InterpreterError::new(
            "tmem_value",
            "TMEM access must be rank-2",
        ));
    }
    for ((o, e), d) in offsets
        .iter()
        .zip(slice_shape.iter())
        .zip(tensor.shape.iter())
    {
        if o + e > *d {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM slice is out of bounds",
            ));
        }
    }
    Ok(())
}

pub fn tmem_slice_cells(
    tensor: &Tensor,
    offsets: &[usize],
    slice_shape: &[usize],
) -> IResult<Vec<TmemCellKey>> {
    check_tmem_slice_bounds(tensor, offsets, slice_shape)?;
    slice_coords(offsets, slice_shape)
        .iter()
        .map(|coord| tmem_cell_for(tensor, coord))
        .collect()
}

fn tmem_slice_half_refs(
    tensor: &Tensor,
    offsets: &[usize],
    slice_shape: &[usize],
) -> IResult<Vec<TmemHalfRef>> {
    check_tmem_slice_bounds(tensor, offsets, slice_shape)?;
    slice_coords(offsets, slice_shape)
        .iter()
        .map(|coord| tmem_half_ref_for(tensor, coord))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(id: u32, dtype: DType, col_start: usize) -> Tensor {
        Tensor {
            id,
            space: MemorySpace::Tmem,
            dtype,
            shape: vec![128, 8],
            layout: Some(Layout::Tmem(TmemLayout {
                kind: TmemLayoutKind::Lane128,
                col_start,
                lane_align: 0,
            })),
            byte_offset: None,
        }
    }

    #[test]
    fn tmem_cells_reinterpret_f32_i32_and_u32_bits() {
        let f = tensor(1, DType::F32, 0);
        let i = tensor(2, DType::I32, 0);
        let u = tensor(3, DType::U32, 0);
        let mut sp = TmemScratchpad::default();

        sp.write_cells(
            &u,
            &[7],
            &[11],
            &ValueArray1::U32(Array1::from(vec![0x3f80_0000])),
        )
        .unwrap();
        assert_eq!(
            sp.read_cells(&f, &[7], &[11])
                .unwrap()
                .to_f32_compute()
                .as_slice()
                .unwrap(),
            &[1.0]
        );

        sp.write_cells(&i, &[8], &[12], &ValueArray1::I32(Array1::from(vec![-1])))
            .unwrap();
        assert_eq!(
            sp.read_cells(&u, &[8], &[12])
                .unwrap()
                .to_i64_compute()
                .as_slice()
                .unwrap(),
            &[0xffff_ffff_i64]
        );
    }

    #[test]
    fn tmem_valid_mask_clear_and_packed_half_values() {
        let f = tensor(1, DType::F32, 0);
        let half = tensor(2, DType::F16, 0);
        let mut sp = TmemScratchpad::default();

        let missing = sp.read_cells(&f, &[0], &[0]).unwrap_err();
        assert_eq!(missing.code, "missing_tmem_value");

        sp.write_cells(&f, &[0], &[0], &ValueArray1::F32(Array1::from(vec![2.0])))
            .unwrap();
        assert!(sp.read_cells(&f, &[0], &[0]).is_ok());
        sp.clear_columns(0, 1);
        assert_eq!(
            sp.read_cells(&f, &[0], &[0]).unwrap_err().code,
            "missing_tmem_value"
        );

        sp.write_slice(
            &half,
            &[0, 0],
            &[1, 2],
            &ValueArray1::from_f32_compute(Array1::from(vec![1.0, 2.0]), DType::F16),
        )
        .unwrap();
        assert_eq!(sp.data[[0, 0]], 0x4000_3c00);
        assert_eq!(
            sp.read_slice(&half, &[0, 0], &[1, 2])
                .unwrap()
                .to_f32_compute()
                .as_slice()
                .unwrap(),
            &[1.0, 2.0]
        );

        sp.write_packed_half_cells(&half, &[1], &[3], &[(3.0, 4.0)])
            .unwrap();
        assert_eq!(
            sp.read_slice(&half, &[1, 6], &[1, 2])
                .unwrap()
                .to_f32_compute()
                .as_slice()
                .unwrap(),
            &[3.0, 4.0]
        );
    }

    #[test]
    fn packed_half_rect_read_preserves_row_major_logical_order() {
        let half = tensor(1, DType::F16, 3);
        let mut sp = TmemScratchpad::default();
        sp.write_slice(
            &half,
            &[1, 2],
            &[2, 4],
            &ValueArray1::from_f32_compute(
                Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                DType::F16,
            ),
        )
        .unwrap();

        let read = sp.read_slice(&half, &[1, 2], &[2, 4]).unwrap();
        assert_eq!(
            read.to_f32_compute().as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn f32_block_accumulate_updates_rect_and_valid_mask() {
        let f = tensor(1, DType::F32, 0);
        let mut sp = TmemScratchpad::default();
        let src = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(sp
            .accumulate_f32_cell_block_from(&f, 10, 2, 20, 3, &src, 0, 0, false)
            .unwrap());

        let read = sp
            .read_cells(&f, &[10, 10, 10, 11, 11, 11], &[20, 21, 22, 20, 21, 22])
            .unwrap();
        assert_eq!(
            read.to_f32_compute().as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        assert!(sp
            .accumulate_f32_cell_block_from(&f, 10, 2, 20, 3, &src, 0, 0, true)
            .unwrap());
        let read = sp
            .read_cells(&f, &[10, 10, 10, 11, 11, 11], &[20, 21, 22, 20, 21, 22])
            .unwrap();
        assert_eq!(
            read.to_f32_compute().as_slice().unwrap(),
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }
}
