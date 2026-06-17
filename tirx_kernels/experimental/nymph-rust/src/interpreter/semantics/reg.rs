//! Register ALU + load/store — port of `semantics/reg.py`, extended with the
//! small REG value language needed by FA-style kernels. Trace-only execution
//! still records only operand reads and destination writes.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::slice_indexing::uniform_contiguous_flat_range;
use super::super::transfer::{read_operand_array, read_transfer_array, write_operand};
use super::super::values::arrays::ValueArray2;
use super::super::values::dtypes::round_scalar;
use super::super::values::registers::{RegisterKey, RegisterTensorValue};
use crate::ir::{
    DType, MemorySpace, RegBinaryOp, RegCondScope, RegLiteral, RegOperand, RegReduceOp, RegUnaryOp,
    Rounding, Stmt, Tensor, TensorSlice,
};
use ndarray::{Array2, Zip};
use std::sync::Arc;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::RegFill, execute_reg_fill);
    reg.register(StmtKind::RegUnary, execute_reg_unary);
    reg.register(StmtKind::RegAdd, execute_reg_add);
    reg.register(StmtKind::RegSub, execute_reg_sub);
    reg.register(StmtKind::RegMul, execute_reg_mul);
    reg.register(StmtKind::RegMax, execute_reg_max);
    reg.register(StmtKind::RegMin, execute_reg_min);
    reg.register(StmtKind::RegFma, execute_reg_fma);
    reg.register(StmtKind::RegBitwise, execute_reg_bitwise);
    reg.register(StmtKind::RegReduce, execute_reg_reduce);
    reg.register(StmtKind::RegCondRescale, execute_reg_cond_rescale);
    reg.register(StmtKind::RegSoftmaxRescale, execute_reg_softmax_rescale);
    reg.register(StmtKind::RegCausalMask, execute_reg_causal_mask);
    reg.register(
        StmtKind::RegCombineIntFracEx2,
        execute_reg_combine_int_frac_ex2,
    );
    reg.register(StmtKind::RegCvt, execute_reg_cvt);
    reg.register(StmtKind::RegLoad, execute_reg_transfer);
    reg.register(StmtKind::RegStore, execute_reg_transfer);
}

fn reg_op_exec(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    operands: &[&RegOperand],
    f: impl FnOnce(&mut CohortContext, &TensorSlice) -> IResult<()>,
) -> IResult<StepStatus> {
    if !ctx.value_mode() {
        let resolved_dst = ctx.eval_slice(dst)?;
        for o in operands {
            emit_operand_read(ctx, o)?;
        }
        ctx.emit_tensor_write(&resolved_dst)?;
        if resolved_dst.tensor.space != MemorySpace::Reg {
            ctx.invalidate_shared_region(&resolved_dst)?;
        }
        return Ok(StepStatus::advance());
    }
    f(ctx, dst)?;
    Ok(StepStatus::advance())
}

fn emit_operand_read(ctx: &mut CohortContext, operand: &RegOperand) -> IResult<()> {
    if let RegOperand::Slice(slice) = operand {
        let resolved = ctx.eval_slice(slice)?;
        ctx.emit_tensor_read(&resolved)?;
    }
    Ok(())
}

fn dst_shape(ctx: &CohortContext, dst: &TensorSlice) -> IResult<(usize, usize)> {
    let timer = super::super::runner::prof_now();
    let resolved = ctx.eval_slice(dst)?;
    let out = Ok((
        ctx.cohort.len(),
        super::super::values::indexing::numel(&resolved.shape),
    ));
    super::super::runner::prof_end("REG:dst_shape", timer);
    out
}

fn read_reg_operand(
    ctx: &CohortContext,
    operand: &RegOperand,
    shape: (usize, usize),
    dst_dtype: DType,
) -> IResult<ValueArray2> {
    let timer = super::super::runner::prof_now();
    let out = match operand {
        RegOperand::Slice(slice) => {
            let arr =
                read_operand_array(ctx, &ctx.eval_slice(slice)?)?.into_coerce_to_dtype(dst_dtype);
            broadcast_array(arr, shape)
        }
        RegOperand::Literal(value) => Ok(literal_array(*value, dst_dtype, shape)),
    };
    super::super::runner::prof_end("REG:read_operand", timer);
    out
}

fn broadcast_array(arr: ValueArray2, shape: (usize, usize)) -> IResult<ValueArray2> {
    if arr.shape() == shape {
        return Ok(arr);
    }
    if arr.nrows() != shape.0 || arr.ncols() != 1 {
        return Err(InterpreterError::new(
            "tensor_value",
            "REG operand shape must match destination or contain one element per thread",
        ));
    }
    Ok(match arr {
        ValueArray2::Bool(a) => ValueArray2::Bool(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::I8(a) => ValueArray2::I8(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::U8(a) => ValueArray2::U8(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::I16(a) => ValueArray2::I16(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::U16(a) => ValueArray2::U16(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::I32(a) => ValueArray2::I32(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::U32(a) => ValueArray2::U32(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::I64(a) => ValueArray2::I64(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::U64(a) => ValueArray2::U64(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::F8E4M3(a) => {
            ValueArray2::F8E4M3(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]]))
        }
        ValueArray2::F16(a) => ValueArray2::F16(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::Bf16(a) => ValueArray2::Bf16(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
        ValueArray2::F32(a) => ValueArray2::F32(Array2::from_shape_fn(shape, |(i, _)| a[[i, 0]])),
    })
}

fn literal_array(value: RegLiteral, dtype: DType, shape: (usize, usize)) -> ValueArray2 {
    match dtype {
        DType::F16 | DType::Bf16 | DType::F32 => {
            ValueArray2::from_f32_compute(Array2::from_elem(shape, value.as_f32()), dtype)
        }
        _ => ValueArray2::from_i64_compute(Array2::from_elem(shape, value.as_i64()), dtype),
    }
}

fn write_reg_result(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    values: &ValueArray2,
) -> IResult<()> {
    let timer = super::super::runner::prof_now();
    let resolved_dst = ctx.eval_slice(dst)?;
    let out = write_operand(ctx, &resolved_dst, values);
    super::super::runner::prof_end("REG:write_result", timer);
    out
}

fn is_float_dtype(d: DType) -> bool {
    matches!(d, DType::F16 | DType::Bf16 | DType::F32)
}

#[derive(Clone)]
struct RegRange {
    tensor: Arc<Tensor>,
    start: usize,
    len: usize,
}

enum DirectFloatInput {
    Values { values: Vec<f32>, len: usize },
    Literal(f32),
}

impl DirectFloatInput {
    fn get(&self, ai: usize, j: usize) -> f32 {
        match self {
            Self::Values { values, len } => {
                if *len == 1 {
                    values[ai]
                } else {
                    values[ai * *len + j]
                }
            }
            Self::Literal(value) => *value,
        }
    }
}

fn resolve_reg_range(ctx: &CohortContext, slice: &TensorSlice) -> IResult<Option<RegRange>> {
    if slice.tensor.space != MemorySpace::Reg {
        return Ok(None);
    }
    let resolved = ctx.eval_slice(slice)?;
    let Some((start, len)) = uniform_contiguous_flat_range(&resolved, &resolved.tensor.shape)?
    else {
        return Ok(None);
    };
    Ok(Some(RegRange {
        tensor: resolved.tensor,
        start,
        len,
    }))
}

fn direct_rows(ctx: &CohortContext) -> Vec<usize> {
    ctx.rows()
}

fn direct_float_input(
    ctx: &CohortContext,
    operand: &RegOperand,
    compute_dtype: DType,
    dst_len: usize,
    rows: &[usize],
) -> IResult<Option<DirectFloatInput>> {
    match operand {
        RegOperand::Literal(value) => Ok(Some(DirectFloatInput::Literal(round_scalar(
            value.as_f32(),
            compute_dtype,
        )))),
        RegOperand::Slice(slice) => {
            let Some(range) = resolve_reg_range(ctx, slice)? else {
                return Ok(None);
            };
            if !is_float_dtype(range.tensor.dtype) || (range.len != dst_len && range.len != 1) {
                return Ok(None);
            }
            let inst = ctx
                .state
                .values
                .registers
                .get(&range.tensor, ctx.stream.cta_id)
                .ok_or_else(|| {
                    InterpreterError::new(
                        "missing_tensor_value",
                        "register instance is not written",
                    )
                })?;
            let mut values = inst.snapshot_float_row_range(rows, range.start, range.len)?;
            if range.tensor.dtype != compute_dtype {
                for value in &mut values {
                    *value = round_scalar(*value, compute_dtype);
                }
            }
            Ok(Some(DirectFloatInput::Values {
                values,
                len: range.len,
            }))
        }
    }
}

fn write_direct_float_result_with(
    ctx: &mut CohortContext,
    dst: &RegRange,
    rows: &[usize],
    mut value_at: impl FnMut(usize, usize) -> f32,
) -> IResult<()> {
    let key = RegisterKey {
        tensor: dst.tensor.clone(),
        cta_id: ctx.stream.cta_id,
    };
    let n = ctx.n_cta_threads();
    let inst = ctx
        .state
        .values
        .registers
        .by_instance
        .entry(key)
        .or_insert_with(|| {
            RegisterTensorValue::empty(dst.tensor.shape.clone(), dst.tensor.dtype, n)
        });
    inst.write_float_row_range_with(rows, dst.start, dst.len, |ai, j| value_at(ai, j))
}

fn try_direct_fill(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    value: &RegOperand,
) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    let Some(input) = direct_float_input(ctx, value, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let timer = super::super::runner::prof_now();
    write_direct_float_result_with(ctx, &dst_range, &rows, |ai, j| input.get(ai, j))?;
    super::super::runner::prof_end("REG:direct_fill", timer);
    Ok(true)
}

fn try_direct_binary(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    lhs: &RegOperand,
    rhs: &RegOperand,
    op: RegBinaryOp,
    rounding: Rounding,
) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) || matches!(op, RegBinaryOp::And | RegBinaryOp::Shl)
    {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    let Some(lhs_v) = direct_float_input(ctx, lhs, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let Some(rhs_v) = direct_float_input(ctx, rhs, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let timer = super::super::runner::prof_now();
    write_direct_float_result_with(ctx, &dst_range, &rows, |ai, j| {
        let value = match op {
            RegBinaryOp::Add => lhs_v.get(ai, j) + rhs_v.get(ai, j),
            RegBinaryOp::Sub => lhs_v.get(ai, j) - rhs_v.get(ai, j),
            RegBinaryOp::Mul => lhs_v.get(ai, j) * rhs_v.get(ai, j),
            RegBinaryOp::Max => lhs_v.get(ai, j).max(rhs_v.get(ai, j)),
            RegBinaryOp::Min => lhs_v.get(ai, j).min(rhs_v.get(ai, j)),
            RegBinaryOp::And | RegBinaryOp::Shl => unreachable!(),
        };
        if rounding == Rounding::Rm {
            value.floor()
        } else {
            value
        }
    })?;
    super::super::runner::prof_end("REG:direct_binary", timer);
    Ok(true)
}

fn try_direct_fma(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    a: &RegOperand,
    b: &RegOperand,
    c: &RegOperand,
) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    let Some(av) = direct_float_input(ctx, a, dst_range.tensor.dtype, dst_range.len, &rows)? else {
        return Ok(false);
    };
    let Some(bv) = direct_float_input(ctx, b, dst_range.tensor.dtype, dst_range.len, &rows)? else {
        return Ok(false);
    };
    let Some(cv) = direct_float_input(ctx, c, dst_range.tensor.dtype, dst_range.len, &rows)? else {
        return Ok(false);
    };
    let timer = super::super::runner::prof_now();
    write_direct_float_result_with(ctx, &dst_range, &rows, |ai, j| {
        av.get(ai, j) * bv.get(ai, j) + cv.get(ai, j)
    })?;
    super::super::runner::prof_end("REG:direct_fma", timer);
    Ok(true)
}

fn try_direct_reduce(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    src: &RegOperand,
    op: RegReduceOp,
) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) || dst_range.len != 1 {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    match src {
        RegOperand::Literal(value) => {
            let value = round_scalar(value.as_f32(), dst_range.tensor.dtype);
            write_direct_float_result_with(ctx, &dst_range, &rows, |_, _| value)?;
        }
        RegOperand::Slice(slice) => {
            let Some(src_range) = resolve_reg_range(ctx, slice)? else {
                return Ok(false);
            };
            if !is_float_dtype(src_range.tensor.dtype) {
                return Ok(false);
            }
            let inst = ctx
                .state
                .values
                .registers
                .get(&src_range.tensor, ctx.stream.cta_id)
                .ok_or_else(|| {
                    InterpreterError::new(
                        "missing_tensor_value",
                        "register instance is not written",
                    )
                })?;
            let mut src_values =
                inst.snapshot_float_row_range(&rows, src_range.start, src_range.len)?;
            if src_range.tensor.dtype != dst_range.tensor.dtype {
                for value in &mut src_values {
                    *value = round_scalar(*value, dst_range.tensor.dtype);
                }
            }
            let timer = super::super::runner::prof_now();
            write_direct_float_result_with(ctx, &dst_range, &rows, |ai, _| {
                let row = &src_values[ai * src_range.len..(ai + 1) * src_range.len];
                match op {
                    RegReduceOp::Max => {
                        row.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b))
                    }
                    RegReduceOp::Sum => row.iter().sum(),
                }
            })?;
            super::super::runner::prof_end("REG:direct_reduce", timer);
        }
    };
    Ok(true)
}

fn try_direct_causal_mask(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    src: &RegOperand,
    query_start: &crate::ir::ScalarValue,
    key_start: &crate::ir::ScalarValue,
    group_size: u32,
    mask_value: &RegOperand,
) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) || group_size == 0 {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    let Some(src_v) = direct_float_input(ctx, src, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let Some(mask_v) = direct_float_input(
        ctx,
        mask_value,
        dst_range.tensor.dtype,
        dst_range.len,
        &rows,
    )?
    else {
        return Ok(false);
    };
    let q0 = ctx.eval_scalar_uniform(
        query_start,
        "reg_causal_mask query_start",
        "divergent_operands",
    )?;
    let k0 =
        ctx.eval_scalar_uniform(key_start, "reg_causal_mask key_start", "divergent_operands")?;
    let tid_in_wg: Vec<usize> = ctx.cohort.iter().map(|thread| thread.tid_in_wg()).collect();
    let timer = super::super::runner::prof_now();
    write_direct_float_result_with(ctx, &dst_range, &rows, |ai, j| {
        let q = q0 + (tid_in_wg[ai] / group_size as usize) as i64;
        let k = k0 + j as i64;
        if k > q {
            mask_v.get(ai, j)
        } else {
            src_v.get(ai, j)
        }
    })?;
    super::super::runner::prof_end("REG:direct_causal_mask", timer);
    Ok(true)
}

fn try_direct_cond_rescale(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    src: &RegOperand,
    scale: &RegOperand,
    threshold: &RegOperand,
    scope: RegCondScope,
) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    let Some(src_v) = direct_float_input(ctx, src, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let Some(scale_v) =
        direct_float_input(ctx, scale, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let Some(threshold_v) =
        direct_float_input(ctx, threshold, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let timer = super::super::runner::prof_now();
    let mut group_needs = std::collections::HashMap::<usize, bool>::new();
    for (ai, &row) in rows.iter().enumerate() {
        let group = match scope {
            RegCondScope::Warp => row / 32,
            RegCondScope::Warpgroup => row / 128,
        };
        let needs = (0..dst_range.len).any(|j| scale_v.get(ai, j) < threshold_v.get(ai, j));
        group_needs
            .entry(group)
            .and_modify(|v| *v |= needs)
            .or_insert(needs);
    }
    write_direct_float_result_with(ctx, &dst_range, &rows, |ai, j| {
        let row = rows[ai];
        let group = match scope {
            RegCondScope::Warp => row / 32,
            RegCondScope::Warpgroup => row / 128,
        };
        let rescale = *group_needs.get(&group).unwrap_or(&false);
        let src_value = src_v.get(ai, j);
        if rescale {
            src_value * scale_v.get(ai, j)
        } else {
            src_value
        }
    })?;
    super::super::runner::prof_end("REG:direct_cond_rescale", timer);
    Ok(true)
}

fn try_direct_cvt(ctx: &mut CohortContext, dst: &TensorSlice, src: &TensorSlice) -> IResult<bool> {
    let Some(dst_range) = resolve_reg_range(ctx, dst)? else {
        return Ok(false);
    };
    if !is_float_dtype(dst_range.tensor.dtype) {
        return Ok(false);
    }
    let rows = direct_rows(ctx);
    let src_op = RegOperand::Slice(src.clone());
    let Some(input) =
        direct_float_input(ctx, &src_op, dst_range.tensor.dtype, dst_range.len, &rows)?
    else {
        return Ok(false);
    };
    let timer = super::super::runner::prof_now();
    write_direct_float_result_with(ctx, &dst_range, &rows, |ai, j| input.get(ai, j))?;
    super::super::runner::prof_end("REG:direct_cvt", timer);
    Ok(true)
}

fn binary_float(
    lhs: Array2<f32>,
    rhs: Array2<f32>,
    op: RegBinaryOp,
    rounding: Rounding,
) -> IResult<Array2<f32>> {
    let mut out = match op {
        RegBinaryOp::Add => &lhs + &rhs,
        RegBinaryOp::Sub => &lhs - &rhs,
        RegBinaryOp::Mul => &lhs * &rhs,
        RegBinaryOp::Max => Zip::from(&lhs)
            .and(&rhs)
            .map_collect(|&a, &b| if b > a { b } else { a }),
        RegBinaryOp::Min => Zip::from(&lhs)
            .and(&rhs)
            .map_collect(|&a, &b| if b < a { b } else { a }),
        RegBinaryOp::And | RegBinaryOp::Shl => {
            return Err(InterpreterError::new(
                "tensor_value",
                "floating REG binary op does not support bitwise operations",
            ))
        }
    };
    if rounding == Rounding::Rm {
        out.mapv_inplace(|v| v.floor());
    }
    Ok(out)
}

fn binary_int(lhs: Array2<i64>, rhs: Array2<i64>, op: RegBinaryOp) -> IResult<Array2<i64>> {
    Ok(match op {
        RegBinaryOp::Add => &lhs + &rhs,
        RegBinaryOp::Sub => &lhs - &rhs,
        RegBinaryOp::Mul => &lhs * &rhs,
        RegBinaryOp::Max => Zip::from(&lhs).and(&rhs).map_collect(|&a, &b| a.max(b)),
        RegBinaryOp::Min => Zip::from(&lhs).and(&rhs).map_collect(|&a, &b| a.min(b)),
        RegBinaryOp::And => Zip::from(&lhs).and(&rhs).map_collect(|&a, &b| a & b),
        RegBinaryOp::Shl => Zip::from(&lhs)
            .and(&rhs)
            .map_collect(|&a, &b| a.wrapping_shl((b & 63) as u32)),
    })
}

fn execute_binary(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    lhs: &RegOperand,
    rhs: &RegOperand,
    op: RegBinaryOp,
    rounding: Rounding,
) -> IResult<()> {
    if try_direct_binary(ctx, dst, lhs, rhs, op, rounding)? {
        return Ok(());
    }
    let shape = dst_shape(ctx, dst)?;
    let dtype = dst.tensor.dtype;
    let lhs_v = read_reg_operand(ctx, lhs, shape, dtype)?;
    let rhs_v = read_reg_operand(ctx, rhs, shape, dtype)?;
    let timer = super::super::runner::prof_now();
    let out = if is_float_dtype(dtype) {
        ValueArray2::from_f32_compute(
            binary_float(lhs_v.to_f32_compute(), rhs_v.to_f32_compute(), op, rounding)?,
            dtype,
        )
    } else {
        ValueArray2::from_i64_compute(
            binary_int(lhs_v.to_i64_compute(), rhs_v.to_i64_compute(), op)?,
            dtype,
        )
    };
    super::super::runner::prof_end("REG:binary_compute", timer);
    write_reg_result(ctx, dst, &out)
}

fn execute_reg_fill<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, value) = match stmt {
        Stmt::RegFill { dst, value } => (dst, value),
        _ => unreachable!(),
    };
    reg_op_exec(ctx, dst, &[value], |ctx, dst| {
        if try_direct_fill(ctx, dst, value)? {
            return Ok(());
        }
        let shape = dst_shape(ctx, dst)?;
        let out = read_reg_operand(ctx, value, shape, dst.tensor.dtype)?;
        write_reg_result(ctx, dst, &out)
    })
}

fn execute_reg_unary<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src, op) = match stmt {
        Stmt::RegUnary { dst, src, op } => (dst, src, *op),
        _ => unreachable!(),
    };
    reg_op_exec(ctx, dst, &[src], |ctx, dst| {
        let shape = dst_shape(ctx, dst)?;
        let dtype = dst.tensor.dtype;
        if !is_float_dtype(dtype) {
            return Err(InterpreterError::new(
                "tensor_value",
                "REG unary op requires a floating destination dtype",
            ));
        }
        let src_v = read_reg_operand(ctx, src, shape, dtype)?.to_f32_compute();
        let out = match op {
            RegUnaryOp::Exp2 => src_v.mapv(|v| v.exp2()),
            RegUnaryOp::Rcp => src_v.mapv(|v| 1.0 / v),
            RegUnaryOp::Neg => src_v.mapv(|v| -v),
        };
        write_reg_result(ctx, dst, &ValueArray2::from_f32_compute(out, dtype))
    })
}

fn execute_reg_add<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegAdd {
            dst,
            lhs,
            rhs,
            rounding,
        } => reg_op_exec(ctx, dst, &[lhs, rhs], |ctx, dst| {
            execute_binary(ctx, dst, lhs, rhs, RegBinaryOp::Add, *rounding)
        }),
        _ => unreachable!(),
    }
}
fn execute_reg_sub<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegSub {
            dst,
            lhs,
            rhs,
            rounding,
        } => reg_op_exec(ctx, dst, &[lhs, rhs], |ctx, dst| {
            execute_binary(ctx, dst, lhs, rhs, RegBinaryOp::Sub, *rounding)
        }),
        _ => unreachable!(),
    }
}
fn execute_reg_mul<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegMul { dst, lhs, rhs } => reg_op_exec(ctx, dst, &[lhs, rhs], |ctx, dst| {
            execute_binary(ctx, dst, lhs, rhs, RegBinaryOp::Mul, Rounding::Rn)
        }),
        _ => unreachable!(),
    }
}
fn execute_reg_max<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegMax { dst, lhs, rhs } => reg_op_exec(ctx, dst, &[lhs, rhs], |ctx, dst| {
            execute_binary(ctx, dst, lhs, rhs, RegBinaryOp::Max, Rounding::Rn)
        }),
        _ => unreachable!(),
    }
}
fn execute_reg_min<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegMin { dst, lhs, rhs } => reg_op_exec(ctx, dst, &[lhs, rhs], |ctx, dst| {
            execute_binary(ctx, dst, lhs, rhs, RegBinaryOp::Min, Rounding::Rn)
        }),
        _ => unreachable!(),
    }
}
fn execute_reg_fma<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegFma { dst, a, b, c } => reg_op_exec(ctx, dst, &[a, b, c], |ctx, dst| {
            if try_direct_fma(ctx, dst, a, b, c)? {
                return Ok(());
            }
            let shape = dst_shape(ctx, dst)?;
            let dtype = dst.tensor.dtype;
            let av = read_reg_operand(ctx, a, shape, dtype)?;
            let bv = read_reg_operand(ctx, b, shape, dtype)?;
            let cv = read_reg_operand(ctx, c, shape, dtype)?;
            let timer = super::super::runner::prof_now();
            let out = if is_float_dtype(dtype) {
                ValueArray2::from_f32_compute(
                    &(&av.to_f32_compute() * &bv.to_f32_compute()) + &cv.to_f32_compute(),
                    dtype,
                )
            } else {
                ValueArray2::from_i64_compute(
                    &(&av.to_i64_compute() * &bv.to_i64_compute()) + &cv.to_i64_compute(),
                    dtype,
                )
            };
            super::super::runner::prof_end("REG:fma_compute", timer);
            write_reg_result(ctx, dst, &out)
        }),
        _ => unreachable!(),
    }
}

fn execute_reg_bitwise<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegBitwise { dst, lhs, rhs, op } => reg_op_exec(ctx, dst, &[lhs, rhs], |ctx, dst| {
            execute_binary(ctx, dst, lhs, rhs, *op, Rounding::Rn)
        }),
        _ => unreachable!(),
    }
}

fn execute_reg_reduce<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src, op) = match stmt {
        Stmt::RegReduce { dst, src, op } => (dst, src, *op),
        _ => unreachable!(),
    };
    reg_op_exec(ctx, dst, &[src], |ctx, dst| {
        if try_direct_reduce(ctx, dst, src, op)? {
            return Ok(());
        }
        let out_shape = dst_shape(ctx, dst)?;
        if out_shape.1 != 1 {
            return Err(InterpreterError::new(
                "tensor_value",
                "REG reduce destination must contain one element per thread",
            ));
        }
        let dtype = dst.tensor.dtype;
        let src_arr = match src {
            RegOperand::Slice(slice) => {
                let resolved = ctx.eval_slice(slice)?;
                read_operand_array(ctx, &resolved)?.into_coerce_to_dtype(dtype)
            }
            RegOperand::Literal(value) => literal_array(*value, dtype, out_shape),
        };
        let timer = super::super::runner::prof_now();
        let out = if is_float_dtype(dtype) {
            let f = src_arr.to_f32_compute();
            let reduced = Array2::from_shape_fn(out_shape, |(i, _)| match op {
                RegReduceOp::Max => {
                    f.row(i)
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a })
                }
                RegReduceOp::Sum => f.row(i).sum(),
            });
            ValueArray2::from_f32_compute(reduced, dtype)
        } else {
            let v = src_arr.to_i64_compute();
            let reduced = Array2::from_shape_fn(out_shape, |(i, _)| match op {
                RegReduceOp::Max => v.row(i).iter().copied().max().unwrap_or(0),
                RegReduceOp::Sum => v.row(i).sum(),
            });
            ValueArray2::from_i64_compute(reduced, dtype)
        };
        super::super::runner::prof_end("REG:reduce_compute", timer);
        write_reg_result(ctx, dst, &out)
    })
}

fn execute_reg_cond_rescale<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src, scale, threshold, scope) = match stmt {
        Stmt::RegCondRescale {
            dst,
            src,
            scale,
            threshold,
            scope,
        } => (dst, src, scale, threshold, *scope),
        _ => unreachable!(),
    };
    reg_op_exec(ctx, dst, &[src, scale, threshold], |ctx, dst| {
        if try_direct_cond_rescale(ctx, dst, src, scale, threshold, scope)? {
            return Ok(());
        }
        let shape = dst_shape(ctx, dst)?;
        let dtype = dst.tensor.dtype;
        if !is_float_dtype(dtype) {
            return Err(InterpreterError::new(
                "tensor_value",
                "reg_cond_rescale requires a floating destination dtype",
            ));
        }
        let src_v = read_reg_operand(ctx, src, shape, dtype)?.to_f32_compute();
        let scale_v = read_reg_operand(ctx, scale, shape, dtype)?.to_f32_compute();
        let threshold_v = read_reg_operand(ctx, threshold, shape, dtype)?.to_f32_compute();
        let timer = super::super::runner::prof_now();
        let rows = ctx.rows();
        let mut group_needs = std::collections::HashMap::<usize, bool>::new();
        for (ai, &row) in rows.iter().enumerate() {
            let group = match scope {
                RegCondScope::Warp => row / 32,
                RegCondScope::Warpgroup => row / 128,
            };
            let needs = (0..shape.1).any(|j| scale_v[[ai, j]] < threshold_v[[ai, j]]);
            group_needs
                .entry(group)
                .and_modify(|v| *v |= needs)
                .or_insert(needs);
        }
        let out = Array2::from_shape_fn(shape, |(ai, j)| {
            let group = match scope {
                RegCondScope::Warp => rows[ai] / 32,
                RegCondScope::Warpgroup => rows[ai] / 128,
            };
            if *group_needs.get(&group).unwrap_or(&false) {
                src_v[[ai, j]] * scale_v[[ai, j]]
            } else {
                src_v[[ai, j]]
            }
        });
        super::super::runner::prof_end("REG:cond_rescale_compute", timer);
        write_reg_result(ctx, dst, &ValueArray2::from_f32_compute(out, dtype))
    })
}

fn execute_reg_softmax_rescale<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (row_max, row_scale, row_max_old, row_max_new, scale_log2, threshold) = match stmt {
        Stmt::RegSoftmaxRescale {
            row_max,
            row_scale,
            row_max_old,
            row_max_new,
            scale_log2,
            threshold,
        } => (
            row_max,
            row_scale,
            row_max_old,
            row_max_new,
            scale_log2,
            threshold,
        ),
        _ => unreachable!(),
    };
    let operands = [row_max_old, row_max_new, scale_log2, threshold];
    if !ctx.value_mode() {
        let max_r = ctx.eval_slice(row_max)?;
        let scale_r = ctx.eval_slice(row_scale)?;
        for operand in operands {
            emit_operand_read(ctx, operand)?;
        }
        ctx.emit_tensor_write(&max_r)?;
        ctx.emit_tensor_write(&scale_r)?;
        return Ok(StepStatus::advance());
    }

    let shape = dst_shape(ctx, row_max)?;
    let scale_shape = dst_shape(ctx, row_scale)?;
    if scale_shape != shape {
        return Err(InterpreterError::new(
            "tensor_value",
            "reg_softmax_rescale row_max and row_scale shapes must match",
        ));
    }
    let dtype = row_max.tensor.dtype;
    if !is_float_dtype(dtype) || !is_float_dtype(row_scale.tensor.dtype) {
        return Err(InterpreterError::new(
            "tensor_value",
            "reg_softmax_rescale requires floating destinations",
        ));
    }
    let old_v = read_reg_operand(ctx, row_max_old, shape, dtype)?.to_f32_compute();
    let new_v = read_reg_operand(ctx, row_max_new, shape, dtype)?.to_f32_compute();
    let scale_log2_v = read_reg_operand(ctx, scale_log2, shape, DType::F32)?.to_f32_compute();
    let threshold_v = read_reg_operand(ctx, threshold, shape, DType::F32)?.to_f32_compute();
    let timer = super::super::runner::prof_now();
    let delta = (&old_v - &new_v) * &scale_log2_v;
    let max_out = Zip::from(&old_v)
        .and(&new_v)
        .and(&delta)
        .and(&threshold_v)
        .map_collect(|&old, &new, &d, &t| if d >= -t { old } else { new });
    let scale_out =
        Zip::from(&delta)
            .and(&threshold_v)
            .map_collect(|&d, &t| if d >= -t { 1.0 } else { d.exp2() });
    super::super::runner::prof_end("REG:softmax_rescale_compute", timer);
    write_reg_result(ctx, row_max, &ValueArray2::from_f32_compute(max_out, dtype))?;
    write_reg_result(
        ctx,
        row_scale,
        &ValueArray2::from_f32_compute(scale_out, row_scale.tensor.dtype),
    )?;
    Ok(StepStatus::advance())
}

fn execute_reg_causal_mask<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src, query_start, key_start, group_size, mask_value) = match stmt {
        Stmt::RegCausalMask {
            dst,
            src,
            query_start,
            key_start,
            group_size,
            mask_value,
        } => (dst, src, query_start, key_start, *group_size, mask_value),
        _ => unreachable!(),
    };
    reg_op_exec(ctx, dst, &[src, mask_value], |ctx, dst| {
        if try_direct_causal_mask(
            ctx,
            dst,
            src,
            query_start,
            key_start,
            group_size,
            mask_value,
        )? {
            return Ok(());
        }
        let shape = dst_shape(ctx, dst)?;
        let dtype = dst.tensor.dtype;
        if !is_float_dtype(dtype) {
            return Err(InterpreterError::new(
                "tensor_value",
                "reg_causal_mask requires a floating destination dtype",
            ));
        }
        if group_size == 0 {
            return Err(InterpreterError::new(
                "tensor_value",
                "reg_causal_mask group_size must be positive",
            ));
        }
        let q0 = ctx.eval_scalar_uniform(
            query_start,
            "reg_causal_mask query_start",
            "divergent_operands",
        )?;
        let k0 =
            ctx.eval_scalar_uniform(key_start, "reg_causal_mask key_start", "divergent_operands")?;
        let src_v = read_reg_operand(ctx, src, shape, dtype)?.to_f32_compute();
        let mask_v = read_reg_operand(ctx, mask_value, shape, dtype)?.to_f32_compute();
        let timer = super::super::runner::prof_now();
        let rows: Vec<usize> = ctx.cohort.iter().map(|thread| thread.tid_in_wg()).collect();
        let out = Array2::from_shape_fn(shape, |(ai, j)| {
            let q = q0 + (rows[ai] / group_size as usize) as i64;
            let k = k0 + j as i64;
            if k > q {
                mask_v[[ai, j]]
            } else {
                src_v[[ai, j]]
            }
        });
        super::super::runner::prof_end("REG:causal_mask_compute", timer);
        write_reg_result(ctx, dst, &ValueArray2::from_f32_compute(out, dtype))
    })
}

fn execute_reg_combine_int_frac_ex2<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, rounded, frac_ex2) = match stmt {
        Stmt::RegCombineIntFracEx2 {
            dst,
            rounded,
            frac_ex2,
        } => (dst, rounded, frac_ex2),
        _ => unreachable!(),
    };
    reg_op_exec(ctx, dst, &[rounded, frac_ex2], |ctx, dst| {
        let shape = dst_shape(ctx, dst)?;
        let dtype = dst.tensor.dtype;
        if !is_float_dtype(dtype) {
            return Err(InterpreterError::new(
                "tensor_value",
                "reg_combine_int_frac_ex2 requires a floating destination dtype",
            ));
        }
        let rounded_v = read_reg_operand(ctx, rounded, shape, DType::F32)?.to_f32_compute();
        let frac_v = read_reg_operand(ctx, frac_ex2, shape, DType::F32)?.to_f32_compute();
        let timer = super::super::runner::prof_now();
        let out = Zip::from(&rounded_v).and(&frac_v).map_collect(|&r, &f| {
            f32::from_bits(r.to_bits().wrapping_shl(23).wrapping_add(f.to_bits()))
        });
        super::super::runner::prof_end("REG:combine_ex2_compute", timer);
        write_reg_result(ctx, dst, &ValueArray2::from_f32_compute(out, dtype))
    })
}

fn execute_reg_cvt<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    match stmt {
        Stmt::RegCvt { dst, src, .. } => {
            let src_op = RegOperand::Slice(src.clone());
            reg_op_exec(ctx, dst, &[&src_op], |ctx, dst| {
                if try_direct_cvt(ctx, dst, src)? {
                    return Ok(());
                }
                let shape = dst_shape(ctx, dst)?;
                let src_v = read_reg_operand(ctx, &src_op, shape, dst.tensor.dtype)?;
                write_reg_result(ctx, dst, &src_v.into_coerce_to_dtype(dst.tensor.dtype))
            })
        }
        _ => unreachable!(),
    }
}

fn execute_reg_transfer<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src) = match stmt {
        Stmt::RegLoad { dst, src } | Stmt::RegStore { dst, src } => (dst, src),
        _ => unreachable!(),
    };
    if !ctx.value_mode() {
        let resolved_src = ctx.eval_slice(src)?;
        let resolved_dst = ctx.eval_slice(dst)?;
        ctx.emit_tensor_read(&resolved_src)?;
        ctx.emit_tensor_write(&resolved_dst)?;
        if resolved_dst.tensor.space != MemorySpace::Reg {
            ctx.invalidate_shared_region(&resolved_dst)?;
        }
        return Ok(StepStatus::advance());
    }
    let resolved_src = ctx.eval_slice(src)?;
    let resolved_dst = ctx.eval_slice(dst)?;
    let dtype = dst.tensor.dtype;
    let values = read_transfer_array(ctx, &resolved_src, dtype)?;
    write_operand(ctx, &resolved_dst, &values)?;
    Ok(StepStatus::advance())
}
