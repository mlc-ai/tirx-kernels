//! Typed per-thread slice transfer for REG<->SMEM/GMEM — port of `transfer.py`.

use super::cohort::CohortContext;
use super::diagnostics::IResult;
use super::slice_indexing::ResolvedSlice;
use super::values::arrays::ValueArray2;
use crate::ir::{DType, MemorySpace};

/// Read one ALU operand slice in the tensor's own container dtype (no upcast).
pub fn read_operand_array(ctx: &CohortContext, resolved: &ResolvedSlice) -> IResult<ValueArray2> {
    if resolved.tensor.space == MemorySpace::Reg {
        ctx.registers_read(resolved)
    } else {
        ctx.shared_read(resolved)
    }
}

/// Read a slice for a RegLoad/RegStore transfer, coerced to the destination dtype.
pub fn read_transfer_array(
    ctx: &CohortContext,
    resolved: &ResolvedSlice,
    dst_dtype: DType,
) -> IResult<ValueArray2> {
    let raw = read_operand_array(ctx, resolved)?;
    Ok(raw.into_coerce_to_dtype(dst_dtype))
}

/// Write one native-array slice directly into its container (REG vs shared by space).
pub fn write_operand(
    ctx: &mut CohortContext,
    resolved: &ResolvedSlice,
    values: &ValueArray2,
) -> IResult<()> {
    if resolved.tensor.space == MemorySpace::Reg {
        ctx.registers_write(resolved, values)
    } else {
        ctx.shared_write(resolved, values)
    }
}
