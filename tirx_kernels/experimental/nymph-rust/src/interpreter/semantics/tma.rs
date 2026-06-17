//! Bulk GMEM<->SMEM TMA — port of `semantics/tma.py`. Loads complete-tx the
//! signaled mbar(s) and (in value mode) copy the tile; stores copy SMEM->GMEM.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::mbar_ops::{
    complete_mbarrier_tx, initialized_mbar_cell, multicast_target_ctas, peer_ctaid_in_cluster,
    retarget_mbar, uniform_mbar_target, MbarTarget,
};
use super::super::outcomes::StepStatus;
use super::super::protocol::{MemoryAccessKind, MemoryProxy, TensorAccessKind, TraceEventKind};
use super::super::region;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::values::indexing::numel;
use super::super::values::mbars::{MbarCell, MbarCellKey};
use super::super::values::tensors::{DenseTensorValue, TensorInstanceKey, TensorOwner};
use crate::ir::{DType, MemorySpace, Stmt, Tensor, TensorSlice};
use std::collections::HashMap;
use std::sync::Arc;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::TmaLoad, execute_tma_load);
    reg.register(StmtKind::TmaStore, execute_tma_store);
}

fn dtype_bytes(d: DType) -> i64 {
    match d {
        DType::Bool | DType::I8 | DType::U8 | DType::F8E4M3 => 1,
        DType::I16 | DType::U16 | DType::F16 | DType::Bf16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 => 8,
    }
}

fn uniform_tuple(
    ctx: &CohortContext,
    values: &[crate::ir::ScalarValue],
    label: &str,
) -> IResult<Vec<usize>> {
    values
        .iter()
        .map(|v| {
            ctx.eval_scalar_uniform(v, label, "divergent_tma_operands")
                .map(|x| x as usize)
        })
        .collect()
}

/// TMA tensormap out-of-bounds semantics: clamp a gmem box to the tensor's
/// extents. Returns the per-dimension VALID extents (loads zero-fill the rest;
/// stores squash it). `None` change = the box is fully in bounds.
fn clamp_gmem_box(tensor: &Tensor, coords: &[usize], shape: &[usize]) -> Vec<usize> {
    coords
        .iter()
        .zip(shape.iter())
        .zip(tensor.shape.iter())
        .map(|((&c, &s), &dim)| dim.saturating_sub(c).min(s))
        .collect()
}

/// Row-major odometer over `shape[..rank-1]` restricted to `valid[..rank-1]`,
/// yielding (flat_index_in_full_box_rows, multi_index). Outer-row granularity:
/// the clamped copy stays a per-row memcpy, never per-element.
fn for_each_valid_outer_row(
    shape: &[usize],
    valid: &[usize],
    mut f: impl FnMut(usize, &[usize]) -> crate::interpreter::diagnostics::IResult<()>,
) -> crate::interpreter::diagnostics::IResult<()> {
    let outer = &valid[..valid.len() - 1];
    if outer.iter().any(|&v| v == 0) {
        return Ok(());
    }
    let mut idx = vec![0usize; outer.len()];
    loop {
        // flat row index within the FULL box's outer dims
        let mut flat = 0usize;
        for (i, &x) in idx.iter().enumerate() {
            flat = flat * shape[i] + x;
        }
        f(flat, &idx)?;
        // odometer increment over the VALID extents
        let mut d = idx.len();
        loop {
            if d == 0 {
                return Ok(());
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < outer[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

fn execute_tma_load<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src, mbar, bytes, coords, shape, gmem_shape, mbar_stage, multicast, cta_group) =
        match stmt {
            Stmt::TmaLoad {
                dst,
                src,
                mbar,
                bytes,
                coords,
                shape,
                gmem_shape,
                mbar_stage,
                multicast_cta_mask,
                cta_group,
            } => (
                dst,
                src,
                mbar,
                bytes,
                coords,
                shape,
                gmem_shape,
                mbar_stage,
                *multicast_cta_mask,
                *cta_group,
            ),
            _ => unreachable!(),
        };
    let byte_count = ctx.eval_scalar_uniform(bytes, "tma_load bytes", "divergent_tma_operands")?;
    if byte_count < 1 {
        return Err(InterpreterError::new(
            "tma_load_bytes",
            "tma_load bytes must be positive",
        ));
    }
    let expected = numel(shape) as i64 * dtype_bytes(src.dtype);
    if byte_count != expected {
        return Err(InterpreterError::new(
            "tma_bytes_mismatch",
            "tma_load bytes mismatch the tile",
        ));
    }
    let coords_r = uniform_tuple(ctx, coords, "tma_load coords")?;
    let dst_offsets = uniform_tuple(ctx, &dst.offsets, "tma_load dst offsets")?;

    let dst_ctas: Vec<usize> = match multicast {
        None => vec![ctx.stream.ctaid_in_cluster],
        Some(mask) => multicast_target_ctas(ctx, mask, "tma", "TmaLoad")?,
    };
    let base = uniform_mbar_target(ctx, mbar, mbar_stage.as_ref())?;
    let targets = mbar_signal_targets(ctx, base, &dst_ctas, cta_group, multicast.is_some())?;

    if ctx.trace_mode() {
        let scope = ctx.access_scope();
        // The tensormap clamps the read box to the tensor; out-of-bounds parts
        // are zero-filled and touch no memory. The footprint is the CLAMPED
        // RECTANGLE decomposed into row runs — a tile only projects to one
        // contiguous interval when it covers full inner rows. A fully-OOB box
        // reads nothing.
        let src_shape = tma_tensor_shape(src, &coords_r, shape, gmem_shape.as_ref(), "tma_load")?;
        if let Some(read_region) =
            region::tensor_rect_region_clamped(src, ctx.stream.cta_id, &coords_r, &src_shape)?
        {
            ctx.emit(TraceEventKind::Read {
                region: read_region,
                proxy: MemoryProxy::Generic,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad),
                scope: scope.clone(),
            })?;
        }
        for &ctaid in &dst_ctas {
            let global_cta = ctx.global_cta_id(ctaid);
            ctx.emit(TraceEventKind::Write {
                region: region::tensor_region_from_uniform(
                    &dst.tensor,
                    global_cta,
                    &dst_offsets,
                    shape,
                )?,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad),
                scope: scope.clone(),
            })?;
            let pool = ctx
                .state
                .values
                .smem
                .pool_for_mut(global_cta, ctx.kernel.smem_size_bytes);
            pool.invalidate_block(&dst.tensor, &dst_offsets, shape)?;
        }
    } else if ctx.value_mode() {
        // value mode: copy the GMEM tile straight into each destination SMEM instance.
        let source_key = TensorInstanceKey {
            tensor: Arc::clone(src),
            owner: TensorOwner::Global,
        };
        if !ctx
            .state
            .values
            .tensors
            .by_instance
            .contains_key(&source_key)
        {
            return Err(InterpreterError::new(
                "missing_tensor_value",
                "tma_load source not loaded",
            ));
        }
        let values = {
            let source = ctx
                .state
                .values
                .tensors
                .by_instance
                .get(&source_key)
                .expect("checked tma source exists");
            let _td = super::super::runner::prof_now();
            let src_shape =
                tma_tensor_shape(src, &coords_r, shape, gmem_shape.as_ref(), "tma_load")?;
            let valid = clamp_gmem_box(src, &coords_r, &src_shape);
            let values = if valid == src_shape {
                source.read_block(&coords_r, &src_shape)
            } else {
                // TMA OOB load: read the in-bounds sub-box with the same per-row
                // bulk path, then place its rows into a zeroed full tile (the
                // hardware zero-fill). No per-element work anywhere.
                let full = numel(&src_shape);
                let mut out =
                    crate::interpreter::values::arrays::ValueArray1::zeros(src.dtype, full);
                let inner_full = *src_shape.last().unwrap();
                let inner_valid = *valid.last().unwrap();
                // A box that is fully out of bounds in ANY dim touches nothing:
                // the whole tile is the zero fill.
                if valid.iter().all(|&v| v > 0) {
                    let sub = source.read_block(&coords_r, &valid)?;
                    let mut sub_row = 0usize;
                    for_each_valid_outer_row(&src_shape, &valid, |row_flat, _| {
                        out.copy_run_from(
                            row_flat * inner_full,
                            &sub,
                            sub_row * inner_valid,
                            inner_valid,
                        )?;
                        sub_row += 1;
                        Ok(())
                    })?;
                }
                Ok(out)
            };
            super::super::runner::prof_end("C:tma_read", _td);
            values?
        };
        let _tw = super::super::runner::prof_now();
        for &ctaid in &dst_ctas {
            let global_cta = ctx.global_cta_id(ctaid);
            let pool = ctx
                .state
                .values
                .smem
                .pool_for_mut(global_cta, ctx.kernel.smem_size_bytes);
            pool.write_block(&dst.tensor, &dst_offsets, shape, &values)?;
        }
        super::super::runner::prof_end("C:tma_write", _tw);
    }

    // complete-tx the signalled mbar cell(s) and wake their waiters.
    let wakes = apply_mbar_complete_tx(ctx, &targets, byte_count)?;
    Ok(StepStatus::advance_wake(wakes))
}

fn mbar_signal_targets(
    ctx: &CohortContext,
    base: MbarTarget,
    dst_ctas: &[usize],
    cta_group: u8,
    multicast: bool,
) -> IResult<Vec<MbarTarget>> {
    if cta_group != 1 && cta_group != 2 {
        return Err(InterpreterError::new(
            "invalid_tma_cta_group",
            "tma cta_group must be 1 or 2",
        ));
    }
    if cta_group == 1 {
        if !multicast {
            let dst_cta = dst_ctas[0];
            if base.identity.ctaid_in_cluster != dst_cta {
                return Err(InterpreterError::new(
                    "tma_mbar_cta_group_mismatch",
                    "TmaLoad cta_group=1 mbar target must match the destination CTA",
                ));
            }
            return Ok(vec![base]);
        }
        return Ok(dst_ctas.iter().map(|&c| retarget_mbar(base, c)).collect());
    }
    if !multicast {
        let dst_cta = dst_ctas[0];
        let peer = peer_ctaid_in_cluster(
            ctx,
            dst_cta,
            "tma_peer_cta_oob",
            "TmaLoad cta_group=2 peer CTA is outside the cluster",
        )?;
        if base.identity.ctaid_in_cluster != dst_cta && base.identity.ctaid_in_cluster != peer {
            return Err(InterpreterError::new(
                "tma_mbar_cta_group_mismatch",
                "TmaLoad cta_group=2 mbar target must be the destination CTA or its peer",
            ));
        }
        return Ok(vec![base]);
    }
    // cta_group == 2 multicast: pick the parity-matching CTA of each pair
    let parity = base.identity.ctaid_in_cluster & 1;
    let mut targets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &c in dst_ctas {
        let target_cta = if c & 1 == parity {
            c
        } else {
            peer_ctaid_in_cluster(
                ctx,
                c,
                "tma_peer_cta_oob",
                "TmaLoad cta_group=2 peer CTA is outside the cluster",
            )?
        };
        if seen.insert(target_cta) {
            targets.push(retarget_mbar(base, target_cta));
        }
    }
    Ok(targets)
}

/// Complete-tx the target mbar cell(s) directly and return their keys (the runner
/// re-checks each key's parked waiters). Duplicate targets coalesce.
fn apply_mbar_complete_tx(
    ctx: &mut CohortContext,
    targets: &[MbarTarget],
    byte_count: i64,
) -> IResult<Vec<MbarCellKey>> {
    let mut cells: HashMap<MbarCellKey, MbarCell> = HashMap::new();
    for target in targets {
        let key = target.key();
        let cell = match cells.get(&key) {
            Some(c) => *c,
            None => initialized_mbar_cell(ctx, key)?,
        };
        if ctx.trace_mode() {
            ctx.emit(TraceEventKind::MbarCompleteTx {
                target: (*target).into(),
                bytes: byte_count,
                scope: ctx.access_scope(),
            })?;
        }
        cells.insert(key, complete_mbarrier_tx(cell, byte_count)?);
    }
    let mut keys = Vec::with_capacity(cells.len());
    for (key, cell) in cells {
        ctx.state.values.mbars.cells.insert(key, cell);
        keys.push(key);
    }
    Ok(keys)
}

fn execute_tma_store<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, src, coords, shape, gmem_shape) = match stmt {
        Stmt::TmaStore {
            dst,
            src,
            coords,
            shape,
            gmem_shape,
        } => (dst, src, coords, shape, gmem_shape),
        _ => unreachable!(),
    };
    if ctx.trace_mode() {
        let coords_r = uniform_tuple(ctx, coords, "tma_store coords")?;
        let src_offsets = uniform_tuple(ctx, &src.offsets, "tma_store src offsets")?;
        let scope = ctx.access_scope();
        ctx.emit(TraceEventKind::Read {
            region: region::tensor_region_from_uniform(
                &src.tensor,
                ctx.stream.cta_id,
                &src_offsets,
                shape,
            )?,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaStore),
            scope: scope.clone(),
        })?;
        // The tensormap squashes out-of-bounds store elements: the write region
        // and the invalidation cover only the in-bounds part of the box, as
        // per-row runs (never one linear interval unless inner rows are full).
        let dst_shape = tma_tensor_shape(dst, &coords_r, shape, gmem_shape.as_ref(), "tma_store")?;
        if let Some(write_region) =
            region::tensor_rect_region_clamped(dst, ctx.stream.cta_id, &coords_r, &dst_shape)?
        {
            ctx.emit(TraceEventKind::Write {
                region: write_region,
                proxy: MemoryProxy::Generic,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaStore),
                scope,
            })?;
        }
        if coords_r.len() == dst.shape.len() && dst_shape.len() == dst.shape.len() {
            let valid = clamp_gmem_box(dst, &coords_r, &dst_shape);
            if valid.iter().all(|&v| v > 0) {
                invalidate_gmem_block(ctx, dst, &coords_r, &valid)?;
            }
        }
        return Ok(StepStatus::advance());
    }
    let coords_r = uniform_tuple(ctx, coords, "tma_store coords")?;
    let src_offsets = uniform_tuple(ctx, &src.offsets, "tma_store src offsets")?;
    store_write(
        ctx,
        dst,
        src,
        &coords_r,
        &src_offsets,
        shape,
        gmem_shape.as_ref(),
    )?;
    Ok(StepStatus::advance())
}

fn invalidate_gmem_block(
    ctx: &mut CohortContext,
    tensor: &Arc<Tensor>,
    offsets: &[usize],
    shape: &[usize],
) -> IResult<()> {
    if tensor.space != MemorySpace::Gmem {
        return Err(InterpreterError::new(
            "unsupported_tensor_space",
            "TMA store destination must be GMEM",
        ));
    }
    let instance = TensorInstanceKey {
        tensor: Arc::clone(tensor),
        owner: TensorOwner::Global,
    };
    let inst = ctx
        .state
        .values
        .tensors
        .by_instance
        .entry(instance)
        .or_insert_with(|| DenseTensorValue::empty(tensor.shape.clone(), tensor.dtype));
    inst.invalidate_slice_inplace(offsets, shape)
}

fn store_write(
    ctx: &mut CohortContext,
    dst: &Arc<crate::ir::Tensor>,
    src: &TensorSlice,
    coords: &[usize],
    src_offsets: &[usize],
    shape: &[usize],
    gmem_shape: Option<&Vec<usize>>,
) -> IResult<()> {
    let instance = TensorInstanceKey {
        tensor: Arc::clone(dst),
        owner: TensorOwner::Global,
    };
    let mut inst = ctx
        .state
        .values
        .tensors
        .by_instance
        .remove(&instance)
        .unwrap_or_else(|| DenseTensorValue::empty(dst.shape.clone(), dst.dtype));
    let values = ctx
        .state
        .values
        .smem
        .pool_for(ctx.stream.cta_id)?
        .read_block(&src.tensor, src_offsets, shape)?;
    let dst_shape = tma_tensor_shape(dst, coords, shape, gmem_shape, "tma_store")?;
    let valid = clamp_gmem_box(dst, coords, &dst_shape);
    let copy_result = if valid == dst_shape {
        inst.write_slice_inplace(coords, &dst_shape, &values)
    } else if valid.iter().any(|&v| v == 0) {
        Ok(()) // fully out of bounds: the tensormap squashes the whole store
    } else {
        // TMA OOB store: compact the in-bounds prefix of each row (one memcpy
        // per row) and write the clamped rectangle through the same bulk path.
        let inner_full = *dst_shape.last().unwrap();
        let inner_valid = *valid.last().unwrap();
        let rows_valid: usize = valid[..valid.len() - 1].iter().product();
        let mut sub = crate::interpreter::values::arrays::ValueArray1::zeros(
            dst.dtype,
            rows_valid * inner_valid,
        );
        let mut sub_row = 0usize;
        for_each_valid_outer_row(&dst_shape, &valid, |row_flat, _| {
            sub.copy_run_from(
                sub_row * inner_valid,
                &values,
                row_flat * inner_full,
                inner_valid,
            )?;
            sub_row += 1;
            Ok(())
        })?;
        inst.write_slice_inplace(coords, &valid, &sub)
    };
    ctx.state.values.tensors.by_instance.insert(instance, inst);
    copy_result
}

fn tma_tensor_shape(
    tensor: &Arc<crate::ir::Tensor>,
    coords: &[usize],
    tile_shape: &[usize],
    explicit_shape: Option<&Vec<usize>>,
    label: &str,
) -> IResult<Vec<usize>> {
    let rank = tensor.shape.len();
    if coords.len() != rank {
        return Err(InterpreterError::new(
            "tma_shape_projection",
            format!("{label} coords rank must match the GMEM tensor rank"),
        ));
    }
    if let Some(shape) = explicit_shape {
        return Ok(shape.clone());
    }
    if tile_shape.len() == rank {
        return Ok(tile_shape.to_vec());
    }
    if tile_shape.len() > rank {
        return Err(InterpreterError::new(
            "tma_shape_projection",
            format!("{label} tile rank exceeds the GMEM tensor rank"),
        ));
    }

    let mut shape = vec![1usize; rank];
    let mut dim = 0usize;
    for &extent in tile_shape {
        let mut matched = None;
        while dim < rank {
            let coord = coords[dim];
            let tensor_dim = tensor.shape[dim];
            if coord <= tensor_dim && extent <= tensor_dim - coord {
                matched = Some(dim);
                dim += 1;
                break;
            }
            dim += 1;
        }
        let Some(axis) = matched else {
            return Err(InterpreterError::new(
                "tma_shape_projection",
                format!("{label} tile shape cannot be projected onto the GMEM tensor"),
            ));
        };
        shape[axis] = extent;
    }
    Ok(shape)
}
