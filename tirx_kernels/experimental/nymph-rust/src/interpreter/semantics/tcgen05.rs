//! tcgen05 MMA + TMEM<->REG datapath + commit — port of `semantics/tcgen05.py`.
//! The GEMM compute core: D = A @ Bᵀ (f32 accumulate from f16/bf16), placed into
//! the TMEM accumulator; ld/st move the accumulator to/from registers.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::mbar_ops::{
    arrive_mbarrier_cell, initialized_mbar_cell, multicast_target_ctas, peer_ctaid_in_cluster,
    retarget_mbar, uniform_mbar_target, MbarTarget,
};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::{
    MemoryAccessKind, MemoryProxy, PoolId, Region, TensorAccessKind, TmemAsyncKind, TraceEventKind,
};
use super::super::region;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scheduler::CtaActivityStatus;
use super::super::slice_indexing::ResolvedSlice;
use super::super::values::arrays::ValueArray1;
use super::super::values::indexing::numel;
use super::super::values::mbars::{MbarCell, MbarCellKey};
use super::super::values::tcgen05_datapath::{
    datapath_has_cell_aliases_cached, datapath_index_arrays_cached, datapath_index_summary_cached,
};
use super::super::values::tmem::{tmem_layout_for, TMEM_COLS, TMEM_ROWS};
use crate::ir::{MemorySpace, Stmt, TensorSlice};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

// Reusable per-MMA f32 operand scratch (no per-call alloc): each MMA gathers A/B from
// the SMEM byte pool into these, then cblas reads the contiguous scratch.
thread_local! {
    static MMA_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        std::cell::RefCell::new((Vec::new(), Vec::new()));
}

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::Tcgen05Commit, execute_commit);
    reg.register(StmtKind::Tcgen05Ld, execute_ld);
    reg.register(StmtKind::Tcgen05WaitLd, execute_wait);
    reg.register(StmtKind::Tcgen05St, execute_st);
    reg.register(StmtKind::Tcgen05WaitSt, execute_wait);
    reg.register(StmtKind::Tcgen05Mma, execute_mma);
    reg.register(StmtKind::Tcgen05Cp, execute_cp);
}

fn execute_wait<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    if ctx.trace_mode() {
        let async_kind = match stmt {
            Stmt::Tcgen05WaitLd => TmemAsyncKind::Ld,
            Stmt::Tcgen05WaitSt => TmemAsyncKind::St,
            _ => unreachable!(),
        };
        ctx.emit(TraceEventKind::TmemWait {
            async_kind,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn execute_commit<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (mbar, stage, cta_group, multicast) = match stmt {
        Stmt::Tcgen05Commit {
            mbar,
            stage,
            cta_group,
            multicast_cta_mask,
        } => (mbar, stage, *cta_group, *multicast_cta_mask),
        _ => unreachable!(),
    };
    let base = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    let targets: Vec<MbarTarget> = match multicast {
        None => vec![base],
        Some(mask) => multicast_target_ctas(ctx, mask, "tcgen05", "Tcgen05Commit")?
            .into_iter()
            .map(|c| retarget_mbar(base, c))
            .collect(),
    };
    // peer-active gate for cta_group=2 (polled — re-checked each round)
    if cta_group == 2 {
        let peer = peer_ctaid_in_cluster(
            ctx,
            ctx.stream.ctaid_in_cluster,
            "tcgen05_peer",
            "tcgen05 peer out of range",
        )?;
        match ctx.cta_activity(ctx.global_cta_id(peer)) {
            CtaActivityStatus::Missing => {
                return Err(InterpreterError::new(
                    "tcgen05_peer_missing",
                    "tcgen05 peer CTA is missing",
                ))
            }
            CtaActivityStatus::Exited => {
                return Err(InterpreterError::new(
                    "tcgen05_peer_exited",
                    "tcgen05 peer CTA has exited",
                ))
            }
            CtaActivityStatus::Active => {}
            CtaActivityStatus::NotStarted => {
                return Ok(StepStatus::block(WakeCondition::Polled));
            }
        }
    }
    let mut cells: HashMap<MbarCellKey, MbarCell> = HashMap::new();
    for target in &targets {
        let key = target.key();
        let cell = match cells.get(&key) {
            Some(c) => *c,
            None => initialized_mbar_cell(ctx, key)?,
        };
        if ctx.trace_mode() {
            ctx.emit(TraceEventKind::MbarArrive {
                target: (*target).into(),
                count: 1,
                scope: ctx.access_scope(),
            })?;
        }
        cells.insert(key, arrive_mbarrier_cell(cell, 1)?);
    }
    let mut keys = Vec::with_capacity(cells.len());
    for (key, cell) in cells {
        ctx.state.values.mbars.cells.insert(key, cell);
        keys.push(key);
    }
    Ok(StepStatus::advance_wake(keys))
}

// ---- datapath (ld/st) ----

#[derive(Clone, Copy)]
struct DatapathBounds {
    reg_size: usize,
    col_start: usize,
    col_end: usize,
}

/// `.32x32b` atoms address a whole 32-lane subpartition, so their taddr lane
/// corner must be 32-aligned (B200-verified: lane=16 faults). The 16-lane
/// `.16x*b` atoms cover HALF a subpartition; a second issue with row=16 covers
/// lanes 16..31 of each warp's partition (the TIRx M=128 two-slab fragment).
fn check_row_alignment(row: i64, shape: &str, label: &str) -> IResult<()> {
    let align = if shape.starts_with("16x") { 16 } else { 32 };
    if row % align != 0 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_misaligned"),
            format!("tcgen05 row must be {align}-aligned for {shape}"),
        ));
    }
    Ok(())
}

fn resolve_datapath(
    ctx: &CohortContext,
    stmt_row: &crate::ir::ScalarValue,
    stmt_col: &crate::ir::ScalarValue,
    shape: &str,
    num: u32,
    tmem_tensor: &crate::ir::Tensor,
    label: &str,
) -> IResult<(Array2<usize>, Array2<usize>, usize)> {
    ctx.check_full_warp_cohort(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    let row = ctx.eval_scalar_uniform(stmt_row, "tcgen05 row", "divergent_operands")?;
    let col = ctx.eval_scalar_uniform(stmt_col, "tcgen05 col", "divergent_operands")?;
    check_row_alignment(row, shape, label)?;
    let (lane_idx, col_idx) = datapath_index_arrays_cached(shape, num as usize)?;
    let reg_size = lane_idx.ncols();
    let col_start = tmem_layout_for(tmem_tensor)?.col_start;
    let a = ctx.cohort.len();
    let mut lanes = Array2::<usize>::zeros((a, reg_size));
    let mut cols = Array2::<usize>::zeros((a, reg_size));
    for (ai, t) in ctx.cohort.iter().enumerate() {
        let subpart = (32 * (t.warp_id % 4)) as i64;
        for r in 0..reg_size {
            let lane = row + subpart + lane_idx[[t.lane_id, r]] as i64;
            let col = col_start as i64 + col + col_idx[[t.lane_id, r]] as i64;
            if lane < 0 || lane >= TMEM_ROWS as i64 || col < 0 || col >= TMEM_COLS as i64 {
                return Err(InterpreterError::new(
                    format!("tcgen05_{label}_out_of_range"),
                    format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
                ));
            }
            lanes[[ai, r]] = lane as usize;
            cols[[ai, r]] = col as usize;
        }
    }
    Ok((lanes, cols, reg_size))
}

fn resolve_datapath_bounds(
    ctx: &CohortContext,
    stmt_row: &crate::ir::ScalarValue,
    stmt_col: &crate::ir::ScalarValue,
    shape: &str,
    num: u32,
    tmem_tensor: &crate::ir::Tensor,
    label: &str,
) -> IResult<DatapathBounds> {
    ctx.check_full_warp_cohort(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    let row = ctx.eval_scalar_uniform(stmt_row, "tcgen05 row", "divergent_operands")?;
    let col = ctx.eval_scalar_uniform(stmt_col, "tcgen05 col", "divergent_operands")?;
    check_row_alignment(row, shape, label)?;
    let summary = datapath_index_summary_cached(shape, num as usize)?;
    let layout_col_start = tmem_layout_for(tmem_tensor)?.col_start;
    let abs_col_start = layout_col_start as i64 + col + summary.col_min as i64;
    let abs_col_end = layout_col_start as i64 + col + summary.col_max as i64;
    if abs_col_start < 0 || abs_col_end >= TMEM_COLS as i64 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_out_of_range"),
            format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
        ));
    }
    let mut checked_warps = Vec::new();
    for t in &ctx.cohort {
        if checked_warps.contains(&t.warp_id) {
            continue;
        }
        checked_warps.push(t.warp_id);
        let subpart = (32 * (t.warp_id % 4)) as i64;
        let lane_start = row + subpart + summary.lane_min as i64;
        let lane_end = row + subpart + summary.lane_max as i64;
        if lane_start < 0 || lane_end >= TMEM_ROWS as i64 {
            return Err(InterpreterError::new(
                format!("tcgen05_{label}_out_of_range"),
                format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
            ));
        }
    }
    if checked_warps.is_empty() {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_mask"),
            format!("tcgen05_{label} must be issued by one or more full warps"),
        ));
    }
    Ok(DatapathBounds {
        reg_size: summary.reg_size,
        col_start: abs_col_start as usize,
        col_end: abs_col_end as usize,
    })
}

fn trace_ldst_tmem_region(
    ctx: &CohortContext,
    stmt_row: &crate::ir::ScalarValue,
    stmt_col: &crate::ir::ScalarValue,
    shape: &str,
    num: u32,
    tmem_tensor: &crate::ir::Tensor,
    label: &str,
) -> IResult<(Region, usize)> {
    ctx.check_full_warp_cohort(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    let row = ctx.eval_scalar_uniform(stmt_row, "tcgen05 row", "divergent_operands")?;
    let col = ctx.eval_scalar_uniform(stmt_col, "tcgen05 col", "divergent_operands")?;
    check_row_alignment(row, shape, label)?;
    let summary = datapath_index_summary_cached(shape, num as usize)?;
    let layout_col_start = tmem_layout_for(tmem_tensor)?.col_start;
    let col_start = layout_col_start as i64 + col + summary.col_min as i64;
    let col_end = layout_col_start as i64 + col + summary.col_max as i64 + 1;
    if col_start < 0 || col_end > TMEM_COLS as i64 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_out_of_range"),
            format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
        ));
    }

    let mut subparts = Vec::new();
    let mut rects = Vec::new();
    for thread in &ctx.cohort {
        let subpart = thread.warp_id % 4;
        if subparts.contains(&subpart) {
            continue;
        }
        subparts.push(subpart);
        let lane_start = row + (32 * subpart) as i64 + summary.lane_min as i64;
        let lane_end = row + (32 * subpart) as i64 + summary.lane_max as i64 + 1;
        if lane_start < 0 || lane_end > TMEM_ROWS as i64 {
            return Err(InterpreterError::new(
                format!("tcgen05_{label}_out_of_range"),
                format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
            ));
        }
        rects.push((
            lane_start as usize,
            (lane_end - lane_start) as usize,
            col_start as usize,
            (col_end - col_start) as usize,
        ));
    }
    if rects.is_empty() {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_mask"),
            format!("tcgen05_{label} must be issued by one or more full warps"),
        ));
    }
    Ok((
        region::tmem_region_from_rects(tmem_tensor.id, ctx.stream.cta_id, rects)?,
        summary.reg_size,
    ))
}

fn execute_ld<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, num, row, col) = match stmt {
        Stmt::Tcgen05Ld {
            dst,
            src,
            shape,
            num,
            row,
            col,
        } => (dst, src, shape_str(shape), *num, row, col),
        _ => unreachable!(),
    };
    if ctx.trace_mode() && !ctx.state.trace.records_events() {
        let t_resolve = super::super::runner::prof_now();
        let bounds = resolve_datapath_bounds(ctx, row, col, shape, num, src, "ld")?;
        super::super::runner::prof_end("TcLd:resolve", t_resolve);
        let t_dst = super::super::runner::prof_now();
        let dst_r = ctx.eval_slice(dst)?;
        check_reg_fragment(&dst_r, src, bounds.reg_size, "ld")?;
        if is_packed_tmem_dtype(src.dtype) {
            let _ = packed_half_register_slice(&dst_r, bounds.reg_size, "ld")?;
        }
        super::super::runner::prof_end("TcLd:dst_check", t_dst);
        let t_alloc = super::super::runner::prof_now();
        if !tmem_col_range_allocated(ctx, ctx.stream.cta_id, bounds.col_start, bounds.col_end) {
            let (_, cols, _) = resolve_datapath(ctx, row, col, shape, num, src, "ld")?;
            check_tmem_cells_allocated(ctx, ctx.stream.cta_id, cols.iter().copied(), "tcgen05_ld")?;
        }
        super::super::runner::prof_end("TcLd:alloc_check", t_alloc);
        return Ok(StepStatus::advance());
    }
    if ctx.trace_mode() {
        let t_resolve = super::super::runner::prof_now();
        let (region, reg_size) = trace_ldst_tmem_region(ctx, row, col, shape, num, src, "ld")?;
        super::super::runner::prof_end("TcLd:resolve", t_resolve);
        let t_dst = super::super::runner::prof_now();
        let dst_r = ctx.eval_slice(dst)?;
        check_reg_fragment(&dst_r, src, reg_size, "ld")?;
        let packed_dst = if is_packed_tmem_dtype(src.dtype) {
            Some(packed_half_register_slice(&dst_r, reg_size, "ld")?)
        } else {
            None
        };
        super::super::runner::prof_end("TcLd:dst_check", t_dst);
        let t_alloc = super::super::runner::prof_now();
        check_tmem_region_allocated(ctx, &region, "tcgen05_ld")?;
        super::super::runner::prof_end("TcLd:alloc_check", t_alloc);
        let scope = ctx.access_scope();
        ctx.emit(TraceEventKind::Read {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Ld),
            scope: scope.clone(),
        })?;
        ctx.emit_tensor_write(packed_dst.as_ref().unwrap_or(&dst_r))?;
        return Ok(StepStatus::advance());
    }
    let t_resolve = super::super::runner::prof_now();
    let (lanes, cols, reg_size) = resolve_datapath(ctx, row, col, shape, num, src, "ld")?;
    super::super::runner::prof_end("TcLd:resolve", t_resolve);
    let t_dst = super::super::runner::prof_now();
    let dst_r = ctx.eval_slice(dst)?;
    check_reg_fragment(&dst_r, src, reg_size, "ld")?;
    let packed_dst = if is_packed_tmem_dtype(src.dtype) {
        Some(packed_half_register_slice(&dst_r, reg_size, "ld")?)
    } else {
        None
    };
    super::super::runner::prof_end("TcLd:dst_check", t_dst);
    let flat_lanes: Vec<usize> = lanes.iter().copied().collect();
    let flat_cols: Vec<usize> = cols.iter().copied().collect();
    let a = ctx.cohort.len();
    if is_packed_tmem_dtype(src.dtype) {
        let values = {
            let scratch = ctx.state.values.tmem.scratchpad_for(ctx.stream.cta_id)?;
            let timer = super::super::runner::prof_now();
            let values = scratch.read_packed_half_cells(src, &flat_lanes, &flat_cols)?;
            super::super::runner::prof_end("TcLd:tmem_read_cells", timer);
            values
        };
        let mut flat = Vec::with_capacity(a * reg_size * 2);
        for (lo, hi) in values {
            flat.push(lo);
            flat.push(hi);
        }
        let native = ValueArray1::from_f32_compute(Array1::from(flat), src.dtype)
            .reshape2((a, reg_size * 2))?;
        let timer = super::super::runner::prof_now();
        ctx.registers_write(packed_dst.as_ref().unwrap(), &native)?;
        super::super::runner::prof_end("TcLd:reg_write", timer);
    } else {
        let values = {
            let scratch = ctx.state.values.tmem.scratchpad_for(ctx.stream.cta_id)?;
            let timer = super::super::runner::prof_now();
            let values = scratch.read_cells(src, &flat_lanes, &flat_cols)?;
            super::super::runner::prof_end("TcLd:tmem_read_cells", timer);
            values
        };
        let native = values.reshape2((a, reg_size))?;
        let timer = super::super::runner::prof_now();
        ctx.registers_write(&dst_r, &native)?;
        super::super::runner::prof_end("TcLd:reg_write", timer);
    }
    Ok(StepStatus::advance())
}

fn execute_st<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, num, row, col) = match stmt {
        Stmt::Tcgen05St {
            dst,
            src,
            shape,
            num,
            row,
            col,
        } => (dst, src, shape_str(shape), *num, row, col),
        _ => unreachable!(),
    };
    if ctx.trace_mode() && !ctx.state.trace.records_events() {
        let t_resolve = super::super::runner::prof_now();
        let bounds = resolve_datapath_bounds(ctx, row, col, shape, num, dst, "st")?;
        super::super::runner::prof_end("TcSt:resolve", t_resolve);
        let t_src = super::super::runner::prof_now();
        let src_r = ctx.eval_slice(src)?;
        check_reg_fragment(&src_r, dst, bounds.reg_size, "st")?;
        if is_packed_tmem_dtype(dst.dtype) {
            let _ = packed_half_register_slice(&src_r, bounds.reg_size, "st")?;
        }
        super::super::runner::prof_end("TcSt:src_check", t_src);
        let t_overlap = super::super::runner::prof_now();
        check_tmem_st_no_overlap(ctx, shape, num)?;
        super::super::runner::prof_end("TcSt:overlap", t_overlap);
        let t_alloc = super::super::runner::prof_now();
        if !tmem_col_range_allocated(ctx, ctx.stream.cta_id, bounds.col_start, bounds.col_end) {
            let (_, cols, _) = resolve_datapath(ctx, row, col, shape, num, dst, "st")?;
            check_tmem_cells_allocated(ctx, ctx.stream.cta_id, cols.iter().copied(), "tcgen05_st")?;
        }
        super::super::runner::prof_end("TcSt:alloc_check", t_alloc);
        return Ok(StepStatus::advance());
    }
    if ctx.trace_mode() {
        let t_resolve = super::super::runner::prof_now();
        let (region, reg_size) = trace_ldst_tmem_region(ctx, row, col, shape, num, dst, "st")?;
        super::super::runner::prof_end("TcSt:resolve", t_resolve);
        let t_src = super::super::runner::prof_now();
        let src_r = ctx.eval_slice(src)?;
        check_reg_fragment(&src_r, dst, reg_size, "st")?;
        let packed_src = if is_packed_tmem_dtype(dst.dtype) {
            Some(packed_half_register_slice(&src_r, reg_size, "st")?)
        } else {
            None
        };
        super::super::runner::prof_end("TcSt:src_check", t_src);
        let t_overlap = super::super::runner::prof_now();
        check_tmem_st_no_overlap(ctx, shape, num)?;
        super::super::runner::prof_end("TcSt:overlap", t_overlap);
        let t_alloc = super::super::runner::prof_now();
        check_tmem_region_allocated(ctx, &region, "tcgen05_st")?;
        super::super::runner::prof_end("TcSt:alloc_check", t_alloc);
        let scope = ctx.access_scope();
        ctx.emit_tensor_read(packed_src.as_ref().unwrap_or(&src_r))?;
        ctx.emit(TraceEventKind::Write {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
            scope,
        })?;
        return Ok(StepStatus::advance());
    }
    let t_resolve = super::super::runner::prof_now();
    let (lanes, cols, reg_size) = resolve_datapath(ctx, row, col, shape, num, dst, "st")?;
    super::super::runner::prof_end("TcSt:resolve", t_resolve);
    let t_src = super::super::runner::prof_now();
    let src_r = ctx.eval_slice(src)?;
    check_reg_fragment(&src_r, dst, reg_size, "st")?;
    let packed_src = if is_packed_tmem_dtype(dst.dtype) {
        Some(packed_half_register_slice(&src_r, reg_size, "st")?)
    } else {
        None
    };
    super::super::runner::prof_end("TcSt:src_check", t_src);
    let t_overlap = super::super::runner::prof_now();
    check_tmem_st_no_overlap(ctx, shape, num)?;
    super::super::runner::prof_end("TcSt:overlap", t_overlap);
    let t_flat = super::super::runner::prof_now();
    let flat_lanes: Vec<usize> = lanes.iter().copied().collect();
    let flat_cols: Vec<usize> = cols.iter().copied().collect();
    super::super::runner::prof_end("TcSt:flatten", t_flat);
    if is_packed_tmem_dtype(dst.dtype) {
        let timer = super::super::runner::prof_now();
        let packed_values =
            read_packed_half_register_pairs(ctx, packed_src.as_ref().unwrap(), reg_size)?;
        super::super::runner::prof_end("TcSt:reg_read", timer);
        let cta_id = ctx.stream.cta_id;
        let sp = ctx
            .state
            .values
            .tmem
            .by_cta
            .get_mut(&cta_id)
            .ok_or_else(|| {
                InterpreterError::new(
                    "missing_tmem_scratchpad",
                    "tcgen05.st writes a missing TMEM scratchpad",
                )
            })?;
        let timer = super::super::runner::prof_now();
        sp.write_packed_half_cells(dst, &flat_lanes, &flat_cols, &packed_values)?;
        super::super::runner::prof_end("TcSt:tmem_write_cells", timer);
    } else {
        let timer = super::super::runner::prof_now();
        let flat_values = ctx.registers_read(&src_r)?.flatten_to_1d();
        super::super::runner::prof_end("TcSt:reg_read", timer);
        let cta_id = ctx.stream.cta_id;
        let sp = ctx
            .state
            .values
            .tmem
            .by_cta
            .get_mut(&cta_id)
            .ok_or_else(|| {
                InterpreterError::new(
                    "missing_tmem_scratchpad",
                    "tcgen05.st writes a missing TMEM scratchpad",
                )
            })?;
        let timer = super::super::runner::prof_now();
        sp.write_cells(dst, &flat_lanes, &flat_cols, &flat_values)?;
        super::super::runner::prof_end("TcSt:tmem_write_cells", timer);
    }
    Ok(StepStatus::advance())
}

fn read_packed_half_register_pairs(
    ctx: &CohortContext,
    src_r: &ResolvedSlice,
    reg_size: usize,
) -> IResult<Vec<(f32, f32)>> {
    let values = ctx.registers_read(src_r)?.to_f32_compute();
    debug_assert_eq!(values.ncols(), reg_size * 2);
    let mut out = Vec::with_capacity(values.nrows() * reg_size);
    for ai in 0..values.nrows() {
        for r in 0..reg_size {
            out.push((values[[ai, 2 * r]], values[[ai, 2 * r + 1]]));
        }
    }
    Ok(out)
}

fn is_packed_tmem_dtype(dtype: crate::ir::DType) -> bool {
    matches!(dtype, crate::ir::DType::F16 | crate::ir::DType::Bf16)
}

fn packed_half_register_slice(
    reg_slice: &ResolvedSlice,
    reg_size: usize,
    label: &str,
) -> IResult<ResolvedSlice> {
    if reg_slice.tensor.shape.len() == 1
        && reg_slice.offsets.ncols() == 1
        && reg_slice.shape.as_slice() == [reg_size]
        && reg_slice
            .offsets
            .column(0)
            .iter()
            .all(|&o| o >= 0 && (o as usize) + reg_size * 2 <= reg_slice.tensor.shape[0])
    {
        return Ok(ResolvedSlice {
            tensor: reg_slice.tensor.clone(),
            offsets: reg_slice.offsets.clone(),
            shape: vec![reg_size * 2],
        });
    }
    Err(InterpreterError::new(
        format!("tcgen05_{label}_shape"),
        format!(
            "tcgen05_{label} packed half register slice requires {reg_size} b32 registers backed by {} half values",
            reg_size * 2
        ),
    ))
}

fn check_reg_fragment(
    reg_slice: &ResolvedSlice,
    tmem_tensor: &crate::ir::Tensor,
    reg_size: usize,
    label: &str,
) -> IResult<()> {
    if reg_slice.tensor.dtype != tmem_tensor.dtype {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_dtype"),
            format!("tcgen05_{label} REG and TMEM operands must share a dtype"),
        ));
    }
    if reg_slice.shape.as_slice() != [reg_size].as_slice() {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_shape"),
            format!("tcgen05_{label} register slice must have shape ({reg_size},)"),
        ));
    }
    Ok(())
}

fn check_tmem_st_no_overlap(ctx: &CohortContext, shape: &str, num: u32) -> IResult<()> {
    if datapath_has_cell_aliases_cached(shape, num as usize)? {
        return Err(InterpreterError::new(
            "overlapping_tmem_write",
            "tcgen05.st overlaps a TMEM cell",
        ));
    }
    let mut subpart_owner = [None; 4];
    for thread in &ctx.cohort {
        let subpart = thread.warp_id % 4;
        match subpart_owner[subpart] {
            Some(owner) if owner == thread.warp_id => {}
            Some(_) => {
                return Err(InterpreterError::new(
                    "overlapping_tmem_write",
                    "tcgen05.st overlaps a TMEM cell",
                ));
            }
            None => subpart_owner[subpart] = Some(thread.warp_id),
        }
    }
    Ok(())
}

fn check_tmem_region_allocated(ctx: &CohortContext, region: &Region, label: &str) -> IResult<()> {
    let PoolId::Tmem { cta_id } = region.owner else {
        return Err(InterpreterError::new(
            "trace_region_owner",
            "TMEM region has a non-TMEM owner",
        ));
    };
    let crate::interpreter::protocol::RegionBoxes::Boxes(region_boxes) = &region.boxes else {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "TMEM region must use (lane, lane_byte) boxes",
        ));
    };
    for b in region_boxes {
        if b.ranges.len() != 2 || b.ranges[1].1 > region::TMEM_LANE_BYTES {
            return Err(InterpreterError::new(
                "trace_region_rank",
                "TMEM region must use (lane, lane_byte) boxes",
            ));
        }
        let col_start = b.ranges[1].0 / 4;
        let col_end = b.ranges[1].1.div_ceil(4);
        let covered = ctx.state.tmem_allocations.keys().any(|key| {
            key.cta_id == cta_id
                && key.col_start <= col_start
                && col_end <= key.col_start + key.n_cols
        });
        if !covered {
            return Err(InterpreterError::new(
                "missing_tmem_allocation",
                format!("{label} accesses a TMEM range without an active allocation"),
            ));
        }
    }
    Ok(())
}

fn tmem_col_range_allocated(
    ctx: &CohortContext,
    cta_id: usize,
    col_start: usize,
    col_end: usize,
) -> bool {
    ctx.state.tmem_allocations.keys().any(|key| {
        key.cta_id == cta_id && key.col_start <= col_start && col_end < key.col_start + key.n_cols
    })
}

fn check_tmem_cells_allocated<I>(
    ctx: &CohortContext,
    cta_id: usize,
    cols: I,
    label: &str,
) -> IResult<()>
where
    I: IntoIterator<Item = usize> + Clone,
{
    let mut range = None::<(usize, usize)>;
    for col in cols.clone() {
        range = Some(match range {
            Some((lo, hi)) => (lo.min(col), hi.max(col)),
            None => (col, col),
        });
    }
    if let Some((col_start, col_end)) = range {
        let covered = ctx.state.tmem_allocations.keys().any(|key| {
            key.cta_id == cta_id
                && key.col_start <= col_start
                && col_end < key.col_start + key.n_cols
        });
        if covered {
            return Ok(());
        }
    } else {
        return Ok(());
    }

    let mut cols: Vec<usize> = cols.into_iter().collect();
    cols.sort_unstable();
    cols.dedup();
    for col in cols {
        let covered = ctx.state.tmem_allocations.keys().any(|key| {
            key.cta_id == cta_id && key.col_start <= col && col < key.col_start + key.n_cols
        });
        if !covered {
            return Err(InterpreterError::new(
                "missing_tmem_allocation",
                format!("{label} accesses a TMEM range without an active allocation"),
            ));
        }
    }
    Ok(())
}

fn shape_str(shape: &crate::ir::LdStShape) -> &'static str {
    shape.as_str()
}

// ---- MMA ----

fn execute_mma<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, a_sl, b_sl, m, n, k, accum, trans_a, trans_b, cta_group, sfa, sfb, sf_byte) =
        match stmt {
            Stmt::Tcgen05Mma {
                dst,
                a,
                b,
                m,
                n,
                k,
                accum,
                trans_a,
                trans_b,
                cta_group,
                sfa,
                sfb,
                sf_byte,
            } => (
                dst,
                a,
                b,
                *m as usize,
                *n as usize,
                *k as usize,
                *accum,
                *trans_a,
                *trans_b,
                *cta_group,
                sfa.as_ref(),
                sfb.as_ref(),
                *sf_byte as usize,
            ),
            _ => unreachable!(),
        };
    ctx.check_full_warp_cohort(
        "tcgen05_mma_mask",
        "tcgen05_mma must be issued by one or more full warps",
    )?;

    // The accumulator CTA(s): cta_group=2's even CTA computes the whole pair; odd is a no-op.
    let cta_ids: Vec<usize> = if cta_group == 2 {
        if m != 128 && m != 256 {
            return Err(InterpreterError::new(
                "tcgen05_mma_unsupported",
                "tcgen05_mma cta_group=2 value model supports only m in {128, 256}",
            ));
        }
        if ctx.cohort[0].ctaid_in_cluster & 1 == 1 {
            return Ok(StepStatus::advance());
        }
        let peer_local = peer_ctaid_in_cluster(
            ctx,
            ctx.cohort[0].ctaid_in_cluster,
            "tcgen05_mma_peer",
            "mma peer out of range",
        )?;
        vec![ctx.stream.cta_id, ctx.global_cta_id(peer_local)]
    } else {
        if m != 64 && m != 128 {
            return Err(InterpreterError::new(
                "tcgen05_mma_unsupported",
                "tcgen05_mma cta_group=1 value model supports only m in {64, 128}",
            ));
        }
        vec![ctx.stream.cta_id]
    };

    if ctx.trace_mode() {
        trace_mma(
            ctx, dst, a_sl, b_sl, m, n, k, cta_group, accum, trans_a, trans_b, &cta_ids, sfa, sfb,
        )?;
        return Ok(StepStatus::advance());
    }
    if dst.tensor.dtype != crate::ir::DType::F32 {
        return Err(InterpreterError::new(
            "tcgen05_mma_dtype",
            "tcgen05_mma dst dtype must be f32",
        ));
    }
    // Contiguous layouts (D: cta_group=1 m=128; A: cta_group=2 m=256). Non-transposed
    // rank-2 operands take the in-place path: SMEM bytes are materialized into
    // reusable f32 scratch, then sgemm writes f32 bits directly into TMEM cells.
    let contiguous = (cta_group == 1 && m == 128) || (cta_group == 2 && m == 256);
    let inplace_ok = contiguous
        && !trans_a
        && !trans_b
        && a_sl.tensor.space == MemorySpace::Smem
        && b_sl.tensor.space == MemorySpace::Smem
        && a_sl.offsets.len() >= 2
        && b_sl.offsets.len() >= 2;
    if inplace_ok {
        let scales = match (sfa, sfb) {
            (Some(sfa_sl), Some(sfb_sl)) => Some((sfa_sl, sfb_sl, sf_byte)),
            _ => None,
        };
        accumulate_inplace(
            ctx, dst, a_sl, b_sl, m, n, k, cta_group, accum, &cta_ids, scales,
        )?;
        return Ok(StepStatus::advance());
    }

    let t_read = super::super::runner::prof_now();
    let a = read_operand_ctas(ctx, a_sl, trans_a, &cta_ids)?;
    super::super::runner::prof_end("MMA:fallback_read_a", t_read);
    let t_read = super::super::runner::prof_now();
    let b = read_operand_ctas(ctx, b_sl, trans_b, &cta_ids)?;
    super::super::runner::prof_end("MMA:fallback_read_b", t_read);
    check_mma_operand_shapes(&a, &b, m, n, k)?;
    // Fallback for everything `accumulate_inplace` can't take (transposed or
    // non-rank-2 operands): a full product then a per-cell scatter via the layout
    // placement table, which handles every (m, cta_group) layout.
    let t_matmul = super::super::runner::prof_now();
    let mut product = matmul_f32(&a, &b);
    super::super::runner::prof_end("MMA:fallback_matmul", t_matmul);
    if let (Some(sfa_sl), Some(sfb_sl)) = (sfa, sfb) {
        // Block-scaled MMA: D += (A . 2^(sfa[m]-127)) @ (B . 2^(sfb[n]-127))^T. The
        // scales are per row and constant over this MMA's k-slice, so scaling the
        // product is exact (powers of two). A's scales are split by M across the
        // CTA pair like A itself; B's scales are duplicated per CTA and each CTA's
        // M-half applies its own copy (what the 2-CTA datapath reads).
        let rows_per_cta = m / cta_ids.len();
        let mut sa: Vec<f32> = Vec::with_capacity(m);
        for &cta in cta_ids.iter() {
            sa.extend(read_scale_rows(ctx, sfa_sl, sf_byte, rows_per_cta, cta)?);
        }
        let mut sb: Vec<Vec<f32>> = Vec::with_capacity(cta_ids.len());
        for &cta in cta_ids.iter() {
            sb.push(read_scale_rows(ctx, sfb_sl, sf_byte, n, cta)?);
        }
        for mm in 0..m {
            let half = mm / rows_per_cta;
            let s_a = sa[mm];
            for nn in 0..n {
                product[[mm, nn]] *= s_a * sb[half][nn];
            }
        }
    }
    let t_acc = super::super::runner::prof_now();
    accumulate_blocks(ctx, dst, &product, m, n, cta_group, accum, &cta_ids)?;
    super::super::runner::prof_end("MMA:fallback_accumulate", t_acc);
    Ok(StepStatus::advance())
}

#[allow(clippy::too_many_arguments)]
fn trace_mma(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    a_sl: &TensorSlice,
    b_sl: &TensorSlice,
    m: usize,
    n: usize,
    k: usize,
    cta_group: u8,
    accum: bool,
    trans_a: bool,
    trans_b: bool,
    cta_ids: &[usize],
    sfa: Option<&TensorSlice>,
    sfb: Option<&TensorSlice>,
) -> IResult<()> {
    let a_r = ctx.eval_slice(a_sl)?;
    let b_r = ctx.eval_slice(b_sl)?;
    let a_eff = trace_mma_effective_shape(&a_r.shape, trans_a, cta_ids.len())?;
    let b_eff = trace_mma_effective_shape(&b_r.shape, trans_b, cta_ids.len())?;
    if a_eff != (m, k) || b_eff != (n, k) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "tcgen05_mma operand shape does not match m/n/k",
        ));
    }
    let dst_off = eval_uniform_usize(ctx, &dst.offsets, "mma dst offset")?;
    if dst_off[0] != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_dst_offset",
            "mma dst row offset must be 0",
        ));
    }
    let layout = tmem_layout_for(&dst.tensor)?;
    let blocks = mma_blocks(
        m,
        n,
        cta_group,
        layout.lane_align,
        layout.col_start,
        dst_off[1],
        TMEM_ROWS,
        TMEM_COLS,
    )?;
    let regions = tmem_regions_from_mma_blocks(&dst.tensor, cta_ids, &blocks)?;
    for region in &regions {
        check_tmem_region_allocated(ctx, region, "tcgen05_mma")?;
    }
    let scope = ctx.access_scope();
    let operand_regions = mma_operand_regions(ctx, a_sl, &a_r, cta_ids)?
        .into_iter()
        .chain(mma_operand_regions(ctx, b_sl, &b_r, cta_ids)?);
    for region in operand_regions {
        ctx.emit(TraceEventKind::Read {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Mma),
            scope: scope.clone(),
        })?;
    }
    // Block-scaled MMA reads its UE8M0 scale vectors from every CTA in the pair
    // (A's scales split by M like A; B's duplicated per CTA): a TMEM read window
    // of the MMA, drained by tcgen05_commit like the accumulator access.
    if let (Some(sfa_sl), Some(sfb_sl)) = (sfa, sfb) {
        for sl in [sfa_sl, sfb_sl] {
            let sf_r = ctx.eval_slice(sl)?;
            let offsets = eval_uniform_usize(ctx, &sl.offsets, "mma scale offset")?;
            for &cta in cta_ids {
                let region =
                    region::tensor_region_from_uniform(&sf_r.tensor, cta, &offsets, &sf_r.shape)?;
                check_tmem_region_allocated(ctx, &region, "tcgen05_mma scale")?;
                ctx.emit(TraceEventKind::Read {
                    region,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                    scope: scope.clone(),
                })?;
            }
        }
    }
    for region in regions {
        if accum {
            ctx.emit(TraceEventKind::Read {
                region: region.clone(),
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                scope: scope.clone(),
            })?;
        }
        ctx.emit(TraceEventKind::Write {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
            scope: scope.clone(),
        })?;
    }
    Ok(())
}

/// Read `rows` per-row UE8M0 scales from a (128, cols) packed-u32 scale-vector
/// TMEM slice on one CTA: logical row r sits at cell (lane = r % 128,
/// col = col_start + col_offset + r / 128); byte `sf_byte` of the cell is the
/// biased exponent, scale = 2^(byte - 127).
fn read_scale_rows(
    ctx: &CohortContext,
    sl: &TensorSlice,
    sf_byte: usize,
    rows: usize,
    cta_id: usize,
) -> IResult<Vec<f32>> {
    let offsets = eval_uniform_usize(ctx, &sl.offsets, "mma scale offset")?;
    let shape = eval_uniform_usize(ctx, &sl.shape, "mma scale shape")?;
    if offsets.len() != 2 || shape.len() != 2 || offsets[0] != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_scale",
            "mma scale slice must be a (128, cols) TMEM slice at lane 0",
        ));
    }
    if rows > shape[0] * shape[1] {
        return Err(InterpreterError::new(
            "tcgen05_mma_scale",
            "mma scale slice does not cover the scaled rows",
        ));
    }
    let layout = tmem_layout_for(&sl.tensor)?;
    let col0 = layout.col_start + offsets[1];
    let sp = ctx.state.values.tmem.scratchpad_for(cta_id)?;
    let mut lanes = Vec::with_capacity(rows);
    let mut cols = Vec::with_capacity(rows);
    for r in 0..rows {
        lanes.push(r % 128);
        cols.push(col0 + r / 128);
    }
    let packed = sp.read_cells(&sl.tensor, &lanes, &cols)?;
    let ValueArray1::U32(packed) = packed else {
        return Err(InterpreterError::new(
            "tcgen05_mma_scale",
            "mma scale tensor dtype must be u32",
        ));
    };
    Ok(packed
        .iter()
        .map(|&cell| {
            let byte = ((cell >> (8 * sf_byte)) & 0xFF) as i32;
            ((byte - 127) as f32).exp2()
        })
        .collect())
}

/// `tcgen05.cp` — copy packed u32 scale cells from SMEM into TMEM. One leader
/// issue drives every CTA in the group: each CTA copies from its OWN SMEM into
/// its OWN TMEM (logical row r -> cell (lane = r % 128, col base + r / 128)).
/// The value lands at issue: the tcgen05 pipe executes its ops in issue order,
/// so a later same-stream MMA read can never observe a stale cell; retirement
/// toward other streams is observed through `tcgen05_commit` (the trace records
/// an async `Tmem(Cp)` write window drained by the commit).
fn execute_cp<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, cta_group) = match stmt {
        Stmt::Tcgen05Cp {
            dst,
            src,
            cta_group,
        } => (dst, src, *cta_group),
        _ => unreachable!(),
    };
    let cta_ids: Vec<usize> = if cta_group == 2 {
        if ctx.cohort[0].ctaid_in_cluster & 1 == 1 {
            return Ok(StepStatus::advance());
        }
        let peer_local = peer_ctaid_in_cluster(
            ctx,
            ctx.cohort[0].ctaid_in_cluster,
            "tcgen05_cp_peer",
            "cp peer out of range",
        )?;
        vec![ctx.stream.cta_id, ctx.global_cta_id(peer_local)]
    } else {
        vec![ctx.stream.cta_id]
    };
    let dst_off = eval_uniform_usize(ctx, &dst.offsets, "tcgen05_cp dst offset")?;
    let dst_r = ctx.eval_slice(dst)?;
    let src_off = eval_uniform_usize(ctx, &src.offsets, "tcgen05_cp src offset")?;
    let src_r = ctx.eval_slice(src)?;
    if dst_off.len() != 2 || dst_r.shape.len() != 2 || dst_off[0] != 0 || dst_r.shape[0] != 128 {
        return Err(InterpreterError::new(
            "tcgen05_cp_shape",
            "tcgen05_cp dst must be a (128, cols) TMEM slice at lane 0",
        ));
    }
    let count = numel(&dst_r.shape);
    if numel(&src_r.shape) != count {
        return Err(InterpreterError::new(
            "tcgen05_cp_shape",
            "tcgen05_cp src/dst element counts must match",
        ));
    }
    let layout = tmem_layout_for(&dst.tensor)?;
    let col0 = layout.col_start + dst_off[1];
    let mut lanes = Vec::with_capacity(count);
    let mut cols = Vec::with_capacity(count);
    for r in 0..count {
        lanes.push(r % 128);
        cols.push(col0 + r / 128);
    }
    let scope = ctx.access_scope();
    for &cta in &cta_ids {
        if ctx.trace_mode() {
            let src_region =
                region::tensor_region_from_uniform(&src_r.tensor, cta, &src_off, &src_r.shape)?;
            let dst_region =
                region::tensor_region_from_uniform(&dst_r.tensor, cta, &dst_off, &dst_r.shape)?;
            check_tmem_region_allocated(ctx, &dst_region, "tcgen05_cp")?;
            ctx.emit(TraceEventKind::Read {
                region: src_region,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Cp),
                scope: scope.clone(),
            })?;
            ctx.emit(TraceEventKind::Write {
                region: dst_region,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Cp),
                scope: scope.clone(),
            })?;
            continue;
        }
        let flat = ctx.state.values.smem.pool_for(cta)?.read_block(
            &src_r.tensor,
            &src_off,
            &src_r.shape,
        )?;
        let ValueArray1::U32(_) = &flat else {
            return Err(InterpreterError::new(
                "tcgen05_cp_dtype",
                "tcgen05_cp moves packed u32 scale cells",
            ));
        };
        ctx.state
            .values
            .tmem
            .by_cta
            .get_mut(&cta)
            .ok_or_else(|| {
                InterpreterError::new(
                    "missing_tmem_scratchpad",
                    "tcgen05_cp writes a missing TMEM scratchpad",
                )
            })?
            .write_cells(&dst_r.tensor, &lanes, &cols, &flat)?;
    }
    Ok(StepStatus::advance())
}

/// THE definition of which pools a pair-MMA operand read touches: the same
/// (offsets, shape) box — per-row runs, never one linear interval — in EVERY
/// CTA of `cta_ids` (cta_group=2 operands are split across the pair's pools;
/// cta_group=1 degenerates to the issuing CTA). The trace events iterate this
/// list and the value reads (`read_operand_ctas`, `accumulate_inplace`)
/// iterate the same `cta_ids` — deriving the CTA set independently per path
/// is how peer-SMEM reads went unrecorded.
fn mma_operand_regions(
    ctx: &CohortContext,
    slice: &TensorSlice,
    resolved: &ResolvedSlice,
    cta_ids: &[usize],
) -> IResult<Vec<Region>> {
    let offsets = eval_uniform_usize(ctx, &slice.offsets, "mma operand offset")?;
    let mut regions = Vec::with_capacity(cta_ids.len());
    for &cta in cta_ids {
        let region =
            region::tensor_region_from_uniform(&resolved.tensor, cta, &offsets, &resolved.shape)?;
        if matches!(resolved.tensor.space, MemorySpace::Tmem) {
            check_tmem_region_allocated(ctx, &region, "tcgen05_mma operand")?;
        }
        regions.push(region);
    }
    Ok(regions)
}

fn trace_mma_effective_shape(
    shape: &[usize],
    transpose: bool,
    cta_count: usize,
) -> IResult<(usize, usize)> {
    let lead = shape.len().saturating_sub(2);
    if shape.len() < 2 || shape[..lead].iter().any(|&d| d != 1) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "tcgen05_mma operand shape does not match m/n/k",
        ));
    }
    let (rows, cols) = (shape[lead], shape[lead + 1]);
    if transpose {
        Ok((cols * cta_count, rows))
    } else {
        Ok((rows * cta_count, cols))
    }
}

/// One TMEM block of an MMA accumulator: the output sub-rectangle
/// `[m_lo, m_lo+rows) x [n_lo, n_lo+cols)` maps element-for-element onto the contiguous
/// TMEM rectangle on `cta_ids[cta_idx]` at lanes `[lane_start, +rows)`, cols
/// `[col_start, +cols)` (element `(mi, ni)` → lane `lane_start + (mi - m_lo)`, col
/// `col_start + (ni - n_lo)`). Every supported MMA layout is 1–4 such blocks — there is
/// no per-element scatter.
#[derive(Debug)]
struct MmaBlock {
    cta_idx: usize,
    m_lo: usize,
    rows: usize,
    n_lo: usize,
    cols: usize,
    lane_start: usize,
    col_start: usize,
}

/// Closed-form block decomposition of an MMA accumulator layout (replaces the per-element
/// `mma_placement`/`mma_scatter_plan` table). `col_start` of each block already folds in
/// the layout's column base and the `dst_col` destination offset.
fn mma_blocks(
    m: usize,
    n: usize,
    cta_group: u8,
    lane_align: u8,
    layout_col_start: usize,
    dst_col: usize,
    n_lanes: usize,
    n_cols: usize,
) -> IResult<Vec<MmaBlock>> {
    let oob = || {
        InterpreterError::new(
            "tcgen05_mma_out_of_range",
            "tcgen05_mma addresses a TMEM cell out of range",
        )
    };
    let la = lane_align as usize;
    let base_col = layout_col_start.checked_add(dst_col).ok_or_else(oob)?;
    let mut blocks: Vec<MmaBlock> = Vec::new();
    if cta_group == 1 && m == 64 {
        // Layout F: four lane-runs of 16 at stride 32, plus the lane-align offset;
        // columns are the full n, contiguous.
        for g in 0..(m / 16) {
            blocks.push(MmaBlock {
                cta_idx: 0,
                m_lo: g * 16,
                rows: 16,
                n_lo: 0,
                cols: n,
                lane_start: 32 * g + la,
                col_start: base_col,
            });
        }
    } else if la != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_lane_align",
            "tcgen05_mma lane_align is only valid for the cta_group=1 m=64 (Layout F) accumulator",
        ));
    } else if cta_group == 1 && m == 128 {
        // Identity (Layout D): one contiguous block.
        blocks.push(MmaBlock {
            cta_idx: 0,
            m_lo: 0,
            rows: 128,
            n_lo: 0,
            cols: n,
            lane_start: 0,
            col_start: base_col,
        });
    } else if cta_group == 2 && m == 256 {
        // Layout A: split m across the CTA pair; each half is contiguous.
        for c in 0..2 {
            blocks.push(MmaBlock {
                cta_idx: c,
                m_lo: c * 128,
                rows: 128,
                n_lo: 0,
                cols: n,
                lane_start: 0,
                col_start: base_col,
            });
        }
    } else if cta_group == 2 && m == 128 {
        // Layout B: split m across the CTA pair, and split n in half — the second n-half
        // stacks into the upper 64 lanes (the "top/bottom" split).
        let half = (n / 2).max(1);
        for c in 0..2 {
            for h in 0..2 {
                let n_lo = h * half;
                if n_lo >= n {
                    continue;
                }
                blocks.push(MmaBlock {
                    cta_idx: c,
                    m_lo: c * 64,
                    rows: 64,
                    n_lo,
                    cols: (n - n_lo).min(half),
                    lane_start: h * 64,
                    col_start: base_col,
                });
            }
        }
    } else {
        return Err(InterpreterError::new(
            "tcgen05_mma_unsupported",
            "unsupported MMA layout",
        ));
    }
    for b in &blocks {
        if b.lane_start + b.rows > n_lanes || b.col_start + b.cols > n_cols {
            return Err(oob());
        }
    }
    Ok(blocks)
}

fn tmem_regions_from_mma_blocks(
    tensor: &crate::ir::Tensor,
    cta_ids: &[usize],
    blocks: &[MmaBlock],
) -> IResult<Vec<Region>> {
    let mut rects_by_cta: HashMap<usize, Vec<(usize, usize, usize, usize)>> = HashMap::new();
    for block in blocks {
        let Some(&cta_id) = cta_ids.get(block.cta_idx) else {
            return Err(InterpreterError::new(
                "tcgen05_mma_unsupported",
                "tcgen05_mma block has no destination CTA",
            ));
        };
        rects_by_cta.entry(cta_id).or_default().push((
            block.lane_start,
            block.rows,
            block.col_start,
            block.cols,
        ));
    }
    rects_by_cta
        .into_iter()
        .map(|(cta_id, rects)| region::tmem_region_from_rects(tensor.id, cta_id, rects))
        .collect()
}

/// (offsets, shape) of a staged operand box with its leading unit dims kept:
/// returns the FULL-rank offsets (for the pool's rect path) plus the trailing
/// (rows, cols) of the box, after checking the leading extents are all 1.
fn squeeze_operand(
    ctx: &CohortContext,
    sl: &TensorSlice,
) -> IResult<(Vec<usize>, Vec<usize>, usize, usize)> {
    let off = eval_uniform_usize(ctx, &sl.offsets, "mma operand offset")?;
    let shape = eval_uniform_usize(ctx, &sl.shape, "mma operand shape")?;
    let lead = shape.len().saturating_sub(2);
    if shape.len() < 2 || shape[..lead].iter().any(|&d| d != 1) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "tcgen05_mma operand shape does not match m/n/k",
        ));
    }
    let rows = shape[lead];
    let cols = shape[lead + 1];
    Ok((off, shape, rows, cols))
}

/// In-place contiguous accumulate: for each accumulator CTA, `sgemm(beta)` reads its
/// A tile and each B segment from SMEM f32 compute slices (base+lda) and writes the
/// result into the column-major TMEM grid. cta_group=2's B spans both CTAs' SMEM,
/// so each B half maps to one n-column range; the CTA set is the same `cta_ids`
/// the trace regions (`mma_operand_regions`) cover.
#[allow(clippy::too_many_arguments)]
fn accumulate_inplace(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    a_sl: &TensorSlice,
    b_sl: &TensorSlice,
    m: usize,
    n: usize,
    k: usize,
    cta_group: u8,
    accum: bool,
    cta_ids: &[usize],
    scales: Option<(&TensorSlice, &TensorSlice, usize)>,
) -> IResult<()> {
    let (a_off, a_box, a_rows, a_cols) = squeeze_operand(ctx, a_sl)?;
    let (b_off, b_box, b_rows, b_cols) = squeeze_operand(ctx, b_sl)?;
    let rows_per_cta = if cta_group == 1 { m } else { m / 2 };
    let n_seg = if cta_group == 1 { n } else { n / 2 };
    if [a_rows, a_cols] != [rows_per_cta, k] || [b_rows, b_cols] != [n_seg, k] {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "tcgen05_mma operand shape does not match m/n/k",
        ));
    }
    // UE8M0 scale vectors: A's split by M across the pair like A itself; B's
    // duplicated per CTA, each acc half applying its own copy. Scaling the
    // materialized operand rows by exact powers of two commutes bit-for-bit
    // with the fallback's product scaling.
    let (sa_scales, sb_scales) = match scales {
        Some((sfa_sl, sfb_sl, sf_byte)) => {
            let mut sa = Vec::with_capacity(cta_ids.len());
            let mut sb = Vec::with_capacity(cta_ids.len());
            for &cta in cta_ids {
                sa.push(read_scale_rows(ctx, sfa_sl, sf_byte, rows_per_cta, cta)?);
                sb.push(read_scale_rows(ctx, sfb_sl, sf_byte, n, cta)?);
            }
            (sa, sb)
        }
        None => (Vec::new(), Vec::new()),
    };
    let (c0, _) = inplace_geometry(ctx, dst, cta_group)?;
    let beta = if accum { 1.0 } else { 0.0 };
    let stride = super::super::values::tmem::TMEM_COLS;

    let n_total = n_seg * cta_ids.len();
    if rows_per_cta > TMEM_ROWS
        || match c0.checked_add(n_total) {
            Some(end) => end > TMEM_COLS,
            None => true,
        }
    {
        return Err(InterpreterError::new(
            "tcgen05_mma_out_of_range",
            "tcgen05_mma addresses a TMEM cell out of range",
        ));
    }
    for &cta_id in cta_ids {
        if !ctx.state.values.tmem.by_cta.contains_key(&cta_id) {
            return Err(InterpreterError::new(
                "missing_tmem_scratchpad",
                "tcgen05_mma writes a missing TMEM scratchpad",
            ));
        }
    }
    MMA_SCRATCH.with(|sc| -> IResult<()> {
        let mut g = sc.borrow_mut();
        let (sa, sb) = &mut *g;
        if scales.is_none() {
            // Unscaled: B is identical for every acc half — materialize once.
            sb.clear();
            for &b_cta in cta_ids {
                let pool = ctx.state.values.smem.pool_for(b_cta)?;
                pool.append_f32_block(&b_sl.tensor, &b_off, &b_box, sb)?;
            }
        }
        for (acc_idx, &acc_cta) in cta_ids.iter().enumerate() {
            if scales.is_some() {
                // Scaled: each acc half applies ITS OWN SFB copy, so the B
                // scratch is rebuilt (and row-scaled) per half.
                sb.clear();
                for &b_cta in cta_ids {
                    let pool = ctx.state.values.smem.pool_for(b_cta)?;
                    pool.append_f32_block(&b_sl.tensor, &b_off, &b_box, sb)?;
                }
                let sbv = &sb_scales[acc_idx];
                for (row, chunk) in sb.chunks_exact_mut(k).enumerate() {
                    let scale = sbv[row];
                    if scale != 1.0 {
                        for v in chunk.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
            }
            sa.clear();
            {
                let pool = ctx.state.values.smem.pool_for(acc_cta)?;
                pool.append_f32_block(&a_sl.tensor, &a_off, &a_box, sa)?;
            }
            if scales.is_some() {
                let sav = &sa_scales[acc_idx];
                for (row, chunk) in sa.chunks_exact_mut(k).enumerate() {
                    let scale = sav[row];
                    if scale != 1.0 {
                        for v in chunk.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
            }
            let sp = ctx
                .state
                .values
                .tmem
                .by_cta
                .get_mut(&acc_cta)
                .ok_or_else(|| {
                    InterpreterError::new(
                        "missing_tmem_scratchpad",
                        "tcgen05_mma writes a missing TMEM scratchpad",
                    )
                })?;
            let grid = sp.data_as_f32_mut();
            let mut col_off = 0usize;
            for seg_idx in 0..cta_ids.len() {
                super::super::blas::sgemm_nt_strided(
                    rows_per_cta,
                    n_seg,
                    k,
                    sa,
                    0,
                    k,
                    sb,
                    seg_idx * n_seg * k,
                    k,
                    grid,
                    c0 + col_off,
                    stride,
                    beta,
                );
                col_off += n_seg;
            }
            sp.valid
                .slice_mut(ndarray::s![0..rows_per_cta, c0..c0 + n_total])
                .fill(true);
        }
        Ok(())
    })
}

fn eval_uniform_usize(
    ctx: &CohortContext,
    vals: &[crate::ir::ScalarValue],
    label: &str,
) -> IResult<Vec<usize>> {
    vals.iter()
        .map(|v| {
            ctx.eval_scalar_uniform(v, label, "divergent_operands")
                .map(|x| x as usize)
        })
        .collect()
}

fn matmul_f32(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    // D[m,n] = A[m,k] @ B[n,k]ᵀ via OpenBLAS sgemm (non-contiguous F/B path).
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.nrows();
    let a_s = a.as_standard_layout();
    let b_s = b.as_standard_layout();
    let mut c = vec![0.0f32; m * n];
    super::super::blas::sgemm_nt(
        m,
        n,
        k,
        a_s.as_slice().unwrap(),
        b_s.as_slice().unwrap(),
        &mut c,
        0.0,
    );
    Array2::from_shape_vec((m, n), c).unwrap()
}

fn check_mma_operand_shapes(
    a: &Array2<f32>,
    b: &Array2<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> IResult<()> {
    if a.dim() != (m, k) || b.dim() != (n, k) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "tcgen05_mma operand shape does not match m/n/k",
        ));
    }
    Ok(())
}

fn read_operand(
    ctx: &CohortContext,
    sl: &TensorSlice,
    transpose: bool,
    cta_id: usize,
) -> IResult<Array2<f32>> {
    let _tr = super::super::runner::prof_now();
    let offsets: Vec<usize> = sl
        .offsets
        .iter()
        .map(|o| {
            ctx.eval_scalar_uniform(o, "mma operand offset", "divergent_operands")
                .map(|x| x as usize)
        })
        .collect::<IResult<_>>()?;
    let shape: Vec<usize> = sl
        .shape
        .iter()
        .map(|s| {
            ctx.eval_scalar_uniform(s, "mma operand shape", "divergent_operands")
                .map(|x| x as usize)
        })
        .collect::<IResult<_>>()?;
    // Staged operands arrive as (1, ..., rows, k) boxes of a stage-major
    // tensor (runtime pipeline stage in the leading offset); squeeze the
    // leading unit dims down to the rank-2 (rows, k) tile the MMA consumes.
    let lead = shape.len().saturating_sub(2);
    if shape.len() < 2 || shape[..lead].iter().any(|&d| d != 1) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "tcgen05_mma operand shape does not match m/n/k",
        ));
    }
    let shape2 = [shape[lead], shape[lead + 1]];
    super::super::runner::prof_end("O:resolve", _tr);
    let _td = super::super::runner::prof_now();
    let flat = match sl.tensor.space {
        MemorySpace::Smem => {
            let t_read = super::super::runner::prof_now();
            let flat = ctx
                .state
                .values
                .smem
                .pool_for(cta_id)?
                .read_block(&sl.tensor, &offsets, &shape)?;
            super::super::runner::prof_end("MMA:smem_read_block", t_read);
            flat
        }
        MemorySpace::Tmem => {
            let t_read = super::super::runner::prof_now();
            let flat = ctx
                .state
                .values
                .tmem
                .scratchpad_for(cta_id)?
                .read_slice(&sl.tensor, &offsets, &shape)?;
            super::super::runner::prof_end("MMA:tmem_read_slice", t_read);
            flat
        }
        _ => {
            return Err(InterpreterError::new(
                "tcgen05_mma_operand_space",
                "tcgen05_mma operands must be SMEM or TMEM in value mode",
            ))
        }
    };
    let t_convert = super::super::runner::prof_now();
    let mat = flat
        .to_f32_compute()
        .into_shape_with_order((shape2[0], shape2[1]))
        .unwrap();
    let out = if transpose { mat.t().to_owned() } else { mat };
    super::super::runner::prof_end("MMA:operand_to_f32", t_convert);
    super::super::runner::prof_end("C:mma_operand_read", _td);
    Ok(out)
}

/// Value-side mirror of `mma_operand_regions`: read the operand box from
/// EVERY CTA in `cta_ids` (row-concatenated in cta order; one CTA for
/// cta_group=1).
fn read_operand_ctas(
    ctx: &CohortContext,
    sl: &TensorSlice,
    transpose: bool,
    cta_ids: &[usize],
) -> IResult<Array2<f32>> {
    let halves: Vec<Array2<f32>> = cta_ids
        .iter()
        .map(|&c| read_operand(ctx, sl, transpose, c))
        .collect::<IResult<_>>()?;
    if halves.len() == 1 {
        return Ok(halves.into_iter().next().unwrap());
    }
    let views: Vec<_> = halves.iter().map(|h| h.view()).collect();
    ndarray::concatenate(ndarray::Axis(0), &views)
        .map_err(|_| InterpreterError::new("tcgen05_mma_shape", "mma operand pair concat failed"))
}

/// For each logical (m,n): (cta_idx, lane, col_local).
fn inplace_geometry(
    ctx: &CohortContext,
    dst: &TensorSlice,
    cta_group: u8,
) -> IResult<(usize, usize)> {
    let dst_off: Vec<usize> = dst
        .offsets
        .iter()
        .map(|o| {
            ctx.eval_scalar_uniform(o, "mma dst offset", "divergent_operands")
                .map(|x| x as usize)
        })
        .collect::<IResult<_>>()?;
    if dst_off[0] != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_dst_offset",
            "mma dst row offset must be 0",
        ));
    }
    let layout = tmem_layout_for(&dst.tensor)?;
    if layout.lane_align != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_lane_align",
            "tcgen05_mma lane_align is only valid for the cta_group=1 m=64 (Layout F) accumulator",
        ));
    }
    // Layout D: the whole accumulator (m rows); Layout A: 128 rows per CTA.
    let rows_per_cta = if cta_group == 1 {
        dst.tensor.shape[0]
    } else {
        128
    };
    let c0 = layout.col_start.checked_add(dst_off[1]).ok_or_else(|| {
        InterpreterError::new(
            "tcgen05_mma_out_of_range",
            "tcgen05_mma addresses a TMEM cell out of range",
        )
    })?;
    Ok((c0, rows_per_cta))
}

/// Non-contiguous Layout F/B (and transposed / non-rank-2 contiguous operands): take the
/// full product, then write each closed-form `mma_blocks` rectangle into TMEM. The block
/// coordinates ARE the layout — no per-element placement table.
#[allow(clippy::too_many_arguments)]
fn accumulate_blocks(
    ctx: &mut CohortContext,
    dst: &TensorSlice,
    product: &Array2<f32>,
    m: usize,
    n: usize,
    cta_group: u8,
    accum: bool,
    cta_ids: &[usize],
) -> IResult<()> {
    let dst_off: Vec<usize> = dst
        .offsets
        .iter()
        .map(|o| {
            ctx.eval_scalar_uniform(o, "mma dst offset", "divergent_operands")
                .map(|x| x as usize)
        })
        .collect::<IResult<_>>()?;
    if dst_off[0] != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_dst_offset",
            "mma dst row offset must be 0",
        ));
    }
    let dst_col = dst_off[1];
    let layout = tmem_layout_for(&dst.tensor)?;
    let first = cta_ids.first().ok_or_else(|| {
        InterpreterError::new(
            "tcgen05_mma_unsupported",
            "tcgen05_mma has no destination CTA",
        )
    })?;
    let scratchpad = ctx.state.values.tmem.by_cta.get(first).ok_or_else(|| {
        InterpreterError::new(
            "missing_tmem_scratchpad",
            "tcgen05_mma writes a missing TMEM scratchpad",
        )
    })?;
    let (n_lanes, n_cols) = scratchpad.data.dim();
    for &cta_id in cta_ids.iter().skip(1) {
        if !ctx.state.values.tmem.by_cta.contains_key(&cta_id) {
            return Err(InterpreterError::new(
                "missing_tmem_scratchpad",
                "tcgen05_mma writes a missing TMEM scratchpad",
            ));
        }
    }
    let blocks = mma_blocks(
        m,
        n,
        cta_group,
        layout.lane_align,
        layout.col_start,
        dst_col,
        n_lanes,
        n_cols,
    )?;

    if dst.tensor.dtype == crate::ir::DType::F32 {
        let timer = super::super::runner::prof_now();
        for block in &blocks {
            let cta_id = cta_ids[block.cta_idx];
            let sp = ctx
                .state
                .values
                .tmem
                .by_cta
                .get_mut(&cta_id)
                .ok_or_else(|| {
                    InterpreterError::new(
                        "missing_tmem_scratchpad",
                        "tcgen05_mma writes a missing TMEM scratchpad",
                    )
                })?;
            if !sp.accumulate_f32_cell_block_from(
                &dst.tensor,
                block.lane_start,
                block.rows,
                block.col_start,
                block.cols,
                product,
                block.m_lo,
                block.n_lo,
                accum,
            )? {
                break;
            }
        }
        super::super::runner::prof_end("MMA:direct_tmem_accumulate", timer);
        return Ok(());
    }

    Err(InterpreterError::new(
        "tcgen05_mma_dtype",
        "tcgen05_mma dst dtype must be f32",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mma_blocks_model_m64_lane_align_halves() {
        // Layout F: four lane-runs of 16 at stride 32, plus the lane-align offset.
        let low = mma_blocks(64, 8, 1, 0, 4, 2, TMEM_ROWS, TMEM_COLS).unwrap();
        assert_eq!(low.len(), 4);
        for (g, b) in low.iter().enumerate() {
            assert_eq!(b.cta_idx, 0);
            assert_eq!(b.m_lo, g * 16);
            assert_eq!(b.rows, 16);
            assert_eq!((b.n_lo, b.cols), (0, 8));
            assert_eq!(b.col_start, 6); // col_start 4 + dst_col 2
            assert_eq!(b.lane_start, 32 * g); // lane_align 0
        }

        let high = mma_blocks(64, 8, 1, 16, 4, 2, TMEM_ROWS, TMEM_COLS).unwrap();
        assert_eq!(high[0].lane_start, 16);
        assert_eq!(high[3].lane_start, 112); // 96 + lane_align 16
    }

    #[test]
    fn mma_blocks_model_cta2_m128_layout_b() {
        // m split across the CTA pair; n split in half into the upper 64 lanes.
        let blocks = mma_blocks(128, 32, 2, 0, 0, 3, TMEM_ROWS, TMEM_COLS).unwrap();
        assert_eq!(blocks.len(), 4);
        let has = |cta, m_lo, n_lo, lane_start| {
            blocks.iter().any(|b| {
                b.cta_idx == cta
                    && b.m_lo == m_lo
                    && b.n_lo == n_lo
                    && b.lane_start == lane_start
                    && b.rows == 64
                    && b.cols == 16
                    && b.col_start == 3
            })
        };
        assert!(has(0, 0, 0, 0)); // cta0, n first half -> lanes [0,64)
        assert!(has(0, 0, 16, 64)); // cta0, n second half -> lanes [64,128)
        assert!(has(1, 64, 0, 0)); // cta1, n first half
        assert!(has(1, 64, 16, 64)); // cta1, n second half
    }

    #[test]
    fn mma_blocks_model_cta2_m256_layout_a_pair_split() {
        let blocks = mma_blocks(256, 32, 2, 0, 7, 5, TMEM_ROWS, TMEM_COLS).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            (blocks[0].cta_idx, blocks[0].m_lo, blocks[0].rows),
            (0, 0, 128)
        );
        assert_eq!(
            (blocks[0].lane_start, blocks[0].col_start, blocks[0].cols),
            (0, 12, 32)
        );
        assert_eq!(
            (blocks[1].cta_idx, blocks[1].m_lo, blocks[1].lane_start),
            (1, 128, 0)
        );
        assert_eq!(blocks[1].col_start, 12); // 7 + 5
    }

    #[test]
    fn mma_blocks_reject_lane_align_for_full_datapath_layouts() {
        for (m, cta_group) in [(128, 1), (128, 2), (256, 2)] {
            let err = mma_blocks(m, 32, cta_group, 16, 0, 0, 128, 512).unwrap_err();
            assert_eq!(err.code, "tcgen05_mma_lane_align");
        }
    }

    #[test]
    fn mma_blocks_reject_tmem_bounds() {
        // Layout F at lane_align 16: the top run reaches lane 128 > 127.
        let err = mma_blocks(64, 8, 1, 16, 0, 0, 127, TMEM_COLS).unwrap_err();
        assert_eq!(err.code, "tcgen05_mma_out_of_range");
        // col_start 500 + 32 cols overruns the 512-column grid.
        let err = mma_blocks(128, 32, 1, 0, 500, 0, TMEM_ROWS, TMEM_COLS).unwrap_err();
        assert_eq!(err.code, "tcgen05_mma_out_of_range");
    }

    #[test]
    fn check_mma_operand_shapes_fails_closed() {
        let a = Array2::<f32>::zeros((128, 16));
        let b = Array2::<f32>::zeros((64, 16));
        let err = check_mma_operand_shapes(&a, &b, 128, 128, 16).unwrap_err();
        assert_eq!(err.code, "tcgen05_mma_shape");
    }
}
