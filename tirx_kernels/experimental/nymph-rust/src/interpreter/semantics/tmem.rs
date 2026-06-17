//! TMEM alloc/dealloc lifecycle + CTA-pair collective — port of `semantics/tmem.py`.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::mbar_ops::peer_ctaid_in_cluster;
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::TraceEventKind;
use super::super::region;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::state::InterpreterState;
use super::super::threads::ThreadMask;
use super::super::tmem::{
    TmemAllocation, TmemAllocationKey, TmemCollectiveArrival, TmemCollectiveKey,
};
use super::super::values::tmem::tmem_physical_range;
use crate::ir::{Stmt, Tensor};
use std::sync::Arc;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::TmemAlloc, execute_tmem_alloc);
    reg.register(StmtKind::TmemDealloc, execute_tmem_dealloc);
}

fn execute_tmem_alloc<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    lifecycle(ctx, stmt, "alloc")
}
fn execute_tmem_dealloc<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    lifecycle(ctx, stmt, "dealloc")
}

fn fields<'k>(stmt: &'k Stmt) -> (&'k Arc<Tensor>, usize, u8) {
    match stmt {
        Stmt::TmemAlloc {
            tensor,
            n_cols,
            cta_group,
        }
        | Stmt::TmemDealloc {
            tensor,
            n_cols,
            cta_group,
        } => (tensor, *n_cols as usize, *cta_group),
        _ => unreachable!(),
    }
}

fn check_full_warp_issue(cohort: &ThreadMask) -> IResult<()> {
    if cohort.len() != 32 {
        return Err(InterpreterError::new(
            "invalid_tmem_issue_mask",
            "tmem alloc/dealloc must be a full warp",
        ));
    }
    let first = &cohort[0];
    for (i, t) in cohort.iter().enumerate() {
        if t.cta_id != first.cta_id || t.warp_id != first.warp_id || t.lane_id != i {
            return Err(InterpreterError::new(
                "invalid_tmem_issue_mask",
                "tmem alloc/dealloc mask must be lanes 0..31",
            ));
        }
    }
    Ok(())
}

fn lifecycle<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
    op: &'static str,
) -> IResult<StepStatus> {
    check_full_warp_issue(&ctx.cohort)?;
    let (tensor, n_cols, cta_group) = fields(stmt);
    let (col_start, n_cols_phys) = tmem_physical_range(tensor, n_cols).map_err(|e| e)?;
    let _ = n_cols_phys;
    if cta_group == 1 {
        let cta_id = ctx.stream.cta_id;
        lifecycle_apply(ctx, stmt, op, &[cta_id], col_start, n_cols)?;
        return Ok(StepStatus::advance());
    }
    cta_group_2(ctx, stmt, op, col_start, n_cols)
}

/// Apply the alloc/dealloc lifecycle directly: allocation records, the value-mode
/// scratchpad ensure/clear, and the last-alloc-cols marker.
fn lifecycle_apply(
    ctx: &mut CohortContext,
    stmt: &Stmt,
    op: &str,
    cta_ids: &[usize],
    col_start: usize,
    n_cols: usize,
) -> IResult<()> {
    let (tensor, _n, cta_group) = fields(stmt);
    let mut sorted = cta_ids.to_vec();
    sorted.sort_unstable();
    if op == "alloc" {
        // Validate all targets first so a conflict aborts before any write.
        for &cta_id in &sorted {
            check_alloc_preconditions(ctx.state, cta_id, col_start, n_cols)?;
        }
        let allocation = TmemAllocation {
            col_start,
            n_cols,
            cta_group,
            collective_cta_ids: sorted.clone(),
        };
        let vm = ctx.value_mode();
        for &cta_id in &sorted {
            let key = TmemAllocationKey {
                cta_id,
                col_start,
                n_cols,
            };
            ctx.state.tmem_allocations.insert(key, allocation.clone());
            if vm {
                ctx.state.values.tmem.by_cta.entry(cta_id).or_default();
            }
            ctx.state.tmem_last_alloc_cols.insert(cta_id, n_cols);
        }
    } else {
        // Validate all targets first so a conflict aborts before any delete/clear.
        for &cta_id in &sorted {
            check_dealloc_preconditions(ctx.state, cta_id, col_start, n_cols, cta_group)?;
        }
        for &cta_id in &sorted {
            let key = TmemAllocationKey {
                cta_id,
                col_start,
                n_cols,
            };
            ctx.state.tmem_allocations.remove(&key);
            if let Some(sp) = ctx.state.values.tmem.by_cta.get_mut(&cta_id) {
                sp.clear_columns(col_start, n_cols);
            }
        }
    }
    if ctx.trace_mode() {
        let scope = ctx.access_scope();
        for &cta_id in &sorted {
            let region = region::tmem_allocation_region(tensor.id, cta_id, col_start, n_cols)?;
            if op == "alloc" {
                ctx.emit(TraceEventKind::TmemAlloc {
                    cta_ids: vec![cta_id],
                    region,
                    scope: scope.clone(),
                })?;
            } else {
                ctx.emit(TraceEventKind::TmemDealloc {
                    cta_ids: vec![cta_id],
                    region,
                    scope: scope.clone(),
                })?;
            }
        }
    }
    Ok(())
}

fn preflight_lifecycle_for_cta(
    state: &InterpreterState,
    op: &str,
    cta_id: usize,
    col_start: usize,
    n_cols: usize,
    cta_group: u8,
) -> IResult<()> {
    if op == "alloc" {
        check_alloc_preconditions(state, cta_id, col_start, n_cols)
    } else {
        check_dealloc_preconditions(state, cta_id, col_start, n_cols, cta_group)
    }
}

fn check_alloc_preconditions(
    state: &InterpreterState,
    cta_id: usize,
    col_start: usize,
    n_cols: usize,
) -> IResult<()> {
    let key = TmemAllocationKey {
        cta_id,
        col_start,
        n_cols,
    };
    if state.tmem_allocations.contains_key(&key) {
        return Err(InterpreterError::new(
            "tmem_already_allocated",
            "TMEM physical range is already allocated",
        ));
    }
    if overlapping_allocation_key(state, key).is_some() {
        return Err(InterpreterError::new(
            "tmem_allocation_overlap",
            "TMEM allocation overlaps an existing physical range",
        ));
    }
    if let Some(&last_n_cols) = state.tmem_last_alloc_cols.get(&cta_id) {
        if n_cols > last_n_cols {
            return Err(InterpreterError::new(
                "tmem_allocation_order",
                "TMEM allocation n_cols cannot increase within a CTA",
            ));
        }
    }
    Ok(())
}

fn check_dealloc_preconditions(
    state: &InterpreterState,
    cta_id: usize,
    col_start: usize,
    n_cols: usize,
    cta_group: u8,
) -> IResult<()> {
    let key = TmemAllocationKey {
        cta_id,
        col_start,
        n_cols,
    };
    let Some(allocation) = state.tmem_allocations.get(&key) else {
        if overlapping_allocation_key(state, key).is_some() {
            return Err(InterpreterError::new(
                "tmem_allocation_mismatch",
                "TMEM deallocation physical range does not match allocation",
            ));
        }
        return Err(InterpreterError::new(
            "missing_tmem_allocation",
            "TMEM physical range is not allocated for this CTA",
        ));
    };
    if allocation.col_start != key.col_start
        || allocation.n_cols != key.n_cols
        || allocation.cta_group != cta_group
    {
        return Err(InterpreterError::new(
            "tmem_allocation_mismatch",
            "TMEM deallocation does not match allocation",
        ));
    }
    Ok(())
}

fn overlapping_allocation_key(
    state: &InterpreterState,
    candidate: TmemAllocationKey,
) -> Option<TmemAllocationKey> {
    state
        .tmem_allocations
        .keys()
        .find(|existing| {
            existing.cta_id == candidate.cta_id
                && ranges_overlap(
                    existing.col_start,
                    existing.n_cols,
                    candidate.col_start,
                    candidate.n_cols,
                )
        })
        .copied()
}

fn ranges_overlap(lhs_start: usize, lhs_cols: usize, rhs_start: usize, rhs_cols: usize) -> bool {
    let lhs_end = lhs_start + lhs_cols;
    let rhs_end = rhs_start + rhs_cols;
    lhs_start < rhs_end && rhs_start < lhs_end
}

fn cta_group_2<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
    op: &'static str,
    col_start: usize,
    n_cols: usize,
) -> IResult<StepStatus> {
    let stmt_id = ctx.stmt_id(stmt);
    let current = ctx.cohort[0].clone();
    let peer_local = peer_ctaid_in_cluster(
        ctx,
        current.ctaid_in_cluster,
        "tmem_collective_peer",
        "TMEM collective peer CTA out of range",
    )?;
    let peer_global = ctx.global_cta_id(peer_local);
    let key = TmemCollectiveKey {
        stmt_id,
        op,
        col_start,
        n_cols,
        cluster_id: current.cluster_id,
        pair_base_ctaid_in_cluster: current.ctaid_in_cluster & !1,
    };
    let (_t, _n, cta_group) = fields(stmt);
    let arrival = TmemCollectiveArrival {
        cta_id: current.cta_id,
        ctaid_in_cluster: current.ctaid_in_cluster,
        stream_id: ctx.stream.stream_id,
        col_start,
        n_cols,
        cta_group,
    };
    let collective = ctx
        .state
        .tmem_collectives
        .get(&key)
        .cloned()
        .unwrap_or_default();
    let existing = collective.arrival_for_cta(current.cta_id).cloned();

    if existing.is_some() {
        // same CTA retrying (polled rendezvous)
        if collective.arrivals.len() < 2 {
            return Ok(StepStatus::block(WakeCondition::Polled));
        }
        // both present → complete this CTA
        let completed = collective.with_completed(current.cta_id);
        if completed.completed_cta_ids.len() >= 2 {
            ctx.state.tmem_collectives.remove(&key);
        } else {
            ctx.state.tmem_collectives.insert(key, completed);
        }
        return Ok(StepStatus::advance());
    }

    preflight_lifecycle_for_cta(ctx.state, op, current.cta_id, col_start, n_cols, cta_group)?;

    // first arrival of this CTA: record it (idempotent — this CTA had none) and park.
    let collective = collective.with_arrival(arrival);
    if collective.arrivals.len() == 1 {
        ctx.state.tmem_collectives.insert(key, collective);
        return Ok(StepStatus::block(WakeCondition::Polled));
    }
    // this arrival completes the pair
    let arrived: Vec<usize> = {
        let mut v: Vec<usize> = collective.arrivals.iter().map(|a| a.cta_id).collect();
        v.sort_unstable();
        v
    };
    let mut expected = vec![current.cta_id, peer_global];
    expected.sort_unstable();
    if arrived != expected {
        return Err(InterpreterError::new(
            "tmem_collective_mismatch",
            "TMEM collective CTA pair mismatch",
        ));
    }
    let completed = collective.with_completed(current.cta_id);
    lifecycle_apply(ctx, stmt, op, &arrived, col_start, n_cols)?;
    ctx.state.tmem_collectives.insert(key, completed);
    Ok(StepStatus::advance())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::protocol::ExecutionMode;

    fn allocation(cta_group: u8) -> TmemAllocation {
        TmemAllocation {
            col_start: 0,
            n_cols: 64,
            cta_group,
            collective_cta_ids: vec![0],
        }
    }

    fn err_code(result: IResult<()>) -> String {
        result.unwrap_err().code
    }

    #[test]
    fn alloc_preconditions_reject_duplicate_overlap_and_increasing_order() {
        let mut state = InterpreterState::new(ExecutionMode::Value);
        state.tmem_allocations.insert(
            TmemAllocationKey {
                cta_id: 0,
                col_start: 0,
                n_cols: 64,
            },
            allocation(1),
        );

        assert_eq!(
            err_code(check_alloc_preconditions(&state, 0, 0, 64)),
            "tmem_already_allocated"
        );
        assert_eq!(
            err_code(check_alloc_preconditions(&state, 0, 32, 64)),
            "tmem_allocation_overlap"
        );
        assert!(check_alloc_preconditions(&state, 1, 32, 64).is_ok());

        let mut ordered = InterpreterState::new(ExecutionMode::Value);
        ordered.tmem_last_alloc_cols.insert(0, 64);
        assert_eq!(
            err_code(check_alloc_preconditions(&ordered, 0, 128, 128)),
            "tmem_allocation_order"
        );
    }

    #[test]
    fn dealloc_preconditions_reject_missing_mismatch_and_cta_group_mismatch() {
        let missing = InterpreterState::new(ExecutionMode::Value);
        assert_eq!(
            err_code(check_dealloc_preconditions(&missing, 0, 0, 64, 1)),
            "missing_tmem_allocation"
        );

        let mut state = InterpreterState::new(ExecutionMode::Value);
        state.tmem_allocations.insert(
            TmemAllocationKey {
                cta_id: 0,
                col_start: 0,
                n_cols: 64,
            },
            allocation(1),
        );
        assert_eq!(
            err_code(check_dealloc_preconditions(&state, 0, 32, 64, 1)),
            "tmem_allocation_mismatch"
        );
        assert_eq!(
            err_code(check_dealloc_preconditions(&state, 0, 0, 64, 2)),
            "tmem_allocation_mismatch"
        );
        assert!(check_dealloc_preconditions(&state, 0, 0, 64, 1).is_ok());
    }
}
