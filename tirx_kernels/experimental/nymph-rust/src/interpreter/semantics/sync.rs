//! Cooperative rendezvous — port of `semantics/sync.py`. CtaSync/WgSync/WarpSync/
//! ClusterSync block until the whole scope arrives — `Block(Polled)`, re-run each
//! round. The arrival/completion writes are direct; re-running is naturally
//! idempotent (the arrival is a set union, completion is gated on `==expected`).

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::TraceEventKind;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scheduler::{flatten_coord, unflatten_coord, CtaActivityStatus};
use super::super::threads::{canonical_thread_mask, ThreadId, ThreadMask};
use crate::ir::Stmt;
use std::collections::HashSet;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::CtaSync, execute_sync);
    reg.register(StmtKind::WgSync, execute_sync);
    reg.register(StmtKind::WarpSync, execute_sync);
    reg.register(StmtKind::ClusterSync, execute_sync);
}

fn cta_threads(template: &ThreadId, warp_ids: &[usize]) -> ThreadMask {
    let mut threads = Vec::new();
    for &w in warp_ids {
        for lane in 0..32 {
            threads.push(ThreadId {
                warp_id: w,
                lane_id: lane,
                ..*template
            });
        }
    }
    canonical_thread_mask(threads)
}

fn cluster_threads(ctx: &CohortContext, template: &ThreadId) -> ThreadMask {
    let cluster = &ctx.kernel.cluster_shape;
    let launch = &ctx.kernel.launch_shape;
    let cluster_grid: Vec<usize> = launch
        .iter()
        .zip(cluster.iter())
        .map(|(l, c)| l / c)
        .collect();
    let num_warps = ctx.kernel.num_warps as usize;
    let mut threads = Vec::new();
    for ctaid_in_cluster in 0..ctx.cluster_cta_count() {
        let local = unflatten_coord(ctaid_in_cluster, cluster);
        let cta_coord: Vec<usize> = cluster
            .iter()
            .zip(template.cluster_coord.as_slice().iter())
            .zip(local.iter())
            .map(|((cl, cc), l)| cl * cc + l)
            .collect();
        let cta_id = flatten_coord(&cta_coord, launch);
        let cluster_id = flatten_coord(template.cluster_coord.as_slice(), &cluster_grid);
        let cta_coord_c = super::super::threads::Coord::from_slice(&cta_coord);
        let local_c = super::super::threads::Coord::from_slice(&local);
        for w in 0..num_warps {
            for lane in 0..32 {
                threads.push(ThreadId {
                    cta_id,
                    cta_coord: cta_coord_c,
                    cluster_id,
                    ctaid_in_cluster,
                    cluster_coord: template.cluster_coord,
                    cta_coord_in_cluster: local_c,
                    warp_id: w,
                    lane_id: lane,
                });
            }
        }
    }
    canonical_thread_mask(threads)
}

fn expected_threads(ctx: &CohortContext, stmt: &Stmt) -> ThreadMask {
    let first = &ctx.cohort[0];
    let num_warps = ctx.kernel.num_warps as usize;
    match stmt {
        Stmt::ClusterSync => cluster_threads(ctx, first),
        Stmt::CtaSync => cta_threads(first, &(0..num_warps).collect::<Vec<_>>()),
        Stmt::WgSync { .. } => {
            let base = first.warpgroup_id() * 4;
            cta_threads(first, &(base..base + 4).collect::<Vec<_>>())
        }
        Stmt::WarpSync => {
            let mut warps: Vec<usize> = ctx.cohort.iter().map(|t| t.warp_id).collect();
            warps.sort_unstable();
            warps.dedup();
            cta_threads(first, &warps)
        }
        _ => unreachable!(),
    }
}

fn sync_key(ctx: &CohortContext, stmt: &Stmt, stmt_id: usize) -> String {
    let first = &ctx.cohort[0];
    match stmt {
        Stmt::ClusterSync => format!("cluster_sync:{stmt_id}:cluster{}", first.cluster_id),
        Stmt::CtaSync => format!("cta_sync:{stmt_id}:cta{}", first.cta_id),
        Stmt::WgSync { barrier_id } => {
            format!(
                "wg_sync:{stmt_id}:cta{}:wg{}:bar{}",
                first.cta_id,
                first.warpgroup_id(),
                barrier_id
            )
        }
        Stmt::WarpSync => {
            let mut warps: Vec<usize> = ctx.cohort.iter().map(|t| t.warp_id).collect();
            warps.sort_unstable();
            warps.dedup();
            let joined = warps
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("warp_sync:{stmt_id}:cta{}:warps{joined}", first.cta_id)
        }
        _ => unreachable!(),
    }
}

fn execute_sync<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let stmt_id = ctx.stmt_id(stmt);
    let expected: HashSet<ThreadId> = expected_threads(ctx, stmt).into_iter().collect();
    let arriving: HashSet<ThreadId> = ctx.cohort.iter().cloned().collect();
    if !arriving.is_subset(&expected) {
        return Err(InterpreterError::new(
            "invalid_sync_scope",
            "sync cohort is outside the sync scope",
        ));
    }
    let key = sync_key(ctx, stmt, stmt_id);
    let cycle = ctx
        .state
        .values
        .cooperative
        .sync_cycles
        .get(&key)
        .copied()
        .unwrap_or(0);
    let arrived: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .syncs
        .get(&key)
        .cloned()
        .unwrap_or_default();
    let completed: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .rendezvous
        .get(&key)
        .cloned()
        .unwrap_or_default();

    if !arriving.is_disjoint(&completed) {
        return Err(InterpreterError::new(
            "cooperative_sync_reentry",
            "thread re-entered a completed sync",
        ));
    }
    let merged: HashSet<ThreadId> = arrived.union(&arriving).cloned().collect();
    if !merged.is_subset(&expected) {
        return Err(InterpreterError::new(
            "cooperative_sync_mismatch",
            "sync arrival set exceeds scope",
        ));
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::SyncArrive {
            sync_kind: sync_kind(stmt).to_string(),
            thread_count: expected.len(),
            count: merged.len(),
            cycle,
            bar_id: sync_bar_id(stmt),
            scope: ctx.access_scope(),
        })?;
    }

    if matches!(stmt, Stmt::ClusterSync) {
        check_cluster_peer_liveness(ctx, &expected, &merged)?;
    }

    if merged != expected {
        // record this cohort's arrival (idempotent set union) and re-poll next round
        ctx.state.values.cooperative.syncs.insert(key, merged);
        return Ok(StepStatus::block(WakeCondition::Polled));
    }

    // all arrived → complete: move this cohort into the rendezvous; the last one
    // (completed == expected) clears both records so the next use starts fresh.
    let completed_next: HashSet<ThreadId> = completed.union(&arriving).cloned().collect();
    if completed_next == expected {
        ctx.state.values.cooperative.syncs.remove(&key);
        ctx.state.values.cooperative.rendezvous.remove(&key);
        ctx.state
            .values
            .cooperative
            .sync_cycles
            .insert(key.clone(), cycle + 1);
    } else {
        ctx.state
            .values
            .cooperative
            .syncs
            .insert(key.clone(), expected.clone());
        ctx.state
            .values
            .cooperative
            .rendezvous
            .insert(key, completed_next);
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::Sync {
            sync_kind: sync_kind(stmt).to_string(),
            thread_count: expected.len(),
            cycle,
            bar_id: sync_bar_id(stmt),
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn sync_kind(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::ClusterSync => "cluster",
        Stmt::CtaSync => "cta",
        Stmt::WgSync { .. } => "warpgroup",
        Stmt::WarpSync => "warp",
        _ => unreachable!(),
    }
}

fn sync_bar_id(stmt: &Stmt) -> Option<u32> {
    match stmt {
        Stmt::WgSync { barrier_id } => Some(*barrier_id as u32),
        _ => None,
    }
}

fn check_cluster_peer_liveness(
    ctx: &CohortContext,
    expected: &HashSet<ThreadId>,
    merged: &HashSet<ThreadId>,
) -> IResult<()> {
    let arrived_ctas: HashSet<usize> = merged.iter().map(|t| t.cta_id).collect();
    let expected_ctas: HashSet<usize> = expected.iter().map(|t| t.cta_id).collect();
    for cta_id in expected_ctas {
        if arrived_ctas.contains(&cta_id) {
            continue;
        }
        match ctx.cta_activity(cta_id) {
            CtaActivityStatus::Missing => {
                return Err(InterpreterError::new(
                    "cluster_sync_peer_missing",
                    "cluster sync peer CTA is missing",
                ))
            }
            CtaActivityStatus::Exited => {
                return Err(InterpreterError::new(
                    "cluster_sync_peer_exited",
                    "cluster sync peer CTA has exited",
                ))
            }
            _ => {}
        }
    }
    Ok(())
}
