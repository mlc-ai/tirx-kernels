//! cp.async.bulk group counter — port of `semantics/cp_async.py`. Never blocks.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::protocol::TraceEventKind;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use crate::ir::Stmt;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::CpAsyncBulkCommitGroup, execute_commit_group);
    reg.register(StmtKind::CpAsyncBulkWaitGroupRead, execute_wait_group_read);
}

fn execute_commit_group<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    _stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let sid = ctx.stream.stream_id;
    let next = ctx
        .state
        .cp_async_bulk_groups
        .get(&sid)
        .copied()
        .unwrap_or(0)
        + 1;
    ctx.state.cp_async_bulk_groups.insert(sid, next);
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::CommitGroup {
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn execute_wait_group_read<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let n = match stmt {
        Stmt::CpAsyncBulkWaitGroupRead { n } => *n,
        _ => unreachable!(),
    };
    if n > 8 {
        return Err(InterpreterError::new(
            "invalid_cp_async_bulk_wait_group_read",
            "cp_async_bulk_wait_group_read n must be <= 8",
        ));
    }
    let sid = ctx.stream.stream_id;
    let retain = i64::from(n);
    match ctx.state.cp_async_bulk_groups.get_mut(&sid) {
        Some(count) if *count > retain => {
            *count = retain;
            if *count == 0 {
                ctx.state.cp_async_bulk_groups.remove(&sid);
            }
        }
        _ => {}
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::WaitGroup {
            n: n as u32,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}
