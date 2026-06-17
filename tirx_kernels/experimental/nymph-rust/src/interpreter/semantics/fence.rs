//! Fence — a trace marker (no value effect) — port of `semantics/fence.py`.

use super::super::cohort::CohortContext;
use super::super::diagnostics::IResult;
use super::super::outcomes::StepStatus;
use super::super::protocol::{FenceEventKind, TraceEventKind};
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use crate::ir::{FenceKind, Stmt};

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::Fence, execute_fence);
}

fn execute_fence<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (kind, scope) = match stmt {
        Stmt::Fence { kind, scope } => (*kind, *scope),
        _ => unreachable!(),
    };
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::Fence {
            fence_kind: fence_event_kind(kind),
            fence_scope: scope,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn fence_event_kind(kind: FenceKind) -> FenceEventKind {
    match kind {
        FenceKind::AsyncProxy => FenceEventKind::ProxyAsync,
        FenceKind::Memory | FenceKind::View => FenceEventKind::Generic,
    }
}
