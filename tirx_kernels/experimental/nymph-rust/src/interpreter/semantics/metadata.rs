//! TensorDef / MBarDef — discovery metadata only (no-op) — port of `metadata.py`.

use super::super::cohort::CohortContext;
use super::super::diagnostics::IResult;
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use crate::ir::Stmt;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::TensorDef, execute_metadata);
    reg.register(StmtKind::MBarDef, execute_metadata);
}

fn execute_metadata<'a, 'k>(
    _ctx: &mut CohortContext<'a, 'k>,
    _stmt: &'k Stmt,
) -> IResult<StepStatus> {
    Ok(StepStatus::advance_continue())
}
