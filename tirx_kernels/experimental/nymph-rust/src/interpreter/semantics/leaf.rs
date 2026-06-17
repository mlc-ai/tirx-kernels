//! Fallback executor — port of `semantics/leaf.py`. In Python this bridges to
//! public per-thread leaf handlers; the Rust port has no public leaf registry
//! yet, so an unregistered statement fails closed (all built-in ops are
//! registered, so this is unreachable for them).

use super::super::cohort::CohortContext;
use super::super::diagnostics::{Diagnostic, IResult};
use super::super::outcomes::StepStatus;
use super::super::registry::StmtExecutorRegistry;
use crate::ir::Stmt;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.set_fallback(execute_leaf);
}

fn execute_leaf<'a, 'k>(_ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let name = format!("{:?}", super::super::registry::stmt_kind(stmt));
    Ok(StepStatus::Fail(Some(Diagnostic::error(
        "unsupported_stmt",
        format!("no executor registered for statement kind {name}"),
    ))))
}
