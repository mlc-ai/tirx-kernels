//! mbarrier executors — port of `semantics/mbarrier.py`. Wait is the one blocking
//! op; it parks on `WakeCondition::Mbar` and is advanced (never re-run) when a
//! later cell write flips the parity. Arrive/expect-tx writes list the touched
//! cell key in `wakes` so the runner re-checks that key's parked waiters.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::mbar_ops::{
    arrive_mbarrier_cell, expect_tx_cell, initialized_mbar_cell, uniform_mbar_target,
};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::TraceEventKind;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scalar_eval;
use super::super::values::mbars::MbarCell;
use crate::ir::{ScalarValue, Stmt};

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::MBarrierInit, execute_mbarrier_init);
    reg.register(StmtKind::MBarrierArrive, execute_mbarrier_arrive);
    reg.register(StmtKind::MBarrierExpectTx, execute_mbarrier_expect_tx);
    reg.register(
        StmtKind::MBarrierArriveExpectTx,
        execute_mbarrier_arrive_expect_tx,
    );
    reg.register(StmtKind::MBarrierWait, execute_mbarrier_wait);
}

fn eval_mbar_wait_phase(ctx: &CohortContext<'_, '_>, phase: &ScalarValue) -> IResult<u8> {
    let timer = super::super::runner::prof_now();
    let proven_uniform = scalar_eval::scalar_is_cohort_uniform(phase);
    super::super::runner::prof_end("MWait:phase_classify", timer);

    let v = if proven_uniform {
        let timer = super::super::runner::prof_now();
        let v = scalar_eval::eval_scalar_at(phase, &ctx.cohort[0], &ctx.state.values.scalars)?;
        super::super::runner::prof_end("MWait:phase_eval_one", timer);
        v
    } else {
        let timer = super::super::runner::prof_now();
        if let Some(v) =
            scalar_eval::eval_scalar_known_uniform(phase, &ctx.cohort, &ctx.state.values.scalars)?
        {
            super::super::runner::prof_end("MWait:phase_fact", timer);
            v
        } else {
            super::super::runner::prof_end("MWait:phase_fact", timer);
            let timer = super::super::runner::prof_now();
            let first =
                scalar_eval::eval_scalar_at(phase, &ctx.cohort[0], &ctx.state.values.scalars)?;
            super::super::runner::prof_end("MWait:phase_eval_one", timer);
            let timer = super::super::runner::prof_now();
            for thread in ctx.cohort.iter().skip(1) {
                if scalar_eval::eval_scalar_at(phase, thread, &ctx.state.values.scalars)? != first {
                    return Err(InterpreterError::new(
                        "divergent_mbarrier_operands",
                        "mbarrier wait phase must be uniform",
                    ));
                }
            }
            super::super::runner::prof_end("MWait:phase_scan", timer);
            first
        }
    };

    if v != 0 && v != 1 {
        return Err(InterpreterError::new(
            "invalid_mbarrier_phase",
            "mbarrier wait phase must be 0 or 1",
        ));
    }
    Ok(v as u8)
}

fn execute_mbarrier_init<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (mbar, count, stage) = match stmt {
        Stmt::MBarrierInit { mbar, count, stage } => (mbar, *count, stage),
        _ => unreachable!(),
    };
    let target = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    if ctx.state.values.mbars.cells.contains_key(&target.key()) {
        return Err(InterpreterError::new(
            "mbarrier_already_initialized",
            "mbarrier cell is already initialized",
        ));
    }
    let cell = MbarCell {
        expected_arrivals: count as i64,
        pending_arrivals: count as i64,
        pending_tx_bytes: 0,
        parity: 0,
        stage: target.stage,
    };
    ctx.state.values.mbars.cells.insert(target.key(), cell);
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::MbarInit {
            target: target.into(),
            count: count as i64,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn execute_mbarrier_arrive<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (mbar, stage, count) = match stmt {
        Stmt::MBarrierArrive { mbar, stage, count } => (mbar, stage, count),
        _ => unreachable!(),
    };
    let target = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    let count = ctx.eval_scalar_uniform(
        count,
        "mbarrier arrive count",
        "divergent_mbarrier_operands",
    )?;
    let cell = initialized_mbar_cell(ctx, target.key())?;
    let updated = arrive_mbarrier_cell(cell, count)?;
    let key = target.key();
    ctx.state.values.mbars.cells.insert(key, updated);
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::MbarArrive {
            target: target.into(),
            count,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance_wake(vec![key]))
}

fn execute_mbarrier_expect_tx<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (mbar, bytes, stage) = match stmt {
        Stmt::MBarrierExpectTx { mbar, bytes, stage } => (mbar, *bytes, stage),
        _ => unreachable!(),
    };
    let target = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    let cell = initialized_mbar_cell(ctx, target.key())?;
    let updated = expect_tx_cell(cell, bytes as i64);
    let key = target.key();
    ctx.state.values.mbars.cells.insert(key, updated);
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::MbarExpectTx {
            target: target.into(),
            bytes: bytes as i64,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance_wake(vec![key]))
}

fn execute_mbarrier_arrive_expect_tx<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (mbar, bytes, stage) = match stmt {
        Stmt::MBarrierArriveExpectTx { mbar, bytes, stage } => (mbar, *bytes, stage),
        _ => unreachable!(),
    };
    let target = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    let cell = initialized_mbar_cell(ctx, target.key())?;
    if cell.pending_arrivals < 1 {
        return Err(InterpreterError::new(
            "mbarrier_arrive_overflow",
            "mbarrier arrive exceeds pending arrivals",
        ));
    }
    let updated = arrive_mbarrier_cell(expect_tx_cell(cell, bytes as i64), 1)?;
    let key = target.key();
    ctx.state.values.mbars.cells.insert(key, updated);
    if ctx.trace_mode() {
        let scope = ctx.access_scope();
        ctx.emit(TraceEventKind::MbarExpectTx {
            target: target.into(),
            bytes: bytes as i64,
            scope: scope.clone(),
        })?;
        ctx.emit(TraceEventKind::MbarArrive {
            target: target.into(),
            count: 1,
            scope,
        })?;
    }
    Ok(StepStatus::advance_wake(vec![key]))
}

fn execute_mbarrier_wait<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (mbar, stage, phase) = match stmt {
        Stmt::MBarrierWait { mbar, stage, phase } => (mbar, stage, phase),
        _ => unreachable!(),
    };
    let timer = super::super::runner::prof_now();
    let target = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    super::super::runner::prof_end("MWait:target", timer);
    let timer = super::super::runner::prof_now();
    let cell = initialized_mbar_cell(ctx, target.key())?;
    super::super::runner::prof_end("MWait:cell_lookup", timer);

    // The awaited phase: explicit, or the parity observed now (phase-less). It rides
    // in the park condition — no state latch, since a parked wait never re-runs.
    let timer = super::super::runner::prof_now();
    let want_phase: u8 = if let Some(p) = phase {
        eval_mbar_wait_phase(ctx, p)?
    } else {
        cell.parity
    };
    super::super::runner::prof_end("MWait:phase", timer);

    if cell.parity != want_phase {
        if ctx.trace_mode() {
            let timer = super::super::runner::prof_now();
            ctx.emit(TraceEventKind::MbarWait {
                target: target.into(),
                phase: want_phase,
                scope: ctx.access_scope(),
            })?;
            super::super::runner::prof_end("MWait:emit_ready", timer);
        }
        Ok(StepStatus::advance())
    } else if ctx.trace_mode() {
        let timer = super::super::runner::prof_now();
        let status = StepStatus::block_with_completion_event(
            WakeCondition::Mbar {
                key: target.key(),
                phase: want_phase,
            },
            ctx.anchored_event(TraceEventKind::MbarWait {
                target: target.into(),
                phase: want_phase,
                scope: ctx.access_scope(),
            }),
        );
        super::super::runner::prof_end("MWait:block_event", timer);
        Ok(status)
    } else {
        let timer = super::super::runner::prof_now();
        let status = StepStatus::block(WakeCondition::Mbar {
            key: target.key(),
            phase: want_phase,
        });
        super::super::runner::prof_end("MWait:block", timer);
        Ok(status)
    }
}
