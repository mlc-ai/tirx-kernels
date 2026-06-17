//! Scalar def/store executors — port of `semantics/scalar.py`. Write each
//! thread's scalar directly (not eager, unlike loop vars).

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scalar_eval::eval_scalar_in_env;
use super::super::transfer::write_operand;
use super::super::values::arrays::ValueArray1;
use super::super::values::tensors::tensor_instance_key;
use crate::ir::{MemorySpace, ScalarInitial, Stmt};
use ndarray::Array1;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::ScalarDef, execute_scalar_def);
    reg.register(StmtKind::ScalarStore, execute_scalar_store);
    reg.register(StmtKind::StoreScalar, execute_store_scalar);
}

fn execute_scalar_def<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (var, initial) = match stmt {
        Stmt::ScalarDef { var, initial } => (var, initial),
        _ => unreachable!(),
    };
    let var_id = var.id.0;
    let values: Vec<i64> = match initial {
        ScalarInitial::Value(v) => ctx.eval_scalar_vec(v)?.to_vec(),
        ScalarInitial::Tensor(slice) => {
            if slice.tensor.space == MemorySpace::Smem {
                let resolved = ctx.eval_slice(slice)?;
                if ctx.trace_mode() {
                    ctx.emit_tensor_read(&resolved)?;
                }
                let values = match ctx.shared_read(&resolved) {
                    Ok(values) => values.to_i64_compute(),
                    Err(e)
                        if ctx.trace_mode()
                            && matches!(e.code.as_str(), "missing_tensor_value") =>
                    {
                        return Err(ctx.trace_inconclusive(
                            "trace_control_from_skipped_payload",
                            "trace control read depends on skipped payload tensor data",
                        ));
                    }
                    Err(e) => return Err(e),
                };
                return Ok(scalar_commit(
                    ctx,
                    var_id,
                    &(0..values.nrows())
                        .map(|i| values[[i, 0]])
                        .collect::<Vec<_>>(),
                ));
            }
            let mut out = Vec::with_capacity(ctx.cohort.len());
            for t in ctx.cohort.clone().iter() {
                out.push(read_scalar_initial(ctx, slice, t)?);
            }
            out
        }
    };
    Ok(scalar_commit(ctx, var_id, &values))
}

fn execute_scalar_store<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (var, value) = match stmt {
        Stmt::ScalarStore { var, value } => (var, value),
        _ => unreachable!(),
    };
    let var_id = var.id.0;
    let values = ctx.eval_scalar_vec(value)?.to_vec();
    Ok(scalar_commit(ctx, var_id, &values))
}

fn execute_store_scalar<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, value) = match stmt {
        Stmt::StoreScalar { dst, value } => (dst, value),
        _ => unreachable!(),
    };
    let resolved_dst = ctx.eval_slice(dst)?;
    let values = ctx.eval_scalar_vec(value)?.to_vec();
    let array = ValueArray1::from_i64_compute(Array1::from(values), dst.tensor.dtype)
        .reshape2((ctx.cohort.len(), 1))?;
    if ctx.trace_mode() {
        ctx.emit_tensor_write(&resolved_dst)?;
    }
    write_operand(ctx, &resolved_dst, &array)?;
    Ok(StepStatus::advance())
}

fn scalar_commit(ctx: &mut CohortContext, var_id: u32, values: &[i64]) -> StepStatus {
    ctx.state
        .values
        .scalars
        .write_values(&ctx.cohort, var_id, values);
    StepStatus::advance()
}

fn read_scalar_initial(
    ctx: &mut CohortContext,
    slice: &crate::ir::TensorSlice,
    thread: &super::super::threads::ThreadId,
) -> IResult<i64> {
    if slice.tensor.space != MemorySpace::Gmem {
        return Err(InterpreterError::new(
            "scalar_load",
            "scalar_def tensor initial must be GMEM",
        ));
    }
    let env = ctx.state.values.scalars.by_thread.get(thread);
    let empty = std::collections::HashMap::new();
    let env = env.unwrap_or(&empty);
    let offsets: Vec<usize> = slice
        .offsets
        .iter()
        .map(|o| eval_scalar_in_env(o, thread, env).map(|v| v as usize))
        .collect::<IResult<_>>()?;
    let shape: Vec<usize> = slice
        .shape
        .iter()
        .map(|s| eval_scalar_in_env(s, thread, env).map(|v| v as usize))
        .collect::<IResult<_>>()?;
    let key = tensor_instance_key(thread.cta_id, &slice.tensor)?;
    let Some(tv) = ctx.state.values.tensors.by_instance.get(&key) else {
        if ctx.trace_mode() {
            return Err(ctx.trace_inconclusive(
                "trace_control_from_skipped_payload",
                "trace control read depends on an unavailable tensor input",
            ));
        }
        return Err(InterpreterError::new(
            "missing_input",
            "scalar_def tensor initial is not loaded",
        ));
    };
    let block = match tv.read_block(&offsets, &shape) {
        Ok(block) => block,
        Err(e)
            if ctx.trace_mode()
                && matches!(e.code.as_str(), "missing_tensor_value" | "missing_input") =>
        {
            return Err(ctx.trace_inconclusive(
                "trace_control_from_skipped_payload",
                "trace control read depends on skipped payload tensor data",
            ));
        }
        Err(e) => return Err(e),
    };
    Ok(block.to_i64_compute()[0])
}
