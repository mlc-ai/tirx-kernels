//! Scalar-IR evaluation — port of `interpreter/scalar_eval.py`.
//! Evaluates ScalarValue/ScalarExpr/Var/ScopeValue to concrete i64, per-thread or
//! vectorized over a cohort. Index ALU uses i64 with Python floor-div/floor-mod
//! (sign follows divisor) — NOT Rust's truncating `/`/`%`.

use super::diagnostics::{IResult, InterpreterError};
use super::threads::{ThreadId, ThreadMask};
use super::values::scalars::ScalarValues;
use crate::ir::{ScalarOp, ScalarValue, ScopeValueKind, VarBinding};
use ndarray::Array1;
use std::collections::HashMap;

type Env = HashMap<u32, i64>;

/// Python floor division (rounds toward -inf).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}
/// Python modulo (sign follows divisor).
fn floor_mod(a: i64, b: i64) -> i64 {
    let m = a % b;
    if m != 0 && ((m < 0) != (b < 0)) {
        m + b
    } else {
        m
    }
}

pub fn eval_scope_value(kind: ScopeValueKind, thread: &ThreadId) -> i64 {
    (match kind {
        ScopeValueKind::TidInWg => thread.tid_in_wg(),
        ScopeValueKind::LaneId => thread.lane_id,
        ScopeValueKind::WarpId => thread.warp_id,
        ScopeValueKind::WarpgroupId => thread.warpgroup_id(),
        ScopeValueKind::CtaidInCluster => thread.ctaid_in_cluster,
        ScopeValueKind::CtaId => thread.cta_id,
        ScopeValueKind::NvshmemMyPe => 0,
    }) as i64
}

fn eval_scalar_op(op: ScalarOp, args: &[i64]) -> IResult<i64> {
    Ok(match op {
        ScalarOp::Add => args[0].wrapping_add(args[1]),
        ScalarOp::Sub => args[0].wrapping_sub(args[1]),
        ScalarOp::Mul => args[0].wrapping_mul(args[1]),
        ScalarOp::Xor => args[0] ^ args[1],
        ScalarOp::And => args[0] & args[1],
        ScalarOp::Or => args[0] | args[1],
        ScalarOp::Eq => (args[0] == args[1]) as i64,
        ScalarOp::Ne => (args[0] != args[1]) as i64,
        ScalarOp::Lt => (args[0] < args[1]) as i64,
        ScalarOp::Le => (args[0] <= args[1]) as i64,
        ScalarOp::Gt => (args[0] > args[1]) as i64,
        ScalarOp::Ge => (args[0] >= args[1]) as i64,
        ScalarOp::FloorDiv => {
            if args[1] == 0 {
                return Err(InterpreterError::new("scalar_eval", "division by zero"));
            }
            floor_div(args[0], args[1])
        }
        ScalarOp::Mod => {
            if args[1] == 0 {
                return Err(InterpreterError::new("scalar_eval", "modulo by zero"));
            }
            floor_mod(args[0], args[1])
        }
        ScalarOp::Neg => -args[0],
        ScalarOp::Not => (args[0] == 0) as i64,
        ScalarOp::Select => {
            if args[0] != 0 {
                args[1]
            } else {
                args[2]
            }
        }
        ScalarOp::Min => args[0].min(args[1]),
        ScalarOp::Max => args[0].max(args[1]),
    })
}

/// Single-thread evaluation against one thread's scalar env.
pub fn eval_scalar_in_env(value: &ScalarValue, thread: &ThreadId, env: &Env) -> IResult<i64> {
    match value {
        ScalarValue::Int(n) => Ok(*n),
        ScalarValue::Var(v) => env.get(&v.id.0).copied().ok_or_else(|| {
            InterpreterError::new("scalar_eval", format!("unresolved scalar var v{}", v.id.0))
        }),
        ScalarValue::Scope(kind) => Ok(eval_scope_value(*kind, thread)),
        ScalarValue::Expr(e) => {
            if e.args.len() != e.op.arity() {
                return Err(InterpreterError::new(
                    "scalar_eval",
                    "scalar expression arity mismatch",
                ));
            }
            match e.op.arity() {
                1 => {
                    let args = [eval_scalar_in_env(&e.args[0], thread, env)?];
                    eval_scalar_op(e.op, &args)
                }
                2 => {
                    let args = [
                        eval_scalar_in_env(&e.args[0], thread, env)?,
                        eval_scalar_in_env(&e.args[1], thread, env)?,
                    ];
                    eval_scalar_op(e.op, &args)
                }
                3 => {
                    let args = [
                        eval_scalar_in_env(&e.args[0], thread, env)?,
                        eval_scalar_in_env(&e.args[1], thread, env)?,
                        eval_scalar_in_env(&e.args[2], thread, env)?,
                    ];
                    eval_scalar_op(e.op, &args)
                }
                _ => unreachable!("validated scalar op arity"),
            }
        }
    }
}

/// Whether `value` is provably identical across a cohort.
pub fn scalar_is_cohort_uniform(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Int(_) => true,
        ScalarValue::Var(v) => matches!(v.binding, VarBinding::Loop | VarBinding::Task),
        ScalarValue::Scope(kind) => !matches!(
            kind,
            ScopeValueKind::TidInWg
                | ScopeValueKind::LaneId
                | ScopeValueKind::WarpId
                | ScopeValueKind::WarpgroupId
        ),
        ScalarValue::Expr(e) => e.args.iter().all(scalar_is_cohort_uniform),
    }
}

fn env_for<'a>(scalars: &'a ScalarValues, thread: &ThreadId) -> &'a Env {
    scalars.by_thread.get(thread).unwrap_or_else(|| empty_env())
}

pub fn eval_scalar_known_uniform(
    value: &ScalarValue,
    cohort: &ThreadMask,
    scalars: &ScalarValues,
) -> IResult<Option<i64>> {
    if cohort.is_empty() {
        return Ok(None);
    }
    let first = &cohort[0];
    match value {
        ScalarValue::Int(n) => Ok(Some(*n)),
        ScalarValue::Var(v) => match v.binding {
            VarBinding::Loop | VarBinding::Task => eval_scalar_at(value, first, scalars).map(Some),
            VarBinding::Scalar => Ok(scalars.uniform_value(cohort, v.id.0)),
        },
        ScalarValue::Scope(kind) => {
            if scalar_is_cohort_uniform(value) {
                Ok(Some(eval_scope_value(*kind, first)))
            } else {
                Ok(None)
            }
        }
        ScalarValue::Expr(e) => {
            if e.args.len() != e.op.arity() {
                return Err(InterpreterError::new(
                    "scalar_eval",
                    "scalar expression arity mismatch",
                ));
            }
            match e.op.arity() {
                1 => {
                    let Some(a0) = eval_scalar_known_uniform(&e.args[0], cohort, scalars)? else {
                        return Ok(None);
                    };
                    eval_scalar_op(e.op, &[a0]).map(Some)
                }
                2 => {
                    let Some(a0) = eval_scalar_known_uniform(&e.args[0], cohort, scalars)? else {
                        return Ok(None);
                    };
                    let Some(a1) = eval_scalar_known_uniform(&e.args[1], cohort, scalars)? else {
                        return Ok(None);
                    };
                    eval_scalar_op(e.op, &[a0, a1]).map(Some)
                }
                3 => {
                    let Some(a0) = eval_scalar_known_uniform(&e.args[0], cohort, scalars)? else {
                        return Ok(None);
                    };
                    let Some(a1) = eval_scalar_known_uniform(&e.args[1], cohort, scalars)? else {
                        return Ok(None);
                    };
                    let Some(a2) = eval_scalar_known_uniform(&e.args[2], cohort, scalars)? else {
                        return Ok(None);
                    };
                    eval_scalar_op(e.op, &[a0, a1, a2]).map(Some)
                }
                _ => unreachable!("validated scalar op arity"),
            }
        }
    }
}

/// Single-thread eval pulling the env from the scalar container.
pub fn eval_scalar_at(
    value: &ScalarValue,
    thread: &ThreadId,
    scalars: &ScalarValues,
) -> IResult<i64> {
    eval_scalar_in_env(value, thread, env_for(scalars, thread))
}

/// Vectorized per-thread eval into an i64 array (uniform values eval once).
pub fn eval_scalar_vec(
    value: &ScalarValue,
    cohort: &ThreadMask,
    scalars: &ScalarValues,
) -> IResult<Array1<i64>> {
    if cohort.is_empty() {
        return Ok(Array1::zeros(0));
    }
    if scalar_is_cohort_uniform(value) {
        let v = eval_scalar_at(value, &cohort[0], scalars)?;
        return Ok(Array1::from_elem(cohort.len(), v));
    }
    let mut out = Array1::<i64>::zeros(cohort.len());
    for (i, t) in cohort.iter().enumerate() {
        out[i] = eval_scalar_in_env(value, t, env_for(scalars, t))?;
    }
    Ok(out)
}

/// Require a value uniform across the cohort, return the single int.
pub fn eval_scalar_uniform(
    value: &ScalarValue,
    cohort: &ThreadMask,
    scalars: &ScalarValues,
    label: &str,
    code: &str,
) -> IResult<i64> {
    if scalar_is_cohort_uniform(value) {
        return eval_scalar_at(value, &cohort[0], scalars);
    }
    if let Some(v) = eval_scalar_known_uniform(value, cohort, scalars)? {
        return Ok(v);
    }
    let first = eval_scalar_at(value, &cohort[0], scalars)?;
    for thread in cohort.iter().skip(1) {
        if eval_scalar_at(value, thread, scalars)? != first {
            return Err(InterpreterError::new(
                code,
                format!("{label} must be uniform"),
            ));
        }
    }
    Ok(first)
}

/// A 'static empty env for threads with no scalar bindings yet.
static EMPTY: std::sync::OnceLock<Env> = std::sync::OnceLock::new();
fn empty_env() -> &'static Env {
    EMPTY.get_or_init(Env::new)
}
