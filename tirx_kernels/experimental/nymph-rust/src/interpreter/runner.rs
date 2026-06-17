//! The main loop, dispatch, precise wake — direct-mutation model.
//! Executors mutate state directly and return a `StepStatus`. A stream that
//! `Block`s is parked with its `WakeCondition` and skipped; when a later step
//! mutates a wake-relevant cell (listed in `Advance { wakes }`), the runner
//! re-checks each parked waiter on that key and advances the satisfied frames
//! directly — the wait never re-runs. No progress in a full round ⇒ deadlock
//! (a wake only follows a mutating step, which is itself progress).

use super::cohort::CohortContext;
use super::diagnostics::{Diagnostic, IResult};
use super::ids::IdSpace;
use super::outcomes::{StepStatus, WakeCondition};
use super::protocol::{ProtocolStatus, TraceEvent};
use super::registry::{StmtExecutorRegistry, StmtKind};
use super::scheduler::{
    advance_frame_at, current_stmt, CtaActivityStatus, ExecutionStream, SchedulerState,
};
use super::semantics::default_executor_registry;
use super::state::{InterpreterState, RunOptions, RunPayload, RunResult};
use super::threads::ThreadId;
use super::values::arrays::ValueArray1;
use super::values::mbars::MbarCellKey;
use super::values::tensors::{DenseTensorValue, TensorInstanceKey, TensorOwner};
use crate::ir::{MemorySpace, Stmt};
use std::collections::HashMap;
use std::sync::Arc;

static PROFILE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! {
    static PROFILE: std::cell::RefCell<std::collections::HashMap<&'static str, (u64, u128)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}
pub(crate) fn prof_add(name: &'static str, nanos: u128) {
    if !PROFILE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    PROFILE.with(|p| {
        let mut m = p.borrow_mut();
        let e = m.entry(name).or_insert((0, 0));
        e.0 += 1;
        e.1 += nanos;
    });
}
/// Start a timer only when profiling is on (so the hot path pays ~one atomic load,
/// not an `Instant::now()`, when stats are off).
#[inline]
pub(crate) fn prof_now() -> Option<std::time::Instant> {
    if PROFILE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        Some(std::time::Instant::now())
    } else {
        None
    }
}
#[inline]
pub(crate) fn prof_end(name: &'static str, start: Option<std::time::Instant>) {
    if let Some(s) = start {
        prof_add(name, s.elapsed().as_nanos());
    }
}
fn kind_name(k: super::registry::StmtKind) -> &'static str {
    use super::registry::StmtKind::*;
    match k {
        Tcgen05Mma => "exec:Tcgen05Mma",
        Tcgen05Ld => "exec:Tcgen05Ld",
        Tcgen05St => "exec:Tcgen05St",
        Tcgen05Commit => "exec:Tcgen05Commit",
        TmaLoad => "exec:TmaLoad",
        TmaStore => "exec:TmaStore",
        RegCvt => "exec:RegCvt",
        RegLoad | RegStore => "exec:RegLoadStore",
        RegFill => "exec:RegFill",
        RegUnary => "exec:RegUnary",
        RegAdd => "exec:RegAdd",
        RegSub => "exec:RegSub",
        RegMul => "exec:RegMul",
        RegFma => "exec:RegFma",
        RegMax => "exec:RegMax",
        RegMin => "exec:RegMin",
        RegBitwise => "exec:RegBitwise",
        RegReduce => "exec:RegReduce",
        RegCondRescale => "exec:RegCondRescale",
        RegSoftmaxRescale => "exec:RegSoftmaxRescale",
        RegCausalMask => "exec:RegCausalMask",
        RegCombineIntFracEx2 => "exec:RegCombineIntFracEx2",
        StoreScalar => "exec:StoreScalar",
        MBarrierWait => "exec:MBarrierWait",
        MBarrierArrive | MBarrierExpectTx | MBarrierArriveExpectTx | MBarrierInit => {
            "exec:MBarrier*"
        }
        CtaSync | WgSync | WarpSync | ClusterSync => "exec:Sync",
        TmemAlloc | TmemDealloc => "exec:Tmem",
        ForLoop | ForEachTask | SchedulerImpl | SchedNext | Loop | BreakIf | If | Role
        | KernelInit | KernelFinalize => "exec:Control",
        _ => "exec:other",
    }
}

fn prof_report() {
    PROFILE.with(|p| {
        let mut rows: Vec<_> = p.borrow().iter().map(|(k, v)| (*k, v.0, v.1)).collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        eprintln!("nymph_rs profile (phase: calls, total_ms):");
        for (name, calls, nanos) in rows {
            eprintln!("  {name:16} {calls:8} {:9.2} ms", nanos as f64 / 1e6);
        }
        p.borrow_mut().clear();
    });
}

enum StepResult {
    Progress {
        wakes: Vec<MbarCellKey>,
    },
    Blocked {
        condition: WakeCondition,
        completion_event: Option<TraceEvent>,
        frame_idx: usize,
        stmt_id: usize,
        stmt_type: String,
    },
    Failed {
        diagnostic: Option<Diagnostic>,
        reason: Option<String>,
    },
}

/// A parked stream + how to decide it is runnable again. `frame_idx` is the frame
/// the wait sits on; the wake path bumps its pc past the wait (no re-run).
struct Parked {
    frame_idx: usize,
    condition: WakeCondition,
    completion_event: Option<TraceEvent>,
    stmt_id: usize,
    stmt_type: String,
}

pub struct Interpreter<'k> {
    pub kernel: &'k crate::ir::Kernel,
    /// Flat, row-major GMEM inputs tagged with each tensor's declared dtype.
    pub inputs: HashMap<u32, ValueArray1>,
    pub options: RunOptions,
}

pub fn interpret(
    kernel: &crate::ir::Kernel,
    inputs: HashMap<u32, ValueArray1>,
    options: RunOptions,
) -> RunResult {
    Interpreter {
        kernel,
        inputs,
        options,
    }
    .run()
}

impl<'k> Interpreter<'k> {
    pub fn run(self) -> RunResult {
        if std::env::var("NYMPH_STATS").is_ok() {
            PROFILE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // Pin OpenBLAS threads (the 224-core default contends on the tiny MMA tiles).
        let bt = std::env::var("NYMPH_BLAS_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        super::blas::set_threads(bt);
        let Interpreter {
            kernel,
            inputs,
            options,
        } = self;
        let mut diagnostics: Vec<Diagnostic> = Vec::new();
        if let Err(e) = kernel.validate() {
            diagnostics.push(Diagnostic::error("ir_validation", e.message.clone()));
            return RunResult {
                completed: false,
                failure_reason: Some("ir_validation".into()),
                diagnostics,
                payload: None,
                blocked_frontier: Vec::new(),
                rounds: 0,
                executed_stmts: 0,
            };
        }
        let _t_init = std::time::Instant::now();
        let ids = IdSpace::discover(kernel);
        let record_trace_events = !options.mode.is_trace() || options.check_protocol;
        let mut state =
            InterpreterState::new_with_trace_recording(options.mode, record_trace_events);
        let mut scheduler = SchedulerState::from_kernel(kernel);
        let registry: StmtExecutorRegistry = default_executor_registry();
        // Parked streams (Mbar/Rendezvous blocks) + the per-key waiter indexes the
        // wake path uses to find and advance them. PeerActive blocks are NOT parked
        // (they re-run each round — polled).
        let mut parked: HashMap<usize, Parked> = HashMap::new();
        let mut mbar_waiters: HashMap<MbarCellKey, Vec<usize>> = HashMap::new();
        let mut rounds = 0usize;
        let mut executed = 0usize;

        // declare mbars
        declare_mbars(kernel, &mut state);
        prof_add("S:init", _t_init.elapsed().as_nanos());

        // Load supplied GMEM inputs. Value mode usually needs payload arrays; trace
        // mode keeps provided arrays only for scalar/control bridges.
        let _t_load = std::time::Instant::now();
        if options.mode.is_value() || !inputs.is_empty() {
            if let Err(e) = load_inputs(kernel, inputs, &mut state) {
                diagnostics.push(Diagnostic::error(e.code.clone(), e.message.clone()));
                return result(
                    false,
                    Some(e.code),
                    state,
                    kernel,
                    &options,
                    diagnostics,
                    Vec::new(),
                    rounds,
                    executed,
                );
            }
        }
        prof_add("P:load_inputs", _t_load.elapsed().as_nanos());

        // ensure scalar envs
        let _t_se = std::time::Instant::now();
        for mask in &scheduler.cta_thread_masks {
            for t in mask {
                state.values.scalars.ensure_thread(t);
            }
        }
        prof_add("P:scalar_envs", _t_se.elapsed().as_nanos());

        loop {
            let t_mgmt = prof_now();
            for i in 0..scheduler.schedules.len() {
                scheduler.ensure_cta_epoch_streams(i);
            }
            if scheduler.all_completed() {
                prof_end("P:round_mgmt", t_mgmt);
                return completed_result(state, kernel, &options, diagnostics, rounds, executed);
            }
            if let Some(mr) = options.max_rounds {
                if rounds >= mr {
                    diagnostics.push(Diagnostic::error("max_rounds", "exceeded max_rounds"));
                    return result(
                        false,
                        Some("max_rounds".into()),
                        state,
                        kernel,
                        &options,
                        diagnostics,
                        Vec::new(),
                        rounds,
                        executed,
                    );
                }
            }

            // snapshot CTA activity for this round
            let mut activity: Vec<CtaActivityStatus> = (0..scheduler.schedules.len())
                .map(|c| scheduler.cta_activity_status(c))
                .collect();

            // collect live stream ids (skip parked — Mbar/Rendezvous waiters)
            let live: Vec<usize> = scheduler
                .schedules
                .iter()
                .flat_map(|s| s.stream_ids.iter().copied())
                .filter(|sid| !scheduler.streams[*sid].completed && !parked.contains_key(sid))
                .collect();
            prof_end("P:round_mgmt", t_mgmt);

            let mut progress = false;
            // PeerActive blocks this round (polled — reported for deadlock detail).
            let mut blocked_frontier: Vec<(usize, usize, String, String)> = Vec::new();

            for sid in live {
                let t_step = prof_now();
                let cta_id = scheduler.streams[sid].cta_id;
                let step = step_stream(
                    &mut scheduler.streams[sid],
                    &mut state,
                    &ids,
                    kernel,
                    &options,
                    &activity,
                    &registry,
                    &mut executed,
                );
                prof_end("P:step_call", t_step);
                if cta_id < activity.len() {
                    activity[cta_id] = scheduler.cta_activity_status(cta_id);
                }
                match step {
                    StepResult::Progress { wakes } => {
                        progress = true;
                        for key in wakes {
                            if let Err(e) = process_wake(
                                key,
                                &mut parked,
                                &mut mbar_waiters,
                                &mut scheduler.streams,
                                &mut state,
                            ) {
                                diagnostics
                                    .push(Diagnostic::error(e.code.clone(), e.message.clone()));
                                return result(
                                    false,
                                    Some(e.code),
                                    state,
                                    kernel,
                                    &options,
                                    diagnostics,
                                    Vec::new(),
                                    rounds,
                                    executed,
                                );
                            }
                        }
                    }
                    StepResult::Blocked {
                        condition,
                        completion_event,
                        frame_idx,
                        stmt_id,
                        stmt_type,
                    } => match condition {
                        WakeCondition::Mbar { key, .. } => {
                            mbar_waiters.entry(key).or_default().push(sid);
                            parked.insert(
                                sid,
                                Parked {
                                    frame_idx,
                                    condition,
                                    completion_event,
                                    stmt_id,
                                    stmt_type,
                                },
                            );
                        }
                        WakeCondition::Polled => {
                            blocked_frontier.push((sid, stmt_id, stmt_type.clone(), stmt_type));
                        }
                    },
                    StepResult::Failed { diagnostic, reason } => {
                        let reason = reason.unwrap_or_else(|| {
                            diagnostic
                                .as_ref()
                                .map(|d| d.code.clone())
                                .unwrap_or_else(|| "failed".into())
                        });
                        if let Some(d) = diagnostic {
                            diagnostics.push(d);
                        }
                        return result(
                            false,
                            Some(reason),
                            state,
                            kernel,
                            &options,
                            diagnostics,
                            Vec::new(),
                            rounds,
                            executed,
                        );
                    }
                }
            }

            if scheduler.all_completed() {
                return completed_result(state, kernel, &options, diagnostics, rounds, executed);
            }
            if !progress {
                let mut frontier = blocked_frontier;
                for (sid, p) in &parked {
                    frontier.push((*sid, p.stmt_id, p.stmt_type.clone(), p.stmt_type.clone()));
                }
                diagnostics.push(
                    Diagnostic::error("deadlock", "no progress in a full round")
                        .with_detail("blocked", frontier.len().to_string()),
                );
                return result(
                    false,
                    Some("deadlock".into()),
                    state,
                    kernel,
                    &options,
                    diagnostics,
                    frontier,
                    rounds,
                    executed,
                );
            }
            rounds += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn step_stream<'k>(
    stream: &mut ExecutionStream<'k>,
    state: &mut InterpreterState,
    ids: &IdSpace,
    kernel: &'k crate::ir::Kernel,
    options: &RunOptions,
    activity: &[CtaActivityStatus],
    registry: &StmtExecutorRegistry,
    executed: &mut usize,
) -> StepResult {
    loop {
        let tcs = prof_now();
        let cs = current_stmt(stream, &mut state.values.scalars);
        prof_end("current_stmt", tcs);
        let (stmt, mask, frame_idx) = match cs {
            None => return StepResult::Progress { wakes: Vec::new() },
            Some(c) => c,
        };
        let current_stmt_id = ids.stmt_id(stmt);
        let current_stmt_kind = super::registry::stmt_kind(stmt);
        let anchor_thread = mask.first().copied();
        let cohort_size = mask.len();
        let executor = match registry.executor_for(stmt) {
            Some(e) => e,
            None => {
                return StepResult::Failed {
                    diagnostic: Some(anchor_step_diagnostic(
                        Diagnostic::error("internal_error", "no executor and no fallback"),
                        stream,
                        current_stmt_id,
                        current_stmt_kind,
                        anchor_thread,
                        cohort_size,
                    )),
                    reason: None,
                }
            }
        };
        // run the executor — it mutates `state` directly via `&mut` ctx
        let texe = prof_now();
        let status = {
            let mut ctx = CohortContext {
                kernel,
                stream,
                cohort: mask,
                state,
                ids,
                options,
                cta_activity: activity,
                current_stmt_id,
                current_stmt_kind,
            };
            executor(&mut ctx, stmt)
        };
        if texe.is_some() {
            prof_end(kind_name(super::registry::stmt_kind(stmt)), texe);
        }
        let status = match status {
            Ok(s) => s,
            Err(e) => {
                if options.mode.is_trace() && e.code == "trace_inconclusive" {
                    return StepResult::Failed {
                        diagnostic: None,
                        reason: Some("trace_inconclusive".into()),
                    };
                }
                return StepResult::Failed {
                    diagnostic: Some(anchor_step_diagnostic(
                        Diagnostic::error(e.code, e.message),
                        stream,
                        current_stmt_id,
                        current_stmt_kind,
                        anchor_thread,
                        cohort_size,
                    )),
                    reason: None,
                };
            }
        };
        match status {
            StepStatus::Advance { wakes } => {
                if let Some(limit) = options.max_executed_stmts {
                    if *executed + 1 > limit {
                        return StepResult::Failed {
                            diagnostic: Some(anchor_step_diagnostic(
                                Diagnostic::error("trace_limit", "exceeded max_executed_stmts"),
                                stream,
                                current_stmt_id,
                                current_stmt_kind,
                                anchor_thread,
                                cohort_size,
                            )),
                            reason: None,
                        };
                    }
                }
                advance_frame_at(stream, frame_idx);
                *executed += 1;
                stream.stream_step += 1;
                return StepResult::Progress { wakes };
            }
            StepStatus::AdvanceContinue => {
                advance_frame_at(stream, frame_idx);
                continue;
            }
            StepStatus::Block {
                condition,
                completion_event,
            } => {
                let stmt_id = ids.stmt_id(stmt); // only computed when actually blocking
                let stmt_type = format!("{:?}", super::registry::stmt_kind(stmt));
                return StepResult::Blocked {
                    condition,
                    completion_event,
                    frame_idx,
                    stmt_id,
                    stmt_type,
                };
            }
            StepStatus::Fail(diagnostic) => {
                return StepResult::Failed {
                    diagnostic: diagnostic.map(|d| {
                        anchor_step_diagnostic(
                            d,
                            stream,
                            current_stmt_id,
                            current_stmt_kind,
                            anchor_thread,
                            cohort_size,
                        )
                    }),
                    reason: None,
                }
            }
        }
    }
}

fn anchor_step_diagnostic(
    mut diagnostic: Diagnostic,
    stream: &ExecutionStream<'_>,
    stmt_id: usize,
    stmt_kind: StmtKind,
    anchor_thread: Option<ThreadId>,
    cohort_size: usize,
) -> Diagnostic {
    if diagnostic.stream_id.is_none() {
        diagnostic.stream_id = Some(stream.stream_id);
    }
    if diagnostic.stmt_id.is_none() {
        diagnostic.stmt_id = Some(stmt_id.to_string());
    }
    if diagnostic.thread.is_none() {
        diagnostic.thread = anchor_thread;
    }
    diagnostic
        .details
        .entry("stmt_kind".into())
        .or_insert_with(|| format!("{stmt_kind:?}"));
    diagnostic
        .details
        .entry("cohort_size".into())
        .or_insert_with(|| cohort_size.to_string());
    diagnostic
        .details
        .entry("stream_step".into())
        .or_insert_with(|| stream.stream_step.to_string());
    diagnostic
        .details
        .entry("cta_id".into())
        .or_insert_with(|| stream.cta_id.to_string());
    diagnostic
}

/// A mutating step touched mbarrier cell `key`; re-check every parked waiter on it
/// and advance the satisfied ones' frames directly (no re-run). Unsatisfied waiters
/// stay parked (re-indexed). Advanced streams step their next stmt next round.
fn process_wake<'k>(
    key: MbarCellKey,
    parked: &mut HashMap<usize, Parked>,
    mbar_waiters: &mut HashMap<MbarCellKey, Vec<usize>>,
    streams: &mut [ExecutionStream<'k>],
    state: &mut InterpreterState,
) -> IResult<()> {
    let total_timer = prof_now();
    let timer = prof_now();
    let sids = match mbar_waiters.remove(&key) {
        Some(v) => v,
        None => {
            prof_end("Wake:lookup", timer);
            prof_end("Wake:process", total_timer);
            return Ok(());
        }
    };
    prof_end("Wake:lookup", timer);
    let mut still: Vec<usize> = Vec::new();
    let scan_timer = prof_now();
    for wsid in sids {
        let timer = prof_now();
        let satisfied = parked
            .get(&wsid)
            .map_or(false, |p| p.condition.satisfied(state));
        prof_end("Wake:satisfied", timer);
        if satisfied {
            let timer = prof_now();
            let p = parked.remove(&wsid).unwrap();
            prof_end("Wake:remove", timer);
            if let Some(event) = p.completion_event {
                let timer = prof_now();
                state.trace.emit(event)?;
                prof_end("Wake:emit_completion", timer);
            }
            let timer = prof_now();
            advance_frame_at(&mut streams[wsid], p.frame_idx);
            prof_end("Wake:advance", timer);
        } else {
            still.push(wsid);
        }
    }
    prof_end("Wake:scan", scan_timer);
    if !still.is_empty() {
        let timer = prof_now();
        mbar_waiters.insert(key, still);
        prof_end("Wake:reinsert", timer);
    }
    prof_end("Wake:process", total_timer);
    Ok(())
}

fn declare_mbars(kernel: &crate::ir::Kernel, state: &mut InterpreterState) {
    fn walk(body: &[Stmt], state: &mut InterpreterState) {
        for stmt in body {
            if let Stmt::MBarDef { mbar } = stmt {
                state
                    .values
                    .mbars
                    .declare(mbar.id, mbar.arrive_count.map(|c| c as i64));
            }
            for child in stmt.child_bodies() {
                walk(child, state);
            }
        }
    }
    walk(&kernel.body, state);
}

/// Seed the GMEM inputs. Takes already-flat native arrays by value and moves
/// each into its tensor instance.
fn load_inputs(
    kernel: &crate::ir::Kernel,
    mut inputs: HashMap<u32, ValueArray1>,
    state: &mut InterpreterState,
) -> super::diagnostics::IResult<()> {
    for tensor in &kernel.args {
        if let Some(flat) = inputs.remove(&tensor.id) {
            if tensor.space != MemorySpace::Gmem {
                return Err(super::diagnostics::InterpreterError::new(
                    "unsupported_input",
                    "input must be GMEM",
                ));
            }
            if flat.dtype() != tensor.dtype {
                return Err(super::diagnostics::InterpreterError::new(
                    "tensor_value",
                    "input array dtype mismatch",
                ));
            }
            if flat.len() != super::values::indexing::numel(&tensor.shape) {
                return Err(super::diagnostics::InterpreterError::new(
                    "tensor_value",
                    "input array shape mismatch",
                ));
            }
            let key = TensorInstanceKey {
                tensor: Arc::clone(tensor),
                owner: TensorOwner::Global,
            };
            let dense = DenseTensorValue::from_native(flat, tensor.shape.clone(), tensor.dtype)?;
            state.values.tensors.by_instance.insert(key, dense);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn result(
    completed: bool,
    failure_reason: Option<String>,
    mut state: InterpreterState,
    kernel: &crate::ir::Kernel,
    options: &RunOptions,
    diagnostics: Vec<Diagnostic>,
    blocked_frontier: Vec<(usize, usize, String, String)>,
    rounds: usize,
    executed: usize,
) -> RunResult {
    let _ = kernel;
    // Direct mutation means a failed run may have partially changed state before
    // discovering the fatal condition. Do not expose that state as a public
    // snapshot; callers may inspect diagnostics, not values/outputs.
    if !completed {
        let payload = if options.mode.is_trace() {
            let status = if failure_reason.as_deref() == Some("trace_inconclusive") {
                ProtocolStatus::Inconclusive
            } else {
                ProtocolStatus::Failed
            };
            let (report, events) = state
                .trace
                .finish(status, diagnostics.clone())
                .expect("protocol trace finish cannot fail");
            Some(RunPayload::Trace { report, events })
        } else {
            None
        };
        return RunResult {
            completed,
            failure_reason,
            diagnostics,
            payload,
            blocked_frontier,
            rounds,
            executed_stmts: executed,
        };
    }

    let payload = if options.mode.is_value() {
        Some(RunPayload::Value {
            outputs: std::mem::take(&mut state.values.tensors.by_instance),
        })
    } else if options.mode.is_trace() {
        let (report, events) = state
            .trace
            .finish(ProtocolStatus::Passed, diagnostics.clone())
            .expect("protocol trace finish cannot fail");
        let report = if options.check_protocol {
            super::checker::run_offline_checkers(kernel, report, &events)
        } else {
            report
        };
        Some(RunPayload::Trace { report, events })
    } else {
        None
    };
    RunResult {
        completed,
        failure_reason,
        diagnostics,
        payload,
        blocked_frontier,
        rounds,
        executed_stmts: executed,
    }
}

fn completed_result(
    state: InterpreterState,
    kernel: &crate::ir::Kernel,
    options: &RunOptions,
    mut diagnostics: Vec<Diagnostic>,
    rounds: usize,
    executed: usize,
) -> RunResult {
    if !state.tmem_allocations.is_empty() {
        diagnostics.push(
            Diagnostic::error(
                "leaked_tmem_allocation",
                "TMEM allocations leaked at completion",
            )
            .with_detail("count", state.tmem_allocations.len().to_string()),
        );
        return result(
            false,
            Some("leaked_tmem_allocation".into()),
            state,
            kernel,
            options,
            diagnostics,
            Vec::new(),
            rounds,
            executed,
        );
    }
    if std::env::var("NYMPH_STATS").is_ok() {
        eprintln!("nymph_rs: rounds={rounds} executed_stmts={executed}");
        prof_report();
    }
    result(
        true,
        None,
        state,
        kernel,
        options,
        diagnostics,
        Vec::new(),
        rounds,
        executed,
    )
}
