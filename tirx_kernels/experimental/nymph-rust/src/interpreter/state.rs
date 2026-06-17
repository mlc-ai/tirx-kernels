//! Simulator state + run options/result.

use super::diagnostics::Diagnostic;
use super::protocol::{ExecutionMode, ProtocolReport, TraceEvent, TraceState};
use super::tmem::{TmemAllocation, TmemAllocationKey, TmemCollective, TmemCollectiveKey};
use super::values::runtime::RuntimeValues;
use super::values::tensors::{DenseTensorValue, TensorInstanceKey};
use std::collections::HashMap;

/// The single mutable simulator state.
#[derive(Default)]
pub struct InterpreterState {
    pub values: RuntimeValues,
    pub mode: ExecutionMode,
    pub trace: TraceState,
    pub tmem_allocations: HashMap<TmemAllocationKey, TmemAllocation>,
    pub tmem_last_alloc_cols: HashMap<usize, usize>,
    pub tmem_collectives: HashMap<TmemCollectiveKey, TmemCollective>,
    pub cp_async_bulk_groups: HashMap<usize, i64>,
    pub scheduler_next_cursors: HashMap<(u32, usize), usize>,
}

impl InterpreterState {
    pub fn new(mode: ExecutionMode) -> Self {
        Self::new_with_trace_recording(mode, true)
    }

    pub fn new_with_trace_recording(mode: ExecutionMode, record_trace_events: bool) -> Self {
        InterpreterState {
            mode,
            trace: TraceState::new(record_trace_events),
            ..Default::default()
        }
    }
}

/// Execution options.
#[derive(Clone, Debug)]
pub struct RunOptions {
    pub max_rounds: Option<usize>,
    pub max_executed_stmts: Option<usize>,
    pub mode: ExecutionMode,
    pub check_protocol: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            max_rounds: None,
            max_executed_stmts: None,
            mode: ExecutionMode::Trace,
            check_protocol: true,
        }
    }
}

/// Mode-specific successful or checker result payload.
pub enum RunPayload {
    Value {
        outputs: HashMap<TensorInstanceKey, DenseTensorValue>,
    },
    Trace {
        report: ProtocolReport,
        events: Vec<TraceEvent>,
    },
}

/// Public run result.
pub struct RunResult {
    pub completed: bool,
    pub failure_reason: Option<String>,
    pub diagnostics: Vec<Diagnostic>,
    pub payload: Option<RunPayload>,
    /// (stream_id, stmt_id, stmt_type, reason) frontier entries on deadlock.
    pub blocked_frontier: Vec<(usize, usize, String, String)>,
    pub rounds: usize,
    pub executed_stmts: usize,
}
