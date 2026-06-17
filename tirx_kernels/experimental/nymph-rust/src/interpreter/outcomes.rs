//! Executor result + wake protocol — the direct-mutation model.
//!
//! Executors mutate `ctx.state` directly and return a light `StepStatus`. The hot
//! blocking op — `mbarrier_wait` — parks on `Block { condition: Mbar { .. } }`
//! and is advanced (never re-run) when a later cell write flips the parity: the
//! mutating op lists the touched cell keys in `Advance { wakes }`, and the runner
//! re-checks each parked waiter on those keys and advances the satisfied frames
//! directly. The rare rendezvous / collective / peer-active blocks use
//! `Block { condition: Polled }` — re-run each round; their re-runs are naturally
//! idempotent (set unions / re-checks), so no first-block dedup is needed.

use super::diagnostics::Diagnostic;
use super::protocol::TraceEvent;
use super::state::InterpreterState;
use super::values::mbars::MbarCellKey;

/// What an executor did this step.
pub enum StepStatus {
    /// Mutated state; the runner advances this frame and re-checks `wakes`' waiters.
    Advance { wakes: Vec<MbarCellKey> },
    /// A structural op pushed a child frame; re-loop without returning.
    AdvanceContinue,
    /// Park the stream on this condition. The block path wrote at most a one-time,
    /// idempotent arrival record before returning.
    Block {
        condition: WakeCondition,
        completion_event: Option<TraceEvent>,
    },
    /// Abort the run.
    Fail(Option<Diagnostic>),
}

impl StepStatus {
    pub fn advance() -> Self {
        StepStatus::Advance { wakes: Vec::new() }
    }
    pub fn advance_wake(wakes: Vec<MbarCellKey>) -> Self {
        StepStatus::Advance { wakes }
    }
    pub fn advance_continue() -> Self {
        StepStatus::AdvanceContinue
    }
    pub fn block(condition: WakeCondition) -> Self {
        StepStatus::Block {
            condition,
            completion_event: None,
        }
    }
    pub fn block_with_completion_event(condition: WakeCondition, event: TraceEvent) -> Self {
        StepStatus::Block {
            condition,
            completion_event: Some(event),
        }
    }
}

/// Why a stream is parked, and how the runner decides it is runnable again.
#[derive(Clone)]
pub enum WakeCondition {
    /// Runnable once `cells[key].parity != phase` (the awaited mbarrier flip). The
    /// runner advances it precisely, without re-running the wait.
    Mbar { key: MbarCellKey, phase: u8 },
    /// Re-run each round (rare rendezvous / tmem collective / peer-active blocks).
    Polled,
}

impl WakeCondition {
    /// Inline re-check the runner runs at mbar wake time (no executor re-run).
    pub fn satisfied(&self, state: &InterpreterState) -> bool {
        match self {
            WakeCondition::Mbar { key, phase } => state
                .values
                .mbars
                .cells
                .get(key)
                .map_or(false, |c| c.parity != *phase),
            WakeCondition::Polled => false,
        }
    }
}
