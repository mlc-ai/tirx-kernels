//! The nymph interpreter — a Rust port of `interpreter/`.
//!
//! Modular per-op dispatch: a registry maps each `Stmt` variant to an executor;
//! semantics modules register their executors, so adding an op does not edit the
//! runner. Executors operate on cohort-vectorized masks, mutate state directly
//! through `&mut state`, and return a light `StepStatus`. The one blocking op on
//! the hot path — `mbarrier_wait` — parks on a `WakeCondition` and is advanced
//! precisely (never re-run) when a later cell write flips its parity; rare
//! rendezvous/collective/peer blocks stay polled.
//!
//! File organization mirrors the Python package.

pub mod blas;
pub mod checker;
pub mod cohort;
pub mod diagnostics;
pub mod elementwise;
pub mod ids;
pub mod mbar_ops;
pub mod outcomes;
pub mod protocol;
pub mod region;
pub mod registry;
pub mod runner;
pub mod scalar_eval;
pub mod scheduler;
pub mod semantics;
pub mod slice_indexing;
pub mod state;
pub mod threads;
pub mod tmem;
pub mod transfer;
pub mod values;

pub use protocol::{
    ExecutionMode, ProtocolReport, ProtocolStatus, ProtocolWarning, TraceEvent, TraceEventKind,
};
pub use runner::interpret;
pub use state::{RunOptions, RunPayload, RunResult};
