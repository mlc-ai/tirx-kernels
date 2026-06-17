//! Diagnostics + the fail-closed interpreter error — port of `diagnostics.py`.

use super::threads::ThreadId;
use std::collections::BTreeMap;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// One anchored interpreter diagnostic (public run record).
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub thread: Option<ThreadId>,
    pub stream_id: Option<usize>,
    pub stmt_id: Option<String>,
    pub details: BTreeMap<String, String>,
}

impl Diagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            thread: None,
            stream_id: None,
            stmt_id: None,
            details: BTreeMap::new(),
        }
    }
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

/// Internal fail-closed error raised before a statement's commit boundary.
/// Carries a stable `code` + a human `message` (mirrors Python's InterpreterError).
#[derive(Clone, Debug)]
pub struct InterpreterError {
    pub code: String,
    pub message: String,
}

impl InterpreterError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        InterpreterError {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for InterpreterError {}

pub type IResult<T> = Result<T, InterpreterError>;

/// Fill missing anchors on diagnostics from the current execution context.
pub fn attach_thread_to_diagnostics(
    diagnostics: &[Diagnostic],
    thread: &ThreadId,
    stream_id: usize,
    stmt_id: &str,
) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .map(|d| {
            let mut nd = d.clone();
            if nd.thread.is_none() {
                nd.thread = Some(thread.clone());
            }
            if nd.stream_id.is_none() {
                nd.stream_id = Some(stream_id);
            }
            if nd.stmt_id.is_none() {
                nd.stmt_id = Some(stmt_id.to_string());
            }
            nd
        })
        .collect()
}
