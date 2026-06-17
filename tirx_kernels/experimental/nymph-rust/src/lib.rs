//! nymph — a Rust port of the nymph GPU-kernel value simulator.
//!
//! The library crate. Pure-Rust modules (the IR, later the interpreter) plus a
//! thin PyO3 layer (gated behind the `python` feature) that exposes them to the
//! Python builder/tests as the `nymph_rs` extension module.

pub mod interpreter;
pub mod ir; // the IR types + validation (faithful port of `ir/ir.py`) // the value simulator (faithful port of `interpreter/`)

#[cfg(feature = "python")]
mod py; // PyO3 bindings (the `nymph_rs` module registration)

/// Crate version marker.
pub fn version() -> &'static str {
    "nymph-rust 0.1.0"
}

// ---------------------------------------------------------------------------
// PyO3 bindings (only compiled with `--features python`, i.e. by maturin).
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Smoke test: prove Python can call into Rust.
#[cfg(feature = "python")]
#[pyfunction]
fn hello() -> String {
    format!("hello from rust — {}", version())
}

/// The `nymph_rs` Python module. Everything Python can use is registered here.
#[cfg(feature = "python")]
#[pymodule]
fn nymph_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    py::register(m)?;
    Ok(())
}
