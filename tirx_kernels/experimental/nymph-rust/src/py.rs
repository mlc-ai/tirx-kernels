//! PyO3 bindings — exposes the Rust IR to Python as the `nymph_rs` module, so the
//! Python builder constructs Rust IR. Only compiled with `--features python`.
//!
//! Built up in chunks: enums → scalars → tensors → mbars → statements → kernel.

use crate::interpreter::diagnostics::{Diagnostic, Severity};
use crate::interpreter::protocol::{
    AccessScope, BoxN, ExecutionMode, MbarTargetEvent, PoolId, ProtocolPassSummary,
    ProtocolWarning, Region, TraceEvent, TraceEventKind,
};
use crate::interpreter::threads::ThreadId;
use crate::interpreter::values::arrays::ValueArray1;
use crate::interpreter::values::tensors::TensorOwner;
use crate::interpreter::{self, RunOptions, RunPayload};
use crate::ir;
use numpy::{Element, IntoPyArray, PyReadonlyArrayDyn};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFrozenSet, PyList, PySlice, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Run the value simulator on a kernel: `inputs` maps each GMEM arg Tensor to a
/// numeric numpy array; returns `{tensor_id: numpy array}` for produced GMEM tensors.
#[pyfunction]
fn interpret(
    py: Python<'_>,
    kernel: &PyKernel,
    inputs: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let stats = std::env::var("NYMPH_STATS").is_ok();
    let t_in = std::time::Instant::now();
    let (rust_inputs, input_ids) = coerce_py_inputs(py, inputs)?;
    if stats {
        eprintln!(
            "  PY:input_copy {:.1} ms",
            t_in.elapsed().as_secs_f64() * 1e3
        );
    }
    let result = interpreter::interpret(
        &kernel.0,
        rust_inputs,
        RunOptions {
            mode: ExecutionMode::Value,
            ..Default::default()
        },
    );
    if !result.completed {
        return Err(PyRuntimeError::new_err(run_failure_message(
            "interpret did not complete",
            &result,
        )));
    }
    let t_out = std::time::Instant::now();
    let out = PyDict::new(py);
    if let Some(RunPayload::Value { outputs }) = result.payload {
        for (key, dense) in outputs {
            // Return only GMEM tensors the kernel produced — skip the inputs we were
            // handed (copying A/B straight back out is pure waste).
            if matches!(key.owner, TensorOwner::Global) && !input_ids.contains(&key.tensor.id) {
                set_output(py, &out, key.tensor.id, &key.tensor.shape, &dense.data)?;
            }
        }
    }
    if stats {
        eprintln!(
            "  PY:output_copy {:.1} ms",
            t_out.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(out.unbind())
}

fn run_failure_message(prefix: &str, result: &interpreter::RunResult) -> String {
    use std::fmt::Write;

    let mut out = format!(
        "{prefix}: reason={:?}, rounds={}, executed_stmts={}",
        result.failure_reason, result.rounds, result.executed_stmts
    );
    if !result.diagnostics.is_empty() {
        let _ = write!(out, "\ndiagnostics:");
        for (idx, diagnostic) in result.diagnostics.iter().enumerate() {
            let _ = write!(
                out,
                "\n  {idx}: [{}] {}",
                diagnostic.code, diagnostic.message
            );
            if let Some(stmt_id) = &diagnostic.stmt_id {
                let _ = write!(out, " stmt_id={stmt_id}");
            }
            if let Some(stream_id) = diagnostic.stream_id {
                let _ = write!(out, " stream_id={stream_id}");
            }
            if let Some(thread) = diagnostic.thread {
                let _ = write!(
                    out,
                    " thread=cta{}:warp{}:lane{}",
                    thread.cta_id, thread.warp_id, thread.lane_id
                );
            }
            if !diagnostic.details.is_empty() {
                let detail = diagnostic
                    .details
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = write!(out, " details={{ {detail} }}");
            }
        }
    }
    if !result.blocked_frontier.is_empty() {
        let _ = write!(
            out,
            "\nblocked_frontier={} entries:",
            result.blocked_frontier.len()
        );
        for (idx, (stream_id, stmt_id, stmt_type, reason)) in
            result.blocked_frontier.iter().take(8).enumerate()
        {
            let _ = write!(
                out,
                "\n  {idx}: stream_id={stream_id} stmt_id={stmt_id} stmt_type={stmt_type} reason={reason}"
            );
        }
        if result.blocked_frontier.len() > 8 {
            let _ = write!(out, "\n  ...");
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (kernel, inputs = None, include_events = false))]
fn check_protocol(
    py: Python<'_>,
    kernel: &PyKernel,
    inputs: Option<&Bound<'_, PyDict>>,
    include_events: bool,
) -> PyResult<Py<PyDict>> {
    let rust_inputs = match inputs {
        Some(inputs) => coerce_py_inputs(py, inputs)?.0,
        None => HashMap::new(),
    };
    let result = interpreter::interpret(
        &kernel.0,
        rust_inputs,
        RunOptions {
            mode: ExecutionMode::Trace,
            ..Default::default()
        },
    );
    match result.payload {
        Some(RunPayload::Trace { report, events }) => {
            let out = PyDict::new(py);
            out.set_item("status", report.status.as_str())?;
            out.set_item("warnings", warnings_to_py(py, &report.warnings)?)?;
            out.set_item("diagnostics", diagnostics_to_py(py, &report.diagnostics)?)?;
            out.set_item(
                "pass_summary",
                pass_summary_to_py(py, &report.pass_summary)?,
            )?;
            out.set_item("event_count", events.len())?;
            // The event stream is the trace itself (and the substrate the future HB/region
            // checker will consume), but marshalling it to Python is opt-in: a large kernel
            // emits ~20k events/task and the default callers (perf, pass/fail) do not need
            // it in Python. Request it with include_events=True (e.g. event-stream tests).
            if include_events {
                out.set_item("events", events_to_py(py, &events)?)?;
            }
            out.set_item("rounds", result.rounds)?;
            out.set_item("executed_stmts", result.executed_stmts)?;
            Ok(out.unbind())
        }
        _ => Err(PyRuntimeError::new_err(run_failure_message(
            "check_protocol did not produce a trace report",
            &result,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (kernel, inputs = None))]
fn trace(
    py: Python<'_>,
    kernel: &PyKernel,
    inputs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDict>> {
    let rust_inputs = match inputs {
        Some(inputs) => coerce_py_inputs(py, inputs)?.0,
        None => HashMap::new(),
    };
    let result = interpreter::interpret(
        &kernel.0,
        rust_inputs,
        RunOptions {
            mode: ExecutionMode::Trace,
            check_protocol: false,
            ..Default::default()
        },
    );
    match result.payload {
        Some(RunPayload::Trace { report, .. }) => {
            let out = PyDict::new(py);
            out.set_item("status", report.status.as_str())?;
            out.set_item("completed", result.completed)?;
            out.set_item("rounds", result.rounds)?;
            out.set_item("executed_stmts", result.executed_stmts)?;
            if let Some(reason) = result.failure_reason {
                out.set_item("failure_reason", reason)?;
            }
            Ok(out.unbind())
        }
        _ => Err(PyRuntimeError::new_err(run_failure_message(
            "trace did not produce a trace report",
            &result,
        ))),
    }
}

fn coerce_py_inputs(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
) -> PyResult<(HashMap<u32, ValueArray1>, std::collections::HashSet<u32>)> {
    let mut rust_inputs: HashMap<u32, ValueArray1> = HashMap::new();
    for (k, v) in inputs.iter() {
        let tensor = k.extract::<PyTensor>()?;
        let flat = coerce_py_input(py, &v, &tensor.0)?;
        rust_inputs.insert(tensor.0.id, flat);
    }
    let input_ids = rust_inputs.keys().copied().collect();
    Ok((rust_inputs, input_ids))
}

fn numpy_dtype_name(dtype: ir::DType) -> &'static str {
    match dtype {
        ir::DType::Bool => "bool",
        ir::DType::I8 => "int8",
        ir::DType::U8 => "uint8",
        ir::DType::I16 => "int16",
        ir::DType::U16 => "uint16",
        ir::DType::I32 => "int32",
        ir::DType::U32 => "uint32",
        ir::DType::I64 => "int64",
        ir::DType::U64 => "uint64",
        ir::DType::F16 => "float16",
        ir::DType::F8E4M3 | ir::DType::Bf16 | ir::DType::F32 => "float32",
    }
}

fn check_input_shape<T: Element>(arr: &PyReadonlyArrayDyn<'_, T>, shape: &[usize]) -> PyResult<()> {
    if arr.as_array().shape() != shape {
        return Err(PyValueError::new_err("input array shape mismatch"));
    }
    Ok(())
}

fn copy_py_array<T: Element + Copy>(arr: &PyReadonlyArrayDyn<'_, T>) -> ndarray::Array1<T> {
    match arr.as_slice() {
        Ok(s) => ndarray::Array1::from(s.to_vec()),
        Err(_) => ndarray::Array1::from_iter(arr.as_array().iter().copied()),
    }
}

fn coerce_py_input(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    tensor: &ir::Tensor,
) -> PyResult<ValueArray1> {
    if tensor.dtype == ir::DType::F16 {
        if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<f32>>() {
            check_input_shape(&arr, &tensor.shape)?;
            return Ok(f16_input_from_f32(&arr));
        }
    }
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy_dtype_name(tensor.dtype))?;
    let coerced = np.getattr("asarray")?.call((value,), Some(&kwargs))?;
    Ok(match tensor.dtype {
        ir::DType::Bool => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<bool>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::Bool(copy_py_array(&arr))
        }
        ir::DType::I8 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<i8>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::I8(copy_py_array(&arr))
        }
        ir::DType::U8 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<u8>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::U8(copy_py_array(&arr))
        }
        ir::DType::I16 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<i16>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::I16(copy_py_array(&arr))
        }
        ir::DType::U16 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<u16>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::U16(copy_py_array(&arr))
        }
        ir::DType::I32 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<i32>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::I32(copy_py_array(&arr))
        }
        ir::DType::U32 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<u32>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::U32(copy_py_array(&arr))
        }
        ir::DType::I64 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<i64>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::I64(copy_py_array(&arr))
        }
        ir::DType::U64 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<u64>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::U64(copy_py_array(&arr))
        }
        ir::DType::F16 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<half::f16>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            f16_input_from_f16(&arr)
        }
        ir::DType::F8E4M3 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<f32>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::from_f32_compute(copy_py_array(&arr), ir::DType::F8E4M3)
        }
        ir::DType::Bf16 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<f32>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::from_f32_compute(copy_py_array(&arr), ir::DType::Bf16)
        }
        ir::DType::F32 => {
            let arr = coerced.extract::<PyReadonlyArrayDyn<f32>>()?;
            check_input_shape(&arr, &tensor.shape)?;
            ValueArray1::F32(copy_py_array(&arr))
        }
    })
}

fn f16_input_from_f32(arr: &PyReadonlyArrayDyn<'_, f32>) -> ValueArray1 {
    let values = match arr.as_slice() {
        Ok(values) => ndarray::Array1::from(
            values
                .par_iter()
                .map(|&x| half::f16::from_f32(x).to_f32())
                .collect::<Vec<f32>>(),
        ),
        Err(_) => ndarray::Array1::from_iter(
            arr.as_array()
                .iter()
                .map(|&x| half::f16::from_f32(x).to_f32()),
        ),
    };
    ValueArray1::F16(values)
}

fn f16_input_from_f16(arr: &PyReadonlyArrayDyn<'_, half::f16>) -> ValueArray1 {
    let native = copy_py_array(arr);
    let values = match native.as_slice() {
        Some(values) => {
            ndarray::Array1::from(values.par_iter().map(|x| x.to_f32()).collect::<Vec<f32>>())
        }
        None => native.mapv(|x| x.to_f32()),
    };
    ValueArray1::F16(values)
}

fn set_output(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    tensor_id: u32,
    shape: &[usize],
    data: &ValueArray1,
) -> PyResult<()> {
    macro_rules! set_arr {
        ($a:expr) => {{
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), $a.to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            out.set_item(tensor_id, arr.into_pyarray(py))?;
        }};
    }
    match data {
        ValueArray1::Bool(a) => set_arr!(a),
        ValueArray1::I8(a) => set_arr!(a),
        ValueArray1::U8(a) => set_arr!(a),
        ValueArray1::I16(a) => set_arr!(a),
        ValueArray1::U16(a) => set_arr!(a),
        ValueArray1::I32(a) => set_arr!(a),
        ValueArray1::U32(a) => set_arr!(a),
        ValueArray1::I64(a) => set_arr!(a),
        ValueArray1::U64(a) => set_arr!(a),
        ValueArray1::F16(a) => {
            let values: Vec<half::f16> = a.iter().map(|&x| half::f16::from_f32(x)).collect();
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), values)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            out.set_item(tensor_id, arr.into_pyarray(py))?;
        }
        ValueArray1::F8E4M3(a) => set_arr!(a),
        ValueArray1::Bf16(a) => set_arr!(a),
        ValueArray1::F32(a) => set_arr!(a),
    }
    Ok(())
}

fn warnings_to_py<'py>(
    py: Python<'py>,
    warnings: &[ProtocolWarning],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for warning in warnings {
        let item = PyDict::new(py);
        item.set_item("code", &warning.code)?;
        item.set_item("message", &warning.message)?;
        let details = PyDict::new(py);
        for (key, value) in &warning.details {
            details.set_item(key, value)?;
        }
        item.set_item("details", details)?;
        out.append(item)?;
    }
    Ok(out)
}

fn pass_summary_to_py<'py>(
    py: Python<'py>,
    summaries: &[ProtocolPassSummary],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for summary in summaries {
        let item = PyDict::new(py);
        item.set_item("name", &summary.name)?;
        item.set_item("status", summary.status.as_str())?;
        item.set_item("diagnostics", summary.diagnostics)?;
        item.set_item("warnings", summary.warnings)?;
        out.append(item)?;
    }
    Ok(out)
}

fn diagnostics_to_py<'py>(
    py: Python<'py>,
    diagnostics: &[Diagnostic],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for diagnostic in diagnostics {
        out.append(diagnostic_to_py(py, diagnostic)?)?;
    }
    Ok(out)
}

fn diagnostic_to_py<'py>(py: Python<'py>, diagnostic: &Diagnostic) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "severity",
        match diagnostic.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        },
    )?;
    out.set_item("code", &diagnostic.code)?;
    out.set_item("message", &diagnostic.message)?;
    out.set_item("stream_id", diagnostic.stream_id)?;
    out.set_item("stmt_id", &diagnostic.stmt_id)?;
    if let Some(thread) = diagnostic.thread {
        out.set_item("thread", thread_to_py(py, &thread)?)?;
    } else {
        out.set_item("thread", py.None())?;
    }
    let details = PyDict::new(py);
    for (key, value) in &diagnostic.details {
        details.set_item(key, value)?;
    }
    out.set_item("details", details)?;
    Ok(out)
}

fn thread_to_py<'py>(py: Python<'py>, thread: &ThreadId) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("cta_id", thread.cta_id)?;
    out.set_item("cta_coord", thread.cta_coord.to_vec())?;
    out.set_item("cluster_id", thread.cluster_id)?;
    out.set_item("ctaid_in_cluster", thread.ctaid_in_cluster)?;
    out.set_item("cluster_coord", thread.cluster_coord.to_vec())?;
    out.set_item("cta_coord_in_cluster", thread.cta_coord_in_cluster.to_vec())?;
    out.set_item("warp_id", thread.warp_id)?;
    out.set_item("lane_id", thread.lane_id)?;
    out.set_item("warpgroup_id", thread.warpgroup_id())?;
    out.set_item("tid_in_wg", thread.tid_in_wg())?;
    Ok(out)
}

fn events_to_py<'py>(py: Python<'py>, events: &[TraceEvent]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for event in events {
        out.append(event_to_py(py, event)?)?;
    }
    Ok(out)
}

fn event_to_py<'py>(py: Python<'py>, event: &TraceEvent) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("stmt_id", event.stmt_id)?;
    out.set_item("stmt_kind", &event.stmt_kind)?;
    match &event.payload {
        TraceEventKind::Read {
            region,
            proxy,
            access_kind,
            scope,
        } => {
            out.set_item("kind", "read")?;
            out.set_item("region", region_to_py(py, region)?)?;
            out.set_item("proxy", proxy.as_str())?;
            out.set_item("access_kind", access_kind.as_str())?;
            out.set_item("access_category", access_kind.category())?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::Write {
            region,
            proxy,
            access_kind,
            scope,
        } => {
            out.set_item("kind", "write")?;
            out.set_item("region", region_to_py(py, region)?)?;
            out.set_item("proxy", proxy.as_str())?;
            out.set_item("access_kind", access_kind.as_str())?;
            out.set_item("access_category", access_kind.category())?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::TmemWait { async_kind, scope } => {
            out.set_item("kind", "tmem_wait")?;
            out.set_item("async_kind", async_kind.as_str())?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::Fence {
            fence_kind,
            fence_scope,
            scope,
        } => {
            out.set_item("kind", "fence")?;
            out.set_item("fence_kind", fence_kind.as_str())?;
            out.set_item("fence_scope", fence_scope_name(*fence_scope))?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::CommitGroup { scope } => {
            out.set_item("kind", "commit_group")?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::WaitGroup { n, scope } => {
            out.set_item("kind", "wait_group")?;
            out.set_item("n", n)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::MbarInit {
            target,
            count,
            scope,
        } => {
            out.set_item("kind", "mbar_init")?;
            out.set_item("target", mbar_target_to_py(py, target)?)?;
            out.set_item("count", count)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::MbarArrive {
            target,
            count,
            scope,
        } => {
            out.set_item("kind", "mbar_arrive")?;
            out.set_item("target", mbar_target_to_py(py, target)?)?;
            out.set_item("count", count)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::MbarExpectTx {
            target,
            bytes,
            scope,
        } => {
            out.set_item("kind", "mbar_expect_tx")?;
            out.set_item("target", mbar_target_to_py(py, target)?)?;
            out.set_item("bytes", bytes)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::MbarCompleteTx {
            target,
            bytes,
            scope,
        } => {
            out.set_item("kind", "mbar_complete_tx")?;
            out.set_item("target", mbar_target_to_py(py, target)?)?;
            out.set_item("bytes", bytes)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::MbarWait {
            target,
            phase,
            scope,
        } => {
            out.set_item("kind", "mbar_wait")?;
            out.set_item("target", mbar_target_to_py(py, target)?)?;
            out.set_item("phase", phase)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::SyncArrive {
            sync_kind,
            thread_count,
            count,
            cycle,
            bar_id,
            scope,
        } => {
            out.set_item("kind", "sync_arrive")?;
            out.set_item("sync_kind", sync_kind)?;
            out.set_item("thread_count", thread_count)?;
            out.set_item("count", count)?;
            out.set_item("cycle", cycle)?;
            out.set_item("bar_id", bar_id)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::Sync {
            sync_kind,
            thread_count,
            cycle,
            bar_id,
            scope,
        } => {
            out.set_item("kind", "sync")?;
            out.set_item("sync_kind", sync_kind)?;
            out.set_item("thread_count", thread_count)?;
            out.set_item("cycle", cycle)?;
            out.set_item("bar_id", bar_id)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::TmemAlloc {
            cta_ids,
            region,
            scope,
        } => {
            out.set_item("kind", "tmem_alloc")?;
            out.set_item("cta_ids", cta_ids)?;
            out.set_item("region", region_to_py(py, region)?)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::TmemDealloc {
            cta_ids,
            region,
            scope,
        } => {
            out.set_item("kind", "tmem_dealloc")?;
            out.set_item("cta_ids", cta_ids)?;
            out.set_item("region", region_to_py(py, region)?)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
        TraceEventKind::SchedulerNext {
            scheduler_id,
            cta_id,
            task_id,
            scope,
        } => {
            out.set_item("kind", "scheduler_next")?;
            out.set_item("scheduler_id", scheduler_id)?;
            out.set_item("cta_id", cta_id)?;
            out.set_item("task_id", task_id)?;
            out.set_item("scope", scope_to_py(py, scope)?)?;
        }
    }
    Ok(out)
}

fn region_to_py<'py>(py: Python<'py>, region: &Region) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("tensor_id", region.tensor_id)?;
    out.set_item("owner", pool_id_to_py(py, &region.owner)?)?;
    let boxes = PyList::empty(py);
    // include_events serialization materializes the strided form so the
    // external JSON shape (one box per run) is unchanged.
    for b in &region.boxes.to_boxes() {
        boxes.append(boxn_to_py(py, b)?)?;
    }
    out.set_item("boxes", boxes)?;
    Ok(out)
}

fn boxn_to_py<'py>(py: Python<'py>, b: &BoxN) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let ranges = PyList::empty(py);
    for &(start, end) in &b.ranges {
        ranges.append(PyTuple::new(py, [start, end])?)?;
    }
    out.set_item("ranges", ranges)?;
    Ok(out)
}

fn mbar_target_to_py<'py>(
    py: Python<'py>,
    target: &MbarTargetEvent,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("mbar_id", target.mbar_id)?;
    out.set_item("cluster_id", target.cluster_id)?;
    out.set_item("ctaid_in_cluster", target.ctaid_in_cluster)?;
    out.set_item("stage", target.stage)?;
    Ok(out)
}

fn scope_to_py<'py>(py: Python<'py>, scope: &AccessScope) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("stream_id", scope.stream_id)?;
    out.set_item("cluster_id", scope.cluster_id)?;
    out.set_item("cta_id", scope.cta_id)?;
    out.set_item("ctaid_in_cluster", scope.ctaid_in_cluster)?;
    out.set_item("cohort_size", scope.cohort_size)?;
    out.set_item("warp_ids", &scope.warp_ids)?;
    Ok(out)
}

fn pool_id_to_py<'py>(py: Python<'py>, owner: &PoolId) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match owner {
        PoolId::Smem { cta_id } => {
            out.set_item("kind", "smem")?;
            out.set_item("cta_id", cta_id)?;
        }
        PoolId::Tmem { cta_id } => {
            out.set_item("kind", "tmem")?;
            out.set_item("cta_id", cta_id)?;
        }
        PoolId::Gmem { tensor_id } => {
            out.set_item("kind", "gmem")?;
            out.set_item("tensor_id", tensor_id)?;
        }
        PoolId::Reg { cta_id, tensor_id } => {
            out.set_item("kind", "reg")?;
            out.set_item("cta_id", cta_id)?;
            out.set_item("tensor_id", tensor_id)?;
        }
    }
    Ok(out)
}

fn fence_scope_name(scope: ir::FenceScope) -> &'static str {
    match scope {
        ir::FenceScope::Cta => "cta",
        ir::FenceScope::Cluster => "cluster",
        ir::FenceScope::Gpu => "gpu",
    }
}

// ===========================================================================
// chunk 1 — enums
// ===========================================================================

/// Generate a Python-visible mirror enum (ALL-CAPS variant names matching the
/// original `ir.py` members) plus both-way `From` conversions to the pure Rust
/// IR enum. PyO3 can't rename variants through a `cfg_attr`, so the mirror lives
/// here where the `python` feature is always on.
macro_rules! py_enum {
    ($pyty:ident = $pyname:literal => $rsty:ident { $($pyvar:ident => $rsvar:ident),* $(,)? }) => {
        // `frozen` + `hash` make the enum usable as a dict key / set member from
        // Python (like the original `str, Enum`), e.g. `_SCALAR_GMEM_DTYPES`.
        #[allow(non_camel_case_types)]
        #[pyclass(eq, eq_int, frozen, hash, name = $pyname)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $pyty { $($pyvar),* }

        impl From<$pyty> for ir::$rsty {
            fn from(p: $pyty) -> ir::$rsty {
                match p { $($pyty::$pyvar => ir::$rsty::$rsvar),* }
            }
        }
        impl From<ir::$rsty> for $pyty {
            fn from(r: ir::$rsty) -> $pyty {
                match r { $(ir::$rsty::$rsvar => $pyty::$pyvar),* }
            }
        }
    };
}

py_enum!(PyMemorySpace = "MemorySpace" => MemorySpace {
    GMEM => Gmem, SMEM => Smem, TMEM => Tmem, REG => Reg,
});
py_enum!(PyDType = "DType" => DType {
    BOOL => Bool, I8 => I8, U8 => U8, I16 => I16, U16 => U16,
    I32 => I32, U32 => U32, I64 => I64, U64 => U64,
    F8E4M3 => F8E4M3, F16 => F16, BF16 => Bf16, F32 => F32,
});
py_enum!(PySwizzle = "Swizzle" => Swizzle {
    NONE => None, B32 => B32, B64 => B64, B128 => B128,
});
py_enum!(PyTmemLayoutKind = "TmemLayoutKind" => TmemLayoutKind {
    LANE_128 => Lane128, LANE_64_UPPER => Lane64Upper, LANE_64_LOWER => Lane64Lower,
    SCALE_VEC_1X => ScaleVec1x, SCALE_VEC_2X => ScaleVec2x, SCALE_VEC_4X => ScaleVec4x,
});
py_enum!(PyMBarKind = "MBarKind" => MBarKind {
    TMA => Tma, TCGEN05 => Tcgen05, THREAD => Thread,
});
py_enum!(PyFenceKind = "FenceKind" => FenceKind {
    MEMORY => Memory, ASYNC_PROXY => AsyncProxy, VIEW => View,
});
py_enum!(PyFenceScope = "FenceScope" => FenceScope {
    CTA => Cta, CLUSTER => Cluster, GPU => Gpu,
});
py_enum!(PyVarBinding = "VarBinding" => VarBinding {
    LOOP => Loop, SCALAR => Scalar, TASK => Task,
});
py_enum!(PyScalarDType = "ScalarDType" => ScalarDType {
    BOOL => Bool, I32 => I32, U32 => U32, I64 => I64, U64 => U64,
});
py_enum!(PyScalarOp = "ScalarOp" => ScalarOp {
    ADD => Add, SUB => Sub, MUL => Mul, FLOORDIV => FloorDiv, MOD => Mod,
    XOR => Xor, AND => And, OR => Or, NEG => Neg, NOT => Not, SELECT => Select,
    MIN => Min, MAX => Max, EQ => Eq, NE => Ne, LT => Lt, LE => Le, GT => Gt, GE => Ge,
});

// ===========================================================================
// chunk 2 — scalars (Var / ScopeValue / ScalarExpr) with operator overloading
// ===========================================================================

// Fresh identity for each `Var()` — mirrors Python's per-object identity.
static NEXT_VAR_ID: AtomicU32 = AtomicU32::new(0);
fn fresh_var_id() -> u32 {
    NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed)
}
static NEXT_TASK_SPACE_ID: AtomicU32 = AtomicU32::new(0);
fn fresh_task_space_id() -> u32 {
    NEXT_TASK_SPACE_ID.fetch_add(1, Ordering::Relaxed)
}
static NEXT_SCHEDULER_ID: AtomicU32 = AtomicU32::new(0);
fn fresh_scheduler_id() -> u32 {
    NEXT_SCHEDULER_ID.fetch_add(1, Ordering::Relaxed)
}

/// `Var` — a scalar variable. Identity is its auto-assigned id (eq/hash by id).
#[pyclass(name = "Var")]
#[derive(Clone)]
pub struct PyVar(pub ir::Var);

impl PyVar {
    fn as_scalar(&self) -> ir::ScalarValue {
        ir::ScalarValue::Var(self.0)
    }
}

/// `ScopeValue` — a per-thread/per-CTA hardware scope value.
#[pyclass(name = "ScopeValue")]
#[derive(Clone)]
pub struct PyScopeValue(pub ir::ScopeValueKind);

impl PyScopeValue {
    fn as_scalar(&self) -> ir::ScalarValue {
        ir::ScalarValue::Scope(self.0)
    }
}

/// `ScalarExpr` — an operation over scalar values.
#[pyclass(name = "ScalarExpr")]
#[derive(Clone)]
pub struct PyScalarExpr(pub ir::ScalarExpr);

impl PyScalarExpr {
    fn as_scalar(&self) -> ir::ScalarValue {
        ir::ScalarValue::Expr(Box::new(self.0.clone()))
    }
}

/// Coerce a Python object (int | Var | ScopeValue | ScalarExpr) into a Rust
/// `ScalarValue` — mirrors `_coerce_scalar_value` (and its bool rejection).
fn coerce_scalar(obj: &Bound<'_, PyAny>) -> PyResult<ir::ScalarValue> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("scalar value cannot be bool"));
    }
    if let Ok(v) = obj.extract::<PyVar>() {
        return Ok(v.as_scalar());
    }
    if let Ok(v) = obj.extract::<PyScopeValue>() {
        return Ok(v.as_scalar());
    }
    if let Ok(v) = obj.extract::<PyScalarExpr>() {
        return Ok(v.as_scalar());
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ir::ScalarValue::Int(i));
    }
    Err(PyTypeError::new_err("invalid scalar value"))
}

fn binop(lhs: ir::ScalarValue, op: ir::ScalarOp, rhs: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> {
    let rhs = coerce_scalar(rhs)?;
    Ok(PyScalarExpr(ir::ScalarExpr {
        op,
        args: vec![lhs, rhs],
    }))
}
fn rbinop(
    lhs: &Bound<'_, PyAny>,
    op: ir::ScalarOp,
    rhs: ir::ScalarValue,
) -> PyResult<PyScalarExpr> {
    let lhs = coerce_scalar(lhs)?;
    Ok(PyScalarExpr(ir::ScalarExpr {
        op,
        args: vec![lhs, rhs],
    }))
}
fn unop(op: ir::ScalarOp, v: ir::ScalarValue) -> PyScalarExpr {
    PyScalarExpr(ir::ScalarExpr { op, args: vec![v] })
}

/// Generate the operator-overloading `#[pymethods]` block shared by every scalar
/// type (mirrors `ScalarExprMixin`). The type-specific `#[new]`/getters are passed
/// in `$specific` so it's all one `#[pymethods]` block (no `multiple-pymethods`).
macro_rules! scalar_class {
    ($ty:ident, { $($specific:tt)* }) => {
        #[pymethods]
        impl $ty {
            $($specific)*

            fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Add, o) }
            fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::Add, self.as_scalar()) }
            fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Sub, o) }
            fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::Sub, self.as_scalar()) }
            fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Mul, o) }
            fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::Mul, self.as_scalar()) }
            fn __floordiv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::FloorDiv, o) }
            fn __rfloordiv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::FloorDiv, self.as_scalar()) }
            fn __mod__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Mod, o) }
            fn __rmod__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::Mod, self.as_scalar()) }
            fn __xor__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Xor, o) }
            fn __rxor__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::Xor, self.as_scalar()) }
            fn __and__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::And, o) }
            fn __rand__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::And, self.as_scalar()) }
            fn __or__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Or, o) }
            fn __ror__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { rbinop(o, ir::ScalarOp::Or, self.as_scalar()) }
            fn __neg__(&self) -> PyScalarExpr { unop(ir::ScalarOp::Neg, self.as_scalar()) }
            fn __invert__(&self) -> PyScalarExpr { unop(ir::ScalarOp::Not, self.as_scalar()) }

            // Comparisons build LT/LE/GT/GE exprs; == / != fall back to identity
            // (Python uses the named `.eq()`/`.ne()` methods for EQ/NE exprs).
            fn __richcmp__(&self, o: &Bound<'_, PyAny>, op: CompareOp, py: Python<'_>) -> PyResult<PyObject> {
                let sop = match op {
                    CompareOp::Lt => ir::ScalarOp::Lt,
                    CompareOp::Le => ir::ScalarOp::Le,
                    CompareOp::Gt => ir::ScalarOp::Gt,
                    CompareOp::Ge => ir::ScalarOp::Ge,
                    CompareOp::Eq | CompareOp::Ne => return Ok(py.NotImplemented()),
                };
                Ok(Py::new(py, binop(self.as_scalar(), sop, o)?)?.into_any())
            }
            fn eq(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Eq, o) }
            fn ne(&self, o: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> { binop(self.as_scalar(), ir::ScalarOp::Ne, o) }
        }
    };
}

fn scope_kind_from_str(kind: &str) -> PyResult<ir::ScopeValueKind> {
    use ir::ScopeValueKind::*;
    Ok(match kind {
        "tid_in_wg" => TidInWg,
        "lane_id" => LaneId,
        "warp_id" => WarpId,
        "warpgroup_id" => WarpgroupId,
        "ctaid_in_cluster" => CtaidInCluster,
        "cta_id" => CtaId,
        "nvshmem_my_pe" => NvshmemMyPe,
        other => {
            return Err(PyTypeError::new_err(format!(
                "unknown scope value kind: {other}"
            )))
        }
    })
}
fn scope_kind_str(kind: ir::ScopeValueKind) -> &'static str {
    use ir::ScopeValueKind::*;
    match kind {
        TidInWg => "tid_in_wg",
        LaneId => "lane_id",
        WarpId => "warp_id",
        WarpgroupId => "warpgroup_id",
        CtaidInCluster => "ctaid_in_cluster",
        CtaId => "cta_id",
        NvshmemMyPe => "nvshmem_my_pe",
    }
}

scalar_class!(PyVar, {
    #[new]
    #[pyo3(signature = (binding = PyVarBinding::LOOP, dtype = PyScalarDType::I32))]
    fn new(binding: PyVarBinding, dtype: PyScalarDType) -> Self {
        PyVar(ir::Var {
            id: ir::VarId(fresh_var_id()),
            binding: binding.into(),
            dtype: dtype.into(),
        })
    }
    #[getter]
    fn binding(&self) -> PyVarBinding {
        self.0.binding.into()
    }
    #[getter]
    fn dtype(&self) -> PyScalarDType {
        self.0.dtype.into()
    }
    #[getter]
    fn id(&self) -> u32 {
        self.0.id.0
    }
});

scalar_class!(PyScopeValue, {
    #[new]
    fn new(kind: &str) -> PyResult<Self> {
        Ok(PyScopeValue(scope_kind_from_str(kind)?))
    }
    #[getter]
    fn kind(&self) -> &'static str {
        scope_kind_str(self.0)
    }
});

scalar_class!(PyScalarExpr, {
    #[new]
    fn new(op: PyScalarOp, args: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut v = Vec::new();
        for item in args.try_iter()? {
            v.push(coerce_scalar(&item?)?);
        }
        Ok(PyScalarExpr(ir::ScalarExpr {
            op: op.into(),
            args: v,
        }))
    }
    #[getter]
    fn op(&self) -> PyScalarOp {
        self.0.op.into()
    }
    #[getter]
    fn args(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.0.args.iter().map(|a| scalar_to_py(py, a)).collect()
    }
});

// Module-level scalar helpers from ir.py (select / min / max / and / or / not).
#[pyfunction]
fn select(
    cond: &Bound<'_, PyAny>,
    t: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
) -> PyResult<PyScalarExpr> {
    Ok(PyScalarExpr(ir::ScalarExpr {
        op: ir::ScalarOp::Select,
        args: vec![coerce_scalar(cond)?, coerce_scalar(t)?, coerce_scalar(f)?],
    }))
}
#[pyfunction]
#[pyo3(name = "min")]
fn scalar_min(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> {
    Ok(PyScalarExpr(ir::ScalarExpr {
        op: ir::ScalarOp::Min,
        args: vec![coerce_scalar(a)?, coerce_scalar(b)?],
    }))
}
#[pyfunction]
#[pyo3(name = "max")]
fn scalar_max(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> {
    Ok(PyScalarExpr(ir::ScalarExpr {
        op: ir::ScalarOp::Max,
        args: vec![coerce_scalar(a)?, coerce_scalar(b)?],
    }))
}
#[pyfunction]
fn scalar_and(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> {
    Ok(PyScalarExpr(ir::ScalarExpr {
        op: ir::ScalarOp::And,
        args: vec![coerce_scalar(a)?, coerce_scalar(b)?],
    }))
}
#[pyfunction]
fn scalar_or(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> {
    Ok(PyScalarExpr(ir::ScalarExpr {
        op: ir::ScalarOp::Or,
        args: vec![coerce_scalar(a)?, coerce_scalar(b)?],
    }))
}
#[pyfunction]
fn scalar_not(a: &Bound<'_, PyAny>) -> PyResult<PyScalarExpr> {
    Ok(PyScalarExpr(ir::ScalarExpr {
        op: ir::ScalarOp::Not,
        args: vec![coerce_scalar(a)?],
    }))
}

/// Convert a Rust `ScalarValue` back into a Python object (int | Var | ScopeValue
/// | ScalarExpr) — used by slice getters.
fn scalar_to_py(py: Python<'_>, sv: &ir::ScalarValue) -> PyResult<PyObject> {
    Ok(match sv {
        ir::ScalarValue::Int(i) => (*i).into_pyobject(py).unwrap().into_any().unbind(),
        ir::ScalarValue::Var(v) => Py::new(py, PyVar(*v))?.into_any(),
        ir::ScalarValue::Scope(k) => Py::new(py, PyScopeValue(*k))?.into_any(),
        ir::ScalarValue::Expr(e) => Py::new(py, PyScalarExpr((**e).clone()))?.into_any(),
    })
}

/// `stop - start`, folding two literals (mirrors `_scalar_sub`).
fn scalar_sub(stop: ir::ScalarValue, start: ir::ScalarValue) -> ir::ScalarValue {
    match (&stop, &start) {
        (ir::ScalarValue::Int(a), ir::ScalarValue::Int(b)) => ir::ScalarValue::Int(a - b),
        _ => ir::ScalarValue::Expr(Box::new(ir::ScalarExpr {
            op: ir::ScalarOp::Sub,
            args: vec![stop, start],
        })),
    }
}

fn coerce_opt_scalar(obj: Option<Bound<'_, PyAny>>) -> PyResult<Option<ir::ScalarValue>> {
    match obj {
        Some(o) if !o.is_none() => Ok(Some(coerce_scalar(&o)?)),
        _ => Ok(None),
    }
}
fn opt_scalar_or(obj: Option<Bound<'_, PyAny>>, default: i64) -> PyResult<ir::ScalarValue> {
    match obj {
        Some(o) => coerce_scalar(&o),
        None => Ok(ir::ScalarValue::Int(default)),
    }
}
fn coerce_scalar_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ir::ScalarValue>> {
    let mut v = Vec::new();
    for item in obj.try_iter()? {
        v.push(coerce_scalar(&item?)?);
    }
    Ok(v)
}

// ===========================================================================
// chunk 3 — tensors, slices, layouts
// ===========================================================================

static NEXT_TENSOR_ID: AtomicU32 = AtomicU32::new(0);
fn fresh_tensor_id() -> u32 {
    NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

/// `SmemSwizzleLayout`.
#[pyclass(name = "SmemSwizzleLayout")]
#[derive(Clone)]
pub struct PySmemSwizzleLayout(pub ir::SmemSwizzleLayout);
#[pymethods]
impl PySmemSwizzleLayout {
    #[new]
    #[pyo3(signature = (swizzle = PySwizzle::NONE))]
    fn new(swizzle: PySwizzle) -> Self {
        PySmemSwizzleLayout(ir::SmemSwizzleLayout {
            swizzle: swizzle.into(),
        })
    }
    #[getter]
    fn swizzle(&self) -> PySwizzle {
        self.0.swizzle.into()
    }
}

/// `TmemLayout`.
#[pyclass(name = "TmemLayout")]
#[derive(Clone)]
pub struct PyTmemLayout(pub ir::TmemLayout);
#[pymethods]
impl PyTmemLayout {
    #[new]
    #[pyo3(signature = (kind = PyTmemLayoutKind::LANE_128, col_start = 0, lane_align = 0))]
    fn new(kind: PyTmemLayoutKind, col_start: usize, lane_align: u8) -> Self {
        PyTmemLayout(ir::TmemLayout {
            kind: kind.into(),
            col_start,
            lane_align,
        })
    }
    #[getter]
    fn kind(&self) -> PyTmemLayoutKind {
        self.0.kind.into()
    }
    #[getter]
    fn col_start(&self) -> usize {
        self.0.col_start
    }
    #[getter]
    fn lane_align(&self) -> u8 {
        self.0.lane_align
    }
}

fn coerce_layout(obj: &Bound<'_, PyAny>) -> PyResult<ir::Layout> {
    if let Ok(l) = obj.extract::<PyTmemLayout>() {
        return Ok(ir::Layout::Tmem(l.0));
    }
    if let Ok(l) = obj.extract::<PySmemSwizzleLayout>() {
        return Ok(ir::Layout::Swizzle(l.0));
    }
    Err(PyTypeError::new_err(
        "expected a Layout (TmemLayout or SmemSwizzleLayout)",
    ))
}

/// `Tensor` — shared via Arc; identity is its auto-assigned id.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor(pub Arc<ir::Tensor>);
#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (space, dtype, shape, layout = None, byte_offset = None))]
    fn new(
        space: PyMemorySpace,
        dtype: PyDType,
        shape: Vec<usize>,
        layout: Option<Bound<'_, PyAny>>,
        byte_offset: Option<usize>,
    ) -> PyResult<Self> {
        let rust_space: ir::MemorySpace = space.into();
        if rust_space == ir::MemorySpace::Smem && byte_offset.is_none() {
            return Err(PyValueError::new_err("smem tensor byte_offset is required"));
        }
        if rust_space != ir::MemorySpace::Smem && byte_offset.is_some() {
            return Err(PyValueError::new_err(
                "byte_offset is only valid for SMEM tensors",
            ));
        }
        let layout = match layout {
            Some(l) => Some(coerce_layout(&l)?),
            None => None,
        };
        Ok(PyTensor(Arc::new(ir::Tensor {
            id: fresh_tensor_id(),
            space: rust_space,
            dtype: dtype.into(),
            shape,
            layout,
            byte_offset,
        })))
    }
    #[getter]
    fn space(&self) -> PyMemorySpace {
        self.0.space.into()
    }
    #[getter]
    fn dtype(&self) -> PyDType {
        self.0.dtype.into()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape.clone()
    }
    #[getter]
    fn id(&self) -> u32 {
        self.0.id
    }
    #[getter]
    fn byte_offset(&self) -> Option<usize> {
        self.0.byte_offset
    }
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyTensorSlice> {
        Ok(PyTensorSlice(tensor_slice_from_key(&self.0, key)?))
    }
}

/// `TensorSlice`.
#[pyclass(name = "TensorSlice")]
#[derive(Clone)]
pub struct PyTensorSlice(pub ir::TensorSlice);
#[pymethods]
impl PyTensorSlice {
    #[new]
    fn new(tensor: PyTensor, offsets: Bound<'_, PyAny>, shape: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTensorSlice(ir::TensorSlice {
            tensor: tensor.0,
            offsets: coerce_scalar_seq(&offsets)?,
            shape: coerce_scalar_seq(&shape)?,
        }))
    }
    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.tensor.clone())
    }
    #[getter]
    fn offsets(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.0.offsets.iter().map(|s| scalar_to_py(py, s)).collect()
    }
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.0.shape.iter().map(|s| scalar_to_py(py, s)).collect()
    }
}

fn is_ellipsis(item: &Bound<'_, PyAny>) -> bool {
    item.get_type()
        .name()
        .ok()
        .and_then(|n| n.extract::<String>().ok())
        .map(|n| n == "ellipsis")
        .unwrap_or(false)
}

fn slice_dim(tdim: usize, item: &Bound<'_, PyAny>) -> PyResult<(ir::ScalarValue, ir::ScalarValue)> {
    if let Ok(sl) = item.downcast::<PySlice>() {
        let step = sl.getattr("step")?;
        if !step.is_none() {
            let s: i64 = step.extract()?;
            if s != 1 {
                return Err(PyValueError::new_err("tensor slice step is not supported"));
            }
        }
        let start = sl.getattr("start")?;
        let start_sv = if start.is_none() {
            ir::ScalarValue::Int(0)
        } else {
            coerce_scalar(&start)?
        };
        let stop = sl.getattr("stop")?;
        let stop_sv = if stop.is_none() {
            ir::ScalarValue::Int(tdim as i64)
        } else {
            coerce_scalar(&stop)?
        };
        let dim = scalar_sub(stop_sv, start_sv.clone());
        Ok((start_sv, dim))
    } else {
        Ok((coerce_scalar(item)?, ir::ScalarValue::Int(1)))
    }
}

fn tensor_slice_from_key(
    tensor: &Arc<ir::Tensor>,
    key: &Bound<'_, PyAny>,
) -> PyResult<ir::TensorSlice> {
    let rank = tensor.shape.len();
    let mut items: Vec<Bound<'_, PyAny>> = if let Ok(tup) = key.downcast::<PyTuple>() {
        tup.iter().collect()
    } else {
        vec![key.clone()]
    };
    // expand a single ellipsis to full slices over the remaining dims
    let ell = items.iter().filter(|it| is_ellipsis(it)).count();
    if ell > 1 {
        return Err(PyValueError::new_err(
            "tensor slice can contain at most one ellipsis",
        ));
    }
    if ell == 1 {
        let fixed = items.len() - 1;
        if fixed > rank {
            return Err(PyValueError::new_err(
                "tensor slice rank must match tensor rank",
            ));
        }
        let fill = rank - fixed;
        let py = key.py();
        let mut expanded = Vec::with_capacity(rank);
        for it in items {
            if is_ellipsis(&it) {
                for _ in 0..fill {
                    expanded.push(PySlice::full(py).into_any());
                }
            } else {
                expanded.push(it);
            }
        }
        items = expanded;
    }
    if items.len() != rank {
        return Err(PyValueError::new_err(
            "tensor slice rank must match tensor rank",
        ));
    }
    let mut offsets = Vec::with_capacity(rank);
    let mut shape = Vec::with_capacity(rank);
    for (i, item) in items.iter().enumerate() {
        let (off, len) = slice_dim(tensor.shape[i], item)?;
        offsets.push(off);
        shape.push(len);
    }
    Ok(ir::TensorSlice {
        tensor: tensor.clone(),
        offsets,
        shape,
    })
}

/// Accept a `Tensor` (→ full slice) or a `TensorSlice` (as-is) — mirrors the
/// builder's `isinstance(x, Tensor)` normalization.
fn coerce_slice(obj: &Bound<'_, PyAny>) -> PyResult<ir::TensorSlice> {
    if let Ok(s) = obj.extract::<PyTensorSlice>() {
        return Ok(s.0);
    }
    if let Ok(t) = obj.extract::<PyTensor>() {
        let offsets = t.0.shape.iter().map(|_| ir::ScalarValue::Int(0)).collect();
        let shape =
            t.0.shape
                .iter()
                .map(|&d| ir::ScalarValue::Int(d as i64))
                .collect();
        return Ok(ir::TensorSlice {
            tensor: t.0.clone(),
            offsets,
            shape,
        });
    }
    Err(PyTypeError::new_err("expected Tensor or TensorSlice"))
}

fn coerce_reg_operand(obj: &Bound<'_, PyAny>) -> PyResult<ir::RegOperand> {
    if let Ok(slice) = coerce_slice(obj) {
        return Ok(ir::RegOperand::Slice(slice));
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("REG operand literal cannot be bool"));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ir::RegOperand::Literal(ir::RegLiteral::Int(i)));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(ir::RegOperand::Literal(ir::RegLiteral::f32(f as f32)));
    }
    Err(PyTypeError::new_err(
        "expected Tensor, TensorSlice, int, or float REG operand",
    ))
}

// ===========================================================================
// chunk 4 — mbarriers
// ===========================================================================

static NEXT_MBAR_ID: AtomicU32 = AtomicU32::new(0);
fn fresh_mbar_id() -> u32 {
    NEXT_MBAR_ID.fetch_add(1, Ordering::Relaxed)
}

/// `MBar` — shared via Arc; identity is its auto-assigned id.
#[pyclass(name = "MBar")]
#[derive(Clone)]
pub struct PyMBar(pub Arc<ir::MBar>);
#[pymethods]
impl PyMBar {
    #[new]
    #[pyo3(signature = (kind, stages = 1, arrive_count = None))]
    fn new(kind: PyMBarKind, stages: u32, arrive_count: Option<u32>) -> PyResult<Self> {
        // Validate eagerly, like Python's MBar.__post_init__.
        if stages < 1 {
            return Err(PyValueError::new_err(
                "mbar stages must be a positive integer",
            ));
        }
        if let Some(c) = arrive_count {
            if c < 1 {
                return Err(PyValueError::new_err(
                    "mbar arrive_count must be a positive integer or None",
                ));
            }
        }
        Ok(PyMBar(Arc::new(ir::MBar {
            id: fresh_mbar_id(),
            kind: kind.into(),
            stages,
            arrive_count,
        })))
    }
    #[getter]
    fn kind(&self) -> PyMBarKind {
        self.0.kind.into()
    }
    #[getter]
    fn stages(&self) -> u32 {
        self.0.stages
    }
    #[getter]
    fn arrive_count(&self) -> Option<u32> {
        self.0.arrive_count
    }
    #[getter]
    fn id(&self) -> u32 {
        self.0.id
    }
}

/// `MBarRef`.
#[pyclass(name = "MBarRef")]
#[derive(Clone)]
pub struct PyMBarRef(pub ir::MBarRef);
#[pymethods]
impl PyMBarRef {
    #[new]
    #[pyo3(signature = (mbar, remote_coord = None))]
    fn new(mbar: PyMBar, remote_coord: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyMBarRef(ir::MBarRef {
            mbar: mbar.0,
            remote_coord: coerce_opt_scalar(remote_coord)?,
        }))
    }
    #[getter]
    fn mbar(&self) -> PyMBar {
        PyMBar(self.0.mbar.clone())
    }
}

/// Accept `MBar` (→ MBarRef) or `MBarRef` — mirrors `_coerce_mbar_ref`.
fn coerce_mbar_ref(obj: &Bound<'_, PyAny>) -> PyResult<ir::MBarRef> {
    if let Ok(r) = obj.extract::<PyMBarRef>() {
        return Ok(r.0);
    }
    if let Ok(b) = obj.extract::<PyMBar>() {
        return Ok(ir::MBarRef {
            mbar: b.0,
            remote_coord: None,
        });
    }
    Err(PyTypeError::new_err("expected MBar or MBarRef"))
}

// ===========================================================================
// chunk 5 — schedulers
// ===========================================================================

#[pyclass(name = "TaskSpace")]
#[derive(Clone)]
pub struct PyTaskSpace(pub Arc<ir::TaskSpace>);

#[pymethods]
impl PyTaskSpace {
    #[new]
    #[pyo3(signature = (grid, fields))]
    fn new(grid: Vec<usize>, fields: Vec<String>) -> Self {
        PyTaskSpace(Arc::new(ir::TaskSpace {
            id: fresh_task_space_id(),
            grid,
            fields,
        }))
    }
    #[getter]
    fn id(&self) -> u32 {
        self.0.id
    }
    #[getter]
    fn grid(&self) -> Vec<usize> {
        self.0.grid.clone()
    }
    #[getter]
    fn fields(&self) -> Vec<String> {
        self.0.fields.clone()
    }
}

#[pyclass(name = "Scheduler")]
#[derive(Clone)]
pub struct PyScheduler(pub Arc<ir::Scheduler>);

#[pymethods]
impl PyScheduler {
    #[new]
    #[pyo3(signature = (space, policy = "grid_stride", scope = "cluster"))]
    fn new(space: PyTaskSpace, policy: &str, scope: &str) -> PyResult<Self> {
        let policy = ir::SchedulerPolicy::parse(policy)
            .ok_or_else(|| PyValueError::new_err(format!("unknown scheduler policy: {policy}")))?;
        let scope = ir::SchedulerScope::parse(scope)
            .ok_or_else(|| PyValueError::new_err(format!("unknown scheduler scope: {scope}")))?;
        Ok(PyScheduler(Arc::new(ir::Scheduler {
            id: fresh_scheduler_id(),
            space: space.0,
            policy,
            scope,
        })))
    }
    #[getter]
    fn id(&self) -> u32 {
        self.0.id
    }
    #[getter]
    fn space(&self) -> PyTaskSpace {
        PyTaskSpace(self.0.space.clone())
    }
    #[getter]
    fn policy(&self) -> &'static str {
        self.0.policy.as_str()
    }
    #[getter]
    fn scope(&self) -> &'static str {
        self.0.scope.as_str()
    }
}

// ===========================================================================
// chunk 6 — statements (one constructor per Stmt variant)
// ===========================================================================

/// `Stmt` — a built statement (wraps the Rust enum). Constructed via the
/// per-type functions below, collected into bodies, and passed to `Kernel`.
#[pyclass(name = "Stmt")]
#[derive(Clone)]
pub struct PyStmt(pub ir::Stmt);

// A handful of field getters — enough for the targeted "constructor round-trips
// its fields" structure tests, not the full dataclass introspection surface.
#[pymethods]
impl PyStmt {
    #[getter]
    fn m(&self) -> PyResult<u32> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { m, .. } => Ok(*m),
            _ => Err(PyAttributeError::new_err("m")),
        }
    }
    #[getter]
    fn n(&self) -> PyResult<u32> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { n, .. } => Ok(*n),
            _ => Err(PyAttributeError::new_err("n")),
        }
    }
    #[getter]
    fn k(&self) -> PyResult<u32> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { k, .. } => Ok(*k),
            _ => Err(PyAttributeError::new_err("k")),
        }
    }
    #[getter]
    fn accum(&self) -> PyResult<bool> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { accum, .. } => Ok(*accum),
            _ => Err(PyAttributeError::new_err("accum")),
        }
    }
    #[getter]
    fn cta_group(&self) -> PyResult<u8> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { cta_group, .. }
            | ir::Stmt::TmemAlloc { cta_group, .. }
            | ir::Stmt::TmemDealloc { cta_group, .. }
            | ir::Stmt::TmaLoad { cta_group, .. }
            | ir::Stmt::Tcgen05Commit { cta_group, .. } => Ok(*cta_group),
            _ => Err(PyAttributeError::new_err("cta_group")),
        }
    }
    #[getter]
    fn dst(&self) -> PyResult<PyTensorSlice> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { dst, .. } | ir::Stmt::StoreScalar { dst, .. } => {
                Ok(PyTensorSlice(dst.clone()))
            }
            _ => Err(PyAttributeError::new_err("dst")),
        }
    }
    #[getter]
    fn a(&self) -> PyResult<PyTensorSlice> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { a, .. } => Ok(PyTensorSlice(a.clone())),
            _ => Err(PyAttributeError::new_err("a")),
        }
    }
    #[getter]
    fn b(&self) -> PyResult<PyTensorSlice> {
        match &self.0 {
            ir::Stmt::Tcgen05Mma { b, .. } => Ok(PyTensorSlice(b.clone())),
            _ => Err(PyAttributeError::new_err("b")),
        }
    }
    #[getter]
    fn var(&self) -> PyResult<PyVar> {
        match &self.0 {
            ir::Stmt::ForLoop { var, .. }
            | ir::Stmt::ForEachTask { var, .. }
            | ir::Stmt::SchedNext { var, .. }
            | ir::Stmt::ScalarDef { var, .. }
            | ir::Stmt::ScalarStore { var, .. } => Ok(PyVar(*var)),
            _ => Err(PyAttributeError::new_err("var")),
        }
    }
    #[getter]
    fn scheduler(&self) -> PyResult<PyScheduler> {
        match &self.0 {
            ir::Stmt::ForEachTask { scheduler, .. }
            | ir::Stmt::SchedulerImpl { scheduler, .. }
            | ir::Stmt::SchedNext { scheduler, .. } => Ok(PyScheduler(scheduler.clone())),
            _ => Err(PyAttributeError::new_err("scheduler")),
        }
    }
    #[getter]
    fn cond(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.0 {
            ir::Stmt::If { cond, .. } | ir::Stmt::BreakIf { cond } => scalar_to_py(py, cond),
            _ => Err(PyAttributeError::new_err("cond")),
        }
    }
    #[getter]
    fn start(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.0 {
            ir::Stmt::ForLoop { start, .. } => scalar_to_py(py, start),
            _ => Err(PyAttributeError::new_err("start")),
        }
    }
    #[getter]
    fn stop(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.0 {
            ir::Stmt::ForLoop { stop, .. } => scalar_to_py(py, stop),
            _ => Err(PyAttributeError::new_err("stop")),
        }
    }
    #[getter]
    fn step(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.0 {
            ir::Stmt::ForLoop { step, .. } => scalar_to_py(py, step),
            _ => Err(PyAttributeError::new_err("step")),
        }
    }
    #[getter]
    fn body(&self) -> PyResult<Vec<PyStmt>> {
        match &self.0 {
            ir::Stmt::ForLoop { body, .. }
            | ir::Stmt::ForEachTask { body, .. }
            | ir::Stmt::SchedulerImpl { body, .. }
            | ir::Stmt::Loop { body }
            | ir::Stmt::Role { body, .. }
            | ir::Stmt::KernelInit { body, .. }
            | ir::Stmt::KernelFinalize { body, .. } => {
                Ok(body.iter().map(|s| PyStmt(s.clone())).collect())
            }
            _ => Err(PyAttributeError::new_err("body")),
        }
    }
}

fn body_to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ir::Stmt>> {
    let mut v = Vec::new();
    for item in obj.try_iter()? {
        v.push(item?.extract::<PyStmt>()?.0);
    }
    Ok(v)
}
fn opt_body(obj: Option<Bound<'_, PyAny>>) -> PyResult<Vec<ir::Stmt>> {
    match obj {
        Some(b) => body_to_vec(&b),
        None => Ok(vec![]),
    }
}

#[pyfunction]
#[pyo3(name = "TensorDef")]
fn tensor_def(tensor: PyTensor) -> PyStmt {
    PyStmt(ir::Stmt::TensorDef { tensor: tensor.0 })
}
#[pyfunction]
#[pyo3(name = "TmemAlloc", signature = (tensor, n_cols, cta_group = 1))]
fn tmem_alloc(tensor: PyTensor, n_cols: u32, cta_group: u8) -> PyStmt {
    PyStmt(ir::Stmt::TmemAlloc {
        tensor: tensor.0,
        n_cols,
        cta_group,
    })
}
#[pyfunction]
#[pyo3(name = "TmemDealloc", signature = (tensor, n_cols, cta_group = 1))]
fn tmem_dealloc(tensor: PyTensor, n_cols: u32, cta_group: u8) -> PyStmt {
    PyStmt(ir::Stmt::TmemDealloc {
        tensor: tensor.0,
        n_cols,
        cta_group,
    })
}
#[pyfunction]
#[pyo3(name = "ScalarDef", signature = (var, initial = None))]
fn scalar_def(var: PyVar, initial: Option<Bound<'_, PyAny>>) -> PyResult<PyStmt> {
    let initial = match initial {
        Some(o) if o.extract::<PyTensorSlice>().is_ok() => {
            ir::ScalarInitial::Tensor(o.extract::<PyTensorSlice>()?.0)
        }
        Some(o) => ir::ScalarInitial::Value(coerce_scalar(&o)?),
        None => ir::ScalarInitial::Value(ir::ScalarValue::Int(0)),
    };
    Ok(PyStmt(ir::Stmt::ScalarDef {
        var: var.0,
        initial,
    }))
}
#[pyfunction]
#[pyo3(name = "ScalarStore")]
fn scalar_store(var: PyVar, value: Bound<'_, PyAny>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::ScalarStore {
        var: var.0,
        value: coerce_scalar(&value)?,
    }))
}
#[pyfunction]
#[pyo3(name = "StoreScalar")]
fn store_scalar(dst: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::StoreScalar {
        dst: coerce_slice(&dst)?,
        value: coerce_scalar(&value)?,
    }))
}
#[pyfunction]
#[pyo3(name = "MBarDef")]
fn mbar_def(mbar: PyMBar) -> PyStmt {
    PyStmt(ir::Stmt::MBarDef { mbar: mbar.0 })
}
#[pyfunction]
#[pyo3(name = "KernelInit", signature = (body = None, warp = None, lane = None, elected = false))]
fn kernel_init(
    body: Option<Bound<'_, PyAny>>,
    warp: Option<u32>,
    lane: Option<u32>,
    elected: bool,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::KernelInit {
        body: opt_body(body)?,
        warp,
        lane,
        elected,
    }))
}
#[pyfunction]
#[pyo3(name = "KernelFinalize", signature = (body = None, warp = None, lane = None, elected = false))]
fn kernel_finalize(
    body: Option<Bound<'_, PyAny>>,
    warp: Option<u32>,
    lane: Option<u32>,
    elected: bool,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::KernelFinalize {
        body: opt_body(body)?,
        warp,
        lane,
        elected,
    }))
}
#[pyfunction]
#[pyo3(name = "Role", signature = (body = None, warp = None, warpgroup = None, elected = false, maxnreg = None))]
fn role(
    body: Option<Bound<'_, PyAny>>,
    warp: Option<u32>,
    warpgroup: Option<u32>,
    elected: bool,
    maxnreg: Option<u32>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::Role {
        body: opt_body(body)?,
        warp,
        warpgroup,
        elected,
        maxnreg,
    }))
}
#[pyfunction]
#[pyo3(name = "ForLoop", signature = (var, start = None, stop = None, step = None, body = None))]
fn for_loop(
    var: PyVar,
    start: Option<Bound<'_, PyAny>>,
    stop: Option<Bound<'_, PyAny>>,
    step: Option<Bound<'_, PyAny>>,
    body: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::ForLoop {
        var: var.0,
        start: opt_scalar_or(start, 0)?,
        stop: opt_scalar_or(stop, 0)?,
        step: opt_scalar_or(step, 1)?,
        body: opt_body(body)?,
    }))
}
#[pyfunction]
#[pyo3(name = "ForEachTask", signature = (scheduler, var, body = None))]
fn for_each_task(
    scheduler: PyScheduler,
    var: PyVar,
    body: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::ForEachTask {
        scheduler: scheduler.0,
        var: var.0,
        body: opt_body(body)?,
    }))
}
#[pyfunction]
#[pyo3(name = "SchedulerImpl", signature = (scheduler, body = None))]
fn scheduler_impl(scheduler: PyScheduler, body: Option<Bound<'_, PyAny>>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::SchedulerImpl {
        scheduler: scheduler.0,
        body: opt_body(body)?,
    }))
}
#[pyfunction]
#[pyo3(name = "SchedNext")]
fn sched_next(scheduler: PyScheduler, var: PyVar) -> PyStmt {
    PyStmt(ir::Stmt::SchedNext {
        scheduler: scheduler.0,
        var: var.0,
    })
}
#[pyfunction]
#[pyo3(name = "Loop", signature = (body = None))]
fn loop_stmt(body: Option<Bound<'_, PyAny>>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::Loop {
        body: opt_body(body)?,
    }))
}
#[pyfunction]
#[pyo3(name = "BreakIf")]
fn break_if(cond: Bound<'_, PyAny>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::BreakIf {
        cond: coerce_scalar(&cond)?,
    }))
}
#[pyfunction]
#[pyo3(name = "If", signature = (cond, then_body = None))]
fn if_stmt(cond: Bound<'_, PyAny>, then_body: Option<Bound<'_, PyAny>>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::If {
        cond: coerce_scalar(&cond)?,
        then_body: opt_body(then_body)?,
    }))
}
#[pyfunction]
#[pyo3(name = "MBarrierInit", signature = (mbar, *, count, stage = None))]
fn mbarrier_init(
    mbar: Bound<'_, PyAny>,
    count: u32,
    stage: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::MBarrierInit {
        mbar: coerce_mbar_ref(&mbar)?,
        count,
        stage: coerce_opt_scalar(stage)?,
    }))
}
#[pyfunction]
#[pyo3(name = "MBarrierArrive", signature = (mbar, stage = None, count = None))]
fn mbarrier_arrive(
    mbar: Bound<'_, PyAny>,
    stage: Option<Bound<'_, PyAny>>,
    count: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::MBarrierArrive {
        mbar: coerce_mbar_ref(&mbar)?,
        stage: coerce_opt_scalar(stage)?,
        count: opt_scalar_or(count, 1)?,
    }))
}
#[pyfunction]
#[pyo3(name = "MBarrierWait", signature = (mbar, stage = None, phase = None))]
fn mbarrier_wait(
    mbar: Bound<'_, PyAny>,
    stage: Option<Bound<'_, PyAny>>,
    phase: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::MBarrierWait {
        mbar: coerce_mbar_ref(&mbar)?,
        stage: coerce_opt_scalar(stage)?,
        phase: coerce_opt_scalar(phase)?,
    }))
}
#[pyfunction]
#[pyo3(name = "MBarrierExpectTx", signature = (mbar, bytes, stage = None))]
fn mbarrier_expect_tx(
    mbar: Bound<'_, PyAny>,
    bytes: u32,
    stage: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::MBarrierExpectTx {
        mbar: coerce_mbar_ref(&mbar)?,
        bytes,
        stage: coerce_opt_scalar(stage)?,
    }))
}
#[pyfunction]
#[pyo3(name = "MBarrierArriveExpectTx", signature = (mbar, bytes, stage = None))]
fn mbarrier_arrive_expect_tx(
    mbar: Bound<'_, PyAny>,
    bytes: u32,
    stage: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::MBarrierArriveExpectTx {
        mbar: coerce_mbar_ref(&mbar)?,
        bytes,
        stage: coerce_opt_scalar(stage)?,
    }))
}
#[pyfunction]
#[pyo3(name = "TmaLoad", signature = (dst, src, mbar, bytes, coords, shape, gmem_shape = None, mbar_stage = None, multicast_cta_mask = None, cta_group = 1))]
#[allow(clippy::too_many_arguments)]
fn tma_load(
    dst: Bound<'_, PyAny>,
    src: PyTensor,
    mbar: Bound<'_, PyAny>,
    bytes: Bound<'_, PyAny>,
    coords: Bound<'_, PyAny>,
    shape: Vec<usize>,
    gmem_shape: Option<Vec<usize>>,
    mbar_stage: Option<Bound<'_, PyAny>>,
    multicast_cta_mask: Option<u16>,
    cta_group: u8,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::TmaLoad {
        dst: coerce_slice(&dst)?,
        src: src.0,
        mbar: coerce_mbar_ref(&mbar)?,
        bytes: coerce_scalar(&bytes)?,
        coords: coerce_scalar_seq(&coords)?,
        shape,
        gmem_shape,
        mbar_stage: coerce_opt_scalar(mbar_stage)?,
        multicast_cta_mask,
        cta_group,
    }))
}
#[pyfunction]
#[pyo3(name = "TmaStore", signature = (dst, src, coords, shape, gmem_shape = None))]
fn tma_store(
    dst: PyTensor,
    src: Bound<'_, PyAny>,
    coords: Bound<'_, PyAny>,
    shape: Vec<usize>,
    gmem_shape: Option<Vec<usize>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::TmaStore {
        dst: dst.0,
        src: coerce_slice(&src)?,
        coords: coerce_scalar_seq(&coords)?,
        shape,
        gmem_shape,
    }))
}
#[pyfunction]
#[pyo3(name = "CpAsyncBulkCommitGroup")]
fn cp_async_bulk_commit_group() -> PyStmt {
    PyStmt(ir::Stmt::CpAsyncBulkCommitGroup)
}
#[pyfunction]
#[pyo3(name = "CpAsyncBulkWaitGroupRead", signature = (n = 0))]
fn cp_async_bulk_wait_group_read(n: u8) -> PyStmt {
    PyStmt(ir::Stmt::CpAsyncBulkWaitGroupRead { n })
}
#[pyfunction]
#[pyo3(name = "Tcgen05Mma", signature = (dst, a, b, m, n, k = 16, accum = false, trans_a = false, trans_b = false, cta_group = 1, sfa = None, sfb = None, sf_byte = 0))]
#[allow(clippy::too_many_arguments)]
fn tcgen05_mma(
    dst: Bound<'_, PyAny>,
    a: Bound<'_, PyAny>,
    b: Bound<'_, PyAny>,
    m: u32,
    n: u32,
    k: u32,
    accum: bool,
    trans_a: bool,
    trans_b: bool,
    cta_group: u8,
    sfa: Option<Bound<'_, PyAny>>,
    sfb: Option<Bound<'_, PyAny>>,
    sf_byte: u8,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::Tcgen05Mma {
        dst: coerce_slice(&dst)?,
        a: coerce_slice(&a)?,
        b: coerce_slice(&b)?,
        m,
        n,
        k,
        accum,
        trans_a,
        trans_b,
        cta_group,
        sfa: sfa.map(|v| coerce_slice(&v)).transpose()?,
        sfb: sfb.map(|v| coerce_slice(&v)).transpose()?,
        sf_byte,
    }))
}
#[pyfunction]
#[pyo3(name = "Tcgen05Cp", signature = (dst, src, cta_group = 1))]
fn tcgen05_cp(dst: Bound<'_, PyAny>, src: Bound<'_, PyAny>, cta_group: u8) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::Tcgen05Cp {
        dst: coerce_slice(&dst)?,
        src: coerce_slice(&src)?,
        cta_group,
    }))
}
#[pyfunction]
#[pyo3(name = "Tcgen05Commit", signature = (mbar, stage = None, cta_group = 1, multicast_cta_mask = None))]
fn tcgen05_commit(
    mbar: Bound<'_, PyAny>,
    stage: Option<Bound<'_, PyAny>>,
    cta_group: u8,
    multicast_cta_mask: Option<u16>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::Tcgen05Commit {
        mbar: coerce_mbar_ref(&mbar)?,
        stage: coerce_opt_scalar(stage)?,
        cta_group,
        multicast_cta_mask,
    }))
}
#[pyfunction]
#[pyo3(name = "Tcgen05Ld", signature = (dst, src, shape = "32x32b", num = 1, row = None, col = None))]
fn tcgen05_ld(
    dst: Bound<'_, PyAny>,
    src: PyTensor,
    shape: &str,
    num: u32,
    row: Option<Bound<'_, PyAny>>,
    col: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    let shape = ir::LdStShape::parse(shape)
        .ok_or_else(|| PyValueError::new_err("tcgen05_ld shape is unsupported"))?;
    Ok(PyStmt(ir::Stmt::Tcgen05Ld {
        dst: coerce_slice(&dst)?,
        src: src.0,
        shape,
        num,
        row: opt_scalar_or(row, 0)?,
        col: opt_scalar_or(col, 0)?,
    }))
}
#[pyfunction]
#[pyo3(name = "Tcgen05WaitLd")]
fn tcgen05_wait_ld() -> PyStmt {
    PyStmt(ir::Stmt::Tcgen05WaitLd)
}
#[pyfunction]
#[pyo3(name = "Tcgen05St", signature = (dst, src, shape = "32x32b", num = 1, row = None, col = None))]
fn tcgen05_st(
    dst: PyTensor,
    src: Bound<'_, PyAny>,
    shape: &str,
    num: u32,
    row: Option<Bound<'_, PyAny>>,
    col: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    let shape = ir::LdStShape::parse(shape)
        .ok_or_else(|| PyValueError::new_err("tcgen05_st shape is unsupported"))?;
    Ok(PyStmt(ir::Stmt::Tcgen05St {
        dst: dst.0,
        src: coerce_slice(&src)?,
        shape,
        num,
        row: opt_scalar_or(row, 0)?,
        col: opt_scalar_or(col, 0)?,
    }))
}
#[pyfunction]
#[pyo3(name = "Tcgen05WaitSt")]
fn tcgen05_wait_st() -> PyStmt {
    PyStmt(ir::Stmt::Tcgen05WaitSt)
}

#[pyfunction]
#[pyo3(name = "LdMatrix", signature = (dst, src, shape = "m8n8", num = 1, trans = false, dtype = "b16"))]
fn ldmatrix(
    dst: Bound<'_, PyAny>,
    src: Bound<'_, PyAny>,
    shape: &str,
    num: u32,
    trans: bool,
    dtype: &str,
) -> PyResult<PyStmt> {
    let shape = ir::MatrixShape::parse(shape)
        .ok_or_else(|| PyValueError::new_err("ldmatrix shape is unsupported"))?;
    let dtype = ir::MatrixDType::parse(dtype)
        .ok_or_else(|| PyValueError::new_err("ldmatrix dtype is unsupported"))?;
    Ok(PyStmt(ir::Stmt::LdMatrix {
        dst: coerce_slice(&dst)?,
        src: coerce_slice(&src)?,
        shape,
        num,
        trans,
        dtype,
    }))
}

#[pyfunction]
#[pyo3(name = "StMatrix", signature = (dst, src, shape = "m8n8", num = 1, trans = false, dtype = "b16"))]
fn stmatrix(
    dst: Bound<'_, PyAny>,
    src: Bound<'_, PyAny>,
    shape: &str,
    num: u32,
    trans: bool,
    dtype: &str,
) -> PyResult<PyStmt> {
    let shape = ir::MatrixShape::parse(shape)
        .ok_or_else(|| PyValueError::new_err("stmatrix shape is unsupported"))?;
    let dtype = ir::MatrixDType::parse(dtype)
        .ok_or_else(|| PyValueError::new_err("stmatrix dtype is unsupported"))?;
    Ok(PyStmt(ir::Stmt::StMatrix {
        dst: coerce_slice(&dst)?,
        src: coerce_slice(&src)?,
        shape,
        num,
        trans,
        dtype,
    }))
}

#[pyfunction]
#[pyo3(name = "RegFill")]
fn reg_fill(dst: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegFill {
        dst: coerce_slice(&dst)?,
        value: coerce_reg_operand(&value)?,
    }))
}

#[pyfunction]
#[pyo3(name = "RegUnary")]
fn reg_unary(dst: Bound<'_, PyAny>, src: Bound<'_, PyAny>, op: &str) -> PyResult<PyStmt> {
    let op = ir::RegUnaryOp::parse(op)
        .ok_or_else(|| PyValueError::new_err("reg_unary op is unsupported"))?;
    Ok(PyStmt(ir::Stmt::RegUnary {
        dst: coerce_slice(&dst)?,
        src: coerce_reg_operand(&src)?,
        op,
    }))
}

/// REG ALU constructors share the (dst, lhs, rhs) shape.
macro_rules! reg_binop_ctor {
    ($fn:ident, $name:literal, $variant:ident, $rounding:expr) => {
        #[pyfunction]
        #[pyo3(name = $name, signature = (dst, lhs, rhs, rounding = "rn"))]
        fn $fn(
            dst: Bound<'_, PyAny>,
            lhs: Bound<'_, PyAny>,
            rhs: Bound<'_, PyAny>,
            rounding: &str,
        ) -> PyResult<PyStmt> {
            let rounding = ir::Rounding::parse(rounding)
                .ok_or_else(|| PyValueError::new_err("REG rounding must be rn or rm"))?;
            if !$rounding && rounding != ir::Rounding::Rn {
                return Err(PyValueError::new_err("REG op only supports rn rounding"));
            }
            Ok(PyStmt(ir::Stmt::$variant {
                dst: coerce_slice(&dst)?,
                lhs: coerce_reg_operand(&lhs)?,
                rhs: coerce_reg_operand(&rhs)?,
                rounding,
            }))
        }
    };
    ($fn:ident, $name:literal, $variant:ident) => {
        #[pyfunction]
        #[pyo3(name = $name)]
        fn $fn(
            dst: Bound<'_, PyAny>,
            lhs: Bound<'_, PyAny>,
            rhs: Bound<'_, PyAny>,
        ) -> PyResult<PyStmt> {
            Ok(PyStmt(ir::Stmt::$variant {
                dst: coerce_slice(&dst)?,
                lhs: coerce_reg_operand(&lhs)?,
                rhs: coerce_reg_operand(&rhs)?,
            }))
        }
    };
}
reg_binop_ctor!(reg_add, "RegAdd", RegAdd, true);
reg_binop_ctor!(reg_sub, "RegSub", RegSub, true);
reg_binop_ctor!(reg_mul, "RegMul", RegMul);
reg_binop_ctor!(reg_max, "RegMax", RegMax);
reg_binop_ctor!(reg_min, "RegMin", RegMin);

#[pyfunction]
#[pyo3(name = "RegFma")]
fn reg_fma(
    dst: Bound<'_, PyAny>,
    a: Bound<'_, PyAny>,
    b: Bound<'_, PyAny>,
    c: Bound<'_, PyAny>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegFma {
        dst: coerce_slice(&dst)?,
        a: coerce_reg_operand(&a)?,
        b: coerce_reg_operand(&b)?,
        c: coerce_reg_operand(&c)?,
    }))
}

#[pyfunction]
#[pyo3(name = "RegBitwise")]
fn reg_bitwise(
    dst: Bound<'_, PyAny>,
    lhs: Bound<'_, PyAny>,
    rhs: Bound<'_, PyAny>,
    op: &str,
) -> PyResult<PyStmt> {
    let op = ir::RegBinaryOp::parse(op)
        .ok_or_else(|| PyValueError::new_err("reg_bitwise op is unsupported"))?;
    Ok(PyStmt(ir::Stmt::RegBitwise {
        dst: coerce_slice(&dst)?,
        lhs: coerce_reg_operand(&lhs)?,
        rhs: coerce_reg_operand(&rhs)?,
        op,
    }))
}

#[pyfunction]
#[pyo3(name = "RegReduce")]
fn reg_reduce(dst: Bound<'_, PyAny>, src: Bound<'_, PyAny>, op: &str) -> PyResult<PyStmt> {
    let op = ir::RegReduceOp::parse(op)
        .ok_or_else(|| PyValueError::new_err("reg_reduce op is unsupported"))?;
    Ok(PyStmt(ir::Stmt::RegReduce {
        dst: coerce_slice(&dst)?,
        src: coerce_reg_operand(&src)?,
        op,
    }))
}

#[pyfunction]
#[pyo3(name = "RegCondRescale", signature = (dst, src, scale, threshold = None, scope = "warpgroup"))]
fn reg_cond_rescale(
    dst: Bound<'_, PyAny>,
    src: Bound<'_, PyAny>,
    scale: Bound<'_, PyAny>,
    threshold: Option<Bound<'_, PyAny>>,
    scope: &str,
) -> PyResult<PyStmt> {
    let scope = ir::RegCondScope::parse(scope)
        .ok_or_else(|| PyValueError::new_err("reg_cond_rescale scope is unsupported"))?;
    Ok(PyStmt(ir::Stmt::RegCondRescale {
        dst: coerce_slice(&dst)?,
        src: coerce_reg_operand(&src)?,
        scale: coerce_reg_operand(&scale)?,
        threshold: match threshold {
            Some(t) => coerce_reg_operand(&t)?,
            None => ir::RegOperand::Literal(ir::RegLiteral::f32(1.0)),
        },
        scope,
    }))
}

#[pyfunction]
#[pyo3(name = "RegSoftmaxRescale", signature = (row_max, row_scale, row_max_old, row_max_new, scale_log2, threshold = None))]
fn reg_softmax_rescale(
    row_max: Bound<'_, PyAny>,
    row_scale: Bound<'_, PyAny>,
    row_max_old: Bound<'_, PyAny>,
    row_max_new: Bound<'_, PyAny>,
    scale_log2: Bound<'_, PyAny>,
    threshold: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegSoftmaxRescale {
        row_max: coerce_slice(&row_max)?,
        row_scale: coerce_slice(&row_scale)?,
        row_max_old: coerce_reg_operand(&row_max_old)?,
        row_max_new: coerce_reg_operand(&row_max_new)?,
        scale_log2: coerce_reg_operand(&scale_log2)?,
        threshold: match threshold {
            Some(t) => coerce_reg_operand(&t)?,
            None => ir::RegOperand::Literal(ir::RegLiteral::f32(8.0)),
        },
    }))
}

#[pyfunction]
#[pyo3(name = "RegCausalMask", signature = (dst, src, query_start, key_start, group_size, mask_value = None))]
fn reg_causal_mask(
    dst: Bound<'_, PyAny>,
    src: Bound<'_, PyAny>,
    query_start: Bound<'_, PyAny>,
    key_start: Bound<'_, PyAny>,
    group_size: u32,
    mask_value: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegCausalMask {
        dst: coerce_slice(&dst)?,
        src: coerce_reg_operand(&src)?,
        query_start: coerce_scalar(&query_start)?,
        key_start: coerce_scalar(&key_start)?,
        group_size,
        mask_value: match mask_value {
            Some(v) => coerce_reg_operand(&v)?,
            None => ir::RegOperand::Literal(ir::RegLiteral::f32(f32::NEG_INFINITY)),
        },
    }))
}

#[pyfunction]
#[pyo3(name = "RegCombineIntFracEx2")]
fn reg_combine_int_frac_ex2(
    dst: Bound<'_, PyAny>,
    rounded: Bound<'_, PyAny>,
    frac_ex2: Bound<'_, PyAny>,
) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegCombineIntFracEx2 {
        dst: coerce_slice(&dst)?,
        rounded: coerce_reg_operand(&rounded)?,
        frac_ex2: coerce_reg_operand(&frac_ex2)?,
    }))
}
#[pyfunction]
#[pyo3(name = "RegCvt", signature = (dst, src, rounding = "rn"))]
fn reg_cvt(dst: Bound<'_, PyAny>, src: Bound<'_, PyAny>, rounding: &str) -> PyResult<PyStmt> {
    if rounding != "rn" {
        return Err(PyValueError::new_err("reg_cvt rounding must be rn"));
    }
    Ok(PyStmt(ir::Stmt::RegCvt {
        dst: coerce_slice(&dst)?,
        src: coerce_slice(&src)?,
        rounding: ir::Rounding::Rn,
    }))
}
#[pyfunction]
#[pyo3(name = "RegLoad")]
fn reg_load(dst: Bound<'_, PyAny>, src: Bound<'_, PyAny>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegLoad {
        dst: coerce_slice(&dst)?,
        src: coerce_slice(&src)?,
    }))
}
#[pyfunction]
#[pyo3(name = "RegStore")]
fn reg_store(dst: Bound<'_, PyAny>, src: Bound<'_, PyAny>) -> PyResult<PyStmt> {
    Ok(PyStmt(ir::Stmt::RegStore {
        dst: coerce_slice(&dst)?,
        src: coerce_slice(&src)?,
    }))
}
#[pyfunction]
#[pyo3(name = "Fence", signature = (kind = PyFenceKind::MEMORY, scope = PyFenceScope::CTA))]
fn fence(kind: PyFenceKind, scope: PyFenceScope) -> PyStmt {
    PyStmt(ir::Stmt::Fence {
        kind: kind.into(),
        scope: scope.into(),
    })
}
#[pyfunction]
#[pyo3(name = "CtaSync")]
fn cta_sync() -> PyStmt {
    PyStmt(ir::Stmt::CtaSync)
}
#[pyfunction]
#[pyo3(name = "WgSync")]
fn wg_sync(barrier_id: u32) -> PyStmt {
    PyStmt(ir::Stmt::WgSync { barrier_id })
}
#[pyfunction]
#[pyo3(name = "WarpSync")]
fn warp_sync() -> PyStmt {
    PyStmt(ir::Stmt::WarpSync)
}
#[pyfunction]
#[pyo3(name = "ClusterSync")]
fn cluster_sync() -> PyStmt {
    PyStmt(ir::Stmt::ClusterSync)
}

// ===========================================================================
// chunk 7 — Kernel
// ===========================================================================

fn tensors_to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Arc<ir::Tensor>>> {
    let mut v = Vec::new();
    for item in obj.try_iter()? {
        v.push(item?.extract::<PyTensor>()?.0);
    }
    Ok(v)
}

/// `Kernel` — the built IR. Construction validates (like Python's
/// `Kernel.__post_init__`); `validate()` re-runs it explicitly.
#[pyclass(name = "Kernel")]
#[derive(Clone)]
pub struct PyKernel(pub ir::Kernel);
#[pymethods]
impl PyKernel {
    #[new]
    #[pyo3(signature = (name, args = None, body = None, num_warps = 12, smem_size_bytes = 0, launch_shape = None, cluster_shape = None))]
    fn new(
        name: String,
        args: Option<Bound<'_, PyAny>>,
        body: Option<Bound<'_, PyAny>>,
        num_warps: u32,
        smem_size_bytes: usize,
        launch_shape: Option<Vec<usize>>,
        cluster_shape: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let kernel = ir::Kernel {
            name,
            args: match args {
                Some(a) => tensors_to_vec(&a)?,
                None => vec![],
            },
            body: opt_body(body)?,
            num_warps,
            smem_size_bytes,
            launch_shape: launch_shape.unwrap_or_else(|| vec![1]),
            cluster_shape: cluster_shape.unwrap_or_else(|| vec![1]),
        };
        kernel
            .validate()
            .map_err(|e| PyValueError::new_err(e.message))?;
        Ok(PyKernel(kernel))
    }
    fn validate(&self) -> PyResult<()> {
        self.0
            .validate()
            .map_err(|e| PyValueError::new_err(e.message))
    }
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }
    #[getter]
    fn args(&self) -> Vec<PyTensor> {
        self.0.args.iter().map(|t| PyTensor(t.clone())).collect()
    }
    #[getter]
    fn body(&self) -> Vec<PyStmt> {
        self.0.body.iter().map(|s| PyStmt(s.clone())).collect()
    }
    #[getter]
    fn num_warps(&self) -> u32 {
        self.0.num_warps
    }
    #[getter]
    fn smem_size_bytes(&self) -> usize {
        self.0.smem_size_bytes
    }
    #[getter]
    fn launch_shape(&self) -> Vec<usize> {
        self.0.launch_shape.clone()
    }
    #[getter]
    fn cluster_shape(&self) -> Vec<usize> {
        self.0.cluster_shape.clone()
    }
    fn launch_cta_count(&self) -> usize {
        self.0.launch_cta_count()
    }
}

// ===========================================================================
// module registration
// ===========================================================================

/// Register every Python-visible class/function into the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // chunk 1 — enums
    m.add_class::<PyMemorySpace>()?;
    m.add_class::<PyDType>()?;
    m.add_class::<PySwizzle>()?;
    m.add_class::<PyTmemLayoutKind>()?;
    m.add_class::<PyMBarKind>()?;
    m.add_class::<PyFenceKind>()?;
    m.add_class::<PyFenceScope>()?;
    m.add_class::<PyVarBinding>()?;
    m.add_class::<PyScalarDType>()?;
    m.add_class::<PyScalarOp>()?;
    // chunk 2 — scalars
    m.add_class::<PyVar>()?;
    m.add_class::<PyScopeValue>()?;
    m.add_class::<PyScalarExpr>()?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_min, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_max, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_and, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_or, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_not, m)?)?;
    // chunk 3 — tensors / slices / layouts
    m.add_class::<PySmemSwizzleLayout>()?;
    m.add_class::<PyTmemLayout>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyTensorSlice>()?;
    // chunk 4 — mbarriers
    m.add_class::<PyMBar>()?;
    m.add_class::<PyMBarRef>()?;
    // chunk 5 — schedulers
    m.add_class::<PyTaskSpace>()?;
    m.add_class::<PyScheduler>()?;
    // chunk 6 — statements
    m.add_class::<PyStmt>()?;
    for f in [
        wrap_pyfunction!(tensor_def, m)?,
        wrap_pyfunction!(tmem_alloc, m)?,
        wrap_pyfunction!(tmem_dealloc, m)?,
        wrap_pyfunction!(scalar_def, m)?,
        wrap_pyfunction!(scalar_store, m)?,
        wrap_pyfunction!(store_scalar, m)?,
        wrap_pyfunction!(mbar_def, m)?,
        wrap_pyfunction!(kernel_init, m)?,
        wrap_pyfunction!(kernel_finalize, m)?,
        wrap_pyfunction!(role, m)?,
        wrap_pyfunction!(for_loop, m)?,
        wrap_pyfunction!(for_each_task, m)?,
        wrap_pyfunction!(scheduler_impl, m)?,
        wrap_pyfunction!(sched_next, m)?,
        wrap_pyfunction!(loop_stmt, m)?,
        wrap_pyfunction!(break_if, m)?,
        wrap_pyfunction!(if_stmt, m)?,
        wrap_pyfunction!(mbarrier_init, m)?,
        wrap_pyfunction!(mbarrier_arrive, m)?,
        wrap_pyfunction!(mbarrier_wait, m)?,
        wrap_pyfunction!(mbarrier_expect_tx, m)?,
        wrap_pyfunction!(mbarrier_arrive_expect_tx, m)?,
        wrap_pyfunction!(tma_load, m)?,
        wrap_pyfunction!(tma_store, m)?,
        wrap_pyfunction!(cp_async_bulk_commit_group, m)?,
        wrap_pyfunction!(cp_async_bulk_wait_group_read, m)?,
        wrap_pyfunction!(tcgen05_mma, m)?,
        wrap_pyfunction!(tcgen05_cp, m)?,
        wrap_pyfunction!(tcgen05_commit, m)?,
        wrap_pyfunction!(tcgen05_ld, m)?,
        wrap_pyfunction!(tcgen05_wait_ld, m)?,
        wrap_pyfunction!(tcgen05_st, m)?,
        wrap_pyfunction!(tcgen05_wait_st, m)?,
        wrap_pyfunction!(ldmatrix, m)?,
        wrap_pyfunction!(stmatrix, m)?,
        wrap_pyfunction!(reg_fill, m)?,
        wrap_pyfunction!(reg_unary, m)?,
        wrap_pyfunction!(reg_add, m)?,
        wrap_pyfunction!(reg_sub, m)?,
        wrap_pyfunction!(reg_mul, m)?,
        wrap_pyfunction!(reg_max, m)?,
        wrap_pyfunction!(reg_min, m)?,
        wrap_pyfunction!(reg_fma, m)?,
        wrap_pyfunction!(reg_bitwise, m)?,
        wrap_pyfunction!(reg_reduce, m)?,
        wrap_pyfunction!(reg_cond_rescale, m)?,
        wrap_pyfunction!(reg_softmax_rescale, m)?,
        wrap_pyfunction!(reg_causal_mask, m)?,
        wrap_pyfunction!(reg_combine_int_frac_ex2, m)?,
        wrap_pyfunction!(reg_cvt, m)?,
        wrap_pyfunction!(reg_load, m)?,
        wrap_pyfunction!(reg_store, m)?,
        wrap_pyfunction!(fence, m)?,
        wrap_pyfunction!(cta_sync, m)?,
        wrap_pyfunction!(wg_sync, m)?,
        wrap_pyfunction!(warp_sync, m)?,
        wrap_pyfunction!(cluster_sync, m)?,
    ] {
        m.add_function(f)?;
    }
    // chunk 6 — kernel
    m.add_class::<PyKernel>()?;
    // the value simulator entry point
    m.add_function(wrap_pyfunction!(interpret, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(check_protocol, m)?)?;

    // chunk 7 — builder compatibility: the GMEM-scalar dtype map (used at runtime
    // by scalar_def) and the names the builder imports but only uses in lazy
    // (`from __future__ import annotations`) type hints, so any importable value
    // suffices.
    let dtypes = PyDict::new(m.py());
    dtypes.set_item(PyDType::BOOL, PyScalarDType::BOOL)?;
    dtypes.set_item(PyDType::I32, PyScalarDType::I32)?;
    dtypes.set_item(PyDType::U32, PyScalarDType::U32)?;
    dtypes.set_item(PyDType::I64, PyScalarDType::I64)?;
    dtypes.set_item(PyDType::U64, PyScalarDType::U64)?;
    m.setattr("_SCALAR_GMEM_DTYPES", dtypes)?;
    let reg_dtypes = PyFrozenSet::new(
        m.py(),
        [
            PyDType::F16,
            PyDType::BF16,
            PyDType::F32,
            PyDType::I32,
            PyDType::U32,
        ],
    )?;
    m.setattr("REG_DTYPES", reg_dtypes)?;
    let builtins = m.py().import("builtins")?;
    let tuple_ty = builtins.getattr("tuple")?;
    let object_ty = builtins.getattr("object")?;
    m.setattr("Shape", &tuple_ty)?;
    m.setattr("LaunchShape", &tuple_ty)?;
    m.setattr("ClusterShape", &tuple_ty)?;
    m.setattr("ScalarValue", &object_ty)?;
    m.setattr("ScalarInitial", &object_ty)?;
    m.setattr("Layout", &object_ty)?;

    // PyO3 sets `__all__` to the registered classes/functions only; the names
    // added via setattr aren't in it, so the package __init__'s `import *` would
    // skip them. Append them so `from nymph_rs import _SCALAR_GMEM_DTYPES` works.
    let all = m.getattr("__all__")?;
    let all = all.downcast::<PyList>()?;
    for n in [
        "_SCALAR_GMEM_DTYPES",
        "REG_DTYPES",
        "Shape",
        "LaunchShape",
        "ClusterShape",
        "ScalarValue",
        "ScalarInitial",
        "Layout",
    ] {
        all.append(n)?;
    }
    Ok(())
}
