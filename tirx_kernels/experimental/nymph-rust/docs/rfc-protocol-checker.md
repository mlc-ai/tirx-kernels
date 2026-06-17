# RFC: Protocol Checker On The Interpreter Runner

**Status:** Draft.
**Scope:** `nymph-rust` interpreter, protocol checker, and Python bridge behavior.
IR syntax and codegen are out of scope.

## 1. Problem

The current interpreter already has the hard parts a protocol checker needs:

- CTA/thread expansion and deterministic stream scheduling;
- nested role/control-flow execution;
- precise mbarrier wake and deadlock detection;
- CTA-local ownership for SMEM/TMEM and cross-CTA cluster state;
- fail-closed runtime diagnostics.

Duplicating that machinery in a separate checker would create a second execution
engine with a second set of synchronization semantics. Instead, the protocol
checker should reuse the interpreter runner and statement handlers, while avoiding
value work that is irrelevant to protocol checking.

The important performance target is that protocol trace mode must not behave like
"value mode plus logging". For example, a GEMM trace must not gather SMEM operands,
call OpenBLAS, or scatter numeric accumulator values into TMEM. It only needs the
protocol facts: which cohort issued the MMA, which SMEM regions were read, which
TMEM region was written, and which barriers/syncs order those effects.

## 2. Design Goals

- Reuse the interpreter scheduler, frame stack, blocking/wake logic, scalar
  evaluation, and fail-closed diagnostics.
- Make execution mode explicit and mutually exclusive: either value simulation or
  protocol trace/checking, never both in one run.
- Let each statement handler decide what to do in each mode. The runner should not
  know op-specific value or trace semantics.
- Keep trace mode fast by computing only values needed to drive control flow,
  addressing, synchronization targets, and protocol metadata.
- Retain normalized events for full `check_protocol` runs so offline passes and
  optional debugging can inspect the trace; let raw `nr.trace()` avoid event
  retention and report only status/progress counters.
- Preserve current value-mode behavior and Python `nr.interpret(kernel, inputs)`
  outputs.

## 3. Non-Goals

- No numeric correctness in trace mode. A trace-mode `tcgen05_mma` does not compute
  accumulator values.
- No combined value+trace mode. Trace computes only control/address scalars and
  protocol state. If a test needs numeric values plus protocol events, run the kernel
  twice under separate modes.
- No per-op value opt-in in v1. Payload-dependent control/addressing is reported as
  inconclusive instead of selectively materializing payload values.
- No independent scheduler/checker engine.
- No formal proof of every legal dynamic task assignment. The checker still runs the
  interpreter's canonical execution, as described in the scheduler RFC.

IR schema invariant: **the IR has no opaque metadata channel.** Any field that can
affect validation, interpretation, scheduling, lowering, codegen, or protocol
checking must be represented as a typed schema field, or as an explicit execution
option outside the IR. Unknown serialized fields are rejected.

## 4. Execution Mode

Replace the two booleans `RunOptions { value_mode, trace }` with one mode:

```rust
pub enum ExecutionMode {
    Value,
    Trace,
}

pub struct RunOptions {
    pub max_rounds: Option<usize>,
    pub max_executed_stmts: Option<usize>,
    pub mode: ExecutionMode,
    pub check_protocol: bool,
}
```

This is a clean, full replacement: no compatibility shims, no legacy accessors,
and no transitional period carrying both `value_mode` and `trace` booleans. Every
call site moves to `ExecutionMode` in the same change. The impossible states this
removes:

- `value_mode=false, trace=false`: unclear purpose;
- `value_mode=true, trace=true`: slow mixed mode with ambiguous ownership of work.

`check_protocol` controls whether trace mode runs the offline checker after trace
execution. `nr.trace()` uses `check_protocol=false` for raw trace timing; the
default remains `true` for protocol reports.

The runner initializes one of two execution payloads:

```text
InterpreterState
  shared protocol state:
    mbar cells
    cooperative sync/rendezvous
    TMEM allocation metadata
    scheduler cursors
    scalar environments
    tensor scalar-cell byte pools and valid masks for ScalarDef-from-tensor
    cp.async group counters

  mode payload:
    ValueState: dense tensors, REG files, TMEM scratchpads
    TraceState: protocol checker state and optional event sink
```

Scalar environments remain shared because trace mode still needs loop counters,
task ids, mbar stages/phases, offsets, and branch conditions.

## 5. Handler Contract

Each statement handler continues to receive `&mut CohortContext` and return
`StepStatus`. The difference is that the context exposes mode-specific helpers:

```rust
impl CohortContext {
    fn mode(&self) -> ExecutionMode;
    fn value(&mut self) -> IResult<&mut ValueState>;
    fn trace(&mut self) -> IResult<&mut TraceState>;
    fn emit(&mut self, event: TraceEvent) -> IResult<()>;
}
```

Handlers own the mode split:

```rust
fn execute_tcgen05_mma(ctx: &mut CohortContext, stmt: &Stmt) -> IResult<StepStatus> {
    let spec = resolve_mma_spec(ctx, stmt)?;
    match ctx.mode() {
        ExecutionMode::Value => execute_mma_value(ctx, spec),
        ExecutionMode::Trace => execute_mma_trace(ctx, spec),
    }
}
```

The shared `resolve_*` phase should do only checks and scalar/address resolution
needed by both modes. Expensive value reads, dtype conversions, BLAS calls, dense
copies, and register/TMEM materialization live only in `execute_*_value`.

## 6. Trace Events

Trace mode should emit normalized protocol events, not raw statement dumps. Memory
events use the unified physical byte `Region`; `tensor_id` is diagnostic metadata,
while aliasing is decided by `owner + boxes`.

```rust
pub struct Region {
    owner: PoolId,
    boxes: Vec<BoxN>,
    tensor_id: u32,
}

pub struct BoxN {
    ranges: Vec<(usize, usize)>, // half-open [start, end)
}

pub enum PoolId {
    Smem { cta_id: usize },
    Tmem { cta_id: usize },
    Gmem { tensor_id: u32 },
    Reg { cta_id: usize, tensor_id: u32 },
}

pub struct AccessScope {
    stream_id: usize,
    cluster_id: usize,
    cta_id: usize,
    ctaid_in_cluster: usize,
    cohort_size: usize,
    warp_ids: Vec<usize>,
}

pub struct MbarTargetEvent {
    mbar_id: u32,
    cluster_id: usize,
    ctaid_in_cluster: usize,
    stage: u32,
}

pub struct TraceEvent {
    stmt_id: usize,
    stmt_kind: String,
    payload: TraceEventKind,
}

pub enum TraceEventKind {
    Read {
        region: Region,
        proxy: MemoryProxy,
        access_kind: MemoryAccessKind,
        scope: AccessScope,
    },
    Write {
        region: Region,
        proxy: MemoryProxy,
        access_kind: MemoryAccessKind,
        scope: AccessScope,
    },
    Fence { fence_kind: FenceEventKind, fence_scope: FenceScope, scope: AccessScope },
    CommitGroup { scope: AccessScope },
    WaitGroup { n: u32, scope: AccessScope },
    MbarInit { target: MbarTargetEvent, count: i64, scope: AccessScope },
    MbarArrive { target: MbarTargetEvent, count: i64, scope: AccessScope },
    MbarExpectTx { target: MbarTargetEvent, bytes: i64, scope: AccessScope },
    MbarCompleteTx { target: MbarTargetEvent, bytes: i64, scope: AccessScope },
    MbarWait { target: MbarTargetEvent, phase: u8, scope: AccessScope },
    SyncArrive {
        sync_kind: String,
        thread_count: usize,
        count: usize,
        cycle: u64,
        bar_id: Option<u32>,
        scope: AccessScope,
    },
    Sync {
        sync_kind: String,
        thread_count: usize,
        cycle: u64,
        bar_id: Option<u32>,
        scope: AccessScope,
    },
    TmemAlloc { cta_ids: Vec<usize>, region: Region, scope: AccessScope },
    TmemDealloc { cta_ids: Vec<usize>, region: Region, scope: AccessScope },
    TmemWait { async_kind: TmemAsyncKind, scope: AccessScope },
    SchedulerNext { scheduler_id: u32, cta_id: usize, task_id: i64, scope: AccessScope },
}
```

Full checker runs retain the event list for offline passes:

```rust
pub struct TraceState {
    events: Option<Vec<TraceEvent>>,
    warnings: Vec<ProtocolWarning>,
}
```

`nr.trace()` disables offline checking and uses a non-recording trace state so
raw trace timing does not retain one `TraceEvent` per protocol action.

## 7. Per-Op Mode Semantics

| Op family | Value mode | Trace mode |
| --- | --- | --- |
| Control, roles, loops, scheduler frames | Push/advance frames and scalar vars. | Same. Control flow must be real in both modes. |
| Scalar ops | Compute scalar env values; tensor scalar initial loads read the cell. | Same — scalars drive control/addressing. `store_scalar` runs in trace too and marks its target scalar cell valid (§8). A tensor scalar load from invalid skipped-payload data makes the report inconclusive. |
| Mbarrier | Mutate phase cells and wake waiters. | Same phase-cell mutation, plus mbar events. This is protocol state, not numeric value state. |
| Sync | Mutate cooperative rendezvous state. | Same, plus sync events. |
| TMA load | Copy GMEM tile into SMEM and complete tx. | Validate uniform operands, emit GMEM read/SMEM write/complete-tx events, update mbar tx. Do not read or write tensor bytes; mark the destination scalar-cell range invalid. |
| TMA store | Copy SMEM tile into GMEM. | Emit SMEM read/GMEM write events. Do not require SMEM bytes to exist; mark the destination scalar-cell range invalid. |
| REG load/store | Move dense values between tensors and register files. | Emit source/destination region events when protocol-visible. Do not materialize register values. A `reg_store` to a tensor marks the destination scalar-cell range invalid. |
| REG ALU/cvt | Compute register values. | No-op for numeric values; register contents become unavailable to trace. |
| TMEM alloc/dealloc | Maintain allocation metadata and scratchpads. | Maintain allocation metadata only, emit alloc/dealloc events. |
| tcgen05.st/ld | Move values between REG and TMEM. | Resolve datapath region and emit REG/TMEM read/write events. Do not read/write TMEM cells. |
| tcgen05.mma | Gather SMEM, run GEMM, write TMEM. | Check issue mask, resolve operand/destination regions, emit SMEM read/TMEM write events. No GEMM, no BLAS, no accumulator scatter. |
| tcgen05.commit | Arrive mbar cell, with peer-active checks. | Same protocol mutation and peer gate, plus event. |
| cp.async/fence | Marker/no-op value behavior. | Emit ordering marker if the checker needs it; otherwise no-op. |

The key rule: trace mode preserves scheduling and protocol causality, not data
contents. Numeric payload writes are treated as writes of unavailable data: they update
protocol state and events, but they invalidate any scalar-cell range they overwrite.

## 8. Control Values Routed Through Memory

Trace mode cannot drop every value. It must compute the scalar subset that drives
execution:

- loop counters and scheduler task ids;
- branch conditions;
- mbar stages and phases;
- tensor offsets, shapes, coordinates, and byte counts;
- CTA/warp/lane scope values.

These are already scalar-evaluation responsibilities. The only bridge from bulk value
storage back into the scalar/control world is `ScalarDef(initial = tensor_cell)`.
`ScalarValue` does not reference REG, TMEM, or MMA results directly; payload can affect
control only by first writing a tensor cell and then being read through this bridge.

That gives trace mode a simple rule: it does not predict whether payload values will
matter later. It checks the bridge when it is actually used.

Trace maintains concrete scalar cells with the tensor byte pool's existing valid mask:

- `store_scalar` writes concrete bytes and marks the target scalar cell valid;
- skipped bulk tensor writes (`TMA load`, `TMA store`, `reg_store`, and similar payload
  writes) do not write bytes, but they invalidate the destination scalar-cell range;
- `ScalarDef(initial = tensor_cell)` may read only a fully valid range;
- if the range is invalid, the trace report records a warning such as
  `trace_control_from_skipped_payload` and becomes `Inconclusive`.

This is not a protocol failure: the checker did not find a bad mbarrier, sync, TMEM
lifetime, or region dependency. It means the canonical trace would need a numeric payload
value that v1 trace intentionally skipped, so the checker cannot claim the protocol run is
proven.

Input tensor cells follow the same bridge rule. If a `ScalarDef` reads a concrete input
cell and the input is provided, trace may read that one scalar cell. If the input is absent
or the cell was overwritten by a skipped payload write, the report becomes inconclusive.

Dynamic schedulers are traceable under this model. `sched_next` is a scalar/control op, so
the canonical task id is concrete. The scheduler can broadcast that id with `StoreScalar`,
and consumers can read it back with `ScalarDef(initial=task_smem[stage])`; the consumer
`break_if(idx < 0)` then evaluates normally. The GEMM tiles moved by TMA and consumed by
MMA remain payload data, so trace mode skips their numeric computation.

## 9. Fixed Trace Algorithm

The v1 algorithm is intentionally fixed. There is no per-op value policy and no
demand-driven value liveness. The pseudocode below is the common shape every handler's
trace branch follows; handlers still own their mode split as described in §5.

```rust
fn execute_trace(stmt: &Stmt, ctx: &mut CohortContext) -> IResult<StepStatus> {
    // This may call scalar_def_from_tensor_cell. If that bridge reads invalid data,
    // the run returns an inconclusive trace report instead of continuing on a guessed
    // control path.
    let resolved = resolve_scalar_and_address_operands(stmt, ctx)?;

    emit_protocol_events(stmt, &resolved, ctx)?;
    mutate_protocol_state(stmt, &resolved, ctx)?;

    match stmt.kind {
        StmtKind::StoreScalar { dst, value } => {
            let bytes = encode_scalar(eval_scalar(value, ctx)?);
            ctx.trace_tensor_cells().write_valid(dst, bytes)?;
        }
        kind if kind.is_skipped_payload_tensor_write() => {
            ctx.trace_tensor_cells().invalidate(kind.tensor_destination_region())?;
        }
        _ => {}
    }

    Ok(StepStatus::Advance)
}

fn scalar_def_from_tensor_cell(cell: TensorCell, ctx: &mut CohortContext) -> IResult<Scalar> {
    if let Some(value) = ctx.trace_tensor_cells().read_valid_scalar(cell)? {
        return Ok(value);
    }
    ctx.trace().warn("trace_control_from_skipped_payload", cell)?;
    Err(IError::TraceInconclusive)
}
```

The algorithm's invariants:

- scalar/control computation always runs;
- payload numeric computation never runs in trace v1;
- skipped payload writes must invalidate their tensor destination range, so stale
  `store_scalar` values cannot survive an overwrite;
- the only place that can turn payload into control is `ScalarDef(initial=tensor_cell)`;
- an invalid bridge read produces an inconclusive report, not a silent pass.

`TraceInconclusive` is not exposed as a runtime failure. The runner stops the canonical
trace, finalizes the sink, and returns a trace payload whose report status is
`Inconclusive`. This is a first-hit result: if a kernel has both a protocol bug and
payload-dependent control, the report exposes whichever condition the canonical trace
encounters first. Fixing or avoiding that condition and rerunning may reveal another one.

REG and TMEM do not need concrete-value stores in v1 because no current `ScalarValue`
reads them directly. They still have protocol state: allocation, ownership, issue masks,
region events, and synchronization semantics remain checked.

## 10. Protocol Checks

The first checker should focus on synchronization protocol properties that match the
existing interpreter state:

- deadlock freedom under the canonical schedule;
- mbar wait/arrive/expect/complete-tx phase consistency;
- missing or mismatched TMEM alloc/dealloc;
- TMEM range overlap and use after dealloc;
- cross-CTA peer missing/exited failures;
- invalid protocol operands surfaced by existing interpreter checks.

The checker should intentionally not infer arbitrary dataflow from register values.
Protocol correctness should be stated in terms of regions, barriers, and happens-before
events.

Event-based HB/region data-race analysis is implemented by
`ordering_analysis` and `memory_race_check`: overlapping SMEM/TMEM accesses
with at least one write must be HB ordered. Output formula injectivity remains
out of scope for the current checker.

Payload-dependent control is represented separately from protocol failure. If trace hits
`trace_control_from_skipped_payload`, the report is `Inconclusive`, not `Failed`. The
checker failed to prove the run under fixed trace semantics; it did not prove a protocol
violation.

## 11. Result Shape

Add a trace/check result separate from value outputs:

```rust
pub enum ProtocolStatus {
    Passed,
    Failed,
    Inconclusive,
}

pub struct ProtocolWarning {
    pub code: String,
    pub message: String,
}

pub struct ProtocolReport {
    pub status: ProtocolStatus,
    pub warnings: Vec<ProtocolWarning>,
    pub diagnostics: Vec<Diagnostic>,
}

pub enum RunPayload {
    Value { outputs: HashMap<TensorInstanceKey, DenseTensorValue> },
    Trace { report: ProtocolReport, events: Vec<TraceEvent> },
}

pub struct RunResult {
    pub completed: bool,
    pub failure_reason: Option<String>,
    pub diagnostics: Vec<Diagnostic>,
    pub payload: Option<RunPayload>,
    pub blocked_frontier: Vec<(usize, usize, String, String)>,
    pub rounds: usize,
    pub executed_stmts: usize,
}
```

Existing call sites migrate to `payload` in the same change. No `values()` /
`outputs()` compatibility accessors or legacy aliases are left behind. Rust-only
internal tests that need final protocol state should read it from an explicit debug
payload/report field, not from legacy result fields.

Python exposes two APIs:

```python
nr.interpret(kernel, inputs)          # value mode, returns GMEM outputs
nr.trace(kernel, inputs={})           # trace mode without offline checkers
nr.check_protocol(kernel, inputs={})  # trace mode, returns a report
```

Trace mode inputs are optional for payload values. A kernel that only needs inputs for
numeric GEMM/TMA payloads can be checked without those arrays. If a tensor input cell is
read through `ScalarDef(initial=tensor_cell)` as a control/address scalar, that specific
input value must be provided; otherwise the report is inconclusive.

Python `nr.check_protocol(...)` returns `Inconclusive` reports normally. It raises
`RuntimeError` only for malformed inputs, invalid IR, or interpreter failures that are not
checker uncertainty.

## 12. Implementation Plan

1. Introduce `ExecutionMode` and migrate internal `RunOptions` off the two booleans.
   Update every Rust and Python call site in the same change; do not keep compatibility
   constructors.
2. Keep scalar/control/protocol state shared, and make numeric payload work
   mode-aware inside handlers. Add `TraceState` with optional event retention,
   `ProtocolReport` status, and warnings.
3. Add trace event types and optional event retention. Wire `ctx.emit(...)`.
4. Add trace scalar-cell validity for tensor-backed `ScalarDef`: `store_scalar` writes
   valid bytes, skipped payload tensor writes invalidate destination ranges, and invalid
   bridge reads produce `Inconclusive`.
5. Convert handlers one family at a time:
   - mbarrier/sync/tmem lifecycle first, because they are already protocol-state
     oriented;
   - scalar next, running `store_scalar` in trace and checking
     `ScalarDef(initial=tensor_cell)`;
   - TMA next, skipping byte copies and invalidating payload tensor destinations;
   - REG next, skipping register values and invalidating `reg_store` tensor
     destinations;
   - tcgen05 last, with `tcgen05.mma` skipping BLAS in trace.
6. Expose `nr.trace()` for raw trace timing and `nr.check_protocol()` for full
   offline checker reports.
7. Add trace-mode benchmarks that compare GEMM value mode, raw trace mode, and
   full protocol checking. The
   expected outcome is that trace mode avoids the dominant GEMM/TMA byte movement and
   BLAS cost. Online bounded event processing can be added later if retained traces
   become the bottleneck.

## 13. Testing Strategy

- Keep existing value-mode Python and Rust tests unchanged.
- Mirror fail-closed interpreter tests in trace mode where the failure is protocol
  relevant.
- Add targeted tests proving trace mode skips numeric work:
  - a GEMM kernel can be checked without A/B numeric inputs if its control/protocol
    scalar sources are available;
  - `tcgen05.mma` trace emits region events and does not require SMEM values;
  - TMA trace updates mbar pending tx/parity without copying GMEM bytes.
- Add one scheduler-protocol trace test for `sched_next -> StoreScalar(SMEM) ->
  ScalarDef(SMEM)`: the CLC consumer's `break_if` must fire in trace mode (the loop
  terminates), proving control routed through SMEM works.
- Add an inconclusive test where control tries to read a scalar from TMA-written payload
  SMEM and trace mode returns `Inconclusive` with
  `trace_control_from_skipped_payload`.
- Add a stale-value test: `store_scalar` writes a valid scalar, a skipped bulk write
  overwrites that range and invalidates it, and a later `ScalarDef` returns
  `Inconclusive` rather than reading the stale scalar.
- Benchmark a GEMM e2e kernel in both modes and assert trace mode does not call the
  BLAS path.

## 14. Open Questions

- Should full `check_protocol` grow a bounded/streaming checker path for very
  large kernels, or is retaining the event stream acceptable for that mode?
- Should Python `check_protocol` return a structured report class immediately, or a
  plain dict until the report stabilizes?
