# Interpreter Architecture

The interpreter runs one bounded Nymph kernel by expanding every modeled GPU
thread, grouping threads into per-CTA execution streams, and stepping those
streams in deterministic source order. It does not add a hidden launch-wide
barrier. Cross-CTA ordering must be expressed by modeled synchronization or
explicit protocol handshakes.

## Modules

- `runner.rs` - initialization, scheduler loop, dispatch, direct mutation, fatal
  failure handling, and precise mbarrier wake.
- `protocol.rs` - execution mode, protocol reports, normalized trace event
  types, and optional trace event retention.
- `scheduler.rs` - CTA epochs, execution streams, frame stacks, thread
  expansion, role masks, and scope matching.
- `state.rs` - `InterpreterState`, `RunOptions`, and `RunResult`.
- `threads.rs` - `ThreadId`, `ThreadMask`, and canonical mask helpers.
- `ids.rs` - discovery-order identity helpers.
- `registry.rs` and `semantics/` - statement executor dispatch and built-in
  per-op executors.
- `cohort.rs`, `slice_indexing.rs`, `transfer.rs`, `elementwise.rs`, and
  `mbar_ops.rs` - vectorized execution surface and shared services used by
  statement executors.
- `values/` - runtime values for GMEM/SMEM tensors, registers, TMEM, mbarriers,
  scalars, cooperative sync, dtype coercion, and tcgen05 datapath helpers.
- top-level `tmem.rs` - internal TMEM allocation and collective bookkeeping.
- `blas.rs` - OpenBLAS-backed SGEMM path for supported tcgen05 MMA cases.

## Execution Surface

Built-in statements execute through `StmtExecutorRegistry` and
`CohortContext`. Each semantics module registers executors for its statement
kinds. `default_executor_registry()` builds the dispatch table by iterating
those registrars; adding a new op family should not require editing the runner.

Executors operate on a whole active cohort. Tensor and register value movement
is vectorized over the cohort; scalar and protocol metadata may still be
resolved per active thread when uniformity or scope values matter.

Rust uses a direct-mutation model. An executor receives `&mut CohortContext`,
mutates `ctx.state` directly after its local checks, and returns a light
`StepStatus`:

- `Advance { wakes }` - the statement completed and the frame should advance.
- `AdvanceContinue` - structural control pushed or advanced a frame without
  recording an executed leaf statement.
- `Block(WakeCondition)` - the stream is parked.
- `Fail` - the run aborts.

Failed value-mode runs do not expose partial value state. Completed value-mode
runs return `RunPayload::Value { outputs }`, where `outputs` contains the final
GMEM tensor instances. Trace-mode runs return `RunPayload::Trace { report,
events }` for Passed, Failed, and Inconclusive protocol outcomes. Full
`check_protocol` runs retain the event stream for offline passes and optional
Python `include_events=True` marshalling. Raw `nr.trace()` disables offline
checking and may return an empty Rust event vector; Python raw trace returns only
status and progress counters.

See [Protocol Trace](protocol-trace.md) for the trace data structures and the
per-statement event contract.

The fallback `leaf` executor fails closed with `unsupported_stmt`. The Rust port
does not expose the historical Python per-thread custom handler registry.

## Basic Model

For each CTA in `kernel.launch_shape`, the scheduler creates one `ThreadId` for
every `(warp_id, lane_id)`. A `ThreadMask` is an ordered set of threads from one
CTA, sorted by `(warp_id, lane_id)`.

An `ExecutionStream` is one scheduled source path for one CTA. It owns:

- a stream id;
- CTA and cluster coordinates;
- the initial active mask;
- a frame stack with source bodies, program counters, and active masks;
- blocked state, if the stream is parked.

The interpreter creates streams for every CTA. It never simulates only CTA 0.

## Scheduler Shape

The scheduler visits CTAs in deterministic order. Within a scheduler round, one
CTA visit can advance each live stream in that CTA's current epoch by at most one
semantic source statement.

```text
while some CTA schedule is not complete:
  snapshot CTA activity for this round
  for each live stream selected from CTA schedules:
    step one source statement
    refresh that CTA's activity before selecting more work
  if no stream progressed during the round:
    fail with deadlock
```

Each CTA owns its own epoch state:

```text
CTA X:
  epoch 0..i   top-level KernelInit scopes
  epoch i+1..j main body role or loose-statement groups
  epoch j+1..n top-level KernelFinalize scopes
```

This ordering is local to the CTA. If CTA 0 finishes init while CTA 1 is still
blocked in init, CTA 0 may enter main. There is no implicit launch-wide,
cluster-wide, or CTA-pair barrier.

Inside the main body:

- consecutive top-level `Role` statements form one concurrent role epoch;
- consecutive top-level non-role executable statements form one loose CTA-wide
  epoch;
- `TensorDef` and `MBarDef` are metadata declarations and do not create streams;
- one CTA's current epoch drains before that CTA starts its next epoch.

## Roles And Frames

Top-level `Role` statements create one stream per matching CTA mask. Role filters
are mechanical:

| Role fields | Active mask per CTA |
| --- | --- |
| `warp=0` | all 32 lanes of warp 0 |
| `warp=0, elected=true` | lane 0 of warp 0 |
| `warpgroup=0` | all lanes of warps 0, 1, 2, and 3 |
| `warpgroup=0, elected=true` | lane 0 of warp 0 |
| `elected=true` | lane 0 of warp 0 |
| no filter | every thread in the CTA |

One top-level `Role` statement materializes as one stream per CTA with the full
matching mask. A warpgroup role is not split into four warp streams; all selected
warps advance through one stream and share its program order. Adjacent top-level
roles in the same role epoch are different streams.

Role-local sync must match the role scope. `cta_sync` is only valid in loose
CTA-scope code, never inside a `Role` body. Warpgroup roles use `wg_sync`, and
warp roles use `warp_sync`.

Nested `Role`, `KernelInit`, `KernelFinalize`, `If`, `ForLoop`, `ForEachTask`,
`SchedulerImpl`, and `Loop` statements do not create new scheduler streams. They
push or advance frames on the current stream.

Divergent `If` does not split a stream. It pushes a child frame with only the
true active lanes, then reconverges to the parent frame when the child drains.
Loops use the same frame stack. Loop bounds and step must be uniform across the
active mask; divergent bounds fail with `divergent_loop_bounds`, and dynamic
non-positive steps fail with `invalid_loop_step`.

## Blocking And Wake

A stream that blocks is parked with a `WakeCondition`.

- `WakeCondition::Mbar { key, phase }` is precise. A later mutating statement
  returns the touched mbar cell keys in `Advance { wakes }`; the runner
  re-checks parked waiters and advances satisfied frames directly. The wait
  statement is not re-run.
- `WakeCondition::Polled` is retried each round. This is used for rare
  rendezvous, TMEM collective, and peer-active gates whose retry paths are
  idempotent.

If an entire scheduler round makes no progress, the run fails with
`failure_reason="deadlock"` and returns `blocked_frontier` entries containing
the stream id, statement id, statement type, and block reason.

## Runtime Values

GMEM and SMEM values are dense tensors keyed by `(tensor, owner)`. GMEM has one
global owner; SMEM is owned by CTA. Missing shared destinations can be created
by value-mode stores when the statement semantics allow it.

Register values are vectorized register files keyed by `(reg tensor, cta_id)`.
Each instance stores rows for CTA-local threads and columns for tensor elements.
REG values are not generic tensor instances.

TMEM values are CTA-local physical scratchpads keyed by `(lane, col)`. TMEM
tensors are views over the scratchpad. `TmemAlloc` and `TmemDealloc` maintain
allocation metadata and ensure or clear scratchpad ranges.

Mbarriers are phase cells keyed by mbar identity, CTA, and stage. The model
tracks expected arrivals, pending arrivals, pending transaction bytes, and
parity. It models phase bookkeeping only, not hardware latency or memory-ordering
proofs.

## Testing Expectations

For interpreter changes, run at least:

```bash
cargo test --test interpreter_runner
cargo test --lib
```

For Python bridge changes, also run:

```bash
cargo test --features python --lib
cargo test --features python --test interpreter_runner
./run_python_tests.sh
```
