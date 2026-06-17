# Interpreter Semantics

This document records the reviewed semantics boundary for the Rust interpreter.
Rows describe deterministic value-simulator behavior, not full hardware
ordering, latency, cache, descriptor, or compiler-lowering semantics.

Structural control statements are interpreter core: `KernelInit`,
`KernelFinalize`, `Role`, `ForLoop`, `ForEachTask`, `SchedulerImpl`, `Loop`,
`BreakIf`, and `If`. Leaf-style operation statements execute through built-in
cohort executors. Unsupported statements fail closed with `unsupported_stmt`.

In `ExecutionMode::Value`, REG/SMEM/GMEM values are stored in typed Rust/ndarray
containers. F16 and BF16 values are rounded at the typed boundary and then held
as f32-backed arrays for hot-path arithmetic. Integer arrays use fixed-width
native integer storage. TMEM is modeled as a CTA-local physical scratchpad of
32-bit cells.

In `ExecutionMode::Trace`, scalar/control/protocol state still executes, but
payload numeric work is skipped. Trace runs return `RunPayload::Trace` for
Passed, Failed, and Inconclusive outcomes; failed value runs return no payload.
The normalized trace data structures and per-statement event emissions are
documented in [Protocol Trace](protocol-trace.md).

## Statement Families

| Statement family | Status | Runtime behavior | Main fail-closed conditions |
| --- | --- | --- | --- |
| `TensorDef`, `MBarDef` | Core metadata | Discovery metadata only; no dynamic execution stream effect. Mbar definitions seed runtime mbar identity availability. | Invalid metadata is rejected by IR validation. |
| `ScalarDef`, `ScalarStore`, `StoreScalar` | Reviewed scalar | Per-thread scalar values in the active cohort. `ScalarDef(initial=TensorSlice)` can load one GMEM/SMEM scalar in value mode, and in trace mode when the scalar cell is concrete and valid. `StoreScalar` writes a scalar expression to an SMEM scalar slice in both modes. | Missing input in value mode, invalid skipped-payload scalar bridge in trace mode, unsupported scalar dtype, unresolved expression, undefined variable. |
| `KernelInit`, `KernelFinalize`, `Role` | Core structural | Push child frames and filter active masks. Top-level init and finalize form per-CTA epochs around the main body. | Invalid scope placement is rejected by IR validation. |
| `ForLoop` | Core structural | Requires uniform bounds and positive step; writes the loop variable before body execution; repeated body statements are distinct dynamic occurrences. | `divergent_loop_bounds`, `invalid_loop_step`, undefined loop variable use. |
| `ForEachTask` | Reviewed scheduler | Functional scheduler consumer loop. `grid_stride` maps each cluster to the canonical task subsequence and writes a task variable before each body execution. | Invalid scheduler metadata, unsupported policy, undefined task variable use. |
| `SchedulerImpl`, `SchedNext`, `Loop`, `BreakIf` | Reviewed scheduler | Concurrent scheduler body executes as ordinary stream code. `SchedNext` returns the canonical flat task index and `-1` terminal sentinel; `Loop`/`BreakIf` model runtime scheduler/consumer loops. | Invalid scheduler policy, missing dynamic loop for `BreakIf`, divergent/unresolved break condition, invalid task space. |
| `If` | Core structural | Evaluates the condition over the active cohort, pushes a true-lane child frame, and reconverges to the parent frame. | Unresolved condition; role-scope conditions rejected by validation where applicable. |
| `TmemAlloc`, `TmemDealloc` | Reviewed lifecycle | Full-warp issue. `cta_group=1` acts on the issuing CTA. `cta_group=2` is a CTA-pair collective that blocks until the peer reaches the same collective occurrence. Allocation ensures a scratchpad; deallocation clears the physical range. | Invalid mask, missing peer, duplicate allocation, non-identical overlap, allocation order violation, missing/mismatched deallocation, leaked allocation. |
| `MBarrierInit` | Reviewed phase | Initializes a mbar cell with expected arrivals, pending arrivals, zero pending tx bytes, and parity 0. | Duplicate init, invalid stage, remote CTA out of range, divergent target. |
| `MBarrierArrive` | Reviewed phase | Decrements pending arrivals once per dynamic statement occurrence; flips parity when arrivals and pending tx bytes reach zero. | Uninitialized cell, non-positive count, arrival underflow, invalid stage, divergent operands. |
| `MBarrierExpectTx` | Reviewed phase | Adds pending transaction bytes to the current phase. | Uninitialized cell, invalid stage, divergent target. |
| `MBarrierArriveExpectTx` | Reviewed phase | Atomically applies expect-tx, then one arrival in the value model. | Uninitialized cell, arrival overflow/underflow, invalid stage, divergent target. |
| `MBarrierWait` | Reviewed phase | Blocks while the requested phase is current. Phase-less waits park on the current parity through a precise `WakeCondition::Mbar` and advance without re-running when parity flips. | Uninitialized cell, invalid phase, invalid stage, divergent target. |
| `TmaLoad` | Reviewed tile copy | Copies a GMEM tile into SMEM in value mode and completes selected mbar tx bytes. Multicast writes all selected destination CTAs. Trace mode emits region events, completes mbar tx, and invalidates destination scalar cells without moving payload bytes. | Missing input in value mode, unsupported rank, OOB slice, byte-count mismatch, tx underflow, invalid multicast mask, missing peer, divergent operands. |
| `TmaStore` | Reviewed tile store | Copies a current-CTA SMEM tile into GMEM in value mode. Existing GMEM values are preserved outside the tile. Trace mode emits region events and invalidates destination scalar cells without reading SMEM bytes. | Missing SMEM source in value mode, unsupported rank, OOB slice, metadata mismatch, divergent operands. |
| `CpAsyncBulkCommitGroup`, `CpAsyncBulkWaitGroupRead` | Reviewed markers | Per-stream group markers only. Commit increments a stream counter; wait with `n=0` clears it. No tensor movement and no blocking in v1. | Nonzero wait count if it bypasses IR validation; trace limit. |
| `Tcgen05Mma` | Reviewed value/trace | Value mode computes dense f16/bf16 MMA values into TMEM cells. Trace mode checks issue shape/range/allocation and emits SMEM-read/TMEM-write events without gathering SMEM operands or calling BLAS. Supports `cta_group=1` with `m=64/128` and `cta_group=2` with `m=128/256`; supported layouts are checked against B200 fixtures. | Unsupported shape/group, missing operands in value mode, invalid issue mask, missing accumulator/allocation, TMEM out of range. |
| `Tcgen05Commit` | Reviewed mbar bookkeeping | Immediately applies one mbar arrival to selected targets. `cta_group=2` is a peer-active gate, not a matched-operation rendezvous. | Uninitialized mbar, arrival overflow, invalid stage, invalid mask, missing/exited peer, divergent operands. |
| `Tcgen05Ld` | Reviewed value | Reads each active thread's datapath-assigned TMEM cells into register slices. Datapath arrays cover all supported shape/num configurations. | Non-full-warp issue, non-uniform or non-32-aligned row, dtype mismatch, wrong register count, out-of-range or unwritten TMEM cell. |
| `Tcgen05St` | Reviewed value | Writes register slices into each active thread's datapath-assigned TMEM cells. Uses the same datapath as `Tcgen05Ld`, reversed. | Non-full-warp issue, non-uniform or non-32-aligned row, dtype mismatch, wrong register count, out-of-range cell, overlapping writes, missing scratchpad. |
| `Tcgen05WaitLd`, `Tcgen05WaitSt` | Reviewed markers | Trace/value markers only. The value model copies synchronously at ld/st. | None beyond trace limit. |
| `LdMatrix` | Reviewed value/trace | Models PTX `ldmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16`. Each active warp uses lane groups 0..7, 8..15, 16..23, and 24..31 as row-address providers for matrices 0..3. Value mode packs two raw b16 SMEM elements into each lane's b32 REG fragment. Trace mode records the exact SMEM row-address footprint and register write without reading payload bytes. | Non-full-warp issue, unsupported shape/num/type, wrong row slice or register fragment size, invalid dtype, OOB or unwritten SMEM cell in value mode. |
| `StMatrix` | Reviewed value/trace | Models PTX `stmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16` as the inverse raw-bit scatter from b32 REG fragments to SMEM row-address slices. Trace mode records the exact register read and SMEM row-address footprint, then invalidates destination SMEM payload cells. | Non-full-warp issue, unsupported shape/num/type, wrong row slice or register fragment size, invalid dtype, overlapping SMEM writes, missing REG source in value mode. |
| `RegLoad`, `RegStore` | Reviewed value | Vectorized movement between register rows and SMEM/GMEM dense values. Values are coerced at the destination dtype boundary. | Missing source, OOB slice, metadata mismatch, overlapping shared/global writes. |
| `RegAdd`, `RegSub`, `RegMul`, `RegMax`, `RegMin` | Reviewed value | Vectorized REG ALU for f16/bf16/f32/i32/u32. Float results round to destination dtype; integer ops wrap to 32 bits, with signed i32 and unsigned u32 comparisons. | Missing source, OOB slice, metadata mismatch, nonnumeric dtype. |
| `RegFma` | Reviewed value | Vectorized REG `a * b + c` for f16/bf16/f32, rounded to destination dtype. | Missing source, OOB slice, metadata mismatch, unsupported dtype. |
| `RegCvt` | Reviewed value | Converts f32 REG values to f16 or bf16 using round-to-nearest-even. | Missing source, OOB slice, metadata mismatch, unsupported conversion. |
| `Fence` | Reviewed marker | Records an ordering marker with kind, scope, and active mask; mutates no runtime values. | Trace limit. |
| `CtaSync` | Reviewed rendezvous | Blocks until the full current CTA reaches the same occurrence. | Partial CTA arrival deadlock. |
| `WgSync` | Reviewed rendezvous | Blocks until the current warpgroup reaches the same occurrence; `barrier_id` is part of the rendezvous key. | Partial warpgroup arrival deadlock; invalid placement rejected by validation. |
| `WarpSync` | Reviewed rendezvous | Blocks until every represented warp has full-lane arrival. | Partial warp arrival deadlock; invalid placement rejected by validation. |
| `ClusterSync` | Reviewed rendezvous | Blocks until every CTA in the current cluster reaches the same occurrence. | Missing or exited peer CTA; partial cluster arrival deadlock. |

## Boundary

The interpreter models deterministic statement-level value effects and protocol
bookkeeping. It does not model async operation queues, hardware instruction
latency, tensor-core exact accumulation order, cache effects, PTX memory
ordering, descriptor encoding, or backend swizzle lowering. Checkers and
hardware tests must consume the executed statement stream and runtime evidence
to validate those properties separately.
