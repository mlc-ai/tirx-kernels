# RFC: Protocol Checker Algorithm

**Status:** Draft.
**Scope:** `nymph-rust` offline protocol checker algorithms, trace schema
requirements, and report contract.

## 1. Problem

The trace interpreter executes one canonical protocol run. The offline checker
must prove stronger schedule-independent conditions: barriers close correctly,
blocking operations cannot deadlock, conflicting shared memory accesses are
happens-before ordered, async/proxy/TMEM windows are drained or ordered, and no
required proof fact is missing from the trace.

The checker does not re-run numeric value simulation or enumerate interleavings.
It consumes `Kernel IR + completed TraceEvent stream`.

Full checker passes run only after trace execution returns `Passed`.

- Trace execution `Failed`: return the trace failure and diagnostics.
- Trace execution `Inconclusive`: return the trace warning and stop.
- Trace execution `Passed`: run the offline checker pipeline.

## 2. Execution Model

- A role/stream executes its dynamic operations in order.
- Each modeled op is atomic at checker granularity.
- Different roles/streams may interleave arbitrarily.
- Barriers, waits, syncs, fences, commits, and async drains constrain legal
  interleavings.
- Event vector order is canonical trace order, not cross-stream
  happens-before.

## 3. Trace Schema Requirements

Every `TraceEvent` carries:

```rust
stmt_id: u32
stmt_kind: String
```

Memory access events use a single physical byte region:

```rust
Region {
    owner: PoolId,
    boxes: Vec<BoxN>,
    tensor_id: u32,
}

BoxN {
    ranges: Vec<(usize, usize)>, // half-open [start, end)
}

PoolId =
    Smem { cta_id }
  | Tmem { cta_id }
  | Gmem { tensor_id }
  | Reg  { cta_id, tensor_id }
```

The memory events are:

```rust
TraceEventKind::Read {
    region: Region,
    proxy: MemoryProxy,
    access_kind: MemoryAccessKind,
    scope: AccessScope,
}

TraceEventKind::Write {
    region: Region,
    proxy: MemoryProxy,
    access_kind: MemoryAccessKind,
    scope: AccessScope,
}
```

`TmemAlloc` and `TmemDealloc` also carry `Region`. `TmemWait` is retained as a
non-memory event.

`AccessScope` includes `stream_id`, `cluster_id`, global `cta_id`,
`ctaid_in_cluster`, `cohort_size`, and `warp_ids`. `MbarTargetEvent` includes
`mbar_id`, `cluster_id`, `ctaid_in_cluster`, and `stage`. Resource keys for
mbarrier, sync, deadlock, and cluster-scope fences must use cluster identity
where applicable.

`MemoryAccessKind` preserves instruction semantics:

```rust
MemoryAccessKind::Tensor(TensorAccessKind)
MemoryAccessKind::Tmem(TmemAsyncKind)
```

`tensor_id` is used for diagnostics and GMEM/REG owner consistency. Alias
identity is `PoolId + boxes`.

## 4. Region Contract

Trace emission must project memory accesses into exact physical byte regions.
The checker does not resolve logical tensor slices into physical footprints.

Validation rules:

- `boxes` is non-empty;
- all boxes in one region have the same rank;
- SMEM, GMEM, and REG regions are rank-1 byte ranges;
- TMEM regions are rank-2 boxes in `(lane, lane_byte)`;
- every dimension satisfies `start < end`;
- all ranges are within owner bounds;
- GMEM/REG owner tensor id matches `region.tensor_id`.

There are no separate alternate region types. A sparse access is `Vec<BoxN>`,
and an individual point is a unit box in every dimension.

Projection rules:

- SMEM tensor slices emit exact rank-1 physical byte boxes.
- GMEM and REG tensor slices emit rank-1 byte boxes for trace completeness.
- TMEM logical/cell accesses emit rank-2 byte boxes; IR column `col` maps to
  lane bytes `col * 4`.
- `tcgen05.mma` emits one box per closed-form layout block:
  `[lane, lane + rows) x [col * 4, (col + cols) * 4)`.
- A contiguous TMA or MMA rectangle must remain one box. Disconnected footprints
  are multiple boxes, not an enclosing box.
- If exact projection is impossible, trace emission returns `Inconclusive`
  rather than emitting an approximate region.

## 5. Result Semantics

The public status remains:

- `Passed`: all implemented required checks completed and found no violation.
- `Failed`: at least one check proved a protocol violation.
- `Inconclusive`: no violation was proved, but a required proof obligation
  could not be discharged from trace/IR facts.

Reports include `pass_summary`, `warnings`, and typed `diagnostics`. Every
failure names the statement/event, resource or region, expected condition, and
witness events when available.

## 6. Checker Architecture

The checker is a pass pipeline over `Kernel IR + TraceEvents`:

- `trace_schema_audit`: validates common event fields and non-region schema.
- `trace_region_audit`: validates `Region` owner/rank/bounds/empty-box rules.
- `barrier_cycle_audit`: audits mbarrier and sync counters/cycles.
- `ordering_analysis`: builds schedule-independent happens-before edges.
- `deadlock_freedom`: proves modeled blocking operations have no wait cycle.
- `async_group_lifetime`: checks cp.async/TMA source windows.
- `tmem_async_hazard`: checks overlapping TMEM async windows.
- `tmem_lifecycle_order`: checks TMEM alloc/use/dealloc coverage.
- `memory_race_check`: checks SMEM/TMEM data-race freedom.
- `proxy_fence`: checks generic/async SMEM proxy transitions.
- `cluster_peer_consistency`, `scheduler_handoff_consistency`,
  `trace_gap_audit`.

Implementations may share scans or helper indexes, but reports should use these
pass names or close equivalents.

## 7. Unified Memory Analysis

Memory helpers operate on `Region` directly:

```text
regions_overlap(a, b):
  a.owner == b.owner
  and any BoxN pair intersects in every dimension

region_covers(a, b):
  a.owner == b.owner
  and every box in b is covered by some box in a
```

`async_group_lifetime`, `proxy_fence`, `tmem_async_hazard`, and
`tmem_lifecycle_order` all use these helpers. Barrier and deadlock passes do not
inspect regions.

`memory_race_check` proves that every conflicting shared memory pair is ordered.
For SMEM/TMEM accesses `A` and `B`, a race is:

```text
A.region.owner == B.region.owner
regions_overlap(A.region, B.region)
A.mode == Write || B.mode == Write
!happens_before(A.event_idx, B.event_idx)
!happens_before(B.event_idx, A.event_idx)
```

The pass keeps per-owner read and write frontiers. Reads query only the write
frontier, may prune older covered reads ordered before the current read, and
then join the read frontier. Writes query both frontiers, then join the write
frontier and may prune older covered frontier entries that are ordered before
the new write. Partial cover is retained conservatively. Reads do not split
writes or drive global spatial partitioning.

The pass does not prove prior-write completeness, read-from identity, or write
consumption. A read without a prior write is not an error by itself, and an
unread write is not an error by itself. Failures use `memory_data_race`.

## 8. Pass Notes

### `trace_region_audit`

Malformed regions indicate instrumentation or projection errors. Out-of-bounds
regions fail. Unknown tensor metadata needed for GMEM/REG bounds is
`Inconclusive`.

### `barrier_cycle_audit`

Mbarriers are keyed by `(mbar_id, cluster_id, ctaid_in_cluster, stage)`. Sync
barriers are keyed by their modeled hardware/resource identity. The pass checks
init before use, counter underflow/overflow, transaction byte balance, and
wait/completion phase consistency.

### `deadlock_freedom`

The pass builds a wait-for graph from blocking waits and their release witness
events. A cycle with no release outside the cycle is a failure. The completed
canonical trace alone is not a proof of deadlock freedom.

### `async_group_lifetime`

The pass tracks committed async source windows. Same-stream overlapping writes
before `wait_group` fail. Cross-stream overlaps without structural ordering are
recorded as trace gaps.

### `tmem_async_hazard`

The pass tracks TMEM async read/write windows by stream and async kind. Same
stream conflicting overlaps before the earlier window is closed fail.
Cross-stream conflicting overlaps without ordering are trace gaps.

### `tmem_lifecycle_order`

Allocation and deallocation regions must match exactly. Every TMEM memory
access must be covered by an active allocation region using byte-box coverage.

### `proxy_fence`

An async-proxy SMEM read that overlaps a prior generic-proxy SMEM write in the
same stream requires an intervening covering `fence.proxy.async`.

## 9. Test Matrix

Required coverage includes:

- `include_events=False` omits Python events by default;
- `include_events=True` returns unified `read`/`write` events with `stmt_id` and
  `stmt_kind`;
- same-owner overlap and different-owner no-alias region helpers;
- rank mismatch and out-of-bounds region audit failures;
- sparse/unit-box overlap and box coverage;
- SMEM physical aliasing where different tensor ids share bytes;
- TMEM lifecycle coverage over byte boxes;
- Layout F MMA emits multiple disjoint TMEM boxes and untouched lane gaps do
  not alias;
- TMA and contiguous MMA footprints remain single boxes when physically
  contiguous;
- `async_group_lifetime`, `tmem_async_hazard`, `tmem_lifecycle_order`,
  `proxy_fence`, and `memory_race_check` use unified overlap helpers;
- value-mode behavior is unchanged.
