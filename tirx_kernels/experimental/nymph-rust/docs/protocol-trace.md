# Protocol Trace

This document is the concrete trace-mode contract for the Rust interpreter. It
records the schema consumed by the offline protocol checker and exposed through
Python when `include_events=True`.

Trace mode executes scalar, control, scheduler, barrier, sync, allocation, and
protocol state. It skips numeric payload work: no TMA payload copy, no REG ALU
payload work, no TMEM value copy, and no tcgen05 MMA BLAS/scatter.

## Public Result Shape

Rust trace runs use:

```rust
RunOptions { mode: ExecutionMode::Trace, .. }
RunPayload::Trace { report, events }
```

Python exposes the same checker through:

```python
nr.trace(kernel, inputs=None)
nr.check_protocol(kernel, inputs=None, include_events=False)
```

`nr.trace()` runs trace mode with offline protocol checking disabled and returns
only status/progress counters; its Rust payload may contain an empty `events`
vector because events are not retained. `nr.check_protocol()` runs the offline checker.
Python returns `status`, `warnings`, `diagnostics`, `pass_summary`, `rounds`, and
`executed_stmts` by default. The `events` list is marshalled only when
`include_events=True`.

## Region Schema

All memory access events use one physical byte region:

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

`tensor_id` is diagnostic metadata. Alias identity is `owner + boxes`; two
regions alias only when their `PoolId` values match and their boxes overlap.

Region validation rules:

- `boxes` must be non-empty.
- all boxes in a region must have the same rank;
- SMEM, GMEM, and REG regions are rank-1 byte ranges;
- TMEM regions are rank-2 boxes in `(lane, lane_byte)`;
- every range uses `start < end` and must be inside the owner pool bounds.

There are no separate alternate region variants. A sparse access is represented
as `Vec<BoxN>`, with unit boxes for individual points.

TMEM columns are 32-bit cells in the IR, but trace regions store lane bytes:
`col` maps to `col * 4`. For example, columns `[8, 16)` become lane-byte range
`[32, 64)`.

Projection must stay exact. A single contiguous TMA or MMA rectangle should be
emitted as one box. Disconnected footprints, such as Layout F MMA lane runs, are
emitted as multiple boxes, not as a larger enclosing box that covers untouched
lanes.

## Event Types

| Event | Fields | Meaning |
| --- | --- | --- |
| `Read` | `region`, `proxy`, `access_kind`, `scope` | A physical region was read. |
| `Write` | `region`, `proxy`, `access_kind`, `scope` | A physical region was written or logically overwritten. |
| `TmemWait` | `async_kind`, `scope` | A TMEM async wait marker. This is not a memory access. |
| `TmemAlloc` | `cta_ids`, `region`, `scope` | TMEM allocation metadata was installed. |
| `TmemDealloc` | `cta_ids`, `region`, `scope` | TMEM allocation metadata was removed. |
| `Fence` | `fence_kind`, `fence_scope`, `scope` | A generic or proxy-async fence was executed. |
| `CommitGroup` | `scope` | A cp.async bulk commit group marker was executed. |
| `WaitGroup` | `n`, `scope` | A cp.async bulk wait group marker was executed. |
| `MbarInit` / `MbarArrive` / `MbarExpectTx` / `MbarCompleteTx` / `MbarWait` | mbar target/count/phase fields plus `scope` | Mbarrier state-machine events. |
| `SyncArrive` / `Sync` | sync kind, count/thread count, cycle, optional bar id, `scope` | Sync arrival and completion events. |
| `SchedulerNext` | scheduler id, CTA id, task id, `scope` | Dynamic scheduler handoff. |

Every event carries `stmt_id` and `stmt_kind`.

Every `scope` carries `stream_id`, `cluster_id`, global `cta_id`,
`ctaid_in_cluster`, `cohort_size`, and participating `warp_ids`. Mbar targets
carry `mbar_id`, `cluster_id`, `ctaid_in_cluster`, and `stage`; the checker uses
the full identity for HB/deadlock resource keys.

`MemoryAccessKind` preserves instruction semantics:

```rust
MemoryAccessKind::Tensor(TensorAccessKind)
MemoryAccessKind::Tmem(TmemAsyncKind)
```

Tensor access kinds include `generic`, `tma_load`, `tma_store`,
`tcgen05_mma`, `ldmatrix`, and `stmatrix`. TMEM async kinds include `ld`,
`st`, `mma`, and reserved `cp`.

## Projection Rules

- SMEM tensor slices project to rank-1 physical byte boxes using tensor
  `byte_offset`, dtype size, and value-mode addressing helpers.
- GMEM and REG tensor slices also project to rank-1 byte boxes. They are kept
  for trace/debug/local def-use completeness; shared race checks target SMEM and
  TMEM by default.
- TMA GMEM endpoints use the uniform byte count and emit one rank-1 box for the
  transfer span.
- TMEM logical and datapath cell accesses project to rank-2 byte boxes in
  `(lane, lane_byte)`. Sparse cells are coalesced when adjacent; otherwise they
  remain multiple unit boxes.
- `tcgen05.mma` projects directly from closed-form MMA layout blocks. Each block
  is one TMEM byte box `[lane, lane + rows) x [col * 4, (col + cols) * 4)`.

## Statement Trace Effects

| Statement family | Trace state update | Events |
| --- | --- | --- |
| `ScalarDef(initial=SMEM tensor cell)` | Reads one scalar cell per active thread. Missing or invalid skipped payload becomes `Inconclusive`. | `Read`. |
| `StoreScalar` | Writes concrete scalar values into a shared tensor cell. | `Write`. |
| `TmaLoad` | Validates uniform operands/byte count, completes selected mbar tx bytes, invalidates destination scalar-cell payload. | GMEM `Read`, SMEM async `Write`, `MbarCompleteTx`. |
| `TmaStore` | Validates uniform operands and invalidates destination GMEM scalar-cell payload. | SMEM async `Read`, GMEM `Write`. |
| `RegLoad`, `RegStore`, register ALU | Resolves operands/destination and skips numeric payload work in trace mode. | Tensor `Read`/`Write` events for referenced regions. |
| `TmemAlloc`, `TmemDealloc` | Validates lifecycle and updates TMEM allocation metadata. | `TmemAlloc` / `TmemDealloc`. |
| `Tcgen05Ld` | Resolves datapath and allocation range, skips TMEM value read and REG write. | TMEM `Read`, REG `Write`. |
| `Tcgen05St` | Resolves datapath, allocation range, and overlap, skips REG value read and TMEM cell write. | REG `Read`, TMEM `Write`. |
| `Tcgen05Mma` | Checks operand shape, issue layout, destination range, and TMEM allocation. | SMEM async `Read` for A/B, TMEM `Read` if accumulating, TMEM `Write`. |
| `LdMatrix`, `StMatrix` | Validates full-warp issue and selected row-address slices. | Tensor `Read`/`Write` with `access_kind` set to `ldmatrix` or `stmatrix`. |
| `CpAsyncBulkCommitGroup`, `CpAsyncBulkWaitGroupRead` | Updates cp.async group markers. | `CommitGroup` / `WaitGroup`. |
| `Fence`, sync, mbarrier, scheduler statements | Update protocol-visible state. | Corresponding non-memory events. |

## Python Event Example

```python
report = nr.check_protocol(kernel, include_events=True)
event = next(e for e in report["events"] if e["kind"] == "read")
```

Typical memory event shape:

```python
{
    "kind": "read",
    "stmt_id": 42,
    "stmt_kind": "Tcgen05Mma",
    "proxy": "async",
    "access_kind": "tcgen05_mma",
    "access_category": "tensor",
    "region": {
        "tensor_id": 7,
        "owner": {"kind": "smem", "cta_id": 0},
        "boxes": [{"ranges": [(0, 8192)]}],
    },
    "scope": {
        "stream_id": 0,
        "cluster_id": 0,
        "cta_id": 0,
        "ctaid_in_cluster": 0,
        "cohort_size": 32,
        "warp_ids": [0],
    },
}
```

TMEM boxes use the same shape, but `owner.kind == "tmem"` and each box has two
ranges: lane range and lane-byte range.

## Test Coverage

```bash
cargo test --no-default-features
cargo build --release --features python
PYTHONPATH="$PWD/_pybuild" python -m pytest tests/interpreter/test_protocol.py -q
PYTHONPATH="$PWD/_pybuild" python -m pytest tests/kernels/test_perf_bench.py -q -s
```
