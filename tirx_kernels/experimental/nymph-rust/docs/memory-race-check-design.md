# Detailed Design: `memory_race_check`

`memory_race_check` proves shared memory data-race freedom for completed trace
executions. It consumes `Kernel IR + TraceEvent stream` after trace execution
has already returned `Passed`.

## Scope

The pass covers memory events expressed as unified physical byte regions:

```rust
Read  { region: Region, proxy, access_kind, scope }
Write { region: Region, proxy, access_kind, scope }
```

The checker target set is:

- SMEM rank-1 byte boxes in `PoolId::Smem { cta_id }`.
- TMEM rank-2 byte boxes in `PoolId::Tmem { cta_id }`, with coordinates
  `(lane, lane_byte)`.

GMEM and REG events remain in the trace for diagnostics and future analyses, but
they are not shared race targets for this pass. Alias identity is `PoolId +
boxes`; `tensor_id` is diagnostic metadata and does not separate aliases.

## Required Result

For every pair of SMEM/TMEM accesses `A` and `B`, the pass fails when all of
these are true:

```text
A.region.owner == B.region.owner
regions_overlap(A.region, B.region)
A.mode == Write || B.mode == Write
!happens_before(A.event_idx, B.event_idx)
!happens_before(B.event_idx, A.event_idx)
```

This pass does not prove prior-write completeness, read-from identity, or write
consumption. A read without a prior write is not an error by itself. A write that
is never read is not an error by itself.

The single race diagnostic code is `memory_data_race`.

## Access Model

The pass normalizes memory events into:

```text
AccessRecord {
  event_idx,
  mode: Read | Write,
  region: Region,
}
```

Only records whose owner is SMEM or TMEM participate. Sparse accesses remain
`Vec<BoxN>` regions. A point is a unit box in every dimension.

## Frontier Algorithm

The checker maintains one `MemoryRaceFrontier` per `PoolId`:

```text
MemoryRaceFrontier {
  reads:  Vec<AccessRecord>,
  writes: Vec<AccessRecord>,
}
```

Events are processed in canonical trace order, but cross-stream trace order is
not an ordering proof.

- `Read`: query overlapping entries in the write frontier. If any overlapping
  write is unordered both ways, report `memory_data_race`. Then prune older
  read-frontier entries that are covered by this read and ordered before it, and
  append the read to the read frontier.
- `Write`: query overlapping entries in both the write frontier and the read
  frontier. If any overlapping access is unordered both ways, report
  `memory_data_race`. Then prune older frontier entries that are fully covered
  by the new write and ordered before it. Finally append the new write.

Reads never split existing writes or define global spatial partitions. A large write followed
by a partial read performs an overlap/HB query against the large write; it does
not create `[read)` and `[remaining)` frontier fragments. Partial cover is retained
conservatively.

## Ordering Inputs

The pass calls shared ordering analysis:

```text
happens_before(a_event_idx, b_event_idx) -> bool
```

Ordering facts are schedule-independent:

- same-stream program order;
- mbar release to matching mbar wait.

Cross-stream trace vector order is not a happens-before edge.

## Projection Requirements

Projection happens before trace emission. The checker does not resolve logical
tensor footprints.

- SMEM tensor slices must emit exact physical byte boxes.
- TMEM logical/cell accesses must emit exact `(lane, lane_byte)` boxes.
- TMA GMEM transfer endpoints should emit one byte-count box for the transfer
  span.
- `tcgen05.mma` must emit one box per closed-form MMA layout block:
  `[lane, lane + rows) x [col * 4, (col + cols) * 4)`.
- If an op cannot project an exact region, trace emission should return
  `Inconclusive`; it must not emit an approximate enclosing box.

## Diagnostics

`memory_data_race` diagnostics include:

- `left_event_idx` and `right_event_idx`;
- left/right `stmt_id` and `stmt_kind` when available;
- left/right access modes;
- owner summary;
- one overlapping witness box.

The pass works on SMEM and TMEM through the same `Region` overlap path. Tests
cover physical SMEM aliasing, write/write and write/read races, read/read
non-conflicts, owner separation, non-overlap, same-stream and mbar HB ordering,
large-write/partial-read behavior, and TMEM Layout F untouched lane gaps.
