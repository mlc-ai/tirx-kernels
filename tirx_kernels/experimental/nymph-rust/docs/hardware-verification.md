# Hardware Verification

This document records the hardware-backed evidence migrated from the Python
prototype docs and updated for the Rust test harnesses.

The goal is to ground the Rust value model for tcgen05 data movement and MMA
accumulator placement against real Blackwell hardware without using CUTLASS or
CuTe as the oracle. The CUDA fixtures use hand-written PTX and are run only by
ignored Cargo tests.

## tcgen05 ld/st Datapath

`src/interpreter/values/tcgen05_datapath.rs` derives the `(lane, col)` TMEM cell
that each `(thread, register)` pair moves for every supported `tcgen05.ld` /
`tcgen05.st` shape and `.xN` count.

The hardware test is `tests/tcgen05_ldst_hardware.rs`. It generates a CUDA
harness under `target/tcgen05_ldst_hardware` by default, compiles it with `nvcc`
for `sm_100a`, and compares both directions:

1. Fill TMEM with sentinel values and verify hardware `tcgen05.ld` returns the
   same cells as `datapath_index_arrays`.
2. Write register sentinels with hardware `tcgen05.st`, dump TMEM with
   `32x32b`, and verify the scatter matches the model.

Run:

```bash
CUDA_VISIBLE_DEVICES=<idle Blackwell> \
  cargo test --test tcgen05_ldst_hardware -- --ignored --nocapture
```

Use `NYMPH_TCGEN05_HW_DIR=<path>` to override the generated fixture/output
directory.

## tcgen05 MMA Accumulator Layouts

`tests/tcgen05_mma_hardware.rs` compiles the hand-written fixtures in
`tests/cuda/` and compares every logical `D[m,n]` against the physical
`(cta, lane, col)` cell assigned by the Rust placement model.

Covered cases:

| Case | Fixture |
| --- | --- |
| `m=64, cta_group=1, lane_align=0` | `tcgen05_mma_m64_dump.cu` |
| `m=64, cta_group=1, lane_align=16` | `tcgen05_mma_m64_align16_dump.cu` |
| `m=128, cta_group=1` | `tcgen05_mma_m128_dump.cu` |
| `m=128, cta_group=2` | `tcgen05_mma_cta2_m128_dump.cu` |
| `m=256, cta_group=2` | `tcgen05_mma_cta2_dump.cu` |
| `accum=true` on the m=128 layout | `tcgen05_mma_accum_dump.cu` |

Run:

```bash
CUDA_VISIBLE_DEVICES=<idle Blackwell> \
  cargo test --test tcgen05_mma_hardware -- --ignored --nocapture
```

Use `NYMPH_TCGEN05_HW_DIR=<path>` to override the compiled binary/output
directory.

## ldmatrix/stmatrix m8n8.b16

`src/interpreter/values/ldstmatrix.rs` derives the PTX fragment mapping for
`ldmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16` and
`stmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16`.

The hardware test is `tests/ldstmatrix_hardware.rs`. It generates a CUDA
harness under `target/ldstmatrix_hardware` by default, compiles it with `nvcc`
for `sm_100a`, and compares both directions:

1. Fill SMEM rows with b16 sentinel values and verify hardware `ldmatrix`
   returns the same packed b32 register fragments as the Rust mapping.
2. Write packed register sentinels with hardware `stmatrix`, dump SMEM, and
   verify the scatter matches the Rust mapping.

Covered cases are `.x1`, `.x2`, and `.x4`, each with and without `.trans`.

Run:

```bash
CUDA_VISIBLE_DEVICES=<idle Blackwell> \
  cargo test --test ldstmatrix_hardware -- --ignored --nocapture
```

Use `NYMPH_LDSTMATRIX_HW_DIR=<path>` to override the generated fixture/output
directory.

## What This Does Not Prove

These tests prove the modeled TMEM datapath and accumulator cell placement for
the covered fixtures, plus the modeled `ldmatrix`/`stmatrix` b16 fragment
mapping. They do not prove async latency, memory ordering, full descriptor
lowering correctness, or every possible dtype/special-value behavior. Those
remain separate compiler and runtime validation concerns.
