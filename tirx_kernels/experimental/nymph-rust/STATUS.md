# nymph-rust — interpreter port status

A Rust port of the nymph GPU-kernel **value simulator**. The Python IRBuilder is
kept as-is and constructs a Rust IR over PyO3; the interpreter is fully reimplemented
in Rust.

For the migrated architecture and semantics docs, start with `README.md`, then
see `docs/ir.md`, `docs/interpreter.md`, `docs/interpreter-semantics.md`, and
`docs/hardware-verification.md`.

## What's done

### IR + builder bridge
- `src/ir/` — the full IR (dtype, scalar, tensor, mbar, stmt, kernel, validate).
- `src/py.rs` — PyO3 bindings: the Python builder constructs Rust IR; `Kernel`
  auto-validates on construction. Plus the `interpret(kernel, inputs)` entry point,
  which marshals typed numpy inputs in and kernel-produced GMEM tensors back out.

### Interpreter
File layout mirrors the Python package; the **modularity contract is preserved**:

- `interpreter/values/` — the simulated memory/value layer: `arrays` (the typed
  `ValueArray1/2` storage), `dtypes` (f16/bf16/f32 rounding incl. the bf16 bit-formula),
  `indexing`, `reg_numerics`, `tensors` (GMEM/SMEM + valid mask), `smem` (byte-pool
  SMEM scratchpad), `registers`
  (per-thread reg file), `tmem` (128×512 cell grid + addressing), `mbars`, `scalars`,
  `cooperative`, `runtime`, `tcgen05_datapath`.
- engine: `ids`, `scalar_eval` (floor-div/mod, scope values, uniform fast path),
  `outcomes` (the `StepStatus` + `WakeCondition` protocol), `state`, `slice_indexing`,
  `cohort` (the vectorized per-thread surface + register/shared read-write), `mbar_ops`
  (pure phase-cell algebra + target resolution), `elementwise` (reg ALU), `transfer`,
  `tmem` (alloc lifecycle), `scheduler` (CTA epochs, frame stack, grid expansion,
  role/scope matching), `runner` (the main loop, dispatch, direct mutation, precise wake).
- `interpreter/registry.rs` + `interpreter/semantics/` — per-op executors. A semantics
  module registers its executors; `default_executor_registry` iterates the registrars.
  **Adding an op = new module + one `register` line; the runner is never edited.**
  Modules: control, scalar, metadata, fence, cp_async, mbarrier, reg, tma, tcgen05,
  tmem, sync, leaf.

Design points:
- **Cohort vectorization** — handlers operate on whole `ThreadMask`s, never loop
  threads in op logic.
- **Direct mutation** — every executor takes `&mut CohortContext` (holding `&mut state`),
  mutates state in place, and returns a light `StepStatus` (`Advance{wakes}` /
  `AdvanceContinue` / `Block(WakeCondition)` / `Fail`). There is no staged commit.
- **Precise wake** — a stream that blocks on `mbarrier_wait` parks on a
  `WakeCondition::Mbar{key,phase}`; when a later mutating step touches that cell, the
  runner re-checks the parked waiters and **advances their frame directly** — the wait
  never re-runs, so there is no phase latch and no idempotency discipline. The rare
  rendezvous / tmem-collective / tcgen05 peer-active blocks use `WakeCondition::Polled`
  (re-run each round; their re-runs are naturally idempotent).
- **Typed values** — `ValueArray1/2` stores each tensor/register in a container chosen
  for value-losslessness: f16/bf16/f32 are **f32-backed** (f16/bf16 rounded on write,
  then held exactly), integers/bool use their **native fixed-width** type. The MMA reads
  its SMEM operands as borrowed `&[f32]` (zero copy) and `sgemm`s straight into the
  column-major f32 TMEM grid; the integer-native storage keeps i64/u64 exact.
- **Fatal failure / protocol report** — failed value runs report diagnostics/frontier
  metadata and no payload, because direct mutation may have left state partial. Trace
  runs return a `RunPayload::Trace` report for Passed, Failed, and Inconclusive
  protocol outcomes.

## Correctness

The fp16/bf16 GEMM (m=512, n=256, k=64, cta_group=2) runs end-to-end and is **cell-exact**:
- vs a numpy reference `round(A @ Bᵀ, dtype)` — 0 mismatches, **fp16 and bf16**.
- vs the **original Python interpreter** — 0 mismatches, cell for cell.

This kernel exercises TMA load/store, the cta_group=2 MMA, the TMEM collective +
scratchpad, mbarrier handshakes (incl. the precise wake), cooperative sync, the
register ALU/cvt, and the direct-mutation runner.

Test coverage: `cargo test` (36 lib unit — IR, the typed value layer + dtype coercion,
engine loop — plus 11 Rust-internal integration tests in `interpreter_runner.rs` for
state the Python API can't observe: mbar-cell parity, no-partial-values-on-failure,
internal commit cells) + `tests/` (98 Python: `ir/` binding/validation/structure,
`interpreter/` per-op value behavior, `kernels/` e2e/parity/determinism).

## Performance

The standing benchmark fixes one GEMM task tile (m=512, n=256, **k=16384**, cta_group=2,
launch=(2,) — one persistent cluster) and scales the task count via
`ForEachTask(grid_stride)`.
Profile small task counts (1/2/4/8/16), linear-fit `total = a + b·tasks`, and extrapolate
the per-task slope `b` to **tasks=2048** (running 2048 for real is too slow — Python
~30 min). Measure clean (no `NYMPH_STATS`; the profiler adds ~3 ms/task).

| backend | per-task (k=16384) | 2048-task extrapolation |
|---|---|---|
| Python interpreter | ~850 ms | ~1745 s (~29 min) |
| Rust interpreter | ~60 ms | ~130 s |
| **speedup** | | **~13–14×** |

Where the per-task time goes (k=16384): the OpenBLAS `sgemm` for the MMA dominates
(~30 ms, the irreducible compute), then the MMA operand read and the TMA gmem↔smem tile
copy (~10 ms each); the interpreter overhead (dispatch, scalar resolve, scheduler,
mbarrier handshakes) is the remaining ~20 ms. The hot path is copy-free: contiguous-layout
MMA borrows the SMEM f32 operands and accumulates in place into the TMEM grid.

`NYMPH_STATS=1` emits a per-phase / per-executor profile to stderr (gated, ~zero cost
when off). `NYMPH_BLAS_THREADS=N` pins the OpenBLAS thread count (default 1).

### Remaining levers (not yet done)
- **Vectorize `scalar_eval::eval_scalar_vec`** — it still loops the cohort for non-uniform
  (lane-dependent) offsets where Python uses numpy; array-based eval would cut `eval_slice`
  and the per-stmt cost across the board.
- **Avoid the per-statement cohort clone** in `current_stmt` (return an index/slice).
- **Multithreading** — the Rust interpreter has no GIL; independent CTAs/tasks could run
  on a thread pool. The Python interpreter cannot.

## Notes
- Non-contiguous MMA layouts (Layout F: cta_group=1 m=64; Layout B: cta_group=2 m=128)
  use the general `matmul_f32` + per-cell scatter path; contiguous D/A layouts use
  the zero-copy in-place sgemm when operands are non-transposed rank-2 SMEM slices.
  Both are cell-exact; the scatter path is the universal fallback and is not on the
  measured GEMM's hot path.
- `[profile.release] debug = 1` is kept for line-table profiling (negligible runtime cost).
- Build/run: `cargo build --release --features python`, copy the `.so` into `_pybuild/`,
  then `PYTHONPATH=_pybuild`. OpenBLAS is linked via `build.rs` (override its directory
  with `BLAS_LIB_DIR`).
