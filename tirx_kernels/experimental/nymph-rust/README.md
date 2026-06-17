# nymph-rust

`nymph-rust` is the Rust port of the Nymph GPU-kernel value simulator. The
Python builder remains the user-facing construction surface and builds the Rust
IR through PyO3; the interpreter, runtime value model, scheduler, and built-in
statement semantics are implemented in Rust.

Nymph IR is a small concurrent language for describing GPU kernel protocols.
Each source statement is a semantic execution unit. The interpreter expands a
bounded launch into concrete CTAs and threads, groups active thread masks into
CTA-local execution streams, and advances those streams under a deterministic
scheduler.

## Documentation

- [IR](docs/ir.md) - the Rust IR data model, validation policy, and extension
  rules.
- [Interpreter Architecture](docs/interpreter.md) - scheduling, roles,
  blocking, value state, and the direct-mutation execution model.
- [Interpreter Semantics](docs/interpreter-semantics.md) - reviewed semantics
  by statement family and the current proof boundary.
- [Hardware Verification](docs/hardware-verification.md) - B200 validation for
  the tcgen05 ld/st datapath and MMA accumulator layouts.
- [Port Status](STATUS.md) - current implementation, correctness, performance,
  and remaining optimization work.

## Execution And Checking Model

The value simulator interprets one concrete execution of a bounded IR program.
Given inputs, the canonical execution engine steps source-ordered execution
streams, executes each statement over the active cohort, and produces terminal
runtime values and GMEM outputs.

This deterministic execution is useful for regression tests and value
validation, but it is not itself a proof of protocol safety. A protocol checker
must prove a stronger property: within a bounded configuration, all legal role
and CTA interleavings are deterministic over observable memory and outputs, and
all modeled waits can make progress. Explicit synchronization in the IR must be
sufficient to make the program confluent.

The intended checker architecture is:

1. Run the IR with a bounded configuration through the canonical execution
   engine.
2. Record executed statement occurrences per stream, including active masks and
   resolved scalar operands.
3. Build a happens-before / visibility model over those dynamic occurrences.
4. Prove that conflicting overlapping SMEM/TMEM accesses are HB ordered and
   that modeled waits can make progress.

The value simulator and any protocol checker should share the same scheduler and
statement semantics. They answer different questions: value simulation asks
whether one modeled execution computes expected values; protocol checking asks
whether every legal bounded execution has the same observable behavior and does
not deadlock.

## Layout

- `src/ir/` - the IR data model and kernel validation.
- `src/interpreter/` - scheduler, runner, statement dispatch, runtime values,
  and built-in semantics.
- `src/py.rs` and `python/nymph_rs/` - PyO3 bindings and the Python builder
  bridge.
- `tests/*.rs` - Rust integration tests for interpreter behavior and hardware
  validation harnesses.
- `tests/*.py` - Python binding, builder, structure, validation, and end-to-end
  tests.
- `tests/cuda/` - hand-written CUDA/PTX fixtures used by ignored hardware tests.

## Build And Test

Run from this directory:

```bash
cargo test
cargo test --features python --lib
cargo test --features python --test interpreter_runner
./run_python_tests.sh
```

Hardware tests require `nvcc` and an sm_100a/sm_100f GPU:

```bash
CUDA_VISIBLE_DEVICES=<idle Blackwell> \
  cargo test --test tcgen05_ldst_hardware -- --ignored --nocapture

CUDA_VISIBLE_DEVICES=<idle Blackwell> \
  cargo test --test tcgen05_mma_hardware -- --ignored --nocapture
```

## Proof Boundary

The interpreter treats each IR statement as one modeled semantic unit. It does
not prove per-thread arithmetic ordering inside a hardware instruction,
PTX/SASS memory-model reorderings, descriptor encoding, cache behavior, async
latency, or backend swizzle correctness. Those belong to lowering, compiler,
runtime, and hardware-validation layers.
