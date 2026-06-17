# Kernel Invariants (B200 / sm_100)

These rules apply by default to every TIRx kernel in this repo. Each one is
**zero-tradeoff**: when the precondition holds, apply it — no analysis required.

**Not EV-weighted.** An invariant encodes what becomes the bottleneck *after*
you optimize today's — which the current BA cannot see. "Low-EV while X
dominates" is never a reason to leave one violated; it bounds the space the EV
comparison ranks within, it is not a candidate in it.

Deviations are allowed but must be flagged in the diff and tracked:

```
# INVARIANT-OVERRIDE I<n>: <one-line reason>
```

…and a row added to `EXEMPTIONS.md` in the kernel's directory.

The `See:` cross-references below point to the `optimization-guide` skill's
`playbook.md` for implementation detail (API forms, performance numbers,
recipes).

---

## I1a. Persistent grid, 1 CTA / SM

`Tx.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})` (right after
`Tx.device_entry()`) + `Tx.cta_id([SM_COUNT])` + a tile scheduler loop.
Scale via additional warpgroups (`Tx.warpgroup_id([w])` with `w > 1`), not
additional CTAs.

**Audit.** Every `Tx.cta_id([N])` must have `N == SM_COUNT`; `SM_COUNT *
k` for `k > 1` is the violation (`k` CTAs per SM). Every kernel that runs a
tile-stride loop must emit `Tx.attr({"tirx.launch_bounds_min_blocks_per_sm":
1})` (→ `__launch_bounds__(threads, 1)`).

**Why.** A software-pipelined kernel relies on a deterministic issue order.
With 2 CTAs/SM the warp scheduler interleaves both freely, the relative phase
drifts, and the steady-state you tuned stops holding. Splitting also loses
intra-CTA operand reuse and replaces SMEM mbarriers with DSMEM/global
handshakes.

**See:** `playbook.md` #14.

---

## I1b. Single-role warp partitioning

Split warps and warpgroups into single-role **branches, each owning its own
tile-stride loop** — NOT one shared `for tile` loop with `if role:` guards
between the steps, which scatters a role's issues across many blocks:

```python
# WRONG — compute-major loop, role-gated ifs: the MMA issues are scattered
for tile in ...:
    if wg == COMPUTE: <prep operand 1>
    if wg == MMA:     <gemm 1>            # MMA issue #1
    if wg == COMPUTE: <read 1; prep 2>
    if wg == MMA:     <gemm 2>            # MMA issue #2  ← different block
    ...                                   #   → each gemm in its own block
```

Instead, each role is one branch owning its tile loop, with its register
budget via `Tx.ptx.setmaxnreg(inc, reg_count)`:

```python
# RIGHT — role-major: each role owns one loop holding all its issues
if wg_id == 3:                              # TMA + MMA warpgroup
    Tx.ptx.setmaxnreg(False, 48)
    if warp_id == 1:                        # role: TMA load
        for tile in ...:  ...               # every Q/K/V copy_async, one block
    elif warp_id == 2:                      # role: TMA store
        for tile in ...:  ...               # every O copy_async, one block
    elif warp_id == 0:                      # role: MMA issue
        for tile in ...:  ...               # every gemm_async, one block
elif wg_id < 2:                             # role: softmax (wg 0, 1)
    Tx.ptx.setmaxnreg(True, 200)
    for tile in ...:  ...
elif wg_id == 2:                            # role: correction + epilogue
    Tx.ptx.setmaxnreg(False, 64)
    for tile in ...:  ...
```

Two hard rules:

1. **TMA, MMA, and CUDA-core compute each need their own role.** TMA
   (`Tx.copy_async(dispatch="tma")`) and MMA
   (`Tx.gemm_async(dispatch="tcgen05")`) are async-engine issues; softmax,
   correction, and epilogue rescale are synchronous CUDA-core compute.
   Mixing any two on one warp serializes them — the async wait stalls the
   math, and the math stalls the next async issue. Give each engine a
   dedicated owner warp(group).

2. **All issues of one role stay in one block** (the WRONG/RIGHT pair above).
   The test is the loop nesting, not the warp guards — separating the engines
   onto their own warps is necessary but not sufficient. Both forms can be
   runtime-equivalent (the same mbarriers serialize them), but only role-major
   shows each engine's full issue stream in one place. Don't scatter a
   "convenience" TMA call into the MMA block, even one.

**Audit.** Find the `wg_id`/`warp_id` guard each engine's issues sit under —
TMA (`copy_async(dispatch="tma")`), MMA (`gemm_async(dispatch="tcgen05")`), and
CUDA-core math — and confirm the three are **pairwise disjoint** (no shared
`wg_id` unless split by disjoint `warp_id`). Two engines under the same
warp(group) is the violation. Write the three guards down; a one-line "looks
fine" verdict is how it slips through.

Reference: `tirx_kernels/attention/flash_attention4.py` `_kernel`.

**See:** `playbook.md` #14.

---

## I2. TMA for bulk GMEM ↔ SMEM

`Tx.copy_async(gmem, smem, dispatch="tma")` (or the reverse) for any bulk
contiguous transfer. Use `use_tma_reduce="add"` for atomic accumulation.

**Why.** TMA does one coalesced async copy from a dedicated engine. Scalar
`ld.global` / `st.global` loops force every lane to issue its own access and
stall on `no eligible`.

**Skip when:** transfer is < ~1 KB or accesses are non-contiguous.

**Boundary tiles in packed/varlen layouts.** When a fixed-shape TMA store
would overrun valid memory belonging to adjacent tiles (sub-tile-sized
sequence end), two options: **pad** the buffer's per-item allocation to
a tile multiple so the overrun lands in padding — works only for
intermediate buffers, since the padding bytes can't be safely read
downstream — or use **scalar** writes, the only correct choice for
externally-visible outputs.

**See:** `playbook.md` #13.

---

## I3. `exp2`, not `exp`

```python
LOG2_E = 1.4426950408889634
Tx.ptx.exp2(Tx.cast(x, 'float32') * LOG2_E)
```

**Why.** `ex2.approx.f32` is one PTX instruction. `Tx.exp(x)` lowers to a
2-op chain `ex2.approx(x * log2(e))`, so it costs ~2× the throughput.
Always run on fp32 — bf16/fp16 saturate the input range.

**See:** `playbook.md` #10.

---

## I4. FMA, not mul + add

Write `Tx.fma(dst, src, scale, bias)` explicitly — computes
`dst = src * scale + bias` in one instruction. For `a*b - c`, negate the bias:
`Tx.fma(dst, a, b, -c)`. Don't rely on ptxas to fuse a separate `mul` and `add`.

**Why.** One instruction with one rounding step on every NVIDIA core. The
separate form doubles instructions and adds a rounding step.

---

## I5. Minimize SMEM access overhead

A SMEM allocation must be justified by **either** a cross-thread /
cross-warp read **or** an engine op that requires SMEM at that endpoint
*with no applicable register-side variant* for this data flow. If the data
fits in registers and stays within one thread's slice end-to-end, store it
in registers — declare `Tx.alloc_local`.

Most engine ops on Blackwell have both a SMEM-endpoint variant and a
register-side variant — picking the variant that matches the data flow is
part of the rule, not orthogonal to it:

| SMEM-endpoint op | Register-side variant | When to pick reg-side |
|---|---|---|
| TMA load / store (GMEM ↔ SMEM, bulk async) | — (no bulk register path; required by I2) | never — TMA's SMEM endpoint is unavoidable for bulk async GMEM transfers |
| MMA from SMEM descriptor | `use_a_tmem=True` (A direct from TMEM), see playbook #4 | A is thread-disjoint and TMEM has the room |
| `tcgen05.cp` (SMEM **src** → TMEM) | `tcgen05.st` (REG → TMEM) | producer thread already holds the data in registers |

**Audit method (per SMEM allocation).**

1. Is one end an engine endpoint that requires SMEM for this data flow,
   with no register-side variant from the table above applicable? → SMEM
   is justified, stop.
2. Otherwise, look at the partition: does thread T's reader access match
   thread T's writer access (same row / slice)? If every slice is read
   only by the thread that wrote it, the buffer is **thread-disjoint** →
   eliminate, move to `Tx.alloc_local`. (Unless the per-thread slice
   exceeds the register budget.)
3. Otherwise (some slice read by multiple threads — warp broadcast,
   warp-cooperative compute): SMEM is justified. If it's hot, apply the
   three sub-actions below — *vectorize*, *minimize bank conflicts
   (swizzle)*, *hoist loop-invariants* — in that order.

**Why.** Every SMEM reference in source emits one LDS or STS — invisible
at source level, dominant in hot loops. The anti-pattern is *one* thing —
"same thread, through SMEM, back to itself" — that surfaces in two shapes:

1. **In one expression (RMW).** `A_smem[r,c] = -w_smem[r] * A_smem[r,c] *
   g_smem[c]` — 3 LDS + 1 STS, the read of `A_smem[r,c]` is on the same
   thread that just wrote it. Visible at one line, easy to grep.
2. **Across two passes (scratch).** Pass 1 writes `s_smem[tid, :]`, pass 2
   reads `s_smem[tid, :]`. Same slice, same thread, invisible at any
   single line — only the partition comparison reveals it. Consumer may
   need a restructure first (readback-compute fusion) to align partitions.

Fix in both cases: keep the value in registers (`Tx.alloc_local` or the
existing `mma_reg`).

**When SMEM is unavoidable but the access is hot, in order:**

1. **Vectorize** with slice-form `Tx.copy(smem[r, c0:c1], reg[i0:i1])`
   (one 16-byte LDS/STS; ptxas does not auto-vectorize scalar SMEM
   loops). Works the same on unswizzled and swizzled SMEM — the CUDA
   copy dispatch picks the right codegen, no helper at the call site.
   See playbook #5 for alignment, #6 for the swizzled-tile mechanism.
2. **Minimize bank conflicts (swizzle)** — reason about per-lane offsets;
   apply `SwizzleLayout` only when multiple lanes in a warp land on the
   same bank. See playbook #6 for the mode/width choice.
3. **Hoist loop-invariants** to per-thread registers.

**See:** `playbook.md` #5 (vectorize details, readback-compute fusion)
and #6 (swizzle).

---

## I6. Iterate the actual tile count

Loop and launch over the real tile count. Never pad the grid or the inner
loop to a multiple of `TILE`.

**Why.** Structurally empty tiles (trailing rows in a varlen-packed batch,
out-of-bounds positions when the grid is rounded up) burn cycles for zero
output and confuse the schedule's steady-state.

**See:** `playbook.md` #12.

---

## I7. One mbarrier per event, one buffer per purpose

Give every distinct sync event its own mbarrier (a distinct object, or a
distinct slot of a multi-slot barrier) and every distinct tensor its own
buffer. Never reuse one `MBarrier`/`TCGen05Bar` for N unrelated handshakes
(e.g. a single `mma_bar` across all 7 GEMMs) or one SMEM/TMEM buffer for two
different tensors. Where two objects have **disjoint lifetimes**, overlap their
*storage* with the pool base so the distinct names cost no extra memory:

```python
base = Tx.meta_var(pool.offset)   # parser-only Python int (see gotcha)
A = pool.alloc_mma(shape_a, dt)   # purpose A
pool.move_base_to(base)
B = pool.alloc_mma(shape_b, dt)   # purpose B reuses A's bytes (A dead before B born)
```

`move_base_to` to a known literal when the base is one — the M=64/M=128 TMEM
regions sit at `move_base_to(0)`.

**Why.** A barrier/buffer is the unit the schedule, racecheck, and a reader
reason about. One object serving many things conflates them: "MMA4 done" is
indistinguishable from "MMA1 done", manual phase juggling stands in for
identity, and a wrong-buffer bug hides behind the shared name. One-per-thing
makes each dependency explicit and each access attributable; aliasing keeps it
free. mbarriers are 8 B — give them distinct objects, don't bother aliasing;
only real data tensors are worth the `move_base_to`.

**Gotcha.** Capture the base with `Tx.meta_var(pool.offset)`, not a bare
`base = pool.offset`. A bare body assignment binds a TIR var, so `move_base_to`
then runs a TIR compare inside a Python `if` and the parser errors with "Cannot
use and/or/not operator to Expr". `Tx.meta_var` keeps it a parser-only Python
value, so `move_base_to(base)` does plain int arithmetic.
