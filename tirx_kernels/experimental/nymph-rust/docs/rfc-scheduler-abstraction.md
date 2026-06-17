# RFC: First-Class Tile Scheduler Abstraction

**Status:** Draft.
**Scope:** Nymph IR, the value simulator (`nymph-rust`), and the bounded
synchronization checker. Codegen is out of nymph's scope but the design must be
codegen-compatible (§8).

## 1. Problem

The old inline grid-stride loop shape baked one static scheduler into each body, and
every body recomputed `task -> (m_idx, n_idx)` by hand. We want to **swap schedulers** —
static grid-stride, atomic work-stealing, hardware CLC — without rewriting the body, and
we want the value simulator and the bounded checker to handle a dynamic scheduler
without proving anything about the scheduler itself.

## 2. Two kinds of scheduler

A scheduler is a swappable component, but schedulers split by **nature**, and the split
matters:

| | **functional** | **concurrent** |
|---|---|---|
| example | `grid_stride` | `clc`, atomic-steal-with-broadcast |
| what it is | a pure index formula `tile = f(cluster, i)` | a real warp with state |
| needs a warp / SMEM / mbar? | no | **yes** — issues an async fetch, waits, broadcasts to consumers, prefetches |
| encapsulable as a black box? | **yes** — inlined, invisible | **no** — it occupies a warp, the consumers engage its protocol |
| IR form | a policy string | a **`SchedulerImpl`** region |

A functional scheduler hides completely. A concurrent one **cannot** — "`policy="clc"`"
instantiates a real scheduler warp into your kernel (it costs a warp, SMEM, and
mbarriers that the consumers wait on). Both are in v1.

For either kind, nymph **trusts** the fetch/allocation — "each tile once / terminates" is
a formula (grid_stride), atomic semantics, or a hardware guarantee — and spends its
effort on the **body's** synchronization, plus, for a concurrent scheduler, its broadcast
plumbing.

## 3. IR (v1)

```
TaskSpace(grid, fields)              e.g. grid=(num_m_tiles, num_n_tiles),
                                     fields=("m_idx","n_idx").

Scheduler(space, policy, scope=cluster)
    policy: "grid_stride"                       functional: inlined formula, no warp
          | "clc" | "atomic_steal"              concurrent: a library-provided SchedulerImpl
          | "custom"                           concurrent: your own SchedulerImpl
                                               (the region is wired separately via
                                               `scheduler_impl(sched)`, not passed into the policy)
    scope:  identifies the participants that share one task stream; default is all roles
            in one cluster using this scheduler. (Same scheduler + same scope => same
            task-field sequence to every participating role.)

ForEachTask(sched) -> token          functional consumer side:
                                     `with k.for_each_task(grid_stride_sched) as t:`
                                     `t` is the flat task index; `t.m_idx`, `t.n_idx`, ... are
                                     projected from it by the builder via scalar ops
                                     (`idx % extent`, `idx // extent`) — not a dedicated IR node.

# concurrent schedulers only:
SchedulerImpl   region whose warp IS the scheduler; the simulator runs it, the checker
                checks its broadcast synchronization with the same HB, deadlock, and
                shared-memory race checks used for the rest of the program.
sched_next      the ONE op whose result is non-deterministic by design (atomic fetch /
                CLC `try_cancel`) — the seam the checker trusts.
loop { ...; break_if(cond) }   explicit scheduler-protocol loops used by the scheduler
                role and by consumer roles that read its broadcast.
```

`policy="clc"` instantiates the library's CLC `SchedulerImpl` — you do not hand-write the
warp, but it is structurally present in your kernel.

## 4. Control flow

- **Functional policy:** the consumer's `ForEachTask` is a **counted loop** (canonical
  count) and branches are the existing per-lane `If` — no new control-flow primitive.
- **Concurrent scheduler protocol:** uses explicit **`loop { ...; break_if(cond) }`** in
  the scheduler role and in any consumer role that waits on the scheduler's broadcast.
  Under the canonical assignment "done" lands at a known iteration, so these loops
  **unroll to a fixed length in the trace** — the checker treats them like counted loops,
  with no termination proof.

## 5. How the simulator and checker use it

v1 runs and checks **one canonical task assignment**.

- **Simulator** runs one execution under a **canonical assignment** (a round-robin
  oracle). A `SchedulerImpl` runs **for real** — its broadcast and mbarrier handshakes
  execute — so the whole program runs with its plumbing real, not abstracted; only
  `sched_next` is the canonical stand-in (**not** the hardware's actual CLC/atomic
  schedule). For kernels whose observable writes are **per-task independent**, this is
  enough for value regression.
- **Checker** checks the **body's** synchronization under the canonical stream
  (mbarrier handshakes, buffer/TMEM reuse, shared-memory data-race freedom,
  deadlock-freedom, and the existing HB checks) and, for a `SchedulerImpl`, checks its
  broadcast plumbing with those same protocol passes. It **trusts** `sched_next`.
  Optional extra check: a kernel may
  **declare its output-region formula** in the task fields, and the checker verifies that
  formula is **injective** over tasks (so two tasks cannot clobber the same output) — a
  check on the declared formula, not generic dataflow.

**What this does and does not give.** v1 deliberately checks a single canonical
assignment and trusts the scheduler policy. That is enough for **value regression** and
**body protocol bugs under that assignment** — the large class of sync bugs that do not
depend on the schedule. It does **not** prove that every legal assignment behaves the
same (stage reuse, tail tasks, task order can in principle change what another assignment
exposes). Full assignment-confluence is out of scope (§9).

## 6. Example 1 — grid_stride (functional, no warp)

```python
space = k.task_space(grid=(num_m_tiles, num_n_tiles), fields=("m_idx", "n_idx"))
sched = k.scheduler(space, policy="grid_stride")

with k.role(warp=TMA_WARP):
    with k.for_each_task(sched) as t:
        k.tma_load(a_smem, a_gmem, coords=(t.m_idx * BLK_M, k_coord), mbar=a_full)
with k.role(warp=MMA_WARP):
    with k.for_each_task(sched) as t:
        k.mbarrier_wait(a_full)
        k.tcgen05_mma(accum, a_smem, b_smem)
        ...                                       # store C[t.m_idx, t.n_idx]
```

Config: grid `(2, 2)` = 4 tasks, 2 clusters, `stride = 2`. Canonical: cluster 0 →
tasks `{0, 2}`.

```
# trace, cluster 0  -- no scheduler ops; for_each_task pulls the canonical tile
[c0/TMA] for_each_task#0   t = {m_idx:0, n_idx:0}      # rr(c0,0)=task0
[c0/TMA] tma_load a_smem <- A[m=0]                      --arrive--> a_full
[c0/MMA] for_each_task#0   t = {m_idx:0, n_idx:0}       # same task0 (same scope -> same stream)
[c0/MMA] wait a_full ; tcgen05_mma accum += ...         <--HB-- a_full
[c0/MMA] store C[0,0]
[c0/*]   for_each_task#1   t = {m_idx:1, n_idx:0}       # rr(c0,1)=task2
[c0/*]   loop ends                                      # canonical count: 2 tasks for c0
```

Checker: HB check on the `a_full`/`a_empty` handshake; declared `C[m_idx, n_idx]` formula
injective. Nothing about the scheduler.

## 7. Example 2 — CLC (concurrent, a `SchedulerImpl`)

`policy="clc"` instantiates the library's CLC scheduler warp. Written out, it is:

```python
task_smem  = k.tensor(SMEM, dtype=I32, shape=(2, 1))   # double-buffered flat task index; -1 means done
task_full  = k.mbar(stages=2)        # scheduler -> consumers: "next tile ready"
task_empty = k.mbar(stages=2)        # consumers -> scheduler: "slot consumed"
clc_done   = k.mbar(kind=TMA)        # CLC -> scheduler: "result landed"

sched = k.scheduler(space, policy="custom")   # task_smem/task_full/task_empty above are the
                                              # kernel's own objects, wired in the SchedulerImpl + consumer below

with k.role(warp=SCHED_WARP):                              # a real warp in the kernel
    with k.scheduler_impl(sched):
        it = k.scalar(initial=0)
        with k.loop():
            stage = it % 2                                 # runtime scalar expr, not Python ^=
            phase = (it // 2) % 2
            k.mbarrier_wait(task_empty, stage=stage, phase=phase)
            idx = k.sched_next(sched).task_id              # <-- CLC try_cancel: the trusted seam
            # Ordinary scalar-to-SMEM store; not a scheduler primitive.
            k.store_scalar(task_smem[stage, 0], idx)       # ordinary SMEM scalar store
            k.mbarrier_arrive(task_full, stage=stage)
            k.break_if(idx < 0)                            # IR condition; never Python `not`
            k.scalar_store(it, it + 1)

with k.role(warp=MMA_WARP):
    it = k.scalar(initial=0)
    with k.loop():                                         # explicit Loop, no sugar
        stage = it % 2                                     # runtime scalar expr
        phase = (it // 2) % 2
        k.mbarrier_wait(task_full, stage=stage, phase=phase)
        idx = k.scalar(initial=task_smem[stage, 0])         # ordinary SMEM scalar read
        k.mbarrier_arrive(task_empty, stage=stage)          # token consumed; slot reusable
        k.break_if(idx < 0)                                # IR condition; never Python `if`
        m_idx = idx % num_m_tiles                          # field projection from flat index
        n_idx = (idx // num_m_tiles) % num_n_tiles
        k.tcgen05_mma(accum, a_smem, b_smem)
        ...                                                # store C[m_idx, n_idx]
        k.scalar_store(it, it + 1)
```

The important rule is that scheduler conditions are IR values. Do **not** write Python
`not idx`, Python `if idx < 0`, or Python `stage ^= 1` inside a builder region: those
execute at construction time, not at runtime. Use IR conditions such as `idx < 0`.

One run; the scheduler warp executes for real, `sched_next` returns the canonical tile:

```
[SCHED] wait task_empty[0]
[SCHED] idx = sched_next -> 0                            # CLC try_cancel; sim: canonical; checker: TRUSTED
[SCHED] copy task_smem[0] <- 0                            --arrive--> task_full[0]
[MMA]   wait task_full[0]                                <--HB-- (checked by ordinary HB/deadlock/race passes)
[MMA]   read idx=0 ; derive (m_idx=0,n_idx=0) ; tcgen05_mma ; store C[0,0]
[MMA]   arrive task_empty[0]                             --HB--> lets SCHED prefetch
[SCHED] wait task_empty[1] ; idx = sched_next -> 1 ; broadcast ; ...
 ...
[SCHED] idx = sched_next -> -1 ; broadcast ; break
[MMA]   reads idx=-1 -> break
```

- **Simulator** runs `SCHED_WARP` for real, so the whole kernel including its scheduler
  warp executes; only `sched_next` is the canonical stand-in.
- **Checker** HB-checks the `task_full`/`task_empty` broadcast handshake (catches a
  plumbing deadlock) and the body's `a_full`/`a_empty`, declared `C` formula injective,
  and **trusts** `sched_next`. The boundary keeps the non-deterministic fetch out of the
  body's reasoning while the rest of the program is fully simulated and checked.

## 8. Codegen

nymph does not codegen, but the same IR is lowered elsewhere, so a policy must have a
**code form**, not only a trusted contract:

| policy | codegen emits |
|---|---|
| `grid_stride` (functional) | the inline index formula in the consumer loop — no warp |
| `clc` / `atomic_steal` (concurrent) | the **real scheduler warp** — i.e. the library's `SchedulerImpl` (the `try_cancel`/atomic + broadcast), plus explicit consumer protocol loops |
| `Custom` | your `SchedulerImpl`, as written |

So a concurrent policy is a **macro that instantiates a `SchedulerImpl`**, not a contract
that hides one: the warp is in the codegen'd kernel and in the simulated IR alike. The
payoff of having one IR for both: a library author can run the CLC `SchedulerImpl`
through nymph, confirm its broadcast handshake is deadlock-free, and only then ship it as
a trusted policy — so "trust" rests on a nymph check, not on faith.

## 9. Non-goals

- **Not** proving every legal schedule produces identical values (formal confluence). The
  body HB check + declared-formula injectivity catch synchronization bugs and the one
  value-clobber risk; full schedule-equivalence is out of scope.
- **Not** verifying the scheduler's allocation. `sched_next` is trusted (formula / atomic
  / hardware); only its synchronization plumbing is checked by the ordinary protocol
  passes.
