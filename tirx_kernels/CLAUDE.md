# tirx_kernels

Canonical TIRx kernel library. Kernels here are the maintained, up-to-date
versions of each workload (`kernel-evolution/` carries the evolution trail and
may use stale APIs).

## Invariants

The rules in `INVARIANTS.md` apply to every kernel in this directory. They are
zero-tradeoff: when the precondition holds, apply the rule.

@INVARIANTS.md

## Optimization catalog

The `optimization-guide` skill includes
`.claude/skills/optimization-guide/kernel-catalog.md` — a per-kernel inventory
of the **unique** optimizations each canonical kernel uses (beyond
`INVARIANTS.md` baseline) plus a reverse index from technique → kernel.

**When adding a new kernel here**, also add a subsection under §1 of
`kernel-catalog.md` listing its beyond-baseline optimizations (one bullet each,
generalization-first + grep anchor) and index each technique into §2. Same
when an existing kernel gains a new unique optimization.
