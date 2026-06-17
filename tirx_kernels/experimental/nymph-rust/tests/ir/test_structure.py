"""Targeted structure tests: a constructor must store its fields where it says,
and the operator overloading must build the right expression tree.

This is the cheap middle ground (not the full dataclass-introspection port): a
few read-back checks on the non-trivial constructors, catching realistic risks
like a swapped field or a mis-nested operator. The builder *logic* itself is
unchanged Python and already covered upstream + by the GEMM integration test.
"""

import pytest

n = pytest.importorskip("nymph_rs")


def smem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.SMEM, dtype=dtype, shape=shape, byte_offset=0)


def tmem(shape, dtype=n.DType.F32):
    return n.Tensor(space=n.MemorySpace.TMEM, dtype=dtype, shape=shape)


def test_mma_constructor_roundtrips_fields():
    # m=128, n=256, k=16; dst is TMEM/f32, operands SMEM/f16 — confirm nothing got
    # swapped between dst/a/b or m/n/k.
    dst = tmem([128, 256])[:, :]
    a = smem([128, 16])[:, :]
    b = smem([256, 16])[:, :]
    mma = n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16, accum=True)
    assert mma.m == 128
    assert mma.n == 256
    assert mma.k == 16
    assert mma.accum is True
    assert mma.cta_group == 1
    assert mma.dst.tensor.dtype == n.DType.F32
    assert mma.a.tensor.dtype == n.DType.F16
    assert mma.b.tensor.dtype == n.DType.F16
    assert mma.a.tensor.shape == [128, 16]
    assert mma.b.tensor.shape == [256, 16]


def test_operator_builds_correct_expr_tree():
    # i*16+4 must nest as ADD( MUL(i,16), 4 ) — precedence + structure.
    i = n.Var()
    e = i * 16 + 4
    assert e.op == n.ScalarOp.ADD
    assert e.args[0].op == n.ScalarOp.MUL  # the (i*16) sub-tree is the first arg
    assert e.args[0].args[1] == 16
    assert e.args[1] == 4
    # reflected operator: 3 + i -> ADD(3, i)
    r = 3 + i
    assert r.op == n.ScalarOp.ADD
    assert r.args[0] == 3


def test_loop_constructor_roundtrips_fields():
    i = n.Var()
    loop = n.ForLoop(var=i, start=2, stop=20, step=4, body=())
    assert loop.start == 2
    assert loop.stop == 20
    assert loop.step == 4
    assert loop.var.id == i.id  # same var (identity by id)


def test_kernel_body_preserves_statements_in_order():
    i = n.Var()
    j = n.Var()
    body = (
        n.ForLoop(var=i, start=0, stop=5, step=1, body=()),
        n.ForLoop(var=j, start=0, stop=10, step=1, body=()),
    )
    k = n.Kernel(name="t", args=(), body=body, num_warps=4, launch_shape=[2], cluster_shape=[2])
    assert len(k.body) == 2
    assert k.body[0].stop == 5
    assert k.body[1].stop == 10


def test_nested_body_roundtrips():
    # a role wrapping a body — confirm the nesting is preserved.
    inner = n.CtaSync()
    role = n.Role(body=(inner,), warp=0)
    assert len(role.body) == 1


def test_scheduler_abstraction_roundtrips_fields():
    space = n.TaskSpace(grid=(2, 3), fields=("m_idx", "n_idx"))
    sched = n.Scheduler(space=space, policy="grid_stride")
    task = n.Var(binding=n.VarBinding.TASK)
    stmt = n.ForEachTask(scheduler=sched, var=task, body=())

    assert sched.space.id == space.id
    assert sched.policy == "grid_stride"
    assert sched.scope == "cluster"
    assert stmt.scheduler.id == sched.id
    assert stmt.var.id == task.id
    assert stmt.body == []


def test_scheduler_impl_loop_break_roundtrips():
    space = n.TaskSpace(grid=(4,), fields=("task",))
    sched = n.Scheduler(space=space, policy="custom")
    task = n.Var(binding=n.VarBinding.TASK)
    next_stmt = n.SchedNext(scheduler=sched, var=task)
    break_stmt = n.BreakIf(task < 0)
    loop = n.Loop(body=(next_stmt, break_stmt))
    impl = n.SchedulerImpl(scheduler=sched, body=(loop,))

    assert next_stmt.scheduler.id == sched.id
    assert next_stmt.var.id == task.id
    assert break_stmt.cond.op == n.ScalarOp.LT
    assert len(loop.body) == 2
    assert impl.scheduler.id == sched.id
    assert len(impl.body) == 1


def test_store_scalar_roundtrips_destination():
    dst = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.I32, shape=[2], byte_offset=0)
    task = n.Var(binding=n.VarBinding.TASK)
    stmt = n.StoreScalar(dst=dst[0], value=task)

    assert stmt.dst.tensor.id == dst.id
    assert stmt.dst.offsets[0] == 0
