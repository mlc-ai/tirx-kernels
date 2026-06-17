import numpy as np
import nymph_rs as nr
from helpers import assert_output_eq, builder, gmem_arg, reg_tensor, run, u32


def test_runs_a_simple_ordered_kernel():
    b = builder("simple_ordered")
    with b.role():
        s = b.scalar(initial=0, dtype=nr.ScalarDType.I32)
        b.scalar_store(s, 5)

    run(b.build())


def test_runs_a_for_loop():
    b = builder("for_loop")
    with b.role(warp=0):
        with b.for_loop(stop=8):
            b.warp_sync()

    run(b.build())


def test_fence_and_cp_async_are_value_noops():
    b = builder("fence_cp_async_noop")
    source = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(4,))
    reg = reg_tensor(b, shape=(4,))

    with b.role(warpgroup=0, elected=True):
        b.cp_async_bulk_commit_group()
        b.cp_async_bulk_commit_group()
        b.cp_async_bulk_wait_group_read()
        b.fence()
        b.reg_load(reg, source)
        b.reg_store(out, reg)

    outputs = run(b.build(), {source: u32([1, 2, 3, 4])})
    assert_output_eq(outputs, out, [1, 2, 3, 4], dtype=np.uint32)
