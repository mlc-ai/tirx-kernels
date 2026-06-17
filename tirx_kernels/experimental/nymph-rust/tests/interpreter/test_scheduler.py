import numpy as np
import nymph_rs as nr
from helpers import assert_output_eq, builder, gmem_arg, reg_tensor, run, smem_tensor, u32


def test_for_each_task_cluster_grid_stride_drives_value_flow():
    b = builder("for_each_task_cluster_grid_stride", launch_shape=(2,), cluster_shape=(2,))
    source = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(4,))
    reg = reg_tensor(b)
    sched = b.scheduler(b.task_space(grid=(4,), fields=("task",)))

    with b.role(warp=0, elected=True):
        with b.for_each_task(sched) as task:
            b.reg_load(reg[0], source[task.task_id])
            b.reg_store(out[task.task_id], reg[0])

    outputs = run(b.build(), {source: u32([10, 11, 12, 13])})
    assert_output_eq(outputs, out, [10, 11, 12, 13], dtype=np.uint32)


def test_for_each_task_grid_stride_shares_canonical_stream_across_roles():
    b = builder("for_each_task_grid_stride", launch_shape=(2,), cluster_shape=(1,))
    source = gmem_arg(b, shape=(4,))
    out0 = gmem_arg(b, shape=(4,))
    out1 = gmem_arg(b, shape=(4,))
    reg0 = reg_tensor(b)
    reg1 = reg_tensor(b)
    sched = b.scheduler(b.task_space(grid=(4,), fields=("task",)))

    with b.role(warp=0, elected=True):
        with b.for_each_task(sched) as task:
            b.reg_load(reg0[0], source[task.task_id])
            b.reg_store(out0[task.task_id], reg0[0])
    with b.role(warp=1, elected=True):
        with b.for_each_task(sched) as task:
            b.reg_load(reg1[0], source[task.task_id])
            b.reg_store(out1[task.task_id], reg1[0])

    outputs = run(b.build(), {source: u32([20, 21, 22, 23])})
    assert_output_eq(outputs, out0, [20, 21, 22, 23], dtype=np.uint32)
    assert_output_eq(outputs, out1, [20, 21, 22, 23], dtype=np.uint32)


def test_sched_next_loop_break_runs_canonical_dynamic_scheduler():
    b = builder("sched_next_loop_break", launch_shape=(2,), cluster_shape=(1,))
    source = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(4,))
    reg = reg_tensor(b)
    sched = b.scheduler(b.task_space(grid=(4,), fields=("task",)), policy="custom")

    with b.scheduler_impl(sched):
        with b.role(warp=0, elected=True):
            with b.loop():
                task = b.sched_next(sched)
                b.break_if(task.task_id < 0)
                b.reg_load(reg[0], source[task.task_id])
                b.reg_store(out[task.task_id], reg[0])

    outputs = run(b.build(), {source: u32([30, 31, 32, 33])})
    assert_output_eq(outputs, out, [30, 31, 32, 33], dtype=np.uint32)


def test_clc_shaped_scheduler_broadcast_consumer_pipeline_completes():
    b = builder("clc_shaped_scheduler_broadcast", smem_size_bytes=8)
    source = gmem_arg(b, shape=(2,))
    out = gmem_arg(b, shape=(2,))
    task_smem = smem_tensor(b, dtype=nr.DType.I32, shape=(2,), byte_offset=0)
    data_reg = reg_tensor(b)
    full = b.mbar(kind=nr.MBarKind.THREAD, stages=2)
    empty = b.mbar(kind=nr.MBarKind.THREAD, stages=2)
    sched = b.scheduler(b.task_space(grid=(2,), fields=("task",)), policy="custom")

    def add_one(var):
        return var + 1

    def stage_of(var):
        return var % 2

    def phase_of(var):
        return (var // 2) % 2

    with b.kernel_init(warp=0):
        b.mbarrier_init(full, count=1, stage=0)
        b.mbarrier_init(full, count=1, stage=1)
        b.mbarrier_init(empty, count=1, stage=0)
        b.mbarrier_init(empty, count=1, stage=1)
        b.mbarrier_arrive(empty, stage=0)
        b.mbarrier_arrive(empty, stage=1)

    with b.role(warp=0, elected=True):
        sched_iter = b.scalar(initial=0, dtype=nr.ScalarDType.I32)
        with b.scheduler_impl(sched):
            with b.loop():
                b.mbarrier_wait(empty, stage=stage_of(sched_iter), phase=phase_of(sched_iter))
                task = b.sched_next(sched)
                b.store_scalar(task_smem[stage_of(sched_iter)], task.task_id)
                b.mbarrier_arrive(full, stage=stage_of(sched_iter))
                b.scalar_store(sched_iter, add_one(sched_iter))
                b.break_if(task.task_id < 0)

    with b.role(warp=1, elected=True):
        consumer_iter = b.scalar(initial=0, dtype=nr.ScalarDType.I32)
        with b.loop():
            b.mbarrier_wait(full, stage=stage_of(consumer_iter), phase=phase_of(consumer_iter))
            task_read = b.scalar(
                initial=task_smem[stage_of(consumer_iter)], dtype=nr.ScalarDType.I32
            )
            b.mbarrier_arrive(empty, stage=stage_of(consumer_iter))
            b.break_if(task_read < 0)
            b.reg_load(data_reg[0], source[task_read])
            b.reg_store(out[task_read], data_reg[0])
            b.scalar_store(consumer_iter, add_one(consumer_iter))

    outputs = run(b.build(), {source: u32([50, 51])})
    assert_output_eq(outputs, out, [50, 51], dtype=np.uint32)
