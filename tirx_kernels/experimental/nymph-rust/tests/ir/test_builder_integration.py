"""Integration test: the Python IRBuilder and the fp16/bf16 GEMM kernel — now
shipped inside the `nymph_rs` package — build a Rust IR `Kernel`.

The builder and kernel live under python/nymph_rs/; their IR imports resolve to
the compiled Rust extension, so building exercises the real PyO3 bindings end to
end and the resulting Kernel is validated by the Rust validator.
"""

import pytest

nymph_rs = pytest.importorskip("nymph_rs")


def test_real_gemm_builds_rust_ir_and_validates():
    from nymph_rs.kernels import build_fp16_bf16_gemm

    k = build_fp16_bf16_gemm()
    assert type(k) is nymph_rs.Kernel  # the Rust Kernel, not a Python dataclass
    assert k.name == "nymph_fp16_bf16_gemm"
    assert k.num_warps > 0
    assert k.launch_cta_count() >= 1
    k.validate()  # the real kernel passes the Rust validator


def test_builder_is_exposed():
    assert hasattr(nymph_rs, "IRBuilder")
    b = nymph_rs.IRBuilder("k", num_warps=4, launch_shape=(2,), cluster_shape=(2,))
    b.cta_sync()
    k = b.build()
    assert type(k) is nymph_rs.Kernel
    assert k.num_warps == 4


def test_builder_for_each_task_emits_scheduler_ir_and_fields():
    b = nymph_rs.IRBuilder("sched", num_warps=4, launch_shape=(2,), cluster_shape=(1,))
    space = b.task_space(grid=(2, 3), fields=("m_idx", "n_idx"))
    sched = b.scheduler(space)
    with b.for_each_task(sched) as task:
        m_idx = task.m_idx
        n_idx = task.n_idx
        b.cta_sync()

    assert m_idx.op == nymph_rs.ScalarOp.MOD
    assert m_idx.args[1] == 2
    assert n_idx.op == nymph_rs.ScalarOp.MOD
    assert n_idx.args[0].op == nymph_rs.ScalarOp.FLOORDIV
    assert n_idx.args[0].args[1] == 2
    assert n_idx.args[1] == 3

    k = b.build()
    assert k.body[0].scheduler.id == sched.id
    assert len(k.body[0].body) == 1


def test_builder_scheduler_impl_loop_sched_next_builds():
    b = nymph_rs.IRBuilder("dyn_sched", num_warps=4, smem_size_bytes=4)
    space = b.task_space(grid=(4,), fields=("task",))
    sched = b.scheduler(space, policy="custom")
    task_smem = b.tensor(
        space=nymph_rs.MemorySpace.SMEM, dtype=nymph_rs.DType.I32, shape=(1,), byte_offset=0
    )
    with b.scheduler_impl(sched):
        with b.loop():
            task = b.sched_next(sched)
            valid = task.valid
            b.store_scalar(task_smem[0], task.task_id)
            b.break_if(~valid)

    assert valid.op == nymph_rs.ScalarOp.GE
    k = b.build()
    impl = k.body[1]
    assert impl.scheduler.id == sched.id
    assert len(impl.body[0].body) == 3
