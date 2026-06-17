import nymph_rs as nr
import pytest
from helpers import builder, expect_runtime_error, gmem_arg, reg_tensor, run, smem_tensor, u32


def test_tma_and_reg_runtime_failures_expose_runtime_codes():
    b = builder("tma_store_missing_source", smem_size_bytes=16)
    out = gmem_arg(b, shape=(4,))
    smem = smem_tensor(b, shape=(4,), byte_offset=0)
    with b.role(warp=0, elected=True):
        b.tma_store(out, smem, coords=(0,), shape=(4,))
    with expect_runtime_error("missing_tensor_value"):
        run(b.build())

    b = builder("tma_store_divergent_coords", smem_size_bytes=16)
    out = gmem_arg(b, shape=(4,))
    smem = smem_tensor(b, shape=(4,), byte_offset=0)
    with b.role(warp=0):
        b.tma_store(out, smem, coords=(b.lane_id(),), shape=(4,))
    with expect_runtime_error("divergent_tma_operands"):
        run(b.build())

    b = builder("reg_load_oob_source")
    short = gmem_arg(b, shape=(8,))
    reg = reg_tensor(b)
    with b.role(warp=0):
        b.reg_load(reg[0], short[b.lane_id()])
    with expect_runtime_error("tensor_value"):
        run(b.build(), {short: u32([0] * 8)})

    b = builder("reg_store_missing_source")
    out = gmem_arg(b, shape=(4,))
    reg = reg_tensor(b)
    with b.role(warp=0, elected=True):
        b.reg_store(out[0], reg[0])
    with expect_runtime_error("missing_tensor_value"):
        run(b.build())


def test_runtime_failure_message_includes_execution_anchor():
    b = builder("reg_load_anchor")
    source = gmem_arg(b, shape=(1,))
    reg = reg_tensor(b)
    with b.role(warp=0, elected=True):
        idx = b.scalar(initial=1, dtype=nr.ScalarDType.I32)
        b.reg_load(reg[0], source[idx])

    with pytest.raises(RuntimeError) as exc_info:
        run(b.build(), {source: u32([0])})

    message = str(exc_info.value)
    assert 'reason=Some("tensor_value")' in message
    assert "diagnostics:" in message
    assert "[tensor_value]" in message
    assert "stmt_id=" in message
    assert "stream_id=" in message
    assert "thread=cta0:warp0:lane0" in message
    assert "stmt_kind=RegLoad" in message
    assert "cohort_size=1" in message


def test_mbarrier_wait_rejects_divergent_phase():
    b = builder("mbarrier_wait_divergent_phase")
    mbar = b.mbar(kind=nr.MBarKind.THREAD)
    with b.kernel_init(warp=0, elected=True):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0):
        b.mbarrier_wait(mbar, phase=b.lane_id() % 2)

    with expect_runtime_error("divergent_mbarrier_operands"):
        run(b.build())


def test_tcgen05_commit_runtime_failures_are_closed():
    b = builder("tcgen05_commit_missing_mbar")
    missing = b.mbar(kind=nr.MBarKind.TCGEN05)
    with b.role(warp=0, elected=True):
        b.tcgen05_commit(missing)
    with expect_runtime_error("uninitialized_mbarrier"):
        run(b.build())

    b = builder("tcgen05_commit_overflow")
    overflow = b.mbar(kind=nr.MBarKind.TCGEN05)
    with b.kernel_init(warp=0):
        b.mbarrier_init(overflow, count=1)
    with b.role(warp=0, elected=True):
        b.mbarrier_expect_tx(overflow, bytes=1)
        b.mbarrier_arrive(overflow)
        b.tcgen05_commit(overflow)
    with expect_runtime_error("mbarrier_arrive_overflow"):
        run(b.build())

    b = builder("tcgen05_commit_peer_exited", launch_shape=(2,), cluster_shape=(2,))
    peer_exited = b.mbar(kind=nr.MBarKind.TCGEN05)
    with b.kernel_init(warp=0):
        b.mbarrier_init(peer_exited, count=1)
    with b.kernel_finalize(warp=0, elected=True):
        with b.if_(b.ctaid_in_cluster().eq(0)):
            b.fence()
    with b.kernel_finalize(warp=0, elected=True):
        with b.if_(b.ctaid_in_cluster().eq(1)):
            b.tcgen05_commit(peer_exited, cta_group=2)
    with expect_runtime_error("tcgen05_peer_exited"):
        run(b.build())

    b = builder("tcgen05_commit_mask_oob", launch_shape=(2,), cluster_shape=(2,))
    mask_oob = b.mbar(kind=nr.MBarKind.TCGEN05)
    with b.kernel_init(warp=0):
        b.mbarrier_init(mask_oob, count=1)
    with b.role(warp=0, elected=True):
        b.tcgen05_commit(mask_oob, multicast_cta_mask=0b100)
    with expect_runtime_error("tcgen05_multicast_cta_mask_oob"):
        run(b.build())
