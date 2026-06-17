import nymph_rs as nr
from helpers import (
    builder,
    expect_runtime_error,
    gmem_arg,
    reg_tensor,
    smem_tensor,
    tmem_tensor,
    u32,
)
from nymph_rs.kernels import build_fp16_bf16_gemm
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig


def _warning_codes(report):
    return {warning["code"] for warning in report["warnings"]}


def _diagnostic_codes(report):
    return {diagnostic["code"] for diagnostic in report["diagnostics"]}


def _event_kinds(report):
    return {event["kind"] for event in report["events"]}


def _events(report, kind):
    return [event for event in report["events"] if event["kind"] == kind]


def _assert_anchored_events(report):
    for event in report["events"]:
        assert isinstance(event["stmt_id"], int)
        assert event["stmt_kind"]


def _pass_status(report, name):
    for item in report["pass_summary"]:
        if item["name"] == name:
            return item["status"]
    raise AssertionError(f"missing pass summary for {name}")


def test_protocol_trace_scheduler_scalar_bridge_passes():
    b = builder("protocol_scheduler_scalar_bridge", smem_size_bytes=8)
    task_smem = smem_tensor(b, dtype=nr.DType.I32, shape=(2,), byte_offset=0)
    full = b.mbar(kind=nr.MBarKind.THREAD, stages=2)
    empty = b.mbar(kind=nr.MBarKind.THREAD, stages=2)
    sched = b.scheduler(b.task_space(grid=(2,), fields=("task",)), policy="custom")

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
                b.scalar_store(sched_iter, sched_iter + 1)
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
            b.scalar_store(consumer_iter, consumer_iter + 1)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    assert _pass_status(report, "trace_schema_audit") == "Passed"
    _assert_anchored_events(report)
    assert {"scheduler_next", "write", "read"} <= _event_kinds(report)


def test_protocol_default_omits_events_but_reports_pass_summary():
    b = builder("protocol_default_omits_events")
    with b.role(warp=0, elected=True):
        b.scalar(initial=0, dtype=nr.ScalarDType.I32)

    report = nr.check_protocol(b.build())
    assert report["status"] == "Passed"
    assert "events" not in report
    assert _pass_status(report, "trace_schema_audit") == "Passed"


def test_protocol_deadlock_freedom_accepts_cta_sync():
    b = builder("protocol_deadlock_cta_sync")
    b.cta_sync()

    report = nr.check_protocol(b.build())
    assert report["status"] == "Passed"
    assert _pass_status(report, "deadlock_freedom") == "Passed"


def test_protocol_deadlock_freedom_rejects_wait_group_without_commit():
    b = builder("protocol_deadlock_wait_group_missing_commit")
    with b.role(warp=0, elected=True):
        b.cp_async_bulk_wait_group_read()

    report = nr.check_protocol(b.build())
    assert report["status"] == "Failed"
    assert _pass_status(report, "deadlock_freedom") == "Failed"
    assert "deadlock_freedom_missing_release_witness" in _diagnostic_codes(report)


def test_protocol_deadlock_freedom_accepts_mixed_supported_blockers():
    b = builder("protocol_deadlock_mixed_supported")
    mbar = b.mbar(kind=nr.MBarKind.THREAD)

    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role():
        b.mbarrier_arrive(mbar)
        b.mbarrier_wait(mbar, phase=0)
        b.cp_async_bulk_commit_group()
        b.cp_async_bulk_wait_group_read()
    b.cta_sync()

    report = nr.check_protocol(b.build())
    assert report["status"] == "Passed"
    assert _pass_status(report, "deadlock_freedom") == "Passed"


def test_protocol_wait_group_read_n_retains_latest_async_source():
    b = builder("protocol_wait_group_read_n_retains_source", smem_size_bytes=4)
    source = smem_tensor(b, shape=(1,), byte_offset=0)
    out = gmem_arg(b, shape=(1,))

    with b.role(warp=0, elected=True):
        b.store_scalar(source[0], 1)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tma_store(out, source, coords=(0,), shape=(1,))
        b.cp_async_bulk_commit_group()
        b.tma_store(out, source, coords=(0,), shape=(1,))
        b.cp_async_bulk_commit_group()
        b.cp_async_bulk_wait_group_read(1)
        b.store_scalar(source[0], 2)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Failed"
    assert _pass_status(report, "async_group_lifetime") == "Failed"
    assert "async_group_source_overwrite" in _diagnostic_codes(report)
    assert _events(report, "wait_group")[0]["n"] == 1


def test_protocol_wait_group_read_zero_drains_retained_async_source():
    b = builder("protocol_wait_group_read_zero_drains_source", smem_size_bytes=4)
    source = smem_tensor(b, shape=(1,), byte_offset=0)
    out = gmem_arg(b, shape=(1,))

    with b.role(warp=0, elected=True):
        b.store_scalar(source[0], 1)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tma_store(out, source, coords=(0,), shape=(1,))
        b.cp_async_bulk_commit_group()
        b.tma_store(out, source, coords=(0,), shape=(1,))
        b.cp_async_bulk_commit_group()
        b.cp_async_bulk_wait_group_read(1)
        b.cp_async_bulk_wait_group_read(0)
        b.store_scalar(source[0], 2)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    assert _pass_status(report, "async_group_lifetime") == "Passed"
    assert [event["n"] for event in _events(report, "wait_group")] == [1, 0]


def test_protocol_deadlock_returns_failed_report():
    b = builder("protocol_deadlock")
    mbar = b.mbar(kind=nr.MBarKind.TMA)
    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0, elected=True):
        b.mbarrier_expect_tx(mbar, bytes=8)
        b.mbarrier_arrive(mbar)
        b.mbarrier_wait(mbar)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Failed"
    assert "deadlock" in _diagnostic_codes(report)
    assert report["events"]
    assert report["pass_summary"] == []


def test_protocol_blocked_mbar_wait_emits_completion_event():
    b = builder("protocol_blocked_mbar_wait_event", smem_size_bytes=16)
    source = gmem_arg(b, shape=(4,))
    smem = smem_tensor(b, shape=(4,), byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.TMA)

    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0, elected=True):
        b.mbarrier_arrive_expect_tx(mbar, bytes=16)
        b.mbarrier_wait(mbar, phase=0)
    with b.role(warp=1, elected=True):
        b.scalar(initial=0, dtype=nr.ScalarDType.I32)
        b.tma_load(smem, source, mbar=mbar, bytes=16, coords=(0,), shape=(4,))

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    _assert_anchored_events(report)
    kinds = [event["kind"] for event in report["events"]]
    assert "mbar_wait" in kinds
    assert kinds.index("mbar_complete_tx") < kinds.index("mbar_wait")
    wait = _events(report, "mbar_wait")[0]
    assert wait["phase"] == 0
    assert wait["target"]["mbar_id"] == mbar.id
    assert wait["stmt_kind"] == "MBarrierWait"


def test_protocol_payload_control_bridge_is_inconclusive():
    b = builder("protocol_payload_bridge", smem_size_bytes=4)
    source = gmem_arg(b, shape=(1,))
    smem = smem_tensor(b, shape=(1,), byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.TMA)
    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0, elected=True):
        b.mbarrier_arrive_expect_tx(mbar, bytes=4)
        b.tma_load(smem, source, mbar=mbar, bytes=4, coords=(0,), shape=(1,))
        b.scalar(initial=smem[0], dtype=nr.ScalarDType.U32)

    report = nr.check_protocol(b.build())
    assert report["status"] == "Inconclusive"
    assert "trace_control_from_skipped_payload" in _warning_codes(report)


def test_protocol_skipped_bulk_write_invalidates_prior_scalar_cell():
    b = builder("protocol_payload_invalidates_scalar", smem_size_bytes=4)
    source = gmem_arg(b, shape=(1,))
    smem = smem_tensor(b, shape=(1,), byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.TMA)
    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0, elected=True):
        b.store_scalar(smem[0], 7)
        b.mbarrier_arrive_expect_tx(mbar, bytes=4)
        b.tma_load(smem, source, mbar=mbar, bytes=4, coords=(0,), shape=(1,))
        b.scalar(initial=smem[0], dtype=nr.ScalarDType.U32)

    report = nr.check_protocol(b.build())
    assert report["status"] == "Inconclusive"
    assert "trace_control_from_skipped_payload" in _warning_codes(report)


def test_protocol_gemm_trace_runs_without_payload_inputs():
    cfg = Fp16Bf16GemmConfig(k=16, block_k=16, launch_shape=(2,))
    kernel = build_fp16_bf16_gemm(cfg)

    report = nr.check_protocol(kernel, include_events=True)
    assert report["status"] == "Passed"
    assert _pass_status(report, "deadlock_freedom") == "Passed"
    _assert_anchored_events(report)
    kinds = _event_kinds(report)
    assert {"read", "write", "tmem_alloc"} <= kinds
    assert any(
        event["proxy"] == "async"
        and event["access_category"] == "tensor"
        and event["region"]["owner"]["kind"] == "smem"
        for event in _events(report, "read")
    )
    assert any(
        event["access_category"] == "tmem" and event["access_kind"] == "mma"
        for event in _events(report, "write")
    )
    assert any(event["async_kind"] == "ld" for event in _events(report, "tmem_wait"))


def test_protocol_tmem_mma_layout_f_emits_union_boxes():
    m, n, k = 64, 16, 16
    a_bytes = m * k * 2
    b_bytes = n * k * 2
    b = builder("protocol_tmem_layout_f_boxes", smem_size_bytes=a_bytes + b_bytes)
    a_s = smem_tensor(b, dtype=nr.DType.F16, shape=(m, k), byte_offset=0)
    b_s = smem_tensor(b, dtype=nr.DType.F16, shape=(n, k), byte_offset=a_bytes)
    dst = tmem_tensor(b, dtype=nr.DType.F32, shape=(m, n), col_start=0, lane_align=0)

    with b.kernel_init(warp=0):
        b.tmem_alloc(dst, n_cols=32)
    with b.role(warpgroup=0):
        b.tcgen05_mma(dst, a_s, b_s, m=m, n=n, k=k, accum=False, cta_group=1)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(dst, n_cols=32)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    assert _pass_status(report, "memory_race_check") == "Passed"
    mma_writes = [
        event
        for event in _events(report, "write")
        if event["access_category"] == "tmem" and event["access_kind"] == "mma"
    ]
    assert len(mma_writes) == 1
    region = mma_writes[0]["region"]
    assert "lane_start" not in region
    assert region["owner"] == {"kind": "tmem", "cta_id": 0}
    assert region["boxes"] == [
        {"ranges": [(0, 16), (0, 64)]},
        {"ranges": [(32, 48), (0, 64)]},
        {"ranges": [(64, 80), (0, 64)]},
        {"ranges": [(96, 112), (0, 64)]},
    ]


def test_protocol_tmem_async_overlap_fails_before_wait():
    b = builder("protocol_tmem_async_overlap")
    tmem = tmem_tensor(b, dtype=nr.DType.F32, shape=(128, 32), col_start=0)
    reg = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))

    with b.kernel_init(warp=0):
        b.tmem_alloc(tmem, n_cols=32)
    with b.role(warp=0):
        b.tcgen05_st(tmem, reg, shape="32x32b", num=1, row=0, col=0)
        b.tcgen05_st(tmem, reg, shape="32x32b", num=1, row=0, col=0)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(tmem, n_cols=32)

    report = nr.check_protocol(b.build())
    assert report["status"] == "Failed"
    assert "tmem_async_overlap" in _diagnostic_codes(report)


def test_protocol_proxy_fence_missing_fails():
    b = builder("protocol_proxy_fence_missing", smem_size_bytes=4)
    source = smem_tensor(b, shape=(1,), byte_offset=0)
    out = gmem_arg(b, shape=(1,))

    with b.role(warp=0, elected=True):
        b.store_scalar(source[0], 1)
        b.tma_store(out, source, coords=(0,), shape=(1,))

    report = nr.check_protocol(b.build())
    assert report["status"] == "Failed"
    assert _pass_status(report, "proxy_fence") == "Failed"
    assert "proxy_fence_missing" in _diagnostic_codes(report)


def test_protocol_proxy_fence_present_passes():
    b = builder("protocol_proxy_fence_present", smem_size_bytes=4)
    source = smem_tensor(b, shape=(1,), byte_offset=0)
    out = gmem_arg(b, shape=(1,))

    with b.role(warp=0, elected=True):
        b.store_scalar(source[0], 1)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tma_store(out, source, coords=(0,), shape=(1,))

    report = nr.check_protocol(b.build())
    assert report["status"] == "Passed"
    assert _pass_status(report, "proxy_fence") == "Passed"


def test_protocol_trace_emits_proxy_fence_group_and_sync_metadata():
    b = builder("protocol_trace_event_metadata", smem_size_bytes=128 * 4)
    source = smem_tensor(b, shape=(128,), byte_offset=0)
    out = gmem_arg(b, shape=(128,))

    with b.role(warpgroup=0):
        b.store_scalar(source[b.tid_in_wg()], b.tid_in_wg())
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CLUSTER)
        b.fence(kind=nr.FenceKind.MEMORY, scope=nr.FenceScope.GPU)
        b.cp_async_bulk_commit_group()
        b.cp_async_bulk_wait_group_read()
        b.tma_store(out, source, coords=(0,), shape=(128,))
        b.wg_sync(barrier_id=7)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    assert _pass_status(report, "deadlock_freedom") == "Passed"
    _assert_anchored_events(report)

    fence = _events(report, "fence")[0]
    assert fence["fence_kind"] == "proxy_async"
    assert fence["fence_scope"] == "cta"
    # the cta/cluster/gpu hierarchy threads from the IR FenceScope into the trace
    assert {"cta", "cluster", "gpu"} <= {ev["fence_scope"] for ev in _events(report, "fence")}

    assert _events(report, "commit_group")
    assert _events(report, "wait_group")[0]["n"] == 0

    sync_arrive = _events(report, "sync_arrive")[0]
    assert sync_arrive["sync_kind"] == "warpgroup"
    assert sync_arrive["bar_id"] == 7
    assert sync_arrive["thread_count"] == 128
    assert sync_arrive["count"] == 128
    assert sync_arrive["cycle"] == 0

    sync = _events(report, "sync")[0]
    assert sync["sync_kind"] == "warpgroup"
    assert sync["bar_id"] == 7
    assert sync["thread_count"] == 128
    assert sync["cycle"] == sync_arrive["cycle"]

    assert any(
        event["proxy"] == "async"
        and event["access_category"] == "tensor"
        and event["region"]["owner"]["kind"] == "smem"
        for event in _events(report, "read")
    )


def test_oob_tensor_slice_caught_in_value_and_trace():
    # Runtime out-of-bounds read: `source` has size 2 but the scalar index is 3.
    b = builder("oob_slice")
    source = gmem_arg(b, shape=(2,))
    out = gmem_arg(b, shape=(1,))
    reg = reg_tensor(b, shape=(1,))
    with b.role(warp=0, elected=True):
        idx = b.scalar(initial=3, dtype=nr.ScalarDType.I32)
        b.reg_load(reg[0], source[idx])
        b.reg_store(out[0], reg[0])
    kernel = b.build()
    # value mode: the read bounds-checks and fails closed.
    with expect_runtime_error("tensor_value"):
        nr.interpret(kernel, {source: u32([10, 20])})
    # trace mode skips the byte read but resolves the same slice to record its footprint,
    # so eval_slice catches the out-of-bounds offset there too — not a silent Passed.
    report = nr.check_protocol(kernel)
    assert report["status"] == "Failed"
    assert "tensor_value" in {d["code"] for d in report["diagnostics"]}
