import numpy as np
import nymph_rs as nr
from helpers import (
    assert_output_eq,
    builder,
    cta_eq,
    expect_runtime_error,
    gmem_arg,
    reg_tensor,
    run,
    smem_tensor,
    u32,
)


def test_tma_load_rejects_non_multicast_mbar_target_mismatch():
    for cluster_shape, remote_coord, cta_group in [((2,), 1, 1), ((4,), 2, 2)]:
        b = builder(
            "tma_load_mbar_target_mismatch",
            smem_size_bytes=4,
            launch_shape=cluster_shape,
            cluster_shape=cluster_shape,
        )
        dst = smem_tensor(b, dtype=nr.DType.F32, shape=(1,), byte_offset=0)
        src = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
        mbar = b.mbar(kind=nr.MBarKind.TMA)
        with b.role():
            b.tma_load(
                dst,
                src,
                mbar=b.mbar_ref(mbar, remote_coord=remote_coord),
                bytes=4,
                coords=(0,),
                shape=(1,),
                cta_group=cta_group,
            )

        with expect_runtime_error("tma_mbar_cta_group_mismatch"):
            run(b.build())


def test_tma_load_store_value_mode_roundtrips_and_preserves_gmem_cells():
    b = builder("tma_roundtrip", smem_size_bytes=16)
    source = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(8,))
    dump = gmem_arg(b, shape=(8,))
    smem = smem_tensor(b, shape=(4,), byte_offset=0)
    reg = reg_tensor(b, shape=(8,))
    mbar = b.mbar(kind=nr.MBarKind.TMA)

    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)

    with b.role(warp=0, elected=True):
        b.mbarrier_arrive_expect_tx(mbar, bytes=16)
        b.tma_load(smem, source, mbar=mbar, bytes=16, coords=(0,), shape=(4,))
        b.tma_store(out, smem, coords=(2,), shape=(4,))
        b.reg_load(reg, out)
        b.reg_store(dump, reg)

    outputs = run(b.build(), {source: u32([10, 11, 12, 13]), out: u32([0, 1, 2, 3, 4, 5, 6, 7])})
    assert_output_eq(outputs, dump, [0, 1, 10, 11, 12, 13, 6, 7], dtype=np.uint32)


def test_tma_value_mode_uses_explicit_full_rank_gmem_shape():
    b = builder("tma_rank_projected_roundtrip", smem_size_bytes=64)
    source = gmem_arg(b, shape=(1, 3, 2, 4))
    out = gmem_arg(b, shape=(1, 3, 2, 4))
    dump = gmem_arg(b, shape=(1, 2, 2, 4))
    smem = smem_tensor(b, shape=(4, 4), byte_offset=0)
    reg = reg_tensor(b, shape=(1, 2, 2, 4))
    mbar = b.mbar(kind=nr.MBarKind.TMA)

    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)

    with b.role(warp=0, elected=True):
        b.mbarrier_arrive_expect_tx(mbar, bytes=64)
        b.tma_load(
            smem,
            source,
            mbar=mbar,
            bytes=64,
            coords=(0, 1, 0, 0),
            shape=(4, 4),
            gmem_shape=(1, 2, 2, 4),
        )
        b.tma_store(out, smem, coords=(0, 0, 0, 0), shape=(4, 4), gmem_shape=(1, 2, 2, 4))
        b.reg_load(reg, out[0:1, 0:2, 0:2, 0:4])
        b.reg_store(dump, reg)

    source_values = np.arange(24, dtype=np.uint32).reshape(1, 3, 2, 4)
    out_values = np.full((1, 3, 2, 4), 1000, dtype=np.uint32)
    expected = source_values[0:1, 1:3, 0:2, :]

    outputs = run(b.build(), {source: source_values, out: out_values})
    assert_output_eq(outputs, dump, expected, dtype=np.uint32)


def test_tma_multicast_writes_each_cta_smem():
    b = builder(
        "tma_multicast_cta_group2", smem_size_bytes=16, launch_shape=(2,), cluster_shape=(2,)
    )
    source = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(2,))
    smem = smem_tensor(b, shape=(4,), byte_offset=0)
    reg = reg_tensor(b)
    mbar = b.mbar(kind=nr.MBarKind.TMA)
    even_mbar = b.mbar_ref(mbar, remote_coord=0)

    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)

    with b.role(warp=0, elected=True):
        with b.if_(cta_eq(b, 0)):
            b.mbarrier_arrive_expect_tx(mbar, bytes=16)
        with b.if_(cta_eq(b, 0)):
            b.tma_load(
                smem,
                source,
                mbar=even_mbar,
                bytes=16,
                coords=(0,),
                shape=(4,),
                multicast_cta_mask=0b11,
                cta_group=2,
            )

    with b.kernel_finalize(warp=0, elected=True):
        b.reg_load(reg[0], smem[0])
        b.reg_store(out[b.ctaid_in_cluster()], reg[0])

    outputs = run(b.build(), {source: u32([20, 21, 22, 23])})
    assert_output_eq(outputs, out, [20, 20], dtype=np.uint32)


def _tma_region_boxes(report, kind, access_kind):
    return [
        e["region"]["boxes"]
        for e in report["events"]
        if e["kind"] == kind and e.get("access_kind") == access_kind
    ]


def _box_ranges(boxes):
    return [tuple(rng) for box in boxes for rng in box["ranges"]]


def test_tma_load_gmem_region_is_per_row_rectangle():
    # A (2,4) tile of a (4,16) u8 tensor at coords (1,3): the footprint is two
    # row runs, NOT one linear interval (which would miss row 2 entirely and
    # cover the gap columns of row 1).
    b = builder("tma_region_rect", smem_size_bytes=64)
    source = gmem_arg(b, dtype=nr.DType.U8, shape=(4, 16))
    smem = smem_tensor(b, dtype=nr.DType.U8, shape=(2, 4), byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.TMA)
    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0):
        b.mbarrier_arrive_expect_tx(mbar, bytes=8)
        b.tma_load(smem, source, mbar=mbar, bytes=8, coords=(1, 3), shape=(2, 4))
        b.mbarrier_wait(mbar, phase=0)
    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    (boxes,) = _tma_region_boxes(report, "read", "tma_load")
    assert _box_ranges(boxes) == [(19, 23), (35, 39)]


def test_tma_load_gmem_region_clamps_partial_tile():
    # coords (3,12) shape (2,8) on (4,16): one in-bounds row, inner clamped to
    # 4 columns — the region must stop at the tensor edge, not run into the
    # next row's bytes.
    b = builder("tma_region_clamp", smem_size_bytes=64)
    source = gmem_arg(b, dtype=nr.DType.U8, shape=(4, 16))
    smem = smem_tensor(b, dtype=nr.DType.U8, shape=(2, 8), byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.TMA)
    with b.kernel_init(warp=0):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0):
        b.mbarrier_arrive_expect_tx(mbar, bytes=16)
        b.tma_load(smem, source, mbar=mbar, bytes=16, coords=(3, 12), shape=(2, 8))
        b.mbarrier_wait(mbar, phase=0)
    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    (boxes,) = _tma_region_boxes(report, "read", "tma_load")
    assert _box_ranges(boxes) == [(60, 64)]


def test_tma_store_gmem_region_is_per_row_rectangle_clamped():
    # Store (2,8) at coords (2,12) on (4,16): two rows, inner clamped to 4.
    b = builder("tma_store_region_rect", smem_size_bytes=64)
    dest = gmem_arg(b, dtype=nr.DType.U8, shape=(4, 16))
    smem = smem_tensor(b, dtype=nr.DType.U8, shape=(2, 8), byte_offset=0)
    with b.role(warp=0):
        with b.if_(b.lane_id().eq(0)):
            for r in range(2):
                for c in range(8):
                    b.store_scalar(nr.TensorSlice(tensor=smem, offsets=(r, c), shape=(1, 1)), 0)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tma_store(dest, smem, coords=(2, 12), shape=(2, 8))
    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    (boxes,) = _tma_region_boxes(report, "write", "tma_store")
    assert _box_ranges(boxes) == [(44, 48), (60, 64)]


def _tma_load_fake_release_kernel(wait_tma):
    """The producer issues a TMA load and signals a consumer via an UNRELATED
    thread barrier without anyone ever waiting the load's own mbarrier. On
    silicon the bulk copy is still in flight — the consumer's read races it."""
    b = builder("tma_fake_release" + ("_wait" if wait_tma else ""), smem_size_bytes=64, num_warps=8)
    source = gmem_arg(b, dtype=nr.DType.F32, shape=(4, 4))
    smem = smem_tensor(b, dtype=nr.DType.F32, shape=(4, 4), byte_offset=0)
    frag = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    tma_mbar = b.mbar(kind=nr.MBarKind.TMA)
    ready = b.mbar(kind=nr.MBarKind.THREAD)
    with b.kernel_init(warp=0):
        b.mbarrier_init(tma_mbar, count=1)
        b.mbarrier_init(ready, count=1)
    with b.role(warp=0):
        with b.if_(b.lane_id().eq(0)):
            b.mbarrier_arrive_expect_tx(tma_mbar, bytes=64)
            b.tma_load(smem, source, mbar=tma_mbar, bytes=64, coords=(0, 0), shape=(4, 4))
            if wait_tma:
                b.mbarrier_wait(tma_mbar, phase=0)
            b.mbarrier_arrive(ready)
    with b.role(warp=4):
        b.mbarrier_wait(ready, phase=0)
        b.reg_load(frag, smem[b.lane_id() % 4, 0:4])
    return b.build()


def test_tma_load_consumer_must_wait_the_loads_mbarrier():
    report = nr.check_protocol(_tma_load_fake_release_kernel(wait_tma=False))
    assert report["status"] == "Failed"
    assert any(d["code"] == "tma_load_access_before_mbar_wait" for d in report["diagnostics"])

    waited = nr.check_protocol(_tma_load_fake_release_kernel(wait_tma=True))
    assert waited["status"] == "Passed", waited["diagnostics"][:2]


def _tma_store_source_reuse_kernel(drain):
    """Overwriting a bulk store's SMEM source is only sound after the store is
    committed into a group and a wait_group retires it — an uncommitted store
    can never be drained."""
    b = builder("tma_store_reuse" + ("_drain" if drain else ""), smem_size_bytes=16)
    dest = gmem_arg(b, dtype=nr.DType.F32, shape=(2, 4))
    smem = smem_tensor(b, dtype=nr.DType.F32, shape=(1, 4), byte_offset=0)
    with b.role(warp=0):
        with b.if_(b.lane_id().eq(0)):
            for c in range(4):
                b.store_scalar(nr.TensorSlice(tensor=smem, offsets=(0, c), shape=(1, 1)), 0)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tma_store(dest, smem, coords=(0, 0), shape=(1, 4))
        if drain:
            b.cp_async_bulk_commit_group()
            b.cp_async_bulk_wait_group_read(0)
        with b.if_(b.lane_id().eq(0)):
            b.store_scalar(nr.TensorSlice(tensor=smem, offsets=(0, 0), shape=(1, 1)), 1)
    return b.build()


def test_tma_store_source_reuse_requires_group_drain():
    report = nr.check_protocol(_tma_store_source_reuse_kernel(drain=False))
    assert report["status"] == "Failed"
    assert any(d["code"] == "async_group_source_overwrite" for d in report["diagnostics"])

    drained = nr.check_protocol(_tma_store_source_reuse_kernel(drain=True))
    assert drained["status"] == "Passed", drained["diagnostics"][:2]
