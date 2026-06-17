import numpy as np
import nymph_rs as nr
from helpers import (
    builder,
    expect_runtime_error,
    gmem_arg,
    output,
    reg_tensor,
    run,
    smem_tensor,
    tmem_tensor,
    u32,
)

DP_B = 1 << 21


def _layout_eval(shape, stride, index):
    offset = 0
    for extent, step in zip(shape, stride, strict=True):
        offset += (index % extent) * step
        index //= extent
    return offset


def _decode(bit_address):
    return bit_address >> 21, (bit_address >> 5) & 0xFFFF, bit_address & 31


def _atom(shape, num):
    if shape == "32x32b" and num in [1, 2, 4, 8, 16, 32, 64, 128]:
        return {
            "val_shape": [32 * num, 32],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [32],
            "dst_thr_stride": [32 * num],
            "dst_val_shape": [32 * num],
            "dst_val_stride": [1],
        }
    if shape == "16x32bx2" and num in [1, 2, 4, 8, 16, 32, 64, 128]:
        return {
            "val_shape": [64 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [16, 2],
            "dst_thr_stride": [64 * num, 32 * num],
            "dst_val_shape": [32 * num],
            "dst_val_stride": [1],
        }
    if shape == "16x64b" and num in [1, 2, 4, 8, 16, 32, 64, 128]:
        return {
            "val_shape": [64 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [2, 2, 8],
            "dst_thr_stride": [512 * num, 32, 64 * num],
            "dst_val_shape": [32, num],
            "dst_val_stride": [1, 64],
        }
    if shape == "16x128b" and num in [1, 2, 4, 8, 16, 32, 64]:
        return {
            "val_shape": [128 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [4, 8],
            "dst_thr_stride": [32, 128 * num],
            "dst_val_shape": [32, 2, num],
            "dst_val_stride": [1, 1024 * num, 128],
        }
    if shape == "16x256b" and num in [1, 2, 4, 8, 16, 32]:
        return {
            "val_shape": [256 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [4, 8],
            "dst_thr_stride": [64, 256 * num],
            "dst_val_shape": [64, 2, num],
            "dst_val_stride": [1, 2048 * num, 256],
        }
    raise ValueError((shape, num))


def register_count(shape, num):
    atom = _atom(shape, num)
    product = 1
    for extent in atom["dst_val_shape"]:
        product *= extent
    return product // 32


def datapath_index_arrays(shape, num):
    atom = _atom(shape, num)
    role = 1
    for extent in atom["dst_thr_shape"]:
        role *= extent
    regs = register_count(shape, num)
    lane_idx = np.zeros((role, regs), dtype=np.int64)
    col_idx = np.zeros((role, regs), dtype=np.int64)
    for thr in range(role):
        thr_off = _layout_eval(atom["dst_thr_shape"], atom["dst_thr_stride"], thr)
        for reg in range(regs):
            val_off = _layout_eval(atom["dst_val_shape"], atom["dst_val_stride"], reg * 32)
            lane, col, bit0 = _decode(
                _layout_eval(atom["val_shape"], atom["val_stride"], thr_off + val_off)
            )
            assert bit0 == 0
            lane_idx[thr, reg] = lane
            col_idx[thr, reg] = col
    return lane_idx, col_idx


def supported_non32_shapes():
    for shape in ["16x32bx2", "16x64b"]:
        for num in [1, 2, 4, 8, 16, 32, 64, 128]:
            yield shape, num
    for num in [1, 2, 4, 8, 16, 32, 64]:
        yield "16x128b", num
    for num in [1, 2, 4, 8, 16, 32]:
        yield "16x256b", num


def u32_sentinels(rows, cols):
    values = [((row << 16) | col) for row in range(rows) for col in range(cols)]
    return u32(values, shape=(rows, cols))


def _tcgen05_role_failure_kernel(op, *, row=0, col=0, elected=False):
    b = builder("tcgen05_datapath_failure")
    src = tmem_tensor(b, dtype=nr.DType.F32, shape=(128, 128), col_start=0)
    dst = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    with b.kernel_init(warp=0):
        b.tmem_alloc(src, n_cols=64)
    with b.role(warp=0, elected=elected):
        if op == "ld":
            b.tcgen05_ld(dst, src, row=row, col=col)
        else:
            b.tcgen05_st(src, dst, row=row, col=col)
    return b.build()


def test_tcgen05_ld_st_reject_partial_warp_and_tmem_bounds_fail_closed():
    with expect_runtime_error("tcgen05_ld_mask"):
        run(_tcgen05_role_failure_kernel("ld", elected=True))
    with expect_runtime_error("tcgen05_ld_out_of_range"):
        run(_tcgen05_role_failure_kernel("ld", row=128))
    with expect_runtime_error("tcgen05_st_mask"):
        run(_tcgen05_role_failure_kernel("st", elected=True))
    with expect_runtime_error("tcgen05_st_out_of_range"):
        run(_tcgen05_role_failure_kernel("st", col=512))


def _reg_row_cols(b, tensor, col, cols):
    return tensor[b.tid_in_wg(), col : col + cols]


def _tcgen05_datapath_kernel(shape, num, mode):
    reg_size = register_count(shape, num)
    b = builder(f"tcgen05_{mode}_datapath")
    tmem = tmem_tensor(b, dtype=nr.DType.U32, shape=(128, 256), col_start=0)
    seed_g = gmem_arg(b, shape=(128, 256))
    source_g = gmem_arg(b, shape=(128, reg_size))
    out = gmem_arg(b, shape=(128, 256 if mode == "st" else reg_size))
    seed_reg = reg_tensor(b, shape=(128,))
    source_reg = reg_tensor(b, shape=(reg_size,))
    out_reg = reg_tensor(b, shape=(reg_size,))
    chunk_reg = reg_tensor(b, shape=(128,))

    with b.kernel_init(warp=0):
        b.tmem_alloc(tmem, n_cols=256)

    with b.role(warpgroup=0):
        for col in [0, 128]:
            b.reg_load(seed_reg, _reg_row_cols(b, seed_g, col, 128))
            b.tcgen05_st(tmem, seed_reg, shape="32x32b", num=128, row=0, col=col)

        if mode == "st":
            b.reg_load(source_reg, _reg_row_cols(b, source_g, 0, reg_size))
            b.tcgen05_st(tmem, source_reg, shape=shape, num=num, row=0, col=0)
            for col in [0, 128]:
                b.tcgen05_ld(chunk_reg, tmem, shape="32x32b", num=128, row=0, col=col)
                b.reg_store(_reg_row_cols(b, out, col, 128), chunk_reg)
        else:
            b.tcgen05_ld(out_reg, tmem, shape=shape, num=num, row=0, col=0)
            b.reg_store(_reg_row_cols(b, out, 0, reg_size), out_reg)

    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(tmem, n_cols=256)

    return b.build(), seed_g, source_g, out


def test_tcgen05_st_non32_shapes_scatter_to_modeled_physical_cells():
    for shape, num in supported_non32_shapes():
        kernel, seed_g, source_g, out = _tcgen05_datapath_kernel(shape, num, "st")
        reg_size = register_count(shape, num)
        outputs = run(
            kernel,
            {seed_g: np.zeros((128, 256), dtype=np.uint32), source_g: u32_sentinels(128, reg_size)},
        )
        dump = output(outputs, out)
        lane_idx, col_idx = datapath_index_arrays(shape, num)
        expected = np.zeros((128, 256), dtype=np.uint32)
        for tid in range(128):
            warp = tid // 32
            lane = tid % 32
            for reg in range(reg_size):
                phys_lane = 32 * (warp % 4) + lane_idx[lane, reg]
                phys_col = col_idx[lane, reg]
                expected[phys_lane, phys_col] = (tid << 16) | reg
        np.testing.assert_array_equal(dump, expected, err_msg=f"{shape}.x{num}")


def test_tcgen05_ld_non32_shapes_gather_from_modeled_physical_cells():
    for shape, num in supported_non32_shapes():
        kernel, seed_g, source_g, out = _tcgen05_datapath_kernel(shape, num, "ld")
        reg_size = register_count(shape, num)
        outputs = run(
            kernel,
            {seed_g: u32_sentinels(128, 256), source_g: np.zeros((128, reg_size), dtype=np.uint32)},
        )
        dump = output(outputs, out)
        lane_idx, col_idx = datapath_index_arrays(shape, num)
        expected = np.zeros((128, reg_size), dtype=np.uint32)
        for tid in range(128):
            warp = tid // 32
            lane = tid % 32
            for reg in range(reg_size):
                phys_lane = 32 * (warp % 4) + lane_idx[lane, reg]
                phys_col = col_idx[lane, reg]
                expected[tid, reg] = (phys_lane << 16) | phys_col
        np.testing.assert_array_equal(dump, expected, err_msg=f"{shape}.x{num}")


def _mma64_kernel(dtype, lane_align, accum, trans_a, trans_b):
    m, n, k = 64, 16, 16
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (k, n) if trans_b else (n, k)
    a_bytes = int(np.prod(a_shape)) * 2
    b_bytes = int(np.prod(b_shape)) * 2
    b = builder("mma64", smem_size_bytes=a_bytes + b_bytes)
    a_g = gmem_arg(b, dtype=dtype, shape=a_shape)
    b_g = gmem_arg(b, dtype=dtype, shape=b_shape)
    zero_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128, n))
    out = gmem_arg(b, dtype=nr.DType.F32, shape=(128, n))
    a_s = smem_tensor(b, dtype=dtype, shape=a_shape, byte_offset=0)
    b_s = smem_tensor(b, dtype=dtype, shape=b_shape, byte_offset=a_bytes)
    dst = tmem_tensor(b, dtype=nr.DType.F32, shape=(m, n), col_start=0, lane_align=lane_align)
    frag = reg_tensor(b, dtype=nr.DType.F32, shape=(n,))
    ma = b.mbar(kind=nr.MBarKind.TMA)
    mb = b.mbar(kind=nr.MBarKind.TMA)

    with b.kernel_init(warp=0):
        b.tmem_alloc(dst, n_cols=32)

    with b.role(warpgroup=0):
        b.reg_load(frag, zero_g[b.tid_in_wg(), 0:n])
        b.tcgen05_st(dst, frag, shape="32x32b", num=n, row=0, col=0)
        b.mbarrier_init(ma, count=1)
        b.mbarrier_expect_tx(ma, bytes=a_bytes)
        b.tma_load(a_s, a_g, mbar=ma, bytes=a_bytes, coords=(0, 0), shape=a_shape)
        b.mbarrier_init(mb, count=1)
        b.mbarrier_expect_tx(mb, bytes=b_bytes)
        b.tma_load(b_s, b_g, mbar=mb, bytes=b_bytes, coords=(0, 0), shape=b_shape)
        b.tcgen05_mma(dst, a_s, b_s, m=m, n=n, k=k, accum=False, trans_a=trans_a, trans_b=trans_b)
        if accum:
            b.tcgen05_mma(
                dst, a_s, b_s, m=m, n=n, k=k, accum=True, trans_a=trans_a, trans_b=trans_b
            )
        b.tcgen05_ld(frag, dst, shape="32x32b", num=n, row=0, col=0)
        b.reg_store(out[b.tid_in_wg(), 0:n], frag)

    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(dst, n_cols=32)

    return b.build(), a_g, b_g, zero_g, out


def _mma64_inputs(trans_b):
    a = np.ones((64, 16), dtype=np.float32)
    b = np.zeros((16, 16), dtype=np.float32)
    if trans_b:
        for kk in range(16):
            for ni in range(16):
                b[kk, ni] = ni + 1
    else:
        for ni in range(16):
            for kk in range(16):
                b[ni, kk] = ni + 1
    return a, b


def test_tcgen05_mma_m64_layout_f_transpose_accum_and_dtype():
    for dtype in [nr.DType.F16, nr.DType.BF16]:
        for lane_align in [0, 16]:
            for trans_a, trans_b in [(False, False), (True, False), (False, True), (True, True)]:
                kernel, a_g, b_g, zero_g, out = _mma64_kernel(
                    dtype, lane_align, True, trans_a, trans_b
                )
                a, b_values = _mma64_inputs(trans_b)
                if trans_a:
                    a = a.T
                outputs = run(
                    kernel,
                    {
                        a_g: a.astype(np.float32),
                        b_g: b_values.astype(np.float32),
                        zero_g: np.zeros((128, 16), dtype=np.float32),
                    },
                )
                dump = output(outputs, out)
                for mi in [0, 15, 16, 63]:
                    for ni in [0, 7, 15]:
                        lane = 32 * (mi // 16) + (mi % 16) + lane_align
                        got = dump[lane, ni]
                        expected = 2.0 * 16.0 * (ni + 1.0)
                        assert got == expected


def _build_mma_failure(
    *,
    dst_shape=(128, 16),
    col_start=0,
    lane_align=0,
    a_shape=(128, 16),
    dst_slice=None,
    allocate=True,
):
    b = builder("mma_failure", smem_size_bytes=1 << 16)
    dst = tmem_tensor(
        b, dtype=nr.DType.F32, shape=dst_shape, col_start=col_start, lane_align=lane_align
    )
    a = smem_tensor(b, dtype=nr.DType.F16, shape=a_shape, byte_offset=0)
    b_s = smem_tensor(b, dtype=nr.DType.F16, shape=(16, 16), byte_offset=4096)
    if allocate:
        with b.kernel_init(warp=0):
            b.tmem_alloc(dst, n_cols=32)
    with b.role(warp=0):
        b.tcgen05_mma(dst if dst_slice is None else dst_slice(dst), a, b_s, m=128, n=16, k=16)
    return b.build()


def test_tcgen05_mma_fail_closed_before_panics_or_wrong_writes():
    with expect_runtime_error("tcgen05_mma_lane_align"):
        run(_build_mma_failure(lane_align=16))

    with expect_runtime_error("tcgen05_mma_dst_offset"):
        run(_build_mma_failure(dst_shape=(129, 16), dst_slice=lambda dst: dst[1:129, :]))

    with expect_runtime_error("tcgen05_mma_out_of_range"):
        run(_build_mma_failure(col_start=500))

    with expect_runtime_error("tcgen05_mma_shape"):
        run(_build_mma_failure(a_shape=(129, 16)))

    with expect_runtime_error("missing_tmem_scratchpad"):
        run(_build_mma_failure(allocate=False))


def _tmem_operand_mma_kernel():
    m, n, k = 64, 16, 16
    b = builder("tmem_operand_mma", smem_size_bytes=n * k * 2)
    p_g = gmem_arg(b, dtype=nr.DType.F16, shape=(128, k))
    b_g = gmem_arg(b, dtype=nr.DType.F16, shape=(n, k))
    out = gmem_arg(b, dtype=nr.DType.F32, shape=(128, n))
    p = tmem_tensor(b, dtype=nr.DType.F16, shape=(m, k), col_start=0)
    dst = tmem_tensor(b, dtype=nr.DType.F32, shape=(m, n), col_start=32)
    b_s = smem_tensor(b, dtype=nr.DType.F16, shape=(n, k), byte_offset=0)
    p_frag = reg_tensor(b, dtype=nr.DType.F16, shape=(k,))
    out_frag = reg_tensor(b, dtype=nr.DType.F32, shape=(n,))
    mb = b.mbar(kind=nr.MBarKind.TMA)
    mc = b.mbar(kind=nr.MBarKind.TCGEN05)

    with b.kernel_init(warp=0):
        b.tmem_alloc(p, n_cols=32)
        b.tmem_alloc(dst, n_cols=32)
    with b.role(warpgroup=0):
        b.reg_fill(out_frag, 0.0)
        b.tcgen05_st(dst, out_frag, shape="32x32b", num=n, row=0, col=0)
        b.tcgen05_wait_st()
        b.reg_load(p_frag, p_g[b.tid_in_wg(), 0:k])
        b.tcgen05_st(p, p_frag[0 : k // 2], shape="32x32b", num=k // 2, row=0, col=0)
        b.tcgen05_wait_st()
        b.mbarrier_init(mb, count=1)
        with b.if_(b.tid_in_wg().eq(0)):
            b.mbarrier_arrive_expect_tx(mb, bytes=n * k * 2)
            b.tma_load(b_s, b_g, mbar=mb, bytes=n * k * 2, coords=(0, 0), shape=(n, k))
        b.mbarrier_wait(mb, phase=0)
        b.mbarrier_init(mc, count=1)
        b.tcgen05_mma(dst, p, b_s, m=m, n=n, k=k)
        b.tcgen05_commit(mc)
        b.tcgen05_ld(out_frag, dst, shape="32x32b", num=n, row=0, col=0)
        b.reg_store(out[b.tid_in_wg(), 0:n], out_frag)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(dst, n_cols=32)
        b.tmem_dealloc(p, n_cols=32)

    return b.build(), p_g, b_g, out


def test_tcgen05_mma_trace_accepts_tmem_operand_region():
    kernel, _, _, _ = _tmem_operand_mma_kernel()
    report = nr.check_protocol(kernel, include_events=True)
    assert report["status"] == "Passed"
    tmem_reads = [
        event
        for event in report["events"]
        if event["kind"] == "read"
        and event["access_category"] == "tensor"
        and event["region"]["owner"] == {"kind": "tmem", "cta_id": 0}
    ]
    assert len(tmem_reads) == 1
    assert tmem_reads[0]["region"]["boxes"] == [{"ranges": [(0, 64), (0, 32)]}]


def test_tcgen05_mma_value_mode_accepts_tmem_operand():
    kernel, p_g, b_g, out = _tmem_operand_mma_kernel()
    p = np.ones((128, 16), dtype=np.float32)
    b_values = np.zeros((16, 16), dtype=np.float32)
    for ni in range(16):
        b_values[ni, :] = ni + 1
    outputs = run(kernel, {p_g: p, b_g: b_values})
    dump = output(outputs, out)
    for mi in [0, 15, 16, 63]:
        for ni in [0, 7, 15]:
            lane = 32 * (mi // 16) + (mi % 16)
            assert dump[lane, ni] == 16.0 * (ni + 1.0)


def _f16_tmem_store_kernel():
    b = builder("f16_tmem_store")
    out = gmem_arg(b, dtype=nr.DType.F16, shape=(32, 2))
    tmem = tmem_tensor(b, dtype=nr.DType.F16, shape=(128, 32), col_start=0)
    frag = reg_tensor(b, dtype=nr.DType.F16, shape=(2,))
    loaded = reg_tensor(b, dtype=nr.DType.F16, shape=(2,))
    with b.kernel_init(warp=0):
        b.tmem_alloc(tmem, n_cols=32)
    with b.role(warp=0):
        b.reg_fill(frag[0], 1.0)
        b.reg_fill(frag[1], 2.0)
        b.tcgen05_st(tmem, frag[0:1], shape="32x32b", num=1, row=0, col=0)
        b.tcgen05_wait_st()
        b.tcgen05_ld(loaded[0:1], tmem, shape="32x32b", num=1, row=0, col=0)
        b.reg_store(out[b.tid_in_wg(), 0:2], loaded)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(tmem, n_cols=32)
    return b.build(), out


def test_tcgen05_st_trace_accepts_f16_tmem_region():
    kernel, _ = _f16_tmem_store_kernel()
    report = nr.check_protocol(kernel, include_events=True)
    assert report["status"] == "Passed"
    tmem_writes = [
        event
        for event in report["events"]
        if event["kind"] == "write"
        and event["access_category"] == "tmem"
        and event["access_kind"] == "st"
    ]
    assert len(tmem_writes) == 1
    assert tmem_writes[0]["region"]["owner"] == {"kind": "tmem", "cta_id": 0}
    reg_reads = [
        event
        for event in report["events"]
        if event["stmt_kind"] == "Tcgen05St"
        and event["kind"] == "read"
        and event["region"]["owner"]["kind"] == "reg"
    ]
    reg_writes = [
        event
        for event in report["events"]
        if event["stmt_kind"] == "Tcgen05Ld"
        and event["kind"] == "write"
        and event["region"]["owner"]["kind"] == "reg"
    ]
    assert reg_reads[0]["region"]["boxes"] == [{"ranges": [(0, 32), (0, 2)]}]
    assert reg_writes[0]["region"]["boxes"] == [{"ranges": [(0, 32), (0, 2)]}]


def test_tcgen05_st_value_mode_accepts_f16_tmem():
    kernel, out = _f16_tmem_store_kernel()
    outputs = run(kernel)
    expected = np.tile(np.array([1.0, 2.0], dtype=np.float16), (32, 1))
    np.testing.assert_array_equal(output(outputs, out), expected)


def _mma_cg2_peer_smem_kernel(synced):
    """One cluster pair; CTA1 writes its own SMEM operand tiles, CTA0 (the
    leader) issues a cta_group=2 MMA that reads BOTH CTAs' tiles. With
    synced=False there is no CTA1 -> CTA0 happens-before at all — the peer-SMEM
    read must surface as a data race, and the events must show an operand read
    on EACH CTA's pool (the old single-CTA trace only recorded smem:cta0)."""
    m, n, mma_k = 256, 32, 16
    a_bytes = 128 * mma_k * 2
    b_bytes = (n // 2) * mma_k * 2
    b = builder(
        "mma_cg2_peer_smem" + ("_sync" if synced else "_race"),
        smem_size_bytes=a_bytes + b_bytes,
        launch_shape=(2,),
        cluster_shape=(2,),
    )
    a_smem = smem_tensor(b, dtype=nr.DType.F16, shape=(128, mma_k), byte_offset=0)
    b_smem = smem_tensor(b, dtype=nr.DType.F16, shape=(n // 2, mma_k), byte_offset=a_bytes)
    accum = b.tensor(
        space=nr.MemorySpace.TMEM,
        dtype=nr.DType.F32,
        shape=(128, n),
        layout=nr.TmemLayout(nr.TmemLayoutKind.LANE_128, col_start=0),
    )
    ready = b.mbar(kind=nr.MBarKind.THREAD)
    ready_leader = b.mbar_ref(ready, remote_coord=0)  # absolute coord: the leader's cell
    with b.kernel_init(warp=0):
        b.tmem_alloc(accum, n_cols=32, cta_group=2)
        b.mbarrier_init(ready, count=1)
    with b.role(warp=0):
        cta = b.ctaid_in_cluster()
        with b.if_(b.lane_id().eq(0)):
            for r in range(2):
                b.store_scalar(nr.TensorSlice(tensor=a_smem, offsets=(r, 0), shape=(1, 1)), 0)
                b.store_scalar(nr.TensorSlice(tensor=b_smem, offsets=(r, 0), shape=(1, 1)), 0)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        if synced:
            # CTA1 -> CTA0 edge: the peer arrives the leader's barrier.
            with b.if_(cta.eq(1)):
                with b.if_(b.lane_id().eq(0)):
                    b.mbarrier_arrive(ready_leader)
        with b.if_(cta.eq(0)):
            if synced:
                b.mbarrier_wait(ready, phase=0)
            b.tcgen05_mma(accum, a_smem, b_smem, m=m, n=n, k=mma_k, accum=False, cta_group=2)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(accum, n_cols=32, cta_group=2)
    return b.build()


def test_mma_cta_group2_records_peer_smem_reads_and_flags_unordered_writes():
    report = nr.check_protocol(_mma_cg2_peer_smem_kernel(synced=False), include_events=True)
    assert report["status"] == "Failed"
    assert any(d["code"] == "memory_data_race" for d in report["diagnostics"])
    owners = {
        e["region"]["owner"].get("cta_id")
        for e in report["events"]
        if e["kind"] == "read"
        and e.get("access_kind") == "tcgen05_mma"
        and e["region"]["owner"]["kind"] == "smem"
    }
    assert owners == {0, 1}, owners

    synced = nr.check_protocol(_mma_cg2_peer_smem_kernel(synced=True))
    assert synced["status"] == "Passed", synced["diagnostics"][:2]


def _mma_operand_overwrite_kernel(drain):
    """An async MMA reads SMEM operands; the same stream overwrites an operand
    byte afterwards. Without a tcgen05_commit drain between them the engine
    may still be reading the tile on silicon — pinning the read at its issue
    point (plain program order) must NOT bless this."""
    b = builder(
        "mma_operand_overwrite" + ("_drain" if drain else ""),
        smem_size_bytes=128 * 16 * 2 + 16 * 16 * 2,
    )
    a = smem_tensor(b, dtype=nr.DType.F16, shape=(128, 16), byte_offset=0)
    bb = smem_tensor(b, dtype=nr.DType.F16, shape=(16, 16), byte_offset=128 * 16 * 2)
    acc = b.tensor(
        space=nr.MemorySpace.TMEM,
        dtype=nr.DType.F32,
        shape=(128, 16),
        layout=nr.TmemLayout(nr.TmemLayoutKind.LANE_128, col_start=0),
    )
    done = b.mbar(kind=nr.MBarKind.TCGEN05)
    with b.kernel_init(warp=0):
        b.tmem_alloc(acc, n_cols=32, cta_group=1)
        b.mbarrier_init(done, count=1)
    with b.role(warp=0):
        with b.if_(b.lane_id().eq(0)):
            b.store_scalar(nr.TensorSlice(tensor=a, offsets=(0, 0), shape=(1, 1)), 0)
            b.store_scalar(nr.TensorSlice(tensor=bb, offsets=(0, 0), shape=(1, 1)), 0)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tcgen05_mma(acc, a, bb, m=128, n=16, k=16, accum=False, cta_group=1)
        if drain:
            b.tcgen05_commit(done, cta_group=1)
            b.mbarrier_wait(done, phase=0)
        with b.if_(b.lane_id().eq(0)):
            b.store_scalar(nr.TensorSlice(tensor=a, offsets=(0, 0), shape=(1, 1)), 1)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(acc, n_cols=32, cta_group=1)
    return b.build()


def test_mma_operand_overwrite_requires_commit_drain():
    report = nr.check_protocol(_mma_operand_overwrite_kernel(drain=False))
    assert report["status"] == "Failed"
    assert any(d["code"] == "tcgen05_operand_overwrite_before_drain" for d in report["diagnostics"])

    drained = nr.check_protocol(_mma_operand_overwrite_kernel(drain=True))
    assert drained["status"] == "Passed", drained["diagnostics"][:2]


def _mma_acc_read_release_kernel(commit_release):
    """Stream A issues an async MMA and signals stream B. If the signal is a
    plain thread arrive (not a tcgen05_commit), B's accumulator read lands
    while the MMA's write window is still open — ordered against the issue,
    not against the drain."""
    b = builder(
        "mma_acc_read_" + ("commit" if commit_release else "arrive"),
        smem_size_bytes=128 * 16 * 2 + 16 * 16 * 2,
        num_warps=8,
    )
    a = smem_tensor(b, dtype=nr.DType.F16, shape=(128, 16), byte_offset=0)
    bb = smem_tensor(b, dtype=nr.DType.F16, shape=(16, 16), byte_offset=128 * 16 * 2)
    acc = b.tensor(
        space=nr.MemorySpace.TMEM,
        dtype=nr.DType.F32,
        shape=(128, 16),
        layout=nr.TmemLayout(nr.TmemLayoutKind.LANE_128, col_start=0),
    )
    frag = reg_tensor(b, dtype=nr.DType.F32, shape=(16,))
    done = b.mbar(kind=nr.MBarKind.TCGEN05 if commit_release else nr.MBarKind.THREAD)
    with b.kernel_init(warp=0):
        b.tmem_alloc(acc, n_cols=32, cta_group=1)
        b.mbarrier_init(done, count=1)
    with b.role(warp=0):
        with b.if_(b.lane_id().eq(0)):
            b.store_scalar(nr.TensorSlice(tensor=a, offsets=(0, 0), shape=(1, 1)), 0)
            b.store_scalar(nr.TensorSlice(tensor=bb, offsets=(0, 0), shape=(1, 1)), 0)
        b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
        b.tcgen05_mma(acc, a, bb, m=128, n=16, k=16, accum=False, cta_group=1)
        if commit_release:
            b.tcgen05_commit(done, cta_group=1)
        else:
            with b.if_(b.lane_id().eq(0)):
                b.mbarrier_arrive(done)
    with b.role(warp=4):
        b.mbarrier_wait(done, phase=0)
        b.tcgen05_ld(frag, acc, num=16, row=0, col=0)
        b.tcgen05_wait_ld()
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(acc, n_cols=32, cta_group=1)
    return b.build()


def test_mma_acc_read_release_must_come_from_commit():
    report = nr.check_protocol(_mma_acc_read_release_kernel(commit_release=False))
    assert report["status"] == "Failed"
    assert any(d["code"] == "tmem_access_before_drain" for d in report["diagnostics"])

    drained = nr.check_protocol(_mma_acc_read_release_kernel(commit_release=True))
    assert drained["status"] == "Passed", drained["diagnostics"][:2]
