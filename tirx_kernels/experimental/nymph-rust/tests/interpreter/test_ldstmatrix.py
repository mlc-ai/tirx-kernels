import numpy as np
import nymph_rs as nr
from helpers import builder, gmem_arg, output, reg_tensor, run, smem_tensor


def _coord(lane, half, trans):
    if trans:
        return 2 * (lane % 4) + half, lane // 4
    return lane // 4, 2 * (lane % 4) + half


def _pack(lo, hi):
    return np.uint32(int(lo) | (int(hi) << 16))


def _source_values(num):
    rows = 8 * num
    return np.asarray(
        [[(row << 8) | col for col in range(8)] for row in range(rows)], dtype=np.uint16
    )


def _ldmatrix_kernel(num, trans):
    rows = 8 * num
    nbytes = rows * 8 * 2
    b = builder(f"ldmatrix_x{num}_{'t' if trans else 'n'}", smem_size_bytes=nbytes)
    source = gmem_arg(b, dtype=nr.DType.U16, shape=(rows, 8))
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(32, num))
    smem = smem_tensor(b, dtype=nr.DType.U16, shape=(rows, 8), byte_offset=0)
    frag = reg_tensor(b, dtype=nr.DType.U32, shape=(num,))
    mbar = b.mbar(kind=nr.MBarKind.TMA)

    with b.kernel_init(warp=0, elected=True):
        b.mbarrier_init(mbar, count=1)
    with b.role(warp=0):
        b.mbarrier_arrive_expect_tx(mbar, bytes=nbytes)
        b.tma_load(smem, source, mbar=mbar, bytes=nbytes, coords=(0, 0), shape=(rows, 8))
        b.ldmatrix(frag, smem[b.lane_id() % rows, 0:8], num=num, trans=trans)
        b.reg_store(out[b.lane_id(), 0:num], frag)
    return b.build(), source, out


def test_ldmatrix_m8n8_b16_matches_ptx_fragment_mapping():
    for num in [1, 2, 4]:
        for trans in [False, True]:
            kernel, source, out = _ldmatrix_kernel(num, trans)
            result = output(run(kernel, {source: _source_values(num)}), out)
            expected = np.zeros((32, num), dtype=np.uint32)
            for lane in range(32):
                for matrix_id in range(num):
                    lo_row, lo_col = _coord(lane, 0, trans)
                    hi_row, hi_col = _coord(lane, 1, trans)
                    expected[lane, matrix_id] = _pack(
                        (matrix_id * 8 + lo_row) << 8 | lo_col,
                        (matrix_id * 8 + hi_row) << 8 | hi_col,
                    )
            np.testing.assert_array_equal(result, expected, err_msg=f"x{num} trans={trans}")


def _stmatrix_kernel(num, trans):
    rows = 8 * num
    nbytes = rows * 8 * 2
    b = builder(f"stmatrix_x{num}_{'t' if trans else 'n'}", smem_size_bytes=nbytes)
    source = gmem_arg(b, dtype=nr.DType.U32, shape=(32, num))
    out = gmem_arg(b, dtype=nr.DType.U16, shape=(rows, 8))
    smem = smem_tensor(b, dtype=nr.DType.U16, shape=(rows, 8), byte_offset=0)
    frag = reg_tensor(b, dtype=nr.DType.U32, shape=(num,))

    with b.role(warp=0):
        b.reg_load(frag, source[b.lane_id(), 0:num])
        b.stmatrix(smem[b.lane_id() % rows, 0:8], frag, num=num, trans=trans)
        b.tma_store(out, smem, coords=(0, 0), shape=(rows, 8))
    return b.build(), source, out


def test_stmatrix_m8n8_b16_matches_ptx_fragment_mapping():
    for num in [1, 2, 4]:
        for trans in [False, True]:
            kernel, source, out = _stmatrix_kernel(num, trans)
            words = np.zeros((32, num), dtype=np.uint32)
            for lane in range(32):
                for matrix_id in range(num):
                    words[lane, matrix_id] = _pack(
                        (matrix_id << 12) | (lane << 1), (matrix_id << 12) | (lane << 1) | 1
                    )
            result = output(run(kernel, {source: words}), out)
            expected = np.zeros((8 * num, 8), dtype=np.uint16)
            for lane in range(32):
                for matrix_id in range(num):
                    halves = [words[lane, matrix_id] & 0xFFFF, words[lane, matrix_id] >> 16]
                    for half, value in enumerate(halves):
                        row, col = _coord(lane, half, trans)
                        expected[matrix_id * 8 + row, col] = value
            np.testing.assert_array_equal(result, expected, err_msg=f"x{num} trans={trans}")


def _stmatrix_b16_kernel(num, trans):
    rows = 8 * num
    nbytes = rows * 8 * 2
    b = builder(f"stmatrix_b16_x{num}_{'t' if trans else 'n'}", smem_size_bytes=nbytes)
    source = gmem_arg(b, dtype=nr.DType.F16, shape=(32, 2 * num))
    out = gmem_arg(b, dtype=nr.DType.U16, shape=(rows, 8))
    smem = smem_tensor(b, dtype=nr.DType.U16, shape=(rows, 8), byte_offset=0)
    frag = reg_tensor(b, dtype=nr.DType.F16, shape=(2 * num,))

    with b.role(warp=0):
        b.reg_load(frag, source[b.lane_id(), 0 : 2 * num])
        b.stmatrix(smem[b.lane_id() % rows, 0:8], frag, num=num, trans=trans)
        b.tma_store(out, smem, coords=(0, 0), shape=(rows, 8))
    return b.build(), source, out


def test_stmatrix_accepts_b16_fragment_as_packed_words():
    # A b16 fragment of 2*num elements is the SAME register file as num packed
    # u32 words (consecutive pairs, little-endian) — the form the f32->b16x2
    # pair cvt produces. Must store identically to the u32-word fragment.
    for num in [1, 2, 4]:
        for trans in [False, True]:
            kernel, source, out = _stmatrix_b16_kernel(num, trans)
            # f16-encodable test pattern: halves are f16 bit patterns of
            # distinct small floats.
            values = np.arange(32 * 2 * num, dtype=np.float16).reshape(32, 2 * num)
            result = output(run(kernel, {source: values.astype(np.float32)}), out)
            bits = values.view(np.uint16)
            expected = np.zeros((8 * num, 8), dtype=np.uint16)
            for lane in range(32):
                for matrix_id in range(num):
                    for half in range(2):
                        row, col = _coord(lane, half, trans)
                        expected[matrix_id * 8 + row, col] = bits[lane, 2 * matrix_id + half]
            np.testing.assert_array_equal(result, expected, err_msg=f"x{num} trans={trans}")


def test_ldstmatrix_trace_events_identify_matrix_accesses():
    num = 2
    rows = 8 * num
    b = builder("ldstmatrix_trace", smem_size_bytes=rows * 8 * 2)
    smem = smem_tensor(b, dtype=nr.DType.U16, shape=(rows, 8), byte_offset=0)
    frag = reg_tensor(b, dtype=nr.DType.U32, shape=(num,))

    with b.role(warp=0):
        b.stmatrix(smem[b.lane_id() % rows, 0:8], frag, num=num, trans=True)
        b.ldmatrix(frag, smem[b.lane_id() % rows, 0:8], num=num, trans=True)
        b.stmatrix(smem[b.lane_id() % rows, 0:8], frag, num=num, trans=True)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    reads = [
        e for e in report["events"] if e["kind"] == "read" and e["access_category"] == "tensor"
    ]
    writes = [
        e for e in report["events"] if e["kind"] == "write" and e["access_category"] == "tensor"
    ]
    assert any(e["access_kind"] == "ldmatrix" and e["proxy"] == "generic" for e in reads)
    assert any(e["access_kind"] == "stmatrix" and e["proxy"] == "generic" for e in writes)
    ldmatrix_read = next(e for e in reads if e["access_kind"] == "ldmatrix")
    assert ldmatrix_read["region"]["owner"]["kind"] == "smem"
    assert ldmatrix_read["region"]["boxes"]
