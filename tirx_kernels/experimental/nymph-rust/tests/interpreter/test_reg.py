import numpy as np
import nymph_rs as nr
from helpers import (
    assert_output_eq,
    builder,
    expect_runtime_error,
    f32,
    gmem_arg,
    i32,
    output,
    reg_tensor,
    run,
    smem_tensor,
    u32,
)


def test_reg_float_and_cvt_value_semantics_round_destination_dtypes():
    b = builder("reg_float_cvt_value")
    lhs_g = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    rhs_g = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    acc_g = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    add_out = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    fma_out = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    f16_out = gmem_arg(b, dtype=nr.DType.F16, shape=(4,))
    bf16_out = gmem_arg(b, dtype=nr.DType.BF16, shape=(4,))
    lhs = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    rhs = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    acc = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    tmp = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    f16_tmp = reg_tensor(b, dtype=nr.DType.F16, shape=(4,))
    bf16_tmp = reg_tensor(b, dtype=nr.DType.BF16, shape=(4,))

    with b.role(warp=0, elected=True):
        b.reg_load(lhs, lhs_g)
        b.reg_load(rhs, rhs_g)
        b.reg_load(acc, acc_g)
        b.reg_add(tmp, lhs, rhs)
        b.reg_store(add_out, tmp)
        b.reg_fma(tmp, lhs, rhs, acc)
        b.reg_store(fma_out, tmp)
        b.reg_cvt(f16_tmp, tmp)
        b.reg_store(f16_out, f16_tmp)
        b.reg_cvt(bf16_tmp, tmp)
        b.reg_store(bf16_out, bf16_tmp)

    outputs = run(
        b.build(),
        {
            lhs_g: f32([1.5, -2.0, 1.0004883, 1.0039063]),
            rhs_g: f32([2.0, 5.0, 0.0, 0.0]),
            acc_g: f32([0.5, 1.0, 0.0, 0.0]),
        },
    )
    np.testing.assert_array_equal(output(outputs, add_out), f32([3.5, 3.0, 1.0004883, 1.0039063]))
    np.testing.assert_array_equal(output(outputs, fma_out), f32([3.5, -9.0, 0.0, 0.0]))
    np.testing.assert_array_equal(
        output(outputs, f16_out).astype(np.float32), f32([3.5, -9.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(output(outputs, bf16_out), f32([3.5, -9.0, 0.0, 0.0]))


def test_reg_integer_alu_value_semantics():
    b = builder("reg_integer_alu_value")
    i_lhs_g = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    i_rhs_g = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    i_add_out = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    i_sub_out = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    i_mul_out = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    i_max_out = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    i_min_out = gmem_arg(b, dtype=nr.DType.I32, shape=(4,))
    u_lhs_g = gmem_arg(b, shape=(4,))
    u_rhs_g = gmem_arg(b, shape=(4,))
    u_add_out = gmem_arg(b, shape=(4,))
    u_sub_out = gmem_arg(b, shape=(4,))
    u_mul_out = gmem_arg(b, shape=(4,))
    u_max_out = gmem_arg(b, shape=(4,))
    u_min_out = gmem_arg(b, shape=(4,))
    i_lhs = reg_tensor(b, dtype=nr.DType.I32, shape=(4,))
    i_rhs = reg_tensor(b, dtype=nr.DType.I32, shape=(4,))
    i_tmp = reg_tensor(b, dtype=nr.DType.I32, shape=(4,))
    u_lhs = reg_tensor(b, shape=(4,))
    u_rhs = reg_tensor(b, shape=(4,))
    u_tmp = reg_tensor(b, shape=(4,))

    with b.role(warp=0, elected=True):
        b.reg_load(i_lhs, i_lhs_g)
        b.reg_load(i_rhs, i_rhs_g)
        for op, out_t in [
            (b.reg_add, i_add_out),
            (b.reg_sub, i_sub_out),
            (b.reg_mul, i_mul_out),
            (b.reg_max, i_max_out),
            (b.reg_min, i_min_out),
        ]:
            op(i_tmp, i_lhs, i_rhs)
            b.reg_store(out_t, i_tmp)

        b.reg_load(u_lhs, u_lhs_g)
        b.reg_load(u_rhs, u_rhs_g)
        for op, out_t in [
            (b.reg_add, u_add_out),
            (b.reg_sub, u_sub_out),
            (b.reg_mul, u_mul_out),
            (b.reg_max, u_max_out),
            (b.reg_min, u_min_out),
        ]:
            op(u_tmp, u_lhs, u_rhs)
            b.reg_store(out_t, u_tmp)

    outputs = run(
        b.build(),
        {
            i_lhs_g: i32([2147483647, -2147483648, 50000, -5]),
            i_rhs_g: i32([1, 1, 50000, 7]),
            u_lhs_g: u32([0xFFFF_FFFF, 0, 2, 5]),
            u_rhs_g: u32([1, 1, 0xFFFF_FFFF, 7]),
        },
    )
    assert_output_eq(outputs, i_add_out, [-2147483648, -2147483647, 100000, 2], dtype=np.int32)
    assert_output_eq(outputs, i_sub_out, [2147483646, 2147483647, 0, -12], dtype=np.int32)
    assert_output_eq(
        outputs, i_mul_out, [2147483647, -2147483648, -1794967296, -35], dtype=np.int32
    )
    assert_output_eq(outputs, i_max_out, [2147483647, 1, 50000, 7], dtype=np.int32)
    assert_output_eq(outputs, i_min_out, [1, -2147483648, 50000, -5], dtype=np.int32)
    assert_output_eq(outputs, u_add_out, [0, 1, 1, 12], dtype=np.uint32)
    assert_output_eq(
        outputs, u_sub_out, [0xFFFF_FFFE, 0xFFFF_FFFF, 3, 0xFFFF_FFFE], dtype=np.uint32
    )
    assert_output_eq(outputs, u_mul_out, [0xFFFF_FFFF, 0, 0xFFFF_FFFE, 35], dtype=np.uint32)
    assert_output_eq(outputs, u_max_out, [0xFFFF_FFFF, 1, 0xFFFF_FFFF, 7], dtype=np.uint32)
    assert_output_eq(outputs, u_min_out, [1, 0, 2, 5], dtype=np.uint32)


def test_reg_extended_float_ops_literals_broadcast_reduce_and_unary():
    b = builder("reg_extended_float_ops")
    src_g = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    shifted_out = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    sum_out = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    max_out = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    exp_out = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    vec = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    scalar = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    max_scalar = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    shifted = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    expv = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))

    with b.role(warp=0, elected=True):
        b.reg_load(vec, src_g)
        b.reg_reduce(scalar, vec, op="sum")
        b.reg_reduce(max_scalar, vec, op="max")
        b.reg_store(sum_out, scalar)
        b.reg_store(max_out, max_scalar)
        b.reg_add(shifted, vec, scalar)
        b.reg_store(shifted_out, shifted)
        b.reg_unary(expv, vec, op="exp2")
        b.reg_store(exp_out, expv)

    outputs = run(b.build(), {src_g: f32([1.0, 2.0, 3.0, 4.0])})
    assert_output_eq(outputs, sum_out, [10.0], dtype=np.float32)
    assert_output_eq(outputs, max_out, [4.0], dtype=np.float32)
    assert_output_eq(outputs, shifted_out, [11.0, 12.0, 13.0, 14.0], dtype=np.float32)
    assert_output_eq(outputs, exp_out, [2.0, 4.0, 8.0, 16.0], dtype=np.float32)


def test_reg_bitwise_shift_and_combine_int_frac_ex2_value_semantics():
    b = builder("reg_bitwise_combine")
    lhs_g = gmem_arg(b, shape=(4,))
    rhs_g = gmem_arg(b, shape=(4,))
    and_out = gmem_arg(b, shape=(4,))
    shl_out = gmem_arg(b, shape=(4,))
    rounded_g = gmem_arg(b, dtype=nr.DType.F32, shape=(2,))
    frac_g = gmem_arg(b, dtype=nr.DType.F32, shape=(2,))
    combine_out = gmem_arg(b, dtype=nr.DType.F32, shape=(2,))
    lhs = reg_tensor(b, shape=(4,))
    rhs = reg_tensor(b, shape=(4,))
    tmp = reg_tensor(b, shape=(4,))
    rounded = reg_tensor(b, dtype=nr.DType.F32, shape=(2,))
    frac = reg_tensor(b, dtype=nr.DType.F32, shape=(2,))
    combined = reg_tensor(b, dtype=nr.DType.F32, shape=(2,))

    with b.role(warp=0, elected=True):
        b.reg_load(lhs, lhs_g)
        b.reg_load(rhs, rhs_g)
        b.reg_bitwise(tmp, lhs, rhs, op="and")
        b.reg_store(and_out, tmp)
        b.reg_bitwise(tmp, lhs, rhs, op="shl")
        b.reg_store(shl_out, tmp)
        b.reg_load(rounded, rounded_g)
        b.reg_load(frac, frac_g)
        b.reg_combine_int_frac_ex2(combined, rounded, frac)
        b.reg_store(combine_out, combined)

    rounded_vals = f32([1.0, 2.0])
    frac_vals = f32([1.25, 0.75])
    rounded_bits = rounded_vals.view(np.uint32)
    frac_bits = frac_vals.view(np.uint32)
    expected_bits = ((rounded_bits << np.uint32(23)) + frac_bits).astype(np.uint32)
    outputs = run(
        b.build(),
        {
            lhs_g: u32([0b1010, 0b1100, 1, 2]),
            rhs_g: u32([0b0110, 0b1010, 3, 4]),
            rounded_g: rounded_vals,
            frac_g: frac_vals,
        },
    )
    assert_output_eq(outputs, and_out, [0b0010, 0b1000, 1, 0], dtype=np.uint32)
    assert_output_eq(outputs, shl_out, [640, 12288, 8, 32], dtype=np.uint32)
    np.testing.assert_array_equal(output(outputs, combine_out).view(np.uint32), expected_bits)


def test_reg_softmax_rescale_keeps_small_row_max_increase():
    b = builder("reg_softmax_rescale")
    old_g = gmem_arg(b, dtype=nr.DType.F32, shape=(32,))
    new_g = gmem_arg(b, dtype=nr.DType.F32, shape=(32,))
    max_out = gmem_arg(b, dtype=nr.DType.F32, shape=(32,))
    scale_out = gmem_arg(b, dtype=nr.DType.F32, shape=(32,))
    old = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    new = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    row_max = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    row_scale = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))

    with b.role(warp=0):
        b.reg_load(old, old_g[b.tid_in_wg()])
        b.reg_load(new, new_g[b.tid_in_wg()])
        b.reg_softmax_rescale(row_max, row_scale, old, new, 1.0, threshold=8.0)
        b.reg_store(max_out[b.tid_in_wg()], row_max)
        b.reg_store(scale_out[b.tid_in_wg()], row_scale)

    old_values = np.full(32, 10.0, dtype=np.float32)
    new_values = np.concatenate(
        [np.full(16, 10.5, dtype=np.float32), np.full(16, 30.0, dtype=np.float32)]
    )
    outputs = run(b.build(), {old_g: old_values, new_g: new_values})
    expected_max = np.concatenate(
        [np.full(16, 10.0, dtype=np.float32), np.full(16, 30.0, dtype=np.float32)]
    )
    expected_scale = np.concatenate(
        [np.ones(16, dtype=np.float32), np.full(16, 2.0**-20, dtype=np.float32)]
    )
    np.testing.assert_array_equal(output(outputs, max_out), expected_max)
    np.testing.assert_allclose(output(outputs, scale_out), expected_scale, rtol=1e-6)


def test_reg_softmax_rescale_accepts_register_threshold():
    b = builder("reg_softmax_rescale_reg_threshold")
    old_g = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    new_g = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    threshold_g = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    max_out = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    scale_out = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    old = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    new = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    threshold = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    row_max = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    row_scale = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))

    with b.role(warp=0, elected=True):
        b.reg_load(old, old_g[0])
        b.reg_load(new, new_g[0])
        b.reg_load(threshold, threshold_g[0])
        b.reg_softmax_rescale(row_max, row_scale, old, new, 1.0, threshold=threshold)
        b.reg_store(max_out[0], row_max)
        b.reg_store(scale_out[0], row_scale)

    outputs = run(b.build(), {old_g: f32([10.0]), new_g: f32([15.0]), threshold_g: f32([8.0])})
    assert_output_eq(outputs, max_out, [10.0])
    assert_output_eq(outputs, scale_out, [1.0])


def test_reg_min_caps_safe_reciprocal_for_fa4_epilogue():
    b = builder("reg_safe_reciprocal_idiom")
    row_sum_g = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    out_g = gmem_arg(b, dtype=nr.DType.F32, shape=(4,))
    row_sum = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    inv = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))

    with b.role(warp=0, elected=True):
        b.reg_load(row_sum, row_sum_g)
        b.reg_unary(inv, row_sum, op="rcp")
        b.reg_min(inv, 1.0, inv)
        b.reg_store(out_g, inv)

    outputs = run(b.build(), {row_sum_g: f32([0.0, np.nan, 1.0, 2.0])})
    np.testing.assert_array_equal(output(outputs, out_g), f32([1.0, 1.0, 1.0, 0.5]))


def test_reg_cond_rescale_uses_warpgroup_any_scope():
    b = builder("reg_cond_rescale_wg", num_warps=4)
    src_g = gmem_arg(b, dtype=nr.DType.F32, shape=(1,))
    scale_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128,))
    out_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128,))
    src = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    scale = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    out = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))

    with b.role(warpgroup=0):
        b.reg_load(src, src_g[0])
        b.reg_load(scale, scale_g[b.tid_in_wg()])
        b.reg_cond_rescale(out, src, scale, threshold=1.0, scope="warpgroup")
        b.reg_store(out_g[b.tid_in_wg()], out)

    scales = np.ones(128, dtype=np.float32)
    scales[7] = 0.5
    scales[9] = 1.5
    outputs = run(b.build(), {src_g: f32([2.0]), scale_g: scales})
    expected = 2.0 * scales
    np.testing.assert_array_equal(output(outputs, out_g), expected)


def test_reg_cond_rescale_direct_matches_slow_dynamic_slot():
    b = builder("reg_cond_rescale_direct_slow", num_warps=4)
    src_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128,))
    scale_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128,))
    direct_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128,))
    slow_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128,))
    src_direct = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    scale_direct = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    out_direct = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    src_slow = reg_tensor(b, dtype=nr.DType.F32, shape=(2,))
    scale_slow = reg_tensor(b, dtype=nr.DType.F32, shape=(2,))
    out_slow = reg_tensor(b, dtype=nr.DType.F32, shape=(2,))
    slot = b.tid_in_wg() % 2

    with b.role(warpgroup=0):
        b.reg_load(src_direct, src_g[b.tid_in_wg()])
        b.reg_load(scale_direct, scale_g[b.tid_in_wg()])
        b.reg_cond_rescale(out_direct, src_direct, scale_direct, threshold=1.0, scope="warpgroup")
        b.reg_store(direct_g[b.tid_in_wg()], out_direct)

        b.reg_load(src_slow[slot], src_g[b.tid_in_wg()])
        b.reg_load(scale_slow[slot], scale_g[b.tid_in_wg()])
        b.reg_cond_rescale(out_slow[slot], src_slow[slot], scale_slow[slot], threshold=1.0)
        b.reg_store(slow_g[b.tid_in_wg()], out_slow[slot])

    src = np.linspace(1.0, 3.0, 128, dtype=np.float32)
    scales = np.ones(128, dtype=np.float32)
    scales[19] = 0.25
    scales[91] = 1.5
    outputs = run(b.build(), {src_g: src, scale_g: scales})
    np.testing.assert_array_equal(output(outputs, direct_g), output(outputs, slow_g))
    np.testing.assert_array_equal(output(outputs, direct_g), src * scales)


def test_reg_causal_mask_uses_thread_row_and_element_index():
    b = builder("reg_causal_mask", num_warps=8)
    src_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128, 4))
    out_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128, 4))
    src = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    out = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))

    with b.role(warpgroup=1):
        b.reg_load(src, src_g[b.tid_in_wg(), 0:4])
        b.reg_causal_mask(out, src, query_start=10, key_start=8, group_size=4, mask_value=-99.0)
        b.reg_store(out_g[b.tid_in_wg(), 0:4], out)

    values = np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
    expected = values.copy()
    for row in range(128):
        q_idx = 10 + row // 4
        for col in range(4):
            if 8 + col > q_idx:
                expected[row, col] = -99.0
    outputs = run(b.build(), {src_g: values})
    np.testing.assert_array_equal(output(outputs, out_g), expected)


def test_reg_causal_mask_direct_matches_slow_dynamic_slot():
    b = builder("reg_causal_mask_direct_slow", num_warps=8)
    src_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128, 4))
    direct_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128, 4))
    slow_g = gmem_arg(b, dtype=nr.DType.F32, shape=(128, 4))
    src_direct = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    out_direct = reg_tensor(b, dtype=nr.DType.F32, shape=(4,))
    src_slow = reg_tensor(b, dtype=nr.DType.F32, shape=(2, 4))
    out_slow = reg_tensor(b, dtype=nr.DType.F32, shape=(2, 4))
    slot = b.tid_in_wg() % 2

    with b.role(warpgroup=1):
        b.reg_load(src_direct, src_g[b.tid_in_wg(), 0:4])
        b.reg_causal_mask(
            out_direct, src_direct, query_start=10, key_start=8, group_size=4, mask_value=-99.0
        )
        b.reg_store(direct_g[b.tid_in_wg(), 0:4], out_direct)

        b.reg_load(src_slow[slot, 0:4], src_g[b.tid_in_wg(), 0:4])
        b.reg_causal_mask(
            out_slow[slot, 0:4],
            src_slow[slot, 0:4],
            query_start=10,
            key_start=8,
            group_size=4,
            mask_value=-99.0,
        )
        b.reg_store(slow_g[b.tid_in_wg(), 0:4], out_slow[slot, 0:4])

    values = np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
    outputs = run(b.build(), {src_g: values})
    np.testing.assert_array_equal(output(outputs, direct_g), output(outputs, slow_g))


def test_reg_smem_is_owned_per_cta():
    b = builder("reg_smem_per_cta", smem_size_bytes=4, launch_shape=(2,), cluster_shape=(2,))
    cta_values = gmem_arg(b, shape=(2,))
    cta_out = gmem_arg(b, shape=(2,))
    cta_smem = smem_tensor(b, shape=(1,), byte_offset=0)
    first = reg_tensor(b)
    second = reg_tensor(b)

    with b.role(warp=0, elected=True):
        b.reg_load(first[0], cta_values[b.cta_id()])
        b.reg_store(cta_smem[0], first[0])
        b.reg_load(second[0], cta_smem[0])
        b.reg_store(cta_out[b.cta_id()], second[0])

    outputs = run(b.build(), {cta_values: u32([10, 20])})
    assert_output_eq(outputs, cta_out, [10, 20], dtype=np.uint32)


def test_reg_overlap_failure_is_fail_closed():
    b = builder("reg_overlap_fail_closed")
    overlap_src = gmem_arg(b, shape=(32,))
    overlap_out = gmem_arg(b, shape=(1,))
    overlap_reg = reg_tensor(b)

    with b.role(warp=0):
        b.reg_load(overlap_reg[0], overlap_src[b.lane_id()])
        b.reg_store(overlap_out[0], overlap_reg[0])

    with expect_runtime_error("overlapping_tensor_write"):
        run(b.build(), {overlap_src: u32(range(32))})


def _reg_write_regions(report, tensor):
    return [
        event["region"]
        for event in report["events"]
        if event["kind"] == "write"
        and event["region"]["owner"]["kind"] == "reg"
        and event["region"]["tensor_id"] == tensor.id
    ]


def _ranges(box_):
    return [tuple(r) for r in box_["ranges"]]


def test_reg_trace_region_uses_register_rows_for_warp_uniform_slice():
    b = builder("reg_region_warp_uniform", num_warps=4)
    reg = reg_tensor(b, shape=(1,))

    with b.role(warp=0):
        b.reg_fill(reg[0], 1)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    regions = _reg_write_regions(report, reg)
    assert len(regions) == 1
    assert regions[0]["owner"]["cta_id"] == 0
    assert [_ranges(box_) for box_ in regions[0]["boxes"]] == [[(0, 32), (0, 1)]]


def test_reg_trace_region_keeps_lane_id_diagonal_exact():
    b = builder("reg_region_lane_diagonal", num_warps=4)
    reg = reg_tensor(b, shape=(32,))

    with b.role(warp=0):
        b.reg_fill(reg[b.lane_id()], 1)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    regions = _reg_write_regions(report, reg)
    assert len(regions) == 1
    boxes = [_ranges(box_) for box_ in regions[0]["boxes"]]
    assert len(boxes) == 32
    assert boxes[0] == [(0, 1), (0, 1)]
    assert boxes[-1] == [(31, 32), (31, 32)]


def test_reg_trace_region_elected_thread_covers_tensor_slice():
    b = builder("reg_region_elected_slice", num_warps=4)
    reg = reg_tensor(b, shape=(4,))

    with b.role(warp=0, elected=True):
        b.reg_fill(reg, 1)

    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed"
    regions = _reg_write_regions(report, reg)
    assert len(regions) == 1
    assert [_ranges(box_) for box_ in regions[0]["boxes"]] == [[(0, 1), (0, 4)]]
