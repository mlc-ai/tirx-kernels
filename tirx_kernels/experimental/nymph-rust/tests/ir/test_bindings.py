"""Unit tests for the `nymph_rs` PyO3 bindings.

Requires the `nymph_rs` extension on the path (build with `run_python_tests.sh`,
which puts it on PYTHONPATH).
"""

import numpy as np
import pytest

nymph_rs = pytest.importorskip("nymph_rs")
n = nymph_rs


def test_enum_names_match_python_and_compare():
    assert str(n.DType.BF16) == "DType.BF16"
    assert n.DType.F16 == n.DType.F16
    assert n.DType.F16 != n.DType.BF16
    assert n.MemorySpace.SMEM is not None
    assert n.VarBinding.SCALAR is not None
    # hashable (used as dict keys, e.g. _SCALAR_GMEM_DTYPES)
    assert {n.DType.F16: 1}[n.DType.F16] == 1


def test_scalar_operators_build_exprs():
    task = n.Var(binding=n.VarBinding.TASK, dtype=n.ScalarDType.I32)
    k = n.Var()  # loop var defaults
    assert (task * 16 + k).op == n.ScalarOp.ADD
    assert (16 * task).op == n.ScalarOp.MUL  # reflected __rmul__
    assert (task < 100).op == n.ScalarOp.LT  # richcmp
    assert task.eq(k).op == n.ScalarOp.EQ
    assert (-task).op == n.ScalarOp.NEG
    assert n.select(task < 100, task, k).op == n.ScalarOp.SELECT
    assert n.min(task, k).op == n.ScalarOp.MIN
    assert n.max(task, k).op == n.ScalarOp.MAX


def test_var_identity_is_unique_per_construction():
    assert n.Var().id != n.Var().id != n.Var().id
    task = n.Var(binding=n.VarBinding.TASK, dtype=n.ScalarDType.I32)
    assert task.binding == n.VarBinding.TASK
    assert task.dtype == n.ScalarDType.I32


def test_isinstance_targets():
    task = n.Var()
    A = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.F16, shape=[256, 16], byte_offset=0)
    assert isinstance(task, n.Var)
    assert isinstance(task * 2, n.ScalarExpr)
    assert isinstance(A, n.Tensor)
    assert isinstance(A[:, :], n.TensorSlice)


def test_tensor_getitem_static_and_symbolic():
    A = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.F16, shape=[256, 16], byte_offset=0)
    k = n.Var(binding=n.VarBinding.LOOP, dtype=n.ScalarDType.I32)
    static = A[0:128, :]
    assert static.tensor.dtype == n.DType.F16
    assert static.shape == [128, 16]
    assert static.offsets == [0, 0]
    symbolic = A[k : k + 8, :]
    assert isinstance(symbolic.shape[0], n.ScalarExpr)  # (k+8) - k
    assert symbolic.shape[0].op == n.ScalarOp.SUB


def test_smem_tensor_requires_byte_offset_and_rejects_non_smem_offset():
    with pytest.raises(ValueError, match="byte_offset is required"):
        n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.F16, shape=[1])
    with pytest.raises(ValueError, match="only valid for SMEM"):
        n.Tensor(space=n.MemorySpace.GMEM, dtype=n.DType.F16, shape=[1], byte_offset=0)


def test_valid_kernel_builds_and_validates():
    s = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    body = (n.ScalarDef(var=s, initial=0), n.ScalarStore(var=s, value=5), n.CtaSync())
    k = n.Kernel(name="t", args=(), body=body, num_warps=4, launch_shape=[2], cluster_shape=[2])
    assert k.name == "t"
    assert k.num_warps == 4
    assert k.launch_cta_count() == 2
    k.validate()  # explicit re-validate is a no-op on a valid kernel


def test_invalid_kernel_rejected_on_construction():
    with pytest.raises(ValueError, match="num_warps"):
        n.Kernel(name="bad", body=(), num_warps=6, launch_shape=[1], cluster_shape=[1])


def _copy_kernel(dtype):
    inp = n.Tensor(space=n.MemorySpace.GMEM, dtype=dtype, shape=[128])
    out = n.Tensor(space=n.MemorySpace.GMEM, dtype=dtype, shape=[128])
    reg = n.Tensor(space=n.MemorySpace.REG, dtype=dtype, shape=[1])
    tid = n.ScopeValue(kind="tid_in_wg")
    body = (
        n.Role(body=(n.RegLoad(dst=reg[:], src=inp[tid]), n.RegStore(dst=out[tid], src=reg[:]))),
    )
    return (
        n.Kernel(
            name="copy",
            args=(inp, out),
            body=body,
            num_warps=4,
            launch_shape=[1],
            cluster_shape=[1],
        ),
        inp,
        out,
    )


def _round_bf16(values):
    f32 = values.astype(np.float32)
    bits = f32.view(np.uint32).astype(np.uint64)
    rounded = (bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))) & np.uint64(
        0xFFFF0000
    )
    return rounded.astype(np.uint32).view(np.float32)


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        (n.DType.F16, np.float16),
        (n.DType.BF16, np.float32),
        (n.DType.F32, np.float32),
        (n.DType.I32, np.int32),
        (n.DType.U32, np.uint32),
    ],
)
def test_interpret_returns_native_output_dtype(dtype, expected_dtype):
    kernel, inp, out = _copy_kernel(dtype)
    if dtype in (n.DType.F16, n.DType.BF16, n.DType.F32):
        values = np.linspace(-3.0, 3.0, 128, dtype=np.float32)
    elif dtype == n.DType.I32:
        values = np.arange(-64, 64, dtype=np.int32)
    else:
        values = np.arange(128, dtype=np.uint32)

    result = n.interpret(kernel, {inp: values})[out.id]
    assert result.dtype == np.dtype(expected_dtype)
    if dtype == n.DType.F16:
        assert np.array_equal(result, values.astype(np.float16))
    elif dtype == n.DType.BF16:
        assert np.array_equal(result, _round_bf16(values))
    else:
        assert np.array_equal(result, values.astype(expected_dtype))


@pytest.mark.parametrize(
    "dtype, scalar_dtype, values",
    [
        (n.DType.BOOL, n.ScalarDType.BOOL, np.array([1], dtype=np.int32)),
        (n.DType.I32, n.ScalarDType.I32, np.array([-7], dtype=np.int32)),
        (n.DType.U32, n.ScalarDType.U32, np.array([2**32 - 1], dtype=np.uint32)),
        (n.DType.I64, n.ScalarDType.I64, np.array([-9], dtype=np.int64)),
        (n.DType.U64, n.ScalarDType.U64, np.array([2**63], dtype=np.uint64)),
    ],
)
def test_interpret_accepts_nonfloat_gmem_scalar_initial(dtype, scalar_dtype, values):
    gmem = n.Tensor(space=n.MemorySpace.GMEM, dtype=dtype, shape=[1])
    scalar = n.Var(binding=n.VarBinding.SCALAR, dtype=scalar_dtype)
    body = (n.Role(body=(n.ScalarDef(var=scalar, initial=gmem[0]),)),)
    kernel = n.Kernel(
        name="scalar_init",
        args=(gmem,),
        body=body,
        num_warps=4,
        launch_shape=[1],
        cluster_shape=[1],
    )

    assert n.interpret(kernel, {gmem: values}) == {}
