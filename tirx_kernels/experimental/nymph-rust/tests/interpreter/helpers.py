import re
from contextlib import contextmanager

import numpy as np
import pytest

nr = pytest.importorskip("nymph_rs")


DEFAULT_KWARGS = {"num_warps": 4, "smem_size_bytes": 0, "launch_shape": (1,), "cluster_shape": (1,)}


def builder(name: str, **kwargs):
    options = dict(DEFAULT_KWARGS)
    options.update(kwargs)
    return nr.IRBuilder(name, **options)


@contextmanager
def expect_runtime_error(code: str):
    with pytest.raises(RuntimeError) as exc_info:
        yield
    message = str(exc_info.value)
    assert re.search(rf'reason=Some\("{re.escape(code)}"\)', message)
    assert "diagnostics:" in message
    assert f"[{code}]" in message


def run(kernel, inputs=None):
    return nr.interpret(kernel, inputs or {})


def output(outputs, tensor):
    return np.asarray(outputs[tensor.id])


def assert_output_eq(outputs, tensor, expected, dtype=None):
    expected = np.asarray(expected, dtype=dtype)
    np.testing.assert_array_equal(output(outputs, tensor), expected)


def u32(values, shape=None):
    arr = np.asarray(values, dtype=np.uint32)
    return arr.reshape(shape) if shape is not None else arr


def i32(values, shape=None):
    arr = np.asarray(values, dtype=np.int32)
    return arr.reshape(shape) if shape is not None else arr


def f32(values, shape=None):
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(shape) if shape is not None else arr


def tmem_layout(*, col_start=0, lane_align=0):
    return nr.TmemLayout(nr.TmemLayoutKind.LANE_128, col_start, lane_align)


def tmem_tensor(b, *, dtype=nr.DType.F32, shape=(128, 128), col_start=0, lane_align=0):
    return b.tensor(
        space=nr.MemorySpace.TMEM,
        dtype=dtype,
        shape=shape,
        layout=tmem_layout(col_start=col_start, lane_align=lane_align),
    )


def reg_tensor(b, *, dtype=nr.DType.U32, shape=(1,)):
    return b.tensor(space=nr.MemorySpace.REG, dtype=dtype, shape=shape)


def smem_tensor(b, *, dtype=nr.DType.U32, shape=(1,), byte_offset=0):
    return b.tensor(space=nr.MemorySpace.SMEM, dtype=dtype, shape=shape, byte_offset=byte_offset)


def gmem_arg(b, *, dtype=nr.DType.U32, shape=(1,)):
    return b.arg(space=nr.MemorySpace.GMEM, dtype=dtype, shape=shape)


def cta_eq(b, ctaid: int):
    return b.ctaid_in_cluster().eq(ctaid)
