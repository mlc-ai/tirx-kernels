import numpy as np
import nymph_rs as nr
import pytest
from helpers import builder, expect_runtime_error, gmem_arg, reg_tensor, run, u32


def test_value_inputs_seed_only_kernel_args_and_ignore_extra_inputs():
    b = builder("input_seed_only_args")
    source = b.tensor(space=nr.MemorySpace.GMEM, dtype=nr.DType.U32, shape=(1,))
    out = gmem_arg(b, shape=(1,))
    reg = reg_tensor(b)

    with b.role(warp=0, elected=True):
        b.reg_load(reg[0], source[0])
        b.reg_store(out[0], reg[0])

    with expect_runtime_error("missing_tensor_value"):
        run(b.build(), {source: u32([7])})


def test_scalar_tensor_initial_missing_input_fails_closed():
    b = builder("scalar_tensor_initial")
    source = gmem_arg(b, shape=(32,))
    with b.role(warp=0):
        b.scalar(initial=source[b.lane_id()], dtype=nr.ScalarDType.U32)

    with expect_runtime_error("missing_input"):
        run(b.build())


def test_bad_input_shape_is_rejected_at_python_boundary():
    b = builder("bad_input_metadata")
    input_tensor = gmem_arg(b, shape=(4,))

    with pytest.raises(ValueError, match="input array shape mismatch"):
        run(b.build(), {input_tensor: np.asarray([1, 2, 3], dtype=np.uint32)})
