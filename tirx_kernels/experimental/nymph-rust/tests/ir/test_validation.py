"""Validation-parity tests: the Rust validator must reject the same invalid IR
the original ir.py did.

This captures the value of test_builder.py's ~180 `pytest.raises` assertions
without re-implementing Python-dataclass introspection: each test builds an
invalid kernel from the nymph_rs constructors and asserts Kernel construction
(which validates) raises with the expected message.
"""

import pytest

n = pytest.importorskip("nymph_rs")

# ---- small tensor/slice helpers -------------------------------------------


def smem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.SMEM, dtype=dtype, shape=shape, byte_offset=0)


def tmem(shape, dtype=n.DType.F32):
    return n.Tensor(space=n.MemorySpace.TMEM, dtype=dtype, shape=shape)


def gmem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.GMEM, dtype=dtype, shape=shape)


def reg(shape, dtype=n.DType.F32):
    return n.Tensor(space=n.MemorySpace.REG, dtype=dtype, shape=shape)


def make(body, *, num_warps=4, launch=(2,), cluster=(2,), args=()):
    """Build a kernel (validates on construction)."""
    return n.Kernel(
        name="t",
        args=args,
        body=tuple(body),
        num_warps=num_warps,
        smem_size_bytes=1 << 20,
        launch_shape=list(launch),
        cluster_shape=list(cluster),
    )


def mma_operands():
    """A valid cta_group=1 MMA's (dst, a, b) slices — m=128, n=256, k=16."""
    return tmem([128, 256])[:, :], smem([128, 16])[:, :], smem([256, 16])[:, :]


# ---- kernel geometry -------------------------------------------------------


def test_rejects_num_warps_not_multiple_of_4():
    with pytest.raises(ValueError, match="num_warps"):
        make([], num_warps=6)


def test_rejects_launch_shape_rank_too_high():
    with pytest.raises(ValueError, match="rank must be in"):
        make([], launch=(2, 2, 2, 2), cluster=(1, 1, 1, 1))


def test_rejects_launch_not_divisible_by_cluster():
    with pytest.raises(ValueError, match="divisible by cluster_shape"):
        make([], launch=(6,), cluster=(4,))


def test_rejects_smem_tensor_outside_pool():
    tensor = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.U32, shape=[4], byte_offset=16)
    with pytest.raises(ValueError, match="byte range exceeds"):
        n.Kernel(
            name="bad_smem_bounds",
            body=(n.TensorDef(tensor),),
            num_warps=4,
            smem_size_bytes=20,
            launch_shape=[1],
            cluster_shape=[1],
        )


# ---- tcgen05_mma -----------------------------------------------------------


def test_rejects_mma_dst_not_tmem():
    _, a, b = mma_operands()
    dst = smem([128, 256])[:, :]
    with pytest.raises(ValueError, match="dst must be TMEM"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_operand_not_smem_or_tmem():
    dst, _, b = mma_operands()
    a = gmem([128, 16])[:, :]
    with pytest.raises(ValueError, match="operands must be SMEM or TMEM"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_accepts_mma_tmem_operand():
    dst, _, b = mma_operands()
    a = tmem([128, 16], dtype=n.DType.F16)[:, :]
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_operand_dtype():
    dst, _, b = mma_operands()
    a = smem([128, 16], dtype=n.DType.F32)[:, :]
    with pytest.raises(ValueError, match="operand dtype must be f16, bf16, or f8e4m3"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_dst_dtype():
    _, a, b = mma_operands()
    dst = tmem([128, 256], dtype=n.DType.F16)[:, :]
    with pytest.raises(ValueError, match="dst dtype must be f32"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_bad_k():
    dst, a, b = mma_operands()
    with pytest.raises(ValueError, match="k must be 16"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=8)])


# ---- tma ------------------------------------------------------------------


def tma_load(dst, src, mbar):
    return n.TmaLoad(dst=dst, src=src, mbar=mbar, bytes=16384, coords=(0, 0), shape=[128, 64])


def test_rejects_tma_dst_not_smem():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="dst must be SMEM"):
        make([tma_load(gmem([128, 64])[:, :], gmem([1024, 1024]), mbar)])


def test_rejects_tma_src_not_gmem():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="src must be GMEM"):
        make([tma_load(smem([128, 64])[:, :], smem([1024, 1024]), mbar)])


def test_rejects_tma_dtype_mismatch():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="dtype must match"):
        make([tma_load(smem([128, 64])[:, :], gmem([1024, 1024], dtype=n.DType.F32), mbar)])


def test_rejects_tma_mbar_kind():
    mbar = n.MBar(kind=n.MBarKind.TCGEN05)
    with pytest.raises(ValueError, match="mbar kind must be tma"):
        make([tma_load(smem([128, 64])[:, :], gmem([1024, 1024]), mbar)])


# ---- reg ops ---------------------------------------------------------------


def test_rejects_reg_add_dst_not_reg():
    s = smem([16, 16])[:, :]
    with pytest.raises(ValueError, match="dst must be REG"):
        make([n.RegAdd(dst=s, lhs=reg([16, 16])[:, :], rhs=reg([16, 16])[:, :])])


def test_rejects_reg_cvt_src_not_f32():
    dst = reg([16, 16], dtype=n.DType.F16)[:, :]
    src = reg([16, 16], dtype=n.DType.F16)[:, :]
    with pytest.raises(ValueError, match="src dtype must be f32"):
        make([n.RegCvt(dst=dst, src=src)])


def test_rejects_reg_cvt_dst_dtype():
    dst = reg([16, 16], dtype=n.DType.F32)[:, :]
    src = reg([16, 16], dtype=n.DType.F32)[:, :]
    with pytest.raises(ValueError, match="dst dtype must be f16 or bf16"):
        make([n.RegCvt(dst=dst, src=src)])


def test_reg_softmax_rescale_accepts_f32_scale_threshold_with_f16_rows():
    row_max = reg([1], dtype=n.DType.F16)[:]
    row_scale = reg([1], dtype=n.DType.F16)[:]
    old = reg([1], dtype=n.DType.F16)[:]
    new = reg([1], dtype=n.DType.F16)[:]
    scale_log2 = reg([1], dtype=n.DType.F32)[:]
    threshold = reg([1], dtype=n.DType.F32)[:]

    make(
        [
            n.RegSoftmaxRescale(
                row_max=row_max,
                row_scale=row_scale,
                row_max_old=old,
                row_max_new=new,
                scale_log2=scale_log2,
                threshold=threshold,
            )
        ]
    )


def test_rejects_reg_softmax_rescale_non_f32_scale_threshold():
    row_max = reg([1], dtype=n.DType.F16)[:]
    row_scale = reg([1], dtype=n.DType.F16)[:]
    old = reg([1], dtype=n.DType.F16)[:]
    new = reg([1], dtype=n.DType.F16)[:]
    bad_scale_log2 = reg([1], dtype=n.DType.F16)[:]
    bad_threshold = reg([1], dtype=n.DType.F16)[:]

    with pytest.raises(ValueError, match="scale_log2 dtype must be F32"):
        make(
            [
                n.RegSoftmaxRescale(
                    row_max=row_max,
                    row_scale=row_scale,
                    row_max_old=old,
                    row_max_new=new,
                    scale_log2=bad_scale_log2,
                    threshold=reg([1], dtype=n.DType.F32)[:],
                )
            ]
        )
    with pytest.raises(ValueError, match="threshold dtype must be F32"):
        make(
            [
                n.RegSoftmaxRescale(
                    row_max=row_max,
                    row_scale=row_scale,
                    row_max_old=old,
                    row_max_new=new,
                    scale_log2=reg([1], dtype=n.DType.F32)[:],
                    threshold=bad_threshold,
                )
            ]
        )


# ---- mbarrier + tmem alloc -------------------------------------------------


def test_rejects_mbarrier_init_zero_count():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="must be a positive integer"):
        make([n.MBarrierInit(mbar, count=0)])


def test_rejects_tmem_alloc_bad_ncols():
    with pytest.raises(ValueError, match=r"power-of-two integer in \[32, 512\]"):
        make([n.TmemAlloc(tmem([128, 256]), n_cols=33)])


# ---- role / scope ----------------------------------------------------------


def test_rejects_role_both_warp_and_warpgroup():
    with pytest.raises(ValueError, match="cannot set both"):
        make([n.Role(body=(), warp=0, warpgroup=0)])


def test_rejects_role_maxnreg_without_warpgroup():
    with pytest.raises(ValueError, match="maxnreg requires"):
        make([n.Role(body=(), warp=0, maxnreg=64)])


def test_rejects_role_warp_out_of_range():
    with pytest.raises(ValueError, match=r"warp must be in \[0, kernel num_warps\)"):
        make([n.Role(body=(), warp=10)])


def test_rejects_wg_sync_bad_barrier_id():
    with pytest.raises(ValueError, match=r"barrier_id must be an integer in \[1, 15\]"):
        make([n.WgSync(barrier_id=99)])


def test_rejects_cta_sync_in_warp_scope():
    with pytest.raises(ValueError, match="cta_sync must be in CTA scope"):
        make([n.KernelInit(body=(n.CtaSync(),), warp=0)])


def test_rejects_cta_sync_inside_role():
    with pytest.raises(ValueError, match="cta_sync cannot be used inside role"):
        make([n.Role(body=(n.CtaSync(),))])


def test_rejects_tmem_alloc_outside_warp_scope():
    with pytest.raises(ValueError, match="must be in warp scope"):
        make([n.TmemAlloc(tmem([128, 256]), n_cols=64)])  # at CTA scope


# ---- tcgen05 ld/st ---------------------------------------------------------


def test_accepts_non_32x32b_tcgen05_ld_st_atom():
    tm = tmem([128, 32], dtype=n.DType.U32)
    frag = reg([4], dtype=n.DType.U32)
    make(
        [
            n.Role(
                body=(
                    n.Tcgen05Ld(dst=frag[:], src=tm, shape="16x128b", num=2),
                    n.Tcgen05St(dst=tm, src=frag[:], shape="16x128b", num=2),
                ),
                warpgroup=0,
            )
        ],
        launch=(1,),
        cluster=(1,),
    )


def test_rejects_invalid_tcgen05_ld_st_atom_num():
    tm = tmem([128, 32], dtype=n.DType.U32)
    frag = reg([128], dtype=n.DType.U32)
    with pytest.raises(ValueError, match="shape/num"):
        make([n.Role(body=(n.Tcgen05Ld(dst=frag[:], src=tm, shape="16x128b", num=128),))])


# ---- ldmatrix / stmatrix ---------------------------------------------------


def test_accepts_ldstmatrix_m8n8_b16_atoms():
    sm = smem([32, 8], dtype=n.DType.U16)
    frag = reg([4], dtype=n.DType.U32)
    make(
        [
            n.Role(
                body=(
                    n.LdMatrix(dst=frag[0:4], src=sm[0, 0:8], num=4, trans=True),
                    n.StMatrix(dst=sm[0, 0:8], src=frag[0:4], num=4, trans=True),
                ),
                warp=0,
            )
        ],
        launch=(1,),
        cluster=(1,),
    )


def test_rejects_ldstmatrix_bad_spaces_shapes_and_dtype():
    sm = smem([8, 8], dtype=n.DType.U16)
    frag = reg([4], dtype=n.DType.U32)
    with pytest.raises(ValueError, match="dst must be REG"):
        make([n.Role(body=(n.LdMatrix(dst=sm[0, 0:4], src=sm[0, 0:8], num=1),), warp=0)])
    with pytest.raises(ValueError, match="src slice must contain one row"):
        make([n.Role(body=(n.LdMatrix(dst=frag[0:1], src=sm[0, 0:4], num=1),), warp=0)])
    with pytest.raises(ValueError, match="src dtype must be i32/u32 words or a b16 fragment"):
        make([n.Role(body=(n.StMatrix(dst=sm[0, 0:8], src=reg([1], dtype=n.DType.F32)[:]),))])


# ---- var definedness -------------------------------------------------------


def test_rejects_scalar_store_to_undefined_var():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="defined before use"):
        make([n.ScalarStore(var=v, value=5)])


def test_rejects_var_defined_twice():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="defined more than once"):
        make([n.ScalarDef(var=v, initial=0), n.ScalarDef(var=v, initial=1)])


# ---- cross-statement walks -------------------------------------------------


def test_rejects_inconsistent_cta_group():
    # two tmem allocs (in warp scope) with different cta_group
    body = (
        n.TmemAlloc(tmem([128, 256]), n_cols=64, cta_group=1),
        n.TmemAlloc(tmem([128, 256]), n_cols=64, cta_group=2),
    )
    with pytest.raises(ValueError, match="cta_group must be consistent"):
        make([n.Role(body=body, warp=0)])


def test_rejects_if_branching_on_role_scope():
    cond = n.ScopeValue(kind="warp_id")
    with pytest.raises(ValueError, match="cannot branch on role scope"):
        make([n.If(cond=cond, then_body=())])


def test_rejects_loop_nonpositive_step():
    i = n.Var()
    with pytest.raises(ValueError, match="step must be positive"):
        make([n.ForLoop(var=i, start=0, stop=10, step=0, body=())])


# ---- #1: scalar_def tensor initial dtype checks ----------------------------


def test_rejects_scalar_def_nonscalar_tensor_dtype():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    init = gmem([1, 1], dtype=n.DType.F16)[:, :]  # f16 is not a scalar integer/bool
    with pytest.raises(ValueError, match="dtype must be scalar integer or bool"):
        make([n.ScalarDef(var=v, initial=init)])


def test_rejects_scalar_def_var_dtype_mismatch():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.U32)
    init = gmem([1, 1], dtype=n.DType.I32)[:, :]  # decodes to i32, but var is u32
    with pytest.raises(ValueError, match="var dtype must match"):
        make([n.ScalarDef(var=v, initial=init)])


# ---- #2: mbar stages / arrive_count (eager, at MBar construction) ----------


def test_rejects_mbar_zero_stages():
    with pytest.raises(ValueError, match="stages must be a positive integer"):
        n.MBar(kind=n.MBarKind.TMA, stages=0)


def test_rejects_mbar_zero_arrive_count():
    with pytest.raises(ValueError, match="arrive_count must be a positive integer"):
        n.MBar(kind=n.MBarKind.TMA, arrive_count=0)
