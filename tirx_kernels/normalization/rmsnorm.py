import math

import numpy as np

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import MBarrier, TMABar

eps = 1e-06
F16_BYTES = 2
F32_BYTES = 4
SM_COUNT = 152
SMEM_SIZE = 232448


def ceildiv(a, b):
    return (a + b - 1) // b


def get_cluster_n(elem, smem_capacity=220, dtype_width=16):
    if dtype_width != 16:
        raise ValueError(f"Unsupported dtype width: {dtype_width}")
    perSMLimit = smem_capacity
    thresholds = [
        (perSMLimit * 1 * 512, 1),
        (perSMLimit * 2 * 512, 2),
        (perSMLimit * 4 * 512, 4),
        (perSMLimit * 8 * 512, 8),
    ]
    for limit, cluster in thresholds:
        if elem <= limit:
            return cluster
    return 16


def prepare_data(batch_size, dim):
    import torch

    torch.manual_seed(42)
    input = torch.randn(batch_size, dim, dtype=torch.float16, device="cuda")
    weights = torch.randn(dim, dtype=torch.float16, device="cuda")
    return (input, weights)


def torch_impl(input, weights):
    import torch

    input_naive = input.clone().to(dtype=torch.float32, device="cuda")
    weights_naive = weights.clone().to(dtype=torch.float32, device="cuda")

    def func():
        variance = input_naive.pow(2).mean(dim=-1, keepdim=True)
        norm_factor = torch.rsqrt(variance + eps)
        scaled = input_naive * norm_factor
        output = (scaled * weights_naive).to(torch.float16)
        return output

    result = bench({"naive": func}, timer="event")
    ms = result["impls"].get("naive", float("nan"))
    print(f"torch time: {ms:.3f} ms")
    return func()


def flashinfer_impl(input, weights, batch_size, dim):
    import flashinfer
    import torch

    out = torch.empty((batch_size, dim), dtype=torch.float16, device="cuda")
    flashinfer_input = input.clone().to(dtype=torch.float16, device="cuda")
    flashinfer_weights = weights.clone().to(dtype=torch.float16, device="cuda")

    def func():
        return flashinfer.norm.rmsnorm(
            flashinfer_input, flashinfer_weights, eps, enable_pdl=False, out=out
        )

    result = bench({"flashinfer": func}, timer="event")
    ms = result["impls"].get("flashinfer", float("nan"))
    print(f"FlashInfer time: {ms:.3f} ms")
    return out


def quack_impl(input, weights, batch_size, dim):
    import quack
    import torch

    quack_input = input.clone().to(dtype=torch.float16, device="cuda")
    quack_weights = weights.clone().to(dtype=torch.float16, device="cuda")

    def func():
        return quack.rmsnorm(quack_input, quack_weights, eps=eps)

    result = bench({"quack": func}, timer="event")
    ms = result["impls"].get("quack", float("nan"))
    print(f"Quack time: {ms:.3f} ms")
    return func()


def tirx_dispatch_rmsnorm(dim: int, batch_size: int, SMEM_PER_CTA=220, MAX_THREADS=256):
    if dim % 256 == 0 and dim <= 8192:
        CLUSTER_N = get_cluster_n(dim, 40)
        MAX_THREADS = 128
        useTMA = 1
    else:
        MAX_THREADS = 512
        useTMA = 0
        CLUSTER_N = get_cluster_n(dim, 110)
        if CLUSTER_N >= 16:
            MAX_THREADS = 1024
            CLUSTER_N = get_cluster_n(dim, 220)
            if CLUSTER_N >= 16:
                raise ValueError(
                    f"Dimension {dim} is too large to fit within SMEM constraints with current cluster reduction scheme"
                )
    print("CLUSTER_N =", CLUSTER_N)
    if useTMA:
        print("Using TMA SMEM load for input")
    else:
        print("Using synchronous SMEM load for input")
    if dim % CLUSTER_N != 0:
        raise ValueError(f"Dimension {dim} must be divisible by cluster size {CLUSTER_N}")
    dim_per_cta = ceildiv(dim, CLUSTER_N)
    if useTMA and dim_per_cta % 256 != 0:
        raise ValueError(f"dim_per_cta={dim_per_cta} must be divisible by 256 for TMA")
    num_clusters = batch_size
    VECTOR_SIZE = math.gcd(16 // F16_BYTES, dim_per_cta, dim - dim_per_cta * (CLUSTER_N - 1))
    BLOCK_SIZE = min(MAX_THREADS, max(32, dim_per_cta // VECTOR_SIZE))
    b_dx = 32
    b_dy = ceildiv(BLOCK_SIZE, b_dx)
    TMA_TILE = min(256, dim_per_cta)
    NUM_TMA_CHUNKS = dim_per_cta // TMA_TILE
    NUM_INPUT_BARS = 1

    @T.prim_func
    def input_SMEM_TMA(input_ptr: T.handle, weight_ptr: T.handle, output_ptr: T.handle):
        """
        RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight
        Uses TMA to load input/weight from GMEM to SMEM.
        For large dim, shards N across a cluster of CTAs with cross-CTA reduction.
        """
        input_global = T.match_buffer(
            input_ptr, shape=[batch_size, dim], dtype="float16", scope="global"
        )
        weight_global = T.match_buffer(weight_ptr, shape=[dim], dtype="float16", scope="global")
        output_global = T.match_buffer(
            output_ptr, shape=[batch_size, dim], dtype="float16", scope="global"
        )
        cbx = T.local_cell("int32")
        cbx = 0
        if CLUSTER_N > 1:
            cbx = T.cta_id_in_cluster([CLUSTER_N])
        b_id = T.cta_id([num_clusters * CLUSTER_N])
        cluster_id = b_id // CLUSTER_N
        t_idx, t_idy = T.thread_id([b_dx, b_dy])
        cta_rank = T.meta_var(cbx)
        col_offset = T.meta_var(cbx * dim_per_cta)
        curr_dim = T.meta_var(dim_per_cta)
        INPUT_SMEM_BYTES = T.meta_var(curr_dim * F16_BYTES)
        CLUSTER_REDUCE_BYTES = T.meta_var(1 * F32_BYTES)
        SMEM_TOTAL = T.meta_var(128 + INPUT_SMEM_BYTES + b_dy * F32_BYTES + CLUSTER_REDUCE_BYTES)
        buf = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        pool = T.meta_var(T.SMEMPool(buf.data))
        input_bar = TMABar(pool, NUM_INPUT_BARS)
        pool.move_base_to(128)
        input_smem = pool.alloc([curr_dim], "float16", align=128)
        sum_sq_smem = pool.alloc([b_dy], "float32")
        cluster_reduce_smem = pool.alloc([1], "float32")
        input_bar.init(1)
        if CLUSTER_N > 1:
            T.cuda.cluster_sync()
        else:
            T.ptx.bar.sync(1, b_dx * b_dy)
        input_vec: T.f16[VECTOR_SIZE]
        weight_vec: T.f16[VECTOR_SIZE]
        x_vec: T.f32[VECTOR_SIZE]
        weight_vec_f32: T.f32[VECTOR_SIZE]
        mul_result: T.f32[VECTOR_SIZE]
        sum_sq: T.f32
        norm_factor: T.f32
        batch_idx = cluster_id
        if t_idx == 0 & t_idy == 0:
            tma_copy_in = T.meta_var({"dispatch": "tma", "mbar": input_bar.ptr_to([0])})
            for tma_chunk in T.serial(NUM_TMA_CHUNKS):
                tma_off = T.meta_var(tma_chunk * TMA_TILE)
                Tx.copy_async(
                    input_smem[tma_off : tma_off + TMA_TILE],
                    input_global[batch_idx, col_offset + tma_off : col_offset + tma_off + TMA_TILE],
                    **tma_copy_in,
                )
            input_bar.arrive(0, INPUT_SMEM_BYTES)
        input_bar.wait(0, 0)
        sum_sq = T.float32(0.0)
        for ki in T.serial(ceildiv(curr_dim, VECTOR_SIZE * b_dx * b_dy)):
            st = T.meta_var((ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE)
            if st < curr_dim:
                Tx.copy(input_vec[:], input_smem[st : st + VECTOR_SIZE])
                Tx.cast(x_vec[:], input_vec[:])
                for v_id in T.unroll(VECTOR_SIZE):
                    weight_vec_f32[v_id] = x_vec[v_id] * x_vec[v_id]
                for v_id in T.unroll(VECTOR_SIZE):
                    sum_sq = sum_sq + weight_vec_f32[v_id]
        sum_sq = T.cuda.cta_sum(sum_sq, b_dy, sum_sq_smem.ptr_to([0]))
        if CLUSTER_N > 1:
            if t_idy == 0 and t_idx == 0:
                cluster_reduce_smem[0] = sum_sq_smem[0]
            T.cuda.cluster_sync()
            if t_idy == 0:
                if t_idx < CLUSTER_N:
                    remote_ptr: T.let = T.reinterpret(
                        PointerType(PrimType("float32")),
                        T.ptx.map_shared_rank(cluster_reduce_smem.ptr_to([0]), t_idx),
                    )
                    remote_buf = T.decl_buffer([1], "float32", scope="shared", data=remote_ptr)
                    sum_sq = remote_buf[0]
                else:
                    sum_sq = T.float32(0)
                sum_sq = T.cuda.warp_sum(sum_sq, width=CLUSTER_N)
                if t_idx == 0:
                    sum_sq_smem[0] = sum_sq
            T.ptx.bar.sync(1, b_dx * b_dy)
            T.ptx.fence.proxy("shared")
            norm_factor = T.rsqrt(sum_sq_smem[0] / dim + eps)
        else:
            norm_factor = T.rsqrt(sum_sq / dim + eps)
        for ki in T.serial(ceildiv(dim_per_cta, VECTOR_SIZE * b_dx * b_dy)):
            st = T.meta_var((ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE)
            if st < dim_per_cta:
                Tx.copy(
                    weight_vec[:], weight_global[col_offset + st : col_offset + st + VECTOR_SIZE]
                )
                Tx.cast(weight_vec_f32[:], weight_vec[:])
                Tx.copy(input_vec[:], input_smem[st : st + VECTOR_SIZE])
                Tx.cast(x_vec[:], input_vec[:])
                Tx.mul(mul_result[:], x_vec[:], norm_factor)
                Tx.mul(mul_result[:], mul_result[:], weight_vec_f32[:])
                Tx.cast(input_vec[:], mul_result[:])
                Tx.copy(
                    output_global[batch_idx, col_offset + st : col_offset + st + VECTOR_SIZE],
                    input_vec[:],
                )

    @T.prim_func
    def input_SMEM_sync(input_ptr: T.handle, weight_ptr: T.handle, output_ptr: T.handle):
        """
        RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight
        Uses TMA to load input/weight from GMEM to SMEM.
        For large dim, shards N across a cluster of CTAs with cross-CTA reduction.
        """
        input_global = T.match_buffer(
            input_ptr, shape=[batch_size, dim], dtype="float16", scope="global"
        )
        weight_global = T.match_buffer(weight_ptr, shape=[dim], dtype="float16", scope="global")
        output_global = T.match_buffer(
            output_ptr, shape=[batch_size, dim], dtype="float16", scope="global"
        )
        cbx = T.local_cell("int32")
        cbx = 0
        if CLUSTER_N > 1:
            cbx = T.cta_id_in_cluster([CLUSTER_N])
        b_id = T.cta_id([num_clusters * CLUSTER_N])
        cluster_id = T.meta_var(b_id // CLUSTER_N)
        t_idx, t_idy = T.thread_id([b_dx, b_dy])
        cta_rank = T.meta_var(cbx)
        col_offset = T.meta_var(cbx * dim_per_cta)
        curr_dim = T.meta_var(dim_per_cta)
        INPUT_SMEM_BYTES = T.meta_var(curr_dim * F16_BYTES)
        CLUSTER_REDUCE_BYTES = T.meta_var(1 * F32_BYTES)
        SMEM_TOTAL = T.meta_var(128 + INPUT_SMEM_BYTES + b_dy * F32_BYTES + CLUSTER_REDUCE_BYTES)
        buf = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        pool = T.meta_var(T.SMEMPool(buf.data))
        pool.move_base_to(128)
        input_smem = pool.alloc([curr_dim], "float16", align=128)
        sum_sq_smem = pool.alloc([b_dy], "float32")
        cluster_reduce_smem = pool.alloc([1], "float32")
        input_vec: T.f16[VECTOR_SIZE]
        weight_vec: T.f16[VECTOR_SIZE]
        x_vec: T.f32[VECTOR_SIZE]
        weight_vec_f32: T.f32[VECTOR_SIZE]
        mul_result: T.f32[VECTOR_SIZE]
        sum_sq: T.f32
        norm_factor: T.f32
        batch_idx = T.meta_var(cluster_id)
        sum_sq = T.float32(0.0)
        for ki in T.serial(ceildiv(curr_dim, VECTOR_SIZE * b_dx * b_dy)):
            st = T.meta_var((ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE)
            if st < curr_dim:
                Tx.copy(
                    input_smem[st : st + VECTOR_SIZE],
                    input_global[batch_idx, col_offset + st : col_offset + st + VECTOR_SIZE],
                )
                Tx.copy(input_vec[:], input_smem[st : st + VECTOR_SIZE])
                Tx.cast(x_vec[:], input_vec[:])
                for v_id in T.unroll(VECTOR_SIZE):
                    sum_sq = sum_sq + x_vec[v_id] * x_vec[v_id]
        sum_sq = T.cuda.cta_sum(sum_sq, b_dy, sum_sq_smem.ptr_to([0]))
        if CLUSTER_N > 1:
            if t_idy == 0 and t_idx == 0:
                cluster_reduce_smem[0] = sum_sq_smem[0]
            T.cuda.cluster_sync()
            if t_idy == 0:
                if t_idx < CLUSTER_N:
                    remote_ptr: T.let = T.reinterpret(
                        PointerType(PrimType("float32")),
                        T.ptx.map_shared_rank(cluster_reduce_smem.ptr_to([0]), t_idx),
                    )
                    remote_buf = T.decl_buffer([1], "float32", scope="shared", data=remote_ptr)
                    sum_sq = remote_buf[0]
                else:
                    sum_sq = T.float32(0)
                sum_sq = T.cuda.warp_sum(sum_sq, width=CLUSTER_N)
                if t_idx == 0:
                    sum_sq_smem[0] = sum_sq
            T.ptx.bar.sync(1, b_dx * b_dy)
            T.ptx.fence.proxy("shared")
            norm_factor = T.rsqrt(sum_sq_smem[0] / dim + eps)
        else:
            norm_factor = T.rsqrt(sum_sq / dim + eps)
        for ki in T.serial(ceildiv(dim_per_cta, VECTOR_SIZE * b_dx * b_dy)):
            st = T.meta_var((ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE)
            if st < dim_per_cta:
                Tx.copy(
                    weight_vec[:], weight_global[col_offset + st : col_offset + st + VECTOR_SIZE]
                )
                Tx.cast(weight_vec_f32[:], weight_vec[:])
                Tx.copy(input_vec[:], input_smem[st : st + VECTOR_SIZE])
                Tx.cast(x_vec[:], input_vec[:])
                Tx.mul(mul_result[:], x_vec[:], norm_factor)
                Tx.mul(mul_result[:], mul_result[:], weight_vec_f32[:])
                Tx.cast(input_vec[:], mul_result[:])
                Tx.copy(
                    output_global[batch_idx, col_offset + st : col_offset + st + VECTOR_SIZE],
                    input_vec[:],
                )

    if useTMA:
        return input_SMEM_TMA
    else:
        return input_SMEM_sync


def tirx_original_impl(hidden_size, batch_size, SMEM_PER_CTA=220, MAX_THREADS=256):
    vec_size = math.gcd(16 // F16_BYTES, hidden_size)
    block_size = min(256, hidden_size // vec_size)
    bdx = 32
    bdy = ceildiv(block_size, 32)
    smem_size = (bdy + hidden_size) * F32_BYTES
    if smem_size > SMEM_SIZE:
        raise ValueError(
            f"SMEM usage for this dim exceeds limit of {SMEM_SIZE} bytes. Consider using a smaller dim."
        )

    @T.prim_func
    def rmsnorm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
        input_global = T.match_buffer(
            input_ptr, [batch_size, hidden_size], "float16", scope="global"
        )
        weight_global = T.match_buffer(weight_ptr, [hidden_size], "float16", scope="global")
        out_global = T.match_buffer(out_ptr, [batch_size, hidden_size], "float16", scope="global")
        bx = T.cta_id([SM_COUNT])
        tx, ty = T.thread_id([bdx, bdy])
        thread_id = T.meta_var(ty * bdx + tx)
        buf = T.alloc_buffer([smem_size], "uint8", scope="shared.dyn")
        pool = T.meta_var(T.SMEMPool(buf.data))
        x_smem = pool.alloc([hidden_size], "float32")
        sum_sq_smem = pool.alloc([bdy], "float32")
        input_vec: T.f16[vec_size]
        weight_vec: T.f16[vec_size]
        input_vec_f32: T.f32[vec_size]
        weight_vec_f32: T.f32[vec_size]
        x_vec: T.f32[vec_size]
        x_tmp: T.f32
        sum_sq: T.f32
        rms_norm: T.f32
        idx: T.i32
        idx = bx
        while idx < batch_size:
            sum_sq = T.float32(0.0)
            for ki in T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                for kv in T.unroll(vec_size):
                    input_vec[kv] = 0.0
                    x_vec[kv] = 0.0
                st = T.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                if st < hidden_size:
                    Tx.copy(input_vec[:], input_global[idx, st : st + vec_size])
                    Tx.cast(input_vec_f32[:], input_vec[:])
                    for kv in T.unroll(vec_size):
                        x_tmp = input_vec_f32[kv]
                        sum_sq = sum_sq + x_tmp * x_tmp
                        x_vec[kv] = x_tmp
                    Tx.copy(x_smem[st : st + vec_size], x_vec[:])
            sum_sq = T.cuda.warp_sum(sum_sq)
            sum_sq_smem[ty] = sum_sq
            T.ptx.bar.sync(1, bdx * bdy)
            T.ptx.fence.proxy("shared")
            if ty == 0:
                if tx < bdy:
                    sum_sq = sum_sq_smem[tx]
                else:
                    sum_sq = T.float32(0.0)
                sum_sq = T.cuda.warp_sum(sum_sq)
                sum_sq_smem[0] = sum_sq
            T.ptx.bar.sync(1, bdx * bdy)
            T.ptx.fence.proxy("shared")
            rms_norm = T.rsqrt(sum_sq_smem[0] / hidden_size + eps)
            for ki in T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                for kv in T.unroll(vec_size):
                    input_vec[kv] = 0.0
                    weight_vec_f32[kv] = 0.0
                    x_vec[kv] = 0.0
                st = T.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                if st < hidden_size:
                    Tx.copy(weight_vec[:], weight_global[st : st + vec_size])
                    Tx.copy(x_vec[:], x_smem[st : st + vec_size])
                    Tx.cast(weight_vec_f32[:], weight_vec[:])
                Tx.mul(input_vec_f32[:], x_vec[:], rms_norm)
                Tx.mul(input_vec_f32[:], input_vec_f32[:], weight_vec_f32[:])
                if st < hidden_size:
                    Tx.cast(input_vec[:], input_vec_f32[:])
                    Tx.copy(out_global[idx, st : st + vec_size], input_vec[:])
            T.ptx.bar.sync(1, bdx * bdy)
            idx = idx + SM_COUNT

    return rmsnorm


def tirx_input_DSMEM_write_TMA_wts_GMEM(
    dim: int, batch_size: int, SMEM_PER_CTA=220, MAX_THREADS=256
):
    CLUSTER_N = get_cluster_n(dim, SMEM_PER_CTA)
    print("tirx_input_DSMEM_TMA_wts_GMEM: CLUSTER_N =", CLUSTER_N)
    if dim % CLUSTER_N != 0:
        raise ValueError(f"Dimension {dim} must be divisible by cluster size {CLUSTER_N}")
    dim_per_cta = ceildiv(dim, CLUSTER_N)
    if dim_per_cta % 256 != 0:
        raise ValueError(f"dim_per_cta={dim_per_cta} must be divisible by 256 for TMA")
    num_clusters = batch_size
    VECTOR_SIZE = math.gcd(16 // F16_BYTES, dim_per_cta, dim - dim_per_cta * (CLUSTER_N - 1))
    BLOCK_SIZE = min(MAX_THREADS, max(32, dim_per_cta // VECTOR_SIZE))
    b_dx = 32
    b_dy = ceildiv(BLOCK_SIZE, b_dx)
    TMA_TILE = min(256, dim_per_cta)
    NUM_TMA_CHUNKS = dim_per_cta // TMA_TILE
    NUM_INPUT_BARS = 1
    NUM_CLUSTER_BARS = 1
    TOTAL_BARS = NUM_INPUT_BARS + NUM_CLUSTER_BARS

    @T.prim_func
    def rms_norm(input_ptr: T.handle, weight_ptr: T.handle, output_ptr: T.handle):
        """
        RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight
        Uses TMA to load input/weight from GMEM to SMEM.
        For large dim, shards N across a cluster of CTAs with cross-CTA reduction.
        """
        input_global = T.match_buffer(
            input_ptr, shape=[batch_size, dim], dtype="float16", scope="global"
        )
        weight_global = T.match_buffer(weight_ptr, shape=[dim], dtype="float16", scope="global")
        output_global = T.match_buffer(
            output_ptr, shape=[batch_size, dim], dtype="float16", scope="global"
        )
        cbx = T.local_cell("int32")
        cbx = 0
        if CLUSTER_N > 1:
            cbx = T.cta_id_in_cluster([CLUSTER_N])
        b_id = T.cta_id([num_clusters * CLUSTER_N])
        cluster_id = b_id // CLUSTER_N
        t_idx, t_idy = T.thread_id([b_dx, b_dy])
        cta_rank = T.meta_var(cbx)
        col_offset = T.meta_var(cbx * dim_per_cta)
        curr_dim = T.meta_var(dim_per_cta)
        INPUT_SMEM_BYTES = T.meta_var(curr_dim * F16_BYTES)
        CLUSTER_REDUCE_BYTES = T.meta_var(CLUSTER_N * F32_BYTES)
        SMEM_TOTAL = T.meta_var(128 + INPUT_SMEM_BYTES + b_dy * F32_BYTES + CLUSTER_REDUCE_BYTES)
        buf = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        pool = T.meta_var(T.SMEMPool(buf.data))
        input_bar = TMABar(pool, NUM_INPUT_BARS)
        cluster_bar = MBarrier(pool, NUM_CLUSTER_BARS)
        pool.move_base_to(128)
        input_smem = pool.alloc([curr_dim], "float16", align=128)
        sum_sq_smem = pool.alloc([b_dy], "float32")
        cluster_reduce_smem = pool.alloc([CLUSTER_N], "float32")
        input_bar.init(1)
        if CLUSTER_N > 1:
            cluster_bar.init(CLUSTER_N)
            T.ptx.fence.mbarrier_init()
        if CLUSTER_N > 1:
            T.cuda.cluster_sync()
        else:
            T.ptx.bar.sync(1, b_dx * b_dy)
        input_vec: T.f16[VECTOR_SIZE]
        weight_vec: T.f16[VECTOR_SIZE]
        x_vec: T.f32[VECTOR_SIZE]
        weight_vec_f32: T.f32[VECTOR_SIZE]
        mul_result: T.f32[VECTOR_SIZE]
        sum_sq: T.f32
        norm_factor: T.f32
        batch_idx = cluster_id
        if t_idx == 0 & t_idy == 0:
            tma_copy_in = T.meta_var({"dispatch": "tma", "mbar": input_bar.ptr_to([0])})
            for tma_chunk in T.serial(NUM_TMA_CHUNKS):
                tma_off = T.meta_var(tma_chunk * TMA_TILE)
                Tx.copy_async(
                    input_smem[tma_off : tma_off + TMA_TILE],
                    input_global[batch_idx, col_offset + tma_off : col_offset + tma_off + TMA_TILE],
                    **tma_copy_in,
                )
            input_bar.arrive(0, INPUT_SMEM_BYTES)
        input_bar.wait(0, 0)
        sum_sq = T.float32(0.0)
        for ki in T.serial(ceildiv(curr_dim, VECTOR_SIZE * b_dx * b_dy)):
            st = T.meta_var((ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE)
            if st < curr_dim:
                Tx.copy(input_vec[:], input_smem[st : st + VECTOR_SIZE])
                Tx.cast(x_vec[:], input_vec[:])
                for v_id in T.unroll(VECTOR_SIZE):
                    sum_sq = sum_sq + x_vec[v_id] * x_vec[v_id]
        sum_sq = T.cuda.warp_sum(sum_sq)
        sum_sq_smem[t_idy] = sum_sq
        T.ptx.bar.sync(1, b_dx * b_dy)
        T.ptx.fence.proxy("shared")
        if t_idy == 0:
            if t_idx < b_dy:
                sum_sq = sum_sq_smem[t_idx]
            else:
                sum_sq = T.float32(0.0)
            sum_sq = T.cuda.warp_sum(sum_sq)
            sum_sq_smem[0] = sum_sq
        T.ptx.bar.sync(1, b_dx * b_dy)
        T.ptx.fence.proxy("shared")
        if CLUSTER_N > 1:
            if t_idy == 0:
                if t_idx < CLUSTER_N:
                    remote_ptr: T.let = T.reinterpret(
                        PointerType(PrimType("float32")),
                        T.ptx.map_shared_rank(cluster_reduce_smem.ptr_to([cta_rank]), t_idx),
                    )
                    remote_buf = T.decl_buffer([1], "float32", scope="shared", data=remote_ptr)
                    remote_buf[0] = sum_sq_smem[0]
                    T.ptx.mbarrier.arrive(cluster_bar.ptr_to([0]), remote=t_idx, pred=True)
            cluster_bar.wait(0, 0)
            if t_idy == 0:
                if t_idx < CLUSTER_N:
                    sum_sq = cluster_reduce_smem[t_idx]
                else:
                    sum_sq = T.float32(0)
                sum_sq = T.cuda.warp_sum(sum_sq, width=CLUSTER_N)
                if t_idx == 0:
                    sum_sq_smem[0] = sum_sq
            T.ptx.bar.sync(1, b_dx * b_dy)
            T.ptx.fence.proxy("shared")
            norm_factor = T.rsqrt(sum_sq_smem[0] / dim + eps)
        else:
            norm_factor = T.rsqrt(sum_sq_smem[0] / dim + eps)
        for ki in T.serial(ceildiv(dim_per_cta, VECTOR_SIZE * b_dx * b_dy)):
            st = T.meta_var((ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE)
            if st < dim_per_cta:
                Tx.copy(
                    weight_vec[:], weight_global[col_offset + st : col_offset + st + VECTOR_SIZE]
                )
                Tx.cast(weight_vec_f32[:], weight_vec[:])
                Tx.copy(input_vec[:], input_smem[st : st + VECTOR_SIZE])
                Tx.cast(x_vec[:], input_vec[:])
                Tx.mul(mul_result[:], x_vec[:], norm_factor)
                Tx.mul(mul_result[:], mul_result[:], weight_vec_f32[:])
                Tx.cast(input_vec[:], mul_result[:])
                Tx.copy(
                    output_global[batch_idx, col_offset + st : col_offset + st + VECTOR_SIZE],
                    input_vec[:],
                )

    return rms_norm


def build_tirx_soln(
    func, input_cat, weights, funcstr: str, dim: int, batch_size: int
) -> tuple[np.ndarray, tvm.runtime.Executable]:
    import torch

    input_cat_tir = input_cat.cuda() if not input_cat.is_cuda else input_cat
    weights_tir = weights.cuda() if not weights.is_cuda else weights
    output_tir = torch.empty((batch_size, dim), dtype=torch.float16, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": func(dim, batch_size)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        def run():
            return mod(input_cat_tir, weights_tir, output_tir)

        result = bench({f"tirx_soln_{funcstr}": run}, timer="event")
        ms = result["impls"].get(f"tirx_soln_{funcstr}", float("nan"))
        print(f"{funcstr} time: {ms:.3f} ms")

    return (output_tir, mod)


def test(batch_size: int, dim: int = 16384):
    import torch

    input, weights = prepare_data(batch_size, dim)
    print(f"----Testing Batch Size {batch_size}, Dim {dim}----")
    output_torch = torch_impl(input, weights)
    output_flashinfer = flashinfer_impl(input, weights, batch_size, dim)
    output_quack = quack_impl(input, weights, batch_size, dim)
    output_tirx_original, tirx_primfunc_1 = build_tirx_soln(
        tirx_original_impl, input, weights, "TIRX_original_impl", dim, batch_size
    )
    output_tirx_dispatch_rmsnorm, tirx_primfunc_2 = build_tirx_soln(
        tirx_dispatch_rmsnorm, input, weights, "TIRX_dispatch_rmsnorm", dim, batch_size
    )
    torch.testing.assert_close(output_flashinfer, output_torch, rtol=0.005, atol=0.005)
    torch.testing.assert_close(output_quack, output_torch, rtol=0.005, atol=0.005)
    torch.testing.assert_close(output_tirx_original, output_torch, rtol=0.005, atol=0.005)
    torch.testing.assert_close(output_tirx_dispatch_rmsnorm, output_torch, rtol=0.005, atol=0.005)


KERNEL_META = {"name": "rmsnorm", "category": "normalization", "compute_capability": 10}
CONFIGS = [
    {"hidden_size": hs, "batch_size": bs, "label": f"hs{hs}_bs{bs}"}
    for hs in [128, 4096, 5120, 8192]
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 4113]
]


def _get_rmsnorm_kernel(hidden_size):
    """Registry-compatible kernel factory (dynamic batch_size)."""
    vec_size = math.gcd(16 // F16_BYTES, hidden_size)
    block_size = min(256, hidden_size // vec_size)
    bdx = 32
    bdy = ceildiv(block_size, 32)

    @T.prim_func
    def rmsnorm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
        batch_size = T.int32()
        input_global = T.match_buffer(
            input_ptr, [batch_size, hidden_size], "float16", scope="global"
        )
        weight_global = T.match_buffer(weight_ptr, [hidden_size], "float16", scope="global")
        out_global = T.match_buffer(out_ptr, [batch_size, hidden_size], "float16", scope="global")
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        tx, ty = T.thread_id([bdx, bdy])
        thread_id = T.meta_var(ty * bdx + tx)
        pool = T.SMEMPool()
        x_smem = pool.alloc([hidden_size], "float32")
        sum_sq_smem = pool.alloc([bdy], "float32")
        pool.commit()
        input_vec: T.f16[vec_size]
        weight_vec: T.f16[vec_size]
        input_vec_f32: T.f32[vec_size]
        weight_vec_f32: T.f32[vec_size]
        x_vec: T.f32[vec_size]
        x_tmp: T.f32
        sum_sq: T.f32
        rms_norm: T.f32
        idx: T.i32
        idx = bx
        while idx < batch_size:
            sum_sq = 0.0
            for ki in T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                for kv in T.unroll(vec_size):
                    input_vec[kv] = 0.0
                    x_vec[kv] = 0.0
                st = T.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                if st < hidden_size:
                    Tx.copy(input_vec[:], input_global[idx, st : st + vec_size])
                    Tx.cast(input_vec_f32[:], input_vec[:])
                    for kv in T.unroll(vec_size):
                        x_tmp = input_vec_f32[kv]
                        sum_sq = sum_sq + x_tmp * x_tmp
                        x_vec[kv] = x_tmp
                    Tx.copy(x_smem[st : st + vec_size], x_vec[:])
            sum_sq = T.cuda.cta_sum(sum_sq, bdy, sum_sq_smem.ptr_to([0]))
            rms_norm = T.rsqrt(sum_sq / hidden_size + eps)
            for ki in T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                for kv in T.unroll(vec_size):
                    input_vec[kv] = 0.0
                    weight_vec_f32[kv] = 0.0
                    x_vec[kv] = 0.0
                st = T.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                if st < hidden_size:
                    Tx.copy(weight_vec[:], weight_global[st : st + vec_size])
                    Tx.copy(x_vec[:], x_smem[st : st + vec_size])
                    Tx.cast(weight_vec_f32[:], weight_vec[:])
                Tx.mul(input_vec_f32[:], x_vec[:], rms_norm)
                Tx.mul(input_vec_f32[:], input_vec_f32[:], weight_vec_f32[:])
                if st < hidden_size:
                    Tx.cast(input_vec[:], input_vec_f32[:])
                    Tx.copy(out_global[idx, st : st + vec_size], input_vec[:])
            T.ptx.bar.sync(1, bdx * bdy)
            idx = idx + SM_COUNT

    return rmsnorm


def get_kernel(hidden_size, **kwargs):
    return _get_rmsnorm_kernel(hidden_size)


def run_test(hidden_size, batch_size, **kwargs):
    """Compile, run, and verify rmsnorm kernel."""
    import torch

    from tirx_kernels.runner import compile_kernel

    input_data, weights = prepare_data(batch_size, hidden_size)
    kernel = _get_rmsnorm_kernel(hidden_size)
    ex = compile_kernel(kernel)
    output_tir = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")
    ex(input_data, weights, output_tir)
    torch.cuda.synchronize()
    input_f32 = input_data.to(torch.float32).cuda()
    variance = input_f32.pow(2).mean(dim=-1, keepdim=True)
    ref = (input_f32 * torch.rsqrt(variance + eps) * weights.float().cuda()).to(torch.float16)
    torch.testing.assert_close(output_tir.cpu(), ref.cpu(), rtol=0.001, atol=0.001)


# timer=None inherits the global default (proton). Proton matters here: rmsnorm is a
# tiny (~2µs) kernel whose event wall is ~3x inflated by launch overhead, and its
# reference is flashinfer (Python-dispatch-heavy). Proton measures the true ~2µs kernel
# time and an undistorted ratio.
def run_bench(hidden_size, batch_size, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark rmsnorm kernel."""

    import torch

    from tirx_kernels.runner import compile_kernel

    kernel = _get_rmsnorm_kernel(hidden_size)
    ex = compile_kernel(kernel)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    input_data, weights = prepare_data(batch_size, hidden_size)
    input_cuda = input_data.cuda()
    weights_cuda = weights.cuda()
    output_cuda = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")

    funcs = {"tir": lambda: ex(input_cuda, weights_cuda, output_cuda)}

    def _flashinfer():
        import flashinfer

        out_fi = torch.zeros_like(input_cuda)
        return lambda: flashinfer.norm.rmsnorm(
            input_cuda, weights_cuda, eps, enable_pdl=False, out=out_fi
        )

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer": _flashinfer},
        **kwargs,
    )


if __name__ == "__main__":
    for batch_size, dim in [(2048, 8192)]:
        test(batch_size, dim)
