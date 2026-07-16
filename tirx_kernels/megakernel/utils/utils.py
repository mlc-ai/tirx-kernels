# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Utility functions for megakernel."""

import numpy as np

import tvm
from tvm.script import tirx as T
from tvm.tirx import Expr


def ceildiv(a, b):
    if isinstance(a, Expr) or isinstance(b, Expr):
        return T.truncdiv(a + b - 1, b)
    return (a + b - 1) // b


# task_type: [0:5], m_idx: [5:18], n_idx: [18:28], k_idx: [28:32]
MAX_TASK_TYPE = 1 << 5
MAX_M_IDX = 1 << 13
MAX_N_IDX = 1 << 10
MAX_K_IDX = 1 << 4


def pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=True, debug=False):
    if host:
        if debug:
            assert (
                task_type < MAX_TASK_TYPE
                and m_idx < MAX_M_IDX
                and n_idx < MAX_N_IDX
                and k_idx < MAX_K_IDX
            )
        return (
            np.int64([task_type | (m_idx << 5) | (n_idx << 18) | (k_idx << 28)])
            .astype(np.int32)
            .item()
        )
    else:
        if debug:
            T.cuda.trap_when_assert_failed(task_type < MAX_TASK_TYPE)
            T.cuda.trap_when_assert_failed(m_idx < MAX_M_IDX)
            T.cuda.trap_when_assert_failed(n_idx < MAX_N_IDX)
            T.cuda.trap_when_assert_failed(k_idx < MAX_K_IDX)
        return task_type | (m_idx << 5) | (n_idx << 18) | (k_idx << 28)


unpack_from_32bit_code = """
__forceinline__ __device__ void unpack_from_32bit(int32_t task_info, int32_t* task_type_ptr, int32_t* m_idx_ptr, int32_t* n_idx_ptr, int32_t* k_idx_ptr) {
    *task_type_ptr = task_info & 0b11111;
    *m_idx_ptr = (task_info >> 5) & 0b1111111111111;
    *n_idx_ptr = (task_info >> 18) & 0b1111111111;
    *k_idx_ptr = (task_info >> 28) & 0b1111;
}
"""


@T.inline
def unpack_from_32bit(task_info, task_type_ptr, m_idx_ptr, n_idx_ptr, k_idx_ptr):
    T.cuda.func_call(
        "unpack_from_32bit",
        task_info,
        task_type_ptr,
        m_idx_ptr,
        n_idx_ptr,
        k_idx_ptr,
        source_code=unpack_from_32bit_code,
    )


def is_power_of_two(n: T.int32):
    return tvm.tirx.all(n > 0, T.bitwise_and(n, n - 1) == 0)


def next_power_of_two(x):
    return 1 << (x - 1).bit_length()


def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1


def rsqrt(x):
    return T.cuda.func_call(
        "_rsqrt",
        x,
        source_code="""
__forceinline__ __device__ float _rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
""",
        return_type="float32",
    )


def exp2(x):
    return T.cuda.func_call(
        "ptx_exp2",
        x,
        source_code="""
    __forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
""",
        return_type="float32",
    )


def silu(x):
    return T.cuda.func_call(
        "silu",
        x,
        source_code="""
    __forceinline__ __device__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}
""",
        return_type="float32",
    )


def threadfence_block():
    return T.cuda.func_call(
        "threadfence_block",
        source_code="""
__forceinline__ __device__ void threadfence_block() {
  __threadfence_block();
}
""",
    )


def any_sync(mask, pred):
    return T.cuda.func_call(
        "any_sync",
        mask,
        pred,
        source_code="""
__forceinline__ __device__ int any_sync(unsigned mask, int pred) {
  return __any_sync(mask, pred);
}
""",
        return_type="int32",
    )


def gt(lhs, rhs):
    return T.cuda.func_call(
        "gt",
        lhs,
        rhs,
        source_code="""
__forceinline__ __device__ bool gt(int32_t a, int32_t b) {
    return a > b;
}
""",
        return_type="bool",
    )


def mbarrier_try_wait(mbarrier, phase):
    return T.cuda.func_call(
        "tvm_builtin_ptx_mbarrier_try_wait_check",
        mbarrier,
        phase,
        source_code="""
__forceinline__ __device__ bool tvm_builtin_ptx_mbarrier_try_wait_check(void* barrier, int phase) {
    uint32_t smem_int_ptr = __cvta_generic_to_shared(barrier);
  uint32_t waitComplete;

  asm volatile(
      "{\\n\\t"
      ".reg .pred P1; \\n\\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \\n\\t"
      "selp.b32 %0, 1, 0, P1; \\n\\t"
      "}"
      : "=r"(waitComplete)
      : "r"(smem_int_ptr), "r"(phase));

  return static_cast<bool>(waitComplete);
}
""",
        return_type="bool",
    )


def atomic_add_int32_remote(addr, value, pe):
    func = """
__forceinline__ __device__ int32_t atomic_add_int32_remote(int32_t* addr, int32_t value, int32_t pe) {
    if (pe >= 0) {
        int32_t* ptr = (int32_t*)(nvshmem_ptr(addr, pe));
        int32_t old_value;
        asm volatile ("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old_value)
                        : "l"(ptr), "r"(value)
                        : "memory");
        return old_value;
    } else {
        return atomicAdd(addr, value);
    }
}
"""
    return T.cuda.func_call(
        "atomic_add_int32_remote", addr, value, pe, source_code=func, return_type="int32"
    )


def atomic_add_int32_local_release(addr, value):
    func = """
__forceinline__ __device__ int32_t atomic_add_int32_release(int32_t* addr, int32_t value) {
    int32_t old_value;
    asm volatile ("atom.release.gpu.global.add.s32 %0, [%1], %2;"
                  : "=r"(old_value)
                  : "l"(addr), "r"(value)
                  : "memory");
    return old_value;
}
"""
    return T.cuda.func_call(
        "atomic_add_int32_release", addr, value, source_code=func, return_type="int32"
    )


def atomic_add_int32_local(addr, value):
    func = """
__forceinline__ __device__ int32_t atomic_add_int32(int32_t* addr, int32_t value) {
    return atomicAdd(addr, value);
}
"""
    return T.cuda.func_call("atomic_add_int32", addr, value, source_code=func, return_type="int32")


def is_const_minus_one(value):
    return (isinstance(value, int) and value == -1) or (
        isinstance(value, tvm.tirx.IntImm) and value.value == -1
    )


def atomic_add_int32(addr, value, pe, release=False):
    if is_const_minus_one(pe):
        if release:
            return atomic_add_int32_local_release(addr, value)
        else:
            return atomic_add_int32_local(addr, value)
    else:
        print(f"pe is not -1: {pe}, {type(pe)}")
        return atomic_add_int32_remote(addr, value, pe)


def stg_remote(v, dst_addr, pe):
    func = """
__forceinline__ __device__ void stg_remote(int32_t v, void* dst_addr, int32_t pe) {
    if (pe >= 0) {
        void* ptr = nvshmem_ptr(dst_addr, pe);
        asm volatile("st.global.release.sys.b32 [%0], %1;"
                     :
                     : "l"(ptr), "r"(v)
                     : "memory");
    } else {
        asm volatile("st.global.release.gpu.b32 [%0], %1;"
                     :
                     : "l"(dst_addr), "r"(v)
                     : "memory");
    }
}
"""
    return T.cuda.func_call("stg_remote", v, dst_addr, pe, source_code=func)


def stg_local(v, dst_addr, pe):
    func = """
    __forceinline__ __device__ void stg_local(int32_t v, void* dst_addr, int32_t pe) {
        asm volatile("st.global.release.gpu.b32 [%0], %1;"
                    :
                    : "l"(dst_addr), "r"(v)
                    : "memory");
    }
    """
    return T.cuda.func_call("stg_local", v, dst_addr, pe, source_code=func)


def stg(v, dst_addr, pe):
    if is_const_minus_one(pe):
        return stg_local(v, dst_addr, pe)
    else:
        return stg_remote(v, dst_addr, pe)


@T.inline
def while_ld_global_acquire(addr, task_info):
    T.cuda.func_call(
        "while_ld_global_acquire",
        addr,
        task_info,
        source_code="""
__forceinline__ __device__ void while_ld_global_acquire(int32_t* addr, int32_t* task_info) {
  asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(*task_info) : "l"(addr) : "memory");
  while (*task_info == -1) {
    __nanosleep(800);
    asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(*task_info) : "l"(addr) : "memory");
  }
}
""",
    )


@T.inline
def sts(value, dst_addr):
    T.cuda.func_call(
        "sts",
        value,
        dst_addr,
        source_code="""
__forceinline__ __device__ void sts(int32_t v, void* dst_addr) {
    asm volatile("st.shared.b32 [%0], %1;"
                 :
                 : "l"(dst_addr), "r"(v)
                 : "memory");
}
""",
    )


def f_init_const(c):
    return lambda *args: c


def f_init_unmatched_dim(dim_len, in_par_size, out_par_size):
    def f_init(i):
        start_out_par = i * out_par_size
        end_out_par = T.min(dim_len, (i + 1) * out_par_size)
        return (end_out_par - 1) // in_par_size - start_out_par // in_par_size + 1

    return f_init


def get_source(module: "tvm.ir.IRModule"):
    target = tvm.target.Target("cuda")
    lib = tvm.compile(module, target, tir_pipeline="tirx")
    src = lib.mod.imports[0].inspect_source()
    return src, lib


def get_source_func(func: "tvm.tirx.PrimFunc"):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod
