//! OpenBLAS `cblas_sgemm` for the MMA hot path — linked via `build.rs`.
//! The MMA dominates at large K; a multithreaded BLAS sgemm (with `beta` to fuse
//! the accumulate) is ~10× the single-thread ndarray/matrixmultiply path.

use std::os::raw::c_int;

extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

extern "C" {
    fn openblas_set_num_threads(n: c_int);
}

const ROW_MAJOR: c_int = 101;
const NO_TRANS: c_int = 111;
const TRANS: c_int = 112;

/// Pin the OpenBLAS thread count once (the 224-core default contends badly on the
/// tiny MMA tiles). Idempotent.
pub fn set_threads(n: usize) {
    unsafe { openblas_set_num_threads(n as c_int) }
}

/// `C[m,n] = beta*C + lhs[m,k] @ rhs[n,k]ᵀ` with arbitrary strides/offsets into the
/// backing buffers — lets sgemm read the operands straight out of the SMEM tensor
/// buffers (lhs/rhs) and write straight into the TMEM grid (c), no temp copies.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_nt_strided(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    lhs_off: usize,
    lda: usize,
    rhs: &[f32],
    rhs_off: usize,
    ldb: usize,
    c: &mut [f32],
    c_off: usize,
    ldc: usize,
    beta: f32,
) {
    if m == 0 || n == 0 {
        return;
    }
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            lhs.as_ptr().add(lhs_off),
            lda as c_int,
            rhs.as_ptr().add(rhs_off),
            ldb as c_int,
            beta,
            c.as_mut_ptr().add(c_off),
            ldc as c_int,
        );
    }
}

/// `C[m,n] = beta*C + A[m,k] @ B[n,k]ᵀ`. All row-major, k-minor (lda=ldb=k, ldc=n).
/// `beta = 1.0` accumulates into C (fuses the MMA accumulate); `0.0` overwrites.
pub fn sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32], beta: f32) {
    assert!(
        a.len() >= m * k && b.len() >= n * k && c.len() >= m * n,
        "sgemm_nt buffer too small"
    );
    if m == 0 || n == 0 {
        return;
    }
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            k as c_int,
            beta,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}
