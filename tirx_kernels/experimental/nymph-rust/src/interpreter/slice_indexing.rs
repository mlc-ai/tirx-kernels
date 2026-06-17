//! Cohort-resolved tensor slices + vectorized flat-index math — port of
//! `interpreter/slice_indexing.py`.

use super::diagnostics::{IResult, InterpreterError};
use super::values::indexing::numel;
use super::values::tensors::row_major_strides;
use crate::ir::Tensor;
use ndarray::Array2;
use std::sync::Arc;

/// A slice resolved over a cohort: per-thread base offsets `[A, rank]` (i64) and a
/// uniform slice `shape`.
#[derive(Clone)]
pub struct ResolvedSlice {
    pub tensor: Arc<Tensor>,
    pub offsets: Array2<i64>,
    pub shape: Vec<usize>,
}

/// Build the `[A, k]` row-major flat-index matrix into a tensor of `tensor_shape`,
/// returning `(idx, k)`. Bounds-checked per dim (overflow-safe `offset > dim - extent`).
pub fn shared_flat_indices(
    resolved: &ResolvedSlice,
    tensor_shape: &[usize],
) -> IResult<(Array2<usize>, usize)> {
    let rank = tensor_shape.len();
    if resolved.shape.len() != rank || resolved.offsets.ncols() != rank {
        return Err(InterpreterError::new(
            "tensor_value",
            "tensor slice rank mismatch",
        ));
    }
    let a = resolved.offsets.nrows();

    // Per-dim out-of-bounds is checked once, at slice resolution
    // (`CohortContext::eval_slice`, which every `ResolvedSlice` flows through and which
    // runs in both value and trace mode), so it is not repeated here.

    let strides = row_major_strides(tensor_shape);
    // intra-slice flat offsets (shared by all threads)
    let intra = intra_offsets(&resolved.shape, &strides);
    let k = intra.len();
    if k == 0 {
        return Ok((Array2::zeros((a, 0)), 0));
    }
    let mut idx = Array2::<usize>::zeros((a, k));
    for ai in 0..a {
        let mut base: i64 = 0;
        for dim in 0..rank {
            base += resolved.offsets[[ai, dim]] * strides[dim] as i64;
        }
        for j in 0..k {
            idx[[ai, j]] = (base + intra[j]) as usize;
        }
    }
    Ok((idx, k))
}

pub fn uniform_contiguous_flat_range(
    resolved: &ResolvedSlice,
    tensor_shape: &[usize],
) -> IResult<Option<(usize, usize)>> {
    let rank = tensor_shape.len();
    if resolved.shape.len() != rank || resolved.offsets.ncols() != rank {
        return Err(InterpreterError::new(
            "tensor_value",
            "tensor slice rank mismatch",
        ));
    }
    if resolved.offsets.nrows() == 0 {
        return Ok(Some((0, 0)));
    }
    for ai in 1..resolved.offsets.nrows() {
        for d in 0..rank {
            if resolved.offsets[[ai, d]] != resolved.offsets[[0, d]] {
                return Ok(None);
            }
        }
    }

    let strides = row_major_strides(tensor_shape);
    let mut base: i64 = 0;
    for d in 0..rank {
        base += resolved.offsets[[0, d]] * strides[d] as i64;
    }
    let len = numel(&resolved.shape);
    if len <= 1 {
        return Ok(Some((base as usize, len)));
    }

    let first_varying = resolved
        .shape
        .iter()
        .position(|&extent| extent > 1)
        .unwrap_or(rank);
    for d in first_varying + 1..rank {
        if resolved.offsets[[0, d]] != 0 || resolved.shape[d] != tensor_shape[d] {
            return Ok(None);
        }
    }
    Ok(Some((base as usize, len)))
}

/// Row-major flat offsets of every slice element relative to the slice base, using
/// the FULL TENSOR strides (so it accounts for tensor layout).
fn intra_offsets(slice_shape: &[usize], strides: &[usize]) -> Vec<i64> {
    if slice_shape.iter().any(|&e| e == 0) {
        return Vec::new();
    }
    let mut coords: Vec<Vec<usize>> = vec![Vec::new()];
    for &extent in slice_shape {
        let mut next = Vec::with_capacity(coords.len() * extent);
        for prefix in &coords {
            for step in 0..extent {
                let mut c = prefix.clone();
                c.push(step);
                next.push(c);
            }
        }
        coords = next;
    }
    coords
        .iter()
        .map(|coord| {
            coord
                .iter()
                .zip(strides.iter())
                .map(|(c, s)| (*c * *s) as i64)
                .sum()
        })
        .collect()
}
