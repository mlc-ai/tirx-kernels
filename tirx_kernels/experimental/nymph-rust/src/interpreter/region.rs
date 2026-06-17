//! Region projection for protocol traces.
//!
//! SMEM/GMEM regions use byte ranges, TMEM regions use physical
//! `(lane, lane_byte)` boxes, and REG regions use per-thread logical tensor
//! coordinates: `(register_row, dim0, dim1, ...)`.

use super::diagnostics::{IResult, InterpreterError};
use super::protocol::{BoxN, PoolId, Region, RegionBoxes};
use super::slice_indexing::{shared_flat_indices, ResolvedSlice};
use super::values::indexing::numel;
use super::values::smem::dtype_size_bytes;
use super::values::tmem::{tmem_layout_for, TMEM_COLS, TMEM_ROWS};
use crate::ir::{MemorySpace, Tensor, TmemLayoutKind};
use std::collections::BTreeSet;
use std::sync::Arc;

const TMEM_CELL_BYTES: usize = 4;
pub const TMEM_LANE_BYTES: usize = TMEM_COLS * TMEM_CELL_BYTES;

pub fn pool_for_tensor(tensor: &Tensor, cta_id: usize) -> PoolId {
    match tensor.space {
        MemorySpace::Smem => PoolId::Smem { cta_id },
        MemorySpace::Tmem => PoolId::Tmem { cta_id },
        MemorySpace::Gmem => PoolId::Gmem {
            tensor_id: tensor.id,
        },
        MemorySpace::Reg => PoolId::Reg {
            cta_id,
            tensor_id: tensor.id,
        },
    }
}

pub fn tensor_region(resolved: &ResolvedSlice, cta_id: usize) -> IResult<Region> {
    let offsets: Vec<Vec<i64>> = resolved
        .offsets
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect();
    if resolved.tensor.space == MemorySpace::Reg {
        let register_rows: Vec<usize> = (0..offsets.len()).collect();
        return reg_tensor_logical_region(
            &resolved.tensor,
            PoolId::Reg {
                cta_id,
                tensor_id: resolved.tensor.id,
            },
            &register_rows,
            &offsets,
            &resolved.shape,
        );
    }
    tensor_region_from_offsets(&resolved.tensor, cta_id, &offsets, &resolved.shape)
}

pub fn reg_tensor_region(
    resolved: &ResolvedSlice,
    cta_id: usize,
    register_rows: &[usize],
) -> IResult<Region> {
    let offsets: Vec<Vec<i64>> = resolved
        .offsets
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect();
    reg_tensor_region_from_offsets(
        &resolved.tensor,
        cta_id,
        register_rows,
        &offsets,
        &resolved.shape,
    )
}

pub fn reg_tensor_region_from_offsets(
    tensor: &Arc<Tensor>,
    cta_id: usize,
    register_rows: &[usize],
    offsets: &[Vec<i64>],
    shape: &[usize],
) -> IResult<Region> {
    reg_tensor_logical_region(
        tensor,
        PoolId::Reg {
            cta_id,
            tensor_id: tensor.id,
        },
        register_rows,
        offsets,
        shape,
    )
}

pub fn reg_tensor_region_from_uniform_offset(
    tensor: &Arc<Tensor>,
    cta_id: usize,
    register_row_start: usize,
    register_row_count: usize,
    offset: &[i64],
    shape: &[usize],
) -> IResult<Region> {
    if register_row_count == 0 {
        return Err(InterpreterError::new(
            "trace_region_empty_box",
            "REG trace box has an empty register-row dimension",
        ));
    }
    if offset.len() != tensor.shape.len() || shape.len() != tensor.shape.len() {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "REG trace region rank does not match tensor rank",
        ));
    }
    let row_end = register_row_start
        .checked_add(register_row_count)
        .ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "REG register-row range overflows")
        })?;
    let mut ranges = Vec::with_capacity(shape.len() + 1);
    ranges.push((register_row_start, row_end));
    for ((&off, &extent), &dim) in offset.iter().zip(shape.iter()).zip(tensor.shape.iter()) {
        if extent == 0 {
            return Err(InterpreterError::new(
                "trace_region_empty_box",
                "REG trace box has an empty dimension",
            ));
        }
        let start = usize::try_from(off).map_err(|_| {
            InterpreterError::new("trace_region_oob", "REG logical offset is negative")
        })?;
        let end = start.checked_add(extent).ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "REG logical range overflows")
        })?;
        if end > dim {
            return Err(InterpreterError::new(
                "trace_region_oob",
                "REG trace box is outside the tensor",
            ));
        }
        ranges.push((start, end));
    }
    non_empty_region(
        PoolId::Reg {
            cta_id,
            tensor_id: tensor.id,
        },
        RegionBoxes::Boxes(vec![BoxN::new(ranges)]),
        tensor.id,
    )
}

pub fn tensor_region_from_uniform(
    tensor: &Arc<Tensor>,
    cta_id: usize,
    offsets: &[usize],
    shape: &[usize],
) -> IResult<Region> {
    let offsets: Vec<i64> = offsets.iter().map(|&x| x as i64).collect();
    tensor_region_from_offsets(tensor, cta_id, &[offsets], shape)
}

/// TMA tensormap GMEM projection: the CLAMPED rectangular byte footprint of a
/// (coords, shape) box. The tensormap zero-fills out-of-bounds loads and
/// squashes out-of-bounds stores, so each dimension's extent clamps to the
/// tensor; the in-bounds part decomposes into contiguous row runs (one box per
/// run via `rect_byte_ranges` — a tile never projects to a single linear
/// interval unless it covers full inner rows). Returns `None` when the whole
/// box is out of bounds (no bytes touched).
pub fn tensor_rect_region_clamped(
    tensor: &Arc<Tensor>,
    cta_id: usize,
    coords: &[usize],
    shape: &[usize],
) -> IResult<Option<Region>> {
    if !matches!(tensor.space, MemorySpace::Gmem) {
        return Err(InterpreterError::new(
            "trace_region_owner",
            "clamped rectangle projection is only valid for GMEM tensors",
        ));
    }
    if coords.len() != tensor.shape.len() || shape.len() != tensor.shape.len() {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "clamped rectangle projection rank mismatch",
        ));
    }
    let valid: Vec<usize> = coords
        .iter()
        .zip(shape.iter())
        .zip(tensor.shape.iter())
        .map(|((&c, &s), &dim)| dim.saturating_sub(c).min(s))
        .collect();
    if valid.iter().any(|&v| v == 0) {
        return Ok(None);
    }
    let offsets: Vec<i64> = coords.iter().map(|&c| c as i64).collect();
    let elem_size = dtype_size_bytes(tensor.dtype);
    let boxes = rect_region_boxes(tensor, &[offsets], &valid, 0, elem_size)?.ok_or_else(|| {
        InterpreterError::new(
            "trace_region_rank",
            "clamped rectangle projection rank mismatch",
        )
    })?;
    let owner = pool_for_tensor(tensor, cta_id);
    Ok(Some(non_empty_region(owner, boxes, tensor.id)?))
}

pub fn tensor_region_from_offsets(
    tensor: &Arc<Tensor>,
    cta_id: usize,
    offsets: &[Vec<i64>],
    shape: &[usize],
) -> IResult<Region> {
    if offsets.is_empty() {
        return Err(InterpreterError::new(
            "trace_region_empty",
            "trace region projection produced no offsets",
        ));
    }
    let owner = pool_for_tensor(tensor, cta_id);
    match tensor.space {
        MemorySpace::Tmem => tmem_logical_region(tensor, owner, offsets, shape),
        MemorySpace::Reg => {
            let register_rows: Vec<usize> = (0..offsets.len()).collect();
            reg_tensor_logical_region(tensor, owner, &register_rows, offsets, shape)
        }
        MemorySpace::Smem | MemorySpace::Gmem => tensor_byte_region(tensor, owner, offsets, shape),
    }
}

pub fn tmem_region_from_cells<I>(tensor_id: u32, cta_id: usize, cells: I) -> IResult<Region>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut cell_vec: Vec<(usize, usize)> = Vec::new();
    for (lane, col) in cells {
        if lane >= TMEM_ROWS || col >= TMEM_COLS {
            return Err(InterpreterError::new(
                "trace_region_oob",
                "TMEM trace cell is outside the scratchpad",
            ));
        }
        cell_vec.push((lane, col));
    }
    cell_vec.sort_unstable();
    cell_vec.dedup();
    let mut boxes = Vec::new();
    let mut run_lane: Option<usize> = None;
    let mut run_start: Option<usize> = None;
    let mut prev_col: Option<usize> = None;
    for (lane, col) in cell_vec {
        match (run_lane, run_start, prev_col) {
            (Some(active_lane), Some(_), Some(last)) if lane == active_lane && col == last + 1 => {
                prev_col = Some(col);
            }
            (Some(active_lane), Some(start), Some(last)) => {
                boxes.push(tmem_box(active_lane, 1, start, last - start + 1));
                run_lane = Some(lane);
                run_start = Some(col);
                prev_col = Some(col);
            }
            _ => {
                run_lane = Some(lane);
                run_start = Some(col);
                prev_col = Some(col);
            }
        }
    }
    if let (Some(lane), Some(start), Some(last)) = (run_lane, run_start, prev_col) {
        boxes.push(tmem_box(lane, 1, start, last - start + 1));
    }
    non_empty_region(
        PoolId::Tmem { cta_id },
        RegionBoxes::Boxes(coalesced_tmem_boxes(boxes)),
        tensor_id,
    )
}

pub fn tmem_region_from_cell_arrays(
    tensor_id: u32,
    cta_id: usize,
    lanes: impl IntoIterator<Item = usize>,
    cols: impl IntoIterator<Item = usize>,
) -> IResult<Region> {
    let cells = lanes.into_iter().zip(cols);
    tmem_region_from_cells(tensor_id, cta_id, cells)
}

pub fn tmem_region_from_rects<I>(tensor_id: u32, cta_id: usize, rects: I) -> IResult<Region>
where
    I: IntoIterator<Item = (usize, usize, usize, usize)>,
{
    let mut boxes = Vec::new();
    for (lane_start, n_lanes, col_start, n_cols) in rects {
        if n_lanes == 0 || n_cols == 0 {
            return Err(InterpreterError::new(
                "trace_region_empty_box",
                "TMEM trace box has an empty dimension",
            ));
        }
        if lane_start + n_lanes > TMEM_ROWS || col_start + n_cols > TMEM_COLS {
            return Err(InterpreterError::new(
                "trace_region_oob",
                "TMEM trace box is outside the scratchpad",
            ));
        }
        boxes.push(tmem_box(lane_start, n_lanes, col_start, n_cols));
    }
    non_empty_region(
        PoolId::Tmem { cta_id },
        RegionBoxes::Boxes(coalesced_tmem_boxes(boxes)),
        tensor_id,
    )
}

pub fn tmem_allocation_region(
    tensor_id: u32,
    cta_id: usize,
    col_start: usize,
    n_cols: usize,
) -> IResult<Region> {
    tmem_region_from_rects(tensor_id, cta_id, [(0, TMEM_ROWS, col_start, n_cols)])
}

pub fn regions_overlap(left: &Region, right: &Region) -> bool {
    if left.owner != right.owner {
        return false;
    }
    // O(1) bounding-span reject (rank-1 sets are sorted by construction).
    if let (Some((la, lb)), Some((ra, rb))) =
        (left.boxes.rank1_bounds(), right.boxes.rank1_bounds())
    {
        if lb <= ra || rb <= la {
            return false;
        }
        // Equal-stride strided sets: decide in O(1) by phase, falling back to
        // direct candidate tests near the span boundary.
        if let (
            RegionBoxes::Strided {
                start: a0,
                len: alen,
                stride: astr,
                count: an,
            },
            RegionBoxes::Strided {
                start: b0,
                len: blen,
                stride: bstr,
                count: bn,
            },
        ) = (&left.boxes, &right.boxes)
        {
            if astr == bstr {
                return strided_same_period_intersect((*a0, *alen, *astr, *an), (*b0, *blen, *bn));
            }
        }
        // Strided vs a small box list: O(1) arithmetic per box.
        if let (
            RegionBoxes::Strided {
                start,
                len,
                stride,
                count,
            },
            RegionBoxes::Boxes(boxes),
        )
        | (
            RegionBoxes::Boxes(boxes),
            RegionBoxes::Strided {
                start,
                len,
                stride,
                count,
            },
        ) = (&left.boxes, &right.boxes)
        {
            return boxes.iter().any(|b| {
                let (bs, be) = b.ranges[0];
                strided_box_intersect((*start, *len, *stride, *count), (bs, be))
            });
        }
        // Mixed / unequal-stride rank-1 sets: lazy two-pointer over the run
        // iterators (no materialization).
        return sorted_rank1_intersect(left.boxes.iter_rank1(), right.boxes.iter_rank1());
    }
    // TMEM (rank-2) and mixed/empty cases: small box counts, order-independent
    // product (2-D rectangle sets have no 1-D sweep equivalent).
    let (RegionBoxes::Boxes(lb), RegionBoxes::Boxes(rb)) = (&left.boxes, &right.boxes) else {
        return false;
    };
    lb.iter().any(|l| rb.iter().any(|r| boxes_overlap(l, r)))
}

/// Equal-period run sets `a_i = a0 + i*s` (len `alen`) vs `b_j = b0 + j*s`
/// (len `blen`): when the spans overlap by at least one full period both
/// sequences place a run inside the shared window, so the phase test decides;
/// a narrower shared window holds at most a couple of candidate runs per
/// side, tested directly. Caller has already rejected disjoint spans.
fn strided_same_period_intersect(
    (a0, alen, s, an): (usize, usize, usize, usize),
    (b0, blen, bn): (usize, usize, usize),
) -> bool {
    let a_end = a0 + (an - 1) * s + alen;
    let b_end = b0 + (bn - 1) * s + blen;
    let lo = a0.max(b0);
    let hi = a_end.min(b_end);
    if hi.saturating_sub(lo) >= 2 * s {
        // Runs repeat every `s` bytes and coalescing guarantees len < stride,
        // so a window of two periods contains a full run of each: overlap is
        // exactly the circular-interval test of the phases.
        let d = (b0 + s - a0 % s) % s; // b's phase relative to a's runs at [0, alen)
        return d < alen || d + blen > s;
    }
    // Narrow shared window: test the few runs that can intersect it.
    let candidates = |x0: usize, xlen: usize, xn: usize| -> Vec<(usize, usize)> {
        let first = lo.saturating_sub(x0 + xlen) / s;
        (first..xn.min(first + 3))
            .map(|i| (x0 + i * s, x0 + i * s + xlen))
            .collect()
    };
    let a_runs = candidates(a0, alen, an);
    let b_runs = candidates(b0, blen, bn);
    a_runs
        .iter()
        .any(|&(asr, aer)| b_runs.iter().any(|&(bsr, ber)| asr < ber && bsr < aer))
}

/// One box `[bs, be)` vs the run progression `start + i*stride` (len `len`,
/// `i < count`): only the runs straddling the box's ends can intersect it,
/// and a box wider than one period always hits a run — O(1) arithmetic.
fn strided_box_intersect(
    (start, len, stride, count): (usize, usize, usize, usize),
    (bs, be): (usize, usize),
) -> bool {
    let last_end = start + (count - 1) * stride + len;
    if be <= start || bs >= last_end {
        return false;
    }
    if be.saturating_sub(bs) >= stride {
        return true; // spans a full period: contains a whole run
    }
    let i = bs.saturating_sub(start) / stride;
    for run in [i, i + 1] {
        if run >= count {
            break;
        }
        let rs = start + run * stride;
        if rs < be && bs < rs + len {
            return true;
        }
    }
    false
}

/// Two-pointer overlap test for two ascending rank-1 run iterators.
fn sorted_rank1_intersect(
    mut a: impl Iterator<Item = (usize, usize)>,
    mut b: impl Iterator<Item = (usize, usize)>,
) -> bool {
    let (mut cur_a, mut cur_b) = (a.next(), b.next());
    while let (Some((a_start, a_end)), Some((b_start, b_end))) = (cur_a, cur_b) {
        // Half-open overlap: [a_start, a_end) ∩ [b_start, b_end) != ∅.
        if a_start < b_end && b_start < a_end {
            return true;
        }
        // The interval ending first cannot meet any later interval of the other
        // list (their starts only increase), so retire it.
        if a_end <= b_end {
            cur_a = a.next();
        } else {
            cur_b = b.next();
        }
    }
    false
}

/// Bounding spans `(dim0, dim1)` of a region's boxes: rank-1 run sets report
/// `((0, 1), byte_span)`, rank-2 sets the per-dim envelope. Empty regions give
/// zero-width spans (touch nothing); mixed-rank sets a full span (reject
/// nothing). Computed once and cached by callers that scan frontiers, so the
/// O(boxes) cost is per-region, not per-comparison.
pub fn region_bounding_spans(region: &Region) -> ((usize, usize), (usize, usize)) {
    if let Some((lo, hi)) = region.boxes.rank1_bounds() {
        return ((0, 1), (lo, hi));
    }
    let RegionBoxes::Boxes(boxes) = &region.boxes else {
        return ((0, 0), (0, 0));
    };
    let mut d0 = (usize::MAX, 0);
    let mut d1 = (usize::MAX, 0);
    for b in boxes {
        if b.ranges.len() != 2 {
            return ((0, usize::MAX), (0, usize::MAX));
        }
        d0 = (d0.0.min(b.ranges[0].0), d0.1.max(b.ranges[0].1));
        d1 = (d1.0.min(b.ranges[1].0), d1.1.max(b.ranges[1].1));
    }
    if boxes.is_empty() {
        return ((0, 0), (0, 0));
    }
    (d0, d1)
}

/// O(1) reject: two regions can only overlap if both bounding spans touch.
pub fn bounding_spans_touch(
    a: ((usize, usize), (usize, usize)),
    b: ((usize, usize), (usize, usize)),
) -> bool {
    a.0 .0 < b.0 .1 && b.0 .0 < a.0 .1 && a.1 .0 < b.1 .1 && b.1 .0 < a.1 .1
}

/// O(1) reject for covers: `inner` can only be covered if its bounding spans
/// sit inside `outer`'s.
pub fn bounding_spans_contain(
    outer: ((usize, usize), (usize, usize)),
    inner: ((usize, usize), (usize, usize)),
) -> bool {
    outer.0 .0 <= inner.0 .0
        && inner.0 .1 <= outer.0 .1
        && outer.1 .0 <= inner.1 .0
        && inner.1 .1 <= outer.1 .1
}

pub fn region_covers(left: &Region, right: &Region) -> bool {
    if left.owner != right.owner {
        return false;
    }
    let (RegionBoxes::Boxes(lb), RegionBoxes::Boxes(rb)) = (&left.boxes, &right.boxes) else {
        // Equal-stride strided sets: covered iff the phases align so every
        // right run sits inside a left run — O(1).
        if let (
            RegionBoxes::Strided {
                start: a0,
                len: alen,
                stride: astr,
                count: an,
            },
            RegionBoxes::Strided {
                start: b0,
                len: blen,
                stride: bstr,
                count: bn,
            },
        ) = (&left.boxes, &right.boxes)
        {
            if astr == bstr {
                return b0 >= a0
                    && (b0 - a0) % astr + blen <= *alen
                    && b0 + (bn - 1) * astr <= a0 + (an - 1) * astr;
            }
        }
        // rank-1 run sets: both iterators ascend, so a single merge pass
        // checks every right run against the one left run that can hold it.
        let mut left_runs = left.boxes.iter_rank1();
        let mut cur = left_runs.next();
        for (rs, re) in right.boxes.iter_rank1() {
            while let Some((_, le)) = cur {
                if le > rs {
                    break;
                }
                cur = left_runs.next();
            }
            match cur {
                Some((ls, le)) if ls <= rs && re <= le => {}
                _ => return false,
            }
        }
        return true;
    };
    rb.iter().all(|r| lb.iter().any(|l| box_covers(l, r)))
}

pub fn boxes_overlap(left: &BoxN, right: &BoxN) -> bool {
    left.rank() == right.rank()
        && left
            .ranges
            .iter()
            .zip(&right.ranges)
            .all(|(&(ls, le), &(rs, re))| ls < re && rs < le)
}

pub fn box_covers(left: &BoxN, right: &BoxN) -> bool {
    left.rank() == right.rank()
        && left
            .ranges
            .iter()
            .zip(&right.ranges)
            .all(|(&(ls, le), &(rs, re))| ls <= rs && re <= le)
}

fn tensor_byte_region(
    tensor: &Arc<Tensor>,
    owner: PoolId,
    offsets: &[Vec<i64>],
    shape: &[usize],
) -> IResult<Region> {
    let offsets = unique_offsets(offsets);
    let elem_size = dtype_size_bytes(tensor.dtype);
    let base = match tensor.space {
        MemorySpace::Smem => tensor.byte_offset.ok_or_else(|| {
            InterpreterError::new("trace_region_layout", "SMEM tensor byte_offset is missing")
        })?,
        MemorySpace::Gmem | MemorySpace::Reg => 0,
        MemorySpace::Tmem => unreachable!(),
    };
    if let Some(boxes) = rect_region_boxes(tensor, &offsets, shape, base, elem_size)? {
        return non_empty_region(owner, boxes, tensor.id);
    }
    let resolved = ResolvedSlice {
        tensor: Arc::clone(tensor),
        offsets: offsets_array(&offsets)?,
        shape: shape.to_vec(),
    };
    let (idx, _) = shared_flat_indices(&resolved, &tensor.shape)?;
    let mut ranges = Vec::with_capacity(idx.len());
    let extent = numel(&tensor.shape).checked_mul(elem_size).ok_or_else(|| {
        InterpreterError::new("trace_region_overflow", "tensor byte extent overflows")
    })?;
    for flat in idx.iter().copied() {
        let start = base
            .checked_add(flat.checked_mul(elem_size).ok_or_else(|| {
                InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
            })?)
            .ok_or_else(|| {
                InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
            })?;
        let end = start.checked_add(elem_size).ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
        })?;
        if matches!(tensor.space, MemorySpace::Gmem | MemorySpace::Reg) && end > extent {
            return Err(InterpreterError::new(
                "trace_region_oob",
                "tensor byte region is outside the tensor",
            ));
        }
        ranges.push((start, end));
    }
    let boxes = coalesced_rank1_boxes(ranges);
    non_empty_region(owner, boxes, tensor.id)
}

fn unique_offsets(offsets: &[Vec<i64>]) -> Vec<Vec<i64>> {
    offsets
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Project a rank-N rectangular box to `RegionBoxes`: trailing dims the box
/// covers fully fold into one run with the first not-fully-covered dim; the
/// remaining outer dims are walked with an odometer, one byte run per outer
/// index — never one per element. With a single offset and at most one outer
/// dim of extent > 1 the runs form one arithmetic progression, so a
/// `RegionBoxes::Strided` is built directly without materializing the runs.
/// `None` only when the slice rank does not match the tensor (the caller
/// takes the per-element path). Bounds are strict: a box outside the tensor
/// errors (TMA's clamped projection handles tensormap OOB upstream).
fn rect_region_boxes(
    tensor: &Tensor,
    offsets: &[Vec<i64>],
    shape: &[usize],
    base: usize,
    elem_size: usize,
) -> IResult<Option<RegionBoxes>> {
    let rank = tensor.shape.len();
    if rank == 0 || shape.len() != rank || offsets.iter().any(|o| o.len() != rank) {
        return Ok(None);
    }
    if shape.iter().any(|&d| d == 0) {
        return Ok(Some(RegionBoxes::Boxes(Vec::new())));
    }
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * tensor.shape[i + 1];
    }
    let mut split = rank - 1;
    while split > 0 && shape[split] == tensor.shape[split] {
        split -= 1;
    }
    let run_bytes = shape[split]
        .checked_mul(strides[split])
        .and_then(|e| e.checked_mul(elem_size))
        .ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
        })?;
    let outer = &shape[..split];
    let corner_of = |offset: &[i64]| -> IResult<usize> {
        let mut corner = 0usize;
        for d in 0..rank {
            let o = usize::try_from(offset[d]).map_err(|_| {
                InterpreterError::new("trace_region_oob", "tensor byte region offset is negative")
            })?;
            if o + shape[d] > tensor.shape[d] {
                return Err(InterpreterError::new(
                    "trace_region_oob",
                    "tensor byte region is outside the tensor",
                ));
            }
            corner = corner
                .checked_add(o.checked_mul(strides[d]).ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
                })?)
                .ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
                })?;
        }
        Ok(corner)
    };
    let one_wide_dim = || -> Option<Option<usize>> {
        let mut wide = (0..split).filter(|&d| outer[d] > 1);
        let d = wide.next();
        match (d, wide.next()) {
            (_, Some(_)) => None,
            (d, None) => {
                if let Some(d) = d {
                    // runs must not overlap, or Strided's len < stride invariant breaks
                    if run_bytes > strides[d].saturating_mul(elem_size) {
                        return None;
                    }
                }
                Some(d)
            }
        }
    };
    if offsets.len() == 1 {
        if let Some(wide) = one_wide_dim() {
            let corner = corner_of(&offsets[0])?;
            let start = base
                .checked_add(corner.checked_mul(elem_size).ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
                })?)
                .ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
                })?;
            let single = |len: usize| -> IResult<Option<RegionBoxes>> {
                let end = start.checked_add(len).ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
                })?;
                Ok(Some(RegionBoxes::Boxes(vec![BoxN::new(vec![(
                    start, end,
                )])])))
            };
            let Some(d) = wide else {
                return single(run_bytes);
            };
            let stride_bytes = strides[d].checked_mul(elem_size).ok_or_else(|| {
                InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
            })?;
            let count = outer[d];
            if run_bytes == stride_bytes {
                // adjacent runs touch: one contiguous interval
                let total = run_bytes.checked_mul(count).ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
                })?;
                return single(total);
            }
            start
                .checked_add(stride_bytes.checked_mul(count - 1).ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
                })?)
                .and_then(|last| last.checked_add(run_bytes))
                .ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
                })?;
            return Ok(Some(RegionBoxes::Strided {
                start,
                len: run_bytes,
                stride: stride_bytes,
                count,
            }));
        }
    }
    let n_runs: usize = outer.iter().product();
    let mut ranges = Vec::with_capacity(offsets.len() * n_runs);
    for offset in offsets {
        let corner = corner_of(offset)?;
        let mut idx = vec![0usize; split];
        'outer: loop {
            let rel: usize = idx
                .iter()
                .zip(&strides[..split])
                .map(|(&i, &st)| i * st)
                .sum();
            let start = base
                .checked_add((corner + rel).checked_mul(elem_size).ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
                })?)
                .ok_or_else(|| {
                    InterpreterError::new("trace_region_overflow", "tensor byte offset overflows")
                })?;
            let end = start.checked_add(run_bytes).ok_or_else(|| {
                InterpreterError::new("trace_region_overflow", "tensor byte range overflows")
            })?;
            ranges.push((start, end));
            let mut d = split;
            loop {
                if d == 0 {
                    break 'outer;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < outer[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
    Ok(Some(coalesced_rank1_boxes(ranges)))
}

fn reg_tensor_logical_region(
    tensor: &Arc<Tensor>,
    owner: PoolId,
    register_rows: &[usize],
    offsets: &[Vec<i64>],
    shape: &[usize],
) -> IResult<Region> {
    let PoolId::Reg { .. } = owner else {
        unreachable!("REG tensor must project to REG owner")
    };
    if register_rows.len() != offsets.len() {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "REG register row count must match tensor offset rows",
        ));
    }
    let mut boxes = Vec::with_capacity(offsets.len());
    for (&register_row, offset) in register_rows.iter().zip(offsets.iter()) {
        boxes.push(reg_tensor_logical_box(tensor, register_row, offset, shape)?);
    }
    non_empty_region(
        owner,
        RegionBoxes::Boxes(coalesced_nd_boxes(boxes)),
        tensor.id,
    )
}

fn reg_tensor_logical_box(
    tensor: &Tensor,
    register_row: usize,
    offset: &[i64],
    shape: &[usize],
) -> IResult<BoxN> {
    if offset.len() != tensor.shape.len() || shape.len() != tensor.shape.len() {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "REG trace region rank does not match tensor rank",
        ));
    }
    let mut ranges = Vec::with_capacity(shape.len() + 1);
    ranges.push((register_row, register_row + 1));
    for ((&off, &extent), &dim) in offset.iter().zip(shape.iter()).zip(tensor.shape.iter()) {
        if extent == 0 {
            return Err(InterpreterError::new(
                "trace_region_empty_box",
                "REG trace box has an empty dimension",
            ));
        }
        let start = usize::try_from(off).map_err(|_| {
            InterpreterError::new("trace_region_oob", "REG logical offset is negative")
        })?;
        let end = start.checked_add(extent).ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "REG logical range overflows")
        })?;
        if end > dim {
            return Err(InterpreterError::new(
                "trace_region_oob",
                "REG trace box is outside the tensor",
            ));
        }
        ranges.push((start, end));
    }
    Ok(BoxN::new(ranges))
}

fn tmem_logical_region(
    tensor: &Arc<Tensor>,
    owner: PoolId,
    offsets: &[Vec<i64>],
    shape: &[usize],
) -> IResult<Region> {
    let PoolId::Tmem { cta_id } = owner else {
        unreachable!("TMEM tensor must project to TMEM owner")
    };
    let offsets = unique_offsets(offsets);
    let mut boxes = Vec::with_capacity(offsets.len());
    for offset in &offsets {
        boxes.push(tmem_logical_box(tensor, offset, shape)?);
    }
    non_empty_region(
        PoolId::Tmem { cta_id },
        RegionBoxes::Boxes(coalesced_tmem_byte_boxes(boxes)),
        tensor.id,
    )
}

fn tmem_logical_box(tensor: &Tensor, offset: &[i64], shape: &[usize]) -> IResult<BoxN> {
    if tensor.shape.len() != 2 || offset.len() != 2 || shape.len() != 2 {
        return Err(InterpreterError::new(
            "tmem_value",
            "TMEM access must be rank-2",
        ));
    }
    let row0 = usize::try_from(offset[0]).map_err(|_| {
        InterpreterError::new("trace_region_oob", "TMEM logical offset is negative")
    })?;
    let col0 = usize::try_from(offset[1]).map_err(|_| {
        InterpreterError::new("trace_region_oob", "TMEM logical offset is negative")
    })?;
    let (rows, cols) = (shape[0], shape[1]);
    if rows == 0 || cols == 0 {
        return Err(InterpreterError::new(
            "trace_region_empty_box",
            "TMEM trace box has an empty dimension",
        ));
    }
    if row0 + rows > tensor.shape[0] || col0 + cols > tensor.shape[1] {
        return Err(InterpreterError::new(
            "trace_region_oob",
            "TMEM slice is out of bounds",
        ));
    }
    let layout = tmem_layout_for(tensor)?;
    let lane_start = match layout.kind {
        TmemLayoutKind::Lane128 => row0,
        TmemLayoutKind::Lane64Upper => {
            if row0 + rows > 64 {
                return Err(InterpreterError::new(
                    "invalid_tmem_row",
                    "TMEM row out of range",
                ));
            }
            row0
        }
        TmemLayoutKind::Lane64Lower => {
            if row0 + rows > 64 {
                return Err(InterpreterError::new(
                    "invalid_tmem_row",
                    "TMEM row out of range",
                ));
            }
            row0 + 64
        }
        _ => {
            return Err(InterpreterError::new(
                "unsupported_tmem_layout",
                "scale-vector TMEM layouts are unsupported for trace access",
            ))
        }
    };
    let elem_size = dtype_size_bytes(tensor.dtype);
    if !matches!(elem_size, 2 | 4) {
        return Err(InterpreterError::new(
            "unsupported_tmem_dtype",
            "TMEM trace access supports 16-bit or 32-bit element widths",
        ));
    }
    let byte_start = layout
        .col_start
        .checked_mul(TMEM_CELL_BYTES)
        .and_then(|base| base.checked_add(col0.checked_mul(elem_size)?))
        .ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "TMEM byte offset overflows")
        })?;
    let byte_end = byte_start
        .checked_add(cols.checked_mul(elem_size).ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "TMEM byte range overflows")
        })?)
        .ok_or_else(|| {
            InterpreterError::new("trace_region_overflow", "TMEM byte range overflows")
        })?;
    if lane_start + rows > TMEM_ROWS || byte_end > TMEM_LANE_BYTES {
        return Err(InterpreterError::new(
            "trace_region_oob",
            "TMEM trace box is outside the scratchpad",
        ));
    }
    Ok(BoxN::new(vec![
        (lane_start, lane_start + rows),
        (byte_start, byte_end),
    ]))
}

fn offsets_array(offsets: &[Vec<i64>]) -> IResult<ndarray::Array2<i64>> {
    let rank = offsets.first().map(|o| o.len()).unwrap_or(0);
    if rank == 0 || offsets.iter().any(|o| o.len() != rank) {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "trace tensor offsets must have a uniform non-zero rank",
        ));
    }
    let values: Vec<i64> = offsets.iter().flat_map(|o| o.iter().copied()).collect();
    ndarray::Array2::from_shape_vec((offsets.len(), rank), values).map_err(|e| {
        InterpreterError::new(
            "trace_region_rank",
            format!("trace tensor offsets cannot form a matrix: {e}"),
        )
    })
}

/// Sort + coalesce a rank-1 run list, then compress: a uniform arithmetic
/// sequence of equal-length runs (the shape every narrow-inner rectangular
/// slice produces) becomes `RegionBoxes::Strided`; anything else stays an
/// explicit box list.
fn coalesced_rank1_boxes(mut ranges: Vec<(usize, usize)>) -> RegionBoxes {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        if let Some((_, last_end)) = out.last_mut() {
            if start <= *last_end {
                *last_end = (*last_end).max(end);
                continue;
            }
        }
        out.push((start, end));
    }
    if out.len() >= 2 {
        let (s0, e0) = out[0];
        let len = e0 - s0;
        let stride = out[1].0 - s0;
        if out
            .iter()
            .enumerate()
            .all(|(i, &(st, en))| st == s0 + i * stride && en - st == len)
        {
            return RegionBoxes::Strided {
                start: s0,
                len,
                stride,
                count: out.len(),
            };
        }
    }
    RegionBoxes::Boxes(
        out.into_iter()
            .map(|(s, e)| BoxN::new(vec![(s, e)]))
            .collect(),
    )
}

fn coalesced_tmem_boxes(mut boxes: Vec<BoxN>) -> Vec<BoxN> {
    boxes.sort_by_key(|b| (b.ranges[1].0, b.ranges[1].1, b.ranges[0].0, b.ranges[0].1));
    let mut out: Vec<BoxN> = Vec::new();
    for b in boxes {
        if let Some(last) = out.last_mut() {
            if last.ranges[1] == b.ranges[1] && last.ranges[0].1 == b.ranges[0].0 {
                last.ranges[0].1 = b.ranges[0].1;
                continue;
            }
        }
        out.push(b);
    }
    out
}

fn coalesced_nd_boxes(mut boxes: Vec<BoxN>) -> Vec<BoxN> {
    boxes.sort_by(|a, b| a.ranges.cmp(&b.ranges));
    let mut changed = true;
    while changed {
        changed = false;
        let mut out: Vec<BoxN> = Vec::new();
        'next_box: for b in boxes {
            for existing in &mut out {
                if try_merge_box(existing, &b) {
                    changed = true;
                    continue 'next_box;
                }
            }
            out.push(b);
        }
        boxes = out;
    }
    boxes
}

fn try_merge_box(left: &mut BoxN, right: &BoxN) -> bool {
    if left.rank() != right.rank() {
        return false;
    }
    let mut merge_dim = None;
    for (dim, (&l, &r)) in left.ranges.iter().zip(right.ranges.iter()).enumerate() {
        if l == r {
            continue;
        }
        if ranges_touch_or_overlap(l, r) {
            if merge_dim.is_some() {
                return false;
            }
            merge_dim = Some(dim);
        } else {
            return false;
        }
    }
    let Some(dim) = merge_dim else {
        return true;
    };
    left.ranges[dim].0 = left.ranges[dim].0.min(right.ranges[dim].0);
    left.ranges[dim].1 = left.ranges[dim].1.max(right.ranges[dim].1);
    true
}

fn ranges_touch_or_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 <= right.1 && right.0 <= left.1
}

fn coalesced_tmem_byte_boxes(mut boxes: Vec<BoxN>) -> Vec<BoxN> {
    boxes.sort_by_key(|b| (b.ranges[0].0, b.ranges[0].1, b.ranges[1].0, b.ranges[1].1));
    let mut row_merged: Vec<BoxN> = Vec::new();
    for b in boxes {
        if let Some(last) = row_merged.last_mut() {
            if last.ranges[0] == b.ranges[0] && b.ranges[1].0 <= last.ranges[1].1 {
                last.ranges[1].1 = last.ranges[1].1.max(b.ranges[1].1);
                continue;
            }
        }
        row_merged.push(b);
    }
    coalesced_tmem_boxes(row_merged)
}

fn tmem_box(lane_start: usize, n_lanes: usize, col_start: usize, n_cols: usize) -> BoxN {
    BoxN::new(vec![
        (lane_start, lane_start + n_lanes),
        (
            col_start * TMEM_CELL_BYTES,
            (col_start + n_cols) * TMEM_CELL_BYTES,
        ),
    ])
}

fn non_empty_region(owner: PoolId, boxes: RegionBoxes, tensor_id: u32) -> IResult<Region> {
    if boxes.is_empty() {
        return Err(InterpreterError::new(
            "trace_region_empty",
            "trace region projection produced no boxes",
        ));
    }
    Ok(Region::from_boxes(owner, boxes, tensor_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, Layout, TmemLayout};

    #[test]
    fn same_owner_regions_overlap() {
        let a = Region::new(PoolId::Smem { cta_id: 0 }, vec![BoxN::new(vec![(0, 8)])], 1);
        let b = Region::new(
            PoolId::Smem { cta_id: 0 },
            vec![BoxN::new(vec![(4, 12)])],
            2,
        );
        assert!(regions_overlap(&a, &b));
    }

    #[test]
    fn different_owner_regions_do_not_alias() {
        let a = Region::new(PoolId::Smem { cta_id: 0 }, vec![BoxN::new(vec![(0, 8)])], 1);
        let b = Region::new(PoolId::Smem { cta_id: 1 }, vec![BoxN::new(vec![(0, 8)])], 1);
        assert!(!regions_overlap(&a, &b));
    }

    #[test]
    fn unit_box_scatter_overlap() {
        let a = Region::new(
            PoolId::Tmem { cta_id: 0 },
            vec![
                BoxN::new(vec![(0, 1), (0, 4)]),
                BoxN::new(vec![(7, 8), (12, 16)]),
            ],
            9,
        );
        let b = Region::new(
            PoolId::Tmem { cta_id: 0 },
            vec![BoxN::new(vec![(7, 8), (12, 16)])],
            9,
        );
        assert!(regions_overlap(&a, &b));
    }

    #[test]
    fn box_cover_uses_all_dimensions() {
        let a = Region::new(
            PoolId::Tmem { cta_id: 0 },
            vec![BoxN::new(vec![(0, 16), (0, 64)])],
            9,
        );
        let b = Region::new(
            PoolId::Tmem { cta_id: 0 },
            vec![BoxN::new(vec![(4, 8), (16, 32)])],
            9,
        );
        assert!(region_covers(&a, &b));
    }

    #[test]
    fn f16_tmem_halves_use_lane_byte_precision() {
        let half = Arc::new(Tensor {
            id: 10,
            space: MemorySpace::Tmem,
            dtype: DType::F16,
            shape: vec![1, 2],
            layout: Some(Layout::Tmem(TmemLayout {
                kind: TmemLayoutKind::Lane128,
                col_start: 0,
                lane_align: 0,
            })),
            byte_offset: None,
        });
        let whole = Arc::new(Tensor {
            id: 11,
            space: MemorySpace::Tmem,
            dtype: DType::F32,
            shape: vec![1, 1],
            layout: Some(Layout::Tmem(TmemLayout {
                kind: TmemLayoutKind::Lane128,
                col_start: 0,
                lane_align: 0,
            })),
            byte_offset: None,
        });
        let lo = tensor_region_from_uniform(&half, 0, &[0, 0], &[1, 1]).unwrap();
        let hi = tensor_region_from_uniform(&half, 0, &[0, 1], &[1, 1]).unwrap();
        let f32_cell = tensor_region_from_uniform(&whole, 0, &[0, 0], &[1, 1]).unwrap();

        assert_eq!(lo.boxes.to_boxes()[0].ranges, vec![(0, 1), (0, 2)]);
        assert_eq!(hi.boxes.to_boxes()[0].ranges, vec![(0, 1), (2, 4)]);
        assert!(!regions_overlap(&lo, &hi));
        assert!(regions_overlap(&lo, &f32_cell));
        assert!(regions_overlap(&hi, &f32_cell));
    }

    #[test]
    fn strided_intersect_matches_materialized_bruteforce() {
        // Sweep (start, len, stride, count) pairs — equal and unequal strides,
        // boundary slivers, single-run degenerates — comparing the compressed
        // path against the same runs materialized as explicit boxes.
        let mk_strided = |start: usize, len: usize, stride: usize, count: usize| {
            let ranges: Vec<(usize, usize)> = (0..count)
                .map(|i| (start + i * stride, start + i * stride + len))
                .collect();
            let compressed = Region::from_boxes(
                PoolId::Smem { cta_id: 0 },
                coalesced_rank1_boxes(ranges.clone()),
                1,
            );
            let explicit = Region::new(
                PoolId::Smem { cta_id: 0 },
                ranges
                    .iter()
                    .map(|&(s, e)| BoxN::new(vec![(s, e)]))
                    .collect(),
                1,
            );
            (compressed, explicit)
        };
        let mut params = Vec::new();
        for &(s0, l, st, n) in &[
            (0usize, 16usize, 256usize, 128usize),
            (16, 16, 256, 128),
            (32, 16, 256, 1),
            (0, 4, 8, 3),
            (100, 10, 64, 5),
            (0, 64, 64, 4), // len == stride: coalesces to one box
        ] {
            params.push((s0, l, st, n));
        }
        // pseudo-random extras
        let mut state = 0x12345678u64;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let s0 = (state >> 33) as usize % 512;
            let st = 8 + (state >> 20) as usize % 120;
            let l = 1 + (state >> 10) as usize % st;
            let n = 1 + (state >> 3) as usize % 12;
            params.push((s0, l, st, n));
        }
        for &(as0, al, ast, an) in &params {
            for &(bs0, bl, bst, bn) in &params {
                let (ca, ea) = mk_strided(as0, al, ast, an);
                let (cb, eb) = mk_strided(bs0, bl, bst, bn);
                let want = regions_overlap(&ea, &eb);
                assert_eq!(
                    regions_overlap(&ca, &cb),
                    want,
                    "strided x strided ({as0},{al},{ast},{an}) vs ({bs0},{bl},{bst},{bn})"
                );
                assert_eq!(regions_overlap(&ca, &eb), want, "strided x boxes");
                assert_eq!(regions_overlap(&ea, &cb), want, "boxes x strided");
            }
        }
    }

    #[test]
    fn rank1_two_pointer_matches_bruteforce() {
        let mk = |ranges: &[(usize, usize)]| {
            Region::new(
                PoolId::Smem { cta_id: 0 },
                ranges
                    .iter()
                    .map(|&(s, e)| BoxN::new(vec![(s, e)]))
                    .collect(),
                1,
            )
        };
        // Reference: the original O(m*n) box-pair product the sweep replaces.
        let brute = |a: &Region, b: &Region| {
            a.owner == b.owner
                && a.boxes
                    .to_boxes()
                    .iter()
                    .any(|l| b.boxes.to_boxes().iter().any(|r| boxes_overlap(l, r)))
        };
        let cases: &[(&[(usize, usize)], &[(usize, usize)], bool)] = &[
            (&[(0, 4)], &[(4, 8)], false), // adjacent half-open: no overlap
            (&[(0, 5)], &[(4, 8)], true),  // genuine overlap
            (&[(0, 4), (8, 12), (16, 20)], &[(5, 7), (13, 15)], false), // strided, all in gaps
            (&[(0, 4), (8, 12), (16, 20)], &[(10, 11)], true), // hits a middle box
            (&[(0, 4), (8, 12)], &[(12, 16), (20, 24)], false), // touch at 12, no overlap
            (&[(100, 200)], &[(0, 50), (60, 90)], false), // entirely before
            (
                &[(0, 4), (8, 12), (16, 20), (24, 28)],
                &[(2, 3), (18, 19), (26, 27)],
                true,
            ),
        ];
        for (a, b, want) in cases {
            let (ra, rb) = (mk(a), mk(b));
            assert_eq!(regions_overlap(&ra, &rb), *want, "a={a:?} b={b:?}");
            assert_eq!(
                regions_overlap(&ra, &rb),
                brute(&ra, &rb),
                "two-pointer vs brute a={a:?} b={b:?}"
            );
            // Overlap is symmetric.
            assert_eq!(regions_overlap(&rb, &ra), *want, "sym a={a:?} b={b:?}");
        }
    }

    #[test]
    fn rank_mismatch_boxes_do_not_overlap() {
        let a = BoxN::new(vec![(0, 8)]);
        let b = BoxN::new(vec![(0, 8), (0, 4)]);
        assert!(!boxes_overlap(&a, &b));
    }
}
