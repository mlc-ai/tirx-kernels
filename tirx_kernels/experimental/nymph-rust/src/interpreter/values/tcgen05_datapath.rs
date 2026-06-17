//! tcgen05 ld/st datapath index arrays — port of
//! `interpreter/values/tcgen05_datapath.py`.
//!
//! The non-`32x32b` atoms are transcribed from the same CUTLASS SM100 TMEM copy
//! traits used by the Python model and verified there against B200 silicon.

use super::super::diagnostics::{IResult, InterpreterError};
use ndarray::Array2;
use std::collections::HashSet;
use std::sync::OnceLock;

const DP_B: usize = 1 << 21;

#[derive(Clone)]
struct Atom {
    val_shape: Vec<usize>,
    val_stride: Vec<usize>,
    dst_thr_shape: Vec<usize>,
    dst_thr_stride: Vec<usize>,
    dst_val_shape: Vec<usize>,
    dst_val_stride: Vec<usize>,
}

impl Atom {
    fn role_size(&self) -> usize {
        product(&self.dst_thr_shape)
    }

    fn reg_size(&self) -> usize {
        product(&self.dst_val_shape) / 32
    }
}

fn product(values: &[usize]) -> usize {
    values.iter().product()
}

fn layout_eval(shape: &[usize], stride: &[usize], mut index: usize) -> usize {
    let mut offset = 0usize;
    for (&extent, &step) in shape.iter().zip(stride.iter()) {
        offset += (index % extent) * step;
        index /= extent;
    }
    offset
}

fn decode(bit_address: usize) -> (usize, usize, usize) {
    (
        bit_address >> 21,
        (bit_address >> 5) & 0xFFFF,
        bit_address & 31,
    )
}

fn valid_num(num: usize, allowed: &[usize]) -> bool {
    allowed.contains(&num)
}

fn atom(shape: &str, num: usize) -> Option<Atom> {
    match (shape, num) {
        ("32x32b", 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128) => Some(Atom {
            val_shape: vec![32 * num, 32],
            val_stride: vec![1, DP_B],
            dst_thr_shape: vec![32],
            dst_thr_stride: vec![32 * num],
            dst_val_shape: vec![32 * num],
            dst_val_stride: vec![1],
        }),
        ("16x32bx2", _) if valid_num(num, &[1, 2, 4, 8, 16, 32, 64, 128]) => Some(Atom {
            val_shape: vec![64 * num, 16],
            val_stride: vec![1, DP_B],
            dst_thr_shape: vec![16, 2],
            dst_thr_stride: vec![64 * num, 32 * num],
            dst_val_shape: vec![32 * num],
            dst_val_stride: vec![1],
        }),
        ("16x64b", _) if valid_num(num, &[1, 2, 4, 8, 16, 32, 64, 128]) => Some(Atom {
            val_shape: vec![64 * num, 16],
            val_stride: vec![1, DP_B],
            dst_thr_shape: vec![2, 2, 8],
            dst_thr_stride: vec![512 * num, 32, 64 * num],
            dst_val_shape: vec![32, num],
            dst_val_stride: vec![1, 64],
        }),
        ("16x128b", _) if valid_num(num, &[1, 2, 4, 8, 16, 32, 64]) => Some(Atom {
            val_shape: vec![128 * num, 16],
            val_stride: vec![1, DP_B],
            dst_thr_shape: vec![4, 8],
            dst_thr_stride: vec![32, 128 * num],
            dst_val_shape: vec![32, 2, num],
            dst_val_stride: vec![1, 1024 * num, 128],
        }),
        ("16x256b", _) if valid_num(num, &[1, 2, 4, 8, 16, 32]) => Some(Atom {
            val_shape: vec![256 * num, 16],
            val_stride: vec![1, DP_B],
            dst_thr_shape: vec![4, 8],
            dst_thr_stride: vec![64, 256 * num],
            dst_val_shape: vec![64, 2, num],
            dst_val_stride: vec![1, 2048 * num, 256],
        }),
        _ => None,
    }
}

pub fn supported_atoms() -> Vec<(&'static str, usize)> {
    let mut out = Vec::new();
    for shape in ["32x32b", "16x32bx2", "16x64b"] {
        out.extend(
            [1usize, 2, 4, 8, 16, 32, 64, 128]
                .into_iter()
                .map(|num| (shape, num)),
        );
    }
    out.extend(
        [1usize, 2, 4, 8, 16, 32, 64]
            .into_iter()
            .map(|num| ("16x128b", num)),
    );
    out.extend(
        [1usize, 2, 4, 8, 16, 32]
            .into_iter()
            .map(|num| ("16x256b", num)),
    );
    out
}

pub fn register_count(shape: &str, num: usize) -> IResult<usize> {
    atom(shape, num).map(|a| a.reg_size()).ok_or_else(|| {
        InterpreterError::new("tcgen05_datapath", "unsupported tcgen05 ld/st shape/num")
    })
}

/// Returns `(lane_idx, col_idx)`, each `[32, reg_size]`: the TMEM cell (lane
/// within the warp subpartition, column word offset) that thread `t` register
/// `r` moves.
pub fn datapath_index_arrays(shape: &str, num: usize) -> IResult<(Array2<usize>, Array2<usize>)> {
    let atom = atom(shape, num).ok_or_else(|| {
        InterpreterError::new("tcgen05_datapath", "unsupported tcgen05 ld/st shape/num")
    })?;
    let role = atom.role_size();
    let regs = atom.reg_size();
    let mut lane_idx = Array2::<usize>::zeros((role, regs));
    let mut col_idx = Array2::<usize>::zeros((role, regs));
    for thr in 0..role {
        let thr_off = layout_eval(&atom.dst_thr_shape, &atom.dst_thr_stride, thr);
        for reg in 0..regs {
            let val_off = layout_eval(&atom.dst_val_shape, &atom.dst_val_stride, reg * 32);
            let (lane, col, bit0) = decode(layout_eval(
                &atom.val_shape,
                &atom.val_stride,
                thr_off + val_off,
            ));
            if bit0 != 0 {
                return Err(InterpreterError::new(
                    "tcgen05_datapath",
                    format!("{shape}.x{num}: register base is not word-aligned"),
                ));
            }
            lane_idx[[thr, reg]] = lane;
            col_idx[[thr, reg]] = col;
        }
    }
    Ok((lane_idx, col_idx))
}

pub fn datapath_index_arrays_cached(
    shape: &str,
    num: usize,
) -> IResult<&'static (Array2<usize>, Array2<usize>)> {
    static CACHE: OnceLock<Vec<((&'static str, usize), (Array2<usize>, Array2<usize>))>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        supported_atoms()
            .into_iter()
            .map(|(shape, num)| {
                (
                    (shape, num),
                    datapath_index_arrays(shape, num)
                        .expect("supported tcgen05 datapath atom must build"),
                )
            })
            .collect()
    });
    cache
        .iter()
        .find(|((s, n), _)| *s == shape && *n == num)
        .map(|(_, arrays)| arrays)
        .ok_or_else(|| {
            InterpreterError::new("tcgen05_datapath", "unsupported tcgen05 ld/st shape/num")
        })
}

#[derive(Clone, Copy)]
pub struct DatapathIndexSummary {
    pub reg_size: usize,
    pub lane_min: usize,
    pub lane_max: usize,
    pub col_min: usize,
    pub col_max: usize,
    pub has_cell_aliases: bool,
}

pub fn datapath_index_summary_cached(shape: &str, num: usize) -> IResult<DatapathIndexSummary> {
    static CACHE: OnceLock<Vec<((&'static str, usize), DatapathIndexSummary)>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        supported_atoms()
            .into_iter()
            .map(|(shape, num)| {
                let (lanes, cols) = datapath_index_arrays(shape, num)
                    .expect("supported tcgen05 datapath atom must build");
                let mut cells = HashSet::new();
                let mut has_cell_aliases = false;
                let mut lane_min = usize::MAX;
                let mut lane_max = 0usize;
                let mut col_min = usize::MAX;
                let mut col_max = 0usize;
                for (&lane, &col) in lanes.iter().zip(cols.iter()) {
                    has_cell_aliases |= !cells.insert((lane, col));
                    lane_min = lane_min.min(lane);
                    lane_max = lane_max.max(lane);
                    col_min = col_min.min(col);
                    col_max = col_max.max(col);
                }
                (
                    (shape, num),
                    DatapathIndexSummary {
                        reg_size: lanes.ncols(),
                        lane_min,
                        lane_max,
                        col_min,
                        col_max,
                        has_cell_aliases,
                    },
                )
            })
            .collect()
    });
    cache
        .iter()
        .find(|((s, n), _)| *s == shape && *n == num)
        .map(|(_, summary)| *summary)
        .ok_or_else(|| {
            InterpreterError::new("tcgen05_datapath", "unsupported tcgen05 ld/st shape/num")
        })
}

pub fn datapath_has_cell_aliases_cached(shape: &str, num: usize) -> IResult<bool> {
    Ok(datapath_index_summary_cached(shape, num)?.has_cell_aliases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_count_matches_ptx_table49() {
        for num in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            for shape in ["32x32b", "16x32bx2", "16x64b"] {
                assert_eq!(register_count(shape, num).unwrap(), num);
            }
        }
        for num in [1usize, 2, 4, 8, 16, 32, 64] {
            assert_eq!(register_count("16x128b", num).unwrap(), 2 * num);
        }
        for num in [1usize, 2, 4, 8, 16, 32] {
            assert_eq!(register_count("16x256b", num).unwrap(), 4 * num);
        }
        assert!(register_count("16x128b", 128).is_err());
        assert!(register_count("16x256b", 64).is_err());
    }

    #[test]
    fn supported_atoms_have_no_cell_aliases() {
        for (shape, num) in supported_atoms() {
            let (lanes, cols) = datapath_index_arrays(shape, num).unwrap();
            assert_eq!(lanes.dim(), cols.dim());
            assert_eq!(lanes.nrows(), 32);
            assert_eq!(lanes.ncols(), register_count(shape, num).unwrap());
            let mut cells = HashSet::new();
            for (&lane, &col) in lanes.iter().zip(cols.iter()) {
                assert!(cells.insert((lane, col)), "{shape}.x{num} aliases");
            }
        }
    }

    #[test]
    fn lanes_stay_in_warp_subpartition() {
        for (shape, num) in supported_atoms() {
            let (lanes, _) = datapath_index_arrays(shape, num).unwrap();
            assert!(lanes.iter().all(|&lane| lane < 32));
            let unique: HashSet<usize> = lanes.iter().copied().collect();
            let expected = if shape == "32x32b" { 32 } else { 16 };
            assert_eq!(unique.len(), expected, "{shape}.x{num}");
        }
    }

    #[test]
    fn non_32x32b_column_scaling_matches_python_model() {
        let (_, col_x1) = datapath_index_arrays("16x256b", 1).unwrap();
        let (_, col_x2) = datapath_index_arrays("16x256b", 2).unwrap();
        assert_eq!(col_x1.ncols(), 4);
        assert_eq!(col_x2.ncols(), 8);
        assert_eq!(*col_x1.iter().max().unwrap(), 7);
        assert_eq!(*col_x2.iter().max().unwrap(), 15);
    }
}
