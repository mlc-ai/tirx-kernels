//! Row-major flat-index helpers — port of `interpreter/values/_indexing.py`.
//! Shared by every per-space value container. Last axis varies fastest (C order).

/// Product of dims; empty shape -> 1 (empty product), a 0 dim -> 0.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major flat offset of `coord` within `shape` (last axis fastest).
pub fn flat_index(coord: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut idx = 0usize;
    for (c, s) in coord.iter().rev().zip(shape.iter().rev()) {
        idx += c * stride;
        stride *= s;
    }
    idx
}

/// Enumerate every absolute coordinate in a slice, row-major (last axis fastest).
/// Returns empty when any extent is 0.
pub fn slice_coords(offsets: &[usize], slice_shape: &[usize]) -> Vec<Vec<usize>> {
    if slice_shape.iter().any(|&e| e == 0) {
        return Vec::new();
    }
    let mut coords: Vec<Vec<usize>> = vec![Vec::new()];
    for (&offset, &extent) in offsets.iter().zip(slice_shape.iter()) {
        let mut next = Vec::with_capacity(coords.len() * extent);
        for prefix in &coords {
            for step in 0..extent {
                let mut c = prefix.clone();
                c.push(offset + step);
                next.push(c);
            }
        }
        coords = next;
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_flat_index() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[2, 3]), 6);
        assert_eq!(flat_index(&[1, 2], &[4, 3]), 1 * 3 + 2);
    }

    #[test]
    fn slice_coords_c_order() {
        let c = slice_coords(&[0, 1], &[2, 2]);
        assert_eq!(c, vec![vec![0, 1], vec![0, 2], vec![1, 1], vec![1, 2]]);
        assert!(slice_coords(&[0], &[0]).is_empty());
    }
}
