//! PTX ldmatrix/stmatrix m8n8.b16 fragment mapping.
//!
//! For one matrix, every lane owns one b32 register holding two b16 elements.
//! A group of four consecutive lanes covers one row in non-transposed form.
//! With `.trans`, the lane fragments are interpreted as column-major: the same
//! two b16 halves map down a column rather than across a row.

use crate::interpreter::diagnostics::{IResult, InterpreterError};

pub const SUPPORTED_NUMS: &[usize] = &[1, 2, 4];
pub const SUPPORTED_TRANS: &[bool] = &[false, true];

pub fn check_num(num: usize, label: &str) -> IResult<()> {
    if SUPPORTED_NUMS.contains(&num) {
        Ok(())
    } else {
        Err(InterpreterError::new(
            format!("{label}_shape"),
            format!("{label} supports only m8n8.x{{1,2,4}}.b16"),
        ))
    }
}

pub fn element_coord(lane: usize, half: usize, trans: bool) -> (usize, usize) {
    debug_assert!(lane < 32);
    debug_assert!(half < 2);
    if trans {
        (2 * (lane % 4) + half, lane / 4)
    } else {
        (lane / 4, 2 * (lane % 4) + half)
    }
}

pub fn row_address_lane(matrix_id: usize, row_in_matrix: usize) -> usize {
    debug_assert!(matrix_id < 4);
    debug_assert!(row_in_matrix < 8);
    matrix_id * 8 + row_in_matrix
}

pub fn pack_b16x2(lo: u16, hi: u16) -> u32 {
    lo as u32 | ((hi as u32) << 16)
}

pub fn unpack_b16x2(word: u32) -> [u16; 2] {
    [(word & 0xffff) as u16, (word >> 16) as u16]
}
