//! REG integer-ALU 32-bit wrapping — port of `interpreter/values/reg_numerics.py`.
//! Operands/products are held in i64 intermediates (enough for an i32*i32 product)
//! and wrapped back to signed/unsigned 32 bits. Both helpers return i64 (only the
//! conditional subtraction in `to_i32` differs).

use ndarray::ArrayD;

/// Wrap to signed 32-bit, returned in an i64 array.
pub fn to_i32(values: &ArrayD<i64>) -> ArrayD<i64> {
    values.mapv(|v| {
        let masked = v & 0xFFFF_FFFF;
        if masked >= 0x8000_0000 {
            masked - 0x1_0000_0000
        } else {
            masked
        }
    })
}

/// Wrap to unsigned 32-bit, returned non-negative in an i64 array (so it stays in
/// [0, 2^32) and compares cleanly with signed intermediates).
pub fn to_u32(values: &ArrayD<i64>) -> ArrayD<i64> {
    values.mapv(|v| v & 0xFFFF_FFFF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn wraps() {
        let a = arr1(&[-1i64, 0x8000_0000, 5]).into_dyn();
        assert_eq!(to_i32(&a).as_slice().unwrap(), &[-1, -2147483648, 5]);
        assert_eq!(
            to_u32(&a).as_slice().unwrap(),
            &[0xFFFF_FFFF, 0x8000_0000, 5]
        );
    }
}
