//! DType numeric policy — a faithful port of `interpreter/values/dtypes.py`.
//!
//! No global compute dtype: each op computes in its own precision and rounds back
//! to the container dtype on write. Float tensors are stored as f32 holding the
//! dtype-rounded value (value-equivalent to numpy's float16 storage / bf16-in-f32
//! container — a representable f16/bf16 value is exact in f32). Integers wrap to
//! their width.

use crate::ir::DType;
use half::f16;
use ndarray::{ArrayD, Zip};

pub fn is_float_dtype(d: DType) -> bool {
    matches!(d, DType::F8E4M3 | DType::F16 | DType::Bf16 | DType::F32)
}

/// Decode one float8 e4m3fn byte via a 256-entry table (the bulk byte-codec
/// hot path). Every finite e4m3 value is exact in f32.
#[inline]
pub fn decode_e4m3(b: u8) -> f32 {
    e4m3_decode_table()[b as usize]
}

fn e4m3_decode_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = decode_e4m3_arith(i as u8);
        }
        t
    })
}

/// Arithmetic e4m3fn decode: 1 sign + 4 exp (bias 7) + 3 mantissa; no inf,
/// NaN = 0x7f/0xff, max finite = ±448 (the table builder).
fn decode_e4m3_arith(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = b & 7;
    if e == 0xF && m == 7 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        // subnormal: m/8 * 2^-6 = m * 2^-9
        m as f32 * (-9f32).exp2()
    } else {
        (8 + m) as f32 * ((e as i32 - 10) as f32).exp2()
    };
    sign * mag
}

/// The 127 non-negative finite e4m3fn magnitudes, indexed by bit pattern
/// 0x00..=0x7E (strictly increasing).
#[cfg(test)]
fn e4m3_magnitudes() -> &'static [f32; 127] {
    static TABLE: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0f32; 127];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = decode_e4m3_arith(i as u8);
        }
        t
    })
}

/// Encode an f32 to e4m3fn: round-to-nearest-even, saturating to ±448 (PTX
/// `cvt.rn.satfinite.e4m3` semantics). NaN -> 0x7f. Closed-form arithmetic
/// (quantize to the value's e4m3 ulp grid with ties-to-even), no table search
/// — this sits on the value simulator's hot byte-codec path.
pub fn encode_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    if exp_f32 == 0 {
        return sign; // f32 subnormal: far below half of e4m3's min subnormal
    }
    let exp = exp_f32 - 127;
    if exp < -10 {
        return sign; // < 2^-10 = half of the min subnormal -> rounds to 0
    }
    let a = f32::from_bits(bits & 0x7FFF_FFFF);
    // ulp exponent: normals (>= 2^-6) step 2^(exp-3); subnormals step 2^-9.
    let q = if exp >= -6 { exp - 3 } else { -9 };
    // exact power of two 2^-q; a/2^q in ulp units, RNE to the grid.
    let inv_ulp = f32::from_bits(((127 - q) as u32) << 23);
    let mant = (a * inv_ulp).round_ties_even() as u32;
    if exp >= -6 {
        let mut e4 = (exp + 7) as u32;
        let mut m = mant; // in [8, 16]; 16 = mantissa overflow into the next exponent
        if m == 16 {
            m = 8;
            e4 += 1;
        }
        if e4 > 15 || (e4 == 15 && m - 8 == 7) {
            return sign | 0x7E; // overflow / the NaN slot -> satfinite 448
        }
        sign | ((e4 as u8) << 3) | ((m - 8) as u8)
    } else if mant == 8 {
        sign | 0x08 // subnormal rounds up to the min normal
    } else {
        sign | mant as u8 // e=0 subnormal
    }
}

/// RNE round of an f32 to e4m3fn precision, result still f32.
#[inline]
pub fn round_e4m3_scalar(x: f32) -> f32 {
    decode_e4m3(encode_e4m3(x))
}

pub fn is_int_dtype(d: DType) -> bool {
    matches!(
        d,
        DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::I32
            | DType::U32
            | DType::I64
            | DType::U64
    )
}

/// bit-width of an integer dtype.
fn int_bits(d: DType) -> u32 {
    match d {
        DType::I8 | DType::U8 => 8,
        DType::I16 | DType::U16 => 16,
        DType::I32 | DType::U32 => 32,
        DType::I64 | DType::U64 => 64,
        _ => unreachable!("int_bits on non-int dtype"),
    }
}

fn is_signed_int(d: DType) -> bool {
    matches!(d, DType::I8 | DType::I16 | DType::I32 | DType::I64)
}

/// Two's-complement wrap of an integer value to `dtype`'s width (mirrors
/// `wrap_int_to_dtype`). Done in i128 to avoid overflow on the 64-bit case.
pub fn wrap_int_to_dtype(value: i128, dtype: DType) -> i64 {
    let bits = int_bits(dtype);
    if bits == 64 {
        // mask is the full 64-bit range; handle signed/unsigned directly.
        let masked = (value as u128) & 0xFFFF_FFFF_FFFF_FFFF;
        return if is_signed_int(dtype) {
            masked as u64 as i64
        } else {
            // unsigned u64 value held in i64 bit pattern (callers treat as unsigned)
            masked as u64 as i64
        };
    }
    let mask: i128 = (1i128 << bits) - 1;
    let mut masked = value & mask;
    if is_signed_int(dtype) && masked >= (1i128 << (bits - 1)) {
        masked -= 1i128 << bits;
    }
    masked as i64
}

/// Round-to-nearest-even of an f32 value to bf16 precision, held in f32 — the
/// exact bit formula from the Python port (add 0x7FFF + round bit, mask low 16).
#[inline]
pub fn round_bf16_scalar(x: f32) -> f32 {
    let bits = x.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 {
        return x; // inf / nan pass through untouched
    }
    let b = bits as u64;
    let rounded = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded as u32)
}

/// Round an f32 value to a float dtype's precision (RNE), result still f32.
#[inline]
pub fn round_scalar(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => x,
        DType::F16 => f16::from_f32(x).to_f32(),
        DType::Bf16 => round_bf16_scalar(x),
        DType::F8E4M3 => round_e4m3_scalar(x),
        _ => x,
    }
}

/// Round a whole f32 array to a float dtype's precision (mirrors `round_to_dtype`).
pub fn round_to_dtype(values: &ArrayD<f32>, dtype: DType) -> ArrayD<f32> {
    match dtype {
        DType::F32 => values.clone(),
        DType::F16 => values.mapv(|x| f16::from_f32(x).to_f32()),
        DType::Bf16 => values.mapv(round_bf16_scalar),
        DType::F8E4M3 => values.mapv(round_e4m3_scalar),
        _ => values.clone(),
    }
}

/// In-place round (avoids an allocation when the caller owns the array).
pub fn round_in_place(values: &mut ArrayD<f32>, dtype: DType) {
    match dtype {
        DType::F32 => {}
        DType::F16 => Zip::from(values).for_each(|x| *x = f16::from_f32(*x).to_f32()),
        DType::Bf16 => Zip::from(values).for_each(|x| *x = round_bf16_scalar(*x)),
        DType::F8E4M3 => Zip::from(values).for_each(|x| *x = round_e4m3_scalar(*x)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn f16_rounds_to_nearest_representable() {
        // 1.0 is exact; a value needing rounding loses precision.
        let a = arr1(&[1.0f32, 1.0 + 1.0 / 2048.0]).into_dyn();
        let r = round_to_dtype(&a, DType::F16);
        assert_eq!(r[0], 1.0);
        // f16 has 10 mantissa bits; 1 + 1/2048 rounds to 1.0 (below the f16 ulp at 1.0 = 1/1024).
        assert_eq!(r[1], 1.0);
    }

    #[test]
    fn bf16_passes_inf_nan_and_rounds_mantissa() {
        let a = arr1(&[f32::INFINITY, 1.0f32]).into_dyn();
        let r = round_to_dtype(&a, DType::Bf16);
        assert!(r[0].is_infinite());
        assert_eq!(r[1], 1.0);
    }

    #[test]
    fn int_wrap_signed_unsigned() {
        assert_eq!(wrap_int_to_dtype(0x1_0000_0000, DType::I32), 0); // 2^32 wraps to 0
        assert_eq!(wrap_int_to_dtype(-1, DType::U32), 0xFFFF_FFFF); // -1 as u32
        assert_eq!(wrap_int_to_dtype(0x8000_0000, DType::I32), -2147483648); // top bit -> negative
    }

    #[test]
    fn e4m3_codec_round_trips_every_finite_encoding() {
        for bits in 0u16..=0xFF {
            let b = bits as u8;
            let v = decode_e4m3(b);
            if v.is_nan() {
                assert!(matches!(b, 0x7F | 0xFF));
                assert_eq!(encode_e4m3(v), 0x7F);
                continue;
            }
            // -0.0 encodes back to +0.0's pattern only via sign; both decode to 0.
            let back = encode_e4m3(v);
            assert_eq!(
                decode_e4m3(back).to_bits(),
                v.to_bits(),
                "byte {b:#04x} -> {v} -> {back:#04x}"
            );
        }
    }

    /// The table-search reference encoder (the original implementation): RNE
    /// over the sorted finite-magnitude table, ties to the even bit pattern.
    fn encode_e4m3_table_ref(x: f32) -> u8 {
        if x.is_nan() {
            return 0x7F;
        }
        let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
        let a = x.abs();
        let t = e4m3_magnitudes();
        if a >= t[126] {
            return sign | 126;
        }
        let mut lo = 0usize;
        let mut hi = 126usize;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if t[mid] <= a {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let below = a - t[lo];
        let above = t[lo + 1] - a;
        let idx = if below < above {
            lo
        } else if below > above {
            lo + 1
        } else if lo % 2 == 0 {
            lo
        } else {
            lo + 1
        };
        sign | idx as u8
    }

    #[test]
    fn e4m3_closed_form_encode_matches_table_reference() {
        // every grid point and every midpoint (the RNE tie cases), both signs
        for i in 0..126u8 {
            let lo = e4m3_magnitudes()[i as usize];
            let hi = e4m3_magnitudes()[i as usize + 1];
            let mid = (lo + hi) / 2.0;
            for v in [lo, hi, mid, mid * (1.0 + 1e-6), mid * (1.0 - 1e-6)] {
                for s in [v, -v] {
                    assert_eq!(
                        encode_e4m3(s),
                        encode_e4m3_table_ref(s),
                        "mismatch at {s} (grid {i})"
                    );
                }
            }
        }
        // dense pseudo-random sweep over the full range + beyond saturation
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..100_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = (state >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
            let v = (frac - 0.5) * 1200.0; // [-600, 600): covers sat + subnormals
            assert_eq!(encode_e4m3(v), encode_e4m3_table_ref(v), "mismatch at {v}");
            let tiny = (frac - 0.5) * 0.01;
            assert_eq!(
                encode_e4m3(tiny),
                encode_e4m3_table_ref(tiny),
                "mismatch at {tiny}"
            );
        }
    }

    #[test]
    fn e4m3_encode_known_values() {
        assert_eq!(decode_e4m3(encode_e4m3(448.0)), 448.0); // max finite
        assert_eq!(decode_e4m3(encode_e4m3(1e9)), 448.0); // satfinite
        assert_eq!(decode_e4m3(encode_e4m3(-1e9)), -448.0);
        assert_eq!(decode_e4m3(encode_e4m3(1.0)), 1.0);
        assert_eq!(decode_e4m3(encode_e4m3(192.0)), 192.0); // 1.5 * 2^7
        assert_eq!(decode_e4m3(encode_e4m3(0.001953125)), 0.001953125); // min subnormal 2^-9
                                                                        // RNE tie: 17 sits halfway between 16 and 18 -> even mantissa (16).
        assert_eq!(decode_e4m3(encode_e4m3(17.0)), 16.0);
        // RNE tie: 19 sits halfway between 18 and 20 -> even mantissa (20).
        assert_eq!(decode_e4m3(encode_e4m3(19.0)), 20.0);
        assert_eq!(encode_e4m3(f32::NAN), 0x7F);
    }
}
