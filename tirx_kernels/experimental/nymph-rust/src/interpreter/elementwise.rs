//! Vectorized register-ALU numeric model — port of `interpreter/elementwise.py`.
//! Float ops round operands to dtype, compute in f32, round the result. `cvt`
//! re-rounds the source through the dtype. `max`/`min` use Python-builtin
//! semantics (`where(b>a,b,a)`), NOT numpy maximum. Integer ops wrap to 32 bits.

use super::diagnostics::{IResult, InterpreterError};
use super::values::arrays::ValueArray2;
use crate::ir::DType;
use ndarray::{Array2, Zip};

fn is_reg_dtype(d: DType) -> bool {
    matches!(
        d,
        DType::F16 | DType::Bf16 | DType::F32 | DType::I32 | DType::U32
    )
}

fn py_max(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut out = a.clone();
    Zip::from(&mut out).and(b).for_each(|o, &bv| {
        if bv > *o {
            *o = bv;
        }
    });
    out
}
fn py_min(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut out = a.clone();
    Zip::from(&mut out).and(b).for_each(|o, &bv| {
        if bv < *o {
            *o = bv;
        }
    });
    out
}

fn wrap_i32(values: Array2<i64>) -> Array2<i64> {
    values.mapv(|v| {
        let masked = v & 0xFFFF_FFFF;
        if masked >= 0x8000_0000 {
            masked - 0x1_0000_0000
        } else {
            masked
        }
    })
}

fn wrap_u32(values: Array2<i64>) -> Array2<i64> {
    values.mapv(|v| v & 0xFFFF_FFFF)
}

pub fn apply(op: &str, operands: &[ValueArray2], dtype: DType) -> IResult<ValueArray2> {
    if !is_reg_dtype(dtype) {
        return Err(InterpreterError::new(
            "tensor_value",
            format!("elementwise compute is not modeled for dtype {dtype:?}"),
        ));
    }
    if matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return apply_float(op, operands, dtype);
    }
    apply_int(op, operands, dtype)
}

fn apply_float(op: &str, operands: &[ValueArray2], dtype: DType) -> IResult<ValueArray2> {
    if op == "cvt" {
        return Ok(operands[0].coerce_to_dtype(dtype));
    }
    let f: Vec<Array2<f32>> = operands
        .iter()
        .map(|o| o.coerce_to_dtype(dtype).to_f32_compute())
        .collect();
    let result = match op {
        "add" => &f[0] + &f[1],
        "sub" => &f[0] - &f[1],
        "mul" => &f[0] * &f[1],
        "max" => py_max(&f[0], &f[1]),
        "min" => py_min(&f[0], &f[1]),
        "fma" => &(&f[0] * &f[1]) + &f[2],
        _ => {
            return Err(InterpreterError::new(
                "tensor_value",
                format!("unsupported floating elementwise op {op}"),
            ))
        }
    };
    Ok(ValueArray2::from_f32_compute(result, dtype))
}

fn apply_int(op: &str, operands: &[ValueArray2], dtype: DType) -> IResult<ValueArray2> {
    let wrap: fn(Array2<i64>) -> Array2<i64> = match dtype {
        DType::I32 => wrap_i32,
        DType::U32 => wrap_u32,
        _ => {
            return Err(InterpreterError::new(
                "tensor_value",
                format!("elementwise compute is not modeled for dtype {dtype:?}"),
            ))
        }
    };
    let left = wrap(operands[0].to_i64_compute());
    let right = wrap(operands[1].to_i64_compute());
    let result = match op {
        "add" => wrap(&left + &right),
        "sub" => wrap(&left - &right),
        "mul" => wrap(&left * &right),
        "max" => Zip::from(&left).and(&right).map_collect(|&a, &b| a.max(b)),
        "min" => Zip::from(&left).and(&right).map_collect(|&a, &b| a.min(b)),
        _ => {
            return Err(InterpreterError::new(
                "tensor_value",
                format!("unsupported integer elementwise op {op}"),
            ))
        }
    };
    Ok(ValueArray2::from_i64_compute(result, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn integer_add_sub_mul_wrap_to_32_bits() {
        let a =
            ValueArray2::from_i64_compute(arr2(&[[0xFFFF_FFFF_i64, 0x8000_0000_i64]]), DType::U32);
        let b = ValueArray2::from_i64_compute(arr2(&[[1_i64, 2_i64]]), DType::U32);
        let add = apply("add", &[a.clone(), b.clone()], DType::U32).unwrap();
        assert_eq!(add.to_i64_compute(), arr2(&[[0_i64, 0x8000_0002_i64]]));

        let signed = ValueArray2::from_i64_compute(arr2(&[[2147483647_i64, -2_i64]]), DType::I32);
        let two = ValueArray2::from_i64_compute(arr2(&[[1_i64, -2147483648_i64]]), DType::I32);
        let mul = apply("mul", &[signed, two], DType::I32).unwrap();
        assert_eq!(mul.to_i64_compute(), arr2(&[[2147483647_i64, 0_i64]]));
    }

    #[test]
    fn integer_max_min_use_wrapped_domain() {
        let a = ValueArray2::from_i64_compute(arr2(&[[-1_i64, 3_i64]]), DType::I32);
        let b = ValueArray2::from_i64_compute(arr2(&[[1_i64, -4_i64]]), DType::I32);
        let max = apply("max", &[a.clone(), b.clone()], DType::I32).unwrap();
        let min = apply("min", &[a, b], DType::I32).unwrap();
        assert_eq!(max.to_i64_compute(), arr2(&[[1_i64, 3_i64]]));
        assert_eq!(min.to_i64_compute(), arr2(&[[-1_i64, -4_i64]]));
    }
}
