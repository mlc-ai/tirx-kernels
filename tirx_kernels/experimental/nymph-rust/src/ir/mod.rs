//! The nymph IR — a faithful Rust port of `ir/ir.py`.
//!
//! Layout (mirrors the Python module structure):
//! - `dtype`  — the simple enums (MemorySpace, DType, ScalarOp, ...)
//! - `scalar` — Var (identity model), ScalarExpr, ScalarValue, ScalarInitial
//! - `tensor` — Tensor (table/id model), TensorSlice, Layout, TmemLayout
//! - `mbar`   — MBar (table/id model), MBarRef
//! - `stmt`   — the big Stmt enum (~41 variants)
//! - `kernel` — Kernel (owns the tensor/mbar tables)

pub mod dtype;
pub mod kernel;
pub mod mbar;
pub mod scalar;
pub mod scheduler;
pub mod stmt;
pub mod tensor;
pub mod validate;

// Re-export everything so callers can write `nymph::ir::Tensor` etc.
pub use dtype::*;
pub use kernel::*;
pub use mbar::*;
pub use scalar::*;
pub use scheduler::*;
pub use stmt::*;
pub use tensor::*;
pub use validate::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Build a tiny IR by hand to prove all the types compose and the identity
    /// model works. (Once the builder exists, this is what it produces.)
    #[test]
    fn assemble_a_tiny_kernel() {
        // A GMEM input tensor C (f32, 256x256) and an SMEM operand A (f16, 256x16),
        // each heap-allocated once and shared via Arc.
        let c = Arc::new(Tensor {
            id: 0,
            space: MemorySpace::Gmem,
            dtype: DType::F32,
            shape: vec![256, 256],
            layout: None,
            byte_offset: None,
        });
        let a = Arc::new(Tensor {
            id: 1,
            space: MemorySpace::Smem,
            dtype: DType::F16,
            shape: vec![256, 16],
            layout: None,
            byte_offset: Some(0),
        });

        // A loop variable `k` (identity is its id).
        let k = Var {
            id: VarId(0),
            binding: VarBinding::Loop,
            dtype: ScalarDType::I32,
        };
        let k_again = k; // a copy refers to the SAME var (equal by id)
        assert_eq!(k, k_again);
        let other = Var {
            id: VarId(1),
            binding: VarBinding::Loop,
            dtype: ScalarDType::I32,
        };
        assert_ne!(k, other); // different id => different var

        // A symbolic offset: k * 16  (a ScalarExpr inside a ScalarValue).
        let offset = ScalarValue::expr(
            ScalarOp::Mul,
            vec![ScalarValue::Var(k), ScalarValue::Int(16)],
        );

        // A slice over A — note we can read the tensor's data right here.
        let a_slice = TensorSlice {
            tensor: Arc::clone(&a),
            offsets: vec![ScalarValue::Int(0), offset.clone()],
            shape: vec![ScalarValue::Int(256), ScalarValue::Int(16)],
        };
        // Cross-reference works at construction (this is the whole point of Arc):
        assert_eq!(a_slice.tensor.space, MemorySpace::Smem);
        assert_eq!(a_slice.tensor.dtype, DType::F16);
        // Two Arc clones of the same tensor are "the same tensor" (equal by id).
        let a2 = Arc::clone(&a);
        assert_eq!(*a_slice.tensor, *a2);

        let body = vec![
            Stmt::ForLoop {
                var: k,
                start: ScalarValue::Int(0),
                stop: ScalarValue::Int(16),
                step: ScalarValue::Int(1),
                body: vec![Stmt::RegStore {
                    dst: a_slice.clone(),
                    src: a_slice.clone(),
                }],
            },
            Stmt::CtaSync,
        ];

        let kernel = Kernel {
            name: "tiny".to_string(),
            args: vec![Arc::clone(&c)],
            body,
            num_warps: 12,
            smem_size_bytes: 256 * 16 * 2,
            launch_shape: vec![2],
            cluster_shape: vec![2],
        };

        // Arg tensor data is reachable directly through its Arc.
        assert_eq!(kernel.args[0].dtype, DType::F32);
        assert_eq!(kernel.launch_cta_count(), 2);
        // The control node exposes its child body for generic walks.
        assert_eq!(kernel.body[0].child_bodies().len(), 1);
        assert_eq!(kernel.body[1].child_bodies().len(), 0);
    }

    fn kernel(body: Vec<Stmt>, num_warps: u32) -> Kernel {
        Kernel {
            name: "t".to_string(),
            args: vec![],
            body,
            num_warps,
            smem_size_bytes: 0,
            launch_shape: vec![2],
            cluster_shape: vec![2],
        }
    }

    #[test]
    fn valid_kernel_passes() {
        let s = Var {
            id: VarId(0),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let body = vec![
            Stmt::ScalarDef {
                var: s,
                initial: ScalarInitial::Value(ScalarValue::Int(0)),
            },
            Stmt::ScalarStore {
                var: s,
                value: ScalarValue::Int(5),
            },
            Stmt::CtaSync,
        ];
        assert!(kernel(body, 4).validate().is_ok());
    }

    #[test]
    fn rejects_bad_num_warps() {
        // 6 is not a multiple of 4 -> local check fails.
        let e = kernel(vec![], 6).validate().unwrap_err();
        assert!(e.message.contains("num_warps"), "{}", e.message);
    }

    #[test]
    fn rejects_undefined_var() {
        // Storing to a var that was never defined -> var-defs walk fails.
        let s = Var {
            id: VarId(7),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let e = kernel(
            vec![Stmt::ScalarStore {
                var: s,
                value: ScalarValue::Int(1),
            }],
            4,
        )
        .validate()
        .unwrap_err();
        assert!(e.message.contains("defined before use"), "{}", e.message);
    }

    #[test]
    fn rejects_cta_sync_in_warp_scope() {
        // cta_sync inside a warp-scope init block -> scope walk fails.
        let body = vec![Stmt::KernelInit {
            body: vec![Stmt::CtaSync],
            warp: Some(0),
            lane: None,
            elected: false,
        }];
        let e = kernel(body, 4).validate().unwrap_err();
        assert!(
            e.message.contains("cta_sync must be in CTA scope"),
            "{}",
            e.message
        );
    }

    #[test]
    fn rejects_cta_sync_inside_role() {
        let body = vec![Stmt::Role {
            body: vec![Stmt::CtaSync],
            warp: None,
            warpgroup: None,
            elected: false,
            maxnreg: None,
        }];
        let e = kernel(body, 4).validate().unwrap_err();
        assert!(
            e.message.contains("cta_sync cannot be used inside role"),
            "{}",
            e.message
        );
    }
}
