//! Scalar expression IR: `Var`, `ScalarExpr`, `ScalarValue`, `ScalarInitial`.
//!
//! KEY DESIGN POINT — identity. In Python `Var` is `eq=False` (two vars with the
//! same fields are still DIFFERENT — identity matters: a var is keyed by object
//! identity in the scalar environment). Rust has no object identity, so we give
//! each `Var` a stable `VarId` and make equality/hash use ONLY that id. The Var
//! also carries its own data (binding, dtype) inline — it's tiny and `Copy`, so we
//! pass it around by value; "refer to the same var" = a `Var` value with the same
//! id. (The builder is the source of fresh ids.)

use super::dtype::{ScalarDType, ScalarOp, ScopeValueKind, VarBinding};
use super::tensor::TensorSlice;
use std::hash::{Hash, Hasher};

/// Stable identity of a `Var` (assigned by the builder from a counter).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarId(pub u32);

/// `Var` — a scalar variable (loop / scalar / task). Carries its data inline; its
/// identity is `id` only (see the module note).
#[derive(Clone, Copy, Debug)]
pub struct Var {
    pub id: VarId,
    pub binding: VarBinding,
    pub dtype: ScalarDType,
}

// Equality/hash by id ONLY — this is the whole point of the identity model.
impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Var {}
impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// `ScalarExpr` — an operation over scalar values (e.g. `task * 16 + k`).
/// Recursive: its args are `ScalarValue`s, which may themselves be exprs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScalarExpr {
    pub op: ScalarOp,
    pub args: Vec<ScalarValue>,
}

/// `ScalarValue = int | Var | ScalarExpr | ScopeValue` — the Python union type,
/// expressed as a Rust enum (Python dispatched on isinstance; Rust matches).
/// `Expr` is boxed because the type is recursive (an expr contains scalar values).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ScalarValue {
    Int(i64),
    Var(Var),
    Expr(Box<ScalarExpr>),
    Scope(ScopeValueKind),
}

impl ScalarValue {
    /// Convenience: wrap an expr (boxes it).
    pub fn expr(op: ScalarOp, args: Vec<ScalarValue>) -> ScalarValue {
        ScalarValue::Expr(Box::new(ScalarExpr { op, args }))
    }
}

// Let plain ints be used wherever a ScalarValue is expected: `ScalarValue::from(5)`.
impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self {
        ScalarValue::Int(v)
    }
}
impl From<Var> for ScalarValue {
    fn from(v: Var) -> Self {
        ScalarValue::Var(v)
    }
}

/// `ScalarInitial = ScalarValue | TensorSlice` — the initial value of a scalar_def
/// (either a scalar, or a 1-element GMEM tensor slice to load from).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ScalarInitial {
    Value(ScalarValue),
    Tensor(TensorSlice),
}
