//! All the simple string-enums from `ir.py`. Each Python `(str, Enum)` becomes a
//! plain Rust `enum`. We derive the traits every enum wants: Clone/Copy (cheap to
//! pass by value), PartialEq/Eq/Hash (use as map keys / compare), Debug (print).
//!
//! These stay pure Rust. The PyO3 layer (`py.rs`) defines mirror enums with the
//! ALL-CAPS Python names + From conversions (a macro keeps that DRY), because
//! `#[pyclass]` can't rename variants through a `cfg_attr`.

/// `MemorySpace` — where a tensor lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemorySpace {
    Gmem,
    Smem,
    Tmem,
    Reg,
}

/// `DType` — element type of a tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    /// float8 e4m3fn (1 sign + 4 exp / bias 7 + 3 mantissa; no inf, NaN = 0x7f/0xff).
    F8E4M3,
    F16,
    Bf16,
    F32,
}

/// `Swizzle` — SMEM swizzle width.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Swizzle {
    None,
    B32,
    B64,
    B128,
}

/// `TmemLayoutKind` — logical row->lane mapping for a TMEM tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemLayoutKind {
    Lane128,
    Lane64Upper,
    Lane64Lower,
    ScaleVec1x,
    ScaleVec2x,
    ScaleVec4x,
}

/// `MBarKind` — what kind of work an mbarrier tracks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MBarKind {
    Tma,
    Tcgen05,
    Thread,
}

/// `FenceKind`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FenceKind {
    Memory,
    AsyncProxy,
    View,
}

/// `FenceScope` — the memory hierarchy level a fence orders. `Cta`/`Cluster` are the
/// shared-memory levels (cpusim's `shared::cta` / `shared::cluster`); `Gpu` is global
/// memory (`global`). The cta-vs-cluster distinction is what a proxy-fence visibility
/// check reads to decide whether a fence covers a peer CTA's shared memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FenceScope {
    Cta,
    Cluster,
    Gpu,
}

/// `VarBinding` — what kind of variable a `Var` is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VarBinding {
    Loop,
    Scalar,
    Task,
}

/// `ScalarDType` — element type of a scalar variable (a subset of DType).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScalarDType {
    Bool,
    I32,
    U32,
    I64,
    U64,
}

/// `ScalarOp` — the operations a `ScalarExpr` can hold.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScalarOp {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Xor,
    And,
    Or,
    Neg,
    Not,
    Select,
    Min,
    Max,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl ScalarOp {
    /// How many args this op takes (mirrors `ScalarExpr.__post_init__`).
    pub fn arity(self) -> usize {
        match self {
            ScalarOp::Neg | ScalarOp::Not => 1,
            ScalarOp::Select => 3,
            _ => 2,
        }
    }
}

/// `ScopeValueKind` — a per-thread/per-CTA hardware scope value (the Python
/// `Literal[...]`). Constructed in Python from its string name (e.g. "lane_id"),
/// so it is not exposed as a Python enum; `py.rs` maps the string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScopeValueKind {
    TidInWg,
    LaneId,
    WarpId,
    WarpgroupId,
    CtaidInCluster,
    CtaId,
    NvshmemMyPe,
}
