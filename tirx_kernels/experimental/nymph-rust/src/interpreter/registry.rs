//! Statement executor dispatch — port of `registry.py`.
//!
//! The modularity contract: a semantics module registers executors for its Stmt
//! kinds; the runner iterates the registrars to build this table and is NEVER
//! edited to add an op. (Rust needs an explicit `StmtKind` discriminant table in
//! place of Python's type→handler dict, but the decoupling is the same.)

use super::cohort::StmtExecutor;
use crate::ir::Stmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StmtKind {
    TensorDef,
    TmemAlloc,
    TmemDealloc,
    ScalarDef,
    ScalarStore,
    StoreScalar,
    MBarDef,
    KernelInit,
    KernelFinalize,
    Role,
    ForLoop,
    ForEachTask,
    SchedulerImpl,
    SchedNext,
    Loop,
    BreakIf,
    If,
    MBarrierInit,
    MBarrierArrive,
    MBarrierWait,
    MBarrierExpectTx,
    MBarrierArriveExpectTx,
    TmaLoad,
    TmaStore,
    CpAsyncBulkCommitGroup,
    CpAsyncBulkWaitGroupRead,
    Tcgen05Mma,
    Tcgen05Cp,
    Tcgen05Commit,
    Tcgen05Ld,
    Tcgen05WaitLd,
    Tcgen05St,
    Tcgen05WaitSt,
    LdMatrix,
    StMatrix,
    RegFill,
    RegUnary,
    RegAdd,
    RegSub,
    RegMul,
    RegFma,
    RegMax,
    RegMin,
    RegBitwise,
    RegReduce,
    RegCondRescale,
    RegSoftmaxRescale,
    RegCausalMask,
    RegCombineIntFracEx2,
    RegCvt,
    RegLoad,
    RegStore,
    Fence,
    CtaSync,
    WgSync,
    WarpSync,
    ClusterSync,
}

pub fn stmt_kind(stmt: &Stmt) -> StmtKind {
    match stmt {
        Stmt::TensorDef { .. } => StmtKind::TensorDef,
        Stmt::TmemAlloc { .. } => StmtKind::TmemAlloc,
        Stmt::TmemDealloc { .. } => StmtKind::TmemDealloc,
        Stmt::ScalarDef { .. } => StmtKind::ScalarDef,
        Stmt::ScalarStore { .. } => StmtKind::ScalarStore,
        Stmt::StoreScalar { .. } => StmtKind::StoreScalar,
        Stmt::MBarDef { .. } => StmtKind::MBarDef,
        Stmt::KernelInit { .. } => StmtKind::KernelInit,
        Stmt::KernelFinalize { .. } => StmtKind::KernelFinalize,
        Stmt::Role { .. } => StmtKind::Role,
        Stmt::ForLoop { .. } => StmtKind::ForLoop,
        Stmt::ForEachTask { .. } => StmtKind::ForEachTask,
        Stmt::SchedulerImpl { .. } => StmtKind::SchedulerImpl,
        Stmt::SchedNext { .. } => StmtKind::SchedNext,
        Stmt::Loop { .. } => StmtKind::Loop,
        Stmt::BreakIf { .. } => StmtKind::BreakIf,
        Stmt::If { .. } => StmtKind::If,
        Stmt::MBarrierInit { .. } => StmtKind::MBarrierInit,
        Stmt::MBarrierArrive { .. } => StmtKind::MBarrierArrive,
        Stmt::MBarrierWait { .. } => StmtKind::MBarrierWait,
        Stmt::MBarrierExpectTx { .. } => StmtKind::MBarrierExpectTx,
        Stmt::MBarrierArriveExpectTx { .. } => StmtKind::MBarrierArriveExpectTx,
        Stmt::TmaLoad { .. } => StmtKind::TmaLoad,
        Stmt::TmaStore { .. } => StmtKind::TmaStore,
        Stmt::CpAsyncBulkCommitGroup => StmtKind::CpAsyncBulkCommitGroup,
        Stmt::CpAsyncBulkWaitGroupRead { .. } => StmtKind::CpAsyncBulkWaitGroupRead,
        Stmt::Tcgen05Mma { .. } => StmtKind::Tcgen05Mma,
        Stmt::Tcgen05Cp { .. } => StmtKind::Tcgen05Cp,
        Stmt::Tcgen05Commit { .. } => StmtKind::Tcgen05Commit,
        Stmt::Tcgen05Ld { .. } => StmtKind::Tcgen05Ld,
        Stmt::Tcgen05WaitLd => StmtKind::Tcgen05WaitLd,
        Stmt::Tcgen05St { .. } => StmtKind::Tcgen05St,
        Stmt::Tcgen05WaitSt => StmtKind::Tcgen05WaitSt,
        Stmt::LdMatrix { .. } => StmtKind::LdMatrix,
        Stmt::StMatrix { .. } => StmtKind::StMatrix,
        Stmt::RegFill { .. } => StmtKind::RegFill,
        Stmt::RegUnary { .. } => StmtKind::RegUnary,
        Stmt::RegAdd { .. } => StmtKind::RegAdd,
        Stmt::RegSub { .. } => StmtKind::RegSub,
        Stmt::RegMul { .. } => StmtKind::RegMul,
        Stmt::RegFma { .. } => StmtKind::RegFma,
        Stmt::RegMax { .. } => StmtKind::RegMax,
        Stmt::RegMin { .. } => StmtKind::RegMin,
        Stmt::RegBitwise { .. } => StmtKind::RegBitwise,
        Stmt::RegReduce { .. } => StmtKind::RegReduce,
        Stmt::RegCondRescale { .. } => StmtKind::RegCondRescale,
        Stmt::RegSoftmaxRescale { .. } => StmtKind::RegSoftmaxRescale,
        Stmt::RegCausalMask { .. } => StmtKind::RegCausalMask,
        Stmt::RegCombineIntFracEx2 { .. } => StmtKind::RegCombineIntFracEx2,
        Stmt::RegCvt { .. } => StmtKind::RegCvt,
        Stmt::RegLoad { .. } => StmtKind::RegLoad,
        Stmt::RegStore { .. } => StmtKind::RegStore,
        Stmt::Fence { .. } => StmtKind::Fence,
        Stmt::CtaSync => StmtKind::CtaSync,
        Stmt::WgSync { .. } => StmtKind::WgSync,
        Stmt::WarpSync => StmtKind::WarpSync,
        Stmt::ClusterSync => StmtKind::ClusterSync,
    }
}

impl StmtKind {
    /// `ClusterSync` is the last variant; the enum is fieldless so `as usize`
    /// gives a contiguous 0..COUNT index — used to dispatch via a flat array.
    pub const COUNT: usize = StmtKind::ClusterSync as usize + 1;
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Direct-index dispatch table: `StmtKind::COUNT` slots of `Option<fn>`. Looking up
/// an executor is one array index off the stmt's kind — no hashing per step.
pub struct StmtExecutorRegistry {
    executors: Vec<Option<StmtExecutor>>,
    fallback: Option<StmtExecutor>,
}

impl Default for StmtExecutorRegistry {
    fn default() -> Self {
        StmtExecutorRegistry {
            executors: vec![None; StmtKind::COUNT],
            fallback: None,
        }
    }
}

impl StmtExecutorRegistry {
    pub fn register(&mut self, kind: StmtKind, executor: StmtExecutor) {
        self.executors[kind.index()] = Some(executor);
    }
    pub fn set_fallback(&mut self, executor: StmtExecutor) {
        self.fallback = Some(executor);
    }
    #[inline]
    pub fn executor_for(&self, stmt: &Stmt) -> Option<StmtExecutor> {
        self.executors[stmt_kind(stmt).index()].or(self.fallback)
    }
}
