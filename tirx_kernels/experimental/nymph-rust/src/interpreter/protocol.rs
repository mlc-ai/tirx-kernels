//! Protocol trace/checker result types.

use super::diagnostics::{Diagnostic, IResult};
use super::mbar_ops::MbarTarget;
use crate::ir::FenceScope;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Value,
    Trace,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Trace
    }
}

impl ExecutionMode {
    pub fn is_value(self) -> bool {
        matches!(self, ExecutionMode::Value)
    }

    pub fn is_trace(self) -> bool {
        matches!(self, ExecutionMode::Trace)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolStatus {
    Passed,
    Failed,
    Inconclusive,
}

impl ProtocolStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ProtocolStatus::Passed => "Passed",
            ProtocolStatus::Failed => "Failed",
            ProtocolStatus::Inconclusive => "Inconclusive",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolWarning {
    pub code: String,
    pub message: String,
    pub details: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolPassSummary {
    pub name: String,
    pub status: ProtocolStatus,
    pub diagnostics: usize,
    pub warnings: usize,
}

#[derive(Clone, Debug)]
pub struct ProtocolReport {
    pub status: ProtocolStatus,
    pub warnings: Vec<ProtocolWarning>,
    pub diagnostics: Vec<Diagnostic>,
    pub pass_summary: Vec<ProtocolPassSummary>,
}

impl ProtocolReport {
    pub fn new(status: ProtocolStatus) -> Self {
        Self {
            status,
            warnings: Vec::new(),
            diagnostics: Vec::new(),
            pass_summary: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessScope {
    pub stream_id: usize,
    pub cluster_id: usize,
    pub cta_id: usize,
    pub ctaid_in_cluster: usize,
    pub cohort_size: usize,
    pub warp_ids: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PoolId {
    Smem { cta_id: usize },
    Tmem { cta_id: usize },
    Gmem { tensor_id: u32 },
    Reg { cta_id: usize, tensor_id: u32 },
}

impl PoolId {
    pub fn kind(&self) -> &'static str {
        match self {
            PoolId::Smem { .. } => "smem",
            PoolId::Tmem { .. } => "tmem",
            PoolId::Gmem { .. } => "gmem",
            PoolId::Reg { .. } => "reg",
        }
    }
}

/// N-dimensional half-open physical byte box.
///
/// SMEM, GMEM, and REG use rank-1 byte ranges. TMEM uses rank-2 boxes in
/// `(lane, lane_byte)`, where a 32-bit TMEM column spans four lane bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoxN {
    pub ranges: Vec<(usize, usize)>,
}

impl BoxN {
    pub fn new(ranges: Vec<(usize, usize)>) -> Self {
        Self { ranges }
    }

    pub fn rank(&self) -> usize {
        self.ranges.len()
    }
}

/// Box set of a region: explicit boxes, or — the hot rank-1 form — a strided
/// run sequence. A rectangular slice whose inner extent is narrower than the
/// tensor row projects to `count` runs of `len` bytes every `stride` bytes
/// (`rect_byte_ranges` walks one non-unit outer dim); storing them as four
/// words instead of `count` boxes keeps trace memory flat and lets the
/// overlap check run in O(1) for the equal-stride case.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionBoxes {
    Boxes(Vec<BoxN>),
    Strided {
        start: usize,
        len: usize,
        stride: usize,
        count: usize,
    },
}

impl RegionBoxes {
    pub fn rank(&self) -> Option<usize> {
        match self {
            RegionBoxes::Boxes(boxes) => boxes.first().map(BoxN::rank),
            RegionBoxes::Strided { .. } => Some(1),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            RegionBoxes::Boxes(boxes) => boxes.is_empty(),
            RegionBoxes::Strided { count, .. } => *count == 0,
        }
    }

    /// Number of rank-1 runs (boxes) in the set.
    pub fn run_count(&self) -> usize {
        match self {
            RegionBoxes::Boxes(boxes) => boxes.len(),
            RegionBoxes::Strided { count, .. } => *count,
        }
    }

    /// Overall byte span `[first.start, last.end)` for rank-1 sets.
    pub fn rank1_bounds(&self) -> Option<(usize, usize)> {
        match self {
            RegionBoxes::Boxes(boxes) => {
                let first = boxes.first()?;
                let last = boxes.last()?;
                if first.ranges.len() != 1 || last.ranges.len() != 1 {
                    return None;
                }
                Some((first.ranges[0].0, last.ranges[last.ranges.len() - 1].1))
            }
            RegionBoxes::Strided {
                start,
                len,
                stride,
                count,
            } => {
                if *count == 0 {
                    return None;
                }
                Some((*start, start + (count - 1) * stride + len))
            }
        }
    }

    /// Iterate the rank-1 byte runs in ascending start order.
    pub fn iter_rank1(&self) -> Rank1Runs<'_> {
        match self {
            RegionBoxes::Boxes(boxes) => Rank1Runs::Boxes(boxes.iter()),
            RegionBoxes::Strided {
                start,
                len,
                stride,
                count,
            } => Rank1Runs::Strided {
                start: *start,
                len: *len,
                stride: *stride,
                remaining: *count,
            },
        }
    }

    /// Materialize as explicit boxes (serialization and diagnostics).
    pub fn to_boxes(&self) -> Vec<BoxN> {
        match self {
            RegionBoxes::Boxes(boxes) => boxes.clone(),
            RegionBoxes::Strided { .. } => self
                .iter_rank1()
                .map(|(s, e)| BoxN::new(vec![(s, e)]))
                .collect(),
        }
    }
}

/// Lazy iterator over a rank-1 run set, ascending by start.
pub enum Rank1Runs<'a> {
    Boxes(std::slice::Iter<'a, BoxN>),
    Strided {
        start: usize,
        len: usize,
        stride: usize,
        remaining: usize,
    },
}

impl Iterator for Rank1Runs<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        match self {
            Rank1Runs::Boxes(iter) => iter.next().map(|b| b.ranges[0]),
            Rank1Runs::Strided {
                start,
                len,
                stride,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                let run = (*start, *start + *len);
                *start += *stride;
                *remaining -= 1;
                Some(run)
            }
        }
    }
}

/// Physical byte footprint touched by a trace event.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub owner: PoolId,
    pub boxes: RegionBoxes,
    pub tensor_id: u32,
}

impl Region {
    pub fn new(owner: PoolId, mut boxes: Vec<BoxN>, tensor_id: u32) -> Self {
        // Construction invariant: rank-1 byte boxes are kept sorted by start, so
        // the rank-1 overlap fast path (`region::regions_overlap`) can sweep them
        // linearly without a per-comparison sortedness check. Production builders
        // already emit coalesced (sorted) boxes, so this is an O(n) no-op there;
        // it only reorders hand-built or future-projection regions. Rank-2 (TMEM)
        // regions use the order-independent product, so their order is untouched.
        if boxes.iter().all(|b| b.ranges.len() == 1) {
            boxes.sort_unstable_by_key(|b| b.ranges[0]);
        }
        Self {
            owner,
            boxes: RegionBoxes::Boxes(boxes),
            tensor_id,
        }
    }

    pub fn from_boxes(owner: PoolId, boxes: RegionBoxes, tensor_id: u32) -> Self {
        Self {
            owner,
            boxes,
            tensor_id,
        }
    }

    pub fn rank(&self) -> Option<usize> {
        self.boxes.rank()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryProxy {
    Generic,
    Async,
}

impl MemoryProxy {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryProxy::Generic => "generic",
            MemoryProxy::Async => "async",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorAccessKind {
    Generic,
    TmaLoad,
    TmaStore,
    Tcgen05Mma,
    Tcgen05Cp,
    LdMatrix,
    StMatrix,
}

impl TensorAccessKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TensorAccessKind::Generic => "generic",
            TensorAccessKind::TmaLoad => "tma_load",
            TensorAccessKind::TmaStore => "tma_store",
            TensorAccessKind::Tcgen05Mma => "tcgen05_mma",
            TensorAccessKind::Tcgen05Cp => "tcgen05_cp",
            TensorAccessKind::LdMatrix => "ldmatrix",
            TensorAccessKind::StMatrix => "stmatrix",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmemAsyncKind {
    Ld,
    St,
    Mma,
    Cp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAccessKind {
    Tensor(TensorAccessKind),
    Tmem(TmemAsyncKind),
}

impl MemoryAccessKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryAccessKind::Tensor(kind) => kind.as_str(),
            MemoryAccessKind::Tmem(kind) => kind.as_str(),
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            MemoryAccessKind::Tensor(_) => "tensor",
            MemoryAccessKind::Tmem(_) => "tmem",
        }
    }
}

impl TmemAsyncKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TmemAsyncKind::Ld => "ld",
            TmemAsyncKind::St => "st",
            TmemAsyncKind::Mma => "mma",
            TmemAsyncKind::Cp => "cp",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceEventKind {
    Generic,
    ProxyAsync,
}

impl FenceEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FenceEventKind::Generic => "generic",
            FenceEventKind::ProxyAsync => "proxy_async",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MbarTargetEvent {
    pub mbar_id: u32,
    pub cluster_id: usize,
    pub ctaid_in_cluster: usize,
    pub stage: u32,
}

impl From<MbarTarget> for MbarTargetEvent {
    fn from(target: MbarTarget) -> Self {
        Self {
            mbar_id: target.identity.mbar_id,
            cluster_id: target.identity.cluster_id,
            ctaid_in_cluster: target.identity.ctaid_in_cluster,
            stage: target.stage as u32,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub stmt_id: usize,
    pub stmt_kind: String,
    pub payload: TraceEventKind,
}

impl TraceEvent {
    pub fn new(stmt_id: usize, stmt_kind: impl Into<String>, payload: TraceEventKind) -> Self {
        Self {
            stmt_id,
            stmt_kind: stmt_kind.into(),
            payload,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEventKind {
    Read {
        region: Region,
        proxy: MemoryProxy,
        access_kind: MemoryAccessKind,
        scope: AccessScope,
    },
    Write {
        region: Region,
        proxy: MemoryProxy,
        access_kind: MemoryAccessKind,
        scope: AccessScope,
    },
    TmemWait {
        async_kind: TmemAsyncKind,
        scope: AccessScope,
    },
    Fence {
        fence_kind: FenceEventKind,
        fence_scope: FenceScope,
        scope: AccessScope,
    },
    CommitGroup {
        scope: AccessScope,
    },
    WaitGroup {
        n: u32,
        scope: AccessScope,
    },
    MbarInit {
        target: MbarTargetEvent,
        count: i64,
        scope: AccessScope,
    },
    MbarArrive {
        target: MbarTargetEvent,
        count: i64,
        scope: AccessScope,
    },
    MbarExpectTx {
        target: MbarTargetEvent,
        bytes: i64,
        scope: AccessScope,
    },
    MbarCompleteTx {
        target: MbarTargetEvent,
        bytes: i64,
        scope: AccessScope,
    },
    MbarWait {
        target: MbarTargetEvent,
        phase: u8,
        scope: AccessScope,
    },
    SyncArrive {
        sync_kind: String,
        thread_count: usize,
        count: usize,
        cycle: u64,
        bar_id: Option<u32>,
        scope: AccessScope,
    },
    Sync {
        sync_kind: String,
        thread_count: usize,
        cycle: u64,
        bar_id: Option<u32>,
        scope: AccessScope,
    },
    TmemAlloc {
        cta_ids: Vec<usize>,
        region: Region,
        scope: AccessScope,
    },
    TmemDealloc {
        cta_ids: Vec<usize>,
        region: Region,
        scope: AccessScope,
    },
    SchedulerNext {
        scheduler_id: u32,
        cta_id: usize,
        task_id: i64,
        scope: AccessScope,
    },
}

#[derive(Clone, Debug)]
pub struct TraceState {
    events: Option<Vec<TraceEvent>>,
    pub warnings: Vec<ProtocolWarning>,
}

impl Default for TraceState {
    fn default() -> Self {
        Self::new(true)
    }
}

impl TraceState {
    pub fn new(record_events: bool) -> Self {
        Self {
            events: record_events.then(Vec::new),
            warnings: Vec::new(),
        }
    }

    pub fn records_events(&self) -> bool {
        self.events.is_some()
    }

    pub fn emit(&mut self, event: TraceEvent) -> IResult<()> {
        if let Some(events) = &mut self.events {
            events.push(event);
        }
        Ok(())
    }

    pub fn warn(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.warnings.push(ProtocolWarning {
            code: code.into(),
            message: message.into(),
            details: BTreeMap::new(),
        });
    }

    pub fn finish(
        &mut self,
        status: ProtocolStatus,
        diagnostics: Vec<Diagnostic>,
    ) -> IResult<(ProtocolReport, Vec<TraceEvent>)> {
        let events = self.events.as_mut().map(std::mem::take).unwrap_or_default();
        let report = ProtocolReport {
            status,
            warnings: std::mem::take(&mut self.warnings),
            diagnostics,
            pass_summary: Vec::new(),
        };
        Ok((report, events))
    }
}
