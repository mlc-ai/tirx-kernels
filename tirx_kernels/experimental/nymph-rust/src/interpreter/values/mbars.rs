//! mbarrier runtime values — port of `interpreter/values/mbars.py`.
//! Snapshot containers + keys only; the phase algebra lives in `mbar_ops`.

use std::collections::HashMap;

/// Barrier identity resolved to a cluster/CTA coordinate. `mbar_id` is the IR
/// mbar's stable id (u32).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MbarIdentity {
    pub mbar_id: u32,
    pub cluster_id: usize,
    pub ctaid_in_cluster: usize,
}

/// (identity, stage). The concrete wake token today.
pub type MbarCellKey = (MbarIdentity, usize);

/// One initialized mbarrier phase cell. Immutable — transitions produce a new cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MbarCell {
    pub expected_arrivals: i64,
    pub pending_arrivals: i64,
    pub pending_tx_bytes: i64,
    pub parity: u8,
    pub stage: usize,
}

#[derive(Clone, Debug, Default)]
pub struct MbarValues {
    pub declared_counts: HashMap<u32, Option<i64>>,
    pub cells: HashMap<MbarCellKey, MbarCell>,
}

impl MbarValues {
    /// First declaration wins (idempotent).
    pub fn declare(&mut self, mbar_id: u32, arrive_count: Option<i64>) {
        self.declared_counts.entry(mbar_id).or_insert(arrive_count);
    }
}
