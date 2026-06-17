//! TMEM allocation lifecycle bookkeeping — port of `interpreter/tmem.py`.
//! Internal scheduling state (internal, not part of the returned RunResult values).

use std::collections::BTreeSet;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TmemAllocationKey {
    pub cta_id: usize,
    pub col_start: usize,
    pub n_cols: usize,
}

#[derive(Clone, Debug)]
pub struct TmemAllocation {
    pub col_start: usize,
    pub n_cols: usize,
    pub cta_group: u8,
    pub collective_cta_ids: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TmemCollectiveKey {
    pub stmt_id: usize,
    pub op: &'static str, // "alloc" | "dealloc"
    pub col_start: usize,
    pub n_cols: usize,
    pub cluster_id: usize,
    pub pair_base_ctaid_in_cluster: usize,
}

#[derive(Clone, Debug)]
pub struct TmemCollectiveArrival {
    pub cta_id: usize,
    pub ctaid_in_cluster: usize,
    pub stream_id: usize,
    pub col_start: usize,
    pub n_cols: usize,
    pub cta_group: u8,
}

#[derive(Clone, Debug, Default)]
pub struct TmemCollective {
    pub arrivals: Vec<TmemCollectiveArrival>,
    pub completed_cta_ids: BTreeSet<usize>,
}

impl TmemCollective {
    pub fn arrival_for_cta(&self, cta_id: usize) -> Option<&TmemCollectiveArrival> {
        self.arrivals.iter().find(|a| a.cta_id == cta_id)
    }
    pub fn with_arrival(&self, arrival: TmemCollectiveArrival) -> TmemCollective {
        if self.arrival_for_cta(arrival.cta_id).is_some() {
            return self.clone();
        }
        let mut next = self.clone();
        next.arrivals.push(arrival);
        next
    }
    pub fn with_completed(&self, cta_id: usize) -> TmemCollective {
        let mut next = self.clone();
        next.completed_cta_ids.insert(cta_id);
        next
    }
}
