//! Cooperative sync/rendezvous arrival sets — port of `values/cooperative.py`.

use super::super::threads::ThreadId;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default)]
pub struct CooperativeValues {
    /// per-occurrence arrival sets.
    pub syncs: HashMap<String, HashSet<ThreadId>>,
    /// per-occurrence completion sets.
    pub rendezvous: HashMap<String, HashSet<ThreadId>>,
    /// completed cycles per sync resource key.
    pub sync_cycles: HashMap<String, u64>,
}
