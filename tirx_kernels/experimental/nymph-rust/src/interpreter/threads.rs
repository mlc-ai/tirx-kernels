//! Thread identity & cohort masks — port of `interpreter/threads.py`.
//!
//! ThreadId is `Copy` (no heap): coords are a fixed-size `Coord` (rank <= 3), so
//! cloning a cohort and hashing the scalar-env key are allocation-free — this is
//! on the hottest path (every statement clones its cohort).

/// A small fixed-capacity coordinate (launch/cluster rank is 1..=3).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Coord {
    data: [usize; 3],
    len: u8,
}

impl Coord {
    pub fn from_slice(s: &[usize]) -> Coord {
        let mut data = [0usize; 3];
        for (i, &v) in s.iter().enumerate() {
            data[i] = v;
        }
        Coord {
            data,
            len: s.len() as u8,
        }
    }
    pub fn as_slice(&self) -> &[usize] {
        &self.data[..self.len as usize]
    }
    pub fn to_vec(&self) -> Vec<usize> {
        self.as_slice().to_vec()
    }
}

/// Modeled GPU thread identity. `Copy` + value-hashable.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ThreadId {
    pub cta_id: usize,
    pub cta_coord: Coord,
    pub cluster_id: usize,
    pub ctaid_in_cluster: usize,
    pub cluster_coord: Coord,
    pub cta_coord_in_cluster: Coord,
    pub warp_id: usize,
    pub lane_id: usize,
}

impl ThreadId {
    pub fn warpgroup_id(&self) -> usize {
        self.warp_id / 4
    }
    pub fn tid_in_wg(&self) -> usize {
        (self.warp_id % 4) * 32 + self.lane_id
    }
}

/// An ordered, deduped cohort of threads (canonical order = sorted by (warp, lane)).
pub type ThreadMask = Vec<ThreadId>;

/// Dedupe then sort by (warp_id, lane_id). `mask[0]` is the canonical first thread.
pub fn canonical_thread_mask(threads: impl IntoIterator<Item = ThreadId>) -> ThreadMask {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<ThreadId> = Vec::new();
    for t in threads {
        if seen.insert((t.cta_id, t.warp_id, t.lane_id)) {
            out.push(t);
        }
    }
    out.sort_by_key(|t| (t.warp_id, t.lane_id));
    out
}

/// Keep threads passing `predicate`, then re-canonicalize.
pub fn filter_thread_mask(mask: &ThreadMask, predicate: impl Fn(&ThreadId) -> bool) -> ThreadMask {
    canonical_thread_mask(mask.iter().filter(|t| predicate(t)).copied())
}
