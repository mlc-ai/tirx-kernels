//! Per-thread scalar environment — port of `interpreter/values/scalars.py`.
//! Keyed by thread, then by Var id (u32). Values are i64 (the index ALU type).

use super::super::threads::{ThreadId, ThreadMask};
use std::collections::HashMap;

const THREAD_BIT_WORDS: usize = 32;

#[derive(Clone, Copy, Debug)]
struct ThreadBits {
    cta_id: usize,
    words: [u64; THREAD_BIT_WORDS],
}

impl ThreadBits {
    fn from_thread(thread: &ThreadId) -> Option<Self> {
        let mut bits = ThreadBits {
            cta_id: thread.cta_id,
            words: [0; THREAD_BIT_WORDS],
        };
        bits.set(thread)?;
        Some(bits)
    }

    fn from_mask(mask: &ThreadMask) -> Option<Self> {
        let first = mask.first()?;
        let mut bits = ThreadBits {
            cta_id: first.cta_id,
            words: [0; THREAD_BIT_WORDS],
        };
        for thread in mask {
            if thread.cta_id != bits.cta_id {
                return None;
            }
            bits.set(thread)?;
        }
        Some(bits)
    }

    fn set(&mut self, thread: &ThreadId) -> Option<()> {
        let bit = thread
            .warp_id
            .checked_mul(32)?
            .checked_add(thread.lane_id)?;
        let word = bit / 64;
        if word >= THREAD_BIT_WORDS {
            return None;
        }
        self.words[word] |= 1u64 << (bit % 64);
        Some(())
    }

    fn overlaps(self, other: Self) -> bool {
        self.cta_id == other.cta_id
            && self
                .words
                .iter()
                .zip(other.words.iter())
                .any(|(a, b)| (a & b) != 0)
    }

    fn contains_all(self, query: Self) -> bool {
        self.cta_id == query.cta_id
            && self
                .words
                .iter()
                .zip(query.words.iter())
                .all(|(fact, q)| (q & !fact) == 0)
    }
}

#[derive(Clone, Copy, Debug)]
struct UniformFact {
    bits: ThreadBits,
    value: i64,
}

#[derive(Clone, Debug, Default)]
pub struct ScalarValues {
    pub by_thread: HashMap<ThreadId, HashMap<u32, i64>>,
    uniform_by_var: HashMap<u32, Vec<UniformFact>>,
}

impl ScalarValues {
    pub fn ensure_thread(&mut self, thread: &ThreadId) {
        self.by_thread.entry(*thread).or_default();
    }
    pub fn write_thread(&mut self, thread: &ThreadId, var_id: u32, value: i64) {
        if let Some(bits) = ThreadBits::from_thread(thread) {
            self.invalidate_overlapping(var_id, bits);
        } else {
            self.uniform_by_var.remove(&var_id);
        }
        self.by_thread
            .entry(*thread)
            .or_default()
            .insert(var_id, value);
    }
    pub fn write_mask(&mut self, mask: &ThreadMask, var_id: u32, value: i64) {
        let bits = ThreadBits::from_mask(mask);
        if let Some(bits) = bits {
            self.invalidate_overlapping(var_id, bits);
        } else {
            self.uniform_by_var.remove(&var_id);
        }
        for t in mask {
            self.by_thread.entry(*t).or_default().insert(var_id, value);
        }
        if let Some(bits) = bits {
            self.uniform_by_var
                .entry(var_id)
                .or_default()
                .push(UniformFact { bits, value });
        }
    }
    pub fn write_values(&mut self, mask: &ThreadMask, var_id: u32, values: &[i64]) {
        let bits = ThreadBits::from_mask(mask);
        if let Some(bits) = bits {
            self.invalidate_overlapping(var_id, bits);
        } else {
            self.uniform_by_var.remove(&var_id);
        }
        let mut first_value = None;
        let mut uniform = true;
        for (t, &value) in mask.iter().zip(values.iter()) {
            if let Some(first) = first_value {
                if value != first {
                    uniform = false;
                }
            } else {
                first_value = Some(value);
            }
            self.by_thread.entry(*t).or_default().insert(var_id, value);
        }
        if uniform {
            if let (Some(bits), Some(value)) = (bits, first_value) {
                self.uniform_by_var
                    .entry(var_id)
                    .or_default()
                    .push(UniformFact { bits, value });
            }
        }
    }
    pub fn get(&self, thread: &ThreadId, var_id: u32) -> Option<i64> {
        self.by_thread
            .get(thread)
            .and_then(|env| env.get(&var_id).copied())
    }
    pub fn uniform_value(&self, mask: &ThreadMask, var_id: u32) -> Option<i64> {
        let query = ThreadBits::from_mask(mask)?;
        self.uniform_by_var
            .get(&var_id)?
            .iter()
            .find_map(|fact| fact.bits.contains_all(query).then_some(fact.value))
    }

    fn invalidate_overlapping(&mut self, var_id: u32, written: ThreadBits) {
        if let Some(facts) = self.uniform_by_var.get_mut(&var_id) {
            facts.retain(|fact| !fact.bits.overlaps(written));
            if facts.is_empty() {
                self.uniform_by_var.remove(&var_id);
            }
        }
    }
}
