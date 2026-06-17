//! Tile scheduler IR nodes.

use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SchedulerPolicy {
    GridStride,
    Clc,
    AtomicSteal,
    Custom,
}

impl SchedulerPolicy {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "grid_stride" => Some(SchedulerPolicy::GridStride),
            "clc" => Some(SchedulerPolicy::Clc),
            "atomic_steal" => Some(SchedulerPolicy::AtomicSteal),
            "custom" => Some(SchedulerPolicy::Custom),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SchedulerPolicy::GridStride => "grid_stride",
            SchedulerPolicy::Clc => "clc",
            SchedulerPolicy::AtomicSteal => "atomic_steal",
            SchedulerPolicy::Custom => "custom",
        }
    }

    pub fn is_functional(self) -> bool {
        matches!(self, SchedulerPolicy::GridStride)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SchedulerScope {
    Cluster,
}

impl SchedulerScope {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "cluster" => Some(SchedulerScope::Cluster),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SchedulerScope::Cluster => "cluster",
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TaskSpace {
    pub id: u32,
    pub grid: Vec<usize>,
    pub fields: Vec<String>,
}

impl TaskSpace {
    pub fn task_count(&self) -> Option<usize> {
        self.grid
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Scheduler {
    pub id: u32,
    pub space: Arc<TaskSpace>,
    pub policy: SchedulerPolicy,
    pub scope: SchedulerScope,
}
