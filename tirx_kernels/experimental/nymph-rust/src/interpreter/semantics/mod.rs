//! Built-in statement executor modules — port of `semantics/__init__.py`.
//!
//! `default_executor_registry` iterates the registrars to build the dispatch
//! table. Adding a new op family = add its module + append its `register` here —
//! the runner is never edited. `leaf` is last (installs the fallback).

pub mod control;
pub mod cp_async;
pub mod fence;
pub mod ldstmatrix;
pub mod leaf;
pub mod mbarrier;
pub mod metadata;
pub mod reg;
pub mod scalar;
pub mod sync;
pub mod tcgen05;
pub mod tma;
pub mod tmem;

use super::registry::StmtExecutorRegistry;

pub fn default_executor_registry() -> StmtExecutorRegistry {
    let mut registry = StmtExecutorRegistry::default();
    metadata::register(&mut registry);
    scalar::register(&mut registry);
    control::register(&mut registry);
    reg::register(&mut registry);
    tmem::register(&mut registry);
    mbarrier::register(&mut registry);
    tma::register(&mut registry);
    tcgen05::register(&mut registry);
    ldstmatrix::register(&mut registry);
    fence::register(&mut registry);
    sync::register(&mut registry);
    cp_async::register(&mut registry);
    leaf::register(&mut registry);
    registry
}
