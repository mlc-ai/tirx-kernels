//! Statement identity — the Rust analog of `interpreter/ids.py`.
//!
//! Tensors/MBars/Vars already carry stable u32 ids from the builder, so those are
//! used directly (with `t{n}`/`m{n}`/`v{n}` formatting for display). Statements
//! have no id field, so — exactly like Python keys them by `id(stmt)` — we assign
//! each a sequential id during one pre-order walk and key by the statement's
//! address (`*const Stmt`). The kernel is immutable during a run, so addresses are
//! stable. (This makes IdSpace !Send, fine for the single-threaded interpreter.)

use crate::ir::{Kernel, Stmt};
use std::collections::HashMap;

#[derive(Default)]
pub struct IdSpace {
    stmt_ids: HashMap<*const Stmt, usize>,
}

impl IdSpace {
    pub fn discover(kernel: &Kernel) -> IdSpace {
        let mut ids = IdSpace::default();
        ids.walk(&kernel.body);
        ids
    }

    fn walk(&mut self, body: &[Stmt]) {
        for stmt in body {
            let next = self.stmt_ids.len();
            self.stmt_ids.insert(stmt as *const Stmt, next);
            for child in stmt.child_bodies() {
                self.walk(child);
            }
        }
    }

    pub fn stmt_id(&self, stmt: &Stmt) -> usize {
        *self
            .stmt_ids
            .get(&(stmt as *const Stmt))
            .expect("stmt not discovered (foreign statement)")
    }
}

pub fn tensor_id_str(id: u32) -> String {
    format!("t{id}")
}
pub fn mbar_id_str(id: u32) -> String {
    format!("m{id}")
}
pub fn var_id_str(id: u32) -> String {
    format!("v{id}")
}
pub fn stmt_id_str(id: usize) -> String {
    format!("s{id}")
}
