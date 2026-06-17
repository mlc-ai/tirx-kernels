//! The cohort-vectorized execution surface — port of `cohort.py` (+ the
//! `cohort_services.py` register/shared read-write helpers, merged here as methods
//! to keep the borrow story simple; the per-op modularity is preserved by the
//! registry, not by splitting the services).

use super::diagnostics::{IResult, InterpreterError};
use super::ids::IdSpace;
use super::outcomes::StepStatus;
use super::protocol::{
    AccessScope, ExecutionMode, MemoryAccessKind, MemoryProxy, Region, TensorAccessKind,
    TraceEvent, TraceEventKind,
};
use super::region;
use super::registry::StmtKind;
use super::scalar_eval;
use super::scheduler::{flatten_coord, unflatten_coord, CtaActivityStatus, ExecutionStream};
use super::slice_indexing::{shared_flat_indices, uniform_contiguous_flat_range, ResolvedSlice};
use super::state::{InterpreterState, RunOptions};
use super::threads::{ThreadId, ThreadMask};
use super::values::arrays::ValueArray2;
use super::values::registers::{register_row, RegisterKey, RegisterTensorValue};
use super::values::tensors::{tensor_instance_key, DenseTensorValue};
use crate::ir::{Kernel, MemorySpace, ScalarValue, Stmt, TensorSlice};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct CohortContext<'a, 'k> {
    pub kernel: &'k Kernel,
    pub stream: &'a mut ExecutionStream<'k>,
    pub cohort: ThreadMask,
    pub state: &'a mut InterpreterState,
    pub ids: &'a IdSpace,
    pub options: &'a RunOptions,
    pub cta_activity: &'a [CtaActivityStatus],
    pub current_stmt_id: usize,
    pub current_stmt_kind: StmtKind,
}

/// Executor signature — registered per Stmt variant. Higher-ranked over `'k` so a
/// control executor can push a `&'k [Stmt]` body onto the stream's frame stack.
/// Executors mutate `ctx.state` directly and return a `StepStatus`.
pub type StmtExecutor = for<'a, 'k> fn(&mut CohortContext<'a, 'k>, &'k Stmt) -> IResult<StepStatus>;

fn resolved_offset_rows(resolved: &ResolvedSlice) -> Vec<Vec<i64>> {
    resolved
        .offsets
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}

fn uniform_offset(resolved: &ResolvedSlice) -> Option<Vec<i64>> {
    if resolved.offsets.nrows() == 0 {
        return None;
    }
    let first: Vec<i64> = resolved.offsets.row(0).iter().copied().collect();
    resolved
        .offsets
        .rows()
        .into_iter()
        .all(|row| row.iter().copied().eq(first.iter().copied()))
        .then_some(first)
}

fn contiguous_register_row_span(cohort: &ThreadMask) -> Option<(usize, usize)> {
    let first = cohort.iter().next().map(register_row)?;
    cohort
        .iter()
        .skip(1)
        .enumerate()
        .all(|(i, thread)| register_row(thread) == first + i + 1)
        .then_some((first, cohort.len()))
}

impl<'a, 'k> CohortContext<'a, 'k> {
    pub fn mode(&self) -> ExecutionMode {
        self.state.mode
    }
    pub fn value_mode(&self) -> bool {
        self.state.mode.is_value()
    }
    pub fn trace_mode(&self) -> bool {
        self.state.mode.is_trace()
    }
    pub fn anchored_event(&self, payload: TraceEventKind) -> TraceEvent {
        TraceEvent::new(
            self.current_stmt_id,
            format!("{:?}", self.current_stmt_kind),
            payload,
        )
    }
    pub fn emit(&mut self, payload: TraceEventKind) -> IResult<()> {
        if self.trace_mode() {
            self.state.trace.emit(self.anchored_event(payload))?;
        }
        Ok(())
    }
    pub fn trace_inconclusive(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> InterpreterError {
        self.state.trace.warn(code, message);
        InterpreterError::new("trace_inconclusive", "protocol trace is inconclusive")
    }
    pub fn access_scope(&self) -> AccessScope {
        let mut warp_ids: Vec<usize> = self.cohort.iter().map(|t| t.warp_id).collect();
        warp_ids.sort_unstable();
        warp_ids.dedup();
        AccessScope {
            stream_id: self.stream.stream_id,
            cluster_id: self.stream.cluster_id,
            cta_id: self.stream.cta_id,
            ctaid_in_cluster: self.stream.ctaid_in_cluster,
            cohort_size: self.cohort.len(),
            warp_ids,
        }
    }
    pub fn tensor_region(&mut self, resolved: &ResolvedSlice) -> IResult<Region> {
        if resolved.tensor.space == MemorySpace::Reg {
            if let (Some(offset), Some((row_start, row_count))) = (
                uniform_offset(resolved),
                contiguous_register_row_span(&self.cohort),
            ) {
                return region::reg_tensor_region_from_uniform_offset(
                    &resolved.tensor,
                    self.stream.cta_id,
                    row_start,
                    row_count,
                    &offset,
                    &resolved.shape,
                );
            }
            let offsets = resolved_offset_rows(resolved);
            let register_rows: Vec<usize> = self.cohort.iter().map(register_row).collect();
            return region::reg_tensor_region_from_offsets(
                &resolved.tensor,
                self.stream.cta_id,
                &register_rows,
                &offsets,
                &resolved.shape,
            );
        }
        let offsets = resolved_offset_rows(resolved);
        region::tensor_region_from_offsets(
            &resolved.tensor,
            self.stream.cta_id,
            &offsets,
            &resolved.shape,
        )
    }
    pub fn tensor_region_with_offsets(
        &mut self,
        resolved: &ResolvedSlice,
        offsets: &[Vec<i64>],
    ) -> IResult<Region> {
        if resolved.tensor.space == MemorySpace::Reg {
            let register_rows: Vec<usize> = (0..offsets.len()).collect();
            region::reg_tensor_region_from_offsets(
                &resolved.tensor,
                self.stream.cta_id,
                &register_rows,
                offsets,
                &resolved.shape,
            )
        } else {
            region::tensor_region_from_offsets(
                &resolved.tensor,
                self.stream.cta_id,
                offsets,
                &resolved.shape,
            )
        }
    }
    pub fn emit_tensor_read(&mut self, resolved: &ResolvedSlice) -> IResult<()> {
        self.emit_tensor_read_with_proxy(resolved, MemoryProxy::Generic)
    }
    pub fn emit_tensor_read_with_proxy(
        &mut self,
        resolved: &ResolvedSlice,
        proxy: MemoryProxy,
    ) -> IResult<()> {
        self.emit_tensor_read_with_kind(resolved, proxy, TensorAccessKind::Generic)
    }
    pub fn emit_tensor_read_with_kind(
        &mut self,
        resolved: &ResolvedSlice,
        proxy: MemoryProxy,
        access_kind: TensorAccessKind,
    ) -> IResult<()> {
        if !self.state.trace.records_events() {
            return Ok(());
        }
        let region = self.tensor_region(resolved)?;
        let scope = self.access_scope();
        self.emit(TraceEventKind::Read {
            region,
            proxy,
            access_kind: MemoryAccessKind::Tensor(access_kind),
            scope,
        })
    }
    pub fn emit_tensor_write(&mut self, resolved: &ResolvedSlice) -> IResult<()> {
        self.emit_tensor_write_with_proxy(resolved, MemoryProxy::Generic)
    }
    pub fn emit_tensor_write_with_proxy(
        &mut self,
        resolved: &ResolvedSlice,
        proxy: MemoryProxy,
    ) -> IResult<()> {
        self.emit_tensor_write_with_kind(resolved, proxy, TensorAccessKind::Generic)
    }
    pub fn emit_tensor_write_with_kind(
        &mut self,
        resolved: &ResolvedSlice,
        proxy: MemoryProxy,
        access_kind: TensorAccessKind,
    ) -> IResult<()> {
        if !self.state.trace.records_events() {
            return Ok(());
        }
        let region = self.tensor_region(resolved)?;
        let scope = self.access_scope();
        self.emit(TraceEventKind::Write {
            region,
            proxy,
            access_kind: MemoryAccessKind::Tensor(access_kind),
            scope,
        })
    }
    pub fn invalidate_shared_region(&mut self, resolved: &ResolvedSlice) -> IResult<()> {
        match resolved.tensor.space {
            MemorySpace::Smem => {
                let (idx, _) = shared_flat_indices(resolved, &resolved.tensor.shape)?;
                let indices: Vec<usize> = idx.into_raw_vec_and_offset().0;
                let pool = self
                    .state
                    .values
                    .smem
                    .pool_for_mut(self.stream.cta_id, self.kernel.smem_size_bytes);
                pool.invalidate_indices(&resolved.tensor, &indices)
            }
            MemorySpace::Gmem => {
                let key = tensor_instance_key(self.stream.cta_id, &resolved.tensor)?;
                let (idx, _) = shared_flat_indices(resolved, &resolved.tensor.shape)?;
                let indices: Vec<usize> = idx.into_raw_vec_and_offset().0;
                let inst = self
                    .state
                    .values
                    .tensors
                    .by_instance
                    .entry(key)
                    .or_insert_with(|| {
                        DenseTensorValue::empty(
                            resolved.tensor.shape.clone(),
                            resolved.tensor.dtype,
                        )
                    });
                inst.invalidate_indices(&indices)
            }
            _ => Ok(()),
        }
    }
    pub fn n_cta_threads(&self) -> usize {
        32 * self.kernel.num_warps as usize
    }
    pub fn cluster_cta_count(&self) -> usize {
        self.kernel.cluster_shape.iter().product()
    }
    pub fn stmt_id(&self, stmt: &Stmt) -> usize {
        self.ids.stmt_id(stmt)
    }
    pub fn check_full_warp_cohort(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> IResult<()> {
        let code = code.into();
        let message = message.into();
        if self.cohort.is_empty() || self.cohort.len() % 32 != 0 {
            return Err(InterpreterError::new(code.clone(), message.clone()));
        }
        let mut lanes_by_warp: HashMap<(usize, usize), u32> = HashMap::new();
        for thread in &self.cohort {
            if thread.lane_id >= 32 {
                return Err(InterpreterError::new(code.clone(), message.clone()));
            }
            *lanes_by_warp
                .entry((thread.cta_id, thread.warp_id))
                .or_insert(0) |= 1u32 << thread.lane_id;
        }
        for lanes in lanes_by_warp.values() {
            if *lanes != u32::MAX {
                return Err(InterpreterError::new(code.clone(), message.clone()));
            }
        }
        Ok(())
    }
    pub fn cta_activity(&self, cta_id: usize) -> CtaActivityStatus {
        self.cta_activity
            .get(cta_id)
            .copied()
            .unwrap_or(CtaActivityStatus::Missing)
    }

    /// Map a cluster-local CTA id to the global launch-grid CTA id.
    pub fn global_cta_id(&self, ctaid_in_cluster: usize) -> usize {
        let cluster = &self.kernel.cluster_shape;
        let local = unflatten_coord(ctaid_in_cluster, cluster);
        let cta_coord: Vec<usize> = cluster
            .iter()
            .zip(self.stream.cluster_coord.iter())
            .zip(local.iter())
            .map(|((cl, cc), l)| cl * cc + l)
            .collect();
        flatten_coord(&cta_coord, &self.kernel.launch_shape)
    }

    pub fn eval_scalar_vec(&self, value: &ScalarValue) -> IResult<Array1<i64>> {
        scalar_eval::eval_scalar_vec(value, &self.cohort, &self.state.values.scalars)
    }
    pub fn eval_scalar_at(&self, value: &ScalarValue, thread: &ThreadId) -> IResult<i64> {
        scalar_eval::eval_scalar_at(value, thread, &self.state.values.scalars)
    }
    pub fn eval_scalar_uniform(
        &self,
        value: &ScalarValue,
        label: &str,
        code: &str,
    ) -> IResult<i64> {
        scalar_eval::eval_scalar_uniform(
            value,
            &self.cohort,
            &self.state.values.scalars,
            label,
            code,
        )
    }

    pub fn rows(&self) -> Vec<usize> {
        self.cohort.iter().map(register_row).collect()
    }

    /// Resolve a slice over the cohort: per-thread offsets `[A, rank]` + uniform shape.
    pub fn eval_slice(&self, value: &TensorSlice) -> IResult<ResolvedSlice> {
        let rank = value.offsets.len();
        let a = self.cohort.len();
        let mut offsets = Array2::<i64>::zeros((a, rank));
        for (d, off) in value.offsets.iter().enumerate() {
            let col = self.eval_scalar_vec(off)?;
            for ai in 0..a {
                offsets[[ai, d]] = col[ai];
            }
        }
        let mut shape = Vec::with_capacity(value.shape.len());
        for dim in &value.shape {
            let s = self.eval_scalar_uniform(dim, "tensor slice shape", "divergent_operands")?;
            shape.push(s as usize);
        }
        // Out-of-bounds: each active thread's offset must keep the slice inside the tensor.
        // This is done here, at slice resolution, so it runs in BOTH modes: trace records an
        // op's read/write footprint by resolving these same offsets, so the check is free and
        // catches the slices a skipped (trace-mode) read/write would otherwise leave unchecked
        // — including per-thread offsets like `[lane_id, i]`, where every lane is verified.
        let tshape = &value.tensor.shape;
        if rank != tshape.len() || shape.len() != tshape.len() {
            return Err(InterpreterError::new(
                "tensor_value",
                "tensor slice rank mismatch",
            ));
        }
        for d in 0..rank {
            let extent = shape[d] as i64;
            let tdim = tshape[d] as i64;
            for ai in 0..a {
                let off = offsets[[ai, d]];
                if off < 0 || off > tdim - extent {
                    return Err(InterpreterError::new(
                        "tensor_value",
                        "tensor slice is out of bounds",
                    ));
                }
            }
        }
        Ok(ResolvedSlice {
            tensor: value.tensor.clone(),
            offsets,
            shape,
        })
    }

    // ---- shared (GMEM/SMEM) ----
    pub fn shared_read(&self, resolved: &ResolvedSlice) -> IResult<ValueArray2> {
        if resolved.tensor.space == MemorySpace::Smem {
            let (idx, k) = shared_flat_indices(resolved, &resolved.tensor.shape)?;
            let indices: Vec<usize> = idx.into_raw_vec_and_offset().0;
            let values = self
                .state
                .values
                .smem
                .pool_for(self.stream.cta_id)?
                .read_indices(&resolved.tensor, &indices)?;
            return values.reshape2((resolved.offsets.nrows(), k));
        }
        let key = tensor_instance_key(self.stream.cta_id, &resolved.tensor)?;
        let value = self
            .state
            .values
            .tensors
            .by_instance
            .get(&key)
            .ok_or_else(|| {
                InterpreterError::new("missing_tensor_value", "tensor instance is not written")
            })?;
        let (idx, k) = shared_flat_indices(resolved, &resolved.tensor.shape)?;
        let a = idx.nrows();
        let av = value.all_valid;
        for ai in 0..a {
            for j in 0..k {
                let flat = idx[[ai, j]];
                if !av && !value.valid[flat] {
                    return Err(InterpreterError::new(
                        "missing_tensor_value",
                        "tensor slice reads unwritten elements",
                    ));
                }
            }
        }
        Ok(value.data.gather2(&idx))
    }

    /// Scatter a cohort's `values` straight into the shared (GMEM/SMEM) tensor
    /// instance.
    pub fn shared_write(&mut self, resolved: &ResolvedSlice, values: &ValueArray2) -> IResult<()> {
        if resolved.tensor.space == MemorySpace::Smem {
            let (idx, k) = shared_flat_indices(resolved, &resolved.tensor.shape)?;
            if values.nrows() != idx.nrows() || values.ncols() != k {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "shared slice write value count mismatch",
                ));
            }
            if values.dtype() != resolved.tensor.dtype {
                return Err(InterpreterError::new(
                    "tensor_value",
                    "shared write dtype must match the destination container",
                ));
            }
            let indices: Vec<usize> = idx.into_raw_vec_and_offset().0;
            let mut sorted = indices.clone();
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                return Err(InterpreterError::new(
                    "overlapping_tensor_write",
                    "cohort shared write overlaps",
                ));
            }
            let flat_values = values.flatten_to_1d();
            let pool = self
                .state
                .values
                .smem
                .pool_for_mut(self.stream.cta_id, self.kernel.smem_size_bytes);
            return pool.write_indices(&resolved.tensor, &indices, &flat_values);
        }
        let key = tensor_instance_key(self.stream.cta_id, &resolved.tensor)?;
        let (idx, k) = shared_flat_indices(resolved, &resolved.tensor.shape)?;
        if values.nrows() != idx.nrows() || values.ncols() != k {
            return Err(InterpreterError::new(
                "tensor_value",
                "shared slice write value count mismatch",
            ));
        }
        if values.dtype() != resolved.tensor.dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "shared write dtype must match the destination container",
            ));
        }
        // contiguous row-major flatten (no per-element 2D indexing)
        let indices: Vec<usize> = idx.into_raw_vec_and_offset().0;
        // overlap check via sort (faster than a SipHash HashSet on the hot path)
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(InterpreterError::new(
                "overlapping_tensor_write",
                "cohort shared write overlaps",
            ));
        }
        let inst = self
            .state
            .values
            .tensors
            .by_instance
            .entry(key)
            .or_insert_with(|| {
                DenseTensorValue::empty(resolved.tensor.shape.clone(), resolved.tensor.dtype)
            });
        if inst.dtype != values.dtype() {
            return Err(InterpreterError::new(
                "tensor_value",
                "shared write dtype must match the destination container",
            ));
        }
        let track = !inst.all_valid;
        let flat_values = values.flatten_to_1d();
        inst.data.scatter_indices(&indices, &flat_values)?;
        if track {
            for &flat in &indices {
                inst.valid[flat] = true;
            }
        }
        Ok(())
    }

    // ---- registers ----
    fn register_cols(&self, resolved: &ResolvedSlice) -> IResult<Array2<usize>> {
        Ok(shared_flat_indices(resolved, &resolved.tensor.shape)?.0)
    }

    fn register_contiguous_range(
        &self,
        resolved: &ResolvedSlice,
    ) -> IResult<Option<(usize, usize)>> {
        uniform_contiguous_flat_range(resolved, &resolved.tensor.shape)
    }

    pub fn registers_read(&self, resolved: &ResolvedSlice) -> IResult<ValueArray2> {
        let inst = self
            .state
            .values
            .registers
            .get(&resolved.tensor, self.stream.cta_id)
            .ok_or_else(|| {
                InterpreterError::new("missing_tensor_value", "register instance is not written")
            })?;
        let rows = self.rows();
        if let Some((start, len)) = self.register_contiguous_range(resolved)? {
            return inst.gather_row_range(&rows, start, len);
        }
        let cols = self.register_cols(resolved)?;
        inst.gather_rows(&rows, &cols)
    }

    /// Scatter a cohort's `values` straight into the register instance.
    pub fn registers_write(
        &mut self,
        resolved: &ResolvedSlice,
        values: &ValueArray2,
    ) -> IResult<()> {
        if values.dtype() != resolved.tensor.dtype {
            return Err(InterpreterError::new(
                "tensor_value",
                "register write dtype must match the destination container",
            ));
        }
        let rows = self.rows();
        let n = self.n_cta_threads();
        let contiguous_range = self.register_contiguous_range(resolved)?;
        let cols = if contiguous_range.is_none() {
            Some(self.register_cols(resolved)?)
        } else {
            None
        };
        let key = RegisterKey {
            tensor: resolved.tensor.clone(),
            cta_id: self.stream.cta_id,
        };
        let inst = self
            .state
            .values
            .registers
            .by_instance
            .entry(key)
            .or_insert_with(|| {
                RegisterTensorValue::empty(resolved.tensor.shape.clone(), resolved.tensor.dtype, n)
            });
        if let Some((start, len)) = contiguous_range {
            inst.scatter_row_range(&rows, start, len, values)?;
            return Ok(());
        }
        let cols = cols.expect("register columns must be available for non-contiguous writes");
        if values.shape() != cols.dim() {
            return Err(InterpreterError::new(
                "tensor_value",
                "register slice write value count mismatch",
            ));
        }
        inst.scatter_rows(&rows, &cols, values)?;
        Ok(())
    }
}

/// Instance-key space dispatch used by the shared read/write (re-exported for
/// tests/semantics that build keys directly).
pub fn shared_space_ok(space: MemorySpace) -> bool {
    matches!(space, MemorySpace::Gmem | MemorySpace::Smem)
}
