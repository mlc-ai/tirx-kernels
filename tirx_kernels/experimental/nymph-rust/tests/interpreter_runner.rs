//! Rust-only interpreter residual tests.
//!
//! Value and fail-closed paths that can be observed through
//! `nr.interpret(kernel, inputs)` live in Python builder tests. This file keeps
//! coverage for interpreter internals that the Python binding does not expose:
//! partial result suppression, mbarrier cells, TMEM scratchpads, blocked
//! frontiers, scalar environments, and cooperative cleanup state.

use nymph_rs::interpreter::interpret;
use nymph_rs::interpreter::values::arrays::ValueArray1;
use nymph_rs::interpreter::{
    ExecutionMode, ProtocolStatus, RunOptions, RunPayload, RunResult, TraceEvent, TraceEventKind,
};
use nymph_rs::ir::*;
use std::collections::HashMap;
use std::sync::Arc;

fn var(id: u32, binding: VarBinding) -> Var {
    Var {
        id: VarId(id),
        binding,
        dtype: ScalarDType::I32,
    }
}

fn tmem_tensor(id: u32, col_start: usize) -> Arc<Tensor> {
    Arc::new(Tensor {
        id,
        space: MemorySpace::Tmem,
        dtype: DType::F32,
        shape: vec![128, 128],
        layout: Some(Layout::Tmem(TmemLayout {
            kind: TmemLayoutKind::Lane128,
            col_start,
            lane_align: 0,
        })),
        byte_offset: None,
    })
}

fn reg_tensor(id: u32, dtype: DType, shape: Vec<usize>) -> Arc<Tensor> {
    Arc::new(Tensor {
        id,
        space: MemorySpace::Reg,
        dtype,
        shape,
        layout: None,
        byte_offset: None,
    })
}

fn gmem_tensor(id: u32, dtype: DType, shape: Vec<usize>) -> Arc<Tensor> {
    Arc::new(Tensor {
        id,
        space: MemorySpace::Gmem,
        dtype,
        shape,
        layout: None,
        byte_offset: None,
    })
}

fn smem_tensor(id: u32, dtype: DType, shape: Vec<usize>, byte_offset: usize) -> Arc<Tensor> {
    Arc::new(Tensor {
        id,
        space: MemorySpace::Smem,
        dtype,
        shape,
        layout: None,
        byte_offset: Some(byte_offset),
    })
}

fn mbar_ref(mbar: &Arc<MBar>) -> MBarRef {
    MBarRef {
        mbar: Arc::clone(mbar),
        remote_coord: None,
    }
}

fn full_slice(tensor: Arc<Tensor>) -> TensorSlice {
    TensorSlice {
        shape: tensor
            .shape
            .iter()
            .map(|&d| ScalarValue::Int(d as i64))
            .collect(),
        offsets: tensor.shape.iter().map(|_| ScalarValue::Int(0)).collect(),
        tensor,
    }
}

fn element_slice(tensor: Arc<Tensor>, offset: ScalarValue) -> TensorSlice {
    TensorSlice {
        tensor,
        offsets: vec![offset],
        shape: vec![ScalarValue::Int(1)],
    }
}

fn cta_eq(ctaid: i64) -> ScalarValue {
    ScalarValue::expr(
        ScalarOp::Eq,
        vec![
            ScalarValue::Scope(ScopeValueKind::CtaidInCluster),
            ScalarValue::Int(ctaid),
        ],
    )
}

fn kernel_init(body: Vec<Stmt>) -> Stmt {
    Stmt::KernelInit {
        body,
        warp: Some(0),
        lane: None,
        elected: false,
    }
}

fn kernel_finalize(body: Vec<Stmt>) -> Stmt {
    Stmt::KernelFinalize {
        body,
        warp: Some(0),
        lane: None,
        elected: false,
    }
}

fn run_value_kernel(kernel: &Kernel, inputs: HashMap<u32, ValueArray1>) -> RunResult {
    interpret(
        kernel,
        inputs,
        RunOptions {
            mode: ExecutionMode::Value,
            ..Default::default()
        },
    )
}

fn run_trace_kernel(kernel: &Kernel, inputs: HashMap<u32, ValueArray1>) -> RunResult {
    interpret(
        kernel,
        inputs,
        RunOptions {
            mode: ExecutionMode::Trace,
            ..Default::default()
        },
    )
}

fn u32_array(values: &[i64]) -> ValueArray1 {
    ValueArray1::from_i64_compute(ndarray::Array1::from(values.to_vec()), DType::U32)
}

fn trace_events(result: &RunResult) -> &[TraceEvent] {
    match result.payload.as_ref().unwrap() {
        RunPayload::Trace { events, .. } => events,
        _ => panic!("expected trace payload"),
    }
}

fn trace_status(result: &RunResult) -> ProtocolStatus {
    match result.payload.as_ref().unwrap() {
        RunPayload::Trace { report, .. } => report.status,
        _ => panic!("expected trace payload"),
    }
}

fn has_mbar_complete_for_cta(result: &RunResult, ctaid_in_cluster: usize) -> bool {
    trace_events(result).iter().any(|event| {
        matches!(
            &event.payload,
            TraceEventKind::MbarCompleteTx { target, .. }
                if target.ctaid_in_cluster == ctaid_in_cluster
        )
    })
}

fn has_mbar_arrive_for_cta(result: &RunResult, ctaid_in_cluster: usize) -> bool {
    trace_events(result).iter().any(|event| {
        matches!(
            &event.payload,
            TraceEventKind::MbarArrive { target, .. }
                if target.ctaid_in_cluster == ctaid_in_cluster
        )
    })
}

#[test]
fn failed_runs_do_not_expose_partial_values() {
    let body = vec![Stmt::Role {
        body: vec![Stmt::ScalarDef {
            var: var(0, VarBinding::Scalar),
            initial: ScalarInitial::Value(ScalarValue::Int(0)),
        }],
        warp: None,
        warpgroup: None,
        elected: false,
        maxnreg: None,
    }];
    let kernel = Kernel {
        name: "t".into(),
        args: vec![],
        body,
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };

    let result = interpret(
        &kernel,
        HashMap::new(),
        RunOptions {
            max_executed_stmts: Some(0),
            ..Default::default()
        },
    );

    assert!(!result.completed);
    assert_eq!(result.failure_reason.as_deref(), Some("trace_limit"));
    assert!(matches!(result.payload, Some(RunPayload::Trace { .. })));
}

#[test]
fn tmem_cta_group2_collective_success_populates_peer_scratchpads() {
    let paired = tmem_tensor(20, 0);
    let kernel = Kernel {
        name: "tmem_cta_group2_success".into(),
        args: vec![],
        body: vec![
            kernel_init(vec![Stmt::TmemAlloc {
                tensor: paired.clone(),
                n_cols: 128,
                cta_group: 2,
            }]),
            kernel_finalize(vec![Stmt::TmemDealloc {
                tensor: paired,
                n_cols: 128,
                cta_group: 2,
            }]),
        ],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![4],
        cluster_shape: vec![2],
    };
    let result = run_value_kernel(&kernel, HashMap::new());
    assert!(kernel.validate().is_ok(), "{:?}", kernel.validate());
    assert!(result.completed, "failed: {:?}", result.failure_reason);
    assert!(matches!(result.payload, Some(RunPayload::Value { .. })));
}

#[test]
fn bad_input_metadata_fail_closed_is_rust_internal() {
    let input = gmem_tensor(303, DType::U32, vec![4]);
    let empty_kernel = Kernel {
        name: "bad_input_metadata".into(),
        args: vec![input.clone()],
        body: vec![],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let wrong_len = run_value_kernel(
        &empty_kernel,
        HashMap::from([(input.id, u32_array(&[1, 2, 3]))]),
    );
    assert!(!wrong_len.completed);
    assert_eq!(wrong_len.failure_reason.as_deref(), Some("tensor_value"));
    assert!(wrong_len.payload.is_none());

    let wrong_dtype = run_value_kernel(
        &empty_kernel,
        HashMap::from([(input.id, ValueArray1::zeros(DType::F32, 4))]),
    );
    assert!(!wrong_dtype.completed);
    assert_eq!(wrong_dtype.failure_reason.as_deref(), Some("tensor_value"));
    assert!(wrong_dtype.payload.is_none());
}

#[test]
fn tma_roundtrip_mbar_cell_parity_is_rust_internal() {
    let source = gmem_tensor(320, DType::U32, vec![4]);
    let out = gmem_tensor(321, DType::U32, vec![8]);
    let smem = smem_tensor(322, DType::U32, vec![4], 0);
    let mbar = Arc::new(MBar {
        id: 320,
        kind: MBarKind::Tma,
        stages: 1,
        arrive_count: None,
    });
    let kernel = Kernel {
        name: "tma_roundtrip".into(),
        args: vec![source.clone()],
        body: vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&mbar),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![
                    Stmt::MBarrierArriveExpectTx {
                        mbar: mbar_ref(&mbar),
                        bytes: 16,
                        stage: None,
                    },
                    Stmt::TmaLoad {
                        dst: full_slice(smem.clone()),
                        src: source.clone(),
                        mbar: mbar_ref(&mbar),
                        bytes: ScalarValue::Int(16),
                        coords: vec![ScalarValue::Int(0)],
                        shape: vec![4],
                        gmem_shape: None,
                        mbar_stage: None,
                        multicast_cta_mask: None,
                        cta_group: 1,
                    },
                    Stmt::MBarrierWait {
                        mbar: mbar_ref(&mbar),
                        stage: None,
                        phase: Some(ScalarValue::Int(0)),
                    },
                    Stmt::TmaStore {
                        dst: out,
                        src: full_slice(smem),
                        coords: vec![ScalarValue::Int(2)],
                        shape: vec![4],
                        gmem_shape: None,
                    },
                ],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 16,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let result = run_trace_kernel(&kernel, HashMap::new());
    assert!(result.completed, "failed: {:?}", result.failure_reason);
    assert_eq!(trace_status(&result), ProtocolStatus::Passed);
    assert!(has_mbar_complete_for_cta(&result, 0));
}

#[test]
fn tma_multicast_group2_mbar_targets_are_deduplicated() {
    let source = gmem_tensor(330, DType::U32, vec![4]);
    let smem = smem_tensor(332, DType::U32, vec![4], 0);
    let mbar = Arc::new(MBar {
        id: 330,
        kind: MBarKind::Tma,
        stages: 1,
        arrive_count: None,
    });
    let even_mbar = MBarRef {
        mbar: mbar.clone(),
        remote_coord: Some(ScalarValue::Int(0)),
    };
    let kernel = Kernel {
        name: "tma_multicast_cta_group2".into(),
        args: vec![source.clone()],
        body: vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&mbar),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![
                    Stmt::If {
                        cond: cta_eq(0),
                        then_body: vec![Stmt::MBarrierArriveExpectTx {
                            mbar: mbar_ref(&mbar),
                            bytes: 16,
                            stage: None,
                        }],
                    },
                    Stmt::If {
                        cond: cta_eq(0),
                        then_body: vec![Stmt::TmaLoad {
                            dst: full_slice(smem),
                            src: source.clone(),
                            mbar: even_mbar,
                            bytes: ScalarValue::Int(16),
                            coords: vec![ScalarValue::Int(0)],
                            shape: vec![4],
                            gmem_shape: None,
                            mbar_stage: None,
                            multicast_cta_mask: Some(0b11),
                            cta_group: 2,
                        }],
                    },
                ],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 16,
        launch_shape: vec![2],
        cluster_shape: vec![2],
    };
    let result = run_trace_kernel(&kernel, HashMap::new());
    assert!(result.completed, "failed: {:?}", result.failure_reason);
    assert_eq!(trace_status(&result), ProtocolStatus::Passed);
    assert!(has_mbar_complete_for_cta(&result, 0));
    assert!(!has_mbar_complete_for_cta(&result, 1));
}

#[test]
fn mbarrier_wait_success_and_blocked_frontier_are_rust_internal() {
    let source = gmem_tensor(340, DType::U32, vec![4]);
    let smem = smem_tensor(341, DType::U32, vec![4], 0);
    let mbar = Arc::new(MBar {
        id: 340,
        kind: MBarKind::Tma,
        stages: 1,
        arrive_count: None,
    });
    let kernel = Kernel {
        name: "mbar_wait_wake".into(),
        args: vec![source.clone()],
        body: vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&mbar),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![
                    Stmt::MBarrierArriveExpectTx {
                        mbar: mbar_ref(&mbar),
                        bytes: 16,
                        stage: None,
                    },
                    Stmt::MBarrierWait {
                        mbar: mbar_ref(&mbar),
                        stage: None,
                        phase: Some(ScalarValue::Int(0)),
                    },
                ],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
            Stmt::Role {
                body: vec![Stmt::TmaLoad {
                    dst: full_slice(smem),
                    src: source.clone(),
                    mbar: mbar_ref(&mbar),
                    bytes: ScalarValue::Int(16),
                    coords: vec![ScalarValue::Int(0)],
                    shape: vec![4],
                    gmem_shape: None,
                    mbar_stage: None,
                    multicast_cta_mask: None,
                    cta_group: 1,
                }],
                warp: Some(1),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 16,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let result = run_trace_kernel(&kernel, HashMap::new());
    assert!(result.completed, "failed: {:?}", result.failure_reason);
    assert_eq!(trace_status(&result), ProtocolStatus::Passed);
    assert!(has_mbar_complete_for_cta(&result, 0));

    let expect_tx = Arc::new(MBar {
        id: 344,
        kind: MBarKind::Tma,
        stages: 1,
        arrive_count: None,
    });
    let expect_tx_kernel = Kernel {
        name: "mbarrier_expect_tx_deadlock".into(),
        args: vec![],
        body: vec![
            Stmt::MBarDef {
                mbar: expect_tx.clone(),
            },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&expect_tx),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![
                    Stmt::MBarrierExpectTx {
                        mbar: mbar_ref(&expect_tx),
                        bytes: 8,
                        stage: None,
                    },
                    Stmt::MBarrierArrive {
                        mbar: mbar_ref(&expect_tx),
                        stage: None,
                        count: ScalarValue::Int(1),
                    },
                    Stmt::MBarrierWait {
                        mbar: mbar_ref(&expect_tx),
                        stage: None,
                        phase: None,
                    },
                ],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let expect_tx_result = run_trace_kernel(&expect_tx_kernel, HashMap::new());
    assert!(!expect_tx_result.completed);
    assert_eq!(expect_tx_result.failure_reason.as_deref(), Some("deadlock"));
    assert!(!expect_tx_result.blocked_frontier.is_empty());
    assert_eq!(trace_status(&expect_tx_result), ProtocolStatus::Failed);

    let remote_success = Arc::new(MBar {
        id: 343,
        kind: MBarKind::Tma,
        stages: 1,
        arrive_count: None,
    });
    let remote_success_kernel = Kernel {
        name: "mbar_remote_arrive_success".into(),
        args: vec![],
        body: vec![
            Stmt::MBarDef {
                mbar: remote_success.clone(),
            },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&remote_success),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![
                    Stmt::If {
                        cond: cta_eq(0),
                        then_body: vec![Stmt::MBarrierArrive {
                            mbar: MBarRef {
                                mbar: remote_success.clone(),
                                remote_coord: Some(ScalarValue::Int(1)),
                            },
                            stage: None,
                            count: ScalarValue::Int(1),
                        }],
                    },
                    Stmt::If {
                        cond: cta_eq(1),
                        then_body: vec![Stmt::MBarrierWait {
                            mbar: mbar_ref(&remote_success),
                            stage: None,
                            phase: Some(ScalarValue::Int(0)),
                        }],
                    },
                ],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![2],
        cluster_shape: vec![2],
    };
    let remote_success_result = run_trace_kernel(&remote_success_kernel, HashMap::new());
    assert!(
        remote_success_result.completed,
        "failed: {:?}",
        remote_success_result.failure_reason
    );
    assert_eq!(trace_status(&remote_success_result), ProtocolStatus::Passed);
    assert!(has_mbar_arrive_for_cta(&remote_success_result, 1));
}

#[test]
fn scalar_tensor_initial_loads_per_thread_gmem() {
    let source = gmem_tensor(345, DType::U32, vec![32]);
    let scalar = Var {
        id: VarId(345),
        binding: VarBinding::Scalar,
        dtype: ScalarDType::U32,
    };
    let kernel = Kernel {
        name: "scalar_tensor_initial".into(),
        args: vec![source.clone()],
        body: vec![Stmt::Role {
            body: vec![Stmt::ScalarDef {
                var: scalar,
                initial: ScalarInitial::Tensor(element_slice(
                    source.clone(),
                    ScalarValue::Scope(ScopeValueKind::LaneId),
                )),
            }],
            warp: Some(0),
            warpgroup: None,
            elected: false,
            maxnreg: None,
        }],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let values: Vec<i64> = (0..32).map(|x| x * 3).collect();
    let result = run_value_kernel(&kernel, HashMap::from([(source.id, u32_array(&values))]));
    assert!(result.completed, "failed: {:?}", result.failure_reason);
    assert!(matches!(result.payload, Some(RunPayload::Value { .. })));
}

#[test]
fn cluster_sync_cleanup_is_rust_internal() {
    let kernel = Kernel {
        name: "cluster_sync_repeated_success".into(),
        args: vec![],
        body: vec![Stmt::ClusterSync, Stmt::ClusterSync],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![2],
        cluster_shape: vec![2],
    };
    let result = run_trace_kernel(&kernel, HashMap::new());
    assert!(result.completed, "failed: {:?}", result.failure_reason);
    assert_eq!(trace_status(&result), ProtocolStatus::Passed);
    assert_eq!(
        trace_events(&result)
            .iter()
            .filter(|event| matches!(&event.payload, TraceEventKind::Sync { .. }))
            .count(),
        4
    );
}

#[test]
fn fence_cp_async_trace_limit_fail_closed() {
    let source = gmem_tensor(348, DType::U32, vec![4]);
    let out = gmem_tensor(349, DType::U32, vec![4]);
    let reg = reg_tensor(350, DType::U32, vec![4]);
    let kernel = Kernel {
        name: "fence_cp_async_noop".into(),
        args: vec![source.clone(), out.clone()],
        body: vec![Stmt::Role {
            body: vec![
                Stmt::CpAsyncBulkCommitGroup,
                Stmt::CpAsyncBulkCommitGroup,
                Stmt::CpAsyncBulkWaitGroupRead { n: 0 },
                Stmt::Fence {
                    kind: FenceKind::Memory,
                    scope: FenceScope::Cta,
                },
                Stmt::RegLoad {
                    dst: full_slice(reg.clone()),
                    src: full_slice(source.clone()),
                },
                Stmt::RegStore {
                    dst: full_slice(out),
                    src: full_slice(reg),
                },
            ],
            warp: None,
            warpgroup: Some(0),
            elected: true,
            maxnreg: None,
        }],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let trace_limited = interpret(
        &kernel,
        HashMap::from([(source.id, u32_array(&[1, 2, 3, 4]))]),
        RunOptions {
            mode: ExecutionMode::Value,
            max_executed_stmts: Some(0),
            ..Default::default()
        },
    );
    assert!(!trace_limited.completed);
    assert_eq!(trace_limited.failure_reason.as_deref(), Some("trace_limit"));
    assert!(trace_limited.payload.is_none());
}

#[test]
fn tma_and_reg_runtime_failures_expose_no_partial_values() {
    let out = gmem_tensor(354, DType::U32, vec![4]);
    let smem = smem_tensor(355, DType::U32, vec![4], 0);
    let missing_source = Kernel {
        name: "tma_store_missing_source".into(),
        args: vec![],
        body: vec![Stmt::Role {
            body: vec![Stmt::TmaStore {
                dst: out.clone(),
                src: full_slice(smem.clone()),
                coords: vec![ScalarValue::Int(0)],
                shape: vec![4],
                gmem_shape: None,
            }],
            warp: Some(0),
            warpgroup: None,
            elected: true,
            maxnreg: None,
        }],
        num_warps: 4,
        smem_size_bytes: 16,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let missing_source_result = run_value_kernel(&missing_source, HashMap::new());
    assert!(!missing_source_result.completed);
    assert_eq!(
        missing_source_result.failure_reason.as_deref(),
        Some("missing_tensor_value")
    );
    assert!(missing_source_result.payload.is_none());

    let divergent = Kernel {
        name: "tma_store_divergent_coords".into(),
        args: vec![],
        body: vec![Stmt::Role {
            body: vec![Stmt::TmaStore {
                dst: out.clone(),
                src: full_slice(smem.clone()),
                coords: vec![ScalarValue::Scope(ScopeValueKind::LaneId)],
                shape: vec![4],
                gmem_shape: None,
            }],
            warp: Some(0),
            warpgroup: None,
            elected: false,
            maxnreg: None,
        }],
        num_warps: 4,
        smem_size_bytes: 16,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let divergent_result = run_value_kernel(&divergent, HashMap::new());
    assert!(!divergent_result.completed);
    assert_eq!(
        divergent_result.failure_reason.as_deref(),
        Some("divergent_tma_operands")
    );
    assert!(divergent_result.payload.is_none());

    let short = gmem_tensor(356, DType::U32, vec![8]);
    let reg = reg_tensor(357, DType::U32, vec![1]);
    let reg_oob = Kernel {
        name: "reg_load_oob_source".into(),
        args: vec![short.clone()],
        body: vec![Stmt::Role {
            body: vec![Stmt::RegLoad {
                dst: element_slice(reg.clone(), ScalarValue::Int(0)),
                src: element_slice(short.clone(), ScalarValue::Scope(ScopeValueKind::LaneId)),
            }],
            warp: Some(0),
            warpgroup: None,
            elected: false,
            maxnreg: None,
        }],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let reg_oob_result =
        run_value_kernel(&reg_oob, HashMap::from([(short.id, u32_array(&[0; 8]))]));
    assert!(!reg_oob_result.completed);
    assert_eq!(
        reg_oob_result.failure_reason.as_deref(),
        Some("tensor_value")
    );
    assert!(reg_oob_result.payload.is_none());

    let reg_missing_store = Kernel {
        name: "reg_store_missing_source".into(),
        args: vec![out.clone()],
        body: vec![Stmt::Role {
            body: vec![Stmt::RegStore {
                dst: element_slice(out, ScalarValue::Int(0)),
                src: element_slice(reg, ScalarValue::Int(0)),
            }],
            warp: Some(0),
            warpgroup: None,
            elected: true,
            maxnreg: None,
        }],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![1],
        cluster_shape: vec![1],
    };
    let reg_missing_store_result = run_value_kernel(&reg_missing_store, HashMap::new());
    assert!(!reg_missing_store_result.completed);
    assert_eq!(
        reg_missing_store_result.failure_reason.as_deref(),
        Some("missing_tensor_value")
    );
    assert!(reg_missing_store_result.payload.is_none());
}

#[test]
fn tcgen05_commit_success_paths_update_rust_internal_mbar_cells() {
    let mbar = Arc::new(MBar {
        id: 350,
        kind: MBarKind::Tcgen05,
        stages: 1,
        arrive_count: None,
    });
    let group2_kernel = Kernel {
        name: "tcgen05_commit_group2".into(),
        args: vec![],
        body: vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&mbar),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![Stmt::If {
                    cond: cta_eq(0),
                    then_body: vec![Stmt::Tcgen05Commit {
                        mbar: mbar_ref(&mbar),
                        stage: None,
                        cta_group: 2,
                        multicast_cta_mask: None,
                    }],
                }],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![2],
        cluster_shape: vec![2],
    };
    let group2 = run_trace_kernel(&group2_kernel, HashMap::new());
    assert!(group2.completed, "failed: {:?}", group2.failure_reason);
    assert_eq!(trace_status(&group2), ProtocolStatus::Passed);
    assert!(has_mbar_arrive_for_cta(&group2, 0));
    assert!(!has_mbar_arrive_for_cta(&group2, 1));

    let multicast = Arc::new(MBar {
        id: 351,
        kind: MBarKind::Tcgen05,
        stages: 1,
        arrive_count: None,
    });
    let multicast_kernel = Kernel {
        name: "tcgen05_commit_multicast_direct_mask".into(),
        args: vec![],
        body: vec![
            Stmt::MBarDef {
                mbar: multicast.clone(),
            },
            kernel_init(vec![Stmt::MBarrierInit {
                mbar: mbar_ref(&multicast),
                count: 1,
                stage: None,
            }]),
            Stmt::Role {
                body: vec![Stmt::If {
                    cond: cta_eq(0),
                    then_body: vec![Stmt::Tcgen05Commit {
                        mbar: MBarRef {
                            mbar: multicast.clone(),
                            remote_coord: Some(ScalarValue::Int(0)),
                        },
                        stage: None,
                        cta_group: 2,
                        multicast_cta_mask: Some(0b10),
                    }],
                }],
                warp: Some(0),
                warpgroup: None,
                elected: true,
                maxnreg: None,
            },
        ],
        num_warps: 4,
        smem_size_bytes: 0,
        launch_shape: vec![2],
        cluster_shape: vec![2],
    };
    let multicast_result = run_trace_kernel(&multicast_kernel, HashMap::new());
    assert!(
        multicast_result.completed,
        "failed: {:?}",
        multicast_result.failure_reason
    );
    assert_eq!(trace_status(&multicast_result), ProtocolStatus::Passed);
    assert!(!has_mbar_arrive_for_cta(&multicast_result, 0));
    assert!(has_mbar_arrive_for_cta(&multicast_result, 1));
}
