from helpers import builder, cta_eq, expect_runtime_error, run


def test_sync_partial_arrival_and_cluster_peer_exit_fail_closed():
    b = builder("partial_warp_sync")
    with b.role(warp=0):
        with b.if_(b.lane_id() < 16):
            b.warp_sync()

    with expect_runtime_error("deadlock"):
        run(b.build())

    b = builder("cluster_sync_peer_exited", launch_shape=(2,), cluster_shape=(2,))
    with b.kernel_finalize():
        with b.if_(cta_eq(b, 0)):
            b.fence()
    with b.kernel_finalize():
        with b.if_(cta_eq(b, 1)):
            b.cluster_sync()

    with expect_runtime_error("cluster_sync_peer_exited"):
        run(b.build())


def test_cluster_sync_success_reuse_after_cleanup_completes():
    b = builder("cluster_sync_repeated_success", launch_shape=(2,), cluster_shape=(2,))
    b.cluster_sync()
    b.cluster_sync()

    run(b.build())
