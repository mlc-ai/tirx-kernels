from helpers import builder, expect_runtime_error, run, tmem_tensor


def test_tmem_lifecycle_failures_are_fail_closed():
    cases = []

    b = builder("tmem_duplicate")
    duplicate = tmem_tensor(b, col_start=0)
    with b.kernel_init(warp=0):
        b.tmem_alloc(duplicate, n_cols=128)
        b.tmem_alloc(duplicate, n_cols=128)
    cases.append((b, "tmem_already_allocated"))

    b = builder("tmem_missing")
    missing = tmem_tensor(b, col_start=0)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(missing, n_cols=128)
    cases.append((b, "missing_tmem_allocation"))

    b = builder("tmem_overlap")
    overlap_a = tmem_tensor(b, col_start=0)
    overlap_b = tmem_tensor(b, col_start=32)
    with b.kernel_init(warp=0):
        b.tmem_alloc(overlap_a, n_cols=64)
        b.tmem_alloc(overlap_b, n_cols=64)
    cases.append((b, "tmem_allocation_overlap"))

    b = builder("tmem_order")
    small = tmem_tensor(b, col_start=128)
    large = tmem_tensor(b, col_start=0)
    with b.kernel_init(warp=0):
        b.tmem_alloc(small, n_cols=64)
        b.tmem_alloc(large, n_cols=128)
    cases.append((b, "tmem_allocation_order"))

    b = builder("tmem_mismatch")
    alloc = tmem_tensor(b, col_start=0)
    mismatch = tmem_tensor(b, col_start=32)
    with b.kernel_init(warp=0):
        b.tmem_alloc(alloc, n_cols=64)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(mismatch, n_cols=64)
    cases.append((b, "tmem_allocation_mismatch"))

    for case, code in cases:
        with expect_runtime_error(code):
            run(case.build())


def test_tmem_cta_group2_missing_peer_fails_closed():
    b = builder("tmem_cta_group2_missing_peer")
    missing = tmem_tensor(b, col_start=0)
    with b.kernel_init(warp=0):
        b.tmem_alloc(missing, n_cols=128, cta_group=2)

    with expect_runtime_error("tmem_collective_peer"):
        run(b.build())
