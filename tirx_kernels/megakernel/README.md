# MegaKernelMOE Task Reference

This document describes the current `MegaKernelMOE` as a set of operators.
Read each task the same way you would read a PyTorch operator reference:

```python
output = op(input0, input1, ...)
```

For each task, the signature says what tensors it consumes and returns.  The
`Scheduler edge` section says which event makes the consumer safe to run, and
the `Tile instances` section says how that logical op is split into
`(m_idx, n_idx, k_idx)` work items.

The current code does not have a standalone `TaskSpec` class.  The operator
contract is implemented by:

- `MegaKernelMOE.task_impl_*`: passes tensors into the tile and issues
  `wait` / `notify` / `pre_notify_and_push`;
- `MegaKernelMOE._set_events`: allocates dependency counters;
- `generate_exec_queue_moe`: creates the static queue and the dynamic initial
  queue;
- `Tile.run(m_idx, n_idx, k_idx, ...)`: implements one task instance.

## Notation

The benchmark shape is Qwen3-30B-A3B MoE:

| Symbol | Meaning | Value |
| --- | --- | --- |
| `B` | input tokens / batch size | runtime config |
| `H` | hidden size | 2048 |
| `I` | intermediate size | 768 |
| `E` | number of experts | 128 |
| `K` | top-k experts per token | 8 |
| `P` | expert-major block size, `MOE_M_PAD_SIZE` | 128 |
| `S` | router split-k factor | 4 |
| `BN` | grouped GEMM output tile columns, `GroupGEMMTileSM100.BLK_N` | 128 |
| `T` | routed entries before padding | `B * K` |
| `TP` | routed entries after expert padding | runtime `num_tokens_post_pad[0]` |
| `TP_MAX` | maximum padded routed entries | `get_max_num_tokens_padded(B, K, E, P)` |

Shapes in this document use logical valid ranges.  Allocated workspaces usually
use `TP_MAX`, while runtime work only covers `TP`.

## SGLang Source Baseline

The optional SGLang fused-MoE baseline is imported lazily from a source
checkout.  It does not require installing the `sglang` package:

```bash
export WORKSPACE=/path/to/workspace
export SGLANG_PATH=$HOME/kernel-libs/sglang
export PYTHONPATH="${SGLANG_PATH}/python:${WORKSPACE}/tirx-kernels:${WORKSPACE}/tvm/python"
export TVM_LIBRARY_PATH="${WORKSPACE}/tvm/build/lib"

python -m tirx_kernels.bench \
  --kernel megakernel_moe \
  --config moe_a3b_bs1_all
```

SGLang module import, runtime-context setup, and Triton JIT compilation happen
outside the timed region.  The timed `sglang_full` callable includes the same
FP32 router matmul, softmax, top-k, and expert computation as the full TIRx MoE
scope.  Both implementations receive the same logical input and weights.  The
result metadata includes one validation record per scheduler; treat a missing or
failed validation as a benchmark failure. The grouped benchmark compiles static,
dynamic, and unfused once and measures them against the same logical input and
the same SGLang and FlashInfer references.

The module includes B200 configs under `sglang_moe_configs/` for SGLang commit
`96a04cb13f9c3ed86028e090784a9eb059cf5318` and Triton 3.6.0.  They were generated
with SGLang's separated fused-MoE tuner over its complete 1920-config search
space for `B = [1, 8, 32, 128, 512, 1024, 2048, 4096]`.  Tuning used 100
fixed-seed, uniformly routed top-k tensors with distinct expert IDs per token.
The two config files independently select gate/up and down-projection settings,
including the down TMA decision.  Set `SGLANG_MOE_CONFIG_DIR` to override the
packaged configs with a retuned directory for another GPU or Triton version.

## Forward Pipeline

The logical computation is:

```python
gating_output = moe_gating(
    hidden_state,          # f16[B, H]
    gate_weight,           # f16[E, H]
)

topk_weights, topk_indices = moe_topk_softmax(
    gating_output,         # f32[B, E]
    top_k=K,
    renormalize=False,
)

routing = moe_align(
    topk_indices,          # i32[B, K]
    num_experts=E,
    block_size=P,
)

sorted_token_ids, reordered_hidden_state = moe_count_and_sort(
    hidden_state,          # f16[B, H]
    topk_indices,          # i32[B, K]
    routing,
)

silu_mul_output = moe_group_gemm_gate_up_silu(
    reordered_hidden_state,  # f16[TP, H]
    grp_gate_up_weight,      # f16[E, 2 * I, H]
    routing,
)

topk_reduce_output = moe_group_gemm_down(
    silu_mul_output,       # f16[TP, I]
    grp_down_weight,       # f16[E, H, I]
    topk_weights,          # f32[B, K]
    routing,
)
```

`routing` is not a Python object in the kernel.  It names the set of routing
metadata produced by `moe_align` and consumed by later tasks:

```python
routing = {
    "num_tokens_post_pad": i32[1],
    "cumsum_buffer": i32[E + 1],
    "expert_ids": i32[TP_MAX / P],
    "num_valid_tokens": i32[TP_MAX / P],
    "sorted_token_ids": i32[TP_MAX],
}
```

`sorted_token_ids[row]` stores a flattened route id:

```python
route_id = token_id * K + route_slot
token_id = route_id // K
route_slot = route_id % K
expert_id = topk_indices[token_id, route_slot]
route_weight = topk_weights[token_id, route_slot]
```

Padding rows use sentinel `T`.

## How Tasks Connect

Two task specs connect when the producer's return value is a named input in the
consumer's signature, and the consumer waits on the event notified by the
producer.

Example:

```python
gating_output = moe_gating(hidden_state, gate_weight)
topk_weights, topk_indices = moe_topk_softmax(gating_output, top_k=K)
```

The connection is valid because:

- `moe_gating` returns `gating_output`;
- `moe_topk_softmax` takes `gating_output`;
- gating has `S * ceildiv(B, 128)` tile instances;
- `evt_gating` is initialized to exactly that producer count;
- every gating instance notifies `evt_gating`;
- every top-k/softmax instance waits on `evt_gating` before reading
  `gating_output`;
- the dynamic scheduler pushes top-k/softmax instances when `evt_gating`
  reaches its trigger.

That is the contract every producer/consumer pair must satisfy.

## Task Reference

### `moe_gating`

Signature:

```python
moe_gating(
    hidden_state: f16[B, H],
    gate_weight: f16[E, H],
) -> gating_output: f32[B, E]
```

Semantics:

```python
gating_output[token, expert] = dot(hidden_state[token, :], gate_weight[expert, :])
```

Tile instances:

```text
JobType.MOE_GATING
m_idx in [0, ceildiv(B, 128))   # token block
n_idx = 0
k_idx in [0, S)                 # split-k shard over H
```

Scheduler edge:

- Waits on: nothing.
- Notifies: `evt_gating`.
- Event count: `S * ceildiv(B, 128)`.
- Dynamic successor: pushes all `moe_topk_softmax` persistent shards.

Implementation:

- Tile: `GemmTile`.
- Wrapper: `MegaKernelMOE.task_impl_moe_gating`.
- Static queue entries: `push_moe_tasks`.

### `moe_topk_softmax`

Signature:

```python
moe_topk_softmax(
    gating_output: f32[B, E],
    *,
    top_k: int = K,
    renormalize: bool = False,
) -> tuple[
    topk_weights: f32[B, K],
    topk_indices: i32[B, K],
]
```

Semantics:

```python
routing = softmax(gating_output[token, :])
topk_weights[token, :], topk_indices[token, :] = topk(routing, K)
```

The current call uses `renormalize=False`, so `topk_weights` are probabilities
from the full expert softmax, not probabilities re-normalized over the selected
top-k experts.

Tile instances:

```text
JobType.MOE_TOPK_SOFTMAX
m_idx in [0, KernelConfig.SM_NUMBER)   # persistent shard id
n_idx = 0
k_idx = 0
```

Scheduler edge:

- Waits on: `evt_gating`.
- Notifies: `evt_topk_softmax`.
- Event count: `KernelConfig.SM_NUMBER`.
- Dynamic successor: pushes one `moe_align` task.

Implementation:

- Tile: `TopkSoftmaxTile`.
- Wrapper: `MegaKernelMOE.task_impl_moe_topk_softmax`.

### `moe_align`

Signature:

```python
moe_align(
    topk_indices: i32[B, K],
    *,
    num_experts: int = E,
    block_size: int = P,
) -> routing
```

Returns:

```python
routing = {
    "num_tokens_post_pad": i32[1],        # TP
    "cumsum_buffer": i32[E + 1],          # expert prefix offsets
    "expert_ids": i32[TP_MAX / P],        # expert id per padded block
    "num_valid_tokens": i32[TP_MAX / P],  # valid rows per padded block
    "sorted_token_ids": i32[TP_MAX],      # initialized to sentinel T for padding
}
```

Semantics:

1. Count routes per expert from `topk_indices.view(-1)`.
2. Round every expert count up to a multiple of `P`.
3. Prefix-sum padded counts into `cumsum_buffer`.
4. Write `num_tokens_post_pad[0] = TP`.
5. For each padded expert-major block, write `expert_ids[block]`.
6. Optionally write `num_valid_tokens[block]`.
7. Initialize padded `sorted_token_ids` rows with sentinel `T`.

Tile instances:

```text
JobType.MOE_ALIGN
m_idx = 0
n_idx = 0
k_idx = 0
```

Scheduler edge:

- Waits on: `evt_topk_softmax`.
- Notifies: `evt_moe_align`.
- Event count: `1`.
- Dynamic successor: pushes all `moe_count_and_sort` persistent shards.

Dynamic scheduler side effect:

- Initializes the dynamic `evt_group_gemm_down` count after runtime `TP` is
  known.

Implementation:

- Tile: `MOEAlignTile`.
- Wrapper: `MegaKernelMOE.task_impl_moe_align`.

### `moe_count_and_sort`

Signature:

```python
moe_count_and_sort(
    hidden_state: f16[B, H],
    topk_indices: i32[B, K],
    routing,
) -> tuple[
    sorted_token_ids: i32[TP_MAX],
    reordered_hidden_state: f16[TP_MAX, H],
]
```

Semantics:

For each flattened route id:

```python
route_id = token_id * K + route_slot
expert_id = topk_indices[token_id, route_slot]
row = atomic_add(routing.cumsum_buffer[expert_id], 1)
sorted_token_ids[row] = route_id
reordered_hidden_state[row, :] = hidden_state[token_id, :]
```

After this task, valid rows of `reordered_hidden_state` are expert-major:
all routes for the same expert occupy contiguous padded blocks.  Padding rows
remain sentinel-filled in `sorted_token_ids`.

`cumsum_buffer` is mutated while assigning rows.  Treat it as scratch after this
task, not as an immutable prefix table.

Tile instances:

```text
JobType.MOE_COUNT_AND_SORT
m_idx in [0, KernelConfig.SM_NUMBER)   # persistent shard id
n_idx = 0
k_idx = 0
```

Scheduler edge:

- Waits on: `evt_moe_align`.
- Notifies: `evt_count_and_sort`.
- Event count: `KernelConfig.SM_NUMBER`.
- Dynamic successor: pushes runtime `moe_group_gemm_gate_up_silu` tiles:

```text
m_idx in [0, TP / P)
n_idx in [0, (2 * I) / BN)
k_idx = 0
```

Implementation:

- Tile: `CountAndSortExpertTokens`.
- Wrapper: `MegaKernelMOE.task_impl_moe_count_and_sort`.

### `moe_group_gemm_gate_up_silu`

Signature:

```python
moe_group_gemm_gate_up_silu(
    reordered_hidden_state: f16[TP_MAX, H],
    grp_gate_up_weight: f16[E, 2 * I, H],
    routing,
) -> silu_mul_output: f16[TP_MAX, I]
```

Semantics:

For each valid expert-major row:

```python
expert = routing.expert_ids[row // P]
gate_up = grp_gate_up_weight[expert] @ reordered_hidden_state[row, :]
gate = gate_up[:I]
up = gate_up[I:]
silu_mul_output[row, :] = silu(gate) * up
```

The implementation computes this as grouped GEMM tiles.  `n_idx` covers a
128-column tile of the `2 * I` gate/up projection; each such tile produces the
corresponding `BN / 2` columns of `silu_mul_output`.

Tile instances:

```text
JobType.MOE_GROUP_GEMM_GATE_UP_SILU
m_idx in [0, TP / P)          # expert-major block
n_idx in [0, (2 * I) / BN)    # gate/up output column tile
k_idx = 0
```

Static scheduling enqueues the max task space for `TP_MAX / P`; runtime-invalid
blocks are skipped by checking `m_idx < num_tokens_post_pad[0] / P`.

Scheduler edge:

- Waits on: `evt_count_and_sort`.
- Notifies: `evt_group_gemm_gate_up[m_idx]` in fused static/dynamic mode.
- Event count per expert-major block: `(2 * I) / BN`.
- Dynamic successor: pushes down-projection tiles for the same `m_idx`.

Implementation note:

- Tile: `GroupGEMMSiluTile`.
- Wrapper: `MegaKernelMOE.task_impl_moe_group_gemm_gate_up_silu`.
- The shared grouped-GEMM call signature passes `topk_weights`, but this stage
  does not apply routing weights.  Routing weights are applied in
  `moe_group_gemm_down`.

### `moe_group_gemm_down`

Signature:

```python
moe_group_gemm_down(
    silu_mul_output: f16[TP_MAX, I],
    grp_down_weight: f16[E, H, I],
    topk_weights: f32[B, K],
    routing,
) -> topk_reduce_output: f16[B, H]
```

Semantics:

For each valid expert-major row:

```python
route_id = routing.sorted_token_ids[row]
token_id = route_id // K
route_slot = route_id % K
expert = routing.expert_ids[row // P]
partial = grp_down_weight[expert] @ silu_mul_output[row, :]
topk_reduce_output[token_id, :] += topk_weights[token_id, route_slot] * partial
```

The output is a reduce-add across the `K` selected experts for every token.

Tile instances:

```text
JobType.MOE_GROUP_GEMM_DOWN
m_idx in [0, TP / P)                         # expert-major block
n_idx in [0, H / BN / down_proj_task_size)   # hidden output tile group
k_idx = 0
```

Inside one scheduled down task, the wrapper may run `down_proj_task_size`
neighboring hidden-column tiles to amortize dynamic scheduling overhead.

Static scheduling enqueues the max task space for `TP_MAX / P`; runtime-invalid
blocks are skipped by checking `m_idx < num_tokens_post_pad[0] / P`.

Scheduler edge:

- Waits on: `evt_group_gemm_gate_up[m_idx]` in fused static/dynamic mode.
- In `unfused` comparison mode, waits on global `evt_group_gemm_gate_up[0]`.
- Dynamic mode notifies `evt_group_gemm_down`.
- Dynamic successor: pushes `END` tasks after all runtime down tiles have been
  dispatched.

Implementation:

- Tile: `GroupGEMMTileSM100`.
- Wrapper: `MegaKernelMOE.task_impl_moe_group_gemm_down`.

## Scheduler Summary

Static scheduling builds the whole work queue before launch:

```text
INIT_ETENSOR*
MOE_GATING*
WAIT_ETENSOR_INIT*
MOE_TOPK_SOFTMAX*
MOE_ALIGN
MOE_COUNT_AND_SORT*
MOE_GROUP_GEMM_GATE_UP_SILU*  # max TP_MAX task space
MOE_GROUP_GEMM_DOWN*          # max TP_MAX task space
END*
```

The queue order is not a cross-SM barrier.  Correctness still comes from the
events in each task wrapper.

Dynamic scheduling starts with only:

```text
INIT_ETENSOR*
MOE_GATING*
```

Then each event trigger pushes the next operator's task instances:

| Trigger | Pushed work |
| --- | --- |
| `evt_gating` | all `moe_topk_softmax` shards |
| `evt_topk_softmax` | one `moe_align` task |
| `evt_moe_align` | all `moe_count_and_sort` shards |
| `evt_count_and_sort` | runtime `moe_group_gemm_gate_up_silu` tiles using `TP` |
| `evt_group_gemm_gate_up[m_idx]` | runtime `moe_group_gemm_down` tiles for that `m_idx` |
| `evt_group_gemm_down` | `END` tasks |

Dynamic notify is two-phase:

1. `pre_notify_and_push` runs before the tile body so successors can be queued
   as soon as the last producer has been dispatched.
2. The normal `notify` runs after the tile body, and consumers still wait for
   that ready state before reading producer outputs.

The push can happen early; the read cannot.

## Adding a Task

Do not start by writing scheduler code.  First write the operator spec:

```python
new_output = new_task(named_input0, named_input1, ...)
```

Then make it executable:

1. Define the output tensor names and shapes.
2. Define the input tensor names and shapes.
3. Define the tile instance space in `(m_idx, n_idx, k_idx)`.
4. Add a `JobType`.
5. Implement `Tile.run(m_idx, n_idx, k_idx, ...)`.
6. Add a `task_impl_*` wrapper that passes the named inputs and outputs.
7. For each producer/consumer edge, allocate an event with the producer task
   count.
8. Make the producer notify that event after writing its output.
9. Make the consumer wait on that event before reading the output.
10. Add static queue entries for the consumer task space.
11. Add a dynamic `pre_notify_and_push` rule for the same consumer task space.
12. Add or update `run_test` coverage so the final output is compared with a
    trusted reference.

If the signature of two neighboring tasks does not line up, or the event count
does not equal the producer task count, the graph is not well specified.
