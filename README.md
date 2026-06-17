# tirx-kernels

High-performance GPU kernels written in [TIRx](https://github.com/mlc-ai/tir).

## Kernels

| Name                 | dtype             | Arch    |
| -------------------- | ----------------- | ------- |
| `fp16_bf16_gemm`     | fp16 / bf16       | sm_100a |
| `fp8_blockwise_gemm` | fp8 (blockwise)   | sm_100a |
| `nvfp4_gemm`         | nvfp4             | sm_100a |
| `flash_attention4`   | bf16              | sm_100a |

## Installation

```bash
pip install tirx-kernels          # from a release
# or, from a checkout:
pip install -e .
```

### External dependencies

These are **not on PyPI** and must be installed/available separately. They are
imported lazily, so `import tirx_kernels` and kernel discovery work without
them — they are only needed to actually compile/run a kernel:

| Dependency       | Needed by                          | Notes                                                  |
| ---------------- | ---------------------------------- | ------------------------------------------------------ |
| `tvm.tirx`       | all kernels (compile + run)        | The TIRx compiler. Put it on `PYTHONPATH`, e.g. `/path/to/tir/python`. |
| `torch`          | all kernels                        | CUDA build matching your GPU.                          |
| `deep_gemm`      | `fp8_blockwise_gemm` data/baseline | Used for fp8 quantization helpers and the reference.   |
| `flashinfer`     | `nvfp4_gemm` data/baseline         | Used for nvfp4 quantization and the baseline.          |

## Usage

### Command line

```bash
# List discovered kernels (with their config labels)
python -m tirx_kernels.registry --format json

# Run correctness tests (optionally filter by kernel / config label)
python -m tirx_kernels.test
python -m tirx_kernels.test --kernel fp16_bf16_gemm
python -m tirx_kernels.test --kernel fp16_bf16_gemm --config bf16_1024x1024x1024

# Benchmark
python -m tirx_kernels.bench --kernel nvfp4_gemm
```

### Programmatic API

Every kernel module exposes a small, uniform interface (see
`tirx_kernels/_protocol.py`):

```python
from tirx_kernels.registry import discover_kernels

kernels = discover_kernels()          # {name: module}
mod = kernels["fp16_bf16_gemm"]

mod.run_test(M=1024, N=1024, K=1024)  # compile + run + correctness check
mod.run_bench(M=1024, N=1024, K=1024) # profile (needs a GPU)

func = mod.get_kernel(M=1024, N=1024, K=1024)  # the TIRx PrimFunc
```

Each module also provides `KERNEL_META` (name / category / `compute_capability`)
and `CONFIGS` (the test/bench parameter sweeps) that the registry and CLI use.

## License

Apache License 2.0. See [LICENSE](LICENSE).
