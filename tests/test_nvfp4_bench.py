import json

import pytest

from tirx_kernels.gemm.nvfp4_gemm import _flashinfer_tuned_choice


def test_flashinfer_tuned_choice_reads_cutlass_tactic(tmp_path) -> None:
    cache = tmp_path / "nvfp4.json"
    cache.write_text(
        json.dumps(
            {
                "_metadata": {},
                "('fp4_gemm', 'CutlassFp4GemmRunner', ((8192, 4096), (4096, 8192)), ())": [
                    "CutlassFp4GemmRunner",
                    2,
                ],
                "('fp4_gemm', 'CudnnFp4GemmRunner', ((4096, 2048), (2048, 4096)), ())": [
                    "CudnnFp4GemmRunner",
                    7,
                ],
            }
        )
    )

    assert _flashinfer_tuned_choice(8192, 8192, 8192, cache) == ("CutlassFp4GemmRunner", 2)


def test_flashinfer_tuned_choice_filters_expected_runner(tmp_path) -> None:
    cache = tmp_path / "nvfp4.json"
    shape = "((8192, 4096), (4096, 8192))"
    cache.write_text(
        json.dumps(
            {
                f"('fp4_gemm', 'CudnnFp4GemmRunner', {shape}, ())": ["CudnnFp4GemmRunner", 8],
                f"('fp4_gemm', 'CutlassFp4GemmRunner', {shape}, ())": ["CutlassFp4GemmRunner", 2],
            }
        )
    )

    assert _flashinfer_tuned_choice(
        8192, 8192, 8192, cache, expected_runner="CutlassFp4GemmRunner"
    ) == ("CutlassFp4GemmRunner", 2)


def test_flashinfer_tuned_choice_rejects_fallback_tactic(tmp_path) -> None:
    cache = tmp_path / "nvfp4.json"
    cache.write_text(
        json.dumps(
            {
                "('fp4_gemm', 'CudnnFp4GemmRunner', ((8192, 4096), (4096, 8192)), ())": [
                    "CudnnFp4GemmRunner",
                    -1,
                ]
            }
        )
    )

    with pytest.raises(RuntimeError, match="tactic=-1"):
        _flashinfer_tuned_choice(8192, 8192, 8192, cache)
