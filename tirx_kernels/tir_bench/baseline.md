# tir-bench baseline view: `tir.json + ref.json`

- Timestamp: `1`
- Label:     `10d2b82a-dirty`
- Git:       `{'tir': '4657e44b-dirty', 'tirx-kernels': 'df166c15-dirty', 'tirx-bench-ci': None}`
- Workloads: 256 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 39.2354 | deepgemm | 41.8448 | 1.067 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 80.4018 | deepgemm | 85.7922 | 1.067 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 39.6526 | deepgemm | 40.8004 | 1.029 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 50.7154 | deepgemm | 52.0008 | 1.025 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 39.3834 | deepgemm | 42.2092 | 1.072 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 81.5792 | deepgemm | 89.1768 | 1.093 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 39.0028 | deepgemm | 38.5782 | 0.989 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 49.5616 | deepgemm | 49.1634 | 0.992 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 67.5070 | deepgemm | 71.4946 | 1.059 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 103.5630 | deepgemm | 109.8914 | 1.061 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 110.4790 | deepgemm | 112.3394 | 1.017 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 105.2148 | deepgemm | 106.4432 | 1.012 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 68.8436 | deepgemm | 74.6672 | 1.085 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 105.5378 | deepgemm | 115.7620 | 1.097 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 107.6930 | deepgemm | 106.8992 | 0.993 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 102.7128 | deepgemm | 102.0052 | 0.993 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 109.3508 | deepgemm | 118.1826 | 1.081 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 68.2424 | deepgemm | 73.8360 | 1.082 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 68.7912 | deepgemm | 71.8610 | 1.045 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 110.4498 | deepgemm | 115.4018 | 1.045 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 110.6046 | deepgemm | 121.4970 | 1.098 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 69.2744 | deepgemm | 75.7784 | 1.094 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 67.0610 | deepgemm | 67.3440 | 1.004 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 107.5652 | deepgemm | 108.8308 | 1.012 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 121.5232 | deepgemm | 129.9796 | 1.070 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 175.0894 | deepgemm | 186.9406 | 1.068 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 123.3298 | deepgemm | 126.1732 | 1.023 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 177.3474 | deepgemm | 180.8738 | 1.020 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 123.5034 | deepgemm | 136.0866 | 1.102 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 177.8788 | deepgemm | 196.3066 | 1.104 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 120.1348 | deepgemm | 120.6848 | 1.005 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 172.2564 | deepgemm | 172.5008 | 1.001 | — |
## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.9768 | deepgemm | 3.9634 | 0.997 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.2434 | deepgemm | 4.2480 | 1.001 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.4558 | deepgemm | 4.4306 | 0.994 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.4556 | deepgemm | 4.4648 | 1.002 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.6808 | deepgemm | 3.8684 | 1.051 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.6698 | deepgemm | 3.8550 | 1.050 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.4882 | deepgemm | 3.6448 | 1.045 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4632 | deepgemm | 3.6368 | 1.050 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.7380 | deepgemm | 3.7110 | 0.993 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6862 | deepgemm | 3.6684 | 0.995 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.8658 | deepgemm | 3.8392 | 0.993 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.8034 | deepgemm | 3.7790 | 0.994 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 6.3864 | deepgemm | 6.3224 | 0.990 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 6.3892 | deepgemm | 6.3430 | 0.993 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.2914 | deepgemm | 6.2304 | 0.990 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.2866 | deepgemm | 6.2536 | 0.995 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.7254 | deepgemm | 3.7172 | 0.998 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.6772 | deepgemm | 3.6698 | 0.998 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.7150 | deepgemm | 3.6958 | 0.995 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.7276 | deepgemm | 3.7180 | 0.997 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.4148 | deepgemm | 3.5970 | 1.053 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 6.0236 | deepgemm | 6.2748 | 1.042 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.4168 | deepgemm | 3.5948 | 1.052 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4618 | deepgemm | 3.6352 | 1.050 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.6046 | deepgemm | 3.5878 | 0.995 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6240 | deepgemm | 3.5996 | 0.993 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6414 | deepgemm | 3.6174 | 0.993 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.5566 | deepgemm | 3.5534 | 0.999 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 6.3572 | deepgemm | 6.2942 | 0.990 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 6.3750 | deepgemm | 6.3150 | 0.991 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.3044 | deepgemm | 6.2290 | 0.988 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.3126 | deepgemm | 6.2574 | 0.991 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.7332 | deepgemm | 3.7256 | 0.998 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.6652 | deepgemm | 3.6504 | 0.996 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.7216 | deepgemm | 3.7086 | 0.997 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.8302 | deepgemm | 3.8206 | 0.997 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.4286 | deepgemm | 3.6368 | 1.061 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.4154 | deepgemm | 3.6070 | 1.056 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.3980 | deepgemm | 3.5890 | 1.056 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.3822 | deepgemm | 3.5900 | 1.061 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.6370 | deepgemm | 3.6084 | 0.992 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6174 | deepgemm | 3.5998 | 0.995 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6082 | deepgemm | 3.5804 | 0.992 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.5760 | deepgemm | 3.5720 | 0.999 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 6.3676 | deepgemm | 6.3122 | 0.991 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 3.7334 | deepgemm | 3.7140 | 0.995 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.3074 | deepgemm | 6.2490 | 0.991 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.3018 | deepgemm | 6.2576 | 0.993 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.7094 | deepgemm | 3.7074 | 0.999 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.6110 | deepgemm | 3.5926 | 0.995 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.8944 | deepgemm | 3.8752 | 0.995 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.8778 | deepgemm | 3.8678 | 0.997 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.4212 | deepgemm | 3.5922 | 1.050 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.5404 | deepgemm | 3.7174 | 1.050 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.4182 | deepgemm | 3.5926 | 1.051 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4202 | deepgemm | 3.5944 | 1.051 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.5880 | deepgemm | 3.5610 | 0.992 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6442 | deepgemm | 3.6240 | 0.994 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6038 | deepgemm | 3.5698 | 0.991 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.5196 | deepgemm | 3.5146 | 0.999 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 6.3654 | deepgemm | 6.3152 | 0.992 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 3.5906 | deepgemm | 3.5734 | 0.995 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.2764 | deepgemm | 6.2182 | 0.991 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.2680 | deepgemm | 6.2278 | 0.994 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.8752 | deepgemm | 3.8578 | 0.996 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.8290 | deepgemm | 3.8088 | 0.995 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.9526 | deepgemm | 3.9306 | 0.994 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.9380 | deepgemm | 3.9222 | 0.996 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.4828 | deepgemm | 3.6528 | 1.049 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.5314 | deepgemm | 3.7318 | 1.057 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.4484 | deepgemm | 3.6630 | 1.062 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4498 | deepgemm | 3.6550 | 1.059 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.6408 | deepgemm | 3.6100 | 0.992 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6392 | deepgemm | 3.6204 | 0.995 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6250 | deepgemm | 3.5994 | 0.993 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.5734 | deepgemm | 3.5782 | 1.001 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 6.3496 | deepgemm | 6.2932 | 0.991 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 3.7260 | deepgemm | 3.7134 | 0.997 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.3534 | deepgemm | 6.2920 | 0.990 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.3622 | deepgemm | 6.3056 | 0.991 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 40.9786 | deepgemm | 43.2588 | 1.056 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 83.3024 | deepgemm | 91.2508 | 1.095 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 41.1424 | deepgemm | 40.8442 | 0.993 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 83.7042 | deepgemm | 84.2848 | 1.007 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 40.6142 | deepgemm | 43.3464 | 1.067 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 81.9182 | deepgemm | 91.0128 | 1.111 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.4332 | deepgemm | 40.0918 | 0.992 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 81.9472 | deepgemm | 82.0190 | 1.001 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 71.5300 | deepgemm | 76.4284 | 1.068 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 107.6200 | deepgemm | 116.0436 | 1.078 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 72.2424 | deepgemm | 72.1952 | 0.999 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 107.7216 | deepgemm | 108.1528 | 1.004 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 71.1096 | deepgemm | 76.1538 | 1.071 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 109.2234 | deepgemm | 115.7616 | 1.060 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 71.2186 | deepgemm | 70.2992 | 0.987 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 110.4270 | deepgemm | 106.2802 | 0.962 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 70.6464 | deepgemm | 77.3186 | 1.094 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 112.0268 | deepgemm | 124.2782 | 1.109 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 70.5780 | deepgemm | 70.8660 | 1.004 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 112.6074 | deepgemm | 113.7106 | 1.010 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 70.3240 | deepgemm | 76.9450 | 1.094 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 110.0212 | deepgemm | 123.4492 | 1.122 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 70.4848 | deepgemm | 69.7416 | 0.989 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 110.5036 | deepgemm | 110.5350 | 1.000 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 126.1058 | deepgemm | 137.3462 | 1.089 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 184.9082 | deepgemm | 198.1844 | 1.072 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 126.8712 | deepgemm | 127.0386 | 1.001 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 183.9772 | deepgemm | 185.5646 | 1.009 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 126.5214 | deepgemm | 137.7232 | 1.089 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 185.7446 | deepgemm | 198.9796 | 1.071 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 126.7194 | deepgemm | 127.1878 | 1.004 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 183.6748 | deepgemm | 185.6682 | 1.011 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.4476 | deepgemm | 4.4196 | 0.994 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 7.3332 | deepgemm | 7.2906 | 0.994 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.6642 | deepgemm | 3.6572 | 0.998 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 6.4090 | deepgemm | 6.3560 | 0.992 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.9808 | deepgemm | 3.9414 | 0.990 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.9662 | deepgemm | 3.9412 | 0.994 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.4222 | deepgemm | 6.3552 | 0.990 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.6732 | deepgemm | 3.6690 | 0.999 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.7244 | deepgemm | 3.6702 | 0.985 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.7358 | deepgemm | 3.7230 | 0.997 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.5696 | deepgemm | 3.5534 | 0.995 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.7136 | deepgemm | 3.7074 | 0.998 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 6.3596 | deepgemm | 6.2722 | 0.986 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6346 | deepgemm | 3.6254 | 0.997 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.7694 | deepgemm | 3.7400 | 0.992 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.3596 | deepgemm | 6.2912 | 0.989 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.5762 | deepgemm | 3.5630 | 0.996 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.7368 | deepgemm | 3.7158 | 0.994 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.6056 | deepgemm | 3.5960 | 0.997 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.6362 | deepgemm | 3.6320 | 0.999 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6848 | deepgemm | 3.6500 | 0.991 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.5322 | deepgemm | 3.5258 | 0.998 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 6.3602 | deepgemm | 6.2954 | 0.990 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.4018 | deepgemm | 6.3328 | 0.989 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.4292 | deepgemm | 6.3188 | 0.983 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.9216 | deepgemm | 3.8998 | 0.994 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 6.3672 | deepgemm | 6.3134 | 0.992 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.5742 | deepgemm | 3.5800 | 1.002 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6488 | deepgemm | 3.6328 | 0.996 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6508 | deepgemm | 3.6530 | 1.001 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6804 | deepgemm | 3.6520 | 0.992 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.3818 | deepgemm | 6.3074 | 0.988 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.0332 | deepgemm | 4.0000 | 0.992 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.2886 | deepgemm | 4.2558 | 0.992 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.7812 | deepgemm | 3.7572 | 0.994 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.7054 | deepgemm | 3.6742 | 0.992 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 6.4390 | deepgemm | 6.3432 | 0.985 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6350 | deepgemm | 3.6360 | 1.000 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6958 | deepgemm | 3.6854 | 0.997 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 6.4172 | deepgemm | 6.3804 | 0.994 | — |
## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.0398 | deepgemm | 5.0444 | 1.001 | — |
| `m13_n24_k7168_s1` | tirx | 20.5056 | deepgemm | 20.6846 | 1.009 | — |
| `m4096_n24_k28672_s16` | tirx | 56.2996 | deepgemm | 56.7726 | 1.008 | — |
| `m4096_n24_k7168_s1` | tirx | 21.7340 | deepgemm | 21.9544 | 1.010 | — |
## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.5962 | flashattn_sm100 | 19.7904 | 1.010 | — |
| `s1024_h32kv16_causal` | tir | 19.8554 | flashattn_sm100 | 19.4234 | 0.978 | — |
| `s1024_h32kv32` | tir | 19.9942 | flashattn_sm100 | 20.0716 | 1.004 | — |
| `s1024_h32kv32_causal` | tir | 31.0104 | flashattn_sm100 | 31.2618 | 1.008 | — |
| `s1024_h32kv4` | tir | 19.2930 | flashattn_sm100 | 19.5710 | 1.014 | — |
| `s1024_h32kv4_causal` | tir | 18.4692 | flashattn_sm100 | 18.9292 | 1.025 | — |
| `s1024_h32kv8` | tir | 29.6066 | flashattn_sm100 | 30.8062 | 1.041 | — |
| `s1024_h32kv8_causal` | tir | 29.5908 | flashattn_sm100 | 30.5240 | 1.032 | — |
| `s2048_h32kv16` | tir | 87.7968 | flashattn_sm100 | 86.2344 | 0.982 | — |
| `s2048_h32kv16_causal` | tir | 35.8128 | flashattn_sm100 | 37.4032 | 1.044 | — |
| `s2048_h32kv32` | tir | 58.0400 | flashattn_sm100 | 58.4660 | 1.007 | — |
| `s2048_h32kv32_causal` | tir | 38.8326 | flashattn_sm100 | 38.9972 | 1.004 | — |
| `s2048_h32kv4` | tir | 55.1594 | flashattn_sm100 | 55.5490 | 1.007 | — |
| `s2048_h32kv4_causal` | tir | 34.5892 | flashattn_sm100 | 36.6120 | 1.058 | — |
| `s2048_h32kv8` | tir | 55.5588 | flashattn_sm100 | 55.9194 | 1.006 | — |
| `s2048_h32kv8_causal` | tir | 54.1202 | flashattn_sm100 | 57.1764 | 1.056 | — |
| `s4096_h32kv16` | tir | 205.0846 | flashattn_sm100 | 206.6698 | 1.008 | — |
| `s4096_h32kv16_causal` | tir | 111.1634 | flashattn_sm100 | 114.3864 | 1.029 | — |
| `s4096_h32kv32` | tir | 210.1902 | flashattn_sm100 | 211.3646 | 1.006 | — |
| `s4096_h32kv32_causal` | tir | 118.7724 | flashattn_sm100 | 118.2650 | 0.996 | — |
| `s4096_h32kv4` | tir | 204.1870 | flashattn_sm100 | 203.4194 | 0.996 | — |
| `s4096_h32kv4_causal` | tir | 109.8378 | flashattn_sm100 | 112.8038 | 1.027 | — |
| `s4096_h32kv8` | tir | 202.3694 | flashattn_sm100 | 201.0826 | 0.994 | — |
| `s4096_h32kv8_causal` | tir | 111.2032 | flashattn_sm100 | 114.4892 | 1.030 | — |
| `s8192_h32kv16` | tir | 827.8186 | flashattn_sm100 | 828.5240 | 1.001 | — |
| `s8192_h32kv16_causal` | tir | 414.5392 | flashattn_sm100 | 434.2396 | 1.048 | — |
| `s8192_h32kv32` | tir | 833.5170 | flashattn_sm100 | 841.3242 | 1.009 | — |
| `s8192_h32kv32_causal` | tir | 442.4606 | flashattn_sm100 | 431.8986 | 0.976 | — |
| `s8192_h32kv4` | tir | 816.9906 | flashattn_sm100 | 814.6276 | 0.997 | — |
| `s8192_h32kv4_causal` | tir | 417.6130 | flashattn_sm100 | 418.9310 | 1.003 | — |
| `s8192_h32kv8` | tir | 835.4284 | flashattn_sm100 | 808.2412 | 0.967 | — |
| `s8192_h32kv8_causal` | tir | 416.9218 | flashattn_sm100 | 407.2240 | 0.977 | — |
## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.3152 | torch-cublas | 5.3254 | 0.843 | deepgemm-bf16=7.3400, deepgemm-cublaslt=5.3696 |
| `bf16_16384x16384x16384` | tir | 6249.7348 | torch-cublas | 6163.9238 | 0.986 | deepgemm-bf16=7410.2664, deepgemm-cublaslt=6168.3394 |
| `bf16_2048x2048x2048` | tir | 15.7870 | deepgemm-cublaslt | 17.0282 | 1.079 | deepgemm-bf16=17.8878, torch-cublas=17.4236 |
| `bf16_4096x4096x4096` | tir | 137.8338 | deepgemm-bf16 | 126.4490 | 0.917 | deepgemm-cublaslt=135.4836, torch-cublas=135.5056 |
| `bf16_8192x8192x8192` | tir | 724.7116 | deepgemm-bf16 | 758.1970 | 1.046 | deepgemm-cublaslt=769.4888, torch-cublas=777.4344 |
| `fp16_1024x1024x1024` | tir | 6.3238 | torch-cublas | 5.3666 | 0.849 | deepgemm-cublaslt=5.4222 |
| `fp16_16384x16384x16384` | tir | 6371.7214 | torch-cublas | 6292.6094 | 0.988 | deepgemm-cublaslt=6310.8330 |
| `fp16_2048x2048x2048` | tir | 16.2970 | deepgemm-cublaslt | 17.6354 | 1.082 | torch-cublas=17.9864 |
| `fp16_4096x4096x4096` | tir | 96.1704 | torch-cublas | 97.9260 | 1.018 | deepgemm-cublaslt=97.9766 |
| `fp16_8192x8192x8192` | tir | 783.3476 | deepgemm-cublaslt | 827.5840 | 1.056 | torch-cublas=834.2480 |
## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 49.1410 | deepgemm | 49.8720 | 1.015 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 116.5584 | deepgemm | 117.1714 | 1.005 | — |
| `deepgemm_m4096_n32768_k512` | tir | 73.0828 | deepgemm | 76.8444 | 1.051 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 83.0924 | deepgemm | 83.5202 | 1.005 | — |
| `deepgemm_m4096_n576_k7168` | tir | 27.4920 | deepgemm | 29.8466 | 1.086 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 370.1946 | deepgemm | 372.3680 | 1.006 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 44.4600 | deepgemm | 44.8032 | 1.008 | — |
## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.0716 | cublaslt_nvfp4 | 4.2224 | 0.833 | flashinfer=4.3438 |
| `16384x16384x16384` | tir | 1631.8342 | flashinfer | 1566.6044 | 0.960 | cublaslt_nvfp4=1577.0186 |
| `2048x2048x2048` | tir | 13.6242 | flashinfer | 12.3292 | 0.905 | cublaslt_nvfp4=15.3006 |
| `4096x4096x4096` | tir | 29.3036 | flashinfer | 28.5838 | 0.975 | cublaslt_nvfp4=30.0556 |
| `8192x8192x8192` | tir | 185.4254 | flashinfer | 182.1540 | 0.982 | cublaslt_nvfp4=182.7252 |
## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1811.7590 | flashmla | 1854.7740 | 1.024 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 2468.3494 | flashmla | 2506.9340 | 1.016 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1784.0872 | flashmla | 1817.6072 | 1.019 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1871.3944 | flashmla | 1918.5914 | 1.025 | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2106.4026 | flashmla | 2115.0278 | 1.004 | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1902.6128 | flashmla | 1947.9200 | 1.024 | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 369.1212 | flashmla | 374.2908 | 1.014 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 584.1140 | flashmla | 593.9050 | 1.017 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 378.8154 | flashmla | 383.0790 | 1.011 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 366.1532 | flashmla | 370.2996 | 1.011 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 384.8004 | flashmla | 396.4802 | 1.030 | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 599.9964 | flashmla | 612.9574 | 1.022 | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 398.4434 | flashmla | 412.8488 | 1.036 | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 373.0568 | flashmla | 378.8500 | 1.016 | — |
