# tir-bench baseline view: `tir.json + ref.json`

- Timestamp: `12`
- Label:     `3a146e3c-dirty`
- Git:       `{'tir': 'c467b1b9', 'tirx-kernels': '3a146e3c-dirty', 'tirx-bench-ci': None}`
- Workloads: 256 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 39.0528 | deepgemm | 41.4492 | 1.061 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 50.4636 | deepgemm | 53.7064 | 1.064 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 39.1974 | deepgemm | 40.2046 | 1.026 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 50.7348 | deepgemm | 51.9132 | 1.023 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 39.3552 | deepgemm | 42.4088 | 1.078 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 51.3034 | deepgemm | 55.6612 | 1.085 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 38.6596 | deepgemm | 38.2982 | 0.991 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 49.6118 | deepgemm | 49.2016 | 0.992 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 67.8256 | deepgemm | 71.5942 | 1.056 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 104.1978 | deepgemm | 110.3214 | 1.059 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 68.4344 | deepgemm | 69.7630 | 1.019 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 105.2054 | deepgemm | 106.6070 | 1.013 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 68.6354 | deepgemm | 74.5248 | 1.086 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 105.7710 | deepgemm | 116.1390 | 1.098 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 66.7216 | deepgemm | 65.8792 | 0.987 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 102.9428 | deepgemm | 102.2286 | 0.993 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 68.5622 | deepgemm | 73.8250 | 1.077 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 68.4964 | deepgemm | 73.8336 | 1.078 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 68.6648 | deepgemm | 71.6704 | 1.044 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 68.6962 | deepgemm | 71.6460 | 1.043 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 69.1526 | deepgemm | 75.4456 | 1.091 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 69.0176 | deepgemm | 75.3618 | 1.092 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 66.9856 | deepgemm | 67.4862 | 1.007 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 66.8244 | deepgemm | 67.2304 | 1.006 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 121.8894 | deepgemm | 129.7596 | 1.065 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 174.6916 | deepgemm | 185.7610 | 1.063 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 123.7708 | deepgemm | 126.5750 | 1.023 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 177.2958 | deepgemm | 180.8310 | 1.020 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 123.9858 | deepgemm | 136.5624 | 1.101 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 178.4298 | deepgemm | 197.0844 | 1.105 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 119.9626 | deepgemm | 120.5018 | 1.004 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 171.7834 | deepgemm | 172.6364 | 1.005 | — |
## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.9608 | deepgemm | 3.9314 | 0.993 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.2380 | deepgemm | 4.2426 | 1.001 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.4350 | deepgemm | 4.4202 | 0.997 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.4806 | deepgemm | 4.4800 | 1.000 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.7690 | deepgemm | 3.9432 | 1.046 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.7932 | deepgemm | 3.9786 | 1.049 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.5372 | deepgemm | 3.6970 | 1.045 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4704 | deepgemm | 3.6774 | 1.060 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.7768 | deepgemm | 3.7482 | 0.992 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.9704 | deepgemm | 3.9496 | 0.995 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.9608 | deepgemm | 3.9400 | 0.995 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.8990 | deepgemm | 3.8880 | 0.997 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.0540 | deepgemm | 4.0362 | 0.996 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 3.9842 | deepgemm | 3.9628 | 0.995 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6162 | deepgemm | 3.5806 | 0.990 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.5948 | deepgemm | 3.5888 | 0.998 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.6582 | deepgemm | 3.6324 | 0.993 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.0358 | deepgemm | 4.0166 | 0.995 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.7036 | deepgemm | 3.6862 | 0.995 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.6448 | deepgemm | 3.6354 | 0.997 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.8584 | deepgemm | 4.0046 | 1.038 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.4554 | deepgemm | 3.6338 | 1.052 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.3990 | deepgemm | 3.5876 | 1.055 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.3826 | deepgemm | 3.6008 | 1.065 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.0328 | deepgemm | 4.0094 | 0.994 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.0212 | deepgemm | 4.0074 | 0.997 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.9136 | deepgemm | 3.8872 | 0.993 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.9312 | deepgemm | 3.9108 | 0.995 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 3.6440 | deepgemm | 3.6248 | 0.995 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 3.7272 | deepgemm | 3.7002 | 0.993 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.9404 | deepgemm | 3.9046 | 0.991 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.6428 | deepgemm | 3.6260 | 0.995 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.0356 | deepgemm | 4.0188 | 0.996 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.7344 | deepgemm | 3.7228 | 0.997 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.9492 | deepgemm | 3.9330 | 0.996 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.8358 | deepgemm | 3.8182 | 0.995 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.5204 | deepgemm | 3.6884 | 1.048 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.4562 | deepgemm | 3.6316 | 1.051 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.7700 | deepgemm | 3.9118 | 1.038 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.8430 | deepgemm | 3.9926 | 1.039 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.9480 | deepgemm | 3.9322 | 0.996 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.9744 | deepgemm | 3.9476 | 0.993 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6194 | deepgemm | 3.5964 | 0.994 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6558 | deepgemm | 3.6364 | 0.995 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 3.7240 | deepgemm | 3.7094 | 0.996 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 3.6542 | deepgemm | 3.6320 | 0.994 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6444 | deepgemm | 3.6200 | 0.993 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.5728 | deepgemm | 3.5672 | 0.998 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.9664 | deepgemm | 3.9562 | 0.997 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.9664 | deepgemm | 3.9550 | 0.997 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.8416 | deepgemm | 3.8192 | 0.994 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.8446 | deepgemm | 3.8398 | 0.999 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.4950 | deepgemm | 3.6558 | 1.046 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.8560 | deepgemm | 4.0200 | 1.043 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.8486 | deepgemm | 3.9966 | 1.038 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4798 | deepgemm | 3.6862 | 1.059 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.6502 | deepgemm | 3.6360 | 0.996 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6248 | deepgemm | 3.6052 | 0.995 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.7512 | deepgemm | 3.7130 | 0.990 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6098 | deepgemm | 3.5962 | 0.996 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 3.7060 | deepgemm | 3.6874 | 0.995 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.0436 | deepgemm | 4.0228 | 0.995 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6030 | deepgemm | 3.5728 | 0.992 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.9936 | deepgemm | 3.9758 | 0.996 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 3.6796 | deepgemm | 3.6552 | 0.993 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 3.8306 | deepgemm | 3.8152 | 0.996 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.0730 | deepgemm | 4.0490 | 0.994 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.8858 | deepgemm | 3.8850 | 1.000 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.8674 | deepgemm | 4.0218 | 1.040 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.5254 | deepgemm | 3.7258 | 1.057 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.4246 | deepgemm | 3.6302 | 1.060 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.4142 | deepgemm | 3.6294 | 1.063 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 3.9808 | deepgemm | 3.9610 | 0.995 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 3.6626 | deepgemm | 3.6512 | 0.997 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.5700 | deepgemm | 3.5460 | 0.993 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.0240 | deepgemm | 4.0102 | 0.997 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 3.6382 | deepgemm | 3.5990 | 0.989 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.0558 | deepgemm | 4.0412 | 0.996 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6986 | deepgemm | 3.6650 | 0.991 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.9604 | deepgemm | 3.9344 | 0.993 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 41.5712 | deepgemm | 43.7146 | 1.052 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 54.5962 | deepgemm | 57.7064 | 1.057 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 41.1702 | deepgemm | 40.9190 | 0.994 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 53.7680 | deepgemm | 53.8580 | 1.002 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 40.7020 | deepgemm | 43.3454 | 1.065 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 53.8916 | deepgemm | 57.2530 | 1.062 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 41.0534 | deepgemm | 40.7790 | 0.993 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 52.5876 | deepgemm | 51.7582 | 0.984 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 70.5332 | deepgemm | 75.7720 | 1.074 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 108.4644 | deepgemm | 116.6622 | 1.076 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 69.9804 | deepgemm | 70.0708 | 1.001 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 109.8610 | deepgemm | 110.6138 | 1.007 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 69.0042 | deepgemm | 75.4814 | 1.094 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 114.7394 | deepgemm | 116.2582 | 1.013 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 69.5186 | deepgemm | 68.6358 | 0.987 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 116.6902 | deepgemm | 106.7686 | 0.915 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 70.1938 | deepgemm | 77.2318 | 1.100 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 70.0840 | deepgemm | 76.9674 | 1.098 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 70.2026 | deepgemm | 70.5664 | 1.005 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 70.3950 | deepgemm | 70.8030 | 1.006 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 69.7520 | deepgemm | 76.7488 | 1.100 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 71.1504 | deepgemm | 77.7792 | 1.093 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 69.9604 | deepgemm | 69.9062 | 0.999 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 69.8496 | deepgemm | 69.3916 | 0.993 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 126.1686 | deepgemm | 137.2644 | 1.088 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 186.8726 | deepgemm | 199.0896 | 1.065 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 125.8938 | deepgemm | 126.1572 | 1.002 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 188.0258 | deepgemm | 188.8176 | 1.004 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 130.1268 | deepgemm | 138.5152 | 1.064 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 183.1572 | deepgemm | 197.4068 | 1.078 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 128.6312 | deepgemm | 129.1630 | 1.004 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 185.8836 | deepgemm | 188.1062 | 1.012 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.4458 | deepgemm | 4.4128 | 0.993 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.4570 | deepgemm | 4.4496 | 0.998 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.0774 | deepgemm | 4.0494 | 0.993 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.6672 | deepgemm | 3.6564 | 0.997 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.9814 | deepgemm | 3.9352 | 0.988 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.0758 | deepgemm | 4.0450 | 0.992 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.9910 | deepgemm | 3.9756 | 0.996 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.6728 | deepgemm | 3.6428 | 0.992 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.7164 | deepgemm | 3.6664 | 0.987 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.6442 | deepgemm | 3.6366 | 0.998 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.5842 | deepgemm | 3.5728 | 0.997 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.7246 | deepgemm | 3.7140 | 0.997 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6312 | deepgemm | 3.6206 | 0.997 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6566 | deepgemm | 3.6460 | 0.997 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.9428 | deepgemm | 3.9424 | 1.000 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.6176 | deepgemm | 3.6136 | 0.999 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.0086 | deepgemm | 3.9892 | 0.995 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.6966 | deepgemm | 3.6968 | 1.000 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.0474 | deepgemm | 4.0252 | 0.995 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.0206 | deepgemm | 4.0064 | 0.996 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.6146 | deepgemm | 3.6052 | 0.997 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.7694 | deepgemm | 3.7454 | 0.994 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6778 | deepgemm | 3.6514 | 0.993 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.6464 | deepgemm | 3.6472 | 1.000 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 3.9240 | deepgemm | 3.8926 | 0.992 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 3.9206 | deepgemm | 3.8920 | 0.993 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.6462 | deepgemm | 3.6290 | 0.995 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.6512 | deepgemm | 3.6446 | 0.998 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.0678 | deepgemm | 4.0132 | 0.987 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.5522 | deepgemm | 3.5496 | 0.999 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.6936 | deepgemm | 3.6478 | 0.988 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.9736 | deepgemm | 3.9584 | 0.996 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.0128 | deepgemm | 3.9904 | 0.994 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.4800 | deepgemm | 4.4646 | 0.997 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.7110 | deepgemm | 3.7026 | 0.998 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.0542 | deepgemm | 4.0408 | 0.997 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 3.8628 | deepgemm | 3.8436 | 0.995 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 3.6420 | deepgemm | 3.6528 | 1.003 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 3.9942 | deepgemm | 3.9924 | 1.000 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 3.6678 | deepgemm | 3.6590 | 0.998 | — |
## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.0322 | deepgemm | 5.0102 | 0.996 | — |
| `m13_n24_k7168_s1` | tirx | 20.4722 | deepgemm | 20.6454 | 1.008 | — |
| `m4096_n24_k28672_s16` | tirx | 56.4038 | deepgemm | 56.8650 | 1.008 | — |
| `m4096_n24_k7168_s1` | tirx | 21.7128 | deepgemm | 21.9608 | 1.011 | — |
## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.6242 | flashattn_sm100 | 19.8208 | 1.010 | — |
| `s1024_h32kv16_causal` | tir | 19.2018 | flashattn_sm100 | 19.3654 | 1.009 | — |
| `s1024_h32kv32` | tir | 20.1002 | flashattn_sm100 | 20.1028 | 1.000 | — |
| `s1024_h32kv32_causal` | tir | 20.5508 | flashattn_sm100 | 20.0346 | 0.975 | — |
| `s1024_h32kv4` | tir | 19.2656 | flashattn_sm100 | 19.5874 | 1.017 | — |
| `s1024_h32kv4_causal` | tir | 18.8606 | flashattn_sm100 | 19.0948 | 1.012 | — |
| `s1024_h32kv8` | tir | 19.6302 | flashattn_sm100 | 19.9304 | 1.015 | — |
| `s1024_h32kv8_causal` | tir | 18.2838 | flashattn_sm100 | 18.9610 | 1.037 | — |
| `s2048_h32kv16` | tir | 56.2074 | flashattn_sm100 | 56.6262 | 1.007 | — |
| `s2048_h32kv16_causal` | tir | 36.1102 | flashattn_sm100 | 37.9766 | 1.052 | — |
| `s2048_h32kv32` | tir | 57.8408 | flashattn_sm100 | 57.9906 | 1.003 | — |
| `s2048_h32kv32_causal` | tir | 38.1550 | flashattn_sm100 | 38.1450 | 1.000 | — |
| `s2048_h32kv4` | tir | 54.4104 | flashattn_sm100 | 54.9310 | 1.010 | — |
| `s2048_h32kv4_causal` | tir | 34.5532 | flashattn_sm100 | 36.6602 | 1.061 | — |
| `s2048_h32kv8` | tir | 55.9074 | flashattn_sm100 | 56.4110 | 1.009 | — |
| `s2048_h32kv8_causal` | tir | 34.6696 | flashattn_sm100 | 36.4542 | 1.051 | — |
| `s4096_h32kv16` | tir | 205.4442 | flashattn_sm100 | 206.4930 | 1.005 | — |
| `s4096_h32kv16_causal` | tir | 111.9902 | flashattn_sm100 | 115.4514 | 1.031 | — |
| `s4096_h32kv32` | tir | 211.0658 | flashattn_sm100 | 212.4378 | 1.007 | — |
| `s4096_h32kv32_causal` | tir | 117.1522 | flashattn_sm100 | 116.4474 | 0.994 | — |
| `s4096_h32kv4` | tir | 204.1070 | flashattn_sm100 | 203.5118 | 0.997 | — |
| `s4096_h32kv4_causal` | tir | 109.3778 | flashattn_sm100 | 111.7976 | 1.022 | — |
| `s4096_h32kv8` | tir | 202.9702 | flashattn_sm100 | 202.7426 | 0.999 | — |
| `s4096_h32kv8_causal` | tir | 110.4246 | flashattn_sm100 | 113.9544 | 1.032 | — |
| `s8192_h32kv16` | tir | 841.0486 | flashattn_sm100 | 830.1202 | 0.987 | — |
| `s8192_h32kv16_causal` | tir | 417.5102 | flashattn_sm100 | 418.7370 | 1.003 | — |
| `s8192_h32kv32` | tir | 841.2190 | flashattn_sm100 | 834.8080 | 0.992 | — |
| `s8192_h32kv32_causal` | tir | 440.2474 | flashattn_sm100 | 438.5840 | 0.996 | — |
| `s8192_h32kv4` | tir | 853.7838 | flashattn_sm100 | 837.4688 | 0.981 | — |
| `s8192_h32kv4_causal` | tir | 408.7442 | flashattn_sm100 | 431.5668 | 1.056 | — |
| `s8192_h32kv8` | tir | 841.1218 | flashattn_sm100 | 832.8190 | 0.990 | — |
| `s8192_h32kv8_causal` | tir | 425.0300 | flashattn_sm100 | 426.0576 | 1.002 | — |
## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.4456 | torch-cublas | 5.3760 | 0.834 | deepgemm-bf16=7.4892, deepgemm-cublaslt=5.4266 |
| `bf16_16384x16384x16384` | tir | 6283.4044 | torch-cublas | 6081.9526 | 0.968 | deepgemm-bf16=6655.4354, deepgemm-cublaslt=6113.4668 |
| `bf16_2048x2048x2048` | tir | 15.8594 | torch-cublas | 17.2414 | 1.087 | deepgemm-bf16=17.9342, deepgemm-cublaslt=17.5336 |
| `bf16_4096x4096x4096` | tir | 92.2864 | deepgemm-bf16 | 91.9502 | 0.996 | deepgemm-cublaslt=93.8422, torch-cublas=93.9048 |
| `bf16_8192x8192x8192` | tir | 770.5386 | deepgemm-cublaslt | 799.6210 | 1.038 | deepgemm-bf16=827.0226, torch-cublas=836.7196 |
| `fp16_1024x1024x1024` | tir | 6.3156 | torch-cublas | 5.3472 | 0.847 | deepgemm-cublaslt=5.3838 |
| `fp16_16384x16384x16384` | tir | 6523.3726 | deepgemm-cublaslt | 6440.7908 | 0.987 | torch-cublas=6444.0982 |
| `fp16_2048x2048x2048` | tir | 16.0552 | torch-cublas | 17.5884 | 1.095 | deepgemm-cublaslt=17.7006 |
| `fp16_4096x4096x4096` | tir | 96.6986 | deepgemm-cublaslt | 99.1310 | 1.025 | torch-cublas=99.2578 |
| `fp16_8192x8192x8192` | tir | 782.9782 | torch-cublas | 836.7684 | 1.069 | deepgemm-cublaslt=875.7346 |
## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 49.0142 | deepgemm | 49.9134 | 1.018 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 117.6808 | deepgemm | 118.1730 | 1.004 | — |
| `deepgemm_m4096_n32768_k512` | tir | 72.7242 | deepgemm | 76.4026 | 1.051 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 82.6742 | deepgemm | 82.9742 | 1.004 | — |
| `deepgemm_m4096_n576_k7168` | tir | 18.1556 | deepgemm | 18.6968 | 1.030 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 327.3532 | deepgemm | 327.6390 | 1.001 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 44.1186 | deepgemm | 44.5254 | 1.009 | — |
## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.1832 | cublaslt_nvfp4 | 4.2432 | 0.819 | flashinfer=4.3290 |
| `16384x16384x16384` | tir | 1667.7848 | cublaslt_nvfp4 | 1569.8754 | 0.941 | flashinfer=1589.4680 |
| `2048x2048x2048` | tir | 8.3540 | cublaslt_nvfp4 | 7.4380 | 0.890 | flashinfer=7.5904 |
| `4096x4096x4096` | tir | 29.1422 | flashinfer | 28.7304 | 0.986 | cublaslt_nvfp4=29.9500 |
| `8192x8192x8192` | tir | 185.4468 | flashinfer | 181.8584 | 0.981 | cublaslt_nvfp4=182.9938 |
## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1805.4558 | flashmla | 1848.2204 | 1.024 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1974.9740 | flashmla | 2003.4210 | 1.014 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1826.5718 | flashmla | 1862.2020 | 1.020 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1927.7038 | flashmla | 1955.0134 | 1.014 | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2134.2220 | flashmla | 2138.2772 | 1.002 | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1897.3562 | flashmla | 1952.2334 | 1.029 | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 369.4776 | flashmla | 373.0770 | 1.010 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 372.6980 | flashmla | 377.4166 | 1.013 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 372.9602 | flashmla | 381.1098 | 1.022 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 365.7822 | flashmla | 369.0582 | 1.009 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 385.2106 | flashmla | 392.2244 | 1.018 | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 388.5462 | flashmla | 399.2042 | 1.027 | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 398.8962 | flashmla | 404.7574 | 1.015 | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 375.5956 | flashmla | 381.4672 | 1.016 | — |
