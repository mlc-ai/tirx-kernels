# bench-suite baseline view: `baseline.json`

- Timestamp: `163`
- Label:     `5ec3e10a`
- Git:       `{'tir': '62fc55f9', 'tirx-kernels': 'fdcc24d8-dirty', 'tirx-bench-ci': None}`
- Workloads: 267 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 42.7153 | deepgemm | 42.0984 | 0.986 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 55.9352 | deepgemm | 54.8431 | 0.980 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.7827 | deepgemm | 41.1750 | 0.962 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 56.1704 | deepgemm | 53.5631 | 0.954 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 41.5035 | deepgemm | 43.5758 | 1.050 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 54.2564 | deepgemm | 56.8055 | 1.047 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.9614 | deepgemm | 39.2701 | 0.959 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 53.1524 | deepgemm | 50.4953 | 0.950 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 74.7531 | deepgemm | 72.5006 | 0.970 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 116.2610 | deepgemm | 111.4883 | 0.959 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 75.4455 | deepgemm | 71.2678 | 0.945 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 115.2981 | deepgemm | 107.7645 | 0.935 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 73.9053 | deepgemm | 76.9218 | 1.041 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 114.2421 | deepgemm | 117.8966 | 1.032 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 72.0730 | deepgemm | 67.9335 | 0.943 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 111.1615 | deepgemm | 104.0997 | 0.936 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 74.4961 | deepgemm | 74.5966 | 1.001 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 74.4294 | deepgemm | 74.5942 | 1.002 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 75.1588 | deepgemm | 73.1997 | 0.974 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 74.6318 | deepgemm | 72.6019 | 0.973 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 72.5012 | deepgemm | 77.0097 | 1.062 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 72.1898 | deepgemm | 76.6127 | 1.061 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 72.2620 | deepgemm | 69.2900 | 0.959 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 71.7013 | deepgemm | 68.7001 | 0.958 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 134.7801 | deepgemm | 131.0728 | 0.972 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 196.1438 | deepgemm | 188.9575 | 0.963 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 133.5682 | deepgemm | 127.1602 | 0.952 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 195.3114 | deepgemm | 183.3781 | 0.939 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 131.8065 | deepgemm | 137.4618 | 1.043 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 190.3468 | deepgemm | 197.7031 | 1.039 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 128.8372 | deepgemm | 121.5097 | 0.943 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 186.0951 | deepgemm | 174.6044 | 0.938 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.5807 | deepgemm | 5.6068 | 1.005 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.9024 | deepgemm | 5.9418 | 1.007 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.4774 | deepgemm | 6.5147 | 1.006 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.3587 | deepgemm | 6.3598 | 1.000 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 5.1442 | deepgemm | 5.6750 | 1.103 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.4625 | deepgemm | 4.7824 | 1.072 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5899 | deepgemm | 5.1214 | 1.116 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4629 | deepgemm | 4.7867 | 1.073 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.2098 | deepgemm | 5.2350 | 1.005 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 5.2710 | deepgemm | 5.2989 | 1.005 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.2924 | deepgemm | 5.3189 | 1.005 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.2305 | deepgemm | 5.2705 | 1.008 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.8671 | deepgemm | 4.9047 | 1.008 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.8483 | deepgemm | 5.1627 | 1.065 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7859 | deepgemm | 5.0501 | 1.055 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7044 | deepgemm | 4.7452 | 1.009 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.8025 | deepgemm | 4.8352 | 1.007 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.6599 | deepgemm | 4.6868 | 1.006 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.6174 | deepgemm | 4.6648 | 1.010 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.6602 | deepgemm | 4.6885 | 1.006 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3908 | deepgemm | 4.7562 | 1.083 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.6435 | deepgemm | 5.0323 | 1.084 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2941 | deepgemm | 4.6846 | 1.091 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6314 | deepgemm | 5.0243 | 1.085 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7800 | deepgemm | 4.8150 | 1.007 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7347 | deepgemm | 4.7960 | 1.013 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.5554 | deepgemm | 4.8227 | 1.059 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7228 | deepgemm | 5.0023 | 1.059 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6251 | deepgemm | 4.7476 | 1.026 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.7857 | deepgemm | 5.1115 | 1.068 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6618 | deepgemm | 4.9850 | 1.069 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7743 | deepgemm | 4.9423 | 1.035 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.8185 | deepgemm | 4.8739 | 1.011 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.8753 | deepgemm | 4.9217 | 1.010 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.8756 | deepgemm | 5.1551 | 1.057 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.8049 | deepgemm | 4.8320 | 1.006 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.4510 | deepgemm | 4.7491 | 1.067 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.6174 | deepgemm | 5.1133 | 1.107 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2802 | deepgemm | 4.5799 | 1.070 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6808 | deepgemm | 5.2024 | 1.111 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7278 | deepgemm | 4.9722 | 1.052 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7091 | deepgemm | 4.7579 | 1.010 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7823 | deepgemm | 4.8516 | 1.014 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7523 | deepgemm | 5.0222 | 1.057 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.5010 | deepgemm | 4.4822 | 0.996 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.7322 | deepgemm | 4.8068 | 1.016 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6657 | deepgemm | 4.9106 | 1.052 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7012 | deepgemm | 4.7743 | 1.016 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.2175 | deepgemm | 5.2436 | 1.005 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.9681 | deepgemm | 4.9943 | 1.005 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3085 | deepgemm | 5.3680 | 1.011 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.2479 | deepgemm | 5.3070 | 1.011 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.6249 | deepgemm | 5.1670 | 1.117 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3053 | deepgemm | 4.8834 | 1.134 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.6540 | deepgemm | 5.2040 | 1.118 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2745 | deepgemm | 4.4586 | 1.043 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.6337 | deepgemm | 4.6662 | 1.007 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 5.1435 | deepgemm | 5.3926 | 1.048 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7032 | deepgemm | 4.7239 | 1.004 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8571 | deepgemm | 5.1342 | 1.057 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7173 | deepgemm | 4.9801 | 1.056 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.7341 | deepgemm | 4.7820 | 1.010 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7647 | deepgemm | 4.8095 | 1.009 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5320 | deepgemm | 4.5604 | 1.006 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.9052 | deepgemm | 5.2172 | 1.064 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.2985 | deepgemm | 5.3699 | 1.013 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.5635 | deepgemm | 5.6094 | 1.008 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.4697 | deepgemm | 5.5226 | 1.010 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.5274 | deepgemm | 4.8666 | 1.075 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.4189 | deepgemm | 4.7203 | 1.068 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2809 | deepgemm | 4.8595 | 1.135 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3869 | deepgemm | 4.9202 | 1.122 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7830 | deepgemm | 4.7779 | 0.999 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 5.2386 | deepgemm | 5.3179 | 1.015 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.9528 | deepgemm | 4.9909 | 1.008 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8306 | deepgemm | 4.8464 | 1.003 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 5.1218 | deepgemm | 5.3698 | 1.048 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.7650 | deepgemm | 5.0352 | 1.057 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6411 | deepgemm | 4.6789 | 1.008 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7631 | deepgemm | 4.8436 | 1.017 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 42.8572 | deepgemm | 44.5161 | 1.039 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 57.1172 | deepgemm | 58.1696 | 1.018 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 43.2241 | deepgemm | 42.5748 | 0.985 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 56.1657 | deepgemm | 55.3137 | 0.985 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 43.1997 | deepgemm | 44.7633 | 1.036 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 57.1400 | deepgemm | 58.0942 | 1.017 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 42.8315 | deepgemm | 41.8194 | 0.976 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 55.3553 | deepgemm | 53.5322 | 0.967 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 74.8012 | deepgemm | 77.8311 | 1.041 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 117.2636 | deepgemm | 120.9458 | 1.031 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 73.3090 | deepgemm | 72.9854 | 0.996 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 114.7257 | deepgemm | 114.9091 | 1.002 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 75.7118 | deepgemm | 77.9454 | 1.030 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 116.3664 | deepgemm | 121.5902 | 1.045 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 74.2522 | deepgemm | 72.8133 | 0.981 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 115.6440 | deepgemm | 113.1928 | 0.979 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 75.8224 | deepgemm | 79.1568 | 1.044 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 76.2796 | deepgemm | 80.0003 | 1.049 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 75.2108 | deepgemm | 73.7631 | 0.981 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 75.3117 | deepgemm | 74.3656 | 0.987 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 75.9823 | deepgemm | 78.8656 | 1.038 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 75.8781 | deepgemm | 78.9897 | 1.041 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 75.7590 | deepgemm | 73.1679 | 0.966 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 75.0381 | deepgemm | 72.3321 | 0.964 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 136.5032 | deepgemm | 143.5287 | 1.051 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 196.4188 | deepgemm | 208.7397 | 1.063 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 135.4681 | deepgemm | 131.3605 | 0.970 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 195.7225 | deepgemm | 193.6134 | 0.989 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 136.4931 | deepgemm | 143.4408 | 1.051 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 198.7638 | deepgemm | 209.1734 | 1.052 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 136.2379 | deepgemm | 127.4390 | 0.935 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 196.4166 | deepgemm | 189.6027 | 0.965 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.8409 | deepgemm | 6.8397 | 1.000 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 7.0550 | deepgemm | 7.0578 | 1.000 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4870 | deepgemm | 4.4927 | 1.001 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6856 | deepgemm | 4.7418 | 1.012 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.3929 | deepgemm | 5.4318 | 1.007 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.5660 | deepgemm | 5.5849 | 1.003 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7838 | deepgemm | 4.7591 | 0.995 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.8026 | deepgemm | 4.7693 | 0.993 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.7059 | deepgemm | 4.7197 | 1.003 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9299 | deepgemm | 5.1556 | 1.046 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 5.1067 | deepgemm | 5.3739 | 1.052 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5055 | deepgemm | 4.6637 | 1.035 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7292 | deepgemm | 4.9879 | 1.055 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.1891 | deepgemm | 5.4250 | 1.045 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.0346 | deepgemm | 5.2671 | 1.046 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5010 | deepgemm | 4.5012 | 1.000 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.2053 | deepgemm | 5.1961 | 0.998 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7397 | deepgemm | 4.7408 | 1.000 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5587 | deepgemm | 4.8153 | 1.056 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.9637 | deepgemm | 5.1270 | 1.033 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7726 | deepgemm | 5.0031 | 1.048 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8914 | deepgemm | 4.8966 | 1.001 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7694 | deepgemm | 4.9793 | 1.044 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7445 | deepgemm | 4.9772 | 1.049 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3274 | deepgemm | 5.3291 | 1.000 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.3018 | deepgemm | 5.3329 | 1.006 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.6258 | deepgemm | 4.8340 | 1.045 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6141 | deepgemm | 4.8269 | 1.046 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7392 | deepgemm | 4.7290 | 0.998 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9639 | deepgemm | 4.9794 | 1.003 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7873 | deepgemm | 4.7982 | 1.002 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5708 | deepgemm | 4.5596 | 0.998 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.9221 | deepgemm | 5.8666 | 0.991 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.2933 | deepgemm | 6.2560 | 0.994 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4943 | deepgemm | 4.4485 | 0.990 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4863 | deepgemm | 4.4393 | 0.990 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.2806 | deepgemm | 5.5331 | 1.048 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.2886 | deepgemm | 5.5569 | 1.051 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8467 | deepgemm | 4.8685 | 1.005 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6270 | deepgemm | 4.6047 | 0.995 | — |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.8711 | deepgemm | 5.5825 | 0.951 | — |
| `m13_n24_k7168_s1` | tirx | 24.3509 | deepgemm | 21.4310 | 0.880 | — |
| `m4096_n24_k28672_s16` | tirx | 65.2296 | deepgemm | 63.2815 | 0.970 | — |
| `m4096_n24_k7168_s1` | tirx | 25.9377 | deepgemm | 23.7780 | 0.917 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.8835 | flashattn_sm100 | 19.9337 | 1.003 | — |
| `s1024_h32kv16_causal` | tir | 20.6334 | flashattn_sm100 | 20.2033 | 0.979 | — |
| `s1024_h32kv32` | tir | 20.3027 | flashattn_sm100 | 20.4006 | 1.005 | — |
| `s1024_h32kv32_causal` | tir | 21.0393 | flashattn_sm100 | 21.5295 | 1.023 | — |
| `s1024_h32kv4` | tir | 19.2423 | flashattn_sm100 | 19.6791 | 1.023 | — |
| `s1024_h32kv4_causal` | tir | 19.0394 | flashattn_sm100 | 19.8235 | 1.041 | — |
| `s1024_h32kv8` | tir | 19.5983 | flashattn_sm100 | 19.6776 | 1.004 | — |
| `s1024_h32kv8_causal` | tir | 19.5914 | flashattn_sm100 | 19.7005 | 1.006 | — |
| `s2048_h32kv16` | tir | 57.3689 | flashattn_sm100 | 58.0667 | 1.012 | — |
| `s2048_h32kv16_causal` | tir | 36.2113 | flashattn_sm100 | 38.7562 | 1.070 | — |
| `s2048_h32kv32` | tir | 59.0573 | flashattn_sm100 | 59.6443 | 1.010 | — |
| `s2048_h32kv32_causal` | tir | 40.7151 | flashattn_sm100 | 40.2601 | 0.989 | — |
| `s2048_h32kv4` | tir | 55.3133 | flashattn_sm100 | 56.0365 | 1.013 | — |
| `s2048_h32kv4_causal` | tir | 34.9291 | flashattn_sm100 | 38.0276 | 1.089 | — |
| `s2048_h32kv8` | tir | 56.8599 | flashattn_sm100 | 57.7193 | 1.015 | — |
| `s2048_h32kv8_causal` | tir | 35.0958 | flashattn_sm100 | 37.8379 | 1.078 | — |
| `s4096_h32kv16` | tir | 209.9647 | flashattn_sm100 | 214.7269 | 1.023 | — |
| `s4096_h32kv16_causal` | tir | 112.1895 | flashattn_sm100 | 114.4095 | 1.020 | — |
| `s4096_h32kv32` | tir | 215.8534 | flashattn_sm100 | 219.9906 | 1.019 | — |
| `s4096_h32kv32_causal` | tir | 122.9837 | flashattn_sm100 | 121.3282 | 0.987 | — |
| `s4096_h32kv4` | tir | 208.1472 | flashattn_sm100 | 210.9866 | 1.014 | — |
| `s4096_h32kv4_causal` | tir | 108.7540 | flashattn_sm100 | 115.6559 | 1.063 | — |
| `s4096_h32kv8` | tir | 206.0879 | flashattn_sm100 | 211.6114 | 1.027 | — |
| `s4096_h32kv8_causal` | tir | 109.4769 | flashattn_sm100 | 116.9152 | 1.068 | — |
| `s8192_h32kv16` | tir | 765.7912 | flashattn_sm100 | 759.8214 | 0.992 | — |
| `s8192_h32kv16_causal` | tir | 467.3904 | flashattn_sm100 | 429.1735 | 0.918 | — |
| `s8192_h32kv32` | tir | 786.8265 | flashattn_sm100 | 803.3429 | 1.021 | — |
| `s8192_h32kv32_causal` | tir | 436.8399 | flashattn_sm100 | 434.9412 | 0.996 | — |
| `s8192_h32kv4` | tir | 753.1620 | flashattn_sm100 | 771.7754 | 1.025 | — |
| `s8192_h32kv4_causal` | tir | 408.1656 | flashattn_sm100 | 412.5219 | 1.011 | — |
| `s8192_h32kv8` | tir | 766.9759 | flashattn_sm100 | 772.4355 | 1.007 | — |
| `s8192_h32kv8_causal` | tir | 411.0292 | flashattn_sm100 | 424.4889 | 1.033 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8238 | deepgemm-cublaslt | 5.9957 | 0.879 | deepgemm-bf16=7.9026, torch-cublas=5.9980 |
| `bf16_16384x16384x16384` | tir | 5895.4776 | torch-cublas | 5584.0874 | 0.947 | deepgemm-bf16=6640.1869, deepgemm-cublaslt=5635.9764 |
| `bf16_2048x2048x2048` | tir | 16.3996 | deepgemm-cublaslt | 15.6963 | 0.957 | deepgemm-bf16=18.4549, torch-cublas=15.6990 |
| `bf16_4096x4096x4096` | tir | 92.1395 | deepgemm-cublaslt | 87.2592 | 0.947 | deepgemm-bf16=90.4326, torch-cublas=88.0935 |
| `bf16_8192x8192x8192` | tir | 683.4882 | deepgemm-cublaslt | 695.3538 | 1.017 | deepgemm-bf16=716.4311, torch-cublas=712.0333 |
| `fp16_1024x1024x1024` | tir | 6.9865 | torch-cublas | 6.0781 | 0.870 | deepgemm-cublaslt=6.0931 |
| `fp16_16384x16384x16384` | tir | 5600.6920 | torch-cublas | 5745.2317 | 1.026 | deepgemm-cublaslt=5994.3154 |
| `fp16_2048x2048x2048` | tir | 16.5632 | torch-cublas | 16.0245 | 0.967 | deepgemm-cublaslt=16.0295 |
| `fp16_4096x4096x4096` | tir | 95.6247 | deepgemm-cublaslt | 91.1782 | 0.954 | torch-cublas=91.4866 |
| `fp16_8192x8192x8192` | tir | 732.3880 | deepgemm-cublaslt | 739.0783 | 1.009 | torch-cublas=763.0512 |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.3019 | deepgemm | 50.7254 | 1.008 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 116.0582 | deepgemm | 116.5337 | 1.004 | — |
| `deepgemm_m4096_n32768_k512` | tir | 67.2035 | deepgemm | 72.3361 | 1.076 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 82.8899 | deepgemm | 82.5567 | 0.996 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.0296 | deepgemm | 20.4921 | 1.023 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 336.4848 | deepgemm | 336.0582 | 0.999 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 42.8906 | deepgemm | 43.2909 | 1.009 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 34.0248 | 38.3974 | 35.7114 | 55.9373 | 63.8725 |
| `moe_a3b_bs8_all` | 102.5422 | 103.3668 | 110.6494 | 144.9019 | 151.2682 |
| `moe_a3b_bs32_all` | 205.2653 | 203.8599 | 206.4944 | 238.0147 | 240.3113 |
| `moe_a3b_bs128_all` | 224.3021 | 220.4371 | 231.1820 | 258.7940 | 258.6643 |
| `moe_a3b_bs512_all` | 238.8456 | 231.1713 | 244.9784 | 309.8942 | 297.4227 |
| `moe_a3b_bs1024_all` | 252.4850 | 252.3479 | 271.7742 | 377.9917 | 338.8048 |
| `moe_a3b_bs2048_all` | 338.9925 | 335.4930 | 351.3748 | 456.8172 | 416.0192 |
| `moe_a3b_bs4096_all` | 525.5962 | 534.2915 | 534.8386 | 664.1469 | 599.7899 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.2126 | flashinfer | 4.4804 | 0.860 | cublaslt_nvfp4=4.5215 |
| `16384x16384x16384` | tir | 1452.6373 | flashinfer | 1398.3717 | 0.963 | cublaslt_nvfp4=1405.9607 |
| `2048x2048x2048` | tir | 8.5739 | cublaslt_nvfp4 | 7.4706 | 0.871 | flashinfer=7.5880 |
| `4096x4096x4096` | tir | 29.6593 | cublaslt_nvfp4 | 28.7817 | 0.970 | flashinfer=30.0261 |
| `8192x8192x8192` | tir | 175.4952 | flashinfer | 184.8774 | 1.053 | cublaslt_nvfp4=188.2353 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1624.1453 | flashmla | 1774.6467 | 1.093 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1795.7931 | flashmla | 1899.9648 | 1.058 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1704.9224 | flashmla | 1758.6548 | 1.032 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1803.4952 | flashmla | 1831.0718 | 1.015 | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1971.9645 | flashmla | 2018.3973 | 1.024 | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1690.7523 | flashmla | 1803.9114 | 1.067 | — |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1105.8868 | flashmla | 1164.5153 | 1.053 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1126.6650 | flashmla | 1180.8836 | 1.048 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1079.1054 | flashmla | 1152.0631 | 1.068 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 380.4536 | flashmla | 380.2676 | 1.000 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 367.8164 | flashmla | 385.2372 | 1.047 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 370.0584 | flashmla | 389.4789 | 1.052 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 364.4971 | flashmla | 373.7882 | 1.025 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 377.2035 | flashmla | 395.2780 | 1.048 | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 381.1478 | flashmla | 409.9546 | 1.076 | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 387.7999 | flashmla | 422.3270 | 1.089 | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 368.3422 | flashmla | 385.6752 | 1.047 | — |
