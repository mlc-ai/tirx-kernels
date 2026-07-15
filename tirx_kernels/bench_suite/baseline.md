# bench-suite baseline view: `baseline.json`

- Timestamp: `41`
- Label:     `all-workloads-current-promote`
- Git:       `{'tir': '0496bb65-dirty', 'tirx-kernels': '1ed585b6-dirty', 'tirx-bench-ci': None}`
- Workloads: 283 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1399.2000 | deepgemm | 1402.2000 | 1.002 | — |
| `t64_m64_h7168_i3072_e384_k6_g2` | tirx | 955.8032 | deepgemm | 944.9782 | 0.989 | — |
| `t64_m64_h7168_i3072_e384_k6_g4` | tirx | 627.5820 | deepgemm | 598.9358 | 0.954 | — |
| `t64_m64_h7168_i3072_e384_k6_g6` | tirx | 475.7514 | deepgemm | 455.0418 | 0.956 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3609.2000 | deepgemm | 3626.2000 | 1.005 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g2` | tirx | 3492.2000 | deepgemm | 3505.8000 | 1.004 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g4` | tirx | 2955.6000 | deepgemm | 2942.4000 | 0.996 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g6` | tirx | 2953.8000 | deepgemm | 2949.6000 | 0.999 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 43.0062 | deepgemm | 41.5223 | 0.965 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 55.8358 | deepgemm | 53.2993 | 0.955 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.7070 | deepgemm | 41.7193 | 0.977 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 55.9296 | deepgemm | 54.5107 | 0.975 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 41.2606 | deepgemm | 41.9477 | 1.017 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 54.3662 | deepgemm | 54.9932 | 1.012 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 41.1534 | deepgemm | 40.3469 | 0.980 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 53.2354 | deepgemm | 51.9914 | 0.977 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 74.9800 | deepgemm | 70.7223 | 0.943 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 116.5681 | deepgemm | 108.9220 | 0.934 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 75.0010 | deepgemm | 72.6260 | 0.968 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 116.4296 | deepgemm | 112.1071 | 0.963 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 73.7235 | deepgemm | 73.6135 | 0.999 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 114.3718 | deepgemm | 112.9516 | 0.988 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 71.7964 | deepgemm | 69.8559 | 0.973 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 110.5012 | deepgemm | 106.8015 | 0.967 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 74.3970 | deepgemm | 71.9727 | 0.967 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 74.7510 | deepgemm | 72.4158 | 0.969 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 74.9382 | deepgemm | 73.6403 | 0.983 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 75.1847 | deepgemm | 74.0376 | 0.985 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 72.3583 | deepgemm | 74.5300 | 1.030 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 72.2611 | deepgemm | 74.6200 | 1.033 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 71.8131 | deepgemm | 70.5902 | 0.983 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 72.2451 | deepgemm | 71.0898 | 0.984 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 134.8443 | deepgemm | 127.0916 | 0.943 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 194.0348 | deepgemm | 181.2173 | 0.934 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 135.1020 | deepgemm | 131.2801 | 0.972 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 194.5108 | deepgemm | 187.8279 | 0.966 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 132.2135 | deepgemm | 132.7432 | 1.004 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 190.6580 | deepgemm | 189.9467 | 0.996 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 129.7662 | deepgemm | 126.0276 | 0.971 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 185.1761 | deepgemm | 178.7103 | 0.965 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.6432 | deepgemm | 5.7489 | 1.019 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.9451 | deepgemm | 5.9318 | 0.998 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.5587 | deepgemm | 6.5596 | 1.000 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.2890 | deepgemm | 6.2676 | 0.997 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.7117 | deepgemm | 5.0226 | 1.066 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.5464 | deepgemm | 5.0773 | 1.117 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4473 | deepgemm | 5.0273 | 1.130 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5053 | deepgemm | 4.7818 | 1.061 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.1004 | deepgemm | 5.2279 | 1.025 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.8428 | deepgemm | 5.0242 | 1.037 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.2102 | deepgemm | 5.3063 | 1.018 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.1565 | deepgemm | 5.2166 | 1.012 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.8436 | deepgemm | 5.1726 | 1.068 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.8272 | deepgemm | 5.1092 | 1.058 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8189 | deepgemm | 5.1335 | 1.065 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.8341 | deepgemm | 5.1119 | 1.057 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.8720 | deepgemm | 4.9249 | 1.011 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.8607 | deepgemm | 5.1131 | 1.052 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.6899 | deepgemm | 4.7577 | 1.014 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.6803 | deepgemm | 4.7328 | 1.011 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3354 | deepgemm | 4.7647 | 1.099 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.4199 | deepgemm | 4.8241 | 1.091 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.3391 | deepgemm | 4.7625 | 1.098 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3911 | deepgemm | 4.7934 | 1.092 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.6285 | deepgemm | 4.6998 | 1.015 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.5052 | deepgemm | 4.5698 | 1.014 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.9858 | deepgemm | 5.3285 | 1.069 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5502 | deepgemm | 4.6057 | 1.012 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6171 | deepgemm | 4.8574 | 1.052 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6819 | deepgemm | 4.8563 | 1.037 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.9659 | deepgemm | 5.2914 | 1.066 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.0868 | deepgemm | 5.4315 | 1.068 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.6639 | deepgemm | 4.7221 | 1.012 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.6841 | deepgemm | 4.7289 | 1.010 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.9839 | deepgemm | 5.2594 | 1.055 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1929 | deepgemm | 5.3472 | 1.030 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.1767 | deepgemm | 4.4593 | 1.068 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.5994 | deepgemm | 5.1300 | 1.115 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.1923 | deepgemm | 4.4772 | 1.068 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2007 | deepgemm | 4.5088 | 1.073 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.6128 | deepgemm | 4.7353 | 1.027 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.6940 | deepgemm | 4.7508 | 1.012 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.6101 | deepgemm | 4.7036 | 1.020 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8567 | deepgemm | 4.9332 | 1.016 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.5554 | deepgemm | 4.6350 | 1.017 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.4978 | deepgemm | 4.4990 | 1.000 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5898 | deepgemm | 4.6194 | 1.006 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5503 | deepgemm | 4.5916 | 1.009 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.1780 | deepgemm | 5.2712 | 1.018 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.8993 | deepgemm | 4.9563 | 1.012 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.2343 | deepgemm | 5.3134 | 1.015 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.2361 | deepgemm | 5.3357 | 1.019 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3351 | deepgemm | 4.8887 | 1.128 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3392 | deepgemm | 4.8805 | 1.125 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4760 | deepgemm | 4.7751 | 1.067 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2773 | deepgemm | 4.8457 | 1.133 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.8769 | deepgemm | 4.9286 | 1.011 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7751 | deepgemm | 5.0129 | 1.050 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.8148 | deepgemm | 5.1007 | 1.059 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8269 | deepgemm | 5.1091 | 1.058 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7276 | deepgemm | 4.9747 | 1.052 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.7589 | deepgemm | 4.8035 | 1.009 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7156 | deepgemm | 4.9447 | 1.049 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6413 | deepgemm | 4.9037 | 1.057 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.0342 | deepgemm | 5.0819 | 1.009 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.2524 | deepgemm | 5.3509 | 1.019 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.5136 | deepgemm | 5.6089 | 1.017 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.4205 | deepgemm | 5.4567 | 1.007 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3125 | deepgemm | 4.6859 | 1.087 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3049 | deepgemm | 4.6331 | 1.076 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2328 | deepgemm | 4.5399 | 1.073 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2706 | deepgemm | 4.5541 | 1.066 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7620 | deepgemm | 4.8429 | 1.017 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7683 | deepgemm | 4.8279 | 1.013 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.1908 | deepgemm | 5.3565 | 1.032 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8594 | deepgemm | 4.8728 | 1.003 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6010 | deepgemm | 4.6739 | 1.016 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.9057 | deepgemm | 4.9676 | 1.013 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.1267 | deepgemm | 5.2168 | 1.018 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.9279 | deepgemm | 5.0035 | 1.015 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 42.8528 | deepgemm | 41.6122 | 0.971 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 56.7340 | deepgemm | 54.0573 | 0.953 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.9226 | deepgemm | 41.4478 | 0.966 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 55.4266 | deepgemm | 52.9173 | 0.955 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 43.7241 | deepgemm | 42.4809 | 0.972 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 56.7987 | deepgemm | 54.6618 | 0.962 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 42.4068 | deepgemm | 39.6624 | 0.935 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 56.1529 | deepgemm | 51.6371 | 0.920 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 75.8712 | deepgemm | 71.1743 | 0.938 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 115.6329 | deepgemm | 110.9118 | 0.959 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 74.2119 | deepgemm | 70.5415 | 0.951 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 115.0241 | deepgemm | 111.0248 | 0.965 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 74.8350 | deepgemm | 72.4768 | 0.968 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 116.9511 | deepgemm | 113.6718 | 0.972 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 74.1868 | deepgemm | 68.4685 | 0.923 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 114.7988 | deepgemm | 107.9853 | 0.941 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 75.9900 | deepgemm | 73.2358 | 0.964 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 75.5921 | deepgemm | 73.1906 | 0.968 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 74.6944 | deepgemm | 71.8495 | 0.962 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 74.9278 | deepgemm | 72.2396 | 0.964 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 75.8642 | deepgemm | 73.7150 | 0.972 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 76.1409 | deepgemm | 73.9917 | 0.972 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 75.4596 | deepgemm | 70.4433 | 0.934 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 74.9051 | deepgemm | 69.8174 | 0.932 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 136.8393 | deepgemm | 131.9712 | 0.964 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 197.4692 | deepgemm | 192.3841 | 0.974 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 135.4557 | deepgemm | 129.3147 | 0.955 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 195.7844 | deepgemm | 189.0655 | 0.966 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 136.0009 | deepgemm | 132.7810 | 0.976 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 198.4929 | deepgemm | 196.4736 | 0.990 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 136.6006 | deepgemm | 127.7822 | 0.935 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 197.8214 | deepgemm | 187.5068 | 0.948 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.8332 | sglang_cutedsl | 6.6443 | 0.972 | deepgemm=6.9014 |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.9599 | deepgemm | 6.9896 | 1.004 | sglang_cutedsl=7.0038 |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.7430 | sglang_cutedsl | 4.7314 | 0.998 | deepgemm=4.7947 |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.7871 | sglang_cutedsl | 4.7277 | 0.988 | deepgemm=5.0606 |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.3638 | sglang_cutedsl | 5.3562 | 0.999 | deepgemm=5.4904 |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.5738 | sglang_cutedsl | 5.4492 | 0.978 | deepgemm=5.6404 |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.0142 | sglang_cutedsl | 4.9452 | 0.986 | deepgemm=5.2740 |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.0620 | sglang_cutedsl | 4.9821 | 0.984 | deepgemm=5.1100 |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.7437 | deepgemm | 4.8393 | 1.020 | sglang_cutedsl=4.8679 |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9025 | sglang_cutedsl | 4.7917 | 0.977 | deepgemm=5.0111 |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5075 | sglang_cutedsl | 4.3912 | 0.974 | deepgemm=4.8006 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6831 | sglang_cutedsl | 4.4084 | 0.941 | deepgemm=4.9663 |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.1278 | sglang_cutedsl | 4.8521 | 0.946 | deepgemm=5.4346 |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8179 | sglang_cutedsl | 4.6800 | 0.971 | deepgemm=5.0335 |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6442 | sglang_cutedsl | 4.4756 | 0.964 | deepgemm=4.8272 |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7075 | sglang_cutedsl | 4.8725 | 1.035 | deepgemm=4.9077 |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.1074 | sglang_cutedsl | 4.9670 | 0.973 | deepgemm=5.4267 |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7729 | deepgemm | 4.7936 | 1.004 | sglang_cutedsl=4.8637 |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5992 | sglang_cutedsl | 4.7029 | 1.023 | deepgemm=4.8277 |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5980 | sglang_cutedsl | 4.7548 | 1.034 | deepgemm=4.8163 |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.6252 | sglang_cutedsl | 4.6495 | 1.005 | deepgemm=4.7135 |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8528 | sglang_cutedsl | 4.8236 | 0.994 | deepgemm=5.0992 |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.9174 | sglang_cutedsl | 4.8935 | 0.995 | deepgemm=4.9472 |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.0304 | sglang_cutedsl | 4.9302 | 0.980 | deepgemm=5.0348 |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.4720 | sglang_cutedsl | 5.2387 | 0.957 | deepgemm=5.5721 |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1629 | sglang_cutedsl | 5.1167 | 0.991 | deepgemm=5.4254 |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4210 | deepgemm | 4.4932 | 1.016 | sglang_cutedsl=4.5653 |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4229 | sglang_cutedsl | 4.4964 | 1.017 | deepgemm=4.4978 |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.9840 | sglang_cutedsl | 4.9509 | 0.993 | deepgemm=5.2911 |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9453 | sglang_cutedsl | 4.8748 | 0.986 | deepgemm=5.0395 |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7704 | sglang_cutedsl | 4.7669 | 0.999 | deepgemm=4.8689 |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7747 | sglang_cutedsl | 4.7632 | 0.998 | deepgemm=4.8634 |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.0033 | sglang_cutedsl | 5.8985 | 0.983 | deepgemm=6.0798 |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.3602 | deepgemm | 6.3890 | 1.005 | sglang_cutedsl=6.4629 |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.8591 | sglang_cutedsl | 4.6988 | 0.967 | deepgemm=5.1544 |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4402 | deepgemm | 4.4909 | 1.011 | sglang_cutedsl=4.5669 |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0373 | deepgemm | 5.1128 | 1.015 | sglang_cutedsl=5.1854 |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.2678 | sglang_cutedsl | 5.0826 | 0.965 | deepgemm=5.3341 |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7179 | deepgemm | 4.7967 | 1.017 | sglang_cutedsl=4.8415 |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.0084 | sglang_cutedsl | 4.9432 | 0.987 | deepgemm=5.2850 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.7316 | deepgemm | 5.4682 | 0.954 | — |
| `m13_n24_k7168_s1` | tirx | 24.3952 | deepgemm | 21.4626 | 0.880 | — |
| `m4096_n24_k28672_s16` | tirx | 65.3790 | deepgemm | 63.3994 | 0.970 | — |
| `m4096_n24_k7168_s1` | tirx | 25.7843 | deepgemm | 23.6655 | 0.918 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.7087 | flashattn_sm100 | 19.9434 | 1.012 | — |
| `s1024_h32kv16_causal` | tir | 20.6421 | flashattn_sm100 | 20.2648 | 0.982 | — |
| `s1024_h32kv32` | tir | 20.2358 | flashattn_sm100 | 20.2495 | 1.001 | — |
| `s1024_h32kv32_causal` | tir | 21.1400 | flashattn_sm100 | 21.3428 | 1.010 | — |
| `s1024_h32kv4` | tir | 19.2840 | flashattn_sm100 | 19.7752 | 1.025 | — |
| `s1024_h32kv4_causal` | tir | 19.2951 | flashattn_sm100 | 19.8401 | 1.028 | — |
| `s1024_h32kv8` | tir | 19.6326 | flashattn_sm100 | 19.8128 | 1.009 | — |
| `s1024_h32kv8_causal` | tir | 19.6657 | flashattn_sm100 | 19.9076 | 1.012 | — |
| `s2048_h32kv16` | tir | 58.1079 | flashattn_sm100 | 58.4977 | 1.007 | — |
| `s2048_h32kv16_causal` | tir | 36.3860 | flashattn_sm100 | 38.8092 | 1.067 | — |
| `s2048_h32kv32` | tir | 59.3849 | flashattn_sm100 | 59.9974 | 1.010 | — |
| `s2048_h32kv32_causal` | tir | 40.5367 | flashattn_sm100 | 39.7649 | 0.981 | — |
| `s2048_h32kv4` | tir | 55.7470 | flashattn_sm100 | 56.6764 | 1.017 | — |
| `s2048_h32kv4_causal` | tir | 35.2782 | flashattn_sm100 | 37.9394 | 1.075 | — |
| `s2048_h32kv8` | tir | 57.1692 | flashattn_sm100 | 57.4633 | 1.005 | — |
| `s2048_h32kv8_causal` | tir | 35.7985 | flashattn_sm100 | 37.8540 | 1.057 | — |
| `s4096_h32kv16` | tir | 214.5373 | flashattn_sm100 | 217.8712 | 1.016 | — |
| `s4096_h32kv16_causal` | tir | 113.0484 | flashattn_sm100 | 118.5909 | 1.049 | — |
| `s4096_h32kv32` | tir | 214.0200 | flashattn_sm100 | 218.8110 | 1.022 | — |
| `s4096_h32kv32_causal` | tir | 123.3154 | flashattn_sm100 | 122.0021 | 0.989 | — |
| `s4096_h32kv4` | tir | 206.7892 | flashattn_sm100 | 209.8065 | 1.015 | — |
| `s4096_h32kv4_causal` | tir | 109.4942 | flashattn_sm100 | 115.1635 | 1.052 | — |
| `s4096_h32kv8` | tir | 207.8481 | flashattn_sm100 | 210.3508 | 1.012 | — |
| `s4096_h32kv8_causal` | tir | 111.1781 | flashattn_sm100 | 116.6816 | 1.050 | — |
| `s8192_h32kv16` | tir | 779.9341 | flashattn_sm100 | 795.2363 | 1.020 | — |
| `s8192_h32kv16_causal` | tir | 462.0088 | flashattn_sm100 | 427.2424 | 0.925 | — |
| `s8192_h32kv32` | tir | 778.9965 | flashattn_sm100 | 798.1348 | 1.025 | — |
| `s8192_h32kv32_causal` | tir | 446.8970 | flashattn_sm100 | 438.4441 | 0.981 | — |
| `s8192_h32kv4` | tir | 763.6209 | flashattn_sm100 | 773.3868 | 1.013 | — |
| `s8192_h32kv4_causal` | tir | 407.4508 | flashattn_sm100 | 421.9564 | 1.036 | — |
| `s8192_h32kv8` | tir | 775.5356 | flashattn_sm100 | 783.5840 | 1.010 | — |
| `s8192_h32kv8_causal` | tir | 409.8918 | flashattn_sm100 | 422.8655 | 1.032 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8958 | torch-cublas | 5.9873 | 0.868 | deepgemm-bf16=6.8004, deepgemm-cublaslt=5.9891 |
| `bf16_16384x16384x16384` | tir | 5969.4004 | torch-cublas | 5675.2168 | 0.951 | deepgemm-bf16=6437.1185, deepgemm-cublaslt=5730.7472 |
| `bf16_2048x2048x2048` | tir | 16.3955 | deepgemm-cublaslt | 15.6693 | 0.956 | deepgemm-bf16=17.3160, torch-cublas=15.6901 |
| `bf16_4096x4096x4096` | tir | 92.9270 | torch-cublas | 89.2582 | 0.961 | deepgemm-bf16=89.5595, deepgemm-cublaslt=89.4955 |
| `bf16_8192x8192x8192` | tir | 685.8558 | torch-cublas | 695.7970 | 1.014 | deepgemm-bf16=711.0258, deepgemm-cublaslt=700.3597 |
| `fp16_1024x1024x1024` | tir | 6.9313 | torch-cublas | 6.0037 | 0.866 | — |
| `fp16_16384x16384x16384` | tir | 5940.4282 | torch-cublas | 5903.0085 | 0.994 | — |
| `fp16_2048x2048x2048` | tir | 16.5062 | torch-cublas | 16.0436 | 0.972 | — |
| `fp16_4096x4096x4096` | tir | 95.6515 | torch-cublas | 92.4653 | 0.967 | — |
| `fp16_8192x8192x8192` | tir | 731.7101 | torch-cublas | 753.1775 | 1.029 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.7700 | deepgemm | 49.6958 | 0.979 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 115.4921 | deepgemm | 114.3964 | 0.991 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.3352 | deepgemm | 71.6330 | 1.048 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 81.6585 | deepgemm | 82.3205 | 1.008 | — |
| `deepgemm_m4096_n576_k7168` | tir | 19.9154 | deepgemm | 18.9649 | 0.952 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 325.9339 | deepgemm | 323.9114 | 0.994 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.4056 | deepgemm | 42.6474 | 0.983 | — |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 162.7028 | deepgemm | 169.4200 | 1.041 | — |
| `large_g4_m8192_n4096_k4096` | tir | 358.0344 | deepgemm | 365.3450 | 1.020 | — |
| `large_g4_m8192_n6144_k7168` | tir | 980.1809 | deepgemm | 989.3948 | 1.009 | — |
| `large_g4_m8192_n7168_k3072` | tir | 489.4461 | deepgemm | 523.6339 | 1.070 | — |
| `large_g8_m4096_n4096_k2048` | tir | 192.8730 | deepgemm | 199.7301 | 1.036 | — |
| `large_g8_m4096_n4096_k4096` | tir | 342.4548 | deepgemm | 345.9934 | 1.010 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1105.9409 | deepgemm | 1107.6434 | 1.002 | — |
| `large_g8_m4096_n7168_k3072` | tir | 508.5867 | deepgemm | 531.6377 | 1.045 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.6804 | 38.2727 | 33.7876 | 53.4710 | 60.2782 |
| `moe_a3b_bs8_all` | 101.7366 | 102.7845 | 109.0687 | 134.2527 | 141.5789 |
| `moe_a3b_bs32_all` | 202.9455 | 203.4063 | 212.2066 | 237.8707 | 239.8744 |
| `moe_a3b_bs128_all` | 224.2337 | 220.1817 | 232.1132 | 255.8537 | 258.1607 |
| `moe_a3b_bs512_all` | 237.7833 | 232.9760 | 243.5980 | 309.8964 | 297.7099 |
| `moe_a3b_bs1024_all` | 253.5489 | 251.8159 | 273.4484 | 370.7522 | 339.3610 |
| `moe_a3b_bs2048_all` | 335.4069 | 338.8775 | 351.7903 | 456.1730 | 413.5496 |
| `moe_a3b_bs4096_all` | 521.3228 | 530.5611 | 539.7104 | 665.4057 | 600.2754 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.4504 | flashinfer | 4.5841 | 0.841 | cublaslt_nvfp4=4.5850 |
| `16384x16384x16384` | tir | 1536.7759 | flashinfer | 1452.5949 | 0.945 | cublaslt_nvfp4=1461.5006 |
| `2048x2048x2048` | tir | 8.5160 | cublaslt_nvfp4 | 7.5888 | 0.891 | flashinfer=7.7881 |
| `4096x4096x4096` | tir | 29.6726 | cublaslt_nvfp4 | 28.9058 | 0.974 | flashinfer=30.9147 |
| `8192x8192x8192` | tir | 188.4441 | flashinfer | 179.4690 | 0.952 | cublaslt_nvfp4=183.6319 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1734.6353 | flashmla | 1764.2080 | 1.017 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1923.6267 | flashmla | 1921.5447 | 0.999 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1718.4375 | flashmla | 1736.0743 | 1.010 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1836.3908 | flashmla | 1852.0205 | 1.009 | trtllm_gen=2076.8240 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2010.7259 | flashmla | 2010.3743 | 1.000 | trtllm_gen=2191.8101 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1795.1110 | flashmla | 1828.5136 | 1.019 | trtllm_gen=2053.3268 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1163.3244 | flashmla | 1175.9770 | 1.011 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1201.6938 | flashmla | 1215.6051 | 1.012 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1146.3656 | flashmla | 1160.1867 | 1.012 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 372.4661 | flashmla | 382.2234 | 1.026 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 373.6835 | flashmla | 383.2534 | 1.026 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 376.7442 | flashmla | 389.0196 | 1.033 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 363.5889 | flashmla | 374.9509 | 1.031 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 383.1649 | flashmla | 397.2809 | 1.037 | trtllm_gen=463.2884 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 394.4668 | flashmla | 410.1573 | 1.040 | trtllm_gen=480.1781 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 401.2921 | flashmla | 416.9654 | 1.039 | trtllm_gen=487.7235 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 374.0787 | flashmla | 383.8616 | 1.026 | trtllm_gen=454.0089 |
