# bench-suite baseline view: `baseline.json`

- Timestamp: `6`
- Label:     `6d46ac95-dirty`
- Git:       `{'tir': '9d215dda', 'tirx-kernels': '6d46ac95-dirty', 'tirx-bench-ci': None}`
- Workloads: 283 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1399.4000 | deepgemm | 1402.6000 | 1.002 | — |
| `t64_m64_h7168_i3072_e384_k6_g2` | tirx | 994.6382 | deepgemm | 987.3286 | 0.993 | — |
| `t64_m64_h7168_i3072_e384_k6_g4` | tirx | 603.9222 | deepgemm | 596.0724 | 0.987 | — |
| `t64_m64_h7168_i3072_e384_k6_g6` | tirx | 486.6194 | deepgemm | 474.4618 | 0.975 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3590.4000 | deepgemm | 3604.4000 | 1.004 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g2` | tirx | 3490.6000 | deepgemm | 3495.2000 | 1.001 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g4` | tirx | 2970.6000 | deepgemm | 2993.8000 | 1.008 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g6` | tirx | 2917.6000 | deepgemm | 2908.6000 | 0.997 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 43.2983 | deepgemm | 41.4927 | 0.958 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 56.5298 | deepgemm | 53.7754 | 0.951 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 43.4012 | deepgemm | 42.2284 | 0.973 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 56.7760 | deepgemm | 55.0184 | 0.969 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 42.1267 | deepgemm | 42.6425 | 1.012 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 55.1717 | deepgemm | 55.6725 | 1.009 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 41.6674 | deepgemm | 40.9117 | 0.982 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 54.1362 | deepgemm | 52.7401 | 0.974 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 76.4061 | deepgemm | 71.9266 | 0.941 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 118.0772 | deepgemm | 110.0149 | 0.932 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 76.3214 | deepgemm | 73.6754 | 0.965 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 117.6396 | deepgemm | 113.2241 | 0.962 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 74.6159 | deepgemm | 74.3541 | 0.996 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 116.0363 | deepgemm | 113.9990 | 0.982 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 73.5882 | deepgemm | 71.2637 | 0.968 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 112.6041 | deepgemm | 108.7477 | 0.966 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 75.9200 | deepgemm | 73.3881 | 0.967 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 75.6416 | deepgemm | 72.9091 | 0.964 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 76.4203 | deepgemm | 75.1188 | 0.983 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 76.0025 | deepgemm | 74.4491 | 0.980 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 73.9838 | deepgemm | 75.9848 | 1.027 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 73.4834 | deepgemm | 75.4814 | 1.027 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 72.8730 | deepgemm | 71.5481 | 0.982 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 73.6753 | deepgemm | 72.2706 | 0.981 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 136.1867 | deepgemm | 128.1030 | 0.941 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 195.9681 | deepgemm | 182.6603 | 0.932 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 135.8271 | deepgemm | 131.7221 | 0.970 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 195.4336 | deepgemm | 188.4316 | 0.964 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 133.1344 | deepgemm | 133.2300 | 1.001 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 191.0887 | deepgemm | 189.8582 | 0.994 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 131.3811 | deepgemm | 127.4267 | 0.970 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 188.6000 | deepgemm | 181.5716 | 0.963 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.3699 | deepgemm | 5.4560 | 1.016 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.8091 | deepgemm | 5.7960 | 0.998 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.2131 | deepgemm | 6.2377 | 1.004 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.1365 | deepgemm | 6.1175 | 0.997 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.4175 | deepgemm | 4.9848 | 1.128 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.4326 | deepgemm | 4.9746 | 1.122 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2695 | deepgemm | 4.8674 | 1.140 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2285 | deepgemm | 4.5566 | 1.078 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.0052 | deepgemm | 5.1372 | 1.026 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.6300 | deepgemm | 4.9263 | 1.064 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0830 | deepgemm | 5.2048 | 1.024 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.0196 | deepgemm | 5.1039 | 1.017 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6883 | deepgemm | 4.9975 | 1.066 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6424 | deepgemm | 4.7320 | 1.019 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7993 | deepgemm | 4.9334 | 1.028 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6985 | deepgemm | 4.9652 | 1.057 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.4185 | deepgemm | 4.5218 | 1.023 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.0833 | deepgemm | 5.3516 | 1.053 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.8719 | deepgemm | 5.1129 | 1.049 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1096 | deepgemm | 5.2384 | 1.025 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.1145 | deepgemm | 4.5300 | 1.101 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3444 | deepgemm | 4.8961 | 1.127 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.1061 | deepgemm | 4.5190 | 1.101 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4910 | deepgemm | 5.0385 | 1.122 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.9686 | deepgemm | 5.2413 | 1.055 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.3371 | deepgemm | 4.3899 | 1.012 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.3213 | deepgemm | 4.3984 | 1.018 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.6943 | deepgemm | 4.7510 | 1.012 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.3827 | deepgemm | 4.5719 | 1.043 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.4313 | deepgemm | 4.6055 | 1.039 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8303 | deepgemm | 5.1274 | 1.062 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.3346 | deepgemm | 4.4820 | 1.034 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.0956 | deepgemm | 5.3836 | 1.057 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.1201 | deepgemm | 5.0371 | 0.984 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.8757 | deepgemm | 5.1371 | 1.054 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1005 | deepgemm | 5.2594 | 1.031 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 3.9407 | deepgemm | 4.3282 | 1.098 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.9377 | deepgemm | 4.2960 | 1.091 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5464 | deepgemm | 5.1524 | 1.133 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.9028 | deepgemm | 4.2665 | 1.093 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7717 | deepgemm | 4.8367 | 1.014 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.4533 | deepgemm | 4.4947 | 1.009 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.4999 | deepgemm | 4.5916 | 1.020 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.4067 | deepgemm | 4.4820 | 1.017 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.8942 | deepgemm | 5.1180 | 1.046 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.9245 | deepgemm | 5.1988 | 1.056 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.3505 | deepgemm | 4.4197 | 1.016 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.3432 | deepgemm | 4.3921 | 1.011 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.0904 | deepgemm | 5.1928 | 1.020 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.0615 | deepgemm | 5.1348 | 1.014 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.2180 | deepgemm | 5.3054 | 1.017 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.0750 | deepgemm | 5.1485 | 1.014 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.1935 | deepgemm | 4.7639 | 1.136 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.1474 | deepgemm | 4.4594 | 1.075 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2407 | deepgemm | 4.6067 | 1.086 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.1747 | deepgemm | 4.5300 | 1.085 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.6348 | deepgemm | 4.7388 | 1.022 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.6226 | deepgemm | 4.8492 | 1.049 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.6354 | deepgemm | 4.9330 | 1.064 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7223 | deepgemm | 4.7882 | 1.014 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.5480 | deepgemm | 4.8283 | 1.062 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.4929 | deepgemm | 4.7324 | 1.053 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5187 | deepgemm | 4.7737 | 1.056 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.4948 | deepgemm | 4.7513 | 1.057 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.1256 | deepgemm | 5.2002 | 1.015 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.1480 | deepgemm | 5.2646 | 1.023 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3635 | deepgemm | 5.4722 | 1.020 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.2879 | deepgemm | 5.3182 | 1.006 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.0946 | deepgemm | 4.4764 | 1.093 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.1018 | deepgemm | 4.4874 | 1.094 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.0348 | deepgemm | 4.4091 | 1.093 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5607 | deepgemm | 5.1308 | 1.125 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.1597 | deepgemm | 5.2864 | 1.025 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 5.0932 | deepgemm | 5.1273 | 1.007 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.1093 | deepgemm | 5.2925 | 1.036 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.6613 | deepgemm | 4.7086 | 1.010 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7958 | deepgemm | 4.8957 | 1.021 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.4125 | deepgemm | 4.4828 | 1.016 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.4910 | deepgemm | 4.5616 | 1.016 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.0280 | deepgemm | 5.2725 | 1.049 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 43.9190 | deepgemm | 42.7889 | 0.974 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 58.0266 | deepgemm | 55.4052 | 0.955 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 43.8993 | deepgemm | 42.2111 | 0.962 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 57.0530 | deepgemm | 54.2935 | 0.952 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 44.0140 | deepgemm | 42.8836 | 0.974 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 57.4364 | deepgemm | 55.5704 | 0.968 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 43.5833 | deepgemm | 40.9358 | 0.939 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 57.7619 | deepgemm | 52.8878 | 0.916 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 77.3160 | deepgemm | 72.7245 | 0.941 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 118.6940 | deepgemm | 110.6860 | 0.933 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 76.7182 | deepgemm | 72.3804 | 0.943 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 116.8648 | deepgemm | 109.1478 | 0.934 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 77.6873 | deepgemm | 74.6709 | 0.961 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 119.0616 | deepgemm | 112.9759 | 0.949 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 76.2089 | deepgemm | 69.7240 | 0.915 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 116.9258 | deepgemm | 105.3425 | 0.901 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 76.9948 | deepgemm | 73.7062 | 0.957 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 77.3611 | deepgemm | 73.6626 | 0.952 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 76.5011 | deepgemm | 72.6601 | 0.950 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 76.7452 | deepgemm | 72.6941 | 0.947 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 77.1845 | deepgemm | 74.3663 | 0.963 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 77.5246 | deepgemm | 74.3325 | 0.959 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 77.0843 | deepgemm | 71.5035 | 0.928 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 77.1038 | deepgemm | 71.2268 | 0.924 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 137.8417 | deepgemm | 129.4573 | 0.939 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 196.7795 | deepgemm | 185.5670 | 0.943 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 136.1458 | deepgemm | 129.1609 | 0.949 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 197.9456 | deepgemm | 187.0002 | 0.945 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 137.8998 | deepgemm | 132.5691 | 0.961 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 197.6921 | deepgemm | 191.8943 | 0.971 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 134.2938 | deepgemm | 122.3584 | 0.911 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 193.7699 | deepgemm | 181.6490 | 0.937 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.6630 | sglang_cutedsl | 6.4732 | 0.972 | deepgemm=6.7575 |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.7638 | sglang_cutedsl | 6.6948 | 0.990 | deepgemm=6.7731 |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4729 | sglang_cutedsl | 4.4701 | 0.999 | deepgemm=4.8085 |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4895 | sglang_cutedsl | 4.3958 | 0.979 | deepgemm=4.8595 |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.2255 | sglang_cutedsl | 5.1027 | 0.977 | deepgemm=5.3518 |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.3646 | sglang_cutedsl | 5.2484 | 0.978 | deepgemm=5.4500 |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8110 | sglang_cutedsl | 4.6424 | 0.965 | deepgemm=5.1543 |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.8705 | sglang_cutedsl | 4.7621 | 0.978 | deepgemm=4.9636 |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.8187 | sglang_cutedsl | 4.5121 | 0.936 | deepgemm=5.0894 |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7870 | sglang_cutedsl | 4.5063 | 0.941 | deepgemm=5.0168 |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5560 | sglang_cutedsl | 4.3985 | 0.965 | deepgemm=4.7804 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4676 | sglang_cutedsl | 4.3569 | 0.975 | deepgemm=4.6820 |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.4017 | sglang_cutedsl | 4.3438 | 0.987 | deepgemm=4.5045 |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.6614 | sglang_cutedsl | 4.4090 | 0.946 | deepgemm=4.8753 |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5160 | sglang_cutedsl | 4.4038 | 0.975 | deepgemm=4.7261 |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5223 | sglang_cutedsl | 4.5538 | 1.007 | deepgemm=4.7306 |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.0576 | sglang_cutedsl | 4.7554 | 0.940 | deepgemm=5.2129 |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7728 | sglang_cutedsl | 4.5546 | 0.954 | deepgemm=4.8760 |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2230 | sglang_cutedsl | 4.2476 | 1.006 | deepgemm=4.3464 |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4614 | sglang_cutedsl | 4.5557 | 1.021 | deepgemm=4.6132 |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.4286 | sglang_cutedsl | 4.3778 | 0.989 | deepgemm=4.5449 |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7161 | sglang_cutedsl | 4.4739 | 0.949 | deepgemm=4.8229 |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7966 | sglang_cutedsl | 4.5786 | 0.955 | deepgemm=4.8267 |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5301 | sglang_cutedsl | 4.5754 | 1.010 | deepgemm=4.7609 |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.2616 | sglang_cutedsl | 4.9545 | 0.942 | deepgemm=5.3594 |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1209 | sglang_cutedsl | 4.8228 | 0.942 | deepgemm=5.1459 |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5737 | sglang_cutedsl | 4.4886 | 0.981 | deepgemm=4.7175 |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2305 | sglang_cutedsl | 4.2284 | 1.000 | deepgemm=4.3423 |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.5835 | sglang_cutedsl | 4.6721 | 1.019 | deepgemm=4.6998 |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5278 | sglang_cutedsl | 4.6056 | 1.017 | deepgemm=4.6173 |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5962 | sglang_cutedsl | 4.5895 | 0.999 | deepgemm=4.8553 |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6812 | sglang_cutedsl | 4.5149 | 0.964 | deepgemm=4.7575 |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.7701 | sglang_cutedsl | 5.6606 | 0.981 | deepgemm=5.8827 |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.4282 | sglang_cutedsl | 6.1161 | 0.951 | deepgemm=6.4307 |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5504 | sglang_cutedsl | 4.4114 | 0.969 | deepgemm=4.6341 |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4869 | sglang_cutedsl | 4.3115 | 0.961 | deepgemm=4.5811 |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.1247 | sglang_cutedsl | 4.8488 | 0.946 | deepgemm=5.1986 |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9978 | sglang_cutedsl | 4.9355 | 0.988 | deepgemm=5.1040 |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.1215 | sglang_cutedsl | 4.7132 | 0.920 | deepgemm=5.4546 |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.1151 | sglang_cutedsl | 4.7118 | 0.921 | deepgemm=5.4176 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.4381 | deepgemm | 5.1751 | 0.952 | — |
| `m13_n24_k7168_s1` | tirx | 23.5442 | deepgemm | 20.6087 | 0.875 | — |
| `m4096_n24_k28672_s16` | tirx | 65.5240 | deepgemm | 63.2597 | 0.965 | — |
| `m4096_n24_k7168_s1` | tirx | 25.8518 | deepgemm | 23.8094 | 0.921 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.6689 | flashattn_sm100 | 20.2116 | 1.028 | — |
| `s1024_h32kv16_causal` | tir | 20.5078 | flashattn_sm100 | 20.4915 | 0.999 | — |
| `s1024_h32kv32` | tir | 20.2300 | flashattn_sm100 | 20.5043 | 1.014 | — |
| `s1024_h32kv32_causal` | tir | 20.7123 | flashattn_sm100 | 21.2563 | 1.026 | — |
| `s1024_h32kv4` | tir | 19.3424 | flashattn_sm100 | 20.0562 | 1.037 | — |
| `s1024_h32kv4_causal` | tir | 18.8113 | flashattn_sm100 | 19.7083 | 1.048 | — |
| `s1024_h32kv8` | tir | 19.7230 | flashattn_sm100 | 20.2036 | 1.024 | — |
| `s1024_h32kv8_causal` | tir | 19.3203 | flashattn_sm100 | 19.6630 | 1.018 | — |
| `s2048_h32kv16` | tir | 58.2121 | flashattn_sm100 | 59.1734 | 1.017 | — |
| `s2048_h32kv16_causal` | tir | 36.9011 | flashattn_sm100 | 39.4810 | 1.070 | — |
| `s2048_h32kv32` | tir | 59.8188 | flashattn_sm100 | 60.4701 | 1.011 | — |
| `s2048_h32kv32_causal` | tir | 41.0778 | flashattn_sm100 | 41.2278 | 1.004 | — |
| `s2048_h32kv4` | tir | 56.2248 | flashattn_sm100 | 57.6953 | 1.026 | — |
| `s2048_h32kv4_causal` | tir | 35.5878 | flashattn_sm100 | 38.7227 | 1.088 | — |
| `s2048_h32kv8` | tir | 56.8717 | flashattn_sm100 | 57.8143 | 1.017 | — |
| `s2048_h32kv8_causal` | tir | 35.7693 | flashattn_sm100 | 38.9340 | 1.088 | — |
| `s4096_h32kv16` | tir | 206.2440 | flashattn_sm100 | 206.9921 | 1.004 | — |
| `s4096_h32kv16_causal` | tir | 109.3491 | flashattn_sm100 | 115.6866 | 1.058 | — |
| `s4096_h32kv32` | tir | 213.0847 | flashattn_sm100 | 212.4507 | 0.997 | — |
| `s4096_h32kv32_causal` | tir | 118.0565 | flashattn_sm100 | 117.3237 | 0.994 | — |
| `s4096_h32kv4` | tir | 200.0160 | flashattn_sm100 | 200.8203 | 1.004 | — |
| `s4096_h32kv4_causal` | tir | 105.8614 | flashattn_sm100 | 112.3307 | 1.061 | — |
| `s4096_h32kv8` | tir | 207.0869 | flashattn_sm100 | 204.8762 | 0.989 | — |
| `s4096_h32kv8_causal` | tir | 106.6751 | flashattn_sm100 | 113.7795 | 1.067 | — |
| `s8192_h32kv16` | tir | 772.5284 | flashattn_sm100 | 775.2997 | 1.004 | — |
| `s8192_h32kv16_causal` | tir | 456.8787 | flashattn_sm100 | 412.9266 | 0.904 | — |
| `s8192_h32kv32` | tir | 778.3614 | flashattn_sm100 | 780.8814 | 1.003 | — |
| `s8192_h32kv32_causal` | tir | 423.6165 | flashattn_sm100 | 412.1708 | 0.973 | — |
| `s8192_h32kv4` | tir | 745.9915 | flashattn_sm100 | 760.7229 | 1.020 | — |
| `s8192_h32kv4_causal` | tir | 403.1465 | flashattn_sm100 | 409.9404 | 1.017 | — |
| `s8192_h32kv8` | tir | 762.1011 | flashattn_sm100 | 764.2141 | 1.003 | — |
| `s8192_h32kv8_causal` | tir | 395.9707 | flashattn_sm100 | 406.7576 | 1.027 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8047 | deepgemm-cublaslt | 5.6751 | 0.834 | deepgemm-bf16=6.7580, torch-cublas=5.6799 |
| `bf16_16384x16384x16384` | tir | 5440.1801 | torch-cublas | 5491.7709 | 1.009 | deepgemm-bf16=6084.4472, deepgemm-cublaslt=5509.1097 |
| `bf16_2048x2048x2048` | tir | 16.3238 | torch-cublas | 15.7965 | 0.968 | deepgemm-bf16=17.2685, deepgemm-cublaslt=15.7991 |
| `bf16_4096x4096x4096` | tir | 90.5814 | deepgemm-bf16 | 84.8929 | 0.937 | deepgemm-cublaslt=85.9197, torch-cublas=85.8411 |
| `bf16_8192x8192x8192` | tir | 665.9228 | torch-cublas | 684.3654 | 1.028 | deepgemm-bf16=700.4187, deepgemm-cublaslt=699.8988 |
| `fp16_1024x1024x1024` | tir | 6.6763 | torch-cublas | 5.6907 | 0.852 | — |
| `fp16_16384x16384x16384` | tir | 5906.1808 | torch-cublas | 5752.0958 | 0.974 | — |
| `fp16_2048x2048x2048` | tir | 16.5622 | torch-cublas | 15.8775 | 0.959 | — |
| `fp16_4096x4096x4096` | tir | 92.9994 | torch-cublas | 87.8782 | 0.945 | — |
| `fp16_8192x8192x8192` | tir | 713.9263 | torch-cublas | 715.0967 | 1.002 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.6382 | deepgemm | 49.2755 | 0.973 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 108.9182 | deepgemm | 107.1198 | 0.983 | — |
| `deepgemm_m4096_n32768_k512` | tir | 65.9984 | deepgemm | 68.8952 | 1.044 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 77.2482 | deepgemm | 76.8766 | 0.995 | — |
| `deepgemm_m4096_n576_k7168` | tir | 19.9765 | deepgemm | 19.0320 | 0.953 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 329.6422 | deepgemm | 330.9794 | 1.004 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 42.3266 | deepgemm | 41.1759 | 0.973 | — |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 155.6456 | deepgemm | 158.4962 | 1.018 | — |
| `large_g4_m8192_n4096_k4096` | tir | 351.3493 | deepgemm | 358.2620 | 1.020 | — |
| `large_g4_m8192_n6144_k7168` | tir | 966.1340 | deepgemm | 977.8325 | 1.012 | — |
| `large_g4_m8192_n7168_k3072` | tir | 484.2481 | deepgemm | 503.3133 | 1.039 | — |
| `large_g8_m4096_n4096_k2048` | tir | 179.0180 | deepgemm | 181.4375 | 1.014 | — |
| `large_g8_m4096_n4096_k4096` | tir | 330.4618 | deepgemm | 352.8749 | 1.068 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1109.7982 | deepgemm | 1138.2328 | 1.026 | — |
| `large_g8_m4096_n7168_k3072` | tir | 512.8243 | deepgemm | 540.0880 | 1.053 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 34.1567 | 38.3291 | 34.7047 | 56.5890 | 67.9720 |
| `moe_a3b_bs8_all` | 101.6912 | 103.0323 | 110.8315 | 133.1151 | 137.6142 |
| `moe_a3b_bs32_all` | 203.3891 | 203.2642 | 212.0484 | 240.8524 | 233.0791 |
| `moe_a3b_bs128_all` | 221.4410 | 219.1773 | 228.8485 | 254.8622 | 255.0274 |
| `moe_a3b_bs512_all` | 231.3416 | 229.4745 | 240.5195 | 309.0578 | 293.4130 |
| `moe_a3b_bs1024_all` | 252.3019 | 248.1186 | 269.1638 | 366.2985 | 336.1219 |
| `moe_a3b_bs2048_all` | 335.5528 | 334.3007 | 354.7237 | 454.3465 | 405.6193 |
| `moe_a3b_bs4096_all` | 512.8225 | 523.8493 | 529.4321 | 648.2106 | 574.2225 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.0545 | cublaslt_nvfp4 | 4.2144 | 0.834 | flashinfer=4.4977 |
| `16384x16384x16384` | tir | 1509.4420 | flashinfer | 1393.8108 | 0.923 | cublaslt_nvfp4=1474.6452 |
| `2048x2048x2048` | tir | 8.3693 | cublaslt_nvfp4 | 7.1390 | 0.853 | flashinfer=7.6177 |
| `4096x4096x4096` | tir | 29.9256 | cublaslt_nvfp4 | 28.9933 | 0.969 | flashinfer=30.9707 |
| `8192x8192x8192` | tir | 180.2704 | flashinfer | 169.7735 | 0.942 | cublaslt_nvfp4=175.5454 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1685.3996 | flashmla | 1715.0027 | 1.018 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1894.6733 | flashmla | 1914.3005 | 1.010 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1679.1788 | flashmla | 1700.6477 | 1.013 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1771.1749 | flashmla | 1808.6809 | 1.021 | trtllm_gen=1948.4338 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1986.3648 | flashmla | 1970.5602 | 0.992 | trtllm_gen=2087.6511 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1789.9039 | flashmla | 1813.1086 | 1.013 | trtllm_gen=1950.4015 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1149.6556 | flashmla | 1148.7789 | 0.999 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1193.5247 | flashmla | 1201.5021 | 1.007 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1146.4572 | flashmla | 1157.4642 | 1.010 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 366.0671 | flashmla | 379.5484 | 1.037 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 368.9547 | flashmla | 379.4678 | 1.028 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 374.8349 | flashmla | 385.5353 | 1.029 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 364.5423 | flashmla | 374.1930 | 1.026 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 381.6064 | flashmla | 400.7907 | 1.050 | trtllm_gen=474.2100 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 385.0149 | flashmla | 401.6918 | 1.043 | trtllm_gen=481.8429 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 396.1295 | flashmla | 413.2136 | 1.043 | trtllm_gen=487.0906 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 369.4077 | flashmla | 381.7007 | 1.033 | trtllm_gen=457.9406 |
