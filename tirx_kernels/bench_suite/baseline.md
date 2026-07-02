# bench-suite baseline view: `tir.json + ref.json`

- Timestamp: `3`
- Label:     `d9a54390-dirty`
- Git:       `{'tir': '3f9ce073-dirty', 'tirx-kernels': 'd9a54390-dirty', 'tirx-bench-ci': None}`
- Workloads: 259 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 42.4724 | deepgemm | 41.8218 | 0.985 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 55.6306 | deepgemm | 54.4145 | 0.978 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.6625 | deepgemm | 41.0675 | 0.963 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 55.7314 | deepgemm | 52.9745 | 0.951 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 41.3631 | deepgemm | 43.4335 | 1.050 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 54.3619 | deepgemm | 57.0345 | 1.049 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.9764 | deepgemm | 39.2146 | 0.957 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 53.3808 | deepgemm | 50.7106 | 0.950 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 75.1229 | deepgemm | 72.7789 | 0.969 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 116.2489 | deepgemm | 111.5121 | 0.959 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 75.2739 | deepgemm | 71.0916 | 0.944 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 115.4503 | deepgemm | 107.9238 | 0.935 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 73.3914 | deepgemm | 76.6287 | 1.044 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 113.7862 | deepgemm | 117.4855 | 1.033 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 72.2390 | deepgemm | 68.1751 | 0.944 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 110.5514 | deepgemm | 103.7497 | 0.938 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 73.9137 | deepgemm | 74.2018 | 1.004 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 73.5261 | deepgemm | 73.9285 | 1.005 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 74.7109 | deepgemm | 72.8777 | 0.975 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 74.0585 | deepgemm | 72.3476 | 0.977 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 72.0417 | deepgemm | 76.4619 | 1.061 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 71.6549 | deepgemm | 76.0950 | 1.062 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 71.2197 | deepgemm | 68.2268 | 0.958 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 71.6510 | deepgemm | 68.6485 | 0.958 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 134.7842 | deepgemm | 131.3786 | 0.975 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 193.6488 | deepgemm | 186.9482 | 0.965 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 134.4953 | deepgemm | 127.7487 | 0.950 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 192.7905 | deepgemm | 181.4789 | 0.941 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 131.3566 | deepgemm | 137.3428 | 1.046 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 189.8828 | deepgemm | 197.7569 | 1.041 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 129.1153 | deepgemm | 121.9132 | 0.944 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 185.7770 | deepgemm | 174.2037 | 0.938 | — |
## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.5616 | deepgemm | 5.5812 | 1.004 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.7915 | deepgemm | 5.8161 | 1.004 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.4502 | deepgemm | 6.4776 | 1.004 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.2242 | deepgemm | 6.2660 | 1.007 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3412 | deepgemm | 4.7021 | 1.083 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3553 | deepgemm | 4.6936 | 1.078 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4695 | deepgemm | 5.0520 | 1.130 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3525 | deepgemm | 4.6778 | 1.075 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.0017 | deepgemm | 4.8985 | 0.979 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.9687 | deepgemm | 4.8296 | 0.972 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.3235 | deepgemm | 5.3748 | 1.010 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.1227 | deepgemm | 5.1698 | 1.009 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6744 | deepgemm | 4.6770 | 1.001 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.5872 | deepgemm | 4.6338 | 1.010 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8348 | deepgemm | 5.1381 | 1.063 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7052 | deepgemm | 4.7285 | 1.005 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.5873 | deepgemm | 4.6046 | 1.004 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.6512 | deepgemm | 4.6959 | 1.010 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.6678 | deepgemm | 4.6845 | 1.004 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.6815 | deepgemm | 4.7078 | 1.006 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3932 | deepgemm | 4.7572 | 1.083 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3128 | deepgemm | 4.6884 | 1.087 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2994 | deepgemm | 4.6422 | 1.080 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3193 | deepgemm | 4.7337 | 1.096 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.5271 | deepgemm | 4.5725 | 1.010 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.5823 | deepgemm | 4.6166 | 1.007 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.6401 | deepgemm | 4.8782 | 1.051 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5163 | deepgemm | 4.5495 | 1.007 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6648 | deepgemm | 4.8170 | 1.033 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6560 | deepgemm | 4.8248 | 1.036 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5352 | deepgemm | 4.6944 | 1.035 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5957 | deepgemm | 4.7742 | 1.039 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.6913 | deepgemm | 4.6913 | 1.000 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.6928 | deepgemm | 4.7030 | 1.002 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.9127 | deepgemm | 5.1788 | 1.054 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7622 | deepgemm | 4.7983 | 1.008 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.2294 | deepgemm | 4.4307 | 1.048 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.2411 | deepgemm | 4.4398 | 1.047 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2312 | deepgemm | 4.4362 | 1.048 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.1650 | deepgemm | 4.3977 | 1.056 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.5622 | deepgemm | 4.5874 | 1.006 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.6405 | deepgemm | 4.6454 | 1.001 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.5219 | deepgemm | 4.6075 | 1.019 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5324 | deepgemm | 4.5572 | 1.005 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.4770 | deepgemm | 4.4802 | 1.001 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.4729 | deepgemm | 4.4888 | 1.004 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.4942 | deepgemm | 4.5143 | 1.004 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5186 | deepgemm | 4.5426 | 1.005 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.9451 | deepgemm | 4.9656 | 1.004 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.9588 | deepgemm | 4.9908 | 1.006 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3145 | deepgemm | 5.3519 | 1.007 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.2055 | deepgemm | 5.2441 | 1.007 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3205 | deepgemm | 4.8730 | 1.128 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3159 | deepgemm | 4.8842 | 1.132 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2354 | deepgemm | 4.4111 | 1.041 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2703 | deepgemm | 4.7849 | 1.120 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.6832 | deepgemm | 4.9548 | 1.058 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7867 | deepgemm | 5.0774 | 1.061 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.6470 | deepgemm | 4.6633 | 1.004 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8338 | deepgemm | 5.1294 | 1.061 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7008 | deepgemm | 5.0190 | 1.068 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6239 | deepgemm | 4.8935 | 1.058 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6194 | deepgemm | 4.6187 | 1.000 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5243 | deepgemm | 4.5617 | 1.008 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.9599 | deepgemm | 5.1580 | 1.040 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.0949 | deepgemm | 5.1436 | 1.010 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.5948 | deepgemm | 5.6398 | 1.008 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.4150 | deepgemm | 5.4566 | 1.008 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3013 | deepgemm | 4.6116 | 1.072 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3432 | deepgemm | 4.6552 | 1.072 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4049 | deepgemm | 4.9339 | 1.120 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2090 | deepgemm | 4.4711 | 1.062 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7140 | deepgemm | 4.7223 | 1.002 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7241 | deepgemm | 4.7194 | 0.999 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.9944 | deepgemm | 5.2481 | 1.051 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9182 | deepgemm | 4.9505 | 1.007 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.5915 | deepgemm | 4.5759 | 0.997 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.5806 | deepgemm | 4.6152 | 1.008 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5422 | deepgemm | 4.6450 | 1.023 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5188 | deepgemm | 4.5898 | 1.016 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 43.1108 | deepgemm | 44.9237 | 1.042 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 56.8612 | deepgemm | 58.1317 | 1.022 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.8805 | deepgemm | 42.3219 | 0.987 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 56.3298 | deepgemm | 55.6615 | 0.988 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 43.6831 | deepgemm | 44.9449 | 1.029 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 56.7806 | deepgemm | 58.1242 | 1.024 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 43.0352 | deepgemm | 41.8744 | 0.973 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 56.1579 | deepgemm | 54.5714 | 0.972 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 75.3799 | deepgemm | 78.1513 | 1.037 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 116.7680 | deepgemm | 121.1505 | 1.038 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 74.1593 | deepgemm | 74.0181 | 0.998 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 114.6278 | deepgemm | 115.2152 | 1.005 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 75.6986 | deepgemm | 77.5140 | 1.024 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 117.5018 | deepgemm | 121.1687 | 1.031 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 74.2356 | deepgemm | 72.7852 | 0.980 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 115.8989 | deepgemm | 112.7934 | 0.973 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 75.6537 | deepgemm | 79.1812 | 1.047 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 75.8933 | deepgemm | 79.1762 | 1.043 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 75.1494 | deepgemm | 74.1301 | 0.986 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 74.8297 | deepgemm | 73.7305 | 0.985 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 75.5236 | deepgemm | 78.7372 | 1.043 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 76.1994 | deepgemm | 79.3050 | 1.041 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 75.3456 | deepgemm | 73.0050 | 0.969 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 75.1505 | deepgemm | 72.6294 | 0.966 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 136.3898 | deepgemm | 143.3177 | 1.051 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 198.1276 | deepgemm | 206.8107 | 1.044 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 135.2201 | deepgemm | 134.0147 | 0.991 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 195.6572 | deepgemm | 194.7769 | 0.996 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 136.7485 | deepgemm | 143.0424 | 1.046 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 198.6531 | deepgemm | 207.2180 | 1.043 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 135.9120 | deepgemm | 131.7728 | 0.970 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 196.3711 | deepgemm | 192.0039 | 0.978 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.8383 | deepgemm | 6.8259 | 0.998 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 7.0938 | deepgemm | 7.0858 | 0.999 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4714 | deepgemm | 4.5041 | 1.007 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 5.1230 | deepgemm | 5.3585 | 1.046 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.3286 | deepgemm | 5.3415 | 1.002 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.4775 | deepgemm | 5.4971 | 1.004 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.3054 | deepgemm | 5.3637 | 1.011 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.9836 | deepgemm | 5.2589 | 1.055 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.6431 | deepgemm | 4.6804 | 1.008 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9367 | deepgemm | 4.9470 | 1.002 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.8054 | deepgemm | 4.9022 | 1.020 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.7969 | deepgemm | 4.9571 | 1.033 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7825 | deepgemm | 5.0262 | 1.051 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5946 | deepgemm | 4.5995 | 1.001 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5722 | deepgemm | 4.7524 | 1.039 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.4783 | deepgemm | 4.4889 | 1.002 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.1242 | deepgemm | 5.3836 | 1.051 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9781 | deepgemm | 5.0195 | 1.008 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5839 | deepgemm | 4.8676 | 1.062 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5379 | deepgemm | 4.7942 | 1.056 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7376 | deepgemm | 5.0433 | 1.065 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8545 | deepgemm | 4.8625 | 1.002 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5454 | deepgemm | 4.5604 | 1.003 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.4411 | deepgemm | 4.4661 | 1.006 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3633 | deepgemm | 5.3774 | 1.003 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1790 | deepgemm | 5.4154 | 1.046 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.9535 | deepgemm | 5.1845 | 1.047 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6047 | deepgemm | 4.7885 | 1.040 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.8022 | deepgemm | 4.7710 | 0.994 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9766 | deepgemm | 5.2273 | 1.050 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5990 | deepgemm | 4.5769 | 0.995 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.1224 | deepgemm | 5.3390 | 1.042 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.8282 | deepgemm | 5.8551 | 1.005 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.5471 | deepgemm | 6.5531 | 1.001 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 5.0738 | deepgemm | 5.3516 | 1.055 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 5.0023 | deepgemm | 5.2507 | 1.050 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0763 | deepgemm | 5.1170 | 1.008 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.2631 | deepgemm | 5.3233 | 1.011 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7154 | deepgemm | 4.6739 | 0.991 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.9804 | deepgemm | 5.1813 | 1.040 | — |
## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.6799 | deepgemm | 5.4221 | 0.955 | — |
| `m13_n24_k7168_s1` | tirx | 23.9149 | deepgemm | 21.0167 | 0.879 | — |
| `m4096_n24_k28672_s16` | tirx | 65.1144 | deepgemm | 63.2028 | 0.971 | — |
| `m4096_n24_k7168_s1` | tirx | 25.8194 | deepgemm | 23.8234 | 0.923 | — |
## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.7282 | flashattn_sm100 | 19.9513 | 1.011 | — |
| `s1024_h32kv16_causal` | tir | 20.4038 | flashattn_sm100 | 20.2978 | 0.995 | — |
| `s1024_h32kv32` | tir | 20.0728 | flashattn_sm100 | 20.3608 | 1.014 | — |
| `s1024_h32kv32_causal` | tir | 21.0569 | flashattn_sm100 | 21.9537 | 1.043 | — |
| `s1024_h32kv4` | tir | 19.0833 | flashattn_sm100 | 19.5518 | 1.025 | — |
| `s1024_h32kv4_causal` | tir | 19.2916 | flashattn_sm100 | 19.8122 | 1.027 | — |
| `s1024_h32kv8` | tir | 19.4733 | flashattn_sm100 | 19.4892 | 1.001 | — |
| `s1024_h32kv8_causal` | tir | 19.8074 | flashattn_sm100 | 19.9971 | 1.010 | — |
| `s2048_h32kv16` | tir | 57.2867 | flashattn_sm100 | 57.5873 | 1.005 | — |
| `s2048_h32kv16_causal` | tir | 36.4395 | flashattn_sm100 | 38.4354 | 1.055 | — |
| `s2048_h32kv32` | tir | 59.3449 | flashattn_sm100 | 59.5780 | 1.004 | — |
| `s2048_h32kv32_causal` | tir | 40.5286 | flashattn_sm100 | 40.1414 | 0.990 | — |
| `s2048_h32kv4` | tir | 55.5671 | flashattn_sm100 | 56.5540 | 1.018 | — |
| `s2048_h32kv4_causal` | tir | 34.8083 | flashattn_sm100 | 37.6594 | 1.082 | — |
| `s2048_h32kv8` | tir | 56.0382 | flashattn_sm100 | 56.4974 | 1.008 | — |
| `s2048_h32kv8_causal` | tir | 35.2730 | flashattn_sm100 | 38.0562 | 1.079 | — |
| `s4096_h32kv16` | tir | 212.5262 | flashattn_sm100 | 214.2213 | 1.008 | — |
| `s4096_h32kv16_causal` | tir | 113.1526 | flashattn_sm100 | 118.2384 | 1.045 | — |
| `s4096_h32kv32` | tir | 215.8744 | flashattn_sm100 | 217.8643 | 1.009 | — |
| `s4096_h32kv32_causal` | tir | 121.8582 | flashattn_sm100 | 120.0687 | 0.985 | — |
| `s4096_h32kv4` | tir | 205.0189 | flashattn_sm100 | 208.6979 | 1.018 | — |
| `s4096_h32kv4_causal` | tir | 109.5566 | flashattn_sm100 | 114.5951 | 1.046 | — |
| `s4096_h32kv8` | tir | 207.9030 | flashattn_sm100 | 210.9836 | 1.015 | — |
| `s4096_h32kv8_causal` | tir | 110.8185 | flashattn_sm100 | 115.8925 | 1.046 | — |
| `s8192_h32kv16` | tir | 770.6608 | flashattn_sm100 | 775.3193 | 1.006 | — |
| `s8192_h32kv16_causal` | tir | 467.6625 | flashattn_sm100 | 424.5643 | 0.908 | — |
| `s8192_h32kv32` | tir | 777.8879 | flashattn_sm100 | 793.8104 | 1.020 | — |
| `s8192_h32kv32_causal` | tir | 438.0190 | flashattn_sm100 | 437.0448 | 0.998 | — |
| `s8192_h32kv4` | tir | 767.9945 | flashattn_sm100 | 767.9092 | 1.000 | — |
| `s8192_h32kv4_causal` | tir | 411.3740 | flashattn_sm100 | 418.7773 | 1.018 | — |
| `s8192_h32kv8` | tir | 765.9781 | flashattn_sm100 | 774.5108 | 1.011 | — |
| `s8192_h32kv8_causal` | tir | 410.8005 | flashattn_sm100 | 420.7198 | 1.024 | — |
## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8093 | torch-cublas | 5.9884 | 0.879 | deepgemm-bf16=7.9185, deepgemm-cublaslt=5.9905 |
| `bf16_16384x16384x16384` | tir | 5643.1360 | torch-cublas | 5514.3591 | 0.977 | deepgemm-bf16=7121.6723, deepgemm-cublaslt=5748.1841 |
| `bf16_2048x2048x2048` | tir | 16.3298 | deepgemm-cublaslt | 15.8561 | 0.971 | deepgemm-bf16=18.4107, torch-cublas=15.8657 |
| `bf16_4096x4096x4096` | tir | 93.9525 | deepgemm-cublaslt | 89.4717 | 0.952 | deepgemm-bf16=89.6780, torch-cublas=89.5411 |
| `bf16_8192x8192x8192` | tir | 670.5453 | torch-cublas | 693.2592 | 1.034 | deepgemm-bf16=698.7956, deepgemm-cublaslt=709.3353 |
| `fp16_1024x1024x1024` | tir | 6.8313 | torch-cublas | 6.0095 | 0.880 | deepgemm-cublaslt=6.0101 |
| `fp16_16384x16384x16384` | tir | 5699.5511 | torch-cublas | 5488.1470 | 0.963 | deepgemm-cublaslt=5767.7040 |
| `fp16_2048x2048x2048` | tir | 16.5019 | deepgemm-cublaslt | 16.0096 | 0.970 | torch-cublas=16.0117 |
| `fp16_4096x4096x4096` | tir | 95.2903 | deepgemm-cublaslt | 91.8152 | 0.964 | torch-cublas=92.2564 |
| `fp16_8192x8192x8192` | tir | 717.8419 | deepgemm-cublaslt | 733.9768 | 1.022 | torch-cublas=741.2138 |
## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.3925 | deepgemm | 50.7063 | 1.006 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 116.3032 | deepgemm | 115.5069 | 0.993 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.6529 | deepgemm | 71.5719 | 1.043 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 81.9528 | deepgemm | 80.6445 | 0.984 | — |
| `deepgemm_m4096_n576_k7168` | tir | 19.9840 | deepgemm | 20.4595 | 1.024 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 336.1156 | deepgemm | 335.6712 | 0.999 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.0887 | deepgemm | 43.3435 | 1.006 | — |
## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.2209 | flashinfer | 4.4884 | 0.860 | cublaslt_nvfp4=4.5231 |
| `16384x16384x16384` | tir | 1482.7563 | flashinfer | 1438.6077 | 0.970 | cublaslt_nvfp4=1457.0015 |
| `2048x2048x2048` | tir | 8.4172 | cublaslt_nvfp4 | 7.5397 | 0.896 | flashinfer=7.6880 |
| `4096x4096x4096` | tir | 29.5334 | cublaslt_nvfp4 | 28.6319 | 0.969 | flashinfer=29.7714 |
| `8192x8192x8192` | tir | 184.5940 | flashinfer | 181.7936 | 0.985 | cublaslt_nvfp4=186.0979 |
## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1721.5591 | flashmla | 1735.5278 | 1.008 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1878.8946 | flashmla | 1905.3873 | 1.014 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1693.1185 | flashmla | 1734.0474 | 1.024 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1800.4858 | flashmla | 1855.7416 | 1.031 | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1996.3684 | flashmla | 2003.0414 | 1.003 | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1774.2175 | flashmla | 1816.5808 | 1.024 | — |
## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1171.0243 | flashmla | 1165.8679 | 0.996 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1208.4711 | flashmla | 1210.3268 | 1.002 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1161.7949 | flashmla | 1153.1358 | 0.993 | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 381.5665 | flashmla | 379.2480 | 0.994 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 384.2981 | flashmla | 383.1284 | 0.997 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 389.7840 | flashmla | 389.8651 | 1.000 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 372.7660 | flashmla | 373.1569 | 1.001 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 401.8188 | flashmla | 398.2406 | 0.991 | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 405.5376 | flashmla | 404.6972 | 0.998 | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 414.6620 | flashmla | 419.3142 | 1.011 | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 383.8136 | flashmla | 382.3221 | 0.996 | — |
