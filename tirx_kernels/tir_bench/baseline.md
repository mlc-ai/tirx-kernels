# tir-bench baseline view: `tir.json + ref.json`

- Timestamp: `20`
- Label:     `ours-x5-full`
- Git:       `{'tir': 'ca00cdc7-dirty', 'tirx-kernels': 'a6c9c52e', 'tirx-bench-ci': None}`
- Workloads: 258 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0390 | deepgemm | 0.0414 | 1.062 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0505 | deepgemm | 0.0536 | 1.061 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0394 | deepgemm | 0.0406 | 1.029 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0509 | deepgemm | 0.0525 | 1.030 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0398 | deepgemm | 0.0426 | 1.071 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0513 | deepgemm | 0.0556 | 1.084 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0388 | deepgemm | 0.0384 | 0.990 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0498 | deepgemm | 0.0495 | 0.994 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0675 | deepgemm | 0.0716 | 1.060 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1036 | deepgemm | 0.1097 | 1.059 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0685 | deepgemm | 0.0697 | 1.018 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1055 | deepgemm | 0.1065 | 1.010 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0690 | deepgemm | 0.0745 | 1.080 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1057 | deepgemm | 0.1160 | 1.098 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0668 | deepgemm | 0.0659 | 0.988 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1029 | deepgemm | 0.1019 | 0.990 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0682 | deepgemm | 0.0734 | 1.076 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0682 | deepgemm | 0.0734 | 1.076 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0689 | deepgemm | 0.0719 | 1.044 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0690 | deepgemm | 0.0720 | 1.043 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0691 | deepgemm | 0.0754 | 1.092 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0693 | deepgemm | 0.0756 | 1.091 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0672 | deepgemm | 0.0675 | 1.003 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0673 | deepgemm | 0.0676 | 1.004 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1220 | deepgemm | 0.1297 | 1.063 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1751 | deepgemm | 0.1859 | 1.062 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1234 | deepgemm | 0.1263 | 1.023 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1777 | deepgemm | 0.1809 | 1.018 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1425 | deepgemm | 0.1362 | 0.956 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1781 | deepgemm | 0.1965 | 1.104 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1200 | deepgemm | 0.1203 | 1.002 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1727 | deepgemm | 0.1725 | 0.999 | — |
## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.993 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0044 | deepgemm | 0.0043 | 0.981 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0044 | deepgemm | 0.0044 | 0.998 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0044 | deepgemm | 0.0044 | 1.002 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.038 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.043 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.057 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.062 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 0.993 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.983 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.003 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0038 | 0.988 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.022 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 0.990 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.026 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.016 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.007 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0036 | 0.961 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.981 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 1.000 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.993 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0038 | 1.077 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.066 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0036 | 1.043 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0036 | 0.962 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.976 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.983 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.020 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.010 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 1.018 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0036 | 0.943 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.002 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.005 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.966 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.991 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.033 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.043 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.074 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0038 | 1.090 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.070 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.992 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.023 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.008 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.967 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.002 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.980 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.978 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.978 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0036 | 0.964 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0036 | 0.963 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0038 | 0.993 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.999 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.066 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.067 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0036 | 1.046 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.064 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.007 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.981 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.032 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.976 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.986 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.012 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.967 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.012 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.028 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.005 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.993 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.997 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0038 | 1.080 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0039 | 1.093 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0034 | deepgemm | 0.0037 | 1.087 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0036 | 1.041 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.981 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.980 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0038 | 1.061 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 1.017 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.979 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.991 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.993 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.022 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0415 | deepgemm | 0.0440 | 1.061 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0543 | deepgemm | 0.0570 | 1.051 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0418 | deepgemm | 0.0409 | 0.977 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0540 | deepgemm | 0.0544 | 1.006 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0414 | deepgemm | 0.0440 | 1.063 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0537 | deepgemm | 0.0566 | 1.055 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0411 | deepgemm | 0.0411 | 1.000 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0533 | deepgemm | 0.0531 | 0.996 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0716 | deepgemm | 0.0762 | 1.065 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1082 | deepgemm | 0.1166 | 1.077 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0712 | deepgemm | 0.0722 | 1.014 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1089 | deepgemm | 0.1100 | 1.010 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0719 | deepgemm | 0.0761 | 1.058 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1093 | deepgemm | 0.1164 | 1.065 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0708 | deepgemm | 0.0697 | 0.985 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1096 | deepgemm | 0.1073 | 0.979 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0713 | deepgemm | 0.0775 | 1.088 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0719 | deepgemm | 0.0776 | 1.080 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0721 | deepgemm | 0.0717 | 0.993 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0713 | deepgemm | 0.0718 | 1.007 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0706 | deepgemm | 0.0773 | 1.094 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0708 | deepgemm | 0.0769 | 1.087 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0704 | deepgemm | 0.0706 | 1.003 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0703 | deepgemm | 0.0703 | 1.000 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1268 | deepgemm | 0.1377 | 1.086 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1837 | deepgemm | 0.1984 | 1.080 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1349 | deepgemm | 0.1279 | 0.948 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1850 | deepgemm | 0.1853 | 1.002 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1264 | deepgemm | 0.1380 | 1.092 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1856 | deepgemm | 0.1979 | 1.066 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1275 | deepgemm | 0.1283 | 1.007 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1840 | deepgemm | 0.1837 | 0.999 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0044 | deepgemm | 0.0044 | 0.995 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0045 | deepgemm | 0.0045 | 0.999 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.000 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.003 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.001 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.995 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.016 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0039 | 1.055 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.002 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0038 | 1.030 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0038 | 1.040 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.025 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.015 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.023 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.968 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.006 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.027 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0038 | 1.037 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.987 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.005 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.978 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.010 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0037 | 0.993 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0038 | 1.028 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.002 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0039 | 1.051 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.006 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.033 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.026 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.998 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.030 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.021 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 1.001 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0042 | deepgemm | 0.0042 | 0.997 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.002 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0038 | 1.029 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.044 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0038 | 1.028 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.002 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.998 | — |
## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 0.0051 | deepgemm | 0.0050 | 0.982 | — |
| `m13_n24_k7168_s1` | tirx | 0.0206 | deepgemm | 0.0207 | 1.005 | — |
| `m4096_n24_k28672_s16` | tirx | 0.0563 | deepgemm | 0.0568 | 1.009 | — |
| `m4096_n24_k7168_s1` | tirx | 0.0218 | deepgemm | 0.0221 | 1.011 | — |
## flash_attention4

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 0.0198 | flashattn_sm100 | 0.0200 | 1.010 | flashinfer=0.0252 |
| `s1024_h32kv16_causal` | tir | 0.0198 | flashattn_sm100 | 0.0195 | 0.983 | flashinfer=0.0251 |
| `s1024_h32kv32` | tir | 0.0201 | flashattn_sm100 | 0.0202 | 1.004 | flashinfer=0.0255 |
| `s1024_h32kv32_causal` | tir | 0.0205 | flashattn_sm100 | 0.0200 | 0.972 | flashinfer=0.0254 |
| `s1024_h32kv4` | tir | 0.0193 | flashattn_sm100 | 0.0198 | 1.022 | flashinfer=0.0250 |
| `s1024_h32kv4_causal` | tir | 0.0183 | flashattn_sm100 | 0.0191 | 1.046 | flashinfer=0.0250 |
| `s1024_h32kv8` | tir | 0.0198 | flashattn_sm100 | 0.0200 | 1.007 | flashinfer=0.0252 |
| `s1024_h32kv8_causal` | tir | 0.0188 | flashattn_sm100 | 0.0192 | 1.019 | flashinfer=0.0253 |
| `s2048_h32kv16` | tir | 0.0562 | flashattn_sm100 | 0.0569 | 1.013 | flashinfer=0.0731 |
| `s2048_h32kv16_causal` | tir | 0.0361 | flashattn_sm100 | 0.0378 | 1.045 | flashinfer=0.0731 |
| `s2048_h32kv32` | tir | 0.0580 | flashattn_sm100 | 0.0585 | 1.009 | flashinfer=0.0733 |
| `s2048_h32kv32_causal` | tir | 0.0387 | flashattn_sm100 | 0.0385 | 0.995 | flashinfer=0.0734 |
| `s2048_h32kv4` | tir | 0.0550 | flashattn_sm100 | 0.0551 | 1.002 | flashinfer=0.0730 |
| `s2048_h32kv4_causal` | tir | 0.0352 | flashattn_sm100 | 0.0369 | 1.048 | flashinfer=0.0729 |
| `s2048_h32kv8` | tir | 0.0555 | flashattn_sm100 | 0.0558 | 1.005 | flashinfer=0.0730 |
| `s2048_h32kv8_causal` | tir | 0.0355 | flashattn_sm100 | 0.0372 | 1.050 | flashinfer=0.0731 |
| `s4096_h32kv16` | tir | 0.2064 | flashattn_sm100 | 0.2084 | 1.010 | flashinfer=0.2621 |
| `s4096_h32kv16_causal` | tir | 0.1122 | flashattn_sm100 | 0.1147 | 1.023 | flashinfer=0.2619 |
| `s4096_h32kv32` | tir | 0.2093 | flashattn_sm100 | 0.2106 | 1.006 | flashinfer=0.2636 |
| `s4096_h32kv32_causal` | tir | 0.1185 | flashattn_sm100 | 0.1178 | 0.994 | flashinfer=0.2635 |
| `s4096_h32kv4` | tir | 0.2020 | flashattn_sm100 | 0.2003 | 0.991 | flashinfer=0.2603 |
| `s4096_h32kv4_causal` | tir | 0.1095 | flashattn_sm100 | 0.1117 | 1.021 | flashinfer=0.2599 |
| `s4096_h32kv8` | tir | 0.2033 | flashattn_sm100 | 0.2028 | 0.997 | flashinfer=0.2619 |
| `s4096_h32kv8_causal` | tir | 0.1100 | flashattn_sm100 | 0.1140 | 1.036 | flashinfer=0.2618 |
| `s8192_h32kv16` | tir | 0.8434 | flashattn_sm100 | 0.8278 | 0.981 | flashinfer=1.0265 |
| `s8192_h32kv16_causal` | tir | 0.4319 | flashattn_sm100 | 0.4219 | 0.977 | flashinfer=1.0207 |
| `s8192_h32kv32` | tir | 0.8511 | flashattn_sm100 | 0.8528 | 1.002 | flashinfer=1.0374 |
| `s8192_h32kv32_causal` | tir | 0.4369 | flashattn_sm100 | 0.4415 | 1.010 | flashinfer=1.0216 |
| `s8192_h32kv4` | tir | 0.8330 | flashattn_sm100 | 0.8191 | 0.983 | flashinfer=1.0148 |
| `s8192_h32kv4_causal` | tir | 0.4146 | flashattn_sm100 | 0.4374 | 1.055 | flashinfer=1.0189 |
| `s8192_h32kv8` | tir | 0.8352 | flashattn_sm100 | 0.8298 | 0.994 | flashinfer=1.0174 |
| `s8192_h32kv8_causal` | tir | 0.4097 | flashattn_sm100 | 0.4240 | 1.035 | flashinfer=1.0159 |
## fp16_bf16_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 0.0063 | torch-cublas | 0.0053 | 0.847 | deepgemm-bf16=0.0074, deepgemm-cublaslt=0.0054 |
| `bf16_16384x16384x16384` | tir | 6.3989 | torch-cublas | 6.1917 | 0.968 | deepgemm-bf16=6.8233, deepgemm-cublaslt=6.1932 |
| `bf16_2048x2048x2048` | tir | 0.0159 | torch-cublas | 0.0175 | 1.100 | deepgemm-bf16=0.0182, deepgemm-cublaslt=0.0177 |
| `bf16_4096x4096x4096` | tir | 0.0916 | deepgemm-bf16 | 0.0910 | 0.994 | deepgemm-cublaslt=0.0925, torch-cublas=0.0925 |
| `bf16_8192x8192x8192` | tir | 0.7776 | deepgemm-cublaslt | 0.7822 | 1.006 | deepgemm-bf16=0.7920, torch-cublas=0.7950 |
| `fp16_1024x1024x1024` | tir | 0.0063 | torch-cublas | 0.0054 | 0.849 | deepgemm-cublaslt=0.0054 |
| `fp16_16384x16384x16384` | tir | 6.5621 | torch-cublas | 6.4814 | 0.988 | deepgemm-cublaslt=6.5096 |
| `fp16_2048x2048x2048` | tir | 0.0161 | torch-cublas | 0.0177 | 1.095 | deepgemm-cublaslt=0.0179 |
| `fp16_4096x4096x4096` | tir | 0.0949 | deepgemm-cublaslt | 0.0968 | 1.021 | torch-cublas=0.0969 |
| `fp16_8192x8192x8192` | tir | 0.6919 | torch-cublas | 0.8217 | 1.188 | deepgemm-cublaslt=0.8291 |
## fp8_blockwise_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 0.0481 | deepgemm | 0.0493 | 1.025 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 0.1160 | deepgemm | 0.1175 | 1.013 | — |
| `deepgemm_m4096_n32768_k512` | tir | 0.0718 | deepgemm | 0.0758 | 1.057 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 0.0828 | deepgemm | 0.0840 | 1.014 | — |
| `deepgemm_m4096_n576_k7168` | tir | 0.0182 | deepgemm | 0.0187 | 1.029 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 0.3239 | deepgemm | 0.3143 | 0.970 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 0.0437 | deepgemm | 0.0445 | 1.018 | — |
| `smoke_1024x1024x1024` | tir | 0.0057 | deepgemm | 0.0062 | 1.075 | — |
| `stress_m8192_n7168_k4096` | tir | 0.1605 | deepgemm | 0.1617 | 1.007 | — |
## nvfp4_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 0.0051 | cublaslt_nvfp4 | 0.0041 | 0.810 | flashinfer=0.0043 |
| `16384x16384x16384` | tir | 1.4669 | flashinfer | 1.3866 | 0.945 | cublaslt_nvfp4=1.3995 |
| `2048x2048x2048` | tir | 0.0083 | cublaslt_nvfp4 | 0.0076 | 0.919 | flashinfer=0.0077 |
| `4096x4096x4096` | tir | 0.0290 | cublaslt_nvfp4 | 0.0300 | 1.036 | — |
| `8192x8192x8192` | tir | 0.1826 | flashinfer | 0.1813 | 0.993 | cublaslt_nvfp4=0.1819 |
## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1.8181 | flashmla | 1.8512 | 1.018 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1.9847 | flashmla | 1.9998 | 1.008 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1.8108 | flashmla | 1.8381 | 1.015 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1.9083 | flashmla | 1.9538 | 1.024 | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2.1154 | flashmla | 2.1146 | 1.000 | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1.8952 | flashmla | 1.9245 | 1.015 | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 0.3701 | flashmla | 0.3763 | 1.017 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 0.3718 | flashmla | 0.3758 | 1.011 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 0.3801 | flashmla | 0.3832 | 1.008 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 0.3650 | flashmla | 0.3707 | 1.016 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 0.3860 | flashmla | 0.3884 | 1.006 | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 0.3875 | flashmla | 0.4047 | 1.044 | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 0.4050 | flashmla | 0.4181 | 1.032 | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 0.3732 | flashmla | 0.3810 | 1.021 | — |
